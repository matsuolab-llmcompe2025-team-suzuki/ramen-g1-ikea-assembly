"""head_left mp4 由来の local png dir を LS project に pre-label 済 task として import。

Cycle 1 (100 frame pilot): v10_hsv07 pseudo → LS で user 修正 (table_top/leg 重点)。
Cycle 2 (残り frame): v11a pseudo → LS で軽微修正 (同 script 再利用、weight と subdir 差替え)。

`predict_and_prelabel.py` は skill/ep 階層 (HF ep0062) 前提だが、head_left は flat な
local png dir なのでこの派生 script が必要。predict / LS API / 座標変換ロジックは
predict_and_prelabel.py と ls_export_to_yolo_obb.py から reuse。

処理:
1. --frames-dir の png を LOCAL_FILES_ROOT/<ls-subdir>/ に symlink (LS が serve できるよう)
2. YOLO OBB で predict (predict_and_prelabel.run_yolo_predict を再利用)
3. LS task に predictions を埋めて import
4. project url を表示 → user は LS UI で修正 → export → ls_export_to_yolo_obb で YOLO 化

Usage (LS は事前に `pixi run serve` で起動、YOLO weight は model/yolo_obb env で pixi 済):
    cd model/yolo_obb  # ultralytics 込みの env
    pixi run python ../../data/annotation_tool/scripts/prelabel_head_left.py \\
        --frames-dir ../../data/yolo_obb/pseudo_source/head_left/frames \\
        --weight runs/m_lowaug_v10_hsv07_imgsz800/weights/best.pt \\
        --conf 0.25 \\
        --project-title upperpolicy_v11_head_left_cycle1 \\
        --ls-subdir head_left_v11_cycle1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv

# 同 dir の predict_and_prelabel.py から LS API + 変換 + predict を reuse
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from predict_and_prelabel import (  # noqa: E402
    LS_BASE,
    LOCAL_FILES_ROOT,
    LABEL_CONFIG_XML,
    build_ls_task,
    get_or_create_project,
    import_tasks,
    run_yolo_predict,
    wait_for_ls,
)
from bootstrap_project import get_or_create_local_storage  # noqa: E402

# ------------------------------------------------------------
# LS storage 登録 (bootstrap_project.get_or_create_local_storage は
# ep=int を title に埋め込むので、head_left 用に薄い wrapper)
# ------------------------------------------------------------
LS_TOKEN = "abcdef0123456789abcdef0123456789abcdef01"
HEADERS = {"Authorization": f"Token {LS_TOKEN}"}


def get_or_create_local_storage_by_title(
    project_id: int, title: str, local_root: Path
) -> int:
    """head_left 用: title 指定で per-project local file storage を登録 or reuse。

    bootstrap_project.get_or_create_local_storage は ep 番号ベースで
    title を組むため、head_left では reuse できない。LS API は同 rel_path prefix。
    """
    # 既存 storage を title で検索
    r = requests.get(
        f"{LS_BASE}/api/storages/localfiles/",
        headers=HEADERS,
        params={"project": project_id},
    )
    r.raise_for_status()
    for s in r.json():
        if s.get("title") == title:
            print(f"  reuse local storage id={s['id']} title='{title}'")
            return int(s["id"])
    # 新規作成
    r = requests.post(
        f"{LS_BASE}/api/storages/localfiles/",
        headers=HEADERS,
        json={
            "project": project_id,
            "title": title,
            "path": str(local_root),
            "use_blob_urls": True,
            "regex_filter": r".*\.(png|jpg|jpeg)$",
        },
    )
    r.raise_for_status()
    sid = r.json()["id"]
    print(f"  created local storage id={sid} title='{title}' path={local_root}")
    return int(sid)


# ------------------------------------------------------------
# frame symlink
# ------------------------------------------------------------
def symlink_frames(frames_dir: Path, dst_subdir: str) -> list[tuple[Path, str]]:
    """--frames-dir の png を LOCAL_FILES_ROOT/<dst_subdir>/ に symlink。

    Returns: [(original_abs_path, rel_path_from_LOCAL_FILES_ROOT), ...]
    """
    dst_dir = LOCAL_FILES_ROOT / dst_subdir
    dst_dir.mkdir(parents=True, exist_ok=True)

    picks: list[tuple[Path, str]] = []
    files = sorted(frames_dir.glob("*.png"))
    for f in files:
        link_path = dst_dir / f.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(f.resolve())
        rel = f"{dst_subdir}/{f.name}"
        picks.append((f.resolve(), rel))
    print(f"  symlinked {len(picks)} frames: {frames_dir} → {dst_dir}")
    return picks


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--frames-dir", type=Path, required=True,
                    help="head_left png dir (extract_frames_for_pseudo.py の出力先)")
    p.add_argument("--weight", type=Path, required=True,
                    help="YOLO OBB best.pt (Cycle 1 = v10_hsv07、Cycle 2 = v11a)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--project-title", type=str, required=True,
                    help="例: upperpolicy_v11_head_left_cycle1")
    p.add_argument("--ls-subdir", type=str, required=True,
                    help="LS DOCUMENT_ROOT/images/<subdir>/ 配下に symlink する dir 名")
    p.add_argument("--model-version", type=str, default=None,
                    help="LS prediction model_version (default: weight run 名)")
    p.add_argument("--dry-run", action="store_true", help="LS import はせず summary のみ")
    args = p.parse_args()

    if not args.weight.exists():
        raise SystemExit(f"weight not found: {args.weight}")
    if not args.frames_dir.is_dir():
        raise SystemExit(f"frames-dir not found: {args.frames_dir}")
    if not LABEL_CONFIG_XML.exists():
        raise SystemExit(f"label config not found: {LABEL_CONFIG_XML}")

    model_version = args.model_version or args.weight.parent.parent.name

    print(f"[prelabel_head_left]")
    print(f"  frames-dir  = {args.frames_dir}")
    print(f"  weight      = {args.weight} (model_version={model_version})")
    print(f"  project     = {args.project_title}")
    print(f"  ls-subdir   = images/{args.ls_subdir}")
    print()

    print("=== 1. symlink frames into LS DOCUMENT_ROOT ===")
    picks = symlink_frames(args.frames_dir, args.ls_subdir)
    if not picks:
        raise SystemExit("no png in frames-dir")
    print()

    print("=== 2. YOLO predict ===")
    predictions = run_yolo_predict(args.weight, [p_[0] for p_ in picks], args.conf)

    tasks: list[dict] = []
    for (_abs_path, rel_path), pred in zip(picks, predictions):
        task = build_ls_task(
            rel_path, pred["boxes"], pred["img_w"], pred["img_h"], model_version
        )
        tasks.append(task)
    print()

    if args.dry_run:
        print("=== 3. dry-run summary ===")
        n_with_preds = sum(1 for t in tasks if "predictions" in t)
        n_boxes = sum(
            len(t["predictions"][0]["result"]) for t in tasks if "predictions" in t
        )
        print(f"  {n_with_preds}/{len(tasks)} tasks have predictions ({n_boxes} boxes)")
        import json
        print(f"  sample task[0]:\n{json.dumps(tasks[0], indent=2)[:800]}")
        return

    print("=== 3. LS import ===")
    wait_for_ls()
    label_config = LABEL_CONFIG_XML.read_text()
    desc = (
        f"IROS 2026 Upper Policy v11 head_left pre-annotation "
        f"({model_version}, {len(tasks)} tasks, ls-subdir={args.ls_subdir})"
    )
    pid = get_or_create_project(args.project_title, desc, label_config)
    storage_title = f"head_left:{args.ls_subdir}"
    get_or_create_local_storage_by_title(pid, storage_title, LOCAL_FILES_ROOT)
    import_tasks(pid, tasks)
    print()
    print("=" * 60)
    print(f"DONE. URL: {LS_BASE}/projects/{pid}/data")
    print("login: admin@example.com / admin1234")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        print(
            f"response: {e.response.text if e.response is not None else 'n/a'}",
            file=sys.stderr,
        )
        sys.exit(1)
