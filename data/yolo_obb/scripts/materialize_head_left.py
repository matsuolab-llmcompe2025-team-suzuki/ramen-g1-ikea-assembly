"""LS project から head_left annotation を fetch → YOLO OBB txt → dataset 配置。

Cycle 1 (100 frame): --cycle cycle1 --project-id 26
Cycle 2 (残り frame): --cycle cycle2 --project-id <new>

処理:
1. LS project の export JSON を fetch (`workspace/exports/head_left_<cycle>_<ts>.json`)
2. parse_ls_export で YOLO OBB 形式に変換
3. `data/yolo_obb/head_left/<cycle>/images/frame_XXXXXX.png` に symlink
   `data/yolo_obb/head_left/<cycle>/labels/frame_XXXXXX.txt` に txt 書き出し
4. summary print (class counts, box 数)

Usage (annotation_tool env で叩く、requests + dotenv 使用):
    cd data/annotation_tool
    pixi run python ../yolo_obb/scripts/materialize_head_left.py \\
        --project-id 26 \\
        --cycle cycle1 \\
        --frames-dir ../yolo_obb/pseudo_source/head_left/frames \\
        --hf-push <your-dataset-repo>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv

# annotation_tool/scripts の関数を再利用
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]  # data/yolo_obb/scripts → iros_2026_ramen
ANNOTATION_TOOL_SCRIPTS = REPO_ROOT / "data" / "annotation_tool" / "scripts"
sys.path.insert(0, str(ANNOTATION_TOOL_SCRIPTS))
from ls_export_to_yolo_obb import parse_ls_export, write_yolo_obb_txts  # noqa: E402
from push_annotations import fetch_ls_export as fetch_ls_export_full  # noqa: E402

LS_BASE = "http://localhost:8080"
LS_TOKEN = "abcdef0123456789abcdef0123456789abcdef01"
HEADERS = {"Authorization": f"Token {LS_TOKEN}"}

EXPORT_DIR = REPO_ROOT / "data" / "annotation_tool" / "workspace" / "exports"
YOLO_OBB_ROOT = REPO_ROOT / "data" / "yolo_obb"


def fetch_export(project_id: int, fallback_to_predictions: bool = False) -> list[dict]:
    """LS export fetch。fallback_to_predictions=True で未 annotation task も引っ張り、
    predictions を実 dict に inline embed する (push_annotations.fetch_ls_export を再利用)。
    """
    return fetch_ls_export_full(project_id, download_all_tasks=fallback_to_predictions)


def flatten_parsed(parsed: dict) -> dict:
    """parse_ls_export の rel_path (`head_left_v11_cycle1/frame_XXX`) を stem のみに flatten。

    head_left 用は skill/ep 階層を持たないため flat な frame_XXX に統一する。
    """
    out = {}
    for rel_path, boxes in parsed.items():
        stem = Path(rel_path).stem
        if stem in out:
            print(f"[warn] duplicate stem {stem}, overwrite", file=sys.stderr)
        out[stem] = boxes
    return out


def symlink_images(frames_dir: Path, dst_images_dir: Path, stems: list[str]) -> int:
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem in stems:
        src = frames_dir / f"{stem}.png"
        if not src.exists():
            print(f"[warn] source frame not found: {src}", file=sys.stderr)
            continue
        link = dst_images_dir / f"{stem}.png"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve())
        n += 1
    return n


def push_to_hf(export_json: Path, label_dir: Path, cycle: str, repo_id: str, token: str | None) -> None:
    """LS export + YOLO txts を HF annotations repo に push。

    HF 側 layout:
      ls_exports/head_left_<cycle>_<ts>.json
      yolo_obb/labels/head_left/<cycle>/<stem>.txt
    """
    from huggingface_hub import CommitOperationAdd, HfApi, create_repo

    create_repo(
        repo_id=repo_id, repo_type="dataset",
        private=True, exist_ok=True, token=token,
    )

    api = HfApi(token=token)
    ops = [
        CommitOperationAdd(
            path_in_repo=f"ls_exports/{export_json.name}",
            path_or_fileobj=str(export_json),
        )
    ]
    for txt in sorted(label_dir.glob("*.txt")):
        ops.append(
            CommitOperationAdd(
                path_in_repo=f"yolo_obb/labels/head_left/{cycle}/{txt.name}",
                path_or_fileobj=str(txt),
            )
        )
    api.create_commit(
        repo_id=repo_id, repo_type="dataset", operations=ops,
        commit_message=f"add head_left {cycle} ({len(ops)-1} txts)",
        token=token,
    )
    print(f"[hf-up] pushed {len(ops)} files → {repo_id}")


def main() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--cycle", type=str, required=True,
                    help="dataset 側 dir 名 (cycle1 / cycle2)")
    p.add_argument("--frames-dir", type=Path, required=True,
                    help="元 frame の png dir (extract_frames_for_pseudo.py 出力)")
    p.add_argument("--hf-push", metavar="REPO", default=None,
                   help="push 先の HF dataset repo (指定したときだけ push。既定なし、Issue #190)")
    p.add_argument("--fallback-to-predictions", action="store_true",
                    help="未 annotation task は predictions (v11a pseudo) を採用")
    args = p.parse_args()

    if not args.frames_dir.is_dir():
        raise SystemExit(f"frames-dir not found: {args.frames_dir}")

    print(f"[materialize_head_left] project={args.project_id} cycle={args.cycle}")

    # 1. fetch + save export
    print("=== 1. fetch LS export ===")
    tasks = fetch_export(args.project_id, fallback_to_predictions=args.fallback_to_predictions)
    n_with_anno = sum(1 for t in tasks if t.get("annotations"))
    n_boxes_anno = sum(len(a.get("result", []))
                       for t in tasks for a in t.get("annotations", []))
    n_only_pred = sum(1 for t in tasks
                      if not t.get("annotations") and t.get("predictions"))
    print(f"  {len(tasks)} tasks, {n_with_anno} with annotations ({n_boxes_anno} boxes), "
          f"{n_only_pred} using pseudo fallback")
    if not args.fallback_to_predictions and n_with_anno < len(tasks):
        print(f"  [warn] {len(tasks) - n_with_anno} tasks lack annotations "
              "(will produce empty label files, pass --fallback-to-predictions to use pseudo)",
              file=sys.stderr)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_json = EXPORT_DIR / f"head_left_{args.cycle}_{ts}.json"
    export_json.write_text(json.dumps(tasks))
    print(f"  saved: {export_json}")

    # 2. parse → flatten
    print("=== 2. parse LS export → YOLO OBB ===")
    parsed = parse_ls_export(export_json, fallback_to_predictions=args.fallback_to_predictions)
    parsed_flat = flatten_parsed(parsed)
    stems = sorted(parsed_flat.keys())
    class_counts = Counter()
    for boxes in parsed_flat.values():
        for cid, _ in boxes:
            class_counts[cid] += 1
    print(f"  {len(parsed_flat)} images, {sum(len(b) for b in parsed_flat.values())} boxes")
    print(f"  class counts: {dict(sorted(class_counts.items()))}")

    # 3. write labels + symlink images
    print("=== 3. write labels + symlink images ===")
    cycle_dir = YOLO_OBB_ROOT / "head_left" / args.cycle
    label_dir = cycle_dir / "labels"
    image_dir = cycle_dir / "images"
    label_dir.mkdir(parents=True, exist_ok=True)
    n_txt = write_yolo_obb_txts(parsed_flat, label_dir)
    n_img = symlink_images(args.frames_dir, image_dir, stems)
    print(f"  labels: {n_txt} → {label_dir}")
    print(f"  images: {n_img} symlinks → {image_dir}")

    # 4. HF push (optional)
    if args.hf_push:
        print("=== 4. HF push ===")
        push_to_hf(export_json, label_dir, args.cycle, args.hf_push, token=None)

    print("[done]")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        print(f"response: {e.response.text if e.response is not None else 'n/a'}",
              file=sys.stderr)
        sys.exit(1)
