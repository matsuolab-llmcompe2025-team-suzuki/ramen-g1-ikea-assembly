"""LS project の annotations を fetch → YOLO OBB 変換 → HF push まで一発。

Epic #43 / Issue #44。

処理:
1. LS API から指定 ep の project の annotations JSON を取得
2. `workspace/exports/ep{XX}_{timestamp}.json` に raw を保存 (timestamped、上書きせず)
3. `ls_export_to_yolo_obb.py` の関数で YOLO OBB txt に変換 →
   `workspace/yolo_obb/{skill}__ep{XX}__frame_{YY}.txt`
4. `--hf-repo` で指定した HF の dataset repo に batch commit で push (指定が無ければ push しない):
   - `ls_exports/ep{XX}_{timestamp}.json` (source of truth、追記)
   - `yolo_obb/labels/*.txt` (training ready、上書き)

Usage (LS 起動中で annotation 済み前提):
    pixi run push-annotations --ep 60 --hf-repo <your-dataset-repo>
    pixi run push-annotations --eps 60,62 --hf-repo <your-dataset-repo>
    pixi run push-annotations --ep 60  # local 変換のみ (--hf-repo なし)

    # v2 semi-supervised (Issue #53) のような custom title project は --project-id 指定:
    pixi run push-annotations --ep 62 --project-id 7
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, create_repo

# scripts/ 配下からの import
from ls_export_to_yolo_obb import parse_ls_export, write_yolo_obb_txts

# ---- 定数 ----
LS_BASE = "http://localhost:8080"
LS_TOKEN = "abcdef0123456789abcdef0123456789abcdef01"
HEADERS = {"Authorization": f"Token {LS_TOKEN}"}


EXP_DIR = Path(__file__).parent.parent.resolve()
EXPORT_DIR = EXP_DIR / "workspace" / "exports"
YOLO_OUT_DIR = EXP_DIR / "workspace" / "yolo_obb"


# ------------------------------------------------------------
# LS API
# ------------------------------------------------------------
def find_project_by_ep(ep: int) -> int | None:
    """title=`upperpolicy_ep{XX:04d}` の project id を返す (無ければ None)。"""
    title = f"upperpolicy_ep{ep:04d}"
    r = requests.get(f"{LS_BASE}/api/projects/", headers=HEADERS)
    r.raise_for_status()
    payload = r.json()
    projects = payload if isinstance(payload, list) else payload.get("results", [])
    for p in projects:
        if p.get("title") == title:
            return p["id"]
    return None


def fetch_ls_export(project_id: int, download_all_tasks: bool = False) -> list[dict]:
    """LS の export API から full JSON export を取得 (task + annotations)。

    download_all_tasks=True: LS default (annotations あり tasks のみ) でなく全 tasks を取得。
    prediction-only tasks (未 review、prelabel のみ) も含めたい時に True にする。
    かつ predictions フィールド (LS export では ID の list) を実 prediction dict に置換
    (parse_ls_export の fallback_to_predictions logic が dict list を期待するため)。
    """
    params = {"exportType": "JSON"}
    if download_all_tasks:
        params["download_all_tasks"] = "true"
    r = requests.get(
        f"{LS_BASE}/api/projects/{project_id}/export",
        headers=HEADERS,
        params=params,
    )
    r.raise_for_status()
    tasks = r.json()

    if download_all_tasks:
        # LS export の predictions は ID の list なので、別 API で実 dict を fetch + inline embed
        # /api/predictions は paginated (dict {"count","next","results"}) or unpaginated (list) の
        # どちらも返し得るため、dict の場合は next を辿って全 page 集める
        preds_list: list[dict] = []
        next_url: str | None = f"{LS_BASE}/api/predictions"
        params_pred: dict | None = {"project": project_id, "page_size": 1000}
        while next_url:
            r2 = requests.get(next_url, headers=HEADERS, params=params_pred)
            r2.raise_for_status()
            data = r2.json()
            if isinstance(data, dict):
                preds_list.extend(data.get("results", []))
                next_url = data.get("next")
                params_pred = None  # next URL 側にクエリ埋め込み済
            else:
                preds_list.extend(data)
                next_url = None
        preds_by_id = {p["id"]: p for p in preds_list}
        # pred_id lookup miss を stderr で明示 (silent drop 防止)
        missing: list[int] = []
        for task in tasks:
            pred_ids = task.get("predictions", [])
            resolved = []
            for pid in pred_ids:
                if pid in preds_by_id:
                    resolved.append(preds_by_id[pid])
                else:
                    missing.append(pid)
            task["predictions"] = resolved
        if missing:
            print(
                f"[warn] {len(missing)} pred_ids not resolved (dropped from fallback): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
                file=sys.stderr,
            )

    return tasks


# ------------------------------------------------------------
# HF push
# ------------------------------------------------------------
def ensure_dest_repo(repo_id: str, token: str | None) -> None:
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=token,
    )


def push_to_hf(
    ls_export_files: dict[Path, str],
    yolo_txt_files: dict[Path, str],
    repo_id: str,
    token: str | None,
) -> None:
    """batch commit で ls_exports/*.json + yolo_obb/labels/*.txt を push。

    Args:
        ls_export_files: {local_path: remote_relpath (under repo)}
        yolo_txt_files: {local_path: remote_relpath}
    """
    api = HfApi(token=token)
    ops = []
    for local, remote in {**ls_export_files, **yolo_txt_files}.items():
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
    if not ops:
        print("[hf-up] nothing to push")
        return
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=f"push {len(ls_export_files)} ls_exports + {len(yolo_txt_files)} yolo_obb txts",
        token=token,
    )
    print(f"[hf-up] pushed {len(ops)} files → {repo_id}")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def resolve_eps(args: argparse.Namespace) -> list[int]:
    if args.ep is not None:
        return [args.ep]
    if args.eps:
        return sorted({int(x.strip()) for x in args.eps.split(",") if x.strip()})
    raise SystemExit("must specify --ep or --eps")


def main() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ep", type=int, help="単一 ep 番号 (e.g. 60)")
    g.add_argument("--eps", type=str, help="複数 ep, comma-separated (e.g. '60,62')")
    p.add_argument(
        "--project-id",
        type=int,
        default=None,
        help=(
            "LS project id (ep-based title lookup を bypass)。"
            "custom title の project (v2 semi-supervised 等) 用。--ep 単一時のみ有効。"
        ),
    )
    p.add_argument(
        "--no-hf-upload",
        dest="hf_upload",
        action="store_false",
        help="HF push を skip (local export + 変換のみ)",
    )
    p.add_argument(
        "--hf-repo",
        default=None,
        help="push 先の HF dataset repo (指定したときだけ push。既定なし、Issue #190)",
    )
    p.add_argument(
        "--fallback-to-predictions",
        action="store_true",
        help=(
            "task の annotations が空の時、predictions フィールド (prelabel 結果) を "
            "fallback として annotation 扱い。user 未 review frame も prelabel のまま採用。"
        ),
    )
    p.set_defaults(hf_upload=True)
    args = p.parse_args()
    if args.hf_upload and not args.hf_repo:
        print("[hf] --hf-repo が無いので HF には push しない (local export + 変換のみ)")
        args.hf_upload = False

    eps = resolve_eps(args)
    if args.project_id is not None and len(eps) != 1:
        raise SystemExit("--project-id は --ep 単一時のみ有効 (--eps 複数指定と併用不可)")
    print(f"[push] target eps: {eps}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    token = None  # dotenv 経由

    if args.hf_upload:
        print(f"[hf] ensuring {args.hf_repo}")
        ensure_dest_repo(args.hf_repo, token)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_ls_files: dict[Path, str] = {}
    all_yolo_files: dict[Path, str] = {}

    for ep in eps:
        print(f"=== ep {ep:04d} ===")

        # 1. LS project 探す (--project-id 指定時はそちらを優先)
        if args.project_id is not None:
            pid = args.project_id
            print(f"  project_id={pid} (from --project-id)")
        else:
            pid = find_project_by_ep(ep)
            if pid is None:
                print(f"  [skip] no LS project for ep {ep:04d}")
                continue
            print(f"  project_id={pid}")

        # 2. LS export fetch
        # fallback_to_predictions=True の時は annotation なし tasks も取得 (prelabel を採用するため)
        export = fetch_ls_export(pid, download_all_tasks=args.fallback_to_predictions)
        if not export:
            print(f"  [skip] LS export empty")
            continue
        ann_count = sum(len(t.get("annotations", [])) for t in export)
        box_count = sum(
            len(a.get("result", []))
            for t in export
            for a in t.get("annotations", [])
        )
        print(f"  tasks={len(export)} annotations={ann_count} boxes={box_count}")

        # 3. LS raw JSON 保存 (timestamped)
        ls_json_path = EXPORT_DIR / f"ep{ep:04d}_{timestamp}.json"
        with open(ls_json_path, "w") as f:
            json.dump(export, f)
        all_ls_files[ls_json_path] = f"ls_exports/{ls_json_path.name}"

        # 4. YOLO OBB txt 変換 (hierarchical: {skill}/ep{XX}/frame_YY.txt)
        parsed = parse_ls_export(ls_json_path, fallback_to_predictions=args.fallback_to_predictions)
        n_txt = write_yolo_obb_txts(parsed, YOLO_OUT_DIR)
        print(f"  wrote {n_txt} txt → {YOLO_OUT_DIR}")
        # HF は yolo_obb/labels/{rel_path}.txt に置く (parsed の key = rel_path)
        for rel_path in parsed:
            local_txt = YOLO_OUT_DIR / f"{rel_path}.txt"
            all_yolo_files[local_txt] = f"yolo_obb/labels/{rel_path}.txt"

    # 5. HF push (batch)
    if args.hf_upload:
        push_to_hf(all_ls_files, all_yolo_files, args.hf_repo, token)

    print(f"[done] ls_exports={len(all_ls_files)} yolo_txt={len(all_yolo_files)}")


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
