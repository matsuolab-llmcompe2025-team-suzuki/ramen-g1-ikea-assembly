"""HF に学習済 YOLO-OBB weight (best.pt) を push する CLI (Issue #58)。

Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb に best.pt を配置し、
新規 team member が hf_hub_download で pull できるようにする。既存 frames dataset
(`_Hara_skillsplitframes_upperpolicy`) と対応する model repo。

Usage:
    pixi run -e train python -m model.yolo_obb.scripts.upload_weight \\
        --run-name m_lowaug_v3

    # dry-run (upload 前に内容確認)
    pixi run -e train python -m model.yolo_obb.scripts.upload_weight \\
        --run-name m_lowaug_v3 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from huggingface_hub import HfApi

RUNS_ROOT = Path("model/yolo_obb/runs")
DEFAULT_REPO_ID = "Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb"


def _default_upload_name() -> str:
    """今日の日付 (UTC) を suffix にした HF 側 file 名 default。

    複数版を HF 上で共存させるため日付付き (`best.pt` 固定だと上書き)。
    別 timezone や別日 upload したい場合は `--upload-name` で明示 override。
    """
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"best_{date}.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="model/yolo_obb/runs/<run-name>/weights/best.pt を upload",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace model repo id (default: {DEFAULT_REPO_ID})",
    )
    p.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="HF commit message (省略 = 'upload <run-name> <upload-name>')",
    )
    p.add_argument(
        "--upload-name",
        type=str,
        default=None,
        help=(
            "HF 側の file 名 (default: 今日の日付付き 'best_YYYYMMDD.pt')。"
            " 上書きしたくない場合 (旧 v3 と共存) の日付付き衝突回避 default"
        ),
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="public repo として作成 (default: private)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="実行内容を print するだけで upload しない",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not found in env / .env (root README の環境構築参照)")

    weight_path = RUNS_ROOT / args.run_name / "weights" / "best.pt"
    if not weight_path.exists():
        sys.exit(f"weight not found: {weight_path}")

    upload_name = args.upload_name or _default_upload_name()

    # HF repo 側 path: runs/<run-name>/weights/<upload-name> の scheme。
    # 日付付き file 名 default (best_YYYYMMDD.pt) なので同 run_name で複数版を
    # 共存できる。旧 upload (best.pt) との後方互換は既存 file を触らない形で確保。
    path_in_repo = f"runs/{args.run_name}/weights/{upload_name}"

    private = not args.public
    print(f"repo_id: {args.repo_id}")
    print(f"local weight: {weight_path} ({weight_path.stat().st_size / 1e6:.1f} MB)")
    print(f"upload_name: {upload_name}")
    print(f"path_in_repo: {path_in_repo}")
    print(f"private: {private}")
    print(f"dry_run: {args.dry_run}")

    if args.dry_run:
        print("[dry-run] skip upload")
        return 0

    api = HfApi(token=token)
    repo_url = api.create_repo(
        args.repo_id, repo_type="model", private=private, exist_ok=True
    )
    print(f"repo ready: {repo_url}")

    commit_msg = args.commit_message or f"upload {args.run_name} {upload_name}"
    commit = api.upload_file(
        path_or_fileobj=str(weight_path),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=commit_msg,
    )
    print(f"commit_url: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
