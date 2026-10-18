"""YOLO OBB precompute cache を skill-merged HF repo に追記 push (Issue #122 D-1')。

skill-merged dataset repo (Team-RAMEN/..._<skill>_merged_v1) の `obb_yolo/<hash>/`
subdir に precompute cache 一式を upload。dataset 本体 (data/, videos/, meta/) には
触らず、YOLO ckpt 差替時は別 hash dir を追加するだけで済む。

# Usage

    python data/yolo_obb/scripts/upload_yolo_obb_to_hf.py \\
        --cache-dir outputs/yolo_obb_cache/<yolo_ckpt_hash>/ \\
        --repo-id Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base

    # path-in-repo は manifest から auto (obb_yolo/<hash>)、override も可
    python ... --path-in-repo custom/path

# 前提

- cache dir 直下に manifest.json 必須 (precompute_yolo_obb.py が書く)
- HF_TOKEN が set 済 + repo write 権限あり (Team-RAMEN org)
- 対象 repo は事前に prepare_merged_skill_dataset.sh で作成済 (skill-merged repo)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# lazy: huggingface_hub は model/yolo_obb pixi env / subtask_policy_training venv のみ install

CACHE_MANIFEST_NAME = "manifest.json"
CACHE_SCHEMA_VERSION = "team_ramen_yolo_obb_cache_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="local precompute cache dir (manifest.json + per-mp4 parquet 含む)",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="skill-merged HF repo id (Team-RAMEN/..._<skill>_merged_v1)",
    )
    p.add_argument(
        "--path-in-repo",
        default=None,
        help="repo 内 subdir path、default = 'obb_yolo/<hash>' (manifest から自動)",
    )
    p.add_argument(
        "--branch",
        default=None,
        help="target branch (default: main)",
    )
    p.add_argument(
        "--commit-message",
        default=None,
        help="commit message (default = 'Add YOLO OBB precompute (hash=..., ckpt=...)')",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="upload せず plan のみ表示",
    )
    return p.parse_args(argv)


def load_manifest(cache_dir: Path) -> dict:
    """cache dir 直下の manifest.json を read + minimum validation。"""
    p = cache_dir / CACHE_MANIFEST_NAME
    if not p.is_file():
        raise FileNotFoundError(
            f"manifest not found: {p} "
            f"(precompute_yolo_obb.py を先に走らせて cache 生成する)"
        )
    manifest = json.loads(p.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema_version: "
            f"got {manifest.get('schema_version')!r}, expected {CACHE_SCHEMA_VERSION!r}"
        )
    for required in ("yolo_ckpt_hash", "yolo_ckpt_ref", "cameras"):
        if required not in manifest:
            raise ValueError(f"manifest missing required field: {required}")
    return manifest


def resolve_path_in_repo(manifest: dict, override: str | None) -> str:
    """path_in_repo を default 'obb_yolo/<hash>' or user override で決める。"""
    if override is not None:
        return override
    return f"obb_yolo/{manifest['yolo_ckpt_hash']}"


def build_commit_message(manifest: dict, override: str | None) -> str:
    if override is not None:
        return override
    return (
        f"Add YOLO OBB precompute (hash={manifest['yolo_ckpt_hash']}, "
        f"ckpt={manifest['yolo_ckpt_ref']}, cameras={manifest['cameras']})"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.cache_dir)
    path_in_repo = resolve_path_in_repo(manifest, args.path_in_repo)
    commit_msg = build_commit_message(manifest, args.commit_message)

    print(f"[plan] repo:          {args.repo_id}")
    print(f"[plan] cache dir:     {args.cache_dir}")
    print(f"[plan] path in repo:  {path_in_repo}")
    print(f"[plan] branch:        {args.branch or 'main'}")
    print(f"[plan] ckpt hash:     {manifest['yolo_ckpt_hash']}")
    print(f"[plan] ckpt ref:      {manifest['yolo_ckpt_ref']}")
    print(f"[plan] cameras:       {manifest['cameras']}")
    print(f"[plan] commit msg:    {commit_msg}")

    if args.dry_run:
        print("[dry-run] upload skipped")
        return 0

    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.upload_folder(
        folder_path=str(args.cache_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        revision=args.branch,
        commit_message=commit_msg,
    )
    branch = args.branch or "main"
    print(
        f"[done] uploaded to "
        f"https://huggingface.co/datasets/{args.repo_id}/tree/{branch}/{path_in_repo}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
