"""
hf_sync.py — Team-RAMEN/ramen-g1-ikea-assembly-curation dataset repo への manual sync。

UI button と CLI (`pixi run sync-hf`) から呼ばれる。preview → confirm → push の
modal 対応 UI からは compute_preview → sync_to_hf の 2 段。CLI は --dry-run と
--commit-msg 対応。

Upload 対象: labels.parquet + labels.jsonl + failure_categories.json +
labels_backup/ 内 全 file。1 commit で atomic upload (create_commit)。
NOT synced: metadata.json (個人 state), videos_cache/ (heavy)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Fix #4 for PR #83 review: HF API 呼び出しに 60s timeout を設定 (HF hub 遅延で
# endpoint indefinite hang を防ぐ)。os.environ.setdefault で user override を許容。
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError


DEFAULT_REPO_ID = "Team-RAMEN/ramen-g1-ikea-assembly-curation"
# .env は iros_2026_ramen/ 直下 (data/curation_tool/ui/ の 3 階層上)
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

SYNC_FILES = ("labels.parquet", "labels.jsonl", "failure_categories.json")
SYNC_BACKUP_DIR = "labels_backup"

_JST = timezone(timedelta(hours=9))


@dataclass
class SyncPreview:
    """UI modal に表示する sync 内容 preview。"""

    remote_exists: bool
    labels_new_rows: int
    labels_total_rows_local: int
    jsonl_new_lines: int
    categories_added: list[dict] = field(default_factory=list)
    files_to_upload: list[str] = field(default_factory=list)


# --- Token 解決 ---

def resolve_token(explicit: Optional[str] = None) -> str:
    """HF_TOKEN を (arg → env → .env → error) 順で解決。"""
    if explicit:
        return explicit
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    if DEFAULT_ENV_PATH.exists():
        # python-dotenv は uvicorn[standard] extras で入る
        from dotenv import load_dotenv

        load_dotenv(DEFAULT_ENV_PATH)
        tok = os.environ.get("HF_TOKEN")
        if tok:
            return tok
    raise RuntimeError(
        f"HF_TOKEN not found (checked arg, env var, .env at {DEFAULT_ENV_PATH})"
    )


# --- Local file helpers ---

def _list_backup_files(workspace: Path) -> list[Path]:
    backup_dir = workspace / SYNC_BACKUP_DIR
    if not backup_dir.exists():
        return []
    return sorted(p for p in backup_dir.iterdir() if p.is_file())


def _count_parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    import pandas as pd

    return len(pd.read_parquet(path))


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _local_categories(workspace: Path) -> list[dict]:
    path = workspace / "failure_categories.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("categories", []))


# --- Remote fetch (test で monkeypatch する level) ---

def _fetch_remote_labels_row_count(
    repo_id: str, token: str
) -> Optional[int]:
    """HF の labels.parquet を download して row 数を返す。存在無しは None。"""
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="labels.parquet",
            repo_type="dataset",
            token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None

    import pandas as pd

    return len(pd.read_parquet(path))


def _fetch_remote_categories(
    repo_id: str, token: str
) -> Optional[list[dict]]:
    """HF の failure_categories.json を download して list 返す。無しは None。"""
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="failure_categories.json",
            repo_type="dataset",
            token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("categories", []))


# --- Preview ---

def compute_preview(
    workspace: Path,
    repo_id: str = DEFAULT_REPO_ID,
    token: Optional[str] = None,
) -> SyncPreview:
    """HF 側と workspace を比較して preview を組み立てる。"""
    tok = resolve_token(token)

    local_labels_rows = _count_parquet_rows(workspace / "labels.parquet")
    local_jsonl_lines = _count_jsonl_lines(workspace / "labels.jsonl")
    local_cats = _local_categories(workspace)

    remote_labels_rows = _fetch_remote_labels_row_count(repo_id, tok)
    remote_cats = _fetch_remote_categories(repo_id, tok)

    remote_exists = remote_labels_rows is not None or remote_cats is not None

    labels_new = local_labels_rows - (remote_labels_rows or 0)
    # jsonl は line 単位 diff を Phase B では取らず、local total を new と仮定表示
    jsonl_new = local_jsonl_lines

    remote_keys = {c["key"] for c in (remote_cats or [])}
    categories_added = [c for c in local_cats if c["key"] not in remote_keys]

    files_to_upload: list[str] = []
    for name in SYNC_FILES:
        if (workspace / name).exists():
            files_to_upload.append(name)
    for backup in _list_backup_files(workspace):
        files_to_upload.append(f"{SYNC_BACKUP_DIR}/{backup.name}")

    return SyncPreview(
        remote_exists=remote_exists,
        labels_new_rows=max(0, labels_new),
        labels_total_rows_local=local_labels_rows,
        jsonl_new_lines=jsonl_new,
        categories_added=categories_added,
        files_to_upload=files_to_upload,
    )


# --- Push ---

def _default_commit_message() -> str:
    now = datetime.now(_JST).isoformat(timespec="seconds")
    return f"curation: sync at {now}"


def sync_to_hf(
    workspace: Path,
    repo_id: str = DEFAULT_REPO_ID,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> str:
    """workspace の SYNC_FILES + labels_backup/ を 1 commit で upload。commit URL 返す。"""
    tok = resolve_token(token)
    api = HfApi(token=tok)

    operations: list[CommitOperationAdd] = []
    for name in SYNC_FILES:
        local = workspace / name
        if local.exists():
            operations.append(
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(local))
            )
    for backup in _list_backup_files(workspace):
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{SYNC_BACKUP_DIR}/{backup.name}",
                path_or_fileobj=str(backup),
            )
        )

    if not operations:
        raise RuntimeError(
            f"no files to upload in {workspace} "
            f"(nothing matches SYNC_FILES / {SYNC_BACKUP_DIR}/)"
        )

    msg = commit_message or _default_commit_message()
    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=msg,
    )
    return commit_info.commit_url


# --- CLI ---

def _print_preview(preview: SyncPreview, repo_id: str, msg: str) -> None:
    print(f"=== HF sync preview: {repo_id} ===")
    print(f"  remote_exists: {preview.remote_exists}")
    print(
        f"  labels.parquet: {preview.labels_new_rows} new rows "
        f"(local total: {preview.labels_total_rows_local})"
    )
    print(f"  labels.jsonl: {preview.jsonl_new_lines} lines (local total)")
    print(f"  categories added: {len(preview.categories_added)}")
    for c in preview.categories_added:
        print(f"    - {c['key']}: {c.get('label', '')}")
    print(f"  files to upload ({len(preview.files_to_upload)}):")
    for f in preview.files_to_upload:
        print(f"    - {f}")
    print(f"  commit message: {msg}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ui.hf_sync",
        description="Sync curation workspace to Hugging Face dataset repo",
    )
    parser.add_argument(
        "--workspace", type=Path, default=None,
        help="curation workspace path (default: this module's ../workspace)",
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--commit-msg", default=None,
        help="override commit message (default: 'curation: sync at <JST ISO8601>')",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="preview only, no upload",
    )
    args = parser.parse_args(argv)

    from ui.labels_io import workspace_dir

    ws = workspace_dir(args.workspace)
    msg = args.commit_msg or _default_commit_message()

    preview = compute_preview(ws, args.repo_id)
    _print_preview(preview, args.repo_id, msg)

    if args.dry_run:
        print("--dry-run, exit without upload")
        return 0

    if not preview.files_to_upload:
        print("nothing to upload, exit")
        return 0

    print(f"→ uploading to {args.repo_id} ...")
    commit_url = sync_to_hf(ws, args.repo_id, commit_message=msg)
    print(f"✓ commit: {commit_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
