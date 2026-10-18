"""frame cache の staging dir を HF dataset repo に同期する (Issue #139)。

staging dir は repo の path をそのまま写した構成
(`obb_yolo/<hash>/...`、`frame_cache_tars/*.tar`、`frame_cache_v2_tars/{overlay,token,common}/*.tar`)。

# subcommand

- upload: `HfApi.upload_large_folder` (途中から再開可能、上げ済みは skip) で staging を上げる。
  `--until-flag` を渡すと、flag が立つまで `--interval` 秒ごとに繰り返す
  (pack と並行して、焼き上がった tar から順に上げる)
- verify: staging の全 file が repo に同じ size で在るか確認
- cleanup: 管理対象 prefix 配下で staging に無い file / dir (旧版・重複) を 1 commit で削除。
  既定は dry-run (削除対象を表示するだけ)、`--execute` で実行。verify が通らないと実行しない

# Usage

    python sync_frame_cache_to_hf.py upload --repo-id <repo> --staging <dir> \\
        --until-flag <work>/.pack.done --interval 300 --num-workers 4
    python sync_frame_cache_to_hf.py verify --repo-id <repo> --staging <dir>
    python sync_frame_cache_to_hf.py cleanup --repo-id <repo> --staging <dir> [--execute]

HF token は環境変数 HF_TOKEN から読む (huggingface_hub の既定動作)。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# upload_large_folder が staging 内に置く resume 用 metadata と、pack 途中の tmp は上げない
IGNORE_PATTERNS = ["*.tmp", "**/*.tmp", ".cache/**"]
DEFAULT_MANAGED_PREFIXES = ("obb_yolo", "frame_cache_tars", "frame_cache_v2_tars", "frame_cache_v2")
PATHS_INFO_BATCH = 200


def list_local_files(staging: Path) -> dict[str, int]:
    """staging 配下の file (repo 相対 posix path → size)。resume metadata と tmp は除く。"""
    files: dict[str, int] = {}
    for p in sorted(staging.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(staging).as_posix()
        if rel.startswith(".cache/") or rel.endswith(".tmp"):
            continue
        files[rel] = p.stat().st_size
    return files


def _api():
    from huggingface_hub import HfApi  # noqa: PLC0415

    return HfApi()


def upload(
    api: Any,
    *,
    repo_id: str,
    staging: Path,
    num_workers: int,
    until_flag: Path | None,
    interval: float,
) -> None:
    """staging を upload。until_flag があれば、flag が立った後の 1 回まで繰り返す。"""
    while True:
        # flag を先に見る: flag が立つ前に書かれた tar は必ずこの回か次の回で上がる
        last_round = until_flag is None or until_flag.exists()
        if list_local_files(staging):
            api.upload_large_folder(
                repo_id=repo_id,
                folder_path=staging,
                repo_type="dataset",
                num_workers=num_workers,
                ignore_patterns=IGNORE_PATTERNS,
                print_report=False,
            )
        if last_round:
            return
        time.sleep(interval)


def verify(api: Any, *, repo_id: str, staging: Path) -> list[str]:
    """staging の file が repo に同じ size で在るか。問題のある path の説明 list を返す。"""
    local = list_local_files(staging)
    paths = sorted(local)
    remote: dict[str, int] = {}
    for start in range(0, len(paths), PATHS_INFO_BATCH):
        batch = paths[start : start + PATHS_INFO_BATCH]
        for info in api.get_paths_info(repo_id, paths=batch, repo_type="dataset"):
            size = getattr(info, "size", None)
            if size is not None:
                remote[info.path] = int(size)
    problems = []
    for path in paths:
        if path not in remote:
            problems.append(f"missing on hub: {path}")
        elif remote[path] != local[path]:
            problems.append(f"size mismatch: {path} (local {local[path]} / hub {remote[path]})")
    return problems


def _is_folder(entry: Any) -> bool:
    from huggingface_hub.hf_api import RepoFolder  # noqa: PLC0415

    return isinstance(entry, RepoFolder)


def compute_stale(
    api: Any,
    *,
    repo_id: str,
    staged_paths: set[str],
    managed_prefixes: tuple[str, ...],
    is_folder=None,
) -> list[tuple[str, bool]]:
    """管理対象 prefix 配下で staging に無い (path, is_folder) を列挙。

    staging 側に 1 file も無い dir は中を列挙せず dir ごと削除対象にする
    (未 tar の旧 frame_cache_v2/ のような大量 file を API で列挙しないため)。
    """
    is_folder = is_folder or _is_folder
    staged_dirs: set[str] = set()
    for path in staged_paths:
        parts = path.split("/")
        for i in range(1, len(parts)):
            staged_dirs.add("/".join(parts[:i]))

    stale: list[tuple[str, bool]] = []

    def walk(prefix: str) -> None:
        for entry in api.list_repo_tree(
            repo_id, path_in_repo=prefix, repo_type="dataset", recursive=False
        ):
            if is_folder(entry):
                if entry.path in staged_dirs:
                    walk(entry.path)
                else:
                    stale.append((entry.path, True))
            elif entry.path not in staged_paths:
                stale.append((entry.path, False))

    for entry in api.list_repo_tree(repo_id, repo_type="dataset", recursive=False):
        if entry.path not in managed_prefixes:
            continue
        if not is_folder(entry):
            stale.append((entry.path, False))
        elif entry.path in staged_dirs:
            walk(entry.path)
        else:
            stale.append((entry.path, True))
    return sorted(stale)


def cleanup(
    api: Any,
    *,
    repo_id: str,
    staging: Path,
    managed_prefixes: tuple[str, ...],
    execute: bool,
) -> int:
    problems = verify(api, repo_id=repo_id, staging=staging)
    if problems:
        print(f"ERROR: verify が通らないので削除しない ({len(problems)} 件):", file=sys.stderr)
        for line in problems[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1
    stale = compute_stale(
        api,
        repo_id=repo_id,
        staged_paths=set(list_local_files(staging)),
        managed_prefixes=managed_prefixes,
    )
    print(f"[cleanup] {repo_id}: 削除対象 {len(stale)} 件")
    for path, folder in stale:
        print(f"  {'dir ' if folder else 'file'} {path}")
    if not stale or not execute:
        if stale:
            print("[cleanup] dry-run (削除するには --execute)")
        return 0
    from huggingface_hub import CommitOperationDelete  # noqa: PLC0415

    ops = [CommitOperationDelete(path_in_repo=path, is_folder=folder) for path, folder in stale]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message="Issue #139: remove superseded frame cache / OBB files",
    )
    print(f"[cleanup] deleted {len(ops)} entries")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("upload", "verify", "cleanup"):
        sp = sub.add_parser(name)
        sp.add_argument("--repo-id", required=True)
        sp.add_argument("--staging", type=Path, required=True)
        if name == "upload":
            sp.add_argument("--num-workers", type=int, default=4)
            sp.add_argument("--until-flag", type=Path, default=None)
            sp.add_argument("--interval", type=float, default=300.0)
        if name == "cleanup":
            sp.add_argument(
                "--managed-prefixes",
                default=",".join(DEFAULT_MANAGED_PREFIXES),
                help="削除判定の対象にする repo 直下の prefix (comma 区切り)",
            )
            sp.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if not args.staging.is_dir():
        print(f"ERROR: staging not found: {args.staging}", file=sys.stderr)
        return 2

    api = _api()
    if args.cmd == "upload":
        upload(
            api,
            repo_id=args.repo_id,
            staging=args.staging,
            num_workers=args.num_workers,
            until_flag=args.until_flag,
            interval=args.interval,
        )
        return 0
    if args.cmd == "verify":
        problems = verify(api, repo_id=args.repo_id, staging=args.staging)
        n = len(list_local_files(args.staging))
        if problems:
            print(f"[verify] {args.repo_id}: NG {len(problems)} / {n} files", file=sys.stderr)
            for line in problems[:20]:
                print(f"  {line}", file=sys.stderr)
            return 1
        print(f"[verify] {args.repo_id}: OK ({n} files)")
        return 0
    prefixes = tuple(x.strip() for x in args.managed_prefixes.split(",") if x.strip())
    return cleanup(
        api,
        repo_id=args.repo_id,
        staging=args.staging,
        managed_prefixes=prefixes,
        execute=args.execute,
    )


if __name__ == "__main__":
    sys.exit(main())
