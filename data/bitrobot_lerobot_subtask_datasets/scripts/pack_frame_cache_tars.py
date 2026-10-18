"""frame cache の mp4 dir を HF upload 用に tar 化する (Issue #139)。

1 tar = 1 mp4 dir (`<cam>/<chunk>/<file>/`)。命名は既存 `frame_cache_tars/` と同じ
`<cam>_<chunk>_<file>.tar` で、member は `<file>/<jpg>`。展開は同梱の
`restore_frame_cache_tars.sh` (HF 上では `restore.sh`)。

# mode

- v0: `_frame_count.txt` がある dir の raw jpg `frame_{0..n-1}.jpg` を詰める
- variants: `_bake_done.json` がある dir の `frame_{i}_v{1..V-1}.jpg` を詰める。
  `--watch` なら bake 中に焼き上がった mp4 から順に詰め、`--done-flag` が立って
  全 mp4 を詰め終えたら終了する

詰める走査の中で、欠けている jpg と 0 byte の jpg を検出して止める
(Issue #139 前の frame_cache_tars に 0 byte jpg が混入していた再発防止)。
書き込みは `<name>.tar.tmp` → rename なので、uploader は完成した tar だけを見る。

# Usage

    # v0 (decode 完了後)
    python pack_frame_cache_tars.py --src-root <work>/v0 --dest-dir <staging>/frame_cache_tars \\
        --mode v0 --jobs 2

    # variants (bake と並行)
    python pack_frame_cache_tars.py --src-root <work>/v2/overlay \\
        --dest-dir <staging>/frame_cache_v2_tars/overlay --mode variants \\
        --cameras observation.images.cam_0 --expect-from <work>/v0 \\
        --watch --done-flag <work>/.bake.done --jobs 2
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

FRAME_COUNT_NAME = "_frame_count.txt"
BAKE_DONE_NAME = "_bake_done.json"
MODES = ("v0", "variants")


class PackError(RuntimeError):
    """tar 化の前提 (枚数 / 0 byte) が崩れている。"""


def list_mp4_dirs(root: Path, cameras: list[str] | None = None) -> list[Path]:
    """`<root>/<cam>/<chunk>/<file>` の root 相対 path を列挙。symlink の cam dir は除く。

    symlink は common root を指す共有 cam で、tar は common 側で作るため。
    """
    if not root.is_dir():
        return []
    rels: list[Path] = []
    for cam_dir in sorted(root.iterdir()):
        if not cam_dir.is_dir() or cam_dir.is_symlink():
            continue
        if cameras is not None and cam_dir.name not in cameras:
            continue
        for chunk_dir in sorted(p for p in cam_dir.iterdir() if p.is_dir()):
            for file_dir in sorted(p for p in chunk_dir.iterdir() if p.is_dir()):
                rels.append(file_dir.relative_to(root))
    return rels


def tar_name(rel: Path) -> str:
    """`<cam>/<chunk>/<file>` → `<cam>_<chunk>_<file>.tar`。"""
    cam, chunk, file_stem = rel.parts
    return f"{cam}_{chunk}_{file_stem}.tar"


def ready_marker(mode: str) -> str:
    return FRAME_COUNT_NAME if mode == "v0" else BAKE_DONE_NAME


def expected_members(src_dir: Path, mode: str) -> list[str]:
    """tar に入れる file 名 (marker を含む) を、marker の記録から組み立てる。"""
    if mode == "v0":
        n = int((src_dir / FRAME_COUNT_NAME).read_text(encoding="utf-8").strip())
        return [f"frame_{i:06d}.jpg" for i in range(n)] + [FRAME_COUNT_NAME]
    done = json.loads((src_dir / BAKE_DONE_NAME).read_text(encoding="utf-8"))
    frames = int(done["frames"])
    num_variants = int(done["num_variants"])
    names = [
        f"frame_{i:06d}_v{v}.jpg" for i in range(frames) for v in range(1, num_variants)
    ]
    return names + [BAKE_DONE_NAME]


def pack_one(src_root: Path, rel: Path, dest_dir: Path, mode: str) -> tuple[Path, int, int]:
    """1 mp4 dir を tar 化。Returns: (tar_path, n_files, n_bytes)。既に tar があれば skip。"""
    dest = dest_dir / tar_name(rel)
    if dest.exists():
        return dest, 0, 0
    src_dir = src_root / rel
    names = expected_members(src_dir, mode)
    file_stem = rel.parts[-1]
    tmp = dest.with_name(dest.name + ".tmp")
    dest_dir.mkdir(parents=True, exist_ok=True)
    n_bytes = 0
    try:
        with tarfile.open(tmp, "w", format=tarfile.GNU_FORMAT) as tf:
            for name in names:
                path = src_dir / name
                try:
                    size = path.stat().st_size
                except FileNotFoundError:
                    raise PackError(f"missing: {path}") from None
                if size == 0 and name.endswith(".jpg"):
                    raise PackError(f"0 byte jpg: {path}")
                tf.add(path, arcname=f"{file_stem}/{name}", recursive=False)
                n_bytes += size
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    return dest, len(names), n_bytes


def _expected_rels(src_root: Path, expect_from: Path | None, cameras: list[str] | None) -> list[Path]:
    """詰めるべき mp4 dir。expect_from (v0 root) があればそちらを正とする。"""
    return list_mp4_dirs(expect_from if expect_from is not None else src_root, cameras)


def run(
    *,
    src_root: Path,
    dest_dir: Path,
    mode: str,
    cameras: list[str] | None,
    expect_from: Path | None,
    watch: bool,
    done_flag: Path | None,
    jobs: int,
    poll_interval: float,
) -> int:
    """詰めるべき mp4 dir を全て tar 化する。watch 時は marker が揃うのを待ちながら詰める。"""
    expected = _expected_rels(src_root, expect_from, cameras)
    if not expected:
        print(f"ERROR: no mp4 dirs to pack under {expect_from or src_root}", file=sys.stderr)
        return 2
    marker = ready_marker(mode)
    pending = set(expected)
    total_files = total_bytes = 0
    t0 = time.monotonic()

    executor = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    running: dict[Path, Future] = {}
    try:
        while pending or running:
            # watch 中の完了判定は「flag を見てから marker を見る」順で行う
            # (flag 後に書かれた marker を取りこぼさない)
            finished_source = (not watch) or (done_flag is not None and done_flag.exists())
            for rel in sorted(pending):
                if rel in running or not (src_root / rel / marker).is_file():
                    continue
                if executor is None:
                    tar_path, n_files, n_bytes = pack_one(src_root, rel, dest_dir, mode)
                    pending.discard(rel)
                    total_files += n_files
                    total_bytes += n_bytes
                    print(f"[pack] {tar_path.name}: {n_files} files, {n_bytes / 1e9:.2f} GB", flush=True)
                else:
                    running[rel] = executor.submit(pack_one, src_root, rel, dest_dir, mode)
            for rel, fut in list(running.items()):
                if not fut.done():
                    continue
                tar_path, n_files, n_bytes = fut.result()  # PackError はここで伝播
                del running[rel]
                pending.discard(rel)
                total_files += n_files
                total_bytes += n_bytes
                print(f"[pack] {tar_path.name}: {n_files} files, {n_bytes / 1e9:.2f} GB", flush=True)
            if not pending and not running:
                break
            if finished_source and not running:
                not_ready = [
                    str(r) for r in sorted(pending) if not (src_root / r / marker).is_file()
                ]
                if not_ready:
                    print(
                        f"ERROR: {marker} が無い mp4 dir が {len(not_ready)} 個: {not_ready[:5]}",
                        file=sys.stderr,
                    )
                    return 3
            if running:
                wait(list(running.values()), timeout=poll_interval, return_when=FIRST_COMPLETED)
            elif watch:
                time.sleep(poll_interval)
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    elapsed = time.monotonic() - t0
    print(
        f"[pack] DONE {len(expected)} tars ({total_files:,} files, {total_bytes / 1e9:.1f} GB "
        f"newly packed) in {elapsed / 60:.1f} min → {dest_dir}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src-root", type=Path, required=True, help="tar 化する cache root")
    p.add_argument("--dest-dir", type=Path, required=True, help="tar の出力先 (upload staging)")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--cameras", default=None, help="対象 cam (comma 区切り)。未指定なら全 cam")
    p.add_argument(
        "--expect-from",
        type=Path,
        default=None,
        help="詰めるべき mp4 dir をこの root (v0) から数える。bake 前の未作成 dir も待てる",
    )
    p.add_argument("--watch", action="store_true", help="marker が揃うのを待ちながら詰める")
    p.add_argument(
        "--done-flag",
        type=Path,
        default=None,
        help="watch 時、この file が現れたら残りの marker を最終確認して終了する",
    )
    p.add_argument("--jobs", type=int, default=2, help="並列 tar 化 process 数 (<=1 で逐次)")
    p.add_argument("--poll-interval", type=float, default=30.0)
    args = p.parse_args(argv)
    if args.watch and args.done_flag is None:
        p.error("--watch には --done-flag が必要")
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()] if args.cameras else None
    try:
        return run(
            src_root=args.src_root,
            dest_dir=args.dest_dir,
            mode=args.mode,
            cameras=cameras,
            expect_from=args.expect_from,
            watch=args.watch,
            done_flag=args.done_flag,
            jobs=args.jobs,
            poll_interval=args.poll_interval,
        )
    except PackError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
