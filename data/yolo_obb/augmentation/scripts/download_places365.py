"""Places365 val_large tarball から指定 category subset を抽出。

Issue #77 Phase 1-1。背景 copy-paste augmentation 用の背景プール準備。

処理:
1. MIT Places365 filelist tarball を DL & cache (~65MB、初回のみ)
   → `places365_val.txt` (filename → category index) と
     `categories_places365.txt` (category name → index) を得る
2. `val_large.tar` (~2.27GB、初回のみ) を DL
   ※個別画像 URL が公開されていない (2026-07 検証)、tar 単位で落とすしかない
3. 指定 category に該当する val filename を filelist から特定
4. tar から該当画像だけ抽出 (val は 1 category = 100 枚固定)
5. seed 固定で N 枚 sample → {out}/{category}/Places365_{category}_NNNNN.jpg
6. tar は cache に残す (次回 category 追加時に再利用、削除は手動)

Usage (root workspace で python3 直、依存追加なし):
    python3 data/yolo_obb/augmentation/scripts/download_places365.py \\
        --categories office living_room \\
        --n-per-cat 25 \\
        --seed 42

注意:
- stdlib のみ (urllib/tarfile) で依存追加なし
- tar cache は 2.27GB あり、削除する場合は `--purge-tar` を後日追加検討
- 落としたファイルは .gitignore で除外前提 (README 記載)
"""

from __future__ import annotations

import argparse
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

FILELIST_URL = "https://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
VAL_TAR_URL = "https://data.csail.mit.edu/places/places365/val_large.tar"
# MIT の Content-Length 実測値。partial DL 検出用に厳密比較する。
EXPECTED_VAL_TAR_SIZE = 2270320640

DEFAULT_OUT = Path("data/yolo_obb/augmentation/backgrounds/places365")
DEFAULT_CACHE = Path("data/yolo_obb/augmentation/.cache")


def _download_with_progress(url: str, dst: Path, label: str) -> None:
    """簡易 progress 表示付き DL。"""
    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = min(100, block_num * block_size * 100 // total_size)
        mb_done = block_num * block_size / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        sys.stderr.write(f"\r  [{label}] {pct:3d}% ({mb_done:.1f}/{mb_total:.1f} MB)")
        sys.stderr.flush()
    print(f"[{label}] downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, dst, reporthook=_hook)
    sys.stderr.write("\n")


def ensure_filelist(cache_dir: Path) -> tuple[Path, Path]:
    """filelist tarball を DL & 展開。val txt と categories txt の path を返す。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "filelist_places365-standard.tar"
    val_txt = cache_dir / "places365_val.txt"
    cat_txt = cache_dir / "categories_places365.txt"
    if val_txt.exists() and cat_txt.exists():
        return val_txt, cat_txt
    if not tar_path.exists():
        _download_with_progress(FILELIST_URL, tar_path, "filelist")
    print("[filelist] extracting txt files ...", file=sys.stderr)
    # filter="data" は PEP 706 対応。Python 3.14+ で default になるが明示指定で
    # DeprecationWarning 回避 + path traversal (CVE-2007-4559) 防止。
    with tarfile.open(tar_path) as tf:
        tf.extract("places365_val.txt", cache_dir, filter="data")
        tf.extract("categories_places365.txt", cache_dir, filter="data")
    return val_txt, cat_txt


def ensure_val_tar(cache_dir: Path) -> Path:
    """val_large.tar を DL (2.27GB、初回のみ)。partial DL cache は削除して再取得。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "val_large.tar"
    if tar_path.exists():
        size = tar_path.stat().st_size
        if size == EXPECTED_VAL_TAR_SIZE:
            print(f"[val_tar] cache hit: {tar_path} ({size / 1024 / 1024:.1f} MB)", file=sys.stderr)
            return tar_path
        # size 不一致 = partial DL、そのまま extract すると tar 破損 error で分かりづらいので削除
        print(f"[val_tar] partial/corrupt cache ({size} bytes, expected {EXPECTED_VAL_TAR_SIZE}), redownloading",
              file=sys.stderr)
        tar_path.unlink()
    _download_with_progress(VAL_TAR_URL, tar_path, "val_tar")
    return tar_path


def load_category_index(cat_txt: Path) -> dict[str, int]:
    """categories_places365.txt を読み、category name → index の dict を返す。

    フォーマット例: `/o/office 244` → {"office": 244}
    """
    index: dict[str, int] = {}
    with cat_txt.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            # `/o/office` → `office`
            name = parts[0].strip("/").split("/", 1)[-1]
            index[name] = int(parts[1])
    return index


def val_files_for_category(val_txt: Path, cat_index: int) -> list[str]:
    """指定 category index に属する val filename を全部返す。"""
    files: list[str] = []
    with val_txt.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and int(parts[1]) == cat_index:
                files.append(parts[0])
    return files


def extract_images(tar_path: Path, targets: dict[str, list[str]], out_root: Path) -> tuple[int, int]:
    """val_large.tar から targets の image だけ抽出。

    targets: {category_name: [filename, ...]} (filename は "Places365_val_00000238.jpg" 形式)
    """
    # filename → (category, dst_index) の逆引き dict
    lookup: dict[str, tuple[str, int]] = {}
    for cat, files in targets.items():
        for i, fn in enumerate(files, 1):
            lookup[fn] = (cat, i)

    ok = 0
    fail = 0
    seen = 0
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            # tar entry の path 例: "val_large/Places365_val_00000238.jpg"
            fn = Path(member.name).name
            hit = lookup.get(fn)
            if hit is None:
                continue
            cat, idx = hit
            dst = out_root / cat / f"Places365_{cat}_{idx:05d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                extracted = tf.extractfile(member)
                if extracted is None:
                    fail += 1
                    continue
                dst.write_bytes(extracted.read())
                ok += 1
                seen += 1
                if seen % 20 == 0:
                    print(f"  extracted {seen}/{sum(len(v) for v in targets.values())}", file=sys.stderr)
            except Exception as e:
                print(f"  FAIL extract {fn}: {e}", file=sys.stderr)
                fail += 1
    return ok, fail


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=["office", "living_room"],
                   help="Places365 category names")
    p.add_argument("--n-per-cat", type=int, default=25, help="1 category あたり保存枚数")
    p.add_argument("--seed", type=int, default=42, help="sampling seed")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="保存 root")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="tarball cache")
    args = p.parse_args()

    val_txt, cat_txt = ensure_filelist(args.cache)
    cat_index_map = load_category_index(cat_txt)

    # category → sampled filenames
    targets: dict[str, list[str]] = {}
    rng = random.Random(args.seed)
    for cat in args.categories:
        if cat not in cat_index_map:
            print(f"[{cat}] SKIP: category 名 typo? (categories.txt に無い)", file=sys.stderr)
            continue
        all_files = val_files_for_category(val_txt, cat_index_map[cat])
        n = min(args.n_per_cat, len(all_files))
        sampled = rng.sample(all_files, n) if n < len(all_files) else all_files
        targets[cat] = sampled
        print(f"[{cat}] index={cat_index_map[cat]} pool={len(all_files)} sample={n}", file=sys.stderr)

    if not targets:
        print("[error] 有効な category が無かった", file=sys.stderr)
        return 1

    tar_path = ensure_val_tar(args.cache)
    print(f"[extract] scanning tar for {sum(len(v) for v in targets.values())} images ...", file=sys.stderr)
    ok, fail = extract_images(tar_path, targets, args.out)
    print(f"\n[done] extracted ok={ok} fail={fail} out={args.out}", file=sys.stderr)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
