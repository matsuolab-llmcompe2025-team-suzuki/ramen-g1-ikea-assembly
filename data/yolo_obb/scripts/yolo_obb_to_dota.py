"""YOLO-OBB dataset (normalized 4-vertex) を DOTA v1.0 形式に変換。

Issue #108 v5 Phase 3b: RF-DETR OBB (PR #921) は DOTA loader しか持たないため、
既存 YOLO-OBB dataset を DOTA v1.0 layout に text 変換 + image copy する。data 内容は
同一 (同じ image + 同じ annotation)、format と layout だけ変換。

処理:
1. YOLO-OBB dataset (images/{split}/{skill}/{ep}/frame.png + labels/{split}/...) 読み込み
2. 追加 augment dir (v5_material_random 等) を train に merge (--src-augmentation 複数可)
3. 各 image を PIL で開いて size 確認、normalized → pixel 座標変換
4. class_id → class_name mapping (0=workspace, 1=leg, ...)
5. DOTA layout に出力:
     {out}/{split}/images/{img_stem}.png
     {out}/{split}/labelTxt/{img_stem}.txt
   label 1 行 = "x1 y1 x2 y2 x3 y3 x4 y4 <class_name> <difficulty>"
   difficulty は全て 0 (easy)。
6. skill/ep 階層は flat 化 (stem 衝突回避のため rename: insert_table_leg_ep0060_frame_00557)
7. image は --copy-mode {copy, symlink}。symlink はディスク節約 (default)

Usage (iros_2026_ramen root から):
    pixi run python data/yolo_obb/scripts/yolo_obb_to_dota.py \\
        --src-yolo-dataset data/yolo_obb/dataset \\
        --src-augmentation data/yolo_obb/augmentation/synthetic/v5_material_random \\
        --out data/yolo_obb/dataset_v5_material_dota \\
        --copy-mode symlink
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CLASS_NAMES = [
    "workspace",   # 0
    "leg",         # 1
    "leg_tip",     # 2
    "hole",        # 3
    "table_top",   # 4
    "hand_right",  # 5
    "hand_left",   # 6
]


def collect_pairs(dataset_root: Path, split: str, augmentation_roots: list[Path] | None = None) -> list[tuple[Path, Path]]:
    """image + label pair を列挙。train split には --src-augmentation を merge。"""
    pairs: list[tuple[Path, Path]] = []
    img_root = dataset_root / "images" / split
    lbl_root = dataset_root / "labels" / split
    if not img_root.exists():
        print(f"[warn] no split '{split}' in {dataset_root}", file=sys.stderr)
        return pairs
    for img in sorted(img_root.rglob("*.png")):
        rel = img.relative_to(img_root)
        lbl = lbl_root / rel.with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    if split == "train" and augmentation_roots:
        for aug_root in augmentation_roots:
            aug_img_root = aug_root / "images"
            aug_lbl_root = aug_root / "labels"
            if not aug_img_root.exists():
                print(f"[warn] augmentation root missing images: {aug_root}", file=sys.stderr)
                continue
            for img in sorted(aug_img_root.rglob("*.png")):
                rel = img.relative_to(aug_img_root)
                lbl = aug_lbl_root / rel.with_suffix(".txt")
                if lbl.exists():
                    pairs.append((img, lbl))
    return pairs


def flat_stem(img_path: Path, root: Path) -> str:
    """skill/ep 階層を flat 化した stem を返す (例: insert_table_leg_ep0060_frame_00557)。"""
    rel = img_path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1].endswith(".png"):
        parts[-1] = parts[-1][:-len(".png")]
    return "_".join(parts)


def convert_label(lbl_path: Path, img_w: int, img_h: int) -> list[str]:
    """YOLO-OBB line → DOTA line list。座標を pixel absolute に、class_id → class_name。"""
    out_lines = []
    for line in lbl_path.read_text().splitlines():
        tokens = line.strip().split()
        if len(tokens) != 9:
            continue
        cls_id = int(tokens[0])
        if not (0 <= cls_id < len(CLASS_NAMES)):
            print(f"[warn] out-of-range class {cls_id} in {lbl_path}, skipping line", file=sys.stderr)
            continue
        class_name = CLASS_NAMES[cls_id]
        coords_norm = [float(t) for t in tokens[1:]]
        # normalized → pixel、[0,1] 外は clip + warn
        clipped = False
        pixels = []
        for i in range(0, 8, 2):
            xn = coords_norm[i]
            yn = coords_norm[i + 1]
            if not (0.0 <= xn <= 1.0) or not (0.0 <= yn <= 1.0):
                clipped = True
                xn = max(0.0, min(1.0, xn))
                yn = max(0.0, min(1.0, yn))
            pixels.append(xn * img_w)
            pixels.append(yn * img_h)
        if clipped:
            print(f"[warn] normalized coords out of [0,1] in {lbl_path}, clipped", file=sys.stderr)
        # DOTA: "x1 y1 x2 y2 x3 y3 x4 y4 category difficulty"
        # difficulty は全て 0 (easy)
        pix_str = " ".join(f"{p:.2f}" for p in pixels)
        out_lines.append(f"{pix_str} {class_name} 0")
    return out_lines


def transfer_image(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_mode == "symlink":
        dst.symlink_to(src.resolve())
    elif copy_mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown copy_mode: {copy_mode}")


def convert_split(
    pairs: list[tuple[Path, Path]],
    src_roots: list[Path],  # image が属する root 候補 list (元 dataset の images/{split} + aug の images)
    out_root: Path,
    split: str,
    copy_mode: str,
) -> tuple[int, Counter]:
    """1 split (train or val) を DOTA 化。stem 生成のため src_roots を参照。"""
    out_img_dir = out_root / split / "images"
    out_lbl_dir = out_root / split / "labelTxt"
    for d in (out_img_dir, out_lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    class_counts: Counter = Counter()
    total = 0
    for img_path, lbl_path in pairs:
        matched_root = next((r for r in src_roots if str(img_path).startswith(str(r))), None)
        if matched_root is None:
            print(f"[warn] {img_path} does not belong to any src root, using file name only", file=sys.stderr)
            stem = img_path.stem
        else:
            stem = flat_stem(img_path, matched_root)
        # size 取得
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        # label 変換
        dota_lines = convert_label(lbl_path, img_w, img_h)
        for line in dota_lines:
            class_counts[line.split()[-2]] += 1
        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(dota_lines) + ("\n" if dota_lines else ""))
        # image 転送
        transfer_image(img_path, out_img_dir / f"{stem}.png", copy_mode)
        total += 1
    return total, class_counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-yolo-dataset", type=Path, required=True,
                   help="YOLO-OBB dataset root (images/{split}/... + labels/{split}/... 構造)")
    p.add_argument("--src-augmentation", type=Path, nargs="*", default=[],
                   help="train に merge する augmentation dir (images/ + labels/ 構造)、複数可")
    p.add_argument("--out", type=Path, required=True, help="DOTA 形式出力 root")
    p.add_argument("--copy-mode", type=str, default="symlink", choices=["copy", "symlink"],
                   help="image 転送方式 (symlink はディスク節約、default)")
    args = p.parse_args()

    print(f"[cfg] src_yolo={args.src_yolo_dataset}, augmentation={args.src_augmentation}, out={args.out}, copy={args.copy_mode}")

    all_class_counts: dict[str, Counter] = defaultdict(Counter)
    for split in ("train", "val"):
        pairs = collect_pairs(args.src_yolo_dataset, split, args.src_augmentation if split == "train" else None)
        print(f"[{split}] {len(pairs)} pairs collected")
        if not pairs:
            continue
        # src_roots: 元 dataset の images/{split} + aug dir の images/
        src_roots = [args.src_yolo_dataset / "images" / split]
        if split == "train":
            src_roots.extend((aug / "images") for aug in args.src_augmentation)
        total, cls_counts = convert_split(pairs, src_roots, args.out, split, args.copy_mode)
        print(f"[{split}] {total} images converted")
        all_class_counts[split] = cls_counts

    # report
    print("\n== class balance ==")
    for split_name, cls_counts in all_class_counts.items():
        print(f"[{split_name}]")
        for cls in CLASS_NAMES:
            print(f"  {cls:11}: {cls_counts.get(cls, 0):5}")

    print(f"\n[done] out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
