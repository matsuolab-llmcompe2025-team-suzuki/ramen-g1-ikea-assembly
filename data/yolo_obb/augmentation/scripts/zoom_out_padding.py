"""zoom-out padding augment: 元 image を factor scale + canvas 内 random 位置に配置、
余白を Places365 bg で埋める。workspace が「常に画面全体」問題への明示的対策。

Issue #108 v8: v6 の built-in geometry aug (random で薄い) から進化し、
「workspace 縮小 + reposition」を **必ず** train image に導入する。
全 OBB 座標を factor scale + offset で label 変換 = アノテーション追加ゼロ。

処理:
1. 元 image (workspace + arm/leg 等含む) を factor (uniform 0.5-0.8 default) で resize
2. canvas (640×480) の random 位置に scaled image を paste
3. 余白は Places365 bg (v4 の bg pool 流用) を配置、feather 境界
4. 全 OBB 座標を: x_new_norm = x_old_norm * factor + offset_x_norm

Usage (iros_2026_ramen root から model/yolo_obb pixi env で):
    cd model/yolo_obb
    # pilot: 4 source × 3 factor pattern の mosaic
    pixi run python ../../data/yolo_obb/augmentation/scripts/zoom_out_padding.py --pilot \\
        --img-root ../../data/yolo_obb/dataset/images/train \\
        --lbl-root ../../data/yolo_obb/dataset/labels/train \\
        --out ../../data/yolo_obb/augmentation/synthetic/v8_zoomout_pilot

    # full: 全 image × N augment (factor は per-augment random)
    pixi run python ../../data/yolo_obb/augmentation/scripts/zoom_out_padding.py --full \\
        --n-per-source 3 \\
        --img-root ../../data/yolo_obb/dataset/images/train \\
        --lbl-root ../../data/yolo_obb/dataset/labels/train \\
        --bg-dir ../../data/yolo_obb/augmentation/backgrounds/places365 \\
        --out ../../data/yolo_obb/augmentation/synthetic/v8_zoomout_random
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DEFAULT_TRAIN_IMG = Path("data/yolo_obb/dataset/images/train")
DEFAULT_TRAIN_LBL = Path("data/yolo_obb/dataset/labels/train")
DEFAULT_BG_DIR = Path("data/yolo_obb/augmentation/backgrounds/places365")
DEFAULT_OUT = Path("data/yolo_obb/augmentation/synthetic/v8_zoomout_pilot")
IMG_W, IMG_H = 640, 480

# Pilot: 4 sample × 3 factor pattern (0.65 = 強縮小、0.75 = 中、0.85 = 弱縮小)
PILOT_SKILLS = ["insert_table_leg", "pick_table_leg", "flip_table", "move_to_table"]
PILOT_FACTORS = [0.65, 0.75, 0.85]

# Full: factor は random range、positioning は user 指示「上詰め」= offset_y=0 固定、offset_x は水平中央
RANDOM_FACTOR = (0.65, 0.85)  # uniform sample
BOUNDARY_FEATHER_PX = 5


def load_source_pairs(img_root: Path, lbl_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img in sorted(img_root.rglob("*.png")):
        rel = img.relative_to(img_root)
        lbl = lbl_root / rel.with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def pick_pilot_sources(pairs, rng, n=4):
    by_skill: dict[str, list[tuple[Path, Path]]] = {}
    for img, lbl in pairs:
        skill = img.parent.parent.name
        by_skill.setdefault(skill, []).append((img, lbl))
    picked = []
    for skill in PILOT_SKILLS[:n]:
        if skill in by_skill and by_skill[skill]:
            picked.append(rng.choice(by_skill[skill]))
    return picked


def parse_obb_label(lbl_path: Path) -> list[tuple[int, list[float]]]:
    """label file から (class_id, [x1n, y1n, ..., x4n, y4n]) list を返す (normalized)."""
    out = []
    for line in lbl_path.read_text().splitlines():
        toks = line.strip().split()
        if len(toks) != 9:
            continue
        cls = int(toks[0])
        coords = [float(t) for t in toks[1:]]
        out.append((cls, coords))
    return out


def transform_obb(coords: list[float], factor: float, offset_x_norm: float, offset_y_norm: float) -> list[float]:
    """normalized OBB 座標を factor scale + offset で変換 (canvas 内座標に射影)。"""
    new = []
    for i in range(0, 8, 2):
        x_new = coords[i] * factor + offset_x_norm
        y_new = coords[i + 1] * factor + offset_y_norm
        new.extend([x_new, y_new])
    return new


def prepare_background(bg_path: Path, target_w: int = IMG_W, target_h: int = IMG_H) -> Image.Image:
    bg = Image.open(bg_path).convert("RGB")
    w, h = bg.size
    if h < w:
        new_h = target_h
        new_w = int(w * new_h / h)
    else:
        new_w = target_w
        new_h = int(h * new_w / w)
    if new_w < target_w or new_h < target_h:
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
    bg = bg.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return bg.crop((left, top, left + target_w, top + target_h))


def synth_one(
    source_img: Path,
    source_lbl: Path,
    bg_path: Path | None,
    factor: float,
    offset_x_pixel: int,
    offset_y_pixel: int,
    out_img: Path,
    out_lbl: Path,
) -> None:
    """1 枚合成 + OBB label 変換。"""
    src = Image.open(source_img).convert("RGB")
    if src.size != (IMG_W, IMG_H):
        src = src.resize((IMG_W, IMG_H), Image.LANCZOS)

    # 1. bg 準備 (bg_path=None なら solid gray)
    if bg_path is not None:
        bg = prepare_background(bg_path)
    else:
        bg = Image.new("RGB", (IMG_W, IMG_H), (128, 128, 128))

    # 2. src を factor で resize
    scaled_w = int(IMG_W * factor)
    scaled_h = int(IMG_H * factor)
    scaled = src.resize((scaled_w, scaled_h), Image.LANCZOS)

    # 3. bg に scaled を paste (feather boundary)
    canvas = bg.copy()
    # mask: scaled region に 255、feather blur で境界柔らかく
    mask = Image.new("L", (IMG_W, IMG_H), 0)
    ImageDraw.Draw(mask).rectangle(
        (offset_x_pixel, offset_y_pixel, offset_x_pixel + scaled_w, offset_y_pixel + scaled_h),
        fill=255,
    )
    if BOUNDARY_FEATHER_PX > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=BOUNDARY_FEATHER_PX))
    # scaled を対応位置に配置した paste_img を作る
    paste_img = bg.copy()
    paste_img.paste(scaled, (offset_x_pixel, offset_y_pixel))
    # mask で blend
    mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
    canvas_arr = np.asarray(canvas, dtype=np.float32)
    paste_arr = np.asarray(paste_img, dtype=np.float32)
    out_arr = paste_arr * mask_arr[..., None] + canvas_arr * (1.0 - mask_arr[..., None])
    result = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8))

    # 4. OBB label 変換
    polygons = parse_obb_label(source_lbl)
    offset_x_norm = offset_x_pixel / IMG_W
    offset_y_norm = offset_y_pixel / IMG_H
    new_lines = []
    for cls, coords in polygons:
        new_coords = transform_obb(coords, factor, offset_x_norm, offset_y_norm)
        # [0, 1] 外は clip
        new_coords = [max(0.0, min(1.0, c)) for c in new_coords]
        new_lines.append(f"{cls} " + " ".join(f"{c:.6f}" for c in new_coords))

    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_img)
    out_lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))


def make_pilot_mosaic(rows, factor_labels, out_path):
    """rows: [(src_path, [result_path, ...]), ...]、factor_labels: ["f=0.5", "f=0.65", "f=0.8"]"""
    thumb_w, thumb_h = 320, 240
    n_cols = 1 + len(factor_labels)
    n_rows = len(rows)
    header_h = 24
    canvas = Image.new("RGB", (thumb_w * n_cols, thumb_h * n_rows + header_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 4), "original", fill=(240, 240, 240))
    for c, lab in enumerate(factor_labels, 1):
        draw.text((c * thumb_w + 10, 4), lab, fill=(240, 240, 240))
    for r, (src, results) in enumerate(rows):
        y0 = header_h + r * thumb_h
        src_img = Image.open(src).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
        canvas.paste(src_img, (0, y0))
        for c, res_path in enumerate(results, 1):
            res_img = Image.open(res_path).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(res_img, (c * thumb_w, y0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-root", type=Path, default=DEFAULT_TRAIN_IMG)
    p.add_argument("--lbl-root", type=Path, default=DEFAULT_TRAIN_LBL)
    p.add_argument("--bg-dir", type=Path, default=DEFAULT_BG_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pilot", action="store_true", help="4 sample × 3 factor pattern + mosaic")
    p.add_argument("--full", action="store_true", help="全 image × N augment (random factor + position)")
    p.add_argument("--n-per-source", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pairs = load_source_pairs(args.img_root, args.lbl_root)
    print(f"[info] sources: {len(pairs)}", file=sys.stderr)
    if not pairs:
        return 1

    bgs = sorted(list(args.bg_dir.rglob("*.jpg")))
    bgs = [b for b in bgs if not b.name.startswith("_")]
    print(f"[info] backgrounds: {len(bgs)}", file=sys.stderr)
    if args.full and not bgs:
        print("[error] full mode requires background images", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    out_img_root = args.out / "images"
    out_lbl_root = args.out / "labels"

    def top_align_offset(factor):
        """user 指示「上詰め」: offset_y=0、offset_x は水平中央 (workspace の左右対称性維持)"""
        scaled_w = int(IMG_W * factor)
        off_x = (IMG_W - scaled_w) // 2
        return off_x, 0

    if args.pilot:
        sources = pick_pilot_sources(pairs, rng, n=4)
        rows = []
        for src_img, src_lbl in sources:
            results = []
            for factor in PILOT_FACTORS:
                local_rng = random.Random(args.seed + hash(f"{src_img.name}:{factor}") % (2**31))
                off_x, off_y = top_align_offset(factor)
                bg = local_rng.choice(bgs) if bgs else None
                rel = src_img.relative_to(args.img_root)
                stem = f"{rel.stem}_zoomout_f{int(factor*100)}"
                out_img = out_img_root / rel.parent / f"{stem}.png"
                out_lbl = out_lbl_root / rel.parent / f"{stem}.txt"
                synth_one(src_img, src_lbl, bg, factor, off_x, off_y, out_img, out_lbl)
                results.append(out_img)
            rows.append((src_img, results))
            print(f"[pilot] {src_img.name} × {len(PILOT_FACTORS)} factors done", file=sys.stderr)
        make_pilot_mosaic(rows, [f"f={f}" for f in PILOT_FACTORS], args.out / "_pilot_mosaic.jpg")
        print(f"[done] pilot={len(rows) * len(PILOT_FACTORS)} images, mosaic={args.out}/_pilot_mosaic.jpg", file=sys.stderr)
        return 0

    if args.full:
        total = 0
        for src_img, src_lbl in pairs:
            rel = src_img.relative_to(args.img_root)
            for k in range(args.n_per_source):
                local_rng = random.Random(args.seed + hash(f"{rel}:{k}") % (2**31))
                factor = local_rng.uniform(*RANDOM_FACTOR)
                off_x, off_y = top_align_offset(factor)
                bg = local_rng.choice(bgs)
                stem = f"{rel.stem}_zoomout{k}"
                out_img = out_img_root / rel.parent / f"{stem}.png"
                out_lbl = out_lbl_root / rel.parent / f"{stem}.txt"
                synth_one(src_img, src_lbl, bg, factor, off_x, off_y, out_img, out_lbl)
                total += 1
            if total > 0 and total % 100 == 0:
                print(f"[full] {total} synthesized", file=sys.stderr)
        print(f"[done] full={total} images, out={args.out}", file=sys.stderr)
        return 0

    print("[error] --pilot か --full を指定してください", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
