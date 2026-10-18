"""ultralytics YOLO の built-in HSV augment (hsv_h/hsv_s/hsv_v) の max range を事前確認する pilot script。

Issue #108 v9 検討: v8_bg + hsv_v 強化を検討中、range max (vgain=0.8-1.0) で
image が破壊されてないか (workspace/leg 見えなくなる) を目視で確認。

処理:
1. 4 sample image を pick
2. 各 image に対し vgain [0.4/0.6/0.8/1.0] を + / - 両方向で apply (計 8 augment)
3. mosaic 出力: row = source (4), col = original + 8 pattern

ultralytics の RandomHSV と同 formula:
    r = np.random.uniform(-1, 1) * [hgain, sgain, vgain] + 1  # multiplier vary
    hsv[..., v_ch] = clip(hsv[..., v_ch] * r[v_ch], 0, 255)

Usage:
    cd model/yolo_obb
    pixi run python ../../data/yolo_obb/augmentation/scripts/hsv_range_pilot.py \\
        --img-root ../../data/yolo_obb/dataset/images/train \\
        --out ../../data/yolo_obb/augmentation/synthetic/v9_hsv_range_pilot
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

IMG_W, IMG_H = 640, 480
PILOT_SKILLS = ["insert_table_leg", "pick_table_leg", "flip_table", "move_to_table"]

# gain range max values to test (each is the max、fixed sign で + / - 両方向 apply)
GAIN_MAXES = {
    "v": [0.4, 0.6, 0.8, 1.0],           # 明度 (default 0.4)
    "h": [0.015, 0.05, 0.1, 0.2, 0.3],   # 色相 (default 0.015、微弱)
    "s": [0.7, 0.9, 1.0],                # 彩度 (default 0.7)
}


def load_source_pairs(img_root: Path) -> dict[str, list[Path]]:
    by_skill: dict[str, list[Path]] = {}
    for img in sorted(img_root.rglob("*.png")):
        skill = img.parent.parent.name
        by_skill.setdefault(skill, []).append(img)
    return by_skill


def apply_hsv(img_bgr: np.ndarray, hgain: float, sgain: float, vgain: float) -> np.ndarray:
    """ultralytics 風 HSV augment (multiplier 変化、fixed direction)。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    # multiplier: 1 + gain (positive) or 1 - gain (negative direction test 用に呼び手で符号決定)
    h_mult = 1 + hgain
    s_mult = 1 + sgain
    v_mult = 1 + vgain
    hsv[..., 0] = np.clip(hsv[..., 0] * h_mult, 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1] * s_mult, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_mult, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def make_mosaic(sources: list[Path], variants_per_source: list[list[tuple[str, Path]]], out_path: Path):
    """rows = source, cols = original + 各 variant"""
    thumb_w, thumb_h = 220, 165  # 8 col + original = 9 col、幅節約
    n_variants = len(variants_per_source[0])
    n_cols = 1 + n_variants
    n_rows = len(sources)
    header_h = 24
    canvas_w = thumb_w * n_cols
    canvas_h = header_h + thumb_h * n_rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 4), "original", fill=(240, 240, 240))
    for c, (label, _) in enumerate(variants_per_source[0], 1):
        draw.text((c * thumb_w + 4, 4), label, fill=(240, 240, 240))
    for r, src in enumerate(sources):
        y0 = header_h + r * thumb_h
        orig = Image.open(src).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
        canvas.paste(orig, (0, y0))
        for c, (_, path) in enumerate(variants_per_source[r], 1):
            img = Image.open(path).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(img, (c * thumb_w, y0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-root", type=Path, default=Path("data/yolo_obb/dataset/images/train"))
    p.add_argument("--out", type=Path, default=Path("data/yolo_obb/augmentation/synthetic/v9_hsv_range_pilot"))
    p.add_argument("--gain-type", type=str, default="v", choices=["v", "h", "s"], help="どの HSV gain を pilot")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    by_skill = load_source_pairs(args.img_root)
    rng = random.Random(args.seed)
    sources = []
    for skill in PILOT_SKILLS:
        if skill in by_skill and by_skill[skill]:
            sources.append(rng.choice(by_skill[skill]))
    if not sources:
        print("[error] no sources", file=__import__("sys").stderr); return 1

    args.out.mkdir(parents=True, exist_ok=True)
    gain_maxes = GAIN_MAXES[args.gain_type]
    variants_per_source = []
    for src in sources:
        img_bgr = cv2.imread(str(src))
        if img_bgr is None:
            continue
        variants = []
        for gmax in gain_maxes:
            # apply_hsv(img, hgain, sgain, vgain) の該当 channel だけ vary
            for sign, sign_label in [(-1, "-"), (+1, "+")]:
                if args.gain_type == "v":
                    im = apply_hsv(img_bgr, 0.0, 0.0, sign * gmax)
                elif args.gain_type == "h":
                    im = apply_hsv(img_bgr, sign * gmax, 0.0, 0.0)
                else:  # s
                    im = apply_hsv(img_bgr, 0.0, sign * gmax, 0.0)
                out_path = args.out / f"{src.stem}_{args.gain_type}{sign_label}{int(gmax*1000)}.png"
                cv2.imwrite(str(out_path), im)
                variants.append((f"{args.gain_type}{sign_label}{gmax}", out_path))
        variants_per_source.append(variants)
        print(f"[pilot] {src.name} × {len(gain_maxes)*2} done", flush=True)

    make_mosaic(sources, variants_per_source, args.out / "_pilot_mosaic.jpg")
    print(f"[done] mosaic: {args.out}/_pilot_mosaic.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
