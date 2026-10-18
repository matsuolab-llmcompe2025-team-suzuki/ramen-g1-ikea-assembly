"""workspace (class 0) の polygon 内で chroma key (緑 mat pixel のみ) を絞り込み、
S=0 (無彩色化 / tint) + V scale (元 pattern 保持) + specular manipulation + noise を
行う material augment。連続 random パラメータを想定した実装。

Issue #77 v5 material variant: workspace polygon 内の「本物のマット pixel だけ」を
色相ベース chroma key で抽出 → arm / leg / hand / hole 等の非マット pixel は
自動的に元のまま保持される。

処理:
1. 元 image + label 読み込み
2. workspace polygon の binary mask 作成
3. **chroma key**: polygon 内で H が緑域 (80°〜150°) かつ S ≥ 20 (PIL scale) の
   pixel を選別 → 「mat pixel」sub-mask。arm (S≈0 の白) / leg (H が緑外) は自動除外
4. mat pixel の色を書き換え:
   - `tint_s == 0`: 完全 desaturate (S=0)
   - `tint_s > 0`: S=tint_s, H は per-augment random (弱 tint 残し)
   - V *= v_scale (元 V pattern と相対テカリを保持したまま暗く/明るく)
5. Specular 処理:
   - none: 何もしない (V scale 適用済み、元 pattern 保持)
   - remove: mat 内 V median で均一化 (テカリ消去、無地マット)
   - shuffle: 均一化 → mat 内 random 位置に Gaussian blob を N 個 overlay
6. noise: mat pixel の V に Gaussian noise σ (grain 感)
7. HSV → RGB
8. mat_bin の soft mask (3px feather) で alpha blend
9. label file はそのままコピー

Modes:
    --pilot         : 5 固定 variant × 4 sample の preset カタログ
    --range-pilot   : 6 軸 × 5 段階の range 端値確認 (1 sample image で全軸展開)
    --full          : 全 image に user 承認済 range から uniform sample で N augment/image 生成
                      (--head N で先頭 N image に限定 = smoke)
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DEFAULT_TRAIN_IMG = Path("data/yolo_obb/dataset/images/train")
DEFAULT_TRAIN_LBL = Path("data/yolo_obb/dataset/labels/train")
DEFAULT_OUT = Path("data/yolo_obb/augmentation/synthetic/v5_material_pilot")
IMG_W, IMG_H = 640, 480
WORKSPACE_CLASS = 0

PILOT_SKILLS = ["insert_table_leg", "pick_table_leg", "flip_table", "move_to_table"]

# preset pilot: 5 固定 variant (label, v_scale, specular_mode)
PILOT_VARIANTS = [
    ("V x0.6 none",    0.6, "none"),
    ("V x1.0 none",    1.0, "none"),
    ("V x1.4 none",    1.4, "none"),
    ("V x0.6 remove",  0.6, "remove"),
    ("V x1.0 shuffle", 1.0, "shuffle"),
]

# range pilot: 各軸 5 段階 + 他軸中央値固定
RANGE_PILOT_AXES = [
    # (axis label, param name, 5 段階の値, base overrides)
    ("V scale",        "v_scale",       [0.4, 0.7, 1.0, 1.3, 1.6],  {"specular_mode": "none"}),
    ("blob count",     "blob_count",    [1, 2, 3, 4, 5],            {"specular_mode": "shuffle"}),
    ("blob radius",    "blob_radius",   [8, 14, 21, 28, 35],        {"specular_mode": "shuffle"}),
    ("blob intensity", "blob_intensity",[15, 30, 45, 60, 80],       {"specular_mode": "shuffle"}),
    ("noise sigma",    "noise_sigma",   [0, 3, 6, 9, 12],           {"specular_mode": "none"}),
    ("tint S",         "tint_s",        [0, 6, 12, 18, 25],         {"specular_mode": "none"}),
]

# 中央値 (range pilot で他軸を固定するときに使う)
CENTER_PARAMS = {
    "v_scale":       1.0,
    "blob_count":    3,
    "blob_radius":   20,
    "blob_intensity":45,
    "noise_sigma":   0,
    "tint_s":        0,
    "specular_mode": "none",
}

# Chroma key: workspace 内 mat pixel を色相で選別
# PIL HSV は H を 0-255 で表現 (通常の 0-360 と違う)
HUE_RANGE_DEG = (80, 150)
HUE_LO_PIL = int(HUE_RANGE_DEG[0] * 256 / 360)
HUE_HI_PIL = int(HUE_RANGE_DEG[1] * 256 / 360)
SAT_MIN_PIL = 20  # S ≥ 20 で arm/白背景 (S≈0) を除外

MAT_MASK_FEATHER_PX = 3

# --full の連続 random 範囲 (user 承認済み range)
RANDOM_V_SCALE = (0.5, 1.3)
RANDOM_BLOB_COUNT = (1, 5)          # inclusive
RANDOM_BLOB_RADIUS = (8, 35)
RANDOM_BLOB_INTENSITY = (15, 60)
RANDOM_NOISE_SIGMA = (0.0, 12.0)
RANDOM_TINT_S = (0, 12)             # inclusive
RANDOM_MODES = ("none", "remove", "shuffle")  # 等確率


def load_source_pairs(img_root: Path, lbl_root: Path) -> list[tuple[Path, Path]]:
    """image + label pair を列挙 (workspace polygon を持つものだけ)。"""
    pairs = []
    for img in sorted(img_root.rglob("*.png")):
        rel = img.relative_to(img_root)
        lbl = lbl_root / rel.with_suffix(".txt")
        if not lbl.exists():
            continue
        if parse_workspace_polys(lbl):
            pairs.append((img, lbl))
    return pairs


def pick_pilot_sources(pairs, rng, n=4):
    """skill を分散させて n 個抽出。"""
    by_skill: dict[str, list[tuple[Path, Path]]] = {}
    for img, lbl in pairs:
        skill = img.parent.parent.name
        by_skill.setdefault(skill, []).append((img, lbl))
    picked: list[tuple[Path, Path]] = []
    for skill in PILOT_SKILLS[:n]:
        if skill in by_skill and by_skill[skill]:
            picked.append(rng.choice(by_skill[skill]))
    remain = [p for s, lst in by_skill.items() if s not in PILOT_SKILLS[:n] for p in lst]
    while len(picked) < n and remain:
        picked.append(rng.choice(remain))
    return picked


def pick_range_pilot_source(pairs):
    """range pilot 用: mat がはっきり見える 1 sample を選ぶ (pick_table_leg 優先)。"""
    for img, lbl in pairs:
        if "pick_table_leg" in str(img):
            return (img, lbl)
    return pairs[0]


def parse_workspace_polys(lbl_path: Path) -> list[list[tuple[float, float]]]:
    polygons = []
    for line in lbl_path.read_text().splitlines():
        tokens = line.strip().split()
        if len(tokens) != 9:
            continue
        cls = int(tokens[0])
        if cls != WORKSPACE_CLASS:
            continue
        coords = [float(t) for t in tokens[1:]]
        poly = [(coords[i] * IMG_W, coords[i + 1] * IMG_H) for i in range(0, 8, 2)]
        polygons.append(poly)
    return polygons


def rasterize_polygons_binary(polygons, w, h):
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        clipped = [(max(0, min(w - 1, x)), max(0, min(h - 1, y))) for x, y in poly]
        draw.polygon(clipped, fill=255)
    return np.asarray(mask_img) > 128


def apply_material_aug(
    source: Image.Image,
    ws_bin: np.ndarray,
    v_scale: float = 1.0,
    specular_mode: str = "none",
    blob_count: int = 3,
    blob_radius: int = 20,
    blob_intensity: int = 45,
    noise_sigma: float = 0.0,
    tint_s: int = 0,
    rng: random.Random = None,
) -> Image.Image:
    """workspace polygon 内で chroma key + S=tint_s + V scale + specular + noise。"""
    if rng is None:
        rng = random.Random()
    src_hsv = np.asarray(source.convert("HSV"), dtype=np.int16).copy()

    # Chroma key
    hue_bin = (src_hsv[..., 0] >= HUE_LO_PIL) & (src_hsv[..., 0] <= HUE_HI_PIL)
    sat_bin = src_hsv[..., 1] >= SAT_MIN_PIL
    mat_bin = ws_bin & hue_bin & sat_bin
    if not mat_bin.any():
        return source.copy()

    new_hsv = src_hsv.copy()

    # 1. mat pixel: S=tint_s (or 0) + V scale
    if tint_s > 0:
        new_hsv[..., 1][mat_bin] = tint_s
        new_hsv[..., 0][mat_bin] = rng.randrange(0, 256)  # per augment で 1 色相
    else:
        new_hsv[..., 1][mat_bin] = 0
    v_scaled = src_hsv[..., 2][mat_bin].astype(np.float32) * v_scale
    new_hsv[..., 2][mat_bin] = np.clip(v_scaled, 0, 255).astype(np.int16)

    # 2. Specular manipulation
    if specular_mode == "none":
        pass
    elif specular_mode == "remove":
        mat_v_median = int(np.median(new_hsv[..., 2][mat_bin]))
        new_hsv[..., 2][mat_bin] = mat_v_median
    elif specular_mode == "shuffle":
        mat_v_median = int(np.median(new_hsv[..., 2][mat_bin]))
        new_hsv[..., 2][mat_bin] = mat_v_median
        overlay = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        mat_indices = np.argwhere(mat_bin)
        y_grid = np.arange(IMG_H)[:, None]
        x_grid = np.arange(IMG_W)[None, :]
        for _ in range(blob_count):
            cy, cx = mat_indices[rng.randrange(len(mat_indices))]
            gauss = np.exp(-((y_grid - cy) ** 2 + (x_grid - cx) ** 2) / (2 * blob_radius ** 2))
            overlay += gauss.astype(np.float32) * blob_intensity
        v_new = new_hsv[..., 2].astype(np.float32) + overlay
        new_hsv[..., 2] = np.where(mat_bin, np.clip(v_new, 0, 255), new_hsv[..., 2])
    else:
        raise ValueError(f"unknown specular_mode: {specular_mode}")

    # 3. Noise (mat pixel V に gaussian)
    if noise_sigma > 0:
        np_rng = np.random.default_rng(rng.randrange(0, 2**31))
        noise = np_rng.normal(0, noise_sigma, size=(IMG_H, IMG_W))
        v_noised = new_hsv[..., 2].astype(np.float32) + noise
        new_hsv[..., 2] = np.where(mat_bin, np.clip(v_noised, 0, 255), new_hsv[..., 2])

    modified = Image.fromarray(new_hsv.astype(np.uint8), mode="HSV").convert("RGB")

    # 4. mat_bin soft mask で alpha blend
    mat_mask_img = Image.fromarray((mat_bin.astype(np.uint8) * 255), mode="L")
    if MAT_MASK_FEATHER_PX > 0:
        mat_mask_img = mat_mask_img.filter(ImageFilter.GaussianBlur(radius=MAT_MASK_FEATHER_PX))
    mat_mask_soft = np.asarray(mat_mask_img, dtype=np.float32) / 255.0
    src_arr = np.asarray(source, dtype=np.float32)
    mod_arr = np.asarray(modified, dtype=np.float32)
    mask3 = mat_mask_soft[..., None]
    out = mod_arr * mask3 + src_arr * (1.0 - mask3)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def load_image_for_aug(src_img_path: Path):
    src = Image.open(src_img_path).convert("RGB")
    if src.size != (IMG_W, IMG_H):
        src = src.resize((IMG_W, IMG_H), Image.LANCZOS)
    return src


# --------------------------------------------------------------------
# preset pilot (5 固定 variant × 4 sample) — 従来 mode
# --------------------------------------------------------------------
def run_preset_pilot(pairs, args, rng):
    sources = pick_pilot_sources(pairs, rng, n=4)
    out_img_root = args.out / "images"
    out_lbl_root = args.out / "labels"
    rows = []
    for src_img, src_lbl in sources:
        src = load_image_for_aug(src_img)
        polys = parse_workspace_polys(src_lbl)
        ws_bin = rasterize_polygons_binary(polys, IMG_W, IMG_H)
        results = []
        for label, v_scale, spec_mode in PILOT_VARIANTS:
            rel = src_img.relative_to(args.img_root)
            stem = f"{rel.stem}_ws_x{v_scale:.1f}_{spec_mode}"
            out_img = out_img_root / rel.parent / f"{stem}.png"
            out_lbl = out_lbl_root / rel.parent / f"{stem}.txt"
            local_rng = random.Random(args.seed + hash(stem) % (2**31))
            result = apply_material_aug(src, ws_bin, v_scale=v_scale, specular_mode=spec_mode, rng=local_rng)
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            result.save(out_img)
            shutil.copyfile(src_lbl, out_lbl)
            results.append(out_img)
        rows.append((src_img, results))
        print(f"[preset] {src_img.name} × {len(results)} variants done", file=sys.stderr)
    mosaic_path = args.out / "_pilot_mosaic.jpg"
    make_preset_mosaic(rows, [v[0] for v in PILOT_VARIANTS], mosaic_path)
    print(f"[done] preset={sum(len(r[1]) for r in rows)} images", file=sys.stderr)
    print(f"[done] mosaic: {mosaic_path}", file=sys.stderr)


def make_preset_mosaic(rows, variant_labels, out_path):
    thumb_w, thumb_h = 320, 240
    n_cols = 1 + len(variant_labels)
    n_rows = len(rows)
    header_h = 24
    canvas = Image.new("RGB", (thumb_w * n_cols, thumb_h * n_rows + header_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 4), "original", fill=(240, 240, 240))
    for c, lab in enumerate(variant_labels, 1):
        draw.text((c * thumb_w + 10, 4), lab, fill=(240, 240, 240))
    for r, (src, results) in enumerate(rows):
        src_img = Image.open(src).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
        canvas.paste(src_img, (0, header_h + r * thumb_h))
        for c, res_path in enumerate(results, 1):
            res_img = Image.open(res_path).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(res_img, (c * thumb_w, header_h + r * thumb_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


# --------------------------------------------------------------------
# range pilot (6 軸 × 5 段階、1 sample image)
# --------------------------------------------------------------------
def run_range_pilot(pairs, args, rng):
    """各軸の 5 段階を 1 image で並べて mosaic 出力 (range 端値確認用)。"""
    src_img_path, src_lbl_path = pick_range_pilot_source(pairs)
    print(f"[range] source: {src_img_path.name}", file=sys.stderr)
    src = load_image_for_aug(src_img_path)
    polys = parse_workspace_polys(src_lbl_path)
    ws_bin = rasterize_polygons_binary(polys, IMG_W, IMG_H)

    out_img_root = args.out / "images"
    result_grid = []  # [(axis_label, [(val, out_path), ...]), ...]
    for axis_label, axis_param, values, base_overrides in RANGE_PILOT_AXES:
        row = []
        for val in values:
            kw = dict(CENTER_PARAMS)
            kw.update(base_overrides)
            kw[axis_param] = val
            local_rng = random.Random(args.seed + hash(f"{axis_param}:{val}") % (2**31))
            result = apply_material_aug(
                src, ws_bin,
                v_scale=kw["v_scale"],
                specular_mode=kw["specular_mode"],
                blob_count=kw["blob_count"],
                blob_radius=kw["blob_radius"],
                blob_intensity=kw["blob_intensity"],
                noise_sigma=kw["noise_sigma"],
                tint_s=kw["tint_s"],
                rng=local_rng,
            )
            stem = f"range_{axis_param}_{val}"
            out_path = out_img_root / f"{stem}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(out_path)
            row.append((val, out_path))
        result_grid.append((axis_label, row))
        print(f"[range] axis={axis_label} 5 vals done", file=sys.stderr)

    mosaic_path = args.out / "_range_pilot_mosaic.jpg"
    make_range_mosaic(src_img_path, result_grid, mosaic_path)
    print(f"[done] range mosaic: {mosaic_path}", file=sys.stderr)


def make_range_mosaic(src_img_path, result_grid, out_path):
    """rows = 6 axes, cols = 5 段階 + 1 (leftmost = original 参照)。"""
    thumb_w, thumb_h = 320, 240
    axis_col_w = 130
    header_h = 30
    n_cols = 1 + 5  # ref (original) + 5 段階
    n_rows = len(result_grid)
    canvas_w = axis_col_w + n_cols * thumb_w
    canvas_h = header_h + n_rows * thumb_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), f"src: {src_img_path.name}   (leftmost = original, others = per-axis sweep)", fill=(240, 240, 240))
    orig = Image.open(src_img_path).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
    for r, (axis_label, row) in enumerate(result_grid):
        y0 = header_h + r * thumb_h
        # left: axis label
        draw.text((10, y0 + thumb_h // 2 - 10), axis_label, fill=(255, 220, 100))
        # col 0: original ref
        canvas.paste(orig, (axis_col_w, y0))
        draw.text((axis_col_w + 5, y0 + 5), "original", fill=(200, 200, 200))
        # col 1..5: per-value results
        for c, (val, res_path) in enumerate(row, 1):
            res_img = Image.open(res_path).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
            x0 = axis_col_w + c * thumb_w
            canvas.paste(res_img, (x0, y0))
            draw.text((x0 + 5, y0 + 5), f"= {val}", fill=(255, 255, 100))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


# --------------------------------------------------------------------
# full (連続 random) mode
# --------------------------------------------------------------------
def sample_random_params(rng: random.Random) -> dict:
    """user 承認済み range から uniform sample。"""
    return dict(
        v_scale=rng.uniform(*RANDOM_V_SCALE),
        specular_mode=rng.choice(RANDOM_MODES),
        blob_count=rng.randint(*RANDOM_BLOB_COUNT),
        blob_radius=rng.randint(*RANDOM_BLOB_RADIUS),
        blob_intensity=rng.randint(*RANDOM_BLOB_INTENSITY),
        noise_sigma=rng.uniform(*RANDOM_NOISE_SIGMA),
        tint_s=rng.randint(*RANDOM_TINT_S),
    )


def run_full_random(pairs, args, rng):
    """全 image に対し N augment/image を連続 random parameter で生成。"""
    out_img_root = args.out / "images"
    out_lbl_root = args.out / "labels"
    total = 0
    skipped = 0
    if args.head is not None:
        pairs = pairs[:args.head]
        print(f"[full] --head={args.head}: subset で smoke", file=sys.stderr)
    for i, (src_img, src_lbl) in enumerate(pairs, 1):
        src = load_image_for_aug(src_img)
        polys = parse_workspace_polys(src_lbl)
        ws_bin = rasterize_polygons_binary(polys, IMG_W, IMG_H)
        # 事前 chroma-key 判定: mat pixel 0 なら apply しない (source そのまま返るだけ)
        src_hsv_arr = np.asarray(src.convert("HSV"), dtype=np.int16)
        hue_bin = (src_hsv_arr[..., 0] >= HUE_LO_PIL) & (src_hsv_arr[..., 0] <= HUE_HI_PIL)
        sat_bin = src_hsv_arr[..., 1] >= SAT_MIN_PIL
        if not (ws_bin & hue_bin & sat_bin).any():
            skipped += args.n_per_source
            continue
        rel = src_img.relative_to(args.img_root)
        for k in range(args.n_per_source):
            local_rng = random.Random(args.seed + hash(f"{rel}:{k}") % (2**31))
            params = sample_random_params(local_rng)
            result = apply_material_aug(src, ws_bin, rng=local_rng, **params)
            stem = f"{rel.stem}_v5r{k}"
            out_img = out_img_root / rel.parent / f"{stem}.png"
            out_lbl = out_lbl_root / rel.parent / f"{stem}.txt"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            result.save(out_img)
            shutil.copyfile(src_lbl, out_lbl)
            total += 1
        if i % 50 == 0:
            print(f"[full] {i}/{len(pairs)} sources ({total} written, {skipped} skipped)", file=sys.stderr)
    print(f"[done] full={total} skipped={skipped} out={args.out}", file=sys.stderr)


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-root", type=Path, default=DEFAULT_TRAIN_IMG)
    p.add_argument("--lbl-root", type=Path, default=DEFAULT_TRAIN_LBL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pilot", action="store_true", help="preset pilot: 5 固定 variant × 4 sample + mosaic")
    p.add_argument("--range-pilot", action="store_true", help="range pilot: 6 軸 × 5 段階 (1 sample image)")
    p.add_argument("--full", action="store_true", help="full mode: user 承認済 range で連続 random N augment/image")
    p.add_argument("--n-per-source", type=int, default=3, help="full mode: 1 image あたり生成する augment 数")
    p.add_argument("--head", type=int, default=None, help="full mode: 先頭 N image に限定 (smoke 用)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pairs = load_source_pairs(args.img_root, args.lbl_root)
    print(f"[info] sources with workspace: {len(pairs)}", file=sys.stderr)
    if not pairs:
        return 1

    rng = random.Random(args.seed)

    if args.range_pilot:
        run_range_pilot(pairs, args, rng)
        return 0

    if args.pilot:
        run_preset_pilot(pairs, args, rng)
        return 0

    if args.full:
        run_full_random(pairs, args, rng)
        return 0

    print("[error] --pilot / --range-pilot / --full のいずれかを指定してください", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
