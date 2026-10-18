"""OBB polygon 単位で前景を切って別背景に合成する copy-paste augmentation。

Issue #77 Phase 1-2 / 1-3。背景 domain gap 対策の合成データ生成。

処理:
1. 元学習画像 + OBB label を読む
2. 背景を resize (短辺=480) → center crop (640×480) で学習解像度に揃える
3. OBB polygon (4点) を rasterize して binary mask を作る
4. feather blur (gaussian) で境界を柔らかく
5. alpha 合成: result = 元 × mask + 背景 × (1 - mask)
6. label file は元をそのままコピー (座標変更なし)

Deterministic per-source ordering:
- 各 source の背景並び順は seed + source path から一意に決まる
- 同じ seed で --n-per-source を変えると、strict subset 関係になる
  (r40=2, r45=3, r50=4 → r40 ⊂ r45 ⊂ r50)
- ablation 比較で「data 追加の効果」だけを純粋に取り出せる

Usage:
    # pilot mode: 4 sources × 5 backgrounds = 20 枚 + mosaic 出力
    cd model/yolo_obb  # PIL 使うので pixi env
    pixi run python ../../data/yolo_obb/augmentation/scripts/copy_paste_bbox.py --pilot

    # full mode: 全 skill × N bg (v4_flat 用)
    pixi run python ../../data/yolo_obb/augmentation/scripts/copy_paste_bbox.py \\
        --n-per-source 3 --out .../v4_flat/

    # full mode: 特定 skill だけ × N bg (v4_r40/45/50 用)
    pixi run python ../../data/yolo_obb/augmentation/scripts/copy_paste_bbox.py \\
        --skill-only move_to_table --n-per-source 3 --out .../v4_r45/

注意:
- 座標は元 OBB のまま (leg 位置動かさない) → label 再計算不要 = アノテーションゼロ
- feather blur は境界の合成感を消すため。デフォルト 5px
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
DEFAULT_OUT = Path("data/yolo_obb/augmentation/synthetic/v4_pilot")
IMG_W, IMG_H = 640, 480

# pilot 時に優先的に選ぶ skill (多様性重視)
PILOT_SKILLS = ["insert_table_leg", "pick_table_leg", "flip_table", "move_to_table"]


def load_source_pairs(img_root: Path, lbl_root: Path) -> list[tuple[Path, Path]]:
    """image + label pair を全部列挙。skill/ep 構造をたどる。"""
    pairs = []
    for img in sorted(img_root.rglob("*.png")):
        rel = img.relative_to(img_root)
        lbl = lbl_root / rel.with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def per_source_bg_order(bgs: list[Path], source_key: str, seed: int) -> list[Path]:
    """source ごとに決定論的な bg 並び順を返す。

    seed + source_key の hashlib.md5 (プロセス跨ぎ deterministic) で local RNG を作る。
    Python 組み込み hash() は PYTHONHASHSEED で run 毎に変わるので使わない。
    同じ seed + source key なら run 間で同じ順序 → --n-per-source 増やしても
    strict subset (r40 ⊂ r45 ⊂ r50) が保証される。
    """
    digest = hashlib.md5(f"{seed}:{source_key}".encode()).digest()
    per_seed = int.from_bytes(digest[:8], "big")
    local_rng = random.Random(per_seed)
    shuffled = bgs.copy()
    local_rng.shuffle(shuffled)
    return shuffled


def pick_pilot_sources(pairs: list[tuple[Path, Path]], rng: random.Random, n: int = 4) -> list[tuple[Path, Path]]:
    """skill を分散させて n 個抽出。"""
    by_skill: dict[str, list[tuple[Path, Path]]] = {}
    for img, lbl in pairs:
        skill = img.parent.parent.name  # .../{skill}/ep{XX}/frame_YY.png
        by_skill.setdefault(skill, []).append((img, lbl))
    picked: list[tuple[Path, Path]] = []
    for skill in PILOT_SKILLS[:n]:
        if skill in by_skill and by_skill[skill]:
            picked.append(rng.choice(by_skill[skill]))
    # 足りなければ他 skill から補充
    remain = [p for s, lst in by_skill.items() if s not in PILOT_SKILLS[:n] for p in lst]
    while len(picked) < n and remain:
        picked.append(rng.choice(remain))
    return picked


def prepare_background(bg_path: Path, target_w: int = IMG_W, target_h: int = IMG_H) -> Image.Image:
    """背景を resize (短辺=target_h) → center crop (target_w × target_h)。"""
    bg = Image.open(bg_path).convert("RGB")
    w, h = bg.size
    # 短辺 = target_h に合わせて等倍 resize
    if h < w:
        new_h = target_h
        new_w = int(w * new_h / h)
    else:
        new_w = target_w
        new_h = int(h * new_w / w)
    # target 4:3 aspect に届かない場合 (短辺リサイズ後の長辺が target_w 未満) は長辺基準に切替
    if new_w < target_w or new_h < target_h:
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
    bg = bg.resize((new_w, new_h), Image.LANCZOS)
    # center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return bg.crop((left, top, left + target_w, top + target_h))


def parse_obb_label(lbl_path: Path, img_w: int, img_h: int) -> list[list[tuple[float, float]]]:
    """label file から polygon (pixel 座標 float、4点) の list を返す。"""
    polygons = []
    for line in lbl_path.read_text().splitlines():
        tokens = line.strip().split()
        if len(tokens) != 9:  # class + 4 points × 2
            continue
        # tokens[0] = class, tokens[1..8] = normalized coords
        coords = [float(t) for t in tokens[1:]]
        poly = [(coords[i] * img_w, coords[i + 1] * img_h) for i in range(0, 8, 2)]
        polygons.append(poly)
    return polygons


def rasterize_polygons(polygons: list[list[tuple[float, float]]], w: int, h: int, feather_px: int = 5) -> np.ndarray:
    """polygon list を 1 枚の binary mask にラスタライズ + feather blur。

    return: shape (h, w) の float32 mask、値域 [0, 1]。
    """
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        # clip 座標 (少しはみ出た polygon の対応)
        clipped = [(max(0, min(w - 1, x)), max(0, min(h - 1, y))) for x, y in poly]
        draw.polygon(clipped, fill=255)
    if feather_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return np.asarray(mask_img, dtype=np.float32) / 255.0


def composite(source: Image.Image, bg: Image.Image, mask: np.ndarray) -> Image.Image:
    """alpha 合成: source * mask + bg * (1 - mask)"""
    src_arr = np.asarray(source, dtype=np.float32)
    bg_arr = np.asarray(bg, dtype=np.float32)
    mask3 = mask[..., None]  # (h, w, 1)
    out = src_arr * mask3 + bg_arr * (1.0 - mask3)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def synth_one(source_img: Path, source_lbl: Path, bg_img: Path, out_img: Path, out_lbl: Path, feather_px: int) -> None:
    """1 枚合成 + label コピー。"""
    src = Image.open(source_img).convert("RGB")
    if src.size != (IMG_W, IMG_H):
        src = src.resize((IMG_W, IMG_H), Image.LANCZOS)
    bg = prepare_background(bg_img)
    polygons = parse_obb_label(source_lbl, IMG_W, IMG_H)
    mask = rasterize_polygons(polygons, IMG_W, IMG_H, feather_px=feather_px)
    result = composite(src, bg, mask)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_img)
    shutil.copyfile(source_lbl, out_lbl)


def make_pilot_mosaic(triples: list[tuple[Path, list[Path], list[Path]]], out_path: Path) -> None:
    """pilot 用 mosaic: 各 row = source | result_1 | result_2 | ... | result_N。

    triples: [(source_img, [bg_paths...], [result_paths...]), ...]
    """
    thumb_w, thumb_h = 320, 240
    if not triples:
        return
    n_results = len(triples[0][2])
    n_cols = 1 + n_results  # source + results
    n_rows = len(triples)
    canvas = Image.new("RGB", (thumb_w * n_cols, thumb_h * n_rows), (30, 30, 30))
    for r, (src, _, results) in enumerate(triples):
        # source (col 0)
        src_img = Image.open(src).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
        canvas.paste(src_img, (0, r * thumb_h))
        for c, res in enumerate(results, 1):
            res_img = Image.open(res).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(res_img, (c * thumb_w, r * thumb_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-root", type=Path, default=DEFAULT_TRAIN_IMG)
    p.add_argument("--lbl-root", type=Path, default=DEFAULT_TRAIN_LBL)
    p.add_argument("--bg-dir", type=Path, default=DEFAULT_BG_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pilot", action="store_true", help="pilot mode: 4 sources × 5 bg = 20 枚 + mosaic")
    p.add_argument("--n-per-source", type=int, default=5, help="full mode: 1 source あたり生成する背景数")
    p.add_argument("--skill-only", type=str, default=None,
                   help="full mode: 特定 skill だけ augment (例: move_to_table)。省略時は全 skill")
    p.add_argument("--feather-px", type=int, default=5, help="mask 境界の gaussian blur px")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pairs = load_source_pairs(args.img_root, args.lbl_root)
    bgs = sorted(list(args.bg_dir.rglob("*.jpg")))
    # preview.jpg などは除外 (先頭 "_" は skip)
    bgs = [b for b in bgs if not b.name.startswith("_")]
    print(f"[info] sources={len(pairs)} backgrounds={len(bgs)}", file=sys.stderr)

    if not pairs or not bgs:
        print("[error] source or background が空", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    out_img_root = args.out / "images"
    out_lbl_root = args.out / "labels"

    if args.pilot:
        sources = pick_pilot_sources(pairs, rng, n=4)
        bgs_per_source = 5
        triples = []
        for src_img, src_lbl in sources:
            # full mode と同じ per_source_bg_order を使う。
            # pilot は full の代表 preview なので determinism を揃える (同 seed で pilot ⊂ full)。
            source_key = str(src_img.relative_to(args.img_root))
            sampled_bgs = per_source_bg_order(bgs, source_key, args.seed)[:bgs_per_source]
            results = []
            for bg_path in sampled_bgs:
                # 出力名: {skill}/{ep}/{frame_stem}_bg-{bg_stem}.png
                rel = src_img.relative_to(args.img_root)
                out_img = out_img_root / rel.parent / f"{rel.stem}_bg-{bg_path.stem}.png"
                out_lbl = out_lbl_root / rel.parent / f"{rel.stem}_bg-{bg_path.stem}.txt"
                synth_one(src_img, src_lbl, bg_path, out_img, out_lbl, args.feather_px)
                results.append(out_img)
            triples.append((src_img, sampled_bgs, results))
            print(f"[pilot] {src_img.name} × {len(sampled_bgs)} bgs done", file=sys.stderr)
        mosaic_path = args.out / "_pilot_mosaic.jpg"
        make_pilot_mosaic(triples, mosaic_path)
        print(f"[done] pilot={sum(len(t[2]) for t in triples)} images", file=sys.stderr)
        print(f"[done] mosaic: {mosaic_path}", file=sys.stderr)
        return 0

    # full mode: skill filter + per-source deterministic bg ordering
    filtered = []
    for src_img, src_lbl in pairs:
        rel = src_img.relative_to(args.img_root)
        skill = rel.parts[0]
        if args.skill_only and skill != args.skill_only:
            continue
        filtered.append((src_img, src_lbl, rel))
    print(f"[filter] skill_only={args.skill_only} → {len(filtered)} sources selected", file=sys.stderr)

    total = 0
    for src_img, src_lbl, rel in filtered:
        source_key = str(rel)
        bg_order = per_source_bg_order(bgs, source_key, args.seed)
        n_take = min(args.n_per_source, len(bg_order))
        for bg_path in bg_order[:n_take]:
            out_img = out_img_root / rel.parent / f"{rel.stem}_bg-{bg_path.stem}.png"
            out_lbl = out_lbl_root / rel.parent / f"{rel.stem}_bg-{bg_path.stem}.txt"
            synth_one(src_img, src_lbl, bg_path, out_img, out_lbl, args.feather_px)
            total += 1
        if total > 0 and total % 100 == 0:
            print(f"[full] {total} synthesized", file=sys.stderr)
    print(f"[done] full={total} images out={args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
