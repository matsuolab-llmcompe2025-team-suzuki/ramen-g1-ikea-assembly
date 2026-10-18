"""YOLO OBB txt の annotation を image に polygon 描画して sanity check する。

Epic #43 / Issue #53 (v2 v1 annotate cycle の一部として、LS → YOLO → dataset 変換の
座標ズレを目視確認する用途)。

処理:
1. `data/yolo_obb/dataset/{images,labels}/{split}/{skill}/ep{XX}/*` を pairing
2. YOLO OBB txt (class_id x1 y1 ... x4 y4、0-1 normalized) を parse
3. 各 class に固定色を割り当て、4 頂点 polygon を image に描画 + class 名テキスト
4. `<out>/{skill}__ep{XX}__frame_YY.png` に保存

Usage:
    pixi run -e train python -m data.yolo_obb.scripts.visualize_labels \\
        --dataset data/yolo_obb/dataset --split train --ep 62 --n-per-skill 1 \\
        --out outputs/yolo_obb/labels_viz/
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = [
    "workspace",   # 0
    "leg",         # 1
    "leg_tip",     # 2
    "hole",        # 3
    "table_top",   # 4
    "hand_right",  # 5
    "hand_left",   # 6
]

# labels_config.xml と近い色調 (BGR)
CLASS_COLORS = {
    0: (253, 196, 161),  # workspace (light blue in RGB → BGR here)
    1: (71, 179, 255),   # leg (orange)
    2: (97, 105, 255),   # leg_tip (red)
    3: (125, 221, 119),  # hole (green)
    4: (214, 174, 195),  # table_top (purple)
    5: (150, 253, 253),  # hand_right (yellow)
    6: (194, 154, 244),  # hand_left (pink)
}


def parse_yolo_obb_txt(txt_path: Path) -> list[tuple[int, np.ndarray]]:
    """YOLO OBB txt を [(class_id, verts_normalized(4, 2))] に parse。"""
    boxes = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 9:
            continue
        cid = int(parts[0])
        verts = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(4, 2)
        boxes.append((cid, verts))
    return boxes


def draw_obb_on_image(img: np.ndarray, boxes: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """OBB を image に描画。normalize 座標を pixel に戻して cv2.polylines。"""
    h, w = img.shape[:2]
    out = img.copy()
    for cid, verts_norm in boxes:
        verts_px = (verts_norm * np.array([w, h])).astype(np.int32)
        color = CLASS_COLORS.get(cid, (200, 200, 200))
        cv2.polylines(out, [verts_px], isClosed=True, color=color, thickness=2)
        # class 名テキストを TL 頂点付近 (verts_px[0]) に置く
        label = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
        tl = verts_px[0]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 背景ボックス (可読性のため)
        cv2.rectangle(
            out,
            (tl[0], tl[1] - th - 4),
            (tl[0] + tw + 4, tl[1]),
            color,
            -1,
        )
        cv2.putText(
            out, label, (tl[0] + 2, tl[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return out


def pair_up(images_root: Path, labels_root: Path, ep: int) -> dict[str, list[tuple[Path, Path]]]:
    """{skill: [(img_path, txt_path), ...]} を ep 単位で pair up。"""
    ep_dir = f"ep{ep:04d}"
    result: dict[str, list[tuple[Path, Path]]] = {}
    for skill_dir in sorted(images_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill = skill_dir.name
        ep_img_dir = skill_dir / ep_dir
        ep_txt_dir = labels_root / skill / ep_dir
        if not ep_img_dir.is_dir() or not ep_txt_dir.is_dir():
            continue
        pairs = []
        for img in sorted(ep_img_dir.glob("*.png")):
            txt = ep_txt_dir / f"{img.stem}.txt"
            if txt.exists():
                pairs.append((img, txt))
        if pairs:
            result[skill] = pairs
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", type=Path, default=Path("data/yolo_obb/dataset"))
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--ep", type=int, required=True, help="target episode (e.g. 62)")
    p.add_argument("--n-per-skill", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("outputs/yolo_obb/labels_viz"))
    args = p.parse_args()

    images_root = args.dataset / "images" / args.split
    labels_root = args.dataset / "labels" / args.split
    if not images_root.is_dir():
        raise SystemExit(f"images root not found: {images_root}")
    if not labels_root.is_dir():
        raise SystemExit(f"labels root not found: {labels_root}")

    pairs_by_skill = pair_up(images_root, labels_root, args.ep)
    if not pairs_by_skill:
        raise SystemExit(f"no image/label pairs for ep {args.ep:04d} in {args.split}")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"[cfg] ep={args.ep:04d} split={args.split} n_per_skill={args.n_per_skill}")
    for skill, pairs in pairs_by_skill.items():
        n = min(args.n_per_skill, len(pairs))
        picked = rng.sample(pairs, n)
        for img_path, txt_path in picked:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [skip] cannot read {img_path}")
                continue
            boxes = parse_yolo_obb_txt(txt_path)
            overlay = draw_obb_on_image(img, boxes)
            out_name = f"{skill}__ep{args.ep:04d}__{img_path.stem}.png"
            out_path = args.out / out_name
            cv2.imwrite(str(out_path), overlay)
            print(f"  wrote {out_path} ({len(boxes)} boxes)")

    print(f"[done] output dir: {args.out}")


if __name__ == "__main__":
    main()
