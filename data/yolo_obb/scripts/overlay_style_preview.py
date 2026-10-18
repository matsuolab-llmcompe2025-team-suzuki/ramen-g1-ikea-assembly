"""Overlay 描画スタイル preview (Issue #122 D-3 pre-work)。

C-11 (Selective overlay) 実装前に、YOLO 出力を画像に重ねた際の視覚 quality を
user が判断するための mosaic を生成する。

# 出力

`<output_dir>/mosaic_conf{c}.jpg` (conf threshold 別に 1 枚) を書き出す。

各 mosaic:
    row = sample frame (N frames)
    col = 6 tiles:
        [ orig 640×480 × 3 thickness (1/2/4) ] + [ resize 224×224 × 3 thickness (1/2/4) ]

これで user が判断する軸:
- 描画スタイル: 全 class 色分け (下記 palette)、fill 無し (枠線のみ)、opaque
- 太さ: 1 / 2 / 4 (どれが認識しやすい / 過剰でない か)
- 解像度: 元 640×480 描画 vs Resize 後 224×224 描画 (raster 粗さ差)
- 信頼度閾値: 0.15 / 0.30 / 0.50 (どこで切ると誤検出が少なく必要検出が残るか)

# Color palette (BGR、cv2 convention)

    workspace   = (128, 128, 128)  gray
    leg         = (  0, 255,   0)  green
    leg_tip     = (255, 255,   0)  cyan
    hole        = (  0,   0, 255)  red
    table_top   = (255,   0,   0)  blue
    hand_right  = (  0, 128, 255)  orange
    hand_left   = (255,   0, 255)  magenta

# Usage (model/yolo_obb pixi env、v11b ckpt 使用)

    cd model/yolo_obb
    pixi run python ../../data/yolo_obb/scripts/overlay_style_preview.py \\
        --yolo-ckpt runs/m_lowaug_v11b/weights/best.pt \\
        --source ../../data/yolo_obb/test_mp4/yolo_test_head_left.mp4 \\
        --output-dir ../../outputs/yolo_obb/overlay_preview/

    # デフォルトは v11b + test_mp4 で走るので上記オプション省略も可
    pixi run python ../../data/yolo_obb/scripts/overlay_style_preview.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np


# BGR (cv2 convention)
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (128, 128, 128),  # workspace  = gray
    1: (0, 255, 0),      # leg        = green
    2: (0, 255, 255),    # leg_tip    = yellow (cyan は leg green と近似で判別困難だったため変更)
    3: (0, 0, 255),      # hole       = red
    4: (255, 0, 0),      # table_top  = blue
    5: (0, 128, 255),    # hand_right = orange
    6: (255, 0, 255),    # hand_left  = magenta
}
CLASS_NAMES: dict[int, str] = {
    0: "workspace",
    1: "leg",
    2: "leg_tip",
    3: "hole",
    4: "table_top",
    5: "hand_right",
    6: "hand_left",
}

REPO_ROOT = Path(__file__).resolve().parents[3]  # iros_2026_ramen/
DEFAULT_CKPT = REPO_ROOT / "model" / "yolo_obb" / "runs" / "m_lowaug_v11b" / "weights" / "best.pt"
DEFAULT_SOURCE = REPO_ROOT / "data" / "yolo_obb" / "test_mp4" / "yolo_test_head_left.mp4"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "yolo_obb" / "overlay_preview"


def pick_frames_from_mp4(mp4_path: Path, n: int, seed: int) -> list[np.ndarray]:
    """mp4 から等間隔サンプリングで n frame ピック (BGR uint8 HWC)。

    先頭/末尾を避けて sensible な frame を選ぶ。seed=42 fixed で決定的。
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open mp4: {mp4_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 先頭 10% / 末尾 10% 除外して中央から等間隔 pick
    lo = int(total * 0.1)
    hi = int(total * 0.9)
    step = max(1, (hi - lo) // max(n, 1))
    indices = [lo + i * step for i in range(n)]
    # deterministic seed 経由で細かい shuffle は避ける (等間隔の方が可読)
    del seed  # 現在未使用、将来 random pick に切替えるときのため保持

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            continue
        frames.append(img)
    cap.release()
    return frames


def pick_frames_from_dir(dir_path: Path, n: int, seed: int) -> list[np.ndarray]:
    """ディレクトリから .jpg / .png を n 枚 pick (seed 固定 shuffle)。"""
    imgs = sorted(list(dir_path.rglob("*.jpg")) + list(dir_path.rglob("*.png")))
    if not imgs:
        raise RuntimeError(f"no jpg/png found under {dir_path}")
    rng = random.Random(seed)
    rng.shuffle(imgs)
    frames: list[np.ndarray] = []
    for p in imgs[:n]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def load_yolo_model(ckpt_path: Path):
    """ultralytics YOLO を lazy import + load。"""
    from ultralytics import YOLO  # noqa: PLC0415

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"YOLO ckpt not found: {ckpt_path}")
    return YOLO(str(ckpt_path))


def predict_obb(model, img_bgr: np.ndarray, imgsz: int, conf: float) -> list[tuple[np.ndarray, float, int]]:
    """YOLO OBB 推論、[(verts (4,2) pixel、conf、class_id)] を返す (conf 降順)。"""
    results = model(img_bgr, imgsz=imgsz, conf=conf, verbose=False)
    r = results[0]
    if r.obb is None or r.obb.cls is None or len(r.obb.cls) == 0:
        return []
    verts_pix = r.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2) pixel coords
    conf_arr = r.obb.conf.cpu().numpy()
    cls_arr = r.obb.cls.cpu().numpy()
    dets = []
    for i in range(len(cls_arr)):
        dets.append((verts_pix[i], float(conf_arr[i]), int(cls_arr[i])))
    dets.sort(key=lambda d: -d[1])
    return dets


def draw_overlay(
    img_bgr: np.ndarray,
    dets: list[tuple[np.ndarray, float, int]],
    thickness: int,
    class_filter: set[int] | None = None,
) -> np.ndarray:
    """OBB を色分けで img_bgr に描画 (opaque 枠線のみ、fill なし)。

    verts は 4-point OBB (回転矩形) を polylines で描画。
    class_filter が指定されていれば該当 class_id のみ描画 (default 全 class)。
    """
    out = img_bgr.copy()
    for verts, conf, cls_id in dets:
        if class_filter is not None and cls_id not in class_filter:
            continue
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        pts = verts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out


def build_tile(img: np.ndarray, label: str, target_size: tuple[int, int] | None) -> np.ndarray:
    """image に上部 label 帯 (30 px 黒地に白文字) を貼って return。target_size 指定で resize。"""
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    header_h = 30
    canvas = np.zeros((H + header_h, W, 3), dtype=np.uint8)
    canvas[:header_h] = (40, 40, 40)
    canvas[header_h:] = img
    cv2.putText(
        canvas, label, (5, header_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA,
    )
    return canvas


def make_legend_row(cols_width: int, tile_height: int) -> np.ndarray:
    """色 palette 凡例 row。"""
    canvas = np.zeros((tile_height + 30, cols_width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    cv2.putText(
        canvas, "Class colors (BGR)", (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA,
    )
    # 色サンプル + name を左から並べる
    x = 5
    y = 50
    for cls_id in sorted(CLASS_COLORS_BGR.keys()):
        color = CLASS_COLORS_BGR[cls_id]
        name = CLASS_NAMES[cls_id]
        cv2.rectangle(canvas, (x, y - 15), (x + 20, y + 5), color, -1)
        cv2.putText(
            canvas, name, (x + 25, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA,
        )
        x += 130
        if x + 130 > cols_width:
            break
    return canvas


def build_mosaic_for_conf(
    frames: list[np.ndarray],
    model,
    imgsz: int,
    conf: float,
    thicknesses: list[int],
    resize_target: tuple[int, int],
) -> np.ndarray:
    """1 mosaic (1 conf threshold 分)。row = frame、col = 元 × thicknesses + resize × thicknesses。"""
    rows: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        dets = predict_obb(model, frame, imgsz=imgsz, conf=conf)
        tiles: list[np.ndarray] = []
        # 元解像度で描画 (thickness 別)
        for t in thicknesses:
            drawn = draw_overlay(frame, dets, thickness=t)
            tiles.append(build_tile(drawn, f"orig t={t}", target_size=None))
        # 224×224 に resize してから描画 (thickness 別)、比較用
        # 注意: この case は "resize → overlay 描画" の順 (raster が粗くなる想定)
        resized = cv2.resize(frame, resize_target, interpolation=cv2.INTER_AREA)
        # verts を resize scale で縮小
        scale_x = resize_target[0] / frame.shape[1]
        scale_y = resize_target[1] / frame.shape[0]
        dets_resized = [
            (v * np.array([[scale_x, scale_y]], dtype=np.float32), c, cid)
            for v, c, cid in dets
        ]
        for t in thicknesses:
            drawn_r = draw_overlay(resized, dets_resized, thickness=t)
            tiles.append(build_tile(drawn_r, f"224 t={t}", target_size=None))
        # 各 tile の高さは frame 元解像度 + 30、224 tile は 224 + 30。
        # 縦の高さを合わせるため 224 tile を frame_h + 30 まで zero-pad (下寄せ)。
        max_h = max(t.shape[0] for t in tiles)
        padded = []
        for t in tiles:
            h_diff = max_h - t.shape[0]
            if h_diff > 0:
                pad = np.zeros((h_diff, t.shape[1], 3), dtype=np.uint8)
                t = np.vstack([t, pad])
            padded.append(t)
        # 横に concat (各 tile の width が違う場合は右詰めで concat)
        # まず全 tile 幅を揃える (min-width) — orig tile と resize tile の幅が違う
        max_w = max(t.shape[1] for t in padded)
        aligned = []
        for t in padded:
            w_diff = max_w - t.shape[1]
            if w_diff > 0:
                pad = np.zeros((t.shape[0], w_diff, 3), dtype=np.uint8)
                t = np.hstack([t, pad])
            aligned.append(t)
        row = np.hstack(aligned)
        # 行頭に sample 番号 label を貼る
        header = np.zeros((row.shape[0], 60, 3), dtype=np.uint8)
        header[:] = (60, 60, 60)
        cv2.putText(
            header, f"#{i}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA,
        )
        row = np.hstack([header, row])
        rows.append(row)
    mosaic = np.vstack(rows)
    # 凡例 row を上に貼る
    legend = make_legend_row(mosaic.shape[1], tile_height=30)
    # legend の幅を mosaic に合わせる
    if legend.shape[1] != mosaic.shape[1]:
        legend = cv2.resize(legend, (mosaic.shape[1], legend.shape[0]))
    return np.vstack([legend, mosaic])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yolo-ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help="mp4 file OR directory of jpg/png (auto-detect)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sample-n", type=int, default=8, help="frame count to pick")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--conf-thresholds", type=str, default="0.15,0.30,0.50")
    p.add_argument("--line-thicknesses", type=str, default="1,2,4")
    p.add_argument("--resize-target", type=str, default="224,224", help="H,W for the resize comparison tile")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source.is_file():
        print(f"[pick] mp4 source: {args.source}")
        frames = pick_frames_from_mp4(args.source, args.sample_n, args.seed)
    elif args.source.is_dir():
        print(f"[pick] dir source: {args.source}")
        frames = pick_frames_from_dir(args.source, args.sample_n, args.seed)
    else:
        raise FileNotFoundError(f"source not found: {args.source}")
    if not frames:
        raise RuntimeError("no frames picked")
    print(f"[pick] picked {len(frames)} frames (shape={frames[0].shape})")

    print(f"[load] YOLO ckpt: {args.yolo_ckpt}")
    model = load_yolo_model(args.yolo_ckpt)

    conf_list = [float(c) for c in args.conf_thresholds.split(",")]
    thick_list = [int(t) for t in args.line_thicknesses.split(",")]
    resize_h, resize_w = (int(x) for x in args.resize_target.split(","))
    resize_target = (resize_w, resize_h)  # cv2 は (W, H)

    print(f"[cfg] conf thresholds: {conf_list}")
    print(f"[cfg] line thicknesses: {thick_list}")
    print(f"[cfg] resize target: {resize_target}")

    for conf in conf_list:
        print(f"[mosaic] conf={conf} ...", flush=True)
        mosaic = build_mosaic_for_conf(
            frames, model,
            imgsz=args.imgsz, conf=conf,
            thicknesses=thick_list,
            resize_target=resize_target,
        )
        out_path = args.output_dir / f"mosaic_conf{conf:.2f}.jpg"
        cv2.imwrite(str(out_path), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"[mosaic] wrote {out_path} (shape={mosaic.shape})")

    print(f"[done] preview at {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
