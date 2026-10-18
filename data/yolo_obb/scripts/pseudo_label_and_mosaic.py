"""frame dir に対して YOLO-OBB pseudo-label 生成 + overlay mosaic 出力。

Cycle 1 pilot 用: pseudo-label (yolo-obb txt) と overlay png、
そして全 frame の 5xN grid mosaic jpg を出力し、user が一目で QC できるようにする。

Usage:
    cd model/yolo_obb
    pixi run python ../../data/yolo_obb/scripts/pseudo_label_and_mosaic.py \\
        --frames-dir ../../data/yolo_obb/pseudo_source/head_left/frames \\
        --weights runs/m_lowaug_v10_hsv07_imgsz800/weights/best.pt \\
        --out-dir ../../data/yolo_obb/pseudo_source/head_left \\
        --imgsz 800 --conf 0.25 --cols 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_model(weights: Path):
    from ultralytics import YOLO
    return YOLO(str(weights))


def obb_result_to_yolo_txt_lines(obb) -> list[str]:
    """ultralytics OBB result → YOLO-OBB txt format (class x1 y1 x2 y2 x3 y3 x4 y4)。

    xyxyxyxyn (normalized 4 corners) を使う。
    """
    if obb is None or obb.cls is None or obb.cls.shape[0] == 0:
        return []
    lines = []
    classes = obb.cls.cpu().numpy().astype(int)
    corners = obb.xyxyxyxyn.cpu().numpy()  # (n, 4, 2)
    for cls, pts in zip(classes, corners):
        flat = pts.reshape(-1).tolist()
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in flat))
    return lines


def run_predict(model, frames_dir: Path, label_dir: Path, overlay_dir: Path,
                imgsz: int, conf: float) -> list[dict]:
    label_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.png"))
    print(f"predicting on {len(frames)} frames (imgsz={imgsz}, conf={conf})", flush=True)

    metadata = []
    for i, frame_path in enumerate(frames):
        results = model.predict(
            source=str(frame_path), imgsz=imgsz, conf=conf,
            verbose=False, save=False,
        )
        r = results[0]
        obb = getattr(r, "obb", None)

        # save label
        lines = obb_result_to_yolo_txt_lines(obb)
        label_path = label_dir / f"{frame_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        # save overlay
        overlay = r.plot()  # BGR numpy
        overlay_path = overlay_dir / f"{frame_path.stem}.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        n_boxes = int(obb.cls.shape[0]) if obb is not None and obb.cls is not None else 0
        mean_conf = float(obb.conf.mean().item()) if (obb is not None and n_boxes > 0) else 0.0
        classes = obb.cls.tolist() if (obb is not None and n_boxes > 0) else []
        metadata.append({
            "frame": frame_path.name,
            "n_boxes": n_boxes,
            "mean_conf": round(mean_conf, 4),
            "classes": [int(c) for c in classes],
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(frames)}", flush=True)
    return metadata


def build_mosaic(overlay_dir: Path, mosaic_path: Path, cols: int, thumb_w: int = 320) -> None:
    """overlay jpg を cols 列で並べて 1 枚の mosaic に。frame index も焼き込む。"""
    files = sorted(overlay_dir.glob("*.jpg"))
    n = len(files)
    rows = (n + cols - 1) // cols
    print(f"mosaic: {rows} rows x {cols} cols = {rows*cols} slots ({n} frames)", flush=True)

    # thumb は元 640x480 → 縮小 (aspect 保持)
    sample = cv2.imread(str(files[0]))
    h0, w0 = sample.shape[:2]
    thumb_h = int(round(thumb_w * h0 / w0))

    canvas = np.full((rows * thumb_h, cols * thumb_w, 3), 30, dtype=np.uint8)
    for i, f in enumerate(files):
        r, c = divmod(i, cols)
        img = cv2.imread(str(f))
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        # frame name を焼き込む
        cv2.putText(thumb, f.stem, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        canvas[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb

    mosaic_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mosaic_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"mosaic saved: {mosaic_path} ({canvas.shape[1]}x{canvas.shape[0]})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="親 dir。labels/, overlays/, mosaic.jpg, metadata.json を作る")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--thumb-w", type=int, default=320)
    args = parser.parse_args()

    label_dir = args.out_dir / "labels"
    overlay_dir = args.out_dir / "overlays"
    mosaic_path = args.out_dir / "mosaic.jpg"
    metadata_path = args.out_dir / "metadata.json"

    model = load_model(args.weights)
    metadata = run_predict(model, args.frames_dir, label_dir, overlay_dir,
                            args.imgsz, args.conf)

    build_mosaic(overlay_dir, mosaic_path, args.cols, args.thumb_w)

    # summary
    n_boxes_arr = np.array([m["n_boxes"] for m in metadata])
    conf_arr = np.array([m["mean_conf"] for m in metadata if m["n_boxes"] > 0])
    class_counts = {}
    for m in metadata:
        for c in m["classes"]:
            class_counts[c] = class_counts.get(c, 0) + 1

    summary = {
        "n_frames": len(metadata),
        "n_boxes_total": int(n_boxes_arr.sum()),
        "n_boxes_mean": float(n_boxes_arr.mean()),
        "n_boxes_min": int(n_boxes_arr.min()),
        "n_boxes_max": int(n_boxes_arr.max()),
        "frames_with_zero_boxes": int((n_boxes_arr == 0).sum()),
        "mean_conf_over_nonempty": float(conf_arr.mean()) if len(conf_arr) > 0 else 0.0,
        "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
    }
    metadata_out = {"summary": summary, "per_frame": metadata}
    metadata_path.write_text(json.dumps(metadata_out, indent=2))

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nlabels: {label_dir}")
    print(f"overlays: {overlay_dir}")
    print(f"mosaic: {mosaic_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
