"""mp4 動画を複数 YOLO-OBB model で推論比較し、differ score top-N frame を side-by-side mosaic 化。

Issue #108 Phase 1/2 評価用。複数 model (v3 baseline + v4 系 + v5 系) を同 動画で inference し、
model 間で結果が大きく違う frame を pick して視覚比較する。

処理:
1. 各 model best.pt で全 frame predict (ultralytics YOLO.predict stream=True)、per-model mp4 出力
2. frame ごとに detection metadata (n_boxes, mean_conf, class 集合) を model 間で集計
3. differ score = normalized(std(n_boxes)) + normalized(std(mean_conf)) で frame 別に計算
4. differ score top N の frame index を pick
5. mosaic 組立て: row = pick frame、col = original + N model overlay
6. 出力: mosaic.jpg + per-model mp4 + differ_scores.csv (再現用)

Usage (iros_2026_ramen root から model/yolo_obb pixi env で):
    cd model/yolo_obb
    pixi run python ../../data/yolo_obb/scripts/predict_and_compare_video.py \\
        --video /path/to/head_left.mp4 \\
        --runs-root model/yolo_obb/runs \\
        --runs m_lowaug_v3 m_lowaug_v4_flat m_lowaug_v4_r50 m_lowaug_v5_material m_lowaug_v5_downsample \\
        --out outputs/yolo_obb/predict/video_compare \\
        --top-n 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# lazy: ultralytics は heavy import
# from ultralytics import YOLO


def load_model(weights_path: Path):
    from ultralytics import YOLO
    return YOLO(str(weights_path))


def predict_video(
    model, video_path: Path, out_video: Path, imgsz: int, conf: float
) -> list[dict]:
    """1 model で全 frame predict → per-model overlay mp4 保存 + 各 frame の detection metadata 返す。"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  video: {w}x{h}, {fps:.1f}fps, {n_frames} frames", flush=True)

    out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    metadata: list[dict] = []
    stream = model.predict(
        source=str(video_path), stream=True, imgsz=imgsz, conf=conf,
        verbose=False, save=False, show=False,
    )
    for i, r in enumerate(stream):
        overlay = r.plot()  # BGR numpy, boxes + label 付き
        writer.write(overlay)
        # OBB detection の場合 r.obb、AABB なら r.boxes
        det = r.obb if getattr(r, "obb", None) is not None else r.boxes
        n_boxes = int(det.cls.shape[0]) if det is not None and det.cls is not None else 0
        mean_conf = float(det.conf.mean().item()) if (det is not None and n_boxes > 0) else 0.0
        classes = det.cls.tolist() if (det is not None and n_boxes > 0) else []
        metadata.append({"frame": i, "n_boxes": n_boxes, "mean_conf": mean_conf, "classes": classes})
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_frames}", flush=True)
    writer.release()
    return metadata


def compute_differ_scores(all_meta: dict[str, list[dict]]) -> list[dict]:
    """全 model の frame 別 metadata から、frame 別 differ score を算出。

    score = z_normalized(std of n_boxes across models) + z_normalized(std of mean_conf across models)
    """
    n_models = len(all_meta)
    if n_models < 2:
        return []
    model_names = list(all_meta.keys())
    n_frames = min(len(all_meta[m]) for m in model_names)

    box_std = np.zeros(n_frames)
    conf_std = np.zeros(n_frames)
    for i in range(n_frames):
        boxes_per = [all_meta[m][i]["n_boxes"] for m in model_names]
        confs_per = [all_meta[m][i]["mean_conf"] for m in model_names]
        box_std[i] = float(np.std(boxes_per))
        conf_std[i] = float(np.std(confs_per))

    # min-max normalize each std to [0, 1] to combine
    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

    combined = norm(box_std) + norm(conf_std)
    return [
        {
            "frame": i,
            "box_std": float(box_std[i]),
            "conf_std": float(conf_std[i]),
            "score": float(combined[i]),
        }
        for i in range(n_frames)
    ]


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """指定 frame を BGR numpy で取り出す。"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"failed to read frame {frame_idx} from {video_path}")
    return frame


def make_mosaic(
    pick_frames: list[int],
    original_video: Path,
    model_videos: dict[str, Path],
    thumb_w: int,
    thumb_h: int,
    out_path: Path,
) -> None:
    """row = pick_frame、col = original + 各 model overlay の mosaic。"""
    model_names = list(model_videos.keys())
    n_cols = 1 + len(model_names)
    n_rows = len(pick_frames)
    header_h = 28
    canvas = Image.new("RGB", (thumb_w * n_cols, thumb_h * n_rows + header_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), "original", fill=(240, 240, 240))
    for c, name in enumerate(model_names, 1):
        draw.text((c * thumb_w + 10, 6), name, fill=(240, 240, 240))
    for r, frame_idx in enumerate(pick_frames):
        y0 = header_h + r * thumb_h
        # original
        orig = extract_frame(original_video, frame_idx)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig_img = Image.fromarray(orig_rgb).resize((thumb_w, thumb_h), Image.LANCZOS)
        canvas.paste(orig_img, (0, y0))
        draw.text((5, y0 + 5), f"f={frame_idx}", fill=(255, 255, 100))
        # each model overlay
        for c, name in enumerate(model_names, 1):
            frame = extract_frame(model_videos[name], frame_idx)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_img = Image.fromarray(frame_rgb).resize((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(frame_img, (c * thumb_w, y0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)
    print(f"[mosaic] wrote {out_path}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="input mp4")
    p.add_argument("--runs-root", type=Path, default=Path("model/yolo_obb/runs"))
    p.add_argument("--runs", type=str, nargs="+", required=True, help="run 名 list (runs-root 下)")
    p.add_argument("--out", type=Path, required=True, help="出力 root (mosaic + per-model mp4 + csv)")
    p.add_argument("--top-n", type=int, default=10, help="differ score top N frame を mosaic に")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--thumb-w", type=int, default=320)
    p.add_argument("--thumb-h", type=int, default=240)
    args = p.parse_args()

    if not args.video.exists():
        raise SystemExit(f"[fail] video not found: {args.video}")

    args.out.mkdir(parents=True, exist_ok=True)
    per_model_dir = args.out / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Predict per model
    all_meta: dict[str, list[dict]] = {}
    model_videos: dict[str, Path] = {}
    for run_name in args.runs:
        weights = args.runs_root / run_name / "weights" / "best.pt"
        if not weights.exists():
            print(f"[warn] weights not found: {weights}, skipping", flush=True)
            continue
        print(f"[predict] {run_name}", flush=True)
        out_video = per_model_dir / f"{run_name}.mp4"
        model = load_model(weights)
        meta = predict_video(model, args.video, out_video, args.imgsz, args.conf)
        all_meta[run_name] = meta
        model_videos[run_name] = out_video

    if len(all_meta) < 2:
        raise SystemExit("[fail] at least 2 models needed for comparison")

    # 2. Differ score
    scores = compute_differ_scores(all_meta)
    scores.sort(key=lambda s: s["score"], reverse=True)
    top_frames = [s["frame"] for s in scores[:args.top_n]]
    print(f"[differ] top {args.top_n} frames: {top_frames}", flush=True)

    # 3. Save csv
    csv_path = args.out / "differ_scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "box_std", "conf_std", "score"])
        for s in sorted(scores, key=lambda s: s["frame"]):
            w.writerow([s["frame"], f"{s['box_std']:.3f}", f"{s['conf_std']:.3f}", f"{s['score']:.4f}"])
    print(f"[csv] wrote {csv_path}", flush=True)

    # 4. Mosaic
    make_mosaic(top_frames, args.video, model_videos, args.thumb_w, args.thumb_h, args.out / "mosaic.jpg")

    print(f"\n[done] out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
