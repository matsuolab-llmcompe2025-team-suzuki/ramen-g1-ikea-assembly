"""mp4 から均等 stride で frame 抽出、pseudo-labeling の入力 png dir を作る。

Cycle 1 (pilot): --n 100 で 3218 frame から stride=32 で ~100 frame
Cycle 2 (full): --fps 6 で downsample

Usage:
    cd model/yolo_obb
    # Cycle 1
    pixi run python ../../data/yolo_obb/scripts/extract_frames_for_pseudo.py \\
        --video data/yolo_obb/test_mp4/yolo_test_head_left.mp4 \\
        --out data/yolo_obb/pseudo_source/head_left/frames \\
        --n 100

    # Cycle 2
    pixi run python ../../data/yolo_obb/scripts/extract_frames_for_pseudo.py \\
        --video data/yolo_obb/test_mp4/yolo_test_head_left.mp4 \\
        --out data/yolo_obb/pseudo_source/head_left/frames \\
        --fps 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, out_dir: Path, n: int | None, fps: float | None,
                    offset: int = 0, exclude_frames: set[int] | None = None) -> list[int]:
    """指定 stride で frame 抽出、出力 frame index list を返す。

    offset: stride 内の start 位置ずらし。Cycle 1 未見 frame 抽出用
    (Cycle 1 train = stride 32 offset 0 → QC 用は stride 32 offset 16 で完全に非重複)。
    exclude_frames: 抽出対象から除外する frame index の set (Cycle 2 で cycle1 と重複回避)。
    """
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / src_fps
    print(f"video: {w}x{h}, {src_fps:.1f}fps, {n_frames} frames, {duration:.1f}sec", flush=True)

    if n is not None:
        stride = max(1, n_frames // n)
        print(f"mode: n={n}, stride={stride}, offset={offset} (~{n_frames // stride} frames)", flush=True)
    elif fps is not None:
        stride = max(1, int(round(src_fps / fps)))
        print(f"mode: fps={fps}, stride={stride}, offset={offset} (~{n_frames // stride} frames)", flush=True)
    else:
        raise ValueError("--n または --fps のどちらかが必要")

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    skipped = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= offset and (idx - offset) % stride == 0:
            if exclude_frames is not None and idx in exclude_frames:
                skipped += 1
            else:
                out_path = out_dir / f"frame_{idx:06d}.png"
                cv2.imwrite(str(out_path), frame)
                saved.append(idx)
        idx += 1
    cap.release()
    if skipped:
        print(f"  skipped {skipped} frames (in exclude_frames)", flush=True)

    print(f"saved {len(saved)} frames to {out_dir}", flush=True)
    print(f"  first: frame_{saved[0]:06d}.png", flush=True)
    print(f"  last:  frame_{saved[-1]:06d}.png", flush=True)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=None, help="均等抽出したい枚数")
    parser.add_argument("--fps", type=float, default=None, help="ターゲット fps (downsample)")
    parser.add_argument("--offset", type=int, default=0,
                        help="stride 内 start 位置ずらし (Cycle 1 未見 frame 抽出用)")
    parser.add_argument("--exclude-dir", type=Path, default=None,
                        help="この dir 内の frame_XXXXXX.png と重複する frame は skip (Cycle 2 用)")
    args = parser.parse_args()

    exclude = None
    if args.exclude_dir is not None:
        exclude = set()
        for f in args.exclude_dir.glob("frame_*.png"):
            try:
                exclude.add(int(f.stem.replace("frame_", "")))
            except ValueError:
                pass
        print(f"exclude {len(exclude)} frames from {args.exclude_dir}", flush=True)

    extract_frames(args.video, args.out, args.n, args.fps, args.offset, exclude)


if __name__ == "__main__":
    main()
