"""学習済 YOLO-OBB weight を使って test frames に inference + 描画 PNG 保存。

Epic #43 / Issue #46。

- Weight: `model/yolo_obb/runs/<run_name>/weights/best.pt` を指す
- Test frames: HF frames repo から 未見 ep の N frame をランダム抽出
- 出力: `outputs/yolo_obb/predict/<run_name>/frame_XX.png` (box overlay 付き)

Usage (model/yolo_obb sub-workspace 経由で叩く):
    cd model/yolo_obb
    pixi run predict-yolo --run-name s_baseline --frames-ep 63 --n 30
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import HfApi, hf_hub_download

# 公開用 dataset repo (Issue #190)。注釈を付けた画像だけがある
HF_FRAMES_REPO = "Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb"
RUNS_ROOT = Path("model/yolo_obb/runs")
PRED_OUT_ROOT = Path("outputs/yolo_obb/predict")

SKILL_NAMES = [
    "insert_table_leg", "move_to_table", "flip_table",
    "finalize_leg", "pick_table_leg", "move_table_base",
]


def sample_test_frames(ep: int, n: int, seed: int, cache: Path, token: str | None) -> list[Path]:
    """指定 ep から HF frames を pull、skill stratified で n 個ランダム抽出。"""
    api = HfApi(token=token)
    files = api.list_repo_files(HF_FRAMES_REPO, repo_type="dataset")
    pat = re.compile(rf"frames/([^/]+)/ep{ep:04d}/frame_\d+\.png$")
    by_skill: dict[str, list[str]] = {}
    for f in files:
        m = pat.match(f)
        if m:
            by_skill.setdefault(m.group(1), []).append(f)
    if not by_skill:
        raise SystemExit(f"no frames on HF for ep {ep:04d}")

    rng = np.random.default_rng(seed)
    per_skill = max(1, n // len(by_skill))
    picked: list[str] = []
    for skill, paths in sorted(by_skill.items()):
        paths_sorted = sorted(paths)
        take = min(per_skill, len(paths_sorted))
        idxs = rng.choice(len(paths_sorted), size=take, replace=False)
        picked.extend(paths_sorted[int(i)] for i in idxs)

    # 足りない分は unused から追加補完
    remaining = n - len(picked)
    if remaining > 0:
        picked_set = set(picked)
        pool = [p for paths in by_skill.values() for p in paths if p not in picked_set]
        if pool:
            take = min(remaining, len(pool))
            extra_idxs = rng.choice(len(pool), size=take, replace=False)
            picked.extend(pool[int(i)] for i in extra_idxs)

    # local DL
    local_paths: list[Path] = []
    for hf_path in picked[:n]:
        local = hf_hub_download(
            repo_id=HF_FRAMES_REPO,
            repo_type="dataset",
            filename=hf_path,
            local_dir=str(cache),
            token=token,
        )
        local_paths.append(Path(local))
    return local_paths


def main() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", type=str, required=True, help="model/yolo_obb/runs/<run-name>/weights/best.pt を使う")
    p.add_argument("--frames-ep", type=int, default=63, help="test 用 ep 番号 (未見前提)")
    p.add_argument("--n", type=int, default=30, help="抽出 frame 数")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence 閾値")
    p.add_argument("--cache", type=Path, default=Path("outputs/yolo_obb/.hf_cache"))
    args = p.parse_args()

    weight_path = RUNS_ROOT / args.run_name / "weights" / "best.pt"
    if not weight_path.exists():
        raise SystemExit(f"weight not found: {weight_path}")

    out_dir = PRED_OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] weight={weight_path}, ep={args.frames_ep}, n={args.n}")

    from ultralytics import YOLO

    model = YOLO(str(weight_path))

    # sample test frames
    print(f"[pull] sampling {args.n} test frames from ep {args.frames_ep:04d}")
    frames = sample_test_frames(args.frames_ep, args.n, args.seed, args.cache, token=None)
    print(f"[pull] got {len(frames)} frames")

    # predict + save overlays
    all_results = []
    for i, frame_path in enumerate(frames):
        results = model(str(frame_path), conf=args.conf, verbose=False)
        r = results[0]
        # ultralytics native plot returns numpy image with boxes drawn
        annotated_np = r.plot()  # BGR
        import cv2
        # save
        stem = Path(frame_path).stem
        out_path = out_dir / f"{i:03d}_{stem}.png"
        cv2.imwrite(str(out_path), annotated_np)
        # collect detection info
        boxes_info = []
        if r.obb is not None:
            for cls_id, conf, xyxyxyxy in zip(
                r.obb.cls.cpu().numpy(),
                r.obb.conf.cpu().numpy(),
                r.obb.xyxyxyxyn.cpu().numpy(),
            ):
                boxes_info.append(
                    {
                        "class_id": int(cls_id),
                        "confidence": float(conf),
                        "verts": xyxyxyxy.tolist(),
                    }
                )
        all_results.append(
            {
                "frame": str(frame_path),
                "out": str(out_path),
                "n_boxes": len(boxes_info),
                "boxes": boxes_info,
            }
        )

    manifest = {
        "run_name": args.run_name,
        "weight": str(weight_path),
        "test_ep": args.frames_ep,
        "n_frames": len(frames),
        "conf": args.conf,
        "results": all_results,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {len(frames)} annotated PNG under {out_dir}")
    print(f"[done] manifest: {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
