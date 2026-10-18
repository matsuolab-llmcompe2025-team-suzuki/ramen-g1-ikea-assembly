"""全 YOLO run (v1 + v2 + v3) を最新 dataset で val 再評価。

Issue #53 の aspect fix の実効果を測定するため、全モデルを同一 val (fixed) で
評価し直す。Retrain 不要、既存 weights を val モードで走らせるだけ。

Usage:
    pixi run -e train python scripts/reval_all_yolo_runs.py \\
        --data data/yolo_obb/configs/dataset.yaml \\
        --out logs/reval_all.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


CONFIGS = [
    "s_baseline", "m_baseline", "l_baseline",
    "s_freeze10", "s_imgsz800", "s_lowaug",
    "m_freeze10", "m_imgsz800", "m_lowaug",
    "l_freeze10", "l_imgsz800", "l_lowaug",
]

SUFFIXES = ["", "_v2", "_v3"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/yolo_obb/configs/dataset.yaml"))
    p.add_argument("--runs-root", type=Path, default=Path("model/yolo_obb/runs"))
    p.add_argument("--out", type=Path, default=Path("logs/reval_all.json"))
    args = p.parse_args()

    from ultralytics import YOLO

    results: dict[str, dict] = {}
    total = len(CONFIGS) * len(SUFFIXES)
    idx = 0

    for suffix in SUFFIXES:
        for cfg in CONFIGS:
            idx += 1
            run_name = f"{cfg}{suffix}"
            weight = args.runs_root / run_name / "weights" / "best.pt"
            if not weight.exists():
                print(f"[{idx}/{total}] {run_name}: SKIP (no weight)")
                continue
            t0 = time.time()
            model = YOLO(str(weight))
            try:
                # val() は run に紐づく既存 dir に書きに行くので dir 汚染回避で
                # project/name を tmp に振る。実 metric は返り値から取る。
                metrics = model.val(
                    data=str(args.data.resolve()),
                    project=str(Path("outputs/yolo_obb/reval").resolve()),
                    name=run_name,
                    exist_ok=True,
                    plots=False,
                    verbose=False,
                )
                map50 = float(metrics.box.map50)
                map50_95 = float(metrics.box.map)
                elapsed = time.time() - t0
                results[run_name] = {
                    "config": cfg,
                    "suffix": suffix or "_v1",
                    "map50": map50,
                    "map50_95": map50_95,
                    "elapsed_sec": round(elapsed, 1),
                }
                print(f"[{idx}/{total}] {run_name}: mAP@0.5={map50:.4f} mAP@0.5:0.95={map50_95:.4f} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"[{idx}/{total}] {run_name}: FAILED — {e}")
                results[run_name] = {"config": cfg, "suffix": suffix or "_v1", "error": str(e)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
