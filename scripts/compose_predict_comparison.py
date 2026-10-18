"""複数モデルの predict 結果を 2x2 grid に合成して比較用 PNG を出力。

Issue #53 の v3 winner 判定用。manifest.json から image path を pair up。

Layout (2x2):
    +--------------------+--------------------+
    | Original (no box)  | Model 1 (top-right)|
    +--------------------+--------------------+
    | Model 2 (bot-left) | Model 3 (bot-right)|
    +--------------------+--------------------+

Usage:
    pixi run -e train python scripts/compose_predict_comparison.py \\
        --runs l_freeze10_v3,m_baseline_v3,m_lowaug_v3 \\
        --out outputs/yolo_obb/predict/comparison_v3_top3/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_map50(run_name: str, runs_root: Path, reval_json: Path | None = None) -> float | None:
    """reval_all.json (fixed val) を優先。無ければ train summary.json (train 時点 val)。"""
    if reval_json and reval_json.exists():
        d = json.loads(reval_json.read_text())
        entry = d.get(run_name)
        if entry and "map50" in entry:
            return entry["map50"]
    sum_path = runs_root / run_name / "summary.json"
    if not sum_path.exists():
        return None
    d = json.loads(sum_path.read_text())
    return d.get("metrics", {}).get("metrics/mAP50(B)")


def label_bar(width: int, text: str, height: int = 32, bg=(40, 40, 40), fg=(255, 255, 255)) -> np.ndarray:
    """text 用の帯を作る (BGR)。"""
    bar = np.full((height, width, 3), bg, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    x = max(6, (width - tw) // 2)
    y = (height + th) // 2 - 2
    cv2.putText(bar, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1, cv2.LINE_AA)
    return bar


def compose_2x2(cells: list[tuple[np.ndarray, str]]) -> np.ndarray:
    """[img, label] 4 個を 2x2 で並べる。全 image を同一 size に resize しておく。"""
    assert len(cells) == 4
    h, w = cells[0][0].shape[:2]
    rows = []
    for r in range(2):
        row_cells = []
        for c in range(2):
            img, label = cells[r * 2 + c]
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            stacked = np.vstack([label_bar(w, label), img])
            row_cells.append(stacked)
        rows.append(np.hstack(row_cells))
    return np.vstack(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=str, required=True, help="comma-separated run names (3 個)")
    p.add_argument("--predict-root", type=Path, default=Path("outputs/yolo_obb/predict"))
    p.add_argument("--runs-root", type=Path, default=Path("model/yolo_obb/runs"))
    p.add_argument(
        "--reval-json",
        type=Path,
        default=Path("logs/reval_all.json"),
        help="fixed val の re-val 結果 (Issue #53、aspect fix 後の真の val)",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    if len(runs) != 3:
        raise SystemExit(f"need exactly 3 runs, got {len(runs)}")

    # 各 run の manifest 読み込み
    manifests = {}
    map50s = {}
    for run in runs:
        m = json.loads((args.predict_root / run / "manifest.json").read_text())
        manifests[run] = m
        map50s[run] = load_map50(run, args.runs_root, args.reval_json)

    # frame 数一致確認
    n = len(manifests[runs[0]]["results"])
    for run in runs[1:]:
        assert len(manifests[run]["results"]) == n, f"frame count mismatch: {run}"

    args.out.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        # 全 run の同 i 番目 result が同じ元 frame か検証 (seed 固定)
        orig_path = Path(manifests[runs[0]]["results"][i]["frame"])
        for run in runs[1:]:
            other = Path(manifests[run]["results"][i]["frame"])
            assert other.name == orig_path.name, f"frame mismatch at idx {i}: {orig_path.name} vs {other.name}"

        # 元画像 (box なし)
        orig_img = cv2.imread(str(orig_path))
        if orig_img is None:
            print(f"[skip {i}] cannot read {orig_path}")
            continue

        # 3 model の predict PNG (box overlay 済)
        model_imgs = []
        for run in runs:
            pred_path = Path(manifests[run]["results"][i]["out"])
            img = cv2.imread(str(pred_path))
            if img is None:
                print(f"[skip {i}] cannot read {pred_path}")
                img = np.zeros_like(orig_img)
            model_imgs.append(img)

        cells = [
            (orig_img, f"Original: {orig_path.stem}"),
            (model_imgs[0], f"{runs[0]}   mAP@0.5 val = {map50s[runs[0]]:.4f}" if map50s[runs[0]] else runs[0]),
            (model_imgs[1], f"{runs[1]}   mAP@0.5 val = {map50s[runs[1]]:.4f}" if map50s[runs[1]] else runs[1]),
            (model_imgs[2], f"{runs[2]}   mAP@0.5 val = {map50s[runs[2]]:.4f}" if map50s[runs[2]] else runs[2]),
        ]
        composed = compose_2x2(cells)
        out_path = args.out / f"{i:03d}_{orig_path.stem}.png"
        cv2.imwrite(str(out_path), composed)
        print(f"  wrote {out_path}")

    print(f"\n[done] {n} composed images → {args.out}")


if __name__ == "__main__":
    main()
