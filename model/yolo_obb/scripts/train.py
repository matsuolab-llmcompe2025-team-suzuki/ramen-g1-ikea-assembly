"""ultralytics YOLO-OBB を fine-tune するトレーナ。

Epic #43 / Issue #46。

- Base model: yolo11{n,s,m,l,x}-obb.pt から選択 (default s)
- Dataset: `data/yolo_obb/configs/dataset.yaml` (Issue #45 で組んだ 90/10 split)
- Freeze: `--freeze N` で先頭 N layer 凍結 (small dataset で backbone overfit 予防)
- wandb logging: project=`iros2026-ramen-yolo`、entity は WANDB_ENTITY 環境変数

Usage (model/yolo_obb sub-workspace 経由で叩く):
    cd model/yolo_obb
    pixi run train-yolo --model s --epochs 100 --batch 8 --imgsz 640 --run-name s_baseline

    # 設定を yaml file 経由で渡す
    pixi run train-yolo --config model/yolo_obb/configs/s_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

# ---- 定数 ----
DATA_YAML = Path("data/yolo_obb/configs/dataset.yaml")
RUNS_ROOT = Path("model/yolo_obb/runs")
WANDB_PROJECT = "iros2026-ramen-yolo"

MODEL_VARIANTS = {
    "n": "yolo11n-obb.pt",
    "s": "yolo11s-obb.pt",
    "m": "yolo11m-obb.pt",
    "l": "yolo11l-obb.pt",
    "x": "yolo11x-obb.pt",
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="yaml config file (他の flag より優先)")
    p.add_argument("--model", type=str, default="s", choices=list(MODEL_VARIANTS.keys()))
    p.add_argument("--data", type=Path, default=DATA_YAML)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--freeze", type=int, default=0, help="先頭 N layer を freeze (0=不使用)")
    p.add_argument("--patience", type=int, default=20, help="early stopping patience")
    p.add_argument("--optimizer", type=str, default="auto")
    p.add_argument("--lr0", type=float, default=None, help="None は ultralytics default")
    p.add_argument("--mosaic", type=float, default=1.0, help="mosaic aug 確率 (1.0=default)")
    p.add_argument("--mixup", type=float, default=0.0, help="mixup aug 確率 (0.0=default)")
    # Augmentation params (None=ultralytics default 保持で既存 config 影響なし、
    # Issue #64 の domain gap 対応で照明 / 幾何 augment 強化用に導入)
    p.add_argument("--degrees", type=float, default=None, help="rotation aug degrees (default 0.0)")
    p.add_argument("--translate", type=float, default=None, help="translate aug fraction (default 0.1)")
    p.add_argument("--scale", type=float, default=None, help="scale aug fraction (default 0.5)")
    p.add_argument("--shear", type=float, default=None, help="shear aug degrees (default 0.0)")
    p.add_argument("--perspective", type=float, default=None, help="perspective aug fraction (default 0.0)")
    p.add_argument("--hsv_h", type=float, default=None, help="HSV hue aug fraction (default 0.015)")
    p.add_argument("--hsv_s", type=float, default=None, help="HSV saturation aug fraction (default 0.7)")
    p.add_argument("--hsv_v", type=float, default=None, help="HSV value aug fraction (default 0.4)")
    p.add_argument("--fliplr", type=float, default=None, help="horizontal flip 確率 (default 0.5、hand_right/left 学習破壊回避で 0.0 指定推奨)")
    p.add_argument("--flipud", type=float, default=None, help="vertical flip 確率 (default 0.0)")
    p.add_argument("--erasing", type=float, default=None, help="random erasing 確率 (default 0.4、object 消失回避で 0.0 指定推奨)")
    p.add_argument("--run-name", type=str, default=None, help="wandb + runs dir 用 name")
    p.add_argument(
        "--run-name-suffix",
        type=str,
        default=None,
        help="run_name に append する suffix (yaml 適用後、v2 batch などで '_v2' 付けて既存を非破壊)",
    )
    p.add_argument("--workers", type=int, default=2, help="dataloader workers (default 2 = WSL2 memory 安全側。大メモリ環境なら config yaml or CLI で明示 override)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        help="既存 run_name dir があっても上書き (default: fail、batch runner の resume 挙動を維持)",
    )
    args = p.parse_args()

    if args.config:
        cfg = load_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k):
                # Path 型 attr は yaml str から Path に coerce (data: 等)
                current = getattr(args, k)
                if isinstance(current, Path) and isinstance(v, str):
                    v = Path(v)
                setattr(args, k, v)

    # suffix は yaml 適用後に append (yaml の run_name を上書きしない)
    if args.run_name_suffix:
        base = args.run_name or args.model  # fallback: run_name 未指定なら model 名
        args.run_name = f"{base}{args.run_name_suffix}"

    return args


def main() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)

    args = parse_args()

    if args.model not in MODEL_VARIANTS:
        raise SystemExit(f"unknown model '{args.model}'")
    model_name = MODEL_VARIANTS[args.model]

    # run name
    if not args.run_name:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_name = f"{args.model}_{ts}"

    print(f"[cfg] model={model_name}, epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}")
    print(f"[cfg] freeze={args.freeze}, patience={args.patience}, lr0={args.lr0}")
    print(f"[cfg] mosaic={args.mosaic}, mixup={args.mixup}, seed={args.seed}")
    aug_overrides = {
        k: getattr(args, k) for k in
        ("degrees", "translate", "scale", "shear", "perspective", "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "erasing")
        if getattr(args, k) is not None
    }
    if aug_overrides:
        print(f"[cfg] aug overrides: {aug_overrides}")
    print(f"[cfg] run_name={args.run_name}, data={args.data}")

    # wandb setup (memory: iros2026-ramen-yolo project, team entity fixed)
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    if os.environ.get("WANDB_ENTITY") in (None, "", "studio-ai"):
        # memory: studio-ai は禁止、team entity 固定
        os.environ["WANDB_ENTITY"] = "ken05-matuo-llm-88_llm_2025_suzuki"

    from ultralytics import YOLO

    model = YOLO(model_name)

    train_kwargs = dict(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        optimizer=args.optimizer,
        mosaic=args.mosaic,
        mixup=args.mixup,
        seed=args.seed,
        project=str(RUNS_ROOT.resolve()),
        name=args.run_name,
        exist_ok=args.force,   # --force で上書き、default False で fail-fast
        deterministic=True,
        plots=True,
        save=True,
        val=True,
    )
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.workers is not None:
        train_kwargs["workers"] = args.workers
    # Augment overrides: None は流さず ultralytics default を維持 (既存 config は無影響)
    for aug_key in ("degrees", "translate", "scale", "shear", "perspective", "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "erasing"):
        val = getattr(args, aug_key)
        if val is not None:
            train_kwargs[aug_key] = val

    print(f"[train] starting: {model_name} on {args.data}")
    results = model.train(**train_kwargs)

    # 結果サマリを json で保存 (config + metrics)
    run_dir = RUNS_ROOT / args.run_name
    metrics = {}
    if hasattr(results, "results_dict"):
        metrics = {k: float(v) for k, v in results.results_dict.items()}
    summary = {
        "run_name": args.run_name,
        "model": model_name,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "metrics": metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] run_dir={run_dir}")
    print(f"[done] key metrics: {metrics}")


if __name__ == "__main__":
    main()
