"""HF から YOLO v11b 学習 dataset を full rebuild する (local cleanup 後の復元用)。

必要な HF の dataset repo `Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb` (画像とラベル、Issue #190):
  - `frames/{skill}/{ep}/frame_XXXXXX.png` (v10 real の source)
  - `augmentation/synthetic/v5_material_random/{skill}/{ep}/*.png`
  - `head_left/frames/frame_XXXXXX.png` (724 flat、cycle1+2 合体)
  - `yolo_obb/labels/{skill}/{ep}/frame_XXXXXX.txt` (v10 real labels)
  - `yolo_obb/labels/augmentation/synthetic/{name}/{skill}/{ep}/*.txt`
  - `yolo_obb/labels/head_left/frame_XXXXXX.txt` (724 flat)

処理:
1. v10 real (966 image + label) は既存 `build_dataset.py` を呼ぶ (HF pull + skill stratified split)
2. augmentation: HF にある v5 を snapshot_download → 適切な images/ labels/ subdir に配置。
   v4_flat・v8_zoomout_random は背景が Places365 (再配布できない) なので HF に無い →
   Places365 を自分で取ってきて `data/yolo_obb/augmentation/scripts/` で作る
3. head_left: snapshot_download → cycle1 (multiples of 32) / cycle2 (rest) に split して配置

出力 layout (dataset_v11b.yaml が期待する形):
    data/yolo_obb/
    ├── dataset/{images,labels}/{train,val}/{skill}/{ep}/*.{png,txt}   (v10 real)
    ├── augmentation/synthetic/{name}/{images,labels}/{skill}/{ep}/*.{png,txt}
    └── head_left/{cycle1,cycle2}/{images,labels}/frame_XXXXXX.{png,txt}

Usage (annotation_tool env で叩く、huggingface_hub 使用):
    cd data/annotation_tool
    pixi run python ../yolo_obb/scripts/rebuild_from_hf.py

    # v10 real は skip (既に build 済) したい時
    pixi run python ../yolo_obb/scripts/rebuild_from_hf.py --skip-real
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from huggingface_hub import snapshot_download

# 画像とラベルは同じ dataset repo (Issue #190)
FRAMES_REPO = ANNOT_REPO = "Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb"

# HF から取る合成。Places365 の背景の合成は HF に無い (自分で作る)
USED_AUG = ["v5_material_random"]
PLACES365_AUG = ["v4_flat", "v8_zoomout_random"]
CYCLE1_STRIDE = 32  # cycle1 = multiples of 32 (v11a bootstrap のときの stride)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
YOLO_OBB_ROOT = REPO_ROOT / "data" / "yolo_obb"


def run_build_dataset() -> None:
    """v10 real dataset は既存 build_dataset.py を呼ぶ。

    `pixi run -e train python -m data.yolo_obb.scripts.build_dataset` 相当を
    subprocess で叩く (rebuild script 側で ultralytics env を仮定しないため)。
    """
    print("=== 1. v10 real dataset (build_dataset.py 呼び出し) ===", flush=True)
    cmd = [
        sys.executable, "-m", "data.yolo_obb.scripts.build_dataset",
        "--out", str(YOLO_OBB_ROOT / "dataset"),
        "--val-ratio", "0.10",
        "--seed", "42",
        "--train-only-eps", "62,120,510,513,515,517,518,519,531",
    ]
    print(f"  running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def rebuild_augmentation(token: str) -> None:
    """HF にある合成 (v5) を pull → images/ labels/ subdir に再構成。"""
    print("\n=== 2. augmentation (HF にある v5) ===", flush=True)

    aug_patterns_frames = [f"augmentation/synthetic/{n}/**/*.png" for n in USED_AUG]
    aug_patterns_labels = [f"yolo_obb/labels/augmentation/synthetic/{n}/**/*.txt" for n in USED_AUG]

    tmp_frames = YOLO_OBB_ROOT / "_hf_pull_frames"
    tmp_labels = YOLO_OBB_ROOT / "_hf_pull_labels"

    print(f"  snapshot_download frames ({len(USED_AUG)} dirs) → {tmp_frames}", flush=True)
    snapshot_download(
        repo_id=FRAMES_REPO, repo_type="dataset",
        local_dir=str(tmp_frames),
        allow_patterns=aug_patterns_frames,
        token=token,
    )
    print(f"  snapshot_download labels ({len(USED_AUG)} dirs) → {tmp_labels}", flush=True)
    snapshot_download(
        repo_id=ANNOT_REPO, repo_type="dataset",
        local_dir=str(tmp_labels),
        allow_patterns=aug_patterns_labels,
        token=token,
    )

    # Reorganize: HF は augmentation/synthetic/{name}/{skill}/{ep}/... で
    # local は augmentation/synthetic/{name}/{images,labels}/{skill}/{ep}/... が要る
    for name in USED_AUG:
        # images: tmp_frames/augmentation/synthetic/{name}/ → augmentation/synthetic/{name}/images/
        src = tmp_frames / "augmentation" / "synthetic" / name
        dst = YOLO_OBB_ROOT / "augmentation" / "synthetic" / name / "images"
        _mv_dir(src, dst)
        n_img = len(list(dst.rglob("*.png")))

        # labels: tmp_labels/yolo_obb/labels/augmentation/synthetic/{name}/ → augmentation/synthetic/{name}/labels/
        src_l = tmp_labels / "yolo_obb" / "labels" / "augmentation" / "synthetic" / name
        dst_l = YOLO_OBB_ROOT / "augmentation" / "synthetic" / name / "labels"
        _mv_dir(src_l, dst_l)
        n_lbl = len(list(dst_l.rglob("*.txt")))
        print(f"    {name}: {n_img} images + {n_lbl} labels", flush=True)

    _rm_dir(tmp_frames)
    _rm_dir(tmp_labels)
    print(
        f"    {', '.join(PLACES365_AUG)}: HF に無い (背景が Places365 で再配布できない)。"
        " dataset_v11b.yaml で学習するなら、Places365 を取ってきて"
        " data/yolo_obb/augmentation/scripts/ で作る",
        flush=True,
    )


def rebuild_head_left(token: str) -> None:
    """head_left 724 (flat) → cycle1 (frame%32==0) / cycle2 (rest) に split して配置。"""
    print("\n=== 3. head_left (split into cycle1/cycle2) ===", flush=True)

    tmp_frames = YOLO_OBB_ROOT / "_hf_pull_head_frames"
    tmp_labels = YOLO_OBB_ROOT / "_hf_pull_head_labels"

    print(f"  snapshot_download head_left frames → {tmp_frames}", flush=True)
    snapshot_download(
        repo_id=FRAMES_REPO, repo_type="dataset",
        local_dir=str(tmp_frames),
        allow_patterns=["head_left/frames/*.png"],
        token=token,
    )
    print(f"  snapshot_download head_left labels → {tmp_labels}", flush=True)
    snapshot_download(
        repo_id=ANNOT_REPO, repo_type="dataset",
        local_dir=str(tmp_labels),
        allow_patterns=["yolo_obb/labels/head_left/*.txt"],
        token=token,
    )

    src_frames = tmp_frames / "head_left" / "frames"
    src_labels = tmp_labels / "yolo_obb" / "labels" / "head_left"

    cycle_dirs = {
        "cycle1": (YOLO_OBB_ROOT / "head_left" / "cycle1"),
        "cycle2": (YOLO_OBB_ROOT / "head_left" / "cycle2"),
    }
    counts = {c: {"img": 0, "lbl": 0} for c in cycle_dirs}
    for c, d in cycle_dirs.items():
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)

    for f in sorted(src_frames.glob("frame_*.png")):
        frame_num = int(f.stem.replace("frame_", ""))
        cycle = "cycle1" if frame_num % CYCLE1_STRIDE == 0 else "cycle2"
        shutil.move(str(f), str(cycle_dirs[cycle] / "images" / f.name))
        counts[cycle]["img"] += 1

    for f in sorted(src_labels.glob("frame_*.txt")):
        frame_num = int(f.stem.replace("frame_", ""))
        cycle = "cycle1" if frame_num % CYCLE1_STRIDE == 0 else "cycle2"
        shutil.move(str(f), str(cycle_dirs[cycle] / "labels" / f.name))
        counts[cycle]["lbl"] += 1

    for c in cycle_dirs:
        print(f"    {c}: {counts[c]['img']} images + {counts[c]['lbl']} labels", flush=True)

    _rm_dir(tmp_frames)
    _rm_dir(tmp_labels)


def _mv_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _rm_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not found in env / .env")

    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-real", action="store_true",
                    help="v10 real dataset (build_dataset.py) を skip")
    p.add_argument("--skip-aug", action="store_true", help="augmentation を skip")
    p.add_argument("--skip-head-left", action="store_true", help="head_left を skip")
    args = p.parse_args()

    if not args.skip_real:
        run_build_dataset()
    if not args.skip_aug:
        rebuild_augmentation(token)
    if not args.skip_head_left:
        rebuild_head_left(token)

    print("\n=== rebuild done ===", flush=True)
    print("次: cd model/yolo_obb && pixi run train-yolo --config model/yolo_obb/configs/m_lowaug_v11b.yaml")


if __name__ == "__main__":
    main()
