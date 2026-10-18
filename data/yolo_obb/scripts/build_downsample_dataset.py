"""move_to_table 50% dataset を組む downsample builder。

Issue #108 v5_downsample: v4_r50 (bg augment ×4 で move 51%) との fair 比較用。
bg augment なし・元 image のみ・非 move skill を downsample して move 比率を 50% に。

処理:
1. 既存 dataset の split_manifest.json を読み込む
2. train entries を skill 別に分類
3. move_to_table = 全保持 (68 枚)
4. non-move skill (5 種) = seed=42 で --non-move-per-skill (default 14) 枚 random 抽出
5. val entries はそのまま
6. 新 dataset dir に元 dataset から shutil.copy2 で copy (HF 再 pull せず)
7. split_manifest_v5.json を出力 (再現性)

Total train = 68 + 5×14 = 138 枚、move ratio = 49.3% (≈50%)。
Val = 元 10 枚保持。

Usage (iros_2026_ramen root から):
    pixi run python data/yolo_obb/scripts/build_downsample_dataset.py \\
        --out data/yolo_obb/dataset_v5_downsample \\
        --non-move-per-skill 14 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_MANIFEST = Path("data/yolo_obb/dataset/split_manifest.json")
DEFAULT_SRC_DATASET = Path("data/yolo_obb/dataset")
DEFAULT_OUT = Path("data/yolo_obb/dataset_v5_downsample")

MOVE_SKILL = "move_to_table"
NON_MOVE_SKILLS = ["finalize_leg", "flip_table", "insert_table_leg", "move_table_base", "pick_table_leg"]


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"[FAIL] manifest not found: {path}")
    with open(path) as f:
        return json.load(f)


def group_by_skill(entries: list[dict]) -> dict[str, list[dict]]:
    by_skill: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_skill[e["skill"]].append(e)
    return by_skill


def downsample_train(train_entries: list[dict], n_per_skill: int, seed: int) -> list[dict]:
    """move は全保持、non-move は各 skill から n_per_skill 枚 random 抽出。"""
    by_skill = group_by_skill(train_entries)
    kept: list[dict] = []
    kept.extend(by_skill.get(MOVE_SKILL, []))
    for skill in NON_MOVE_SKILLS:
        pool = by_skill.get(skill, [])
        if len(pool) < n_per_skill:
            print(f"[warn] {skill}: only {len(pool)} available (< {n_per_skill}), taking all", file=sys.stderr)
            kept.extend(pool)
        else:
            # skill 名も seed に混ぜて skill 間独立に (numpy hash と違い python hash は
            # PYTHONHASHSEED 依存だが random.Random は再現性あるので mix 用途に問題ない)
            skill_rng = random.Random(seed + hash(skill) % (2**31))
            idxs = skill_rng.sample(range(len(pool)), n_per_skill)
            kept.extend(pool[i] for i in sorted(idxs))
    return kept


def copy_entries(entries: list[dict], src_dataset: Path, out_root: Path, split: str) -> None:
    """entries の image + label を新 dir に copy。"""
    for e in entries:
        skill = e["skill"]
        ep = f"ep{e['ep']:04d}"
        frame_stem = f"frame_{e['frame']:05d}"
        src_img = src_dataset / "images" / split / skill / ep / f"{frame_stem}.png"
        src_lbl = src_dataset / "labels" / split / skill / ep / f"{frame_stem}.txt"
        if not src_img.exists() or not src_lbl.exists():
            print(f"[warn] source missing: {skill}/{ep}/{frame_stem}", file=sys.stderr)
            continue
        dst_img = out_root / "images" / split / skill / ep / f"{frame_stem}.png"
        dst_lbl = out_root / "labels" / split / skill / ep / f"{frame_stem}.txt"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)


def report(train: list[dict], val: list[dict]) -> None:
    print("\n== dataset summary ==")
    for split_name, entries in [("train", train), ("val", val)]:
        by_skill = group_by_skill(entries)
        print(f"[{split_name}] {len(entries)} entries")
        for skill in sorted(by_skill.keys()):
            print(f"  {skill:20}: {len(by_skill[skill]):4}")
    move_ratio = sum(1 for e in train if e["skill"] == MOVE_SKILL) / max(1, len(train))
    print(f"[train] move_to_table ratio = {move_ratio:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--src-dataset", type=Path, default=DEFAULT_SRC_DATASET)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--non-move-per-skill", type=int, default=14, help="non-move skill から抽出する枚数 (default 14, 総 train ≈ 138)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"[cfg] manifest={args.manifest}, out={args.out}, non_move_per_skill={args.non_move_per_skill}, seed={args.seed}")
    print(f"[cfg] src train={len(manifest['train'])}, src val={len(manifest['val'])}")

    train_kept = downsample_train(manifest["train"], args.non_move_per_skill, args.seed)
    val_kept = list(manifest["val"])

    # 出力 clean (既存 file 混在防止)
    for sub in ("images", "labels"):
        target = args.out / sub
        if target.exists():
            print(f"[clean] removing existing {target}")
            shutil.rmtree(target)

    copy_entries(train_kept, args.src_dataset, args.out, "train")
    copy_entries(val_kept, args.src_dataset, args.out, "val")

    v5_manifest = {
        "seed": args.seed,
        "non_move_per_skill": args.non_move_per_skill,
        "source_manifest": str(args.manifest.resolve()),
        "train": [{"skill": e["skill"], "ep": e["ep"], "frame": e["frame"]} for e in train_kept],
        "val": [{"skill": e["skill"], "ep": e["ep"], "frame": e["frame"]} for e in val_kept],
    }
    (args.out / "split_manifest_v5.json").write_text(json.dumps(v5_manifest, indent=2))
    print(f"[manifest] wrote {args.out}/split_manifest_v5.json")

    report(train_kept, val_kept)


if __name__ == "__main__":
    main()
