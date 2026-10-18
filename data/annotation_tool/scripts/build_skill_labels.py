"""YOLO の注釈の候補を選ぶための、frame ごとの skill の表を作る (Issue #190)。

`dump_head_frames.py` が skill ごとに同じ数ずつ head camera の画像を抜き出すときに読む。
元データ (`BitRobot/G1_WBT_Dex1_Building-Children-Table`) の frame ごとの `task_index` から、
`episode_index`・`frame_index`・`skill_id`・`split` の 4 列を次の決まりで作る:

- 30 秒未満の episode (収録の中断など) を外す
- `task_index` を 6 class に読み替える (`SKILL_REMAP`)。rotate_table_base (5) は
  rotate_leg_to_tighten (3) とまとめて finalize_leg、move_table_base (7) は 5 にする。
  building_children_table (6) の frame は除く
- episode 単位で train / val / test を 8:1:1 に分ける (seed 42)

以前は ViT の前処理 (公開しない `data/vit_phase1/`) が同じ 4 列を作っていた。決まりはそれと同じ。

Usage (data/annotation_tool で):
    # HF から meta と data の 3 列だけを読む (data 全体は落とさない)
    pixi run python scripts/build_skill_labels.py
    # 手元の LeRobot dataset を使う
    pixi run python scripts/build_skill_labels.py --root /path/to/G1_WBT_Dex1_Building-Children-Table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SOURCE_REPO = "BitRobot/G1_WBT_Dex1_Building-Children-Table"
SOURCE_REVISION = "4b5961c6f8b97ececa75ee73041b30bcbf463ae2"
DEFAULT_OUT = Path("workspace/skill_labels.parquet")

MIN_DURATION_SEC = 30.0
# 元データの task_index → skill_id (dump_head_frames.py の SKILL_NAMES の番号)
SKILL_REMAP: dict[int, int] = {
    0: 0,  # insert_table_leg
    1: 1,  # move_to_table
    2: 2,  # flip_table
    3: 3,  # rotate_leg_to_tighten → finalize_leg
    4: 4,  # pick_table_leg
    5: 3,  # rotate_table_base → finalize_leg (3 とまとめる)
    7: 5,  # move_table_base
}
DROP_TASK_INDEX = 6  # building_children_table (組み立て全体)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)
DATA_COLUMNS = ["episode_index", "frame_index", "task_index"]


def kept_episodes(
    episodes: pa.Table, fps: int, min_duration_sec: float = MIN_DURATION_SEC
) -> list[int]:
    """meta/episodes の表から、min_duration_sec 以上の episode_index を sort して返す。"""
    frame_count = pc.subtract(
        episodes["dataset_to_index"], episodes["dataset_from_index"]
    )
    duration = pc.divide(pc.cast(frame_count, pa.float64()), float(fps))
    kept = episodes.filter(pc.greater_equal(duration, min_duration_sec))
    return sorted(kept["episode_index"].to_pylist())


def split_episodes(
    kept: list[int],
    seed: int = SPLIT_SEED,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> dict[int, str]:
    """episode 単位で train / val / test に分ける (端数は test)。episode_index → split 名。"""
    if not kept:
        raise ValueError("no episode to split")
    perm = np.random.default_rng(seed).permutation(np.array(kept, dtype=np.int64))
    n_train = int(len(perm) * ratios[0])
    n_val = int(len(perm) * ratios[1])
    names = (
        ["train"] * n_train + ["val"] * n_val + ["test"] * (len(perm) - n_train - n_val)
    )
    return {int(ep): name for ep, name in zip(perm.tolist(), names)}


def skill_labels(
    frames: pa.Table, kept: list[int], split_of: dict[int, str]
) -> pa.Table:
    """frame ごとの表 (episode_index・frame_index・task_index) から 4 列の表を作る。

    kept の episode だけを残し、task_index を読み替え、building_children_table の frame を除く。
    並びは episode_index、frame_index の順。
    """
    frames = frames.filter(
        pc.is_in(
            frames["episode_index"], pa.array(kept, type=frames["episode_index"].type)
        )
    )
    order = pc.sort_indices(
        frames, sort_keys=[("episode_index", "ascending"), ("frame_index", "ascending")]
    )
    frames = frames.take(order)

    task_index = frames["task_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    unknown = sorted(
        set(np.unique(task_index).tolist()) - set(SKILL_REMAP) - {DROP_TASK_INDEX}
    )
    if unknown:
        raise ValueError(f"unknown task_index values: {unknown}")
    keep = task_index != DROP_TASK_INDEX
    skill_id = np.array(
        [SKILL_REMAP.get(int(t), -1) for t in task_index[keep]], dtype=np.int8
    )
    episode_index = (
        frames["episode_index"].to_numpy(zero_copy_only=False).astype(np.int64)[keep]
    )
    frame_index = (
        frames["frame_index"].to_numpy(zero_copy_only=False).astype(np.int64)[keep]
    )
    return pa.table(
        {
            "episode_index": pa.array(episode_index),
            "frame_index": pa.array(frame_index),
            "skill_id": pa.array(skill_id),
            "split": pa.array(
                [split_of[int(ep)] for ep in episode_index], type=pa.string()
            ),
        }
    )


def _read_source(root: Path | None) -> tuple[int, pa.Table, pa.Table]:
    """(fps, meta/episodes の表, data の 3 列の表) を返す。root が無ければ HF から必要な所だけ読む。"""
    if root is not None:
        fps = int(json.loads((root / "meta" / "info.json").read_text())["fps"])
        episodes = pa.concat_tables(
            pq.read_table(f)
            for f in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
        )
        frames = pa.concat_tables(
            pq.read_table(f, columns=DATA_COLUMNS)
            for f in sorted(root.glob("data/chunk-*/file-*.parquet"))
        )
        return fps, episodes, frames

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{SOURCE_REPO}@{SOURCE_REVISION}"
    fps = int(json.loads(fs.read_text(f"{base}/meta/info.json"))["fps"])

    def read(pattern: str, columns: list[str] | None = None) -> pa.Table:
        tables = []
        for path in sorted(fs.glob(f"{base}/{pattern}")):
            with fs.open(path, "rb") as f:
                tables.append(pq.read_table(f, columns=columns))
        if not tables:
            raise FileNotFoundError(f"no files match {base}/{pattern}")
        return pa.concat_tables(tables)

    return (
        fps,
        read("meta/episodes/chunk-*/file-*.parquet"),
        read("data/chunk-*/file-*.parquet", DATA_COLUMNS),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="手元の LeRobot dataset (無ければ HF から読む)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    fps, episodes, frames = _read_source(args.root)
    kept = kept_episodes(episodes, fps)
    split_of = split_episodes(kept)
    table = skill_labels(frames, kept, split_of)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    counts = {
        name: sum(1 for s in split_of.values() if s == name)
        for name in ("train", "val", "test")
    }
    print(
        f"[skill_labels] {args.out}: {table.num_rows} frames / episodes kept {len(kept)} of "
        f"{episodes.num_rows} (split {counts})"
    )


if __name__ == "__main__":
    main()
