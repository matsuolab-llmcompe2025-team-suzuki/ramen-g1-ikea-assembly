"""frame ごとの skill の表 (build_skill_labels.py) の test (Issue #190)。

以前の ViT の前処理 (公開しない data/vit_phase1/) と同じ決まりで作ること:
30 秒未満の episode を外す、task_index の読み替え、6 を除く、episode 単位の 8:1:1 (seed 42)。
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")

from scripts.build_skill_labels import (  # noqa: E402
    DROP_TASK_INDEX,
    SKILL_REMAP,
    kept_episodes,
    skill_labels,
    split_episodes,
)


def _episodes(lengths: list[int]) -> "pa.Table":
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    return pa.table(
        {
            "episode_index": list(range(len(lengths))),
            "dataset_from_index": starts,
            "dataset_to_index": [s + n for s, n in zip(starts, lengths)],
        }
    )


def test_episodes_shorter_than_30_seconds_are_dropped():
    # 30 fps: 899 frame = 29.97 秒 (外す)、900 frame = 30 秒 (残す)
    assert kept_episodes(_episodes([899, 900, 20000, 10]), fps=30) == [1, 2]


def test_split_is_8_1_1_by_episode_and_reproducible():
    kept = list(range(100, 120))
    split_of = split_episodes(kept)
    assert split_of == split_episodes(kept)
    assert sorted(split_of) == kept
    names = list(split_of.values())
    assert (names.count("train"), names.count("val"), names.count("test")) == (16, 2, 2)


def test_split_puts_the_remainder_in_test():
    names = list(split_episodes(list(range(13))).values())
    assert (names.count("train"), names.count("val"), names.count("test")) == (10, 1, 2)


def test_task_index_is_remapped_and_building_children_table_is_dropped():
    frames = pa.table(
        {
            "episode_index": [0] * 8,
            "frame_index": list(range(8)),
            "task_index": [0, 1, 2, 3, 4, 5, 6, 7],
        }
    )
    table = skill_labels(frames, kept=[0], split_of={0: "train"})
    assert table["skill_id"].to_pylist() == [0, 1, 2, 3, 4, 3, 5]
    assert table["frame_index"].to_pylist() == [0, 1, 2, 3, 4, 5, 7]
    assert table["split"].to_pylist() == ["train"] * 7
    assert str(table["skill_id"].type) == "int8"
    assert (
        set(SKILL_REMAP.values()) == set(range(6))
        and DROP_TASK_INDEX not in SKILL_REMAP
    )


def test_only_kept_episodes_remain_in_episode_then_frame_order():
    frames = pa.table(
        {
            "episode_index": [2, 2, 0, 0, 1],
            "frame_index": [1, 0, 1, 0, 0],
            "task_index": [4, 4, 0, 0, 7],
        }
    )
    table = skill_labels(frames, kept=[0, 2], split_of={0: "val", 2: "test"})
    assert list(
        zip(table["episode_index"].to_pylist(), table["frame_index"].to_pylist())
    ) == [
        (0, 0),
        (0, 1),
        (2, 0),
        (2, 1),
    ]
    assert table["split"].to_pylist() == ["val", "val", "test", "test"]


def test_an_unknown_task_index_is_an_error():
    frames = pa.table({"episode_index": [0], "frame_index": [0], "task_index": [8]})
    with pytest.raises(ValueError, match="unknown task_index"):
        skill_labels(frames, kept=[0], split_of={0: "train"})


def test_the_skill_names_of_the_frame_dump_match_the_remap():
    pytest.importorskip("cv2")
    from scripts.dump_head_frames import SKILL_NAMES

    assert set(SKILL_NAMES) == set(SKILL_REMAP.values())
