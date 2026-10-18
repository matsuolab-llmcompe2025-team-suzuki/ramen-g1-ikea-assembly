"""segments.py の unit test (pure function 部分のみ)。

実 LeRobot dataset load / hf_hub download / ffmpeg 実行は Phase B ep0 dogfood で
smoke 確認する (network + heavy)。ここでは pure logic のみ検証。
"""

import pytest

from ui.segments import Segment, detect_segments, segment_tile_cache_path


# --- detect_segments (pure) ---

def test_detect_segments_empty():
    assert detect_segments([]) == []


def test_detect_segments_single_frame():
    assert detect_segments([5]) == [(5, 0, 0)]


def test_detect_segments_all_same():
    assert detect_segments([2, 2, 2, 2]) == [(2, 0, 3)]


def test_detect_segments_two_runs():
    assert detect_segments([2, 2, 3, 3, 3]) == [(2, 0, 1), (3, 2, 4)]


def test_detect_segments_multiple_runs():
    assert detect_segments([2, 2, 3, 3, 3, 2, 2, 1]) == [
        (2, 0, 1), (3, 2, 4), (2, 5, 6), (1, 7, 7),
    ]


def test_detect_segments_alternating():
    assert detect_segments([1, 2, 1, 2, 1]) == [
        (1, 0, 0), (2, 1, 1), (1, 2, 2), (2, 3, 3), (1, 4, 4),
    ]


def test_detect_segments_return_type():
    """return は list of 3-tuple (int, int, int)。"""
    result = detect_segments([0, 0, 1, 1])
    assert isinstance(result, list)
    for t in result:
        assert isinstance(t, tuple)
        assert len(t) == 3
        assert all(isinstance(x, int) for x in t)


def test_detect_segments_realistic_example():
    """G1_WBT の想定 pattern (pick → rotate_leg → insertion 系) を模擬。"""
    seq = [0] * 5 + [1] * 100 + [2] * 300 + [3] * 50 + [2] * 100
    result = detect_segments(seq)
    assert result == [
        (0, 0, 4), (1, 5, 104), (2, 105, 404), (3, 405, 454), (2, 455, 554),
    ]
    # 合計 frame 数一致
    total = sum(e - s + 1 for _, s, e in result)
    assert total == len(seq)


# --- Segment dataclass ---

def test_segment_frame_count_consistency():
    seg = Segment(
        episode_id=0, segment_id=3, task_index=2,
        frame_start=100, frame_end=250, frame_count=151,
    )
    assert seg.frame_count == seg.frame_end - seg.frame_start + 1


def test_segment_single_frame():
    seg = Segment(
        episode_id=0, segment_id=0, task_index=0,
        frame_start=42, frame_end=42, frame_count=1,
    )
    assert seg.frame_count == 1


# --- segment_tile_cache_path ---

def test_segment_tile_cache_path_format(tmp_workspace):
    p = segment_tile_cache_path(tmp_workspace, episode_id=0, segment_id=3)
    assert p.parent.name == "ep000"
    assert p.name == "seg03_tile.mp4"
    assert p.parent.parent.name == "videos_cache"


def test_segment_tile_cache_path_large_indices(tmp_workspace):
    p = segment_tile_cache_path(tmp_workspace, episode_id=532, segment_id=14)
    assert p.parent.name == "ep532"
    assert p.name == "seg14_tile.mp4"


def test_segment_tile_cache_path_is_under_workspace(tmp_workspace):
    p = segment_tile_cache_path(tmp_workspace, episode_id=0, segment_id=0)
    assert tmp_workspace in p.parents
