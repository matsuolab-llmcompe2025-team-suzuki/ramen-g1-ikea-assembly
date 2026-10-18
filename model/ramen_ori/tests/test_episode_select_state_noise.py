"""手順の揃った区間の選別 (episode_select.py) と腕の状態のずらし (state_noise.py) の test (Issue #188)。numpy だけ。"""

from __future__ import annotations

import numpy as np
import pytest

from model.ramen_ori.episode_select import (
    FPS,
    LEFT_SHOULDER_PITCH,
    RIGHT_SHOULDER_PITCH,
    EpisodeSelectConfig,
    select_episodes,
)
from model.ramen_ori.state_noise import StateNoiseConfig, add_arm_offset


def _episode(
    n: int, t_left: int, t_right: int, left_min: float = -1.0, right_min: float = -1.1
) -> np.ndarray:
    """左右の肩 pitch がそれぞれの時刻で最小になる (n, 36)。"""
    q = np.zeros((n, 36), dtype=np.float32)
    q[:, LEFT_SHOULDER_PITCH] = -0.2
    q[:, RIGHT_SHOULDER_PITCH] = -0.2
    q[t_left, LEFT_SHOULDER_PITCH] = left_min
    q[t_right, RIGHT_SHOULDER_PITCH] = right_min
    return q


# 左先・右押・差 1 秒 / 右先 / 右が押さない (-0.5) / 左先・右押・差 3 秒
EPISODES = [
    _episode(300, 100, 130),
    _episode(300, 150, 100),
    _episode(300, 100, 130, right_min=-0.5),
    _episode(300, 50, 140),
]
Q = np.concatenate(EPISODES)
LENGTHS = [len(e) for e in EPISODES]


def test_shoulder_pitch_columns() -> None:
    """robot_q_current の肩 pitch = 胴体の位置・向き 7 + 関節 15 (左) / 22 (右)。"""
    assert (LEFT_SHOULDER_PITCH, RIGHT_SHOULDER_PITCH) == (22, 29)


@pytest.mark.parametrize(
    "cfg, expected",
    [
        ({}, [True, True, True, True]),
        ({"left_peak_first": True}, [True, False, True, True]),
        ({"right_push_max_rad": -0.8}, [True, True, False, True]),
        (
            {"left_peak_first": True, "right_push_max_rad": -0.8},
            [True, False, False, True],
        ),  # A
        (
            {
                "left_peak_first": True,
                "right_push_max_rad": -0.8,
                "max_peak_gap_s": 2.0,
            },
            [True, False, False, False],
        ),  # C
    ],
)
def test_select_rules(cfg, expected) -> None:
    np.testing.assert_array_equal(
        select_episodes(Q, LENGTHS, EpisodeSelectConfig.from_dict(cfg)), expected
    )


def test_peak_gap_boundary_is_inclusive_in_frames() -> None:
    """差がちょうど max_peak_gap_s (60 frame = 2.0 秒) なら残す (秒に直す前の frame で比べる)。"""
    q = _episode(300, 100, 100 + 2 * FPS)
    keep = select_episodes(
        q, [300], EpisodeSelectConfig.from_dict({"max_peak_gap_s": 2.0})
    )
    assert keep.tolist() == [True]


def test_select_rejects_bad_config_and_row_mismatch() -> None:
    with pytest.raises(ValueError, match="知らない key"):
        EpisodeSelectConfig.from_dict({"right_push": -0.8})
    with pytest.raises(ValueError, match="0 以上"):
        EpisodeSelectConfig.from_dict({"max_peak_gap_s": -1})
    with pytest.raises(ValueError, match="行数"):
        select_episodes(Q, [300, 300], EpisodeSelectConfig())


class TestStateNoise:
    def test_offset_is_scaled_and_clipped(self) -> None:
        cfg = StateNoiseConfig.from_dict({"sigma_rad": 0.15, "max_rad": 0.4})
        z = np.array([1.0, -1.0, 2.0, 10.0, -10.0] + [0.0] * 9)
        np.testing.assert_allclose(
            cfg.offset_from_normal(z)[:5], [0.15, -0.15, 0.3, 0.4, -0.4], atol=1e-6
        )

    def test_adds_the_same_offset_to_both_rows_of_the_arms_only(self) -> None:
        q = np.random.default_rng(0).normal(size=(2, 36)).astype(np.float32)
        offset = np.linspace(-0.3, 0.3, 14).astype(np.float32)
        out = add_arm_offset(q, offset)
        np.testing.assert_allclose(
            out[:, 22:36] - q[:, 22:36], np.stack([offset, offset]), atol=1e-6
        )
        np.testing.assert_array_equal(out[:, :22], q[:, :22])
        np.testing.assert_allclose(
            out[1] - out[0], q[1] - q[0], atol=1e-6
        )  # 速度は変わらない
        assert not np.shares_memory(out, q)

    @pytest.mark.parametrize(
        "cfg, match",
        [
            ({"sigma": 0.1}, "知らない key"),
            ({"prob": 2}, "prob"),
            ({"max_rad": 0}, "正の値"),
        ],
    )
    def test_rejects_bad_config(self, cfg, match) -> None:
        with pytest.raises(ValueError, match=match):
            StateNoiseConfig.from_dict(cfg)
