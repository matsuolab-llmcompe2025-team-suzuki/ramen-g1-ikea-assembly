"""会場の腕の表 (venue_arm.py、Issue #188) の test。numpy だけで動く (Mac では --noconftest で回せる)。"""

from __future__ import annotations

import numpy as np
import pytest

from model.ramen_ori.venue_arm import (
    ARM_DIM,
    ARM_SLICE_Q36,
    FPS,
    VenueArmConfig,
    build_tables,
    command19_from,
    replace_arms,
    simulate_episode,
)
from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    UPPER_BODY_SOURCE_INDEX_MAP,
)


def _episode(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """腕がゆっくり往復する教師の指令 (n, 19) と、指令どおりに動いた (硬い) 腕 (n, 14)。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    phase = rng.uniform(0, np.pi, size=ARM_DIM)
    arms = 0.4 * np.sin(2 * np.pi * 0.5 * t[:, None] + phase)
    command = np.zeros((n, 19))
    command[:, 3:17] = arms
    return command, arms.copy()


def _lag_frames(command_arms: np.ndarray, arm: np.ndarray, max_shift: int = 30) -> int:
    """arm が command_arms から何 frame 遅れているか (ずれの平均が最小になる shift)。"""
    errors = [
        np.abs(arm[s:] - command_arms[: len(arm) - s]).mean() for s in range(max_shift)
    ]
    return int(np.argmin(errors))


class TestConfig:
    def test_defaults(self) -> None:
        cfg = VenueArmConfig.from_dict({})
        assert cfg.prob == 0.5 and cfg.preset == "standard" and cfg.eval_index is None
        assert cfg.taus == (0.05, 0.1, 0.15, 0.2, 0.3)

    def test_eval_index(self) -> None:
        assert (
            VenueArmConfig.from_dict({"taus": [0.1, 0.15], "eval_tau": 0.15}).eval_index
            == 1
        )

    @pytest.mark.parametrize(
        "cfg, match",
        [
            ({"probability": 0.5}, "知らない key"),
            ({"prob": 1.5}, "prob"),
            ({"taus": []}, "taus"),
            ({"taus": [0.1, -0.2]}, "taus"),
            ({"taus": [0.1], "eval_tau": 0.15}, "eval_tau"),
        ],
    )
    def test_rejects_bad_values(self, cfg, match) -> None:
        with pytest.raises(ValueError, match=match):
            VenueArmConfig.from_dict(cfg)


def test_arm_slice_matches_the_upper_body_map() -> None:
    """robot_q_current の腕 = [22:36] (胴体の位置・向き 7 + 関節 15..28)。state の腕 (UPPER_BODY の 3:17) と同じ所。"""
    assert ARM_SLICE_Q36 == slice(22, 36)
    assert list(range(36)[ARM_SLICE_Q36]) == list(UPPER_BODY_SOURCE_INDEX_MAP[3:17])


def test_command19_follows_the_inference_order() -> None:
    n = 4
    q_desired = np.arange(n * 36, dtype=np.float32).reshape(n, 36)
    hand = -np.arange(n * 2, dtype=np.float32).reshape(n, 2) - 1
    c19 = command19_from(q_desired, hand)
    np.testing.assert_array_equal(c19[:, 0:3], q_desired[:, 19:22])  # waist
    np.testing.assert_array_equal(c19[:, 3:17], q_desired[:, 22:36])  # 腕
    np.testing.assert_array_equal(c19[:, 17:19], hand)


class TestSimulate:
    @pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
    def test_starts_at_the_recorded_arm(self) -> None:
        command, measured = _episode(60)
        measured[0] += (
            0.05  # 記録された腕の 1 frame 目が指令とずれていても、そこから始まる
        )
        arm = simulate_episode(command, measured, tau=0.15, preset="standard")
        assert arm.shape == (60, ARM_DIM) and arm.dtype == np.float32
        np.testing.assert_allclose(arm[0], measured[0], atol=1e-6)

    @pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
    def test_slower_arm_lags_more(self) -> None:
        command, measured = _episode(150)
        lags = [
            _lag_frames(
                command[:, 3:17],
                simulate_episode(command, measured, tau=tau, preset="standard"),
            )
            for tau in (0.05, 0.15, 0.3)
        ]
        assert lags[0] < lags[1] < lags[2]
        # 0.15 s の腕で数 frame 以上 (09-29 の台を回す 270 ms ≈ 8 frame と同じ桁)
        assert 3 <= lags[1] <= 15

    def test_empty_episode(self) -> None:
        assert simulate_episode(
            np.zeros((0, 19)), np.zeros((0, 14)), 0.15, "standard"
        ).shape == (0, ARM_DIM)


class TestBuildTables:
    @pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
    def test_each_episode_starts_fresh(self) -> None:
        """区間の境目で腕の状態を引き継がない: 表の各区間 = その区間だけを模擬に通した結果。"""
        (c0, m0), (c1, m1) = _episode(40, seed=0), _episode(50, seed=1)
        cfg = VenueArmConfig.from_dict({"taus": [0.1, 0.2], "workers": 1})
        tables = build_tables(
            np.concatenate([c0, c1]), np.concatenate([m0, m1]), [40, 50], cfg
        )
        assert tables.shape == (2, 90, ARM_DIM)
        for k, tau in enumerate(cfg.taus):
            np.testing.assert_allclose(
                tables[k, :40], simulate_episode(c0, m0, tau, "standard")
            )
            np.testing.assert_allclose(
                tables[k, 40:], simulate_episode(c1, m1, tau, "standard")
            )

    @pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
    def test_processes_give_the_same_table(self) -> None:
        episodes = [_episode(30 + 5 * i, seed=i) for i in range(5)]
        command = np.concatenate([c for c, _ in episodes])
        measured = np.concatenate([m for _, m in episodes])
        lengths = [len(c) for c, _ in episodes]
        serial = build_tables(
            command,
            measured,
            lengths,
            VenueArmConfig.from_dict({"taus": [0.1, 0.3], "workers": 1}),
        )
        pooled = build_tables(
            command,
            measured,
            lengths,
            VenueArmConfig.from_dict({"taus": [0.1, 0.3], "workers": 2}),
        )
        np.testing.assert_array_equal(serial, pooled)

    def test_row_count_mismatch_raises(self) -> None:
        command, measured = _episode(10)
        with pytest.raises(ValueError, match="行数"):
            build_tables(
                command, measured, [9], VenueArmConfig.from_dict({"workers": 1})
            )


class TestReplaceArms:
    def test_only_the_arms_change(self) -> None:
        q = np.random.default_rng(0).normal(size=(2, 36)).astype(np.float32)
        cur, prev = np.full(14, 1.0, np.float32), np.full(14, 2.0, np.float32)
        out = replace_arms(q, cur, prev)
        np.testing.assert_array_equal(out[1, 22:36], cur)
        np.testing.assert_array_equal(out[0, 22:36], prev)
        np.testing.assert_array_equal(out[:, :22], q[:, :22])
        assert not np.shares_memory(out, q)

    def test_episode_start_keeps_the_previous_row(self) -> None:
        q = np.random.default_rng(1).normal(size=(2, 36)).astype(np.float32)
        out = replace_arms(q, np.zeros(14, np.float32), None)
        np.testing.assert_array_equal(out[0], q[0])
        np.testing.assert_array_equal(out[1, 22:36], 0.0)
