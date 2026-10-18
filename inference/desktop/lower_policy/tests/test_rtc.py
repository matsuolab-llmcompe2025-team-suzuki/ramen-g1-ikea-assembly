"""RTC 共有部品の unit test (Issue #137 Phase A)。

重点は `build_velocity_strength` が **本家 GR00T の式と数値一致する**こと。
GR00T 経路では重みの適用が LeRobot 内部 (`policies/groot/groot_n1_7.py`) で
行われるため、我々の式がずれると同じ YAML でも GR00T と RAMEN-Ori の挙動が
別物になり実機比較が成立しない。parity test はその契約を機械的に固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.rtc import (
    AUTO_FROZEN_STEPS,
    DEFAULT_RAMP_RATE,
    ChunkLeftoverBuffer,
    DelayEstimator,
    RtcConfig,
    build_velocity_strength,
)


class TestBuildVelocityStrength:
    def test_three_regions(self) -> None:
        strength = build_velocity_strength(
            chunk_len=16, frozen_steps=3, overlap_steps=8
        )
        assert strength.shape == (16,)
        assert strength.dtype == np.float32
        # 凍結区間は完全に 0 (前 chunk の値がそのまま残る)
        assert np.all(strength[:3] == 0.0)
        # 緩衝区間は (0, 1) の開区間
        assert np.all(strength[3:8] > 0.0)
        assert np.all(strength[3:8] < 1.0)
        # 重ならない区間は自由生成
        assert np.all(strength[8:] == 1.0)

    def test_ramp_is_strictly_increasing(self) -> None:
        strength = build_velocity_strength(
            chunk_len=40, frozen_steps=2, overlap_steps=20
        )
        ramp = strength[2:20]
        assert np.all(np.diff(ramp) > 0.0)

    def test_frozen_zero_has_no_hard_mask(self) -> None:
        strength = build_velocity_strength(
            chunk_len=16, frozen_steps=0, overlap_steps=8
        )
        assert np.all(strength > 0.0)
        assert np.all(strength[8:] == 1.0)

    def test_frozen_equals_overlap_is_pure_hard_mask(self) -> None:
        # ramp 区間が空になる境界。空 slice 代入で落ちないこと。
        strength = build_velocity_strength(
            chunk_len=16, frozen_steps=5, overlap_steps=5
        )
        expected = np.ones(16, dtype=np.float32)
        expected[:5] = 0.0
        np.testing.assert_array_equal(strength, expected)

    def test_overlap_equals_chunk_len(self) -> None:
        strength = build_velocity_strength(
            chunk_len=8, frozen_steps=1, overlap_steps=8
        )
        assert strength.shape == (8,)
        assert strength[0] == 0.0
        # overlap が chunk 全体を覆う場合、自由区間は存在しない
        assert np.all(strength[1:] < 1.0)

    def test_higher_ramp_rate_frees_earlier(self) -> None:
        slow = build_velocity_strength(
            chunk_len=16, frozen_steps=2, overlap_steps=10, ramp_rate=2.0
        )
        fast = build_velocity_strength(
            chunk_len=16, frozen_steps=2, overlap_steps=10, ramp_rate=12.0
        )
        assert np.all(fast[2:10] > slow[2:10])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"chunk_len": 0, "frozen_steps": 0, "overlap_steps": 0}, "chunk_len"),
            ({"chunk_len": 16, "frozen_steps": -1, "overlap_steps": 4}, "frozen_steps"),
            ({"chunk_len": 16, "frozen_steps": 6, "overlap_steps": 4}, "overlap_steps"),
            ({"chunk_len": 16, "frozen_steps": 2, "overlap_steps": 17}, "chunk_len"),
        ],
    )
    def test_invalid_bounds_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            build_velocity_strength(**kwargs)

    @pytest.mark.parametrize("ramp_rate", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_ramp_rate_rejected(self, ramp_rate: float) -> None:
        with pytest.raises(ValueError, match="ramp_rate"):
            build_velocity_strength(
                chunk_len=16, frozen_steps=2, overlap_steps=8, ramp_rate=ramp_rate
            )

    @pytest.mark.parametrize(
        ("chunk_len", "frozen_steps", "overlap_steps", "ramp_rate"),
        [
            (16, 3, 8, 6.0),    # GR00T rotate/insert 実測想定 (d=3、execution 8)
            (16, 0, 8, 6.0),
            (16, 8, 8, 6.0),
            (40, 2, 20, 6.0),   # furniture_groot H40
            (16, 1, 4, 2.5),
        ],
    )
    def test_matches_lerobot_reference(
        self, chunk_len: int, frozen_steps: int, overlap_steps: int, ramp_rate: float
    ) -> None:
        """本家 `groot_n1_7.py` の vel_strength 構築と数値一致すること。

        参照実装は LeRobot の RTC 分岐をそのまま torch で再現したもの。式を
        変更したい場合は、GR00T 側の挙動も同時に変わることを理解した上で
        この test を更新すること。
        """
        torch = pytest.importorskip("torch")

        vel_strength = torch.ones(1, chunk_len, 1)
        vel_strength[:, :frozen_steps, :] = 0.0
        intermediate_steps = overlap_steps - frozen_steps
        t = torch.linspace(0.0, 1.0, intermediate_steps + 2)
        ramp = 1 - torch.exp(-ramp_rate * t)
        ramp = ramp / ramp[-1].clamp_min(1e-8)
        vel_strength[:, frozen_steps:overlap_steps, :] = ramp[1:-1][None, :, None]
        reference = vel_strength[0, :, 0].numpy()

        ours = build_velocity_strength(
            chunk_len=chunk_len,
            frozen_steps=frozen_steps,
            overlap_steps=overlap_steps,
            ramp_rate=ramp_rate,
        )
        np.testing.assert_allclose(ours, reference, rtol=1e-6, atol=1e-6)


class TestDelayEstimator:
    def test_returns_none_without_samples(self) -> None:
        estimator = DelayEstimator()
        assert len(estimator) == 0
        assert estimator.max_latency_s is None
        assert estimator.frozen_steps(0.0333, cap=8) is None

    def test_uses_max_not_mean(self) -> None:
        estimator = DelayEstimator()
        for latency in (0.020, 0.030, 0.098):
            estimator.add(latency)
        assert estimator.max_latency_s == pytest.approx(0.098)
        # mean (0.049) なら 2 tick、max (0.098) なら 3 tick
        assert estimator.frozen_steps(0.0333, cap=8) == 3

    def test_rounds_up(self) -> None:
        estimator = DelayEstimator()
        estimator.add(0.0853)  # 実測 GR00T rotate: 85.3ms / 33.3ms = 2.56 tick
        assert estimator.frozen_steps(0.0333, cap=8) == 3

    def test_ramen_ori_measured_case(self) -> None:
        # 実測 RAMEN-Ori: max 79.8ms / 61.9ms tick = 1.29 tick
        estimator = DelayEstimator()
        estimator.add(0.0798)
        assert estimator.frozen_steps(0.0619, cap=8) == 2

    def test_capped(self) -> None:
        estimator = DelayEstimator()
        estimator.add(10.0)
        assert estimator.frozen_steps(0.0333, cap=8) == 8
        assert estimator.frozen_steps(0.0333, cap=0) == 0

    def test_window_drops_old_samples(self) -> None:
        estimator = DelayEstimator(window=3)
        estimator.add(0.500)
        for _ in range(3):
            estimator.add(0.010)
        assert len(estimator) == 3
        assert estimator.max_latency_s == pytest.approx(0.010)

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_ignores_invalid_samples(self, bad: float) -> None:
        estimator = DelayEstimator()
        estimator.add(bad)
        assert len(estimator) == 0

    def test_reset(self) -> None:
        estimator = DelayEstimator()
        estimator.add(0.05)
        estimator.reset()
        assert estimator.max_latency_s is None

    @pytest.mark.parametrize("tick_period_s", [0.0, -0.01, float("nan")])
    def test_invalid_tick_period_rejected(self, tick_period_s: float) -> None:
        estimator = DelayEstimator()
        estimator.add(0.05)
        with pytest.raises(ValueError, match="tick_period_s"):
            estimator.frozen_steps(tick_period_s, cap=8)

    def test_invalid_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="window"):
            DelayEstimator(window=0)

    def test_default_window_recovers_from_an_outlier(self) -> None:
        # latency は predict 1 回につき 1 サンプル。async replan では 8 tick に
        # 1 回しか predict しないので、window が長いと 1 回の遅い推論が frozen を
        # 上げたまま長く戻らない (実 ckpt smoke で実測)。既定 8 = 約 8 replan 分。
        estimator = DelayEstimator()
        estimator.add(1.5)  # 外れ値
        for _ in range(8):
            estimator.add(0.09)
        assert estimator.max_latency_s == pytest.approx(0.09)


class TestChunkLeftoverBuffer:
    def _chunk(self, chunk_len: int = 16, action_dim: int = 53) -> np.ndarray:
        return np.arange(chunk_len * action_dim, dtype=np.float32).reshape(
            chunk_len, action_dim
        )

    def test_empty_returns_none(self) -> None:
        buffer = ChunkLeftoverBuffer(action_dim=53)
        assert buffer.origin_step is None
        assert buffer.remaining(0) is None

    def test_returns_unconsumed_tail(self) -> None:
        buffer = ChunkLeftoverBuffer(action_dim=53)
        chunk = self._chunk()
        buffer.store(chunk, origin_step=100)
        # 8 tick 消費した時点 (execution_steps=8 の replan タイミング)
        remaining = buffer.remaining(108)
        assert remaining is not None
        assert remaining.shape == (8, 53)
        np.testing.assert_array_equal(remaining, chunk[8:])

    def test_full_chunk_when_nothing_consumed(self) -> None:
        buffer = ChunkLeftoverBuffer(action_dim=53)
        chunk = self._chunk()
        buffer.store(chunk, origin_step=5)
        remaining = buffer.remaining(5)
        assert remaining is not None
        assert remaining.shape == (16, 53)

    def test_none_when_exhausted(self) -> None:
        buffer = ChunkLeftoverBuffer(action_dim=53)
        buffer.store(self._chunk(), origin_step=0)
        assert buffer.remaining(16) is None
        assert buffer.remaining(99) is None

    def test_returns_copy(self) -> None:
        buffer = ChunkLeftoverBuffer(action_dim=53)
        buffer.store(self._chunk(), origin_step=0)
        first = buffer.remaining(4)
        assert first is not None
        first[:] = -1.0
        second = buffer.remaining(4)
        assert second is not None
        assert not np.any(second == -1.0)

    def test_store_copies_input(self) -> None:
        buffer = ChunkLeftoverBuffer(action_dim=53)
        chunk = self._chunk()
        buffer.store(chunk, origin_step=0)
        chunk[:] = -1.0
        remaining = buffer.remaining(0)
        assert remaining is not None
        assert not np.any(remaining == -1.0)

    def test_store_replaces_previous(self) -> None:
        buffer = ChunkLeftoverBuffer(action_dim=53)
        buffer.store(self._chunk(), origin_step=0)
        buffer.store(np.zeros((4, 53), dtype=np.float32), origin_step=8)
        assert buffer.origin_step == 8
        remaining = buffer.remaining(8)
        assert remaining is not None
        assert remaining.shape == (4, 53)

    def test_reset(self) -> None:
        buffer = ChunkLeftoverBuffer(action_dim=53)
        buffer.store(self._chunk(), origin_step=0)
        buffer.reset()
        assert buffer.remaining(0) is None
        assert buffer.origin_step is None

    def test_step_before_origin_rejected(self) -> None:
        buffer = ChunkLeftoverBuffer(action_dim=53)
        buffer.store(self._chunk(), origin_step=10)
        with pytest.raises(ValueError, match="precedes stored origin_step"):
            buffer.remaining(9)

    @pytest.mark.parametrize(
        "bad_shape", [(16,), (16, 19), (16, 53, 1), (0, 53)]
    )
    def test_invalid_chunk_shape_rejected(self, bad_shape: tuple) -> None:
        buffer = ChunkLeftoverBuffer(action_dim=53)
        with pytest.raises(ValueError):
            buffer.store(np.zeros(bad_shape, dtype=np.float32), origin_step=0)

    def test_invalid_action_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="action_dim"):
            ChunkLeftoverBuffer(action_dim=0)


class TestRtcConfig:
    def test_defaults_are_disabled(self) -> None:
        cfg = RtcConfig()
        assert cfg.enabled is False
        assert cfg.frozen_steps == AUTO_FROZEN_STEPS
        assert cfg.overlap_steps is None
        assert cfg.ramp_rate == DEFAULT_RAMP_RATE

    def test_explicit_values(self) -> None:
        cfg = RtcConfig(
            enabled=True, frozen_steps=3, overlap_steps=8, ramp_rate=4.0
        )
        assert cfg.enabled is True
        assert cfg.frozen_steps == 3
        assert cfg.overlap_steps == 8

    def test_is_hashable_for_frozen_policy_config_default(self) -> None:
        # PolicyConfig(frozen=True) の default 値として持たせるため
        assert hash(RtcConfig()) == hash(RtcConfig())

    @pytest.mark.parametrize(
        ("kwargs", "exc", "match"),
        [
            ({"enabled": "yes"}, TypeError, "enabled"),
            ({"frozen_steps": "always"}, TypeError, "frozen_steps"),
            ({"frozen_steps": 1.5}, TypeError, "frozen_steps"),
            ({"frozen_steps": -1}, ValueError, "frozen_steps"),
            ({"overlap_steps": 0}, ValueError, "overlap_steps"),
            ({"overlap_steps": "8"}, TypeError, "overlap_steps"),
            ({"ramp_rate": 0.0}, ValueError, "ramp_rate"),
            ({"ramp_rate": -1.0}, ValueError, "ramp_rate"),
            ({"frozen_steps": 9, "overlap_steps": 8}, ValueError, "frozen_steps"),
        ],
    )
    def test_invalid_values_rejected(
        self, kwargs: dict, exc: type[Exception], match: str
    ) -> None:
        with pytest.raises(exc, match=match):
            RtcConfig(**kwargs)

    def test_auto_frozen_with_overlap_is_allowed(self) -> None:
        # "auto" は実行時決定なので overlap との静的比較はしない
        cfg = RtcConfig(enabled=True, overlap_steps=4)
        assert cfg.frozen_steps == AUTO_FROZEN_STEPS
