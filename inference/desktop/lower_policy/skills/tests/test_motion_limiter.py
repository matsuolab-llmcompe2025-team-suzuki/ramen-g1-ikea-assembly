"""MotionLimiter (Phase 3、Issue #128) の contract test。

issue-70 UpperBodySafetyLimiter port の velocity + acceleration + position clip
3 段パイプラインの correctness。実 VlaSkill 統合は test_vla_skill.py 側で verify。
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.skills.motion_limiter import MotionLimiter


DIM = 19
ARM_SLICE = slice(0, 17)   # waist3 + arm14
HAND_SLICE = slice(17, 19)  # dex1 L/R


def _make(
    arm_velocity: float = 2.0,
    arm_acceleration: float = 10.0,
    hand_velocity: float = 6.0,
    hand_acceleration: float = 30.0,
    control_hz: float = 30.0,
    lower=None,
    upper=None,
    reject_position_slice=None,
) -> MotionLimiter:
    return MotionLimiter(
        dim=DIM,
        arm_slice=ARM_SLICE,
        hand_slice=HAND_SLICE,
        arm_velocity=arm_velocity,
        arm_acceleration=arm_acceleration,
        hand_velocity=hand_velocity,
        hand_acceleration=hand_acceleration,
        control_hz=control_hz,
        lower=lower,
        upper=upper,
        reject_position_slice=reject_position_slice,
    )


class TestConstructor:
    def test_valid_construction(self) -> None:
        lim = _make()
        assert lim.dim == DIM
        assert lim.control_hz == 30.0

    def test_rejects_non_positive_dim(self) -> None:
        with pytest.raises(ValueError):
            MotionLimiter(
                dim=0, arm_slice=ARM_SLICE, hand_slice=HAND_SLICE,
                arm_velocity=1, arm_acceleration=1, hand_velocity=1,
                hand_acceleration=1, control_hz=30.0,
            )

    def test_rejects_non_positive_velocity(self) -> None:
        with pytest.raises(ValueError):
            _make(arm_velocity=0.0)
        with pytest.raises(ValueError):
            _make(hand_velocity=-1.0)

    def test_rejects_non_positive_acceleration(self) -> None:
        with pytest.raises(ValueError):
            _make(arm_acceleration=0.0)

    def test_rejects_non_positive_hz(self) -> None:
        with pytest.raises(ValueError):
            _make(control_hz=0.0)

    def test_rejects_lower_greater_than_upper(self) -> None:
        lower = np.zeros(DIM); lower[0] = 5.0
        upper = np.ones(DIM); upper[0] = 1.0  # lower[0] > upper[0]
        with pytest.raises(ValueError):
            _make(lower=lower, upper=upper)


class TestPerDimEnvelopeConstruction:
    """arm_slice / hand_slice で異なる envelope が per-dim vector に展開される。"""

    def test_arm_slice_gets_arm_velocity(self) -> None:
        # arm_velocity=1.5、hand_velocity=6.0 で構成、arm slice の内部 max_velocity
        # が 1.5、hand slice が 6.0 になっていること
        lim = _make(arm_velocity=1.5, hand_velocity=6.0)
        # 内部 attr 経由 (test 特権)
        assert np.all(lim._max_velocity[ARM_SLICE] == 1.5)
        assert np.all(lim._max_velocity[HAND_SLICE] == 6.0)

    def test_arm_slice_gets_arm_acceleration(self) -> None:
        lim = _make(arm_acceleration=8.0, hand_acceleration=30.0)
        assert np.all(lim._max_acceleration[ARM_SLICE] == 8.0)
        assert np.all(lim._max_acceleration[HAND_SLICE] == 30.0)


class TestVelocityLimit:
    """velocity clip: |target - reference| / dt <= max_velocity。"""

    def test_no_limit_when_delta_within_envelope(self) -> None:
        # arm_velocity=2.0、dt=1/30 → max_step = 2/30 = 0.0667 rad
        # measured=0、target=0.05 (< 0.0667) → そのまま返る
        # 初回は acceleration limit も効くので、arm_acceleration も緩め (1e6) に
        # して velocity limit だけの効果を isolate する
        lim = _make(arm_velocity=2.0, arm_acceleration=1e6)
        measured = np.zeros(DIM)
        target = np.zeros(DIM); target[0] = 0.05
        out = lim.apply(target=target, measured=measured)
        assert abs(out[0] - 0.05) < 1e-9

    def test_clips_when_delta_exceeds_velocity(self) -> None:
        # arm_velocity=2.0、dt=1/30 → max_step = 0.0667 rad
        # target=1.0 に一気に飛ぶ → 1 step で max_step までしか動けない
        lim = _make(arm_velocity=2.0, arm_acceleration=1e6)  # accel は大きく (無効化)
        measured = np.zeros(DIM)
        target = np.zeros(DIM); target[0] = 1.0
        out = lim.apply(target=target, measured=measured)
        max_step = 2.0 / 30.0
        assert abs(out[0] - max_step) < 1e-6, f"got {out[0]}, expected {max_step}"

    def test_hand_uses_hand_velocity_not_arm(self) -> None:
        # hand slice ([17:19]) が hand_velocity で clip されること (arm と別 envelope)
        lim = _make(arm_velocity=2.0, hand_velocity=6.0, arm_acceleration=1e6, hand_acceleration=1e6)
        measured = np.zeros(DIM)
        target = np.zeros(DIM); target[17] = 1.0  # left hand grip
        out = lim.apply(target=target, measured=measured)
        max_step_hand = 6.0 / 30.0
        assert abs(out[17] - max_step_hand) < 1e-6


class TestAccelerationLimit:
    """acceleration clip: |velocity - prev_velocity| / dt <= max_acceleration。"""

    def test_first_apply_no_acceleration_limit(self) -> None:
        # 初回 apply: prev_velocity=0、reference=measured。max_acceleration が
        # 有限でも、|desired_velocity - 0| / dt <= max_acceleration 条件のみ
        # (=max_acceleration*dt を上限とする velocity)。
        lim = _make(arm_velocity=100.0, arm_acceleration=5.0)  # arm_vel は緩、accel が effective cap
        measured = np.zeros(DIM)
        target = np.zeros(DIM); target[0] = 100.0  # very far
        out = lim.apply(target=target, measured=measured)
        # accel = 5 rad/s²、dt = 1/30、max velocity_delta = 5/30 = 0.1667 rad/s
        # 初回 velocity = 0.1667 rad/s (accel * dt)
        # position 変化 = velocity * dt = 0.1667/30 = 0.00556 rad
        expected = (5.0 / 30.0) / 30.0
        assert abs(out[0] - expected) < 1e-6, f"got {out[0]}, expected {expected}"

    def test_second_apply_uses_previous_velocity(self) -> None:
        # 2 回目 apply では前 tick velocity 基準で accel clip
        lim = _make(arm_velocity=100.0, arm_acceleration=5.0)
        measured = np.zeros(DIM)
        target = np.zeros(DIM); target[0] = 100.0
        out1 = lim.apply(target=target, measured=measured)
        out2 = lim.apply(target=target, measured=measured)
        # 2 回目は position が 2 * accel * dt² = 2 * (5/30/30) = 0.0111 rad へ
        # 実際は accel * dt² * (1 + 2) / 2 みたいな話じゃなく、
        # velocity(t) = t * accel、position(t) = 0.5 * accel * t²
        # ここは fixed dt での integer step、position(2) = out1 + velocity(2) * dt
        # velocity(2) = velocity(1) + accel * dt = 0.1667 + 0.1667 = 0.333
        # out2 = out1 + 0.333 * dt = 0.00556 + 0.333/30 = 0.00556 + 0.0111 = 0.0167
        expected = out1[0] + 2 * (5.0 / 30.0) / 30.0
        assert abs(out2[0] - expected) < 1e-6


class TestPositionClip:
    def test_clips_to_upper(self) -> None:
        upper = np.full(DIM, 0.5); lower = np.full(DIM, -0.5)
        lim = _make(arm_velocity=1e6, arm_acceleration=1e6, lower=lower, upper=upper)
        measured = np.zeros(DIM)
        target = np.full(DIM, 10.0)  # 大幅超過
        out = lim.apply(target=target, measured=measured)
        assert np.all(out <= 0.5 + 1e-9)

    def test_clips_to_lower(self) -> None:
        upper = np.full(DIM, 0.5); lower = np.full(DIM, -0.5)
        lim = _make(arm_velocity=1e6, arm_acceleration=1e6, lower=lower, upper=upper)
        measured = np.zeros(DIM)
        target = np.full(DIM, -10.0)
        out = lim.apply(target=target, measured=measured)
        assert np.all(out >= -0.5 - 1e-9)

    def test_default_position_bounds_are_permissive(self) -> None:
        # default lower/upper = ±100 rad = 事実上 no-op
        lim = _make()
        measured = np.zeros(DIM)
        target = np.full(DIM, 50.0)
        # velocity/acceleration で clip されるので position clip は発火しない、
        # 例外なく通ること (=default bounds が過剰に狭くない)
        out = lim.apply(target=target, measured=measured)
        assert np.all(np.abs(out) < 100.0)

    def test_arm_limit_violation_holds_entire_previous_target(self) -> None:
        lower = np.full(DIM, -1.0)
        upper = np.full(DIM, 1.0)
        lim = _make(
            arm_velocity=1e6,
            arm_acceleration=1e6,
            lower=lower,
            upper=upper,
            reject_position_slice=slice(3, 17),
        )
        measured = np.linspace(-0.2, 0.2, DIM)
        valid = measured.copy()
        valid[3:17] += 0.01
        previous = lim.apply(target=valid, measured=measured)
        unsafe = valid.copy()
        unsafe[6] = -1.2
        unsafe[7] += 0.5
        held = lim.apply(target=unsafe, measured=measured)
        np.testing.assert_allclose(held, previous)
        assert lim.last_target_held is True
        assert lim.last_position_violation_indices == (6,)

    def test_hand_endpoint_is_clipped_without_rejecting_safe_arms(self) -> None:
        lower = np.concatenate((np.full(17, -2.0), np.zeros(2)))
        upper = np.concatenate((np.full(17, 2.0), np.full(2, 5.4)))
        lim = _make(
            arm_velocity=1e6,
            arm_acceleration=1e6,
            hand_velocity=1e6,
            hand_acceleration=1e6,
            lower=lower,
            upper=upper,
            reject_position_slice=slice(3, 17),
        )
        target = np.zeros(DIM)
        target[17:] = (-0.5, 5.9)
        out = lim.apply(target=target, measured=np.zeros(DIM))
        np.testing.assert_allclose(out[17:], [0.0, 5.4])
        assert lim.last_target_held is False


class TestReset:
    def test_reset_returns_to_measured_reference(self) -> None:
        # reset 前: 前 target が reference、reset 後: measured が reference
        lim = _make(arm_velocity=2.0, arm_acceleration=1e6)
        measured = np.zeros(DIM)
        target = np.zeros(DIM); target[0] = 1.0
        # 1 tick 走らせて trajectory 状態を作る
        out1 = lim.apply(target=target, measured=measured)
        assert out1[0] > 0.0  # progressed
        # reset
        lim.reset()
        # 直後の apply は measured (=0) を reference にするので、また max_step = 2/30 で clip
        out2 = lim.apply(target=target, measured=measured)
        max_step = 2.0 / 30.0
        assert abs(out2[0] - max_step) < 1e-6


class TestInputValidation:
    def test_rejects_wrong_dim_target(self) -> None:
        lim = _make()
        with pytest.raises(ValueError):
            lim.apply(target=np.zeros(DIM - 1), measured=np.zeros(DIM))
        with pytest.raises(ValueError):
            lim.apply(target=np.zeros(DIM + 1), measured=np.zeros(DIM))

    def test_rejects_wrong_dim_measured(self) -> None:
        lim = _make()
        with pytest.raises(ValueError):
            lim.apply(target=np.zeros(DIM), measured=np.zeros(DIM - 1))

    def test_rejects_nan_target(self) -> None:
        lim = _make()
        target = np.zeros(DIM); target[3] = float("nan")
        with pytest.raises(ValueError):
            lim.apply(target=target, measured=np.zeros(DIM))

    def test_rejects_nan_measured(self) -> None:
        lim = _make()
        measured = np.zeros(DIM); measured[5] = float("nan")
        with pytest.raises(ValueError):
            lim.apply(target=np.zeros(DIM), measured=measured)


class TestOutputIndependence:
    def test_output_is_independent_copy(self) -> None:
        # 呼出後に返却 array を書き換えても limiter 内部 state が壊れないこと
        lim = _make()
        measured = np.zeros(DIM)
        target = np.zeros(DIM); target[0] = 0.5
        out = lim.apply(target=target, measured=measured)
        original_prev = lim._previous_target.copy()
        out[:] = -999.0
        assert np.array_equal(lim._previous_target, original_prev), (
            "apply が内部 _previous_target を返却 array と共有している (state 破壊 risk)"
        )


class TestBrakingDeceleration:
    """Issue #137: 制動距離による速度上限 (減速計画)。

    これが無いと目標到達時点で速度が残り行き過ぎる。chunk 切替のたびに乗るため
    実機の手先振動の増幅要因になっていた。
    """

    @staticmethod
    def _step_response(step: float, n: int = 90, **kw) -> np.ndarray:
        lim = _make(arm_velocity=1.0, arm_acceleration=8.0, **kw)
        measured = np.zeros(DIM)
        target = np.zeros(DIM)
        target[3] = step
        return np.array(
            [lim.apply(target=target, measured=measured)[3] for _ in range(n)]
        )

    @pytest.mark.parametrize("step", [0.05, 0.10, 0.20])
    def test_overshoot_stays_under_25_percent(self, step: float) -> None:
        # 修正前は 0.05 rad で 40%、0.10 rad で 33%。
        y = self._step_response(step)
        overshoot = (y.max() - step) / step
        assert overshoot < 0.25, f"step={step} overshoot={overshoot:.3f}"

    @pytest.mark.parametrize("step", [0.05, 0.10, 0.20])
    def test_settles_on_target(self, step: float) -> None:
        y = self._step_response(step)
        assert abs(y[-1] - step) < 1e-6

    def test_far_target_still_reaches_velocity_limit(self) -> None:
        """遠い目標では制動 cap が効かず、従来どおり最大速度まで出る。"""
        y = self._step_response(2.0, n=20)
        peak_velocity = np.diff(y).max() * 30.0
        assert peak_velocity > 0.9, peak_velocity   # arm_velocity=1.0 近傍

    def test_does_not_reverse_direction(self) -> None:
        """減速は行き過ぎを減らすためで、目標を跨いで往復してはいけない。"""
        step = 0.10
        y = self._step_response(step)
        crossings = int(np.sum(np.sign(y[1:] - step) * np.sign(y[:-1] - step) < 0))
        assert crossings <= 1, crossings


class TestPositionViolationDiagnostics:
    """Issue #137: position clip が効いた事実を外から観測できるようにする。"""

    def test_no_violation_when_within_limits(self) -> None:
        lim = _make(lower=np.full(DIM, -1.0), upper=np.full(DIM, 1.0))
        lim.apply(target=np.zeros(DIM), measured=np.zeros(DIM))
        assert lim.last_position_violation_indices == ()
        assert lim.position_violation_ticks == 0

    def test_records_violating_dim(self) -> None:
        lower = np.full(DIM, -100.0)
        upper = np.full(DIM, 100.0)
        upper[3] = 0.5
        lim = _make(lower=lower, upper=upper)
        target = np.zeros(DIM)
        target[3] = 0.8            # upper を超える要求
        lim.apply(target=target, measured=np.zeros(DIM))
        assert lim.last_position_violation_indices == (3,)
        assert lim.position_violation_ticks == 1

    def test_counter_accumulates_and_resets(self) -> None:
        lower = np.full(DIM, -100.0)
        upper = np.full(DIM, 100.0)
        upper[3] = 0.5
        lim = _make(lower=lower, upper=upper)
        target = np.zeros(DIM)
        target[3] = 0.8
        for _ in range(4):
            lim.apply(target=target, measured=np.zeros(DIM))
        assert lim.position_violation_ticks == 4
        lim.reset()
        assert lim.position_violation_ticks == 0
        assert lim.last_position_violation_indices == ()
