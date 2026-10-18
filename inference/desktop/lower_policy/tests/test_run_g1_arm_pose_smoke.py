from __future__ import annotations

import numpy as np

from inference.desktop.lower_policy.scripts.run_g1_arm_pose_smoke import (
    LEFT_SHOULDER_ROLL,
    RIGHT_SHOULDER_ROLL,
    _build_outward_target,
    _pose_error,
    _ramp_action,
)


class _FakeArm:
    def __init__(self) -> None:
        self.targets: list[np.ndarray] = []

    def send_action(self, target: np.ndarray) -> None:
        self.targets.append(np.asarray(target, dtype=np.float64).copy())


def test_ramp_action_is_monotonic_and_finishes_exactly(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    arm = _FakeArm()
    start = np.zeros(14, dtype=np.float64)
    target = np.zeros(14, dtype=np.float64)
    target[1] = 0.08
    target[8] = -0.08

    _ramp_action(arm, start, target, duration_s=1.0, update_hz=10.0)

    assert len(arm.targets) == 10
    assert np.all(np.diff([q[1] for q in arm.targets]) > 0.0)
    assert np.all(np.diff([q[8] for q in arm.targets]) < 0.0)
    assert np.array_equal(arm.targets[-1], target)


def test_pose_error_uses_all_fourteen_joints() -> None:
    actual = np.zeros(14, dtype=np.float64)
    target = np.zeros(14, dtype=np.float64)
    actual[13] = -0.11
    assert _pose_error(actual, target) == 0.11


def test_outward_target_applies_delta_once_and_symmetrically() -> None:
    before = np.linspace(-0.3, 0.3, 14)
    target = _build_outward_target(before, 0.08)
    delta = target - before

    expected = np.zeros(14)
    expected[LEFT_SHOULDER_ROLL] = 0.08
    expected[RIGHT_SHOULDER_ROLL] = -0.08
    assert np.allclose(delta, expected)
