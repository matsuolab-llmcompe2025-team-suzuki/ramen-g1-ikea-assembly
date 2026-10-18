"""HoldPoseSkill の unit test (Issue #141 束 1-8 / D2)。

開始姿勢で N 秒待つ有限の準備 skill。Policy 開始の Enter gate は別にある。
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.skills.hold_pose import HoldPoseSkill
from inference.desktop.perception.g1_urdf_fk import G1_JOINT_NAMES


class _Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


class _JointState:
    def __init__(self, arm_value: float) -> None:
        self.name = G1_JOINT_NAMES
        self.position = np.zeros(29)
        self.position[15:] = arm_value
        self.velocity = np.zeros(29)
        self.effort = np.zeros(29)
        self.t = 0


def _obs(arm_value: float = 0.3) -> dict:
    return {"joint_state": _JointState(arm_value), "t": 0}


def test_holds_the_first_measured_pose_until_the_time_passes():
    clock = _Clock()
    skill = HoldPoseSkill(3.0, time_fn=clock)
    skill.start({})
    first = skill.step(_obs(0.3))
    np.testing.assert_allclose(first, 0.3)
    assert not skill.is_complete

    clock.now += 2.9
    np.testing.assert_allclose(skill.step(_obs(0.9)), 0.3)   # 測定が変わっても指令は同じ
    assert not skill.is_complete

    clock.now += 0.2
    skill.step(_obs(0.9))
    assert skill.is_complete


def test_retry_hold_uses_last_published_target_instead_of_lagging_measurement():
    target = np.full(14, 0.45)
    skill = HoldPoseSkill(
        0.0, hold_arm_target_provider=lambda: target.copy(), time_fn=_Clock()
    )
    skill.start({})
    np.testing.assert_allclose(skill.step(_obs(0.30)), target)


def test_zero_seconds_completes_on_the_first_tick():
    skill = HoldPoseSkill(0.0, time_fn=_Clock())
    skill.start({})
    skill.step(_obs())
    assert skill.is_complete


def test_restart_clears_the_previous_hold():
    clock = _Clock()
    skill = HoldPoseSkill(1.0, time_fn=clock)
    skill.start({})
    skill.step(_obs(0.3))
    clock.now += 2.0
    skill.step(_obs(0.3))
    assert skill.is_complete
    skill.stop()

    skill.start({})
    assert not skill.is_complete
    np.testing.assert_allclose(skill.step(_obs(0.7)), 0.7)


def test_negative_hold_is_rejected():
    with pytest.raises(ValueError, match="hold_sec must be >= 0"):
        HoldPoseSkill(-1.0)


def test_missing_joint_state_is_rejected():
    skill = HoldPoseSkill(1.0)
    skill.start({})
    with pytest.raises(RuntimeError, match="live joint state"):
        skill.step({"joint_state": None})


def test_async_completion_gate_holds_until_ready():
    clock = _Clock()
    ready = {"value": False}
    skill = HoldPoseSkill(
        0.0,
        time_fn=clock,
        completion_ready_fn=lambda: ready["value"],
        completion_description="pick model loads",
    )
    skill.start({})
    np.testing.assert_allclose(skill.step(_obs(0.3)), 0.3)
    assert not skill.is_complete

    ready["value"] = True
    np.testing.assert_allclose(skill.step(_obs(0.9)), 0.3)
    assert skill.is_complete


def test_async_completion_gate_waits_for_a_slow_load_without_a_time_limit():
    """読み込み中の model は待つしかない。何分かかっても故障にしない。"""
    clock = _Clock()
    ready = {"value": False}
    skill = HoldPoseSkill(0.0, time_fn=clock)
    skill.set_completion_gate(
        lambda: ready["value"], description="insert model loads"
    )
    skill.start({})
    skill.step(_obs(0.3))
    clock.now = 600.0
    np.testing.assert_allclose(skill.step(_obs(0.9)), 0.3)
    assert skill.failure_reason is None
    assert not skill.is_complete

    ready["value"] = True
    skill.step(_obs(0.9))
    assert skill.is_complete


def test_async_completion_gate_fails_closed_on_load_error():
    skill = HoldPoseSkill(
        0.0,
        time_fn=_Clock(),
        completion_ready_fn=lambda: False,
        completion_error_fn=lambda: "checkpoint corrupt",
        completion_description="pick model loads",
    )
    skill.start({})
    np.testing.assert_allclose(skill.step(_obs()), 0.3)
    assert "checkpoint corrupt" in (skill.failure_reason or "")
    assert not skill.is_complete
