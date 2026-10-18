"""開始待ちで U / D キーで手の高さを変える (`skills/start_lift.py`、Issue #188)。"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from inference.desktop import entrypoint
from inference.desktop.lower_policy.initial_pose import (
    apply_boundary_profiles,
    initial_pose_from_config,
)
from inference.desktop.lower_policy.pose_utils import ROS_ARM_JOINT_NAMES
from inference.desktop.lower_policy.skills.operator_gate import (
    OperatorConfirmationHoldSkill,
)
from inference.desktop.lower_policy.skills.start_lift import (
    StartLiftAdjuster,
    StartLiftConfig,
    lift_arm_pose,
)
from inference.desktop.perception.g1_urdf_fk import G1WristFK

SKILL_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "skill_config.yaml"
FK = G1WristFK.from_urdf()


def _config() -> dict:
    return yaml.safe_load(SKILL_CONFIG.read_text(encoding="utf-8"))


def _venue_pick() -> tuple[dict, np.ndarray]:
    venue, _ = apply_boundary_profiles(_config())
    return venue, initial_pose_from_config(venue, "pick_table_leg").arm_position_rad


def _right_tool(arm14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.zeros(29)
    q[15:29] = arm14
    transform = FK._fk_chain(FK._right_chain, q)
    return transform[:3, 3] + transform[:3, :3] @ FK._right_offset, transform[:3, :3]


def _adjuster(**overrides) -> StartLiftAdjuster:
    venue, base = _venue_pick()
    config = StartLiftConfig.from_skill(
        venue["skills"]["pick_table_leg"], "pick_table_leg"
    )
    values = {**config.__dict__, **overrides}
    teacher = venue["skills"]["pick_table_leg"]["teacher_joint_range"][
        "arm_position_rad"
    ]
    return StartLiftAdjuster(
        base, StartLiftConfig(**values), fk=FK, teacher_range=teacher
    )


# ---------------------------------------------------------------- IK


@pytest.mark.parametrize("dz", [0.01, -0.01, 0.03])
def test_lift_moves_the_hand_straight_up_keeping_its_orientation(dz):
    _, base = _venue_pick()
    lifted = lift_arm_pose(base, "right", dz, FK)
    p0, r0 = _right_tool(base)
    p1, r1 = _right_tool(lifted)
    np.testing.assert_allclose(p1 - p0, [0.0, 0.0, dz], atol=5e-4)
    assert np.degrees(np.arccos(np.clip((np.trace(r0.T @ r1) - 1) / 2, -1, 1))) < 0.2
    np.testing.assert_array_equal(lifted[:7], base[:7])


def test_an_unreachable_lift_is_refused():
    _, base = _venue_pick()
    with pytest.raises(ValueError, match="cannot move"):
        lift_arm_pose(base, "right", 0.6, FK)


# ---------------------------------------------------------------- 設定


def test_only_pick_and_flip_have_the_keys_in_the_real_config():
    cfg = _config()
    config = StartLiftConfig.from_skill(
        cfg["skills"]["pick_table_leg"], "pick_table_leg"
    )
    assert config.hand == "right" and config.step_m == pytest.approx(0.01)
    assert config.base_lift_m == pytest.approx(0.025)
    flip = StartLiftConfig.from_skill(cfg["skills"]["flip_table"], "flip_table")
    assert flip.hand == "both" and flip.base_lift_m == pytest.approx(0.03)
    for name, section in cfg["skills"].items():
        if name not in ("pick_table_leg", "flip_table") and isinstance(section, dict):
            assert StartLiftConfig.from_skill(section, name) is None, name


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"hand": "neither"}, "left, right or both"),
        ({"step_m": 0.05}, "step_m"),
        ({"base_lift_m": 0.09}, "min_lift_m <= base_lift_m"),
        ({"joint_speed_rad_s": 2.0}, "joint_speed_rad_s"),
        ({"typo": 1}, "exactly"),
    ],
)
def test_a_wrong_config_is_rejected(change, message):
    section = {
        "start_lift_keys": {
            **_config()["skills"]["pick_table_leg"]["start_lift_keys"],
            **change,
        }
    }
    with pytest.raises(ValueError, match=message):
        StartLiftConfig.from_skill(section, "pick_table_leg")


# ---------------------------------------------------------------- 上げ下げ


def test_keys_move_the_target_one_step_and_stop_at_the_ends():
    adjuster = _adjuster()
    _, base = _venue_pick()
    np.testing.assert_array_equal(adjuster.target(), base)
    assert "+3.5 cm" in adjuster.adjust("u") and adjuster.lift_m == pytest.approx(0.035)
    p0, _ = _right_tool(base)
    p1, _ = _right_tool(adjuster.target())
    np.testing.assert_allclose(p1 - p0, [0.0, 0.0, 0.01], atol=5e-4)
    for _ in range(10):
        adjuster.adjust("u")
    assert adjuster.lift_m == pytest.approx(0.08)
    assert "範囲の端" in adjuster.adjust("u")
    for _ in range(12):
        adjuster.adjust("d")
    assert adjuster.lift_m == pytest.approx(0.0)
    assert "範囲の端" in adjuster.adjust("d")


def test_a_step_outside_the_joint_range_is_refused():
    adjuster = _adjuster()
    # 教師の範囲を今の姿勢ちょうどに狭めると、どちらにも動かせない
    base = adjuster.target()
    adjuster._lower, adjuster._upper = base - 1e-6, base + 1e-6
    message = adjuster.adjust("u")
    assert "範囲を越える" in message
    assert adjuster.lift_m == pytest.approx(0.025)
    np.testing.assert_array_equal(adjuster.target(), base)


def test_ramp_limits_the_joint_speed_and_reaches_the_target():
    adjuster = _adjuster()
    held = adjuster.target()
    adjuster.adjust("u")
    goal = adjuster.target()
    step = adjuster.ramp(held, 1 / 30)
    assert np.max(np.abs(step - held)) <= 0.2 / 30 + 1e-12
    for _ in range(200):
        held = adjuster.ramp(held, 1 / 30)
    np.testing.assert_allclose(held, goal)


# ---------------------------------------------------------------- 開始待ちの保持


def _obs(arm14: np.ndarray) -> dict:
    names = tuple(f"unused_{i}" for i in range(15)) + tuple(ROS_ARM_JOINT_NAMES)
    return {
        "joint_state": SimpleNamespace(name=names, position=(0.0,) * 15 + tuple(arm14))
    }


def test_the_gate_moves_to_the_adjusted_pose_and_starts_only_after_arriving():
    adjuster = _adjuster()
    keys: "queue.Queue[str]" = queue.Queue()
    prompts: list[str] = []

    def read(prompt, cancel):
        prompts.append(prompt)
        while not cancel.is_set():
            try:
                return keys.get(timeout=0.05)
            except queue.Empty:
                continue
        raise EOFError("cancelled")

    def arrived(measured):
        return bool(
            np.max(np.abs(np.asarray(measured) - adjuster.target())) < 1e-3
        ), "check"

    base = adjuster.target()
    gate = OperatorConfirmationHoldSkill(
        name="operator_gate_for_pick_table_leg",
        next_skill_name="pick_table_leg",
        cancellable_input_fn=read,
        arrival_check=arrived,
        require_arrival=True,
        hold_arm_target_provider=lambda: base,
        start_adjuster=adjuster,
    )
    gate.start({})
    measured = base.copy()
    held = gate.step(_obs(measured))
    keys.put("u")
    deadline = time.monotonic() + 2.0
    while adjuster.lift_m < 0.03 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert adjuster.lift_m == pytest.approx(0.035)
    goal = adjuster.target()
    # 着く前の Enter では始めない
    keys.put("enter")
    time.sleep(0.2)
    assert not gate.is_complete
    # 保持している目標が、速さを抑えて新しい目標へ近づく (実測もそれについてくるとする)
    for _ in range(200):
        held = gate.step(_obs(held))
        time.sleep(0.002)
        if np.allclose(held, goal):
            break
    np.testing.assert_allclose(held, goal)
    keys.put("enter")
    deadline = time.monotonic() + 2.0
    while not gate.is_complete and time.monotonic() < deadline:
        gate.step(_obs(held))
        time.sleep(0.01)
    assert gate.is_complete
    assert any("[start-lift] 右手 +2.5 cm" in p for p in prompts)
    gate.stop()


# ---------------------------------------------------------------- entrypoint


def test_only_the_venue_pick_gets_the_keys():
    venue, base = _venue_pick()
    assert (
        entrypoint.build_start_lift_adjuster(venue, "pick_table_leg", "sdk", base)
        is None
    )
    assert (
        entrypoint.build_start_lift_adjuster(
            venue, "rotate_table_base", "boundary", base
        )
        is None
    )
    adjuster = entrypoint.build_start_lift_adjuster(
        venue, "pick_table_leg", "boundary", base
    )
    assert isinstance(adjuster, StartLiftAdjuster)
    np.testing.assert_array_equal(adjuster.target(), base)


def test_the_arrival_check_follows_a_changing_target():
    target = {"arm": np.zeros(14)}
    check = entrypoint._gate_arrival_check(lambda: target["arm"], None, 0.05)
    assert check(np.zeros(14))[0]
    target["arm"] = np.full(14, 0.2)
    assert not check(np.zeros(14))[0]
    assert check(np.full(14, 0.2))[0]


def test_the_console_waits_for_any_of_the_keys():
    from inference.desktop.operator_console import OperatorConsole

    console = OperatorConsole()
    result: dict = {}

    def wait():
        result["key"] = console.wait_for_any(
            ("enter", "u", "d"), stage=1, phase="pick／開始待ち"
        )

    thread = threading.Thread(target=wait)
    thread.start()
    deadline = time.monotonic() + 1.0
    while console._view is None and time.monotonic() < deadline:
        time.sleep(0.005)
    console._events.put((console._generation, "u"))
    thread.join(1.0)
    assert result["key"] == "u"


def _left_tool(arm14: np.ndarray) -> np.ndarray:
    q = np.zeros(29)
    q[15:29] = arm14
    transform = FK._fk_chain(FK._left_chain, q)
    return transform[:3, 3] + transform[:3, :3] @ FK._left_offset


def test_flip_moves_both_hands_by_the_same_step():
    """flip は両手を一緒に上げ下げする (量は右手の元の開始姿勢から)。"""
    venue, _ = apply_boundary_profiles(_config())
    base = initial_pose_from_config(venue, "flip_table").arm_position_rad
    adjuster = entrypoint.build_start_lift_adjuster(venue, "flip_table", "boundary", base)
    assert "両手（右手の量） +3.0 cm" in adjuster.status()
    assert "+4.0 cm" in adjuster.adjust("u")
    moved = adjuster.target()
    np.testing.assert_allclose(_right_tool(moved)[0] - _right_tool(base)[0], [0, 0, 0.01], atol=5e-4)
    np.testing.assert_allclose(_left_tool(moved) - _left_tool(base), [0, 0, 0.01], atol=5e-4)
    adjuster.adjust("d")
    adjuster.adjust("d")
    lowered = adjuster.target()
    np.testing.assert_allclose(_left_tool(lowered) - _left_tool(base), [0, 0, -0.01], atol=5e-4)
