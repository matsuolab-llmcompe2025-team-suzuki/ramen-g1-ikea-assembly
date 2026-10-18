"""Physical preparation opening, distinct from the dataset's hand calibration."""

import copy
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from inference.desktop.lower_policy.actuators.hand import MockHandActuator
from inference.desktop.lower_policy.policies.taskspace_adapter import dex1_model_to_taskspace
from inference.desktop.lower_policy.skills.hand_pre_motion import HandPreMotionSkill
from inference.desktop.lower_policy.skills.hand_ramp import (
    build_hand_open_ramp,
    resolve_hand_opening_rad,
)
from inference.desktop.perception.g1_urdf_fk import G1_JOINT_NAMES


@pytest.mark.parametrize("lane,expected", [("sdk", 5.4), ("boundary", 5.3)])
def test_opening_matches_lane_without_rescaling_dataset(lane, expected):
    config = yaml.safe_load(Path("inference/desktop/lower_policy/configs/skill_config.yaml").read_text())
    original = copy.deepcopy(config)
    opening = resolve_hand_opening_rad(config, action_sink=lane)
    assert config == original
    assert opening == pytest.approx(expected)
    config["hand_pre_motion"]["open_rad"] = opening
    hand = MockHandActuator()
    for mode in ("open", "release"):
        skill = HandPreMotionSkill.from_config(config, "pick_table_leg", target=mode, hand_actuator=hand)
        assert skill._target == pytest.approx([expected, expected])
    for mode in ("pose", "grasp"):
        a = HandPreMotionSkill.from_config(config, "insert_table_leg", target=mode, hand_actuator=hand)
        b = HandPreMotionSkill.from_config(original, "insert_table_leg", target=mode, hand_actuator=hand)
        np.testing.assert_array_equal(a._target, b._target)
    assert build_hand_open_ramp([1.0, 2.0], open_rad=opening)[-1] == pytest.approx([expected] * 2)


def test_boundary_full_open_physically_reaches_the_same_target_and_completes():
    """The official relay's -5.30 endpoint must satisfy the preparation gate."""
    now = [0.0]
    hand = MockHandActuator()
    opening = resolve_hand_opening_rad({}, action_sink="boundary")
    config = {"hand_pre_motion": {"open_rad": opening}}
    skill = HandPreMotionSkill.from_config(
        config, "pick_table_leg", target="open", hand_actuator=hand, time_fn=lambda: now[0]
    )
    measured = np.array([1.0, 2.0])
    joint = SimpleNamespace(name=G1_JOINT_NAMES, position=np.zeros(29), velocity=np.zeros(29))
    skill.start({})
    for _ in range(240):
        received = time.monotonic_ns()
        obs = {"joint_state": joint, "hand_state": SimpleNamespace(
            position_rad=measured.copy(), t=received,
            left_received_monotonic_ns=received, right_received_monotonic_ns=received,
        )}
        skill.step(obs)
        command = hand.latest
        if command is not None:
            normalized = np.array([dex1_model_to_taskspace(x) for x in command])
            measured = 5.3 * (1.0 - normalized) / 2.0
        now[0] += 1 / 30
        if skill.is_complete or skill.failure_reason:
            break
    assert skill.failure_reason is None
    assert skill.is_complete
    assert measured == pytest.approx([5.3, 5.3])


@pytest.mark.parametrize("opening", [0, -1, 6, float("nan"), float("inf")])
def test_invalid_preparation_opening_is_rejected(opening):
    with pytest.raises(ValueError, match="open_rad"):
        resolve_hand_opening_rad({"hand_pre_motion": {"open_rad": opening}}, action_sink="boundary")


def test_explicit_smaller_opening_is_preserved():
    assert resolve_hand_opening_rad({"hand_pre_motion": {"open_rad": 4.0}}, action_sink="boundary") == 4.0
