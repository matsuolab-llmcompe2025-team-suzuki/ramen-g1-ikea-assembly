"""会場の経路だけの開始姿勢の上書き (`boundary_profile`、Issue #188)。

運営 WBC では、肘の重力補正を入れた後も開始の手が学習の開始の中央値より低い skill がある
(handoff の 10-2)。会場では、その手を向きを保ったまま真上へ上げた開始姿勢を使う。
書いた値がその通りになっているか・会場の経路でだけ効くか・別の開始姿勢には重ねないかを確かめる。
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from inference.desktop import entrypoint
from inference.desktop.lower_policy.actuators.g1_arm_sdk import (
    G1_ARM_POSITION_LOWER_RAD,
    G1_ARM_POSITION_UPPER_RAD,
)
from inference.desktop.lower_policy.initial_pose import (
    ARM_JOINT_ORDER,
    apply_boundary_profiles,
    apply_policy_variant_profile,
    initial_pose_from_config,
)
from inference.desktop.perception.g1_urdf_fk import G1WristFK

SKILL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "skill_config.yaml"
#: skill → {手: 上げた量 [m]}。skill_config の boundary_profile のコメントと同じ。
LIFTS = {
    "rotate_table_base": {"left": 0.025},
    "pick_table_leg": {"right": 0.025, "left": 0.025},
    "rotate_leg_to_tighten": {"right": 0.025},
    "flip_table": {"right": 0.030, "left": 0.040},
}
FK = G1WristFK.from_urdf()


def _config() -> dict:
    return yaml.safe_load(SKILL_CONFIG.read_text(encoding="utf-8"))


def _tool(arms14: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """胴基準 (腰 0) の手先の位置と向き。"""
    q = np.zeros(29)
    q[15:29] = arms14
    chain, offset = (
        (FK._left_chain, FK._left_offset)
        if side == "left"
        else (FK._right_chain, FK._right_offset)
    )
    transform = FK._fk_chain(chain, q)
    return transform[:3, 3] + transform[:3, :3] @ offset, transform[:3, :3]


def _poses(skill: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = _config()
    base = initial_pose_from_config(cfg, skill).arm_position_rad
    venue, _ = apply_boundary_profiles(cfg)
    return base, initial_pose_from_config(venue, skill).arm_position_rad


@pytest.mark.parametrize(
    ("skill", "side", "lift"),
    [
        (skill, side, lift)
        for skill, hands in LIFTS.items()
        for side, lift in hands.items()
    ],
)
def test_each_lifted_hand_moves_straight_up_with_the_same_orientation(
    skill, side, lift
):
    base, venue = _poses(skill)
    p0, r0 = _tool(base, side)
    p1, r1 = _tool(venue, side)
    np.testing.assert_allclose(p1 - p0, [0.0, 0.0, lift], atol=1e-3)
    angle = np.degrees(np.arccos(np.clip((np.trace(r0.T @ r1) - 1.0) / 2.0, -1.0, 1.0)))
    assert angle < 0.1


@pytest.mark.parametrize("skill", list(LIFTS))
def test_the_other_hand_stays_where_it_was(skill):
    base, venue = _poses(skill)
    for side, arm in (("left", slice(0, 7)), ("right", slice(7, 14))):
        if side not in LIFTS[skill]:
            np.testing.assert_array_equal(venue[arm], base[arm])


@pytest.mark.parametrize("skill", list(LIFTS))
def test_the_venue_pose_stays_inside_the_teacher_range_and_the_urdf_limits(skill):
    _, venue = _poses(skill)
    teacher = (_config()["skills"][skill].get("teacher_joint_range") or {}).get(
        "arm_position_rad", {}
    )
    for index, name in enumerate(ARM_JOINT_ORDER):
        if name in teacher:
            low, high = teacher[name]
            assert low <= venue[index] <= high, (skill, name)
        assert G1_ARM_POSITION_LOWER_RAD[index] + 0.03 <= venue[index], (skill, name)
        assert venue[index] <= G1_ARM_POSITION_UPPER_RAD[index] - 0.03, (skill, name)


def test_only_the_skills_with_a_profile_change():
    cfg = _config()
    venue, skills = apply_boundary_profiles(cfg)
    assert set(skills) == set(LIFTS)
    for name in cfg["skills"]:
        if name not in LIFTS:
            assert venue["skills"][name] == cfg["skills"][name], name
    for name in LIFTS:
        before = cfg["skills"][name]["initial_pose"]
        after = venue["skills"][name]["initial_pose"]
        assert after["dex1_opening_fraction"] == before["dex1_opening_fraction"], name
    assert cfg == _config()  # 元の設定は書き換えない


def test_only_the_venue_lane_uses_it():
    cfg = _config()
    base, venue = _poses("pick_table_leg")
    sdk = entrypoint.apply_venue_start_poses(copy.deepcopy(cfg), "sdk")
    boundary = entrypoint.apply_venue_start_poses(copy.deepcopy(cfg), "boundary")
    np.testing.assert_array_equal(
        initial_pose_from_config(sdk, "pick_table_leg").arm_position_rad, base
    )
    np.testing.assert_array_equal(
        initial_pose_from_config(boundary, "pick_table_leg").arm_position_rad, venue
    )


def test_a_variant_with_another_start_pose_is_left_alone(capsys):
    """flip の n17_4 は別のデータの開始姿勢に差し替える。そこには会場の上書きを重ねない。"""
    cfg = apply_policy_variant_profile(
        _config(), "flip_table", "groot_flip_table_n17_4"
    )
    before = initial_pose_from_config(cfg, "flip_table").arm_position_rad
    venue, skills = apply_boundary_profiles(cfg)
    assert "flip_table" not in skills
    np.testing.assert_array_equal(
        initial_pose_from_config(venue, "flip_table").arm_position_rad, before
    )
    assert "not applied" in capsys.readouterr().err
    # 既定の variant (legacy_fk) は開始姿勢を変えないので重なる
    legacy = apply_policy_variant_profile(
        _config(), "flip_table", "groot_flip_table_n17_2_legacy_fk"
    )
    assert "flip_table" in apply_boundary_profiles(legacy)[1]


def _profile(**initial_pose) -> dict:
    ip = _config()["skills"]["pick_table_leg"]["initial_pose"]
    return {
        "source": f"{ip['dataset_repo_id']}@{ip['dataset_revision']}",
        "initial_pose": initial_pose,
    }


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (["not", "a", "mapping"], "may only override"),
        ({**_profile(), "max_seconds_hard": 5}, "may only override"),
        ({"initial_pose": {"arm_position_rad": {"R.elbow": 0.2}}}, "source must name"),
        (_profile(dex1_opening_fraction={"left": 0.5}), "may only override"),
        (_profile(arm_position_rad={"R.elbw": 0.2}), "unknown joints"),
        (_profile(arm_position_rad={"R.elbow": float("nan")}), "finite"),
    ],
)
def test_a_wrong_profile_is_rejected(profile, message):
    cfg = _config()
    cfg["skills"]["pick_table_leg"]["boundary_profile"] = profile
    with pytest.raises(ValueError, match=message):
        apply_boundary_profiles(cfg)


@pytest.mark.parametrize(
    ("skill", "side"), [("pick_table_leg", "right"), ("flip_table", "right")]
)
def test_the_start_keys_base_matches_the_lift_written_in_the_profile(skill, side):
    """開始待ちの U / D の基準 (base_lift_m) と、boundary_profile で実際に上げた量が同じ。"""
    keys = _config()["skills"][skill]["start_lift_keys"]
    assert keys["base_lift_m"] == pytest.approx(LIFTS[skill][side])
