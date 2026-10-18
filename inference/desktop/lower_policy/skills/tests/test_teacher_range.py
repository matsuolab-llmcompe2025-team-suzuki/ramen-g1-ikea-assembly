"""TeacherJointRange の unit test (Issue #141 束 1-4 / D5-1)。

主眼:
    - 範囲の外に出た関節だけを端から margin_rad 内側へ戻す (内側にいる関節は触らない)
    - 補正した関節名とはみ出した量が分かる
    - `skill_config.yaml` の読み込み (欠けた関節・上下逆・margin 無しは error)
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from inference.desktop.lower_policy.pose_utils import JOINT_NAMES, NUM_ARM_JOINTS
from inference.desktop.lower_policy.skills.teacher_range import (
    ARMS_19D,
    TeacherJointRange,
    load_teacher_joint_range_for_skill,
)

SKILL_CONFIG_PATH = (
    "inference/desktop/lower_policy/configs/skill_config.yaml"
)


def _range(margin_rad: float = 0.02) -> TeacherJointRange:
    return TeacherJointRange(
        np.full(NUM_ARM_JOINTS, -1.0), np.full(NUM_ARM_JOINTS, 1.0), margin_rad=margin_rad
    )


def _target(**arms: float) -> np.ndarray:
    target = np.zeros(19)
    for name, value in arms.items():
        target[ARMS_19D.start + JOINT_NAMES.index(name)] = value
    return target


def _arm(target: np.ndarray, name: str) -> float:
    return float(target[ARMS_19D.start + JOINT_NAMES.index(name)])


class TestApply:
    def test_inside_range_is_untouched(self):
        result = _range().apply(_target(**{"R.elbow": 0.99}))
        assert not result.bind
        assert result.corrected_joints == ()
        assert _arm(result.target_19d, "R.elbow") == pytest.approx(0.99)

    def test_above_max_is_pulled_to_upper_minus_margin(self):
        result = _range().apply(_target(**{"R.elbow": 1.5}))
        assert result.corrected_joints == ("R.elbow",)
        assert _arm(result.target_19d, "R.elbow") == pytest.approx(0.98)
        assert result.max_excursion_rad == pytest.approx(0.5)

    def test_below_min_is_pulled_to_lower_plus_margin(self):
        result = _range().apply(_target(**{"L.wrist_yaw": -1.2}))
        assert _arm(result.target_19d, "L.wrist_yaw") == pytest.approx(-0.98)

    def test_only_out_of_range_joints_move(self):
        target = _target(**{"R.elbow": 1.5, "L.elbow": 0.5})
        result = _range().apply(target)
        assert result.corrected_joints == ("R.elbow",)
        assert _arm(result.target_19d, "L.elbow") == pytest.approx(0.5)

    def test_waist_and_hand_are_never_touched(self):
        target = _target(**{"R.elbow": 1.5})
        target[0:3] = 9.0    # waist
        target[17:19] = 9.0  # hand
        result = _range().apply(target)
        assert result.target_19d[0:3].tolist() == [9.0, 9.0, 9.0]
        assert result.target_19d[17:19].tolist() == [9.0, 9.0]

    def test_input_is_not_modified_in_place(self):
        target = _target(**{"R.elbow": 1.5})
        _range().apply(target)
        assert _arm(target, "R.elbow") == pytest.approx(1.5)

    def test_wrong_dimension_raises(self):
        with pytest.raises(ValueError, match=r"\(19,\)"):
            _range().apply(np.zeros(16))


class TestConstruction:
    def test_margin_wider_than_half_the_range_raises(self):
        with pytest.raises(ValueError, match="narrower"):
            TeacherJointRange(
                np.full(NUM_ARM_JOINTS, -0.01),
                np.full(NUM_ARM_JOINTS, 0.01),
                margin_rad=0.02,
            )

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="must be"):
            TeacherJointRange(np.zeros(3), np.ones(3), margin_rad=0.02)


class TestLoadFromConfig:
    def _config(self, **overrides) -> dict:
        arm = {name: [-1.0, 1.0] for name in JOINT_NAMES}
        block = {"margin_rad": 0.02, "arm_position_rad": arm}
        block.update(overrides)
        return {"skills": {"demo": {"teacher_joint_range": block}}}

    def test_missing_block_returns_none(self):
        assert load_teacher_joint_range_for_skill({"skills": {"demo": {}}}, "demo") is None

    def test_loads_all_14_joints(self):
        loaded = load_teacher_joint_range_for_skill(self._config(), "demo")
        assert loaded is not None and loaded.margin_rad == pytest.approx(0.02)

    def test_missing_joint_raises(self):
        config = self._config()
        del config["skills"]["demo"]["teacher_joint_range"]["arm_position_rad"]["R.elbow"]
        with pytest.raises(ValueError, match="missing="):
            load_teacher_joint_range_for_skill(config, "demo")

    def test_unknown_joint_raises(self):
        config = self._config()
        config["skills"]["demo"]["teacher_joint_range"]["arm_position_rad"]["R.thumb"] = [0.0, 1.0]
        with pytest.raises(ValueError, match="unknown="):
            load_teacher_joint_range_for_skill(config, "demo")

    def test_reversed_bounds_raise(self):
        config = self._config()
        config["skills"]["demo"]["teacher_joint_range"]["arm_position_rad"]["R.elbow"] = [1.0, -1.0]
        with pytest.raises(ValueError, match="min must be < max"):
            load_teacher_joint_range_for_skill(config, "demo")

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown keys"):
            load_teacher_joint_range_for_skill(self._config(p99=True), "demo")


@pytest.fixture(scope="module")
def config() -> dict:
    with open(SKILL_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestShippedConfig:
    """実際に配る `skill_config.yaml` の値 (教師データからの実測) を確かめる。"""

    @pytest.mark.parametrize(
        "skill_name",
        ["rotate_table_base", "pick_table_leg", "insert_table_leg",
         "rotate_leg_to_tighten", "flip_table"],
    )
    def test_learned_skills_have_a_range(self, config, skill_name):
        loaded = load_teacher_joint_range_for_skill(config, skill_name)
        assert loaded is not None
        assert loaded.margin_rad == pytest.approx(0.02)

    def test_rotate_matches_the_measured_audit_numbers(self, config):
        """09-09 の監査 (計測 M-5、merged_v1 146,873 frame) と同じ値であること。"""
        block = config["skills"]["rotate_table_base"]["teacher_joint_range"]
        assert block["source"]["frame_count"] == 146873
        assert block["arm_position_rad"]["R.elbow"] == pytest.approx([-0.5969154239, 1.3200325966])
        assert block["arm_position_rad"]["R.wrist_yaw"] == pytest.approx([-0.2097411752, 1.4965564013])
        assert block["arm_position_rad"]["L.wrist_yaw"] == pytest.approx([-1.5679877996, 0.4148552418])
