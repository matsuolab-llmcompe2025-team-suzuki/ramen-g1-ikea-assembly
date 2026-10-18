"""SkillInitialPose loader unit test。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from evaluate.model_evaluation.common.initial_pose import (
    ARM_JOINT_ORDER,
    DEX1_MAX_RAD,
    SkillInitialPose,
    load_initial_pose,
)


def _write_yaml(path: Path, skill_name: str, initial_pose: dict | None) -> None:
    cfg = {"skills": {skill_name: {}}}
    if initial_pose is not None:
        cfg["skills"][skill_name]["initial_pose"] = initial_pose
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _valid_arm() -> dict[str, float]:
    return {name: 0.1 * i for i, name in enumerate(ARM_JOINT_ORDER)}


class TestLoadInitialPose:
    def test_valid_load(self, tmp_path):
        cfg_path = tmp_path / "skill_config.yaml"
        _write_yaml(cfg_path, "rotate_table_base", {
            "arm_position_rad": _valid_arm(),
            "dex1_opening_fraction": {"left": 0.2, "right": 0.8},
            "dataset_repo_id": "example/repo",
            "dataset_revision": "deadbeef",
            "training_episode_count": 42,
            "statistic": "median_test",
        })
        pose = load_initial_pose(cfg_path, "rotate_table_base")
        assert isinstance(pose, SkillInitialPose)
        assert pose.arm_position_rad.shape == (14,)
        np.testing.assert_array_almost_equal(
            pose.arm_position_rad, np.arange(14) * 0.1
        )
        assert pose.dex1_opening_fraction == (0.2, 0.8)
        assert pose.requires_separate_hand_initialization is False
        assert pose.training_episode_count == 42

    def test_dex1_target_rad_conversion(self, tmp_path):
        cfg_path = tmp_path / "skill_config.yaml"
        _write_yaml(cfg_path, "test_skill", {
            "arm_position_rad": _valid_arm(),
            "dex1_opening_fraction": {"left": 0.5, "right": 1.0},
        })
        pose = load_initial_pose(cfg_path, "test_skill")
        left_rad, right_rad = pose.dex1_target_rad
        assert left_rad == pytest.approx(0.5 * DEX1_MAX_RAD)
        assert right_rad == pytest.approx(1.0 * DEX1_MAX_RAD)

    def test_hand_initialization_flag(self, tmp_path):
        cfg_path = tmp_path / "skill_config.yaml"
        _write_yaml(cfg_path, "insert_table_leg", {
            "arm_position_rad": _valid_arm(),
            "dex1_opening_fraction": {"left": 0.68, "right": 0.42},
            "requires_separate_hand_initialization": True,
            "hand_initialization_instruction": "Place the leg first.",
        })
        pose = load_initial_pose(cfg_path, "insert_table_leg")
        assert pose.requires_separate_hand_initialization is True
        assert pose.hand_initialization_instruction == "Place the leg first."

    def test_missing_skill_raises(self, tmp_path):
        cfg_path = tmp_path / "skill_config.yaml"
        _write_yaml(cfg_path, "another_skill", {
            "arm_position_rad": _valid_arm(),
            "dex1_opening_fraction": {"left": 0.0, "right": 0.0},
        })
        with pytest.raises(KeyError, match="not registered"):
            load_initial_pose(cfg_path, "missing_skill")

    def test_missing_initial_pose_raises(self, tmp_path):
        cfg_path = tmp_path / "skill_config.yaml"
        _write_yaml(cfg_path, "move_table_base", None)
        with pytest.raises(KeyError, match="no initial_pose"):
            load_initial_pose(cfg_path, "move_table_base")

    def test_wrong_arm_keys_raises(self, tmp_path):
        cfg_path = tmp_path / "skill_config.yaml"
        _write_yaml(cfg_path, "s", {
            "arm_position_rad": {"L.shoulder_pitch": 0.0},  # 1 of 14
            "dex1_opening_fraction": {"left": 0.0, "right": 0.0},
        })
        with pytest.raises(ValueError, match="14 keys"):
            load_initial_pose(cfg_path, "s")

    def test_out_of_range_dex1_raises(self, tmp_path):
        cfg_path = tmp_path / "skill_config.yaml"
        _write_yaml(cfg_path, "s", {
            "arm_position_rad": _valid_arm(),
            "dex1_opening_fraction": {"left": -0.1, "right": 0.5},
        })
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            load_initial_pose(cfg_path, "s")

    def test_actual_repo_config_loads_all_evaluation_skills(self):
        """Regression: all standalone evaluation skills have frame-0 poses."""
        repo_root = Path(__file__).resolve().parents[4]
        cfg_path = (
            repo_root
            / "inference/desktop/lower_policy/configs/skill_config.yaml"
        )
        assert cfg_path.exists(), f"skill_config.yaml not found: {cfg_path}"
        for skill in (
            "move_table_base", "rotate_table_base", "pick_table_leg",
            "insert_table_leg", "rotate_leg_to_tighten", "insert_and_tighten",
            "flip_table",
        ):
            pose = load_initial_pose(cfg_path, skill)
            assert pose.arm_position_rad.shape == (14,)
            assert np.all(np.isfinite(pose.arm_position_rad))
            assert 0.0 <= pose.dex1_opening_fraction[0] <= 1.0
            assert 0.0 <= pose.dex1_opening_fraction[1] <= 1.0

    def test_insert_and_tighten_uses_training_frame0_median(self):
        repo_root = Path(__file__).resolve().parents[4]
        cfg_path = (
            repo_root
            / "inference/desktop/lower_policy/configs/skill_config.yaml"
        )
        pose = load_initial_pose(cfg_path, "insert_and_tighten")
        assert pose.requires_separate_hand_initialization is True
        assert pose.training_episode_count == 531
        assert pose.dataset_revision == "86f3b57dd55ef60b2cd665c7ec49a2380ee8837f"
        np.testing.assert_allclose(
            pose.arm_position_rad[[0, 6, 7, 13]],
            [-0.5541704297065735, -0.9824245572090149,
             -0.3572963774204254, 0.5678691267967224],
        )
        np.testing.assert_allclose(
            pose.dex1_opening_fraction,
            [0.668033070034451, 0.42090747091505265],
        )
