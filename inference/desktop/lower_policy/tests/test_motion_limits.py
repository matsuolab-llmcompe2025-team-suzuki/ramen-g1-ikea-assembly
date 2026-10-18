"""MotionLimits + load_motion_limits_for_skill contract test (Phase 3、Issue #128)。

Layer 1 の per-skill envelope schema (dataclass + YAML loader) の unit test。
実 apply の smoothing 挙動は test_motion_limiter.py が担当、ここは config 解決の
correctness のみ。
"""

from __future__ import annotations

import math

import pytest

from inference.desktop.lower_policy.motion_limits import (
    DEFAULT_MOTION_LIMITS,
    MotionLimits,
    load_motion_limits_for_skill,
)


class TestMotionLimitsDataclass:
    def _valid_dict(self, **overrides) -> dict:
        base = {
            "arm_velocity_rad_s": 2.0,
            "arm_acceleration_rad_s2": 10.0,
            "hand_velocity_rad_s": 6.0,
            "hand_acceleration_rad_s2": 30.0,
        }
        base.update(overrides)
        return base

    def test_default_module_constant(self) -> None:
        assert isinstance(DEFAULT_MOTION_LIMITS, MotionLimits)
        assert DEFAULT_MOTION_LIMITS.arm_velocity_rad_s == 2.0
        assert DEFAULT_MOTION_LIMITS.arm_acceleration_rad_s2 == 10.0
        assert DEFAULT_MOTION_LIMITS.hand_velocity_rad_s == 6.0
        assert DEFAULT_MOTION_LIMITS.hand_acceleration_rad_s2 == 30.0

    def test_from_config_dict_all_fields(self) -> None:
        limits = MotionLimits.from_config_dict(self._valid_dict())
        assert limits.arm_velocity_rad_s == 2.0
        assert limits.arm_acceleration_rad_s2 == 10.0
        assert limits.hand_velocity_rad_s == 6.0
        assert limits.hand_acceleration_rad_s2 == 30.0

    def test_from_config_dict_missing_key_raises(self) -> None:
        for missing in (
            "arm_velocity_rad_s",
            "arm_acceleration_rad_s2",
            "hand_velocity_rad_s",
            "hand_acceleration_rad_s2",
        ):
            cfg = self._valid_dict()
            del cfg[missing]
            with pytest.raises(KeyError, match=missing):
                MotionLimits.from_config_dict(cfg)

    def test_rejects_non_positive(self) -> None:
        for key in (
            "arm_velocity_rad_s",
            "arm_acceleration_rad_s2",
            "hand_velocity_rad_s",
            "hand_acceleration_rad_s2",
        ):
            with pytest.raises(ValueError):
                MotionLimits.from_config_dict(self._valid_dict(**{key: 0.0}))
            with pytest.raises(ValueError):
                MotionLimits.from_config_dict(self._valid_dict(**{key: -1.0}))

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError):
            MotionLimits.from_config_dict(
                self._valid_dict(arm_velocity_rad_s=float("nan"))
            )
        with pytest.raises(ValueError):
            MotionLimits.from_config_dict(
                self._valid_dict(arm_acceleration_rad_s2=float("inf"))
            )

    def test_immutable(self) -> None:
        limits = MotionLimits.from_config_dict(self._valid_dict())
        with pytest.raises(Exception):
            limits.arm_velocity_rad_s = 999.0  # type: ignore[misc]


class TestLoadMotionLimitsForSkill:
    """`load_motion_limits_for_skill` の resolution order + merge behavior。"""

    def test_hard_fallback_when_no_config(self) -> None:
        cfg: dict = {}
        limits = load_motion_limits_for_skill(cfg, "any_skill")
        assert limits == DEFAULT_MOTION_LIMITS

    def test_uses_motion_limits_default_when_present(self) -> None:
        cfg = {
            "motion_limits_default": {
                "arm_velocity_rad_s": 3.0,
                "arm_acceleration_rad_s2": 15.0,
                "hand_velocity_rad_s": 7.0,
                "hand_acceleration_rad_s2": 35.0,
            },
        }
        limits = load_motion_limits_for_skill(cfg, "any_skill")
        assert limits.arm_velocity_rad_s == 3.0
        assert limits.arm_acceleration_rad_s2 == 15.0
        assert limits.hand_velocity_rad_s == 7.0
        assert limits.hand_acceleration_rad_s2 == 35.0

    def test_per_skill_override_full_replaces_default(self) -> None:
        cfg = {
            "motion_limits_default": {
                "arm_velocity_rad_s": 2.0,
                "arm_acceleration_rad_s2": 10.0,
                "hand_velocity_rad_s": 6.0,
                "hand_acceleration_rad_s2": 30.0,
            },
            "skills": {
                "flip_table": {
                    "motion_limits": {
                        "arm_velocity_rad_s": 1.5,
                        "arm_acceleration_rad_s2": 10.0,
                        "hand_velocity_rad_s": 5.0,
                        "hand_acceleration_rad_s2": 25.0,
                    },
                },
            },
        }
        limits = load_motion_limits_for_skill(cfg, "flip_table")
        assert limits.arm_velocity_rad_s == 1.5
        assert limits.arm_acceleration_rad_s2 == 10.0
        assert limits.hand_velocity_rad_s == 5.0
        assert limits.hand_acceleration_rad_s2 == 25.0

    def test_per_skill_partial_override_merges_with_default(self) -> None:
        # user 判断: YAML の使い勝手向上 (partial override は default に merge)。
        cfg = {
            "motion_limits_default": {
                "arm_velocity_rad_s": 2.0,
                "arm_acceleration_rad_s2": 10.0,
                "hand_velocity_rad_s": 6.0,
                "hand_acceleration_rad_s2": 30.0,
            },
            "skills": {
                "flip_table": {
                    "motion_limits": {
                        # arm_velocity のみ override
                        "arm_velocity_rad_s": 1.5,
                    },
                },
            },
        }
        limits = load_motion_limits_for_skill(cfg, "flip_table")
        assert limits.arm_velocity_rad_s == 1.5
        # 残 3 field は default から継承
        assert limits.arm_acceleration_rad_s2 == 10.0
        assert limits.hand_velocity_rad_s == 6.0
        assert limits.hand_acceleration_rad_s2 == 30.0

    def test_per_skill_override_uses_hard_fallback_when_no_default(self) -> None:
        # motion_limits_default が無いが per-skill が partial → hard fallback を base に
        cfg = {
            "skills": {
                "flip_table": {
                    "motion_limits": {
                        "arm_velocity_rad_s": 1.5,
                    },
                },
            },
        }
        limits = load_motion_limits_for_skill(cfg, "flip_table")
        assert limits.arm_velocity_rad_s == 1.5
        # 残 3 field は DEFAULT_MOTION_LIMITS 継承
        assert limits.arm_acceleration_rad_s2 == DEFAULT_MOTION_LIMITS.arm_acceleration_rad_s2
        assert limits.hand_velocity_rad_s == DEFAULT_MOTION_LIMITS.hand_velocity_rad_s

    def test_missing_skill_falls_back_to_default(self) -> None:
        cfg = {
            "motion_limits_default": {
                "arm_velocity_rad_s": 3.0,
                "arm_acceleration_rad_s2": 15.0,
                "hand_velocity_rad_s": 7.0,
                "hand_acceleration_rad_s2": 35.0,
            },
            "skills": {"other_skill": {"motion_limits": {"arm_velocity_rad_s": 99.0}}},
        }
        limits = load_motion_limits_for_skill(cfg, "flip_table")
        assert limits.arm_velocity_rad_s == 3.0  # default

    def test_rejects_unknown_key_in_override(self) -> None:
        # typo protection: unknown key があれば error (YAML の凡ミスを早期検出)
        cfg = {
            "skills": {
                "flip_table": {
                    "motion_limits": {
                        "arm_veloctiy_rad_s": 1.5,  # typo: veloctiy
                    },
                },
            },
        }
        with pytest.raises(KeyError, match="arm_veloctiy_rad_s"):
            load_motion_limits_for_skill(cfg, "flip_table")

    def test_rejects_non_mapping_override(self) -> None:
        cfg = {"skills": {"flip_table": {"motion_limits": [1, 2, 3]}}}
        with pytest.raises(TypeError):
            load_motion_limits_for_skill(cfg, "flip_table")


class TestRealYAMLFileStructure:
    """実 skill_config.yaml が期待通り load できることの smoke test。"""

    def test_load_real_yaml_all_skills(self) -> None:
        # inference/desktop/lower_policy/configs/skill_config.yaml を open し、
        # 全 skill (setup / move_to_table 除く VLA evaluation skill) について load OK。
        import pathlib
        import yaml

        yaml_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "configs" / "skill_config.yaml"
        )
        assert yaml_path.exists(), f"skill_config not found: {yaml_path}"
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert "motion_limits_default" in cfg
        # 各 skill が正しく load できること (partial override 含む)
        for skill_name in (
            "move_table_base",
            "rotate_table_base",
            "pick_table_leg",
            "insert_table_leg",
            "rotate_leg_to_tighten",
            "insert_and_tighten",
            "flip_table",
        ):
            limits = load_motion_limits_for_skill(cfg, skill_name)
            # 全 field が finite / positive
            for name in (
                "arm_velocity_rad_s",
                "arm_acceleration_rad_s2",
                "hand_velocity_rad_s",
                "hand_acceleration_rad_s2",
            ):
                value = getattr(limits, name)
                assert math.isfinite(value) and value > 0.0, (
                    f"{skill_name}.{name} = {value}"
                )

    def test_flip_table_matches_issue70_verified_values(self) -> None:
        # flip_table の arm_velocity_rad_s = 1.5、arm_acceleration_rad_s2 = 10.0 は
        # issue-70 で dataset-derived verified 値。YAML から load しても保たれること。
        import pathlib
        import yaml

        yaml_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "configs" / "skill_config.yaml"
        )
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        limits = load_motion_limits_for_skill(cfg, "flip_table")
        assert limits.arm_velocity_rad_s == 1.5
        assert limits.arm_acceleration_rad_s2 == 10.0

    def test_insert_table_leg_more_conservative_than_default(self) -> None:
        # insert = 精密 (最難関) skill、default より envelope 狭い
        import pathlib
        import yaml

        yaml_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "configs" / "skill_config.yaml"
        )
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        limits = load_motion_limits_for_skill(cfg, "insert_table_leg")
        default_arm_vel = cfg["motion_limits_default"]["arm_velocity_rad_s"]
        assert limits.arm_velocity_rad_s < default_arm_vel, (
            f"insert velocity {limits.arm_velocity_rad_s} not < default {default_arm_vel}"
        )
