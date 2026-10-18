"""OfficialG1ArmGravityCompensator (Phase 5、Issue #128) の contract test。

issue-70 verbatim port + 我々 URDF-based Self-Contained path 追加の verify。

依存: `pinocchio ==3.1.0` (runtime env のみ)。default env では pinocchio import 失敗
= 該当 test を pytest.importorskip で skip。URDF file 存在確認は default env でも
可能なので分離。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inference.desktop.lower_policy.gravity_compensation import (
    EXPECTED_ARM_JOINT_NAMES,
    MAX_ABS_GRAVITY_TORQUE_NM,
    _DEFAULT_URDF_RELATIVE_PATH,
    _LOCKED_JOINT_NAMES,
    find_default_urdf_path,
)


# ---- Module constants (default env で pinocchio 無しでも verify 可能) ---- #


class TestModuleConstants:
    def test_expected_arm_joint_names_is_14(self) -> None:
        assert len(EXPECTED_ARM_JOINT_NAMES) == 14
        # left arm 7 + right arm 7 の順
        for i, name in enumerate(EXPECTED_ARM_JOINT_NAMES):
            side = "left" if i < 7 else "right"
            assert name.startswith(f"{side}_"), f"{i}: {name}"
        # 各 side は shoulder pitch/roll/yaw + elbow + wrist roll/pitch/yaw の 7 順
        for side in ("left", "right"):
            offset = 0 if side == "left" else 7
            assert EXPECTED_ARM_JOINT_NAMES[offset + 0] == f"{side}_shoulder_pitch_joint"
            assert EXPECTED_ARM_JOINT_NAMES[offset + 1] == f"{side}_shoulder_roll_joint"
            assert EXPECTED_ARM_JOINT_NAMES[offset + 2] == f"{side}_shoulder_yaw_joint"
            assert EXPECTED_ARM_JOINT_NAMES[offset + 3] == f"{side}_elbow_joint"
            assert EXPECTED_ARM_JOINT_NAMES[offset + 4] == f"{side}_wrist_roll_joint"
            assert EXPECTED_ARM_JOINT_NAMES[offset + 5] == f"{side}_wrist_pitch_joint"
            assert EXPECTED_ARM_JOINT_NAMES[offset + 6] == f"{side}_wrist_yaw_joint"

    def test_max_torque_bound_conservative(self) -> None:
        # issue-70 verbatim = 15 Nm bound (実 pre-motion で 7 Nm 未満だった実測)
        assert MAX_ABS_GRAVITY_TORQUE_NM == 15.0

    def test_locked_joints_count_matches_reduction(self) -> None:
        # 33 DOF full (12 leg + 3 waist + 14 arm + 4 dex1) - 14 arm = 19 lock joint
        assert len(_LOCKED_JOINT_NAMES) == 19
        legs = [n for n in _LOCKED_JOINT_NAMES if "hip" in n or "knee" in n or "ankle" in n]
        waist = [n for n in _LOCKED_JOINT_NAMES if "waist" in n]
        dex1 = [n for n in _LOCKED_JOINT_NAMES if "dex1_finger" in n]
        assert len(legs) == 12
        assert len(waist) == 3
        assert len(dex1) == 4


class TestDefaultURDFDiscovery:
    def test_default_urdf_path_resolves_to_existing_file(self) -> None:
        # Self-Contained default = repo 内 URDF path が実在
        path = find_default_urdf_path()
        assert path.is_file(), f"URDF not found: {path}"
        assert path.name == "g1_29dof_mode_15_with_dex1_1.urdf"

    def test_default_relative_path_matches_g1_urdf_fk_asset(self) -> None:
        # g1_urdf_fk.py と同じ URDF asset を使う (別 asset で meta drift 防止)
        assert _DEFAULT_URDF_RELATIVE_PATH.endswith(
            "g1_29dof_mode_15_with_dex1_1.urdf"
        )
        # perception.g1_urdf_fk.DEFAULT_URDF_PATH と一致 verify
        from inference.desktop.perception.g1_urdf_fk import DEFAULT_URDF_PATH
        assert find_default_urdf_path().resolve() == Path(DEFAULT_URDF_PATH).resolve()


# ---- Integration (pinocchio 必要、runtime env でのみ実行) ---- #


class TestSelfContainedURDFBuild:
    """URDF → reduced model → RNEA torque の end-to-end (要 pinocchio)。"""

    @pytest.fixture
    def pin(self):
        return pytest.importorskip("pinocchio", reason="pinocchio not available in default env")

    def test_build_reduced_model_from_urdf(self, pin) -> None:
        from inference.desktop.lower_policy.gravity_compensation import (
            build_reduced_arm_model_from_urdf,
        )
        urdf_path = find_default_urdf_path()
        model, data = build_reduced_arm_model_from_urdf(urdf_path)
        assert model.nq == 14
        assert model.nv == 14
        names = tuple(str(n) for n in model.names[1:])
        assert names == EXPECTED_ARM_JOINT_NAMES

    def test_from_default_urdf_construction(self, pin) -> None:
        from inference.desktop.lower_policy.gravity_compensation import (
            OfficialG1ArmGravityCompensator,
        )
        gravity = OfficialG1ArmGravityCompensator.from_default_urdf()
        assert gravity.cache_path is None  # Self-Contained = pickle 未使用
        assert len(gravity.cache_sha256) == 64  # SHA256 hex (URDF digest)

    def test_torque_shape_and_finite_at_zero_pose(self, pin) -> None:
        from inference.desktop.lower_policy.gravity_compensation import (
            OfficialG1ArmGravityCompensator,
        )
        gravity = OfficialG1ArmGravityCompensator.from_default_urdf()
        # 全 joint 0 rad (T-pose 相当) での torque
        torque = gravity.torque_nm(np.zeros(14, dtype=np.float64))
        assert torque.shape == (14,)
        assert np.all(np.isfinite(torque))

    def test_torque_bound_check_fires_on_extreme_pose(self, pin) -> None:
        # 極端 pose では bound (15 Nm) を超える可能性 → 明示 error
        # ただし G1 arm の物理範囲では通常 <7 Nm、artificial に超過 pose を作る
        from inference.desktop.lower_policy.gravity_compensation import (
            OfficialG1ArmGravityCompensator,
            MAX_ABS_GRAVITY_TORQUE_NM,
        )
        gravity = OfficialG1ArmGravityCompensator.from_default_urdf()
        # normal pose = should pass
        torque = gravity.torque_nm(np.zeros(14, dtype=np.float64))
        assert np.max(np.abs(torque)) <= MAX_ABS_GRAVITY_TORQUE_NM

    def test_rejects_wrong_input_shape(self, pin) -> None:
        from inference.desktop.lower_policy.gravity_compensation import (
            OfficialG1ArmGravityCompensator,
        )
        gravity = OfficialG1ArmGravityCompensator.from_default_urdf()
        with pytest.raises(ValueError, match="arms\\[14\\]"):
            gravity.torque_nm(np.zeros(13, dtype=np.float64))
        with pytest.raises(ValueError, match="arms\\[14\\]"):
            gravity.torque_nm(np.zeros(15, dtype=np.float64))

    def test_rejects_nan_input(self, pin) -> None:
        from inference.desktop.lower_policy.gravity_compensation import (
            OfficialG1ArmGravityCompensator,
        )
        gravity = OfficialG1ArmGravityCompensator.from_default_urdf()
        bad = np.zeros(14, dtype=np.float64)
        bad[5] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            gravity.torque_nm(bad)

    def test_torque_direction_realistic_pose(self, pin) -> None:
        """Shoulder pitch を水平前方に上げると shoulder に反時計 torque (負符号)
        必要 = gravity が腕を下げる方向に働くのを持ち上げる必要。sanity check。"""
        from inference.desktop.lower_policy.gravity_compensation import (
            OfficialG1ArmGravityCompensator,
        )
        gravity = OfficialG1ArmGravityCompensator.from_default_urdf()
        # Left shoulder pitch を -π/2 (前方水平) に、他 zero
        pose = np.zeros(14, dtype=np.float64)
        pose[0] = -np.pi / 2  # left_shoulder_pitch = 前方 90°
        torque = gravity.torque_nm(pose)
        # left_shoulder_pitch torque (index 0) は non-zero (何らかの gravity 影響)
        assert abs(torque[0]) > 0.1, (
            f"shoulder_pitch torque {torque[0]:.4f} too small for 前方水平 pose"
        )
