"""kinematics の型 / 回転ユーティリティ / mock の unit test (Issue #123)。"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.kinematics.mock import MockArmKinematics
from inference.desktop.lower_policy.kinematics.types import (
    EEPose,
    IKResult,
    IKStatus,
    Side,
    matrix_to_rpy,
    rpy_to_matrix,
)


# ---------------------------------------------------------------- rotation


@pytest.mark.parametrize(
    "rpy",
    [
        (0.0, 0.0, 0.0),
        (0.2, 0.83, -0.5),          # 把持姿勢の実測レンジ (pitch ~0.83)
        (-1.2, 0.4, 2.9),
        (0.5, -1.4, -3.0),
        (0.199, 0.828, 0.234),      # dataset の把持時 rpy median
    ],
)
def test_rpy_matrix_roundtrip(rpy):
    R = rpy_to_matrix(np.array(rpy))
    np.testing.assert_allclose(matrix_to_rpy(R), np.array(rpy), atol=1e-9)


def test_rpy_to_matrix_is_orthonormal():
    R = rpy_to_matrix(np.array([0.7, -0.3, 1.1]))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_matrix_to_rpy_gimbal_lock_is_finite():
    """pitch = +-pi/2 でも NaN を出さず、roll を 0 に寄せる。"""
    for pitch in (np.pi / 2, -np.pi / 2):
        R = rpy_to_matrix(np.array([0.4, pitch, 1.0]))
        out = matrix_to_rpy(R)
        assert np.all(np.isfinite(out))
        assert np.isclose(out[0], 0.0, atol=1e-8)
        assert np.isclose(abs(out[1]), np.pi / 2, atol=1e-8)


# ---------------------------------------------------------------- EEPose


def test_eepose_vec6_roundtrip():
    vec = [0.351, -0.187, 0.117, 0.199, 0.828, 0.234]
    pose = EEPose.from_vec6(vec)
    np.testing.assert_allclose(pose.to_vec6(), np.array(vec))


def test_eepose_rejects_bad_shape_and_nan():
    with pytest.raises(ValueError, match="shape"):
        EEPose(position=np.zeros(2), rpy=np.zeros(3))
    with pytest.raises(ValueError, match="NaN"):
        EEPose(position=np.array([0.0, np.nan, 0.0]), rpy=np.zeros(3))
    with pytest.raises(ValueError, match="6 values"):
        EEPose.from_vec6([1.0, 2.0, 3.0])


def test_eepose_position_and_rotation_error():
    a = EEPose.from_vec6([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = EEPose.from_vec6([0.03, 0.04, 0.0, 0.0, 0.0, 0.5])
    assert np.isclose(a.position_error(b), 0.05)
    assert np.isclose(a.rotation_error(b), 0.5)
    # 同一 pose なら誤差ゼロ
    assert np.isclose(a.rotation_error(a), 0.0, atol=1e-9)


def test_eepose_rotation_error_is_not_rpy_difference():
    """rpy の各成分差ではなく相対回転角であることを確認。

    roll=pi, yaw=pi は rpy 上は離れているが、実際の回転は pitch 軸まわり pi の
    1 回転ぶんしか離れていない。
    """
    a = EEPose.from_vec6([0, 0, 0, np.pi, 0.0, 0.0])
    b = EEPose.from_vec6([0, 0, 0, 0.0, 0.0, np.pi])
    err = a.rotation_error(b)
    assert 0.0 <= err <= np.pi + 1e-9
    assert np.isclose(err, np.pi, atol=1e-6)


def test_eepose_interpolate_endpoints():
    a = EEPose.from_vec6([0.35, -0.19, 0.12, 0.2, 0.83, 0.23])
    b = EEPose.from_vec6([0.30, -0.14, 0.19, 0.0, 0.30, -0.10])
    np.testing.assert_allclose(a.interpolate(b, 0.0).to_vec6(), a.to_vec6(), atol=1e-9)
    np.testing.assert_allclose(a.interpolate(b, 1.0).to_vec6(), b.to_vec6(), atol=1e-9)


def test_eepose_interpolate_clips_out_of_range_t():
    a = EEPose.from_vec6([0, 0, 0, 0, 0, 0])
    b = EEPose.from_vec6([1, 0, 0, 0, 0, 1.0])
    np.testing.assert_allclose(a.interpolate(b, -5.0).to_vec6(), a.to_vec6(), atol=1e-9)
    np.testing.assert_allclose(a.interpolate(b, 9.0).to_vec6(), b.to_vec6(), atol=1e-9)


def test_eepose_interpolate_position_is_linear():
    a = EEPose.from_vec6([0.0, 0.0, 0.0, 0, 0, 0])
    b = EEPose.from_vec6([0.4, 0.2, -0.1, 0, 0, 0])
    mid = a.interpolate(b, 0.5)
    np.testing.assert_allclose(mid.position, np.array([0.2, 0.1, -0.05]), atol=1e-12)


def test_eepose_interpolate_rotation_is_monotonic_geodesic():
    """補間点が始点から離れ、終点に近づくこと (rpy 線形補間だと保証されない)。"""
    a = EEPose.from_vec6([0, 0, 0, 0.0, 0.0, 0.0])
    b = EEPose.from_vec6([0, 0, 0, 0.5, 0.9, -2.4])
    total = a.rotation_error(b)
    prev_from_a = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mid = a.interpolate(b, float(t))
        from_a = a.rotation_error(mid)
        assert from_a >= prev_from_a - 1e-9
        prev_from_a = from_a
        # 測地線上なので「始点まで + 終点まで」が全体距離に一致する
        assert np.isclose(from_a + mid.rotation_error(b), total, atol=1e-6)
    assert np.isclose(prev_from_a, total, atol=1e-9)


def test_eepose_interpolate_stays_valid_rotation():
    a = EEPose.from_vec6([0, 0, 0, 0.0, 0.0, 0.0])
    b = EEPose.from_vec6([0, 0, 0, 3.0, -1.2, 2.5])
    for t in np.linspace(0, 1, 9):
        R = a.interpolate(b, float(t)).matrix
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_eepose_interpolate_handles_pi_rotation():
    """theta ~ pi の分岐 (対称行列からの軸復元) が破綻しないこと。"""
    a = EEPose.from_vec6([0, 0, 0, 0.0, 0.0, 0.0])
    b = EEPose.from_vec6([0, 0, 0, np.pi, 0.0, 0.0])
    mid = a.interpolate(b, 0.5)
    R = mid.matrix
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(a.rotation_error(mid), np.pi / 2, atol=1e-6)


# ---------------------------------------------------------------- IKResult


def test_ik_result_ok_property_and_shape_validation():
    r = IKResult(IKStatus.OK, np.zeros(7), 0.0, 0.0, 3)
    assert r.ok
    assert not IKResult(IKStatus.JOINT_LIMIT, np.zeros(7), 0.0, 0.0, 3).ok
    with pytest.raises(ValueError, match="shape"):
        IKResult(IKStatus.OK, np.zeros(6), 0.0, 0.0, 1)


# ---------------------------------------------------------------- Mock


def test_mock_fk_ik_roundtrip_both_sides():
    kin = MockArmKinematics()
    for side in (Side.LEFT, Side.RIGHT):
        target = EEPose.from_vec6([0.30, 0.05, 0.20, 0.1, 0.4, -0.2])
        res = kin.ik(target, seed=np.zeros(7), side=side)
        assert res.ok
        back = kin.fk(res.q, side)
        np.testing.assert_allclose(back.to_vec6(), target.to_vec6(), atol=1e-12)


def test_mock_ik_preserves_redundant_axis_from_seed():
    kin = MockArmKinematics()
    seed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.77])
    res = kin.ik(EEPose.from_vec6([0.2, 0, 0.1, 0, 0, 0]), seed, Side.RIGHT)
    assert np.isclose(res.q[6], 0.77)


def test_mock_ik_reports_joint_limit():
    kin = MockArmKinematics(joint_limit_rad=0.5)
    res = kin.ik(EEPose.from_vec6([0, 0, 0, 0, 0, 3.0]), np.zeros(7), Side.RIGHT)
    assert res.status is IKStatus.JOINT_LIMIT
    assert not res.ok


def test_mock_ik_reports_unreachable():
    kin = MockArmKinematics(unreachable_radius=0.5)
    res = kin.ik(EEPose.from_vec6([2.0, 0, 0, 0, 0, 0]), np.zeros(7), Side.RIGHT)
    assert res.status is IKStatus.NOT_CONVERGED


def test_mock_ik_force_status_overrides():
    kin = MockArmKinematics(force_status=IKStatus.SINGULAR)
    res = kin.ik(EEPose.from_vec6([0.1, 0, 0, 0, 0, 0]), np.zeros(7), Side.LEFT)
    assert res.status is IKStatus.SINGULAR


def test_mock_records_ik_calls():
    kin = MockArmKinematics()
    kin.ik(EEPose.from_vec6([0.1, 0, 0, 0, 0, 0]), np.zeros(7), Side.LEFT)
    kin.ik(EEPose.from_vec6([0.2, 0, 0, 0, 0, 0]), np.zeros(7), Side.RIGHT)
    assert len(kin.ik_calls) == 2
    assert kin.ik_calls[1][1] is Side.RIGHT


def test_mock_rejects_bad_shapes():
    kin = MockArmKinematics()
    with pytest.raises(ValueError, match="shape"):
        kin.fk(np.zeros(6), Side.LEFT)
    with pytest.raises(ValueError, match="shape"):
        kin.ik(EEPose.from_vec6([0, 0, 0, 0, 0, 0]), np.zeros(3), Side.LEFT)


def test_mock_satisfies_protocol():
    from inference.desktop.lower_policy.kinematics.base import ArmKinematics

    assert isinstance(MockArmKinematics(), ArmKinematics)
