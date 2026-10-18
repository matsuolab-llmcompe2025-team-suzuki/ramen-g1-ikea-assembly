"""G1ArmKinematics (URDF 由来の実 FK/IK) の unit test (Issue #123)。"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.kinematics.base import ArmKinematics
from inference.desktop.lower_policy.kinematics.g1_arm import (
    OPERATIONAL_LIMIT_RAD,
    G1ArmKinematics,
)
from inference.desktop.lower_policy.kinematics.types import (
    EEPose,
    IKStatus,
    Side,
    rpy_to_matrix,
)

# dataset (robot_q_current) から実際に観測された把持前後の腕姿勢。
# 「実機が実際に取る配置」で FK/IK を確かめるための代表点。
DATASET_POSES: dict[Side, list[list[float]]] = {
    Side.RIGHT: [
        [-0.509, -0.294, 0.014, 0.032, 0.424, 0.661, 0.650],
        [-0.312, -0.140, 0.101, 0.560, 0.281, 0.902, 0.884],
        [-0.884, -0.402, -0.115, 0.744, 0.612, 0.399, 0.331],
    ],
    Side.LEFT: [
        [-0.673, 0.200, -0.004, 0.001, -0.530, 0.679, -0.639],
        [-0.451, 0.348, 0.121, 0.503, -0.284, 0.902, -0.401],
        [-0.912, 0.115, -0.220, 0.680, -0.771, 0.401, -0.884],
    ],
}


@pytest.fixture
def kin() -> G1ArmKinematics:
    return G1ArmKinematics()


def test_satisfies_protocol(kin):
    assert isinstance(kin, ArmKinematics)


# ---------------------------------------------------------------- FK


def test_fk_zero_pose_extends_forearm_forward(kin):
    """ゼロ姿勢では前腕が前方 (+x) に伸び、左右で y の符号が分かれる。

    URDF の肘以降のリンクは全て +x オフセット (0.100 + 0.038 + 0.051 + 0.0415)
    なので、全関節ゼロだと腕は水平前方を向く。垂れ下がる姿勢ではない。
    """
    left = kin.fk(np.zeros(7), Side.LEFT)
    right = kin.fk(np.zeros(7), Side.RIGHT)
    assert left.position[1] > 0.05, "左手は y>0 側"
    assert right.position[1] < -0.05, "右手は y<0 側"
    assert 0.20 < left.position[0] < 0.30, "前腕が前方に伸びる"
    np.testing.assert_allclose(left.position[0], right.position[0], atol=2e-3)
    # 肩 (torso から z=0.248) より下に来る
    assert 0.0 < left.position[2] < 0.30


def test_fk_is_mirror_symmetric_between_sides(kin):
    """左右対称な関節指令なら EE も y について鏡像になる。"""
    q = np.array([-0.5, 0.3, 0.2, 0.6, -0.4, 0.5, -0.3])
    q_mirror = q * np.array([1, -1, -1, 1, -1, 1, -1])
    left = kin.fk(q, Side.LEFT)
    right = kin.fk(q_mirror, Side.RIGHT)
    np.testing.assert_allclose(left.position[0], right.position[0], atol=2e-3)
    np.testing.assert_allclose(left.position[1], -right.position[1], atol=2e-3)
    np.testing.assert_allclose(left.position[2], right.position[2], atol=2e-3)


def test_waist_yaw_rotates_the_end_effector(kin):
    """腰を無視できないこと (root_link 基準の EE は腰に依存する)。"""
    q = np.array(DATASET_POSES[Side.RIGHT][0])
    base = kin.fk(q, Side.RIGHT, waist=np.zeros(3))
    yawed = kin.fk(q, Side.RIGHT, waist=np.array([0.15, 0.0, 0.0]))
    shift = base.position_error(yawed)
    assert shift > 0.02, f"腰 0.15 rad が EE に効いていない (shift={shift:.4f} m)"


def test_waist_default_is_zero(kin):
    q = np.array(DATASET_POSES[Side.LEFT][0])
    np.testing.assert_allclose(
        kin.fk(q, Side.LEFT).to_vec6(),
        kin.fk(q, Side.LEFT, waist=np.zeros(3)).to_vec6(),
    )


def test_tcp_offset_shifts_along_gripper_axis(kin):
    q = np.array(DATASET_POSES[Side.RIGHT][0])
    shifted = G1ArmKinematics(tcp_offset=(0.05, 0.0, 0.0))
    a, b = kin.fk(q, Side.RIGHT), shifted.fk(q, Side.RIGHT)
    np.testing.assert_allclose(a.position_error(b), 0.05, atol=1e-9)
    # 姿勢は変わらない。rotation_error は arccos 経由なので恒等回転の近傍で
    # 1e-8 rad 程度の数値ノイズが乗る (stage 許容 0.12 rad に対し無視できる)。
    assert a.rotation_error(b) < 1e-6


# ---------------------------------------------------------------- Jacobian


@pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
def test_jacobian_matches_numerical_derivative(kin, side):
    q = np.array(DATASET_POSES[side][0])
    waist = np.array([0.05, -0.02, 0.03])
    J = kin.jacobian(q, side, waist)
    eps = 1e-6
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        p1 = kin.fk(q + dq, side, waist)
        p0 = kin.fk(q - dq, side, waist)
        # 並進
        np.testing.assert_allclose(
            (p1.position - p0.position) / (2 * eps), J[0:3, i], atol=1e-4
        )
        # 回転 (相対回転の回転ベクトル / 2eps)
        R_rel = p1.matrix @ p0.matrix.T
        w = np.array([R_rel[2, 1] - R_rel[1, 2],
                      R_rel[0, 2] - R_rel[2, 0],
                      R_rel[1, 0] - R_rel[0, 1]]) / 2.0
        np.testing.assert_allclose(w / (2 * eps), J[3:6, i], atol=1e-4)


# ---------------------------------------------------------------- IK


@pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
def test_ik_recovers_dataset_poses_from_warm_seed(kin, side):
    for q in DATASET_POSES[side]:
        q = np.array(q)
        target = kin.fk(q, side)
        seed = np.clip(q + 0.1, -OPERATIONAL_LIMIT_RAD, OPERATIONAL_LIMIT_RAD)
        res = kin.ik(target, seed, side)
        assert res.ok, f"{side} {q}: {res.status}"
        got = kin.fk(res.q, side)
        assert got.position_error(target) < 1e-3
        assert got.rotation_error(target) < 1e-2


@pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
def test_ik_converges_from_cold_seed(kin, side):
    q = np.array(DATASET_POSES[side][0])
    target = kin.fk(q, side)
    res = kin.ik(target, np.zeros(7), side)
    assert res.ok
    assert kin.fk(res.q, side).position_error(target) < 1e-3


def test_ik_respects_waist(kin):
    """腰を渡した FK の目標は、同じ腰を渡した IK で解けること。"""
    q = np.array(DATASET_POSES[Side.RIGHT][0])
    waist = np.array([0.12, 0.03, -0.05])
    target = kin.fk(q, Side.RIGHT, waist)
    res = kin.ik(target, q, Side.RIGHT, waist)
    assert res.ok
    assert kin.fk(res.q, Side.RIGHT, waist).position_error(target) < 1e-3


def test_ik_output_is_inside_joint_limits(kin):
    lim = kin.joint_limits(Side.RIGHT)
    for q in DATASET_POSES[Side.RIGHT]:
        res = kin.ik(kin.fk(np.array(q), Side.RIGHT), np.zeros(7), Side.RIGHT)
        assert np.all(res.q >= lim[:, 0] - 1e-9)
        assert np.all(res.q <= lim[:, 1] + 1e-9)


def test_ik_reports_failure_instead_of_raising_for_unreachable_target(kin):
    far = EEPose.from_vec6([2.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    res = kin.ik(far, np.zeros(7), Side.RIGHT)
    assert not res.ok
    assert res.status in (IKStatus.NOT_CONVERGED, IKStatus.JOINT_LIMIT)
    assert np.isfinite(res.position_error)


def test_ik_never_returns_nan(kin):
    """特異点近傍でも NaN を返さない (actuator に NaN が流れると危険)。"""
    for target in (
        EEPose.from_vec6([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),      # 骨盤原点 = 到達不能
        EEPose.from_vec6([0.35, -0.18, 0.10, 0.19, 0.87, 0.30]),
        EEPose.from_vec6([1.5, 1.5, 1.5, 3.0, 1.5, -3.0]),
    ):
        res = kin.ik(target, np.zeros(7), Side.RIGHT)
        assert np.all(np.isfinite(res.q))


def test_ik_is_fast_enough_for_30hz(kin):
    """1 tick 33.3ms の予算に対し、IK 1 回が十分速いこと。"""
    import time

    q = np.array(DATASET_POSES[Side.RIGHT][0])
    target = kin.fk(q, Side.RIGHT)
    t0 = time.perf_counter()
    for _ in range(50):
        kin.ik(target, np.zeros(7), Side.RIGHT)
    per_call_ms = (time.perf_counter() - t0) / 50 * 1000
    assert per_call_ms < 15.0, f"IK が遅い: {per_call_ms:.1f} ms/call"


# ---------------------------------------------------------------- limits


def test_limits_are_urdf_intersected_with_operational_clamp(kin):
    for side in (Side.LEFT, Side.RIGHT):
        lim = kin.joint_limits(side)
        assert np.all(lim[:, 0] >= -OPERATIONAL_LIMIT_RAD - 1e-12)
        assert np.all(lim[:, 1] <= OPERATIONAL_LIMIT_RAD + 1e-12)
        assert np.all(lim[:, 0] < lim[:, 1])
    # 肩 roll は URDF 側で左右非対称 (左 -1.5882..2.2515 / 右 -2.2515..1.5882)。
    # ±1.5 でクランプすると両方 -1.5..1.5 になるが、非対称性の元データは保つ。
    assert kin.joint_limits(Side.LEFT).shape == (7, 2)


def test_joint_limits_returns_a_copy(kin):
    a = kin.joint_limits(Side.RIGHT)
    a[0, 0] = 99.0
    assert kin.joint_limits(Side.RIGHT)[0, 0] != 99.0


# ---------------------------------------------------------------- validation


def test_rejects_bad_shapes_and_nan(kin):
    with pytest.raises(ValueError, match="shape"):
        kin.fk(np.zeros(6), Side.RIGHT)
    with pytest.raises(ValueError, match="NaN"):
        kin.fk(np.full(7, np.nan), Side.RIGHT)
    with pytest.raises(ValueError, match="shape"):
        kin.fk(np.zeros(7), Side.RIGHT, waist=np.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        kin.ik(EEPose.from_vec6([0.3, 0, 0.2, 0, 0, 0]), np.zeros(3), Side.RIGHT)
    with pytest.raises(ValueError, match="shape"):
        G1ArmKinematics(tcp_offset=(0.0, 0.0))


def test_fk_matches_hand_rolled_reference_at_a_known_configuration(kin):
    """FK を独立計算 (単純な連鎖) と突き合わせ、実装の取り違えを検出する。"""
    q = np.array([-0.3, 0.2, 0.1, 0.5, -0.2, 0.3, -0.1])
    waist = np.array([0.1, 0.05, -0.02])

    def rot(axis, th):
        a = np.asarray(axis, float)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

    chain = [
        ([0, 0, 0], [0, 0, 0], [0, 0, 1], waist[0]),
        ([-0.0039635, 0, 0.044], [0, 0, 0], [1, 0, 0], waist[1]),
        ([0, 0, 0], [0, 0, 0], [0, 1, 0], waist[2]),
        ([0.0039563, 0.10022, 0.24778], [0.27931, 5.4949e-05, -0.00019159], [0, 1, 0], q[0]),
        ([0, 0.038, -0.013831], [-0.27925, 0, 0], [1, 0, 0], q[1]),
        ([0, 0.00624, -0.1032], [0, 0, 0], [0, 0, 1], q[2]),
        ([0.015783, 0, -0.080518], [0, 0, 0], [0, 1, 0], q[3]),
        ([0.100, 0.00188791, -0.010], [0, 0, 0], [1, 0, 0], q[4]),
        ([0.038, 0, 0], [0, 0, 0], [0, 1, 0], q[5]),
        ([0.051, 0, 0], [0, 0, 0], [0, 0, 1], q[6]),
    ]
    p = np.zeros(3)
    R = np.eye(3)
    for xyz, rpy, axis, th in chain:
        p = p + R @ np.asarray(xyz, float)
        R = R @ rpy_to_matrix(np.asarray(rpy)) @ rot(axis, th)
    p = p + R @ np.array([0.0415, 0.0, 0.0])

    got = kin.fk(q, Side.LEFT, waist)
    np.testing.assert_allclose(got.position, p, atol=1e-12)
