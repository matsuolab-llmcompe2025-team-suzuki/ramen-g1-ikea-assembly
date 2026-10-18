"""Issue #129 Phase 0 D0-2: g1_urdf_fk_torch (torch autograd 対応 FK) の test。

Coverage:
1. numpy 版 (`g1_urdf_fk.G1WristFK.compute_ee_state`) と numerical 一致 (rand q_groot 経由)
2. Batch dim 対応 (chunk_len=16 shape で回るか)
3. Gradient flow (autograd.grad で joint 角度 → 3D 位置の Jacobian が非ゼロ)
4. Constants (buffer shape, chain 長 10 = waist 3 + arm 7)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# torch の無い環境 (root の default env など) では、この file ごと skip する (Issue #192)
torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[4]
URDF_PATH = (
    REPO_ROOT
    / "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf"
)


@pytest.fixture(scope="module")
def fk_torch():
    if not URDF_PATH.exists():
        pytest.skip(f"URDF not found: {URDF_PATH}")
    from inference.desktop.perception.g1_urdf_fk_torch import G1WristFKTorch
    return G1WristFKTorch.from_urdf(urdf_path=str(URDF_PATH), dtype=torch.float64)


@pytest.fixture(scope="module")
def fk_numpy():
    if not URDF_PATH.exists():
        pytest.skip(f"URDF not found: {URDF_PATH}")
    from inference.desktop.perception.g1_urdf_fk import G1WristFK
    return G1WristFK.from_urdf(urdf_path=str(URDF_PATH))


def _groot_to_sdk29(q_groot: np.ndarray) -> np.ndarray:
    """(19,) GR00T action → (29,) SDK joint positions、legs/gripper は 0 で padding。"""
    from model.subtask_policy_training.joint_layout import GROOT_TO_SDK29_INDEX
    q_sdk = np.zeros(29, dtype=np.float64)
    for groot_idx, sdk_idx in enumerate(GROOT_TO_SDK29_INDEX):
        if sdk_idx >= 0:
            q_sdk[sdk_idx] = q_groot[groot_idx]
    return q_sdk


def test_chain_lengths(fk_torch) -> None:
    """chain 長 = waist 3 + arm 7 = 10。"""
    assert fk_torch._left_fixed_Ts.shape == (10, 4, 4)
    assert fk_torch._left_axes.shape == (10, 3)
    assert fk_torch._left_groot_idx.shape == (10,)
    assert fk_torch._right_fixed_Ts.shape == (10, 4, 4)


def test_numerical_agreement_single(fk_torch, fk_numpy) -> None:
    """rand q で torch FK と numpy FK が numerical 一致 (< 1e-6)。"""
    rng = np.random.default_rng(42)
    q_groot = rng.uniform(-0.5, 0.5, size=19)
    # torch forward
    q_t = torch.tensor(q_groot, dtype=torch.float64)
    left_t, right_t = fk_torch(q_t)
    # numpy forward
    q_sdk = _groot_to_sdk29(q_groot)
    ee_np = fk_numpy.compute_ee_state(q_sdk)  # (12,) = left_xyz(3) + left_euler(3) + right_xyz(3) + right_euler(3)
    left_np = ee_np[:3]
    right_np = ee_np[6:9]
    # 3D 位置一致
    np.testing.assert_allclose(left_t.detach().numpy(), left_np, atol=1e-6)
    np.testing.assert_allclose(right_t.detach().numpy(), right_np, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_numerical_agreement_random(fk_torch, fk_numpy, seed: int) -> None:
    """複数 seed で numerical 一致 (regression 予防)。"""
    rng = np.random.default_rng(seed)
    q_groot = rng.uniform(-1.0, 1.0, size=19)
    q_t = torch.tensor(q_groot, dtype=torch.float64)
    left_t, right_t = fk_torch(q_t)
    q_sdk = _groot_to_sdk29(q_groot)
    ee_np = fk_numpy.compute_ee_state(q_sdk)
    np.testing.assert_allclose(left_t.detach().numpy(), ee_np[:3], atol=1e-6)
    np.testing.assert_allclose(right_t.detach().numpy(), ee_np[6:9], atol=1e-6)


def test_batch_dim(fk_torch) -> None:
    """(B, chunk_len, 19) batch shape で forward 通る。"""
    B, T = 2, 16
    q = torch.zeros(B, T, 19, dtype=torch.float64)
    left, right = fk_torch(q)
    assert left.shape == (B, T, 3)
    assert right.shape == (B, T, 3)


def test_batch_matches_single(fk_torch) -> None:
    """batch forward の各要素が single forward と一致。"""
    rng = np.random.default_rng(7)
    q_np = rng.uniform(-0.5, 0.5, size=(3, 19))
    q = torch.tensor(q_np, dtype=torch.float64)
    left_b, right_b = fk_torch(q)
    for i in range(3):
        left_s, right_s = fk_torch(q[i])
        np.testing.assert_allclose(left_b[i].detach().numpy(), left_s.detach().numpy(), atol=1e-9)
        np.testing.assert_allclose(right_b[i].detach().numpy(), right_s.detach().numpy(), atol=1e-9)


def test_gradient_flow(fk_torch) -> None:
    """joint 角度 → 3D 位置の gradient が non-zero (autograd flow 確認)。"""
    q = torch.zeros(19, dtype=torch.float64, requires_grad=True)
    left, right = fk_torch(q)
    loss = (left ** 2).sum() + (right ** 2).sum()
    loss.backward()
    # 左腕 dim 3-9 に非ゼロ gradient (waist も左に効く場合あるが、左腕 index の少なくとも 1 個は非ゼロ)
    assert q.grad is not None
    # waist + left arm + right arm dim (0-16) に少なくとも 1 個は非ゼロ gradient あるはず
    body_grads = q.grad[:17].abs()
    assert (body_grads > 1e-9).any(), f"no gradient flow: {q.grad}"
    # gripper 17,18 は wrist FK 不参加 → gradient == 0
    np.testing.assert_allclose(q.grad[17:19].detach().numpy(), np.zeros(2), atol=1e-12)


def test_left_arm_only_affects_left_wrist(fk_torch) -> None:
    """左腕 index (3-9) だけ動かすと右 wrist は不変、左 wrist だけ動く。"""
    q0 = torch.zeros(19, dtype=torch.float64)
    left0, right0 = fk_torch(q0)
    # 左腕 shoulder pitch (index 3) だけ +0.3 rad
    q1 = q0.clone()
    q1[3] = 0.3
    left1, right1 = fk_torch(q1)
    # 右 wrist は変化なし (< 1e-9)
    np.testing.assert_allclose(right0.numpy(), right1.numpy(), atol=1e-9)
    # 左 wrist は明確に動く (> 1e-3)
    assert (left0 - left1).abs().max() > 1e-3


def test_waist_affects_both_wrists(fk_torch) -> None:
    """waist joint (0-2) を動かすと左右両方の wrist が動く (chain 共通 root)。"""
    q0 = torch.zeros(19, dtype=torch.float64)
    left0, right0 = fk_torch(q0)
    q1 = q0.clone()
    q1[0] = 0.5  # waist_yaw
    left1, right1 = fk_torch(q1)
    # 両手 wrist が動く
    assert (left0 - left1).abs().max() > 1e-3
    assert (right0 - right1).abs().max() > 1e-3


def test_tool_offset_buffers_not_persistent(fk_torch) -> None:
    """Issue #134: tool_offset buffer は persistent=False なので state_dict に
    含まれない。既存 GR00T rotate specialist ckpt (buffer 追加前学習) を新 fk
    定義で load しても missing key にならないことを保証する。"""
    state_keys = set(fk_torch.state_dict().keys())
    assert "_left_tool_offset" not in state_keys, (
        f"_left_tool_offset must be persistent=False (regression: got in state_dict). "
        f"keys={sorted(state_keys)}"
    )
    assert "_right_tool_offset" not in state_keys, (
        f"_right_tool_offset must be persistent=False (regression: got in state_dict). "
        f"keys={sorted(state_keys)}"
    )
    # ただし instance attribute としては存在 (register_buffer は attr 登録もする)
    assert hasattr(fk_torch, "_left_tool_offset")
    assert hasattr(fk_torch, "_right_tool_offset")
