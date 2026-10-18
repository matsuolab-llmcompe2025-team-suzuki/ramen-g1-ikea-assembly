"""obb_warp.py の unit test (Issue #129 Phase E、2026-08-31)。

Coord token を geometric aug (RandomAffine) と一貫させるための helper の検証:
- Identity affine → verts 不変
- Pure translation → verts が同 (tx, ty) 平行移動
- Pure rotation → 特定 point が回転後の期待位置に到達
- Combined transformation
- Batch shape (top_K 単独、(B, top_K)、(B, N_cams, top_K))
- clamp フラグ、shape/dim mismatch raise
"""

from __future__ import annotations

import math

import pytest
import torch

from model.ramen_ori.obb_warp import (
    make_rotation_translation_affine,
    warp_obb_coord_under_affine,
)


class TestIdentity:
    def test_identity_affine_preserves_verts(self) -> None:
        verts = torch.rand(4, 8)  # (top_K=4, 8)
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = warp_obb_coord_under_affine(verts, identity)
        torch.testing.assert_close(out, verts)

    def test_identity_preserves_dtype(self) -> None:
        verts_f64 = torch.rand(4, 8, dtype=torch.float64)
        identity = torch.eye(2, 3, dtype=torch.float64)
        out = warp_obb_coord_under_affine(verts_f64, identity)
        assert out.dtype == torch.float64


class TestTranslation:
    def test_translation_shifts_all_verts_by_same_offset(self) -> None:
        verts = torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4]])  # (1, 8)
        # Pure translation +0.5 in x, +0.1 in y
        trans = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.1]])
        out = warp_obb_coord_under_affine(verts, trans)
        expected = torch.tensor([[0.6, 0.3, 0.8, 0.3, 0.8, 0.5, 0.6, 0.5]])
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


class TestRotation:
    def test_rotation_90_around_origin(self) -> None:
        """90° rotation around origin: (x, y) → (-y, x)。"""
        # 1 vert at (1, 0): 90° → (0, 1)
        # verts (top_K=1, 8) = 4 頂点 全て同 (1, 0) で簡単化
        verts = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
        rot90 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
        out = warp_obb_coord_under_affine(verts, rot90)
        expected = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)

    def test_rotation_around_image_center_preserves_center(self) -> None:
        """画像中心 (0.5, 0.5) 周りの rotation で center vert は不変。"""
        # verts: center (0.5, 0.5) + 他 3 頂点
        verts = torch.tensor([[0.5, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6]])
        affine = make_rotation_translation_affine(angle_rad=math.radians(30), center=(0.5, 0.5))
        out = warp_obb_coord_under_affine(verts, affine)
        # center 頂点 (index 0, 1) は不変
        torch.testing.assert_close(out[0, :2], verts[0, :2], atol=1e-6, rtol=0)


class TestCombined:
    def test_rotation_plus_translation_via_helper(self) -> None:
        """`make_rotation_translation_affine` の結果を verify (analytic 期待値と一致)。"""
        verts = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])  # 全 center
        # center (0.5, 0.5) 周り 0° rotation + tx=0.1, ty=-0.05
        affine = make_rotation_translation_affine(angle_rad=0.0, tx=0.1, ty=-0.05)
        out = warp_obb_coord_under_affine(verts, affine)
        # center → (0.6, 0.45)
        torch.testing.assert_close(
            out, torch.tensor([[0.6, 0.45, 0.6, 0.45, 0.6, 0.45, 0.6, 0.45]]),
            atol=1e-6, rtol=0,
        )


class TestBatchShapes:
    def test_top_K_only(self) -> None:
        verts = torch.rand(4, 8)  # (top_K,)
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = warp_obb_coord_under_affine(verts, identity)
        assert out.shape == (4, 8)

    def test_batch_dim(self) -> None:
        verts = torch.rand(2, 4, 8)  # (B, top_K, 8)
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = warp_obb_coord_under_affine(verts, identity)
        assert out.shape == (2, 4, 8)

    def test_multi_batch_dim(self) -> None:
        verts = torch.rand(2, 3, 4, 8)  # (B, N_cams, top_K, 8)
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = warp_obb_coord_under_affine(verts, identity)
        assert out.shape == (2, 3, 4, 8)


class TestClampFlag:
    def test_clamp_true_clips_outside(self) -> None:
        verts = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        # +1.0 translate → 座標が [0,1] を越える
        trans = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        out = warp_obb_coord_under_affine(verts, trans, clamp=True)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    def test_clamp_false_allows_outside(self) -> None:
        verts = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        trans = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        out = warp_obb_coord_under_affine(verts, trans, clamp=False)
        # (0.5 + 1.0) = 1.5 が確保される
        assert out.max().item() == pytest.approx(1.5, rel=1e-6)


class TestRaises:
    def test_wrong_verts_dim_raises(self) -> None:
        verts_bad = torch.rand(4, 6)  # 8 でない
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="verts last dim must be 8"):
            warp_obb_coord_under_affine(verts_bad, identity)

    def test_wrong_affine_shape_raises(self) -> None:
        verts = torch.rand(4, 8)
        affine_bad = torch.eye(3)  # (3, 3) not (2, 3)
        with pytest.raises(ValueError, match="affine_matrix must be"):
            warp_obb_coord_under_affine(verts, affine_bad)


class TestGradient:
    def test_gradient_flows_through_verts(self) -> None:
        """verts.grad が流れる (autograd 対応、正常な affine 演算)。"""
        verts = torch.rand(4, 8, requires_grad=True)
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = warp_obb_coord_under_affine(verts, identity)
        loss = out.sum()
        loss.backward()
        assert verts.grad is not None
        assert verts.grad.abs().sum().item() > 0


class TestMakeRotationTranslationAffine:
    def test_identity_case(self) -> None:
        affine = make_rotation_translation_affine(angle_rad=0.0, tx=0.0, ty=0.0)
        assert affine.shape == (2, 3)
        # Identity 相当: (x, y) → (x, y)
        verts = torch.tensor([[0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7]])
        out = warp_obb_coord_under_affine(verts, affine)
        torch.testing.assert_close(out, verts, atol=1e-6, rtol=0)

    def test_translation_only(self) -> None:
        affine = make_rotation_translation_affine(angle_rad=0.0, tx=0.1, ty=-0.05)
        verts = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        out = warp_obb_coord_under_affine(verts, affine)
        torch.testing.assert_close(
            out, torch.tensor([[0.6, 0.45, 0.6, 0.45, 0.6, 0.45, 0.6, 0.45]]),
            atol=1e-6, rtol=0,
        )

    def test_180_rotation_around_center_flips_verts(self) -> None:
        """180° rotation around (0.5, 0.5): (0.1, 0.2) → (0.9, 0.8)。"""
        affine = make_rotation_translation_affine(angle_rad=math.pi, center=(0.5, 0.5))
        verts = torch.tensor([[0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]])
        out = warp_obb_coord_under_affine(verts, affine)
        expected = torch.tensor([[0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8]])
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)
