"""TemporalEncoder + DeltaViT の unit test (Issue #115、Phase 1 M5)。

- DeltaViT: init 構造 / forward shape / pos_embed / gradient flow
- TemporalEncoder: forward shape / cam_id 効果 / delta 計算確認 / gradient flow
- Param count が design doc "small ViT ~20M" range (14-24M) に収まる sanity check
"""

from __future__ import annotations

import pytest
import torch

from model.ramen_ori.temporal import DeltaViT, TemporalEncoder


class TestDeltaViT:
    def test_init_pos_embed_shape(self) -> None:
        vit = DeltaViT(img_size=224, patch_size=16, embed_dim=384)
        assert vit.pos_embed.shape == (1, 196, 384)  # 14*14 patch

    def test_forward_shape(self) -> None:
        vit = DeltaViT(img_size=224, patch_size=16, embed_dim=384, depth=2)
        x = torch.randn(3, 3, 224, 224)
        out = vit(x)
        assert out.shape == (3, 196, 384)

    def test_forward_with_non_224_resolution(self) -> None:
        # pos_embed は 224 固定なので、他解像度では pos_embed size 不整合 (Phase 1 は 224 固定運用)
        vit = DeltaViT(img_size=320, patch_size=16, embed_dim=384, depth=2)
        x = torch.randn(2, 3, 320, 320)
        out = vit(x)
        assert out.shape == (2, 20 * 20, 384)


class TestTemporalEncoderInit:
    def test_layer_types(self) -> None:
        module = TemporalEncoder(num_cams=6, d_model=512)
        assert isinstance(module.delta_encoder, DeltaViT)
        assert isinstance(module.pool, torch.nn.AdaptiveAvgPool2d)
        assert isinstance(module.projection, torch.nn.Linear)
        assert isinstance(module.cam_embed, torch.nn.Embedding)
        assert isinstance(module.cam_lift, torch.nn.Linear)

    def test_cam_embed_vocab_size(self) -> None:
        module = TemporalEncoder(num_cams=6, cam_embed_dim=16, d_model=512)
        assert module.cam_embed.weight.shape == (6, 16)
        assert module.cam_lift.in_features == 16
        assert module.cam_lift.out_features == 512

    def test_param_count_in_design_range(self) -> None:
        """~20M target、embed=384 depth=8 で 14-16M 想定 (design doc "small ViT ~20M" range)。"""
        module = TemporalEncoder(
            num_cams=6, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4.0, d_model=512
        )
        n = sum(p.numel() for p in module.parameters())
        # 14M ≤ n ≤ 24M (design doc の "~20M" 記述に対する ±30% tolerance)
        assert 14e6 <= n <= 24e6, f"param count {n / 1e6:.1f}M out of 14-24M range"


class TestTemporalEncoderForward:
    @pytest.mark.parametrize(
        "B,N_cams,pool_size", [(1, 4, 8), (2, 4, 8), (2, 2, 4), (1, 6, 8)]
    )
    def test_output_shape(self, B: int, N_cams: int, pool_size: int) -> None:
        module = TemporalEncoder(
            num_cams=6, embed_dim=384, depth=2, pool_size=pool_size, d_model=512
        )
        images = torch.randn(B, N_cams, 3, 224, 224)
        images_prev = torch.randn(B, N_cams, 3, 224, 224)
        cam_id = torch.randint(0, 6, (B, N_cams), dtype=torch.long)
        tokens = module(images, images_prev, cam_id)
        assert tokens.shape == (B, N_cams * pool_size**2, 512)

    def test_delta_zero_when_input_equals_prev(self) -> None:
        """images == images_prev のとき delta=0、出力は pos_embed + cam_embed 由来のみ。"""
        module = TemporalEncoder(
            num_cams=6, embed_dim=384, depth=2, pool_size=4, d_model=64
        )
        module.eval()
        images = torch.randn(1, 2, 3, 224, 224)
        cam_id = torch.tensor([[0, 1]], dtype=torch.long)
        # 同じ images を prev に渡す → delta = 0
        out_zero_delta = module(images, images, cam_id)
        out_random_prev = module(images, torch.randn_like(images), cam_id)
        # 出力は異なる (delta の情報が別)
        assert not torch.allclose(out_zero_delta, out_random_prev)

    def test_cam_id_affects_output(self) -> None:
        """cam_id を変えると cam_embed が変わり output token に影響する。"""
        module = TemporalEncoder(
            num_cams=6, embed_dim=384, depth=2, pool_size=4, d_model=64
        )
        module.eval()
        images = torch.randn(1, 2, 3, 224, 224)
        images_prev = torch.randn(1, 2, 3, 224, 224)
        out_a = module(images, images_prev, torch.tensor([[0, 1]], dtype=torch.long))
        out_b = module(images, images_prev, torch.tensor([[2, 3]], dtype=torch.long))
        assert not torch.allclose(out_a, out_b)


class TestGradient:
    def test_backward_populates_grad_for_all_params(self) -> None:
        module = TemporalEncoder(
            num_cams=6, embed_dim=64, depth=2, num_heads=2, pool_size=4, d_model=32
        )
        images = torch.randn(1, 2, 3, 224, 224)
        images_prev = torch.randn(1, 2, 3, 224, 224)
        cam_id = torch.tensor([[0, 1]], dtype=torch.long)
        tokens = module(images, images_prev, cam_id)
        loss = tokens.sum()
        loss.backward()
        for name, param in module.named_parameters():
            assert param.grad is not None, f"{name} has no grad"
            # cam_embed の未 access index は grad が 0 だが、access index は非 0
            if "cam_embed" in name:
                # cam_id ∈ {0, 1} のみ access、他 index の grad は 0 でも良い
                assert param.grad[[0, 1]].abs().sum().item() > 0
            else:
                assert param.grad.abs().sum().item() > 0, f"{name} grad all zero"
