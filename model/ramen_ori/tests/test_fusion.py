"""Fusion variants の unit test (Issue #115 G-1、Issue #120 Alt-1 で G-2/G-3 追加)。

- G-1 FusionTransformer: forward shape 保存、mask 効果、gradient flow、param count
- G-3 FusionCrossAttn: 出力 = num_queries、mask 効果、query learnable
- G-2 FusionInterleaved: 出力 = num_queries、cross+self 交互、even layers 要求
- 全 variant: forward の返り値 = (context, context_mask)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.ramen_ori.fusion import (
    FusionCrossAttn,
    FusionInterleaved,
    FusionTransformer,
    _CrossAttnBlock,
    _SelfAttnBlock,
)


class TestFusionTransformer:
    """G-1 Standard self-attn。"""

    @pytest.mark.parametrize("B,N,d", [(1, 10, 64), (2, 530, 512), (4, 100, 256)])
    def test_forward_shape_preserved(self, B: int, N: int, d: int) -> None:
        module = FusionTransformer(d_model=d, num_layers=2, num_heads=4, mlp_dim=d * 2)
        tokens = torch.randn(B, N, d)
        out, mask = module(tokens)
        assert out.shape == tokens.shape
        assert mask is None

    def test_attention_mask_changes_output(self) -> None:
        """attention_mask 有無で output が変わる (半分 mask したら残り 5 token の attention 相手が減る)。"""
        torch.manual_seed(0)
        module = FusionTransformer(d_model=64, num_layers=2, num_heads=4, mlp_dim=128)
        module.eval()
        tokens = torch.randn(1, 10, 64)
        mask_all = torch.ones(1, 10, dtype=torch.bool)
        mask_half = torch.tensor([[True] * 5 + [False] * 5])
        out_all, _ = module(tokens, mask_all)
        out_half, _ = module(tokens, mask_half)
        assert not torch.allclose(out_all, out_half)

    def test_none_mask_equivalent_to_all_true_mask(self) -> None:
        """attention_mask=None は "全 token 有効" と等価であるべき。"""
        torch.manual_seed(0)
        module = FusionTransformer(d_model=64, num_layers=2, num_heads=4, mlp_dim=128)
        module.eval()
        tokens = torch.randn(2, 8, 64)
        out_none, _ = module(tokens, None)
        out_all, _ = module(tokens, torch.ones(2, 8, dtype=torch.bool))
        assert torch.allclose(out_none, out_all, atol=1e-6)

    def test_mask_passthrough(self) -> None:
        module = FusionTransformer(d_model=32, num_layers=2, num_heads=4, mlp_dim=64)
        tokens = torch.randn(2, 5, 32)
        mask = torch.tensor([[True, True, False, True, True], [True, False, False, True, True]])
        _, out_mask = module(tokens, mask)
        assert torch.equal(out_mask, mask)


class TestFusionQKNorm:
    """Issue #141: qk_norm=True は nn.TransformerEncoderLayer (pre-LN) と同じ計算 + q/k の RMSNorm。"""

    @staticmethod
    def _pair() -> tuple[FusionTransformer, FusionTransformer, list[str], list[str]]:
        torch.manual_seed(0)
        plain = FusionTransformer(d_model=32, num_layers=2, num_heads=4, mlp_dim=64).eval()
        qk = FusionTransformer(d_model=32, num_layers=2, num_heads=4, mlp_dim=64, qk_norm=True).eval()
        missing, unexpected = qk.load_state_dict(plain.state_dict(), strict=False)
        return plain, qk, missing, unexpected

    def test_default_keeps_torch_encoder(self) -> None:
        module = FusionTransformer(d_model=32, num_layers=1, num_heads=4, mlp_dim=64)
        assert isinstance(module.encoder, nn.TransformerEncoder)

    def test_state_dict_adds_only_qk_norm(self) -> None:
        _, _, missing, unexpected = self._pair()
        assert unexpected == []
        assert sorted(missing) == sorted(
            f"encoder.layers.{i}.{n}_norm.weight" for i in range(2) for n in ("q", "k")
        )

    def test_identity_norm_matches_torch_encoder(self) -> None:
        plain, qk, _, _ = self._pair()
        for layer in qk.encoder.layers:
            layer.q_norm, layer.k_norm = nn.Identity(), nn.Identity()
        tokens = torch.randn(2, 7, 32)
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[0, 5:] = False
        ref, _ = plain(tokens, mask)
        out, _ = qk(tokens, mask)
        torch.testing.assert_close(out[mask], ref[mask], atol=1e-5, rtol=1e-5)

    def test_backward_reaches_qk_norm(self) -> None:
        module = FusionTransformer(d_model=32, num_layers=2, num_heads=4, mlp_dim=64, qk_norm=True)
        out, _ = module(torch.randn(2, 5, 32))
        out.sum().backward()
        for layer in module.encoder.layers:
            assert layer.q_norm.weight.grad is not None and layer.k_norm.weight.grad is not None


class TestFusionCrossAttn:
    """G-3 Cross-attn only。"""

    @pytest.mark.parametrize("B,N,d,Q", [(1, 10, 64, 8), (2, 530, 256, 32), (4, 50, 128, 16)])
    def test_output_shape_is_num_queries(self, B: int, N: int, d: int, Q: int) -> None:
        module = FusionCrossAttn(
            d_model=d, num_layers=2, num_heads=4, mlp_dim=d * 2, num_queries=Q
        )
        tokens = torch.randn(B, N, d)
        out, mask = module(tokens)
        assert out.shape == (B, Q, d)
        assert mask.shape == (B, Q)
        assert mask.all(), "queries は全 valid のはず"

    def test_attention_mask_changes_output(self) -> None:
        torch.manual_seed(0)
        module = FusionCrossAttn(d_model=64, num_layers=2, num_heads=4, mlp_dim=128, num_queries=8)
        module.eval()
        tokens = torch.randn(1, 10, 64)
        mask_all = torch.ones(1, 10, dtype=torch.bool)
        mask_half = torch.tensor([[True] * 5 + [False] * 5])
        out_all, _ = module(tokens, mask_all)
        out_half, _ = module(tokens, mask_half)
        assert not torch.allclose(out_all, out_half)

    def test_queries_are_learnable(self) -> None:
        module = FusionCrossAttn(d_model=32, num_layers=2, num_heads=4, mlp_dim=64, num_queries=8)
        assert module.queries.requires_grad
        assert module.queries.shape == (8, 32)

    def test_gradient_flows_to_queries(self) -> None:
        module = FusionCrossAttn(d_model=32, num_layers=2, num_heads=4, mlp_dim=64, num_queries=8)
        tokens = torch.randn(2, 10, 32)
        out, _ = module(tokens)
        out.sum().backward()
        assert module.queries.grad is not None
        assert module.queries.grad.abs().sum().item() > 0

    def test_no_self_attn_layers(self) -> None:
        """G-3 は cross-attn only、self-attn block 無し。"""
        module = FusionCrossAttn(d_model=32, num_layers=4, num_heads=4, mlp_dim=64, num_queries=8)
        assert len(module.layers) == 4
        assert all(isinstance(layer, _CrossAttnBlock) for layer in module.layers)


class TestFusionInterleaved:
    """G-2 Interleaved self-cross-attn。"""

    @pytest.mark.parametrize("B,N,d,Q", [(1, 10, 64, 8), (2, 530, 256, 32)])
    def test_output_shape_is_num_queries(self, B: int, N: int, d: int, Q: int) -> None:
        module = FusionInterleaved(
            d_model=d, num_layers=4, num_heads=4, mlp_dim=d * 2, num_queries=Q
        )
        tokens = torch.randn(B, N, d)
        out, mask = module(tokens)
        assert out.shape == (B, Q, d)
        assert mask.shape == (B, Q)
        assert mask.all()

    def test_odd_num_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="even num_layers"):
            FusionInterleaved(d_model=32, num_layers=3, num_heads=4, mlp_dim=64, num_queries=8)

    def test_layers_alternate_cross_self(self) -> None:
        module = FusionInterleaved(d_model=32, num_layers=6, num_heads=4, mlp_dim=64, num_queries=8)
        # layer 0,2,4 = cross、1,3,5 = self
        for i, layer in enumerate(module.layers):
            if i % 2 == 0:
                assert isinstance(layer, _CrossAttnBlock), f"layer {i} should be cross-attn"
            else:
                assert isinstance(layer, _SelfAttnBlock), f"layer {i} should be self-attn"

    def test_attention_mask_changes_output(self) -> None:
        torch.manual_seed(0)
        module = FusionInterleaved(d_model=64, num_layers=4, num_heads=4, mlp_dim=128, num_queries=8)
        module.eval()
        tokens = torch.randn(1, 10, 64)
        out_all, _ = module(tokens, torch.ones(1, 10, dtype=torch.bool))
        out_half, _ = module(tokens, torch.tensor([[True] * 5 + [False] * 5]))
        assert not torch.allclose(out_all, out_half)

    def test_gradient_flows_to_queries(self) -> None:
        module = FusionInterleaved(d_model=32, num_layers=4, num_heads=4, mlp_dim=64, num_queries=8)
        tokens = torch.randn(2, 10, 32)
        out, _ = module(tokens)
        out.sum().backward()
        assert module.queries.grad is not None
        assert module.queries.grad.abs().sum().item() > 0


class TestFusionGradient:
    def test_g1_backward_all_params(self) -> None:
        module = FusionTransformer(d_model=32, num_layers=2, num_heads=4, mlp_dim=64)
        tokens = torch.randn(2, 8, 32, requires_grad=True)
        out, _ = module(tokens)
        out.sum().backward()
        for name, p in module.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.abs().sum().item() > 0, f"{name} grad all zero"


class TestFusionParamCount:
    def test_g1_default_in_expected_range(self) -> None:
        """d=512, 6-layer, 8-head, mlp=2048 で ~19M。"""
        module = FusionTransformer()
        n = sum(p.numel() for p in module.parameters())
        assert 15e6 <= n <= 25e6, f"G-1 param count {n / 1e6:.1f}M out of 15-25M range"

    def test_g3_default_in_expected_range(self) -> None:
        """G-3 も d=512/6L/8H で 15-25M 範囲 (queries 32 分の追加 ~16K は無視できる小)。"""
        module = FusionCrossAttn(d_model=512, num_layers=6, num_heads=8, mlp_dim=2048, num_queries=32)
        n = sum(p.numel() for p in module.parameters())
        assert 15e6 <= n <= 25e6, f"G-3 param count {n / 1e6:.1f}M out of 15-25M range"

    def test_g2_default_in_expected_range(self) -> None:
        """G-2 も d=512/6L/8H で 15-25M 範囲 (cross/self 各 3 層、G-1 に近い)。"""
        module = FusionInterleaved(d_model=512, num_layers=6, num_heads=8, mlp_dim=2048, num_queries=32)
        n = sum(p.numel() for p in module.parameters())
        assert 15e6 <= n <= 25e6, f"G-2 param count {n / 1e6:.1f}M out of 15-25M range"

    def test_g3_has_query_params(self) -> None:
        """G-3 は learnable queries を持つ (G-1 は持たない)。"""
        g1 = FusionTransformer(d_model=512, num_layers=6, num_heads=8, mlp_dim=2048)
        g3 = FusionCrossAttn(d_model=512, num_layers=6, num_heads=8, mlp_dim=2048, num_queries=32)
        assert not any("queries" in n for n, _ in g1.named_parameters())
        assert any("queries" in n for n, _ in g3.named_parameters())
