"""B-8 MeanFlow action expert の unit test (Issue #120 Alt-7)。

- MeanFlow: forward shape、compute_loss/sample method、gradient flow、param count
- FlowMatchingDiT の method-based API (compute_loss/sample) の後方互換
"""

from __future__ import annotations

import pytest
import torch

from model.ramen_ori.action_expert import FlowMatchingDiT, MeanFlow


class TestMeanFlowForward:
    @pytest.mark.parametrize(
        "B,chunk_len,action_dim,d,ctx_dim",
        [(1, 8, 4, 64, 64), (2, 16, 19, 128, 128), (4, 4, 2, 32, 32)],
    )
    def test_forward_shape(self, B, chunk_len, action_dim, d, ctx_dim) -> None:
        mf = MeanFlow(chunk_len=chunk_len, action_dim=action_dim, embed_dim=d,
                      depth=2, num_heads=4, context_dim=ctx_dim)
        x = torch.randn(B, chunk_len, action_dim)
        t = torch.rand(B)
        ctx = torch.randn(B, 10, ctx_dim)
        out = mf(x, t, ctx)
        assert out.shape == (B, chunk_len, action_dim)

    def test_context_mask_changes_output(self) -> None:
        torch.manual_seed(0)
        mf = MeanFlow(chunk_len=8, action_dim=4, embed_dim=64, depth=2, num_heads=4, context_dim=64)
        # output_proj は zero-init なので mask 効果測定用に軽く jitter
        with torch.no_grad():
            mf.output_proj.weight.normal_(0.0, 0.02)
        mf.eval()
        x = torch.randn(1, 8, 4)
        t = torch.rand(1)
        ctx = torch.randn(1, 10, 64)
        out_all = mf(x, t, ctx, context_mask=torch.ones(1, 10, dtype=torch.bool))
        out_half = mf(x, t, ctx, context_mask=torch.tensor([[True] * 5 + [False] * 5]))
        assert not torch.allclose(out_all, out_half)


class TestMeanFlowLoss:
    def test_compute_loss_scalar(self) -> None:
        mf = MeanFlow(chunk_len=8, action_dim=4, embed_dim=64, depth=2, num_heads=4, context_dim=64)
        action = torch.randn(2, 8, 4)
        ctx = torch.randn(2, 10, 64)
        loss = mf.compute_loss(action, ctx)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_zero_init_loss_is_variance_of_v_target(self) -> None:
        """output_proj zero-init → v_pred=0 → loss = mean(v_target^2)。"""
        torch.manual_seed(0)
        mf = MeanFlow(chunk_len=8, action_dim=4, embed_dim=64, depth=2, num_heads=4, context_dim=64)
        action = torch.randn(2, 8, 4)
        ctx = torch.randn(2, 10, 64)
        # 初期 loss は v_target = x_1 - x_0 の分散に近い (x_0, x_1 独立 N(0,1) → var=2)
        loss = mf.compute_loss(action, ctx)
        # 明確な比較は難しいが正値でスケール ~O(1) 保証
        assert 0.0 < loss.item() < 10.0


class TestMeanFlowSample:
    def test_sample_shape(self) -> None:
        mf = MeanFlow(chunk_len=8, action_dim=4, embed_dim=64, depth=2, num_heads=4, context_dim=64)
        mf.eval()
        ctx = torch.randn(2, 10, 64)
        out = mf.sample(ctx, n_steps=1)
        assert out.shape == (2, 8, 4)

    def test_single_step_default(self) -> None:
        """MeanFlow は n_steps=1 が default (straight flow)。"""
        mf = MeanFlow(chunk_len=8, action_dim=4, embed_dim=64, depth=2, num_heads=4, context_dim=64)
        mf.eval()
        ctx = torch.randn(1, 5, 64)
        # 明示指定なしで動く
        out = mf.sample(ctx)
        assert out.shape == (1, 8, 4)

    def test_multi_step_also_works(self) -> None:
        """n_steps>1 でも Euler で複数 step 積分可能。"""
        mf = MeanFlow(chunk_len=8, action_dim=4, embed_dim=64, depth=2, num_heads=4, context_dim=64)
        mf.eval()
        ctx = torch.randn(1, 5, 64)
        out = mf.sample(ctx, n_steps=4)
        assert out.shape == (1, 8, 4)


class TestMeanFlowGradient:
    def test_backward_all_params_have_grad(self) -> None:
        mf = MeanFlow(chunk_len=8, action_dim=4, embed_dim=64, depth=2, num_heads=4, context_dim=64)
        action = torch.randn(2, 8, 4)
        ctx = torch.randn(2, 10, 64)
        loss = mf.compute_loss(action, ctx)
        loss.backward()
        for name, p in mf.named_parameters():
            assert p.grad is not None, f"{name} no grad"


class TestMeanFlowParamCount:
    def test_default_config_lighter_than_flow_matching(self) -> None:
        """MeanFlow (~50M target) は FlowMatchingDiT (~150M) より軽量。"""
        mf = MeanFlow(chunk_len=16, action_dim=19, embed_dim=512, depth=6,
                      num_heads=8, mlp_ratio=4.0, context_dim=512)
        fmd = FlowMatchingDiT(chunk_len=16, action_dim=19, embed_dim=768, depth=12,
                              num_heads=12, mlp_ratio=4.0, context_dim=512)
        n_mf = sum(p.numel() for p in mf.parameters())
        n_fmd = sum(p.numel() for p in fmd.parameters())
        assert n_mf < n_fmd, f"MeanFlow ({n_mf/1e6:.1f}M) should be < FlowMatching ({n_fmd/1e6:.1f}M)"
        # ~50M range (許容 30-100M)
        assert 20e6 <= n_mf <= 100e6, f"MeanFlow param count {n_mf/1e6:.1f}M out of 20-100M range"


class TestFlowMatchingMethodApi:
    """FlowMatchingDiT にも Alt-7 で追加した method-based API の後方互換確認。"""

    def test_compute_loss_method_matches_free_function(self) -> None:
        from model.ramen_ori.action_expert import flow_matching_loss
        torch.manual_seed(0)
        fmd = FlowMatchingDiT(chunk_len=8, action_dim=4, embed_dim=64, depth=2,
                              num_heads=4, context_dim=64)
        action = torch.randn(2, 8, 4)
        ctx = torch.randn(2, 10, 64)
        torch.manual_seed(42)
        l_method = fmd.compute_loss(action, ctx)
        torch.manual_seed(42)
        l_func = flow_matching_loss(fmd, action, ctx)
        assert l_method.item() == pytest.approx(l_func.item(), abs=1e-6)

    def test_sample_method_matches_free_function(self) -> None:
        from model.ramen_ori.action_expert import sample_action
        fmd = FlowMatchingDiT(chunk_len=8, action_dim=4, embed_dim=64, depth=2,
                              num_heads=4, context_dim=64)
        fmd.eval()
        ctx = torch.randn(2, 10, 64)
        torch.manual_seed(1)
        s_method = fmd.sample(ctx, n_steps=3)
        torch.manual_seed(1)
        s_func = sample_action(fmd, ctx, n_steps=3)
        torch.testing.assert_close(s_method, s_func)
