"""FlowMatchingDiT + helper の unit test (Issue #115、Phase 1 M6)。

- TimeEmbedding: (B,) → (B, dim)、t が異なれば output も異なる
- DiTBlock: forward shape 保存、AdaLN-Zero init で最初は identity 相当
- FlowMatchingDiT: forward shape、init で v_pred ≈ 0 (zero-init pattern)、
  context_mask 効果、gradient flow、param count sanity
- flow_matching_loss: scalar 返す、backward で grad populated
- sample_action: shape 正しい、seed で deterministic
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.ramen_ori.action_expert import (
    DiTBlock,
    FlowMatchingDiT,
    TimeEmbedding,
    flow_matching_loss,
    sample_action,
)


class TestTimeEmbedding:
    def test_output_shape(self) -> None:
        te = TimeEmbedding(dim=512, freq_dim=256)
        t = torch.rand(4)
        out = te(t)
        assert out.shape == (4, 512)

    def test_different_t_gives_different_output(self) -> None:
        te = TimeEmbedding(dim=64, freq_dim=32)
        te.eval()
        t1 = torch.tensor([0.1])
        t2 = torch.tensor([0.9])
        out1 = te(t1)
        out2 = te(t2)
        assert not torch.allclose(out1, out2)


class TestDiTBlock:
    def test_forward_shape_preserved(self) -> None:
        block = DiTBlock(dim=64, num_heads=4, mlp_ratio=2.0)
        x = torch.randn(2, 8, 64)  # (B, chunk_len, dim)
        t_e = torch.randn(2, 64)
        context = torch.randn(2, 10, 64)  # (B, N_context, dim)
        out = block(x, t_e, context)
        assert out.shape == x.shape

    def test_adaln_zero_init_is_identity(self) -> None:
        """AdaLN-Zero: 最終 Linear が zero-init なので、初期状態は out ≈ x (identity)。"""
        block = DiTBlock(dim=64, num_heads=4, mlp_ratio=2.0)
        block.eval()
        x = torch.randn(2, 8, 64)
        t_e = torch.randn(2, 64)
        context = torch.randn(2, 10, 64)
        out = block(x, t_e, context)
        assert torch.allclose(out, x, atol=1e-6)


class TestDiTQKNorm:
    """Issue #141: qk_norm=True は self-attn と cross-attn に q/k の RMSNorm を足すだけ。"""

    NORMS = ("attn_q_norm", "attn_k_norm", "cross_q_norm", "cross_k_norm")

    def test_adds_only_norm_params(self) -> None:
        plain = set(DiTBlock(dim=32, num_heads=4).state_dict())
        qk = set(DiTBlock(dim=32, num_heads=4, qk_norm=True).state_dict())
        assert plain < qk and qk - plain == {f"{n}.weight" for n in self.NORMS}

    def test_identity_norm_matches_default_block(self) -> None:
        torch.manual_seed(0)
        plain = DiTBlock(dim=32, num_heads=4).eval()
        # AdaLN-Zero のままだと gate が 0 で attention が出力に効かないので、modulation を 0 以外に
        nn.init.normal_(plain.adaln[-1].weight, std=0.1)
        qk = DiTBlock(dim=32, num_heads=4, qk_norm=True).eval()
        qk.load_state_dict(plain.state_dict(), strict=False)
        for n in self.NORMS:
            setattr(qk, n, nn.Identity())
        x, t_embed, context = torch.randn(2, 6, 32), torch.randn(2, 32), torch.randn(2, 9, 32)
        kpm = torch.zeros(2, 9, dtype=torch.bool)
        kpm[1, 7:] = True
        torch.testing.assert_close(
            qk(x, t_embed, context, kpm), plain(x, t_embed, context, kpm), atol=1e-5, rtol=1e-5
        )

    def test_flow_matching_dit_uses_qk_norm_in_all_blocks(self) -> None:
        model = FlowMatchingDiT(
            chunk_len=4, action_dim=3, embed_dim=32, depth=2, num_heads=4, context_dim=16, qk_norm=True
        )
        assert all(block.qk_norm for block in model.blocks)
        loss = flow_matching_loss(model, torch.randn(2, 4, 3), torch.randn(2, 5, 16))
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())


class TestFlowMatchingDiTInit:
    def test_forward_shape(self) -> None:
        model = FlowMatchingDiT(
            chunk_len=16, action_dim=19, embed_dim=128, depth=2,
            num_heads=4, mlp_ratio=2.0, context_dim=64,
        )
        model.eval()
        x = torch.randn(2, 16, 19)
        t = torch.rand(2)
        context = torch.randn(2, 20, 64)
        v_pred = model(x, t, context)
        assert v_pred.shape == (2, 16, 19)

    def test_zero_init_output_at_init(self) -> None:
        """DiT-Zero pattern: init 時は v_pred = 0 (output_proj + adaln_final gate zero-init)。"""
        model = FlowMatchingDiT(
            chunk_len=16, action_dim=19, embed_dim=128, depth=2,
            num_heads=4, mlp_ratio=2.0, context_dim=64,
        )
        model.eval()
        x = torch.randn(2, 16, 19)
        t = torch.rand(2)
        context = torch.randn(2, 20, 64)
        v_pred = model(x, t, context)
        assert torch.allclose(v_pred, torch.zeros_like(v_pred), atol=1e-6)

    def test_param_count_in_dit_b_range(self) -> None:
        """DiT-B/2 相当の canonical config で param count が 100M-200M range。

        design doc "~150M" は AdaLN modulation の overhead を含めない見積、
        実 param は AdaLN の 9*dim*dim/block で +30-50M 増える。
        """
        model = FlowMatchingDiT(
            chunk_len=16, action_dim=19, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4.0, context_dim=512,
        )
        n = sum(p.numel() for p in model.parameters())
        assert 100e6 <= n <= 200e6, f"param count {n / 1e6:.1f}M out of 100-200M range"


class TestFlowMatchingDiTForward:
    def test_context_mask_affects_output(self) -> None:
        """異なる context_mask で v_pred が変わる (masked context は attend されない)。

        AdaLN-Zero init のまま testing すると block の cross-attn contribution が gate=0
        で殺されて mask 効果が output に届かない。ここでは AdaLN の最終 Linear を non-zero
        に上書きして "modulation が active" な状態で mask 挙動を検証する
        (training 後の実運用状態を smoke test で近似)。
        """
        torch.manual_seed(0)
        model = FlowMatchingDiT(
            chunk_len=8, action_dim=4, embed_dim=64, depth=2,
            num_heads=4, mlp_ratio=2.0, context_dim=32,
        )
        # AdaLN-Zero init を上書き、block modulation を alive に
        for blk in model.blocks:
            nn.init.normal_(blk.adaln[-1].weight, std=0.5)
            nn.init.normal_(blk.adaln[-1].bias, std=0.5)
        # output_proj も上書き (最終 v_pred が非 0 になるように)
        nn.init.normal_(model.output_proj.weight, std=0.1)

        model.eval()
        x = torch.randn(1, 8, 4)
        t = torch.tensor([0.5])
        context = torch.randn(1, 10, 32)
        mask_all = torch.ones(1, 10, dtype=torch.bool)
        mask_half = torch.tensor([[True] * 5 + [False] * 5])
        out_all = model(x, t, context, mask_all)
        out_half = model(x, t, context, mask_half)
        assert not torch.allclose(out_all, out_half)


class TestFlowMatchingLoss:
    def test_loss_is_scalar_and_grad_populates(self) -> None:
        model = FlowMatchingDiT(
            chunk_len=8, action_dim=4, embed_dim=64, depth=2,
            num_heads=4, mlp_ratio=2.0, context_dim=32,
        )
        target = torch.randn(2, 8, 4)
        context = torch.randn(2, 10, 32)
        loss = flow_matching_loss(model, target, context)
        assert loss.dim() == 0
        assert loss.item() >= 0  # MSE ≥ 0
        loss.backward()
        # 全 param に grad populate (zero-init な output_proj も含めて)
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no grad"


class TestSampleAction:
    def test_sample_shape(self) -> None:
        model = FlowMatchingDiT(
            chunk_len=16, action_dim=19, embed_dim=64, depth=2,
            num_heads=4, mlp_ratio=2.0, context_dim=32,
        )
        model.eval()
        context = torch.randn(3, 10, 32)
        action = sample_action(model, context, n_steps=4)
        assert action.shape == (3, 16, 19)

    def test_sample_deterministic_with_seed(self) -> None:
        model = FlowMatchingDiT(
            chunk_len=4, action_dim=2, embed_dim=32, depth=1,
            num_heads=2, mlp_ratio=2.0, context_dim=16,
        )
        model.eval()
        context = torch.randn(1, 5, 16)
        torch.manual_seed(0)
        a1 = sample_action(model, context, n_steps=4)
        torch.manual_seed(0)
        a2 = sample_action(model, context, n_steps=4)
        assert torch.allclose(a1, a2)


class TestRtcGuidance:
    """RTC (Issue #137) の prefix / velocity_strength。

    既定 (両方 None) では従来経路と完全に一致する必要がある。学習は
    `sample_action` を通らないが、推論と同じ関数なので既定経路を壊さないことを
    機械的に固定しておく。
    """

    CHUNK_LEN = 16
    ACTION_DIM = 16

    def _model(self) -> FlowMatchingDiT:
        model = FlowMatchingDiT(
            chunk_len=self.CHUNK_LEN, action_dim=self.ACTION_DIM, embed_dim=64,
            depth=2, num_heads=4, mlp_ratio=2.0, context_dim=32,
        )
        # zero-init のままだと v_pred = 0 で x が動かず、ramp の効果を検証できない
        torch.nn.init.normal_(model.output_proj.weight, std=0.05)
        torch.nn.init.normal_(model.output_proj.bias, std=0.05)
        model.eval()
        return model

    def _context(self, B: int = 1) -> torch.Tensor:
        torch.manual_seed(99)
        return torch.randn(B, 12, 32)

    def _sample(self, model, context, *, seed: int = 7, **kwargs) -> torch.Tensor:
        torch.manual_seed(seed)
        with torch.no_grad():
            return sample_action(model, context, n_steps=kwargs.pop("n_steps", 4), **kwargs)

    def test_supports_rtc_guidance_flag(self) -> None:
        assert FlowMatchingDiT.SUPPORTS_RTC_GUIDANCE is True

    def test_strength_of_ones_is_bit_identical_to_default(self) -> None:
        # 追加した分岐が既定経路の数値を一切変えないこと (1.0 倍は厳密)
        model, context = self._model(), self._context()
        baseline = self._sample(model, context)
        ones = self._sample(
            model, context, velocity_strength=torch.ones(self.CHUNK_LEN)
        )
        assert torch.equal(baseline, ones)

    def test_frozen_rows_stay_exactly_at_the_prefix(self) -> None:
        model, context = self._model(), self._context()
        prefix = torch.full((6, self.ACTION_DIM), 0.25)
        strength = torch.ones(self.CHUNK_LEN)
        strength[:3] = 0.0
        out = self._sample(
            model, context, prefix=prefix, velocity_strength=strength
        )
        # 凍結区間は速度場が乗らないので初期値 = prefix のまま
        assert torch.equal(out[0, :3], prefix[:3])
        # 凍結外は動く
        assert not torch.allclose(out[0, 3:6], prefix[3:6])

    def test_displacement_is_proportional_to_strength(self) -> None:
        # n_steps=1 なら out = x0 + dt * v(x0) * strength なので厳密に比例する。
        # ramp の形がそのまま軌道への影響度になっていることの確認。
        model, context = self._model(), self._context()
        prefix = torch.zeros(self.CHUNK_LEN, self.ACTION_DIM)
        base = torch.linspace(0.1, 1.0, self.CHUNK_LEN)
        out_a = self._sample(
            model, context, n_steps=1, prefix=prefix, velocity_strength=base
        )
        out_b = self._sample(
            model, context, n_steps=1, prefix=prefix, velocity_strength=base * 2.0
        )
        torch.testing.assert_close(out_b, out_a * 2.0, rtol=1e-5, atol=1e-6)

    def test_prefix_accepts_batched_and_unbatched(self) -> None:
        model, context = self._model(), self._context()
        prefix = torch.full((4, self.ACTION_DIM), 0.5)
        strength = torch.zeros(self.CHUNK_LEN)
        flat = self._sample(model, context, prefix=prefix, velocity_strength=strength)
        batched = self._sample(
            model, context, prefix=prefix.unsqueeze(0), velocity_strength=strength
        )
        assert torch.equal(flat, batched)

    def test_full_length_prefix_allowed(self) -> None:
        model, context = self._model(), self._context()
        prefix = torch.zeros(self.CHUNK_LEN, self.ACTION_DIM)
        out = self._sample(
            model, context, prefix=prefix,
            velocity_strength=torch.zeros(self.CHUNK_LEN),
        )
        assert torch.equal(out[0], prefix)

    @pytest.mark.parametrize(
        "prefix_shape", [(17, 16), (4, 19), (2, 4, 16, 1)]
    )
    def test_invalid_prefix_rejected(self, prefix_shape: tuple) -> None:
        model, context = self._model(), self._context()
        with pytest.raises(ValueError):
            self._sample(model, context, prefix=torch.zeros(prefix_shape))

    @pytest.mark.parametrize("strength_shape", [(15,), (17,), (16, 1)])
    def test_invalid_strength_rejected(self, strength_shape: tuple) -> None:
        model, context = self._model(), self._context()
        with pytest.raises(ValueError):
            self._sample(model, context, velocity_strength=torch.ones(strength_shape))


class TestMaskedMse:
    """Issue #141 RO-2: 区間末尾の埋め草の行を loss から外す。"""

    def test_none_mask_equals_plain_mse(self) -> None:
        from model.ramen_ori.action_expert import masked_mse

        a, b = torch.randn(2, 5, 3), torch.randn(2, 5, 3)
        torch.testing.assert_close(masked_mse(a, b, None), torch.nn.functional.mse_loss(a, b))

    def test_pad_rows_do_not_change_loss(self) -> None:
        from model.ramen_ori.action_expert import masked_mse

        pred, target = torch.randn(2, 5, 3), torch.randn(2, 5, 3)
        mask = torch.tensor([[True, True, True, False, False], [True] * 5])
        target2 = target.clone()
        target2[~mask] += 10.0
        torch.testing.assert_close(masked_mse(pred, target, mask), masked_mse(pred, target2, mask))
        # 有効な行だけの平均
        expected = ((pred - target) ** 2)[mask].mean()
        torch.testing.assert_close(masked_mse(pred, target, mask), expected)

    def test_row_mask_reaches_flow_matching_and_meanflow(self) -> None:
        """全行 True の mask は mask なしと同じ値 (同じ乱数)。FlowMatchingDiT と MeanFlow の両方。"""
        from model.ramen_ori.action_expert import FlowMatchingDiT, MeanFlow

        ctx = torch.randn(2, 6, 32)
        target = torch.randn(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        for expert in (
            FlowMatchingDiT(chunk_len=4, action_dim=3, embed_dim=32, depth=1, num_heads=4, context_dim=32),
            MeanFlow(chunk_len=4, action_dim=3, embed_dim=32, depth=1, num_heads=4, context_dim=32),
        ):
            torch.manual_seed(0)
            a = expert.compute_loss(target, ctx)
            torch.manual_seed(0)
            b = expert.compute_loss(target, ctx, row_mask=mask)
            torch.testing.assert_close(a, b)
