"""QK-norm の attention (attention.py、Issue #141) の unit test。

- norm を恒等にすれば nn.MultiheadAttention と同じ出力 (q/k/v の分け方・mask の向き・out_proj)
- mask した key は出力に効かない
- RMSNorm なら q/k の重みを何倍にしても出力が変わらない (logit が q/k の大きさで増えない)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.ramen_ori.attention import qk_norm_attention

D, H = 32, 4


def _mha() -> nn.MultiheadAttention:
    torch.manual_seed(0)
    return nn.MultiheadAttention(D, H, batch_first=True).eval()


class TestQkNormAttention:
    @pytest.mark.parametrize("with_mask", [False, True])
    def test_identity_norm_matches_multihead_attention(self, with_mask: bool) -> None:
        mha = _mha()
        query, key = torch.randn(2, 5, D), torch.randn(2, 7, D)
        kpm = torch.tensor([[False] * 5 + [True] * 2, [False] * 7]) if with_mask else None
        ref, _ = mha(query, key, key, key_padding_mask=kpm, need_weights=False)
        out = qk_norm_attention(mha, nn.Identity(), nn.Identity(), query, key, kpm)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_masked_keys_do_not_change_output(self) -> None:
        mha = _mha()
        q_norm, k_norm = nn.RMSNorm(D // H, eps=1e-6), nn.RMSNorm(D // H, eps=1e-6)
        query, key = torch.randn(1, 3, D), torch.randn(1, 5, D)
        kpm = torch.tensor([[False, False, False, True, True]])
        out = qk_norm_attention(mha, q_norm, k_norm, query, key, kpm)
        key2 = key.clone()
        key2[:, 3:] = 1000.0 * torch.randn(1, 2, D)
        torch.testing.assert_close(qk_norm_attention(mha, q_norm, k_norm, query, key2, kpm), out)

    def test_output_does_not_depend_on_qk_weight_scale(self) -> None:
        mha = _mha()
        with torch.no_grad():
            mha.in_proj_bias[: 2 * D].zero_()  # q と k の bias を 0 に (RMSNorm が消すのは大きさだけ)
        q_norm, k_norm = nn.RMSNorm(D // H, eps=1e-6), nn.RMSNorm(D // H, eps=1e-6)
        x = torch.randn(2, 6, D)
        before = qk_norm_attention(mha, q_norm, k_norm, x, x)
        plain_before, _ = mha(x, x, x, need_weights=False)
        with torch.no_grad():
            mha.in_proj_weight[: 2 * D].mul_(100.0)  # q と k の重みだけ 100 倍
        torch.testing.assert_close(qk_norm_attention(mha, q_norm, k_norm, x, x), before, atol=1e-5, rtol=1e-4)
        # norm なしの attention は同じ変更で出力が変わる (比較の対照)
        plain_after, _ = mha(x, x, x, need_weights=False)
        assert not torch.allclose(plain_after, plain_before, atol=1e-3)
