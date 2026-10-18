"""DepthHead (I-6 Depth aux) unit test (Issue #120 Alt-3)。

- output shape: (B, N_cams, 1, H_out, W_out)
- reshape 正しさ (vision_features → per-cam spatial)
- output_size と pool_size の関係 validation
- gradient flow
- param count ~1-5M range
"""

from __future__ import annotations

import pytest
import torch

from model.ramen_ori.aux_depth import DepthHead


class TestInit:
    def test_default_output_size(self) -> None:
        head = DepthHead(vision_embed_dim=512, num_cams=4, pool_size=8, output_size=(32, 32))
        assert head.output_size == (32, 32)
        # 8 → 16 → 32 = 2 段 upsample、Conv2d 数 = 2 + 1 (最終 1ch) = 3
        conv_layers = [l for l in head.decoder if isinstance(l, torch.nn.Conv2d)]
        assert len(conv_layers) == 3

    def test_output_size_must_be_pool_size_x_2k(self) -> None:
        with pytest.raises(ValueError, match="pool_size .* 2\\^k"):
            DepthHead(vision_embed_dim=64, num_cams=2, pool_size=8, output_size=(24, 24))

    def test_output_must_be_multiple_of_pool_size(self) -> None:
        with pytest.raises(ValueError, match="multiple"):
            DepthHead(vision_embed_dim=64, num_cams=2, pool_size=8, output_size=(20, 20))

    def test_non_square_output_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            DepthHead(vision_embed_dim=64, num_cams=2, pool_size=8, output_size=(32, 16))


class TestForward:
    @pytest.mark.parametrize(
        "B,N_cams,P,d,H_out",
        [(1, 4, 8, 512, 32), (2, 4, 4, 128, 16), (4, 2, 8, 256, 32)],
    )
    def test_output_shape(self, B, N_cams, P, d, H_out) -> None:
        head = DepthHead(
            vision_embed_dim=d, num_cams=N_cams, pool_size=P, output_size=(H_out, H_out)
        )
        vision_features = torch.randn(B, N_cams * P * P, d)
        out = head(vision_features)
        assert out.shape == (B, N_cams, 1, H_out, H_out)

    def test_vision_features_N_mismatch_raises(self) -> None:
        head = DepthHead(vision_embed_dim=64, num_cams=4, pool_size=8, output_size=(32, 32))
        wrong = torch.randn(1, 4 * 8 * 8 + 1, 64)  # extra token
        with pytest.raises(ValueError, match="N mismatch"):
            head(wrong)

    def test_vision_features_d_mismatch_raises(self) -> None:
        head = DepthHead(vision_embed_dim=64, num_cams=4, pool_size=8, output_size=(32, 32))
        wrong = torch.randn(1, 4 * 8 * 8, 128)  # wrong d
        with pytest.raises(ValueError, match="d_model"):
            head(wrong)


class TestReshape:
    def test_per_cam_independence_across_passes(self) -> None:
        """per-cam 独立性の regression guard。

        VisionEncoder 出力順序 (`[batch][cam][row][col][dim]`、vision.py:82-87) と
        `DepthHead.forward` の reshape (`.view(B, N_cams, P, P, D)`) が同 mapping で
        あることを 2 pass で verify。**vision.py の permute order 変更や aux_depth の
        reshape 破損時にこの test で silent bug を捕まえる**。

        方針: 2 pass で cam 0/2 の input を同一に、cam 1 だけ変える →
        - cam 0/2 の output は 2 pass で同一 (per-cam 独立性、input 同じ)
        - cam 1 の output は 2 pass で異なる (input が違う、per-cam mixing 無)
        """
        head = DepthHead(
            vision_embed_dim=8, num_cams=3, pool_size=2, output_size=(4, 4), hidden_dim=16,
        )
        # DepthHead の最終 conv は default Kaiming init (非 zero) なので追加 jitter 不要。
        head.eval()

        B, N_cams, P, d = 1, 3, 2, 8

        # Pass A: cam 0 = 0.5, cam 1 = 1.0, cam 2 = -0.5
        vf_a = torch.zeros(B, N_cams * P * P, d)
        vf_a[0, 0 * P * P : 1 * P * P, :] = 0.5
        vf_a[0, 1 * P * P : 2 * P * P, :] = 1.0
        vf_a[0, 2 * P * P : 3 * P * P, :] = -0.5
        out_a = head(vf_a)
        assert out_a.shape == (1, 3, 1, 4, 4)

        # Pass B: cam 0 = 0.5 (same), cam 1 = 2.0 (DIFFERENT), cam 2 = -0.5 (same)
        vf_b = torch.zeros(B, N_cams * P * P, d)
        vf_b[0, 0 * P * P : 1 * P * P, :] = 0.5
        vf_b[0, 1 * P * P : 2 * P * P, :] = 2.0
        vf_b[0, 2 * P * P : 3 * P * P, :] = -0.5
        out_b = head(vf_b)

        # per-cam 独立性: 同 cam input → 同 cam output (2 pass 間で一致)
        torch.testing.assert_close(
            out_a[0, 0], out_b[0, 0],
            msg="cam 0 output differs across passes but input was identical — per-cam mixing suspected",
        )
        torch.testing.assert_close(
            out_a[0, 2], out_b[0, 2],
            msg="cam 2 output differs across passes but input was identical — per-cam mixing suspected",
        )
        # cam 1 は input が違うので output も違うはず (aux head が input に反応することの確認)
        assert not torch.allclose(out_a[0, 1], out_b[0, 1]), (
            "cam 1 output identical across passes but input differed — aux head not responsive to input"
        )


class TestGradient:
    def test_backward_all_params_have_grad(self) -> None:
        head = DepthHead(vision_embed_dim=64, num_cams=2, pool_size=4, output_size=(8, 8))
        vf = torch.randn(2, 2 * 4 * 4, 64, requires_grad=True)
        out = head(vf)
        out.sum().backward()
        for name, p in head.named_parameters():
            assert p.grad is not None, f"{name} has no grad"


class TestParamCount:
    def test_default_config_in_expected_range(self) -> None:
        """d=512, num_cams=4, pool_size=8, output=(32,32), hidden=256 で ~1-5M。"""
        head = DepthHead(
            vision_embed_dim=512, num_cams=4, pool_size=8, output_size=(32, 32), hidden_dim=256
        )
        n = sum(p.numel() for p in head.parameters())
        # 1M-5M range (design doc の "~5M" 見積、hidden=256 で ~1.5M 実測)
        assert 1e6 <= n <= 5e6, f"DepthHead param count {n / 1e6:.2f}M out of 1-5M range"

    def test_hidden_dim_scales_params(self) -> None:
        h_low = DepthHead(vision_embed_dim=512, num_cams=4, pool_size=8, hidden_dim=64)
        h_high = DepthHead(vision_embed_dim=512, num_cams=4, pool_size=8, hidden_dim=256)
        n_low = sum(p.numel() for p in h_low.parameters())
        n_high = sum(p.numel() for p in h_high.parameters())
        assert n_high > n_low
