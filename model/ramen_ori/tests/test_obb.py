"""ObbTokenizer の unit test (Issue #115、Phase 1 M3)。

- init: embedding shape, feature_dim=57 (verts 8 + conf 1 + class 32 + cam 16)
- forward: (B, N_cams, top_K, ...) input → (B, N_cams*top_K, d_model) + mask
- valid_mask propagation: pad detection が output mask で False
- training mode: detection dropout が確率的に mask を False 反転
- eval mode: dropout 適用されず、mask は input valid_mask のまま
- switchable num_cams / num_classes / dims
- gradient flow: 全 param に grad populate
"""

from __future__ import annotations

import pytest
import torch

from model.ramen_ori.obb import ObbTokenizer


def _make_inputs(
    B: int = 2,
    N_cams: int = 4,
    top_K: int = 4,
    num_classes: int = 7,
    num_cams: int = 6,
    all_valid: bool = True,
) -> dict:
    """テスト用の合成 input 一式を返す。"""
    verts = torch.rand(B, N_cams, top_K, 8)
    conf = torch.rand(B, N_cams, top_K, 1)
    class_id = torch.randint(0, num_classes, (B, N_cams, top_K), dtype=torch.long)
    cam_id = torch.randint(0, num_cams, (B, N_cams, top_K), dtype=torch.long)
    if all_valid:
        valid_mask = torch.ones(B, N_cams, top_K, dtype=torch.bool)
    else:
        valid_mask = torch.rand(B, N_cams, top_K) > 0.5
    return dict(
        verts=verts, conf=conf, class_id=class_id, cam_id=cam_id, valid_mask=valid_mask
    )


class TestInit:
    def test_feature_dim_is_57_with_default_config(self) -> None:
        module = ObbTokenizer(
            num_classes=7, num_cams=6, class_embed_dim=32, cam_embed_dim=16, d_model=512
        )
        assert module.feature_dim == 57
        assert module.projection.in_features == 57
        assert module.projection.out_features == 512

    def test_embedding_shapes_match_vocab(self) -> None:
        module = ObbTokenizer(num_classes=7, num_cams=6, d_model=512)
        assert module.class_embed.weight.shape == (7, 32)
        assert module.cam_embed.weight.shape == (6, 16)


class TestForward:
    @pytest.mark.parametrize("B,N_cams,top_K", [(1, 4, 4), (2, 4, 4), (4, 2, 8)])
    def test_output_shape(self, B: int, N_cams: int, top_K: int) -> None:
        module = ObbTokenizer(num_classes=7, num_cams=6, d_model=512, drop_rate=0.0)
        module.eval()
        inputs = _make_inputs(B=B, N_cams=N_cams, top_K=top_K)
        tokens, mask = module(**inputs)
        assert tokens.shape == (B, N_cams * top_K, 512)
        assert mask.shape == (B, N_cams * top_K)
        assert mask.dtype == torch.bool

    def test_valid_mask_propagates_to_output(self) -> None:
        module = ObbTokenizer(num_classes=7, num_cams=6, d_model=512, drop_rate=0.0)
        module.eval()
        # 特定 pattern: (B=1, N_cams=2, top_K=2)、valid_mask = [[T,F],[F,T]]
        inputs = _make_inputs(B=1, N_cams=2, top_K=2, all_valid=False)
        inputs["valid_mask"] = torch.tensor([[[True, False], [False, True]]])
        _, mask = module(**inputs)
        # flatten 順は (cam0_det0, cam0_det1, cam1_det0, cam1_det1)
        assert mask.tolist() == [[True, False, False, True]]


class TestDetectionDropout:
    def test_training_mode_reduces_true_count_on_average(self) -> None:
        module = ObbTokenizer(num_classes=7, num_cams=6, d_model=64, drop_rate=0.5)
        module.train()
        torch.manual_seed(0)
        # 全 valid で 1000 detection、drop_rate=0.5 → 期待 True 数 ~500
        inputs = _make_inputs(B=10, N_cams=10, top_K=10, all_valid=True)
        _, mask = module(**inputs)
        true_frac = mask.float().mean().item()
        # 大数の法則で 0.5 ± 0.03 に入る (1000 sample)
        assert true_frac == pytest.approx(0.5, abs=0.05)

    def test_eval_mode_preserves_valid_mask_exactly(self) -> None:
        module = ObbTokenizer(num_classes=7, num_cams=6, d_model=64, drop_rate=0.5)
        module.eval()
        inputs = _make_inputs(B=2, N_cams=4, top_K=4, all_valid=False)
        expected = inputs["valid_mask"].flatten(1, 2)
        _, mask = module(**inputs)
        assert torch.equal(mask, expected)


class TestSwitchable:
    @pytest.mark.parametrize("num_cams", [4, 6])
    @pytest.mark.parametrize("num_classes", [7, 10])
    def test_various_vocab_sizes(self, num_cams: int, num_classes: int) -> None:
        module = ObbTokenizer(
            num_classes=num_classes, num_cams=num_cams, d_model=256, drop_rate=0.0
        )
        module.eval()
        inputs = _make_inputs(
            B=2, N_cams=num_cams, top_K=3, num_classes=num_classes, num_cams=num_cams
        )
        tokens, mask = module(**inputs)
        assert tokens.shape == (2, num_cams * 3, 256)
        assert mask.shape == (2, num_cams * 3)


class TestGradient:
    def test_backward_populates_grad_for_all_params(self) -> None:
        module = ObbTokenizer(num_classes=7, num_cams=6, d_model=32, drop_rate=0.0)
        inputs = _make_inputs(B=2, N_cams=4, top_K=4)
        tokens, _ = module(**inputs)
        loss = tokens.sum()
        loss.backward()
        for name, param in module.named_parameters():
            assert param.grad is not None, f"{name} has no grad"
            assert param.grad.abs().sum().item() > 0, f"{name} grad is all zero"
