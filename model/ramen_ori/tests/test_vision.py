"""VisionEncoder の unit test (Issue #115、Phase 1 M4)。

- 高速 unit test は stub backbone を注入 (LingBot 実 DL 不要)
- optional integration test は from_lingbot で実 backbone、HF cache 有無で skip
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from model.ramen_ori.vision import VisionEncoder


class StubViT(nn.Module):
    """LingBot backbone API を最小模擬する stub。

    - `.patch_size` 属性 (int)
    - `.forward(x, is_training=True)` が dict を返し `"x_norm_patchtokens"` に
      shape (B, h*w, embed_dim) の tensor を持つ
    """

    def __init__(self, patch_size: int = 16, embed_dim: int = 768) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # gradient path を作るために dummy な conv-like variable を含める
        self.token_generator = nn.Linear(3, embed_dim)

    def forward(self, x: torch.Tensor, is_training: bool = False) -> dict:
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        # channel-avg してから embed_dim に project、grid に broadcast
        avg = x.mean(dim=(2, 3))  # (B, 3)
        feat = self.token_generator(avg)  # (B, embed_dim)
        tokens = feat.unsqueeze(1).expand(-1, h * w, -1).contiguous()
        return {"x_norm_patchtokens": tokens}


class TestInitAndFreeze:
    def test_frozen_backbone_params_have_no_grad(self) -> None:
        stub = StubViT(patch_size=16, embed_dim=768)
        module = VisionEncoder(backbone=stub, embed_dim=768, d_model=512, freeze=True)
        for name, p in module.backbone.named_parameters():
            assert p.requires_grad is False, f"backbone.{name} not frozen"

    def test_non_frozen_backbone_params_keep_grad(self) -> None:
        stub = StubViT(patch_size=16, embed_dim=768)
        module = VisionEncoder(backbone=stub, embed_dim=768, d_model=512, freeze=False)
        for name, p in module.backbone.named_parameters():
            assert p.requires_grad is True, f"backbone.{name} unexpectedly frozen"

    def test_train_mode_does_not_unfreeze_backbone(self) -> None:
        stub = StubViT()
        module = VisionEncoder(backbone=stub, embed_dim=768, d_model=512, freeze=True)
        module.train()  # user が train() 呼んでも frozen 側 backbone は eval のまま
        assert module.backbone.training is False
        # projection 側は train mode に入る
        assert module.projection.training is True

    def test_projection_shape(self) -> None:
        stub = StubViT()
        module = VisionEncoder(backbone=stub, embed_dim=768, d_model=512)
        assert module.projection.in_features == 768
        assert module.projection.out_features == 512


class TestForward:
    @pytest.mark.parametrize(
        "B,N_cams,pool_size", [(1, 4, 8), (2, 4, 8), (2, 2, 4), (1, 6, 8)]
    )
    def test_output_shape(self, B: int, N_cams: int, pool_size: int) -> None:
        stub = StubViT(patch_size=16, embed_dim=768)
        module = VisionEncoder(
            backbone=stub, embed_dim=768, d_model=512, pool_size=pool_size
        )
        module.eval()
        images = torch.randn(B, N_cams, 3, 224, 224)
        tokens = module(images)
        assert tokens.shape == (B, N_cams * pool_size**2, 512)

    def test_forward_with_non_224_resolution(self) -> None:
        # LingBot は任意の H,W (patch の倍数) をサポート、pool で 8×8 に正規化
        stub = StubViT(patch_size=16, embed_dim=768)
        module = VisionEncoder(backbone=stub, embed_dim=768, d_model=512, pool_size=8)
        module.eval()
        images = torch.randn(2, 4, 3, 320, 320)  # 20×20 patch → pool 8×8
        tokens = module(images)
        assert tokens.shape == (2, 4 * 64, 512)


class TestGradient:
    def test_frozen_backbone_no_grad_projection_grad_ok(self) -> None:
        stub = StubViT()
        module = VisionEncoder(backbone=stub, embed_dim=768, d_model=32, freeze=True)
        module.train()  # projection は grad on、backbone は frozen
        images = torch.randn(1, 2, 3, 224, 224)
        tokens = module(images)
        loss = tokens.sum()
        loss.backward()
        # projection grad populated
        assert module.projection.weight.grad is not None
        assert module.projection.weight.grad.abs().sum().item() > 0
        # backbone grad は無し (requires_grad=False なので .grad は None のまま)
        for p in module.backbone.parameters():
            assert p.grad is None


@pytest.mark.skipif(
    os.environ.get("RAMEN_ORI_INTEGRATION") != "1",
    reason="LingBot-B 実 DL & forward 用の integration test、env RAMEN_ORI_INTEGRATION=1 で有効",
)
class TestLingBotIntegration:
    def test_from_lingbot_loads_and_forwards(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        module = VisionEncoder.from_lingbot(
            variant="base", d_model=512, pool_size=8, freeze=True, device=device
        )
        module.eval()
        images = torch.randn(1, 4, 3, 224, 224, device=device)
        tokens = module(images)
        assert tokens.shape == (1, 4 * 64, 512)
        assert tokens.device.type == device
