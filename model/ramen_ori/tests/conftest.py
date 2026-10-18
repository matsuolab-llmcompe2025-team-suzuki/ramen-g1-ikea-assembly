"""pytest fixture: 全 ramen_ori test で使う共通 helper。

- StubViT: LingBot backbone API を最小模擬、HF DL 不要で VisionEncoder / RamenOriPolicy を
  fast unit test 可能にする
- make_batch: RamenOriPolicy.compute_loss / predict_action に流す batch dict の合成
- make_default_policy: DI 化された RamenOriPolicy (Phase 2 A-0) を全 sub-module 込みで組む helper
- baked の dataset が os.environ に立てる frame cache の env を、test ごとに元へ戻す
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

_FRAME_CACHE_ENV = (
    "FRAME_CACHE_ROOT_OVERRIDE",
    "FRAME_CACHE_NUM_VARIANTS",
    "LEROBOT_FRAME_CACHE_ENABLE",
    "FRAME_CACHE_STRICT",
)


@pytest.fixture(autouse=True)
def _restore_frame_cache_env():
    """baked の dataset は frame cache の env (strict など) を立てる。後の test (frame cache patch の test など) に持ち越さない。"""
    saved = {key: os.environ.get(key) for key in _FRAME_CACHE_ENV}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

from model.ramen_ori.action_expert import FlowMatchingDiT
from model.ramen_ori.fusion import FusionTransformer
from model.ramen_ori.model import RamenOriPolicy
from model.ramen_ori.obb import ObbTokenizer
from model.ramen_ori.skill import SkillEmbedding
from model.ramen_ori.state import StateEncoder
from model.ramen_ori.temporal import TemporalEncoder
from model.ramen_ori.vision import VisionEncoder


class StubViT(nn.Module):
    def __init__(self, patch_size: int = 16, embed_dim: int = 768) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.token_generator = nn.Linear(3, embed_dim)

    def forward(self, x: torch.Tensor, is_training: bool = False) -> dict:
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        avg = x.mean(dim=(2, 3))
        feat = self.token_generator(avg)
        tokens = feat.unsqueeze(1).expand(-1, h * w, -1).contiguous()
        return {"x_norm_patchtokens": tokens}


@pytest.fixture
def stub_vit() -> StubViT:
    return StubViT(patch_size=16, embed_dim=768)


def make_default_policy(
    stub_vit_module: nn.Module | None = None,
    d_model: int = 512,
    # Issue #129 Phase A (2026-08-31): defaults を base.yaml と整合 (num_cams 6→4、action_dim 19→16)。
    num_cams: int = 4,
    chunk_len: int = 16,
    action_dim: int = 16,
    state_dim: int = 71,
    num_skills: int = 6,
    num_classes: int = 7,
    # Issue #129 Phase B (2026-08-31): L4 FK anchor 有効化オプション
    l4_enabled: bool = False,
    fk: nn.Module | None = None,
) -> RamenOriPolicy:
    """DI 型 RamenOriPolicy を default 構成 (Phase 1 A-1 LingBot + G-1 + B-7 + ...) で組む。

    test では stub_vit_module を渡して HF DL を回避。渡さない場合は StubViT を new。
    """
    backbone = stub_vit_module if stub_vit_module is not None else StubViT(patch_size=16, embed_dim=768)
    vision = VisionEncoder(
        backbone=backbone, embed_dim=backbone.embed_dim, d_model=d_model, pool_size=8, freeze=True
    )
    temporal = TemporalEncoder(
        num_cams=num_cams, embed_dim=384, depth=8, num_heads=6, pool_size=8,
        cam_embed_dim=16, d_model=d_model,
    )
    obb = ObbTokenizer(
        num_classes=num_classes, num_cams=num_cams, class_embed_dim=32,
        cam_embed_dim=16, d_model=d_model, drop_rate=0.05,
    )
    state = StateEncoder(state_dim=state_dim, d_model=d_model)
    skill = SkillEmbedding(num_skills=num_skills, d_model=d_model)
    fusion = FusionTransformer(d_model=d_model, num_layers=6, num_heads=8, mlp_dim=2048)
    action_expert = FlowMatchingDiT(
        chunk_len=chunk_len, action_dim=action_dim, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4.0, context_dim=d_model,
    )
    return RamenOriPolicy(
        vision=vision, temporal=temporal, obb=obb, state=state, skill=skill,
        fusion=fusion, action_expert=action_expert,
        d_model=d_model, num_cams=num_cams, chunk_len=chunk_len,
        action_dim=action_dim, sample_n_steps=6,
        fk=fk, l4_enabled=l4_enabled,
    )


def make_batch(
    B: int = 2,
    # Issue #129 Phase A (2026-08-31): defaults を base.yaml と整合
    # (N_cams 4→3、num_cams 6→4、action_dim 19→16、depth_target_num_cams 4→3)。
    N_cams: int = 3,
    top_K: int = 4,
    num_cams: int = 4,
    num_classes: int = 7,
    num_skills: int = 6,
    state_dim: int = 71,
    chunk_len: int = 16,
    action_dim: int = 16,
    img_size: int = 224,
    include_action: bool = True,
    include_action_waist_teacher: bool = False,  # Issue #129 Phase B: L4 FK anchor 用
    include_depth_target: bool = False,          # Alt-3 I-6 aux 用
    depth_target_size: tuple[int, int] = (32, 32),
    depth_target_num_cams: int = 3,
) -> dict:
    """RamenOriPolicy.compute_loss / predict_action 用の合成 batch。"""
    batch = {
        "images": torch.randn(B, N_cams, 3, img_size, img_size),
        "images_prev": torch.randn(B, N_cams, 3, img_size, img_size),
        "cam_id": torch.randint(0, num_cams, (B, N_cams), dtype=torch.long),
        "obb_verts": torch.rand(B, N_cams, top_K, 8),
        "obb_conf": torch.rand(B, N_cams, top_K, 1),
        "obb_class_id": torch.randint(0, num_classes, (B, N_cams, top_K), dtype=torch.long),
        "obb_cam_id": torch.randint(0, num_cams, (B, N_cams, top_K), dtype=torch.long),
        "obb_valid_mask": torch.rand(B, N_cams, top_K) > 0.3,
        "state": torch.randn(B, state_dim),
        "skill_id": torch.randint(0, num_skills, (B,), dtype=torch.long),
    }
    if include_action:
        batch["action"] = torch.randn(B, chunk_len, action_dim)
        # Issue #141 RO-2: 区間末尾の埋め草の行 (既定は全行が実データ)
        batch["action_is_pad"] = torch.zeros(B, chunk_len, dtype=torch.bool)
    if include_action_waist_teacher:
        # L4 FK anchor 用: teacher waist target (chunk_len, 3)
        batch["action_waist_teacher"] = torch.randn(B, chunk_len, 3)
    if include_depth_target:
        H, W = depth_target_size
        batch["depth_target"] = torch.randn(B, depth_target_num_cams, 1, H, W)
    return batch
