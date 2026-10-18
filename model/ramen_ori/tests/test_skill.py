"""SkillEmbedding の unit test (Issue #115、Phase 1 M1)。

- init: weight shape (num_skills, d_model)、std ≈ init_std
- forward: (B,) → (B, 1, d_model)
- num_skills 切替 (6 inference aligned / 7 curation aligned) の A/B 両方 pass
- gradient flow: backward で weight.grad が populate される
"""

from __future__ import annotations

import pytest
import torch

from model.ramen_ori.skill import SkillEmbedding


class TestInit:
    @pytest.mark.parametrize("num_skills", [6, 7])
    def test_weight_shape_matches_vocab(self, num_skills: int) -> None:
        module = SkillEmbedding(num_skills=num_skills, d_model=512)
        assert module.embedding.weight.shape == (num_skills, 512)

    def test_init_std_close_to_configured(self) -> None:
        # d_model=512, num_skills=6 → 3072 サンプル、population std 0.02 に対する
        # 経験的な sample std deviation は ~2-3%。rel=0.1 で余裕を持たせて確認
        torch.manual_seed(0)
        module = SkillEmbedding(num_skills=6, d_model=512, init_std=0.02)
        actual_std = module.embedding.weight.std().item()
        assert actual_std == pytest.approx(0.02, rel=0.1)


class TestForward:
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_output_shape_is_batch_1_d_model(self, batch_size: int) -> None:
        module = SkillEmbedding(num_skills=6, d_model=512)
        skill_id = torch.zeros(batch_size, dtype=torch.long)
        out = module(skill_id)
        assert out.shape == (batch_size, 1, 512)

    @pytest.mark.parametrize("num_skills", [6, 7])
    def test_switchable_vocab_forward(self, num_skills: int) -> None:
        module = SkillEmbedding(num_skills=num_skills, d_model=256)
        skill_id = torch.arange(num_skills, dtype=torch.long)
        out = module(skill_id)
        assert out.shape == (num_skills, 1, 256)


class TestGradient:
    def test_backward_populates_grad(self) -> None:
        module = SkillEmbedding(num_skills=6, d_model=32)
        skill_id = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        out = module(skill_id)  # (4, 1, 32)
        loss = out.sum()
        loss.backward()
        assert module.embedding.weight.grad is not None
        assert module.embedding.weight.grad.abs().sum().item() > 0
