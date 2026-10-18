"""optim_utils.py の unit test (Issue #129 Phase C、2026-08-31)。

- find_backbone_blocks: LingBot 直、RADIO wrap の 2 path
- apply_partial_train_freeze: non-block frozen、lower blocks frozen、last N unfrozen
- make_partial_train_backbone_groups: LLRD LR (base_lr * decay^dist_from_last)、name field
- make_optimizer:
  - partial_train.enabled=false: single group、frozen 以外の全 trainable
  - partial_train.enabled=true : head group + backbone LLRD groups、head と backbone の param 分離
- WarmupCosineScheduler (train.py) の multi-group 対応: 各 group が initial LR を保ちつつ ratio でスケール
- Muon (Issue #141): split=1 で torch.optim.Muon と一致、split=n は block ごとの Muon と一致、
  学習する param の Muon / AdamW への振り分け、scheduler と再開 (load_state_dict の後も lr が届く)
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from model.ramen_ori.action_expert import FlowMatchingDiT
from model.ramen_ori.fusion import FusionTransformer
from model.ramen_ori.obb import ObbTokenizer
from model.ramen_ori.optim_utils import (
    SplitMuon,
    _orthogonalize,
    apply_partial_train_freeze,
    find_backbone_blocks,
    make_optimizer,
    make_partial_train_backbone_groups,
)
from model.ramen_ori.skill import SkillEmbedding
from model.ramen_ori.state import StateEncoder
from model.ramen_ori.train import WarmupCosineScheduler


# ============================================================================
# Stub backbones: LingBot / RADIO structure を模倣 (real HF DL 不要)
# ============================================================================


class _StubBlock(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.ffn = nn.Linear(dim, dim)


class _StubLingBotBackbone(nn.Module):
    """LingBot 構造模倣: .blocks を直接持つ。"""

    def __init__(self, n_blocks: int = 12, dim: int = 32) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(3, dim)
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([_StubBlock(dim) for _ in range(n_blocks)])


class _StubRadioModel(nn.Module):
    """RADIO 内部 VisionTransformer 模倣: .blocks を持つ。"""

    def __init__(self, n_blocks: int = 12, dim: int = 32) -> None:
        super().__init__()
        self.patch_generator = nn.Linear(3, dim)
        self.norm_pre = nn.LayerNorm(dim)
        self.blocks = nn.Sequential(*[_StubBlock(dim) for _ in range(n_blocks)])


class _StubRadioAdapter(nn.Module):
    """RadioAdapter wrap 相当: .radio.model.blocks path。"""

    def __init__(self, n_blocks: int = 12, dim: int = 32) -> None:
        super().__init__()
        self.radio = _RadioContainer(_StubRadioModel(n_blocks, dim))


class _RadioContainer(nn.Module):
    """RadioAdapter.radio = raw RADIOModel、その .model が VisionTransformer。"""

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.model = vit


class _StubUnknownBackbone(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.some_other_attr = nn.Linear(dim, dim)


# ============================================================================
# TestFindBackboneBlocks
# ============================================================================


class TestFindBackboneBlocks:
    def test_lingbot_returns_direct_blocks(self) -> None:
        bb = _StubLingBotBackbone(n_blocks=12)
        blocks = find_backbone_blocks(bb)
        assert len(blocks) == 12
        assert blocks is bb.blocks

    def test_radio_returns_nested_blocks(self) -> None:
        bb = _StubRadioAdapter(n_blocks=12)
        blocks = find_backbone_blocks(bb)
        assert len(blocks) == 12
        assert blocks is bb.radio.model.blocks

    def test_unknown_backbone_raises(self) -> None:
        bb = _StubUnknownBackbone()
        with pytest.raises(ValueError, match="cannot locate .blocks"):
            find_backbone_blocks(bb)


# ============================================================================
# TestApplyPartialTrainFreeze
# ============================================================================


class TestApplyPartialTrainFreeze:
    def test_all_non_block_frozen(self) -> None:
        bb = _StubLingBotBackbone(n_blocks=12)
        apply_partial_train_freeze(bb, last_n_blocks=4)
        # patch_embed / norm は freeze 継続
        for p in bb.patch_embed.parameters():
            assert p.requires_grad is False
        for p in bb.norm.parameters():
            assert p.requires_grad is False

    def test_lower_blocks_frozen(self) -> None:
        bb = _StubLingBotBackbone(n_blocks=12)
        apply_partial_train_freeze(bb, last_n_blocks=4)
        # blocks[0..7] は freeze 継続 (total 12 - last 4 = 8)
        for i in range(8):
            for p in bb.blocks[i].parameters():
                assert p.requires_grad is False, f"block[{i}] should be frozen"

    def test_last_n_blocks_unfrozen(self) -> None:
        bb = _StubLingBotBackbone(n_blocks=12)
        apply_partial_train_freeze(bb, last_n_blocks=4)
        # blocks[8..11] は unfrozen
        for i in range(8, 12):
            for p in bb.blocks[i].parameters():
                assert p.requires_grad is True, f"block[{i}] should be unfrozen"

    def test_idempotent(self) -> None:
        """複数回呼んでも状態は同じ。"""
        bb = _StubLingBotBackbone(n_blocks=12)
        apply_partial_train_freeze(bb, last_n_blocks=4)
        state1 = [p.requires_grad for p in bb.parameters()]
        apply_partial_train_freeze(bb, last_n_blocks=4)
        state2 = [p.requires_grad for p in bb.parameters()]
        assert state1 == state2

    def test_last_n_zero_freezes_all(self) -> None:
        bb = _StubLingBotBackbone(n_blocks=12)
        apply_partial_train_freeze(bb, last_n_blocks=0)
        for p in bb.parameters():
            assert p.requires_grad is False

    def test_last_n_all_unfreezes_all_blocks(self) -> None:
        """last_n_blocks=total で blocks は全 unfrozen (non-block は依然 frozen)。"""
        bb = _StubLingBotBackbone(n_blocks=12)
        apply_partial_train_freeze(bb, last_n_blocks=12)
        for i in range(12):
            for p in bb.blocks[i].parameters():
                assert p.requires_grad is True
        for p in bb.patch_embed.parameters():
            assert p.requires_grad is False

    def test_out_of_range_raises(self) -> None:
        bb = _StubLingBotBackbone(n_blocks=12)
        with pytest.raises(ValueError, match="out of range"):
            apply_partial_train_freeze(bb, last_n_blocks=13)
        with pytest.raises(ValueError, match="out of range"):
            apply_partial_train_freeze(bb, last_n_blocks=-1)


# ============================================================================
# TestMakePartialTrainBackboneGroups (LLRD)
# ============================================================================


class TestMakePartialTrainBackboneGroups:
    @pytest.fixture
    def frozen_bb(self) -> _StubLingBotBackbone:
        bb = _StubLingBotBackbone(n_blocks=12)
        apply_partial_train_freeze(bb, last_n_blocks=4)
        return bb

    def test_returns_last_n_groups(self, frozen_bb) -> None:
        groups = make_partial_train_backbone_groups(
            frozen_bb, last_n_blocks=4, base_lr=2e-5, llrd_decay=0.75
        )
        assert len(groups) == 4

    def test_llrd_lr_matches_formula(self, frozen_bb) -> None:
        """block[11] = base_lr、block[10] = base_lr * 0.75、block[9] = base_lr * 0.75^2、block[8] = base_lr * 0.75^3。"""
        base_lr = 2e-5
        decay = 0.75
        groups = make_partial_train_backbone_groups(
            frozen_bb, last_n_blocks=4, base_lr=base_lr, llrd_decay=decay
        )
        # groups 順は block_8, 9, 10, 11 (loop iterates from total-last_n=8 to total-1=11)
        expected = [
            base_lr * (decay ** 3),  # block 8: dist=3
            base_lr * (decay ** 2),  # block 9: dist=2
            base_lr * (decay ** 1),  # block 10: dist=1
            base_lr * (decay ** 0),  # block 11: dist=0
        ]
        for g, exp_lr in zip(groups, expected):
            assert g["lr"] == pytest.approx(exp_lr, rel=1e-6)

    def test_group_names_indicate_block_index(self, frozen_bb) -> None:
        groups = make_partial_train_backbone_groups(
            frozen_bb, last_n_blocks=4, base_lr=2e-5, llrd_decay=0.75
        )
        names = [g["name"] for g in groups]
        assert names == [
            "backbone.block_8",
            "backbone.block_9",
            "backbone.block_10",
            "backbone.block_11",
        ]

    def test_lr_decreases_toward_earlier_blocks(self, frozen_bb) -> None:
        """後の block ほど LR 高い (last=base_lr、以前は decay かけて小さくなる)。"""
        groups = make_partial_train_backbone_groups(
            frozen_bb, last_n_blocks=4, base_lr=2e-5, llrd_decay=0.75
        )
        lrs = [g["lr"] for g in groups]
        # block_8 < block_9 < block_10 < block_11
        assert lrs[0] < lrs[1] < lrs[2] < lrs[3]

    def test_frozen_block_excluded(self) -> None:
        """apply_partial_train_freeze を skip、全 backbone frozen の状態で呼ぶと空 group リスト。"""
        bb = _StubLingBotBackbone(n_blocks=12)
        for p in bb.parameters():
            p.requires_grad = False
        groups = make_partial_train_backbone_groups(
            bb, last_n_blocks=4, base_lr=2e-5, llrd_decay=0.75
        )
        # 全 block frozen → block_params 空 → skip されて groups は空
        assert len(groups) == 0


# ============================================================================
# TestMakeOptimizer (統合)
# ============================================================================


class _StubPolicy(nn.Module):
    """head (linear) + vision.backbone (StubLingBot or StubRadio) を持つ minimal model。"""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        # RamenOriPolicy と同じく model.vision.backbone path
        self.vision = _VisionContainer(backbone)
        self.head = nn.Linear(32, 16)


class _VisionContainer(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone


class TestMakeOptimizer:
    def test_legacy_single_group(self) -> None:
        """partial_train.enabled=false → single group AdamW、head + backbone 全 trainable。"""
        model = _StubPolicy(_StubLingBotBackbone(n_blocks=12))
        cfg = OmegaConf.create(
            {"lr": 1e-4, "betas": [0.9, 0.95], "weight_decay": 1e-2}
        )
        opt = make_optimizer(model, cfg)
        assert isinstance(opt, torch.optim.AdamW)
        assert len(opt.param_groups) == 1

    def test_legacy_disabled_partial_train(self) -> None:
        """partial_train.enabled=false でも single group、backbone param は不変。"""
        model = _StubPolicy(_StubLingBotBackbone(n_blocks=12))
        # 事前に全 True (default) と assume
        before = [p.requires_grad for p in model.vision.backbone.parameters()]
        cfg = OmegaConf.create(
            {
                "lr": 1e-4,
                "partial_train": {"enabled": False, "last_n_blocks": 4},
            }
        )
        make_optimizer(model, cfg)
        after = [p.requires_grad for p in model.vision.backbone.parameters()]
        assert before == after, "partial_train.enabled=false で backbone requires_grad は不変"

    def test_partial_train_creates_multiple_groups(self) -> None:
        """partial_train.enabled=true で head + backbone LLRD の 1 + last_n groups。"""
        model = _StubPolicy(_StubLingBotBackbone(n_blocks=12))
        cfg = OmegaConf.create(
            {
                "lr": 1e-4,
                "partial_train": {
                    "enabled": True,
                    "last_n_blocks": 4,
                    "llrd_decay": 0.75,
                    "backbone_lr_ratio": 0.2,
                },
            }
        )
        opt = make_optimizer(model, cfg)
        # head (1) + backbone LLRD (4) = 5 groups
        assert len(opt.param_groups) == 5
        # 順序: head first, then backbone_8, 9, 10, 11
        assert opt.param_groups[0]["name"] == "head"
        assert opt.param_groups[-1]["name"] == "backbone.block_11"

    def test_partial_train_head_lr_matches_optim_lr(self) -> None:
        model = _StubPolicy(_StubLingBotBackbone(n_blocks=12))
        cfg = OmegaConf.create(
            {
                "lr": 1e-4,
                "partial_train": {"enabled": True, "last_n_blocks": 4, "llrd_decay": 0.75, "backbone_lr_ratio": 0.2},
            }
        )
        opt = make_optimizer(model, cfg)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)

    def test_partial_train_backbone_last_block_lr(self) -> None:
        """Last block (block_11) LR = optim.lr * backbone_lr_ratio (LLRD dist=0)。"""
        model = _StubPolicy(_StubLingBotBackbone(n_blocks=12))
        cfg = OmegaConf.create(
            {
                "lr": 1e-4,
                "partial_train": {"enabled": True, "last_n_blocks": 4, "llrd_decay": 0.75, "backbone_lr_ratio": 0.2},
            }
        )
        opt = make_optimizer(model, cfg)
        expected_last = 1e-4 * 0.2  # 2e-5
        # backbone.block_11 は最後の group
        last_g = [g for g in opt.param_groups if g["name"] == "backbone.block_11"][0]
        assert last_g["lr"] == pytest.approx(expected_last)

    def test_partial_train_freezes_backbone_correctly(self) -> None:
        """partial_train enabled 後、backbone は last N のみ unfrozen。"""
        model = _StubPolicy(_StubLingBotBackbone(n_blocks=12))
        cfg = OmegaConf.create(
            {
                "lr": 1e-4,
                "partial_train": {"enabled": True, "last_n_blocks": 4, "llrd_decay": 0.75, "backbone_lr_ratio": 0.2},
            }
        )
        make_optimizer(model, cfg)
        for i in range(8):
            for p in model.vision.backbone.blocks[i].parameters():
                assert p.requires_grad is False
        for i in range(8, 12):
            for p in model.vision.backbone.blocks[i].parameters():
                assert p.requires_grad is True

    def test_partial_train_head_backbone_no_param_overlap(self) -> None:
        """head group と backbone group で param が重複しない (id で set 引き)。"""
        model = _StubPolicy(_StubLingBotBackbone(n_blocks=12))
        cfg = OmegaConf.create(
            {
                "lr": 1e-4,
                "partial_train": {"enabled": True, "last_n_blocks": 4, "llrd_decay": 0.75, "backbone_lr_ratio": 0.2},
            }
        )
        opt = make_optimizer(model, cfg)
        head_ids = {id(p) for p in opt.param_groups[0]["params"]}
        for g in opt.param_groups[1:]:
            for p in g["params"]:
                assert id(p) not in head_ids, f"backbone param leaked into head group ({g['name']})"


# ============================================================================
# TestWarmupCosineSchedulerMultiGroup
# ============================================================================


class TestWarmupCosineSchedulerMultiGroup:
    def _make_multi_group_optim(self) -> torch.optim.AdamW:
        p1 = nn.Parameter(torch.zeros(4))
        p2 = nn.Parameter(torch.zeros(4))
        p3 = nn.Parameter(torch.zeros(4))
        return torch.optim.AdamW(
            [
                {"params": [p1], "lr": 1e-4, "name": "head"},
                {"params": [p2], "lr": 2e-5, "name": "backbone.block_11"},
                {"params": [p3], "lr": 1.5e-5, "name": "backbone.block_10"},
            ],
            lr=1e-4,
        )

    def test_warmup_preserves_group_ratio(self) -> None:
        opt = self._make_multi_group_optim()
        sched = WarmupCosineScheduler(
            opt, warmup_steps=100, max_steps=1000, base_lr=1e-4, end_lr=1e-5
        )
        # step 50 (mid warmup): ratio = 51/100 = 0.51
        for _ in range(51):
            sched.step()
        expected_ratio = 51 / 100
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4 * expected_ratio, rel=1e-6)
        assert opt.param_groups[1]["lr"] == pytest.approx(2e-5 * expected_ratio, rel=1e-6)
        assert opt.param_groups[2]["lr"] == pytest.approx(1.5e-5 * expected_ratio, rel=1e-6)

    def test_end_of_schedule_hits_end_lr_ratio(self) -> None:
        opt = self._make_multi_group_optim()
        end_lr_ratio = 0.1  # 1e-5 / 1e-4
        sched = WarmupCosineScheduler(
            opt, warmup_steps=100, max_steps=1000, base_lr=1e-4, end_lr=1e-5
        )
        # step すべて消費 (max_steps)
        for _ in range(1000):
            sched.step()
        # cosine 到達点 = initial_lr * end_lr_ratio
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4 * end_lr_ratio, rel=1e-4)
        assert opt.param_groups[1]["lr"] == pytest.approx(2e-5 * end_lr_ratio, rel=1e-4)
        assert opt.param_groups[2]["lr"] == pytest.approx(1.5e-5 * end_lr_ratio, rel=1e-4)

    def test_single_group_backward_compat(self) -> None:
        """Legacy single-group: 従来 API (base_lr / end_lr) と同じ結果。"""
        p = nn.Parameter(torch.zeros(4))
        opt = torch.optim.AdamW([p], lr=1e-4)
        sched = WarmupCosineScheduler(
            opt, warmup_steps=100, max_steps=1000, base_lr=1e-4, end_lr=1e-5
        )
        # warmup step 50
        for _ in range(51):
            sched.step()
        expected = 1e-4 * (51 / 100)  # 従来 formula
        assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6)

    def test_end_lr_ratio_direct_override(self) -> None:
        opt = self._make_multi_group_optim()
        # end_lr_ratio 直接指定 (legacy base_lr / end_lr は無視)
        sched = WarmupCosineScheduler(
            opt, warmup_steps=10, max_steps=100, end_lr_ratio=0.05
        )
        assert sched.end_lr_ratio == pytest.approx(0.05)

    def test_cosine_ratio_monotonic_decrease(self) -> None:
        """warmup 後 cosine phase は monotonically decreasing ratio。"""
        opt = self._make_multi_group_optim()
        sched = WarmupCosineScheduler(
            opt, warmup_steps=10, max_steps=100, base_lr=1e-4, end_lr=1e-5
        )
        # skip warmup
        for _ in range(10):
            sched.step()
        prev_ratio = 1.0
        for _ in range(90):
            r = sched.get_ratio(sched._step)
            assert r <= prev_ratio + 1e-9, f"ratio 単調減少崩れ: {prev_ratio} -> {r}"
            prev_ratio = r
            sched.step()

    def test_invalid_schedule_raises(self) -> None:
        p = nn.Parameter(torch.zeros(4))
        opt = torch.optim.AdamW([p], lr=1e-4)
        with pytest.raises(ValueError, match="invalid schedule"):
            WarmupCosineScheduler(opt, warmup_steps=100, max_steps=50, base_lr=1e-4)


# ============================================================================
# TestRadioPathIntegration
# ============================================================================


class TestRadioPathIntegration:
    """RADIO wrap path (backbone.radio.model.blocks) が make_optimizer 経由でも動く。"""

    def test_radio_backbone_partial_train(self) -> None:
        model = _StubPolicy(_StubRadioAdapter(n_blocks=12))
        cfg = OmegaConf.create(
            {
                "lr": 1e-4,
                "partial_train": {"enabled": True, "last_n_blocks": 4, "llrd_decay": 0.75, "backbone_lr_ratio": 0.2},
            }
        )
        opt = make_optimizer(model, cfg)
        assert len(opt.param_groups) == 5  # head + 4 backbone
        # RADIO 側の blocks は Sequential、freeze / unfreeze 動作確認
        radio_blocks = model.vision.backbone.radio.model.blocks
        for i in range(8):
            for p in radio_blocks[i].parameters():
                assert p.requires_grad is False
        for i in range(8, 12):
            for p in radio_blocks[i].parameters():
                assert p.requires_grad is True


# ============================================================================
# Muon (Issue #141 6 本目、optim.name=muon)
# ============================================================================

BASE_OPTIM = OmegaConf.load(Path(__file__).parents[1] / "configs" / "base.yaml").optim


def _muon_cfg(**over):
    return OmegaConf.merge(BASE_OPTIM, {"name": "muon", **over})


class _ToyRamenOri(nn.Module):
    """RAMEN-Ori と同じ module 名の小さな model (base.yaml の optim.muon の名前がそのまま当たる)。"""

    def __init__(self) -> None:
        super().__init__()
        d = 32
        self.obb = ObbTokenizer(
            num_classes=7, num_cams=4, class_embed_dim=8, cam_embed_dim=4, d_model=d, drop_rate=0.0
        )
        self.state = StateEncoder(state_dim=71, d_model=d)
        self.skill = SkillEmbedding(num_skills=6, d_model=d)
        self.fusion = FusionTransformer(d_model=d, num_layers=2, num_heads=4, mlp_dim=64, qk_norm=True)
        self.action_expert = FlowMatchingDiT(
            chunk_len=4, action_dim=3, embed_dim=d, depth=2, num_heads=4, context_dim=d, qk_norm=True
        )


class TestSplitMuon:
    def test_split1_matches_torch_muon(self) -> None:
        torch.manual_seed(0)
        shapes = [(16, 8), (8, 24)]
        ours = [nn.Parameter(torch.randn(s)) for s in shapes]
        ref = [nn.Parameter(p.detach().clone()) for p in ours]
        opt = SplitMuon(ours, lr=0.02, weight_decay=0.01)
        opt_ref = torch.optim.Muon(ref, lr=0.02, weight_decay=0.01, adjust_lr_fn="match_rms_adamw")
        for _ in range(3):
            for p, q in zip(ours, ref):
                g = torch.randn_like(p)
                p.grad, q.grad = g.clone(), g.clone()
            opt.step()
            opt_ref.step()
        for p, q in zip(ours, ref):
            torch.testing.assert_close(p, q, atol=1e-5, rtol=1e-4)

    def test_split_equals_torch_muon_on_each_block(self) -> None:
        # (3m, n) を split=3 で更新 = 3 つの (m, n) を別々に torch の Muon で更新 (更新の大きさも block の形で決まる)
        torch.manual_seed(0)
        fused = nn.Parameter(torch.randn(3 * 8, 12))
        blocks = [nn.Parameter(b.clone()) for b in fused.detach().chunk(3)]
        opt = SplitMuon([{"params": [fused], "split": 3}], lr=0.02, weight_decay=0.01)
        opt_ref = torch.optim.Muon(blocks, lr=0.02, weight_decay=0.01, adjust_lr_fn="match_rms_adamw")
        for _ in range(3):
            g = torch.randn_like(fused)
            fused.grad = g.clone()
            for b, gb in zip(blocks, g.chunk(3)):
                b.grad = gb.clone()
            opt.step()
            opt_ref.step()
        torch.testing.assert_close(fused.detach(), torch.cat([b.detach() for b in blocks]), atol=1e-5, rtol=1e-4)

    def test_rejects_rows_not_divisible_by_split(self) -> None:
        with pytest.raises(ValueError, match="split=3"):
            SplitMuon([{"params": [nn.Parameter(torch.randn(10, 4))], "split": 3}], lr=0.01, weight_decay=0.0)

    # Issue #141: c32_muon が 2 回とも直交化の辺りの GPU の kernel の異常で落ちたので、計算の経路を選べるようにした

    @pytest.mark.parametrize("shape, split", [((24, 8), 3), ((8, 24), 1), ((16, 16), 2), ((18, 6), 1)])
    def test_matmul_impl_matches_baddbmm(self, shape, split) -> None:
        # 同じ fp32 なら、baddbmm と普通の行列積 (転置の view を渡さない) は丸めの差だけ。縦長 (転置する) も横長も
        torch.manual_seed(0)
        update = torch.randn(shape)
        ref = _orthogonalize(update, split, 5, 1e-7, torch.float32, "baddbmm")
        out = _orthogonalize(update, split, 5, 1e-7, torch.float32, "matmul")
        assert out.shape == update.shape and out.dtype == torch.float32
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)

    def test_float32_is_close_to_bfloat16(self) -> None:
        # fp32 は bf16 の丸めが無いだけで、同じ直交化 (更新の向きはほぼ同じ)
        torch.manual_seed(0)
        update = torch.randn(3 * 16, 32)
        bf16 = _orthogonalize(update, 3, 5, 1e-7).float()
        fp32 = _orthogonalize(update, 3, 5, 1e-7, torch.float32, "matmul")
        cos = torch.nn.functional.cosine_similarity(bf16.flatten(), fp32.flatten(), dim=0)
        assert cos > 0.99

    def test_ns_options_keep_torch_muon_default_and_survive_resume(self) -> None:
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(16, 8))
        old = SplitMuon([p], lr=0.02, weight_decay=0.0)
        assert (old.ns_dtype, old.ns_impl) == ("bfloat16", "baddbmm")  # torch.optim.Muon と同じ既定
        # 既定のままの ckpt から、fp32 + matmul の設定で再開しても設定は残る (load_state_dict は group の値を上書きする)
        p.grad = torch.randn_like(p)
        old.step()
        new = SplitMuon([p], lr=0.02, weight_decay=0.0, ns_dtype="float32", ns_impl="matmul")
        new.load_state_dict(copy.deepcopy(old.state_dict()))
        assert (new.ns_dtype, new.ns_impl) == ("float32", "matmul")
        new.step()

    def test_rejects_unknown_ns_options(self) -> None:
        p = nn.Parameter(torch.randn(8, 8))
        with pytest.raises(ValueError, match="ns_dtype"):
            SplitMuon([p], lr=0.01, weight_decay=0.0, ns_dtype="float16")
        with pytest.raises(ValueError, match="ns_impl"):
            SplitMuon([p], lr=0.01, weight_decay=0.0, ns_impl="bmm")


class TestMakeMuonOptimizer:
    def test_assignment(self) -> None:
        model = _ToyRamenOri()
        params = dict(model.named_parameters())
        name_of = {id(p): n for n, p in params.items()}
        opt = make_optimizer(model, _muon_cfg())
        groups = {g["name"]: [name_of[id(p)] for p in g["params"]] for g in opt.param_groups}
        # 学習する param はどれもちょうど 1 つの group に入る
        assert sorted(n for ns in groups.values() for n in ns) == sorted(params)
        entry = tuple(f"{m}." for m in BASE_OPTIM.muon.adamw_modules)
        embeddings = {"skill.embedding.weight", "obb.class_embed.weight", "obb.cam_embed.weight"}
        assert set(groups["adamw"]) == {
            n for n, p in params.items() if n.startswith(entry) or p.ndim != 2 or n in embeddings
        }
        assert set(groups["muon_split3"]) == {n for n in params if n.endswith("in_proj_weight")}
        assert len(groups["muon_split3"]) == 2 + 2 * 2  # fusion 2 層 + action expert 2 block の self / cross
        assert set(groups["muon_split9"]) == {f"action_expert.blocks.{i}.adaln.1.weight" for i in range(2)}
        assert groups["muon_split2"] == ["action_expert.adaln_final.1.weight"]
        assert {"state.fc2.weight", "action_expert.context_proj.weight", "action_expert.time_embed.mlp.2.weight"} <= set(
            groups["muon_split1"]
        )
        # QK-norm の scale は 1D なので AdamW
        assert "fusion.encoder.layers.0.q_norm.weight" in groups["adamw"]

    def test_ns_options_from_config(self) -> None:
        opt = make_optimizer(_ToyRamenOri(), _muon_cfg())
        assert (opt.muon.ns_dtype, opt.muon.ns_impl) == ("bfloat16", "baddbmm")  # base.yaml の既定
        opt = make_optimizer(_ToyRamenOri(), _muon_cfg(muon={"ns_dtype": "float32", "ns_impl": "matmul"}))
        assert (opt.muon.ns_dtype, opt.muon.ns_impl) == ("float32", "matmul")

    def test_unknown_names_raise(self) -> None:
        cfg = _muon_cfg(muon={"adamw_modules": ["state.fc9"], "split": [["qkv_weight", 3]]})
        with pytest.raises(ValueError, match="state.fc9.*qkv_weight"):
            make_optimizer(_ToyRamenOri(), cfg)

    def test_partial_train_with_muon_raises(self) -> None:
        with pytest.raises(ValueError, match="partial_train"):
            make_optimizer(_ToyRamenOri(), _muon_cfg(partial_train={"enabled": True}))

    def test_scheduler_and_resume(self) -> None:
        torch.manual_seed(0)
        model = _ToyRamenOri()
        opt = make_optimizer(model, _muon_cfg())
        sched = WarmupCosineScheduler(opt, warmup_steps=2, max_steps=10, base_lr=1e-4, end_lr=1e-5)
        sched.step()
        assert all(g["lr"] == pytest.approx(5e-5) for g in opt.param_groups)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step()

        # train.py の再開と同じ順: optimizer → scheduler → ckpt の state を読む
        model2 = copy.deepcopy(model)
        opt2 = make_optimizer(model2, _muon_cfg())
        sched2 = WarmupCosineScheduler(opt2, warmup_steps=2, max_steps=10, base_lr=1e-4, end_lr=1e-5)
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
        sched2._step = sched._step
        sched.step()
        sched2.step()
        # load_state_dict で差し替わった group にも scheduler の lr が届く
        assert [g["lr"] for g in opt2.param_groups] == [g["lr"] for g in opt.param_groups]
        assert opt2.muon.state[opt2.muon.param_groups[0]["params"][0]]  # momentum も引き継ぐ
        for (_, p), (_, q) in zip(model.named_parameters(), model2.named_parameters()):
            g = torch.randn_like(p)
            p.grad, q.grad = g.clone(), g.clone()
        opt.step()
        opt2.step()
        for p, q in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p, q)
