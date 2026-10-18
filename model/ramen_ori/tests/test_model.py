"""RamenOriPolicy 統合 test (Issue #115、Phase 1 M8、Phase 2 A-0 で DI 化に更新)。

- Init with stub vision_backbone (HF DL 不要)
- compute_loss: scalar 返す、backward で 非-frozen param に grad populate
- predict_action: (B, chunk_len, action_dim) shape
- Frozen backbone param が model init 後も requires_grad=False
- Fast smoke なので minimal config (embed 小、depth 少) を使う
- Sub-module は個別 instantiate して DI (Phase 2 A-0)、alternative は _target_ swap で追加
"""

from __future__ import annotations

import pytest
import torch

from model.ramen_ori.action_expert import FlowMatchingDiT
from model.ramen_ori.aux_depth import DepthHead
from model.ramen_ori.fusion import FusionTransformer
from model.ramen_ori.model import RamenOriPolicy
from model.ramen_ori.obb import ObbTokenizer
from model.ramen_ori.skill import SkillEmbedding
from model.ramen_ori.state import StateEncoder
from model.ramen_ori.temporal import TemporalEncoder
from model.ramen_ori.tests.conftest import make_batch
from model.ramen_ori.vision import VisionEncoder


def _make_small_policy(
    stub_vit,
    chunk_len: int = 8,
    action_dim: int = 4,
    N_cams: int = 4,
    sample_n_steps: int = 6,
    aux_head: torch.nn.Module | None = None,
    aux_weight: float = 0.1,
    # FK の loss (Issue #141 RO-16: 3 項 + 毎 step の重み)
    l4_enabled: bool = False,
    fk_weight_max: float = 1.2,
    fk: torch.nn.Module | None = None,
    # Issue #141 RO-4 / RO-6: model の中の正規化と、StateEncoder 入口の LayerNorm
    normalization: bool = False,
    state_input_norm: bool = True,
    use_temporal: bool = True,
    freeze_vision: bool = True,
    memory: torch.nn.Module | None = None,
    state_clip: float | None = None,
) -> RamenOriPolicy:
    """Fast test 用の minimal config policy (DI 型)。"""
    d_model = 64
    num_cams = 6
    vision = VisionEncoder(backbone=stub_vit, embed_dim=768, d_model=d_model, pool_size=4, freeze=freeze_vision)
    temporal = TemporalEncoder(
        num_cams=num_cams, embed_dim=64, depth=2, num_heads=4, pool_size=4,
        cam_embed_dim=16, d_model=d_model,
    ) if use_temporal else None
    obb = ObbTokenizer(
        num_classes=7, num_cams=num_cams, class_embed_dim=32,
        cam_embed_dim=16, d_model=d_model, drop_rate=0.05,
    )
    state = StateEncoder(state_dim=71, d_model=d_model, input_norm=state_input_norm)
    skill = SkillEmbedding(num_skills=6, d_model=d_model)
    fusion = FusionTransformer(d_model=d_model, num_layers=2, num_heads=4, mlp_dim=128)
    action_expert = FlowMatchingDiT(
        chunk_len=chunk_len, action_dim=action_dim, embed_dim=64, depth=2,
        num_heads=4, mlp_ratio=4.0, context_dim=d_model,
    )
    return RamenOriPolicy(
        vision=vision, temporal=temporal, obb=obb, state=state, skill=skill,
        fusion=fusion, action_expert=action_expert,
        aux_head=aux_head, aux_weight=aux_weight,
        fk=fk, l4_enabled=l4_enabled, fk_weight_max=fk_weight_max,
        normalization=normalization, memory=memory, state_clip=state_clip,
        d_model=d_model, num_cams=num_cams, chunk_len=chunk_len,
        action_dim=action_dim, sample_n_steps=sample_n_steps,
    )


def _set_random_stats(policy: RamenOriPolicy, action_dim: int, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    policy.set_normalization_stats(
        torch.randn(71, generator=g), torch.rand(71, generator=g) + 0.1,
        torch.randn(action_dim, generator=g), torch.rand(action_dim, generator=g) + 0.1,
    )


def _make_depth_aux_head(N_cams: int = 4, pool_size: int = 4, output_size: int = 8) -> DepthHead:
    """test 用 minimal DepthHead (d=64, pool=4, output=8)。"""
    return DepthHead(
        vision_embed_dim=64, num_cams=N_cams, pool_size=pool_size,
        output_size=(output_size, output_size), hidden_dim=32,
    )


class TestInit:
    def test_all_submodules_instantiated(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit)
        assert policy.vision is not None
        assert policy.temporal is not None
        assert policy.obb is not None
        assert policy.state is not None
        assert policy.skill is not None
        assert policy.fusion is not None
        assert policy.action_expert is not None

    def test_frozen_vision_backbone_after_init(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit)
        for name, p in policy.vision.backbone.named_parameters():
            assert p.requires_grad is False, f"vision.backbone.{name} not frozen"


class TestComputeLoss:
    def test_loss_is_scalar(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit)
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=4)
        loss = policy.compute_loss(batch)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_backward_grads_non_frozen_params(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit)
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=4)
        loss = policy.compute_loss(batch)
        loss.backward()
        # 学習可能な param が非 zero grad を持つ
        learnable = [
            (n, p) for n, p in policy.named_parameters() if p.requires_grad
        ]
        for name, p in learnable:
            assert p.grad is not None, f"{name} has no grad"
        # frozen backbone (vision) は grad=None のまま
        for name, p in policy.vision.backbone.named_parameters():
            assert p.grad is None, f"frozen vision.backbone.{name} unexpectedly has grad"


class TestPredictAction:
    @pytest.mark.parametrize(
        "B,N_cams,chunk_len,action_dim", [(1, 4, 8, 4), (2, 4, 16, 19)]
    )
    def test_output_shape(
        self, stub_vit, B: int, N_cams: int, chunk_len: int, action_dim: int
    ) -> None:
        policy = _make_small_policy(
            stub_vit, chunk_len=chunk_len, action_dim=action_dim, sample_n_steps=2,
        )
        policy.eval()
        batch = make_batch(
            B=B, N_cams=N_cams, chunk_len=chunk_len, action_dim=action_dim, include_action=False
        )
        action = policy.predict_action(batch)
        assert action.shape == (B, chunk_len, action_dim)


class TestAuxDepthLoss:
    """I-6 Depth aux (Alt-3) の compute_loss 統合。"""

    def test_no_aux_head_default_action_loss_only(self, stub_vit) -> None:
        """aux_head=None (I-1 baseline) では action_loss のみ、depth_target 提供しても影響ゼロ。"""
        policy = _make_small_policy(stub_vit)
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=4, include_depth_target=True)
        # Flow matching は内部で torch.randn を呼ぶので seed 固定して 2 回比較
        torch.manual_seed(42)
        loss = policy.compute_loss(batch)
        torch.manual_seed(42)
        loss_no_target = policy.compute_loss({k: v for k, v in batch.items() if k != "depth_target"})
        assert loss.item() == pytest.approx(loss_no_target.item(), abs=1e-6)

    def test_aux_head_with_target_adds_loss(self, stub_vit) -> None:
        """aux_head 有 + depth_target 有 → aux loss が加算される (action_loss 単体より大)。"""
        torch.manual_seed(0)
        policy_no_aux = _make_small_policy(stub_vit)
        torch.manual_seed(0)
        aux_head = _make_depth_aux_head(N_cams=4, pool_size=4, output_size=8)
        policy_with_aux = _make_small_policy(stub_vit, aux_head=aux_head, aux_weight=1.0)
        batch = make_batch(
            B=2, N_cams=4, chunk_len=8, action_dim=4,
            include_depth_target=True, depth_target_size=(8, 8), depth_target_num_cams=4,
        )
        # 同 seed の noise で action_loss は同一、aux 分だけ差が付くはず
        torch.manual_seed(1)
        loss_no_aux = policy_no_aux.compute_loss(batch)
        torch.manual_seed(1)
        loss_with_aux = policy_with_aux.compute_loss(batch)
        # aux 込みの方が大きい (aux_weight=1、L1 loss は非負)
        assert loss_with_aux.item() > loss_no_aux.item()

    def test_aux_head_without_target_skips_aux(self, stub_vit) -> None:
        """aux_head 有だが batch に depth_target 無し → aux loss skip、action_loss のみ。"""
        aux_head = _make_depth_aux_head(N_cams=4, pool_size=4, output_size=8)
        policy = _make_small_policy(stub_vit, aux_head=aux_head, aux_weight=1.0)
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=4)  # no depth_target
        loss = policy.compute_loss(batch)
        assert loss.item() >= 0  # 正常完了、raise しない

    def test_aux_weight_zero_no_effect(self, stub_vit) -> None:
        """aux_weight=0 で aux head + target 有でも loss 影響ゼロ。"""
        aux_head = _make_depth_aux_head(N_cams=4, pool_size=4, output_size=8)
        policy_zero = _make_small_policy(stub_vit, aux_head=aux_head, aux_weight=0.0)
        policy_no_aux = _make_small_policy(stub_vit)
        batch = make_batch(
            B=2, N_cams=4, chunk_len=8, action_dim=4,
            include_depth_target=True, depth_target_size=(8, 8), depth_target_num_cams=4,
        )
        torch.manual_seed(1)
        loss_zero = policy_zero.compute_loss(batch)
        torch.manual_seed(1)
        loss_no_aux = policy_no_aux.compute_loss(batch)
        assert loss_zero.item() == pytest.approx(loss_no_aux.item(), abs=1e-6)

    def test_aux_head_gradient_flows(self, stub_vit) -> None:
        """aux_head の param に grad が populated される (backbone → aux → loss)。"""
        aux_head = _make_depth_aux_head(N_cams=4, pool_size=4, output_size=8)
        policy = _make_small_policy(stub_vit, aux_head=aux_head, aux_weight=1.0)
        batch = make_batch(
            B=2, N_cams=4, chunk_len=8, action_dim=4,
            include_depth_target=True, depth_target_size=(8, 8), depth_target_num_cams=4,
        )
        loss = policy.compute_loss(batch)
        loss.backward()
        for name, p in aux_head.named_parameters():
            assert p.grad is not None, f"aux_head.{name} no grad"
            assert p.grad.abs().sum().item() > 0, f"aux_head.{name} grad all zero"

    def test_depth_target_mask_masks_invalid_pixels(self, stub_vit) -> None:
        """depth_target_mask 提供時、mask=False pixel は loss から除外。"""
        aux_head = _make_depth_aux_head(N_cams=4, pool_size=4, output_size=8)
        policy = _make_small_policy(stub_vit, aux_head=aux_head, aux_weight=1.0)
        batch = make_batch(
            B=2, N_cams=4, chunk_len=8, action_dim=4,
            include_depth_target=True, depth_target_size=(8, 8), depth_target_num_cams=4,
        )
        # 全 True mask + 全 False mask で loss 差 (前者は普通 L1、後者は 0-count guard で valid=1 fallback で結果ゼロ近い)
        batch_all = {**batch, "depth_target_mask": torch.ones(2, 4, 1, 8, 8, dtype=torch.bool)}
        batch_none = {**batch, "depth_target_mask": torch.zeros(2, 4, 1, 8, 8, dtype=torch.bool)}
        torch.manual_seed(1)
        loss_all = policy.compute_loss(batch_all)
        torch.manual_seed(1)
        loss_none = policy.compute_loss(batch_none)
        # 全 False は valid_count 0→ clamp(min=1) で分母 1、numerator は 0 → aux 部分 0 に近い
        assert loss_none.item() < loss_all.item()


class TestTokenCount:
    def test_encoded_context_matches_expected_token_count(self, stub_vit) -> None:
        """Vision (N_cams * pool_size**2) + Delta (同) + OBB (N_cams * top_K) + State (1) + Skill (1)。"""
        N_cams, pool_size, top_K = 4, 4, 4
        policy = _make_small_policy(stub_vit, chunk_len=8, action_dim=4)
        policy.eval()
        batch = make_batch(
            B=2, N_cams=N_cams, top_K=top_K, chunk_len=8, action_dim=4, include_action=False
        )
        context, mask, _vis_tokens = policy._encode(batch)
        expected_N = N_cams * pool_size**2 * 2 + N_cams * top_K + 1 + 1  # 32+32+16+1+1=82
        assert context.shape == (2, expected_N, 64)
        assert mask.shape == (2, expected_N)


def _set_fk_stats(policy: RamenOriPolicy, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    policy.set_fk_stats(torch.randn(27, generator=g) * 0.1, torch.rand(27, generator=g) * 0.1 + 0.02)


class TestFKLoss:
    """Issue #141 RO-16 / RO-12: FK の loss (左側・右側・両側の 3 項、毎 step 自動の重み、上限)。"""

    def _make_fk_policy(self, stub_vit, fk_weight_max: float = 100.0, normalization: bool = False):
        policy = _make_small_policy(
            stub_vit, chunk_len=4, action_dim=16, N_cams=4,
            l4_enabled=True, fk_weight_max=fk_weight_max, normalization=normalization,
        )
        _set_fk_stats(policy)
        return policy

    @staticmethod
    def _batch():
        batch = make_batch(B=2, N_cams=4, chunk_len=4, action_dim=16, include_action_waist_teacher=True)
        batch["action"] = batch["action"] * 0.3        # 関節角らしい大きさ (rad)
        batch["action_waist_teacher"] = batch["action_waist_teacher"] * 0.05
        return batch

    def test_disabled_by_default(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit, chunk_len=4, action_dim=16)
        assert policy.l4_enabled is False
        assert policy.fk is None and policy.fk_normalizer is None
        parts = policy.compute_loss_parts(make_batch(B=2, N_cams=4, chunk_len=4, action_dim=16))
        assert set(parts) == {"loss", "bc", "loss_action"}

    def test_enabled_auto_loads_fk_from_default_urdf(self, stub_vit) -> None:
        from model.ramen_ori.fk import G1WristFKTorch

        policy = _make_small_policy(stub_vit, chunk_len=4, action_dim=16, l4_enabled=True)
        assert isinstance(policy.fk, G1WristFKTorch)
        assert policy.fk_normalizer is not None

    def test_parts_and_total(self, stub_vit) -> None:
        policy = self._make_fk_policy(stub_vit)
        parts = policy.compute_loss_parts(self._batch())
        assert set(parts) == {
            "loss", "bc", "fk_left", "fk_right", "fk_both", "w_left", "w_right", "w_both", "loss_action"
        }
        expected = parts["bc"] + sum(parts[f"w_{k}"] * parts[f"fk_{k}"] for k in ("left", "right", "both"))
        torch.testing.assert_close(parts["loss"].detach(), expected)

    def test_weights_keep_each_term_at_ratio_over_three_of_bc(self, stub_vit) -> None:
        """上限に当たらなければ、各項は重みを掛けて BC の 0.2 / 3 になる (3 項の合計で BC の 0.2)。"""
        policy = self._make_fk_policy(stub_vit, fk_weight_max=1e9)
        parts = policy.compute_loss_parts(self._batch())
        for k in ("left", "right", "both"):
            torch.testing.assert_close(parts[f"w_{k}"] * parts[f"fk_{k}"], parts["bc"] * 0.2 / 3)

    def test_weights_are_capped(self, stub_vit) -> None:
        policy = self._make_fk_policy(stub_vit, fk_weight_max=1e-6)
        parts = policy.compute_loss_parts(self._batch())
        for k in ("left", "right", "both"):
            assert parts[f"w_{k}"].item() == pytest.approx(1e-6)

    def test_weights_carry_no_gradient_and_fk_reaches_action_expert(self, stub_vit) -> None:
        policy = self._make_fk_policy(stub_vit)
        parts = policy.compute_loss_parts(self._batch())
        assert parts["loss"].requires_grad
        assert not any(parts[f"w_{k}"].requires_grad for k in ("left", "right", "both"))
        parts["loss"].backward()
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in policy.action_expert.parameters()
        )

    def test_row_mask_reaches_the_losses(self, stub_vit, monkeypatch) -> None:
        """埋め草の行 (action_is_pad) が BC と FK の 3 項に row_mask として渡る。"""
        policy = self._make_fk_policy(stub_vit)
        batch = self._batch()
        batch["action_is_pad"][:, 2:] = True
        seen = {}
        orig_bc = policy.action_expert.compute_loss_with_prediction
        orig_fk = policy._fk_loss_terms

        def _bc(target, context, mask, row_mask=None):
            seen["bc"] = row_mask
            return orig_bc(target, context, mask, row_mask=row_mask)

        def _fk(pred, teacher, waist, row_mask):
            seen["fk"] = row_mask
            return orig_fk(pred, teacher, waist, row_mask)

        monkeypatch.setattr(policy.action_expert, "compute_loss_with_prediction", _bc)
        monkeypatch.setattr(policy, "_fk_loss_terms", _fk)
        policy.compute_loss_parts(batch)
        for k in ("bc", "fk"):
            torch.testing.assert_close(seen[k], ~batch["action_is_pad"])

    def test_fk_terms_ignore_pad_rows(self, stub_vit) -> None:
        """FK の 3 項は行ごとの誤差の平均で、埋め草の行の値 (予測・教師・腰) を変えても変わらない。"""
        policy = self._make_fk_policy(stub_vit)
        g = torch.Generator().manual_seed(1)
        pred = torch.randn(2, 4, 16, generator=g) * 0.3
        teacher = torch.randn(2, 4, 16, generator=g) * 0.3
        waist = torch.randn(2, 4, 3, generator=g) * 0.05
        row_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        a = policy._fk_loss_terms(pred, teacher, waist, row_mask)
        pred2, teacher2, waist2 = pred.clone(), teacher.clone(), waist.clone()
        pred2[~row_mask] += 1.0
        teacher2[~row_mask] -= 1.0
        waist2[~row_mask] += 0.2
        b = policy._fk_loss_terms(pred2, teacher2, waist2, row_mask)
        for k in ("left", "right", "both"):
            torch.testing.assert_close(a[k], b[k])

    def test_left_and_right_terms_are_separated(self, stub_vit) -> None:
        """右腕だけずらすと、左側の項は 0 のまま、右側と両側の項が増える (腰は教師のものを共有)。"""
        policy = self._make_fk_policy(stub_vit)
        g = torch.Generator().manual_seed(2)
        teacher = torch.randn(2, 4, 16, generator=g) * 0.3
        waist = torch.randn(2, 4, 3, generator=g) * 0.05
        pred = teacher.clone()
        pred[..., 7:14] += 0.2       # 右腕 7 関節
        pred[..., 14:16] += 1.0      # hand は FK に入らない
        terms = policy._fk_loss_terms(pred, teacher, waist, torch.ones(2, 4, dtype=torch.bool))
        assert terms["left"].item() == pytest.approx(0.0, abs=1e-8)
        assert terms["right"].item() > 0.0 and terms["both"].item() > 0.0

    def test_missing_fk_stats_raises(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit, chunk_len=4, action_dim=16, l4_enabled=True)
        with pytest.raises(RuntimeError, match="before stats"):
            policy.compute_loss_parts(self._batch())

    def test_set_fk_stats_without_l4_raises(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit, chunk_len=4, action_dim=16)
        with pytest.raises(RuntimeError, match="l4_enabled=False"):
            _set_fk_stats(policy)

    def test_meanflow_with_l4_raises_at_construction(self, stub_vit) -> None:
        """MeanFlow は 1-step 復元の予測を返さないので、学習の最初の step で落ちる前に止める (RO-7)。"""
        from model.ramen_ori.action_expert import MeanFlow

        with pytest.raises(ValueError, match="compute_loss_with_prediction"):
            RamenOriPolicy(
                vision=torch.nn.Identity(), temporal=torch.nn.Identity(), obb=torch.nn.Identity(),
                state=StateEncoder(state_dim=71, d_model=64), skill=torch.nn.Identity(),
                fusion=torch.nn.Identity(),
                action_expert=MeanFlow(chunk_len=4, action_dim=16, embed_dim=64, depth=1, num_heads=4, context_dim=64),
                fk=torch.nn.Identity(), l4_enabled=True, chunk_len=4, action_dim=16,
            )


class TestRelativeAction:
    """Issue #129 Phase F: use_relative_action=True (Issue #141 で保留、BC のみ。FK の loss とは併用しない)。"""

    def _submodules(self, stub_vit):
        from model.ramen_ori.fusion import FusionTransformer
        from model.ramen_ori.obb import ObbTokenizer
        from model.ramen_ori.skill import SkillEmbedding

        d_model, num_cams = 64, 6
        return dict(
            vision=VisionEncoder(backbone=stub_vit, embed_dim=768, d_model=d_model, pool_size=4, freeze=True),
            temporal=TemporalEncoder(num_cams=num_cams, embed_dim=64, depth=2, num_heads=4, pool_size=4, cam_embed_dim=16, d_model=d_model),
            obb=ObbTokenizer(num_classes=7, num_cams=num_cams, class_embed_dim=32, cam_embed_dim=16, d_model=d_model, drop_rate=0.05),
            state=StateEncoder(state_dim=71, d_model=d_model),
            skill=SkillEmbedding(num_skills=6, d_model=d_model),
            fusion=FusionTransformer(d_model=d_model, num_layers=2, num_heads=4, mlp_dim=128),
            action_expert=FlowMatchingDiT(chunk_len=4, action_dim=16, embed_dim=64, depth=2, num_heads=4, mlp_ratio=4.0, context_dim=d_model),
            d_model=d_model, num_cams=num_cams, chunk_len=4, action_dim=16,
        )

    def test_relative_action_requires_stats(self, stub_vit) -> None:
        with pytest.raises(ValueError, match="relative_stats"):
            RamenOriPolicy(**self._submodules(stub_vit), use_relative_action=True, relative_stats=None)

    def test_relative_action_registers_stats_buffers(self, stub_vit) -> None:
        stats = {"mean": torch.arange(14, dtype=torch.float32), "std": torch.ones(14) * 0.05}
        policy = RamenOriPolicy(**self._submodules(stub_vit), use_relative_action=True, relative_stats=stats)
        assert policy.use_relative_action is True
        torch.testing.assert_close(policy._relative_arms_mean, stats["mean"])
        # std は clamp(min=1e-6) が入るが 0.05 は無影響
        torch.testing.assert_close(policy._relative_arms_std, stats["std"])

    def test_relative_bc_only_is_finite(self, stub_vit) -> None:
        stats = {"mean": torch.zeros(14), "std": torch.ones(14) * 0.01}
        policy = RamenOriPolicy(**self._submodules(stub_vit), use_relative_action=True, relative_stats=stats)
        loss = policy.compute_loss(make_batch(B=2, N_cams=4, chunk_len=4, action_dim=16))
        assert torch.isfinite(loss)

    def test_relative_with_l4_raises(self, stub_vit) -> None:
        stats = {"mean": torch.zeros(14), "std": torch.ones(14) * 0.01}
        with pytest.raises(ValueError, match="abs action only"):
            RamenOriPolicy(
                **self._submodules(stub_vit), use_relative_action=True, relative_stats=stats,
                fk=torch.nn.Identity(), l4_enabled=True,
            )


class TestPredictActionRtcGuidance:
    """RTC (Issue #137) の kwargs が predict_action を通ること + 非対応 expert の拒否。"""

    def test_defaults_take_the_unchanged_path(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit)
        batch = make_batch(B=1, N_cams=4, chunk_len=8, action_dim=4)
        torch.manual_seed(3)
        baseline = policy.predict_action(batch)
        torch.manual_seed(3)
        again = policy.predict_action(batch, prefix=None, velocity_strength=None)
        assert torch.equal(baseline, again)

    def test_prefix_and_strength_reach_the_sampler(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit)
        batch = make_batch(B=1, N_cams=4, chunk_len=8, action_dim=4)
        prefix = torch.full((3, 4), 0.75)
        strength = torch.ones(8)
        strength[:3] = 0.0
        torch.manual_seed(5)
        out = policy.predict_action(
            batch, prefix=prefix, velocity_strength=strength
        )
        # 凍結区間は prefix のまま届く = kwargs が sampler まで通っている
        assert torch.equal(out[0, :3], prefix)

    def test_expert_without_rtc_support_is_rejected(self, stub_vit) -> None:
        # MeanFlow のような 1-step sampler は ramp を刻めない。silent に無視すると
        # 「RTC を有効にしたのに効いていない」事故になるので落とす。
        policy = _make_small_policy(stub_vit)
        batch = make_batch(B=1, N_cams=4, chunk_len=8, action_dim=4)
        policy.action_expert.SUPPORTS_RTC_GUIDANCE = False
        with pytest.raises(ValueError, match="does not support RTC guidance"):
            policy.predict_action(batch, velocity_strength=torch.ones(8))

    def test_meanflow_does_not_declare_rtc_support(self) -> None:
        from model.ramen_ori.action_expert import MeanFlow

        assert getattr(MeanFlow, "SUPPORTS_RTC_GUIDANCE", False) is False


class TestNormalization:
    """Issue #141 RO-4 / RO-6: model の中の正規化 (入出力は元の単位、中で z-score)。"""

    def test_default_keeps_phase_k_structure(self, stub_vit) -> None:
        """既定 (normalization=False、input_norm=True) は Phase K と同じ state_dict の key。
        推論は base.yaml から組み立てて Phase K の ckpt を読むので、既定の構造を変えない。"""
        policy = _make_small_policy(stub_vit)
        assert policy.state_normalizer is None
        assert policy.action_normalizer is None
        keys = set(policy.state_dict())
        assert not any("normalizer" in k for k in keys)
        assert {"state.input_ln.weight", "state.input_ln.bias"} <= keys

    def test_state_dropout_requires_normalization(self, stub_vit) -> None:
        """Issue #141 Phase 6: 隠した値の 0 を平均として使うので、正規化なしでは組み立てない。"""
        policy = _make_small_policy(stub_vit, normalization=True, state_input_norm=False)
        state = StateEncoder(state_dim=71, d_model=64, input_norm=False, dropout={"joint": 0.1})
        with pytest.raises(ValueError, match="正規化"):
            RamenOriPolicy(
                vision=policy.vision, temporal=None, obb=policy.obb, state=state, skill=policy.skill,
                fusion=policy.fusion, action_expert=policy.action_expert,
                d_model=64, num_cams=6, chunk_len=8, action_dim=4, normalization=False,
            )

    @pytest.mark.parametrize("normalization, state_clip, expected_max", [(True, 10.0, 10.0), (True, None, 100.0), (False, 10.0, None)])
    def test_normalized_state_is_clipped(self, stub_vit, normalization, state_clip, expected_max) -> None:
        """正規化した後の state を ±state_clip に切り詰める (正規化のときだけ、Issue #141 の c32 の発散の対策)。"""
        policy = _make_small_policy(
            stub_vit, normalization=normalization, state_input_norm=False, state_clip=state_clip, use_temporal=False
        )
        if normalization:
            policy.set_normalization_stats(torch.zeros(71), torch.ones(71), torch.zeros(4), torch.ones(4))
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=4)
        batch.pop("images_prev")
        batch["state"] = torch.zeros(2, 71)
        batch["state"][:, 40] = 100.0
        captured = {}
        handle = policy.state.fc1.register_forward_hook(lambda m, inp, out: captured.update(x=inp[0]))
        policy._encode(batch)
        handle.remove()
        if expected_max is None:
            assert policy.state_clip is None   # 正規化なしでは切り詰めない (生の値に掛けない)
        else:
            assert float(captured["x"].abs().max()) == pytest.approx(expected_max)

    def test_hand_change_does_not_move_arm_inputs(self, stub_vit) -> None:
        """次元ごとの正規化 + LN なしでは、hand だけを変えても腕の関節角の入力は同じ。
        sample ごとの LN (Phase K) では腕の入力も動く (RO-6)。"""
        s1 = torch.randn(1, 71)
        s2 = s1.clone()
        s2[:, 17:19] += 4.0   # joint 末尾の hand
        s2[:, 57:59] += 4.0   # hand_state

        policy = _make_small_policy(stub_vit, normalization=True, state_input_norm=False)
        _set_random_stats(policy, action_dim=4)
        x1 = policy.state.input_ln(policy.state_normalizer.normalize(s1))
        x2 = policy.state.input_ln(policy.state_normalizer.normalize(s2))
        torch.testing.assert_close(x1[:, 3:17], x2[:, 3:17])

        ln = torch.nn.LayerNorm(71)
        assert not torch.allclose(ln(s1)[:, 3:17], ln(s2)[:, 3:17])

    def test_compute_loss_uses_normalized_target(self, stub_vit, monkeypatch) -> None:
        policy = _make_small_policy(stub_vit, normalization=True)
        _set_random_stats(policy, action_dim=4)
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=4)
        seen = {}

        def _capture(target, context, mask, row_mask=None):
            seen["target"] = target
            return target.sum() * 0.0

        monkeypatch.setattr(policy.action_expert, "compute_loss", _capture)
        policy.compute_loss(batch)
        n = policy.action_normalizer
        torch.testing.assert_close(seen["target"], (batch["action"] - n.mean) / n.std)

    def test_state_is_normalized_before_state_encoder(self, stub_vit, monkeypatch) -> None:
        policy = _make_small_policy(stub_vit, normalization=True, state_input_norm=False)
        _set_random_stats(policy, action_dim=4)
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=4)
        seen = {}
        orig = policy.state.forward

        def _capture(state):
            seen["state"] = state
            return orig(state)

        monkeypatch.setattr(policy.state, "forward", _capture)
        policy.predict_action(batch)
        n = policy.state_normalizer
        torch.testing.assert_close(seen["state"], (batch["state"] - n.mean) / n.std)

    def test_predict_action_returns_original_units(self, stub_vit, monkeypatch) -> None:
        policy = _make_small_policy(stub_vit, normalization=True)
        _set_random_stats(policy, action_dim=4)
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=4)
        monkeypatch.setattr(
            policy.action_expert, "sample",
            lambda context, mask, n_steps, **kw: torch.zeros(2, 8, 4),
        )
        out = policy.predict_action(batch)
        # 正規化空間の 0 = 元の単位の平均
        torch.testing.assert_close(out, policy.action_normalizer.mean.expand(2, 8, 4))

    def test_rtc_prefix_is_given_in_original_units(self, stub_vit) -> None:
        """prefix は元の単位で渡し、凍結区間は元の単位のまま返る (中で正規化 → 逆変換)。"""
        policy = _make_small_policy(stub_vit, normalization=True)
        _set_random_stats(policy, action_dim=4)
        batch = make_batch(B=1, N_cams=4, chunk_len=8, action_dim=4)
        prefix = torch.full((3, 4), 0.75)
        strength = torch.ones(8)
        strength[:3] = 0.0
        out = policy.predict_action(batch, prefix=prefix, velocity_strength=strength)
        torch.testing.assert_close(out[0, :3], prefix)

    def test_l4_gets_denormalized_prediction(self, stub_vit, monkeypatch) -> None:
        """FK は関節角 (元の単位) で計算する: 予測は逆変換、教師は元の単位のまま渡る。"""
        policy = _make_small_policy(
            stub_vit, action_dim=16, normalization=True, l4_enabled=True, fk=torch.nn.Identity()
        )
        _set_random_stats(policy, action_dim=16)
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=16, include_action_waist_teacher=True)
        x1_hat_norm = torch.randn(2, 8, 16)
        monkeypatch.setattr(
            policy.action_expert, "compute_loss_with_prediction",
            lambda target, context, mask, row_mask=None: (target.sum() * 0.0 + 1.0, x1_hat_norm),
        )
        seen = {}

        def _capture(pred_arms_hand, teacher_arms_hand, teacher_waist, row_mask):
            seen["pred"] = pred_arms_hand
            seen["teacher"] = teacher_arms_hand
            one = pred_arms_hand.sum() * 0.0 + 1.0
            return {"left": one, "right": one, "both": one}

        monkeypatch.setattr(policy, "_fk_loss_terms", _capture)
        policy.compute_loss(batch)
        torch.testing.assert_close(seen["pred"], policy.action_normalizer.denormalize(x1_hat_norm))
        torch.testing.assert_close(seen["teacher"], batch["action"])

    def test_stats_survive_state_dict_roundtrip(self, stub_vit) -> None:
        """学習が入れた統計が ckpt (model_state_dict) 経由で同じ値になる。"""
        src = _make_small_policy(stub_vit, normalization=True)
        _set_random_stats(src, action_dim=4, seed=7)
        dst = _make_small_policy(stub_vit, normalization=True)
        dst.load_state_dict(src.state_dict())
        for a, b in ((src.state_normalizer, dst.state_normalizer), (src.action_normalizer, dst.action_normalizer)):
            torch.testing.assert_close(a.mean, b.mean)
            torch.testing.assert_close(a.std, b.std)
            assert bool(b.ready)

    def test_use_without_stats_raises(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit, normalization=True)
        batch = make_batch(B=1, N_cams=4, chunk_len=8, action_dim=4)
        with pytest.raises(RuntimeError, match="before stats"):
            policy.predict_action(batch)

    def test_set_stats_without_normalization_raises(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit)
        with pytest.raises(RuntimeError, match="normalization=False"):
            _set_random_stats(policy, action_dim=4)

    def test_relative_action_with_normalization_raises(self, stub_vit) -> None:
        with pytest.raises(ValueError, match="abs action only"):
            RamenOriPolicy(
                vision=torch.nn.Identity(), temporal=torch.nn.Identity(), obb=torch.nn.Identity(),
                state=StateEncoder(state_dim=71, d_model=64), skill=torch.nn.Identity(),
                fusion=torch.nn.Identity(), action_expert=torch.nn.Identity(),
                use_relative_action=True,
                relative_stats={"mean": torch.zeros(14), "std": torch.ones(14)},
                normalization=True,
            )


class TestNoTemporal:
    """Issue #141 RO-11: 画像差分 (TemporalEncoder) なしの model。"""

    def test_token_count_drops_delta_tokens(self, stub_vit) -> None:
        batch = make_batch(B=2, N_cams=3, chunk_len=8, action_dim=4)
        with_t = _make_small_policy(stub_vit)
        without_t = _make_small_policy(stub_vit, use_temporal=False)
        ctx_with, _, _ = with_t._encode(batch)
        del batch["images_prev"]
        ctx_without, mask_without, _ = without_t._encode(batch)
        # 差分の token は 1 カメラ pool_size² = 16 個 × 3 カメラ
        assert ctx_with.shape[1] - ctx_without.shape[1] == 3 * 16
        assert mask_without.shape == ctx_without.shape[:2]

    def test_loss_and_predict_without_images_prev(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit, use_temporal=False)
        batch = make_batch(B=2, N_cams=3, chunk_len=8, action_dim=4)
        del batch["images_prev"]
        assert torch.isfinite(policy.compute_loss(batch))
        assert policy.predict_action(batch).shape == (2, 8, 4)
        assert not any(k.startswith("temporal.") for k in policy.state_dict())

    def test_chunk32(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit, chunk_len=32, use_temporal=False)
        batch = make_batch(B=2, N_cams=3, chunk_len=32, action_dim=4)
        batch["action_is_pad"][:, 20:] = True
        del batch["images_prev"]
        assert torch.isfinite(policy.compute_loss(batch))
        assert policy.predict_action(batch).shape == (2, 32, 4)


class _StubViTWithBlocks(torch.nn.Module):
    """partial-train の block 探索 (`backbone.blocks`) に応える stub (forward は StubViT と同じ)。"""

    def __init__(self, n_blocks: int = 12) -> None:
        super().__init__()
        from model.ramen_ori.tests.conftest import StubViT

        self.inner = StubViT(patch_size=16, embed_dim=768)
        self.patch_size = 16
        self.embed_dim = 768
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(n_blocks)])

    def forward(self, x, is_training: bool = False):
        return self.inner(x, is_training=is_training)


class TestPartialTrainOnNewModel:
    """Issue #141 RO-13: 新しい model (差分なし・正規化・FK の loss) でも partial-train が param を正しく分ける。"""

    def test_groups_cover_every_learnable_param_once(self) -> None:
        from omegaconf import OmegaConf

        from model.ramen_ori.optim_utils import make_optimizer

        policy = _make_small_policy(
            _StubViTWithBlocks(), action_dim=16, use_temporal=False, normalization=True,
            l4_enabled=True, freeze_vision=False,
        )
        cfg = OmegaConf.create({"lr": 1e-4, "partial_train": {
            "enabled": True, "last_n_blocks": 4, "llrd_decay": 0.75, "backbone_lr_ratio": 0.2}})
        opt = make_optimizer(policy, cfg)
        in_groups = [id(p) for g in opt.param_groups for p in g["params"]]
        learnable = [p for p in policy.parameters() if p.requires_grad]
        assert len(in_groups) == len(set(in_groups)) == len(learnable)
        assert set(in_groups) == {id(p) for p in learnable}
        blocks = policy.vision.backbone.blocks
        assert all(not p.requires_grad for b in blocks[:8] for p in b.parameters())
        assert all(p.requires_grad for b in blocks[8:] for p in b.parameters())
        # 正規化と FK の統計は buffer で、optimizer に入らない
        assert not any(n.startswith(("state_normalizer", "action_normalizer", "fk_normalizer"))
                       for n, _ in policy.named_parameters())


class TestCompileSubmodules:
    """Issue #141: 重い部分をその場で compile (key を変えない、loss の計算は外)。"""

    def test_compiles_heavy_parts_and_keeps_state_dict_keys(self, stub_vit) -> None:
        from model.ramen_ori.train import _compile_submodules

        policy = _make_small_policy(stub_vit, action_dim=16, use_temporal=False, l4_enabled=True)
        keys = list(policy.state_dict())
        names = _compile_submodules(policy, "default")
        assert names == ["vision", "fusion", "action_expert", "fk.forward_detailed"]
        assert list(policy.state_dict()) == keys
        assert not any("_orig_mod" in k for k in keys)
        for name in ("vision", "fusion", "action_expert"):
            assert getattr(policy, name)._compiled_call_impl is not None


class TestVisionBackboneMode:
    """Issue #141: backbone は学習中も常に eval mode (LingBot の RoPE の乱れを入れない)。一部解放では勾配が流れる。"""

    def test_backbone_stays_eval_in_train_for_both_freeze_settings(self) -> None:
        for freeze in (True, False):
            policy = _make_small_policy(_StubViTWithBlocks(), use_temporal=False, freeze_vision=freeze)
            policy.train()
            assert policy.vision.training
            assert not policy.vision.backbone.training
            assert all(not m.training for m in policy.vision.backbone.modules())

    def test_partial_unfreeze_gets_gradients(self) -> None:
        """一部解放 (freeze=False + 解放した param): eval mode のままでも勾配が流れる。"""
        backbone = _StubViTWithBlocks()
        for p in backbone.parameters():
            p.requires_grad_(False)
        policy = _make_small_policy(backbone, use_temporal=False, freeze_vision=False)
        tg = policy.vision.backbone.inner.token_generator     # forward で使われる param を「解放した block」に見立てる
        for p in tg.parameters():
            p.requires_grad_(True)
        # 学習前の DiT は出力層などが 0 初期化で、出力が文脈に依存しない (勾配も 0) ので乱数を入れる
        with torch.no_grad():
            for p in policy.action_expert.parameters():
                if p.abs().max() == 0:
                    p.normal_(0, 0.02)
        batch = make_batch(B=2, N_cams=3, chunk_len=8, action_dim=4)
        del batch["images_prev"]
        policy.compute_loss(batch).backward()
        assert tg.weight.grad is not None and tg.weight.grad.abs().sum() > 0

    def test_frozen_backbone_has_no_gradients(self) -> None:
        policy = _make_small_policy(_StubViTWithBlocks(), use_temporal=False, freeze_vision=True)
        batch = make_batch(B=2, N_cams=3, chunk_len=8, action_dim=4)
        del batch["images_prev"]
        policy.compute_loss(batch).backward()
        assert all(p.grad is None for p in policy.vision.backbone.parameters())


class TestMemory:
    """Issue #141 Phase 7: memory の token (state の token の次に 1 個)。"""

    @staticmethod
    def _policy(stub_vit, memory: bool):
        from model.ramen_ori.memory import MemoryEncoder

        enc = None
        if memory:
            enc = MemoryEncoder(d_model=64)
            enc.set_stats(-torch.ones(51), torch.ones(51), torch.zeros(51), torch.ones(51))
        return _make_small_policy(stub_vit, use_temporal=False, memory=enc)

    @staticmethod
    def _batch() -> dict:
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=4)
        batch.pop("images_prev")
        batch["memory"] = torch.randn(2, 51) * 0.5
        return batch

    def test_adds_one_token_that_depends_on_memory(self, stub_vit) -> None:
        with_memory = self._policy(stub_vit, memory=True).eval()
        without = self._policy(stub_vit, memory=False).eval()
        batch = self._batch()
        ctx, mask, _ = with_memory._encode(batch)
        _, mask_without, _ = without._encode(batch)
        assert mask.shape[1] == mask_without.shape[1] + 1
        assert bool(mask[:, -2].all())   # memory の token (skill の token の直前) は常に有効
        moved = {**batch, "memory": batch["memory"] + 0.3}
        assert not torch.allclose(with_memory._encode(moved)[0], ctx)

    def test_compute_loss_runs(self, stub_vit) -> None:
        policy = self._policy(stub_vit, memory=True)
        loss = policy.compute_loss(self._batch())
        assert torch.isfinite(loss)

    def test_default_has_no_memory(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit)
        assert policy.memory is None
        assert not any(k.startswith("memory.") for k in policy.state_dict())
        with pytest.raises(RuntimeError, match="without memory"):
            policy.set_memory_stats(None, None, None, None)
