"""区間の進みの補助 head (aux_progress.ProgressHead、Issue #183) の test。"""

from __future__ import annotations

import pytest
import torch

from model.ramen_ori.aux_progress import ProgressHead
from model.ramen_ori.tests.conftest import make_batch
from model.ramen_ori.tests.test_model import _make_small_policy


def _head() -> ProgressHead:
    return ProgressHead(d_model=64, hidden_dim=32, num_heads=4)


class TestHead:
    def test_output_is_one_value_per_sample_in_0_1(self) -> None:
        out = _head()(torch.randn(3, 50, 64) * 10)
        assert out.shape == (3,)
        assert torch.all((out > 0) & (out < 1))

    def test_loss_is_l1(self) -> None:
        pred = torch.tensor([0.2, 0.9])
        batch = {"progress_target": torch.tensor([0.5, 0.4])}
        assert ProgressHead.loss(pred, batch).item() == pytest.approx((0.3 + 0.5) / 2)

    def test_gradient_reaches_the_tokens(self) -> None:
        """勾配が入力の token (= VisionEncoder の投影) まで届く (画像の表現に効かせるのが狙い)。"""
        tokens = torch.randn(2, 16, 64, requires_grad=True)
        ProgressHead.loss(_head()(tokens), {"progress_target": torch.tensor([0.1, 0.8])}).backward()
        assert tokens.grad is not None and tokens.grad.abs().sum() > 0

    def test_d_model_must_split_into_heads(self) -> None:
        with pytest.raises(ValueError, match="割り切れ"):
            ProgressHead(d_model=64, num_heads=5)


class TestPolicyLoss:
    def _batch(self, with_target: bool) -> dict:
        batch = make_batch(B=2, N_cams=4, chunk_len=8, action_dim=4)
        if with_target:
            batch["progress_target"] = torch.tensor([0.25, 0.75])
        return batch

    def test_adds_weighted_l1_to_the_action_loss(self, stub_vit) -> None:
        """正解があれば loss = 行動の loss + aux_weight × 進みの L1、内訳に aux_progress が出る。"""
        policy = _make_small_policy(stub_vit, aux_head=_head(), aux_weight=0.5)
        batch = self._batch(with_target=True)
        torch.manual_seed(0)
        parts = policy.compute_loss_parts(batch)
        assert "aux_progress" in parts and "aux_depth" not in parts
        assert parts["loss"].item() == pytest.approx(parts["bc"].item() + 0.5 * parts["aux_progress"].item(), rel=1e-5)
        # 補助 loss を除いた値 (補助 head の無い run と比べる用)
        assert parts["loss_action"].item() == pytest.approx(parts["bc"].item(), rel=1e-6)
        # 補助 head の param と、画像の投影に勾配が届く
        parts["loss"].backward()
        assert all(p.grad is not None for p in policy.aux_head.parameters())
        assert policy.vision.projection.weight.grad is not None

    def test_without_target_is_the_action_loss_only(self, stub_vit) -> None:
        policy = _make_small_policy(stub_vit, aux_head=_head(), aux_weight=0.5)
        torch.manual_seed(0)
        parts = policy.compute_loss_parts(self._batch(with_target=False))
        assert "aux_progress" not in parts
        assert parts["loss"].item() == pytest.approx(parts["bc"].item())
        assert parts["loss_action"].item() == pytest.approx(parts["loss"].item())

    def test_predict_action_does_not_use_the_head(self, stub_vit) -> None:
        """推論 (predict_action) は補助 head を呼ばない。"""
        policy = _make_small_policy(stub_vit, aux_head=_head(), aux_weight=0.5)
        policy.eval()

        def _fail(*args, **kwargs):
            raise AssertionError("predict_action called the aux head")

        policy.aux_head.forward = _fail
        out = policy.predict_action(self._batch(with_target=False))
        assert out.shape == (2, 8, 4)
