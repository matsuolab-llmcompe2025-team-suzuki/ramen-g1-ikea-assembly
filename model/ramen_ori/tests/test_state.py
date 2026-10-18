"""StateEncoder の unit test (Issue #115、Phase 1 M2)。

- init: layer 構造 (LN → Linear → GELU → Linear)
- forward: (B, state_dim) → (B, 1, d_model)
- state_dim / d_model / hidden_dim の カスタマイズ切替
- 入口 LN が scale ばらばら input を正規化 (batch 内 zero-mean、unit-var)
- gradient flow: backward で LN と両 Linear の grad が populate される
"""

from __future__ import annotations

import pytest
import torch

from model.ramen_ori.state import StateEncoder


class TestInit:
    def test_default_hidden_dim_equals_d_model(self) -> None:
        module = StateEncoder(state_dim=71, d_model=512)
        assert module.hidden_dim == 512
        assert module.fc1.out_features == 512
        assert module.fc2.in_features == 512

    def test_custom_hidden_dim(self) -> None:
        module = StateEncoder(state_dim=71, d_model=512, hidden_dim=256)
        assert module.hidden_dim == 256
        assert module.fc1.out_features == 256
        assert module.fc2.in_features == 256

    def test_layer_types_match_design(self) -> None:
        module = StateEncoder(state_dim=71, d_model=512)
        assert isinstance(module.input_ln, torch.nn.LayerNorm)
        assert isinstance(module.fc1, torch.nn.Linear)
        assert isinstance(module.act, torch.nn.GELU)
        assert isinstance(module.fc2, torch.nn.Linear)


class TestForward:
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_output_shape_is_batch_1_d_model(self, batch_size: int) -> None:
        module = StateEncoder(state_dim=71, d_model=512)
        state = torch.randn(batch_size, 71)
        out = module(state)
        assert out.shape == (batch_size, 1, 512)

    @pytest.mark.parametrize("state_dim,d_model", [(71, 512), (23, 256), (100, 1024)])
    def test_state_and_dmodel_switchable(self, state_dim: int, d_model: int) -> None:
        module = StateEncoder(state_dim=state_dim, d_model=d_model)
        state = torch.randn(4, state_dim)
        out = module(state)
        assert out.shape == (4, 1, d_model)


class TestInputLayerNorm:
    def test_ln_normalizes_mixed_scale_input(self) -> None:
        """入口 LN が per-sample で zero-mean, unit-var に正規化することを確認。

        joint (rad, ±3)、hand (0-1)、EE (m + rad 混在) の scale mix を模擬した
        input を作り、input_ln 直後の tensor が feature 軸で zero-mean unit-var
        になることを検証する (LN の定義)。
        """
        module = StateEncoder(state_dim=71, d_model=512)
        # scale mix: 0..19 = joint rad ±3、19..38 = tracking_err ±1、38..57 = vel ±5、
        # 57..59 = hand 0..1、59..71 = EE (m + rad 混在)
        state = torch.zeros(2, 71)
        state[:, 0:19] = torch.randn(2, 19) * 3.0
        state[:, 19:38] = torch.randn(2, 19) * 1.0
        state[:, 38:57] = torch.randn(2, 19) * 5.0
        state[:, 57:59] = torch.rand(2, 2)
        state[:, 59:71] = torch.randn(2, 12) * 0.5

        normalized = module.input_ln(state)
        # per-sample の mean/var 検証 (LN は feature 軸 = -1 で正規化)
        mean = normalized.mean(dim=-1)
        var = normalized.var(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


class TestGradient:
    def test_backward_populates_grad_for_all_params(self) -> None:
        module = StateEncoder(state_dim=71, d_model=32)
        state = torch.randn(4, 71)
        out = module(state)
        loss = out.sum()
        loss.backward()
        for name, param in module.named_parameters():
            assert param.grad is not None, f"{name} has no grad"
            assert param.grad.abs().sum().item() > 0, f"{name} grad is all zero"


class TestDropout:
    """Issue #141 Phase 6: 学習時に state を group ごとに隠し、隠した印を足す。"""

    PROBS = {"joint": 0.1, "hand": 0.1, "tracking_err": 0.3, "velocity": 0.3, "ee_pose": 0.3}

    @staticmethod
    def _fc1_input(enc: StateEncoder, state: torch.Tensor) -> torch.Tensor:
        captured = {}
        handle = enc.fc1.register_forward_hook(lambda m, inp, out: captured.update(x=inp[0]))  # None を返す (出力を置き換えない)
        enc(state)
        handle.remove()
        return captured["x"]

    def test_eval_does_not_hide_and_flags_are_zero(self) -> None:
        enc = StateEncoder(state_dim=71, d_model=32, input_norm=False, dropout=self.PROBS).eval()
        state = torch.randn(8, 71)
        x = self._fc1_input(enc, state)
        assert x.shape == (8, 76)
        torch.testing.assert_close(x[:, :71], state)
        assert torch.all(x[:, 71:] == 0)
        torch.testing.assert_close(enc(state), enc(state))

    def test_group_with_prob_one_is_zeroed_and_flagged(self) -> None:
        probs = {**{g: 0.0 for g in self.PROBS}, "hand": 1.0}
        enc = StateEncoder(state_dim=71, d_model=32, input_norm=False, dropout=probs).train()
        state = torch.randn(16, 71) + 5.0
        x = self._fc1_input(enc, state)
        hand_dims = [17, 18, 57, 58]
        assert torch.all(x[:, hand_dims] == 0)
        other = [d for d in range(71) if d not in hand_dims]
        torch.testing.assert_close(x[:, other], state[:, other])
        # 印の並びは STATE71_DROPOUT_GROUPS の並び (joint, hand, tracking_err, velocity, ee_pose)
        assert enc.dropout_groups == ("joint", "hand", "tracking_err", "velocity", "ee_pose")
        torch.testing.assert_close(x[:, 71:], torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]).expand(16, 5))

    def test_hide_rate_per_group(self) -> None:
        torch.manual_seed(0)
        enc = StateEncoder(state_dim=71, d_model=32, input_norm=False, dropout=self.PROBS).train()
        x = self._fc1_input(enc, torch.ones(40000, 71))
        rates = x[:, 71:].mean(dim=0)
        torch.testing.assert_close(rates, torch.tensor(list(self.PROBS.values())), atol=0.01, rtol=0)
        # 印が立った group の次元だけが 0
        assert torch.equal(x[:, 17] == 0, x[:, 72] == 1)
        assert torch.equal(x[:, 20] == 0, x[:, 73] == 1)

    def test_no_dropout_keeps_phase_k_shape(self) -> None:
        enc = StateEncoder(state_dim=71, d_model=32)
        assert enc.dropout_groups == ()
        assert enc.fc1.in_features == 71
        assert not any("dropout" in k for k in enc.state_dict())   # 印の mask は ckpt に入れない

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"dropout": {"hands": 0.1}}, "group"),
            ({"dropout": {"joint": 0.1}, "state_dim": 73}, "71D"),
            ({"dropout": {"joint": 0.1}, "input_norm": True}, "LayerNorm"),
            ({"dropout": {"joint": 1.5}}, "0〜1"),
        ],
    )
    def test_invalid_config_raises(self, kwargs, match) -> None:
        kwargs = {"state_dim": 71, "input_norm": False, **kwargs}
        with pytest.raises(ValueError, match=match):
            StateEncoder(d_model=32, **kwargs)
