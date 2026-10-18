"""vision_backbone loader/adapter の unit test (Issue #120 Alt-4)。

RADIO の実 DL 無しで、adapter output format と variant dispatch を test。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from model.ramen_ori.vision_backbone import (
    RadioAdapter,
    _extract_radio_embed_dim,
    _extract_radio_patch_size,
    _resolve_dtype,
    load_vision_backbone,
)


class StubRadioModel(nn.Module):
    """RADIO を模擬する stub。forward は (summary, features) tuple を返す。"""

    def __init__(self, embed_dim: int = 768, n_patches: int = 196, patch_size: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches = n_patches
        self.patch_size = patch_size
        # 最小限の学習可能 param (adapter test で backbone を freeze 確認)
        self.dummy = nn.Linear(3, embed_dim)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        avg = x.mean(dim=(2, 3))  # (B, 3)
        summary = self.dummy(avg)  # (B, D)
        features = summary.unsqueeze(1).expand(-1, self.n_patches, -1).contiguous()
        return summary, features


class TestRadioAdapter:
    def test_forward_returns_lingbot_compat_dict(self) -> None:
        stub = StubRadioModel(embed_dim=768, n_patches=196)
        adapter = RadioAdapter(radio_model=stub)
        x = torch.randn(2, 3, 224, 224)
        out = adapter(x)
        assert isinstance(out, dict)
        assert "x_norm_patchtokens" in out
        assert out["x_norm_patchtokens"].shape == (2, 196, 768)

    def test_is_training_kwarg_accepted(self) -> None:
        """LingBot API 互換で is_training kwarg を受けるが実装は無視。"""
        stub = StubRadioModel()
        adapter = RadioAdapter(radio_model=stub)
        # kwarg 受け入れて raise しない
        out = adapter(torch.randn(1, 3, 224, 224), is_training=True)
        assert "x_norm_patchtokens" in out

    def test_embed_dim_exposed(self) -> None:
        stub = StubRadioModel(embed_dim=1024)
        adapter = RadioAdapter(radio_model=stub)
        assert adapter.embed_dim == 1024

    def test_namedtuple_output_supported(self) -> None:
        """RADIO 公式は NamedTuple `RadioOutput(summary, features)` を返す (features 属性)。"""
        from collections import namedtuple
        RadioOutput = namedtuple("RadioOutput", ["summary", "features"])

        class NTStub(nn.Module):
            embed_dim = 768
            patch_size = 16
            def forward(self, x):
                B = x.shape[0]
                return RadioOutput(
                    summary=torch.zeros(B, 768),
                    features=torch.ones(B, 100, 768),
                )
        adapter = RadioAdapter(radio_model=NTStub())
        out = adapter(torch.randn(1, 3, 224, 224))
        assert out["x_norm_patchtokens"].shape == (1, 100, 768)

    def test_patch_size_exposed(self) -> None:
        """Issue #122 fix: RadioAdapter は VisionEncoder が触る patch_size 属性を expose する。

        Regression: vision.py:79 で `self.backbone.patch_size` を触る際に
        AttributeError で train が落ちていた (PR #121 未検証部分)。
        """
        stub = StubRadioModel(patch_size=16)
        adapter = RadioAdapter(radio_model=stub)
        assert adapter.patch_size == 16

    def test_patch_size_missing_raises(self) -> None:
        """radio_model が patch_size を expose しないと明示的 error (silent fallback しない)。"""
        class NoPatchSize(nn.Module):
            embed_dim = 768
        with pytest.raises(RuntimeError, match="cannot infer patch_size"):
            RadioAdapter(radio_model=NoPatchSize())


class TestExtractEmbedDim:
    def test_direct_embed_dim(self) -> None:
        stub = StubRadioModel(embed_dim=512)
        assert _extract_radio_embed_dim(stub) == 512

    def test_nested_model_embed_dim(self) -> None:
        class Nested(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = StubRadioModel(embed_dim=1024)
        assert _extract_radio_embed_dim(Nested()) == 1024

    def test_missing_raises(self) -> None:
        class Empty(nn.Module):
            pass
        with pytest.raises(RuntimeError, match="cannot infer embed_dim"):
            _extract_radio_embed_dim(Empty())


class TestExtractPatchSize:
    def test_direct_patch_size(self) -> None:
        stub = StubRadioModel(patch_size=16)
        assert _extract_radio_patch_size(stub) == 16

    def test_nested_model_patch_size(self) -> None:
        """公式 API の `model.model.patch_size` 経路も pick できる。"""
        class Nested(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = StubRadioModel(patch_size=14)
        assert _extract_radio_patch_size(Nested()) == 14

    def test_patch_generator_path(self) -> None:
        """`patch_generator.patch_size` 経路 (RADIO 別 build)。"""
        class WithGen(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_generator = nn.Module()
                self.patch_generator.patch_size = 32
        assert _extract_radio_patch_size(WithGen()) == 32

    def test_missing_raises(self) -> None:
        class Empty(nn.Module):
            pass
        with pytest.raises(RuntimeError, match="cannot infer patch_size"):
            _extract_radio_patch_size(Empty())


class TestLoadVisionBackbone:
    def test_lingbot_variant_dispatches_to_lingbot_loader(self) -> None:
        """variant='lingbot-base' → lingbot_vision.load_pretrained_backbone を call。"""
        with patch("lingbot_vision.load_pretrained_backbone") as mock:
            mock.return_value = (nn.Linear(1, 1), 42)
            model, embed_dim = load_vision_backbone("lingbot-base", device="cpu", dtype="fp32")
            mock.assert_called_once_with(variant="base", device="cpu", dtype="fp32")
            assert embed_dim == 42

    def test_lingbot_arbitrary_variant_passthrough(self) -> None:
        with patch("lingbot_vision.load_pretrained_backbone") as mock:
            mock.return_value = (nn.Linear(1, 1), 42)
            load_vision_backbone("lingbot-large", device="cpu", dtype="fp32")
            mock.assert_called_once_with(variant="large", device="cpu", dtype="fp32")

    def test_radio_variant_dispatches_to_radio_loader(self) -> None:
        """variant='radio-b' → torch.hub.load 経由の _load_radio を通る。"""
        stub = StubRadioModel(embed_dim=768)
        with patch("torch.hub.load", return_value=stub) as mock_hub:
            model, embed_dim = load_vision_backbone("radio-b", device="cpu", dtype="fp32")
            mock_hub.assert_called_once()
            _, kwargs = mock_hub.call_args
            # torch.hub.load(repo, name, version=..., progress=..., force_reload=...)
            args, _ = mock_hub.call_args
            assert args[0] == "NVlabs/RADIO"
            assert args[1] == "radio_model"
            assert kwargs.get("version") == "radio_v2.5-b"
            assert kwargs.get("trust_repo") is True
            assert embed_dim == 768
            assert isinstance(model, RadioAdapter)

    def test_radio_l_variant(self) -> None:
        stub = StubRadioModel(embed_dim=1024)
        with patch("torch.hub.load", return_value=stub) as mock_hub:
            _, embed_dim = load_vision_backbone("radio-l", device="cpu", dtype="fp32")
            _, kwargs = mock_hub.call_args
            assert kwargs.get("version") == "radio_v2.5-l"
            assert embed_dim == 1024

    def test_unknown_radio_size_raises(self) -> None:
        with patch("torch.hub.load") as mock_hub:
            with pytest.raises(ValueError, match="radio size"):
                load_vision_backbone("radio-x", device="cpu", dtype="fp32")
            mock_hub.assert_not_called()

    def test_unknown_variant_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown variant"):
            load_vision_backbone("clip-b", device="cpu", dtype="fp32")

    def test_radio_backbone_is_frozen(self) -> None:
        stub = StubRadioModel(embed_dim=768)
        with patch("torch.hub.load", return_value=stub):
            adapter, _ = load_vision_backbone("radio-b", device="cpu", dtype="fp32")
            for name, p in adapter.radio.named_parameters():
                assert not p.requires_grad, f"radio.{name} not frozen"

    def test_radio_input_conditioner_dtype_overridden(self) -> None:
        """Issue #122: InputConditioner.dtype は Python 属性で Module.to() で変化しない。
        _load_radio が明示的に target dtype に上書きするか確認 (bf16 forward で
        mat1/mat2 dtype mismatch の RuntimeError 回避のため必須)。"""
        stub = StubRadioModel(embed_dim=768)
        # 実 RADIO を模した input_conditioner (Python 属性の dtype)
        conditioner = nn.Module()
        conditioner.dtype = torch.float32   # 初期値、backbone dtype と mismatch させる
        stub.input_conditioner = conditioner
        with patch("torch.hub.load", return_value=stub):
            load_vision_backbone("radio-b", device="cpu", dtype="bf16")
        assert stub.input_conditioner.dtype == torch.bfloat16, (
            "InputConditioner.dtype が bf16 に更新されていない (fix 未適用または壊れた)"
        )

    def test_radio_no_input_conditioner_ok(self) -> None:
        """input_conditioner を持たない RADIO variant でも AttributeError にならない。"""
        stub = StubRadioModel(embed_dim=768)  # input_conditioner 属性なし
        with patch("torch.hub.load", return_value=stub):
            adapter, _ = load_vision_backbone("radio-b", device="cpu", dtype="bf16")
        assert isinstance(adapter, RadioAdapter)


class TestResolveDtype:
    @pytest.mark.parametrize(
        "s,expected",
        [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)],
    )
    def test_known(self, s: str, expected: torch.dtype) -> None:
        assert _resolve_dtype(s) == expected

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            _resolve_dtype("int8")
