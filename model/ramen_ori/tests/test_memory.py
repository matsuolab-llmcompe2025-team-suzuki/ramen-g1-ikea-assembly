"""memory の token (memory.MemoryEncoder、Issue #141 Phase 7) の test。"""

from __future__ import annotations

import pytest
import torch

from model.ramen_ori.memory import MemoryEncoder
from model.ramen_ori.memory_features import MEMORY_DIM, MOTION_SLICE


def _encoder(motion_dropout: float = 0.3) -> MemoryEncoder:
    enc = MemoryEncoder(d_model=32, motion_dropout=motion_dropout)
    enc.set_stats(
        clip_low=-torch.ones(MEMORY_DIM), clip_high=torch.ones(MEMORY_DIM),
        mean=torch.full((MEMORY_DIM,), 0.5), std=torch.full((MEMORY_DIM,), 2.0),
    )
    return enc


def _fc1_input(enc: MemoryEncoder, memory: torch.Tensor) -> torch.Tensor:
    captured = {}
    handle = enc.fc1.register_forward_hook(lambda m, inp, out: captured.update(x=inp[0]))  # None を返す
    out = enc(memory)
    handle.remove()
    assert out.shape == (memory.shape[0], 1, 32)
    return captured["x"]


def test_clip_then_normalize_and_flag_zero_in_eval() -> None:
    enc = _encoder().eval()
    memory = torch.full((4, MEMORY_DIM), 3.0)          # 切り詰めの上限 1 を超える
    x = _fc1_input(enc, memory)
    assert x.shape == (4, MEMORY_DIM + 1)
    torch.testing.assert_close(x[:, :MEMORY_DIM], torch.full((4, MEMORY_DIM), (1.0 - 0.5) / 2.0))
    assert torch.all(x[:, MEMORY_DIM] == 0)


def test_motion_is_hidden_and_flagged_in_train() -> None:
    enc = _encoder(motion_dropout=1.0).train()
    memory = torch.zeros(3, MEMORY_DIM)
    x = _fc1_input(enc, memory)
    assert torch.all(x[:, MOTION_SLICE] == 0)
    torch.testing.assert_close(x[:, :MOTION_SLICE.start], torch.full((3, MOTION_SLICE.start), -0.25))
    assert torch.all(x[:, MEMORY_DIM] == 1)


def test_hide_rate() -> None:
    torch.manual_seed(0)
    enc = _encoder(motion_dropout=0.3).train()
    x = _fc1_input(enc, torch.zeros(20000, MEMORY_DIM))
    assert float(x[:, MEMORY_DIM].mean()) == pytest.approx(0.3, abs=0.01)


def test_no_dropout_has_no_flag() -> None:
    enc = MemoryEncoder(d_model=32, motion_dropout=0.0)
    assert enc.fc1.in_features == MEMORY_DIM


def test_stats_are_buffers_in_state_dict() -> None:
    keys = set(_encoder().state_dict())
    assert {"clip_low", "clip_high", "normalizer.mean", "normalizer.std", "normalizer.ready"} <= keys


def test_used_before_stats_raises() -> None:
    with pytest.raises(RuntimeError, match="stats"):
        MemoryEncoder(d_model=32)(torch.zeros(1, MEMORY_DIM))


def test_invalid_stats_raise() -> None:
    enc = MemoryEncoder(d_model=32)
    with pytest.raises(ValueError, match="clip"):
        enc.set_stats(torch.ones(MEMORY_DIM), torch.zeros(MEMORY_DIM), torch.zeros(MEMORY_DIM), torch.ones(MEMORY_DIM))
