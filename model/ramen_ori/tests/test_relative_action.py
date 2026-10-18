"""relative_action.py の unit test (Issue #129 Phase F、2026-08-31)。

- load_relative_stats: {mean, std} 形式読込 + shape 検証 + numpy/torch dtype 変換
- compute_teacher_arms_dq: 特定 teacher trajectory → 期待 Δq
- normalize / denormalize round-trip: normalize(denorm(x)) ≈ x
- reconstruct_arms_abs_from_dq_norm: normalized Δq → absolute q 復元 (cumsum)
- Shape / dim mismatch raise
- autograd 通過
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from model.ramen_ori.relative_action import (
    compute_teacher_arms_dq,
    denormalize_arms_dq,
    load_relative_stats,
    normalize_arms_dq,
    reconstruct_arms_abs_from_dq_norm,
)


@pytest.fixture
def sample_stats() -> dict[str, torch.Tensor]:
    """Test 用の per-dim (14,) stats。"""
    mean = torch.arange(14, dtype=torch.float32) * 0.01   # 0..0.13
    std = torch.ones(14, dtype=torch.float32) * 0.05
    return {"mean": mean, "std": std}


class TestLoadRelativeStats:
    def test_roundtrip_from_torch_save(self, tmp_path: Path) -> None:
        stats_in = {"mean": torch.arange(14, dtype=torch.float32), "std": torch.ones(14) * 0.1}
        path = tmp_path / "stats.pt"
        torch.save(stats_in, path)
        stats = load_relative_stats(path)
        assert stats["mean"].shape == (14,)
        assert stats["std"].shape == (14,)
        torch.testing.assert_close(stats["mean"], stats_in["mean"])
        torch.testing.assert_close(stats["std"], stats_in["std"])

    def test_numpy_input_converted_to_torch(self, tmp_path: Path) -> None:
        stats_in = {"mean": np.arange(14, dtype=np.float32), "std": np.ones(14, dtype=np.float32) * 0.1}
        path = tmp_path / "stats_np.pt"
        torch.save(stats_in, path)
        stats = load_relative_stats(path)
        assert isinstance(stats["mean"], torch.Tensor)
        assert isinstance(stats["std"], torch.Tensor)
        assert stats["mean"].dtype == torch.float32

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save({"mean": torch.zeros(14)}, path)  # missing "std"
        with pytest.raises(ValueError, match="mean.*std"):
            load_relative_stats(path)

    def test_wrong_shape_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.pt"
        torch.save({"mean": torch.zeros(19), "std": torch.zeros(19)}, path)  # 19 でなく 14
        with pytest.raises(ValueError, match=r"must be \(14,\)"):
            load_relative_stats(path)


class TestComputeTeacherArmsDq:
    def test_dq_zero_at_step_zero_when_teacher_equals_current(self) -> None:
        """teacher[0] = current → Δq[0] = 0。"""
        arms_current = torch.zeros(14)
        arms_teacher = torch.zeros(4, 14)  # 全 0 (chunk_len=4)
        dq = compute_teacher_arms_dq(arms_teacher, arms_current)
        assert dq.shape == (4, 14)
        torch.testing.assert_close(dq, torch.zeros(4, 14))

    def test_dq_shape_and_first_step(self) -> None:
        """Δq[0] = teacher[0] - current。"""
        arms_current = torch.ones(14) * 0.5
        arms_teacher = torch.arange(4 * 14, dtype=torch.float32).reshape(4, 14) * 0.1
        dq = compute_teacher_arms_dq(arms_teacher, arms_current)
        assert dq.shape == (4, 14)
        # dq[0] = teacher[0] - current
        expected_0 = arms_teacher[0] - arms_current
        torch.testing.assert_close(dq[0], expected_0)

    def test_dq_rest_is_consecutive_diff(self) -> None:
        """Δq[k>0] = teacher[k] - teacher[k-1]。"""
        arms_current = torch.zeros(14)
        arms_teacher = torch.tensor(
            [[float(i)] * 14 for i in range(4)], dtype=torch.float32
        )   # [[0,..0], [1,..1], [2,..2], [3,..3]]
        dq = compute_teacher_arms_dq(arms_teacher, arms_current)
        # dq[1] = teacher[1] - teacher[0] = 1
        torch.testing.assert_close(dq[1], torch.ones(14))
        torch.testing.assert_close(dq[2], torch.ones(14))
        torch.testing.assert_close(dq[3], torch.ones(14))

    def test_batch_shape(self) -> None:
        arms_current = torch.zeros(2, 14)  # (B=2, 14)
        arms_teacher = torch.zeros(2, 4, 14)  # (B=2, chunk=4, 14)
        dq = compute_teacher_arms_dq(arms_teacher, arms_current)
        assert dq.shape == (2, 4, 14)

    def test_wrong_dim_raises(self) -> None:
        arms_current = torch.zeros(14)
        arms_teacher_bad = torch.zeros(4, 13)  # last dim 13
        with pytest.raises(ValueError, match="14"):
            compute_teacher_arms_dq(arms_teacher_bad, arms_current)


class TestNormalizeDenormalizeRoundtrip:
    def test_normalize_then_denormalize_identity(self, sample_stats) -> None:
        mean, std = sample_stats["mean"], sample_stats["std"]
        dq = torch.randn(4, 14)
        norm = normalize_arms_dq(dq, mean, std)
        denorm = denormalize_arms_dq(norm, mean, std)
        torch.testing.assert_close(dq, denorm, atol=1e-5, rtol=1e-5)

    def test_normalize_shifts_to_unit_variance(self, sample_stats) -> None:
        """Δq が mean=0, std=1 に normalize されるはず (dq が sample_stats を仮定した分布)。"""
        mean, std = sample_stats["mean"], sample_stats["std"]
        # dq をランダム生成、期待するのは normalize 後 approx unit variance
        torch.manual_seed(0)
        dq_raw = torch.randn(1000, 14) * std + mean   # stats を反映した分布
        norm = normalize_arms_dq(dq_raw, mean, std)
        # normalize 後 mean ≈ 0、std ≈ 1
        assert norm.mean().abs().item() < 0.1
        assert (norm.std() - 1.0).abs().item() < 0.1

    def test_std_zero_uses_eps_fallback(self, sample_stats) -> None:
        """std=0 の dim があっても NaN にならない (eps clamp)。"""
        mean = torch.zeros(14)
        std = torch.zeros(14)   # 全 dim 0
        dq = torch.randn(4, 14)
        norm = normalize_arms_dq(dq, mean, std, eps=1e-6)
        assert torch.isfinite(norm).all()


class TestReconstructArmsAbs:
    def test_reconstruct_recovers_teacher(self, sample_stats) -> None:
        """teacher → Δq → normalize → reconstruct = teacher (round-trip)。"""
        mean, std = sample_stats["mean"], sample_stats["std"]
        torch.manual_seed(0)
        arms_current = torch.randn(14)
        arms_teacher = torch.randn(4, 14)
        dq = compute_teacher_arms_dq(arms_teacher, arms_current)
        dq_norm = normalize_arms_dq(dq, mean, std)
        recovered = reconstruct_arms_abs_from_dq_norm(dq_norm, arms_current, mean, std)
        torch.testing.assert_close(recovered, arms_teacher, atol=1e-4, rtol=1e-4)

    def test_batch_reconstruction(self, sample_stats) -> None:
        mean, std = sample_stats["mean"], sample_stats["std"]
        torch.manual_seed(1)
        arms_current = torch.randn(3, 14)  # B=3
        arms_teacher = torch.randn(3, 4, 14)  # (B=3, chunk=4, 14)
        dq = compute_teacher_arms_dq(arms_teacher, arms_current)
        dq_norm = normalize_arms_dq(dq, mean, std)
        recovered = reconstruct_arms_abs_from_dq_norm(dq_norm, arms_current, mean, std)
        assert recovered.shape == (3, 4, 14)
        torch.testing.assert_close(recovered, arms_teacher, atol=1e-4, rtol=1e-4)

    def test_zero_dq_yields_arms_current_at_all_steps(self, sample_stats) -> None:
        """Δq normalized = -mean/std (raw Δq = 0) → 全 step で arms_current。"""
        mean, std = sample_stats["mean"], sample_stats["std"]
        arms_current = torch.ones(14) * 0.3
        # Δq_raw = 0 → normalized = -mean/std
        dq_norm = -mean.expand(4, 14) / std.expand(4, 14)
        recovered = reconstruct_arms_abs_from_dq_norm(dq_norm, arms_current, mean, std)
        # cumsum(0) = 0 なので全 step で arms_current
        expected = arms_current.expand(4, 14)
        torch.testing.assert_close(recovered, expected, atol=1e-5, rtol=1e-5)


class TestAutograd:
    def test_gradient_flows_through_reconstruct(self, sample_stats) -> None:
        mean, std = sample_stats["mean"], sample_stats["std"]
        dq_norm = torch.randn(4, 14, requires_grad=True)
        arms_current = torch.zeros(14)
        recovered = reconstruct_arms_abs_from_dq_norm(dq_norm, arms_current, mean, std)
        loss = (recovered ** 2).sum()
        loss.backward()
        assert dq_norm.grad is not None
        assert dq_norm.grad.abs().sum().item() > 0
