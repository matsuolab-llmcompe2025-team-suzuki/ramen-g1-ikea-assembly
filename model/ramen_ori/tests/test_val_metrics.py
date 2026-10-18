"""val_metrics.py の unit test (Issue #129 Phase H、2026-08-31)。

- V1 (compute_ee_error_mm):
  - Absolute mode で pred == teacher → EE error mm ≈ 0
  - Absolute mode で pred ≠ teacher → EE error > 0
  - Relative mode で reconstruct → 同 mode 内で pred == teacher → EE error ≈ 0
  - Left/right EE を各々 log
- V2 (compute_motion_energy):
  - Constant chunk (全 step 同一) → dq_L=dq_R=0
  - Ramp chunk (linear ↑) → dq 非ゼロ、ptp > 0
  - Left/right asymmetry: left が大きく動く → LR_dq > 1
  - Shape/dim raise
"""

from __future__ import annotations

import pytest
import torch

from model.ramen_ori.fk import G1WristFKTorch
from model.ramen_ori.val_metrics import (
    compute_ee_error_mm,
    compute_motion_energy,
)


@pytest.fixture(scope="module")
def fk() -> G1WristFKTorch:
    return G1WristFKTorch.from_default_urdf()


class TestComputeEEErrorMm:
    def test_pred_equals_teacher_zero_error(self, fk) -> None:
        """Absolute mode で pred == teacher → EE error mm は 0 (数値誤差以下)。"""
        B, chunk_len = 2, 4
        teacher = torch.zeros(B, chunk_len, 16)
        pred = teacher.clone()
        waist = torch.zeros(B, chunk_len, 3)
        ee = compute_ee_error_mm(
            pred_action=pred, teacher_action=teacher, teacher_waist=waist, fk=fk
        )
        assert ee["left_mm"] == pytest.approx(0.0, abs=1e-3)
        assert ee["right_mm"] == pytest.approx(0.0, abs=1e-3)
        assert ee["avg_mm"] == pytest.approx(0.0, abs=1e-3)

    def test_pred_differs_teacher_nonzero_error(self, fk) -> None:
        B, chunk_len = 2, 4
        teacher = torch.zeros(B, chunk_len, 16)
        pred = teacher.clone()
        # arms 部分に 0.1 rad ズラす → EE 位置は数 cm shift
        pred[:, -1, 0] = 0.1
        waist = torch.zeros(B, chunk_len, 3)
        ee = compute_ee_error_mm(
            pred_action=pred, teacher_action=teacher, teacher_waist=waist, fk=fk
        )
        assert ee["left_mm"] > 0.0
        assert ee["avg_mm"] > 0.0

    def test_returns_left_right_and_avg_keys(self, fk) -> None:
        B, chunk_len = 1, 4
        pred = torch.zeros(B, chunk_len, 16)
        teacher = torch.zeros(B, chunk_len, 16)
        waist = torch.zeros(B, chunk_len, 3)
        ee = compute_ee_error_mm(pred, teacher, waist, fk=fk)
        assert set(ee.keys()) == {"left_mm", "right_mm", "avg_mm"}

    def test_relative_mode_zero_error_with_perfect_pred(self, fk) -> None:
        """Relative mode: pred と teacher が同 normalized Δq → reconstruct 後も同一、EE error 0。"""
        B, chunk_len = 2, 4
        # Normalized Δq (arms 14) と absolute hand (2) が混じった 16D
        pred = torch.randn(B, chunk_len, 16) * 0.1
        teacher = pred.clone()
        waist = torch.zeros(B, chunk_len, 3)
        arms_current = torch.zeros(B, 14)
        rel_mean = torch.zeros(14)
        rel_std = torch.ones(14) * 0.01
        ee = compute_ee_error_mm(
            pred_action=pred, teacher_action=teacher, teacher_waist=waist,
            fk=fk, arms_current=arms_current, rel_mean=rel_mean, rel_std=rel_std,
            use_relative_action=True,
        )
        assert ee["avg_mm"] == pytest.approx(0.0, abs=1e-3)

    def test_relative_mode_requires_stats(self, fk) -> None:
        B, chunk_len = 1, 4
        pred = torch.zeros(B, chunk_len, 16)
        teacher = torch.zeros(B, chunk_len, 16)
        waist = torch.zeros(B, chunk_len, 3)
        with pytest.raises(ValueError, match="arms_current"):
            compute_ee_error_mm(
                pred_action=pred, teacher_action=teacher, teacher_waist=waist,
                fk=fk, use_relative_action=True,
                # arms_current, rel_mean, rel_std を敢えて渡さない
            )


class TestComputeMotionEnergy:
    def test_constant_chunk_zero_motion(self) -> None:
        """全 step 同一 = 静止 → dq_L=dq_R=0、ptp=0。"""
        B, chunk_len = 2, 4
        action = torch.ones(B, chunk_len, 16) * 0.5
        m = compute_motion_energy(action)
        assert m["dq_L"] == pytest.approx(0.0)
        assert m["dq_R"] == pytest.approx(0.0)
        assert m["ptp_L"] == pytest.approx(0.0)
        assert m["ptp_R"] == pytest.approx(0.0)

    def test_ramp_chunk_nonzero_motion(self) -> None:
        """chunk_step ごと linear ↑ → dq 非ゼロ、ptp > 0。"""
        B, chunk_len = 1, 4
        action = torch.zeros(B, chunk_len, 16)
        # step k で arms[0..13] = k * 0.1
        for k in range(chunk_len):
            action[:, k, :14] = k * 0.1
        m = compute_motion_energy(action)
        # dq (per-tick) = 0.1 (arms 部分)
        assert m["dq_L"] == pytest.approx(0.1, abs=1e-5)
        assert m["dq_R"] == pytest.approx(0.1, abs=1e-5)
        # ptp per-dim = (chunk_len - 1) * 0.1 = 0.3
        assert m["ptp_L"] == pytest.approx(0.3, abs=1e-5)
        assert m["ptp_R"] == pytest.approx(0.3, abs=1e-5)

    def test_left_only_motion_LR_dq_gt_one(self) -> None:
        """Left arm のみ動く → LR_dq >> 1、LR_asym > 0。"""
        B, chunk_len = 1, 4
        action = torch.zeros(B, chunk_len, 16)
        for k in range(chunk_len):
            action[:, k, 0:7] = k * 0.1   # left arm のみ
        m = compute_motion_energy(action)
        assert m["dq_L"] > 0.0
        assert m["dq_R"] == pytest.approx(0.0, abs=1e-6)
        assert m["LR_dq"] > 1e6, "R が 0 なら LR_dq は極大値 (eps 分母 clip)"
        # LR asym = |ΔL - ΔR| mean = |0.1 - 0| = 0.1
        assert m["LR_asym"] == pytest.approx(0.1, abs=1e-5)

    def test_symmetric_motion_LR_dq_near_one(self) -> None:
        """Left/right 同じ動き → LR_dq ≈ 1、LR_asym = 0。"""
        B, chunk_len = 1, 4
        action = torch.zeros(B, chunk_len, 16)
        for k in range(chunk_len):
            action[:, k, 0:14] = k * 0.1   # left + right arms 同じ
        m = compute_motion_energy(action)
        assert m["LR_dq"] == pytest.approx(1.0, abs=1e-5)
        assert m["LR_asym"] == pytest.approx(0.0, abs=1e-6)

    def test_chunk_len_1_returns_zeros(self) -> None:
        """chunk_len=1 で Δq 定義できず、全 metric 0 or 1 返す。"""
        action = torch.rand(2, 1, 16)
        m = compute_motion_energy(action)
        assert m["dq_L"] == 0.0
        assert m["dq_R"] == 0.0
        assert m["LR_dq"] == 1.0

    def test_wrong_shape_raises(self) -> None:
        bad = torch.zeros(2, 16)  # 2D not 3D
        with pytest.raises(ValueError, match=r"\(B, chunk, 16\)"):
            compute_motion_energy(bad)

    def test_hand_dims_excluded_from_motion(self) -> None:
        """Hand 2 dim (index 14, 15) は motion metric 対象外 (arms のみ)。"""
        B, chunk_len = 1, 4
        action = torch.zeros(B, chunk_len, 16)
        # hand のみ動かす (arms は固定)
        for k in range(chunk_len):
            action[:, k, 14:16] = k * 0.5
        m = compute_motion_energy(action)
        # arms 部分は不変
        assert m["dq_L"] == pytest.approx(0.0, abs=1e-6)
        assert m["dq_R"] == pytest.approx(0.0, abs=1e-6)
        assert m["ptp_L"] == pytest.approx(0.0, abs=1e-6)


class TestEEErrorRows:
    """Issue #141: 推論で実行する先頭の行 (n_rows) のうち、埋め草でない行だけで測る。"""

    def test_rows_beyond_n_rows_are_ignored(self, fk) -> None:
        teacher = torch.zeros(2, 6, 16)
        pred = teacher.clone()
        pred[:, 4:, 0] = 0.3   # 5 行目以降だけずれる
        waist = torch.zeros(2, 6, 3)
        ee = compute_ee_error_mm(pred, teacher, waist, fk=fk, n_rows=4)
        assert ee["avg_mm"] == pytest.approx(0.0, abs=1e-3)
        assert compute_ee_error_mm(pred, teacher, waist, fk=fk)["avg_mm"] > 0.0

    def test_pad_rows_are_ignored(self, fk) -> None:
        teacher = torch.zeros(2, 6, 16)
        pred = teacher.clone()
        pred[:, 2:, 0] = 0.3
        waist = torch.zeros(2, 6, 3)
        pad = torch.zeros(2, 6, dtype=torch.bool)
        pad[:, 2:] = True
        ee = compute_ee_error_mm(pred, teacher, waist, fk=fk, row_mask=~pad, n_rows=4)
        assert ee["avg_mm"] == pytest.approx(0.0, abs=1e-3)

    def test_mean_over_valid_rows(self, fk) -> None:
        """有効な行の L2 距離の平均 (1 行だけずれて、有効 2 行なら半分)。"""
        teacher = torch.zeros(1, 3, 16)
        pred = teacher.clone()
        pred[:, 1, 0] = 0.3
        waist = torch.zeros(1, 3, 3)
        one_row = compute_ee_error_mm(pred[:, 1:2], teacher[:, 1:2], waist[:, 1:2], fk=fk)
        two_rows = compute_ee_error_mm(pred, teacher, waist, fk=fk, n_rows=2)
        assert two_rows["left_mm"] == pytest.approx(one_row["left_mm"] / 2, rel=1e-5)
