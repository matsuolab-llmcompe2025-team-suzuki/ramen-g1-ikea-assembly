"""Task chunk balancing (compute_task_uniform_weights) の unit test (Issue #120 Phase A-5)。"""

from __future__ import annotations

import numpy as np

from model.ramen_ori.data_lerobot import compute_task_uniform_weights


class TestTaskUniformWeights:
    def test_two_tasks_8_to_1_ratio_balances_to_1_to_1(self) -> None:
        """task 5 が 8 (task 7 の 8 倍) → weight で per-task total = 1/2 に補正。"""
        task_ids = np.array([5] * 800 + [7] * 100, dtype=np.int64)
        weights = compute_task_uniform_weights(task_ids)

        assert weights.shape == (900,)
        # per-task 合計は 1/N_task = 1/2 で等しい
        w_task5 = weights[task_ids == 5].sum()
        w_task7 = weights[task_ids == 7].sum()
        np.testing.assert_allclose(w_task5, 0.5, atol=1e-9)
        np.testing.assert_allclose(w_task7, 0.5, atol=1e-9)

    def test_uniform_task_distribution(self) -> None:
        """3 task 均等頻度 → 各 sample の weight が同じ (1/(3*N_per_task))。"""
        task_ids = np.array([0, 1, 2] * 100, dtype=np.int64)
        weights = compute_task_uniform_weights(task_ids)
        expected = 1.0 / (3 * 100)
        np.testing.assert_allclose(weights, expected, atol=1e-12)

    def test_all_same_task(self) -> None:
        """全 sample 同 task → 均等 weight。"""
        task_ids = np.zeros(50, dtype=np.int64)
        weights = compute_task_uniform_weights(task_ids)
        np.testing.assert_allclose(weights, 1.0 / 50, atol=1e-12)

    def test_dtype_is_float64(self) -> None:
        task_ids = np.array([5, 5, 7], dtype=np.int64)
        assert compute_task_uniform_weights(task_ids).dtype == np.float64
