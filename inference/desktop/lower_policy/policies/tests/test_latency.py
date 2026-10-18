"""Phase 8 smoke: latency budget constants + benchmark helper (Issue #125)。

- Budget 定数 (STEP_TIME_MS / MODEL_FORWARD_BUDGET_MS / etc.) の値検証
- measure_latency helper が numpy stats を正しく返す
- check_budget の within_budget 判定 + summary format
- 実 policy latency 測定は `@pytest.mark.integration` marker (Sakura 送り、
  実機で調整前提)
"""

from __future__ import annotations

import time

import pytest

from inference.desktop.lower_policy.policies.latency import (
    ACTION_CHUNK_LEN,
    CONTROL_FPS,
    MODEL_FORWARD_BUDGET_MS,
    PER_STEP_CPU_BUDGET_MS,
    STEP_TIME_MS,
    YOLO_FORWARD_BUDGET_MS,
    check_budget,
    measure_latency,
)


class TestBudgetConstants:
    def test_control_fps_matches_g1_wbt(self) -> None:
        # G1_WBT dataset は 30fps (data_lerobot.py + CLAUDE.md 明記)
        assert CONTROL_FPS == 30

    def test_step_time_from_fps(self) -> None:
        assert STEP_TIME_MS == pytest.approx(1000.0 / 30)

    def test_action_chunk_matches_config(self) -> None:
        # RAMEN-Ori base.yaml + GR00T subtask_training.json 両方 16
        assert ACTION_CHUNK_LEN == 16

    def test_model_forward_budget(self) -> None:
        # chunk 消費期間内に predict 完了する必要 = 16 × 33.33 = 533.33ms
        assert MODEL_FORWARD_BUDGET_MS == pytest.approx(16 * 1000.0 / 30)

    def test_yolo_budget_equals_model_forward(self) -> None:
        # async 前提 = 両方とも chunk 消費期間内に完了
        assert YOLO_FORWARD_BUDGET_MS == MODEL_FORWARD_BUDGET_MS

    def test_per_step_cpu_budget(self) -> None:
        # tick loop の壁時計制約 = 1 step 分
        assert PER_STEP_CPU_BUDGET_MS == STEP_TIME_MS


class TestMeasureLatency:
    def test_returns_expected_keys(self) -> None:
        stats = measure_latency(time.sleep, 0, n_iter=3, warmup=0)
        for key in ("min_ms", "mean_ms", "p50_ms", "p95_ms", "max_ms", "all_ms"):
            assert key in stats

    def test_measures_positive_duration(self) -> None:
        # sleep 0.005 sec = 5ms、warmup 除いた 3 iter で mean ≈ 5ms
        stats = measure_latency(time.sleep, 0.005, n_iter=3, warmup=1)
        # 実 sleep は OS jitter で ±数ms、range 内で verify
        assert stats["mean_ms"] > 4.0
        assert stats["mean_ms"] < 30.0

    def test_all_ms_length(self) -> None:
        stats = measure_latency(time.sleep, 0, n_iter=7, warmup=2)
        # warmup 2 は計測に含まれない
        assert len(stats["all_ms"]) == 7

    def test_percentiles_ordered(self) -> None:
        stats = measure_latency(time.sleep, 0, n_iter=10, warmup=0)
        assert stats["min_ms"] <= stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]


class TestCheckBudget:
    def test_within_budget(self) -> None:
        stats = {"mean_ms": 10.0, "p95_ms": 15.0}
        result = check_budget(stats, budget_ms=33.33, tag="preprocess")
        assert result["within_budget"] is True
        assert result["margin_ms"] == pytest.approx(23.33)
        assert result["ratio"] == pytest.approx(10.0 / 33.33)
        assert "preprocess" in result["summary"]
        assert "OK" in result["summary"]

    def test_over_budget(self) -> None:
        stats = {"mean_ms": 50.0, "p95_ms": 55.0}
        result = check_budget(stats, budget_ms=33.33, tag="preprocess")
        assert result["within_budget"] is False
        assert result["margin_ms"] < 0
        assert "OVER" in result["summary"]

    def test_summary_format(self) -> None:
        stats = {"mean_ms": 3.2, "p95_ms": 4.5}
        result = check_budget(stats, budget_ms=33.33, tag="build_batch")
        assert "build_batch" in result["summary"]
        assert "3.20ms" in result["summary"]
        assert "4.50ms" in result["summary"]


@pytest.mark.integration
class TestRealPolicyLatency:
    """実 policy 上での latency measurement (Sakura or Orin 送り)。"""

    def test_ramen_ori_predict_within_chunk_budget(self) -> None:
        # 実機で走る想定: RamenOriPolicy.from_ckpt(cfg) 後に predict latency 測定
        # → mean_ms < MODEL_FORWARD_BUDGET_MS (533.33ms) が期待値
        pytest.skip(
            "integration: requires torch + LeRobot fork + real ckpt (Sakura or "
            "model/ramen_ori env)。local 8GB VRAM では実測不可、実機で tuning"
        )

    def test_groot_predict_within_chunk_budget(self) -> None:
        pytest.skip(
            "integration: requires lerobot + isaac-gr00t + real ckpt。GR00T 3B は "
            "local 8GB VRAM 不足、Sakura H100 or Orin で実測 + tuning"
        )
