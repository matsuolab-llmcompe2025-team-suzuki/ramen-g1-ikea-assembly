"""ChunkExecutor (Issue #139) の unit test。偽の推論関数で実行層の意味論だけを見る。"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from inference.desktop.lower_policy.chunk_executor import ChunkExecutor


# diffusion_absolute_v1 = lead 2 / max_age 0.20s。execution_steps 4 なら
# replan_after = 2 で、seed 直後の step 1 に async 推論が submit される。
_FAMILY = "diffusion_absolute_v1"


def _executor(
    *,
    replan_family: str | None = _FAMILY,
    temporal_lambda: float | None = -0.1,
    execution_steps: int = 4,
) -> ChunkExecutor:
    return ChunkExecutor(
        action_dim=3,
        temporal_lambda=temporal_lambda,
        replan_family=replan_family,
        execution_steps=execution_steps,
        thread_name_prefix="chunk-executor-test",
    )


def _wait_for_async(executor: ChunkExecutor, timeout_s: float = 2.0) -> None:
    """submit 済みの async 推論が完了するまで待つ (promote はしない)。"""
    pending = executor._pipeline._pending
    deadline = time.monotonic() + timeout_s
    while pending is not None and not pending.done():
        assert time.monotonic() < deadline
        time.sleep(0.001)


def test_unknown_replan_family_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="no asynchronous replanning profile"):
        _executor(replan_family="no_such_family")


def test_sync_mode_blends_overlapping_chunks_every_tick() -> None:
    executor = _executor(replan_family=None)
    executor.step(lambda: (np.zeros((16, 3)), 2.0))
    target, meta = executor.step(lambda: (np.ones((16, 3)), 2.0))

    # step 1 の候補 = seed の row 1 (age 1) + 新 chunk の row 0 (age 0)
    weight_old = math.exp(-0.1)
    np.testing.assert_allclose(target, 1.0 / (weight_old + 1.0))
    assert meta["chunk_source"] == "sync"
    assert meta["blended_from_n_candidates"] == 2
    assert meta["async_deadline_miss_ticks"] is None
    assert meta["predict_latency_ms"] == 2.0


def test_async_promotes_at_boundary_and_skips_elapsed_prefix() -> None:
    seed = np.repeat(np.arange(16, dtype=np.float64)[:, None], 3, axis=1)
    replacement = seed + 100.0
    chunks = iter([seed, replacement, replacement + 100.0])
    executor = _executor(temporal_lambda=None)
    try:
        target, meta = executor.step(lambda: (next(chunks), 5.0))  # step 0
        assert meta["chunk_source"] == "sync"
        np.testing.assert_allclose(target, seed[0])

        _, meta = executor.step(lambda: (next(chunks), 5.0))  # step 1
        assert meta["pending_submit_step"] == 1
        _wait_for_async(executor)
        executor.step(lambda: (next(chunks), 5.0))  # step 2
        executor.step(lambda: (next(chunks), 5.0))  # step 3

        target, meta = executor.step(lambda: (next(chunks), 5.0))  # step 4
        assert meta["chunk_source"] == "async_promoted"
        # 差し替え chunk は step 1 基準。step 4 は row 3 で、経過 3 行を飛ばした後に 1 行進む
        np.testing.assert_allclose(target, replacement[3])
        assert meta["pipeline_index"] == 3
        # 境界で次の推論がすぐ出る
        assert meta["pending_submit_step"] == 4
    finally:
        executor.close()


def test_late_prediction_holds_last_target_without_sync_fallback() -> None:
    release = threading.Event()
    calls = 0

    def predict() -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        if calls > 1:
            assert release.wait(timeout=2.0)
        return np.full((16, 3), float(calls)), 1.0

    executor = _executor()
    try:
        # 新しい chunk が届かないまま、seed の origin (step 0) から 12 tick
        # (= execution_steps 4 + lead 2 + max_age 0.20s × 30Hz) までは seed を blend する
        for _ in range(13):
            _, meta = executor.step(predict)
            assert meta["chunk_source"] != "async_hold"
        started = time.monotonic()
        held, meta = executor.step(predict)  # step 13
        assert time.monotonic() - started < 0.05
        np.testing.assert_allclose(held, 1.0)
        assert meta["chunk_source"] == "async_hold"
        assert meta["async_hold_ticks"] == 1
        assert meta["fresh_chunk_age_ticks"] == 13
        assert meta["async_deadline_miss_ticks"] > 0
        # seed + 走行中の async 1 本だけ。制御 thread での同期推論は無い
        assert calls == 2
    finally:
        release.set()
        executor.reset()


def test_stalled_inference_holds_before_a_long_chunk_runs_out() -> None:
    """ACT のように chunk が長くても、止まった推論の古い計画を先まで実行しない。"""
    release = threading.Event()
    calls = 0
    seed = np.repeat(np.arange(100, dtype=np.float64)[:, None], 3, axis=1)

    def predict() -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        if calls > 1:
            assert release.wait(timeout=2.0)
        return seed.copy(), 1.0

    executor = _executor(temporal_lambda=None)
    try:
        for step in range(13):
            target, _ = executor.step(predict)
            np.testing.assert_allclose(target, seed[step])
        for step in range(13, 20):
            held, meta = executor.step(predict)
            # seed はまだ 87 行残っているが、step 12 の目標で止まる
            np.testing.assert_allclose(held, seed[12])
            assert meta["chunk_source"] == "async_hold"
            assert meta["blended_from_n_candidates"] == 1
    finally:
        release.set()
        executor.reset()


@pytest.mark.parametrize(("family", "limit"), [("act", 18), ("diffusion", 21)])
def test_fresh_limit_for_act_and_diffusion_slots(family: str, limit: int) -> None:
    # slot の execution_steps 8。ACT = 8 + lead 2 + 0.25s×30、DP = 8 + lead 4 + 0.30s×30
    executor = _executor(replan_family=family, execution_steps=8)
    assert executor._fresh_limit_ticks == limit
    assert executor._max_hold_ticks == 30


def test_hold_longer_than_max_hold_raises() -> None:
    release = threading.Event()
    calls = 0

    def predict() -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        if calls > 1:
            assert release.wait(timeout=2.0)
        return np.zeros((100, 3)), 1.0

    # max_hold 0.1 s = 3 tick。step 13 から保持して、4 tick 目 (step 16) で止める
    executor = ChunkExecutor(
        action_dim=3,
        temporal_lambda=-0.1,
        replan_family=_FAMILY,
        execution_steps=4,
        thread_name_prefix="chunk-executor-test",
        max_hold_s=0.1,
    )
    try:
        for _ in range(16):
            executor.step(predict)
        with pytest.raises(RuntimeError, match="inference stalled"):
            executor.step(predict)
    finally:
        release.set()
        executor.reset()


def test_wall_clock_delay_alone_does_not_discard_a_prediction() -> None:
    """古さは tick で数える (Issue #159 B3b-1)。

    大会経路は tick が推論に律速されて 20 Hz 以下になる。壁時計で数えると、
    何 step も遅れていない予測まで捨てて最後の target を保持し続けていた。
    """
    release = threading.Event()
    calls = 0

    def predict() -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert release.wait(timeout=2.0)
        return np.full((16, 3), float(calls)), 1.0

    executor = _executor()
    try:
        executor.step(predict)  # step 0
        _, meta = executor.step(predict)  # step 1
        assert meta["pending_submit_step"] == 1
        time.sleep(0.25)  # 壁時計では max_age 0.20s を超える
        release.set()
        _wait_for_async(executor)
        executor.step(predict)  # step 2
        executor.step(predict)  # step 3

        _, meta = executor.step(predict)  # step 4 (境界): submit から 3 tick
        assert meta["async_stale_discard_count"] == 0
        assert meta["chunk_source"] == "async_promoted"
    finally:
        executor.close()


def test_stale_prediction_is_discarded_and_resubmitted_in_same_tick() -> None:
    release = threading.Event()
    calls = 0

    def predict() -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert release.wait(timeout=2.0)
        return np.full((16, 3), float(calls)), 1.0

    executor = _executor()
    try:
        executor.step(predict)  # step 0
        _, meta = executor.step(predict)  # step 1: tick 1 で submit
        assert meta["pending_submit_step"] == 1
        # max_age 0.20s = 6 tick。step 9 の promote は submit から 7 tick 後
        for _ in range(2, 9):
            executor.step(predict)
        release.set()
        _wait_for_async(executor)

        _, meta = executor.step(predict)  # step 9
        assert meta["async_stale_discard_count"] == 1
        assert meta["chunk_source"] == "async_none_this_tick"
        assert meta["pending_submit_step"] == 9
    finally:
        executor.close()


def test_async_prediction_failure_surfaces_at_boundary() -> None:
    calls = 0

    def predict() -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("worker died")
        return np.zeros((16, 3)), 1.0

    executor = _executor()
    try:
        executor.step(predict)  # step 0
        executor.step(predict)  # step 1: async 側で失敗する
        _wait_for_async(executor)
        executor.step(predict)  # step 2
        executor.step(predict)  # step 3
        with pytest.raises(RuntimeError, match="worker died"):
            executor.step(predict)  # step 4: promote 時に再送出
    finally:
        executor.reset()


def test_reset_restarts_from_sync_seed() -> None:
    calls = 0

    def predict() -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        return np.full((16, 3), float(calls)), 1.0

    executor = _executor()
    executor.step(predict)
    executor.step(predict)
    _wait_for_async(executor)
    executor.reset()
    assert executor._pipeline is None

    target, meta = executor.step(predict)
    try:
        assert meta["chunk_source"] == "sync"
        assert meta["async_hold_ticks"] == 0
        # reset 前の chunk は候補に残っていない
        assert meta["blended_from_n_candidates"] == 1
        np.testing.assert_allclose(target, 3.0)
    finally:
        executor.close()


def test_close_aborts_unresponsive_prediction() -> None:
    release = threading.Event()
    calls = 0

    def predict() -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            release.wait()
            raise EOFError("worker terminated")
        return np.zeros((16, 3)), 1.0

    executor = _executor()
    executor.step(predict)
    executor.step(predict)  # async 1 本が応答しないまま
    started = time.monotonic()
    executor.close(abort_pending=release.set)
    assert release.is_set()
    assert time.monotonic() - started < 1.5
    assert executor._pipeline is None


# ---- 先の予定 (Issue #188): 腕の遅れる会場で、予定の少し先を送るための口 ----


def test_preview_target_reads_the_plan_ahead_of_the_emitted_step() -> None:
    executor = _executor(replan_family=None, temporal_lambda=None)
    chunk = np.arange(16 * 3, dtype=np.float64).reshape(16, 3)
    target, _ = executor.step(lambda: (chunk, 1.0))
    np.testing.assert_allclose(target, chunk[0])
    preview, ahead = executor.preview_target(6)
    assert ahead == 6
    np.testing.assert_allclose(preview, chunk[6])
    assert executor.preview_target(0) is None
    executor.reset()
    assert executor.preview_target(6) is None


def test_preview_target_is_none_while_holding_the_last_target() -> None:
    release = threading.Event()
    calls = 0

    def predict() -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        if calls > 1:
            assert release.wait(timeout=2.0)
        return np.full((16, 3), float(calls)), 1.0

    executor = _executor()
    try:
        for _ in range(13):
            executor.step(predict)
        assert executor.preview_target(2) is not None
        _, meta = executor.step(predict)
        assert meta["chunk_source"] == "async_hold"
        assert executor.preview_target(2) is None
    finally:
        release.set()
        executor.reset()
