from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from inference.desktop.lower_policy.async_replanning import (
    AsyncActionChunkPipeline,
    AsyncPredictionTask,
    advance_periodic_deadline,
    family_replanning_schedule,
    run_cleanup_steps,
)


@pytest.mark.parametrize(
    ("family", "steps", "replan", "max_age"),
    [
        ("act_absolute_joint16_v1", 30, 26, 0.30),
        ("groot_absolute_joint_v1", 8, 4, 0.30),
        ("groot_relative_eef_v1", 8, 4, 0.35),
        ("groot_relative_eef_v1", 4, 0, 0.35),
        ("groot_relative_eef_v1", 16, 12, 0.35),
        ("diffusion_chunk_relative_v1", 8, 6, 0.20),
        ("act", 8, 6, 0.25),
        ("diffusion", 8, 4, 0.30),
    ],
)
def test_family_replanning_schedule_is_centralized(
    family: str,
    steps: int,
    replan: int,
    max_age: float,
) -> None:
    assert family_replanning_schedule(family, steps) == (replan, max_age)


def test_generic_prediction_task_has_bounded_abort() -> None:
    release = threading.Event()

    def blocked() -> int:
        release.wait()
        raise EOFError("worker terminated")

    task = AsyncPredictionTask(blocked, thread_name="generic-replan-test")
    started = time.monotonic()
    task.close(timeout_s=0.02, abort_pending=release.set)
    assert time.monotonic() - started < 0.2
    assert task.shutdown_aborted is True


def test_pipeline_accepts_model_specific_horizon_and_action_dimension() -> None:
    pipeline = AsyncActionChunkPipeline(
        np.zeros((30, 16)),
        execution_steps=30,
        replan_after_steps=1,
        max_prediction_age_s=2.0,
        thread_name_prefix="act-test",
    )
    try:
        assert pipeline.chunk_shape == (30, 16)
        assert pipeline.execution_steps == 30
    finally:
        pipeline.close()


def test_pipeline_promotes_prefetched_chunk_at_boundary() -> None:
    initial = np.tile(np.arange(16, dtype=np.float64), (16, 1))
    replacement = initial + 100.0
    pipeline = AsyncActionChunkPipeline(
        initial,
        execution_steps=8,
        replan_after_steps=1,
        max_prediction_age_s=2.0,
    )
    try:
        pipeline.next_action()
        pipeline.submit(
            lambda: (replacement, 12.0, {"adapter": "test"}),
            anchor_generation=(1, 2, 3),
        )
        completed = None
        deadline = time.monotonic() + 2.0
        while completed is None and time.monotonic() < deadline:
            if pipeline.action_index < pipeline.execution_steps:
                pipeline.next_action()
            completed = pipeline.promote_if_ready()
            time.sleep(0.001)
        assert completed is not None
        assert completed.anchor_generation == (1, 2, 3)
        assert completed.diagnostics == {"adapter": "test"}
        action, index = pipeline.next_action()
        assert index == 0
        np.testing.assert_allclose(action, replacement[0])
    finally:
        pipeline.close()


def test_pipeline_holds_last_target_without_blocking_when_prediction_is_late() -> None:
    release = threading.Event()
    initial = np.arange(4 * 2, dtype=np.float64).reshape(4, 2)
    replacement = initial + 10.0
    pipeline = AsyncActionChunkPipeline(
        initial,
        execution_steps=2,
        replan_after_steps=1,
        max_prediction_age_s=2.0,
    )
    try:
        pipeline.next_action()

        def slow_prediction() -> tuple[np.ndarray, float, dict[str, float]]:
            assert release.wait(timeout=2.0)
            return replacement, 100.0, {}

        pipeline.submit(slow_prediction, anchor_generation=(4,))
        last, index = pipeline.next_action()
        assert index == 1
        started = time.monotonic()
        assert pipeline.promote_if_ready() is None
        held, index = pipeline.next_action()
        assert time.monotonic() - started < 0.05
        assert index is None
        np.testing.assert_allclose(held, last)
        assert pipeline.deadline_miss_ticks == 1
        release.set()
    finally:
        release.set()
        pipeline.close()


def test_pipeline_rejects_shape_or_nonfinite_contract_changes() -> None:
    with pytest.raises(ValueError, match="rank-2"):
        AsyncActionChunkPipeline(
            np.zeros(16), execution_steps=1, replan_after_steps=0,
            max_prediction_age_s=2.0,
        )
    with pytest.raises(ValueError, match="finite"):
        AsyncActionChunkPipeline(
            np.asarray([[np.nan]]), execution_steps=1, replan_after_steps=0,
            max_prediction_age_s=2.0,
        )

    pipeline = AsyncActionChunkPipeline(
        np.zeros((16, 16)), execution_steps=1, replan_after_steps=0,
        max_prediction_age_s=2.0,
    )
    pipeline.next_action()
    pipeline.submit(
        lambda: (np.zeros((30, 16)), 1.0, {}),
        anchor_generation=(),
    )
    deadline = time.monotonic() + 2.0
    while pipeline.prediction_pending and time.monotonic() < deadline:
        try:
            pipeline.promote_if_ready()
        except ValueError:
            break
        time.sleep(0.001)
    else:
        pytest.fail("shape-changing predictor did not fail")
    with pytest.raises(ValueError, match="shape changed"):
        pipeline.close()


def test_pipeline_surfaces_final_prediction_failure() -> None:
    pipeline = AsyncActionChunkPipeline(
        np.zeros((16, 16)), execution_steps=8, replan_after_steps=1,
        max_prediction_age_s=2.0,
    )
    pipeline.next_action()

    def fail() -> tuple[np.ndarray, float, dict[str, float]]:
        raise RuntimeError("synthetic inference failure")

    pipeline.submit(fail, anchor_generation=(7, 8, 9))
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        pipeline.close()


def test_pipeline_close_finishes_a_valid_pending_prediction() -> None:
    """A normal time-limit exit must not become a CancelledError."""

    pipeline = AsyncActionChunkPipeline(
        np.zeros((16, 16)), execution_steps=8, replan_after_steps=1,
        max_prediction_age_s=2.0,
    )
    pipeline.next_action()
    pipeline.submit(
        lambda: (np.ones((16, 16)), 1.0, {}),
        anchor_generation=(1,),
    )
    pipeline.close()


def test_stale_prediction_is_discarded_instead_of_promoted() -> None:
    pipeline = AsyncActionChunkPipeline(
        np.zeros((2, 1)),
        execution_steps=1,
        replan_after_steps=0,
        max_prediction_age_s=0.001,
    )
    pipeline.submit(
        lambda: (time.sleep(0.01) or np.ones((2, 1)), 10.0, {}),
        anchor_generation=(1,),
    )
    pipeline.next_action()
    time.sleep(0.02)
    pipeline.promote_if_ready()
    assert pipeline.stale_discard_count == 1
    assert pipeline.completed_chunks == 0
    held, index = pipeline.next_action()
    assert index is None
    np.testing.assert_allclose(held, np.zeros(1))
    pipeline.close()


def test_unfinished_prediction_exceeding_age_budget_keeps_nonblocking_hold() -> None:
    release = threading.Event()
    pipeline = AsyncActionChunkPipeline(
        np.zeros((2, 1)),
        execution_steps=1,
        replan_after_steps=0,
        max_prediction_age_s=0.01,
    )
    pipeline.submit(
        lambda: (release.wait() or np.ones((2, 1)), 10.0, {}),
        anchor_generation=(1,),
    )
    pipeline.next_action()
    time.sleep(0.02)
    started = time.monotonic()
    assert pipeline.promote_if_ready() is None
    held, index = pipeline.next_action()
    assert time.monotonic() - started < 0.05
    assert index is None
    np.testing.assert_allclose(held, np.zeros(1))
    pipeline.close(timeout_s=0.02, abort_pending=release.set)


def test_promoted_chunk_can_skip_already_consumed_prefix() -> None:
    pipeline = AsyncActionChunkPipeline(
        np.arange(16, dtype=np.float64).reshape(16, 1),
        execution_steps=16,
        replan_after_steps=12,
        max_prediction_age_s=2.0,
    )
    pipeline.skip_consumed_prefix(4)
    action, index = pipeline.next_action()
    assert index == 4
    np.testing.assert_allclose(action, [4.0])
    pipeline.skip_consumed_prefix(999)
    held, index = pipeline.next_action()
    assert index is None
    np.testing.assert_allclose(held, [4.0])
    pipeline.close()


def test_close_aborts_a_stuck_predictor_within_a_bound() -> None:
    release = threading.Event()
    pipeline = AsyncActionChunkPipeline(
        np.zeros((2, 1)),
        execution_steps=1,
        replan_after_steps=0,
        max_prediction_age_s=1.0,
    )

    def blocked() -> tuple[np.ndarray, float, dict[str, float]]:
        release.wait()
        raise EOFError("synthetic worker terminated")

    pipeline.submit(blocked, anchor_generation=(1,))
    started = time.monotonic()
    pipeline.close(timeout_s=0.02, abort_pending=release.set)
    assert time.monotonic() - started < 0.2
    assert pipeline.shutdown_aborted_prediction is True


def test_periodic_deadline_skips_to_next_grid_not_a_full_extra_period() -> None:
    assert advance_periodic_deadline(0.033, 0.040, 0.033) == pytest.approx(0.066)
    assert advance_periodic_deadline(0.033, 0.070, 0.033) == pytest.approx(0.099)


def test_cleanup_runs_every_step_and_preserves_primary_failure() -> None:
    calls: list[str] = []
    primary = RuntimeError("primary")

    def fail() -> None:
        calls.append("fail")
        raise RuntimeError("cleanup")

    run_cleanup_steps(
        [("first", fail), ("second", lambda: calls.append("second"))],
        primary_exception=primary,
    )
    assert calls == ["fail", "second"]
    assert "first: RuntimeError: cleanup" in primary.cleanup_failures


def test_cleanup_finishes_after_second_keyboard_interrupt() -> None:
    calls: list[str] = []

    def interrupt() -> None:
        calls.append("interrupt")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_cleanup_steps(
            [("interrupt", interrupt), ("release", lambda: calls.append("release"))],
            primary_exception=None,
        )
    assert calls == ["interrupt", "release"]


# ---------------------------------------------------------------- 古さは tick で数える (Issue #159 B3b-1)
#
# 大会経路は 1 tick = 教師 1 step だけ進み、tick は最大 20 Hz で推論に律速される。
# 採用は tick (lead_steps 後) で決まるのに古さを壁時計で測ると、推論が一瞬でも
# 「lead ÷ tick レート」秒は経ち、遅い tick では毎回の予測が捨てられていた。


def _wait_until_done(pipeline: AsyncActionChunkPipeline) -> None:
    deadline = time.monotonic() + 2.0
    while not pipeline._pending.done() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert pipeline._pending.done()


def test_a_slow_tick_rate_does_not_make_a_fresh_prediction_stale() -> None:
    """8 Hz 相当で 4 tick。壁時計では 0.35 s を超えるが、4 step しか経っていない。"""
    pipeline = AsyncActionChunkPipeline(
        np.zeros((8, 2)),
        execution_steps=4,
        replan_after_steps=0,
        max_prediction_age_s=0.35,
    )
    pipeline.submit(lambda: (np.ones((8, 2)), 1.0, {}), anchor_generation=(0,))
    _wait_until_done(pipeline)
    for _ in range(4):
        time.sleep(0.12)
        assert pipeline.promote_if_ready() is None
        pipeline.next_action()
    time.sleep(0.12)

    promoted = pipeline.promote_if_ready()

    assert promoted is not None, "4 tick しか経っていない予測を古いとして捨てた"
    assert pipeline.stale_discard_count == 0


def test_a_prediction_older_than_the_budget_in_ticks_is_still_discarded() -> None:
    """本当に遅れた予測 (0.35 s × 30 = 10.5 tick を超えた) は今までどおり捨てる。"""
    release = threading.Event()
    pipeline = AsyncActionChunkPipeline(
        np.zeros((16, 2)),
        execution_steps=12,
        replan_after_steps=0,
        max_prediction_age_s=0.35,
    )

    def late():
        release.wait(2.0)
        return np.ones((16, 2)), 1.0, {}

    pipeline.submit(late, anchor_generation=(0,))
    for _ in range(12):  # 壁時計はほとんど経たないが、12 step 進んだ
        pipeline.next_action()
    release.set()
    _wait_until_done(pipeline)

    assert pipeline.promote_if_ready() is None
    assert pipeline.stale_discard_count == 1
    assert pipeline.last_stale_discard_age_ticks == 12
