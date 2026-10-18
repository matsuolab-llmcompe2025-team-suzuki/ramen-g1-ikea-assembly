"""RAMEN-Ori: 古いとして捨てられた予測の後も async 再計画が続くこと (Issue #159 B3d-01)。

pipeline は古い予測を捨てると自分の `_pending` を消すが、policy 側の latch
(`_pending_submit_step`) を外すのは promote が成功したときだけだった。1 回捨てると
`wants_prediction` が立っても submit せず、その skill の間 async 再計画が止まり、
seed を使い切るたびに tick を塞ぐ同期推論 (sync_fallback) だけで動いていた。
GR00T には同じ PR #138 で解除が入っていて、RAMEN-Ori にだけ無かった。

torch は要らない (推論関数だけ差し替え、pipeline / temporal ensemble は本物)。
"""

from __future__ import annotations

import threading
import time

import numpy as np

from inference.desktop.lower_policy.policies.base import PolicyConfig
from inference.desktop.lower_policy.policies.ramen_ori import (
    CAMERAS_3CAM_PHASE_K,
    RamenOriPolicy,
)

_CHUNK = 32


def _shipped_policy(first_async: threading.Event):
    """会場の slot (rotate_table_base_ramen_ori_venue_all6_20k) と同じ実行設定。

    最初の async 予測だけ `first_async` が立つまで返らない。
    """
    cfg = PolicyConfig(
        mode="overlay",
        ckpt_ref="x",
        dtype="fp32",
        cams=CAMERAS_3CAM_PHASE_K,
        overlay_jpeg_subsampling="4:4:4",
        action_space="abs",
        skill_id=4,
        replan_family="ramen_ori_flow_v1",
        execution_steps=8,
        temporal_lambda=-0.1,
    )
    policy = RamenOriPolicy(cfg)
    policy._model = object()
    calls = {"n": 0}

    def inner(_obs, *, rtc_step=None):
        calls["n"] += 1
        if calls["n"] == 2:  # 1 本目 = sync の seed、2 本目 = 最初の async
            first_async.wait(5.0)
        values = np.tile(
            (np.arange(_CHUNK, dtype=np.float32) + float(rtc_step))[:, None], (1, 19)
        )
        return values, 5.0

    policy._in_process_predict_chunk_19d = inner
    return policy


def _wait_for_the_pending_prediction(policy) -> None:
    pending = policy._pipeline._pending
    deadline = time.monotonic() + 2.0
    while pending is not None and not pending.done() and time.monotonic() < deadline:
        time.sleep(0.001)


def test_async_replanning_resumes_after_a_stale_discard():
    first_async = threading.Event()
    policy = _shipped_policy(first_async)
    obs = policy._make_dummy_observation()
    sources: dict[str, int] = {}
    try:
        # 最初の async 予測を 12 tick 返さない (max age 0.25 s × 30 = 7.5 tick を超える)
        for _ in range(18):
            source = policy.predict(obs).metadata["chunk_source"]
            sources[source] = sources.get(source, 0) + 1
        first_async.set()
        _wait_for_the_pending_prediction(policy)
        for _ in range(80):
            source = policy.predict(obs).metadata["chunk_source"]
            sources[source] = sources.get(source, 0) + 1
            _wait_for_the_pending_prediction(policy)

        pipeline = policy._pipeline
        assert pipeline.stale_discard_count == 1, sources
        assert pipeline.completed_chunks >= 5, (
            f"捨てた後に async 再計画が戻っていない: promoted={pipeline.completed_chunks} "
            f"sources={sources} pending_submit_step={policy._pending_submit_step}"
        )
    finally:
        first_async.set()
        policy.reset()
