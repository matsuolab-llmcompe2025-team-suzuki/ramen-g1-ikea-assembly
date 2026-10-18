"""policy の先の予定の口 `preview_target` (Issue #188)。

会場の腕は指令から遅れるので、VlaSkill が腕だけを予定の少し先に差し替えて送る。
その「少し先」を、ensemble を中に持つ policy (GR00T・RAMEN-Ori・DP) が答える。
GR00T・RAMEN-Ori は model を読まずに、推論を差し替えた instance で predict を通す
(Mac の torch 無しでも回る)。
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np

from inference.desktop.lower_policy.policies.act_diffusion import ActDiffusionPolicy
from inference.desktop.lower_policy.policies.deferred import DeferredPolicy
from inference.desktop.lower_policy.policies.groot import Gr00tPolicy
from inference.desktop.lower_policy.policies.ramen_ori import RamenOriPolicy
from model.subtask_policy_training.gr00t.temporal_ensemble import (
    TargetTemporalEnsembler,
)

DIM = 19
CHUNK = np.arange(16 * DIM, dtype=np.float64).reshape(16, DIM)


def _groot(replan_family=None) -> Gr00tPolicy:
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.cfg = SimpleNamespace(
        replan_family=replan_family, mode="none", temporal_lambda=-0.1
    )
    policy._ensembler = TargetTemporalEnsembler(dim=DIM, decay_lambda=-0.1)
    policy._current_step = 0
    policy._preview_origin_step = None
    policy._last_emitted_target = None
    policy._async_hold_ticks = 0
    policy._pipeline = None
    policy._pending_submit_step = None
    policy._last_sync_metadata = {}
    policy._rtc_last_tick_ns = None
    policy._rtc_tick_deltas = deque(maxlen=32)
    policy._with_model_history = lambda obs: obs
    policy._sync_predict_chunk_19d = lambda obs, rtc_step: (
        CHUNK.astype(np.float32),
        1.0,
        np.zeros((16, 53)),
    )
    return policy


def _ramen_ori(replan_family="ramen_ori_flow_v1") -> RamenOriPolicy:
    policy = RamenOriPolicy.__new__(RamenOriPolicy)
    policy.cfg = SimpleNamespace(
        replan_family=replan_family,
        mode="overlay",
        action_space="abs",
        temporal_lambda=-0.1,
    )
    policy._ensembler = TargetTemporalEnsembler(dim=DIM, decay_lambda=-0.1)
    policy._current_step = 0
    policy._preview_origin_step = None
    policy._pipeline = None
    policy._pending_submit_step = None
    policy._last_seen_obs = None
    policy._last_rtc_metadata = {}
    policy._rtc_last_tick_ns = None
    policy._rtc_tick_deltas = deque(maxlen=32)
    policy._sync_predict_chunk_19d = lambda obs, rtc_step: (
        CHUNK.astype(np.float32),
        1.0,
    )

    def init_pipeline(seed):
        policy._pipeline = SimpleNamespace(
            deadline_miss_ticks=0, stale_discard_count=0, last_stale_discard_age_ms=None
        )

    policy._init_pipeline = init_pipeline
    return policy


def test_groot_previews_from_the_step_it_just_emitted():
    policy = _groot()
    assert policy.preview_target(6) is None  # まだ何も返していない
    action = policy.predict(SimpleNamespace())
    np.testing.assert_allclose(action.action_chunk[0], CHUNK[0])
    target, ahead = policy.preview_target(6)
    assert ahead == 6
    np.testing.assert_allclose(target, CHUNK[6])
    assert policy.preview_target(0) is None


def test_groot_does_not_preview_while_holding_the_last_target():
    policy = _groot()
    policy.predict(SimpleNamespace())
    policy._ensembler.reset()  # 予定が尽きた = 保持する tick
    policy._sync_predict_chunk_19d = lambda obs, rtc_step: (
        np.zeros((0, DIM), dtype=np.float32),
        1.0,
        np.zeros((0, 53)),
    )
    action = policy.predict(SimpleNamespace())
    assert action.metadata["chunk_source"] == "async_hold"
    assert policy.preview_target(6) is None


def test_ramen_ori_async_previews_from_the_emitted_step():
    policy = _ramen_ori()
    action = policy.predict(SimpleNamespace(skill_id=4))
    np.testing.assert_allclose(action.action_chunk[0], CHUNK[0])
    target, ahead = policy.preview_target(6)
    assert ahead == 6
    np.testing.assert_allclose(target, CHUNK[6])
    policy._rtc_leftover = None
    policy._rtc_delay = SimpleNamespace(reset=lambda: None)
    policy._memory_tracker = None
    policy.reset()
    assert policy.preview_target(6) is None


def test_ramen_ori_sync_path_leaves_the_preview_to_the_chunk_rows():
    """同期経路は chunk 全体を返すので、VlaSkill が chunk の行から先を読む。"""
    policy = _ramen_ori(replan_family=None)
    action = policy.predict(SimpleNamespace(skill_id=4))
    assert action.action_chunk.shape == (16, DIM)
    assert policy.preview_target(6) is None


def test_act_diffusion_and_deferred_delegate():
    executor = SimpleNamespace(preview_target=lambda steps: (CHUNK[steps], steps))
    policy = ActDiffusionPolicy.__new__(ActDiffusionPolicy)
    policy._executor = executor
    target, ahead = policy.preview_target(3)
    assert ahead == 3
    np.testing.assert_allclose(target, CHUNK[3])

    deferred = DeferredPolicy(ActDiffusionPolicy, label="dp", loader=lambda: policy)
    assert (
        deferred.preview_target(3) is None
    )  # 読み込み前は答えない (読み込みも起こさない)
    assert deferred.is_loaded is False
    deferred.prepare()
    target, ahead = deferred.preview_target(3)
    assert ahead == 3
    deferred_without = DeferredPolicy(object, label="x", loader=lambda: object())
    deferred_without.prepare()
    assert deferred_without.preview_target(3) is None
