"""TargetTemporalEnsembler port + 19D adaptation の verify (Issue #128 Phase 2)。

issue-70-flip-table-data-augmentation branch の PhysicalTargetTemporalEnsembler
(dim=16 固定) を dim=19 (waist3 + arms14 + hand2) にパラメタ化して port した
実装の contract test。

## テスト設計

- **単一 candidate**: add 1 chunk → target[k] は chunk[k-origin] そのもの
- **複数 candidate**: 2 chunk overlap → weighted mean が正しい
- **decay_lambda=None**: blend disable、最新 candidate のみ返す
- **prune**: target(k) 呼出後、k 未満の step は破棄される
- **reset**: 全 candidate クリア
- **input validation**: shape / NaN の rejection
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from model.subtask_policy_training.gr00t.temporal_ensemble import (
    TargetTemporalEnsembler,
)


DIM = 19  # VlaSkill contract (waist3 + arms14 + hand2)


class TestConstructor:
    def test_default_decay_lambda_is_neg_zero_one(self) -> None:
        # issue-70 検証値 -0.1 を module default に据える (実装は明示的引数、
        # ここは PolicyConfig default と揃ってることの確認は別 test)。
        # ensembler 自体の default 引数は -0.1
        ens = TargetTemporalEnsembler(dim=DIM)
        assert ens.decay_lambda == -0.1

    def test_can_disable_blend_with_none(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM, decay_lambda=None)
        assert ens.decay_lambda is None

    def test_rejects_non_positive_dim(self) -> None:
        with pytest.raises(ValueError):
            TargetTemporalEnsembler(dim=0)
        with pytest.raises(ValueError):
            TargetTemporalEnsembler(dim=-1)

    def test_rejects_non_finite_lambda(self) -> None:
        with pytest.raises(ValueError):
            TargetTemporalEnsembler(dim=DIM, decay_lambda=float("nan"))
        with pytest.raises(ValueError):
            TargetTemporalEnsembler(dim=DIM, decay_lambda=float("inf"))


class TestSingleChunkFlow:
    def test_target_from_single_chunk_at_origin(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM)
        chunk = np.arange(3 * DIM, dtype=np.float64).reshape(3, DIM)
        ens.add_chunk(origin_step=10, absolute_targets=chunk)
        # step 10 = chunk[0]
        got = ens.target(step=10)
        assert np.allclose(got, chunk[0])

    def test_target_from_single_chunk_at_offset(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM)
        chunk = np.arange(4 * DIM, dtype=np.float64).reshape(4, DIM)
        ens.add_chunk(origin_step=5, absolute_targets=chunk)
        # step 5+2 = chunk[2]
        got = ens.target(step=7)
        assert np.allclose(got, chunk[2])

    def test_target_missing_step_raises(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM)
        chunk = np.zeros((2, DIM), dtype=np.float64)
        ens.add_chunk(origin_step=0, absolute_targets=chunk)
        with pytest.raises(KeyError, match="step 99"):
            ens.target(step=99)


class TestBlendWeights:
    """複数 chunk が overlap する時の weighted mean が正しいこと。"""

    def test_two_candidates_use_exp_decay_weights(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM, decay_lambda=-0.1)
        # 古い chunk (origin=0): step 5 の予測 = value 1.0 (全 dim)
        old_chunk = np.ones((10, DIM), dtype=np.float64) * 1.0
        ens.add_chunk(origin_step=0, absolute_targets=old_chunk)
        # 新 chunk (origin=5): step 5 の予測 = value 5.0
        new_chunk = np.ones((5, DIM), dtype=np.float64) * 5.0
        ens.add_chunk(origin_step=5, absolute_targets=new_chunk)

        # 期待 weighted mean at step 5:
        #   old: age=5, weight = exp(-0.1 * 5) = 0.6065...
        #   new: age=0, weight = exp(0) = 1.0
        #   result = (0.6065 * 1.0 + 1.0 * 5.0) / (0.6065 + 1.0) = 5.618... / 1.6065
        w_old = math.exp(-0.1 * 5)
        w_new = math.exp(0)
        expected = (w_old * 1.0 + w_new * 5.0) / (w_old + w_new)

        got = ens.target(step=5)
        assert np.allclose(got, expected, atol=1e-9), (
            f"step 5 blend: got {got[0]:.6f}, expected {expected:.6f}"
        )

    def test_three_candidates_all_weighted(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM, decay_lambda=-0.1)
        # 3 chunks all overlapping step 10, targets 1.0 / 2.0 / 3.0
        ens.add_chunk(origin_step=0, absolute_targets=np.ones((11, DIM)) * 1.0)
        ens.add_chunk(origin_step=5, absolute_targets=np.ones((11, DIM)) * 2.0)
        ens.add_chunk(origin_step=10, absolute_targets=np.ones((11, DIM)) * 3.0)

        w = [math.exp(-0.1 * age) for age in (10, 5, 0)]
        expected = sum(wi * vi for wi, vi in zip(w, [1.0, 2.0, 3.0])) / sum(w)

        got = ens.target(step=10)
        assert np.allclose(got, expected, atol=1e-9)

    def test_decay_lambda_none_returns_latest(self) -> None:
        # decay_lambda=None: blend disable、origin_step 最大 (最新) を採用
        ens = TargetTemporalEnsembler(dim=DIM, decay_lambda=None)
        ens.add_chunk(origin_step=0, absolute_targets=np.ones((5, DIM)) * 1.0)
        ens.add_chunk(origin_step=2, absolute_targets=np.ones((5, DIM)) * 5.0)
        # step 3 に対して: origin=0 の chunk[3]=1.0、origin=2 の chunk[1]=5.0
        # → 最新 origin=2 を採用 → 5.0
        got = ens.target(step=3)
        assert np.allclose(got, 5.0)

    def test_zero_decay_lambda_gives_equal_weights(self) -> None:
        # decay_lambda=0 = 全 candidate 等重み (単純平均)
        ens = TargetTemporalEnsembler(dim=DIM, decay_lambda=0.0)
        ens.add_chunk(origin_step=0, absolute_targets=np.ones((5, DIM)) * 1.0)
        ens.add_chunk(origin_step=2, absolute_targets=np.ones((5, DIM)) * 5.0)
        # step 3: 2 candidates、equal weight → (1.0 + 5.0) / 2 = 3.0
        got = ens.target(step=3)
        assert np.allclose(got, 3.0)


class TestPruneBehavior:
    """target(step) 呼出後、step 未満の candidate が破棄される (メモリ枠固定)。"""

    def test_query_step_drops_older_steps(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM)
        chunk = np.zeros((10, DIM), dtype=np.float64)
        ens.add_chunk(origin_step=0, absolute_targets=chunk)
        # 予備: 全 step 存在
        assert ens.candidate_count(step=0) == 1
        assert ens.candidate_count(step=5) == 1
        # target(5) 呼出後、step<5 は破棄
        _ = ens.target(step=5)
        assert ens.candidate_count(step=0) == 0
        assert ens.candidate_count(step=4) == 0
        # step>=5 は残る
        assert ens.candidate_count(step=5) == 1
        assert ens.candidate_count(step=9) == 1

    def test_same_step_query_reuses_remaining_candidates(self) -> None:
        # target(k) 呼出後、step=k の candidate は残る (同 step 再問い合わせ可能)
        ens = TargetTemporalEnsembler(dim=DIM)
        ens.add_chunk(origin_step=0, absolute_targets=np.ones((3, DIM)) * 1.0)
        first = ens.target(step=1)
        second = ens.target(step=1)
        assert np.allclose(first, second)


class TestReset:
    def test_reset_clears_all_candidates(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM)
        ens.add_chunk(origin_step=0, absolute_targets=np.zeros((5, DIM)))
        ens.add_chunk(origin_step=10, absolute_targets=np.zeros((5, DIM)))
        ens.reset()
        for step in (0, 5, 10, 14):
            assert ens.candidate_count(step) == 0
        # target 呼び出しも KeyError
        with pytest.raises(KeyError):
            ens.target(step=0)


class TestInputValidation:
    def test_rejects_wrong_dim_chunk(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM)
        with pytest.raises(ValueError, match=str(DIM)):
            ens.add_chunk(origin_step=0, absolute_targets=np.zeros((3, DIM + 1)))
        with pytest.raises(ValueError, match=str(DIM)):
            ens.add_chunk(origin_step=0, absolute_targets=np.zeros((3, DIM - 1)))

    def test_rejects_1d_targets(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM)
        with pytest.raises(ValueError):
            ens.add_chunk(origin_step=0, absolute_targets=np.zeros(DIM))

    def test_rejects_nan_targets(self) -> None:
        ens = TargetTemporalEnsembler(dim=DIM)
        bad = np.zeros((3, DIM))
        bad[1, 5] = float("nan")
        with pytest.raises(ValueError):
            ens.add_chunk(origin_step=0, absolute_targets=bad)


class TestOverlappingChunksScenario:
    """実 GR00T 使用シナリオ相当 (30Hz predict、chunk_len=16 overlap)。"""

    def test_streaming_add_predict_pattern(self) -> None:
        # 30 tick 分 predict、各 tick で origin=t の chunk_len=16 を add、
        # ensembler.target(t) を取得 (VlaSkill.step の per-tick pattern)。
        # 各 tick で不同定な値 (t) を chunk に入れて、blend が期待通り効くこと。
        # chunk[k] = t + k * 0.01 で全 dim 同値。
        # step t での候補 = {(origin=t', value=t' + (t-t')*0.01) : t-15 <= t' <= t}
        # min value = 最古 origin での予測、max value = 最新 origin での予測
        # weighted mean は必ず [min, max] 内に入る (数値的自明)。
        ens = TargetTemporalEnsembler(dim=DIM, decay_lambda=-0.1)
        prev_target_0 = None
        for t in range(30):
            chunk = np.zeros((16, DIM), dtype=np.float64)
            for k in range(16):
                chunk[k, :] = t + k * 0.01
            ens.add_chunk(origin_step=t, absolute_targets=chunk)
            target = ens.target(step=t)
            # step t 候補 (t' ∈ [max(0, t-15), t]) の value 範囲を計算
            oldest_t = max(0, t - 15)
            oldest_value = oldest_t + (t - oldest_t) * 0.01
            newest_value = t + 0.0
            lo, hi = min(oldest_value, newest_value), max(oldest_value, newest_value)
            assert lo - 1e-9 <= target[0] <= hi + 1e-9, (
                f"tick {t}: blend {target[0]:.4f} outside candidate range "
                f"[{lo:.4f}, {hi:.4f}]"
            )
            # 全 dim uniform に blend されてる (broadcast bug 検出)
            assert np.allclose(target, target[0])
            # blend が最新方向へ引っ張られる (decay_lambda<0 の効果): 単調増加
            if prev_target_0 is not None:
                assert target[0] > prev_target_0, (
                    f"tick {t}: {target[0]:.4f} <= prev {prev_target_0:.4f} "
                    f"(decay_lambda<0 なら最新方向へ引っ張られるはず)"
                )
            prev_target_0 = target[0]

    def test_reset_between_skill_transitions(self) -> None:
        # skill A で 5 tick 走らせて reset、skill B で 1 tick 走らせたら
        # blend は skill B の 1 candidate のみに戻ってること
        ens = TargetTemporalEnsembler(dim=DIM, decay_lambda=-0.1)
        for t in range(5):
            ens.add_chunk(origin_step=t, absolute_targets=np.ones((16, DIM)) * 100.0)
            ens.target(step=t)
        ens.reset()
        ens.add_chunk(origin_step=0, absolute_targets=np.ones((16, DIM)) * 5.0)
        got = ens.target(step=0)
        # 前 skill の 100.0 が漏れず、5.0 単独になってること
        assert np.allclose(got, 5.0)
        assert ens.candidate_count(step=0) == 1


# ---------------------------------------------------------------- peek_ahead (Issue #188)
#
# 会場の腕は指令から遅れるので、少し先の予定を今送る。覗いても候補を捨てない
# (次の tick は今の step+1 を target() で読む)。


def _numbered_chunk(length: int) -> np.ndarray:
    return np.arange(length * DIM, dtype=np.float64).reshape(length, DIM)


def test_peek_ahead_returns_the_future_step_without_discarding():
    ensembler = TargetTemporalEnsembler(dim=DIM, decay_lambda=-0.1)
    chunk = _numbered_chunk(10)
    ensembler.add_chunk(origin_step=0, absolute_targets=chunk)
    ensembler.target(step=0)
    target, ahead = ensembler.peek_ahead(0, 3)
    assert ahead == 3
    np.testing.assert_allclose(target, chunk[3])
    np.testing.assert_allclose(ensembler.target(step=1), chunk[1])
    assert ensembler.candidate_count(2) == 1


def test_peek_ahead_falls_back_to_the_furthest_step_it_has():
    ensembler = TargetTemporalEnsembler(dim=DIM, decay_lambda=-0.1)
    chunk = _numbered_chunk(4)
    ensembler.add_chunk(origin_step=0, absolute_targets=chunk)
    ensembler.target(step=0)
    target, ahead = ensembler.peek_ahead(0, 6)
    assert ahead == 3
    np.testing.assert_allclose(target, chunk[3])
    assert ensembler.peek_ahead(3, 2) is None


def test_peek_ahead_blends_like_target_will_on_that_step():
    ensembler = TargetTemporalEnsembler(dim=DIM, decay_lambda=-0.1)
    ensembler.add_chunk(origin_step=0, absolute_targets=_numbered_chunk(10))
    ensembler.add_chunk(origin_step=2, absolute_targets=_numbered_chunk(10) + 100.0)
    ensembler.target(step=2)
    peeked, ahead = ensembler.peek_ahead(2, 3)
    assert ahead == 3
    for step in (3, 4):
        ensembler.target(step=step)
    np.testing.assert_allclose(peeked, ensembler.target(step=5))
