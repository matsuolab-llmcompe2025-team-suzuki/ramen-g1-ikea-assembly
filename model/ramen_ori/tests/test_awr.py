"""H-7 AWR twist の unit test (Issue #120 Alt-6)。

- compute_duration_advantage: 3 method (inverse / z_score / rank) の分布性質
- broadcast_to_frames: ep-level weight → frame-level weight
- CurriculumSampler + advantage_weights の統合 (Alt-5 + Alt-6)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.ramen_ori.awr import broadcast_to_frames, compute_duration_advantage
from model.ramen_ori.curriculum import CurriculumSampler, CurriculumStage


class TestComputeDurationAdvantageInverse:
    def test_shorter_ep_higher_weight(self) -> None:
        lengths = np.array([100, 200, 400], dtype=np.int64)
        w = compute_duration_advantage(lengths, method="inverse")
        # ep 0 (100 frames) が一番高い weight
        assert w[0] > w[1] > w[2]

    def test_mean_normalized_to_1(self) -> None:
        lengths = np.array([50, 100, 150, 200], dtype=np.int64)
        w = compute_duration_advantage(lengths, method="inverse")
        assert w.mean() == pytest.approx(1.0)

    def test_uniform_length_yields_uniform_weight(self) -> None:
        lengths = np.array([100, 100, 100], dtype=np.int64)
        w = compute_duration_advantage(lengths, method="inverse")
        np.testing.assert_allclose(w, np.ones(3))


class TestComputeDurationAdvantageZScore:
    def test_shorter_ep_higher_weight(self) -> None:
        lengths = np.array([100, 200, 400], dtype=np.int64)
        w = compute_duration_advantage(lengths, method="z_score", temperature=1.0)
        assert w[0] > w[1] > w[2]
        assert w.mean() == pytest.approx(1.0)

    def test_uniform_length_yields_uniform_weight(self) -> None:
        lengths = np.array([100, 100, 100], dtype=np.int64)
        w = compute_duration_advantage(lengths, method="z_score")
        np.testing.assert_allclose(w, np.ones(3))

    def test_high_temperature_flatter(self) -> None:
        """temperature 大きくすると uniform 寄り (weight 差が小さくなる)。"""
        lengths = np.array([100, 200, 400], dtype=np.int64)
        w_low_t = compute_duration_advantage(lengths, method="z_score", temperature=0.5)
        w_high_t = compute_duration_advantage(lengths, method="z_score", temperature=5.0)
        # low t: 差が大きい、high t: 差が小さい
        assert (w_low_t.max() - w_low_t.min()) > (w_high_t.max() - w_high_t.min())


class TestComputeDurationAdvantageRank:
    def test_shorter_ep_higher_weight(self) -> None:
        lengths = np.array([50, 200, 100, 400], dtype=np.int64)
        w = compute_duration_advantage(lengths, method="rank")
        # 短い順: 50 (idx 0), 100 (idx 2), 200 (idx 1), 400 (idx 3)
        assert w[0] > w[2] > w[1] > w[3]
        assert w.mean() == pytest.approx(1.0)


class TestComputeDurationAdvantageErrors:
    def test_zero_length_raises(self) -> None:
        with pytest.raises(ValueError, match="episode_lengths must be > 0"):
            compute_duration_advantage(np.array([100, 0, 50]))

    def test_negative_length_raises(self) -> None:
        with pytest.raises(ValueError, match="episode_lengths must be > 0"):
            compute_duration_advantage(np.array([100, -50, 50]))

    def test_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D"):
            compute_duration_advantage(np.array([[100, 200], [50, 150]]))

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            compute_duration_advantage(np.array([100]), method="foo")

    def test_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be"):
            compute_duration_advantage(np.array([100]), method="z_score", temperature=0)


class TestBroadcastToFrames:
    def test_basic_broadcast(self) -> None:
        ep_w = np.array([2.0, 0.5], dtype=np.float64)
        lens = np.array([3, 2], dtype=np.int64)
        frame_w = broadcast_to_frames(ep_w, lens)
        np.testing.assert_array_equal(frame_w, [2.0, 2.0, 2.0, 0.5, 0.5])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            broadcast_to_frames(np.array([1.0, 2.0]), np.array([5, 3, 2]))

    def test_frame_count_sums_correctly(self) -> None:
        ep_w = np.array([1.0, 1.0, 1.0])
        lens = np.array([10, 20, 30])
        frame_w = broadcast_to_frames(ep_w, lens)
        assert len(frame_w) == 60


class TestSamplerAdvantageIntegration:
    def test_curriculum_sampler_with_advantage_weights(self) -> None:
        """CurriculumSampler に advantage_weights inject、high-weight sample が多く選ばれる。"""
        skill_ids = np.array([5, 5, 5, 5, 5, 5], dtype=np.int64)  # 全 skill=5
        # ep 0: 3 frame, weight = 3.0 (高)、ep 1: 3 frame, weight = 0.1 (低)
        ep_lengths = np.array([3, 3])
        ep_w = np.array([3.0, 0.1])
        frame_adv = broadcast_to_frames(ep_w, ep_lengths)
        stages = [CurriculumStage(step=0, skills=[5])]
        sampler = CurriculumSampler(
            skill_ids_per_sample=skill_ids,
            stages=stages,
            step_ref=lambda: 0,
            num_samples_per_epoch=10000,
            generator=torch.Generator().manual_seed(0),
            advantage_weights=frame_adv,
        )
        indices = list(sampler)
        # ep 0 (index 0..2) が ep 1 (index 3..5) より圧倒的に多く出るはず
        count_ep0 = sum(1 for i in indices if i < 3)
        count_ep1 = sum(1 for i in indices if i >= 3)
        assert count_ep0 > count_ep1 * 20  # weight 差 30x なので少なくとも 20x 差

    def test_curriculum_sampler_advantage_none_equals_no_advantage(self) -> None:
        """advantage_weights=None のとき Alt-5 と挙動一致。"""
        skill_ids = np.array([5, 5, 5], dtype=np.int64)
        stages = [CurriculumStage(step=0, skills=[5])]
        s_no = CurriculumSampler(skill_ids, stages, step_ref=lambda: 0,
                                 num_samples_per_epoch=50,
                                 generator=torch.Generator().manual_seed(0))
        s_adv = CurriculumSampler(skill_ids, stages, step_ref=lambda: 0,
                                  num_samples_per_epoch=50,
                                  generator=torch.Generator().manual_seed(0),
                                  advantage_weights=np.ones(3))
        # 全 1 の advantage は影響なし → 同 seed で同一分布
        assert list(s_no) == list(s_adv)

    def test_shape_mismatch_raises(self) -> None:
        skill_ids = np.array([5, 5, 5])
        stages = [CurriculumStage(step=0, skills=[5])]
        with pytest.raises(ValueError, match="advantage_weights shape"):
            CurriculumSampler(skill_ids, stages, step_ref=lambda: 0,
                              advantage_weights=np.array([1.0, 2.0]))

    def test_negative_advantage_raises(self) -> None:
        skill_ids = np.array([5, 5, 5])
        stages = [CurriculumStage(step=0, skills=[5])]
        with pytest.raises(ValueError, match="non-negative"):
            CurriculumSampler(skill_ids, stages, step_ref=lambda: 0,
                              advantage_weights=np.array([1.0, -0.5, 2.0]))
