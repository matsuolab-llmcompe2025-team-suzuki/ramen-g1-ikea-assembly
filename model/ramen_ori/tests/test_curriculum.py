"""H-3 Curriculum sampler の unit test (Issue #120 Alt-5)。

- CurriculumStage の validation (step, skills)
- resolve_active_skills の step lookup
- compute_stage_weights の active per-skill uniform 分布
- CurriculumSampler の __iter__ 再計算 (step 変化で active 切替)
- Integration: 2 stage で 1 epoch 目/2 epoch 目の分布差
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.ramen_ori.curriculum import (
    CurriculumSampler,
    CurriculumStage,
    compute_stage_weights,
    resolve_active_skills,
)


class TestCurriculumStage:
    def test_valid_stage(self) -> None:
        s = CurriculumStage(step=0, skills=[5])
        assert s.step == 0 and s.skills == [5]

    def test_negative_step_raises(self) -> None:
        with pytest.raises(ValueError, match="step must be"):
            CurriculumStage(step=-1, skills=[5])

    def test_empty_skills_raises(self) -> None:
        with pytest.raises(ValueError, match="skills must be non-empty"):
            CurriculumStage(step=0, skills=[])


class TestResolveActiveSkills:
    def test_single_stage_before_start(self) -> None:
        stages = [CurriculumStage(step=100, skills=[5])]
        # current_step < first stage → まだ stage 0 に到達してない → stages[0] 採用 (Phase 0)
        active = resolve_active_skills(current_step=0, stages=stages)
        assert active == {5}

    def test_advances_through_stages(self) -> None:
        stages = [
            CurriculumStage(step=0, skills=[5]),
            CurriculumStage(step=100, skills=[4, 5]),
            CurriculumStage(step=200, skills=[0, 4, 5]),
        ]
        assert resolve_active_skills(0, stages) == {5}
        assert resolve_active_skills(50, stages) == {5}
        assert resolve_active_skills(100, stages) == {4, 5}
        assert resolve_active_skills(199, stages) == {4, 5}
        assert resolve_active_skills(200, stages) == {0, 4, 5}
        assert resolve_active_skills(9999, stages) == {0, 4, 5}

    def test_empty_stages_raises(self) -> None:
        with pytest.raises(ValueError, match="stages must be non-empty"):
            resolve_active_skills(0, [])


class TestComputeStageWeights:
    def test_uniform_within_active(self) -> None:
        # 3 sample of skill 5, 2 sample of skill 7, 1 sample of skill 0
        skill_ids = np.array([5, 5, 5, 7, 7, 0], dtype=np.int64)
        weights = compute_stage_weights(skill_ids, active_skills={5, 7})
        # active per-skill 合計 = 1/2 (2 active skills)
        assert weights[skill_ids == 5].sum() == pytest.approx(0.5)
        assert weights[skill_ids == 7].sum() == pytest.approx(0.5)
        # inactive skill (0) は 0
        assert weights[skill_ids == 0].sum() == 0.0

    def test_single_active_skill(self) -> None:
        skill_ids = np.array([5, 5, 7, 7], dtype=np.int64)
        weights = compute_stage_weights(skill_ids, active_skills={5})
        assert weights[skill_ids == 5].sum() == pytest.approx(1.0)
        assert weights[skill_ids == 7].sum() == 0.0

    def test_active_skill_with_zero_samples_raises(self) -> None:
        skill_ids = np.array([5, 5, 7], dtype=np.int64)
        with pytest.raises(ValueError, match="skill_id=0 has 0 samples"):
            compute_stage_weights(skill_ids, active_skills={0, 5})

    def test_empty_active_raises(self) -> None:
        skill_ids = np.array([5, 7], dtype=np.int64)
        with pytest.raises(ValueError, match="active_skills is empty"):
            compute_stage_weights(skill_ids, active_skills=set())


class TestCurriculumSampler:
    def test_iter_returns_only_active_indices(self) -> None:
        skill_ids = np.array([5, 5, 5, 5, 7, 7, 7, 7], dtype=np.int64)
        stages = [CurriculumStage(step=0, skills=[5])]
        sampler = CurriculumSampler(
            skill_ids_per_sample=skill_ids,
            stages=stages,
            step_ref=lambda: 0,
            num_samples_per_epoch=100,
            generator=torch.Generator().manual_seed(0),
        )
        indices = list(sampler)
        assert len(indices) == 100
        # active skill = {5} なので index 0..3 (skill=5) だけ出るはず
        for idx in indices:
            assert skill_ids[idx] == 5

    def test_stages_sorted_by_step(self) -> None:
        skill_ids = np.array([5, 7], dtype=np.int64)
        stages_unsorted = [
            CurriculumStage(step=100, skills=[5, 7]),
            CurriculumStage(step=0, skills=[5]),
        ]
        sampler = CurriculumSampler(
            skill_ids_per_sample=skill_ids,
            stages=stages_unsorted,
            step_ref=lambda: 0,
            num_samples_per_epoch=10,
        )
        assert sampler.stages[0].step == 0
        assert sampler.stages[1].step == 100

    def test_stage_transition_on_step_change(self) -> None:
        """step_ref の値変化で active skills が切替、次 __iter__ に反映。"""
        skill_ids = np.array([5, 5, 7, 7], dtype=np.int64)
        stages = [
            CurriculumStage(step=0, skills=[5]),
            CurriculumStage(step=100, skills=[5, 7]),
        ]
        current = {"v": 0}
        sampler = CurriculumSampler(
            skill_ids_per_sample=skill_ids,
            stages=stages,
            step_ref=lambda: current["v"],
            num_samples_per_epoch=200,
            generator=torch.Generator().manual_seed(0),
        )
        # Stage 1: skill=5 のみ (index 0, 1)
        indices_s1 = list(sampler)
        for i in indices_s1:
            assert skill_ids[i] == 5, f"stage 1 should sample only skill=5, got {skill_ids[i]}"

        # step 進めて stage 2
        current["v"] = 150
        indices_s2 = list(sampler)
        skills_s2 = set(int(skill_ids[i]) for i in indices_s2)
        # active = {5, 7} なので両方の skill が出るはず (100 sample あれば十分)
        assert skills_s2 == {5, 7}, f"stage 2 should sample both 5 and 7, got {skills_s2}"

    def test_len_matches_num_samples_per_epoch(self) -> None:
        skill_ids = np.array([5, 7], dtype=np.int64)
        stages = [CurriculumStage(step=0, skills=[5, 7])]
        sampler = CurriculumSampler(
            skill_ids_per_sample=skill_ids, stages=stages, step_ref=lambda: 0,
            num_samples_per_epoch=42,
        )
        assert len(sampler) == 42

    def test_default_num_samples_is_dataset_size(self) -> None:
        skill_ids = np.array([5, 5, 5, 7, 7], dtype=np.int64)
        stages = [CurriculumStage(step=0, skills=[5, 7])]
        sampler = CurriculumSampler(
            skill_ids_per_sample=skill_ids, stages=stages, step_ref=lambda: 0,
        )
        assert len(sampler) == 5

    def test_empty_stages_raises(self) -> None:
        with pytest.raises(ValueError, match="stages must be non-empty"):
            CurriculumSampler(
                skill_ids_per_sample=np.array([5]), stages=[], step_ref=lambda: 0,
            )
