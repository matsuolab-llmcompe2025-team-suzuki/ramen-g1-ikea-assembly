"""normalization の unit test (Issue #141 RO-4 / RO-6)。

- skill ごとに同じ重みのまとめ方 (frame 数で混ぜない)、std の下限、分割して貯めても同じ
- Normalizer の往復、統計を入れる前はエラー、state_dict で統計が保存・復元される
- compute_normalization_moments が dataset の state_action_item を skill ごとに集計する (FK の量も)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.ramen_ori.normalization import (
    Normalizer,
    SkillBalancedMoments,
    compute_normalization_moments,
    normalized_std_by_skill,
)


class TestSkillBalancedMoments:
    def test_single_skill_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(2.0, 3.0, size=(500, 3))
        m = SkillBalancedMoments(3)
        m.update(x, np.full(500, 4))
        mean, std = m.finalize(std_min=1e-3)
        np.testing.assert_allclose(mean, x.mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(std, x.std(axis=0), rtol=1e-5)

    def test_skills_are_weighted_equally_not_by_frame_count(self) -> None:
        """frame 数が 100 倍違っても、各 skill が 1/2 ずつ効く (β で insert が統計を決めない)。"""
        rng = np.random.default_rng(1)
        a = rng.normal(0.0, 1.0, size=(1000, 2))
        b = rng.normal(10.0, 2.0, size=(10, 2))
        m = SkillBalancedMoments(2)
        m.update(np.concatenate([a, b]), np.array([0] * 1000 + [4] * 10))
        mean, std = m.finalize(std_min=1e-3)
        # 同じ重みの分布 = 各 frame に 1 / (2 n_skill) の重み
        w = np.array([1 / 2000] * 1000 + [1 / 20] * 10)
        x = np.concatenate([a, b])
        exp_mean = (w[:, None] * x).sum(axis=0)
        exp_std = np.sqrt((w[:, None] * (x - exp_mean) ** 2).sum(axis=0))
        np.testing.assert_allclose(mean, exp_mean, rtol=1e-5)
        np.testing.assert_allclose(std, exp_std, rtol=1e-5)
        assert mean[0] > 4.0  # frame 数で混ぜると 0.1 付近

    def test_constant_dim_gets_std_min(self) -> None:
        x = np.stack([np.zeros(100), np.linspace(-1, 1, 100)], axis=1)
        m = SkillBalancedMoments(2)
        m.update(x, np.zeros(100, dtype=int))
        _, std = m.finalize(std_min=1e-3)
        assert std[0] == pytest.approx(1e-3)
        assert std[1] > 0.5

    def test_chunked_updates_equal_single_update(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=(300, 4))
        s = rng.integers(0, 3, size=300)
        m1 = SkillBalancedMoments(4)
        m1.update(x, s)
        m2 = SkillBalancedMoments(4)
        for i in range(0, 300, 7):
            m2.update(x[i:i + 7], s[i:i + 7])
        for a, b in zip(m1.finalize(1e-3), m2.finalize(1e-3), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-6)
        assert m1.counts == m2.counts

    def test_normalized_std_by_skill(self) -> None:
        rng = np.random.default_rng(3)
        m = SkillBalancedMoments(1)
        m.update(rng.normal(0, 1, size=(4000, 1)), np.zeros(4000, dtype=int))
        m.update(rng.normal(0, 3, size=(4000, 1)), np.ones(4000, dtype=int))
        _, std = m.finalize(1e-3)
        nstd = normalized_std_by_skill(m, std)
        # 全体の分散 = (1 + 9) / 2 = 5 → skill 0 は 1/√5、skill 1 は 3/√5
        assert nstd[0][0] == pytest.approx(1 / np.sqrt(5), rel=0.05)
        assert nstd[1][0] == pytest.approx(3 / np.sqrt(5), rel=0.05)

    def test_wrong_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="must be"):
            SkillBalancedMoments(3).update(np.zeros((5, 2)), np.zeros(5))


class TestNormalizer:
    def test_roundtrip(self) -> None:
        n = Normalizer(3)
        n.set_stats([1.0, -2.0, 0.5], [2.0, 0.1, 4.0])
        x = torch.randn(5, 7, 3)
        torch.testing.assert_close(n.denormalize(n.normalize(x)), x)
        torch.testing.assert_close(
            n.normalize(torch.tensor([[1.0, -2.0, 0.5]])), torch.zeros(1, 3)
        )

    def test_use_before_stats_raises(self) -> None:
        n = Normalizer(3)
        with pytest.raises(RuntimeError, match="before stats"):
            n.normalize(torch.zeros(1, 3))

    def test_state_dict_carries_stats(self) -> None:
        """ckpt に保存した統計が、読み直した先で同じ値になる (学習と推論が同じ統計を使う)。"""
        src = Normalizer(2)
        src.set_stats([0.3, -0.7], [1.5, 0.2])
        dst = Normalizer(2)
        dst.load_state_dict(src.state_dict())
        torch.testing.assert_close(dst.mean, src.mean)
        torch.testing.assert_close(dst.std, src.std)
        x = torch.randn(4, 2)
        torch.testing.assert_close(dst.normalize(x), src.normalize(x))

    def test_loading_state_without_stats_keeps_raising(self) -> None:
        dst = Normalizer(2)
        dst.load_state_dict(Normalizer(2).state_dict())
        with pytest.raises(RuntimeError, match="before stats"):
            dst.denormalize(torch.zeros(1, 2))

    def test_set_stats_validates(self) -> None:
        n = Normalizer(2)
        with pytest.raises(ValueError, match="stats must be"):
            n.set_stats([0.0], [1.0])
        with pytest.raises(ValueError, match="positive"):
            n.set_stats([0.0, 0.0], [1.0, 0.0])


class _FakeStateActionDataset:
    """state_action_item を持つ fake (state 3 次元、action chunk 2 行 × 16 次元、腰の指令 2 行 × 3)。"""

    def __init__(self, n_per_skill: dict[int, int]) -> None:
        rng = np.random.default_rng(4)
        self.items = []
        for sid, n in n_per_skill.items():
            for _ in range(n):
                self.items.append({
                    "state": rng.normal(sid, 1.0, size=3).astype(np.float32),
                    "action": rng.normal(-sid, 2.0, size=(2, 16)).astype(np.float32) * 0.1,
                    "action_waist_teacher": rng.normal(0.0, 0.05, size=(2, 3)).astype(np.float32),
                    "skill_id": sid,
                })

    def __len__(self) -> int:
        return len(self.items)

    def state_action_item(self, i: int) -> dict:
        return self.items[i]


class TestComputeNormalizationMoments:
    def test_collects_state_and_first_action_row_by_skill(self) -> None:
        ds = _FakeStateActionDataset({4: 30, 5: 12})
        moments = compute_normalization_moments(ds, batch_size=8)
        assert set(moments) == {"state", "action"}
        assert moments["state"].counts == {4: 30, 5: 12}
        assert moments["action"].counts == {4: 30, 5: 12}
        rows = np.stack([it["action"][0] for it in ds.items if it["skill_id"] == 5])
        np.testing.assert_allclose(moments["action"].per_skill()[5][0], rows.mean(axis=0), rtol=1e-5)
        states = np.stack([it["state"] for it in ds.items if it["skill_id"] == 4])
        np.testing.assert_allclose(moments["state"].per_skill()[4][0], states.mean(axis=0), rtol=1e-5)

    def test_fk_features_of_teacher_command(self) -> None:
        """fk を渡すと、教師の指令 (腰 + 腕 + hand) を FK に通した 27 次元も同じ pass で集める。"""
        from model.ramen_ori.fk import FK_FEATURE_DIM, G1WristFKTorch, assemble_action19

        fk = G1WristFKTorch.from_default_urdf()
        ds = _FakeStateActionDataset({4: 10, 0: 6})
        moments = compute_normalization_moments(ds, fk=fk, batch_size=4)
        assert moments["fk"].dim == FK_FEATURE_DIM
        assert moments["fk"].counts == {0: 6, 4: 10}
        items = [it for it in ds.items if it["skill_id"] == 0]
        q19 = assemble_action19(
            torch.from_numpy(np.stack([it["action_waist_teacher"][0] for it in items])),
            torch.from_numpy(np.stack([it["action"][0] for it in items])),
        )
        expected = fk.features(q19).numpy().mean(axis=0)
        np.testing.assert_allclose(moments["fk"].per_skill()[0][0], expected, rtol=1e-4, atol=1e-6)
