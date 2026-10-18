"""dataset の区間の選別と腕の状態のずらし (Issue #188、flip の追加学習) の test。

腕の列 (hf_dataset) で区間ごとの手順を決めた偽の LeRobot で dataset を作り、train / val / test に残るのが揃った区間の
frame だけか、ずらした腕から追従のずれ・手先 (FK) が作られ速度は変わらないか、学習以外ではずらさないかを見る。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.ramen_ori.data_lerobot import RamenOriLerobotDataset, RamenOriMultiDataset
from model.ramen_ori.state_derive import STATE71_GROUPS
from model.ramen_ori.tests.test_data_lerobot import FakeLeRobotBase

LENGTHS = [20, 20, 20, 20, 20]
ARMS = slice(3, 17)  # state の各 19-D group (joint / tracking_err / velocity) の中の腕
SELECT_A = {"left_peak_first": True, "right_push_max_rad": -0.8}


class _Columns(dict):
    def with_format(self, _fmt):
        return self


class _FakeBaseWithShoulders(FakeLeRobotBase):
    """区間 1・3 は右腕が先に最高点 (揃っていない)、残りは左腕が先・右腕も押す。"""

    RIGHT_FIRST = (1, 3)

    def __init__(self) -> None:
        super().__init__(
            num_samples=sum(LENGTHS), episode_lengths=LENGTHS, n_image_frames=1
        )
        q = np.zeros((self.num_samples, 36), dtype=np.float32)
        for ep, start in enumerate(np.cumsum([0] + LENGTHS[:-1])):
            t_left, t_right = (12, 6) if ep in self.RIGHT_FIRST else (6, 12)
            q[start + t_left, 22] = -1.0
            q[start + t_right, 29] = -1.1
        self.hf_dataset = _Columns(
            {
                "observation.state.robot_q_current": q,
                "episode_index": np.repeat(np.arange(len(LENGTHS)), LENGTHS),
            }
        )

    def _ensure_hf_dataset_loaded(self) -> None:
        pass


def _dataset(**kwargs) -> RamenOriLerobotDataset:
    return RamenOriLerobotDataset(
        base_dataset=_FakeBaseWithShoulders(),
        load_prev_image=False,
        ee_state_source="fk",
        **kwargs,
    )


def _groups(state) -> dict[str, np.ndarray]:
    return {name: np.asarray(state[sl]) for name, sl in STATE71_GROUPS.items()}


class TestEpisodeSelect:
    def test_views_keep_only_the_selected_episodes(self) -> None:
        ds = _dataset(episode_select=SELECT_A)
        train, val, test = ds.make_train_val_test_split(
            val_ratio=0.2, test_ratio=0.2, seed=0
        )
        kept = {i for ep in (0, 2, 4) for i in range(ep * 20, ep * 20 + 20)}
        rows = set(train.indices) | set(val.indices) | set(test.indices)
        assert rows == kept
        assert not set(train.indices) & set(val.indices)

    def test_same_split_as_without_selection_minus_the_dropped(self) -> None:
        """分け方 (seed) は同じで、揃っていない区間を落とすだけ (val が選別前の val の部分集合)。"""
        plain = _dataset().make_train_val_test_split(
            val_ratio=0.2, test_ratio=0.2, seed=0
        )
        sel = _dataset(episode_select=SELECT_A).make_train_val_test_split(
            val_ratio=0.2, test_ratio=0.2, seed=0
        )
        for p, s in zip(plain, sel):
            assert set(s.indices) == {
                i
                for i in p.indices
                if (i // 20) not in _FakeBaseWithShoulders.RIGHT_FIRST
            }

    def test_without_selection_keeps_everything(self) -> None:
        train, val, test = _dataset().make_train_val_test_split(
            val_ratio=0.2, test_ratio=0.2, seed=0
        )
        assert len(train) + len(val) + len(test) == sum(LENGTHS)


class TestStateNoise:
    NOISE = {"prob": 1.0, "sigma_rad": 0.2, "max_rad": 0.5}

    def test_train_sample_shifts_arms_but_not_velocity(self) -> None:
        plain = _dataset()
        noisy = _dataset(state_noise=self.NOISE)
        idx = 7
        torch.manual_seed(0)
        a = _groups(plain._state_action(plain._base[idx], idx, is_train=True)["state"])
        b = _groups(noisy._state_action(noisy._base[idx], idx, is_train=True)["state"])
        shift = b["joint"][ARMS] - a["joint"][ARMS]
        assert np.abs(shift).max() > 1e-3 and np.abs(shift).max() <= 0.5 + 1e-6
        np.testing.assert_allclose(
            b["velocity"], a["velocity"], atol=1e-6
        )  # 同じずれなので速度は同じ
        np.testing.assert_allclose(
            b["tracking_err"][ARMS], a["tracking_err"][ARMS] - shift, atol=1e-5
        )
        np.testing.assert_array_equal(b["joint"][:3], a["joint"][:3])  # 腰はずらさない
        assert not np.allclose(b["ee_pose"], a["ee_pose"])  # 手先はずらした腕の FK

    def test_hand_pose_is_the_fk_of_the_shifted_arm(self) -> None:
        noisy = _dataset(state_noise=self.NOISE)
        idx = 9
        raw = noisy._base[idx]
        torch.manual_seed(1)
        g = _groups(noisy._state_action(raw, idx, is_train=True)["state"])
        q = raw["observation.state.robot_q_current"].numpy().copy()
        q[1, 22:36] = g["joint"][ARMS]
        fk = noisy._ee_fk_by_skill[noisy._skill_id_by_episode[0]]
        np.testing.assert_allclose(
            g["ee_pose"], fk.compute_ee_state(q[1][7:36]), atol=1e-5
        )

    def test_targets_unchanged_and_no_noise_outside_training(self) -> None:
        plain = _dataset()
        noisy = _dataset(state_noise=self.NOISE)
        idx = 5
        a = plain._state_action(plain._base[idx], idx)
        b_train = noisy._state_action(noisy._base[idx], idx, is_train=True)
        b_eval = noisy._state_action(noisy._base[idx], idx, is_train=False)
        np.testing.assert_array_equal(a["action"], b_train["action"])
        np.testing.assert_array_equal(a["state"], b_eval["state"])

    def test_prob_zero_is_the_plain_dataset(self) -> None:
        plain = _dataset()
        never = _dataset(state_noise={**self.NOISE, "prob": 0.0})
        np.testing.assert_array_equal(
            never._state_action(never._base[4], 4, is_train=True)["state"],
            plain._state_action(plain._base[4], 4)["state"],
        )

    def test_requires_fk_hand_input(self) -> None:
        with pytest.raises(ValueError, match="ee_state_source"):
            RamenOriLerobotDataset(
                base_dataset=_FakeBaseWithShoulders(),
                load_prev_image=False,
                state_noise=self.NOISE,
            )


def test_multi_keeps_both_settings() -> None:
    subs = [
        _dataset(episode_select=SELECT_A, state_noise=TestStateNoise.NOISE)
        for _ in range(2)
    ]
    multi = RamenOriMultiDataset(
        sub_datasets=subs,
        ee_state_source="fk",
        episode_select=SELECT_A,
        state_noise=TestStateNoise.NOISE,
    )
    assert (
        multi.episode_select == subs[0].episode_select
        and multi.episode_select["left_peak_first"] is True
    )
    assert multi.state_noise["sigma_rad"] == 0.2
    train, _, _ = multi.make_train_val_test_split(val_ratio=0.2, test_ratio=0.2, seed=0)
    assert len(train) > 0
