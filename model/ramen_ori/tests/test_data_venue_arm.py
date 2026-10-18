"""dataset の会場の腕の置き換え (Issue #188) の test。

指令と腕の列 (hf_dataset) を持つ偽の LeRobot で dataset を作り、表が区間ごとの模擬と一致するか、
item の腕・追従のずれ・速度・手先 (FK) が置き換えた腕から作られるか、正解の指令が変わらないかを見る。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.ramen_ori.data_lerobot import RamenOriLerobotDataset, RamenOriMultiDataset
from model.ramen_ori.state_derive import STATE71_GROUPS
from model.ramen_ori.tests.test_data_lerobot import FakeLeRobotBase
from model.ramen_ori.venue_arm import command19_from, simulate_episode

LENGTHS = [20, 25]
ARMS = slice(3, 17)  # state の各 19-D group (joint / tracking_err / velocity) の中の腕


class _Columns(dict):
    def with_format(self, _fmt):
        return self


class _FakeBaseWithArmColumns(FakeLeRobotBase):
    """FakeLeRobotBase + 教師の指令と腕の列 (hf_dataset)。2 区間目の先頭の 1 frame 前を pad にする。"""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            num_samples=sum(LENGTHS),
            episode_lengths=LENGTHS,
            n_image_frames=1,
            prev_pad_at_idx=LENGTHS[0],
        )
        rng = np.random.default_rng(seed)
        n = self.num_samples
        t = np.arange(n)[:, None] / 30.0
        q_desired = (
            0.3 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, np.pi, 36))
        ).astype(np.float32)
        self.hf_dataset = _Columns(
            {
                "action.robot_q_desired": q_desired,
                "action.hand_cmd": rng.uniform(0, 1, size=(n, 2)).astype(np.float32),
                "observation.state.robot_q_current": q_desired.copy(),
                "episode_index": np.repeat(np.arange(len(LENGTHS)), LENGTHS),
            }
        )

    def _ensure_hf_dataset_loaded(self) -> None:
        pass


VENUE = {"taus": [0.1, 0.2], "prob": 0.5, "workers": 1}


def _dataset(**venue) -> RamenOriLerobotDataset:
    return RamenOriLerobotDataset(
        base_dataset=_FakeBaseWithArmColumns(),
        load_prev_image=False,
        ee_state_source="fk",
        venue_arm={**VENUE, **venue},
    )


def _groups(state: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(state[sl]) for name, sl in STATE71_GROUPS.items()}


@pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
def test_table_matches_simulating_each_episode() -> None:
    ds = _dataset()
    cols = ds._base.hf_dataset
    command19 = command19_from(cols["action.robot_q_desired"], cols["action.hand_cmd"])
    measured = cols["observation.state.robot_q_current"][:, 22:36]
    assert ds._venue_arm_tables.shape == (2, sum(LENGTHS), 14)
    for k, tau in enumerate(VENUE["taus"]):
        start = 0
        for n in LENGTHS:
            np.testing.assert_allclose(
                ds._venue_arm_tables[k, start : start + n],
                simulate_episode(
                    command19[start : start + n],
                    measured[start : start + n],
                    tau,
                    "standard",
                ),
            )
            start += n


@pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
def test_eval_tau_rebuilds_arm_tracking_velocity_and_hand_from_the_venue_arm() -> None:
    ds = _dataset(eval_tau=0.2)
    idx = 7
    raw = ds._base[idx]
    arm, arm_prev = ds._venue_arm_tables[1, idx], ds._venue_arm_tables[1, idx - 1]
    g = _groups(ds._state_action(raw, idx, is_train=False)["state"])
    np.testing.assert_allclose(g["joint"][ARMS], arm, atol=1e-6)
    np.testing.assert_allclose(g["velocity"][ARMS], arm - arm_prev, atol=1e-6)
    action_prev = raw["action.robot_q_desired"].numpy()[0, 22:36]
    np.testing.assert_allclose(g["tracking_err"][ARMS], action_prev - arm, atol=1e-6)
    q = raw["observation.state.robot_q_current"].numpy().copy()
    q[1, 22:36] = arm
    fk = ds._ee_fk_by_skill[ds._skill_id_by_episode[0]]
    np.testing.assert_allclose(g["ee_pose"], fk.compute_ee_state(q[1][7:36]), atol=1e-5)


@pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
def test_episode_start_has_no_velocity() -> None:
    ds = _dataset(eval_tau=0.1)
    idx = LENGTHS[0]  # 2 区間目の先頭 (1 frame 前が区間の外)
    g = _groups(ds._state_action(ds._base[idx], idx, is_train=False)["state"])
    np.testing.assert_allclose(
        g["joint"][ARMS], ds._venue_arm_tables[0, idx], atol=1e-6
    )
    np.testing.assert_array_equal(g["velocity"], 0.0)


@pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
def test_targets_and_non_arm_state_are_unchanged() -> None:
    plain = RamenOriLerobotDataset(
        base_dataset=_FakeBaseWithArmColumns(),
        load_prev_image=False,
        ee_state_source="fk",
    )
    venue = _dataset(eval_tau=0.2)
    idx = 9
    a = plain._state_action(plain._base[idx], idx)
    b = venue._state_action(venue._base[idx], idx, is_train=False)
    np.testing.assert_array_equal(a["action"], b["action"])
    np.testing.assert_array_equal(a["action_waist_teacher"], b["action_waist_teacher"])
    ga, gb = _groups(a["state"]), _groups(b["state"])
    np.testing.assert_array_equal(ga["joint"][:3], gb["joint"][:3])  # 腰
    np.testing.assert_array_equal(ga["hand_state"], gb["hand_state"])
    np.testing.assert_array_equal(ga["joint"][17:19], gb["joint"][17:19])   # joint の末尾の hand
    assert not np.allclose(ga["joint"][ARMS], gb["joint"][ARMS])


@pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
def test_prob_decides_training_samples() -> None:
    idx = 5
    never = _dataset(prob=0.0)
    plain = RamenOriLerobotDataset(
        base_dataset=_FakeBaseWithArmColumns(),
        load_prev_image=False,
        ee_state_source="fk",
    )
    np.testing.assert_array_equal(
        never._state_action(never._base[idx], idx, is_train=True)["state"],
        plain._state_action(plain._base[idx], idx)["state"],
    )
    always = _dataset(prob=1.0)
    torch.manual_seed(0)
    picked = set()
    for _ in range(20):
        arm = always._state_action(always._base[idx], idx, is_train=True)["state"][:19][
            ARMS
        ]
        matches = [
            k
            for k in range(2)
            if np.allclose(arm, always._venue_arm_tables[k, idx], atol=1e-6)
        ]
        assert len(matches) == 1
        picked.add(matches[0])
    assert picked == {0, 1}  # tau は両方とも引かれる


@pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
def test_val_without_eval_tau_is_the_recorded_arm() -> None:
    ds = _dataset()  # eval_tau なし
    plain = RamenOriLerobotDataset(
        base_dataset=_FakeBaseWithArmColumns(),
        load_prev_image=False,
        ee_state_source="fk",
    )
    idx = 3
    np.testing.assert_array_equal(
        ds._state_action(ds._base[idx], idx, is_train=False)["state"],
        plain._state_action(plain._base[idx], idx)["state"],
    )


@pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
def test_transform_item_passes_is_train() -> None:
    ds = _dataset(prob=0.0, eval_tau=0.2)
    idx = 8
    train, val = (
        ds._transform_item(ds._base[idx], idx, is_train=True),
        ds._transform_item(ds._base[idx], idx, is_train=False),
    )
    np.testing.assert_allclose(
        val["state"][:19][ARMS].numpy(), ds._venue_arm_tables[1, idx], atol=1e-6
    )
    assert not np.allclose(
        train["state"][:19][ARMS].numpy(), ds._venue_arm_tables[1, idx]
    )


def test_requires_fk_hand_input() -> None:
    with pytest.raises(ValueError, match="ee_state_source"):
        RamenOriLerobotDataset(
            base_dataset=_FakeBaseWithArmColumns(),
            load_prev_image=False,
            venue_arm=VENUE,
        )


@pytest.mark.requires_boundary  # 会場の腕の模擬は運営の package を読む (Issue #196)
def test_multi_keeps_the_setting() -> None:
    subs = [_dataset(), _dataset()]
    multi = RamenOriMultiDataset(
        sub_datasets=subs, ee_state_source="fk", venue_arm=VENUE
    )
    assert multi.venue_arm == subs[0].venue_arm
    assert (
        multi.venue_arm["taus"] == (0.1, 0.2)
        and multi.venue_arm["preset"] == "standard"
    )
