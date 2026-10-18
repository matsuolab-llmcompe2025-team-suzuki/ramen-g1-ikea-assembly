"""state_derive の unit test (Issue #120 Phase A-1)。

- shape / dim contract (71D 出力、slice 内訳)
- ep 先頭 (state_prev=None) の velocity=zeros
- suzuki UPPER_BODY_SOURCE_INDEX_MAP と cross-check (joint 19 が期待 index から抽出)
- dim mismatch / NaN / Inf の raise
"""

from __future__ import annotations

import numpy as np
import pytest

from model.ramen_ori.state_derive import (
    ARMS_HAND_SOURCE_INDEX_MAP,
    DEPTH_CONTACT_DIM,
    RAMEN_ORI_ACTION_DIM,
    RAMEN_ORI_STATE_DIM,
    RAMEN_ORI_STATE_DIM_WITH_DEPTH_CONTACT,
    derive_state_71d,
    derive_state_73d,
)
from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    SOURCE_EEF_DIM,
    SOURCE_JOINT_SLICES,
    SOURCE_ROBOT_Q_DIM,
    SOURCE_ROOT_POSE_DIM,
    SOURCE_STATE_DIM,
    UPPER_BODY_SOURCE_INDEX_MAP,
    UPPER_BODY_STATE_DIM,
)


def _make_row(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(state_current, action_prev, ee_state, state_prev) の合成 row を作る。"""
    rng = np.random.default_rng(seed)
    state_current = rng.standard_normal(SOURCE_STATE_DIM).astype(np.float32)
    action_prev = rng.standard_normal(SOURCE_STATE_DIM).astype(np.float32)
    ee_state = rng.standard_normal(SOURCE_EEF_DIM).astype(np.float32)
    state_prev = rng.standard_normal(SOURCE_STATE_DIM).astype(np.float32)
    return state_current, action_prev, ee_state, state_prev


class TestShape:
    def test_output_is_71d_float32(self) -> None:
        sc, ac, ee, sp = _make_row(seed=0)
        out = derive_state_71d(sc, ac, ee, sp)
        assert out.shape == (RAMEN_ORI_STATE_DIM,)
        assert out.dtype == np.float32

    def test_upper_body_dim_matches_suzuki(self) -> None:
        """joint / tracking_err / velocity slice はいずれも UPPER_BODY_STATE_DIM (=19)。"""
        assert UPPER_BODY_STATE_DIM == 19
        # 19 + 19 + 19 + 2 + 12 = 71
        assert (
            UPPER_BODY_STATE_DIM * 3 + 2 + SOURCE_EEF_DIM == RAMEN_ORI_STATE_DIM
        )


class TestSliceContent:
    def test_joint_slice_equals_upper_body_of_state_current(self) -> None:
        sc, ac, ee, sp = _make_row(seed=1)
        out = derive_state_71d(sc, ac, ee, sp)
        expected = sc[np.asarray(UPPER_BODY_SOURCE_INDEX_MAP)]
        np.testing.assert_allclose(out[0:19], expected)

    def test_tracking_err_is_prev_action_minus_state_with_waist_zero(self) -> None:
        """Issue #141 RO-3: tracking_err = 1 frame 前の指令 − 今、腰 3 次元は 0。"""
        sc, ap, ee, sp = _make_row(seed=2)
        out = derive_state_71d(sc, ap, ee, sp)
        idx = np.asarray(UPPER_BODY_SOURCE_INDEX_MAP)
        expected = ap[idx] - sc[idx]
        expected[:3] = 0.0
        np.testing.assert_allclose(out[19:38], expected)

    def test_action_prev_none_yields_tracking_err_zeros(self) -> None:
        """区間先頭 (1 frame 前が区間の外) → 0 (推論の 1 tick 目と同じ)。"""
        sc, _, ee, sp = _make_row(seed=9)
        out = derive_state_71d(sc, None, ee, sp)
        np.testing.assert_array_equal(out[19:38], np.zeros(19, dtype=np.float32))

    def test_velocity_slice_equals_state_minus_prev_on_upper_body(self) -> None:
        sc, ac, ee, sp = _make_row(seed=3)
        out = derive_state_71d(sc, ac, ee, sp)
        idx = np.asarray(UPPER_BODY_SOURCE_INDEX_MAP)
        expected = sc[idx] - sp[idx]
        np.testing.assert_allclose(out[38:57], expected)

    def test_hand_state_slice_equals_state_current_indices_36_37(self) -> None:
        sc, ac, ee, sp = _make_row(seed=4)
        out = derive_state_71d(sc, ac, ee, sp)
        np.testing.assert_allclose(
            out[57:59], sc[SOURCE_ROBOT_Q_DIM : SOURCE_ROBOT_Q_DIM + 2]
        )

    def test_ee_pose_slice_equals_ee_state(self) -> None:
        sc, ac, ee, sp = _make_row(seed=5)
        out = derive_state_71d(sc, ac, ee, sp)
        np.testing.assert_allclose(out[59:71], ee)

    def test_hand_state_slice_duplicates_last_2_of_joint_slice(self) -> None:
        """設計上、hand_state 2D は joint 19 末尾 2 と同信号 (design doc 通り explicit token)。"""
        sc, ac, ee, sp = _make_row(seed=6)
        out = derive_state_71d(sc, ac, ee, sp)
        np.testing.assert_allclose(out[17:19], out[57:59])


class TestEpBoundary:
    def test_state_prev_none_yields_velocity_zeros(self) -> None:
        sc, ac, ee, _ = _make_row(seed=7)
        out = derive_state_71d(sc, ac, ee, state_prev=None)
        np.testing.assert_array_equal(out[38:57], np.zeros(19, dtype=np.float32))

    def test_state_prev_equal_current_yields_velocity_zeros(self) -> None:
        sc, ac, ee, _ = _make_row(seed=8)
        out = derive_state_71d(sc, ac, ee, state_prev=sc.copy())
        np.testing.assert_allclose(out[38:57], np.zeros(19, dtype=np.float32))


class TestArmsHandActionMap:
    """Issue #129 Phase A (2026-08-31): action target 用 ARMS_HAND_SOURCE_INDEX_MAP
    (=UPPER_BODY_SOURCE_INDEX_MAP[3:]、waist 3 除外の arms 14 + hand 2 = 16 index)。"""

    def test_action_dim_is_16(self) -> None:
        assert RAMEN_ORI_ACTION_DIM == 16
        assert len(ARMS_HAND_SOURCE_INDEX_MAP) == RAMEN_ORI_ACTION_DIM

    def test_arms_hand_map_is_upper_body_minus_waist(self) -> None:
        """ARMS_HAND_SOURCE_INDEX_MAP は UPPER_BODY_SOURCE_INDEX_MAP から先頭 3 (waist) を除いたもの。"""
        assert list(ARMS_HAND_SOURCE_INDEX_MAP) == list(UPPER_BODY_SOURCE_INDEX_MAP[3:])

    def test_first_7_indices_are_left_arm(self) -> None:
        """ARMS_HAND[0:7] は left_arm (UPPER_BODY[3:10] と同じ)。"""
        assert list(ARMS_HAND_SOURCE_INDEX_MAP[0:7]) == list(UPPER_BODY_SOURCE_INDEX_MAP[3:10])

    def test_next_7_indices_are_right_arm(self) -> None:
        assert list(ARMS_HAND_SOURCE_INDEX_MAP[7:14]) == list(UPPER_BODY_SOURCE_INDEX_MAP[10:17])

    def test_last_2_indices_are_hand(self) -> None:
        assert list(ARMS_HAND_SOURCE_INDEX_MAP[14:16]) == list(UPPER_BODY_SOURCE_INDEX_MAP[17:19])


class TestUpperBodyIndexMap:
    """UPPER_BODY_SOURCE_INDEX_MAP の内訳 (suzuki の source of truth) が
    waist(3)+left_arm(7)+right_arm(7)+hand(2)=19 であること (state 71D の joint slice 用に継続使用)。"""

    def test_length_is_19(self) -> None:
        assert len(UPPER_BODY_SOURCE_INDEX_MAP) == 19

    def test_first_3_indices_are_waist(self) -> None:
        expected = list(
            range(
                SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["waist"][0],
                SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["waist"][1],
            )
        )
        assert list(UPPER_BODY_SOURCE_INDEX_MAP[0:3]) == expected

    def test_next_7_indices_are_left_arm(self) -> None:
        expected = list(
            range(
                SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["left_arm"][0],
                SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["left_arm"][1],
            )
        )
        assert list(UPPER_BODY_SOURCE_INDEX_MAP[3:10]) == expected

    def test_next_7_indices_are_right_arm(self) -> None:
        expected = list(
            range(
                SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["right_arm"][0],
                SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["right_arm"][1],
            )
        )
        assert list(UPPER_BODY_SOURCE_INDEX_MAP[10:17]) == expected

    def test_last_2_indices_are_hand(self) -> None:
        assert list(UPPER_BODY_SOURCE_INDEX_MAP[17:19]) == [
            SOURCE_ROBOT_Q_DIM,
            SOURCE_ROBOT_Q_DIM + 1,
        ]


class TestRaises:
    @pytest.mark.parametrize(
        "arg,dim_expected",
        [
            ("state_current", SOURCE_STATE_DIM),
            ("action_prev", SOURCE_STATE_DIM),
            ("ee_state", SOURCE_EEF_DIM),
        ],
    )
    def test_raises_on_wrong_dim(self, arg: str, dim_expected: int) -> None:
        sc, ac, ee, sp = _make_row(seed=9)
        kwargs = {"state_current": sc, "action_prev": ac, "ee_state": ee, "state_prev": sp}
        kwargs[arg] = np.zeros(dim_expected - 1, dtype=np.float32)
        with pytest.raises(ValueError, match=f"expected {dim_expected}"):
            derive_state_71d(**kwargs)

    def test_raises_on_prev_wrong_dim(self) -> None:
        sc, ac, ee, _ = _make_row(seed=10)
        with pytest.raises(ValueError, match="state_prev"):
            derive_state_71d(sc, ac, ee, state_prev=np.zeros(SOURCE_STATE_DIM - 1))

    def test_raises_on_nan(self) -> None:
        sc, ac, ee, sp = _make_row(seed=11)
        sc[0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            derive_state_71d(sc, ac, ee, sp)

    def test_raises_on_inf(self) -> None:
        sc, ac, ee, sp = _make_row(seed=12)
        ac[5] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            derive_state_71d(sc, ac, ee, sp)

    def test_raises_on_2d_input(self) -> None:
        sc = np.zeros((2, SOURCE_STATE_DIM), dtype=np.float32)
        ac = np.zeros(SOURCE_STATE_DIM, dtype=np.float32)
        ee = np.zeros(SOURCE_EEF_DIM, dtype=np.float32)
        with pytest.raises(ValueError, match="must be 1-D"):
            derive_state_71d(sc, ac, ee, state_prev=None)


class TestDeriveState73d:
    """E-β 73D (E-δ 71D + depth contact bit 2D)。"""

    def test_output_is_73d_float32(self) -> None:
        sc, ac, ee, sp = _make_row(seed=20)
        dc = np.array([0.3, 0.7], dtype=np.float32)
        out = derive_state_73d(sc, ac, ee, sp, dc)
        assert out.shape == (RAMEN_ORI_STATE_DIM_WITH_DEPTH_CONTACT,)
        assert out.dtype == np.float32
        assert DEPTH_CONTACT_DIM == 2

    def test_first_71_dims_equal_derive_state_71d(self) -> None:
        sc, ac, ee, sp = _make_row(seed=21)
        dc = np.array([0.5, 0.2], dtype=np.float32)
        out_73 = derive_state_73d(sc, ac, ee, sp, dc)
        out_71 = derive_state_71d(sc, ac, ee, sp)
        np.testing.assert_allclose(out_73[:71], out_71)

    def test_last_2_dims_equal_depth_contact(self) -> None:
        sc, ac, ee, sp = _make_row(seed=22)
        dc = np.array([0.9, 0.1], dtype=np.float32)
        out = derive_state_73d(sc, ac, ee, sp, dc)
        np.testing.assert_allclose(out[71:73], dc)

    def test_ep_boundary_still_yields_velocity_zeros_in_73d(self) -> None:
        sc, ac, ee, _ = _make_row(seed=23)
        dc = np.zeros(2, dtype=np.float32)
        out = derive_state_73d(sc, ac, ee, state_prev=None, depth_contact=dc)
        np.testing.assert_allclose(out[38:57], np.zeros(19, dtype=np.float32))

    def test_raises_on_wrong_depth_contact_dim(self) -> None:
        sc, ac, ee, sp = _make_row(seed=24)
        with pytest.raises(ValueError, match="expected 2"):
            derive_state_73d(sc, ac, ee, sp, depth_contact=np.zeros(3))

    def test_raises_on_nan_depth_contact(self) -> None:
        sc, ac, ee, sp = _make_row(seed=25)
        dc = np.array([np.nan, 0.5], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            derive_state_73d(sc, ac, ee, sp, dc)

    def test_zero_depth_contact_placeholder(self) -> None:
        """Alt-2 の placeholder path: depth_contact=zeros で state[71:73]=0。"""
        sc, ac, ee, sp = _make_row(seed=26)
        out = derive_state_73d(sc, ac, ee, sp, depth_contact=np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(out[71:73], np.zeros(2, dtype=np.float32))


class TestDeterminism:
    def test_same_input_yields_same_output(self) -> None:
        sc, ac, ee, sp = _make_row(seed=13)
        out1 = derive_state_71d(sc, ac, ee, sp)
        out2 = derive_state_71d(sc, ac, ee, sp)
        np.testing.assert_array_equal(out1, out2)

    def test_output_not_shared_memory_with_input(self) -> None:
        """出力配列が入力を書き換えないこと (mutability guard)。"""
        sc, ac, ee, sp = _make_row(seed=14)
        sc_orig = sc.copy()
        out = derive_state_71d(sc, ac, ee, sp)
        out[0] = 999.0
        np.testing.assert_array_equal(sc, sc_orig)
