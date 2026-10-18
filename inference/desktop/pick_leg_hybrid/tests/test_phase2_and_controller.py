"""区間 2 の補間と、3 区間 controller の unit test (Issue #136)。"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.pick_leg_hybrid.boundary import (
    BoundaryConfig,
    GraspBoundaryDetector,
)
from inference.desktop.pick_leg_hybrid.config import PickLegHybridConfig
from inference.desktop.pick_leg_hybrid.controller import PickLegHybridController
from inference.desktop.pick_leg_hybrid.phase2 import (
    IDX_BASE_HEIGHT,
    SLICE_EE_LEFT_POS,
    SLICE_EE_LEFT_QUAT,
    SLICE_EE_RIGHT_POS,
    SLICE_NAVIGATE,
    EePose,
    Phase2Config,
    Phase2Motion,
    quat_angle_between,
    slerp,
    taskspace_row,
)
from inference.desktop.pick_leg_hybrid.phases import Phase
from inference.desktop.pick_leg_hybrid.vlm import VlmAnswer

# derive_thresholds.py が実データから出した値 (config と同じ)
GOAL_L = EePose.of([0.3783, 0.1172, 0.3107], [0.8964, -0.3106, 0.0475, -0.3127])
GOAL_R = EePose.of([0.3970, -0.0434, 0.1382], [0.8752, 0.0888, 0.0505, 0.4728])
START_L = EePose.of([0.20, 0.05, 0.15], [1.0, 0.0, 0.0, 0.0])
START_R = EePose.of([0.25, -0.10, 0.10], [1.0, 0.0, 0.0, 0.0])

RAW_CONFIG = {
    "interlock": {"follow_error_min": 0.2, "hold_min": 1.45, "hold_max": 2.35},
    "boundary": {"confirm_count": 2, "min_interval_sec": 1.0},
    "vlm": {"timeout_sec": 5.0},
    "overlay": {"enabled": True, "class_ids": [1, 5, 6]},
    "references": {"before": None, "after": None},
    "runtime": {
        "hard_timeout_sec": 65.0,
        "phase1_timeout_sec": 18.0,
        "phase2_timeout_sec": 16.0,
    },
    "phase2": {
        "ee_frame": "g1_wrist_fk",
        "duration_sec": 2.7,
        "pos_tol": 0.02,
        "rot_tol": 0.12,
        "goal_left": {"pos": list(GOAL_L.pos), "quat": list(GOAL_L.quat)},
        "goal_right": {"pos": list(GOAL_R.pos), "quat": list(GOAL_R.quat)},
    },
}


class TestSlerp:
    def test_endpoints(self) -> None:
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0])
        assert np.allclose(slerp(a, b, 0.0), a)
        assert np.allclose(slerp(a, b, 1.0), b)

    def test_stays_unit_length(self) -> None:
        a, b = GOAL_L.quat, GOAL_R.quat
        for s in np.linspace(0, 1, 11):
            assert np.isclose(np.linalg.norm(slerp(a, b, s)), 1.0)

    def test_takes_the_short_way_around(self) -> None:
        """q と −q は同じ回転。近い方を選ぶこと。"""
        a = np.array([1.0, 0.0, 0.0, 0.0])
        assert quat_angle_between(slerp(a, -a, 0.5), a) < 1e-6

    def test_angle_between_identical_is_zero(self) -> None:
        assert quat_angle_between(GOAL_L.quat, GOAL_L.quat) == pytest.approx(
            0.0, abs=1e-7
        )


class TestTaskspaceRow:
    def test_layout_matches_the_boundary_contract(self) -> None:
        row = taskspace_row(GOAL_L, GOAL_R, hand_left_model=2.2, hand_right_model=2.2)
        assert row.shape == (25,)
        assert np.allclose(row[SLICE_EE_LEFT_POS], GOAL_L.pos, atol=1e-6)
        assert np.allclose(row[SLICE_EE_LEFT_QUAT], GOAL_L.quat, atol=1e-6)
        assert np.allclose(row[SLICE_EE_RIGHT_POS], GOAL_R.pos, atol=1e-6)

    def test_lower_body_is_zero_held_for_pick(self) -> None:
        row = taskspace_row(GOAL_L, GOAL_R, hand_left_model=2.2, hand_right_model=2.2)
        assert np.allclose(row[SLICE_NAVIGATE], 0.0)

    def test_base_height_defaults_to_neutral_not_zero(self) -> None:
        """col [21] の既定は 0 ではなく 0.74 であること。

        運営の adapter は col [21] をそのまま WBC へ渡す
        (`wbc_driver.run_decoupled` の `base_height_command=[[float(r[21])] ...]`)。0 は「指定なし」ではなく
        **骨盤高さ 0 m** の指令。運営既定は `wbc_goal.py:46` の 0.74。
        """
        row = taskspace_row(GOAL_L, GOAL_R, hand_left_model=2.2, hand_right_model=2.2)
        assert row[IDX_BASE_HEIGHT] == pytest.approx(0.74)

    def test_hand_is_mapped_into_boundary_units(self) -> None:
        """model 空間 0..4.5 → boundary の +1(閉) .. -1(開)。"""
        closed = taskspace_row(
            GOAL_L, GOAL_R, hand_left_model=0.0, hand_right_model=0.0
        )
        opened = taskspace_row(
            GOAL_L, GOAL_R, hand_left_model=4.5, hand_right_model=4.5
        )
        assert closed[0] > opened[0]
        assert -1.0 <= float(opened[0]) <= 1.0
        assert -1.0 <= float(closed[0]) <= 1.0


class TestPhase2Motion:
    def _motion(self) -> Phase2Motion:
        m = Phase2Motion(Phase2Config(goal_left=GOAL_L, goal_right=GOAL_R))
        m.start(0.0, START_L, START_R)
        return m

    def test_starts_at_the_captured_pose(self) -> None:
        """可変始点: 1→2 遷移時の実姿勢から始まる (D-10)。"""
        left, right = self._motion().pose_at(0.0)
        assert np.allclose(left.pos, START_L.pos)
        assert np.allclose(right.pos, START_R.pos)

    def test_ends_at_the_fixed_goal(self) -> None:
        left, right = self._motion().pose_at(999.0)
        assert np.allclose(left.pos, GOAL_L.pos, atol=1e-6)
        assert np.allclose(right.pos, GOAL_R.pos, atol=1e-6)

    def test_progress_is_clamped(self) -> None:
        m = self._motion()
        assert m.progress(-5.0) == 0.0
        assert m.progress(999.0) == 1.0
        assert m.progress(1.35) == pytest.approx(0.5, abs=1e-6)

    def test_step_returns_a_25d_row_holding_the_leg(self) -> None:
        row = self._motion().step(1.0)
        assert row.shape == (25,)
        # 保持し続ける = 閉じ側 (hold_hand_value=2.2 は開 4.5 より閉じている)
        opened = taskspace_row(
            GOAL_L, GOAL_R, hand_left_model=4.5, hand_right_model=4.5
        )
        assert row[0] > opened[0]

    def test_reached_uses_the_measured_pose_not_the_schedule(self) -> None:
        """指令どおり動いていない場合に誤って進まないこと。"""
        m = self._motion()
        m.step(999.0)  # 予定上は終点
        assert m.reached(START_L, START_R) is False  # 実測はまだ始点
        assert m.reached(GOAL_L, GOAL_R) is True

    def test_reached_respects_the_position_tolerance(self) -> None:
        m = self._motion()
        cfg = m.cfg
        near = EePose.of(GOAL_L.pos + np.array([cfg.pos_tol * 0.5, 0, 0]), GOAL_L.quat)
        far = EePose.of(GOAL_L.pos + np.array([cfg.pos_tol * 2.0, 0, 0]), GOAL_L.quat)
        assert m.reached(near, GOAL_R) is True
        assert m.reached(far, GOAL_R) is False

    def test_both_arms_must_arrive(self) -> None:
        m = self._motion()
        assert m.reached(GOAL_L, START_R) is False

    def test_pose_before_start_raises(self) -> None:
        m = Phase2Motion(Phase2Config(goal_left=GOAL_L, goal_right=GOAL_R))
        with pytest.raises(RuntimeError, match="start"):
            m.pose_at(0.0)

    def test_ee_pose_validates_input(self) -> None:
        with pytest.raises(ValueError, match="pos"):
            EePose.of([0.0, 0.0], [1, 0, 0, 0])
        with pytest.raises(ValueError, match="quat"):
            EePose.of([0, 0, 0], [0, 0, 0, 0])


class TestConfig:
    def test_from_dict_round_trips(self) -> None:
        cfg = PickLegHybridConfig.from_dict(RAW_CONFIG)
        assert cfg.interlock.follow_error_min == 0.2
        assert cfg.boundary.confirm_count == 2
        assert cfg.overlay.class_ids == (1, 5, 6)
        assert cfg.runtime.hard_timeout_sec == 65.0
        assert np.allclose(cfg.phase2.goal_left.pos, GOAL_L.pos)

    def test_requires_a_positive_production_timeout(self) -> None:
        with pytest.raises(ValueError, match="hard_timeout_sec"):
            PickLegHybridConfig.from_dict(
                {**RAW_CONFIG, "runtime": {**RAW_CONFIG["runtime"], "hard_timeout_sec": 0.0}}
            )

    def test_runtime_leaves_time_for_the_rule_based_handover(self) -> None:
        with pytest.raises(ValueError, match="leave time for phase 3"):
            PickLegHybridConfig.from_dict(
                {**RAW_CONFIG, "runtime": {**RAW_CONFIG["runtime"], "hard_timeout_sec": 34.0}}
            )

    def test_rejects_an_unknown_ee_frame(self) -> None:
        """どの FK 定義で書いた値かを取り違えると 10cm ずれる (D-3)。"""
        raw = {**RAW_CONFIG, "phase2": {**RAW_CONFIG["phase2"], "ee_frame": "nope"}}
        with pytest.raises(ValueError, match="ee_frame"):
            PickLegHybridConfig.from_dict(raw)

    def test_requires_the_phase2_goals(self) -> None:
        raw = {**RAW_CONFIG, "phase2": {"ee_frame": "g1_wrist_fk"}}
        with pytest.raises(ValueError, match="goal_left"):
            PickLegHybridConfig.from_dict(raw)

    def test_references_report_readiness(self) -> None:
        cfg = PickLegHybridConfig.from_dict(RAW_CONFIG)
        assert cfg.references.is_ready is False  # 参照画像は未作成


class _StubVlm:
    def __init__(self, values) -> None:
        self._values = list(values)
        self.calls = 0

    def ask(self, question, images, *, current_phase_text=""):
        self.calls += 1
        v = self._values.pop(0) if self._values else 0
        return VlmAnswer(value=v)


class TestController:
    def _controller(self, vlm_values, ee_probe=None):
        cfg = PickLegHybridConfig.from_dict(RAW_CONFIG)
        detector = GraspBoundaryDetector(
            cfg=BoundaryConfig(confirm_count=2, min_interval_sec=1.0),
            interlock_cfg=cfg.interlock,
            vlm_client=_StubVlm(vlm_values),
        )
        self.vla_calls: list[dict] = []

        def vla(obs):
            self.vla_calls.append(obs)
            return np.zeros((16, 25), dtype=np.float32)

        return PickLegHybridController(
            cfg,
            vla_step=vla,
            ee_probe=ee_probe or (lambda obs: (START_L, START_R)),
            detector=detector,
        )

    def _obs(self, t, holding=True):
        return {
            "t": t,
            "hand_state": [2.19, 2.24] if holding else [4.49, 4.49],
            "hand_cmd": [1.84, 1.89] if holding else [4.50, 4.50],
            "images_b64": ["past", "now"],
        }

    def test_phase1_runs_the_vla(self) -> None:
        c = self._controller([0])
        out = c.step(self._obs(0.0, holding=False))
        assert out.phase is Phase.APPROACH_GRASP
        assert out.executor == "vla"
        assert out.action.shape == (16, 25)
        assert len(self.vla_calls) == 1

    def test_advances_to_phase2_after_two_confirmations(self) -> None:
        c = self._controller([1, 1])
        assert c.step(self._obs(0.0)).advanced is False
        out = c.step(self._obs(1.0))
        assert out.advanced is True
        assert c.state.phase is Phase.CARRY_TO_LEFT

    def test_phase2_emits_interpolated_taskspace_rows(self) -> None:
        c = self._controller([1, 1])
        c.step(self._obs(0.0))
        c.step(self._obs(1.0))  # → 区間 2
        out = c.step(self._obs(1.5))
        assert out.phase is Phase.CARRY_TO_LEFT
        assert out.executor == "mp"
        assert out.action.shape == (1, 25)
        # 区間 2 では VLA を呼ばない
        assert len(self.vla_calls) == 2

    def test_phase2_captures_the_pose_at_the_moment_of_transition(self) -> None:
        c = self._controller([1, 1])
        c.step(self._obs(0.0))
        c.step(self._obs(1.0))
        left, _ = c.motion.pose_at(1.0)
        assert np.allclose(left.pos, START_L.pos)

    def test_advances_to_phase3_when_the_measured_pose_arrives(self) -> None:
        poses = {"value": (START_L, START_R)}
        c = self._controller([1, 1], ee_probe=lambda obs: poses["value"])
        c.step(self._obs(0.0))
        c.step(self._obs(1.0))
        assert c.step(self._obs(1.5)).advanced is False
        poses["value"] = (GOAL_L, GOAL_R)
        out = c.step(self._obs(2.0))
        assert out.reached is True and out.advanced is True
        assert c.state.phase is Phase.HANDOVER_ONWARD

    def test_phase3_runs_the_vla_and_never_advances(self) -> None:
        c = self._controller([1, 1], ee_probe=lambda obs: (GOAL_L, GOAL_R))
        c.step(self._obs(0.0))
        c.step(self._obs(1.0))
        c.step(self._obs(1.5))
        assert c.state.phase is Phase.HANDOVER_ONWARD
        for i in range(3):
            out = c.step(self._obs(2.0 + i))
            assert out.phase is Phase.HANDOVER_ONWARD
            assert out.advanced is False
            assert out.boundary is None  # 最終区間は境界検出しない

    def test_never_goes_backwards_even_if_the_vlm_says_no(self) -> None:
        """単調性の保証はコード側 (D-9)。VLM が何を返しても戻らない。"""
        c = self._controller([1, 1] + [0] * 20, ee_probe=lambda obs: (GOAL_L, GOAL_R))
        c.step(self._obs(0.0))
        c.step(self._obs(1.0))
        assert c.state.phase is Phase.CARRY_TO_LEFT
        for i in range(10):
            c.step(self._obs(2.0 + i))
            assert c.state.phase >= Phase.CARRY_TO_LEFT

    def test_reset_returns_to_phase1(self) -> None:
        c = self._controller([1, 1])
        c.step(self._obs(0.0))
        c.step(self._obs(1.0))
        c.reset()
        assert c.state.phase is Phase.APPROACH_GRASP
        assert c.motion.started is False
        assert c.detector.hits == 0
