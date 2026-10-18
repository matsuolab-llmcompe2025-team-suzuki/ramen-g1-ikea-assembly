"""Orchestrator core の unit test。

perception は StubPerception (frame_id → 予め決めた OBBDetection list)、
dispatcher は 6 MockSkill 積んだ実 SkillDispatchLowerPolicy を使う。
"""

from __future__ import annotations

from typing import Optional

import io
import time
import json

import numpy as np
import pytest

from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
from inference.desktop.lower_policy.skills.base import Skill
from inference.desktop.lower_policy.skills.mock import MockSkill
from inference.desktop.orchestrator import (
    DEFAULT_ENTER_CHECK,
    DEFAULT_TRANSITIONS,
    LiveSourceSafetyError,
    Orchestrator,
    TickResult,
    build_stage_transitions,
    enter_never,
    stage_enter_check,
)
from inference.desktop.perception.frame_source import FrameData, LerobotFrameSource
from inference.desktop.perception.dex1_state_source import Dex1StateData
from inference.desktop.perception.joint_state_source import JointStateData
from inference.desktop.perception.stream import DetectionStream
from inference.desktop.perception.yolo_obb import OBBDetection


# =========================================================================
# helpers
# =========================================================================
def _det(
    class_name: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    confidence: float = 0.9,
) -> OBBDetection:
    verts = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
    )
    return OBBDetection(
        class_id=0, class_name=class_name, confidence=confidence, verts=verts
    )


def _empty_frame(t: int) -> FrameData:
    """t を rgb[0,0,0] に埋め込んだ 1x1 frame (Stub がこれで frame_id を復元)。"""
    rgb = np.array([[[t % 256]]], dtype=np.uint8) * np.ones(
        (1, 1, 3), dtype=np.uint8
    )
    return FrameData(rgb=rgb, t=t)


class _StubPerception:
    """t (FrameData.t) → 予め決めた OBBDetection list を返す stub。"""

    def __init__(self, dets_by_t: dict[int, list[OBBDetection]]):
        self._dets_by_t = dets_by_t
        self._last_t: int = -1
        self._call_history: list[int] = []

    def predict(self, rgb: np.ndarray) -> list[OBBDetection]:
        # rgb は 1x1x3、rgb[0,0,0] を frame_t として復元
        # ただし next tick で受け取った rgb は必ずしも tick から直接来ないので、
        # 呼び出し順序で t を assign する形が確実
        self._last_t += 1
        self._call_history.append(self._last_t)
        return self._dets_by_t.get(self._last_t, [])


def _base_cleanup_config() -> dict:
    return {
        "max_count": {"leg": 4, "hand_left": 1, "hand_right": 1, "table_top": 1, "workspace": 1, "hole": 8, "leg_tip": 4},
        "over_max_continue_iou": 0.3,
        "under_max_similar_iou": 0.3,
        "median_filter": {"enabled": False, "iou_match_min": 0.5},  # test は cleaner を透過寄りで
    }


def _make_registry() -> dict[str, MockSkill]:
    return {
        name: MockSkill(name)
        for name in [
            "move_to_table",
            "move_table_base",
            "pick_table_leg",
            "insert_table_leg",
            "rotate_leg_to_tighten",
            "flip_table",
        ]
    }


# =========================================================================
# Orchestrator の基本挙動
# =========================================================================
class TestInitialization:
    def test_initial_state_before_any_tick(self):
        """init 時点で dispatcher はまだ start されてない (lazy start)。"""
        perception = _StubPerception({})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)
        assert orch.state.current_skill == "move_to_table"
        assert dispatcher.active_skill_name is None  # lazy: tick まで start しない


def test_operator_policy_never_advances_on_completion_or_old_deadline():
    class CompletedPolicy(MockSkill):
        @property
        def is_complete(self):
            return True

    registry = {
        "pick_table_leg": CompletedPolicy("pick_table_leg", max_dwell_sec=1),
        "retry_open_pick_table_leg": MockSkill("retry_open_pick_table_leg"),
        "arm_transition_pick_table_leg_to_insert_table_leg": MockSkill(
            "arm_transition_pick_table_leg_to_insert_table_leg"
        ),
    }
    dispatcher = SkillDispatchLowerPolicy(registry)
    orch = Orchestrator(
        _StubPerception({}), DetectionStream(_base_cleanup_config()), dispatcher,
        initial_skill="pick_table_leg",
        transitions={
            "pick_table_leg": ["arm_transition_pick_table_leg_to_insert_table_leg"],
        },
        hard_timeout_by_skill={"pick_table_leg": 1},
        operator_console=object(), operator_stage=1,
        operator_retry_routes={"pick_table_leg": "retry_open_pick_table_leg"},
    )
    dispatcher.start("pick_table_leg", {})
    assert orch.advance_finished_skill(now=1).fired_to is None
    assert orch.advance_finished_skill(now=10_000).fired_to is None
    assert dispatcher.active_skill_name == "pick_table_leg"
    orch._operator_command("r")
    assert dispatcher.active_skill_name == "retry_open_pick_table_leg"


def test_operator_next_is_disabled_for_single_stage_terminal_policy():
    dispatcher = SkillDispatchLowerPolicy({"flip_table": MockSkill("flip_table")})
    orch = Orchestrator(
        _StubPerception({}), DetectionStream(_base_cleanup_config()), dispatcher,
        initial_skill="flip_table", transitions={"flip_table": []},
        operator_console=object(), operator_stage=5,
    )
    dispatcher.start("flip_table", {})
    orch._operator_command("n")
    assert dispatcher.active_skill_name == "flip_table"
    assert orch._operator_finished is False


def test_operator_preparation_dwell_cannot_skip_unreached_pose():
    dispatcher = SkillDispatchLowerPolicy({
        "arm_transition_pick_table_leg_to_insert_table_leg": MockSkill(
            "arm_transition_pick_table_leg_to_insert_table_leg", max_dwell_sec=1
        ),
        "operator_gate_for_insert_table_leg": MockSkill(
            "operator_gate_for_insert_table_leg"
        ),
    })
    source = "arm_transition_pick_table_leg_to_insert_table_leg"
    orch = Orchestrator(
        _StubPerception({}), DetectionStream(_base_cleanup_config()), dispatcher,
        initial_skill=source,
        transitions={source: ["operator_gate_for_insert_table_leg"]},
        operator_console=object(), operator_stage=2,
    )
    dispatcher.start(source, {})
    assert orch.advance_finished_skill(now=1).fired_to is None
    with pytest.raises(LiveSourceSafetyError, match="without verified completion"):
        orch.advance_finished_skill(now=3)
    assert dispatcher.active_skill_name == source


def test_operator_next_moves_then_waits_for_explicit_enter_gate():
    class Finite(MockSkill):
        def __init__(self, name, complete=False):
            super().__init__(name)
            self.complete = complete

        @property
        def is_complete(self):
            return self.complete

    transition = Finite("arm_transition_pick_table_leg_to_insert_table_leg", True)
    gate = Finite("operator_gate_for_insert_table_leg")
    registry = {
        "pick_table_leg": Finite("pick_table_leg", True),
        transition.name: transition,
        gate.name: gate,
        "insert_table_leg": Finite("insert_table_leg"),
    }
    dispatcher = SkillDispatchLowerPolicy(registry)
    orch = Orchestrator(
        _StubPerception({}), DetectionStream(_base_cleanup_config()), dispatcher,
        initial_skill="pick_table_leg",
        transitions={
            "pick_table_leg": [transition.name],
            transition.name: [gate.name],
            gate.name: ["insert_table_leg"],
            "insert_table_leg": [],
        },
        operator_console=object(), operator_stage=2,
    )
    dispatcher.start("pick_table_leg", {})
    assert orch.advance_finished_skill(now=1).fired_to is None
    orch._operator_command("n")
    assert dispatcher.active_skill_name == transition.name
    assert orch.advance_finished_skill(now=2).fired_to == gate.name
    assert orch.advance_finished_skill(now=3).fired_to is None
    assert dispatcher.active_skill_name == gate.name
    gate.complete = True  # a fresh Enter at the reached initial pose
    assert orch.advance_finished_skill(now=4).fired_to == "insert_table_leg"


def test_operator_retry_discards_inference_result_that_arrived_after_keypress():
    class SlowResult(MockSkill):
        def step(self, obs):
            return np.full(14, 0.9)

    class Console:
        def __init__(self):
            self.pending = "r"

        def poll(self):
            key, self.pending = self.pending, None
            return key

    commands = []
    dispatcher = SkillDispatchLowerPolicy({
        "pick_table_leg": SlowResult("pick_table_leg"),
        "retry_open_pick_table_leg": MockSkill("retry_open_pick_table_leg"),
    })
    orch = Orchestrator(
        _StubPerception({}), DetectionStream(_base_cleanup_config()), dispatcher,
        initial_skill="pick_table_leg",
        transitions={"pick_table_leg": []},
        actuator_send_fn=lambda target: commands.append(np.asarray(target).copy()),
        operator_console=Console(), operator_stage=1,
        operator_retry_routes={"pick_table_leg": "retry_open_pick_table_leg"},
        operator_hold_target=lambda: np.full(14, 0.1),
    )
    orch.tick(_empty_frame(t=0))
    assert dispatcher.active_skill_name == "retry_open_pick_table_leg"
    np.testing.assert_allclose(commands[-1], 0.1)


@pytest.mark.parametrize(
    "gate_name, phase",
    [
        # R のやり直しで脚を置いた後の Enter (insert / tighten)。名前が `retry_hand_` で
        # 始まるので、以前は「ハンド初期姿勢へ移動中 (キー無し)」に上書きされ、Enter が
        # 永久に捨てられていた (Issue #172)。
        ("retry_hand_gate_insert_table_leg", "insert_table_leg／脚配置・ハンド初期幅待ち"),
        ("operator_gate_for_insert_table_leg", "insert_table_leg／開始待ち"),
        ("retry_wait_insert_table_leg", "insert_table_leg／腕保持・ハンド全開"),
    ],
)
def test_operator_view_of_a_gate_is_not_overwritten(gate_name, phase):
    from inference.desktop.lower_policy.skills.operator_gate import (
        OperatorConfirmationHoldSkill,
    )
    from inference.desktop.operator_console import OperatorConsole

    console = OperatorConsole()  # start() しない: キーの受け付け判定だけを見る
    gate = OperatorConfirmationHoldSkill(name=gate_name, next_skill_name="insert_table_leg")
    dispatcher = SkillDispatchLowerPolicy({gate_name: gate})
    orch = Orchestrator(
        _StubPerception({}), DetectionStream(_base_cleanup_config()), dispatcher,
        initial_skill=gate_name, transitions={gate_name: []},
        operator_console=console, operator_stage=2,
    )
    dispatcher.start(gate_name, {})
    key = "r" if gate_name.startswith("retry_wait_") else "enter"
    console.show(2, phase, (key,))  # gate の入力 thread が出す問い
    for _ in range(3):
        orch._show_operator_view()  # run_live が毎周期呼ぶ
    assert console._view == (2, phase, (key,))


def test_run_live_does_not_take_keys_while_a_gate_waits(monkeypatch):
    """gate の入力 thread と run_live が同じキーを取り合わない (Issue #172)。"""
    _patch_clock(monkeypatch)

    class Gate(MockSkill):
        owns_operator_view = True

    class Console:
        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None

        def show(self, *_args):
            pass

    frames = iter([_empty_frame(t=i) for i in range(5)])

    class _Source:
        def get(self):
            try:
                return next(frames)
            except StopIteration:
                raise KeyboardInterrupt

    console = Console()
    dispatcher = SkillDispatchLowerPolicy({
        "operator_gate_for_pick_table_leg": Gate("operator_gate_for_pick_table_leg"),
    })
    orch = Orchestrator(
        _StubPerception({}), DetectionStream(_base_cleanup_config()), dispatcher,
        initial_skill="operator_gate_for_pick_table_leg",
        transitions={"operator_gate_for_pick_table_leg": []},
        operator_console=console, operator_stage=1,
    )
    with pytest.raises(KeyboardInterrupt):
        orch.run_live(_Source(), hz=30.0, startup_timeout=1.0)
    assert dispatcher.active_skill_name == "operator_gate_for_pick_table_leg"
    assert console.polls == 0

def test_rejects_unknown_head_perception_view():
    with pytest.raises(ValueError, match="head_perception_view"):
        Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
            head_perception_view="center",
        )


class TestHeadStereoRouting:
    def _make_orch(self, view: str) -> Orchestrator:
        return Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
            head_perception_view=view,
        )

    def test_left_detector_view_preserves_packed_policy_observation(self):
        left = np.full((4, 5, 3), 11, dtype=np.uint8)
        right = np.full((4, 5, 3), 22, dtype=np.uint8)
        packed = np.concatenate([left, right], axis=1)
        orch = self._make_orch("left")

        assert np.array_equal(orch._perception_rgb(packed), left)
        obs = orch._build_obs(FrameData(rgb=packed, t=1), cleaned=[])
        assert obs["head_rgb"] is packed

    def test_right_detector_view_selects_right_eye(self):
        left = np.full((4, 5, 3), 11, dtype=np.uint8)
        right = np.full((4, 5, 3), 22, dtype=np.uint8)
        packed = np.concatenate([left, right], axis=1)
        assert np.array_equal(
            self._make_orch("right")._perception_rgb(packed), right
        )

    def test_odd_width_is_rejected_when_split_requested(self):
        with pytest.raises(ValueError, match="positive even width"):
            self._make_orch("left")._perception_rgb(
                np.zeros((4, 5, 3), dtype=np.uint8)
            )


class TestTickWarmup:
    def test_first_tick_auto_starts_initial_skill(self):
        """初回 tick で dispatcher が initial skill を auto start。"""
        perception = _StubPerception({0: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        result = orch.tick(_empty_frame(t=0))
        # cleaner が溜まるまで Planner の判定は無いが (cleaned=None)、skill は止めない
        # (Issue #141 D3)。dispatcher は auto start してる。
        assert result is not None and result.cleaned is None
        assert result.detection_refreshed is True
        assert dispatcher.active_skill_name == "move_to_table"

    def test_second_tick_returns_result_median_disabled(self):
        """median disabled 時、2 push 目 (T=1) で TickResult 返る。"""
        perception = _StubPerception({0: [], 1: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        orch.tick(_empty_frame(t=0))
        result = orch.tick(_empty_frame(t=1))
        assert result is not None
        assert result.t == 1
        assert result.current_skill == "move_to_table"
        assert result.fire_transition_to is None

    def test_duplicate_t_skips_detection_but_still_steps_the_skill(self):
        """同じ frame.t の tick では YOLO を掛け直さない (Issue #141 D4)。

        同じ frame を filter に 2 度入れると「前の frame にも出ていた」の判定が崩れる。
        skill は 30 Hz で進めたいので、tick 自体は止めない。
        """
        perception = _StubPerception({0: [], 1: [], 2: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        orch.tick(_empty_frame(t=0))
        orch.tick(_empty_frame(t=1))  # ここで warmup 完了
        fresh = orch.tick(_empty_frame(t=2))
        assert fresh.detection_refreshed is True and fresh.cleaned is not None
        # 重複: 同じ t を再度渡すと検出は据え置き (Planner の判定もしない)
        dup = orch.tick(_empty_frame(t=2))
        assert dup.detection_refreshed is False
        assert dup.cleaned is None
        assert dup.policy_cleaned is fresh.policy_cleaned


# =========================================================================
# Action forwarding (Type B skill の action tensor が actuator に流れるか)
# =========================================================================
class _TypeBStub(Skill):
    def __init__(self, name: str, action: np.ndarray) -> None:
        super().__init__()
        self.name = name  # type: ignore[misc]
        self._action = action

    def _on_start(self, params: dict) -> None: pass
    def _on_stop(self) -> None: pass
    def step(self, obs: dict) -> Optional[np.ndarray]:
        return self._action


class TestActionForwarding:
    def test_action_forwarded_when_send_fn_set(self):
        """Type B skill の action が actuator_send_fn に流れる。"""
        action = np.array([1.0, 2.0], dtype=np.float32)
        registry = _make_registry()
        # move_to_table を Type B stub に差し替え
        registry["move_to_table"] = _TypeBStub("move_to_table", action)
        dispatcher = SkillDispatchLowerPolicy(registry)

        sent: list[np.ndarray] = []
        perception = _StubPerception({0: [], 1: []})
        cleaner = DetectionStream(_base_cleanup_config())
        orch = Orchestrator(
            perception, cleaner, dispatcher,
            actuator_send_fn=lambda a: sent.append(a),
        )
        orch.tick(_empty_frame(t=0))  # cleaner は溜まっていないが skill は動く
        orch.tick(_empty_frame(t=1))
        assert len(sent) == 2
        np.testing.assert_array_equal(sent[0], action)

    def test_action_dropped_when_send_fn_none(self):
        """actuator_send_fn=None なら action は silently drop (log 経由で確認可)。"""
        action = np.array([1.0], dtype=np.float32)
        registry = _make_registry()
        registry["move_to_table"] = _TypeBStub("move_to_table", action)
        dispatcher = SkillDispatchLowerPolicy(registry)
        perception = _StubPerception({0: [], 1: []})
        cleaner = DetectionStream(_base_cleanup_config())
        orch = Orchestrator(
            perception, cleaner, dispatcher, actuator_send_fn=None,
        )
        orch.tick(_empty_frame(t=0))
        result = orch.tick(_empty_frame(t=1))
        assert result is not None
        np.testing.assert_array_equal(result.action, action)


# =========================================================================
# JSONL log sink
# =========================================================================
class TestLogSink:
    def test_log_writes_jsonl_per_tick(self):
        perception = _StubPerception({0: [], 1: [], 2: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        sink = io.StringIO()
        orch = Orchestrator(perception, cleaner, dispatcher, log_sink=sink)

        orch.tick(_empty_frame(t=0))  # cleaner が溜まる前の tick も記録する
        orch.tick(_empty_frame(t=1))
        orch.tick(_empty_frame(t=2))

        # tick log は current_skill を持つが event field は無い (skill_transition_snapshot
        # と区別)。tick log 行だけ拾う (event を持つ行は除外)。
        lines = sink.getvalue().strip().split("\n")
        tick_lines = [
            line for line in lines
            if line.strip() and '"event"' not in line
        ]
        assert len(tick_lines) == 3
        for line in tick_lines:
            payload = json.loads(line)
            assert "t" in payload
            assert "current_skill" in payload
            assert "fire_transition_to" in payload
            assert "n_cleaned" in payload



# =========================================================================
# run loop
# =========================================================================
class TestRunLoop:
    def test_run_loop_ends_on_source_exhaustion(self):
        """LerobotFrameSource が exhaust したら run() が return する。"""
        frames = [_empty_frame(t=i) for i in range(5)]
        source = LerobotFrameSource(frames)
        perception = _StubPerception({i: [] for i in range(5)})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        # hz=None で sleep なし、5 tick で source 尽きて自然終了
        orch.run(source, hz=None)
        assert dispatcher.active_skill_name == "move_to_table"


# =========================================================================
# JointState source integration (Issue #75)
# =========================================================================
class _StubJointStateSource:
    """`.get() → Optional[JointStateData]` を返す stub。"""

    def __init__(self, snapshot: Optional[JointStateData]):
        self._snapshot = snapshot

    def get(self) -> Optional[JointStateData]:
        return self._snapshot


class TestJointStateInObs:
    """orchestrator._build_obs が joint_state_source の値を obs に載せることを確認。"""

    def _make_orch(self, joint_state_source: Optional[object] = None) -> Orchestrator:
        perception = _StubPerception({})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        return Orchestrator(
            perception, cleaner, dispatcher,
            joint_state_source=joint_state_source,
        )

    def test_no_source_no_joint_state_field(self):
        """joint_state_source=None なら obs に joint_state field を追加しない (後方互換)。"""
        orch = self._make_orch(joint_state_source=None)
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert "joint_state" not in obs
        # 既存 field は維持
        assert "head_rgb" in obs and "t" in obs and "cleaned" in obs

    def test_source_snapshot_populates_obs(self):
        """source.get() が JointStateData を返せば obs["joint_state"] にそのまま乗る。"""
        snapshot = JointStateData(
            name=("j0", "j1"),
            position=np.array([0.1, 0.2], dtype=np.float64),
            velocity=np.zeros(2),
            effort=np.zeros(2),
            t=123_456_789,
        )
        source = _StubJointStateSource(snapshot=snapshot)
        orch = self._make_orch(joint_state_source=source)
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert "joint_state" in obs
        assert obs["joint_state"] is snapshot

    def test_source_returns_none_still_adds_field(self):
        """source.get() が None (未受信) でも obs に field は追加 (値 None)。
        skill 側で「obs['joint_state'] is None なら未受信」を明示 handle させる。"""
        source = _StubJointStateSource(snapshot=None)
        orch = self._make_orch(joint_state_source=source)
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert "joint_state" in obs
        assert obs["joint_state"] is None


class TestJointStateFreshness:
    """受信時刻を持つ関節 state (公式 :5557) が止まったら安全停止する。"""

    @staticmethod
    def _snapshot(age_s: float) -> JointStateData:
        now = time.monotonic_ns()
        return JointStateData(
            name=("j0",),
            position=np.zeros(1),
            velocity=np.zeros(1),
            effort=np.zeros(1),
            t=now,
            received_monotonic_ns=now - int(age_s * 1e9),
        )

    def _orch(self, snapshot) -> Orchestrator:
        return Orchestrator(
            _StubPerception({0: []}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
            joint_state_source=_StubJointStateSource(snapshot=snapshot),
        )

    def test_stale_joint_state_stops_the_run(self):
        orch = self._orch(self._snapshot(age_s=2.0))
        with pytest.raises(LiveSourceSafetyError, match="joint state is stale"):
            orch.tick(_empty_frame(t=0))

    def test_fresh_joint_state_passes(self):
        orch = self._orch(self._snapshot(age_s=0.0))
        orch.tick(_empty_frame(t=0))

    def test_state_without_receive_time_is_not_judged(self):
        import dataclasses

        snapshot = dataclasses.replace(
            self._snapshot(age_s=0.0), received_monotonic_ns=None
        )
        orch = self._orch(snapshot)
        orch.tick(_empty_frame(t=0))


class TestHandStateFreshness:
    """実測の Dex1 state (会場の gripper_q) が止まったら安全停止する。"""

    @staticmethod
    def _orch(snapshot) -> Orchestrator:
        return Orchestrator(
            _StubPerception({0: []}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
            dex1_state_source=_StubDex1StateSource(snapshot),
        )

    @staticmethod
    def _state(age_s: float, *, measured: bool) -> Dex1StateData:
        received = time.monotonic_ns() - int(age_s * 1e9)
        return Dex1StateData(
            position_rad=np.asarray([2.0, 2.0]),
            left_received_monotonic_ns=received,
            right_received_monotonic_ns=received,
            t=received,
            measured=measured,
        )

    def test_stale_measured_hand_stops_the_run(self):
        orch = self._orch(self._state(2.0, measured=True))
        with pytest.raises(LiveSourceSafetyError, match="Dex1 state is stale"):
            orch.tick(_empty_frame(t=0))

    def test_synthetic_hand_is_not_judged(self):
        self._orch(self._state(2.0, measured=False)).tick(_empty_frame(t=0))

    def test_fresh_measured_hand_passes(self):
        self._orch(self._state(0.0, measured=True)).tick(_empty_frame(t=0))


class _StubDex1StateSource:
    def __init__(self, snapshot: Optional[Dex1StateData]) -> None:
        self._snapshot = snapshot

    def get(self) -> Optional[Dex1StateData]:
        return self._snapshot


class TestDex1StateInObs:
    def test_bilateral_state_is_forwarded_without_conversion(self):
        snapshot = Dex1StateData(
            position_rad=np.asarray([5.1, 0.3], dtype=np.float32),
            left_received_monotonic_ns=10,
            right_received_monotonic_ns=11,
            t=11,
        )
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
            dex1_state_source=_StubDex1StateSource(snapshot),
        )
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert obs["hand_state"] is snapshot

    def test_no_source_keeps_backward_compatible_obs(self):
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
        )
        assert "hand_state" not in orch._build_obs(_empty_frame(t=0), cleaned=[])


# =========================================================================
# Wrist camera source integration (Issue #75)
# =========================================================================
class _StubFrameSource:
    """`.get() → Optional[FrameData]` を返す stub (Ros2FrameSource duck-type)。"""

    def __init__(self, snapshot: Optional[FrameData]):
        self._snapshot = snapshot

    def get(self) -> Optional[FrameData]:
        return self._snapshot


class TestWristSourceInObs:
    """orchestrator._build_obs が wrist_{left,right}_source の値を obs に載せることを確認。
    SampleVLASkill (Mock) は obs 使わないが、real VLA drop-in で obs pass-through が
    通ることを事前 verify する pipeline test。"""

    def _make_orch(
        self,
        wrist_left_source=None,
        wrist_right_source=None,
    ) -> Orchestrator:
        perception = _StubPerception({})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        return Orchestrator(
            perception, cleaner, dispatcher,
            wrist_left_source=wrist_left_source,
            wrist_right_source=wrist_right_source,
        )

    def test_no_wrist_sources_no_wrist_fields(self):
        """両 wrist source=None なら obs に wrist field 追加しない (後方互換)。"""
        orch = self._make_orch()
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert "wrist_left_rgb" not in obs
        assert "wrist_right_rgb" not in obs
        # 既存 field は維持
        assert "head_rgb" in obs and "t" in obs and "cleaned" in obs

    def test_wrist_sources_populate_obs(self):
        """wrist_{left,right}_source が渡されれば obs にそれぞれの latest snapshot 追加。"""
        left_frame = _empty_frame(t=100)
        right_frame = _empty_frame(t=200)
        orch = self._make_orch(
            wrist_left_source=_StubFrameSource(snapshot=left_frame),
            wrist_right_source=_StubFrameSource(snapshot=right_frame),
        )
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert obs["wrist_left_rgb"] is left_frame
        assert obs["wrist_right_rgb"] is right_frame

    def test_wrist_source_returns_none_still_adds_field(self):
        """source.get() が None (未受信) でも obs に field は追加 (値 None)。
        real VLA 側で「obs['wrist_left_rgb'] is None なら未受信」を明示 handle させる。"""
        orch = self._make_orch(
            wrist_left_source=_StubFrameSource(snapshot=None),
            wrist_right_source=_StubFrameSource(snapshot=None),
        )
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert obs["wrist_left_rgb"] is None
        assert obs["wrist_right_rgb"] is None

    def test_one_side_only_adds_only_that_side(self):
        """片方だけ渡した場合は片方のみ obs 追加 (別々 opt-in)。"""
        left_frame = _empty_frame(t=100)
        orch = self._make_orch(
            wrist_left_source=_StubFrameSource(snapshot=left_frame),
            # right は None
        )
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert obs["wrist_left_rgb"] is left_frame
        assert "wrist_right_rgb" not in obs


# =========================================================================
# Live source safety (Issue #64: run_live + auto-transition)
# =========================================================================
class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _ScriptedLiveSource:
    def __init__(self, values: list[Optional[FrameData]]) -> None:
        self._values = iter(values)

    def get(self) -> Optional[FrameData]:
        try:
            return next(self._values)
        except StopIteration:
            raise KeyboardInterrupt


def _patch_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(
        "inference.desktop.orchestrator.time.monotonic", clock.monotonic
    )
    monkeypatch.setattr("inference.desktop.orchestrator.time.sleep", clock.sleep)
    return clock


class TestLiveRunLoop:
    def test_waits_for_first_frame_before_starting_skill(self, monkeypatch):
        _patch_clock(monkeypatch)
        source = _ScriptedLiveSource([None, None, _empty_frame(t=0)])
        perception = _StubPerception({0: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        with pytest.raises(KeyboardInterrupt):
            orch.run_live(source, hz=10.0, startup_timeout=1.0)

        assert dispatcher.active_skill_name == "move_to_table"

    def test_startup_timeout_does_not_start_skill(self, monkeypatch):
        _patch_clock(monkeypatch)

        class _NoFrameSource:
            def get(self):
                return None

        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
        )

        with pytest.raises(LiveSourceSafetyError, match="startup timeout"):
            orch.run_live(_NoFrameSource(), hz=10.0, startup_timeout=0.2)

        assert dispatcher.active_skill_name is None

    def _tick_starts(self, monkeypatch, work_s: float, hz: float = 30.0) -> list[float]:
        """1 tick の処理に work_s かかる source で run_live を回し、各 tick の開始時刻を返す。"""
        clock = _patch_clock(monkeypatch)
        starts: list[float] = []
        frames = iter([_empty_frame(t=i) for i in range(40)])

        class _WorkingSource:
            def get(self):
                starts.append(clock.now)
                clock.now += work_s
                try:
                    return next(frames)
                except StopIteration:
                    raise KeyboardInterrupt

        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
        )
        with pytest.raises(KeyboardInterrupt):
            orch.run_live(_WorkingSource(), hz=hz, startup_timeout=1.0, frame_timeout=0.5)
        return starts

    def test_an_overrun_tick_does_not_add_an_extra_period(self, monkeypatch):
        """処理が周期を超えても次の周に dt を足さない: 40 ms の処理なら 40 ms 間隔 (B2a-06)。"""
        intervals = np.diff(self._tick_starts(monkeypatch, work_s=0.040)[2:])
        assert intervals == pytest.approx(np.full(len(intervals), 0.040), abs=1e-9)

    def test_a_short_tick_keeps_the_period_without_catching_up(self, monkeypatch):
        """処理が周期より短ければ従来どおり 1/hz。速くはならない。"""
        intervals = np.diff(self._tick_starts(monkeypatch, work_s=0.010)[2:])
        assert intervals == pytest.approx(np.full(len(intervals), 1.0 / 30.0), abs=1e-9)

    def test_stale_frame_raises_safety_error(self, monkeypatch):
        _patch_clock(monkeypatch)
        frame = _empty_frame(t=0)

        class _StaleSource:
            def get(self):
                return frame

        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({0: []}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
        )

        with pytest.raises(LiveSourceSafetyError, match="frame timeout"):
            orch.run_live(
                _StaleSource(),
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.2,
            )

        assert dispatcher.active_skill_name == "move_to_table"

    def _make_registry_with_dwell(self, dwell_map: dict[str, float]) -> dict[str, MockSkill]:
        """dwell 秒を skill instance に注入した registry (Issue #81、Skill.max_dwell_sec 経由)。"""
        return {
            name: MockSkill(name, max_dwell_sec=dwell_map.get(name))
            for name in [
                "move_to_table", "move_table_base", "pick_table_leg",
                "insert_table_leg", "rotate_leg_to_tighten", "flip_table",
            ]
        }

    def test_skill_max_dwell_auto_transitions_to_next_candidate(self, monkeypatch):
        """Skill.max_dwell_sec 到達で TRANSITIONS graph の first candidate に
        auto-transition される (Issue #64 決め打ちルール、Issue #81 で skill が
        自分の dwell を持つ形に refactor)。domain gap で YOLO fire が来なくても
        pipeline を止めずに次 skill = move_table_base へ進む。"""
        _patch_clock(monkeypatch)

        dispatcher = SkillDispatchLowerPolicy(
            self._make_registry_with_dwell({"move_to_table": 0.2})
        )
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
        )

        # dwell 到達後は auto-transition = 例外 raise せず pipeline 継続。
        # test 側から強制終了させるため source が途中で KeyboardInterrupt を
        # 出す form (`_ScriptedLiveSource` を N frame → StopIteration → interrupt)。
        n_frames = 200
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(n_frames)])
        with pytest.raises(KeyboardInterrupt):
            orch.run_live(
                source,
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.5,
            )

        # move_to_table → move_table_base に auto-transition された
        assert dispatcher.active_skill_name == "move_table_base"

    def test_hard_timeout_stops_without_transition_when_configured(self, monkeypatch):
        """`on_timeout: stop` の skill は時間切れで止まる (Issue #141 D4)。

        時間切れは「うまくいった」ではないので、止める側を選んだ skill では遷移しない。
        """
        _patch_clock(monkeypatch)
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="move_table_base",
            hard_timeout_by_skill={"move_table_base": 0.2},
            timeout_action_by_skill={"move_table_base": "stop"},
        )
        source = _ScriptedLiveSource(
            [_empty_frame(t=i) for i in range(20)]
        )

        with pytest.raises(LiveSourceSafetyError, match="without transition"):
            orch.run_live(
                source,
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.5,
            )

        assert dispatcher.active_skill_name == "move_table_base"

    def test_hard_timeout_advances_to_the_next_skill_by_default(self, monkeypatch):
        """既定は「次の skill へ進む」(Issue #141 束 1-13 / D4、ユーザー決定)。"""
        _patch_clock(monkeypatch)
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="move_table_base",
            hard_timeout_by_skill={"move_table_base": 0.2},
        )
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(20)])

        with pytest.raises(KeyboardInterrupt):        # source 尽きるまで走り続ける
            orch.run_live(source, hz=10.0, startup_timeout=1.0, frame_timeout=0.5)

        assert dispatcher.active_skill_name == "pick_table_leg"

    def test_hard_timeout_advance_is_recorded_as_a_timeout_fallback(self, monkeypatch):
        """時間切れで進んだ遷移は、成功で進んだ遷移と記録で区別できること。

        端末には 1 行出るだけなので、run の後で `[orch] skill_timeout` を
        読み返すことになる。JSONL に残しておけば機械的に分けられる (Issue #152)。
        """
        _patch_clock(monkeypatch)
        sink = io.StringIO()
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="move_table_base",
            hard_timeout_by_skill={"move_table_base": 0.2},
            log_sink=sink,
        )
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(20)])

        with pytest.raises(KeyboardInterrupt):
            orch.run_live(source, hz=10.0, startup_timeout=1.0, frame_timeout=0.5)

        events = [
            json.loads(line)
            for line in sink.getvalue().splitlines()
            if '"event"' in line
        ]
        fallback = [e for e in events if e["event"] == "skill_timeout_fallback"]
        assert fallback, "時間切れの遷移が記録されていない"
        assert fallback[0]["skill"] == "move_table_base"
        assert fallback[0]["next_skill"] == "pick_table_leg"
        assert fallback[0]["timeout_s"] == 0.2

    def test_rejects_unknown_timeout_action(self):
        with pytest.raises(ValueError, match="on_timeout"):
            Orchestrator(
                _StubPerception({}),
                DetectionStream(_base_cleanup_config()),
                SkillDispatchLowerPolicy(_make_registry()),
                timeout_action_by_skill={"move_table_base": "retry"},
            )

    def test_rejects_nonpositive_hard_timeout(self):
        with pytest.raises(ValueError, match="hard timeout"):
            Orchestrator(
                _StubPerception({}),
                DetectionStream(_base_cleanup_config()),
                SkillDispatchLowerPolicy(_make_registry()),
                hard_timeout_by_skill={"move_table_base": 0.0},
            )

    def test_terminal_hard_timeout_finishes_the_stage(self, monkeypatch):
        _patch_clock(monkeypatch)
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="move_table_base",
            transitions={"move_table_base": []},
            hard_timeout_by_skill={"move_table_base": 0.2},
        )
        source = _ScriptedLiveSource(
            [_empty_frame(t=i) for i in range(20)]
        )

        # 進む先が無い skill の時間切れは stage の終わり (既定 advance でも同じ)。
        orch.run_live(
            source,
            hz=10.0,
            startup_timeout=1.0,
            frame_timeout=0.5,
        )

        assert dispatcher.active_skill_name == "move_table_base"

    def test_terminal_hard_timeout_stops_when_the_yaml_says_stop(self, monkeypatch):
        """終端の skill でも `on_timeout: stop` を書いたら止まること (review 指摘 10)。

        起動側の都合 (stage を続けたい) が、YAML に書いた「止める」を
        黙って上書きしないようにする。
        """
        _patch_clock(monkeypatch)
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="move_table_base",
            transitions={"move_table_base": []},
            hard_timeout_by_skill={"move_table_base": 0.2},
            timeout_action_by_skill={"move_table_base": "stop"},
        )
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(20)])
        with pytest.raises(LiveSourceSafetyError, match="max_seconds_hard"):
            orch.run_live(source, hz=10.0, startup_timeout=1.0, frame_timeout=0.5)

    def test_observation_driven_skill_transitions_only_after_complete(self, monkeypatch):
        _patch_clock(monkeypatch)

        class _CompleteAfterStep(Skill):
            name = "arm_pre_motion"

            def __init__(self):
                super().__init__()
                self.complete = False

            def _on_start(self, params):
                self.complete = False

            def _on_stop(self):
                pass

            def step(self, obs):
                self.complete = True
                return np.zeros(14)

            @property
            def is_complete(self):
                return self.complete

        dispatcher = SkillDispatchLowerPolicy(
            {
                "arm_pre_motion": _CompleteAfterStep(),
                "move_table_base": MockSkill("move_table_base"),
            }
        )
        log = io.StringIO()
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="arm_pre_motion",
            transitions={"arm_pre_motion": ["move_table_base"]},
            enter_check={"move_table_base": lambda _d, _s: False},
            log_sink=log,
        )
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(5)])
        with pytest.raises(KeyboardInterrupt):
            orch.run_live(source, hz=10.0, startup_timeout=1.0)

        assert dispatcher.active_skill_name == "move_table_base"

    def test_completed_terminal_skill_finishes_stage_cleanly(self, monkeypatch):
        _patch_clock(monkeypatch)

        class _CompleteAfterStep(Skill):
            name = "terminal_pre_motion"

            def __init__(self):
                super().__init__()
                self.complete = False

            def _on_start(self, params):
                self.complete = False

            def _on_stop(self):
                pass

            def step(self, obs):
                self.complete = True
                return np.zeros(14)

            @property
            def is_complete(self):
                return self.complete

        dispatcher = SkillDispatchLowerPolicy(
            {"terminal_pre_motion": _CompleteAfterStep()}
        )
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="terminal_pre_motion",
            transitions={"terminal_pre_motion": []},
            enter_check={},
        )
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(5)])

        orch.run_live(source, hz=10.0, startup_timeout=1.0)

        assert dispatcher.active_skill_name == "terminal_pre_motion"

    @staticmethod
    def _short_of_target_skill(name: str):
        """目標に届かないまま終わった (failure_reason を立てた) finite skill。"""

        class _ShortSkill(Skill):
            def __init__(self):
                super().__init__()
                self.name = name
                self.failure_reason = None

            def _on_start(self, params):
                self.failure_reason = None

            def _on_stop(self):
                pass

            @property
            def is_complete(self):
                return False

            def step(self, obs):
                self.failure_reason = "IK unreachable"
                return np.zeros(14)

        return _ShortSkill()

    def test_a_skill_that_ends_short_of_its_target_holds(self, monkeypatch):
        """Finite preparation failures never dispatch the next policy."""
        _patch_clock(monkeypatch)
        dispatcher = SkillDispatchLowerPolicy(
            {
                "arm_transition_a_to_b": self._short_of_target_skill("arm_transition_a_to_b"),
                "hold_transition_a_to_b": MockSkill("hold_transition_a_to_b"),
            }
        )
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="arm_transition_a_to_b",
            transitions={
                "arm_transition_a_to_b": ["hold_transition_a_to_b"],
                "hold_transition_a_to_b": [],
            },
            enter_check={},
        )
        orch.tick(_empty_frame(t=1))
        with pytest.raises(LiveSourceSafetyError, match="IK unreachable"):
            orch.advance_finished_skill(now=1.0)
        assert dispatcher.active_skill_name == "arm_transition_a_to_b"

    def test_a_terminal_skill_that_ends_short_holds(self, monkeypatch):
        _patch_clock(monkeypatch)
        dispatcher = SkillDispatchLowerPolicy(
            {"pick_table_leg": self._short_of_target_skill("pick_table_leg")}
        )
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="pick_table_leg",
            transitions={"pick_table_leg": []},
            enter_check={},
        )
        with pytest.raises(LiveSourceSafetyError, match="IK unreachable"):
            orch.run_live(
                _ScriptedLiveSource([_empty_frame(t=1), _empty_frame(t=2)]),
                hz=10.0,
                startup_timeout=1.0,
            )

    @staticmethod
    def _timed_out_skill(name: str):
        """締め切りまでに目標へ届かなかった (timeout_reason を立てた) finite skill。"""

        class _TimedOutSkill(Skill):
            def __init__(self):
                super().__init__()
                self.name = name
                self.timeout_reason = None

            def _on_start(self, params):
                self.timeout_reason = None

            def _on_stop(self):
                pass

            @property
            def is_complete(self):
                return False

            def step(self, obs):
                self.timeout_reason = "did not converge within 15s"
                return np.zeros(14)

        return _TimedOutSkill()

    def test_a_skill_that_times_out_short_of_its_target_moves_on(self, monkeypatch):
        """目標に届かないまま締め切りを過ぎたのは故障ではない。時間切れと同じく次へ。"""
        _patch_clock(monkeypatch)
        log = io.StringIO()
        dispatcher = SkillDispatchLowerPolicy(
            {
                "arm_transition_a_to_b": self._timed_out_skill("arm_transition_a_to_b"),
                "hold_transition_a_to_b": MockSkill("hold_transition_a_to_b"),
            }
        )
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="arm_transition_a_to_b",
            transitions={
                "arm_transition_a_to_b": ["hold_transition_a_to_b"],
                "hold_transition_a_to_b": [],
            },
            enter_check={},
            log_sink=log,
        )
        orch.tick(_empty_frame(t=1))
        advance = orch.advance_finished_skill(now=1.0)
        assert advance.fired_to == "hold_transition_a_to_b"
        assert advance.reason == "timeout"
        assert dispatcher.active_skill_name == "hold_transition_a_to_b"
        events = [
            json.loads(line)
            for line in log.getvalue().splitlines()
            if '"event"' in line
        ]
        fallback = [e for e in events if e["event"] == "skill_timeout_fallback"]
        assert fallback and fallback[-1]["skill"] == "arm_transition_a_to_b"
        assert fallback[-1]["next_skill"] == "hold_transition_a_to_b"
        assert "did not converge" in fallback[-1]["detail"]

    def test_a_terminal_skill_that_times_out_finishes_the_stage(self, monkeypatch):
        _patch_clock(monkeypatch)
        dispatcher = SkillDispatchLowerPolicy(
            {"pick_table_leg": self._timed_out_skill("pick_table_leg")}
        )
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="pick_table_leg",
            transitions={"pick_table_leg": []},
            enter_check={},
        )
        orch.tick(_empty_frame(t=1))
        advance = orch.advance_finished_skill(now=1.0)
        assert advance.stage_finished
        assert advance.reason == "timeout"
        assert advance.fired_to is None

    def test_skill_max_dwell_raises_safety_error_when_no_transition_candidate(self, monkeypatch):
        """TRANSITIONS graph に candidates が無い skill が dwell 超過した場合、
        auto-transition できないので fallback で LiveSourceSafetyError を raise。"""
        _patch_clock(monkeypatch)

        class _FreshSource:
            def __init__(self):
                self.t = 0

            def get(self):
                frame = _empty_frame(t=self.t)
                self.t += 1
                return frame

        dispatcher = SkillDispatchLowerPolicy(
            self._make_registry_with_dwell({"move_to_table": 0.2})
        )
        # transitions = 空 dict → 遷移候補なし
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            transitions={},
        )

        with pytest.raises(LiveSourceSafetyError, match="no transition candidates"):
            orch.run_live(
                _FreshSource(),
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.5,
            )

        # 遷移不能、initial skill のまま残る
        assert dispatcher.active_skill_name == "move_to_table"

    def test_skill_max_dwell_fires_for_multiple_chained_skills(self, monkeypatch):
        """chain 対応 (Issue #81): 複数 skill が max_dwell_sec を expose していれば
        全て順に auto-transition が発火する (旧 initial_skill_timeout の 1 回きり制約撤廃)。
        pick_table_leg → insert_table_leg → rotate_leg_to_tighten の 2 段 chain を verify。"""
        _patch_clock(monkeypatch)

        dispatcher = SkillDispatchLowerPolicy(
            self._make_registry_with_dwell({
                "pick_table_leg": 0.2,
                "insert_table_leg": 0.2,
            })
        )
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="pick_table_leg",
        )

        # pick_table_leg → insert_table_leg → rotate_leg_to_tighten の 2 段 chain
        # (rotate_leg_to_tighten は max_dwell_sec None なので停止したまま)
        n_frames = 200
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(n_frames)])
        with pytest.raises(KeyboardInterrupt):
            orch.run_live(
                source,
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.5,
            )

        # 2 段 chain 経過、rotate_leg_to_tighten に到達
        assert dispatcher.active_skill_name == "rotate_leg_to_tighten"

    def test_skill_with_none_max_dwell_does_not_auto_transition(self, monkeypatch):
        """Skill.max_dwell_sec が None なら auto-transition しない (enter_check ベースのみ)。
        従来 initial_skill_timeout=None の挙動と等価。"""
        _patch_clock(monkeypatch)

        # 全 skill max_dwell_sec=None (registry default)
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
        )

        n_frames = 200
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(n_frames)])
        with pytest.raises(KeyboardInterrupt):
            orch.run_live(
                source,
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.5,
            )

        # timeout 発火せず、move_to_table のまま (YOLO fire 無しなので)
        assert dispatcher.active_skill_name == "move_to_table"

    def test_operator_boundary_returns_after_named_skill_runtime(self, monkeypatch):
        """Phase smoke ends cleanly instead of entering a later MockSkill."""
        _patch_clock(monkeypatch)
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="move_table_base",
        )
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(20)])

        orch.run_live(
            source,
            hz=10.0,
            startup_timeout=1.0,
            stop_after_skill="move_table_base",
            stop_after_s=0.3,
        )

        assert dispatcher.active_skill_name == "move_table_base"

    def test_operator_boundary_returns_immediately_on_pick(self, monkeypatch):
        _patch_clock(monkeypatch)
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="pick_table_leg",
        )
        source = _ScriptedLiveSource([_empty_frame(t=0)])

        orch.run_live(
            source,
            hz=10.0,
            startup_timeout=1.0,
            stop_on_skills=frozenset({"pick_table_leg"}),
        )

        assert dispatcher.active_skill_name == "pick_table_leg"

    def test_operator_boundary_requires_complete_runtime_pair(self):
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
        )
        with pytest.raises(ValueError, match="specified together"):
            orch.run_live(
                _ScriptedLiveSource([]),
                stop_after_skill="move_table_base",
            )


# =========================================================================
# Production 配線の regression (initial_skill="setup" + DEFAULT_*)
# =========================================================================
def _production_registry() -> dict[str, MockSkill]:
    """entrypoint.py と同じ skill 名集合 (setup を含む production 配線)。"""
    return {
        name: MockSkill(name)
        for name in [
            "setup",
            "move_to_table",
            "move_table_base",
            "pick_table_leg",
            "insert_table_leg",
            "rotate_leg_to_tighten",
            "flip_table",
        ]
    }


class TestProductionWiring:
    """production の唯一経路 (initial_skill='setup' + DEFAULT_TRANSITIONS/ENTER_CHECK)
    を通す regression。setup の遷移先 move_to_table が DEFAULT_ENTER_CHECK に無いと、
    warmup 明けの tick で enter_check[cand] 直接 index が KeyError 即死する。"""

    def test_setup_start_survives_warmup_tick_without_keyerror(self):
        """initial_skill='setup' + DEFAULT_* で warmup 明けの tick が例外死しない。"""
        perception = _StubPerception({0: [], 1: [], 2: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_production_registry())
        orch = Orchestrator(
            perception, cleaner, dispatcher, initial_skill="setup"
        )

        orch.tick(_empty_frame(t=0))  # warmup (cleaned=None)
        # 修正前: ここで KeyError 'move_to_table' (setup の遷移候補を index)
        result = orch.tick(_empty_frame(t=1))
        assert result is not None
        assert result.current_skill == "setup"
        # YOLO fire 無し・dwell 前なので setup に留まる (enter_check では進まない)
        assert result.fire_transition_to is None

    def test_default_graph_is_constructible(self):
        """production 配線 (DEFAULT_TRANSITIONS + DEFAULT_ENTER_CHECK + 全 skill 登録)
        で Orchestrator が例外なく構築できる (production wiring smoke)。"""
        perception = _StubPerception({})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_production_registry())
        # 例外を投げずに構築できれば OK
        Orchestrator(perception, cleaner, dispatcher, initial_skill="setup")


class TestBuildStageTransitions:
    """stage の skill 列と遷移 (Issue #128 Phase 3 → #141 D7-1 で頭の手順を追加)。"""

    def test_stage_0_walks_then_runs_the_head_procedure_for_pick(self):
        from inference.desktop.orchestrator import (
            build_stage_skill_sequence,
            build_stage_transitions,
        )

        # 先頭で、ロボットが指令に従い始めるのを待つ (Issue #159 T5)。腕を下ろす・
        # 歩くのは go-live の後 (go-live 前の adapter は WBC に何も送らない、2026-09-24)
        for include_hand in (True, False):
            sequence = build_stage_skill_sequence(0, include_hand=include_hand)
            assert sequence[:4] == [
                "wait_for_go_live_pick_table_leg",
                "setup",
                "move_to_table",
                "post_walk_settle",
            ], include_hand
        trans = build_stage_transitions(0)
        assert trans["wait_for_go_live_pick_table_leg"] == ["setup"]
        assert trans["setup"] == ["move_to_table"]
        assert trans["move_to_table"] == ["post_walk_settle"]
        # 歩き終わってから 1 本目の pick の開始姿勢へ (Issue #141 D2 #6)
        assert trans["post_walk_settle"] == ["hand_open_pick_table_leg"]
        assert trans["hand_open_pick_table_leg"] == ["arm_pre_motion_for_pick_table_leg"]
        assert trans["arm_pre_motion_for_pick_table_leg"] == ["hand_pose_pick_table_leg"]
        assert trans["hand_pose_pick_table_leg"] == ["hold_pose_for_pick_table_leg"]
        assert trans["hold_pose_for_pick_table_leg"] == []

    def test_stage_1_starts_at_pick_when_continued_from_stage_0(self):
        """1 本目に rotate は無い (1-6)。頭の手順は stage 0 で済んでいる。"""
        from inference.desktop.orchestrator import build_stage_transitions

        trans = build_stage_transitions(1, is_start_stage=False)
        assert trans["pick_table_leg"] == [
            "arm_transition_pick_table_leg_to_insert_table_leg"
        ]
        assert trans["hold_transition_pick_table_leg_to_insert_table_leg"] == [
            "insert_table_leg"
        ]
        assert trans["insert_table_leg"] == [
            "hand_release_insert_table_leg_to_rotate_leg_to_tighten"
        ]
        assert trans[
            "hold_transition_insert_table_leg_to_rotate_leg_to_tighten"
        ] == ["rotate_leg_to_tighten"]
        assert trans["rotate_leg_to_tighten"] == []
        assert "arm_pre_motion_for_pick_table_leg" not in trans

    def test_stage_1_gets_the_head_procedure_when_started_from_there(self):
        from inference.desktop.orchestrator import build_stage_transitions

        trans = build_stage_transitions(1, is_start_stage=True)
        assert trans["hold_pose_for_pick_table_leg"] == ["pick_table_leg"]

    def test_stages_2_3_4_start_with_the_rotate_head_procedure(self):
        from inference.desktop.orchestrator import build_stage_transitions

        baseline = build_stage_transitions(2)
        assert baseline["hold_pose_for_rotate_table_base"] == ["rotate_table_base"]
        assert baseline["rotate_table_base"] == [
            "hand_release_rotate_table_base_to_pick_table_leg"
        ]
        assert baseline[
            "hold_transition_rotate_table_base_to_pick_table_leg"
        ] == ["pick_table_leg"]
        assert baseline["rotate_leg_to_tighten"] == []
        for stage in (3, 4):
            assert build_stage_transitions(stage) == baseline

    def test_only_pick_to_insert_keeps_the_grip(self):
        """境界は腕の前に手を離すのが既定。例外は脚を運ぶ pick -> insert だけ (Issue #159 T3)。"""
        from inference.desktop.orchestrator import (
            KEEP_GRIP_MODEL_TRANSITIONS,
            model_transition_skill_names,
        )

        assert KEEP_GRIP_MODEL_TRANSITIONS == {("pick_table_leg", "insert_table_leg")}
        assert model_transition_skill_names("pick_table_leg", "insert_table_leg") == [
            "arm_transition_pick_table_leg_to_insert_table_leg",
            "hand_transition_pick_table_leg_to_insert_table_leg",
            "hold_transition_pick_table_leg_to_insert_table_leg",
        ]
        assert model_transition_skill_names("rotate_leg_to_tighten", "flip_table")[0] == (
            "hand_release_rotate_leg_to_tighten_to_flip_table"
        )
        # 手を動かさない構成では離す手順も入らない
        assert model_transition_skill_names(
            "insert_table_leg", "rotate_leg_to_tighten", include_hand=False
        ) == [
            "arm_transition_insert_table_leg_to_rotate_leg_to_tighten",
            "hold_transition_insert_table_leg_to_rotate_leg_to_tighten",
        ]

    def test_continuous_next_stage_uses_model_boundary_not_startup_hand_open(self):
        from inference.desktop.orchestrator import build_stage_skill_sequence

        sequence = build_stage_skill_sequence(2, is_start_stage=False)
        assert sequence[:4] == [
            "hand_release_rotate_leg_to_tighten_to_rotate_table_base",
            "arm_transition_rotate_leg_to_tighten_to_rotate_table_base",
            "hand_transition_rotate_leg_to_tighten_to_rotate_table_base",
            "hold_transition_rotate_leg_to_tighten_to_rotate_table_base",
        ]
        assert "hand_open_rotate_table_base" not in sequence

    def test_hybrid_internal_pick_to_insert_transition_is_not_duplicated(self):
        from inference.desktop.orchestrator import build_stage_skill_sequence

        sequence = build_stage_skill_sequence(
            2,
            skip_model_transition_pairs=frozenset(
                {("pick_table_leg", "insert_table_leg")}
            ),
        )
        pick_index = sequence.index("pick_table_leg")
        # 腕・手の移動は hybrid が済ませるので省くが、model 載せ替えの HOLD は残す
        # (Codex #5: HOLD が無いと insert を pick と同時に GPU へ載せていた)。
        assert sequence[pick_index + 1 : pick_index + 3] == [
            "hold_transition_pick_table_leg_to_insert_table_leg",
            "insert_table_leg",
        ]
        assert [
            name for name in sequence if "pick_table_leg_to_insert_table_leg" in name
        ] == ["hold_transition_pick_table_leg_to_insert_table_leg"]

    def test_self_bridged_pair_without_its_hold_is_rejected(self):
        from inference.desktop.orchestrator import (
            build_stage_skill_sequence,
            validate_stage_model_transition_chain,
        )

        hybrid_pair = frozenset({("pick_table_leg", "insert_table_leg")})
        sequence = build_stage_skill_sequence(
            2, is_start_stage=True, skip_model_transition_pairs=hybrid_pair
        )
        broken = [
            name
            for name in sequence
            if name != "hold_transition_pick_table_leg_to_insert_table_leg"
        ]
        with pytest.raises(ValueError, match="model switch HOLD"):
            validate_stage_model_transition_chain(
                2, broken, is_start_stage=True, self_bridged_pairs=hybrid_pair
            )

    def test_stage_5_is_the_head_procedure_then_flip(self):
        from inference.desktop.orchestrator import build_stage_transitions

        trans = build_stage_transitions(5)
        assert trans["hold_pose_for_flip_table"] == ["flip_table"]
        assert trans["flip_table"] == []

    def test_without_a_real_hand_the_hand_skills_are_dropped(self):
        from inference.desktop.orchestrator import build_stage_transitions

        trans = build_stage_transitions(5, include_hand=False)
        assert trans["arm_pre_motion_for_flip_table"] == ["hold_pose_for_flip_table"]
        assert not any(name.startswith("hand_") for name in trans)

    def test_invalid_stage_raises(self):
        from inference.desktop.orchestrator import build_stage_transitions
        with pytest.raises(KeyError, match="stage 6 not registered"):
            build_stage_transitions(6)

    def test_all_stages_have_exactly_one_terminal_skill(self):
        from inference.desktop.orchestrator import (
            STAGE_SKILL_SEQUENCES,
            build_stage_skill_sequence,
            build_stage_transitions,
        )
        for stage in STAGE_SKILL_SEQUENCES:
            trans = build_stage_transitions(stage)
            sequence = build_stage_skill_sequence(stage)
            terminals = [name for name, nxt in trans.items() if not nxt]
            assert terminals == [sequence[-1]]

    @pytest.mark.parametrize("stage", range(6))
    def test_every_stage_model_boundary_has_a_verified_pose_scaffold(self, stage):
        from inference.desktop.orchestrator import (
            build_stage_skill_sequence,
            validate_stage_model_transition_chain,
        )

        sequence = build_stage_skill_sequence(stage, is_start_stage=True)
        validate_stage_model_transition_chain(
            stage, sequence, is_start_stage=True
        )

    @pytest.mark.parametrize("stage", (2, 3, 4, 5))
    def test_every_continuous_stage_boundary_has_a_verified_pose_scaffold(
        self, stage
    ):
        from inference.desktop.orchestrator import (
            build_stage_skill_sequence,
            validate_stage_model_transition_chain,
        )

        sequence = build_stage_skill_sequence(stage, is_start_stage=False)
        validate_stage_model_transition_chain(
            stage, sequence, is_start_stage=False
        )

    def test_direct_policy_to_policy_transition_is_rejected(self):
        from inference.desktop.orchestrator import (
            build_stage_skill_sequence,
            validate_stage_model_transition_chain,
        )

        sequence = build_stage_skill_sequence(2, is_start_stage=True)
        start = sequence.index("rotate_table_base") + 1
        end = sequence.index("pick_table_leg")
        broken = sequence[:start] + sequence[end:]
        with pytest.raises(ValueError, match="missing its verified initial-pose"):
            validate_stage_model_transition_chain(
                2, broken, is_start_stage=True
            )

    def test_only_declared_hybrid_pair_may_self_bridge(self):
        from inference.desktop.orchestrator import (
            build_stage_skill_sequence,
            validate_stage_model_transition_chain,
        )

        hybrid_pair = frozenset({("pick_table_leg", "insert_table_leg")})
        sequence = build_stage_skill_sequence(
            2,
            is_start_stage=True,
            skip_model_transition_pairs=hybrid_pair,
        )
        validate_stage_model_transition_chain(
            2,
            sequence,
            is_start_stage=True,
            self_bridged_pairs=hybrid_pair,
        )
        with pytest.raises(ValueError, match="missing its verified initial-pose"):
            validate_stage_model_transition_chain(
                2, sequence, is_start_stage=True
            )


# =========================================================================
# Issue #141 D3 / D4 / 束 1-6: 検出の 2 系統・カメラの停止・記録の hook・天板の基準
# =========================================================================
def _rotated_table_top(angle_deg: float, half_w: float = 0.2, half_h: float = 0.1):
    """原点まわりに angle_deg 回した table_top。"""
    corners = np.array(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]],
        dtype=np.float64,
    )
    rad = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    verts = (corners @ rot.T) + np.array([0.5, 0.5])
    return OBBDetection(
        class_id=0, class_name="table_top", confidence=0.9,
        verts=verts.astype(np.float32),
    )


class _StubPolicyFilter:
    """policy 用 filter の stub (push した生の検出をそのまま覚えて返す)。"""

    def __init__(self) -> None:
        self.pushes: list[list[OBBDetection]] = []

    def push(self, raw: list[OBBDetection]) -> list[OBBDetection]:
        self.pushes.append(list(raw))
        return list(raw)


class TestDetectionSplit:
    """YOLO は 1 tick 1 回、出力を Planner と policy に分ける (D3)。"""

    def _orch(self, policy_filter=None, dets_by_t=None):
        perception = _StubPerception(dets_by_t or {0: [], 1: [], 2: []})
        return Orchestrator(
            perception,
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
            policy_filter=policy_filter,
        )

    def test_policy_gets_the_filter_output_not_the_cleaner_output(self):
        table = _det("table_top", 0.1, 0.1, 0.5, 0.3)
        policy_filter = _StubPolicyFilter()
        orch = self._orch(policy_filter, {0: [table], 1: [table]})
        first = orch.tick(_empty_frame(t=0))
        # cleaner はまだ溜まっていないが、policy 用は 1 frame 目から出る
        assert first.cleaned is None
        assert len(first.policy_cleaned) == 1
        assert len(policy_filter.pushes) == 1

    def test_detector_runs_once_per_new_frame(self):
        policy_filter = _StubPolicyFilter()
        orch = self._orch(policy_filter)
        orch.tick(_empty_frame(t=0))
        orch.tick(_empty_frame(t=0))     # 同じ frame
        orch.tick(_empty_frame(t=1))
        assert len(policy_filter.pushes) == 2

    def test_without_a_filter_the_cleaner_output_is_used_for_both(self):
        orch = self._orch(None)
        orch.tick(_empty_frame(t=0))
        result = orch.tick(_empty_frame(t=1))
        assert result.policy_cleaned is result.cleaned


class TestCameraFreshness:
    """4 台のカメラの受信時刻で止まりを見る (D3)。"""

    class _StaleFrame:
        def __init__(self, t: int) -> None:
            self.rgb = np.zeros((2, 4, 3), dtype=np.uint8)
            self.t = t
            self.received_monotonic_ns = time.monotonic_ns() - 3_000_000_000

    def test_stale_head_stops_the_run(self):
        orch = Orchestrator(
            _StubPerception({0: []}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
        )
        with pytest.raises(LiveSourceSafetyError, match="camera frames are stale"):
            orch.tick(self._StaleFrame(t=0))

    def test_stale_frame_is_rejected_before_the_first_skill_starts(self):
        """Codex #6: 古い画像のまま skill を start しない (歩行の vx を出さない)。"""
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({0: []}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
        )
        with pytest.raises(LiveSourceSafetyError, match="camera frames are stale"):
            orch.tick(self._StaleFrame(t=0))
        assert dispatcher.active_skill_name is None

    def test_stale_joint_state_is_rejected_before_the_first_skill_starts(self):
        now = time.monotonic_ns()
        stale = JointStateData(
            name=("j0",),
            position=np.zeros(1),
            velocity=np.zeros(1),
            effort=np.zeros(1),
            t=now,
            received_monotonic_ns=now - 2_000_000_000,
        )
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({0: []}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            joint_state_source=_StubJointStateSource(snapshot=stale),
        )
        with pytest.raises(LiveSourceSafetyError, match="joint state is stale"):
            orch.tick(_empty_frame(t=0))
        assert dispatcher.active_skill_name is None

    def test_stale_wrist_stops_the_run(self):
        orch = Orchestrator(
            _StubPerception({0: []}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
            wrist_left_source=_StubFrameSource(snapshot=self._StaleFrame(t=0)),
            wrist_right_source=_StubFrameSource(snapshot=_empty_frame(t=0)),
        )
        with pytest.raises(LiveSourceSafetyError, match="left_wrist"):
            orch.tick(_empty_frame(t=0))

    def test_fresh_cameras_pass(self):
        orch = Orchestrator(
            _StubPerception({0: []}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(_make_registry()),
            wrist_left_source=_StubFrameSource(snapshot=_empty_frame(t=0)),
            wrist_right_source=_StubFrameSource(snapshot=_empty_frame(t=0)),
        )
        assert orch.tick(_empty_frame(t=0)) is not None


class TestOnTickHook:
    """記録の hook は毎 tick 呼ばれる (D3 / D1: 本番でも states/actions を残す)。"""

    def test_hook_receives_result_obs_and_skill(self):
        calls: list[tuple] = []
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({0: [], 1: []}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            on_tick=lambda result, obs, skill: calls.append((result, obs, skill)),
        )
        orch.tick(_empty_frame(t=0))
        orch.tick(_empty_frame(t=1))
        assert len(calls) == 2
        result, obs, skill = calls[-1]
        assert result.t == 1
        assert "head_rgb" in obs and "t" in obs
        assert skill is dispatcher.active_skill


class TestRotateTransitionContext:
    """rotate に入るときも天板の基準を記録する (束 1-6 / INF-7)。"""

    def _orch(self, dets_by_t, *, after_rotate: list[str]):
        from inference.desktop.skill_planner.enter_conditions import enter_pick_table_leg

        registry = _make_registry()
        registry["rotate_table_base"] = MockSkill("rotate_table_base")
        return Orchestrator(
            _StubPerception(dets_by_t),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(registry),
            initial_skill="move_to_table",
            transitions={
                "move_to_table": ["rotate_table_base"],
                "rotate_table_base": after_rotate,
                "pick_table_leg": [],
            },
            enter_check={
                "rotate_table_base": lambda dets, state: True,
                "pick_table_leg": enter_pick_table_leg,
            },
        )

    def test_reference_is_recorded_when_entering_rotate(self):
        table = _rotated_table_top(0.0)
        orch = self._orch({t: [table] for t in range(4)}, after_rotate=[])
        assert orch.state.base_rotation_start_table_top_verts is None
        orch.tick(_empty_frame(t=0))     # cleaner が溜まるまでは遷移しない
        orch.tick(_empty_frame(t=1))     # ここで move_to_table -> rotate_table_base
        assert orch.state.current_skill == "rotate_table_base"
        assert orch.state.base_rotation_start_table_top_verts is not None

    def test_rotate_to_pick_fires_once_the_table_has_turned(self):
        """stage の列 (rotate -> pick) で、天板が回れば pick が発火する。"""
        straight = _rotated_table_top(0.0)
        turned = _rotated_table_top(30.0)     # 閾値 18 度を超える
        dets = {t: [straight] for t in range(4)}
        dets.update({t: [turned] for t in range(4, 10)})
        orch = self._orch(dets, after_rotate=["pick_table_leg"])
        fired = None
        for t in range(10):
            result = orch.tick(_empty_frame(t=t))
            if result.fire_transition_to == "pick_table_leg":
                fired = t
                break
        assert fired is not None, "rotate -> pick が発火しなかった"
        assert orch.state.current_skill == "pick_table_leg"


class TestOperatorBoundaryDoesNotStepTheNextSkill:
    """区切りで戻るとき、次の skill を 1 tick も動かさない (Issue #141 D6-1)。

    評価は「頭の手順 → [Enter] → policy」の順で、Enter の前に policy が 1 tick でも
    動くと、操作者の確認前に指令が出てしまう。
    """

    class _CompleteImmediately(Skill):
        name = "head_step"

        def __init__(self) -> None:
            super().__init__()
            self.steps = 0

        def _on_start(self, params: dict) -> None:
            self.steps = 0

        def _on_stop(self) -> None:
            pass

        def step(self, obs: dict) -> np.ndarray:
            self.steps += 1
            return np.zeros(14)

        @property
        def is_complete(self) -> bool:
            return self.steps >= 1

    class _CountingSkill(Skill):
        name = "evaluated_skill"

        def __init__(self) -> None:
            super().__init__()
            self.steps = 0

        def _on_start(self, params: dict) -> None:
            pass

        def _on_stop(self) -> None:
            pass

        def step(self, obs: dict) -> np.ndarray:
            self.steps += 1
            return np.zeros(14)

    def test_next_skill_is_started_but_never_stepped(self, monkeypatch):
        _patch_clock(monkeypatch)
        head = self._CompleteImmediately()
        evaluated = self._CountingSkill()
        dispatcher = SkillDispatchLowerPolicy(
            {head.name: head, evaluated.name: evaluated}
        )
        orch = Orchestrator(
            _StubPerception({t: [] for t in range(10)}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill=head.name,
            transitions={head.name: [evaluated.name], evaluated.name: []},
            enter_check={head.name: enter_never, evaluated.name: enter_never},
        )
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(10)])

        orch.run_live(
            source, hz=10.0, startup_timeout=1.0, frame_timeout=0.5,
            stop_on_skills=frozenset({evaluated.name}),
        )

        assert dispatcher.active_skill_name == evaluated.name
        assert evaluated.steps == 0
        assert head.steps >= 1

    def test_the_run_continues_from_the_boundary(self, monkeypatch):
        _patch_clock(monkeypatch)
        head = self._CompleteImmediately()
        evaluated = self._CountingSkill()
        dispatcher = SkillDispatchLowerPolicy(
            {head.name: head, evaluated.name: evaluated}
        )
        orch = Orchestrator(
            _StubPerception({t: [] for t in range(20)}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill=head.name,
            transitions={head.name: [evaluated.name], evaluated.name: []},
            enter_check={head.name: enter_never, evaluated.name: enter_never},
        )
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(20)])
        orch.run_live(
            source, hz=10.0, startup_timeout=1.0, frame_timeout=0.5,
            stop_on_skills=frozenset({evaluated.name}),
        )
        # 区切りの後、同じ orchestrator で続きを回すと policy が動く
        with pytest.raises(KeyboardInterrupt):
            orch.run_live(source, hz=10.0, startup_timeout=1.0, frame_timeout=0.5)
        assert evaluated.steps > 0


# =========================================================================
# 初期 skill が rotate_table_base のときの基準 seed
# =========================================================================
def _rotated_table_top(deg: float, cx: float = 320.0, cy: float = 240.0) -> OBBDetection:
    """中心 (cx, cy) の矩形を deg だけ回した table_top 検出。"""
    half_w, half_h = 140.0, 90.0
    corners = np.array(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]],
        dtype=np.float64,
    )
    rad = np.deg2rad(deg)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    verts = (corners @ rot.T + np.array([cx, cy])).astype(np.float32)
    return OBBDetection(
        class_id=0, class_name="table_top", confidence=0.95, verts=verts
    )


class TestInitialSkillBaseRotationReference:
    """`enter_pick_table_leg` の基準は遷移経由でしか入らなかった。

    `rotate_table_base` が initial_skill だと tick() 冒頭の auto start が
    `state.transition()` を通らないので基準が永久に None になり、
    `enter_pick_table_leg` は必ず False を返して
    **orchestrator が rotate_table_base から出られない**。

    大会経路そのもの (boundary の OrchestratorDriver は
    initial_skill="rotate_table_base"、自前経路の --stage 1..4 も同じ)。
    実 image を載せた pod で、実フレームを 57.5° 回しても遷移 0 回だった
    (threshold は 18°、2026-09-21)。
    """

    def _make_orch(self, dets_by_t: dict[int, list[OBBDetection]]) -> Orchestrator:
        registry = _make_registry()
        registry["rotate_table_base"] = MockSkill("rotate_table_base")
        return Orchestrator(
            _StubPerception(dets_by_t),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(registry),
            initial_skill="rotate_table_base",
            transitions={"rotate_table_base": ["pick_table_leg"], "pick_table_leg": []},
            enter_check={"pick_table_leg": DEFAULT_ENTER_CHECK["pick_table_leg"]},
        )

    def test_reference_is_seeded_without_a_transition(self):
        """遷移していなくても ring が満ちた時点で基準が入る。"""
        orch = self._make_orch({t: [_rotated_table_top(0.0)] for t in range(6)})

        assert orch.state.base_rotation_start_table_top_verts is None
        for t in range(6):
            orch.tick(_empty_frame(t))

        assert orch.state.base_rotation_start_table_top_verts is not None

    def test_pick_fires_once_the_table_top_has_turned(self):
        """基準が入るので 18° を超えたところで pick へ遷移する。"""
        dets = {t: [_rotated_table_top(0.0)] for t in range(4)}
        for i, t in enumerate(range(4, 12)):
            dets[t] = [_rotated_table_top(5.0 * (i + 1))]  # 5°,10°,...,40°
        orch = self._make_orch(dets)

        fired: list[str] = []
        for t in range(12):
            result = orch.tick(_empty_frame(t))
            if result.fire_transition_to is not None:
                fired.append(result.fire_transition_to)

        assert fired == ["pick_table_leg"], f"遷移しなかった: {fired}"
        assert orch.state.current_skill == "pick_table_leg"

    def test_reference_is_not_reseeded_after_it_exists(self):
        """一度入った基準を後の tick が上書きしないこと (上書きすると常に 0° 扱い)。"""
        orch = self._make_orch({t: [_rotated_table_top(0.0)] for t in range(4)})
        for t in range(4):
            orch.tick(_empty_frame(t))
        first = orch.state.base_rotation_start_table_top_verts.copy()

        for t in range(4, 10):
            orch.tick(_empty_frame(t))

        np.testing.assert_allclose(
            orch.state.base_rotation_start_table_top_verts, first
        )

    def test_no_seed_for_an_unrelated_initial_skill(self):
        """天板を回さない skill から始まったときは seed しない。"""
        registry = _make_registry()
        orch = Orchestrator(
            _StubPerception({t: [_rotated_table_top(0.0)] for t in range(6)}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(registry),
            initial_skill="move_to_table",
            transitions={"move_to_table": []},
            enter_check={},
        )
        for t in range(6):
            orch.tick(_empty_frame(t))

        assert orch.state.base_rotation_start_table_top_verts is None


# =========================================================================
# advance_finished_skill: tick() の受け皿 (is_complete / timeout / dwell)
# =========================================================================
class _CompletableSkill(MockSkill):
    """is_complete を外から切り替えられる Mock。"""

    def __init__(self, skill_name: str, max_dwell_sec: float | None = None) -> None:
        super().__init__(skill_name, max_dwell_sec=max_dwell_sec)
        self.complete = False

    @property
    def is_complete(self) -> bool:
        return self.complete


class TestAdvanceFinishedSkill:
    """`tick()` は YOLO の enter_check しか見ない。その受け皿。

    元は `run_live()` の局所変数に閉じており、`tick()` を直接回す boundary の
    `OrchestratorDriver` には受け皿が無かった。YOLO が落とすと
    `rotate_table_base` から永久に出られない (2026-09-21 に pod で実測)。
    両方の経路がこの method を通ることで、片方だけが取り残されないようにする。
    """

    def _make_orch(self, skill: MockSkill, **kwargs) -> Orchestrator:
        registry = _make_registry()
        registry[skill.name] = skill
        registry.setdefault("pick_table_leg", MockSkill("pick_table_leg"))
        return Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(registry),
            initial_skill=skill.name,
            transitions={skill.name: ["pick_table_leg"], "pick_table_leg": []},
            enter_check={"pick_table_leg": enter_never},
            **kwargs,
        )

    def test_nothing_fires_while_the_skill_is_still_running(self):
        skill = _CompletableSkill("rotate_table_base")
        orch = self._make_orch(skill, hard_timeout_by_skill={"rotate_table_base": 30.0})
        orch.tick(_empty_frame(0))

        assert orch.advance_finished_skill(now=100.0).fired_to is None
        assert orch.advance_finished_skill(now=110.0).fired_to is None

    def test_is_complete_advances(self):
        skill = _CompletableSkill("rotate_table_base")
        orch = self._make_orch(skill)
        orch.tick(_empty_frame(0))
        orch.advance_finished_skill(now=100.0)

        skill.complete = True
        advance = orch.advance_finished_skill(now=101.0)

        assert advance.fired_to == "pick_table_leg"
        assert advance.reason == "complete"
        assert orch.state.current_skill == "pick_table_leg"

    def test_hard_timeout_advances(self):
        """YOLO が落としても max_seconds_hard で次へ進む (これが受け皿の本体)。"""
        skill = _CompletableSkill("rotate_table_base")
        orch = self._make_orch(skill, hard_timeout_by_skill={"rotate_table_base": 30.0})
        orch.tick(_empty_frame(0))
        orch.advance_finished_skill(now=100.0)

        assert orch.advance_finished_skill(now=129.9).fired_to is None
        advance = orch.advance_finished_skill(now=130.1)

        assert advance.fired_to == "pick_table_leg"
        assert advance.reason == "timeout"

    def test_on_timeout_stop_raises_instead_of_advancing(self):
        """`on_timeout: stop` は進めずに止める (握って進むより安全な場合)。"""
        skill = _CompletableSkill("rotate_table_base")
        orch = self._make_orch(
            skill,
            hard_timeout_by_skill={"rotate_table_base": 30.0},
            timeout_action_by_skill={"rotate_table_base": "stop"},
        )
        orch.tick(_empty_frame(0))
        orch.advance_finished_skill(now=100.0)

        with pytest.raises(LiveSourceSafetyError, match="max_seconds_hard"):
            orch.advance_finished_skill(now=131.0)

    def test_max_dwell_advances(self):
        skill = _CompletableSkill("rotate_table_base", max_dwell_sec=5.0)
        orch = self._make_orch(skill)
        orch.tick(_empty_frame(0))
        orch.advance_finished_skill(now=100.0)

        advance = orch.advance_finished_skill(now=106.0)

        assert advance.fired_to == "pick_table_leg"
        assert advance.reason == "dwell"

    def test_timer_restarts_when_the_skill_changes(self):
        """遷移直後に前の skill の経過時間で二重発火しないこと。"""
        skill = _CompletableSkill("rotate_table_base")
        orch = self._make_orch(
            skill,
            hard_timeout_by_skill={"rotate_table_base": 30.0, "pick_table_leg": 30.0},
        )
        orch.tick(_empty_frame(0))
        orch.advance_finished_skill(now=100.0)
        assert orch.advance_finished_skill(now=131.0).fired_to == "pick_table_leg"

        # 直後は pick の timer が 0 から始まるので進まない
        assert orch.advance_finished_skill(now=132.0).fired_to is None
        # pick 自身が 30 秒経って初めて終端扱い
        assert orch.advance_finished_skill(now=162.0).stage_finished is True

    def test_terminal_skill_reports_stage_finished(self):
        skill = _CompletableSkill("rotate_table_base")
        registry = _make_registry()
        registry[skill.name] = skill
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(registry),
            initial_skill=skill.name,
            transitions={skill.name: []},
            enter_check={},
            hard_timeout_by_skill={"rotate_table_base": 10.0},
        )
        orch.tick(_empty_frame(0))
        orch.advance_finished_skill(now=100.0)

        advance = orch.advance_finished_skill(now=111.0)

        assert advance.fired_to is None
        assert advance.stage_finished is True


class TestResetEpisode:
    """boundary の server は 1 process が動き続け、運営が episode 間に reset を呼ぶ。

    戻さないと 2 本目が skill を一切進めないまま終わる:
      - `n_legs_completed` が残ると `enter_pick_table_leg` が常に False
      - `base_rotation_start_table_top_verts` が残ると 1 本目の天板が基準になる
      - `dispatcher` が前の skill を握ったままだと initial_skill に戻らない
    """

    def _make_orch(self) -> Orchestrator:
        registry = _make_registry()
        registry["rotate_table_base"] = MockSkill("rotate_table_base")
        return Orchestrator(
            _StubPerception({t: [_rotated_table_top(0.0)] for t in range(8)}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(registry),
            initial_skill="rotate_table_base",
            transitions={"rotate_table_base": ["pick_table_leg"], "pick_table_leg": []},
            enter_check={"pick_table_leg": enter_never},
        )

    def test_reset_returns_to_the_initial_skill(self):
        orch = self._make_orch()
        for t in range(4):
            orch.tick(_empty_frame(t))
        orch.state.transition("pick_table_leg")
        assert orch.state.current_skill == "pick_table_leg"

        orch.reset_episode()

        assert orch.state.current_skill == "rotate_table_base"
        assert orch.dispatcher.active_skill_name is None

    def test_reset_clears_the_leg_counter_and_the_rotation_reference(self):
        orch = self._make_orch()
        for t in range(4):
            orch.tick(_empty_frame(t))
        orch.state.n_legs_completed = 3
        assert orch.state.base_rotation_start_table_top_verts is not None

        orch.reset_episode()

        assert orch.state.n_legs_completed == 0
        assert orch.state.base_rotation_start_table_top_verts is None

    def test_reset_clears_the_pick_contact_flags(self):
        orch = self._make_orch()
        orch.state.left_hand_touched_leg_since_pick = True
        orch.state.leg_tip_touched_hole_since_pick = True
        orch.state.left_right_hand_overlapped_since_pick = True

        orch.reset_episode()

        assert orch.state.left_hand_touched_leg_since_pick is False
        assert orch.state.leg_tip_touched_hole_since_pick is False
        assert orch.state.left_right_hand_overlapped_since_pick is False

    def test_reset_clears_the_skill_timer(self):
        """前の episode の経過時間で即 timeout しないこと。"""
        orch = self._make_orch()
        orch.tick(_empty_frame(0))
        orch.advance_finished_skill(now=100.0)
        assert orch._active_skill_started_at is not None

        orch.reset_episode()

        assert orch._active_skill_name is None
        assert orch._active_skill_started_at is None
        assert orch._active_skill_dwell_fired is False

    def test_the_next_tick_starts_the_initial_skill_again(self):
        """reset の後、tick が initial_skill を start し直すこと。"""
        orch = self._make_orch()
        for t in range(4):
            orch.tick(_empty_frame(t))
        orch.reset_episode()

        orch.tick(_empty_frame(10))

        assert orch.dispatcher.active_skill_name == "rotate_table_base"
        assert orch.state.current_skill == "rotate_table_base"

    def test_reset_stops_the_active_skill(self):
        """`_on_stop()` を通すこと (skill の buffer を畳む)。

        副作用として active だった skill の model は解放される。docstring が
        そう書いていることと実装が一致しているかを固定する。
        """
        registry = _make_registry()
        stopped: list[str] = []

        class _Recording(MockSkill):
            def _on_stop(self) -> None:
                stopped.append(self.name)

        registry["rotate_table_base"] = _Recording("rotate_table_base")
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(registry),
            initial_skill="rotate_table_base",
            transitions={"rotate_table_base": []},
            enter_check={},
        )
        orch.tick(_empty_frame(0))
        assert orch.dispatcher.active_skill_name == "rotate_table_base"

        orch.reset_episode()

        assert stopped == ["rotate_table_base"]


class _ObsRecordingSkill(MockSkill):
    """step に届いた ``obs["cleaned"]`` (overlay の入力) を記録する。"""

    def __init__(self, skill_name: str) -> None:
        super().__init__(skill_name)
        self.cleaned_seen: list = []

    def step(self, obs: dict):
        self.cleaned_seen.append(obs.get("cleaned"))
        return None


class TestStageRunDoesNotAdvanceOnPerception:
    """stage の run は YOLO で次へ進まない。検出は overlay 用に skill へ届く (2026-09-24)。"""

    def test_every_way_of_looking_up_a_skill_never_fires(self):
        """`[]`・`.get()`・`in` のどれで引いても「進まない」(dict の既定は `[]` だけ)。"""
        enter_check = stage_enter_check()
        for lookup in (
            enter_check["pick_table_leg"],
            enter_check.get("pick_table_leg"),
            enter_check.get("a_skill_added_later", None),
        ):
            assert lookup is enter_never
        assert "a_skill_added_later" in enter_check

    _BOUNDARY = "hand_release_rotate_table_base_to_pick_table_leg"
    _PICK_SKIPS = frozenset({("pick_table_leg", "insert_table_leg")})

    def test_every_candidate_of_every_stage_never_fires(self):
        enter_check = stage_enter_check()
        for stage in range(6):
            for is_start_stage in (True, False):
                for include_hand in (True, False):
                    for skips in (frozenset(), self._PICK_SKIPS):
                        transitions = build_stage_transitions(
                            stage,
                            is_start_stage=is_start_stage,
                            include_hand=include_hand,
                            skip_model_transition_pairs=skips,
                        )
                        for candidates in transitions.values():
                            for cand in candidates:
                                assert enter_check[cand] is enter_never, (stage, cand)

    def _run(self, enter_check) -> tuple[Orchestrator, _ObsRecordingSkill, Optional[int]]:
        """stage 2 の境界 (台を回す -> pick) を、天板が回った検出で 16 tick 回す。"""
        from inference.desktop.perception.policy_filter import (
            PolicyDetectionFilter,
            load_policy_filter_config,
        )

        dets = {t: [_rotated_table_top(0.0)] for t in range(6)}
        dets.update({t: [_rotated_table_top(30.0)] for t in range(6, 16)})  # 閾値 18 度超
        rotate = _ObsRecordingSkill("rotate_table_base")
        orch = Orchestrator(
            _StubPerception(dets),
            DetectionStream(_base_cleanup_config()),
            SkillDispatchLowerPolicy(
                {"rotate_table_base": rotate, self._BOUNDARY: MockSkill(self._BOUNDARY)}
            ),
            initial_skill="rotate_table_base",
            transitions={"rotate_table_base": [self._BOUNDARY], self._BOUNDARY: []},
            enter_check=enter_check,
            hard_timeout_by_skill={"rotate_table_base": 10.0},
            policy_filter=PolicyDetectionFilter(load_policy_filter_config()),
        )
        fired = None
        for t in range(16):
            if orch.tick(_empty_frame(t=t)).fire_transition_to is not None:
                fired = t
                break
        return orch, rotate, fired

    def test_the_old_wiring_would_have_advanced_on_these_detections(self):
        """対照: 境界の先頭に YOLO の enter_pick を置くと、この検出で次へ進む。"""
        from inference.desktop.skill_planner.enter_conditions import enter_pick_table_leg

        _orch, _rotate, fired = self._run({self._BOUNDARY: enter_pick_table_leg})
        assert fired is not None

    def test_yolo_does_not_advance_but_still_reaches_the_overlay(self):
        orch, rotate, fired = self._run(stage_enter_check())
        assert fired is None
        assert orch.state.current_skill == "rotate_table_base"
        # overlay の入力 (obs["cleaned"]) には、天板が回った後の検出がそのまま届く
        assert len(rotate.cleaned_seen) == 16
        last = rotate.cleaned_seen[-1]
        assert [d.class_name for d in last] == ["table_top"]
        np.testing.assert_allclose(last[0].verts, _rotated_table_top(30.0).verts)

    def test_the_timeout_still_advances(self):
        orch, _rotate, _fired = self._run(stage_enter_check())
        assert orch.advance_finished_skill(now=100.0).fired_to is None
        advance = orch.advance_finished_skill(now=110.0)
        assert advance.fired_to == self._BOUNDARY
        assert advance.reason == "timeout"
