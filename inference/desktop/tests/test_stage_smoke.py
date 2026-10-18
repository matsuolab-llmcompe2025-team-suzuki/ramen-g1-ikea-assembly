"""stage を端から端まで通す smoke (Issue #141 D7-3)。

実機に持って行く前に、頭の手順 → 学習 skill → 次の skill という流れが実際につながるかを
確かめられるのはここだけ (unit test は部品ごとの確認しかしない)。

合成の frame + 指令をそのまま返す偽のロボット + stub の policy で、stage 2 の列
(手を開く → 腕の pre-motion → 手を開始の開度へ → N 秒保持 → rotate → pick → insert →
rotate_leg) を回し、次を確かめる:

    - 列の全部が順番に active になり、最後まで進む
    - 学習 skill の tick だけが states / actions に **対で** 残る
    - 時間切れ (`on_timeout: advance`) で次の skill へ進む
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from inference.desktop import assembly
from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
from inference.desktop.lower_policy.initial_pose import initial_pose_from_config
from inference.desktop.lower_policy.skills.base import Skill
from inference.desktop.orchestrator import (
    Orchestrator,
    build_stage_skill_sequence,
    build_stage_transitions,
    enter_never,
    model_transition_pairs_for_stage,
)
from inference.desktop.perception.frame_source import FrameData
from inference.desktop.perception.g1_urdf_fk import G1_JOINT_NAMES
from inference.desktop.perception.stream import DetectionStream
from inference.desktop.recording import RunRecorder, record_policy_tick

SKILL_CONFIG_PATH = Path("inference/desktop/lower_policy/configs/skill_config.yaml")
CLEANUP_CONFIG_PATH = Path("inference/desktop/perception/configs/cleanup.yaml")
LEARNED_SKILLS = frozenset({
    "rotate_table_base", "pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten",
})


class _FakeRobot:
    """指令をそのまま実測として返す偽のロボット (腕 14 + Dex1 2)。"""

    def __init__(self) -> None:
        self.arm = np.zeros(14)
        self.hand = np.zeros(2)

    # --- actuator 側 ---
    def send_arm(self, target_14: np.ndarray) -> None:
        self.arm = np.asarray(target_14, dtype=np.float64)[:14].copy()

    def send_action(self, target: np.ndarray) -> None:      # hand actuator の API
        self.hand = np.asarray(target, dtype=np.float64).copy()

    # --- sensor 側 ---
    def joint_state(self):
        class _JointState:
            name = G1_JOINT_NAMES

        state = _JointState()
        state.position = np.concatenate((np.zeros(15), self.arm))
        state.velocity = np.zeros(29)
        state.effort = np.full(29, 0.5)
        state.t = time.monotonic_ns()
        return state

    def hand_state(self):
        class _HandState:
            pass

        state = _HandState()
        state.position_rad = self.hand.copy()
        state.left_received_monotonic_ns = time.monotonic_ns()
        state.right_received_monotonic_ns = time.monotonic_ns()
        state.t = time.monotonic_ns()
        return state


class _Source:
    def __init__(self, robot: _FakeRobot, kind: str) -> None:
        self._robot = robot
        self._kind = kind

    def get(self):
        return getattr(self._robot, self._kind)()


class _FrameSource:
    """毎回新しい t の packed stereo frame を返す。"""

    def __init__(self) -> None:
        self._t = 0

    def get(self) -> FrameData:
        self._t += 1
        return FrameData(rgb=np.zeros((4, 8, 3), dtype=np.uint8), t=self._t)


class _StubPerception:
    def predict(self, rgb):
        return []


class _StubPolicyAction:
    def __init__(self) -> None:
        self.action_chunk = np.zeros((16, 19), dtype=np.float32)
        self.latency_ms = 1.0
        self.metadata = {"stub": True}


class _StubLearnedSkill(Skill):
    """学習 skill の代わり。腕を保持し、last_action を持つ (記録の対を作る)。"""

    def __init__(self, name: str, robot: _FakeRobot) -> None:
        super().__init__()
        self.name = name
        self._robot = robot
        self.last_action = None
        self.steps = 0

    def _on_start(self, params: dict) -> None:
        self.steps = 0
        self.last_action = None

    def _on_stop(self) -> None:
        pass

    def step(self, obs: dict) -> np.ndarray:
        self.steps += 1
        self.last_action = _StubPolicyAction()
        return self._robot.arm.copy()


@pytest.fixture
def fast_skill_config() -> dict:
    """実物の skill_config を読み、頭の手順だけ test 用に速くする。"""
    config = copy.deepcopy(yaml.safe_load(SKILL_CONFIG_PATH.read_text()))
    config["arm_pre_motion"] = {
        "velocity_limit_rad_s": 50.0,
        "acceleration_limit_rad_s2": 500.0,
        "measured_tolerance_rad": 0.10,
        "stage_timeout_s": 5.0,
    }
    config["hand_pre_motion"] = {
        "velocity_limit_rad_s": 54.0,      # 1 tick で開ききる
        "tolerance_rad": 0.15,
        "required_stable_samples": 2,
        "minimum_timeout_sec": 5.0,
        "timeout_margin_sec": 3.0,
        "state_max_age_s": 5.0,
    }
    return config


def _build_stage_2(config: dict, robot: _FakeRobot, recorder: RunRecorder):
    sequence = build_stage_skill_sequence(2, is_start_stage=True, include_hand=True)
    registry: dict[str, Skill] = {}
    for head in assembly.build_head_procedure(
        skill_config=config,
        skill_name="rotate_table_base",
        initial_pose=initial_pose_from_config(config, "rotate_table_base"),
        hand_actuator=robot,
        hold_sec=0.0,
        include_hand=True,
    ):
        registry[head.name] = head
    for previous_skill, next_skill in model_transition_pairs_for_stage(
        2, is_start_stage=True
    ):
        for transition in assembly.build_model_transition_procedure(
            skill_config=config,
            previous_skill=previous_skill,
            next_skill=next_skill,
            initial_pose=initial_pose_from_config(config, next_skill),
            hand_actuator=robot,
            hold_sec=0.0,
            include_hand=True,
        ):
            registry[transition.name] = transition
    for name in LEARNED_SKILLS:
        registry[name] = _StubLearnedSkill(name, robot)

    dispatcher = SkillDispatchLowerPolicy(registry)
    orch = Orchestrator(
        _StubPerception(),
        DetectionStream(yaml.safe_load(CLEANUP_CONFIG_PATH.read_text())),
        dispatcher,
        initial_skill=sequence[0],
        transitions=build_stage_transitions(2, is_start_stage=True, include_hand=True),
        enter_check={name: enter_never for name in sequence},
        actuator_send_fn=robot.send_arm,
        joint_state_source=_Source(robot, "joint_state"),
        dex1_state_source=_Source(robot, "hand_state"),
        hard_timeout_by_skill={name: 0.05 for name in LEARNED_SKILLS},
        on_tick=lambda result, obs, skill: record_policy_tick(
            recorder, obs, skill, learned_skills=LEARNED_SKILLS
        ),
    )
    return orch, dispatcher, registry, sequence


def test_stage_2_runs_from_the_head_procedure_to_the_last_learned_skill(
    tmp_path: Path, fast_skill_config: dict
):
    robot = _FakeRobot()
    with RunRecorder(tmp_path) as recorder:
        orch, dispatcher, registry, sequence = _build_stage_2(
            fast_skill_config, robot, recorder
        )
        seen: list[str] = []
        original_tick = orch.tick

        def _tick(frame):
            active = dispatcher.active_skill_name
            if active is not None and (not seen or seen[-1] != active):
                seen.append(active)
            return original_tick(frame)

        orch.tick = _tick   # type: ignore[method-assign]
        orch.run_live(_FrameSource(), hz=200.0, startup_timeout=1.0, frame_timeout=1.0)

    # 列の全部を順番に通り、最後の skill まで到達する
    assert seen == sequence, f"{seen} != {sequence}"

    # 学習 skill の tick だけが states / actions に対で残る
    states = [json.loads(line) for line in (tmp_path / "states.jsonl").read_text().splitlines()]
    actions = [json.loads(line) for line in (tmp_path / "actions.jsonl").read_text().splitlines()]
    assert len(states) == len(actions) > 0
    assert {row["skill"] for row in states} <= LEARNED_SKILLS
    for state, action in zip(states, actions):
        assert state["t"] == action["t"]
        assert state["skill"] == action["skill"]

    # 時間切れ (on_timeout: advance) で次へ進んだので、全部の学習 skill が動いている
    assert all(registry[name].steps > 0 for name in LEARNED_SKILLS)


def test_the_hand_reaches_the_teacher_opening_before_the_policy_starts(
    tmp_path: Path, fast_skill_config: dict
):
    """頭の手順が終わった時点で、手は教師 frame 0 の開度になっている (INF-14)。"""
    robot = _FakeRobot()
    with RunRecorder(tmp_path) as recorder:
        orch, dispatcher, _, _ = _build_stage_2(fast_skill_config, robot, recorder)
        orch.run_live(
            _FrameSource(), hz=200.0, startup_timeout=1.0, frame_timeout=1.0,
            stop_on_skills=frozenset({"rotate_table_base"}),
        )

    expected = initial_pose_from_config(fast_skill_config, "rotate_table_base").dex1_target_rad
    np.testing.assert_allclose(robot.hand, expected, atol=1e-6)
    assert dispatcher.active_skill_name == "rotate_table_base"


# =========================================================================
# 会場経路 (boundary) の Stage 通し (Codex レビュー後の結合 test)
# =========================================================================
def _official_joint_sink_class(monkeypatch):
    """公式 `boundary/actions.py` の JointSink を、socket 無しで使う。

    この env には zmq / msgpack が無いので import だけ通す stand-in を入れる。
    検証 (`validate_chunk`) は公式実装そのまま。publish だけ記録に差し替える。
    """
    import importlib
    import sys
    import types

    monkeypatch.setitem(sys.modules, "zmq", types.ModuleType("zmq"))
    fake_msgpack = types.ModuleType("msgpack")
    fake_msgpack.packb = lambda *a, **k: b""
    fake_msgpack.unpackb = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "msgpack", fake_msgpack)
    fake_cv2 = types.ModuleType("cv2")
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    for name in [m for m in sys.modules if m.startswith("inference.desktop.boundary")]:
        monkeypatch.delitem(sys.modules, name)
    # ここで import する boundary は偽の zmq / msgpack に結び付く。この test より前に
    # import されていなかった module も後で消えるように、無い key を setitem で記録する
    # (undo は元が無い key を消す)。残すと後続の実 socket の test が
    # `zmq has no attribute 'Context'` で落ちる (Issue #172)。
    import inference.desktop as desktop_pkg

    monkeypatch.setattr(
        desktop_pkg, "boundary", getattr(desktop_pkg, "boundary", None), raising=False
    )
    for name in (
        "inference.desktop.boundary",
        "inference.desktop.boundary.actions",
        "inference.desktop.boundary.cameras",
        "inference.desktop.boundary.states",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        del sys.modules[name]
    actions = importlib.import_module("inference.desktop.boundary.actions")

    class _RecordingSink(actions.JointSink):
        chunks: list = []

        def __init__(self, port=5556, host="*"):  # noqa: D401 - no socket
            self.published = 0

        def send_chunk(self, chunk, issued_at=None):
            arr = self.validate_chunk(chunk)  # 公式の検証
            type(self).chunks.append(arr.copy())
            self.published += 1

        def close(self):
            pass

    boundary_pkg = types.ModuleType("inference.desktop.boundary")
    boundary_pkg.JointSink = _RecordingSink
    monkeypatch.setitem(sys.modules, "inference.desktop.boundary", boundary_pkg)
    _RecordingSink.chunks = []
    return _RecordingSink


@pytest.mark.requires_boundary
def test_boundary_lane_stage_2_publishes_only_validated_rows_and_keeps_targets(
    tmp_path: Path, fast_skill_config: dict, monkeypatch
):
    from inference.desktop.lower_policy.actuators.boundary_sink import (
        BoundaryActionSink,
        BoundaryArmActuator,
        BoundaryWalkActuator,
        assemble_action19,
    )
    from inference.desktop.lower_policy.skills.hold_pose import HoldPoseSkill
    from inference.desktop.orchestrator import self_bridged_transition_skill_names

    sink_cls = _official_joint_sink_class(monkeypatch)
    robot = _FakeRobot()
    hybrid_pair = frozenset({("pick_table_leg", "insert_table_leg")})
    sequence = build_stage_skill_sequence(
        2, is_start_stage=True, include_hand=True,
        skip_model_transition_pairs=hybrid_pair,
    )
    assert "hold_transition_pick_table_leg_to_insert_table_leg" in sequence

    sink = BoundaryActionSink(lane="joint")
    arm = BoundaryArmActuator()
    walk = BoundaryWalkActuator()
    sent_arms: list[np.ndarray] = []

    def _send_to_boundary(arms14) -> bool:
        body = np.concatenate((np.zeros(15), robot.arm))
        action19 = assemble_action19(
            None, arms14, tuple(robot.hand), measured_waist3=body[12:15]
        )
        sink.send_action(action19, body, navigate_cmd=walk.latest)
        sent_arms.append(np.asarray(arms14, dtype=np.float64).copy())
        robot.send_arm(arms14)  # WBC が追従したことにする
        return True

    arm.configure(sender=_send_to_boundary, state_getter=lambda: robot.arm.copy())
    walk.set_publish_callback(
        lambda: arm.send_action(arm.read_last_published_targets()[0])
    )
    arm.start()

    registry: dict[str, Skill] = {}
    for head in assembly.build_head_procedure(
        skill_config=fast_skill_config,
        skill_name="rotate_table_base",
        initial_pose=initial_pose_from_config(fast_skill_config, "rotate_table_base"),
        hand_actuator=robot,
        hold_sec=0.0,
        include_hand=True,
    ):
        registry[head.name] = head
    for previous_skill, next_skill in model_transition_pairs_for_stage(
        2, is_start_stage=True, skip_model_transition_pairs=hybrid_pair
    ):
        for transition in assembly.build_model_transition_procedure(
            skill_config=fast_skill_config,
            previous_skill=previous_skill,
            next_skill=next_skill,
            initial_pose=initial_pose_from_config(fast_skill_config, next_skill),
            hand_actuator=robot,
            hold_sec=0.0,
            include_hand=True,
            published_arm_target_provider=lambda: arm.read_last_published_targets()[0],
        ):
            registry[transition.name] = transition
    for name in self_bridged_transition_skill_names(*next(iter(hybrid_pair))):
        registry[name] = HoldPoseSkill(0.0, name=name)
    for name in LEARNED_SKILLS:
        registry[name] = _StubLearnedSkill(name, robot)

    dispatcher = SkillDispatchLowerPolicy(registry)
    seen: list[str] = []
    orch = Orchestrator(
        _StubPerception(),
        DetectionStream(yaml.safe_load(CLEANUP_CONFIG_PATH.read_text())),
        dispatcher,
        initial_skill=sequence[0],
        transitions=build_stage_transitions(
            2, is_start_stage=True, include_hand=True,
            skip_model_transition_pairs=hybrid_pair,
        ),
        enter_check={name: enter_never for name in sequence},
        actuator_send_fn=arm.send_action,  # 本番と同じ唯一の送信窓口
        joint_state_source=_Source(robot, "joint_state"),
        dex1_state_source=_Source(robot, "hand_state"),
        hard_timeout_by_skill={name: 0.05 for name in LEARNED_SKILLS},
        on_tick=lambda _r, _o, skill: (
            seen.append(skill.name)
            if skill is not None and (not seen or seen[-1] != skill.name)
            else None
        ),
    )
    orch.run_live(_FrameSource(), hz=200.0, startup_timeout=1.0, frame_timeout=1.0)

    assert seen == sequence, f"{seen} != {sequence}"
    chunks = sink_cls.chunks
    assert len(chunks) == len(sent_arms) > 0
    rows = np.concatenate(chunks)
    assert rows.shape[1] == 22
    np.testing.assert_allclose(rows[:, 4:18], np.asarray(sent_arms), atol=1e-6)
    np.testing.assert_allclose(rows[:, 21], 0.74)          # 骨盤高さは中立
    np.testing.assert_allclose(rows[:, 18:21], 0.0)        # 歩行指令なし
    assert np.all(np.abs(rows[:, 0:4]) <= 1.0)

    # 終了時の歩行停止は「最後に publish した target」を再送する (Codex #1)
    last = arm.read_last_published_targets()[0].copy()
    np.testing.assert_allclose(last, sent_arms[-1])
    walk.set_velocity(0.0, 0.0, 0.0)
    np.testing.assert_allclose(sent_arms[-1], last)
    np.testing.assert_allclose(sink_cls.chunks[-1][0, 18:21], 0.0)


@pytest.mark.requires_boundary
def test_the_fake_boundary_does_not_leak_into_later_tests():
    """偽の zmq に結び付いた boundary を後続の test に残さない (Issue #172)。

    残ると、この後の実 socket の test が `zmq has no attribute 'Context'` で落ちる。
    """
    import sys

    def _boundary_modules():
        return {
            name: module for name, module in sys.modules.items()
            if name.startswith("inference.desktop.boundary")
        }

    before = _boundary_modules()
    with pytest.MonkeyPatch.context() as mp:
        _official_joint_sink_class(mp)
        assert _boundary_modules() != before
    assert _boundary_modules() == before
