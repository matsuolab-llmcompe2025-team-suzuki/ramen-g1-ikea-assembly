from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from inference.desktop.lower_policy.kinematics.g1_arm import (
    DEX1_BASE_OFFSET,
    G1ArmKinematics,
)
from inference.desktop.lower_policy.kinematics.types import EEPose, Side, matrix_to_rpy
from inference.desktop.perception.g1_urdf_fk import (
    LEFT_WRIST_TOOL_OFFSET_M,
    RIGHT_WRIST_TOOL_OFFSET_M,
)
from inference.desktop.pick_leg_hybrid.config import DEFAULT_CONFIG_PATH
from inference.desktop.pick_leg_hybrid.boundary import (
    BoundaryConfig,
    BoundaryDecision,
    GraspBoundaryDetector,
)
from inference.desktop.pick_leg_hybrid.real_skill import (
    RealPickLegHybridVlaSkill,
    _quat_to_matrix_wxyz,
    build_vlm_self_check_images,
    load_reference_images,
)
from inference.desktop.pick_leg_hybrid.phases import Phase
from inference.desktop.pick_leg_hybrid.vlm import VlmAnswer
from inference.desktop.pick_leg_hybrid.overlay_input import FrameHistory
from inference.desktop.perception.g1_urdf_fk import G1WristFK
from inference.desktop.lower_policy.policies.base import PolicyAction


def test_pinned_vlm_references_exist_and_are_distinct() -> None:
    cfg, images = load_reference_images(DEFAULT_CONFIG_PATH)
    assert cfg.references.is_ready
    assert len(images) == 2
    assert all(len(image) > 10_000 for image in images)
    assert images[0] != images[1]
    assert cfg.phase2.hold_left == 4.5
    assert cfg.phase2.hold_right == 1.85


def test_vlm_history_owns_each_camera_generation() -> None:
    history = FrameHistory(2)
    reused = np.zeros((4, 4, 3), dtype=np.uint8)
    history.push(reused)
    reused.fill(255)
    history.push(reused)
    assert int(history._frames[0][0, 0, 0]) == 0
    assert int(history._frames[1][0, 0, 0]) == 255


def test_vlm_preflight_uses_the_same_five_images_as_runtime() -> None:
    cfg, _ = load_reference_images(DEFAULT_CONFIG_PATH)
    images = build_vlm_self_check_images(cfg, ["before", "after"])
    assert images == ["before", "after", "after", "after", "after"]
    assert len(images) == 2 + cfg.vlm.history_frames + 1 == 5


def test_phase2_goals_are_reachable_by_real_g1_joint_ik() -> None:
    cfg, _ = load_reference_images(DEFAULT_CONFIG_PATH)
    zero_seed = np.zeros(7, dtype=np.float64)
    cases = (
        (
            Side.LEFT,
            cfg.phase2.goal_left,
            LEFT_WRIST_TOOL_OFFSET_M,
        ),
        (
            Side.RIGHT,
            cfg.phase2.goal_right,
            RIGHT_WRIST_TOOL_OFFSET_M,
        ),
    )
    for side, goal, tool_offset in cases:
        kin = G1ArmKinematics(
            tcp_offset=tool_offset - np.asarray(DEX1_BASE_OFFSET)
        )
        target = EEPose(
            position=goal.pos,
            rpy=matrix_to_rpy(_quat_to_matrix_wxyz(goal.quat)),
        )
        result = kin.ik(target, zero_seed, side, np.zeros(3))
        assert result.ok, (side, result)
        assert result.position_error < 0.001
        assert result.rotation_error < 0.001


class _Policy:
    EXECUTION_HORIZON = 1

    def __init__(self) -> None:
        self.predict_calls = 0

    def reset(self) -> None:
        pass

    @staticmethod
    def build_state_from_raw(_raw) -> np.ndarray:
        return np.zeros(38, dtype=np.float32)

    def predict(self, _observation) -> PolicyAction:
        self.predict_calls += 1
        action = np.zeros((1, 19), dtype=np.float32)
        action[0, 17:19] = [4.5, 1.8]
        return PolicyAction(action_chunk=action, latency_ms=1.0)


class _Actuator:
    def __init__(self) -> None:
        self.actions: list[list[float]] = []

    def send_action(self, action) -> None:
        self.actions.append(list(action))


class _YesVlm:
    def ask(self, *_args, **_kwargs) -> VlmAnswer:
        return VlmAnswer(value=1)


def _observation(t_ns: int) -> dict:
    frame = np.zeros((480, 1280, 3), dtype=np.uint8)
    wrist = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "t": t_ns,
        "head_rgb": frame,
        "wrist_left_rgb": wrist,
        "wrist_right_rgb": wrist,
        "joint_state": SimpleNamespace(
            position=np.zeros(29, dtype=np.float64),
            velocity=np.zeros(29, dtype=np.float64),
        ),
        "hand_state": SimpleNamespace(
            position_rad=np.array([4.5, 2.2], dtype=np.float64)
        ),
        "cleaned": [],
    }


def test_real_adapter_switches_from_vla_to_joint_ik_without_reloading_policy() -> None:
    hand = _Actuator()
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(),
        waist_actuator=_Actuator(),
        hand_actuator=hand,
        fk=G1WristFK.from_urdf(),
        dispatch_waist=False,
        hybrid_config_path=DEFAULT_CONFIG_PATH,
        phase3_executor="vla",  # Explicit legacy comparison; production defaults to rules.
    )
    skill._boundary = GraspBoundaryDetector(
        cfg=BoundaryConfig(confirm_count=1, min_interval_sec=0.0),
        interlock_cfg=skill._hybrid_cfg.interlock,
        vlm_client=_YesVlm(),
        reference_images_b64=["A", "B"],
    )
    skill.start({})
    first = skill.step(_observation(1_000_000_000))
    assert first is not None and first.shape == (14,)
    assert skill.hybrid_phase is Phase.CARRY_TO_LEFT

    second = skill.step(_observation(2_000_000_000))
    assert second.shape == (14,)
    assert np.isfinite(second).all()
    assert skill.last_action is not None
    assert skill.last_action.metadata["hybrid_executor"] == "g1_joint_ik"
    assert hand.actions[-1] == [4.5, 1.85]
    skill.stop()


def test_rule_based_phase3_does_not_call_groot() -> None:
    policy = _Policy()
    hand = _Actuator()
    fk = G1WristFK.from_urdf()
    skill = RealPickLegHybridVlaSkill(
        policy=policy,
        waist_actuator=_Actuator(),
        hand_actuator=hand,
        fk=fk,
        dispatch_waist=False,
        hybrid_config_path=DEFAULT_CONFIG_PATH,
        next_initial_arm_target=np.zeros(14),
        next_initial_hand_target=np.array([3.0, 1.85]),
    )
    assert skill._phase3_executor == "rule_based"
    skill.start({})
    obs = _observation(1_000_000_000)
    left, right = skill._measured_ee(obs)
    skill._hybrid_state.advance()
    skill._hybrid_state.advance()
    skill._phase3_rule.start(
        skill._time_seconds(obs), left, right, np.zeros(14)
    )
    skill._last_safe_arm = np.zeros(14, dtype=np.float64)

    arm = skill.step(obs)

    assert arm is not None and arm.shape == (14,)
    assert policy.predict_calls == 0
    assert skill.last_action is not None
    assert skill.last_action.metadata["hybrid_executor"] == "rule_based_phase3"
    assert skill.last_action.metadata["hybrid_phase3_stage"] == "close_left"
    assert len(skill.last_action.metadata["hybrid_phase3_measured_arm"]) == 14
    assert len(
        skill.last_action.metadata["hybrid_phase3_measured_arm_velocity"]
    ) == 14
    assert len(skill.last_action.metadata["hybrid_phase3_transmitted_arm"]) == 14
    assert hand.actions[-1] == [0.0, 1.85]
    skill.stop()


def _rule_phase3_skill():
    """rule-based phase 3 に入った直後の skill (test_rule_based_phase3_does_not_call_groot と同じ組み方)。"""
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(),
        waist_actuator=_Actuator(),
        hand_actuator=_Actuator(),
        fk=G1WristFK.from_urdf(),
        dispatch_waist=False,
        hybrid_config_path=DEFAULT_CONFIG_PATH,
        next_initial_arm_target=np.zeros(14),
        next_initial_hand_target=np.array([3.0, 1.85]),
    )
    skill.start({})
    obs = _observation(1_000_000_000)
    left, right = skill._measured_ee(obs)
    skill._hybrid_state.advance()
    skill._hybrid_state.advance()
    skill._phase3_rule.start(skill._time_seconds(obs), left, right, np.zeros(14))
    skill._last_safe_arm = np.zeros(14, dtype=np.float64)
    return skill, obs, left, right


def _phase3_command(left, right, **overrides):
    from inference.desktop.pick_leg_hybrid.phase3_rule import (
        Phase3RuleCommand,
        Phase3RuleStage,
    )

    fields = dict(
        left=left,
        right=right,
        hand=np.array([4.5, 1.85]),
        stage=Phase3RuleStage.MOVE_RIGHT_REGRASP,
        moving_side=1,
        progress=0.5,
        complete=False,
        failure_reason=None,
        arm_target=None,
        timeout_reason=None,
    )
    fields.update(overrides)
    return Phase3RuleCommand(**fields)


def test_rule_phase3_ik_failure_holds_and_retries_instead_of_failing(
    monkeypatch, capsys
) -> None:
    """IK が解けないのは故障ではない。止めずに直前の安全な腕を保って解き直す。"""
    skill, obs, left, right = _rule_phase3_skill()
    monkeypatch.setattr(
        skill._phase3_rule, "step", lambda **_: _phase3_command(left, right)
    )
    unreachable = SimpleNamespace(
        ok=False,
        q=None,
        status=SimpleNamespace(value="not_converged"),
        position_error=0.2,
        rotation_error=0.1,
    )
    monkeypatch.setattr(skill._ik[Side.RIGHT], "ik", lambda *_, **__: unreachable)

    for _ in range(3):
        arm = skill.step(obs)
        assert arm is not None and arm.shape == (14,)
        assert skill.failure_reason is None
        assert skill.timeout_reason is None
        assert skill.last_action.metadata["hybrid_ik_failures"]
    # 解けない間の stderr は最初の 1 回だけ (30 Hz で流さない)。
    assert capsys.readouterr().err.count("IK cannot reach") == 1
    skill.stop()


def test_rule_phase3_stage_timeout_is_reported_to_orchestrator(monkeypatch) -> None:
    skill, obs, left, right = _rule_phase3_skill()
    reason = "phase3 rule stage 'move_right_regrasp' timed out after 6.10s"
    monkeypatch.setattr(
        skill._phase3_rule,
        "step",
        lambda **_: _phase3_command(
            left, right, moving_side=None, timeout_reason=reason
        ),
    )
    skill.step(obs)
    assert skill.failure_reason is None
    assert skill.timeout_reason == reason
    skill.stop()


def test_phase1_elapsed_time_does_not_advance_when_vlm_misses(monkeypatch) -> None:
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(), waist_actuator=_Actuator(), hand_actuator=_Actuator(),
        fk=G1WristFK.from_urdf(), hybrid_config_path=DEFAULT_CONFIG_PATH,
        phase3_executor="vla",
    )
    monkeypatch.setattr(
        skill._boundary, "update", lambda **_: BoundaryDecision(interlock=True)
    )
    skill.start({})
    for t in (1, 3, 19):
        skill.step(_observation(t * 1_000_000_000))
    assert skill.hybrid_phase is Phase.APPROACH_GRASP
    assert skill.failure_reason is None
    skill.stop()


def test_phase1_elapsed_time_without_grasp_does_not_advance(monkeypatch) -> None:
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(), waist_actuator=_Actuator(), hand_actuator=_Actuator(),
        fk=G1WristFK.from_urdf(), hybrid_config_path=DEFAULT_CONFIG_PATH,
        phase3_executor="vla",
    )
    monkeypatch.setattr(
        skill._boundary, "update", lambda **_: BoundaryDecision(interlock=False)
    )
    skill.start({})
    first = _observation(1_000_000_000)
    first["hand_state"].position_rad[1] = 4.5
    last = _observation(19_000_000_000)
    last["hand_state"].position_rad[1] = 4.5
    skill.step(first)
    skill.step(last)
    assert skill.hybrid_phase is Phase.APPROACH_GRASP
    assert skill.failure_reason is None
    assert "hybrid_phase_transition_reason" not in skill.last_action.metadata
    skill.stop()


def test_phase1_synthetic_hand_elapsed_time_does_not_advance(monkeypatch) -> None:
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(), waist_actuator=_Actuator(), hand_actuator=_Actuator(),
        fk=G1WristFK.from_urdf(), hybrid_config_path=DEFAULT_CONFIG_PATH,
        phase3_executor="vla",
    )
    answers = iter((BoundaryDecision(), BoundaryDecision(), BoundaryDecision()))
    monkeypatch.setattr(skill._boundary, "update", lambda **_: next(answers))
    skill.start({})
    for t in (1, 3, 19):
        obs = _observation(t * 1_000_000_000)
        obs["hand_state"].measured = False
        skill.step(obs)
    assert skill.hybrid_phase is Phase.APPROACH_GRASP
    assert "hybrid_phase_transition_reason" not in skill.last_action.metadata
    skill.stop()


@pytest.mark.parametrize("near_goal", [True, False])
def test_phase2_elapsed_time_does_not_advance_without_measured_goal(monkeypatch, near_goal: bool) -> None:
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(), waist_actuator=_Actuator(), hand_actuator=_Actuator(),
        fk=G1WristFK.from_urdf(), hybrid_config_path=DEFAULT_CONFIG_PATH,
        phase3_executor="vla",
    )
    skill.start({})
    first = _observation(1_000_000_000)
    left, right = skill._measured_ee(first)
    skill._hybrid_state.advance()
    skill._phase2.start(skill._time_seconds(first), left, right)
    skill._last_safe_arm = np.zeros(14)
    monkeypatch.setattr(
        skill._ik[Side.LEFT], "ik", lambda *_, **__: SimpleNamespace(ok=True, q=np.zeros(7))
    )
    monkeypatch.setattr(
        skill._ik[Side.RIGHT], "ik", lambda *_, **__: SimpleNamespace(ok=True, q=np.zeros(7))
    )
    offset = 0.03 if near_goal else 0.10
    goals = (
        skill._hybrid_cfg.phase2.goal_left,
        skill._hybrid_cfg.phase2.goal_right,
    )
    monkeypatch.setattr(
        skill, "_measured_ee",
        lambda _: tuple(
            type(goal).of(goal.pos + [offset, 0, 0], goal.quat)
            for goal in goals
        ),
    )
    skill.step(_observation(17_100_000_000))
    assert skill.hybrid_phase is Phase.CARRY_TO_LEFT
    assert skill.failure_reason is None
    assert "hybrid_phase_transition_reason" not in skill.last_action.metadata
    skill.stop()


# ---- 実測の手先はツール点 (B8a-01) ------------------------------------------------
# hybrid の目標・IK はツール点。実測を運営へ送る用の wrist_yaw_link 原点で出すと 13 cm
# ずれ、区間 2→3 が成立せず、区間 2 の頭で脚を持った右手を引き戻す。09-20 の実機 run
# (WandB kjyk6gw9) はツール点の実測で区間 3 へ進んでいた。


def _goal_joints() -> np.ndarray:
    """区間 2 の終点 (YAML) を IK で解いた 29 関節 = 腕が終点に着いた状態。"""
    cfg, _ = load_reference_images(DEFAULT_CONFIG_PATH)
    q = np.zeros(29, dtype=np.float64)
    cases = (
        (Side.LEFT, cfg.phase2.goal_left, LEFT_WRIST_TOOL_OFFSET_M, slice(15, 22)),
        (Side.RIGHT, cfg.phase2.goal_right, RIGHT_WRIST_TOOL_OFFSET_M, slice(22, 29)),
    )
    for side, goal, tool_offset, arm in cases:
        kin = G1ArmKinematics(tcp_offset=tool_offset - np.asarray(DEX1_BASE_OFFSET))
        target = EEPose(
            position=goal.pos, rpy=matrix_to_rpy(_quat_to_matrix_wxyz(goal.quat))
        )
        result = kin.ik(target, np.zeros(7), side, np.zeros(3))
        assert result.ok, (side, result)
        q[arm] = result.q
    return q


#: monotonic ns の時刻 (起動から 1 万秒) と 1 tick (30 Hz)。
_T0_NS = 10_000 * 1_000_000_000
_TICK_NS = 33_000_000


def _observation_with_joints(t_ns: int, q: np.ndarray) -> dict:
    obs = _observation(t_ns)
    obs["joint_state"].position = np.asarray(q, dtype=np.float64).copy()
    return obs


def _hybrid_skill_before_grasp(**kwargs) -> RealPickLegHybridVlaSkill:
    """区間 1 の skill。VLM は必ず「掴んだ」と答える。"""
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(),
        waist_actuator=_Actuator(),
        hand_actuator=_Actuator(),
        fk=G1WristFK.from_urdf(),
        dispatch_waist=False,
        hybrid_config_path=DEFAULT_CONFIG_PATH,
        next_initial_arm_target=np.zeros(14),
        next_initial_hand_target=np.array([3.0, 1.85]),
        **kwargs,
    )
    skill._boundary = GraspBoundaryDetector(
        cfg=BoundaryConfig(confirm_count=1, min_interval_sec=0.0),
        interlock_cfg=skill._hybrid_cfg.interlock,
        vlm_client=_YesVlm(),
        reference_images_b64=["A", "B"],
    )
    skill.start({})
    return skill


def test_phase2_advances_when_the_arms_reach_the_goal() -> None:
    q = _goal_joints()
    skill = _hybrid_skill_before_grasp()
    skill.step(_observation_with_joints(_T0_NS, q))
    assert skill.hybrid_phase is Phase.CARRY_TO_LEFT
    skill.step(_observation_with_joints(_T0_NS + _TICK_NS, q))
    assert skill.hybrid_phase is Phase.CARRY_TO_LEFT  # Pose alone is not enough.
    skill.step(_observation_with_joints(_T0_NS + 2_800_000_000, q))
    assert skill.hybrid_phase is Phase.HANDOVER_ONWARD
    skill.stop()


def test_phase2_starts_from_where_the_hands_are(monkeypatch) -> None:
    """区間 2 の最初の IK 目標は今の手先 (ツール点)。脚を持った手を引き戻さない。"""
    q = _goal_joints()
    skill = _hybrid_skill_before_grasp()
    skill.step(_observation_with_joints(_T0_NS, q))
    assert skill.hybrid_phase is Phase.CARRY_TO_LEFT
    targets: dict[Side, EEPose] = {}
    for side, kin in skill._ik.items():
        solve = kin.ik

        def record(target, *args, _side=side, _solve=solve, **kw):
            targets[_side] = target
            return _solve(target, *args, **kw)

        monkeypatch.setattr(kin, "ik", record)
    skill.step(_observation_with_joints(_T0_NS + _TICK_NS, q))
    for side, kin in skill._ik.items():
        hand = kin.fk(q[15:22] if side is Side.LEFT else q[22:29], side, q[12:15])
        assert np.linalg.norm(targets[side].position - hand.position) < 0.01, side
    skill.stop()


def test_boundary_insert_arrival_compares_tool_points_on_both_sides(monkeypatch) -> None:
    """会場の insert 到着判定: 目標関節にいるなら、目標と実測は同じツール点になる。"""
    insert_goal = _goal_joints()[15:29]
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(),
        waist_actuator=_Actuator(),
        hand_actuator=_Actuator(),
        fk=G1WristFK.from_urdf(),
        dispatch_waist=False,
        hybrid_config_path=DEFAULT_CONFIG_PATH,
        next_initial_arm_target=insert_goal,
        next_initial_hand_target=np.array([3.0, 1.85]),
        boundary_taskspace_arrival=True,
    )
    skill.start({})
    q = np.zeros(29, dtype=np.float64)
    q[15:29] = insert_goal
    obs = _observation_with_joints(_T0_NS, q)
    left, right = skill._measured_ee(obs)
    skill._hybrid_state.advance()
    skill._hybrid_state.advance()
    skill._phase3_rule.start(skill._time_seconds(obs), left, right, insert_goal)
    skill._last_safe_arm = insert_goal.copy()
    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return _phase3_command(kwargs["left"], kwargs["right"], moving_side=None)

    monkeypatch.setattr(skill._phase3_rule, "step", capture)
    skill.step(obs)
    target_left, target_right = seen["insert_target_ee"]
    for actual, target in ((seen["left"], target_left), (seen["right"], target_right)):
        assert np.linalg.norm(actual.pos - target.pos) < 1e-6
    skill.stop()


# ---- 区間 2→3 で握りを抜かない (B8a-04) ------------------------------------------------


def _pick_motion_limiter():
    """skill_config.yaml の motion_limits_default と同じ値 (30 Hz)。"""
    from inference.desktop.lower_policy.skills.motion_limiter import MotionLimiter

    return MotionLimiter(
        dim=19,
        arm_slice=slice(0, 17),
        hand_slice=slice(17, 19),
        arm_velocity=2.0,
        arm_acceleration=10.0,
        hand_velocity=6.0,
        hand_acceleration=30.0,
        control_hz=30.0,
    )


def test_phase2_to_phase3_keeps_the_right_hand_grip() -> None:
    """脚を右手 1 本で持ったまま区間 3 に入る。1 tick 目も締め付け (実測 − 指令) を保つ。"""
    q = _goal_joints()
    hand = _Actuator()
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(),
        waist_actuator=_Actuator(),
        hand_actuator=hand,
        fk=G1WristFK.from_urdf(),
        dispatch_waist=False,
        hybrid_config_path=DEFAULT_CONFIG_PATH,
        next_initial_arm_target=np.zeros(14),
        next_initial_hand_target=np.array([3.0, 1.85]),
        motion_limiter=_pick_motion_limiter(),
    )
    skill._boundary = GraspBoundaryDetector(
        cfg=BoundaryConfig(confirm_count=1, min_interval_sec=0.0),
        interlock_cfg=skill._hybrid_cfg.interlock,
        vlm_client=_YesVlm(),
        reference_images_b64=["A", "B"],
    )
    skill.start({})
    measured_right = 2.2  # 脚に当たって止まっている (_observation の hand_state)
    # 区間 1 で右手が締まり (指令が実測から limiter で下がる)、区間 2 を経て区間 3 に入るまで回す
    t = _T0_NS
    while skill.hybrid_phase is not Phase.HANDOVER_ONWARD:
        assert t < _T0_NS + 5 * 1_000_000_000, skill.hybrid_phase
        skill.step(_observation_with_joints(t, q))
        t += _TICK_NS
    grip_before = measured_right - hand.actions[-1][1]  # 区間 2 の最後の指令
    assert grip_before > 0.3
    skill.step(_observation_with_joints(t, q))  # 区間 3 の 1 tick 目
    grip_after = measured_right - hand.actions[-1][1]
    assert grip_after >= 0.9 * grip_before, (grip_before, grip_after)
    skill.stop()


def test_flush_resets_the_motion_limiter_unless_told_not_to() -> None:
    """やり直しの境目は従来どおり実測から (reset)。区間 2→3 だけ前の指令を保つ。"""
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(),
        waist_actuator=_Actuator(),
        hand_actuator=_Actuator(),
        fk=G1WristFK.from_urdf(),
        dispatch_waist=False,
        hybrid_config_path=DEFAULT_CONFIG_PATH,
        next_initial_arm_target=np.zeros(14),
        next_initial_hand_target=np.array([3.0, 1.85]),
        motion_limiter=_pick_motion_limiter(),
    )
    limiter = skill._motion_limiter
    limiter.apply(target=np.ones(19), measured=np.zeros(19))
    skill._flush_execution_state(reset_motion_limiter=False)
    assert limiter._previous_target is not None
    skill._flush_execution_state()
    assert limiter._previous_target is None


# ---- obs["t"] は常に monotonic ns (B8a-02) --------------------------------------------


def test_time_is_read_as_monotonic_ns_even_right_after_boot() -> None:
    """起動から 500 s (5e11 ns < 1e12): 区間 2 の 1 tick 後の進みは 33 ms 分。"""
    q = _goal_joints()
    skill = _hybrid_skill_before_grasp()
    t0 = 500 * 1_000_000_000
    skill.step(_observation_with_joints(t0, q))
    assert skill.hybrid_phase is Phase.CARRY_TO_LEFT
    now = skill._time_seconds(_observation_with_joints(t0 + _TICK_NS, q))
    assert now == pytest.approx(500.033)
    assert skill._phase2.progress(now) == pytest.approx(0.033 / 2.7, rel=1e-6)
    skill.stop()
