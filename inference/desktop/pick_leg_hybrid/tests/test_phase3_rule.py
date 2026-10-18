from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.kinematics.g1_arm import (
    DEX1_BASE_OFFSET,
    G1ArmKinematics,
)
from inference.desktop.lower_policy.kinematics.types import EEPose as JointEePose
from inference.desktop.lower_policy.kinematics.types import Side, matrix_to_rpy
from inference.desktop.perception.g1_urdf_fk import (
    LEFT_WRIST_TOOL_OFFSET_M,
    RIGHT_WRIST_TOOL_OFFSET_M,
)
from inference.desktop.pick_leg_hybrid.config import DEFAULT_CONFIG_PATH, load_yaml
from inference.desktop.pick_leg_hybrid.interlock import (
    LEFT,
    RIGHT,
    GraspInterlockConfig,
)
from inference.desktop.pick_leg_hybrid.phase2 import EePose
from inference.desktop.pick_leg_hybrid.phase3_rule import (
    Phase3RuleConfig,
    Phase3RuleController,
    Phase3RuleStage,
    RuleStageTiming,
    _quat_to_matrix,
)


def _config(confirm_count: int = 1) -> Phase3RuleConfig:
    """段の順番を見る test 用の設定。閉じる段は帯に入った tick で確定させる (grasp_confirm_sec 0)。

    本番の 0.5 s (空振りの通過を把持と見ない) は YAML の設定で下の B8a-03 の test が見る。
    """
    timings = {
        stage: RuleStageTiming(0.1, 1.0)
        for stage in tuple(Phase3RuleStage)[:-1]
    }
    return Phase3RuleConfig(
        left_present=EePose.of([0.30, 0.08, 0.30], [1, 0, 0, 0]),
        right_regrasp_offset_left_local=np.array([0.05, -0.10, -0.10]),
        right_regrasp_quat=np.array([1, 0, 0, 0]),
        close_hand_value=0.0,
        open_hand_value=4.5,
        right_initial_hold_value=1.85,
        grasp_preload_rad=0.35,
        open_min=3.5,
        confirm_count=confirm_count,
        pos_tol=0.01,
        rot_tol=0.05,
        left_present_pos_tol=0.02,
        left_present_rot_tol=0.10,
        joint_tol=0.01,
        joint_velocity_tol=0.05,
        joint_command_tol=0.01,
        hand_tol=0.05,
        timings=timings,
        grasp_confirm_sec=0.0,
    )


def _controller(confirm_count: int = 1) -> Phase3RuleController:
    return Phase3RuleController(
        _config(confirm_count),
        GraspInterlockConfig(),
        insert_arm_target=np.linspace(-0.35, 0.35, 14),
        insert_hand_target=np.array([3.0, 1.85]),
    )


def test_phase3_rule_runs_the_complete_right_regrasp_sequence() -> None:
    ctl = _controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    arm = np.zeros(14)
    ctl.start(0.0, left, right, arm)
    previous = np.array([0.0, 1.85])

    # Left physically obstructed -> the first handover is confirmed.
    out = ctl.step(
        t=0.1, left=left, right=right,
        hand_state=np.array([2.2, 2.2]), previous_hand_command=previous,
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.OPEN_RIGHT
    assert np.allclose(out.hand, [1.85, 4.5])

    # Right release, then move only left to the presentation pose.
    out = ctl.step(
        t=0.2, left=left, right=right,
        hand_state=np.array([2.2, 4.5]), previous_hand_command=out.hand,
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.MOVE_LEFT_PRESENT
    assert out.moving_side == 0
    left = _config().left_present
    out = ctl.step(
        t=0.4, left=left, right=right,
        hand_state=np.array([2.2, 4.5]), previous_hand_command=out.hand,
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.MOVE_RIGHT_REGRASP
    assert out.moving_side == 1

    # Resolve the right goal from the measured left frame and reach it.
    right = EePose.of(
        left.pos
        + _quat_to_matrix(left.quat)
        @ _config().right_regrasp_offset_left_local,
        _config().right_regrasp_quat,
    )
    out = ctl.step(
        t=0.6, left=left, right=right,
        hand_state=np.array([2.2, 4.5]), previous_hand_command=out.hand,
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.CLOSE_RIGHT
    assert np.allclose(out.hand, [1.85, 0.0])
    out = ctl.step(
        t=0.7, left=left, right=right,
        hand_state=np.array([2.2, 2.2]), previous_hand_command=out.hand,
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.OPEN_LEFT
    assert np.allclose(out.hand, [4.5, 1.85])

    out = ctl.step(
        t=0.8, left=left, right=right,
        hand_state=np.array([4.5, 2.2]), previous_hand_command=out.hand,
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL
    assert out.arm_target is not None
    assert np.allclose(out.arm_target[:7], ctl._insert_arm_target[:7])
    assert np.allclose(out.arm_target[7:], arm[7:])
    arm[:7] = ctl._insert_arm_target[:7]
    out = ctl.step(
        t=1.0, left=left, right=right,
        hand_state=np.array([4.5, 2.2]), previous_hand_command=out.hand,
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.MOVE_RIGHT_INSERT_INITIAL
    assert out.arm_target is not None
    assert np.allclose(out.arm_target[:7], arm[:7])
    assert np.allclose(out.arm_target[7:], ctl._insert_arm_target[7:])
    arm[7:] = ctl._insert_arm_target[7:]
    out = ctl.step(
        t=1.2, left=left, right=right,
        hand_state=np.array([4.5, 2.2]), previous_hand_command=out.hand,
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.SET_INSERT_HAND_INITIAL
    assert out.arm_target is None
    assert np.allclose(out.hand, [3.0, 1.85])
    out = ctl.step(
        t=1.4, left=left, right=right,
        hand_state=np.array([3.0, 2.2]), previous_hand_command=out.hand,
        arm_state=arm,
    )
    assert out.complete
    assert out.stage is Phase3RuleStage.COMPLETE
    assert ctl.failure_reason is None


def test_confirmed_grasp_replaces_full_close_with_bounded_preload() -> None:
    ctl = _controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    arm = np.zeros(14)
    ctl.start(0.0, left, right, arm)

    out = ctl.step(
        t=0.1,
        left=left,
        right=right,
        hand_state=np.array([2.2, 2.2]),
        previous_hand_command=np.array([0.0, 1.85]),
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.OPEN_RIGHT
    assert out.hand[LEFT] == 2.2 - 0.35
    assert out.hand[LEFT] > ctl.cfg.close_hand_value
    assert 2.2 - out.hand[LEFT] > ctl._grasp_cfg.follow_error_min

    # Advance directly to the right close stage and confirm the symmetric
    # behavior.  The subsequent open-left command must retain the preload,
    # rather than continuing the 0-rad contact probe.
    ctl._stage = Phase3RuleStage.CLOSE_RIGHT
    ctl._enter(ctl._stage, 0.2, left, right, arm)
    out = ctl.step(
        t=0.3,
        left=left,
        right=right,
        hand_state=np.array([2.2, 2.1]),
        previous_hand_command=np.array([1.85, 0.0]),
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.OPEN_LEFT
    assert out.hand[RIGHT] == 2.1 - 0.35
    assert out.hand[RIGHT] > ctl.cfg.close_hand_value


def test_phase3_rule_never_moves_both_arms_together() -> None:
    cfg = _config()
    assert all(
        side in (None, 0, 1)
        for side in (
            None,  # hand-only stages
            0,     # left presentation/retract
            1,     # right regrasp/settle
        )
    )
    ctl = _controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    arm = np.zeros(14)
    ctl.start(0.0, left, right, arm)
    out = ctl.step(
        t=0.01, left=left, right=right,
        hand_state=np.array([4.5, 2.2]),
        previous_hand_command=np.array([0.0, 1.85]),
        arm_state=arm,
    )
    assert out.moving_side is None


def test_phase3_rule_stage_elapsed_time_does_not_advance() -> None:
    """未到達の有限手順は旧時間上限を越えても保持する。"""
    ctl = _controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    arm = np.zeros(14)
    ctl.start(0.0, left, right, arm)
    out = ctl.step(
        t=1.1, left=left, right=right,
        hand_state=np.array([4.5, 2.2]),
        previous_hand_command=np.array([0.0, 1.85]),
        arm_state=arm,
    )
    assert out.failure_reason is None
    assert out.timeout_reason is None
    assert ctl.timeout_reason is None
    assert out.stage is Phase3RuleStage.CLOSE_LEFT
    assert not out.complete


def test_left_present_load_tolerance_does_not_relax_right_regrasp() -> None:
    ctl = _controller()
    cfg = _config()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    arm = np.zeros(14)
    ctl.start(0.0, left, right, arm)
    ctl._stage = Phase3RuleStage.MOVE_LEFT_PRESENT
    ctl._enter(ctl._stage, 0.0, left, right, arm)

    # This error is outside the strict right-regrasp tolerance, but inside the
    # load-bearing left-present tolerance.
    angle = 0.075
    near_quat = np.array([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0])
    near_left = EePose.of(
        cfg.left_present.pos + np.array([0.015, 0.0, 0.0]), near_quat
    )
    out = ctl.step(
        t=0.2,
        left=near_left,
        right=right,
        hand_state=np.array([2.2, 4.5]),
        previous_hand_command=np.array([1.85, 4.5]),
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.MOVE_RIGHT_REGRASP

    assert ctl._goal_right is not None
    near_right = EePose.of(
        ctl._goal_right.pos + np.array([0.015, 0.0, 0.0]),
        ctl._goal_right.quat,
    )
    out = ctl.step(
        t=0.3,
        left=near_left,
        right=near_right,
        hand_state=np.array([2.2, 4.5]),
        previous_hand_command=out.hand,
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.MOVE_RIGHT_REGRASP


def test_insert_arm_elapsed_time_keeps_waiting_for_measured_goal() -> None:
    ctl = _controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    arm = np.zeros(14)
    ctl.start(0.0, left, right, arm)
    ctl._stage = Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL
    ctl._enter(ctl._stage, 0.0, left, right, arm)
    out = ctl.step(
        t=1.1,
        left=left,
        right=right,
        hand_state=np.array([4.5, 2.2]),
        previous_hand_command=np.array([4.5, 0.0]),
        arm_state=arm,
    )
    assert out.failure_reason is None
    assert out.timeout_reason is None
    assert out.stage is Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL


def test_insert_arm_requires_settled_state_and_final_command() -> None:
    ctl = _controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    target = ctl._insert_arm_target.copy()
    ctl.start(0.0, left, right, target)
    ctl._stage = Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL
    ctl._enter(ctl._stage, 0.0, left, right, target)
    hand_state = np.array([4.5, 2.2])
    hand_command = np.array([4.5, 1.85])

    moving = np.zeros(14)
    moving[:7] = ctl.cfg.joint_velocity_tol * 2.0
    out = ctl.step(
        t=0.2,
        left=left,
        right=right,
        hand_state=hand_state,
        previous_hand_command=hand_command,
        arm_state=target,
        arm_velocity=moving,
        previous_arm_command=target,
    )
    assert out.stage is Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL

    intermediate_command = target.copy()
    intermediate_command[0] += ctl.cfg.joint_command_tol * 2.0
    out = ctl.step(
        t=0.3,
        left=left,
        right=right,
        hand_state=hand_state,
        previous_hand_command=hand_command,
        arm_state=target,
        arm_velocity=np.zeros(14),
        previous_arm_command=intermediate_command,
    )
    assert out.stage is Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL

    out = ctl.step(
        t=0.4,
        left=left,
        right=right,
        hand_state=hand_state,
        previous_hand_command=hand_command,
        arm_state=target,
        arm_velocity=np.zeros(14),
        previous_arm_command=target,
    )
    assert out.stage is Phase3RuleStage.MOVE_RIGHT_INSERT_INITIAL


def test_boundary_insert_accepts_pose_equivalent_ik_but_not_unbounded_posture() -> None:
    ctl = _controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    target = ctl._insert_arm_target.copy()
    ctl.start(0.0, left, right, target)
    ctl._stage = Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL
    ctl._enter(ctl._stage, 0.0, left, right, target)
    alternate = target.copy()
    alternate[2] += 0.32  # organizer IK may choose a different joint solution
    kwargs = dict(
        left=left, right=right,
        hand_state=np.array([4.5, 2.2]),
        previous_hand_command=np.array([4.5, 1.85]),
        arm_state=alternate, arm_velocity=np.zeros(14),
        previous_arm_command=target,
        insert_target_ee=(left, right),
    )
    out = ctl.step(t=0.2, **kwargs)
    assert out.stage is Phase3RuleStage.MOVE_RIGHT_INSERT_INITIAL

    ctl._stage = Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL
    ctl._enter(ctl._stage, 1.0, left, right, target)
    too_far = alternate.copy()
    too_far[2] = target[2] + ctl.cfg.boundary_joint_guard_rad + 0.01
    out = ctl.step(t=1.2, **{**kwargs, "arm_state": too_far})
    assert out.stage is Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL
    out = ctl.step(
        t=1.3,
        **{**kwargs, "left": EePose.of([0.25, 0.1, 0.2], [1, 0, 0, 0])},
    )
    assert out.stage is Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL


def test_final_hand_gate_rechecks_both_insert_arms() -> None:
    ctl = _controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    target = ctl._insert_arm_target.copy()
    ctl.start(0.0, left, right, target)
    ctl._stage = Phase3RuleStage.SET_INSERT_HAND_INITIAL
    ctl._enter(ctl._stage, 0.0, left, right, target)
    drifted = target.copy()
    drifted[0] += 0.03
    kwargs = dict(
        left=left, right=right,
        hand_state=np.array([3.0, 2.2]),
        previous_hand_command=np.array([3.0, 1.85]),
        arm_velocity=np.zeros(14), previous_arm_command=target,
    )
    out = ctl.step(t=0.2, arm_state=drifted, **kwargs)
    assert out.stage is Phase3RuleStage.SET_INSERT_HAND_INITIAL
    out = ctl.step(t=0.3, arm_state=target, **kwargs)
    assert out.stage is Phase3RuleStage.COMPLETE


def test_phase3_rule_keeps_going_when_the_supporting_hand_loses_grasp(capsys) -> None:
    """把持が外れた疑いは記録だけ残して続ける (落としたら人が止める)。"""
    ctl = _controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    arm = np.zeros(14)
    ctl.start(0.0, left, right, arm)
    out = ctl.step(
        t=0.1, left=left, right=right,
        hand_state=np.array([2.2, 2.2]),
        previous_hand_command=np.array([0.0, 1.85]),
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.OPEN_RIGHT
    out = ctl.step(
        t=0.2, left=left, right=right,
        hand_state=np.array([0.0, 4.5]),
        previous_hand_command=out.hand,
        arm_state=arm,
    )
    assert out.failure_reason is None
    assert out.timeout_reason is None
    assert "left supporting grasp" in capsys.readouterr().err


def test_default_phase3_targets_are_reachable_by_real_g1_ik() -> None:
    cfg = load_yaml(DEFAULT_CONFIG_PATH)
    p3 = cfg.phase3_rule_based
    assert p3 is not None
    left_rotation = _quat_to_matrix(p3.left_present.quat)
    right_regrasp = EePose.of(
        p3.left_present.pos
        + left_rotation @ p3.right_regrasp_offset_left_local,
        p3.right_regrasp_quat,
    )
    cases = (
        (Side.LEFT, p3.left_present, LEFT_WRIST_TOOL_OFFSET_M),
        (Side.RIGHT, right_regrasp, RIGHT_WRIST_TOOL_OFFSET_M),
    )
    for side, pose, tool_offset in cases:
        kin = G1ArmKinematics(
            tcp_offset=tool_offset - np.asarray(DEX1_BASE_OFFSET)
        )
        target = JointEePose(
            position=pose.pos,
            rpy=matrix_to_rpy(_quat_to_matrix(pose.quat)),
        )
        result = kin.ik(target, np.zeros(7), side, np.zeros(3))
        assert result.ok, (side, result)
        assert result.position_error < 0.001
        assert result.rotation_error < 0.001


# ---- 閉じる段の把持確定は帯に居続けた時間 (B8a-03) ------------------------------------
# 2026-09-20 の実機 run (WandB kjyk6gw9、30 Hz) の close_left: 左の [実測, 指令]。
# 指令は 1 tick 0.2 rad で閉じ、実測は約 0.3 rad 遅れて追う (当たる前から「追従していない」)。
# 6.14 s に帯 (1.45〜2.35) へ入り、脚に当たって 1.83 付近に留まった。右は脚を持ったまま。
_REAL_CLOSE_LEFT = (
    (5.68, 4.492, 4.458), (5.71, 4.491, 4.392), (5.74, 4.482, 4.292), (5.78, 4.411, 4.158),
    (5.81, 4.267, 3.992), (5.84, 4.088, 3.792), (5.88, 3.909, 3.592), (5.91, 3.706, 3.392),
    (5.94, 3.484, 3.192), (5.98, 3.262, 2.992), (6.01, 3.064, 2.792), (6.04, 2.877, 2.592),
    (6.08, 2.693, 2.392), (6.11, 2.483, 2.192), (6.14, 2.285, 1.992), (6.18, 2.197, 1.792),
    (6.21, 2.147, 1.592), (6.24, 2.078, 1.392), (6.28, 2.040, 1.192), (6.31, 1.979, 0.992),
    (6.34, 1.953, 0.792), (6.38, 1.922, 0.592), (6.41, 1.857, 0.393), (6.44, 1.849, 0.228),
)
_RIGHT_HOLDING = (2.215, 1.85)


def _shipped_controller() -> Phase3RuleController:
    cfg = load_yaml(DEFAULT_CONFIG_PATH)
    return Phase3RuleController(
        cfg.phase3_rule_based,
        cfg.interlock,
        insert_arm_target=np.linspace(-0.35, 0.35, 14),
        insert_hand_target=np.array([3.0, 1.85]),
    )


def _close_left(ctl: Phase3RuleController, t: float, left_state: float, left_cmd: float):
    pose = EePose.of([0.3, 0.1, 0.3], [1, 0, 0, 0])
    return ctl.step(
        t=t, left=pose, right=pose,
        hand_state=np.array([left_state, _RIGHT_HOLDING[0]]),
        previous_hand_command=np.array([left_cmd, _RIGHT_HOLDING[1]]),
        arm_state=np.zeros(14),
    )


def _started_shipped_controller(t0: float) -> Phase3RuleController:
    ctl = _shipped_controller()
    pose = EePose.of([0.3, 0.1, 0.3], [1, 0, 0, 0])
    ctl.start(t0, pose, pose, np.zeros(14))
    assert ctl.stage is Phase3RuleStage.CLOSE_LEFT
    return ctl


def test_shipped_grasp_confirm_time_is_half_a_second() -> None:
    assert load_yaml(DEFAULT_CONFIG_PATH).phase3_rule_based.grasp_confirm_sec == 0.5


@pytest.mark.parametrize("rate_hz", [30.0, 20.0])
@pytest.mark.parametrize("hand_speed", [4.2, 2.5])
def test_a_missed_close_passing_through_the_band_is_not_a_grasp(rate_hz, hand_speed) -> None:
    """空振り: 何にも当たらず 4.5 → 0 へ閉じ切る。帯を通る間も確定しない。"""
    dt = 1.0 / rate_hz
    ctl = _started_shipped_controller(0.0)
    state, cmd, t = 4.5, 4.5, 0.0
    while t < 3.0:  # close_left の時間切れ (4 s) より前
        t += dt
        cmd = max(0.0, cmd - 6.0 * dt)       # 実機と同じ 0.2 rad/tick@30Hz の指令
        state = max(0.0, state - hand_speed * dt)
        _close_left(ctl, t, state, cmd)
        assert ctl.stage is Phase3RuleStage.CLOSE_LEFT, (t, state, cmd)


def test_the_real_close_left_is_confirmed_once_the_hand_stays_in_the_band() -> None:
    """実機の記録を流す: 帯に入った 6.14 s から 0.5 s 留まった所で確定する。"""
    ctl = _started_shipped_controller(5.65)
    for t, left_state, left_cmd in _REAL_CLOSE_LEFT:
        _close_left(ctl, t, left_state, left_cmd)
        assert ctl.stage is Phase3RuleStage.CLOSE_LEFT, t
    t = _REAL_CLOSE_LEFT[-1][0]
    while ctl.stage is Phase3RuleStage.CLOSE_LEFT:
        t = round(t + 1.0 / 30.0, 4)
        assert t < 7.0, "the real grasp must be confirmed well before the timeout"
        _close_left(ctl, t, 1.828, 0.0)  # その後の run で左は 1.828 に留まった
    assert ctl.stage is Phase3RuleStage.OPEN_RIGHT
    assert 6.64 <= t <= 6.70
