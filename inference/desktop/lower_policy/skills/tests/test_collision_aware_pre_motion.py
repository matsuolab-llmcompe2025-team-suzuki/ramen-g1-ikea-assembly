from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from inference.desktop.lower_policy.actuators.g1_arm_sdk import (
    G1_ARM_POSITION_LOWER_RAD,
    G1_ARM_POSITION_UPPER_RAD,
)
from inference.desktop.lower_policy.skills.collision_aware_pre_motion import (
    ArmWaypoint,
    CollisionAwareArmPreMotionSkill,
    GoLiveWaitSkill,
    MeasuredArmWalkHoldSkill,
    PostWalkArmSettleSkill,
    build_boundary_lift_waypoints,
    lowered_walk_pose_violation,
    build_collision_aware_waypoints,
    validate_lowered_walk_pose,
)
from inference.desktop.lower_policy.actuators.mock import MockWalkActuator


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def obs_for_arm(arm: np.ndarray) -> dict:
    whole = np.zeros(29, dtype=np.float64)
    whole[15:29] = arm
    return {
        "joint_state": SimpleNamespace(
            position=whole,
            velocity=np.zeros(29, dtype=np.float64),
        )
    }


def obs_for_arm_and_leg_speed(arm: np.ndarray, leg_speed: float) -> dict:
    obs = obs_for_arm(arm)
    obs["joint_state"].velocity[:12] = leg_speed
    return obs


def obs_for_arm_and_arm_speed(arm: np.ndarray, arm_speed: float) -> dict:
    obs = obs_for_arm(arm)
    obs["joint_state"].velocity[15:29] = arm_speed
    return obs


def test_startup_abort_retreat_uses_only_completed_waypoints():
    start = np.zeros(14)
    skill = CollisionAwareArmPreMotionSkill(
        tuple(np.full(14, 0.1)),
        skill_name="startup",
        waypoint_profile="overhead_clearance",
    )
    skill.start({})
    skill.step(obs_for_arm(start))
    assert len(skill.retreat_waypoints) == 1
    np.testing.assert_allclose(skill.retreat_waypoints[-1], start)
    skill._stage_index = 2  # first two stages reached, third remains unconfirmed
    retreat = skill.retreat_waypoints
    assert len(retreat) == 3
    np.testing.assert_allclose(retreat[0], skill._targets[1])
    np.testing.assert_allclose(retreat[1], skill._targets[0])
    np.testing.assert_allclose(retreat[2], start)


def test_walk_hold_latches_measured_lowered_pose() -> None:
    arm = np.linspace(-0.2, 0.2, 14)
    skill = MeasuredArmWalkHoldSkill(dwell_sec=0.5)
    skill.start({})
    assert np.allclose(skill.step(obs_for_arm(arm)), arm)
    # dwell は「下がった姿勢を latch した時点」から数える (Issue #152)。
    # 最初の tick で latch するので、ほぼ dwell そのもの。
    assert skill.max_dwell_sec == pytest.approx(0.5, abs=0.05)


def test_walk_hold_rejects_raised_lateral_arms_before_command() -> None:
    arm = np.zeros(14)
    arm[1] = 1.0
    with pytest.raises(RuntimeError, match="pre-motion is blocked"):
        validate_lowered_walk_pose(arm)


def test_walk_envelope_accepts_small_real_tracking_overshoot() -> None:
    arm = np.zeros(14)
    arm[3] = 1.0005
    arm[10] = -1.0005
    assert np.array_equal(validate_lowered_walk_pose(arm), arm)


def test_walk_envelope_accepts_the_venue_arms_hanging_by_their_weight() -> None:
    """09-29 の会場 (G1 (3)・運営 WBC): 歩行用の姿勢へ送っても肘は 1.12〜1.13 で止まった (Issue #188)。"""
    arm = np.zeros(14)
    arm[3] = 1.130388
    arm[10] = 1.120908
    assert np.array_equal(validate_lowered_walk_pose(arm), arm)


def test_walk_envelope_still_rejects_material_elbow_excess() -> None:
    arm = np.zeros(14)
    arm[3] = 1.301
    with pytest.raises(RuntimeError, match="elbow"):
        validate_lowered_walk_pose(arm)


def test_post_walk_settle_reasserts_zero_and_keeps_arms_lowered() -> None:
    actuator = MockWalkActuator()
    arm = np.linspace(-0.2, 0.2, 14)
    clock = Clock()
    skill = PostWalkArmSettleSkill(
        actuator,
        minimum_settle_sec=1.0,
        required_stable_samples=2,
        timeout_sec=3.0,
        time_fn=clock,
    )

    skill.start({})
    assert actuator.events == [("set_velocity", 0.0, 0.0, 0.0, 1.0)]
    assert np.allclose(skill.step(obs_for_arm(arm)), arm)
    assert not skill.is_complete
    clock.now = 1.0
    skill.step(obs_for_arm(arm))
    assert skill.is_complete

    skill.stop()
    assert actuator.events == [
        ("set_velocity", 0.0, 0.0, 0.0, 1.0),
        ("set_velocity", 0.0, 0.0, 0.0, 1.0),
    ]


def test_post_walk_settle_rejects_arms_that_rose_during_walk() -> None:
    actuator = MockWalkActuator()
    arm = np.zeros(14)
    arm[1] = 1.0
    skill = PostWalkArmSettleSkill(actuator)
    skill.start({})
    with pytest.raises(RuntimeError, match="lowered walk envelope"):
        skill.step(obs_for_arm(arm))


def test_post_walk_settle_does_not_complete_while_legs_are_moving() -> None:
    actuator = MockWalkActuator()
    clock = Clock()
    arm = np.zeros(14)
    skill = PostWalkArmSettleSkill(
        actuator,
        minimum_settle_sec=0.5,
        max_leg_speed_rad_s=0.25,
        required_stable_samples=2,
        timeout_sec=2.0,
        time_fn=clock,
    )
    skill.start({})
    for timestamp in (0.0, 0.5, 1.0):
        clock.now = timestamp
        skill.step(obs_for_arm_and_leg_speed(arm, 0.4))
        assert not skill.is_complete


def test_post_walk_settle_fails_closed_if_motion_never_stops() -> None:
    actuator = MockWalkActuator()
    clock = Clock()
    arm = np.zeros(14)
    skill = PostWalkArmSettleSkill(
        actuator,
        minimum_settle_sec=0.5,
        timeout_sec=1.0,
        time_fn=clock,
    )
    skill.start({})
    skill.step(obs_for_arm_and_leg_speed(arm, 0.4))
    clock.now = 1.01
    with pytest.raises(TimeoutError, match="did not measurably settle"):
        skill.step(obs_for_arm_and_leg_speed(arm, 0.4))


def test_waypoint_order_matches_proven_collision_avoidance_path() -> None:
    waypoints = build_collision_aware_waypoints(np.zeros(14))
    assert [waypoint.name for waypoint in waypoints] == [
        "shoulder_pitch_backward_clearance",
        "lateral_high_clearance",
        "forward_outward_clearance",
        "policy_initial_pose",
    ]
    initial = np.zeros(14)
    lateral = waypoints[1].resolve(initial)
    assert lateral[1] == pytest.approx(1.5)
    assert lateral[8] == pytest.approx(-1.5)
    assert lateral[3] == pytest.approx(0.4)
    assert lateral[10] == pytest.approx(0.4)


def test_pre_motion_supports_phase3_registry_name() -> None:
    skill = CollisionAwareArmPreMotionSkill(
        np.zeros(14), skill_name="arm_pre_motion_for_rotate_table_base"
    )
    assert skill.name == "arm_pre_motion_for_rotate_table_base"
    with pytest.raises(ValueError, match="skill_name"):
        CollisionAwareArmPreMotionSkill(np.zeros(14), skill_name="")


def test_model_boundary_starts_from_last_published_target() -> None:
    measured = np.zeros(14)
    published = np.linspace(-0.07, 0.07, 14)
    skill = CollisionAwareArmPreMotionSkill(
        np.zeros(14), published_target_provider=lambda: published
    )
    skill.start({})
    np.testing.assert_allclose(skill.step(obs_for_arm(measured)), published)


def test_pre_motion_converges_all_stages_with_bounded_command_steps() -> None:
    clock = Clock()
    final = np.array(
        [-0.2, 1.2, 0, 0, 0, 0, 0, -0.2, -1.2, 0, 0, 0, 0, 0],
        dtype=np.float64,
    )
    skill = CollisionAwareArmPreMotionSkill(
        final,
        velocity_limit_rad_s=0.5,
        acceleration_limit_rad_s2=1.0,
        measured_tolerance_rad=0.1,
        stage_timeout_s=20.0,
        time_fn=clock,
    )
    skill.start({})
    measured = np.zeros(14, dtype=np.float64)
    previous = skill.step(obs_for_arm(measured))
    for _ in range(2000):
        clock.now += 0.02
        command = skill.step(obs_for_arm(measured))
        # Simulate a responsive but not instantaneous arm.
        measured += np.clip(command - measured, -0.02, 0.02)
        assert np.max(np.abs(command - previous)) <= 0.5 * 0.02 + 1e-9
        previous = command
        if skill.is_complete:
            break
    assert skill.is_complete
    assert np.max(np.abs(previous - final)) <= 1e-3


def test_pre_motion_timeout_does_not_report_complete() -> None:
    clock = Clock()
    skill = CollisionAwareArmPreMotionSkill(
        np.zeros(14), stage_timeout_s=0.2, time_fn=clock
    )
    skill.start({})
    measured = np.zeros(14)
    skill.step(obs_for_arm(measured))
    # Timeout is accumulated from bounded control updates, not one wall-clock
    # jump.  Three 0.1 s updates exceed this 0.2 s test budget.
    for timestamp in (0.1, 0.2, 0.3):
        clock.now = timestamp
        skill.step(obs_for_arm(measured))
    assert not skill.is_complete
    # 収束しないのは故障ではなく時間切れ。orchestrator は次へ進む。
    assert skill.failure_reason is None
    assert skill.timeout_reason is not None
    assert "did not converge" in skill.timeout_reason


def test_pre_motion_model_load_gap_does_not_consume_waypoint_timeout() -> None:
    """A background expert load may starve ticks while arm DDS keeps holding.

    The trajectory integrator already limits a resumed tick to 0.1 s.  Its
    timeout must use the same clock or a 30 s load would fail a 0.2 s waypoint
    without giving it the control updates needed to move.
    """

    clock = Clock()
    skill = CollisionAwareArmPreMotionSkill(
        np.zeros(14), stage_timeout_s=0.2, time_fn=clock
    )
    skill.start({})
    measured = np.zeros(14)
    skill.step(obs_for_arm(measured))

    clock.now = 30.0
    skill.step(obs_for_arm(measured))
    assert skill.timeout_reason is None

    # Once actual control updates consume the budget, a non-converging arm
    # times out (the orchestrator moves on; it is not a fault).
    clock.now = 30.1
    skill.step(obs_for_arm(measured))
    clock.now = 30.2
    skill.step(obs_for_arm(measured))
    assert skill.timeout_reason is not None
    assert skill.failure_reason is None


def test_pre_motion_requires_measured_arm_velocity_to_settle() -> None:
    clock = Clock()
    skill = CollisionAwareArmPreMotionSkill(
        np.zeros(14),
        velocity_limit_rad_s=50.0,
        acceleration_limit_rad_s2=500.0,
        measured_tolerance_rad=0.1,
        measured_velocity_tolerance_rad_s=0.05,
        stable_samples_required=2,
        stage_timeout_s=5.0,
        time_fn=clock,
    )
    skill.start({})
    measured = np.zeros(14)
    command = skill.step(obs_for_arm(measured))
    for _ in range(200):
        clock.now += 0.01
        command = skill.step(obs_for_arm_and_arm_speed(measured, 0.2))
        measured = command.copy()
        if np.max(np.abs(command - skill._targets[0])) <= 1e-3:  # noqa: SLF001
            break
    for _ in range(10):
        clock.now += 0.01
        skill.step(obs_for_arm_and_arm_speed(measured, 0.2))
    assert skill._stage_index == 0  # noqa: SLF001

    for _ in range(10):
        clock.now += 0.01
        skill.step(obs_for_arm_and_arm_speed(measured, 0.0))
        if skill._stage_index == 1:  # noqa: SLF001
            break
    assert skill._stage_index == 1  # noqa: SLF001


def test_boundary_lift_waypoints_raise_the_shoulders_then_settle() -> None:
    """境界の経路は「今の姿勢を少し上げる → 上げたまま次の開始姿勢 → 下ろす」。

    歩行後の頭上・側方の退避 (roll ±1.5) は skill 間には広すぎ、脚を保持した腕を
    大きく開くと持った脚がテーブルや反対の腕に当たる (Issue #152)。
    """
    final = np.linspace(-0.4, 0.4, 14)
    waypoints = build_boundary_lift_waypoints(final, 0.15)
    assert [w.name for w in waypoints] == [
        "boundary_lift_from_measured",
        "boundary_lifted_policy_initial_pose",
        "policy_initial_pose",
    ]

    measured = np.full(14, 0.2)
    lifted_now = waypoints[0].resolve(measured)
    # 肩 pitch だけ実測から 0.15 上がり (符号は負が上)、他は実測のまま
    assert lifted_now[0] == pytest.approx(0.2 - 0.15)
    assert lifted_now[7] == pytest.approx(0.2 - 0.15)
    assert lifted_now[1] == pytest.approx(0.2)
    assert lifted_now[3] == pytest.approx(0.2)

    lifted_target = waypoints[1].resolve(measured)
    assert lifted_target[0] == pytest.approx(final[0] - 0.15)
    assert lifted_target[7] == pytest.approx(final[7] - 0.15)
    # 最後は開始姿勢そのもの (roll を大きく振る経由点は無い)
    np.testing.assert_allclose(waypoints[2].resolve(measured), final)
    assert max(abs(float(v)) for w in waypoints for v in w.resolve(measured)) < 1.5


def test_boundary_lift_of_zero_goes_straight_to_the_initial_pose() -> None:
    final = np.zeros(14)
    waypoints = build_boundary_lift_waypoints(final, 0.0)
    measured = np.full(14, 0.3)
    np.testing.assert_allclose(waypoints[0].resolve(measured), measured)


def test_boundary_lift_rejects_a_negative_lift() -> None:
    with pytest.raises(ValueError, match="boundary_lift_rad"):
        build_boundary_lift_waypoints(np.zeros(14), -0.1)


def test_unknown_waypoint_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="waypoint_profile"):
        CollisionAwareArmPreMotionSkill(np.zeros(14), waypoint_profile="banzai")


# ---- 実測を引き継ぐ関節と 1.5 rad の検査 (Issue #159 T4) ----
# rotate_leg は締める最中に手首 yaw を 1.5〜1.61 まで捻る (教師の全 frame の 2.5%)。
# 大会経路の rotate_leg は時間切れでしか終わらないので、その姿勢のまま境界に入りうる。
_L_WRIST_YAW, _R_WRIST_YAW, _L_SHOULDER_PITCH = 6, 13, 0


def _tightening_pose() -> np.ndarray:
    measured = np.full(14, 0.2)
    measured[_L_WRIST_YAW] = -1.58
    measured[_R_WRIST_YAW] = 1.58
    return measured


def test_a_measured_wrist_beyond_1_5_is_carried_through_the_lift() -> None:
    lift = build_boundary_lift_waypoints(np.zeros(14), 0.15)[0]
    resolved = lift.resolve(_tightening_pose())
    assert resolved[_L_WRIST_YAW] == pytest.approx(-1.58)
    assert resolved[_R_WRIST_YAW] == pytest.approx(1.58)


def test_a_low_shoulder_pitch_can_still_be_lifted() -> None:
    """B11-06: 肩 pitch -1.40 から -0.15 上げると -1.55 (URDF の中)。"""
    measured = np.full(14, 0.2)
    measured[_L_SHOULDER_PITCH] = -1.40
    resolved = build_boundary_lift_waypoints(np.zeros(14), 0.15)[0].resolve(measured)
    assert resolved[_L_SHOULDER_PITCH] == pytest.approx(-1.55)


def test_an_absolute_waypoint_beyond_1_5_is_still_rejected() -> None:
    """経由点が自分で決めた値は、今までどおり 1.5 rad で検査する。"""
    target = np.zeros(14)
    target[_R_WRIST_YAW] = 1.6
    with pytest.raises(ValueError, match="1.5rad smoke limit"):
        ArmWaypoint("typo", tuple(target)).resolve(np.zeros(14))


def test_measured_joints_are_clipped_to_the_urdf_limits() -> None:
    """自前経路の arm actuator (G1ArmActuator.send_action) と同じ限界に収める。"""
    measured = np.full(14, 0.2)
    measured[_R_WRIST_YAW] = G1_ARM_POSITION_UPPER_RAD[_R_WRIST_YAW] + 0.05
    resolved = build_boundary_lift_waypoints(np.zeros(14), 0.15)[0].resolve(measured)
    assert np.all(resolved <= G1_ARM_POSITION_UPPER_RAD)
    assert np.all(resolved >= G1_ARM_POSITION_LOWER_RAD)
    assert resolved[_R_WRIST_YAW] == pytest.approx(G1_ARM_POSITION_UPPER_RAD[_R_WRIST_YAW])


def test_a_boundary_started_mid_tightening_completes() -> None:
    clock = Clock()
    skill = CollisionAwareArmPreMotionSkill(
        np.zeros(14),
        skill_name="arm_transition_rotate_leg_to_tighten_to_rotate_table_base",
        waypoint_profile="boundary_lift",
        time_fn=clock,
    )
    skill.start({})
    measured = _tightening_pose()
    for _ in range(3000):
        command = skill.step(obs_for_arm(measured))
        assert skill.failure_reason is None, skill.failure_reason
        measured = command.copy()
        clock.now += 1.0 / 30.0
        if skill.is_complete:
            break
    assert skill.is_complete


# ---- go-live 待ち (Issue #159 T5) ----
# 大会経路は client が繋がってから人が go-live を打つまで、運営 adapter が (T,25) を
# ロボットに渡さない。その間も act は 20 Hz で呼ばれる。頭の手順の先頭で、ロボットが
# 指令に従い始めるまで待つ。肩 pitch を少し上げる指令 (負が上) を出し、実測が
# ついてきたら完了。
_PITCH = (0, 7)


class WallClock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _go_live_skill(tmp_path, clock, wall) -> GoLiveWaitSkill:
    return GoLiveWaitSkill(
        name="wait_for_go_live_pick_table_leg",
        lift_rad=0.05,
        detect_rad=0.02,
        flag_paths=(str(tmp_path / "go_live"), str(tmp_path / "tmp_go_live")),
        time_fn=clock,
        wall_time_fn=wall,
    )


def test_go_live_wait_holds_forever_while_the_robot_does_not_follow(tmp_path) -> None:
    clock, wall = Clock(), WallClock()
    skill = _go_live_skill(tmp_path, clock, wall)
    skill.start({})
    measured = np.full(14, 0.1)
    for _ in range(20 * 120):  # 20 Hz で 2 分
        command = skill.step(obs_for_arm(measured))
        clock.now += 0.05
        wall.now += 0.05
    assert not skill.is_complete
    assert skill.failure_reason is None
    # 指令は「実測から肩を 0.05 上げた姿勢」から先に進まない
    expected = measured.copy()
    expected[list(_PITCH)] -= 0.05
    np.testing.assert_allclose(command, expected)


def test_go_live_wait_completes_when_the_arm_follows(tmp_path) -> None:
    clock, wall = Clock(), WallClock()
    skill = _go_live_skill(tmp_path, clock, wall)
    skill.start({})
    measured = np.full(14, 0.1)
    skill.step(obs_for_arm(measured))
    measured[0] -= 0.025  # go-live: 左肩が指令の方へ動いた
    skill.step(obs_for_arm(measured))
    assert skill.is_complete
    assert skill.completion_mode == "followed"


def test_go_live_wait_ignores_the_arm_drooping(tmp_path) -> None:
    """重力で腕が垂れる向き (pitch が増える) は、従い始めたとはみなさない。"""
    clock, wall = Clock(), WallClock()
    skill = _go_live_skill(tmp_path, clock, wall)
    skill.start({})
    measured = np.full(14, 0.1)
    skill.step(obs_for_arm(measured))
    measured[list(_PITCH)] += 0.05
    skill.step(obs_for_arm(measured))
    assert not skill.is_complete


def test_go_live_can_be_forced_by_touching_the_flag_file(tmp_path) -> None:
    """検出できないとき、人が `touch ~/ramen_logs/go_live` で強制できる。見たら消す。"""
    clock, wall = Clock(), WallClock()
    skill = _go_live_skill(tmp_path, clock, wall)
    skill.start({})
    measured = np.full(14, 0.1)
    skill.step(obs_for_arm(measured))
    flag = tmp_path / "go_live"
    flag.touch()
    os.utime(flag, (wall.now + 1.0, wall.now + 1.0))
    wall.now += 1.0
    skill.step(obs_for_arm(measured))
    assert skill.is_complete
    assert skill.completion_mode == "forced"
    assert not flag.exists()


def test_a_flag_left_from_an_earlier_run_is_removed_and_ignored(tmp_path) -> None:
    clock, wall = Clock(), WallClock()
    stale = tmp_path / "tmp_go_live"
    stale.touch()
    os.utime(stale, (wall.now - 600.0, wall.now - 600.0))
    skill = _go_live_skill(tmp_path, clock, wall)
    skill.start({})
    skill.step(obs_for_arm(np.full(14, 0.1)))
    assert not skill.is_complete
    assert not stale.exists()


def test_go_live_wait_rejects_a_lift_that_cannot_be_detected() -> None:
    with pytest.raises(ValueError, match="detect_rad"):
        GoLiveWaitSkill(name="x", lift_rad=0.02, detect_rad=0.02)
def test_walk_hold_lowers_the_arms_when_they_are_above_the_envelope() -> None:
    """歩く前に腕が上がっていたら、止めずに下ろしてから歩く (Issue #152)。

    stage 0 を人の手当て無しで始められるようにする。下ろし終わるまでは
    dwell で次 (歩行) へ行かせない。
    """
    clock = Clock()
    raised = np.zeros(14)
    raised[0] = -1.2  # 肩 pitch が上がっている
    skill = MeasuredArmWalkHoldSkill(
        dwell_sec=0.5,
        lowered_pose_rad=np.zeros(14),
        lowering_settings={
            "velocity_limit_rad_s": 50.0,
            "acceleration_limit_rad_s2": 500.0,
            "measured_tolerance_rad": 0.1,
            "measured_velocity_tolerance_rad_s": 0.05,
            "stable_samples_required": 2,
        },
        time_fn=clock,
    )
    skill.start({})

    measured = raised.copy()
    command = skill.step(obs_for_arm(measured))
    # 下ろす指令が出る (止まらない)。まだ歩かせない
    assert skill.max_dwell_sec is None

    for _ in range(200):
        clock.now += 0.033
        command = skill.step(obs_for_arm(measured))
        measured += np.clip(command - measured, -0.05, 0.05)
        if skill.max_dwell_sec is not None:
            break

    assert skill.max_dwell_sec is not None, "下ろし終わらなかった"
    assert lowered_walk_pose_violation(measured) is None
    # latch した後は dwell が効く (latch までの時間 + dwell)
    assert skill.max_dwell_sec > 0.5


_FAST_LOWERING = {
    "velocity_limit_rad_s": 50.0,
    "acceleration_limit_rad_s2": 500.0,
    "measured_tolerance_rad": 0.1,
    "measured_velocity_tolerance_rad_s": 0.05,
    "stable_samples_required": 2,
}
# 運営 WBC の既定の姿勢 (decoupled_wbc の ElbowPose.LOW: 肩 roll ±0.2、他 0)。
# 肘 0 なので前腕が水平に前を向く「腕が前の HOME」。歩行の範囲の中にある。
_WBC_HOME = np.zeros(14)
_WBC_HOME[1], _WBC_HOME[8] = 0.2, -0.2
_LOWERED = _WBC_HOME.copy()
_LOWERED[3] = _LOWERED[10] = 0.9


def _drive_walk_hold(skill, clock, measured, *, follow=True, ticks=300):
    """setup を回す。follow=True なら実測が指令へ 1 tick 0.05 rad ずつついてくる。"""
    for _ in range(ticks):
        clock.now += 0.033
        command = skill.step(obs_for_arm(measured))
        if follow:
            measured += np.clip(command - measured, -0.05, 0.05)
        if skill.max_dwell_sec is not None:
            return command, measured
    raise AssertionError("setup が保持に移らなかった")


def test_walk_hold_lowers_the_arms_from_the_wbc_home() -> None:
    """会場の開始姿勢 (腕が前の HOME) は歩行の範囲の中でも、下ろしてから歩く。"""
    assert lowered_walk_pose_violation(_WBC_HOME) is None  # 範囲の中
    clock = Clock()
    skill = MeasuredArmWalkHoldSkill(
        dwell_sec=0.5,
        lowered_pose_rad=_LOWERED,
        lowering_settings=_FAST_LOWERING,
        time_fn=clock,
    )
    skill.start({})
    skill.step(obs_for_arm(_WBC_HOME.copy()))  # 1 tick 目は実測から始める
    clock.now += 0.033
    second = skill.step(obs_for_arm(_WBC_HOME.copy()))
    assert skill.max_dwell_sec is None, "HOME のまま歩行へ進ませない"
    assert second[3] > _WBC_HOME[3] and second[10] > _WBC_HOME[10], "肘を下ろす向き"
    held, measured = _drive_walk_hold(skill, clock, _WBC_HOME.copy())
    np.testing.assert_allclose(measured, _LOWERED, atol=0.1)
    np.testing.assert_allclose(held, measured)


def test_walk_hold_keeps_arms_that_are_already_lowered() -> None:
    clock = Clock()
    skill = MeasuredArmWalkHoldSkill(
        dwell_sec=0.5,
        lowered_pose_rad=_LOWERED,
        lowering_settings=_FAST_LOWERING,
        time_fn=clock,
    )
    skill.start({})
    near = _LOWERED + 0.05
    np.testing.assert_allclose(skill.step(obs_for_arm(near)), near)
    assert skill.max_dwell_sec == pytest.approx(0.5)


def test_walk_hold_confirms_lowering_with_the_venue_checker() -> None:
    """会場は運営 IK が別の関節角で同じ手首に届く。関節が目標と違っても task-space で確かめる。"""
    clock = Clock()
    calls = []

    def checker(goal, measured, velocity):
        calls.append(goal.copy())
        return True, "ee_pos_error=0.0010m ee_rot_error=0.0020rad speed=0.000rad/s"

    skill = MeasuredArmWalkHoldSkill(
        dwell_sec=0.5,
        lowered_pose_rad=_LOWERED,
        lowering_settings=_FAST_LOWERING,
        measured_convergence_checker=checker,
        time_fn=clock,
    )
    skill.start({})
    ik_solution = _LOWERED.copy()
    ik_solution[2] = 0.3  # 肩 yaw が違う冗長な解 (関節の許容 0.1 rad の外)
    _drive_walk_hold(skill, clock, ik_solution, follow=False)
    assert calls, "下ろし終わりの判定に checker を使っていない"


def _ik_solution_outside_the_envelope() -> np.ndarray:
    """手先では下ろせたが、運営 IK の選んだ肘が歩行の範囲 (1.25 + 0.05) をわずかに超える解。"""
    solution = _LOWERED.copy()
    solution[3] = 1.35
    assert lowered_walk_pose_violation(solution) is not None
    return solution


def _converging_walk_hold(latch_check: str, clock: Clock) -> MeasuredArmWalkHoldSkill:
    skill = MeasuredArmWalkHoldSkill(
        dwell_sec=0.5,
        lowered_pose_rad=_LOWERED,
        lowering_settings=_FAST_LOWERING,
        measured_convergence_checker=lambda goal, measured, velocity: (True, "ee ok"),
        latch_check=latch_check,
        time_fn=clock,
    )
    skill.start({})
    return skill


def test_joint_latch_check_stops_before_walking_outside_the_envelope() -> None:
    """既定 (joint): 下ろす動きが収束しても、関節が歩行の範囲の外なら歩かずに止める。"""
    clock = Clock()
    skill = _converging_walk_hold("joint", clock)
    with pytest.raises(RuntimeError, match="lowered walk envelope"):
        _drive_walk_hold(skill, clock, _ik_solution_outside_the_envelope(), follow=False)


def test_converged_latch_check_holds_the_measured_pose_and_warns(capsys) -> None:
    """converged: 収束していれば範囲をわずかに外れても記録して保持する (接続テストで決める)。"""
    clock = Clock()
    skill = _converging_walk_hold("converged", clock)
    solution = _ik_solution_outside_the_envelope()
    held, _ = _drive_walk_hold(skill, clock, solution.copy(), follow=False)
    np.testing.assert_allclose(held, solution)
    assert "outside the walk envelope" in capsys.readouterr().err


def test_an_unknown_latch_check_is_rejected() -> None:
    with pytest.raises(ValueError, match="latch_check"):
        MeasuredArmWalkHoldSkill(dwell_sec=0.5, latch_check="skip")


def test_the_configured_lowered_pose_really_lowers_the_arms() -> None:
    """skill_config.yaml の walk_lowered_pose は、前腕を前に出した姿勢ではない。"""
    from pathlib import Path

    import yaml

    from inference.desktop.lower_policy.kinematics.g1_arm import G1ArmKinematics
    from inference.desktop.lower_policy.kinematics.types import Side
    from inference.desktop.lower_policy.pose_utils import densify_pose

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs" / "skill_config.yaml").read_text()
    )
    pose = densify_pose(
        config["skills"]["walk_lowered_pose"]["pose_rad"], context="walk_lowered_pose"
    )
    kin = G1ArmKinematics()
    home_z = kin.fk(np.zeros(7), Side.LEFT).position[2]
    for side, arm in ((Side.LEFT, pose[:7]), (Side.RIGHT, pose[7:])):
        wrist = kin.fk(arm, side).position
        assert wrist[0] < 0.20, f"{side}: 手首が前に出ている ({wrist[0]:.3f} m)"
        assert wrist[2] < home_z - 0.10, f"{side}: HOME より 10 cm 以上下がっていない"
    # 運営 WBC の肩 roll の可動範囲 (左 >= 0.19 / 右 <= -0.19) の中
    assert pose[1] >= 0.19 and pose[8] <= -0.19
    # 到達判定の許容 (arm_pre_motion.measured_tolerance_rad) だけずれても歩行の範囲の中
    tolerance = float(config["arm_pre_motion"]["measured_tolerance_rad"])
    for shift in (-tolerance, tolerance):
        assert lowered_walk_pose_violation(pose + shift) is None, shift


def test_walk_hold_still_blocks_when_no_lowered_pose_is_configured() -> None:
    """従来どおり、下ろす姿勢が設定されていなければ止める。"""
    raised = np.zeros(14)
    raised[1] = 1.0
    skill = MeasuredArmWalkHoldSkill(dwell_sec=0.5)
    skill.start({})
    with pytest.raises(RuntimeError, match="lowered walk envelope"):
        skill.step(obs_for_arm(raised))


def test_lowered_pose_outside_the_envelope_is_rejected() -> None:
    bad = np.zeros(14)
    bad[1] = 1.2
    with pytest.raises(ValueError, match="outside the walk envelope"):
        MeasuredArmWalkHoldSkill(dwell_sec=0.5, lowered_pose_rad=bad)


def test_boundary_can_confirm_equivalent_taskspace_ik_solution() -> None:
    """Organizer IK may reach the same wrist pose with a different redundant q."""

    clock = Clock()
    details = []

    def checker(goal, measured, velocity):
        details.append((goal.copy(), measured.copy(), velocity.copy()))
        return True, "ee_pos_error=0.0010m ee_rot_error=0.0020rad speed=0.000rad/s"

    skill = CollisionAwareArmPreMotionSkill(
        np.full(14, 0.2),
        waypoint_profile="direct",
        velocity_limit_rad_s=10.0,
        acceleration_limit_rad_s2=100.0,
        stable_samples_required=2,
        measured_convergence_checker=checker,
        time_fn=clock,
    )
    skill.start({})
    measured = np.zeros(14)
    for _ in range(20):
        clock.now += 0.1
        skill.step(obs_for_arm(measured))
        if skill.is_complete:
            break
    assert skill.is_complete
    assert details


def test_stalled_ticks_time_out_on_the_wall_clock_limit() -> None:
    """Codex #12: tick が 1 s に 1 回しか来なくても実時間の上限で時間切れにする。

    制御時間は 1 tick 0.1 s までしか数えないので、実時間 60 s でも 6 s。
    """
    clock = Clock()
    target = np.full(14, 0.5)
    skill = CollisionAwareArmPreMotionSkill(
        target,
        stage_timeout_s=15.0,
        stage_wall_timeout_s=20.0,
        waypoint_profile="direct",
        time_fn=clock,
    )
    skill.start({})
    stuck = obs_for_arm(np.zeros(14))  # 実測が動かない
    skill.step(stuck)
    for _ in range(19):
        clock.now += 1.0
        skill.step(stuck)
    assert skill.timeout_reason is None  # 実時間 19 s、制御時間 1.9 s
    clock.now += 2.0
    skill.step(stuck)
    assert skill.failure_reason is None
    assert skill.timeout_reason is not None
    assert "wall" in skill.timeout_reason


def test_wall_limit_must_not_undercut_the_control_limit() -> None:
    with pytest.raises(ValueError, match="stage_wall_timeout_s"):
        CollisionAwareArmPreMotionSkill(
            np.zeros(14), stage_timeout_s=15.0, stage_wall_timeout_s=5.0
        )


def test_default_wall_limit_tolerates_a_model_load_gap() -> None:
    assert CollisionAwareArmPreMotionSkill(
        np.zeros(14), stage_timeout_s=15.0
    )._stage_wall_timeout == 60.0
    assert CollisionAwareArmPreMotionSkill(
        np.zeros(14), stage_timeout_s=30.0
    )._stage_wall_timeout == 120.0
