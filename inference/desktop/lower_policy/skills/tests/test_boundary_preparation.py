from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from inference.desktop.lower_policy.actuators.tests.test_boundary_sink import _make_joint_sink
from inference.desktop.lower_policy.skills.boundary_preparation import BoundaryPreparation, PreparationSettings
from inference.desktop.lower_policy.skills.collision_aware_pre_motion import (
    ArmWaypoint, CollisionAwareArmPreMotionSkill, build_collision_aware_waypoints,
)


class Console:
    stage = 1

    def __init__(self):
        self.view = None
        self.key = None

    def show(self, stage, phase, allowed):
        view = (phase, allowed)
        if view != self.view:
            self.key = None
            self.view = view

    def poll(self):
        key, self.key = self.key, None
        return key


def setup(waypoints=None):
    config = yaml.safe_load((Path(__file__).parents[2] / "configs/skill_config.yaml").read_text())
    cfg = PreparationSettings.from_config(config)
    sink, transport = _make_joint_sink()
    gotos, events, holds = [], [], []
    transport.send_goto = lambda left, right, **kw: gotos.append((np.r_[left, right], kw))
    sink._log_fn = events.append
    clock = SimpleNamespace(now=10.0)
    sample = SimpleNamespace(current=None)
    console = Console()
    engine = BoundaryPreparation(
        waypoints or (ArmWaypoint("target", (0.5,) * 14),), sink=sink, settings=cfg,
        record_hold=lambda q: holds.append(q.copy()), state_getter=lambda: sample.current,
        console=console, clock=lambda: clock.now,
    )

    def tick(value=0.0, *, speed=0.0, dt=0.05, duplicate=False):
        clock.now += dt
        if not duplicate:
            sample.current = SimpleNamespace(
                position=np.r_[np.zeros(15), np.broadcast_to(value, (14,))],
                velocity=np.r_[np.zeros(15), np.full(14, speed)],
                received_monotonic_ns=int(round(clock.now * 1e9)),
            )
        engine.step({"joint_state": sample.current})

    return SimpleNamespace(**locals())


@pytest.mark.requires_boundary
def test_one_shot_no_ordinary_chunk_even_when_forced_or_navigation_changes():
    s = setup()
    s.tick()
    for _ in range(40):
        s.tick(0.5)
        if s.engine.complete:
            break
        assert not s.sink.send_action(
            np.zeros(19), np.zeros(29), force=True, navigate_cmd=(1, 0, 0)
        )
    assert s.engine.complete
    assert len(s.gotos) == 1
    assert s.gotos[0][1]["hands"] is None
    np.testing.assert_allclose(s.holds[-1], 0.5)
    assert s.transport.chunks == []
    assert s.sink.send_action(np.r_[np.zeros(3), np.full(14, 0.5), np.zeros(2)], np.zeros(29))


@pytest.mark.parametrize("residual", [0.1087, 0.1108, 0.1162])
def test_late_arrival_recovers_without_reloading_or_rerouting(residual):
    s = setup()
    s.tick()
    for _ in range(160):
        s.tick(0.5 - residual)
    assert s.engine.phase == "holding"
    assert not s.engine.complete
    s.console.key = "enter"
    s.tick(0.5 - residual)
    assert not s.engine.complete
    for _ in range(10):
        s.tick(0.5 - 0.08445)
    assert s.engine.phase == "ready"
    assert not s.engine.complete
    s.console.key = "enter"
    s.tick(0.5 - 0.08445)
    assert s.engine.complete
    assert len(s.gotos) == 1


def test_unreachable_or_clamped_pose_holds_and_retry_is_same_waypoint():
    s = setup((ArmWaypoint("first", (0.5,) * 14), ArmWaypoint("second", (0.7,) * 14)))
    s.tick()
    for _ in range(170):
        s.tick(0.2)
    assert s.engine.phase == "holding" and s.engine.index == 0
    assert len(s.gotos) == 1  # No ACK / no progress never causes blind retransmission.
    s.console.key = "r"
    s.tick(0.2)
    assert len(s.gotos) == 2
    np.testing.assert_allclose(s.gotos[1][0], 0.5)
    assert s.engine.index == 0


def test_duplicate_state_cannot_satisfy_dwell_and_stale_state_is_rejected():
    s = setup()
    s.tick()
    for _ in range(40):
        s.tick(0.2)
    s.tick(0.5)
    for _ in range(8):
        s.tick(duplicate=True)
    assert not s.engine.complete
    with pytest.raises(RuntimeError, match="fresh joint state"):
        s.tick(duplicate=True, dt=0.2)


def test_moving_through_target_is_not_arrival():
    s = setup()
    s.tick()
    for _ in range(160):
        s.tick(0.5, speed=0.1)
    assert s.engine.phase == "holding" and not s.engine.complete


def test_cancel_uses_measured_pose_never_final_target_or_double_compensation():
    s = setup()
    s.tick()
    s.tick(0.12)
    s.engine.cancel()
    np.testing.assert_allclose(s.gotos[-1][0], 0.12)
    np.testing.assert_allclose(s.holds[-1], 0.12)
    assert s.engine.phase == "interrupted"
    assert s.sink._preparation_owner is None
    s.engine.cancel()
    assert len(s.gotos) == 2


def test_long_move_split_on_line_below_official_limit():
    goal = np.zeros(14)
    goal[0] = 1.5
    start = np.zeros(14)
    start[0] = -3.0
    s = setup((ArmWaypoint("long", tuple(goal)),))
    s.tick(start)
    first = s.gotos[0][0]
    assert -3 < first[0] < 1.5
    assert np.count_nonzero(first) == 1
    assert s.engine.segment["planned_duration_s"] <= 15
    for _ in range(310):
        s.tick(first)
        if len(s.gotos) > 1:
            break
    assert len(s.gotos) == 2
    assert s.engine.segment["planned_duration_s"] <= 15
    np.testing.assert_allclose(s.gotos[-1][0], goal)


def test_forward_outward_preserves_yaw_and_wrists_but_monitors_them():
    s = setup(build_collision_aware_waypoints(np.zeros(14)))
    s.tick(0.2)
    indices = [2, 4, 5, 6, 9, 11, 12, 13]
    np.testing.assert_allclose(s.engine.targets[2][indices], 0.2)
    assert "home" not in " ".join(w.name for w in s.engine.waypoints)


def test_sdk_follower_is_opt_in_and_joint_follower_returns_no_chunk():
    s = setup()
    skill = CollisionAwareArmPreMotionSkill((0.5,) * 14, waypoint_profile="direct")
    assert not skill.owns_operator_view
    skill.configure_boundary_preparation(
        sink=s.sink, settings=s.cfg, record_hold=s.holds.append,
        state_getter=lambda: s.sample.current, console=s.console, clock=lambda: s.clock.now,
    )
    skill.start({})
    s.sample.current = SimpleNamespace(position=np.zeros(29), velocity=np.zeros(29), received_monotonic_ns=int(s.clock.now * 1e9))
    assert skill.step({"joint_state": s.sample.current}) is None
    skill.stop()
    assert len(s.gotos) == 2


@pytest.mark.parametrize("stage", range(6))
def test_all_stage_head_poses_with_lagged_tracking(stage):
    from inference.desktop.lower_policy.initial_pose import initial_pose_from_config
    from inference.desktop.orchestrator import STAGE_HEAD_SKILL
    config = yaml.safe_load((Path(__file__).parents[2] / "configs/skill_config.yaml").read_text())
    final = initial_pose_from_config(config, STAGE_HEAD_SKILL[stage]).arm_position_rad
    s = setup(build_collision_aware_waypoints(final))
    position = np.zeros(14)
    s.tick(position)
    for _ in range(3000):
        # First-order tracking lag with a bounded velocity, not immediate echo.
        target = s.gotos[-1][0]
        delta = np.clip((target - position) * 0.2, -0.015, 0.015)
        position += delta
        s.tick(position, speed=float(np.max(np.abs(delta))) / 0.05)
        if s.engine.complete:
            break
    assert s.engine.complete, (stage, s.engine.phase, s.engine.index)
    assert len(s.gotos) == 4
    np.testing.assert_allclose(position, final, atol=0.1)
    assert s.transport.chunks == []


def test_gravity_is_used_only_on_wire_and_arrival_uses_model_intent():
    s = setup()
    s.sink._gravity_offset = SimpleNamespace(apply=lambda q: (q + 0.12, np.full(14, 0.12)))
    s.tick()
    np.testing.assert_allclose(s.gotos[0][0], 0.62)
    for _ in range(60):
        s.tick(0.5)
        if s.engine.complete:
            break
    assert s.engine.complete
    np.testing.assert_allclose(s.holds[-1], 0.5)


def test_preserved_joint_drift_is_not_ignored():
    s = setup(build_collision_aware_waypoints(np.zeros(14)))
    s.tick()
    actual = s.engine.targets[0].copy()
    actual[5] += 0.12
    for _ in range(200):
        s.tick(actual)
    assert s.engine.phase == "holding" and s.engine.index == 0


def test_venue_model_pose_above_smoke_limit_is_not_clipped():
    goal = np.zeros(14)
    goal[0] = 2.5
    s = setup((ArmWaypoint("model_pose", tuple(goal)),))
    s.tick()
    np.testing.assert_array_equal(s.engine.targets[0], goal)
    goal[0] = 3.0
    s = setup((ArmWaypoint("invalid_model_pose", tuple(goal)),))
    with pytest.raises(ValueError, match="URDF joint limits"):
        s.tick()
    assert s.gotos == []


def test_dwell_counts_measurement_time_not_a_delayed_freshness_stamp():
    """A PC2-guard reply delayed in transit gets an early freshness stamp (request time
    minus source age). Its gap to the next sample must not count as stillness: two
    samples 40 ms apart are not 0.3 s of dwell (Codex review of #185)."""
    s = setup()
    s.tick()
    for _ in range(80):  # wait out the planned trajectory away from the target
        s.tick(0.0)
    assert not s.engine.complete and s.engine.phase == "moving"

    def deliver(received_s, measured_s):
        s.sample.current = SimpleNamespace(
            position=np.r_[np.zeros(15), np.full(14, 0.5)],
            velocity=np.zeros(29),
            received_monotonic_ns=int(round(received_s * 1e9)),
            measured_monotonic_ns=int(round(measured_s * 1e9)),
        )
        s.engine.step({"joint_state": s.sample.current})

    now = s.clock.now
    deliver(now - 0.300, now)          # reply delayed 300 ms: freshness stamp early
    s.clock.now += 0.040
    deliver(now + 0.040, now + 0.040)  # next reply on time, 40 ms of measurement
    assert not s.engine.complete, "a delayed stamp satisfied the 0.3 s dwell"
    for step in range(1, 9):
        s.clock.now += 0.040
        deliver(s.clock.now, now + 0.040 * (step + 1))
    assert s.engine.complete
