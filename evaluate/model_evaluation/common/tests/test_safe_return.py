from __future__ import annotations

import numpy as np

from evaluate.model_evaluation.common.safe_return import (
    BoundedArmWaypointFollower,
    ReturnArmWaypoint,
    build_return_arm_waypoints,
)
from inference.desktop.lower_policy.skills.collision_aware_pre_motion import (
    build_collision_aware_waypoints,
)


def test_return_path_is_exact_resolved_startup_reverse() -> None:
    measured = np.array(
        [0.2, 0.1, -0.1, 0.4, 0.2, -0.1, 0.0] * 2,
        dtype=np.float64,
    )
    policy_initial = np.linspace(-0.3, 0.3, 14)
    startup = build_collision_aware_waypoints(policy_initial)
    resolved = [waypoint.resolve(measured) for waypoint in startup]

    returned = build_return_arm_waypoints(measured, policy_initial)

    assert [waypoint.name for waypoint in returned] == [
        "return_policy_initial_pose",
        "return_forward_outward_clearance",
        "return_lateral_high_clearance",
        "return_shoulder_pitch_backward_clearance",
        "return_measured_initial_pose",
    ]
    for actual, expected in zip(returned[:-1], reversed(resolved), strict=True):
        np.testing.assert_allclose(actual.as_array(), expected)
    np.testing.assert_allclose(returned[-1].as_array(), measured)


def test_follower_respects_velocity_and_acceleration_limits() -> None:
    goal = np.full(14, 0.20, dtype=np.float64)
    follower = BoundedArmWaypointFollower(
        (ReturnArmWaypoint("small_move", tuple(goal)),),
        velocity_limit_rad_s=0.5,
        acceleration_limit_rad_s2=1.0,
        measured_tolerance_rad=0.01,
        stage_timeout_s=5.0,
        stable_samples_required=2,
    )
    dt = 0.02
    now = 0.0
    measured = np.zeros(14, dtype=np.float64)
    previous = follower.start(measured, now)
    previous_velocity = np.zeros(14, dtype=np.float64)
    for _ in range(300):
        now += dt
        command = follower.step(measured, now)
        velocity = (command - previous) / dt
        assert np.max(np.abs(velocity)) <= 0.5 + 1e-9
        assert np.max(np.abs(velocity - previous_velocity)) <= 1.0 * dt + 1e-8
        measured = command.copy()
        previous = command
        previous_velocity = velocity
        if follower.complete:
            break

    assert follower.complete
    np.testing.assert_allclose(previous, goal, atol=1e-3)


def test_waypoint_specific_measured_tolerance_matches_issue70_return() -> None:
    goal = np.zeros(14, dtype=np.float64)
    measured = goal.copy()
    measured[4] = 0.19
    follower = BoundedArmWaypointFollower(
        (
            ReturnArmWaypoint(
                "return_forward_outward_clearance", tuple(goal)
            ),
        ),
        measured_tolerance_rad=0.10,
        measured_tolerance_overrides={
            "return_forward_outward_clearance": 0.20
        },
        stable_samples_required=1,
    )
    follower.start(goal, 0.0)
    follower.step(measured, 0.02)
    assert follower.complete
