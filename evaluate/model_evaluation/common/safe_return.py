"""Evaluation-only arm/Dex1 return helpers.

This module deliberately stays under :mod:`evaluate`: the production
orchestrator and policy implementations are not changed by model-evaluation
shutdown behavior.  The arm path is the exact reverse of the collision-aware
startup path that the shared inference skill resolves from the measured
pre-takeover pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from inference.desktop.lower_policy.skills.collision_aware_pre_motion import (
    build_collision_aware_waypoints,
)


ARM_DOF = 14


def _arm_vector(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (ARM_DOF,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite 14-D")
    return result


@dataclass(frozen=True)
class ReturnArmWaypoint:
    name: str
    target_rad: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        return _arm_vector(self.target_rad, f"waypoint {self.name!r}")


def build_return_arm_waypoints(
    measured_initial_arm_rad: Sequence[float],
    policy_initial_arm_rad: Sequence[float],
) -> tuple[ReturnArmWaypoint, ...]:
    """Resolve once and reverse the exact arm startup path.

    The policy may finish far from its training frame-zero pose, so frame zero
    is restored first.  The forward-outward, lateral-high, and
    shoulder-backward clearance poses are then traversed in reverse before the
    measured lowered pose captured before arm-sdk takeover is restored.
    """

    measured = _arm_vector(measured_initial_arm_rad, "measured initial arm pose")
    policy_initial = _arm_vector(policy_initial_arm_rad, "policy initial arm pose")
    startup = build_collision_aware_waypoints(policy_initial)
    resolved = tuple(waypoint.resolve(measured) for waypoint in startup)
    return tuple(
        ReturnArmWaypoint(f"return_{waypoint.name}", tuple(target.tolist()))
        for waypoint, target in zip(reversed(startup), reversed(resolved), strict=True)
    ) + (
        ReturnArmWaypoint(
            "return_measured_initial_pose", tuple(measured.tolist())
        ),
    )


class BoundedArmWaypointFollower:
    """Measured-convergence waypoint follower with velocity/acceleration caps."""

    def __init__(
        self,
        waypoints: Sequence[ReturnArmWaypoint],
        *,
        velocity_limit_rad_s: float = 0.5,
        acceleration_limit_rad_s2: float = 1.0,
        measured_tolerance_rad: float = 0.10,
        measured_tolerance_overrides: Mapping[str, float] | None = None,
        stage_timeout_s: float = 15.0,
        stable_samples_required: int = 5,
    ) -> None:
        self.waypoints = tuple(waypoints)
        if not self.waypoints:
            raise ValueError("return waypoint sequence must not be empty")
        for waypoint in self.waypoints:
            waypoint.as_array()
        for label, value in (
            ("velocity_limit_rad_s", velocity_limit_rad_s),
            ("acceleration_limit_rad_s2", acceleration_limit_rad_s2),
            ("measured_tolerance_rad", measured_tolerance_rad),
            ("stage_timeout_s", stage_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
        if stable_samples_required < 1:
            raise ValueError("stable_samples_required must be positive")
        self.velocity_limit_rad_s = float(velocity_limit_rad_s)
        self.acceleration_limit_rad_s2 = float(acceleration_limit_rad_s2)
        self.measured_tolerance_rad = float(measured_tolerance_rad)
        self.measured_tolerance_overrides = dict(
            measured_tolerance_overrides or {}
        )
        known_names = {waypoint.name for waypoint in self.waypoints}
        unknown_names = set(self.measured_tolerance_overrides) - known_names
        if unknown_names:
            raise ValueError(
                "return tolerance override names are not in the waypoint set: "
                f"{sorted(unknown_names)}"
            )
        for name, tolerance in self.measured_tolerance_overrides.items():
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(
                    f"return tolerance override for {name!r} must be positive"
                )
        self.stage_timeout_s = float(stage_timeout_s)
        self.stable_samples_required = int(stable_samples_required)
        self.command_rad: np.ndarray | None = None
        self.velocity_rad_s = np.zeros(ARM_DOF, dtype=np.float64)
        self.stage_index = 0
        self.stage_started_at: float | None = None
        self.last_at: float | None = None
        self.stable_samples = 0
        self.complete = False

    @property
    def current_waypoint(self) -> ReturnArmWaypoint:
        index = min(self.stage_index, len(self.waypoints) - 1)
        return self.waypoints[index]

    def start(self, measured_arm_rad: Sequence[float], now: float) -> np.ndarray:
        measured = _arm_vector(measured_arm_rad, "measured arm pose")
        if not math.isfinite(now):
            raise ValueError("start time must be finite")
        self.command_rad = measured.copy()
        self.velocity_rad_s.fill(0.0)
        self.stage_index = 0
        self.stage_started_at = float(now)
        self.last_at = float(now)
        self.stable_samples = 0
        self.complete = False
        return self.command_rad.copy()

    def step(self, measured_arm_rad: Sequence[float], now: float) -> np.ndarray:
        measured = _arm_vector(measured_arm_rad, "measured arm pose")
        if self.command_rad is None or self.stage_started_at is None or self.last_at is None:
            return self.start(measured, now)
        if self.complete:
            return self.command_rad.copy()
        if not math.isfinite(now) or now < self.last_at:
            raise ValueError("waypoint time must be finite and monotonic")
        waypoint = self.current_waypoint
        goal = waypoint.as_array()
        if now - self.stage_started_at > self.stage_timeout_s:
            errors = np.abs(goal - measured)
            worst = int(np.argmax(errors))
            raise TimeoutError(
                f"return stage {waypoint.name!r} did not converge within "
                f"{self.stage_timeout_s:g}s (max_arm_error="
                f"{errors[worst]:.4f}rad, worst_joint={worst}, "
                f"target={goal[worst]:+.4f}rad, measured={measured[worst]:+.4f}rad)"
            )

        dt = min(max(float(now) - self.last_at, 1e-4), 0.1)
        error = goal - self.command_rad
        # Discrete stopping bound.  The continuous sqrt(2*a*d) bound can
        # overshoot during the final sample and then snap velocity to zero,
        # which itself violates the requested acceleration limit.  This bound
        # reserves one full ``v*dt`` step before the remaining braking distance:
        #     v*dt + v^2/(2a) <= distance.
        acceleration_step = self.acceleration_limit_rad_s2 * dt
        braking_speed = np.maximum(
            0.0,
            np.sqrt(
                acceleration_step**2
                + 2.0 * self.acceleration_limit_rad_s2 * np.abs(error)
            )
            - acceleration_step,
        )
        desired_velocity = np.sign(error) * np.minimum(
            self.velocity_limit_rad_s, braking_speed
        )
        velocity_delta = np.clip(
            desired_velocity - self.velocity_rad_s,
            -self.acceleration_limit_rad_s2 * dt,
            self.acceleration_limit_rad_s2 * dt,
        )
        self.velocity_rad_s += velocity_delta
        command_step = self.velocity_rad_s * dt
        # The discrete braking construction should make this a numerical-only
        # guard.  Keeping the commanded velocity consistent with the emitted
        # position step avoids a hidden acceleration discontinuity.
        overshoot = np.abs(command_step) > np.abs(error)
        command_step = np.where(overshoot, error, command_step)
        self.velocity_rad_s = command_step / dt
        self.command_rad += command_step
        self.last_at = float(now)

        command_error = float(np.max(np.abs(goal - self.command_rad)))
        measured_error = float(np.max(np.abs(goal - measured)))
        measured_tolerance = self.measured_tolerance_overrides.get(
            waypoint.name, self.measured_tolerance_rad
        )
        stable = (
            command_error <= 1e-3
            and measured_error <= measured_tolerance
            and float(np.max(np.abs(self.velocity_rad_s)))
            <= self.acceleration_limit_rad_s2 * dt + 1e-9
        )
        self.stable_samples = self.stable_samples + 1 if stable else 0
        if self.stable_samples >= self.stable_samples_required:
            self.stage_index += 1
            self.velocity_rad_s.fill(0.0)
            self.stable_samples = 0
            self.stage_started_at = float(now)
            if self.stage_index >= len(self.waypoints):
                self.complete = True
        return self.command_rad.copy()
