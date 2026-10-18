#!/usr/bin/env python3
"""Fail the read-only camera bringup when a required stream stops updating.

The RealSense ROS driver can remain alive after the kernel has returned
ENODEV for a disconnected D405.  In that state systemd sees the launch as
healthy and therefore never applies Restart=on-failure.  This watchdog is
part of the camera-only bringup: it observes compressed image topics and
exits non-zero when one is missing or stale.  A launch OnProcessExit handler
then shuts down the camera container so systemd can restart it.

The watchdog intentionally subscribes to the small ``camera_info`` messages
that are emitted for each captured frame.  Subscribing to three JPEG topics a
second time doubles DDS serialization/deserialization work and can reduce the
Desktop image rate on Orin.  CameraInfo preserves frame liveness without
copying image payloads.  This node never publishes a robot command.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class CameraLiveness:
    """Monotonic-time health state, kept independent of ROS for unit tests."""

    roles: tuple[str, ...]
    startup_grace_s: float
    stale_timeout_s: float
    started_s: float
    last_frame_s: dict[str, float | None] = field(init=False)

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError("at least one camera role is required")
        if self.startup_grace_s <= 0.0 or self.stale_timeout_s <= 0.0:
            raise ValueError("camera watchdog timeouts must be positive")
        self.last_frame_s = {role: None for role in self.roles}

    def observe(self, role: str, now_s: float) -> None:
        if role not in self.last_frame_s:
            raise KeyError(role)
        self.last_frame_s[role] = float(now_s)

    def failure(self, now_s: float) -> str | None:
        missing = [role for role, stamp in self.last_frame_s.items() if stamp is None]
        if missing:
            age_s = float(now_s) - self.started_s
            if age_s >= self.startup_grace_s:
                return (
                    "required camera streams never became ready within "
                    f"{self.startup_grace_s:.1f}s: {','.join(missing)}"
                )
            return None

        stale = {
            role: float(now_s) - float(stamp)
            for role, stamp in self.last_frame_s.items()
            if float(now_s) - float(stamp) >= self.stale_timeout_s
        }
        if not stale:
            return None
        details = ",".join(f"{role}={age_s:.3f}s" for role, age_s in stale.items())
        return f"required camera stream stale: {details}"


def main(args: list[str] | None = None) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo

    rclpy.init(args=args)
    node = Node("camera_stream_watchdog")
    node.declare_parameter("startup_grace_s", 20.0)
    node.declare_parameter("stale_timeout_s", 2.0)
    node.declare_parameter(
        "roles",
        ["head_stereo", "wrist_left", "wrist_right"],
    )
    node.declare_parameter(
        "topics",
        [
            "/head/camera/color/camera_info",
            "/wrist_left/camera/color/camera_info",
            "/wrist_right/camera/color/camera_info",
        ],
    )

    roles = tuple(str(value) for value in node.get_parameter("roles").value)
    topics = tuple(str(value) for value in node.get_parameter("topics").value)
    if len(roles) != len(topics) or not roles:
        node.get_logger().fatal(
            "camera watchdog roles/topics must be non-empty and have equal length"
        )
        node.destroy_node()
        rclpy.shutdown()
        return 2

    health = CameraLiveness(
        roles=roles,
        startup_grace_s=float(node.get_parameter("startup_grace_s").value),
        stale_timeout_s=float(node.get_parameter("stale_timeout_s").value),
        started_s=time.monotonic(),
    )
    subscriptions = []
    for role, topic in zip(roles, topics):
        subscriptions.append(
            node.create_subscription(
                CameraInfo,
                topic,
                lambda _message, camera_role=role: health.observe(
                    camera_role, time.monotonic()
                ),
                qos_profile_sensor_data,
            )
        )

    node.get_logger().info(
        "read-only camera watchdog armed: "
        f"roles={list(roles)}, startup_grace={health.startup_grace_s:.1f}s, "
        f"stale_timeout={health.stale_timeout_s:.1f}s"
    )
    exit_code = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            failure = health.failure(time.monotonic())
            if failure is not None:
                node.get_logger().fatal(failure)
                exit_code = 2
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Keep subscription objects alive until after the spin loop.
        subscriptions.clear()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
