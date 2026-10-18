"""Read-only sustained four-camera preflight for physical evaluations."""

from __future__ import annotations

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--minimum-hz", type=float, default=25.0)
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    from evaluate.model_evaluation.runners.run_skill import (
        DEFAULT_HEAD_TOPIC,
        DEFAULT_WRIST_L_TOPIC,
        DEFAULT_WRIST_R_TOPIC,
        _collect_obs,
        _validate_camera_stream_window,
    )
    from inference.desktop.perception.frame_source import Ros2FrameSource

    ChannelFactoryInitialize(0, args.interface)
    sources = [
        Ros2FrameSource(topic=DEFAULT_HEAD_TOPIC, stereo_view="packed"),
        Ros2FrameSource(topic=DEFAULT_WRIST_L_TOPIC),
        Ros2FrameSource(topic=DEFAULT_WRIST_R_TOPIC),
    ]

    def collect():
        return _collect_obs(None, None, *sources)

    print(
        "[read-only] waiting for head stereo + bilateral wrist streams; "
        "no robot command is sent"
    )
    try:
        deadline = time.monotonic() + 15.0
        missing: list[str] = []
        while time.monotonic() < deadline:
            observation = collect()
            frames = observation.get("_camera_frames", {})
            missing = [
                role
                for role in ("head_left", "head_right", "left_wrist", "right_wrist")
                if role not in frames
            ]
            if not missing:
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(
                "four-camera warmup did not complete within 15s; "
                f"missing={','.join(missing)}"
            )
        rates = _validate_camera_stream_window(
            collect,
            require_wrist=True,
            duration_s=args.duration_seconds,
            minimum_hz=args.minimum_hz,
        )
    finally:
        for source in reversed(sources):
            source.close()
    print(
        "PASS: sustained four-camera validation completed: "
        + " ".join(f"{role}={rate:.2f}Hz" for role, rate in rates.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
