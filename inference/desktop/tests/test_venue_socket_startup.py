"""Loopback-only venue wire startup test; never touches a robot or DDS."""

from __future__ import annotations

import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
msgpack = pytest.importorskip("msgpack")
zmq = pytest.importorskip("zmq")
# 運営の interface package。公開用 repo には無い (Issue #190、conftest.py)
pytest.importorskip("inference.desktop.boundary")

from inference.desktop.boundary.actions import DecoupledSink, TASKSPACE_TOPIC
from inference.desktop.perception.boundary_state_source import BoundaryJointStateSource
from inference.desktop.perception.frame_source import ZmqFrameSource


def test_real_venue_sockets_start_and_exchange_contract_messages() -> None:
    context = zmq.Context.instance()
    camera_pub = context.socket(zmq.PUB)
    state_pub = context.socket(zmq.PUB)
    camera_port = camera_pub.bind_to_random_port("tcp://127.0.0.1")
    state_port = state_pub.bind_to_random_port("tcp://127.0.0.1")
    action_port = None
    camera = state = action = action_sub = None
    wrists = []
    try:
        camera = ZmqFrameSource(
            f"tcp://127.0.0.1:{camera_port}", rcvtimeo_ms=100
        )
        wrists = [
            ZmqFrameSource(
                f"tcp://127.0.0.1:{camera_port}", rcvtimeo_ms=100,
                stereo_view="single", image_key=key,
            )
            for key in ("left_wrist", "right_wrist")
        ]
        state = BoundaryJointStateSource(host="127.0.0.1", port=state_port)
        action = DecoupledSink(port=0, host="127.0.0.1")
        action_port = action._socket.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[1]
        action_sub = context.socket(zmq.SUB)
        action_sub.setsockopt(zmq.SUBSCRIBE, TASKSPACE_TOPIC)
        action_sub.setsockopt(zmq.LINGER, 0)
        action_sub.connect(f"tcp://127.0.0.1:{action_port}")

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 1] = 80
        encoded = cv2.imencode(".jpg", image)[1].tobytes()
        left_wrist = cv2.imencode(".jpg", np.full_like(image, 30))[1].tobytes()
        right_wrist = cv2.imencode(".jpg", np.full_like(image, 160))[1].tobytes()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            now = time.time()
            camera_pub.send(msgpack.packb({
                "timestamps": {"ego_view_left": now, "ego_view_right": now},
                "images": {
                    "ego_view_left": encoded, "ego_view_right": encoded,
                    "left_wrist": left_wrist, "right_wrist": right_wrist,
                },
            }, use_bin_type=True))
            state_pub.send(b"g1_debug" + msgpack.packb({
                "body_q": [0.0] * 29,
                "base_quat": [1.0, 0.0, 0.0, 0.0],
                "gripper_q": {"left": {"q": -5.3}, "right": {"q": -5.3}},
            }, use_bin_type=True))
            if camera.get() is not None and state.get() is not None and all(w.get() is not None for w in wrists):
                break
            time.sleep(0.05)
        assert camera.get() is not None, "camera socket did not receive stereo frames"
        assert camera.get().rgb.shape == (480, 1280, 3)
        for wrist, value in zip(wrists, (30, 160)):
            assert wrist.get() is not None
            assert wrist.get().rgb.shape == (480, 640, 3)
            assert np.mean(wrist.get().rgb) == pytest.approx(value, abs=2)
        assert state.get() is not None, "state socket did not receive 29D body state"
        assert state.get().position.shape == (29,)
        assert state.latest_gripper() is not None

        row = np.zeros((1, 25), dtype=np.float32)
        row[0, 7] = 1.0
        row[0, 14] = 1.0
        row[0, 21] = 0.74
        received = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            action.send_chunk(row)
            if action_sub.poll(50):
                received = action_sub.recv()
                break
        assert received is not None, "action subscriber did not receive taskspace"
        payload = msgpack.unpackb(received[len(TASKSPACE_TOPIC):], raw=False)
        assert payload["shape"] == [1, 25]
        assert np.isclose(np.frombuffer(payload["actions"], dtype=np.float32)[21], 0.74)
        assert abs(time.time() - payload["issued_at"]) < 1.0
    finally:
        if camera is not None:
            camera.close()
        for wrist in wrists:
            wrist.close()
        if state is not None:
            state.close()
        if action is not None:
            action.close()
        if action_sub is not None:
            action_sub.close(linger=0)
        camera_pub.close(linger=0)
        state_pub.close(linger=0)
