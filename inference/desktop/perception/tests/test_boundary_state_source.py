from __future__ import annotations

import time

import numpy as np
import pytest

from inference.desktop.perception.boundary_state_source import (
    BoundaryDex1StateSource,
    BoundaryJointStateSource,
    parse_gripper_q,
)


class _State:
    body_q = np.arange(29, dtype=np.float32)
    base_quat = np.asarray([0.9, 0.1, 0.2, 0.3], dtype=np.float32)
    gripper_q = None


class _FakeStream:
    def __init__(self, state=None):
        self.closed = False
        self._state = state if state is not None else _State()
        self._sent = False

    def read(self, timeout_ms=200):
        del timeout_ms
        if not self._sent:
            self._sent = True
            return self._state
        time.sleep(0.005)
        return None

    def close(self):
        self.closed = True


def _wait(fn, timeout=1.0):
    deadline = time.monotonic() + timeout
    value = fn()
    while value is None and time.monotonic() < deadline:
        time.sleep(0.005)
        value = fn()
    return value


def test_boundary_state_is_exposed_in_canonical_order():
    stream = _FakeStream()
    source = BoundaryJointStateSource(host="robot", port=5557, stream=stream)
    snapshot = _wait(source.get)
    assert snapshot is not None
    assert snapshot.position.tolist() == list(range(29))
    assert snapshot.velocity.shape == (29,)
    assert np.allclose(snapshot.base_quat_wxyz, _State.base_quat)
    assert snapshot.received_monotonic_ns is not None
    assert source.latest_gripper() is None
    source.close()
    assert stream.closed


def test_gripper_q_is_kept_with_its_receive_time():
    state = _State()
    state.gripper_q = np.asarray([-0.5, -4.0])
    source = BoundaryJointStateSource(stream=_FakeStream(state))
    sample = _wait(source.latest_gripper)
    assert sample is not None
    np.testing.assert_allclose(sample[0], [-0.5, -4.0])
    assert sample[1] > 0
    source.close()


@pytest.mark.parametrize(
    "message, expected",
    [
        ({"gripper_q": {"left": {"q": -1.0}, "right": {"q": -2.5}}}, [-1.0, -2.5]),
        ({"gripper_q": None}, None),
        ({}, None),
        ({"gripper_q": {"left": {"q": -1.0}}}, None),
        ({"gripper_q": {"left": {"q": float("nan")}, "right": {"q": 0.0}}}, None),
        ("not a dict", None),
    ],
)
def test_parse_gripper_q(message, expected):
    got = parse_gripper_q(message)
    if expected is None:
        assert got is None
    else:
        np.testing.assert_allclose(got, expected)


@pytest.mark.requires_boundary
def test_official_decoder_still_validates_and_gripper_q_is_extracted():
    """公式 StateStream._decode の検証を通したうえで gripper_q を拾う。"""
    msgpack = pytest.importorskip("msgpack")
    pytest.importorskip("zmq")
    from inference.desktop.perception.boundary_state_source import (
        _gripper_aware_stream,
    )

    stream = _gripper_aware_stream("127.0.0.1", 59157)
    try:
        payload = {
            "body_q": list(range(29)),
            "base_quat": [1.0, 0.0, 0.0, 0.0],
            "gripper_q": {
                "left": {"q": -0.25, "dq": 0.0, "tau_est": 0.0},
                "right": {"q": -4.5, "dq": 0.0, "tau_est": 0.0},
            },
        }
        state = stream._decode(b"g1_debug" + msgpack.packb(payload))
        np.testing.assert_allclose(state.gripper_q, [-0.25, -4.5])
        assert state.body_q.shape == (29,)

        bad = dict(payload, body_q=list(range(28)))
        with pytest.raises(ValueError):
            stream._decode(b"g1_debug" + msgpack.packb(bad))
    finally:
        stream.close()


@pytest.mark.requires_boundary
def test_official_decoder_path_without_the_wire_libraries(monkeypatch):
    """zmq / msgpack が無い env でも、公式 _decode → gripper_q 抽出の順を確かめる。"""
    import importlib
    import json
    import sys
    import types

    fake_msgpack = types.ModuleType("msgpack")
    fake_msgpack.unpackb = lambda data, raw=False: json.loads(data.decode())
    fake_msgpack.packb = lambda obj, **_: json.dumps(obj).encode()
    monkeypatch.setitem(sys.modules, "msgpack", fake_msgpack)
    monkeypatch.setitem(sys.modules, "zmq", types.ModuleType("zmq"))
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    for name in [m for m in sys.modules if m.startswith("inference.desktop.boundary")]:
        monkeypatch.delitem(sys.modules, name)
    states = importlib.import_module("inference.desktop.boundary.states")
    from inference.desktop.perception.boundary_state_source import (
        gripper_aware_stream_class,
    )

    cls = gripper_aware_stream_class()
    payload = {
        "body_q": list(range(29)),
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "gripper_q": {"left": {"q": -0.25}, "right": {"q": -4.5}},
    }
    state = cls._decode(None, b"g1_debug" + json.dumps(payload).encode())
    np.testing.assert_allclose(state.gripper_q, [-0.25, -4.5])
    with pytest.raises(states.StateError):
        cls._decode(
            None,
            b"g1_debug" + json.dumps(dict(payload, body_q=[0.0] * 28)).encode(),
        )


class _Joint:
    def __init__(self, sample=None):
        self.sample = sample

    def latest_gripper(self):
        return self.sample


class _Synthetic:
    def __init__(self):
        self.bound = None
        self.closed = False

    def bind_command_source(self, source):
        self.bound = source

    def get(self):
        from inference.desktop.perception.dex1_state_source import Dex1StateData

        return Dex1StateData(
            position_rad=np.asarray([4.5, 4.5]),
            left_received_monotonic_ns=1,
            right_received_monotonic_ns=1,
            t=1,
            measured=False,
        )

    def close(self):
        self.closed = True


def test_dex1_falls_back_to_the_synthetic_echo_without_gripper_q():
    synthetic = _Synthetic()
    source = BoundaryDex1StateSource(_Joint(None), synthetic)
    state = source.get()
    assert state.measured is False
    assert not source.measured_available
    source.bind_command_source("actuator")
    assert synthetic.bound == "actuator"
    source.close()
    assert synthetic.closed


def test_dex1_uses_measured_gripper_q_as_the_physical_opening():
    """運営の写像は motor q = -(開き)。model 座標の開き [rad] に戻す。"""
    joint = _Joint((np.asarray([-2.3, -0.1]), 123))
    source = BoundaryDex1StateSource(joint, _Synthetic())
    state = source.get()
    assert state.measured is True
    np.testing.assert_allclose(state.position_rad, [2.3, 0.1])
    assert state.left_received_monotonic_ns == 123
    assert source.measured_available
