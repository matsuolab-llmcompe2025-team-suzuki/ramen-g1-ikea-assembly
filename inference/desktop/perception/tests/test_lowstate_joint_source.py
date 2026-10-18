"""LowStateJointSource の unit test (Issue #141 束 1-14 / INF-18)。

SDK 無しで回すため、subscriber は差し替え口から注入する。
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.perception.g1_urdf_fk import G1_JOINT_NAMES
from inference.desktop.perception.lowstate_joint_source import LowStateJointSource


class _Motor:
    def __init__(self, q: float) -> None:
        self.q = q
        self.dq = q * 0.1
        self.tau_est = q * 0.01


class _LowState:
    """SDK の LowState_ の必要な部分だけ (motor_state は 35 要素)。"""

    def __init__(self, tick: int, base: float = 0.0) -> None:
        self.tick = tick
        self.motor_state = [_Motor(base + i) for i in range(35)]


class _FakeSubscriber:
    """`Read()` が渡した列を順に返し、尽きたら None を返す。"""

    def __init__(self, topic: str, message_type: object) -> None:
        self.topic = topic
        self.messages: list[object] = []
        self.closed = False
        self.init_calls = 0

    def Init(self) -> None:
        self.init_calls += 1

    def Read(self):
        return self.messages.pop(0) if self.messages else None

    def Close(self) -> None:
        self.closed = True


@pytest.fixture
def source_and_sub():
    created: list[_FakeSubscriber] = []

    def factory(topic, message_type):
        sub = _FakeSubscriber(topic, message_type)
        created.append(sub)
        return sub

    source = LowStateJointSource(subscriber_factory=factory, message_type=object)
    return source, created[0]


class TestSnapshot:
    def test_returns_none_before_first_sample(self, source_and_sub):
        source, _ = source_and_sub
        assert source.get() is None

    def test_maps_first_29_motors_in_ros_joint_order(self, source_and_sub):
        source, sub = source_and_sub
        sub.messages.append(_LowState(tick=7))
        snapshot = source.get()
        assert snapshot.name == G1_JOINT_NAMES
        assert snapshot.position.shape == (29,)
        assert snapshot.position[0] == pytest.approx(0.0)
        assert snapshot.position[28] == pytest.approx(28.0)   # 29..34 は読まない
        assert snapshot.velocity[10] == pytest.approx(1.0)
        assert snapshot.effort[10] == pytest.approx(0.1)

    def test_keeps_tick_and_receive_time(self, source_and_sub):
        source, sub = source_and_sub
        sub.messages.append(_LowState(tick=1234))
        snapshot = source.get()
        assert snapshot.tick == 1234
        assert snapshot.received_monotonic_ns == snapshot.t > 0

    def test_holds_last_snapshot_when_nothing_new_arrived(self, source_and_sub):
        source, sub = source_and_sub
        sub.messages.append(_LowState(tick=1))
        first = source.get()
        assert source.get() is first          # Read() が None でも前の値を返す

    def test_newer_sample_replaces_the_previous_one(self, source_and_sub):
        source, sub = source_and_sub
        sub.messages.extend([_LowState(tick=1), _LowState(tick=2, base=100.0)])
        source.get()
        snapshot = source.get()
        assert snapshot.tick == 2
        assert snapshot.position[0] == pytest.approx(100.0)

    def test_broken_sample_keeps_the_previous_snapshot(self, source_and_sub, capsys):
        source, sub = source_and_sub
        sub.messages.append(_LowState(tick=1))
        good = source.get()
        broken = _LowState(tick=2)
        broken.motor_state = broken.motor_state[:3]      # 29 に足りない
        sub.messages.append(broken)
        assert source.get() is good
        assert "invalid lowstate sample" in capsys.readouterr().err


class TestLifecycle:
    def test_init_registers_the_reader_once(self, source_and_sub):
        _, sub = source_and_sub
        assert sub.init_calls == 1
        assert sub.topic == "rt/lowstate"

    def test_wait_for_first_returns_as_soon_as_a_sample_arrives(self, source_and_sub):
        source, sub = source_and_sub
        sub.messages.append(_LowState(tick=1))
        assert source.wait_for_first(timeout_s=0.1).tick == 1

    def test_wait_for_first_times_out_without_samples(self, source_and_sub):
        source, _ = source_and_sub
        with pytest.raises(RuntimeError, match="timed out"):
            source.wait_for_first(timeout_s=0.05)

    def test_close_is_idempotent(self, source_and_sub):
        source, sub = source_and_sub
        source.close()
        source.close()
        assert sub.closed
        assert source.get() is None
