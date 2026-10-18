from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from inference.desktop.perception.dex1_state_source import (
    Dex1StateSource,
    SyntheticDex1StateSource,
)


class _FakeSubscriber:
    instances: dict[str, "_FakeSubscriber"] = {}

    def __init__(self, topic: str, message_type: object) -> None:
        self.topic = topic
        self.message_type = message_type
        self.callback = None
        self.closed = False
        self.instances[topic] = self

    def Init(self, callback, queue_len: int) -> None:  # noqa: N802, ANN001
        assert queue_len == 1
        self.callback = callback

    def Close(self) -> None:  # noqa: N802
        self.closed = True

    def emit(
        self,
        q: float,
        *,
        mode: int = 0,
        temperature: int = 0,
        reserve: tuple[int, int] = (0, 0),
    ) -> None:
        assert self.callback is not None
        self.callback(
            SimpleNamespace(
                states=[
                    SimpleNamespace(
                        q=q,
                        mode=mode,
                        temperature=temperature,
                        reserve=reserve,
                    )
                ]
            )
        )


def _make_source() -> Dex1StateSource:
    _FakeSubscriber.instances.clear()
    return Dex1StateSource(
        subscriber_factory=_FakeSubscriber,
        message_type=object(),
    )


def test_waits_for_atomic_bilateral_sample() -> None:
    source = _make_source()
    _FakeSubscriber.instances["rt/dex1/left/state"].emit(4.2)
    assert source.get() is None
    _FakeSubscriber.instances["rt/dex1/right/state"].emit(1.3)
    snapshot = source.get()
    assert snapshot is not None
    np.testing.assert_allclose(snapshot.position_rad, [4.2, 1.3])
    assert snapshot.left_received_monotonic_ns > 0
    assert snapshot.right_received_monotonic_ns > 0
    assert not snapshot.health_diagnostics_available
    np.testing.assert_array_equal(snapshot.fault_code, [0, 0])


def test_decodes_versioned_bridge_health_diagnostics() -> None:
    from inference.desktop.perception.dex1_state_source import (
        DEX1_HEALTH_PROTOCOL_MAGIC,
    )

    source = _make_source()
    left = _FakeSubscriber.instances["rt/dex1/left/state"]
    right = _FakeSubscriber.instances["rt/dex1/right/state"]
    left.emit(
        5.3,
        mode=1,
        temperature=33,
        reserve=(0, DEX1_HEALTH_PROTOCOL_MAGIC),
    )
    right.emit(
        3.2,
        mode=0,
        temperature=34,
        reserve=(0x200, DEX1_HEALTH_PROTOCOL_MAGIC),
    )
    snapshot = source.get()
    assert snapshot is not None
    assert snapshot.health_diagnostics_available
    np.testing.assert_array_equal(snapshot.motor_mode, [1, 0])
    np.testing.assert_array_equal(snapshot.shell_temperature_c, [33, 34])
    np.testing.assert_array_equal(snapshot.fault_code, [0, 0x200])


def test_invalid_sample_does_not_replace_last_valid_pair() -> None:
    source = _make_source()
    left = _FakeSubscriber.instances["rt/dex1/left/state"]
    right = _FakeSubscriber.instances["rt/dex1/right/state"]
    left.emit(3.0)
    right.emit(4.0)
    before = source.get()
    left.emit(float("nan"))
    assert source.get() is before


def test_close_is_idempotent_and_closes_both_readers() -> None:
    source = _make_source()
    left = _FakeSubscriber.instances["rt/dex1/left/state"]
    right = _FakeSubscriber.instances["rt/dex1/right/state"]
    source.close()
    source.close()
    assert left.closed
    assert right.closed
    assert source.get() is None


# =========================================================================
# SyntheticDex1StateSource (会場のリグは Dex1 state を配信しないため)
# =========================================================================
class _FakeHandActuator:
    """MockHandActuator の `latest` 契約だけを満たす最小 fake。"""

    def __init__(self) -> None:
        self.latest: tuple[float, float] | None = None


def test_synthetic_seeds_from_the_skill_frame0_value() -> None:
    """指令が無いうちは seed した frame-0 値を返す (定数 4.5 固定にしない)。"""
    source = SyntheticDex1StateSource((1.2, 0.8))
    sample = source.get()
    assert sample is not None
    np.testing.assert_allclose(sample.position_rad, [1.2, 0.8])


def test_synthetic_never_returns_none_before_any_command() -> None:
    """実 source と違い「まだ来ていない」状態が無いので preflight を素通りする。"""
    assert SyntheticDex1StateSource((4.5, 4.5)).get() is not None


def test_synthetic_echoes_the_commanded_value() -> None:
    source = SyntheticDex1StateSource((4.5, 4.5))
    source.update((0.3, 0.4))
    np.testing.assert_allclose(source.get().position_rad, [0.3, 0.4])


def test_synthetic_follows_the_bound_actuator() -> None:
    """繋いだ actuator の直近指令を毎回読み直す (skill をまたいで追従する)。"""
    actuator = _FakeHandActuator()
    source = SyntheticDex1StateSource((4.5, 4.5), command_source=actuator)

    np.testing.assert_allclose(source.get().position_rad, [4.5, 4.5])  # 指令前は seed
    actuator.latest = (2.0, 2.1)
    np.testing.assert_allclose(source.get().position_rad, [2.0, 2.1])
    actuator.latest = (0.0, 0.1)
    np.testing.assert_allclose(source.get().position_rad, [0.0, 0.1])


def test_bound_actuator_wins_over_a_stale_update() -> None:
    """actuator の方が live なので、古い update() には引きずられない。"""
    actuator = _FakeHandActuator()
    source = SyntheticDex1StateSource((4.5, 4.5))
    source.update((1.0, 1.0))
    source.bind_command_source(actuator)
    actuator.latest = (3.0, 3.0)
    np.testing.assert_allclose(source.get().position_rad, [3.0, 3.0])


def test_synthetic_clamps_to_the_hardware_range() -> None:
    source = SyntheticDex1StateSource((4.5, 4.5))
    source.update((99.0, -5.0))
    np.testing.assert_allclose(source.get().position_rad, [5.4, 0.0])


def test_synthetic_rejects_malformed_seeds() -> None:
    for bad in [(1.0,), (1.0, 2.0, 3.0)]:
        with pytest.raises(ValueError, match="length 2"):
            SyntheticDex1StateSource(bad)
    with pytest.raises(ValueError, match="finite"):
        SyntheticDex1StateSource((float("nan"), 1.0))


def test_synthetic_close_is_idempotent() -> None:
    source = SyntheticDex1StateSource((4.5, 4.5))
    source.close()
    source.close()
    assert source.get() is None
