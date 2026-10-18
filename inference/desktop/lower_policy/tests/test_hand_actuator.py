"""Hand actuator (MockHandActuator + G1HandActuator real 実装) の unit test
(Issue #125 Phase B-2/B-3)。

- MockHandActuator: shape/length + clamp + lifecycle
- G1HandActuator: DI 経路 (fake left/right publisher + fake thread + fake cmd
  factory) で send_action → 2 publisher write + lifecycle 検証。SDK-free でも走る
  (real DDS 経路は SDK 経由 lazy import で init される、DI 注入で回避)。
"""

from __future__ import annotations

from typing import Any

import pytest

from inference.desktop.lower_policy.actuators.hand import (
    G1_DEX1_CONTROL_FREQ_HZ,
    G1_DEX1_POSITION_KD,
    G1_DEX1_POSITION_KP,
    G1_NUM_HAND_JOINTS,
    HAND_GRIP_MAX,
    HAND_GRIP_MIN,
    G1HandActuator,
    MockHandActuator,
)


# ---- shared fake helpers ---- #


class _FakePublisher:
    def __init__(self) -> None:
        self.written: list[Any] = []

    def Write(self, msg: Any, timeout: float | None = None) -> bool:
        self.written.append(msg)
        return True


class _FailedPublisher(_FakePublisher):
    def Write(self, msg: Any, timeout: float | None = None) -> bool:
        self.written.append(msg)
        return False


class _FakeThread:
    def __init__(self) -> None:
        self.started = False
        self.waited = False

    def Start(self) -> None:
        self.started = True

    def Wait(self, timeout: float = 1.0) -> None:
        self.waited = True


# ---- constants ---- #


class TestConstants:
    def test_num_hand_joints(self) -> None:
        assert G1_NUM_HAND_JOINTS == 2

    def test_grip_range(self) -> None:
        assert HAND_GRIP_MIN == 0.0
        assert HAND_GRIP_MAX == 5.4

    def test_official_position_controller_parameters(self) -> None:
        assert G1_DEX1_CONTROL_FREQ_HZ == 200.0
        assert G1_DEX1_POSITION_KP == 5.0
        assert G1_DEX1_POSITION_KD == 0.05


# ---- MockHandActuator ---- #


class TestMockHandActuator:
    def test_send_action_stores_history(self) -> None:
        m = MockHandActuator()
        m.send_action([0.3, 0.7])
        assert m.latest == (0.3, 0.7)
        assert len(m.history) == 1
        m.send_action([0.5, 0.2])
        assert m.latest == (0.5, 0.2)
        assert len(m.history) == 2

    def test_reject_wrong_length(self) -> None:
        m = MockHandActuator()
        with pytest.raises(ValueError, match="length 2"):
            m.send_action([0.5])
        with pytest.raises(ValueError, match="length 2"):
            m.send_action([0.1, 0.2, 0.3])

    def test_clamp_out_of_range(self) -> None:
        # Phase B-2 追加: range [0, 1] clamp
        m = MockHandActuator()
        m.send_action([-0.5, 7.0])
        assert m.latest == (0.0, 5.4)  # physical-radian envelope

    def test_lifecycle_no_op(self) -> None:
        # Mock は start/stop no-op、既存 test との互換保つ
        m = MockHandActuator()
        assert m.started is False
        m.start()
        assert m.started is True
        m.stop()
        assert m.started is False


# ---- G1HandActuator (skeleton) ---- #


class _FakeMotorCmd:
    """MotorCmd_ 互換 (mode, q, dq, tau, kp, kd 属性)。"""

    def __init__(self):
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0


class _FakeMotorCmds:
    """MotorCmds_ 互換 (cmds sequence with 1 slot 初期化済)。"""

    def __init__(self):
        self.cmds = [_FakeMotorCmd()]


def _fake_cmd_factory():
    return _FakeMotorCmds()


class TestG1HandActuatorInitAndSend:
    def _make(self):
        lp = _FakePublisher()
        rp = _FakePublisher()
        thread = _FakeThread()
        act = G1HandActuator(
            left_publisher=lp,
            right_publisher=rp,
            publish_thread=thread,
            cmd_factory=_fake_cmd_factory,
        )
        return act, lp, rp, thread

    def test_send_action_stores_target(self) -> None:
        act, _, _, _ = self._make()
        act.send_action([0.3, 0.7])
        assert act._target.received
        assert act._target.positions == (0.3, 0.7)

    def test_reject_wrong_length(self) -> None:
        act, _, _, _ = self._make()
        with pytest.raises(ValueError, match="length 2"):
            act.send_action([0.1])

    def test_clamp_out_of_range(self) -> None:
        act, _, _, _ = self._make()
        act.send_action([-0.3, 7.0])
        assert act._target.positions == (0.0, 5.4)

    def test_prime_hold_seeds_measured_pose(self) -> None:
        act, _, _, _ = self._make()
        act.prime_hold([5.2, 0.1])
        assert act._target.positions == (5.2, 0.1)
        assert act._commanded_positions == (5.2, 0.1)

    def test_reject_nonpositive_velocity_limit(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            G1HandActuator(velocity_limit_rad_s=0.0)


class TestG1HandActuatorLifecycle:
    """DI 経路の start/stop + real path publish (MotorCmds_ に相当する Fake) の smoke。"""

    def _make(self):
        lp = _FakePublisher()
        rp = _FakePublisher()
        thread = _FakeThread()
        act = G1HandActuator(
            left_publisher=lp,
            right_publisher=rp,
            publish_thread=thread,
            cmd_factory=_fake_cmd_factory,
        )
        return act, lp, rp, thread

    def test_start_calls_thread_start(self) -> None:
        act, _, _, thread = self._make()
        act.start()
        assert act._running is True
        assert thread.started is True

    def test_start_is_idempotent(self) -> None:
        act, _, _, thread = self._make()
        act.start()
        act.start()  # 2nd start → no-op
        assert thread.started is True  # 1 回だけ

    def test_stop_calls_thread_wait(self) -> None:
        act, _, _, thread = self._make()
        act.start()
        act.stop()
        assert act._running is False
        assert thread.waited is True

    def test_stop_is_idempotent(self) -> None:
        act, _, _, _ = self._make()
        act.stop()  # start 前でも例外なし

    def test_publish_once_writes_both_left_and_right_with_gripper_q(self) -> None:
        # Real path: cmd_factory 経由で MotorCmds_ 生成 → cmds[0].q に gripper q 書く
        # → 2 publisher に別々 Write
        act, lp, rp, _ = self._make()
        act.send_action([0.4, 0.6])
        act.start()
        act._publish_once()
        assert len(lp.written) == 1
        assert len(rp.written) == 1
        # cmds[0].q が期待 gripper q になっている
        assert lp.written[0].cmds[0].q == 0.4
        assert rp.written[0].cmds[0].q == 0.6
        assert act.read_last_commanded_positions() == pytest.approx([0.4, 0.6])

    def test_publish_once_skip_when_no_target(self) -> None:
        act, lp, rp, _ = self._make()
        act.start()
        # target 送信前
        act._publish_once()
        assert lp.written == []
        assert rp.written == []

    def test_publish_once_skip_when_not_running(self) -> None:
        act, lp, rp, _ = self._make()
        act.send_action([0.5, 0.5])
        # start() 呼ばず _publish_once
        act._publish_once()
        assert lp.written == []
        assert rp.written == []

    def test_publish_writes_latest_only(self) -> None:
        # 複数回 send_action、_publish_once で最新のみ反映される (latest-only pattern)
        act, lp, rp, _ = self._make()
        act.start()
        act.send_action([0.1, 0.2])
        act.send_action([0.9, 0.8])  # 上書き
        act._publish_once()
        assert lp.written[0].cmds[0].q == 0.9
        assert rp.written[0].cmds[0].q == 0.8

    def test_failed_dds_write_is_latched_for_foreground_health_check(self) -> None:
        act = G1HandActuator(
            left_publisher=_FailedPublisher(),
            right_publisher=_FakePublisher(),
            publish_thread=_FakeThread(),
            cmd_factory=_fake_cmd_factory,
        )
        act.start()
        act.send_action([0.4, 0.6])
        act._publish_once()
        assert act.read_last_commanded_positions() is None
        with pytest.raises(RuntimeError, match="Dex1 DDS publisher failed"):
            act.raise_if_unhealthy()

    def test_slew_limited_publish_starts_from_primed_measured_pose(self) -> None:
        lp = _FakePublisher()
        rp = _FakePublisher()
        act = G1HandActuator(
            control_freq_hz=10.0,
            velocity_limit_rad_s=1.0,
            left_publisher=lp,
            right_publisher=rp,
            publish_thread=_FakeThread(),
            cmd_factory=_fake_cmd_factory,
        )
        act.prime_hold([5.0, 0.0])
        act.start()
        act.send_action([1.0, 3.0])
        act._publish_once()
        assert lp.written[-1].cmds[0].q == pytest.approx(4.9)
        assert rp.written[-1].cmds[0].q == pytest.approx(0.1)


# ---- Topic constants ---- #


class TestDex1Topics:
    """Dex1-1 DDS topic 名 (docs/inference/apple_vision_pro_upper_body_teleop.md §4.4)。"""

    def test_cmd_topics(self) -> None:
        from inference.desktop.lower_policy.actuators.hand import (
            G1_DEX1_LEFT_CMD_TOPIC,
            G1_DEX1_RIGHT_CMD_TOPIC,
        )
        assert G1_DEX1_LEFT_CMD_TOPIC == "rt/dex1/left/cmd"
        assert G1_DEX1_RIGHT_CMD_TOPIC == "rt/dex1/right/cmd"

    def test_state_topics(self) -> None:
        from inference.desktop.lower_policy.actuators.hand import (
            G1_DEX1_LEFT_STATE_TOPIC,
            G1_DEX1_RIGHT_STATE_TOPIC,
        )
        assert G1_DEX1_LEFT_STATE_TOPIC == "rt/dex1/left/state"
        assert G1_DEX1_RIGHT_STATE_TOPIC == "rt/dex1/right/state"
