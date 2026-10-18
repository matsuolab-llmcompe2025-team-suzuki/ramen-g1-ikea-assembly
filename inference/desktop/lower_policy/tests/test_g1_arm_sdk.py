"""G1ArmActuator unit test (Issue #75)。

SDK / cyclonedds install 不要な形で書く: publisher / crc_fn / publish_thread を
inject 経路で mock 化 → 実 SDK に触らずに build_lowcmd / send_action / lifecycle
の behavior を検証。
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from inference.desktop.lower_policy.actuators.g1_arm_sdk import (
    G1_ARM_JOINT_INDICES,
    G1_NUM_ARM_JOINTS,
    G1_NUM_WAIST_JOINTS,
    G1_WAIST_JOINT_INDICES,
    G1_ARM_POSITION_LOWER_RAD,
    G1_ARM_POSITION_UPPER_RAD,
    G1_WAIST_POSITION_LOWER_RAD,
    G1_WAIST_POSITION_UPPER_RAD,
    ArmControlGains,
    ArmSafetyLimits,
    G1ArmActuator,
)


# =========================================================================
# helpers: fake SDK layer (publisher / CRC / RecurrentThread)
# =========================================================================
class _FakePublisher:
    def __init__(self) -> None:
        self.written: list[Any] = []

    def Write(self, msg: Any) -> None:
        self.written.append(msg)


class _FakeCRC:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def Crc(self, msg: Any) -> int:
        self.calls.append(msg)
        return 0xDEADBEEF


class _FakeRecurrentThread:
    """`Start()` / `Wait()` だけ satisfy する fake、実 thread は起動しない。"""

    def __init__(self) -> None:
        self.started = False
        self.waited = False

    def Start(self) -> None:
        self.started = True

    def Wait(self, timeout: float = 1.0) -> None:
        self.waited = True


class _FakeGravityCompensator:
    def __init__(self, torque: np.ndarray) -> None:
        self._torque = np.asarray(torque, dtype=np.float64)
        self.calls: list[np.ndarray] = []

    def torque_nm(self, arm_position_rad: np.ndarray) -> np.ndarray:
        self.calls.append(np.asarray(arm_position_rad, dtype=np.float64).copy())
        return self._torque.copy()


def _make_actuator(**overrides: Any) -> tuple[G1ArmActuator, _FakePublisher, _FakeCRC, _FakeRecurrentThread]:
    pub = _FakePublisher()
    crc = _FakeCRC()
    thread = _FakeRecurrentThread()
    actuator = G1ArmActuator(
        publisher=pub,
        crc_fn=crc,
        publish_thread=thread,
        initial_positions=np.linspace(-0.28, 0.28, 29),
        **overrides,
    )
    return actuator, pub, crc, thread


# =========================================================================
# G1_ARM_JOINT_INDICES 定数
# =========================================================================
class TestJointIndexConstants:
    def test_arm_joint_indices_len(self):
        assert G1_NUM_ARM_JOINTS == 14
        assert len(G1_ARM_JOINT_INDICES) == 14

    def test_arm_joint_indices_are_15_to_28(self):
        assert G1_ARM_JOINT_INDICES == tuple(range(15, 29))


# =========================================================================
# send_action shape + clamp
# =========================================================================
class TestSendAction:
    def test_invalid_safety_limit_configuration_is_rejected(self):
        with pytest.raises(ValueError, match="joint_max_abs"):
            ArmSafetyLimits(joint_max_abs=float("nan"))
        with pytest.raises(ValueError, match="velocity_limit"):
            ArmSafetyLimits(velocity_limit_rad_s=0.0)

    def test_wrong_shape_raises(self):
        actuator, _, _, _ = _make_actuator()
        with pytest.raises(ValueError, match="shape"):
            actuator.send_action(np.zeros(13))
        with pytest.raises(ValueError, match="shape"):
            actuator.send_action(np.zeros(29))

    def test_target_stored_after_send_action(self):
        actuator, _, _, _ = _make_actuator()
        positions = np.linspace(-0.5, 0.5, G1_NUM_ARM_JOINTS)
        actuator.send_action(positions)
        assert actuator._target.received
        assert np.allclose(actuator._target.positions, positions)

    def test_clamp_applied(self):
        """|target| が limits.joint_max_abs を超えたら clamp される。"""
        actuator, _, _, _ = _make_actuator(
            limits=ArmSafetyLimits(joint_max_abs=1.0)
        )
        positions = np.array([2.0] * G1_NUM_ARM_JOINTS)  # 全部 clamp 対象
        actuator.send_action(positions)
        assert actuator._target.positions is not None
        assert np.all(np.abs(actuator._target.positions) <= 1.0)

    def test_default_uses_joint_specific_urdf_limits(self):
        actuator, _, _, _ = _make_actuator()
        actuator.send_action(np.full(G1_NUM_ARM_JOINTS, -10.0))
        np.testing.assert_allclose(
            actuator._target.positions, G1_ARM_POSITION_LOWER_RAD
        )
        actuator.send_action(np.full(G1_NUM_ARM_JOINTS, 10.0))
        np.testing.assert_allclose(
            actuator._target.positions, G1_ARM_POSITION_UPPER_RAD
        )

    def test_nonfinite_target_is_rejected(self):
        actuator, _, _, _ = _make_actuator()
        target = np.zeros(G1_NUM_ARM_JOINTS)
        target[3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            actuator.send_action(target)


# =========================================================================
# publish loop (_publish_once) の LowCmd_ 組み立て
# =========================================================================
class TestPublishOnce:
    def test_idle_before_first_send_action(self):
        """未受信なら _publish_once は publisher に何も送らない。"""
        actuator, pub, _, _ = _make_actuator()
        actuator.start()
        actuator._publish_once()
        assert pub.written == []

    def test_publish_only_after_start_and_send_action(self):
        actuator, pub, crc, _ = _make_actuator()
        actuator.start()
        actuator.send_action(np.ones(G1_NUM_ARM_JOINTS) * 0.3)
        actuator._publish_once()
        assert len(pub.written) == 1
        # CRC が計算されて lowcmd.crc に注入されてる
        assert crc.calls == pub.written  # same instance passed to CRC then Write
        published_arm, published_waist = actuator.read_last_published_targets()
        assert published_arm is not None
        assert published_waist is None

    def test_failed_write_does_not_advance_published_target_telemetry(self):
        class _RaisingPublisher:
            def Write(self, msg):
                raise RuntimeError("simulated DDS failure")

        actuator = G1ArmActuator(
            publisher=_RaisingPublisher(),
            crc_fn=_FakeCRC(),
            publish_thread=_FakeRecurrentThread(),
            initial_positions=np.zeros(29),
        )
        actuator.start()
        actuator.send_action(np.ones(G1_NUM_ARM_JOINTS) * 0.1)
        actuator._publish_once()
        assert actuator.read_last_published_targets() == (None, None)

    def test_publish_no_op_if_stopped(self):
        actuator, pub, _, _ = _make_actuator()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))
        # start 呼ばず _publish_once するのは、running=False なので何もしない
        actuator._publish_once()
        assert pub.written == []

    def test_lowcmd_matches_official_motion_mode_contract(self):
        actuator, pub, _, _ = _make_actuator()
        actuator.start()
        targets = np.linspace(-0.05, 0.05, G1_NUM_ARM_JOINTS)
        actuator.send_action(targets)
        actuator._publish_once()
        assert len(pub.written) == 1
        lowcmd = pub.written[0]
        # Official motion mode locks every physical G1 joint at startup pose.
        for i in range(29):
            assert lowcmd.motor_cmd[i].mode == 1
        # Arms overwrite only their own initial hold target.
        initial_arm = np.linspace(-0.28, 0.28, 29)[15:29]
        max_step = 20.0 / 250.0
        scale = max(float(np.max(np.abs(targets - initial_arm))) / max_step, 1.0)
        expected = initial_arm + (targets - initial_arm) / scale
        for slot, joint_idx in enumerate(G1_ARM_JOINT_INDICES):
            m = lowcmd.motor_cmd[joint_idx]
            assert m.q == pytest.approx(expected[slot])
        assert lowcmd.motor_cmd[0].q == pytest.approx(-0.28)
        assert lowcmd.motor_cmd[29].q == pytest.approx(1.0)
        assert lowcmd.motor_cmd[15].kp == pytest.approx(80.0)
        assert lowcmd.motor_cmd[15].kd == pytest.approx(3.0)
        assert lowcmd.motor_cmd[19].kp == pytest.approx(40.0)
        assert lowcmd.motor_cmd[19].kd == pytest.approx(1.5)

    def test_custom_gains_applied(self):
        actuator, pub, _, _ = _make_actuator(
            gains=ArmControlGains(
                body_kp=25.0, body_kd=0.5,
                weak_kp=26.0, weak_kd=0.6,
                wrist_kp=27.0, wrist_kd=0.7,
            )
        )
        actuator.start()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))
        actuator._publish_once()
        lowcmd = pub.written[0]
        assert lowcmd.motor_cmd[0].kp == 25.0
        assert lowcmd.motor_cmd[0].kd == 0.5
        assert lowcmd.motor_cmd[15].kp == 26.0
        assert lowcmd.motor_cmd[15].kd == 0.6
        assert lowcmd.motor_cmd[19].kp == 27.0
        assert lowcmd.motor_cmd[19].kd == 0.7

    def test_official_gravity_feedforward_is_written_to_arm_slots_only(self):
        torque = np.linspace(-5.0, 5.0, G1_NUM_ARM_JOINTS)
        gravity = _FakeGravityCompensator(torque)
        actuator, pub, _, _ = _make_actuator(gravity_compensator=gravity)
        actuator.start()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))
        actuator._publish_once()

        assert len(gravity.calls) == 1
        lowcmd = pub.written[0]
        for slot, joint_idx in enumerate(G1_ARM_JOINT_INDICES):
            assert lowcmd.motor_cmd[joint_idx].tau == pytest.approx(torque[slot])
        for joint_idx in range(15):
            assert lowcmd.motor_cmd[joint_idx].tau == pytest.approx(0.0)

    def test_mode_machine_forwarded(self):
        actuator, pub, _, _ = _make_actuator(mode_machine=5)
        actuator.start()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))
        actuator._publish_once()
        assert pub.written[0].mode_machine == 5

    def test_large_target_is_velocity_limited(self):
        actuator, pub, _, _ = _make_actuator()
        actuator.start()
        actuator.send_action(np.ones(G1_NUM_ARM_JOINTS))
        actuator._publish_once()
        # The official global scale obeys 20 rad/s at 250 Hz (= 0.08 rad)
        # for the furthest arm joint in each message.
        initial_arm = np.linspace(-0.28, 0.28, 29)[15:29]
        target = np.ones(G1_NUM_ARM_JOINTS)
        scale = np.max(np.abs(target - initial_arm)) / (20.0 / 250.0)
        expected = initial_arm + (target - initial_arm) / scale
        assert pub.written[0].motor_cmd[15].q == pytest.approx(expected[0])

    def test_velocity_limiter_advances_from_last_successful_command_when_measured_stalls(self):
        """A loaded joint must not be trapped at one max_step from measured.

        The injected actuator has a fixed measured pose, which models a motor
        that has not moved yet.  Consecutive successful publications must
        nevertheless advance the command trajectory toward the target.
        """
        actuator, pub, _, _ = _make_actuator(
            limits=ArmSafetyLimits(velocity_limit_rad_s=1.0)
        )
        actuator.start()
        actuator.send_action(np.ones(G1_NUM_ARM_JOINTS))
        initial_arm = actuator._hold_positions[list(G1_ARM_JOINT_INDICES)]
        actuator._publish_once()
        first = actuator.read_last_published_targets()[0]
        actuator._publish_once()
        second = actuator.read_last_published_targets()[0]

        assert first is not None and second is not None
        assert np.max(np.abs(first - initial_arm)) == pytest.approx(1.0 / 250.0)
        assert np.max(np.abs(second - first)) == pytest.approx(1.0 / 250.0)
        assert np.linalg.norm(np.ones(G1_NUM_ARM_JOINTS) - second) < np.linalg.norm(
            np.ones(G1_NUM_ARM_JOINTS) - first
        )
        assert len(pub.written) == 2

    def test_publish_exception_caught(self):
        """publisher.Write() が raise しても process 死しない (listener thread 保護と同 pattern)。"""

        class _RaisingPublisher:
            def Write(self, msg):
                raise RuntimeError("simulated DDS failure")

        actuator = G1ArmActuator(
            publisher=_RaisingPublisher(),
            crc_fn=_FakeCRC(),
            publish_thread=_FakeRecurrentThread(),
        )
        actuator.start()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))
        actuator._publish_once()  # 例外が伝播しないことだけ確認 (test fail しなければ OK)


# =========================================================================
# lifecycle (start / stop / idempotent)
# =========================================================================
class TestLifecycle:
    def test_start_starts_publish_thread_once(self):
        actuator, _, _, thread = _make_actuator()
        actuator.start()
        assert thread.started is True
        assert actuator._running is True
        # 二重 start は fake_thread の start が呼ばれない (started は既に True)
        thread.started = False
        actuator.start()
        assert thread.started is False  # 呼ばれない = idempotent

    def test_stop_calls_wait_and_marks_stopped(self):
        actuator, _, _, thread = _make_actuator()
        actuator.start()
        actuator.stop()
        assert thread.waited is True
        assert actuator._running is False
        # 二重 stop は wait 呼ばない (idempotent)
        thread.waited = False
        actuator.stop()
        assert thread.waited is False

    def test_controlled_release_reaches_zero_before_stop(self):
        actuator, pub, _, _ = _make_actuator()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))
        actuator.start()
        actuator.controlled_release(duration_s=0.0)
        assert actuator.motion_weight == pytest.approx(0.0)
        actuator._publish_once()
        assert pub.written[-1].motor_cmd[29].q == pytest.approx(0.0)
        actuator.stop()

    def test_controlled_release_requires_running_actuator(self):
        actuator, _, _, _ = _make_actuator()
        with pytest.raises(RuntimeError, match="must be running"):
            actuator.controlled_release(duration_s=0.0)

    def test_controlled_release_rejects_negative_duration(self):
        actuator, _, _, _ = _make_actuator()
        actuator.start()
        with pytest.raises(ValueError, match="non-negative"):
            actuator.controlled_release(duration_s=-0.1)


# =========================================================================
# thread safety の smoke (競合状態を粗く確認)
# =========================================================================
class TestThreadSafety:
    def test_concurrent_send_action_and_publish(self):
        """send_action と _publish_once が並列でも lock で protect されて壊れない。"""
        actuator, pub, _, _ = _make_actuator()
        actuator.start()

        # send_action を多数回投げる thread と、_publish_once を回す thread を並列
        stop = threading.Event()

        def sender():
            i = 0
            while not stop.is_set():
                actuator.send_action(np.ones(G1_NUM_ARM_JOINTS) * (i % 10) * 0.01)
                i += 1

        def publisher_loop():
            while not stop.is_set():
                actuator._publish_once()

        threads = [threading.Thread(target=sender), threading.Thread(target=publisher_loop)]
        for t in threads:
            t.start()
        threading.Event().wait(0.05)  # 50ms concurrent 実行
        stop.set()
        for t in threads:
            t.join(timeout=1.0)
        # 生きて完走できれば OK (deadlock なし、例外 crash なし)
        assert not any(t.is_alive() for t in threads)


# =========================================================================
# Phase B-1: waist target dispatch (Issue #125)
# =========================================================================
class TestSendWaistAction:
    def test_waist_joint_indices_are_12_to_14(self):
        assert G1_NUM_WAIST_JOINTS == 3
        assert G1_WAIST_JOINT_INDICES == (12, 13, 14)

    def test_wrong_shape_raises(self):
        actuator, _, _, _ = _make_actuator()
        with pytest.raises(ValueError, match="waist positions"):
            actuator.send_waist_action(np.zeros(2))
        with pytest.raises(ValueError, match="waist positions"):
            actuator.send_waist_action(np.zeros(4))

    def test_target_stored_after_send(self):
        actuator, _, _, _ = _make_actuator()
        actuator.send_waist_action(np.array([0.1, -0.2, 0.3]))
        assert actuator._waist_target.received
        assert actuator._waist_target.positions is not None
        np.testing.assert_allclose(
            actuator._waist_target.positions, [0.1, -0.2, 0.3]
        )

    def test_clamp_applied(self):
        # Production defaults use the asymmetric pinned URDF waist bounds.
        actuator, _, _, _ = _make_actuator()
        actuator.send_waist_action(np.array([2.5, -2.5, 0.1]))
        np.testing.assert_allclose(
            actuator._waist_target.positions,
            [2.5, G1_WAIST_POSITION_LOWER_RAD[1], 0.1],
        )
        assert G1_WAIST_POSITION_UPPER_RAD[0] > 2.5

    def test_clear_waist_action_drops_stale_target_at_ownership_boundary(self):
        actuator, _, _, _ = _make_actuator()
        actuator.send_waist_action(np.array([0.1, -0.2, 0.3]))
        actuator._last_published_waist_positions = np.array([0.1, -0.2, 0.3])

        actuator.clear_waist_action()

        assert actuator._waist_target.received is False
        assert actuator._waist_target.positions is None
        assert actuator.read_last_published_targets()[1] is None
        np.testing.assert_allclose(
            actuator._hold_positions[list(G1_WAIST_JOINT_INDICES)],
            np.linspace(-0.28, 0.28, 29)[list(G1_WAIST_JOINT_INDICES)],
        )


class TestPublishWithWaist:
    def _prep(self):
        actuator, pub, _, _ = _make_actuator()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))  # arm target 必要
        actuator.start()
        return actuator, pub

    def test_lowcmd_waist_slot_overridden_when_send_waist_action_called(self):
        actuator, pub = self._prep()
        # initial hold_positions は linspace(-0.28, 0.28, 29)、waist [12,13,14] ≈ [-0.04, -0.02, 0.00]。
        # rate limit (20 rad/s @ 250Hz = 0.08 rad/tick) 内に収まる target を選ぶ。
        hold_waist = actuator._hold_positions[list(G1_WAIST_JOINT_INDICES)]
        target = hold_waist + np.array([0.01, -0.01, 0.02])  # 全 |delta| < 0.08
        actuator.send_waist_action(target)
        actuator._publish_once()
        assert len(pub.written) == 1
        cmd = pub.written[0]
        assert cmd.motor_cmd[12].q == pytest.approx(float(target[0]), abs=1e-6)
        assert cmd.motor_cmd[13].q == pytest.approx(float(target[1]), abs=1e-6)
        assert cmd.motor_cmd[14].q == pytest.approx(float(target[2]), abs=1e-6)

    def test_waist_target_is_velocity_limited(self):
        # 大 target 送信で rate-limited されること (arm 側の test_large_target_is_velocity_limited と対称)
        actuator, pub = self._prep()
        hold_waist = actuator._hold_positions[list(G1_WAIST_JOINT_INDICES)]
        far_target = hold_waist + np.array([1.0, 1.0, 1.0])  # delta >> max_step 0.08
        actuator.send_waist_action(far_target)
        actuator._publish_once()
        cmd = pub.written[0]
        # 実際の write は current + max_step 方向 (全 waist joint で ~0.08 進む)
        max_step = 20.0 / 250.0
        for slot, joint_idx in enumerate(G1_WAIST_JOINT_INDICES):
            actual_step = cmd.motor_cmd[joint_idx].q - float(hold_waist[slot])
            assert 0.0 < actual_step <= max_step + 1e-6, (
                f"waist joint {joint_idx}: expected step in (0, {max_step}], got {actual_step}"
            )

    def test_waist_limiter_advances_from_last_successful_command(self):
        actuator, _ = self._prep()
        hold = actuator._hold_positions[list(G1_WAIST_JOINT_INDICES)]
        # Keep every joint inside the asymmetric waist bounds while remaining
        # farther than one 0.08-rad publication step.
        actuator.send_waist_action(hold + 0.3)
        actuator._publish_once()
        first = actuator.read_last_published_targets()[1]
        actuator._publish_once()
        second = actuator.read_last_published_targets()[1]

        assert first is not None and second is not None
        np.testing.assert_allclose(first - hold, np.full(3, 20.0 / 250.0))
        np.testing.assert_allclose(second - first, np.full(3, 20.0 / 250.0))

    def test_lowcmd_waist_slot_keeps_hold_pose_when_no_send_waist_action(self):
        actuator, pub = self._prep()
        # waist target なし → publish → lowcmd waist slots は hold_positions (initial) を維持
        expected_hold = actuator._hold_positions[list(G1_WAIST_JOINT_INDICES)]
        actuator._publish_once()
        assert len(pub.written) == 1
        cmd = pub.written[0]
        for slot, joint_idx in enumerate(G1_WAIST_JOINT_INDICES):
            assert cmd.motor_cmd[joint_idx].q == pytest.approx(
                float(expected_hold[slot]), abs=1e-6
            )

    def test_arm_slots_still_overridden_when_waist_target_present(self):
        # waist target 追加後も arm target が上書きされることを確認 (regression 防止)
        actuator, pub = self._prep()
        actuator.send_action(np.linspace(0.1, 0.4, G1_NUM_ARM_JOINTS))
        actuator.send_waist_action(np.array([0.02, 0.03, 0.04]))
        actuator._publish_once()
        cmd = pub.written[0]
        # arm joint 15 が 0.1、joint 28 が 0.4 相当 (rate-limit で 1 step 進む)
        # 初期 latest_arm_positions は linspace(-0.28, 0.28) なので rate limit 内
        assert cmd.motor_cmd[15].q != cmd.motor_cmd[12].q  # arm と waist は独立


class TestWaistOnlyDispatch:
    """Phase B-1 Finding 4 (PR #126): waist target のみで publish 発火 + hold pose fallback。"""

    def _prep(self):
        actuator, pub, _, _ = _make_actuator()
        actuator.start()
        return actuator, pub

    def test_waist_only_dispatch_uses_hold_pose_for_arm(self):
        # arm target 未送信、waist だけ送る → publish 発火 + arm slot は hold_positions
        actuator, pub = self._prep()
        hold_waist = actuator._hold_positions[list(G1_WAIST_JOINT_INDICES)]
        target = hold_waist + np.array([0.01, -0.01, 0.02])  # rate-limit 内
        actuator.send_waist_action(target)
        actuator._publish_once()
        assert len(pub.written) == 1
        cmd = pub.written[0]
        # waist は target で override
        for slot, joint_idx in enumerate(G1_WAIST_JOINT_INDICES):
            assert cmd.motor_cmd[joint_idx].q == pytest.approx(float(target[slot]), abs=1e-6)
        # arm は hold_positions を維持 (init 時 initial_positions=linspace(-0.28, 0.28))
        expected_arm_hold = actuator._hold_positions[list(G1_ARM_JOINT_INDICES)]
        for slot, joint_idx in enumerate(G1_ARM_JOINT_INDICES):
            assert cmd.motor_cmd[joint_idx].q == pytest.approx(
                float(expected_arm_hold[slot]), abs=1e-6
            )

    def test_neither_target_skips_publish(self):
        # arm / waist 両方未送信 → publish 呼んでも write 発生しない
        actuator, pub = self._prep()
        actuator._publish_once()
        assert pub.written == []


class TestG1WaistActuator:
    """Waist wrapper class の smoke: send_action が arm_actuator.send_waist_action に流れる。"""

    def test_wrapper_delegates_to_arm_actuator(self):
        from inference.desktop.lower_policy.actuators.waist import G1WaistActuator

        actuator, _, _, _ = _make_actuator()
        wrapper = G1WaistActuator(actuator)
        wrapper.send_action([0.11, -0.22, 0.33])
        np.testing.assert_allclose(
            actuator._waist_target.positions, [0.11, -0.22, 0.33]
        )

    def test_wrapper_rejects_wrong_length(self):
        from inference.desktop.lower_policy.actuators.waist import G1WaistActuator

        actuator, _, _, _ = _make_actuator()
        wrapper = G1WaistActuator(actuator)
        with pytest.raises(ValueError, match="waist positions"):
            wrapper.send_action([0.1, 0.2])

    def test_wrapper_rejects_arm_actuator_without_send_waist_action(self):
        from inference.desktop.lower_policy.actuators.waist import G1WaistActuator

        class _Bad:
            def send_action(self, positions):
                pass
        with pytest.raises(TypeError, match="send_waist_action"):
            G1WaistActuator(_Bad())
