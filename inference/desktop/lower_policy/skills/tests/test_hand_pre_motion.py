"""HandPreMotionSkill の unit test (Issue #141 束 1-8 / D2)。

主眼:
    - 腕は最初の tick の測定値を保持し続ける (手が動く間、腕は動かさない)
    - Dex1 は ramp で目標へ動き、到達が 5 tick 続いたら完了
    - `grasp` だけは接触して止まっても完了 (力センサが無いので位置で判定)
    - Dex1 の state が古い / 来ない tick は数えない。締め切りを過ぎたら failure_reason
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import yaml

from inference.desktop.lower_policy.actuators.hand import HAND_GRIP_MAX, MockHandActuator
from inference.desktop.lower_policy.skills.hand_pre_motion import HandPreMotionSkill
from inference.desktop.perception.g1_urdf_fk import G1_JOINT_NAMES

SKILL_CONFIG_PATH = "inference/desktop/lower_policy/configs/skill_config.yaml"
ARM_START = 15


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class _JointState:
    def __init__(self, arm_value: float = 0.25) -> None:
        self.name = G1_JOINT_NAMES
        self.position = np.zeros(29)
        self.position[ARM_START:] = arm_value
        self.velocity = np.zeros(29)
        self.effort = np.zeros(29)
        self.t = 0


class _HandState:
    def __init__(
        self,
        position,
        age_s: float = 0.0,
        *,
        fault_code=(0, 0),
        health_available: bool = False,
    ) -> None:
        self.position_rad = np.asarray(position, dtype=np.float64)
        self.fault_code = np.asarray(fault_code, dtype=np.uint32)
        self.health_diagnostics_available = health_available
        received = time.monotonic_ns() - int(age_s * 1e9)
        self.left_received_monotonic_ns = received
        self.right_received_monotonic_ns = received
        self.t = received


def _obs(hand, *, arm_value: float = 0.25, age_s: float = 0.0) -> dict:
    return {
        "joint_state": _JointState(arm_value),
        "hand_state": None if hand is None else _HandState(hand, age_s),
        "t": time.monotonic_ns(),
    }


def _fault_obs(hand, fault_code) -> dict:
    obs = _obs(hand)
    obs["hand_state"] = _HandState(
        hand, fault_code=fault_code, health_available=True
    )
    return obs


def _skill(clock: _Clock, **kwargs) -> tuple[HandPreMotionSkill, MockHandActuator]:
    hand = MockHandActuator()
    defaults = dict(
        target_rad=(2.0, 2.0),
        velocity_limit_rad_s=1.5,
        tolerance_rad=0.15,
        required_stable_samples=5,
        time_fn=clock,
    )
    defaults.update(kwargs)
    return HandPreMotionSkill("hand_test", hand_actuator=hand, **defaults), hand


def _tick_for(skill, clock, seconds: float, make_obs, hz: float = 30.0) -> None:
    """`hz` で tick を刻みながら `seconds` 進める。失敗したらそこで止める。

    締め切りと watchdog は「指令を進められた制御時間」(1 tick 最大 0.1 s) で数える
    (Issue #159 T6) ので、時計を 1 回で跳ばしても進まない。
    """
    for _ in range(int(round(seconds * hz))):
        clock.now += 1.0 / hz
        skill.step(make_obs())
        if skill.failure_reason is not None or skill.is_complete:
            return


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


class TestArmHold:
    def test_arm_holds_the_pose_measured_on_the_first_tick(self, clock):
        skill, _ = _skill(clock)
        skill.start({})
        first = skill.step(_obs((5.4, 5.4), arm_value=0.25))
        later = skill.step(_obs((4.0, 4.0), arm_value=0.90))   # 腕が動いても指令は変えない
        assert first.shape == (14,)
        np.testing.assert_allclose(first, 0.25)
        np.testing.assert_allclose(later, 0.25)

    def test_missing_joint_state_is_rejected(self, clock):
        skill, _ = _skill(clock)
        skill.start({})
        obs = _obs((5.4, 5.4))
        obs["joint_state"] = None
        with pytest.raises(RuntimeError, match="live joint state"):
            skill.step(obs)


class TestRamp:
    def test_faulted_motor_is_rejected_before_first_command(self, clock):
        skill, hand = _skill(clock, target_rad=(5.4, 5.4))
        skill.start({})
        skill.step(_fault_obs((5.3, 3.2), (0, 0x200)))
        assert hand.history == []
        assert skill.failure_reason is not None
        assert "right=0x200" in skill.failure_reason
        assert "winding overheating" in skill.failure_reason

    def test_hand_command_is_bounded_by_the_velocity_limit(self, clock):
        skill, hand = _skill(clock, target_rad=(0.0, 0.0))
        skill.start({})
        for _ in range(4):
            skill.step(_obs((5.4, 5.4)))
        commands = np.asarray(hand.history, dtype=np.float64)
        steps = np.abs(np.diff(np.vstack(([[5.4, 5.4]], commands)), axis=0))
        assert steps.max() <= 1.5 / 30.0 + 1e-9

    def test_ramp_ends_at_the_target_and_then_repeats_it(self, clock):
        skill, hand = _skill(clock, target_rad=(5.3, 5.3))
        skill.start({})
        for _ in range(20):                     # 5.4 -> 5.3 は 2 step で終わる
            skill.step(_obs((5.4, 5.4)))
        np.testing.assert_allclose(hand.history[-1], (5.3, 5.3))

    def test_no_command_until_the_dex1_state_arrives(self, clock):
        skill, hand = _skill(clock)
        skill.start({})
        skill.step(_obs(None))
        assert hand.history == []


class TestCompletion:
    def test_completes_after_five_stable_ticks_at_the_target(self, clock):
        skill, _ = _skill(clock, target_rad=(2.0, 2.0))
        skill.start({})
        for i in range(4):
            skill.step(_obs((2.0, 2.0)))
            assert not skill.is_complete, f"tick {i}"
        skill.step(_obs((2.0, 2.0)))
        assert skill.is_complete
        assert skill.completion_mode == "target"

    def test_stale_dex1_state_does_not_count(self, clock):
        skill, _ = _skill(clock)
        skill.start({})
        for _ in range(10):
            skill.step(_obs((2.0, 2.0), age_s=0.9))
        assert not skill.is_complete

    def test_grasp_accepts_a_hand_stopped_by_the_object(self, clock):
        """閉じる途中で物に当たって止まった (ストロークの 80% 以上進んで安定)。"""
        skill, _ = _skill(clock, target_rad=(1.0, 1.0), allow_closing_contact=True)
        skill.start({})
        skill.step(_obs((5.0, 5.0)))            # ここが start 位置
        for _ in range(78):
            skill.step(_obs((1.3, 1.3)))        # final command 前は完了しない
        assert not skill.is_complete
        for _ in range(12):
            skill.step(_obs((1.3, 1.3)))        # 80% 進んで動かない
        assert skill.is_complete
        assert skill.completion_mode == "closing_contact"

    def test_open_move_does_not_accept_contact(self, clock):
        skill, _ = _skill(clock, target_rad=(1.0, 1.0), allow_closing_contact=False)
        skill.start({})
        skill.step(_obs((5.0, 5.0)))
        for _ in range(12):
            skill.step(_obs((1.3, 1.3)))
        assert not skill.is_complete


class TestDeadline:
    def test_fresh_but_frozen_feedback_fails_with_bridge_diagnostic(self, clock):
        skill, _ = _skill(clock, target_rad=(5.4, 5.4))
        skill.start({})
        skill.step(_obs((2.0, 0.0)))
        _tick_for(skill, clock, 2.1, lambda: _obs((2.0, 0.0)))
        assert "hardware did not move" in skill.failure_reason
        assert "dex1_1_gripper_server" in skill.failure_reason

    def test_one_moving_hand_does_not_mask_the_other_stalled_hand(self, clock):
        """左右の最大進捗では片側断線を見逃すため、各手を独立監視する。"""
        skill, _ = _skill(clock, target_rad=(5.4, 5.4))
        skill.start({})
        skill.step(_obs((2.0, 3.2)))
        _tick_for(skill, clock, 2.1, lambda: _obs((5.35, 3.2)))
        assert "right" in skill.failure_reason
        assert "left(" not in skill.failure_reason
        assert "progress=0.0000rad" in skill.failure_reason

    def test_timeout_reason_is_set_after_the_deadline(self, clock):
        """実測は来ていて目標に届かないだけ = 時間切れ (故障ではない、次へ進む)。"""
        skill, _ = _skill(clock, target_rad=(0.0, 0.0))
        skill.start({})
        skill.step(_obs((5.4, 5.4)))
        assert skill.timeout_reason is None
        _tick_for(skill, clock, 60.0, lambda: _obs((5.0, 5.0)))
        assert "did not reach" in skill.timeout_reason
        assert skill.failure_reason is None
        assert not skill.is_complete

    def test_failure_reason_is_set_when_the_dex1_state_never_arrives(self, clock):
        """state が一度も来なくても締め切りで失敗すること (review 指摘 5)。

        締め切りを ramp の開始時にしか張らないと、Dex1 の state が途切れたときに
        完了も失敗もせず、`max_dwell_sec` も無いので腕を持ったまま止まらなくなる。
        """
        skill, hand = _skill(clock)
        skill.start({})
        skill.step(_obs(None))
        assert skill.failure_reason is None
        _tick_for(skill, clock, 60.0, lambda: _obs(None))
        # 実測が一度も来ない = センサーが来ていない故障。時間切れにはしない。
        assert "no fresh Dex1 state" in skill.failure_reason
        assert skill.timeout_reason is None
        assert not skill.is_complete
        assert hand.history == []

    def test_reaching_the_target_leaves_no_failure(self, clock):
        skill, _ = _skill(clock, target_rad=(5.4, 5.4))
        skill.start({})
        for _ in range(6):
            skill.step(_obs((5.4, 5.4)))
        _tick_for(skill, clock, 60.0, lambda: _obs((5.4, 5.4)))
        assert skill.is_complete and skill.failure_reason is None


class _Dex1WithTelemetry(MockHandActuator):
    """直前の指令を返す hand actuator (自前経路の Dex1 と同じ API)。

    `previous` は、この skill より前 (直前の model) が最後に出した指令。
    """

    def __init__(self, previous) -> None:
        super().__init__()
        self._commanded = np.asarray(previous, dtype=np.float64)

    def send_action(self, positions) -> None:
        super().send_action(positions)
        self._commanded = np.asarray(self.latest, dtype=np.float64)

    def read_last_commanded_positions(self):
        return self._commanded.copy()


def _run_gripping_hand(clock, hand, *, start, contact_floor, target, seconds=6.0):
    """指令へ 1.5 rad/s で動くが、脚に当たると contact_floor より閉じない手で回す。"""
    skill = HandPreMotionSkill(
        "hand_transition_pick_table_leg_to_insert_table_leg",
        hand_actuator=hand,
        target_rad=target,
        allow_closing_contact=True,
        time_fn=clock,
    )
    skill.start({})
    measured = np.asarray(start, dtype=np.float64)
    floor = np.asarray(contact_floor, dtype=np.float64)
    dt = 1.0 / 30.0
    end = clock.now + seconds
    while clock.now < end:
        skill.step(_obs(tuple(measured)))
        if skill.is_complete or skill.failure_reason is not None:
            break
        command = hand._commanded if hasattr(hand, "_commanded") else hand.latest
        if command is not None:
            measured = measured + np.clip(np.asarray(command) - measured, -1.5 * dt, 1.5 * dt)
            measured = np.maximum(measured, floor)
        clock.now += dt
    return skill


class TestHandAlreadyGripping:
    """pick -> insert の境界: pick の終わりに脚を握った手 (Issue #159 B4a-04)。

    実測は脚に当たって 2.19 / 2.23 で止まり、直前の指令は 1.9。insert の開始開度は
    左 3.057 (開く) / 右 1.896 (握ったまま)。右手は 1 mm も閉じられない。
    """

    PICK_END = (2.19, 2.23)
    PREVIOUS_COMMAND = (1.9, 1.9)
    INSERT_FRAME_ZERO = (3.057, 1.896)

    def test_completes_while_still_holding_the_leg(self, clock):
        hand = _Dex1WithTelemetry(self.PREVIOUS_COMMAND)
        skill = _run_gripping_hand(
            clock,
            hand,
            start=self.PICK_END,
            contact_floor=(0.0, 2.23),
            target=self.INSERT_FRAME_ZERO,
        )
        assert skill.failure_reason is None
        assert skill.is_complete
        assert skill.completion_mode == "holding_contact"

    def test_the_gripping_hand_is_never_loosened(self, clock):
        """ramp を実測 (2.23) から始めると指令が 1.9 -> 2.2 に緩み、把持力が抜ける。"""
        hand = _Dex1WithTelemetry(self.PREVIOUS_COMMAND)
        _run_gripping_hand(
            clock,
            hand,
            start=self.PICK_END,
            contact_floor=(0.0, 2.23),
            target=self.INSERT_FRAME_ZERO,
        )
        right_commands = np.asarray(hand.history, dtype=np.float64)[:, 1]
        assert right_commands.max() <= self.PREVIOUS_COMMAND[1] + 1e-9

    def test_an_empty_frozen_hand_still_trips_the_watchdog(self, clock):
        """握っていない (実測 = 指令) のに動かない手は、今までどおり異常として止める。"""
        hand = _Dex1WithTelemetry((3.0, 3.0))
        skill = _run_gripping_hand(
            clock,
            hand,
            start=(3.0, 3.0),
            contact_floor=(3.0, 3.0),
            target=self.INSERT_FRAME_ZERO,
        )
        assert not skill.is_complete
        assert "did not move" in skill.failure_reason

    def test_without_command_telemetry_the_old_rule_applies(self, clock):
        """直前の指令が読めない actuator では「握っている」と判断しない。"""
        hand = MockHandActuator()
        skill = _run_gripping_hand(
            clock,
            hand,
            start=self.PICK_END,
            contact_floor=(0.0, 2.23),
            target=self.INSERT_FRAME_ZERO,
        )
        assert not skill.is_complete
        assert "did not move" in skill.failure_reason

    def test_opening_from_full_closure_is_unchanged(self, clock):
        """頭の手順 (全開から脚へ閉じる) は今までどおり closing_contact で完了する。"""
        hand = _Dex1WithTelemetry((HAND_GRIP_MAX, HAND_GRIP_MAX))
        skill = _run_gripping_hand(
            clock,
            hand,
            start=(HAND_GRIP_MAX, HAND_GRIP_MAX),
            contact_floor=(0.0, 2.23),
            target=self.INSERT_FRAME_ZERO,
        )
        assert skill.failure_reason is None
        assert skill.completion_mode == "closing_contact"


def _release_skill(clock, hand) -> HandPreMotionSkill:
    return HandPreMotionSkill(
        "hand_release_insert_table_leg_to_rotate_leg_to_tighten",
        hand_actuator=hand,
        target_rad=(4.5, 4.5),
        release=True,
        time_fn=clock,
    )


def _run_release(clock, hand, *, start, seconds=6.0):
    """指令へ 1.5 rad/s で追従する (開く向きは物に邪魔されない) 手で回す。"""
    skill = _release_skill(clock, hand)
    skill.start({})
    measured = np.asarray(start, dtype=np.float64)
    dt = 1.0 / 30.0
    end = clock.now + seconds
    while clock.now < end:
        skill.step(_obs(tuple(measured)))
        if skill.is_complete or skill.failure_reason is not None:
            break
        if hand.latest is not None:
            measured = measured + np.clip(
                np.asarray(hand.latest) - measured, -1.5 * dt, 1.5 * dt
            )
        clock.now += dt
    return skill


class TestReleaseBeforeTheArmMoves:
    """pick -> insert 以外の境界: 腕を上げる前に、握っている手だけ離す (Issue #159 T3)。

    教師 episode は pick 以外、手を離して終わる。離す前に時間切れで切られたまま
    腕を上げると、挿した脚や締めた脚を引き上げる。
    """

    def test_only_the_gripping_hand_is_opened(self, clock):
        # 右手は締めた脚を握ったまま (実測 2.2 / 指令 1.9)、左手は握り拳 (実測 0.2 / 指令 0.0)
        hand = _Dex1WithTelemetry((0.0, 1.9))
        skill = _run_release(clock, hand, start=(0.2, 2.2))
        assert skill.failure_reason is None
        assert skill.is_complete
        commands = np.asarray(hand.history, dtype=np.float64)
        np.testing.assert_allclose(commands[:, 0], 0.0)   # 握り拳は触らない
        assert commands[-1, 1] == pytest.approx(4.5)      # 右手は離す

    def test_free_hands_complete_at_once_without_a_command(self, clock):
        """教師どおりに離して終わっていれば、時間も指令も使わない。"""
        hand = _Dex1WithTelemetry((1.49, 3.89))
        skill = _release_skill(clock, hand)
        skill.start({})
        skill.step(_obs((1.40, 3.87)))
        assert skill.is_complete
        assert skill.completion_mode == "nothing_to_release"
        assert hand.history == []

    def test_without_command_telemetry_any_open_hand_is_released(self, clock):
        """直前の指令が読めないときは、物を持てる開度 (1.0 rad 以上) の手を開く。"""
        hand = MockHandActuator()
        skill = _run_release(clock, hand, start=(0.2, 2.2))
        assert skill.is_complete
        commands = np.asarray(hand.history, dtype=np.float64)
        assert commands[-1, 1] == pytest.approx(4.5)
        np.testing.assert_allclose(commands[:, 0], 0.2)   # 握り拳は今の位置のまま


def _run_following_hand(clock, hz: float, *, start, target, seconds=20.0):
    """指令にすぐ追従する手で `hz` の tick を刻む。(skill, 送った指令の列) を返す。"""
    hand = _Dex1WithTelemetry(start)
    skill = HandPreMotionSkill(
        "hand_pose_test", hand_actuator=hand, target_rad=target, time_fn=clock
    )
    skill.start({})
    measured = np.asarray(start, dtype=np.float64)
    end = clock.now + seconds
    while clock.now < end:
        skill.step(_obs(tuple(measured)))
        if skill.is_complete or skill.failure_reason is not None:
            break
        if hand.latest is not None:
            measured = np.asarray(hand.latest, dtype=np.float64)
        clock.now += 1.0 / hz
    return skill, np.asarray(hand.history, dtype=np.float64)


class TestControlTime:
    """ramp・締め切り・watchdog は「指令を進められた制御時間」で数える (Issue #159 T6)。

    腕の pre-motion (#152 97e502d) と同じ数え方。ramp は 1 tick に 1 段ではなく
    制御時間に応じて進むので、tick が遅くても 1.5 rad/s を保つ。1 tick の時間は最大
    0.1 s で頭打ちなので、tick が止まった間は数えない。
    """

    STROKE_START = (0.0, 0.0)
    STROKE_TARGET = (2.07, 4.47)   # B4a-09 の probe と同じ大きいストローク

    @pytest.mark.parametrize("hz", [30.0, 20.0, 12.0])
    def test_a_long_stroke_completes_at_low_tick_rates(self, clock, hz):
        started = clock.now
        skill, _ = _run_following_hand(
            clock, hz, start=self.STROKE_START, target=self.STROKE_TARGET
        )
        assert skill.failure_reason is None
        assert skill.is_complete
        # 4.47 rad を 1.5 rad/s = 約 3 s。tick が遅くても速さは変わらない
        assert clock.now - started < 3.0 + 0.5

    def test_the_30_hz_command_sequence_is_one_ramp_step_per_tick(self, clock):
        from inference.desktop.lower_policy.skills.hand_ramp import build_hand_target_ramp

        _, commands = _run_following_hand(
            clock, 30.0, start=self.STROKE_START, target=self.STROKE_TARGET
        )
        ramp = np.asarray(
            build_hand_target_ramp(self.STROKE_START, self.STROKE_TARGET), dtype=np.float64
        )
        np.testing.assert_allclose(commands[: len(ramp)], ramp)

    def test_one_long_stall_does_not_trip_the_deadline(self, clock):
        hand = _Dex1WithTelemetry((0.0, 0.0))
        skill = HandPreMotionSkill(
            "hand_pose_test", hand_actuator=hand, target_rad=(2.0, 2.0), time_fn=clock
        )
        skill.start({})
        skill.step(_obs((0.0, 0.0)))
        clock.now += 30.0            # 読み込みなどで tick が 30 s 止まった
        skill.step(_obs((0.0, 0.0)))
        assert skill.failure_reason is None


@pytest.fixture(scope="module")
def config() -> dict:
    with open(SKILL_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestFromConfig:
    def test_open_target_is_the_physical_full_open(self, config):
        """Preparation opens fully; the dataset's 4.5-rad scale is not a hardware limit."""
        assert config["hand_pre_motion"]["open_rad"] == pytest.approx(HAND_GRIP_MAX)
        skill = HandPreMotionSkill.from_config(
            config, "rotate_table_base", target="open", hand_actuator=MockHandActuator()
        )
        np.testing.assert_allclose(skill._target, (HAND_GRIP_MAX, HAND_GRIP_MAX))

    def test_an_already_open_hand_completes_without_the_watchdog(self, config, clock):
        skill = HandPreMotionSkill.from_config(
            config,
            "rotate_table_base",
            target="open",
            hand_actuator=MockHandActuator(),
            time_fn=clock,
        )
        skill.start({})
        _tick_for(skill, clock, 3.0, lambda: _obs((HAND_GRIP_MAX, HAND_GRIP_MAX)))
        assert skill.failure_reason is None
        assert skill.is_complete

    def test_pose_target_comes_from_the_teacher_frame_zero(self, config):
        skill = HandPreMotionSkill.from_config(
            config, "rotate_table_base", target="pose", hand_actuator=MockHandActuator()
        )
        fraction = config["skills"]["rotate_table_base"]["initial_pose"]["dex1_opening_fraction"]
        np.testing.assert_allclose(
            skill._target, (fraction["left"] * 4.5, fraction["right"] * 4.5)
        )

    def test_grasp_allows_contact_and_pose_does_not(self, config):
        kwargs = dict(skill_config=config, skill_name="insert_table_leg", hand_actuator=MockHandActuator())
        grasp = HandPreMotionSkill.from_config(target="grasp", **kwargs)
        pose = HandPreMotionSkill.from_config(target="pose", **kwargs)
        assert grasp._allow_closing_contact and not pose._allow_closing_contact
        np.testing.assert_allclose(grasp._target, pose._target)

    def test_shipped_config_values_are_used(self, config):
        skill = HandPreMotionSkill.from_config(
            config, "rotate_table_base", target="open", hand_actuator=MockHandActuator()
        )
        assert skill._velocity_limit_rad_s == pytest.approx(1.5)
        assert skill._tolerance_rad == pytest.approx(0.05)
        assert skill._required_stable_samples == 5
        assert skill._holding_contact_gap_rad == pytest.approx(0.10)
        assert skill._holding_min_opening_rad == pytest.approx(1.0)

    def test_release_opens_to_the_physical_maximum(self, config):
        skill = HandPreMotionSkill.from_config(
            config, "rotate_leg_to_tighten", target="release", hand_actuator=MockHandActuator()
        )
        np.testing.assert_allclose(skill._target, (HAND_GRIP_MAX, HAND_GRIP_MAX))
        assert skill._release and not skill._allow_closing_contact

    def test_unknown_target_is_rejected(self, config):
        with pytest.raises(ValueError, match="target must be one of"):
            HandPreMotionSkill.from_config(
                config, "rotate_table_base", target="close", hand_actuator=MockHandActuator()
            )
