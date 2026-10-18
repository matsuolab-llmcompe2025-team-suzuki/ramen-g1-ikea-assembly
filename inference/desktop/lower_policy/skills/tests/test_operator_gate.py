from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np

from inference.desktop.lower_policy.pose_utils import ROS_ARM_JOINT_NAMES
from inference.desktop.lower_policy.skills.operator_gate import (
    OperatorConfirmationHoldSkill,
)


def _obs() -> dict:
    names = tuple(f"unused_{i}" for i in range(15)) + tuple(ROS_ARM_JOINT_NAMES)
    positions = tuple(0.0 for _ in range(15)) + tuple(
        np.linspace(-0.7, 0.7, len(ROS_ARM_JOINT_NAMES))
    )
    return {"joint_state": SimpleNamespace(name=names, position=positions)}


def test_stop_cancels_the_exact_reader_without_flushing_other_views():
    entered = threading.Event()
    finished = threading.Event()
    events = []

    def read(_prompt, cancel):
        events.append(cancel)
        entered.set()
        cancel.wait(1)
        finished.set()
        raise EOFError("cancelled")

    def unexpected_flush():
        raise AssertionError("must not flush stdin")

    gate = OperatorConfirmationHoldSkill(
        name="gate", next_skill_name="pick", cancellable_input_fn=read,
        discard_pending_input_fn=unexpected_flush,
    )
    gate.start({})
    gate.step(_obs())
    assert entered.wait(1)
    gate.stop()
    assert finished.wait(1)
    assert events[0].is_set()
    assert not gate.is_complete
    gate.start({})
    assert gate._input_cancel is not events[0]
    assert not gate._input_cancel.is_set()
    gate.stop()


def test_operator_gate_holds_while_input_is_pending_then_completes() -> None:
    release = threading.Event()

    def input_fn(_prompt: str) -> str:
        release.wait(timeout=1.0)
        return ""

    skill = OperatorConfirmationHoldSkill(
        name="operator_gate_for_pick_table_leg",
        next_skill_name="pick_table_leg",
        input_fn=input_fn,
    )
    skill.start({})
    first = skill.step(_obs())
    assert first.shape == (14,)
    assert not skill.is_complete
    np.testing.assert_allclose(skill.step(_obs()), first)

    release.set()
    deadline = time.monotonic() + 1.0
    while not skill.is_complete and time.monotonic() < deadline:
        time.sleep(0.001)
    assert skill.is_complete
    assert skill.failure_reason is None


def test_operator_gate_input_failure_is_fail_closed() -> None:
    def input_fn(_prompt: str) -> str:
        raise EOFError("stdin closed")

    skill = OperatorConfirmationHoldSkill(
        name="operator_gate_for_flip_table",
        next_skill_name="flip_table",
        input_fn=input_fn,
    )
    skill.start({})
    skill.step(_obs())
    deadline = time.monotonic() + 1.0
    while skill.failure_reason is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert not skill.is_complete
    assert "input failed" in (skill.failure_reason or "")


def test_operator_gate_discards_enter_typed_before_the_prompt() -> None:
    """手前の prompt で余分に押された Enter で policy を始めない。"""
    calls: list[str] = []
    done = threading.Event()

    def input_fn(_prompt: str) -> str:
        calls.append("input")
        done.set()
        return ""

    skill = OperatorConfirmationHoldSkill(
        name="operator_gate_for_pick_table_leg",
        next_skill_name="pick_table_leg",
        input_fn=input_fn,
        discard_pending_input_fn=lambda: calls.append("discard"),
    )
    skill.start({})
    skill.step(_obs())
    assert done.wait(timeout=1.0)
    assert calls == ["discard", "input"]


def test_old_reader_cannot_confirm_a_restarted_gate():
    releases = [threading.Event(), threading.Event()]
    started = [threading.Event(), threading.Event()]
    calls = []

    def input_fn(prompt):
        index = len(calls)
        calls.append(prompt)
        started[index].set()
        releases[index].wait(1.0)
        return ""

    skill = OperatorConfirmationHoldSkill(name="gate", next_skill_name="pick", input_fn=input_fn)
    skill.start({})
    skill.step(_obs())
    assert started[0].wait(1)
    skill.stop()
    skill.start({})
    skill.step(_obs())
    assert started[1].wait(1)
    releases[0].set()
    time.sleep(0.02)
    assert not skill.is_complete
    releases[1].set()
    deadline = time.monotonic() + 1
    while not skill.is_complete and time.monotonic() < deadline:
        time.sleep(0.001)
    assert skill.is_complete


def test_venue_gate_rechecks_fresh_arrival_after_enter(capsys):
    state = _obs()["joint_state"]
    state.received_monotonic_ns = time.monotonic_ns()
    state.velocity = np.zeros(29)
    skill = OperatorConfirmationHoldSkill(
        name="gate", next_skill_name="pick", require_arrival=True,
        input_fn=lambda _: "", arrival_check=lambda q: (True, "arrived"),
    )
    skill.require_fresh_state(lambda: state, max_age_s=0.5, max_speed_rad_s=0.05)
    skill.start({})
    skill.step({"joint_state": state})
    deadline = time.monotonic() + 1
    while not skill.is_complete and time.monotonic() < deadline:
        time.sleep(0.001)
    assert skill.is_complete
    state.received_monotonic_ns -= 1_000_000_000
    assert not skill.is_complete
    message = capsys.readouterr().err
    assert "changed before transition; Enter ignored" in message
    assert "press Enter again" in message
    state.received_monotonic_ns = time.monotonic_ns()
    assert not skill.is_complete  # Restoring state is not a new Enter.
    assert "Enter ignored" not in capsys.readouterr().err


# ------------------------------------------------ 開始姿勢に届いたかで問いを変える (2026-09-25)
#
# 準備動作が時間切れでも run は先へ進む。GB10 の Stage 5 (模擬の PC2 は指令に従わない) で、
# 届いていないのに「initial arm/hand pose is reached」と問いが出た。


def _prompt_of(arrival_check) -> tuple[str, list]:
    prompts: list[str] = []
    seen: list[np.ndarray] = []
    done = threading.Event()

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        done.set()
        return ""

    def check(arm):
        seen.append(np.asarray(arm).copy())
        return arrival_check(arm)

    skill = OperatorConfirmationHoldSkill(
        name="operator_gate_for_flip_table",
        next_skill_name="flip_table",
        input_fn=input_fn,
        discard_pending_input_fn=lambda: None,
        arrival_check=check,
    )
    skill.start({})
    held = skill.step(_obs())
    assert done.wait(timeout=1.0)
    np.testing.assert_allclose(seen[0], held)  # 判定は保持している実測の腕
    return prompts[0], seen


def test_the_prompt_says_reached_only_when_the_check_passes() -> None:
    prompt, _ = _prompt_of(lambda _arm: (True, "worst=L.elbow error=0.040rad"))
    assert "initial arm/hand pose is reached (worst=L.elbow error=0.040rad)" in prompt
    assert "NOT" not in prompt


def test_the_prompt_warns_when_the_pose_was_not_reached() -> None:
    prompt, _ = _prompt_of(lambda _arm: (False, "worst=R.shoulder_pitch error=0.979rad"))
    assert prompt.startswith("[gate] WARNING: flip_table initial arm pose is NOT reached")
    assert "worst=R.shoulder_pitch error=0.979rad" in prompt
    assert "is reached" not in prompt


def test_joint_space_arrival_names_the_worst_joint() -> None:
    from inference.desktop.lower_policy.initial_pose import ARM_JOINT_ORDER
    from inference.desktop.lower_policy.skills.operator_gate import joint_space_arrival

    target = np.zeros(14)
    measured = np.zeros(14)
    measured[3] = 0.09
    reached, detail = joint_space_arrival(measured, target, 0.10, ARM_JOINT_ORDER)
    assert reached and "worst=L.elbow error=0.090rad" in detail
    measured[7] = -0.2
    reached, detail = joint_space_arrival(measured, target, 0.10, ARM_JOINT_ORDER)
    assert not reached and "worst=R.shoulder_pitch error=0.200rad" in detail


def test_live_state_reader_failure_is_reported_without_starting() -> None:
    def broken_state():
        raise ValueError("malformed state")

    skill = OperatorConfirmationHoldSkill(
        name="operator_gate_for_flip_table", next_skill_name="flip_table",
        input_fn=lambda _: "", discard_pending_input_fn=lambda: None,
        require_arrival=True,
    )
    skill.require_fresh_state(broken_state, max_age_s=0.5, max_speed_rad_s=0.05)
    skill.start({})
    skill._read_confirmation("", skill._generation)
    assert not skill.is_complete
    assert "malformed state" in skill.failure_reason


def test_gate_holds_the_last_command_not_the_sagging_measurement() -> None:
    """保持 target は最後に送った指令 (Issue #172)。

    実測を保持すると、WBC は重力で遅れて止まる実測をさらに目標にするので腕が下がり、
    開始姿勢の到達判定を満たせなくなる。
    """
    commanded = np.full(14, 0.3)
    skill = OperatorConfirmationHoldSkill(
        name="operator_gate_for_insert_table_leg",
        next_skill_name="insert_table_leg",
        input_fn=lambda _prompt: threading.Event().wait(1.0),
        discard_pending_input_fn=lambda: None,
        hold_arm_target_provider=lambda: commanded,
    )
    skill.start({})
    np.testing.assert_allclose(skill.step(_obs()), commanded)
    np.testing.assert_allclose(skill.step(_obs()), commanded)


def test_gate_falls_back_to_the_measurement_without_a_command() -> None:
    skill = OperatorConfirmationHoldSkill(
        name="operator_gate_for_insert_table_leg",
        next_skill_name="insert_table_leg",
        input_fn=lambda _prompt: threading.Event().wait(1.0),
        discard_pending_input_fn=lambda: None,
        hold_arm_target_provider=lambda: None,
    )
    skill.start({})
    np.testing.assert_allclose(skill.step(_obs())[:2], np.linspace(-0.7, 0.7, 14)[:2])


def test_ignored_enter_names_the_worst_joint_and_reprompts(capsys) -> None:
    """到達前の Enter は理由 (一番ずれた関節) を出して捨て、次の問いも今の実測で作る。"""
    reached = {"ok": False}
    prompts: list[str] = []
    second = threading.Event()

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        if len(prompts) == 2:
            second.set()
        return ""

    def check(_arm):
        if reached["ok"]:
            return True, "worst=L.elbow error=0.030rad"
        return False, "worst=R.shoulder_pitch error=0.140rad"

    skill = OperatorConfirmationHoldSkill(
        name="operator_gate_for_insert_table_leg",
        next_skill_name="insert_table_leg",
        input_fn=input_fn,
        discard_pending_input_fn=lambda: None,
        arrival_check=check,
        require_arrival=True,
    )
    skill.start({})
    skill.step(_obs())
    assert second.wait(timeout=1.0)
    reached["ok"] = True
    deadline = time.monotonic() + 1.0
    while not skill.is_complete and time.monotonic() < deadline:
        skill.step(_obs())
        time.sleep(0.001)
    err = capsys.readouterr().err
    assert "initial pose not reached (worst=R.shoulder_pitch error=0.140rad); Enter ignored" in err
    assert "worst=R.shoulder_pitch error=0.140rad" in prompts[1]
    assert skill.owns_operator_view


def test_after_enter_the_gate_holds_until_the_leave_barrier_opens(capsys) -> None:
    """読みかけの model があれば、Enter の後もそれが終わるまで動き出さない (Issue #188)。"""
    barrier = {"open": False}
    skill = OperatorConfirmationHoldSkill(
        name="operator_gate_for_pick_table_leg",
        next_skill_name="pick_table_leg",
        input_fn=lambda _prompt: "",
    )
    skill.set_leave_barrier(lambda: barrier["open"], description="the model load finishes")
    skill.start({})
    held = skill.step(_obs())
    deadline = time.monotonic() + 1.0
    while not skill._confirmed.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert skill._confirmed.is_set()
    assert not skill.is_complete
    assert not skill.is_complete  # 待つ間は保持を続ける (1 回だけ知らせる)
    np.testing.assert_allclose(skill.step(_obs()), held)
    assert capsys.readouterr().err.count("holding until the model load finishes") == 1
    barrier["open"] = True
    assert skill.is_complete


def test_the_leave_barrier_is_not_asked_before_enter() -> None:
    asked = []
    release = threading.Event()
    skill = OperatorConfirmationHoldSkill(
        name="retry_wait_pick_table_leg",
        next_skill_name="retry_arm_pick_table_leg",
        input_fn=lambda _prompt: release.wait(1.0) and "",
    )
    skill.set_leave_barrier(lambda: asked.append(1) or True, description="x")
    skill.start({})
    skill.step(_obs())
    assert not skill.is_complete
    assert asked == []  # Enter の前は聞かない (聞くと読み込みが止まる)
    release.set()
    deadline = time.monotonic() + 1.0
    while not skill.is_complete and time.monotonic() < deadline:
        time.sleep(0.001)
    assert skill.is_complete and asked


class _Choice:
    keys = ("0", "1")

    def __init__(self, accept: dict[str, bool]) -> None:
        self.accept = accept
        self.asked = []

    def choose(self, key):
        self.asked.append(key)
        return self.accept[key], f"[model] chose {key}"


def _wait_until_complete(skill, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not skill.is_complete and time.monotonic() < deadline:
        time.sleep(0.001)
    return skill.is_complete


def test_a_digit_that_is_accepted_proceeds_like_r(capsys) -> None:
    keys = iter(["1"])
    choice = _Choice({"0": True, "1": True})
    skill = OperatorConfirmationHoldSkill(
        name="retry_wait_rotate_table_base",
        next_skill_name="retry_arm_rotate_table_base",
        cancellable_input_fn=lambda _prompt, _cancel: next(keys),
        choice=choice,
    )
    skill.start({})
    skill.step(_obs())
    assert _wait_until_complete(skill)
    assert choice.asked == ["1"]
    assert "[model] chose 1" in capsys.readouterr().err


def test_a_digit_that_is_refused_keeps_waiting_until_r() -> None:
    keys = iter(["1", "1", "r"])
    choice = _Choice({"0": True, "1": False})
    skill = OperatorConfirmationHoldSkill(
        name="retry_wait_rotate_table_base",
        next_skill_name="retry_arm_rotate_table_base",
        cancellable_input_fn=lambda _prompt, _cancel: next(keys),
        choice=choice,
    )
    skill.start({})
    skill.step(_obs())
    assert _wait_until_complete(skill)
    assert choice.asked == ["1", "1"]  # 読み込み中で断られた後、R でいまの model のまま進む


def test_a_cancelled_enter_tells_the_barrier_to_stay() -> None:
    """Enter の後に到達・静止が崩れて確認を取り消したら cancel_fn を呼ぶ (LoadPlan.stay、Issue #188)。"""
    stays = []
    fresh = {"ok": True}
    names = tuple(f"unused_{i}" for i in range(15)) + tuple(ROS_ARM_JOINT_NAMES)

    def state():
        speed = 0.0 if fresh["ok"] else 1.0
        return SimpleNamespace(
            name=names,
            position=tuple(0.0 for _ in names),
            velocity=tuple(speed for _ in names),
            received_monotonic_ns=time.monotonic_ns(),
        )

    skill = OperatorConfirmationHoldSkill(
        name="operator_gate_for_pick_table_leg",
        next_skill_name="pick_table_leg",
        input_fn=lambda _prompt: "",
        require_arrival=True,
    )
    skill.require_fresh_state(state, max_age_s=1.0, max_speed_rad_s=0.1)
    skill.set_leave_barrier(lambda: False, description="x", cancel_fn=lambda: stays.append(1))
    skill.start({})
    skill.step(_obs())
    deadline = time.monotonic() + 1.0
    while not skill._confirmed.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert skill._confirmed.is_set() and not skill.is_complete and stays == []
    fresh["ok"] = False  # 腕が動いた = 到達・静止が崩れた
    assert not skill.is_complete
    assert stays == [1]
