"""No-robot tests for the one-reader production key protocol."""

from __future__ import annotations

import os
import threading
import time

import pytest

from inference.desktop.operator_console import OperatorConsole


def test_replaced_waiter_cannot_consume_the_next_views_enter():
    console = OperatorConsole()
    outcome = []

    def wait():
        try:
            console.wait_for("enter", stage=5, phase="old gate")
            outcome.append("accepted")
        except EOFError:
            outcome.append("cancelled")

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    deadline = time.monotonic() + 1
    while console._view is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert console._view is not None
    console.show(5, "safety decision", ("enter",))
    console._events.put((console._generation, "enter"))
    thread.join(timeout=1)
    console._stop.set()
    assert outcome == ["cancelled"]
    assert console._events.get_nowait() == (console._generation, "enter")


def test_cancelled_waiter_cannot_replace_the_current_view():
    console = OperatorConsole()
    console.show(5, "safety decision", ("enter",))
    view = console._view
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(EOFError, match="cancelled"):
        console.wait_for("enter", stage=5, phase="old gate", cancel_event=cancel)
    assert console._view == view


def test_cancellation_does_not_consume_pending_input():
    console = OperatorConsole()
    cancel = threading.Event()
    outcome = []

    def wait():
        try:
            console.wait_for("enter", stage=5, phase="gate", cancel_event=cancel)
        except EOFError:
            outcome.append("cancelled")

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    deadline = time.monotonic() + 1
    while console._view is None and time.monotonic() < deadline:
        time.sleep(0.005)
    with console._lock:
        cancel.set()
        console._events.put((console._generation, "enter"))
    thread.join(timeout=1)
    console._stop.set()
    assert outcome == ["cancelled"]
    assert console._events.get_nowait() == (console._generation, "enter")


def _wait_key(console: OperatorConsole, timeout: float = 0.5):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        key = console.poll()
        if key is not None:
            return key
        time.sleep(0.005)
    return None


def test_single_tty_reader_rejects_unavailable_keys_and_queued_repeat(monkeypatch):
    master, slave = os.openpty()
    reader = os.fdopen(slave, "r", buffering=1)
    monkeypatch.setattr("sys.stdin", reader)
    console = OperatorConsole()
    try:
        console.start()
        console.show(2, "pick／初回把持", ("n", "r"))
        time.sleep(0.16)
        os.write(master, b"n")
        until = time.monotonic() + 0.5
        while not console.has_pending() and time.monotonic() < until:
            time.sleep(0.005)
        assert console.has_pending()
        assert _wait_key(console) == "n"
        assert not console.has_pending()
        os.write(master, b"rr")
        time.sleep(0.03)
        console.show(2, "pick／腕保持・ハンド全開", ("r",))
        assert console.poll() is None  # old R presses are never carried over
        time.sleep(0.36)
        os.write(master, b"r")
        assert _wait_key(console) == "r"
        console.show(2, "pick／開始待ち", ("enter",))
        os.write(master, b"nR")
        time.sleep(0.16)
        assert console.poll() is None
        os.write(master, b"\r")
        assert _wait_key(console) == "enter"
    finally:
        console.close()
        reader.close()
        os.close(master)


def test_terminal_disconnect_fails_closed(monkeypatch):
    master, slave = os.openpty()
    reader = os.fdopen(slave, "r", buffering=1)
    monkeypatch.setattr("sys.stdin", reader)
    console = OperatorConsole()
    try:
        console.start()
        console.show(1, "pick", ("n", "r"))
        os.close(master)
        until = time.monotonic() + 0.5
        while not console._stop.is_set() and time.monotonic() < until:
            time.sleep(0.005)
        try:
            console.poll()
        except EOFError:
            pass
        else:
            raise AssertionError("terminal disconnect must stop the run")
    finally:
        console.close()
        reader.close()


def test_safety_stop_view_says_what_enter_and_ctrl_c_do(capsys):
    from inference.desktop.operator_console import SAFETY_STOP_PHASE

    console = OperatorConsole()
    console.show(2, f"{SAFETY_STOP_PHASE}／保持中・判断待ち", ("enter",))
    err = capsys.readouterr().err
    assert "Enter 戻す（初期姿勢→ハンド全開→腕下ろし）" in err
    assert "Ctrl+C その場で終了" in err


def test_wait_for_prints_the_gate_detail_and_accepts_the_key(monkeypatch, capsys):
    master, slave = os.openpty()
    reader = os.fdopen(slave, "r", buffering=1)
    monkeypatch.setattr("sys.stdin", reader)
    console = OperatorConsole()
    try:
        assert not console.running
        console.start()
        assert console.running

        def _press():
            time.sleep(0.25)
            os.write(master, b"\n")

        import threading

        threading.Thread(target=_press, daemon=True).start()
        assert console.wait_for(
            "enter", stage=2, phase="insert_table_leg／開始待ち",
            detail="[gate] WARNING: ... NOT reached (worst=L.elbow error=0.120rad)",
        ) == "enter"
        assert "worst=L.elbow error=0.120rad" in capsys.readouterr().err
    finally:
        console.close()
        reader.close()
        os.close(master)


def test_callers_can_name_keys_the_table_does_not_know(capsys):
    """R の後に model を選ぶ数字キー (Issue #188)。呼び出し側の名前が既定の名前より優先する。"""
    console = OperatorConsole()
    console.show(
        2, "rotate_table_base／腕保持・ハンド全開", ("r", "0", "1"),
        {"r": "R 初期姿勢へ（a のまま）", "0": "0 a（既定）", "1": "1 b"},
    )
    out = capsys.readouterr().err
    assert "操作：R 初期姿勢へ（a のまま） ｜ 0 a（既定） ｜ 1 b ｜ Ctrl+C 終了" in out


def test_wait_for_any_passes_the_names_and_returns_a_digit():
    console = OperatorConsole()
    got = []
    thread = threading.Thread(
        target=lambda: got.append(
            console.wait_for_any(("r", "1"), stage=2, phase="x", labels={"1": "1 b"})
        ),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 1
    while console._view is None and time.monotonic() < deadline:
        time.sleep(0.005)
    console._events.put((console._generation, "1"))
    thread.join(timeout=1)
    console._stop.set()
    assert got == ["1"]
