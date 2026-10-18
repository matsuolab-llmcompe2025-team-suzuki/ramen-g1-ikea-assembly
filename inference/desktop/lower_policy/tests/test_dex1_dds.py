"""Dex1DdsGripper の unit test (Issue #123)。

SDK / cyclonedds install 不要な形で書く: publisher / subscriber / cmd factory を
inject して payload 構築・安全クランプ・状態取り込み・lifecycle を検証する
(`test_g1_arm_sdk.py` と同じ流儀)。
"""

from __future__ import annotations

import types
import time
from typing import Any, Callable

import pytest

from inference.desktop.lower_policy.actuators.dex1 import (
    Dex1Gripper,
    GraspState,
    classify_grasp,
)
from inference.desktop.lower_policy.actuators.dex1_dds import (
    DEFAULT_CMD_TOPIC,
    DEFAULT_STATE_TOPIC,
    Dex1DdsGripper,
    Dex1Gains,
    Dex1Limits,
)
from inference.desktop.lower_policy.kinematics.types import Side


class _FakePublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.written: list[Any] = []
        self.closed = False

    def Write(self, msg: Any) -> None:
        self.written.append(msg)

    def Close(self) -> None:
        self.closed = True


class _FakeSubscriber:
    def __init__(self, topic: str, handler: Callable[[Any], None]) -> None:
        self.topic = topic
        self.handler = handler
        self.closed = False

    def Close(self) -> None:
        self.closed = True


def _fake_cmd() -> Any:
    motor = types.SimpleNamespace(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=0)
    return types.SimpleNamespace(cmds=[motor])


def _state_msg(q: float) -> Any:
    return types.SimpleNamespace(states=[types.SimpleNamespace(q=q)])


def build(**kwargs: Any):
    pubs: dict[str, _FakePublisher] = {}
    subs: dict[str, _FakeSubscriber] = {}

    def pf(topic: str) -> _FakePublisher:
        pubs[topic] = _FakePublisher(topic)
        return pubs[topic]

    def sf(topic: str, handler: Callable[[Any], None]) -> _FakeSubscriber:
        subs[topic] = _FakeSubscriber(topic, handler)
        return subs[topic]

    g = Dex1DdsGripper(
        publisher_factory=pf,
        subscriber_factory=sf,
        cmd_factory=_fake_cmd,
        continuous_publish_hz=None,
        **kwargs,
    )
    return g, pubs, subs


# ---------------------------------------------------------------- topics


def test_uses_official_dex1_topics():
    _, pubs, subs = build()
    assert set(pubs) == {"rt/dex1/left/cmd", "rt/dex1/right/cmd"}
    assert set(subs) == {"rt/dex1/left/state", "rt/dex1/right/state"}
    assert DEFAULT_CMD_TOPIC[Side.RIGHT] == "rt/dex1/right/cmd"
    assert DEFAULT_STATE_TOPIC[Side.LEFT] == "rt/dex1/left/state"


def test_topic_override():
    g, pubs, subs = build(
        cmd_topic={Side.LEFT: "a", Side.RIGHT: "b"},
        state_topic={Side.LEFT: "c", Side.RIGHT: "d"},
    )
    assert set(pubs) == {"a", "b"}
    assert set(subs) == {"c", "d"}


def test_rejects_incomplete_topic_mapping():
    with pytest.raises(ValueError, match="both sides"):
        build(cmd_topic={Side.LEFT: "a"})


# ---------------------------------------------------------------- command


def test_command_publishes_to_the_matching_side():
    g, pubs, _ = build()
    g.command(Side.RIGHT, 4.5)
    assert len(pubs["rt/dex1/right/cmd"].written) == 1
    assert len(pubs["rt/dex1/left/cmd"].written) == 0
    assert pubs["rt/dex1/right/cmd"].written[0].cmds[0].q == 4.5


def test_command_payload_carries_gains_and_zero_velocity():
    g, pubs, _ = build(gains=Dex1Gains(kp=7.0, kd=0.25))
    g.command(Side.LEFT, 1.5)
    m = pubs["rt/dex1/left/cmd"].written[0].cmds[0]
    assert (m.q, m.dq, m.tau, m.kp, m.kd) == (1.5, 0.0, 0.0, 7.0, 0.25)


def test_runtime_sdk_factory_builds_the_required_single_motor_slot():
    """Exercise the real SDK message factory while keeping DDS fully mocked."""
    # 本物の SDK の message を作るので、SDK の無い default 環境では skip する (Issue #192)
    pytest.importorskip("unitree_sdk2py")
    pubs: dict[str, _FakePublisher] = {}

    def pf(topic: str) -> _FakePublisher:
        pubs[topic] = _FakePublisher(topic)
        return pubs[topic]

    gripper = Dex1DdsGripper(
        publisher_factory=pf,
        subscriber_factory=lambda topic, handler: _FakeSubscriber(topic, handler),
        cmd_factory=None,
        continuous_publish_hz=None,
    )
    gripper.command(Side.LEFT, 1.25)

    message = pubs[DEFAULT_CMD_TOPIC[Side.LEFT]].written[0]
    assert len(message.cmds) == 1
    assert message.cmds[0].mode == 1
    assert message.cmds[0].q == pytest.approx(1.25)


@pytest.mark.parametrize("raw,expected", [(-3.0, 0.0), (0.0, 0.0), (2.2, 2.2), (9.9, 4.5)])
def test_command_is_clamped_to_safe_range(raw, expected):
    g, pubs, _ = build()
    g.command(Side.RIGHT, raw)
    assert pubs["rt/dex1/right/cmd"].written[0].cmds[0].q == expected
    assert g.last_command(Side.RIGHT) == expected


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), float("-inf")])
def test_command_rejects_non_finite_positions(raw):
    g, pubs, _ = build()
    with pytest.raises(ValueError, match="finite"):
        g.command(Side.RIGHT, raw)
    assert pubs[DEFAULT_CMD_TOPIC[Side.RIGHT]].written == []


def test_continuous_publisher_repeats_latest_target_and_stops_cleanly():
    pubs: dict[str, _FakePublisher] = {}
    subs: dict[str, _FakeSubscriber] = {}

    def pf(topic: str) -> _FakePublisher:
        pubs[topic] = _FakePublisher(topic)
        return pubs[topic]

    def sf(topic: str, handler: Callable[[Any], None]) -> _FakeSubscriber:
        subs[topic] = _FakeSubscriber(topic, handler)
        return subs[topic]

    g = Dex1DdsGripper(
        publisher_factory=pf,
        subscriber_factory=sf,
        cmd_factory=_fake_cmd,
        continuous_publish_hz=500.0,
    )
    g.command(Side.LEFT, 1.25)
    deadline = time.monotonic() + 0.2
    while len(pubs[DEFAULT_CMD_TOPIC[Side.LEFT]].written) < 3:
        if time.monotonic() >= deadline:
            pytest.fail("continuous publisher did not repeat the latest target")
        time.sleep(0.002)
    g.close()
    count = len(pubs[DEFAULT_CMD_TOPIC[Side.LEFT]].written)
    time.sleep(0.01)
    assert len(pubs[DEFAULT_CMD_TOPIC[Side.LEFT]].written) == count
    assert all(msg.cmds[0].q == pytest.approx(1.25) for msg in pubs[DEFAULT_CMD_TOPIC[Side.LEFT]].written)


def test_custom_limits_are_honored():
    g, pubs, _ = build(limits=Dex1Limits(min_position=1.0, max_position=3.0))
    g.command(Side.RIGHT, 0.0)
    assert pubs["rt/dex1/right/cmd"].written[0].cmds[0].q == 1.0


def test_limits_reject_inverted_range():
    with pytest.raises(ValueError, match="min_position < max_position"):
        Dex1Limits(min_position=4.0, max_position=1.0)


def test_last_command_is_none_before_any_command():
    g, _, _ = build()
    assert g.last_command(Side.LEFT) is None
    assert g.last_command(Side.RIGHT) is None


def test_build_command_rejects_empty_message():
    g, _, _ = build()
    g._cmd_factory = lambda: types.SimpleNamespace(cmds=[])  # noqa: SLF001
    with pytest.raises(RuntimeError, match="without 'cmds'"):
        g.build_command(1.0)


# ---------------------------------------------------------------- state


def test_state_arrives_through_the_subscriber_handler():
    g, _, subs = build()
    assert g.read(Side.RIGHT) is None
    subs["rt/dex1/right/state"].handler(_state_msg(2.23))
    assert g.read(Side.RIGHT) == 2.23
    assert g.read(Side.LEFT) is None


def test_state_is_latest_only():
    g, _, subs = build()
    for v in (4.5, 3.1, 2.2):
        subs["rt/dex1/left/state"].handler(_state_msg(v))
    assert g.read(Side.LEFT) == 2.2


@pytest.mark.parametrize("msg", [
    types.SimpleNamespace(states=[]),
    types.SimpleNamespace(states=None),
    types.SimpleNamespace(),
    types.SimpleNamespace(states=[types.SimpleNamespace()]),
    types.SimpleNamespace(states=[types.SimpleNamespace(q="nope")]),
    types.SimpleNamespace(states=[types.SimpleNamespace(q=float("nan"))]),
    types.SimpleNamespace(states=[types.SimpleNamespace(q=float("inf"))]),
])
def test_malformed_state_is_ignored_without_raising(msg):
    """壊れた sample で制御ループを落とさない (未受信と同じ扱い)。"""
    g, _, subs = build()
    subs["rt/dex1/right/state"].handler(_state_msg(2.2))
    subs["rt/dex1/right/state"].handler(msg)
    assert g.read(Side.RIGHT) == 2.2   # 直前の正常値を保つ


def test_end_to_end_grasp_interlock_with_dds_layer():
    """全閉指令 → 脚が止める → HOLDING、が DDS 層越しに成立すること。"""
    g, _, subs = build()
    g.command(Side.RIGHT, 0.0)
    subs["rt/dex1/right/state"].handler(_state_msg(2.23))
    assert classify_grasp(g.last_command(Side.RIGHT), g.read(Side.RIGHT)) is GraspState.HOLDING

    # 空振りなら実測が全閉まで届く
    subs["rt/dex1/right/state"].handler(_state_msg(0.03))
    assert classify_grasp(g.last_command(Side.RIGHT), g.read(Side.RIGHT)) is GraspState.EMPTY_CLOSED


# ---------------------------------------------------------------- lifecycle


def test_close_is_idempotent_and_tears_down_handles():
    g, pubs, subs = build()
    g.close()
    assert g.closed
    assert all(p.closed for p in pubs.values())
    assert all(s.closed for s in subs.values())
    g.close()   # 2 回目は no-op


def test_command_after_close_raises():
    g, _, _ = build()
    g.close()
    with pytest.raises(RuntimeError, match="closed"):
        g.command(Side.RIGHT, 1.0)


def test_close_tolerates_handles_without_close_method():
    class Bare:
        def __init__(self, *a: Any) -> None:
            pass

        def Write(self, msg: Any) -> None:
            pass

    g = Dex1DdsGripper(
        publisher_factory=lambda t: Bare(),
        subscriber_factory=lambda t, h: Bare(),
        cmd_factory=_fake_cmd,
    )
    g.close()
    assert g.closed


def test_satisfies_protocol():
    g, _, _ = build()
    assert isinstance(g, Dex1Gripper)


# ---------------------------------------------------------------- config


def test_from_config_defaults_and_overrides():
    g = Dex1DdsGripper.from_config(
        {"gains": {"kp": 8.0}, "limits": {"max_position": 4.0}},
        publisher_factory=lambda t: _FakePublisher(t),
        subscriber_factory=lambda t, h: _FakeSubscriber(t, h),
        cmd_factory=_fake_cmd,
        continuous_publish_hz=None,
    )
    assert g.build_command(1.0).cmds[0].kp == 8.0
    assert g.build_command(1.0).cmds[0].kd == Dex1Gains().kd
    g.command(Side.RIGHT, 9.0)
    assert g.last_command(Side.RIGHT) == 4.0


@pytest.mark.parametrize("cfg,match", [
    ({"bogus": 1}, "unknown key"),
    ({"gains": {"kx": 1}}, "unknown key"),
    ({"limits": {"lo": 1}}, "unknown key"),
    ({"gains": []}, "must be a mapping"),
])
def test_from_config_rejects_invalid(cfg, match):
    with pytest.raises(ValueError, match=match):
        Dex1DdsGripper.from_config(
            cfg,
            publisher_factory=lambda t: _FakePublisher(t),
            subscriber_factory=lambda t, h: _FakeSubscriber(t, h),
            cmd_factory=_fake_cmd,
        )


def test_module_imports_without_sdk():
    """default env から import できること (lazy import 方針の確認)。"""
    import importlib

    mod = importlib.import_module(
        "inference.desktop.lower_policy.actuators.dex1_dds"
    )
    assert hasattr(mod, "Dex1DdsGripper")


# ---------------------------------------------------------------- 速度制限
#
# 自前機のグリッパには制限が無く、実測 5.43 unit/s (WandB の 09-20 pick run) で
# 動いていた。学習データの実測最速は 4.20 unit/s (0.30 秒窓) なので、**学習より
# 1.3 倍速い条件で評価してきた**ことになる。policy は学習時の速度で腕とグリッパの
# タイミングを学んでいるので、そこへ揃える (Issue #154)。
#
# 実装は運営の `run_wbc_with_dex1.py` の `Dex1Injector.apply` と同じ形にしてある。


def _q(pub: _FakePublisher) -> list[float]:
    return [m.cmds[0].q for m in pub.written]


def _at(monkeypatch, now: list[float]) -> None:
    import inference.desktop.lower_policy.actuators.dex1_dds as mod

    monkeypatch.setattr(mod.time, "monotonic", lambda: now[0])


def test_the_first_command_is_published_as_is(monkeypatch):
    """追従元が無い状態で slew しても意味が無い。運営の実装も初回は即採用。"""
    now = [100.0]
    _at(monkeypatch, now)
    g, pubs, _ = build(limits=Dex1Limits(max_speed=4.2))
    g.command(Side.LEFT, 0.0)  # 全開 4.5 から全閉 0.0 へ、いきなり
    assert _q(pubs["rt/dex1/left/cmd"]) == [0.0]


def test_a_later_jump_is_rate_limited(monkeypatch):
    now = [100.0]
    _at(monkeypatch, now)
    g, pubs, _ = build(limits=Dex1Limits(max_speed=4.2))
    g.command(Side.LEFT, 4.5)  # 初回 = そのまま
    now[0] += 0.05  # 50ms 後
    g.command(Side.LEFT, 0.0)  # 全閉を要求
    sent = _q(pubs["rt/dex1/left/cmd"])
    assert sent[0] == 4.5
    # 1 tick で動けるのは 4.2 * 0.05 = 0.21 だけ
    assert sent[1] == pytest.approx(4.5 - 0.21)


def test_repeated_calls_converge_to_the_target(monkeypatch):
    now = [100.0]
    _at(monkeypatch, now)
    g, pubs, _ = build(limits=Dex1Limits(max_speed=4.2))
    g.command(Side.LEFT, 4.5)
    for _ in range(60):  # 60 * 50ms = 3s。全ストローク 4.5/4.2 = 1.07s より十分長い
        now[0] += 0.05
        g.command(Side.LEFT, 0.0)
    assert _q(pubs["rt/dex1/left/cmd"])[-1] == pytest.approx(0.0, abs=1e-9)


def test_last_command_is_the_target_and_current_is_what_ships(monkeypatch):
    """把持判定は `current_command` を見ること (目標と比べると誤判定する)。"""
    now = [100.0]
    _at(monkeypatch, now)
    g, _, _ = build(limits=Dex1Limits(max_speed=4.2))
    g.command(Side.RIGHT, 4.5)
    now[0] += 0.05
    g.command(Side.RIGHT, 0.0)
    assert g.last_command(Side.RIGHT) == 0.0
    assert g.current_command(Side.RIGHT) == pytest.approx(4.5 - 0.21)


def test_max_speed_none_disables_the_limit(monkeypatch):
    now = [100.0]
    _at(monkeypatch, now)
    g, pubs, _ = build(limits=Dex1Limits(max_speed=None))
    g.command(Side.LEFT, 4.5)
    now[0] += 0.05
    g.command(Side.LEFT, 0.0)
    assert _q(pubs["rt/dex1/left/cmd"]) == [4.5, 0.0]


def test_the_hold_publisher_also_slews(monkeypatch):
    """200Hz の保持 publish が目標を素通しすると、制限が無意味になる。"""
    now = [100.0]
    _at(monkeypatch, now)
    g, pubs, _ = build(limits=Dex1Limits(max_speed=4.2))
    g.command(Side.LEFT, 4.5)
    now[0] += 0.05
    g.command(Side.LEFT, 0.0)
    before = g.current_command(Side.LEFT)
    now[0] += 0.05
    # `_publisher_loop` が 1 周する分だけ手で回す (スレッドは止めてある)
    with g._lock:  # noqa: SLF001 - 保持経路の slew を直接確かめる
        moved = g._slew(Side.LEFT, g._last_cmd[Side.LEFT])  # noqa: SLF001
    assert moved == pytest.approx(before - 0.21)
    assert moved > 0.0  # まだ目標には着いていない


def test_an_invalid_max_speed_is_rejected():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            Dex1Limits(max_speed=bad)


def test_the_default_matches_the_measured_training_ceiling():
    from inference.desktop.lower_policy.actuators.dex1_dds import (
        TRAIN_MAX_SPEED_UNITS_PER_S,
    )

    assert Dex1Limits().max_speed == TRAIN_MAX_SPEED_UNITS_PER_S == 4.2


# ---------------------------------------------------------------- 運営との一致
#
# 会場のグリッパは運営の injector が `rt/lowcmd` の slot 31/33 へ自分の kp/kd を
# 書くので、ここの値は **自前機だけ**に効く。合わせる理由は「自前機の評価が会場の
# 挙動を予測できるようにする」こと。ずれていると、速度のときと同じで
# **前の run と条件が違うのに気づけない**。
#
# 出典 (運営 interface package 2026.09.21):
#   tools/run_wbc_with_dex1.py   GRIPPER_KP = 5.0 / GRIPPER_KD = 0.05
#                                DEX1_CLOSED_Q = 0.0 / DEX1_OPEN_Q = -5.30
#                                GRIPPER_INDEX = {"left": 31, "right": 33}
#   tools/diagnose_dex1.py       "trust these, not the docs"


def test_gains_match_the_organizer_reference_calibration():
    """kp / kd を運営の reference calibration に固定する。

    kd はかつて 0.1 だった (2026-09-11 の暫定値、「実機で要調整」と書いたまま
    一度も調整されていなかった)。速度と違って **ゲインは学習が持っている量では
    ない** ので、会場の値が唯一の実行可能なアンカー (Issue #154)。
    """
    g = Dex1Gains()
    assert (g.kp, g.kd) == (5.0, 0.05), (
        f"運営の reference calibration (kp=5.0, kd=0.05) と違う: {g}。"
        " 会場は tools/run_wbc_with_dex1.py の GRIPPER_KP/KD を書き込むので、"
        " ここがずれていると自前機の評価が会場の挙動を予測しなくなる"
    )


def test_the_open_and_closed_ends_match_the_organizer():
    """model 空間の端点が運営の生モータ角と対応していること。

    `taskspace_adapter.dex1_model_to_taskspace` → 運営 `hand_norm_to_dex1_q` の
    往復が壊れていないかは `components/ramen` 側の test が見る。ここでは
    **自前機の可動域の端**だけを固定する。
    """
    lim = Dex1Limits()
    assert (lim.min_position, lim.max_position) == (0.0, 4.5)
