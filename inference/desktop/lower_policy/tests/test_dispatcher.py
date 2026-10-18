"""SkillDispatchLowerPolicy の unit tests。

dispatch routing の契約:
  - registry key と skill.name の一致を fail-fast (init)
  - 同 skill 再 start は no-op (caller から見て idempotent)
  - 別 skill への遷移は「旧を stop → 新を start」の順
  - DONE_SENTINEL は現 active を stop
  - 未知 skill 名は KeyError
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from inference.desktop.lower_policy.actuators.mock import MockWalkActuator
from inference.desktop.lower_policy.dispatcher import (
    DONE_SENTINEL,
    SkillDispatchLowerPolicy,
)
from inference.desktop.lower_policy.skills.base import Skill
from inference.desktop.lower_policy.skills.mock import MockSkill
from inference.desktop.lower_policy.skills.move_to_table import MoveToTable


class DummySkill(Skill):
    """Dispatch routing の test 用 stub。actuator に依存せず、event ログのみ。"""

    def __init__(self, skill_name: str) -> None:
        super().__init__()
        self.name = skill_name  # type: ignore[misc]
        self.events: list[str] = []

    def _on_start(self, params: dict) -> None:
        self.events.append("start")

    def _on_stop(self) -> None:
        self.events.append("stop")


def make_dispatcher() -> tuple[SkillDispatchLowerPolicy, MockWalkActuator]:
    """MoveToTable(vx=0.1) を積んだ dispatcher と actuator の pair を返す。"""
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator, vx=0.1)
    return SkillDispatchLowerPolicy({"move_to_table": skill}), actuator


def test_registry_key_must_match_skill_name():
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator)  # skill.name = "move_to_table"
    with pytest.raises(ValueError, match="skill.name mismatch"):
        SkillDispatchLowerPolicy({"other_name": skill})


def test_start_dispatches_to_skill():
    policy, actuator = make_dispatcher()
    policy.start("move_to_table", {})
    assert actuator.events == [("set_velocity", 0.1, 0.0, 0.0)]
    assert policy.active_skill_name == "move_to_table"


def test_active_skill_returns_instance(monkeypatch):
    """dispatcher.active_skill が Skill instance を返す (Issue #81、Orchestrator が
    Skill.max_dwell_sec を lookup するために使う)。"""
    from inference.desktop.lower_policy.skills.mock import MockSkill
    from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy

    skill_a = MockSkill("skill_a", max_dwell_sec=1.5)
    skill_b = MockSkill("skill_b")  # None
    policy = SkillDispatchLowerPolicy({"skill_a": skill_a, "skill_b": skill_b})
    # 未 start
    assert policy.active_skill is None
    # start 後
    policy.start("skill_a", {})
    assert policy.active_skill is skill_a
    assert policy.active_skill.max_dwell_sec == 1.5
    # switch
    policy.start("skill_b", {})
    assert policy.active_skill is skill_b
    assert policy.active_skill.max_dwell_sec is None
    # stop
    policy.stop()
    assert policy.active_skill is None


def test_stop_after_start():
    policy, actuator = make_dispatcher()
    policy.start("move_to_table", {})
    policy.stop()
    assert actuator.events == [
        ("set_velocity", 0.1, 0.0, 0.0),
        ("set_velocity", 0.0, 0.0, 0.0, 1.0),
    ]
    assert policy.active_skill_name is None


def test_stop_without_active_is_noop():
    policy, actuator = make_dispatcher()
    policy.stop()
    assert actuator.events == []
    assert policy.active_skill_name is None


def test_restart_same_skill_is_noop():
    policy, actuator = make_dispatcher()
    policy.start("move_to_table", {})
    n = len(actuator.events)
    policy.start("move_to_table", {})
    assert len(actuator.events) == n
    assert policy.active_skill_name == "move_to_table"


def test_done_sentinel_stops_active_skill():
    policy, actuator = make_dispatcher()
    policy.start("move_to_table", {})
    policy.start(DONE_SENTINEL, {})
    assert policy.active_skill_name is None
    # active skill stop の events もこの順で出ている
    assert actuator.events[-1:] == [("set_velocity", 0.0, 0.0, 0.0, 1.0)]


def test_done_sentinel_when_no_active_is_noop():
    policy, actuator = make_dispatcher()
    policy.start(DONE_SENTINEL, {})
    assert actuator.events == []
    assert policy.active_skill_name is None


def test_unknown_skill_raises_keyerror():
    policy, _ = make_dispatcher()
    with pytest.raises(KeyError, match="unknown skill"):
        policy.start("unknown_skill", {})


def test_unknown_skill_does_not_stop_active_skill():
    """caller が未知 skill を渡した時、現 active skill は stop されない
    (atomic 挙動)。KeyError 検証を「old skill を stop する前」に行うことで
    state 不一致を防ぐ。
    """
    skill_a = DummySkill("skill_a")
    policy = SkillDispatchLowerPolicy({"skill_a": skill_a})
    policy.start("skill_a", {})
    assert skill_a.events == ["start"]
    with pytest.raises(KeyError):
        policy.start("unknown_skill", {})
    # old skill は stop されていない
    assert skill_a.events == ["start"]
    assert policy.active_skill_name == "skill_a"


def test_transition_to_different_skill_stops_first():
    """A → B の遷移: A の stop → B の start がこの順で起きる。"""
    skill_a = DummySkill("skill_a")
    skill_b = DummySkill("skill_b")
    policy = SkillDispatchLowerPolicy({"skill_a": skill_a, "skill_b": skill_b})

    policy.start("skill_a", {})
    assert skill_a.events == ["start"]
    assert skill_b.events == []

    policy.start("skill_b", {})
    assert skill_a.events == ["start", "stop"]
    assert skill_b.events == ["start"]
    assert policy.active_skill_name == "skill_b"


# =========================================================================
# Skill.step + Dispatcher.step (Type A / Type B の pass-through 検証)
# =========================================================================
class _TypeBStub(Skill):
    """VLA / GR00T 系を模した stub。step で固定 action tensor を返す。"""

    def __init__(self, skill_name: str, action: np.ndarray) -> None:
        super().__init__()
        self.name = skill_name  # type: ignore[misc]
        self._action = action

    def _on_start(self, params: dict) -> None:
        pass

    def _on_stop(self) -> None:
        pass

    def step(self, obs: dict) -> Optional[np.ndarray]:
        return self._action


def test_dispatcher_step_returns_none_when_no_active():
    """未 start 時、Dispatcher.step は None を返す。"""
    policy = SkillDispatchLowerPolicy({"m": MockSkill("m")})
    assert policy.step({"any": "obs"}) is None


def test_dispatcher_step_type_a_returns_none():
    """Type A (Skill.step default): 自前で actuator を叩く前提、caller には None。"""
    policy = SkillDispatchLowerPolicy({"m": MockSkill("m")})
    policy.start("m", {})
    assert policy.step({"any": "obs"}) is None


def test_dispatcher_step_type_b_returns_action_tensor():
    """Type B (step override): action tensor を pass-through で返す。"""
    action = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    stub = _TypeBStub("vla_skill", action=action)
    policy = SkillDispatchLowerPolicy({"vla_skill": stub})
    policy.start("vla_skill", {})
    out = policy.step({"obs": "any"})
    assert out is not None
    np.testing.assert_array_equal(out, action)


def test_dispatcher_step_switches_after_transition():
    """Type B → Type A へ切替後、以降 step は None (Type A の default)。"""
    action = np.array([1.0], dtype=np.float32)
    type_b = _TypeBStub("skill_b", action=action)
    type_a = MockSkill("skill_a")
    policy = SkillDispatchLowerPolicy({"skill_a": type_a, "skill_b": type_b})

    policy.start("skill_b", {})
    np.testing.assert_array_equal(policy.step({}), action)

    policy.start("skill_a", {})
    assert policy.step({}) is None


# =========================================================================
# Action history buffer (skill 遷移時 snapshot 用)
# =========================================================================
class TestActionHistory:
    """SkillDispatchLowerPolicy._action_history の契約。"""

    def test_action_history_empty_before_any_step(self):
        policy = SkillDispatchLowerPolicy({"m": MockSkill("m")})
        assert policy.action_history == []

    def test_type_a_step_none_does_not_add_to_history(self):
        """Type A (step が None) は action_history に何も足さない。"""
        policy = SkillDispatchLowerPolicy({"m": MockSkill("m")})
        policy.start("m", {})
        for _ in range(5):
            policy.step({})
        assert policy.action_history == []

    def test_type_b_step_action_appended_to_history(self):
        """Type B (action tensor 返す) は毎 step で history に append。"""
        action = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        stub = _TypeBStub("vla", action=action)
        policy = SkillDispatchLowerPolicy({"vla": stub})
        policy.start("vla", {})

        for _ in range(3):
            policy.step({})

        history = policy.action_history
        assert len(history) == 3
        for hist_a in history:
            np.testing.assert_array_equal(hist_a, action)

    def test_action_history_maxlen_drops_oldest(self):
        """maxlen 超えたら古いのが drop される (ring buffer 挙動)。"""
        action = np.array([1.0], dtype=np.float32)
        stub = _TypeBStub("vla", action=action)
        policy = SkillDispatchLowerPolicy(
            {"vla": stub}, action_history_maxlen=5
        )
        policy.start("vla", {})

        for _ in range(10):  # maxlen=5 を超えて 10 tick 回す
            policy.step({})

        assert len(policy.action_history) == 5

    def test_action_history_stores_defensive_copy(self):
        """呼び出し側が返り値を in-place 変更しても history は汚染されない。"""
        action = np.array([1.0, 2.0], dtype=np.float32)
        stub = _TypeBStub("vla", action=action)
        policy = SkillDispatchLowerPolicy({"vla": stub})
        policy.start("vla", {})

        out = policy.step({})
        out[0] = 999.0  # 返り値を破壊

        # history 側は元の値のまま (copy されたはず)
        history = policy.action_history
        assert len(history) == 1
        np.testing.assert_array_equal(history[0], [1.0, 2.0])

    def test_action_history_property_returns_new_list(self):
        """property は毎回 new list を返す (外部変更で内部 deque が汚染されない)。"""
        action = np.array([1.0], dtype=np.float32)
        stub = _TypeBStub("vla", action=action)
        policy = SkillDispatchLowerPolicy({"vla": stub})
        policy.start("vla", {})
        policy.step({})

        h1 = policy.action_history
        h1.append(np.array([99.0]))  # 外部で変更

        h2 = policy.action_history
        assert len(h2) == 1  # 内部 deque は影響を受けない


class TestRestoreActionHistory:
    """resume 経路で action history を snapshot から復元する contract。"""

    def test_restore_overwrites_existing_history(self):
        """既に step で埋まってた history を restore で上書きする。"""
        action = np.array([1.0], dtype=np.float32)
        stub = _TypeBStub("vla", action=action)
        policy = SkillDispatchLowerPolicy({"vla": stub})
        policy.start("vla", {})
        policy.step({})
        policy.step({})
        assert len(policy.action_history) == 2

        # restore で全 clear + 新規 埋め
        restored = [np.array([9.0, 9.0], dtype=np.float32)]
        policy.restore_action_history(restored)

        history = policy.action_history
        assert len(history) == 1
        np.testing.assert_array_equal(history[0], [9.0, 9.0])

    def test_restore_truncates_to_maxlen(self):
        """maxlen を超える list を渡しても deque が自動 truncate (先頭側 drop)。"""
        policy = SkillDispatchLowerPolicy(
            {"m": MockSkill("m")}, action_history_maxlen=3
        )
        actions = [np.array([float(i)], dtype=np.float32) for i in range(5)]
        policy.restore_action_history(actions)

        history = policy.action_history
        assert len(history) == 3
        # 最新 3 個 (index 2, 3, 4) が残る
        np.testing.assert_array_equal(history[0], [2.0])
        np.testing.assert_array_equal(history[-1], [4.0])

    def test_restore_empty_list_clears_history(self):
        """empty list で restore すると history は空になる。"""
        action = np.array([1.0], dtype=np.float32)
        stub = _TypeBStub("vla", action=action)
        policy = SkillDispatchLowerPolicy({"vla": stub})
        policy.start("vla", {})
        policy.step({})
        assert len(policy.action_history) == 1

        policy.restore_action_history([])
        assert policy.action_history == []

    def test_restore_defensive_copy_protects_from_mutation(self):
        """restore に渡した list の要素を後で変更しても history は汚染されない。"""
        policy = SkillDispatchLowerPolicy({"m": MockSkill("m")})
        original = np.array([1.0, 2.0], dtype=np.float32)
        policy.restore_action_history([original])

        original[0] = 999.0  # 外部で破壊

        history = policy.action_history
        np.testing.assert_array_equal(history[0], [1.0, 2.0])


# ---- R の後に model を選ぶ (Issue #188 ② 段 3) ----


def test_replace_skill_swaps_the_next_start_and_keeps_the_name():
    main, candidate = DummySkill("rotate_table_base"), DummySkill("rotate_table_base")
    dispatcher = SkillDispatchLowerPolicy({"rotate_table_base": main, "gate": DummySkill("gate")})
    dispatcher.start("gate", {})
    dispatcher.replace_skill("rotate_table_base", candidate)
    dispatcher.start("rotate_table_base", {})
    assert dispatcher.active_skill is candidate
    assert candidate.events == ["start"] and main.events == []


def test_replace_skill_refuses_the_running_skill_and_a_wrong_name():
    main = DummySkill("rotate_table_base")
    dispatcher = SkillDispatchLowerPolicy({"rotate_table_base": main})
    dispatcher.start("rotate_table_base", {})
    with pytest.raises(RuntimeError, match="while it is running"):
        dispatcher.replace_skill("rotate_table_base", DummySkill("rotate_table_base"))
    with pytest.raises(ValueError, match="skill.name mismatch"):
        dispatcher.replace_skill("rotate_table_base", DummySkill("pick_table_leg"))
    with pytest.raises(KeyError):
        dispatcher.replace_skill("flip_table", DummySkill("flip_table"))
    assert dispatcher.active_skill is main
