"""ModelResidency の unit test (Issue #141 束 1-12 / D7-2)。

主眼:
    - GPU に置く数を N に保つ (範囲外は解放、範囲内でまだなら先読み)
    - N=1 は今までどおり「実行中の 1 個だけ」
    - 先読みは別 thread。失敗しても run を落とさない
"""

from __future__ import annotations

import threading
import time

import pytest

from inference.desktop.lower_policy.policies.deferred import DeferredPolicy
from inference.desktop.lower_policy.policies.residency import ModelResidency


class _StubPolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.prepared = 0
        self.closed = 0
        self.fail = fail
        self.loaded = threading.Event()

    def prepare(self) -> None:
        if self.fail:
            raise RuntimeError("boom")
        self.prepared += 1
        self.loaded.set()

    def close(self) -> None:
        self.closed += 1


def _residency(resident: int, *, order=("a", "b", "c"), fail_on=()):
    policies = {name: _StubPolicy(fail=name in fail_on) for name in order}
    return ModelResidency(list(order), policies, resident=resident), policies


def _settle(manager: ModelResidency) -> None:
    manager._queue.join()  # 積んだ読み込みが終わるのを待つ (worker は止めない)


def test_venue_never_loads_a_model_inside_preparation():
    policies = {name: _StubPolicy() for name in ("a", "b")}
    manager = ModelResidency(
        ["arm_pre_motion_for_a", "a", "arm_transition_a_to_b", "hold_transition_b", "b"],
        policies, resident=2, preparation_loads_only_at_hold=True,
    )
    try:
        manager.prime()
        _settle(manager)
        assert policies["a"].prepared == 1 and policies["b"].prepared == 0
        manager.on_skill_started("arm_pre_motion_for_a")
        manager.on_skill_started("arm_transition_a_to_b")
        _settle(manager)
        assert policies["b"].prepared == 0
        manager.on_skill_started("hold_transition_b")
        _settle(manager)
        assert policies["b"].prepared == 1
    finally:
        manager.close()


class TestResidentCount:
    def test_one_keeps_only_the_active_model(self):
        manager, policies = _residency(1)
        manager.on_skill_started("a")
        manager.on_skill_started("b")
        _settle(manager)
        assert policies["a"].closed == 1
        assert policies["b"].prepared == 1
        assert policies["c"].prepared == 0

    def test_slow_release_does_not_block_the_control_caller(self):
        manager, policies = _residency(1)
        manager.on_skill_started("a")
        _settle(manager)
        original_close = policies["a"].close

        def _slow_close() -> None:
            time.sleep(0.15)
            original_close()

        policies["a"].close = _slow_close  # type: ignore[method-assign]
        started = time.monotonic()
        manager.on_skill_started("b")
        elapsed = time.monotonic() - started

        assert elapsed < 0.05
        _settle(manager)
        assert policies["a"].closed == 1

    def test_two_preloads_the_next_model(self):
        manager, policies = _residency(2)
        manager.on_skill_started("a")
        _settle(manager)
        assert policies["a"].prepared == 1
        assert policies["b"].prepared == 1
        assert policies["c"].prepared == 0
        assert policies["a"].closed == 0

    def test_moving_on_releases_what_fell_out_of_range(self):
        manager, policies = _residency(2)
        manager.on_skill_started("a")
        manager.on_skill_started("b")
        _settle(manager)
        assert policies["a"].closed == 1
        assert manager.loaded == ("b", "c")

    def test_all_keeps_everything(self):
        manager, policies = _residency(3)
        manager.on_skill_started("a")
        _settle(manager)
        assert all(p.prepared == 1 for p in policies.values())
        assert all(p.closed == 0 for p in policies.values())

    def test_close_releases_what_it_loaded(self):
        """stage ごとに作り直すので、close は載せた分を解放する (review 指摘 9)。

        解放しないと、次の stage の `_loaded` が空のまま前の stage の model が
        GPU に残り、誰も解放できなくなる。
        """
        manager, policies = _residency(2)
        manager.on_skill_started("a")
        _settle(manager)
        assert manager.loaded == ("a", "b")
        manager.close()
        assert manager.loaded == ()
        assert policies["a"].closed == 1 and policies["b"].closed == 1
        assert policies["c"].closed == 0

    def test_zero_is_rejected(self):
        with pytest.raises(ValueError, match="resident must be >= 1"):
            _residency(0)

    def test_finite_transition_loads_only_after_frame_zero_hold(self):
        policies = {"a": _StubPolicy(), "b": _StubPolicy()}
        manager = ModelResidency(
            [
                "a",
                "hand_release_a_to_b",
                "arm_transition_a_to_b",
                "hand_transition_a_to_b",
                "hold_transition_a_to_b",
                "b",
            ],
            policies,
            resident=1,
        )
        manager.prime()

        manager.on_skill_started("hand_release_a_to_b")
        manager.on_skill_started("arm_transition_a_to_b")
        manager.on_skill_started("hand_transition_a_to_b")
        _settle(manager)
        assert policies["a"].closed == 0
        assert policies["b"].prepared == 0

        manager.on_skill_started("hold_transition_a_to_b")
        assert not manager.is_ready("b")
        _settle(manager)

        assert policies["a"].closed == 1
        assert policies["b"].prepared == 1
        assert manager.loaded == ("b",)
        assert manager.is_ready("b")

    def test_resident_two_does_not_cross_a_physical_transition(self):
        policies = {"a": _StubPolicy(), "b": _StubPolicy()}
        manager = ModelResidency(
            ["a", "hold_transition_a_to_b", "b"], policies, resident=2
        )
        manager.prime()
        _settle(manager)
        assert policies["a"].prepared == 1
        assert policies["b"].prepared == 0

    def test_release_failure_prevents_loading_the_next_model(self):
        policies = {"a": _StubPolicy(), "b": _StubPolicy()}
        manager = ModelResidency(
            ["a", "hold_transition_a_to_b", "b"], policies, resident=1
        )
        manager.prime()

        def _fail_close() -> None:
            raise RuntimeError("cannot release a")

        policies["a"].close = _fail_close  # type: ignore[method-assign]
        manager.on_skill_started("hold_transition_a_to_b")
        _settle(manager)

        assert policies["b"].prepared == 0
        assert "refusing to load" in (manager.error_for("b") or "")
        assert not manager.is_ready("b")


class TestPrime:
    def test_first_model_is_loaded_before_returning(self):
        manager, policies = _residency(2)
        manager.prime()
        assert policies["a"].prepared == 1  # 同期で読む
        _settle(manager)
        assert policies["b"].prepared == 1  # 残りは裏で

    def test_prime_without_any_model_is_a_no_op(self):
        manager = ModelResidency(["x"], {}, resident=2)
        manager.prime()
        assert manager.loaded == ()


class TestRobustness:
    def test_a_failed_preload_does_not_raise_and_is_retryable(self, capsys):
        manager, policies = _residency(2, fail_on=("b",))
        manager.on_skill_started("a")
        _settle(manager)
        assert "preload failed" in capsys.readouterr().err
        assert "b" not in manager.loaded  # 次に来たときもう一度読む

    def test_unknown_skill_is_ignored(self):
        manager, policies = _residency(2)
        manager.on_skill_started("hold_pose_for_x")
        manager.on_skill_started(None)
        _settle(manager)
        assert all(p.prepared == 0 for p in policies.values())

    def test_repeating_the_same_skill_does_not_reload(self):
        manager, policies = _residency(2)
        for _ in range(5):
            manager.on_skill_started("a")
        _settle(manager)
        assert policies["a"].prepared == 1


class _SelfReleasingPolicy:
    """`VlaSkill._on_stop()` のように residency を通さず自分で解放する policy。"""

    def __init__(self) -> None:
        self.prepared = 0
        self.closed = 0
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def prepare(self) -> None:
        self.prepared += 1
        self._loaded = True

    def close(self) -> None:
        self.closed += 1
        self._loaded = False

    def release_behind_the_managers_back(self) -> None:
        """skill の停止処理が呼ぶ経路 (residency は知らない)。"""
        self._loaded = False


def test_reloads_a_model_that_was_released_behind_its_back():
    """帳簿ではなく実体を見ること。

    `VlaSkill._on_stop()` は skill 停止時に `release_after_skill()` を呼び、
    residency を通さず解放する。集合だけを信じると「載っているつもりで実は無い」
    になり、先読みが積まれず、次に要る tick でその場で読むことになる
    (実測 8 秒、2026-09-21 に実 image で確認)。
    """
    policies = {name: _SelfReleasingPolicy() for name in ("a", "b", "c")}
    manager = ModelResidency(["a", "b", "c"], policies, resident=2)

    manager.on_skill_started("a")
    _settle(manager)
    assert policies["b"].prepared == 1

    # skill が裏で解放する (residency の集合はまだ「載っている」と思っている)
    policies["b"].release_behind_the_managers_back()

    manager.on_skill_started("a")
    _settle(manager)

    assert policies["b"].prepared == 2, "解放に気付かず先読みし直していない"


def test_a_policy_without_is_loaded_keeps_the_old_behaviour():
    """`is_loaded` を持たない policy は今までどおり集合を信じる。"""
    manager, policies = _residency(2)

    manager.on_skill_started("a")
    _settle(manager)
    manager.on_skill_started("a")
    _settle(manager)

    assert policies["b"].prepared == 1, "同じ model を 2 度読んでいる"


# ---------------------------------------------------------------- 本物の DeferredPolicy で (Issue #159)
#
# 上の `_StubPolicy.is_loaded` は lock を取らないので、「先読み中に呼び出し元が
# 止まる」(B1b-1) をこのファイルは検出できなかった。ここでは本物の DeferredPolicy
# に、gate が開くまで戻らない loader を入れる。


class _SlowInner:
    def close(self) -> None:
        pass


def _slow_deferred(label, *, gate, calls, fail=False):
    def loader():
        calls.append(label)
        gate.wait(5.0)
        if fail:
            raise RuntimeError(f"{label}: load failed")
        return _SlowInner()

    return DeferredPolicy(_SlowInner, label=label, loader=loader)


def test_a_running_preload_never_blocks_the_caller():
    """先読み中の model に `is_loaded` を聞いて lock を待たないこと (B1b-1)。

    大会経路の driver は毎 tick `on_skill_started` を呼んでいた。以前は切替の次の
    tick で、先読み中の model の `is_loaded` が `_load()` の lock を待ち、request
    thread が読み込み完了まで止まっていた (pod 実測 8〜27 s)。
    """
    gate, calls = threading.Event(), []
    policies = {n: _slow_deferred(n, gate=gate, calls=calls) for n in ("a", "b", "c")}
    manager = ModelResidency(["a", "b", "c"], policies, resident=2)
    manager.on_skill_started(
        "a"
    )  # a と b を先読みに積む (gate が閉じているので終わらない)
    try:
        # worker が a の `_load()` に入る (= a の lock を握る) まで待つ。入る前に
        # 呼ぶと lock が空いていて、直っていなくても待ちが起きない。
        deadline = time.monotonic() + 5.0
        while "a" not in calls and time.monotonic() < deadline:
            time.sleep(0.001)
        assert "a" in calls, "worker が読み込みに入らない"
        worst = 0.0
        for _ in range(10):
            started = time.monotonic()
            manager.on_skill_started("a")
            worst = max(worst, time.monotonic() - started)
        assert worst < 0.05, f"呼び出し元が先読みの lock を {worst:.2f}s 待った"
    finally:
        gate.set()
        _settle(manager)
    assert calls.count("a") == 1 and calls.count("b") == 1, calls


def test_a_failed_preload_is_not_retried_while_the_same_skill_runs():
    """失敗した先読みを毎回積み直さないこと (B3a-01)。skill が変われば 1 回だけ試す。"""
    gate, calls = threading.Event(), []
    gate.set()
    policies = {
        "a": _slow_deferred("a", gate=gate, calls=calls),
        "b": _slow_deferred("b", gate=gate, calls=calls, fail=True),
    }
    manager = ModelResidency(["a", "b"], policies, resident=2)
    for _ in range(5):
        manager.on_skill_started("a")
        _settle(manager)
    assert calls.count("b") == 1, f"同じ skill の間に {calls.count('b')} 回読み直した"

    manager.on_skill_started("b")
    _settle(manager)
    assert calls.count("b") == 2, "skill が変わったのに試し直していない"


def test_a_preload_that_falls_out_of_range_is_released_once_loaded():
    """読んでいる間に範囲から外れた model は、読み終わった時点で解放すること。

    呼び出し元で close すると `_load()` の lock を待つので、worker が後始末する
    (N 個を超えて載せたままにしない)。
    """
    gate, calls = threading.Event(), []
    policies = {n: _slow_deferred(n, gate=gate, calls=calls) for n in ("a", "b", "c")}
    manager = ModelResidency(["a", "b", "c"], policies, resident=1)
    manager.on_skill_started("a")  # a を読み始める
    manager.on_skill_started("b")  # a は範囲外になったが読み込み中
    gate.set()
    _settle(manager)
    assert manager.loaded == ("b",)
    assert not policies["a"].is_loaded, "範囲外の a が載ったまま"


def test_self_bridged_pick_to_insert_switches_models_inside_the_hold():
    """Codex #5: hybrid pick -> insert も HOLD の中で「解放 → 読込」を行う。

    `--gpu-models 1` で、insert の開始 (`reset`) が pick を載せたまま同期で
    読み込み、2 個が同時に載っていた。HOLD を挟めば常に 1 個以下。
    """
    from inference.desktop.orchestrator import self_bridged_transition_skill_names

    loaded: set[str] = set()
    peak = [0]
    lock = threading.Lock()

    class _Inner:
        def __init__(self, label):
            self.label = label

        def reset(self):
            pass

        def close(self):
            with lock:
                loaded.discard(self.label)

    def _deferred(label):
        def loader():
            with lock:
                loaded.add(label)
                peak[0] = max(peak[0], len(loaded))
            return _Inner(label)

        return DeferredPolicy(_Inner, label=label, loader=loader)

    hold = self_bridged_transition_skill_names("pick_table_leg", "insert_table_leg")[0]
    timeline = ["pick_table_leg", hold, "insert_table_leg"]
    policies = {
        "pick_table_leg": _deferred("pick_table_leg"),
        "insert_table_leg": _deferred("insert_table_leg"),
    }
    manager = ModelResidency(timeline, policies, resident=1)
    manager.prime()
    manager.on_skill_started(hold)
    _settle(manager)
    assert manager.is_ready("insert_table_leg")
    policies["insert_table_leg"].reset()  # insert の開始
    manager.on_skill_started("insert_table_leg")
    _settle(manager)
    assert peak[0] == 1
    assert loaded == {"insert_table_leg"}


def test_continuous_stage_first_model_loads_inside_its_transition_hold():
    """Codex #9: Stage 2..5 の最初の model は腕の遷移の後、HOLD の中で読む。"""
    suffix = "rotate_leg_to_tighten_to_rotate_table_base"
    timeline = [
        f"hand_release_{suffix}",
        f"arm_transition_{suffix}",
        f"hand_transition_{suffix}",
        f"hold_transition_{suffix}",
        "rotate_table_base",
    ]
    policy = _StubPolicy()
    manager = ModelResidency(timeline, {"rotate_table_base": policy}, resident=1)
    assert manager.first_model_waits_for_switch_hold
    for name in timeline[:3]:
        manager.on_skill_started(name)
    _settle(manager)
    assert policy.prepared == 0  # 腕が動いている間は読まない
    manager.on_skill_started(timeline[3])
    _settle(manager)
    assert policy.prepared == 1
    assert manager.is_ready("rotate_table_base")


def test_stage_that_starts_with_its_model_still_primes():
    manager, _ = _residency(1, order=("pick_table_leg",))
    assert not manager.first_model_waits_for_switch_hold
