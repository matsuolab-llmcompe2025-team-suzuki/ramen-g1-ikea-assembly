"""LoadPlan の unit test (Issue #188 ② 段 2、`--gpu-models plan`)。

主眼:
    - 起動時は最初の skill の main だけを読み、残りは決めた順番に 1 本ずつ
    - 読むのは腕が止まっている間だけ。動き出す関所は読みかけの 1 本を待つ
    - 後ろの skill は保持に入ったときに main → 候補。保持が待つのは main だけ
    - 保持に入ったら前の skill の model を解放する
"""

from __future__ import annotations

import threading
import time

import pytest

from inference.desktop.lower_policy.policies.deferred import DeferredPolicy
from inference.desktop.lower_policy.policies.load_plan import LoadPlan, PlannedModel
from inference.desktop.orchestrator import build_stage_skill_sequence

STAGE2 = build_stage_skill_sequence(2, is_start_stage=True, include_hand=True)
HOLD_TO_PICK = "hold_transition_rotate_table_base_to_pick_table_leg"
HOLD_TO_INSERT = "hold_transition_pick_table_leg_to_insert_table_leg"
HOLD_TO_TIGHTEN = "hold_transition_insert_table_leg_to_rotate_leg_to_tighten"


class _Stub:
    """読み込みの順番を記録する偽の policy。`gate` を渡すと、その Event まで読み込みを止める。"""

    def __init__(self, name: str, log: list, *, fail: bool = False, gate=None) -> None:
        self.name = name
        self.log = log
        self.fail = fail
        self.gate = gate
        self.prepared = 0
        self.closed = 0
        self.managed = None

    def prepare(self) -> None:
        if self.gate is not None:
            assert self.gate.wait(5.0)
        if self.fail:
            raise RuntimeError(f"{self.name} broke")
        self.prepared += 1
        self.log.append(("load", self.name))

    def close(self) -> None:
        self.closed += 1
        self.log.append(("release", self.name))

    def set_residency_managed(self, managed: bool) -> None:
        self.managed = managed


def _stage2(log, *, kinds=None, candidates=None, **stub_kwargs):
    """Stage 2 の既定の並び。skill → (種類, 候補の数)。"""
    kinds = kinds or {
        "rotate_table_base": "ramen_ori",
        "pick_table_leg": "groot_pick_legs",
        "insert_table_leg": "act_diffusion",
        "rotate_leg_to_tighten": "groot",
    }
    candidates = candidates or {
        "rotate_table_base": 2,
        "pick_table_leg": 1,
        "insert_table_leg": 1,
    }
    models, stubs = [], {}
    for skill, kind in kinds.items():
        names = [f"{skill}:main"] + [
            f"{skill}:cand{i}" for i in range(1, candidates.get(skill, 0) + 1)
        ]
        for name in names:
            stub = _Stub(name, log, **stub_kwargs.get(name, {}))
            stubs[name] = stub
            models.append(
                PlannedModel(
                    skill=skill,
                    label=name,
                    kind=kind if name.endswith("main") else "ramen_ori",
                    policy=stub,
                    required=name.endswith("main"),
                )
            )
    return models, stubs


def _plan(models, **kwargs) -> LoadPlan:
    kwargs.setdefault("preload_through_skill", "pick_table_leg")
    return LoadPlan(STAGE2, models, **kwargs)


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _idle(plan: LoadPlan) -> bool:
    """読みかけが無く、今やってよい仕事も残っていない (積みは覗くだけで動かさない)。"""
    with plan._cond:
        pending = plan._releases + plan._queue
        return plan._busy is None and (
            plan._leaving or not any(plan._may_run(entry) for entry in pending)
        )


def _settle(plan: LoadPlan) -> None:
    """やってよい仕事が無くなるまで待つ。"""
    _wait_until(lambda: _idle(plan))


class TestOrder:
    def test_start_loads_only_the_first_main_then_waits_for_the_arms_to_stop(self):
        log = []
        models, stubs = _stage2(log)
        plan = _plan(models)
        try:
            plan.prime()
            assert log == [("load", "rotate_table_base:main")]
            # 頭の手順 (腕・手を動かしている間) は読まない
            for name in STAGE2[:4]:
                plan.on_skill_started(name)
            time.sleep(0.05)
            assert log == [("load", "rotate_table_base:main")]
            assert plan.is_ready("rotate_table_base")
            assert not plan.is_ready("pick_table_leg")
        finally:
            plan.close()

    def test_stage2_reads_the_rotate_candidates_then_pick_while_the_arms_are_still(
        self,
    ):
        log = []
        models, _ = _stage2(log)
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            _settle(plan)
            assert [name for _, name in log] == [
                "rotate_table_base:main",
                "rotate_table_base:cand1",
                "rotate_table_base:cand2",
                "pick_table_leg:main",
                "pick_table_leg:cand1",
            ]
            # insert・締め付けは保持に入るまで読まない
            assert not plan.is_ready("insert_table_leg")
        finally:
            plan.close()

    def test_a_later_skill_loads_main_then_candidates_at_its_hold(self):
        log = []
        models, _ = _stage2(log)
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            _settle(plan)
            log.clear()
            plan.on_skill_started("pick_table_leg")
            plan.on_skill_started(HOLD_TO_INSERT)
            _settle(plan)
            assert log == [
                ("release", "rotate_table_base:main"),
                ("release", "rotate_table_base:cand1"),
                ("release", "rotate_table_base:cand2"),
                ("release", "pick_table_leg:main"),
                ("release", "pick_table_leg:cand1"),
                ("load", "insert_table_leg:main"),
                ("load", "insert_table_leg:cand1"),
            ]
            assert plan.is_ready("insert_table_leg")
        finally:
            plan.close()

    def test_stage1_starts_with_pick_and_stage5_with_flip(self):
        for stage, first in ((1, "pick_table_leg"), (5, "flip_table")):
            log = []
            order = build_stage_skill_sequence(
                stage, is_start_stage=True, include_hand=True
            )
            learned = [
                name
                for name in order
                if not name.startswith(("hand_", "arm_", "hold_", "wait_"))
            ]
            models = [
                PlannedModel(
                    skill=skill,
                    label=f"{skill}:main",
                    kind="groot",
                    policy=_Stub(f"{skill}:main", log),
                )
                for skill in learned
            ] + [
                PlannedModel(
                    skill=first,
                    label=f"{first}:cand1",
                    kind="ramen_ori",
                    policy=_Stub(f"{first}:cand1", log),
                    required=False,
                )
            ]
            plan = LoadPlan(order, models, preload_through_skill="pick_table_leg")
            try:
                plan.prime()
                plan.on_skill_started(f"hold_pose_for_{first}")
                _settle(plan)
                assert [name for _, name in log] == [f"{first}:main", f"{first}:cand1"]
            finally:
                plan.close()


class TestArmsStill:
    def test_nothing_starts_while_a_policy_or_a_transition_moves_the_arms(self):
        log = []
        models, _ = _stage2(log)
        plan = _plan(models)
        try:
            plan.prime()
            for name in (
                "rotate_table_base",
                "hand_release_rotate_table_base_to_pick_table_leg",
                "arm_transition_rotate_table_base_to_pick_table_leg",
                "retry_open_rotate_table_base",
                "retry_arm_rotate_table_base",
                "wait_for_go_live_rotate_table_base",
            ):
                plan.on_skill_started(name)
                time.sleep(0.02)
            assert log == [("load", "rotate_table_base:main")]
            plan.on_skill_started("retry_wait_rotate_table_base")
            _wait_until(lambda: len(log) >= 2)
            assert log[1] == ("load", "rotate_table_base:cand1")
        finally:
            plan.close()

    def test_leaving_waits_for_the_one_in_progress_and_starts_nothing_new(self):
        log = []
        gate = threading.Event()
        models, stubs = _stage2(log, **{"rotate_table_base:cand1": {"gate": gate}})
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("operator_gate_for_rotate_table_base")
            _wait_until(lambda: plan._busy is not None)
            assert not plan.may_leave()  # Enter: 読みかけの候補 1 を待つ
            gate.set()
            _wait_until(plan.may_leave)
            time.sleep(0.05)
            # 待った後は新しく読み始めない (候補 2 は次の止まっている区間まで)
            assert [name for _, name in log] == [
                "rotate_table_base:main",
                "rotate_table_base:cand1",
            ]
            plan.on_skill_started("rotate_table_base")
            time.sleep(0.05)
            assert stubs["rotate_table_base:cand2"].prepared == 0
            plan.on_skill_started("retry_hold_rotate_table_base")
            _wait_until(lambda: stubs["rotate_table_base:cand2"].prepared == 1)
        finally:
            plan.close()

    def test_listed_kinds_may_load_while_a_policy_runs(self):
        log = []
        models, stubs = _stage2(log)
        plan = _plan(models, load_while_policy_runs=("groot_pick_legs",))
        try:
            plan.prime()
            plan.on_skill_started("rotate_table_base")
            _wait_until(lambda: stubs["pick_table_leg:main"].prepared == 1)
            time.sleep(0.05)
            # RAMEN-Ori の候補は止まっている間を待つ
            assert stubs["rotate_table_base:cand1"].prepared == 0
            assert stubs["pick_table_leg:cand1"].prepared == 0
        finally:
            plan.close()

    def test_unknown_kinds_are_rejected(self):
        models, _ = _stage2([])
        with pytest.raises(ValueError, match="unknown kinds"):
            _plan(models, load_while_policy_runs=("gr00t",))


class TestHold:
    def test_the_hold_waits_only_for_the_main(self):
        log = []
        gate = threading.Event()
        models, _ = _stage2(log, **{"insert_table_leg:cand1": {"gate": gate}})
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("pick_table_leg")
            plan.on_skill_started(HOLD_TO_INSERT)
            _wait_until(lambda: plan.is_ready("insert_table_leg"))
            assert ("load", "insert_table_leg:cand1") not in log  # 候補はまだ読みかけ
            gate.set()
        finally:
            plan.close()

    def test_a_failed_main_is_reported_to_the_hold_and_a_failed_candidate_is_not(self):
        log = []
        models, _ = _stage2(
            log,
            **{
                "insert_table_leg:main": {"fail": True},
                "rotate_table_base:cand1": {"fail": True},
            },
        )
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            _settle(plan)
            assert plan.error_for("rotate_table_base") is None
            assert ("load", "rotate_table_base:cand2") in log  # 候補の失敗で止めない
            plan.on_skill_started("pick_table_leg")
            plan.on_skill_started(HOLD_TO_INSERT)
            _wait_until(lambda: plan.error_for("insert_table_leg") is not None)
            assert "insert_table_leg:main broke" in plan.error_for("insert_table_leg")
            assert not plan.is_ready("insert_table_leg")
        finally:
            plan.close()

    def test_a_failed_first_main_raises_before_the_stage(self):
        models, _ = _stage2([], **{"rotate_table_base:main": {"fail": True}})
        plan = _plan(models)
        with pytest.raises(RuntimeError, match="broke"):
            plan.prime()
        plan.close()


class TestRelease:
    def test_a_model_still_loading_is_released_after_it_finishes(self):
        log = []
        gate = threading.Event()
        models, stubs = _stage2(log, **{"rotate_table_base:cand1": {"gate": gate}})
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            _wait_until(lambda: plan._busy is not None)
            plan.on_skill_started("rotate_table_base")
            plan.on_skill_started(HOLD_TO_PICK)
            gate.set()
            _settle(plan)
            assert stubs["rotate_table_base:cand1"].closed == 1
            assert (
                stubs["rotate_table_base:cand2"].prepared == 0
            )  # 積んでいただけの物は外す
            assert stubs["rotate_table_base:main"].closed == 1
            assert plan.is_ready("pick_table_leg")
        finally:
            plan.close()

    def test_close_releases_what_is_loaded_and_hands_back_ownership(self):
        log = []
        models, stubs = _stage2(log)
        plan = _plan(models)
        plan.prime()
        assert stubs["rotate_table_base:main"].managed is True
        plan.close()
        assert stubs["rotate_table_base:main"].closed == 1
        assert stubs["pick_table_leg:main"].closed == 0
        assert all(stub.managed is False for stub in stubs.values())

    def test_a_deferred_policy_is_not_closed_by_the_skill_while_the_plan_owns_it(self):
        loads = []
        policy = DeferredPolicy(
            object, label="x", loader=lambda: loads.append(1) or object()
        )
        order = ["hold_pose_for_a", "a"]
        plan = LoadPlan(
            order, [PlannedModel(skill="a", label="a", kind="groot", policy=policy)]
        )
        try:
            plan.prime()
            policy.release_after_skill()
            assert policy.is_loaded and plan.is_ready("a")
        finally:
            plan.close()
        assert not policy.is_loaded


def test_models_for_a_skill_outside_the_stage_are_rejected():
    with pytest.raises(ValueError, match="not in the stage"):
        LoadPlan(
            ["a"],
            [PlannedModel(skill="b", label="b", kind="groot", policy=_Stub("b", []))],
        )


def test_a_skill_without_a_main_is_rejected():
    with pytest.raises(ValueError, match="no main"):
        LoadPlan(
            ["a"],
            [
                PlannedModel(
                    skill="a",
                    label="a",
                    kind="groot",
                    policy=_Stub("a", []),
                    required=False,
                )
            ],
        )


class TestHybridPickVlm:
    """hybrid pick の VLM は pick の GR00T の前に読み、保持は両方を待つ。"""

    def _models(self, log, **stub_kwargs):
        models, stubs = _stage2(log, **stub_kwargs)
        vlm = _Stub("pick_table_leg:vlm", log, **stub_kwargs.get("pick_table_leg:vlm", {}))
        index = next(i for i, m in enumerate(models) if m.label == "pick_table_leg:main")
        models.insert(
            index, PlannedModel(skill="pick_table_leg", label="pick_table_leg:vlm", kind="vlm", policy=vlm)
        )
        stubs["pick_table_leg:vlm"] = vlm
        return models, stubs

    def test_the_vlm_comes_right_before_the_pick_groot(self):
        log = []
        models, _ = self._models(log)
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            _settle(plan)
            names = [name for _, name in log]
            assert names.index("pick_table_leg:vlm") == names.index("pick_table_leg:main") - 1
            assert names.index("rotate_table_base:cand2") < names.index("pick_table_leg:vlm")
        finally:
            plan.close()

    def test_the_hold_before_pick_waits_for_the_vlm_too(self):
        log = []
        gate = threading.Event()
        models, _ = self._models(log, **{"pick_table_leg:vlm": {"gate": gate}})
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("rotate_table_base")
            plan.on_skill_started(HOLD_TO_PICK)
            _wait_until(lambda: plan._busy is not None)
            assert not plan.is_ready("pick_table_leg")
            gate.set()
            _wait_until(lambda: plan.is_ready("pick_table_leg"))
        finally:
            plan.close()

    def test_a_vlm_that_does_not_start_stops_the_hold_before_pick(self):
        log = []
        models, stubs = self._models(log, **{"pick_table_leg:vlm": {"fail": True}})
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("rotate_table_base")
            plan.on_skill_started(HOLD_TO_PICK)
            _wait_until(lambda: plan.error_for("pick_table_leg") is not None)
            assert "pick_table_leg:vlm broke" in plan.error_for("pick_table_leg")
            assert not plan.is_ready("pick_table_leg")
        finally:
            plan.close()

    def test_the_vlm_is_released_with_pick_at_the_insert_hold(self):
        log = []
        models, stubs = self._models(log)
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            _settle(plan)
            plan.on_skill_started("pick_table_leg")
            plan.on_skill_started(HOLD_TO_INSERT)
            _settle(plan)
            assert stubs["pick_table_leg:vlm"].closed == 1
        finally:
            plan.close()


class TestPageCache:
    def test_each_load_drops_its_weight_files_from_the_page_cache(self, monkeypatch):
        from inference.desktop.lower_policy.policies import load_plan

        dropped = []
        monkeypatch.setattr(
            load_plan, "drop_page_cache", lambda paths: dropped.append(list(paths)) or (1, 10)
        )
        log = []
        models = [
            PlannedModel(
                skill="rotate_table_base", label="rotate", kind="ramen_ori",
                policy=_Stub("rotate", log), weight_files=lambda: ["rotate.pt"],
            ),
            PlannedModel(
                skill="rotate_table_base", label="cand", kind="ramen_ori",
                policy=_Stub("cand", log), required=False, weight_files=lambda: ["cand.pt"],
            ),
            PlannedModel(
                skill="pick_table_leg", label="pick", kind="groot_pick_legs",
                policy=_Stub("pick", log),
            ),
        ]
        plan = _plan(models)
        try:
            plan.prime()
            assert dropped == [["rotate.pt"]]  # 読み終わった直後
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            _settle(plan)
            assert dropped == [["rotate.pt"], ["cand.pt"]]  # weight_files の無い物は落とさない
        finally:
            plan.close()

    def test_a_failed_drop_keeps_the_model_loaded(self, monkeypatch):
        from inference.desktop.lower_policy.policies import load_plan

        def boom(paths):
            raise OSError("no such device")

        monkeypatch.setattr(load_plan, "drop_page_cache", boom)
        model = PlannedModel(
            skill="rotate_table_base", label="rotate", kind="ramen_ori",
            policy=_Stub("rotate", []), weight_files=lambda: ["rotate.pt"],
        )
        plan = _plan([model])
        try:
            plan.prime()
            assert plan.is_ready("rotate_table_base")
        finally:
            plan.close()


class TestRecord:
    """1 本ごとの記録 (orch log の "model_load") と、読んでいる間の制御の周期。"""

    def test_each_load_is_recorded_from_the_tick_thread_with_memory(self):
        events = []
        memory = iter([{"MemAvailable": 90.0, "Cached": 2.0}, {"MemAvailable": 80.0, "Cached": 9.0}] * 10)
        log = []
        models, _ = _stage2(log)
        plan = _plan(
            models[:2], event_fn=events.append, meminfo=lambda: next(memory),
        )
        try:
            plan.prime()
            assert events == []  # worker・prime からは渡さない
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            assert [e["label"] for e in events] == ["rotate_table_base:main"]
            first = events[0]
            assert first["event"] == "model_load"
            assert first["operation"] == "load" and first["ok"] is True
            assert first["role"] == "main" and first["kind"] == "ramen_ori"
            assert first["during"] == "before the stage" and first["while_moving"] is False
            assert first["mem_available_gb"] == [90.0, 80.0]
            assert first["cached_gb"] == [2.0, 9.0]
            _settle(plan)
            plan.on_skill_started("operator_gate_for_rotate_table_base")
            candidate = events[-1]
            assert candidate["label"] == "rotate_table_base:cand1"
            assert candidate["role"] == "candidate"
            assert candidate["during"] == "hold_pose_for_rotate_table_base"
        finally:
            plan.close()

    def test_the_control_period_while_loading_is_measured(self):
        events = []
        gate = threading.Event()
        clock = {"now": 0.0}
        log = []
        models, _ = _stage2(log, **{"rotate_table_base:cand1": {"gate": gate}})
        plan = _plan(models[:2], event_fn=events.append, clock=lambda: clock["now"], meminfo=lambda: None)
        try:
            plan.prime()
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            _wait_until(lambda: plan._busy is not None)
            for step in (0.033, 0.034, 0.120):
                clock["now"] += step
                plan.on_skill_started("hold_pose_for_rotate_table_base")
            gate.set()
            _settle(plan)
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            period = events[-1]["tick_period_ms"]
            assert period["n"] == 3
            assert period["max"] == pytest.approx(120.0)
            assert period["p95"] == pytest.approx(120.0)
            assert events[-1]["mem_available_gb"] == [None, None]
        finally:
            plan.close()

    def test_failures_and_releases_are_recorded_too(self):
        events = []
        log = []
        models, _ = _stage2(log, **{"insert_table_leg:main": {"fail": True}})
        plan = _plan(models, event_fn=events.append, meminfo=lambda: None)
        try:
            plan.prime()
            plan.on_skill_started("pick_table_leg")
            plan.on_skill_started(HOLD_TO_INSERT)
            _wait_until(lambda: plan.error_for("insert_table_leg") is not None)
            plan.on_skill_started(HOLD_TO_TIGHTEN)
            released = [e for e in events if e["operation"] == "release"]
            assert released and released[0]["label"] == "rotate_table_base:main"
            failed = [e for e in events if not e["ok"]]
            assert failed[0]["label"] == "insert_table_leg:main"
            assert "insert_table_leg:main broke" in failed[0]["error"]
        finally:
            plan.close()

    def test_a_broken_event_sink_does_not_stop_the_run(self):
        def broken(_record):
            raise OSError("disk full")

        models, _ = _stage2([])
        plan = _plan(models[:1], event_fn=broken, meminfo=lambda: None)
        try:
            plan.prime()
            plan.on_skill_started("hold_pose_for_rotate_table_base")
            assert plan.is_ready("rotate_table_base")
        finally:
            plan.close()

    def test_close_hands_over_the_events_left(self):
        events = []
        models, _ = _stage2([])
        plan = _plan(models[:1], event_fn=events.append, meminfo=lambda: None)
        plan.prime()
        plan.close()
        assert [e["label"] for e in events] == ["rotate_table_base:main"]


def test_meminfo_is_read_in_gb(tmp_path):
    from inference.desktop.lower_policy.policies.load_plan import read_meminfo_gb

    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 131072000 kB\nMemAvailable: 97656250 kB\nCached: 1953125 kB\n")
    assert read_meminfo_gb(path) == {"MemAvailable": 100.0, "Cached": 2.0}
    assert read_meminfo_gb(tmp_path / "missing") is None


def test_the_dry_run_check_reads_and_releases_each_model_not_checked_yet():
    log = []
    models, stubs = _stage2(log)
    plan = _plan(models)
    try:
        skip = {id(stubs["rotate_table_base:main"])}
        checked = plan.check_each(skip_ids=skip)
        assert "rotate_table_base:main" not in checked
        assert checked[0] == "rotate_table_base:cand1"
        assert all(stubs[name].prepared == 1 and stubs[name].closed == 1 for name in checked)
        assert stubs["rotate_table_base:main"].prepared == 0
    finally:
        plan.close()


def test_the_dry_run_check_uses_validate_load_and_release_when_it_exists():
    loads = []
    policy = DeferredPolicy(object, label="x", loader=lambda: loads.append(1) or object())
    plan = LoadPlan(["a"], [PlannedModel(skill="a", label="a", kind="groot", policy=policy)])
    try:
        assert plan.check_each() == ["a"]
        assert loads == [1] and not policy.is_loaded
    finally:
        plan.close()


class TestModelState:
    """R の後に model を選ぶ画面の「読み込み済み・読み込み中・失敗」(Issue #188 段 3)。"""

    def _plan(self, log, **stub_kwargs):
        models, stubs = _stage2(log, **stub_kwargs)
        models = [
            PlannedModel(**{**m.__dict__, "variant": m.label}) for m in models
        ]
        return _plan(models), stubs

    def test_states_follow_the_loads(self):
        gate = threading.Event()
        plan, _ = self._plan([], **{"rotate_table_base:cand2": {"gate": gate}, "pick_table_leg:cand1": {"fail": True}})
        try:
            plan.prime()
            assert plan.model_state("rotate_table_base", "rotate_table_base:main") == "loaded"
            assert plan.model_state("rotate_table_base", "rotate_table_base:cand1") == "loading"
            plan.on_skill_started("retry_wait_rotate_table_base")
            _wait_until(lambda: plan.model_state("rotate_table_base", "rotate_table_base:cand1") == "loaded")
            assert plan.model_state("rotate_table_base", "rotate_table_base:cand2") == "loading"
            gate.set()
            _wait_until(lambda: plan.model_state("pick_table_leg", "pick_table_leg:cand1") == "failed")
            plan.on_skill_started("rotate_table_base")
            plan.on_skill_started(HOLD_TO_PICK)
            _settle(plan)
            assert plan.model_state("rotate_table_base", "rotate_table_base:cand1") == "released"
        finally:
            plan.close()

    def test_the_main_is_loaded_only_with_the_models_loaded_with_it(self):
        log = []
        gate = threading.Event()
        models, _ = _stage2(log)
        models = [PlannedModel(**{**m.__dict__, "variant": m.label}) for m in models]
        vlm = PlannedModel(
            skill="pick_table_leg", label="vlm", kind="vlm",
            policy=_Stub("vlm", log, gate=gate), variant="",
        )
        index = next(i for i, m in enumerate(models) if m.label == "pick_table_leg:main")
        models.insert(index, vlm)
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("rotate_table_base")
            plan.on_skill_started(HOLD_TO_PICK)
            _wait_until(lambda: plan._busy is not None)
            assert plan.model_state("pick_table_leg", "pick_table_leg:main") == "loading"
            gate.set()
            _wait_until(lambda: plan.model_state("pick_table_leg", "pick_table_leg:main") == "loaded")
        finally:
            plan.close()

    def test_an_unknown_variant_is_an_error(self):
        plan, _ = self._plan([])
        with pytest.raises(KeyError):
            plan.model_state("rotate_table_base", "nothing")
        plan.close()


class TestReviewFixes:
    """PR #189 の review の直し。"""

    def test_stay_lets_the_loads_go_on_after_a_cancelled_enter(self):
        log = []
        gate = threading.Event()
        models, _ = _stage2(log, **{"rotate_table_base:cand1": {"gate": gate}})
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("operator_gate_for_rotate_table_base")
            _wait_until(lambda: plan._busy is not None)
            assert not plan.may_leave()  # Enter: 読みかけ (候補 1) を待つ。新しくは読み始めない
            gate.set()
            _wait_until(lambda: plan._busy is None)
            time.sleep(0.05)
            assert ("load", "rotate_table_base:cand2") not in log
            plan.stay()  # 到達・静止が崩れて関所が確認を取り消した
            _wait_until(lambda: ("load", "rotate_table_base:cand2") in log)
        finally:
            plan.close()

    def test_stay_does_nothing_while_the_arms_move(self):
        models, _ = _stage2([])
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("rotate_table_base")
            plan.may_leave()
            plan.stay()
            assert plan._leaving  # 動いている間は触らない
        finally:
            plan.close()

    def test_close_stops_a_vlm_that_is_still_starting_without_waiting(self):
        released = threading.Event()

        class _StartingVlm:
            is_loaded = False

            def prepare(self):
                # 起動待ち (時間制限なし)。止められたら抜けて失敗する
                assert released.wait(10.0)
                raise RuntimeError("the VLM server was stopped while loading")

            def close(self):
                released.set()

        models, _ = _stage2([])
        vlm = PlannedModel(skill="pick_table_leg", label="vlm", kind="vlm", policy=_StartingVlm())
        index = next(i for i, m in enumerate(models) if m.label == "pick_table_leg:main")
        models.insert(index, vlm)
        models = [m for m in models if m.skill in ("rotate_table_base", "pick_table_leg") and m.required]
        plan = _plan(models)
        plan.prime()
        plan.on_skill_started("hold_pose_for_rotate_table_base")
        _wait_until(lambda: plan._busy is not None and plan._busy.model.kind == "vlm")
        started = time.monotonic()
        plan.close()
        assert released.is_set()
        assert time.monotonic() - started < 5.0  # worker の 30 s を待たない


class TestVlmAsACandidatePart:
    """既定の pick が学習の policy (hybrid なし) の run: VLM は hybrid の候補の一部 (Issue #188)。"""

    def _models(self, log, **stub_kwargs):
        models, stubs = _stage2(log, candidates={"pick_table_leg": 1})
        vlm = _Stub("vlm", log, **stub_kwargs)
        index = next(i for i, m in enumerate(models) if m.label == "pick_table_leg:cand1")
        models.insert(
            index,
            PlannedModel(skill="pick_table_leg", label="vlm", kind="vlm", policy=vlm, required=False, variant="vlm"),
        )
        models = [PlannedModel(**{**m.__dict__, "variant": m.variant or m.label}) for m in models]
        return models, vlm

    def test_the_hold_before_pick_waits_only_for_the_default(self):
        log = []
        gate = threading.Event()
        models, _ = self._models(log, gate=gate)
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("rotate_table_base")
            plan.on_skill_started(HOLD_TO_PICK)
            _wait_until(lambda: plan.is_ready("pick_table_leg"))  # VLM はまだ起動中
            assert plan.model_state("pick_table_leg", "vlm") == "loading"
            gate.set()
            _wait_until(lambda: plan.model_state("pick_table_leg", "vlm") == "loaded")
        finally:
            plan.close()

    def test_a_vlm_that_fails_does_not_stop_the_hold(self):
        log = []
        models, _ = self._models(log, fail=True)
        plan = _plan(models)
        try:
            plan.prime()
            plan.on_skill_started("rotate_table_base")
            plan.on_skill_started(HOLD_TO_PICK)
            _wait_until(lambda: plan.model_state("pick_table_leg", "vlm") == "failed")
            assert plan.error_for("pick_table_leg") is None
            assert plan.is_ready("pick_table_leg")
        finally:
            plan.close()
