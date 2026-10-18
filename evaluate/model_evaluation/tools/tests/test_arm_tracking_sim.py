"""arm_tracking_sim: 会場の log を本物の送り方に流す模擬 (Issue #188)。"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from evaluate.model_evaluation.tools import arm_tracking_sim as sim


def _cfg() -> dict:
    return yaml.safe_load(sim.SKILL_CONFIG.read_text())


def _segment(seconds: float = 6.0, *, pushing: bool = False):
    """30 Hz の 19-D 指令と、会場の実測に見立てた腕。pushing なら右肘が指令より 0.15 伸びたまま。"""
    t = np.arange(0.0, seconds, 1 / 30.0)
    arms = 0.3 + 0.2 * np.sin(2 * np.pi * 0.3 * t)[:, None] * np.ones(14)
    if pushing:
        arms[:, 10] = 0.3  # 押している間、policy の右肘の指令はほぼ一定
    action = np.zeros((len(t), 19))
    action[:, 3:17] = arms
    action[:, 17:19] = 4.5
    measured = arms.copy()
    if pushing:
        measured[:, 10] = 0.45
    return t, action, measured


def _run(case: sim.Case, *, pushing: bool = False, wall: bool = False):
    t, action, measured = _segment(pushing=pushing)
    assist = sim._assist(case, _cfg())
    return sim.simulate(
        t,
        action,
        measured,
        assist,
        tau=0.15,
        case=case,
        wall_joint=10 if wall else None,
    )


@pytest.mark.requires_boundary
def test_the_arm_follows_with_a_lag_and_the_sink_publishes():
    result = _run(sim.Case("today", "today"))
    t, action, _ = _segment()
    assert result["published"] > 20
    assert np.all(np.isfinite(result["measured"])) and result["measured"][-1].any()
    metrics = sim.tracking_metrics([(t, action[:, 3:17], result["measured"])])
    # 一次遅れ 0.15 s + adapter の段 (1 行目 50 ms 後、実測への合わせ直し)
    assert 150.0 <= metrics["lag_ms"] <= 450.0
    assert metrics["mean_abs_error_rad"] < 0.03


@pytest.mark.requires_boundary
def test_standard_pushes_harder_than_today_into_a_wall():
    today = _run(sim.Case("today", "today"), pushing=True, wall=True)
    standard = _run(sim.Case("standard", "standard"), pushing=True, wall=True)
    assert standard["push"].mean() > today["push"].mean() * 1.2


@pytest.mark.requires_boundary
def test_a_higher_adapter_cap_pushes_harder():
    base = _run(sim.Case("standard", "standard"), pushing=True, wall=True)
    capped = _run(
        sim.Case("standard 1.5", "standard", max_joint_vel=1.5), pushing=True, wall=True
    )
    assert capped["push"].mean() > base["push"].mean()


def test_default_cases_use_every_preset():
    labels = [case.arm for case in sim.default_cases(_cfg())]
    assert {"today", "standard", "lead"} <= set(labels)


def test_unknown_skill_is_an_error(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "actions.jsonl").write_text("")
    (run / "states.jsonl").write_text("")
    with pytest.raises(ValueError, match="no pick_table_leg rows"):
        sim.load_segments(run, "pick_table_leg")


# ---- 先の予定 (Issue #188) ----

def _write_preview_run(root, *, sent_ahead_s: float, lag_s: float, with_plan: bool = True):
    """予定 (ゆっくりした正弦波)、送った腕 = 予定の sent_ahead_s 先、実測 = 予定の lag_s 遅れ。"""
    run = root / "production_runs" / "2026-09-30T01-00-00"
    run.mkdir(parents=True)
    t = np.arange(0.0, 20.0, 1 / 30.0)
    phase = np.linspace(0.0, 1.3, 14)

    def plan_at(times):
        return 0.3 * np.sin(2 * np.pi * 0.25 * times[:, None] + phase)

    plan, sent, measured = plan_at(t), plan_at(t + sent_ahead_s), plan_at(t - lag_s)
    t0 = 80_000_000_000_000
    with (run / "actions.jsonl").open("w") as a, (run / "states.jsonl").open("w") as s:
        for k, tk in enumerate(t):
            stamp = t0 + int(tk * 1e9)
            action = {
                "t": stamp,
                "skill": "pick_table_leg",
                "chunk_0": [0.0, 0.0, 0.0, *sent[k].tolist(), 4.5, 4.5],
            }
            if with_plan:
                action["policy_metadata"] = {
                    "arm_preview_steps": 6,
                    "arm_preview_used_steps": 6,
                    "plan_target_arms": plan[k].tolist(),
                }
            a.write(json.dumps(action) + "\n")
            s.write(json.dumps({"t": stamp, "joint_position": [0.0] * 15 + measured[k].tolist()}) + "\n")
    return run


@pytest.mark.requires_boundary
def test_compare_sends_the_plan_ahead_only_for_presets_with_preview(tmp_path):
    run = _write_preview_run(tmp_path, sent_ahead_s=0.0, lag_s=0.3, with_plan=False)
    cfg = _cfg()
    cases = [case for case in sim.default_cases(cfg) if case.label.startswith("standard")][:2]
    assert [case.label for case in cases] == ["standard (先の予定なし)", "standard"]
    without, with_preview = sim.compare(
        run, "pick_table_leg", cases, tau=0.15, wall_joint=None, skill_config=cfg
    )
    assert with_preview["lag_ms"] < without["lag_ms"] - 100.0
    assert with_preview["instant_error_rad"] < without["instant_error_rad"]


def test_load_segments_uses_the_plan_of_a_preview_run(tmp_path):
    run = _write_preview_run(tmp_path, sent_ahead_s=0.2, lag_s=0.1)
    ((t, action, _),) = sim.load_segments(run, "pick_table_leg")
    phase = np.linspace(0.0, 1.3, 14)
    np.testing.assert_allclose(
        action[:, 3:17], 0.3 * np.sin(2 * np.pi * 0.25 * t[:, None] + phase), atol=1e-6
    )
