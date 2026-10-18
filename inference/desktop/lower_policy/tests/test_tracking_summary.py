"""run の記録から腕の遅れ・押し負けを出す道具 (Issue #188)。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inference.desktop.lower_policy.scripts import tracking_summary


def _write_run(
    root: Path, *, lag_s: float, r_elbow_deficit: float, seconds: float = 20.0
) -> Path:
    """30 Hz の指令 (ゆっくりした正弦波) と、lag_s 遅れて右肘だけ押し負けた実測を書く。"""
    run = root / "production_runs" / "2026-09-30T00-00-00"
    run.mkdir(parents=True)
    t = np.arange(0.0, seconds, 1 / 30.0)
    phase = np.linspace(0.0, 1.3, 14)
    intent = 0.3 * np.sin(2 * np.pi * 0.25 * t[:, None] + phase)
    delayed = 0.3 * np.sin(2 * np.pi * 0.25 * (t[:, None] - lag_s) + phase)
    delayed[:, 10] -= r_elbow_deficit  # 右肘 (腕 14-D の 10 番) が指令より手前で止まる
    t0 = 80_000_000_000_000
    with (run / "actions.jsonl").open("w") as a, (run / "states.jsonl").open("w") as s:
        for k, tk in enumerate(t):
            stamp = t0 + int(tk * 1e9)
            chunk = [0.0, 0.0, 0.0, *intent[k].tolist(), 4.5, 4.5]
            body = [0.0] * 15 + delayed[k].tolist()
            a.write(
                json.dumps({"t": stamp, "skill": "rotate_table_base", "chunk_0": chunk})
                + "\n"
            )
            s.write(
                json.dumps(
                    {"t": stamp, "skill": "rotate_table_base", "joint_position": body}
                )
                + "\n"
            )
    logs = root / "orch_logs"
    logs.mkdir()
    with (logs / f"orch_{run.name}.jsonl").open("w") as f:
        for k in range(40):
            f.write(
                json.dumps(
                    {
                        "event": "boundary_joint",
                        "issued_at": 1_790_000_000.0 + k * 0.1,
                        "publish_reason": "changed",
                        "arm_blocked": k % 4 == 0,
                        "arm_lead_rad": [0.02] * 14,
                        "arm_tracking_preset": "standard",
                    }
                )
                + "\n"
            )
        f.write("not json (書きかけの行)\n")
    return run


def test_lag_and_deficit_are_recovered(tmp_path):
    _write_run(tmp_path, lag_s=0.3, r_elbow_deficit=0.12)
    report = tracking_summary.summarize(tmp_path)  # outputs を渡すと最新の run
    m = report["skills"]["rotate_table_base"]
    assert m["lag_ms"] == pytest.approx(300.0, abs=20.0)
    # 遅れをそろえた後は、押し負けた右肘だけが残る (指令 − 実測 = +0.12)
    assert m["deficit_rad"]["R.el"] == pytest.approx(0.12, abs=0.01)
    assert abs(m["deficit_rad"]["L.el"]) < 0.01
    assert m["mean_abs_error_rad"] < 0.02


def test_publish_stats_come_from_the_orch_log(tmp_path):
    _write_run(tmp_path, lag_s=0.1, r_elbow_deficit=0.0)
    publish = tracking_summary.summarize(tmp_path)["publish"]
    assert publish["published"] == 40
    assert publish["preset"] == "standard"
    assert publish["blocked_fraction"] == pytest.approx(0.25)
    assert publish["max_lead_rad"] == pytest.approx(0.02)


def test_the_text_report_names_the_pushed_joint(tmp_path):
    run = _write_run(tmp_path, lag_s=0.2, r_elbow_deficit=0.1)
    text = tracking_summary.format_report(tracking_summary.summarize(run))
    assert "rotate_table_base" in text
    assert "R.el=+0.1" in text
    assert "preset=standard" in text


def test_a_run_without_motion_is_reported_as_such(tmp_path):
    run = tmp_path / "production_runs" / "r"
    run.mkdir(parents=True)
    (run / "actions.jsonl").write_text("")
    (run / "states.jsonl").write_text("")
    text = tracking_summary.format_report(tracking_summary.summarize(run))
    assert "動いた skill が無い" in text


def test_a_directory_without_runs_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        tracking_summary.resolve_run(tmp_path)


# ---- 先の予定を送った run (Issue #188) ----

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


def test_a_preview_run_is_measured_against_the_plan(tmp_path):
    run = _write_preview_run(tmp_path, sent_ahead_s=0.2, lag_s=0.1)
    report = tracking_summary.summarize(run)
    skill = report["skills"]["pick_table_leg"]
    # 送った値 (0.2 s 先) と比べると 300 ms に見えるが、予定の今の値と比べて 100 ms
    assert skill["lag_ms"] == pytest.approx(100.0, abs=20.0)
    assert skill["preview"]["requested_ticks"] == 6
    assert skill["preview"]["used_ms_mean"] == pytest.approx(200.0, abs=5.0)
    assert skill["preview"]["unused_fraction"] == 0.0
    text = tracking_summary.format_report(report)
    assert "先の予定: 腕だけ平均 200 ms 先" in text


def test_an_old_run_without_the_plan_uses_what_was_sent(tmp_path):
    run = _write_preview_run(tmp_path, sent_ahead_s=0.2, lag_s=0.1, with_plan=False)
    skill = tracking_summary.summarize(run)["skills"]["pick_table_leg"]
    assert skill["lag_ms"] == pytest.approx(300.0, abs=20.0)
    assert "preview" not in skill
