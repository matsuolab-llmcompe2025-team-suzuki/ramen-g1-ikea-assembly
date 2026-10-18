"""check_boundary_hold: 保持中に腕が指令から離れていくかを boundary ログから判定する (Issue #172)。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from evaluate.model_evaluation.tools import check_boundary_hold as tool


def _record(t_s: float, command: np.ndarray, measured: np.ndarray, **extra) -> str:
    row = np.zeros(22)
    row[0:4] = -1.0
    row[4:18] = command
    row[21] = 0.74
    body = np.zeros(29)
    body[15:29] = measured
    return json.dumps({
        "event": "boundary_joint", "joint_22": row.tolist(),
        "measured_body_q29": body.tolist(), "monotonic_ns": int(t_s * 1e9), **extra,
    })


def _hold(t0: float, seconds: float, command: np.ndarray, sag_fn) -> list[str]:
    return [
        _record(t0 + k * 0.5, command, command - sag_fn(k * 0.5))
        for k in range(int(seconds / 0.5) + 1)
    ]


def _results(lines: list[str]) -> list[tool.HoldResult]:
    samples = tool.load_samples(lines)
    return [
        tool.assess_hold(seg, clamp_rad=0.05, drift_rad=0.03, drift_rate=0.01)
        for seg in tool.hold_segments(samples, min_hold_s=2.0)
    ]


def test_a_steady_small_sag_is_ok():
    command = np.full(14, 0.3)
    [result] = _results(_hold(0.0, 5.0, command, lambda _t: np.full(14, 0.03)))
    assert result.verdict == "OK"
    assert abs(result.end_error_rad - 0.03) < 1e-9


def test_a_steady_sag_beyond_the_adapter_clamp_is_flagged():
    command = np.full(14, 0.3)
    sag = np.zeros(14)
    sag[0] = 0.07  # 左 shoulder pitch
    [result] = _results(_hold(0.0, 5.0, command, lambda _t: sag))
    assert result.verdict == "SAG"
    assert result.worst_joint == "L.shoulder_pitch"


def test_an_arm_moving_away_from_a_constant_command_is_drift():
    command = np.full(14, 0.3)

    def sag(t):
        out = np.full(14, 0.02)
        out[7] = 0.06 + 0.12 * t  # 右 shoulder pitch が下がり続ける
        return out

    [result] = _results(_hold(0.0, 6.0, command, sag))
    assert result.verdict == "DRIFT"
    assert result.worst_joint == "R.shoulder_pitch"
    assert result.drift_rate_rad_s > 0.1


def test_holds_are_split_by_command_changes_and_stale_or_foreign_records_are_ignored():
    first = np.full(14, 0.3)
    second = np.full(14, 0.5)
    lines = (
        _hold(0.0, 3.0, first, lambda _t: np.full(14, 0.01))
        + ["not json", json.dumps({"event": "boundary_taskspace"})]
        + [_record(3.2, first, first - 0.5, state_stale=True)]
        + _hold(3.5, 3.0, second, lambda _t: np.full(14, 0.02))
    )
    results = _results(lines)
    assert [r.verdict for r in results] == ["OK", "OK"]
    assert [round(r.duration_s, 1) for r in results] == [3.0, 3.0]


def test_main_reports_and_fails_only_on_drift(tmp_path, capsys):
    command = np.full(14, 0.3)
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(_hold(0.0, 5.0, command, lambda _t: np.full(14, 0.03))))
    assert tool.main(["--log", str(log)]) == 0
    assert "OK: every hold stays within the adapter clamp" in capsys.readouterr().out

    log.write_text("\n".join(_hold(0.0, 6.0, command, lambda t: np.full(14, 0.06 + 0.1 * t))))
    assert tool.main(["--log", str(log)]) == 1
    assert "DRIFT" in capsys.readouterr().out


def test_with_the_gravity_offset_it_judges_the_intent_and_suggests_a_scale():
    """補正ありの記録: policy の目標と実測で判定し、実際の垂れ / 補正量から倍率を出す。"""
    intent = np.full(14, 0.3)
    predicted = np.zeros(14)
    predicted[3] = 0.05  # 倍率 1 の補正量 (左 elbow)
    actual_sag = 1.2 * predicted  # 会場の kp が YAML より 2 割低い
    sent = intent + predicted  # scale 1.0 で送った
    lines = []
    for k in range(11):
        record = json.loads(_record(k * 0.5, sent, intent + predicted - actual_sag))
        record.update(arms_intent=intent.tolist(), gravity_offset_rad=predicted.tolist(),
                      gravity_offset_scale=1.0)
        lines.append(json.dumps(record))
    [result] = _results(lines)
    assert result.verdict == "OK"  # 目標とのずれは 0.01 rad
    assert result.end_error_rad == pytest.approx(0.01)
    assert result.suggested_offset_scale == pytest.approx(1.2)
