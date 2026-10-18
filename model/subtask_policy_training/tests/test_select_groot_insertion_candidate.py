from __future__ import annotations

from model.subtask_policy_training.scripts.select_groot_insertion_candidate import select


def _candidate(step: int, *, right_f1: float, arm_error: float) -> dict:
    metrics = {
        "left_support_f1": 0.95,
        "right_insert_f1": right_f1,
        "arm_p95_error_rad": arm_error,
        "dex1_transition_f1": 0.9,
        "replan_discontinuity_p99_rad": 0.04,
        "nonfinite_values": 0,
        "official_limit_violations": 0,
    }
    return {
        "step": step,
        "metrics": metrics,
        "gate_failures": [] if right_f1 >= 0.9 and arm_error <= 0.08 else ["gate"],
        "offline_gate_passed": right_f1 >= 0.9 and arm_error <= 0.08,
    }


def test_selector_prefers_gate_passing_candidate_over_lower_arm_error() -> None:
    result = select(
        [
            _candidate(500, right_f1=0.89, arm_error=0.02),
            _candidate(1000, right_f1=0.93, arm_error=0.06),
            _candidate(2000, right_f1=0.92, arm_error=0.04),
        ]
    )

    assert result["selected_step"] == 1000
    assert result["offline_gate_passed"] is True
    assert result["physical_release_eligible"] is False
