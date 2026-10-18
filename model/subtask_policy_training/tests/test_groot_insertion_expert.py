from __future__ import annotations

import pytest

from model.subtask_policy_training.gr00t.insertion_expert import (
    build_insertion_sampling_plan,
    build_finish_supervision,
    build_hard_negative_supervision,
    build_success_supervision,
    validate_supervision,
)


def progress_record(*, reviewed: bool = True) -> dict:
    source = "reviewed_phase_frames" if reviewed else "hand_cmd+hand_state"
    length = 100
    return {
        "episode_index": 7,
        "length": length,
        "milestones": {
            f"M{index}": {
                "frame": frame,
                "confidence": 1.0,
                "source": source,
                "valid": True,
            }
            for index, frame in enumerate((1, 10, 20, 35, 55, 75, 90))
        },
        "dex1_transition_mask": [index in {52, 53, 54} for index in range(length)],
    }


def test_success_interval_uses_only_reviewed_m3_m4() -> None:
    record = build_success_supervision(
        progress_record(), trajectory_type="full_success", eef_loss_eligible=False
    ).as_dict()
    validate_supervision(record)
    assert record["expert_role"] == "insertion"
    assert record["interval_start"] == 20
    assert record["interval_end"] == 66
    assert sum(record["action_loss_eligible"]) == 46
    assert not any(record["eef_loss_eligible"])


def test_heuristic_milestones_are_not_action_ground_truth() -> None:
    with pytest.raises(ValueError, match="reviewed"):
        build_success_supervision(
            progress_record(reviewed=False), trajectory_type="full_success"
        )


def test_finish_interval_covers_m4_through_episode_end() -> None:
    record = build_finish_supervision(
        progress_record(), trajectory_type="recovery_success"
    ).as_dict()
    validate_supervision(record)
    assert record["expert_role"] == "finish"
    assert record["interval_start"] == 45
    assert record["interval_end"] == 100
    assert sum(record["action_loss_eligible"]) == 55


def test_hard_negative_never_provides_action_teacher() -> None:
    record = build_hard_negative_supervision(
        episode_index=91,
        length=12,
        reason="right_hand_above_tabletop",
        left_support_ready=[1.0] * 12,
        right_insert_complete=[0.0] * 12,
    ).as_dict()
    validate_supervision(record)
    assert not any(record["action_loss_eligible"])
    assert not any(record["eef_loss_eligible"])
    assert record["policy_input"] is False


def test_insertion_sampling_excludes_unreviewed_frames() -> None:
    record = build_success_supervision(
        progress_record(), trajectory_type="full_success"
    ).as_dict()
    base = {
        "schema_version": "team_ramen_phase_sampling_plan/v3",
        "progress_sidecar_sha256": "a" * 64,
        "episodes": [
            {"episode_index": 0, "length": 4},
            {"episode_index": 1, "length": 100},
        ],
    }
    record["episode_index"] = 1
    plan = build_insertion_sampling_plan(
        base, [record], insertion_sidecar_sha256="b" * 64
    )
    assert plan["episodes"][0]["train_eligible"] is False
    assert not any(plan["episodes"][0]["frame_weights"])
    reviewed = plan["episodes"][1]
    assert reviewed["train_eligible"] is True
    assert sum(weight > 0 for weight in reviewed["frame_weights"]) == 46
    assert max(reviewed["frame_weights"]) == 4.0
    assert plan["summary"]["action_loss_frame_count"] == 46


def test_insertion_sampling_rejects_unknown_episode() -> None:
    record = build_success_supervision(
        progress_record(), trajectory_type="full_success"
    ).as_dict()
    base = {
        "schema_version": "team_ramen_phase_sampling_plan/v3",
        "episodes": [{"episode_index": 0, "length": 100}],
    }
    with pytest.raises(ValueError, match="unknown episodes"):
        build_insertion_sampling_plan(
            base, [record], insertion_sidecar_sha256="b" * 64
        )
