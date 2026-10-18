from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.upper_policy.flip_table_expert_router import (
    Expert,
    FlipTableExpertRouter,
    InsertionEstimate,
    blend_absolute_targets,
    gate_right_dex1_target,
)


def estimate(
    *,
    support: float = 0.0,
    inserted: float = 0.0,
    visual: float = 1.0,
    margin: float = 0.3,
) -> InsertionEstimate:
    return InsertionEstimate(
        left_support_ready=support,
        right_insert_complete=inserted,
        tabletop_edge_in_grasp_region=visual,
        tabletop_between_fingers=visual,
        insertion_depth_ready=visual,
        right_wrist_limit_margin_rad=margin,
    )


def test_router_latches_early_insertion_finish_without_reverse_transition() -> None:
    router = FlipTableExpertRouter()
    for _ in range(4):
        assert router.update(estimate(support=1.0)).expert is Expert.EARLY
    decision = router.update(estimate(support=1.0))
    assert decision.expert is Expert.INSERTION
    assert decision.execution_steps == 4

    for _ in range(4):
        assert router.update(estimate(inserted=1.0)).expert is Expert.INSERTION
    decision = router.update(estimate(inserted=1.0))
    assert decision.expert is Expert.FINISH
    assert decision.execution_steps == 5
    assert router.update(estimate()).expert is Expert.FINISH


@pytest.mark.parametrize("visual,margin", [(0.89, 0.3), (1.0, 0.149)])
def test_right_hand_close_requires_every_observable_gate(
    visual: float, margin: float
) -> None:
    router = FlipTableExpertRouter()
    for _ in range(5):
        decision = router.update(estimate(support=1.0, visual=visual, margin=margin))
    assert decision.expert is Expert.INSERTION
    assert not decision.right_hand_close_allowed


def test_absolute_target_blending_has_no_boundary_jump() -> None:
    previous = np.zeros(16)
    following = np.arange(16, dtype=np.float64)
    blended = blend_absolute_targets(previous, following, steps=10)
    assert blended.shape == (10, 16)
    np.testing.assert_allclose(blended[0], following / 10.0)
    np.testing.assert_allclose(blended[-1], following)
    np.testing.assert_allclose(
        np.diff(blended, axis=0),
        np.repeat(following[None, :] / 10.0, 9, axis=0),
    )


def test_right_dex1_close_is_blocked_until_visual_gates_pass() -> None:
    target = np.arange(16, dtype=np.float64) / 10.0
    target[14:] = [1.2, 0.4]
    measured = target.copy()
    measured[15] = 4.3

    gated = gate_right_dex1_target(target, measured, close_allowed=False)
    np.testing.assert_allclose(gated[:15], target[:15])
    assert gated[15] == pytest.approx(4.3)
    np.testing.assert_allclose(
        gate_right_dex1_target(target, measured, close_allowed=True),
        target,
    )


def test_right_dex1_gate_enforces_physical_contract() -> None:
    with pytest.raises(ValueError, match="physical 16D"):
        gate_right_dex1_target(
            np.zeros(15),
            np.zeros(16),
            close_allowed=False,
        )
    with pytest.raises(ValueError, match=r"\[0, 4\.5\]"):
        gate_right_dex1_target(
            np.zeros(16),
            np.zeros(16),
            close_allowed=False,
            minimum_open_value=4.6,
        )
