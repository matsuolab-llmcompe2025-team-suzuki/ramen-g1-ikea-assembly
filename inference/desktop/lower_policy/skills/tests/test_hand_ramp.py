"""hand_ramp の unit test (Issue #141 束 1-8 / D2)。

`evaluate/model_evaluation/common/tests/test_safe_return.py` と
`runners/tests/test_run_skill.py` から移設 (関数を inference 側へ移したため)。
"""

from __future__ import annotations

import numpy as np

from inference.desktop.lower_policy.actuators.hand import HAND_GRIP_MAX
from inference.desktop.lower_policy.skills.hand_ramp import (
    build_hand_open_ramp,
    build_hand_target_ramp,
    hand_target_completion_mode,
)

DEX1_OPEN_MAX_RAD = HAND_GRIP_MAX


def _hand_history(value: tuple[float, float], count: int = 10) -> list[np.ndarray]:
    return [np.asarray(value, dtype=np.float64) for _ in range(count)]

def test_hand_open_ramp_is_monotonic_and_reaches_physical_maximum() -> None:
    start = np.array([1.2, 4.8], dtype=np.float64)
    ramp = build_hand_open_ramp(start, command_hz=30.0, velocity_limit_rad_s=1.5)
    stacked = np.vstack((start, *ramp))

    assert np.all(np.diff(stacked, axis=0) >= -1e-12)
    assert np.max(np.diff(stacked, axis=0)) <= 1.5 / 30.0 + 1e-12
    np.testing.assert_allclose(stacked[-1], [DEX1_OPEN_MAX_RAD] * 2)


def test_hand_target_ramp_closes_independently_to_dataset_width() -> None:
    start = np.array([5.4, 5.4], dtype=np.float64)
    target = np.array([3.6679358482, 2.2749712097], dtype=np.float64)
    ramp = build_hand_target_ramp(
        start, target, command_hz=30.0, velocity_limit_rad_s=1.5
    )
    stacked = np.vstack((start, *ramp))

    assert np.all(np.diff(stacked, axis=0) <= 1e-12)
    assert np.max(np.abs(np.diff(stacked, axis=0))) <= 1.5 / 30.0 + 1e-12
    np.testing.assert_allclose(stacked[-1], target)


def test_hand_completion_accepts_exact_target() -> None:
    mode = hand_target_completion_mode(
        start=np.asarray([5.4, 5.4]),
        target=np.asarray([0.0, 3.84]),
        measured_history=_hand_history((0.10, 3.86), count=1),
        tolerance_rad=0.15,
        allow_closing_contact=False,
    )
    assert mode == "target"

def test_hand_completion_accepts_stable_bounded_closing_contact() -> None:
    mode = hand_target_completion_mode(
        start=np.asarray([5.4, 5.4]),
        target=np.asarray([0.0, 3.84]),
        measured_history=_hand_history((0.23, 3.87)),
        tolerance_rad=0.15,
        allow_closing_contact=True,
    )
    assert mode == "closing_contact"

def test_hand_completion_accepts_recorded_insert_and_tighten_contact() -> None:
    """Regression: a real leg stopped the right hand after 86% of its stroke."""

    mode = hand_target_completion_mode(
        start=np.asarray([5.385760307312012, 5.4030866622924805]),
        target=np.asarray([3.0189805030822754, 1.8930855989456177]),
        measured_history=_hand_history(
            (3.0363693237304688, 2.3773865699768066)
        ),
        tolerance_rad=0.15,
        allow_closing_contact=True,
    )
    assert mode == "closing_contact"

def test_hand_completion_rejects_contact_with_excessive_absolute_residual() -> None:
    mode = hand_target_completion_mode(
        start=np.asarray([5.4, 5.4]),
        target=np.asarray([0.0, 0.0]),
        measured_history=_hand_history((1.0, 1.0)),
        tolerance_rad=0.15,
        allow_closing_contact=True,
    )
    assert mode is None

def test_hand_completion_rejects_a_hand_that_did_not_close() -> None:
    mode = hand_target_completion_mode(
        start=np.asarray([5.4, 5.4]),
        target=np.asarray([0.0, 3.84]),
        measured_history=_hand_history((4.8, 5.0)),
        tolerance_rad=0.15,
        allow_closing_contact=True,
    )
    assert mode is None

def test_hand_completion_rejects_contact_during_opening() -> None:
    mode = hand_target_completion_mode(
        start=np.asarray([0.0, 0.0]),
        target=np.asarray([5.4, 5.4]),
        measured_history=_hand_history((5.15, 5.20)),
        tolerance_rad=0.15,
        allow_closing_contact=True,
    )
    assert mode is None

def test_hand_completion_rejects_unstable_contact() -> None:
    history = _hand_history((0.23, 3.87))
    history[-1] = np.asarray([0.30, 3.87])
    mode = hand_target_completion_mode(
        start=np.asarray([5.4, 5.4]),
        target=np.asarray([0.0, 3.84]),
        measured_history=history,
        tolerance_rad=0.15,
        allow_closing_contact=True,
    )
    assert mode is None
