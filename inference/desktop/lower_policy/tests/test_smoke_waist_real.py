from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.scripts.smoke_waist_real import (
    _Step,
    _parse_sequence,
    _target_for_step,
)


def test_parse_sequence_accepts_bounded_relative_offsets() -> None:
    assert _parse_sequence("yaw:0.08,pause:1,zero:0") == [
        _Step("yaw", 0.08),
        _Step("pause", 1.0),
        _Step("zero", 0.0),
    ]


@pytest.mark.parametrize(
    "raw",
    ["yaw:0.101", "roll:nan", "pitch:inf", "pause:0", "zero:1"],
)
def test_parse_sequence_rejects_unsafe_values(raw: str) -> None:
    with pytest.raises(ValueError):
        _parse_sequence(raw)


def test_each_motion_step_offsets_only_one_axis_from_measured_initial() -> None:
    initial = np.asarray([0.02, -0.03, 0.04])
    assert np.allclose(
        _target_for_step(initial, _Step("yaw", 0.08)),
        [0.10, -0.03, 0.04],
    )
    assert np.allclose(
        _target_for_step(initial, _Step("roll", -0.08)),
        [0.02, -0.11, 0.04],
    )
    assert np.allclose(
        _target_for_step(initial, _Step("pitch", 0.08)),
        [0.02, -0.03, 0.12],
    )
    assert np.array_equal(
        _target_for_step(initial, _Step("zero", 0.0)), initial
    )
