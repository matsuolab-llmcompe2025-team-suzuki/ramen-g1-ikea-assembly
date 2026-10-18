"""Four-camera Desktop preview unit tests."""

from __future__ import annotations

import numpy as np

from evaluate.model_evaluation.common.desktop_preview import (
    CAMERA_ROLES,
    _encode_preview_update,
    _fit_preview_frame,
    compose_four_camera_view_bgr,
)


class _JointState:
    position = np.linspace(-1.0, 1.0, 29, dtype=np.float64)


def test_compositor_keeps_canonical_role_layout() -> None:
    values = {
        "head_left": 20,
        "head_right": 60,
        "left_wrist": 100,
        "right_wrist": 140,
    }
    frames = {
        role: np.full((480, 640, 3), value, dtype=np.uint8)
        for role, value in values.items()
    }

    view = compose_four_camera_view_bgr(
        frames, np.zeros(14, dtype=np.float64), "POLICY INFERENCE"
    )

    assert view.shape == (960, 1280, 3)
    assert view.dtype == np.uint8
    # Sample away from title/HUD overlays to prove left/right role placement.
    np.testing.assert_array_equal(view[400, 500], [20, 20, 20])
    np.testing.assert_array_equal(view[400, 1100], [60, 60, 60])
    np.testing.assert_array_equal(view[900, 500], [100, 100, 100])
    np.testing.assert_array_equal(view[900, 1100], [140, 140, 140])


def test_preview_fit_preserves_non_native_frame_without_affecting_source() -> None:
    source = np.full((480, 848, 3), 77, dtype=np.uint8)
    source_copy = source.copy()

    fitted = _fit_preview_frame(source)

    assert fitted.shape == (480, 640, 3)
    np.testing.assert_array_equal(source, source_copy)
    # Letterboxing preserves the 848:480 aspect ratio instead of cropping data.
    assert np.all(fitted[0, 320] == 0)
    assert np.all(fitted[239, 320] == 77)


def test_preview_update_contains_all_cameras_and_arm14() -> None:
    observation = {
        "joint_state": _JointState(),
        "_camera_frames": {
            role: np.full((48, 64, 3), index * 30, dtype=np.uint8)
            for index, role in enumerate(CAMERA_ROLES, start=1)
        },
    }

    update = _encode_preview_update(observation, "PRE-MOTION")

    assert update is not None
    assert set(update["camera_jpeg"]) == set(CAMERA_ROLES)
    assert all(update["camera_jpeg"][role] for role in CAMERA_ROLES)
    np.testing.assert_allclose(
        update["arm_joint_position_rad"], _JointState.position[15:29]
    )
    assert update["status"] == "PRE-MOTION"


def test_preview_skips_incomplete_camera_set() -> None:
    observation = {
        "joint_state": _JointState(),
        "_camera_frames": {
            role: np.zeros((48, 64, 3), dtype=np.uint8)
            for role in CAMERA_ROLES[:-1]
        },
    }

    assert _encode_preview_update(observation, "CAMERAS READY") is None
