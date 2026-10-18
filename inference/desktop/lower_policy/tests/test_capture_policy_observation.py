from types import SimpleNamespace

import numpy as np
import pytest

from inference.desktop.lower_policy.scripts.capture_policy_observation import (
    _received_ns,
    _split_head_bundle,
    _validate_frames,
)
from inference.desktop.perception.frame_source import FrameData


def test_received_ns_requires_host_monotonic_timestamp():
    assert _received_ns(SimpleNamespace(received_monotonic_ns=123)) == 123
    with pytest.raises(ValueError, match="host receive timestamp"):
        _received_ns(SimpleNamespace(received_monotonic_ns=None))


def test_validate_frames_accepts_native_d405_width():
    frames = {
        "head_left": SimpleNamespace(
            rgb=np.ones((480, 640, 3), dtype=np.uint8)
        ),
        "head_right": SimpleNamespace(
            rgb=np.ones((480, 640, 3), dtype=np.uint8)
        ),
        "wrist_left": SimpleNamespace(
            rgb=np.ones((480, 848, 3), dtype=np.uint8)
        ),
        "wrist_right": SimpleNamespace(
            rgb=np.ones((480, 848, 3), dtype=np.uint8)
        ),
    }

    _validate_frames(frames)


def test_split_head_bundle_subscribes_once_but_produces_two_logical_views():
    left = np.full((480, 640, 3), 10, dtype=np.uint8)
    right = np.full((480, 640, 3), 220, dtype=np.uint8)
    wrist = np.ones((480, 848, 3), dtype=np.uint8)
    bundle = {
        "head_stereo": FrameData(
            rgb=np.concatenate([left, right], axis=1),
            t=42,
            received_monotonic_ns=100,
        ),
        "wrist_left": FrameData(wrist, 1, 101),
        "wrist_right": FrameData(wrist, 2, 102),
    }

    logical = _split_head_bundle(bundle)

    assert logical["head_left"].rgb.mean() == 10
    assert logical["head_right"].rgb.mean() == 220
    assert logical["head_left"].t == logical["head_right"].t == 42
    assert logical["head_left"].received_monotonic_ns == 100
