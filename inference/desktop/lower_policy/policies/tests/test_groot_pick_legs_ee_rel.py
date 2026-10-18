"""No-actuation regression tests for the 56-D/20-D pick-leg model."""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.kinematics.g1_arm import (
    DEX1_BASE_OFFSET,
    G1ArmKinematics,
)
from inference.desktop.lower_policy.kinematics.types import Side
from inference.desktop.lower_policy.policies.base import Observation, RawRobotState
from inference.desktop.lower_policy.policies.groot_pick_legs_ee_rel import (
    _EERelativePickWorkerClient,
    build_state_from_raw,
    rot6d_rows_to_matrix,
)
from inference.desktop.perception.g1_urdf_fk import (
    G1WristFK,
    LEFT_WRIST_TOOL_OFFSET_M,
    RIGHT_WRIST_TOOL_OFFSET_M,
)
from inference.desktop.upper_policy.groot_pick_leg_ee_rel_contract import (
    MODEL_REPO_ID,
    MODEL_REVISION,
    validate_ee_relative_checkpoint_metadata,
)


def _client_without_worker() -> _EERelativePickWorkerClient:
    client = object.__new__(_EERelativePickWorkerClient)
    client._ik = {
        Side.LEFT: G1ArmKinematics(
            tcp_offset=LEFT_WRIST_TOOL_OFFSET_M - np.asarray(DEX1_BASE_OFFSET),
            limit_rad=2.1,
        ),
        Side.RIGHT: G1ArmKinematics(
            tcp_offset=RIGHT_WRIST_TOOL_OFFSET_M - np.asarray(DEX1_BASE_OFFSET),
            limit_rad=2.1,
        ),
    }
    return client


def _observation() -> Observation:
    joints = np.zeros(29, dtype=np.float32)
    ee_state = G1WristFK.from_urdf().compute_ee_state(joints)
    state = build_state_from_raw(
        RawRobotState(
            joint_positions=joints,
            hand_state=np.asarray((4.5, 4.5), dtype=np.float32),
            ee_state=ee_state,
        )
    )
    return Observation(
        frames_bgr={},
        frames_bgr_prev=None,
        state=state,
        skill_id=None,
        language="pick table leg",
        obb_detections=None,
        timestamp_ns=1,
    )


def test_state_order_and_identity_action_returns_measured_pose() -> None:
    obs = _observation()
    assert obs.state.shape == (56,)
    np.testing.assert_allclose(obs.state[18:20], (4.5, 4.5))
    raw = np.tile(obs.state[:20], (16, 1))
    action = _client_without_worker()._decode_actions(raw, obs, 5.0)
    assert action.action_chunk.shape == (4, 19)
    np.testing.assert_allclose(action.action_chunk[:, :17], 0, atol=1e-5)
    np.testing.assert_allclose(action.action_chunk[:, 17:], 4.5)
    assert action.metadata["lower_body_command_dimensions"] == 0


def test_degenerate_rotation_is_rejected() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        rot6d_rows_to_matrix(np.zeros(6))


def test_ik_frame_mismatch_fails_closed() -> None:
    obs = _observation()
    bad_state = obs.state.copy()
    bad_state[0] += 0.1
    obs = Observation(
        frames_bgr=obs.frames_bgr,
        frames_bgr_prev=None,
        state=bad_state,
        skill_id=None,
        language=obs.language,
        obb_detections=None,
        timestamp_ns=1,
    )
    raw = np.tile(obs.state[:20], (16, 1))
    with pytest.raises(RuntimeError, match="frame mismatch"):
        _client_without_worker()._decode_actions(raw, obs, 5.0)


def test_checkpoint_metadata_is_pinned() -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        checkpoint = snapshot_download(
            MODEL_REPO_ID, revision=MODEL_REVISION, local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pytest.skip("sealed HF checkpoint is not cached")
    contract = validate_ee_relative_checkpoint_metadata(checkpoint)
    assert contract["state_dim"] == 56
    assert contract["decoded_action_dim"] == 20
    assert contract["action_horizon"] == 16
    assert contract["lower_body_command_dimensions"] == 0
