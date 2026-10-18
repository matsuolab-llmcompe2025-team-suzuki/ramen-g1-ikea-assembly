from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

FEATURE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FEATURE_ROOT))

from gr00t.g1_full_body_mapping import (
    REAL_G1_DEX1_ACTION_LOSS_EXCLUDED_INDICES,
    dex1_to_hand,
)
from inference.coarse_insert import (
    COARSE_INSERT_SDK_MOTOR_IDS,
    build_coarse_insert_state,
    decode_coarse_insert_action,
)


def test_build_coarse_insert_state_uses_sdk_order_and_dex1_synergy() -> None:
    state = build_coarse_insert_state(
        eef_pose_xyz_euler=[
            0.1,
            0.2,
            0.3,
            0.0,
            0.0,
            0.0,
            0.4,
            0.5,
            0.6,
            0.0,
            0.0,
            0.0,
        ],
        sdk_joint_positions=np.arange(29, dtype=np.float32),
        dex1_open_close=[4.5, 2.25],
    )

    assert state.shape == (49,)
    assert state[:9] == pytest.approx([0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0])
    assert state[9:18] == pytest.approx([0.4, 0.5, 0.6, 1, 0, 0, 0, 1, 0])
    assert state[18:25] == pytest.approx(dex1_to_hand(4.5, side="left", kind="state"))
    assert state[25:32] == pytest.approx(dex1_to_hand(2.25, side="right", kind="state"))
    assert state[32:39] == pytest.approx(np.arange(15, 22))
    assert state[39:46] == pytest.approx(np.arange(22, 29))
    assert state[46:49] == pytest.approx(np.arange(12, 15))


def test_decode_coarse_insert_action_outputs_only_sdk_body_and_dex1_commands() -> None:
    action = np.zeros((2, 53), dtype=np.float32)
    action[:, 18:25] = np.asarray(
        [dex1_to_hand(value, side="left", kind="action") for value in (1.5, 4.5)]
    )
    action[:, 25:32] = np.asarray(
        [dex1_to_hand(value, side="right", kind="action") for value in (1.5, 4.5)]
    )
    action[:, 32:39] = np.arange(15, 22)
    action[:, 39:46] = np.arange(22, 29)
    action[:, 46:49] = np.arange(12, 15)
    action[:, list(REAL_G1_DEX1_ACTION_LOSS_EXCLUDED_INDICES)] = 99.0
    limits = {sdk_id: (-50.0, 50.0) for sdk_id in COARSE_INSERT_SDK_MOTOR_IDS}
    limits[12] = (-1.0, 1.0)

    command = decode_coarse_insert_action(action, joint_limits=limits)

    assert command.sdk_motor_ids == tuple(range(12, 29))
    assert command.sdk_position_targets.shape == (2, 17)
    assert command.sdk_position_targets[:, 0] == pytest.approx([1.0, 1.0])
    assert command.sdk_position_targets[:, 3:] == pytest.approx(
        np.tile(np.arange(15, 29), (2, 1))
    )
    np.testing.assert_allclose(
        command.dex1_open_close_targets,
        [[1.5, 1.5], [4.5, 4.5]],
    )
    assert command.clipped_joint_values == 2
    # Base/navigation values are retained only in diagnostics and never enter
    # either executable command array.
    assert not np.any(command.sdk_position_targets == 99.0)


def test_coarse_insert_state_rejects_wrong_sdk_vector_size() -> None:
    with pytest.raises(ValueError, match="sdk_joint_positions"):
        build_coarse_insert_state(
            eef_pose_xyz_euler=np.zeros(12),
            sdk_joint_positions=np.zeros(36),
            dex1_open_close=np.zeros(2),
        )
