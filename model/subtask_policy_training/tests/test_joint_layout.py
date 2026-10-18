"""Issue #129 Phase 0 D0-4: joint_layout.py の自己整合性 + 外部 config との一致 test。

- validate(): 定数間の一貫性 (重複なし、部位別 index の総和 = 19 dim)
- test_names_match_subtask_config: subtask_training.json の action.names / state.names と一致
- test_sdk29_mapping_matches_urdf_module: g1_urdf_fk.G1_JOINT_NAMES の該当 name と一致
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model.subtask_policy_training import joint_layout


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBTASK_CONFIG_PATH = REPO_ROOT / "model" / "subtask_policy_training" / "configs" / "subtask_training.json"


def test_validate_self_consistency() -> None:
    """定数間の一貫性、joint_layout.validate() が例外出さない。"""
    joint_layout.validate()


def test_names_match_subtask_training_json() -> None:
    """subtask_training.json の action.names と GROOT_ACTION_NAMES が一致。"""
    cfg = json.loads(SUBTASK_CONFIG_PATH.read_text(encoding="utf-8"))
    action_names = cfg["action"]["names"]
    # joint_layout 側は "*_target" 接尾辞なしで保持、config は "*_target" 付き
    expected = [n + "_target" if not n.endswith("_q") else n + "_cmd" for n in joint_layout.GROOT_ACTION_NAMES]
    assert action_names == expected, f"mismatch:\n  config={action_names}\n  layout={expected}"

    state_names = cfg["state"]["names"]
    assert list(joint_layout.GROOT_ACTION_NAMES) == state_names


def test_sdk29_mapping_matches_g1_joint_names() -> None:
    """GROOT_TO_SDK29_INDEX が g1_urdf_fk.G1_JOINT_NAMES と一致。"""
    try:
        from inference.desktop.perception.g1_urdf_fk import G1_JOINT_NAMES
    except ImportError:
        pytest.skip("g1_urdf_fk import 不可 (env 依存)、skip")

    for groot_idx, sdk_idx in enumerate(joint_layout.GROOT_TO_SDK29_INDEX):
        if sdk_idx == -1:
            # gripper は SDK 外なので skip
            groot_name = joint_layout.GROOT_ACTION_NAMES[groot_idx]
            assert "gripper" in groot_name, f"non-gripper joint {groot_name} has -1 SDK index"
            continue
        assert G1_JOINT_NAMES[sdk_idx] == joint_layout.GROOT_ACTION_NAMES[groot_idx], (
            f"GR00T[{groot_idx}]={joint_layout.GROOT_ACTION_NAMES[groot_idx]} "
            f"vs SDK[{sdk_idx}]={G1_JOINT_NAMES[sdk_idx]}"
        )


def test_left_right_symmetry() -> None:
    """LEFT_ARM と RIGHT_ARM が同数、LEFT_HAND と RIGHT_HAND も同数。"""
    assert len(joint_layout.LEFT_ARM_INDICES) == len(joint_layout.RIGHT_ARM_INDICES) == 7
    assert len(joint_layout.LEFT_HAND_INDICES) == len(joint_layout.RIGHT_HAND_INDICES) == 8
    assert len(joint_layout.BILATERAL_LEFT_INDICES) == len(joint_layout.BILATERAL_RIGHT_INDICES)
