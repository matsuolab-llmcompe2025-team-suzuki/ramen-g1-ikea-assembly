"""G1 29dof の SDK motor index ↔ URDF joint 名 の mapping。

Unitree SDK の `unitree_hg.msg.dds_.LowState_.motor_state[i]` (i = 0..28) は
`G1JointIndex` enum の順で入っている (source: unitree_sdk2_python の
`example/g1/low_level/g1_low_level_example.py:34-69`)。ここではその index 順を
URDF `g1_29dof_mode_15_with_dex1_1.urdf` の revolute joint 名と一対一で対応させる。

`floating_base_joint` (type=floating) と Dex1-1 の hand joint は SDK 経由では
取れないので、この bridge の scope 外 (hand は Dex1-1 SDK が別途担当)。
"""

from __future__ import annotations

# SDK motor_state[i] index i = 0..28 に対応する URDF joint 名。
# 順序は unitree_sdk2py の G1JointIndex enum と URDF の revolute joint 順が
# 一致していることを前提とする (2026-07-15 検証: `grep type="revolute"` で
# 29 joint、G1JointIndex 0..28 と一対一)。
G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",       # 0  LeftHipPitch
    "left_hip_roll_joint",        # 1  LeftHipRoll
    "left_hip_yaw_joint",         # 2  LeftHipYaw
    "left_knee_joint",            # 3  LeftKnee
    "left_ankle_pitch_joint",     # 4  LeftAnklePitch (mode_pr) / LeftAnkleB (mode_ab)
    "left_ankle_roll_joint",      # 5  LeftAnkleRoll  (mode_pr) / LeftAnkleA (mode_ab)
    "right_hip_pitch_joint",      # 6  RightHipPitch
    "right_hip_roll_joint",       # 7  RightHipRoll
    "right_hip_yaw_joint",        # 8  RightHipYaw
    "right_knee_joint",           # 9  RightKnee
    "right_ankle_pitch_joint",    # 10 RightAnklePitch / RightAnkleB
    "right_ankle_roll_joint",     # 11 RightAnkleRoll  / RightAnkleA
    "waist_yaw_joint",            # 12 WaistYaw
    "waist_roll_joint",           # 13 WaistRoll  (29dof waist-locked では常に 0)
    "waist_pitch_joint",          # 14 WaistPitch (29dof waist-locked では常に 0)
    "left_shoulder_pitch_joint",  # 15 LeftShoulderPitch
    "left_shoulder_roll_joint",   # 16 LeftShoulderRoll
    "left_shoulder_yaw_joint",    # 17 LeftShoulderYaw
    "left_elbow_joint",           # 18 LeftElbow
    "left_wrist_roll_joint",      # 19 LeftWristRoll
    "left_wrist_pitch_joint",     # 20 LeftWristPitch
    "left_wrist_yaw_joint",       # 21 LeftWristYaw
    "right_shoulder_pitch_joint", # 22 RightShoulderPitch
    "right_shoulder_roll_joint",  # 23 RightShoulderRoll
    "right_shoulder_yaw_joint",   # 24 RightShoulderYaw
    "right_elbow_joint",          # 25 RightElbow
    "right_wrist_roll_joint",     # 26 RightWristRoll
    "right_wrist_pitch_joint",    # 27 RightWristPitch
    "right_wrist_yaw_joint",      # 28 RightWristYaw
)

G1_NUM_MOTOR: int = len(G1_JOINT_NAMES)


def extract_positions(motor_states) -> list[float]:
    """`LowState_.motor_state` から先頭 29 個の `q` を list[float] で返す。

    Args:
        motor_states: `LowState_.motor_state` (実 SDK では固定長 array、
            test では `dataclass(q=float)` の list など indexable + `.q` を持つ
            もの)。

    Returns:
        `G1_JOINT_NAMES` と同じ順の joint position (radian)。長さは
        `G1_NUM_MOTOR` (=29)。

    Note:
        SDK の `motor_state` array は 35 要素 (unitree_hg で SDK 側は
        余分な要素あり) だが、G1 body が使うのは先頭 29 個のみ。
    """
    return [float(motor_states[i].q) for i in range(G1_NUM_MOTOR)]
