"""joint_mapping.py の unit test (pure Python、rclpy / SDK 依存なし)。

real_hw_bridge_node.py は rclpy import を top-level に持つので main pixi env では
collect できない。ここでは SDK / rclpy から切り離した pure module (joint_mapping) を
テストして、joint 名の順序 / SDK motor_state からの positions 抽出 logic を守る。
"""

from __future__ import annotations

from dataclasses import dataclass

from g1_hw_bridge.joint_mapping import G1_JOINT_NAMES, G1_NUM_MOTOR, extract_positions


@dataclass
class _FakeMotorState:
    q: float
    dq: float = 0.0
    tau_est: float = 0.0


def test_joint_names_has_29_entries():
    assert G1_NUM_MOTOR == 29
    assert len(G1_JOINT_NAMES) == 29


def test_joint_names_are_unique():
    assert len(set(G1_JOINT_NAMES)) == len(G1_JOINT_NAMES)


def test_joint_names_match_urdf_convention():
    # URDF g1_29dof_mode_15_with_dex1_1.urdf の revolute joint と一対一対応であること
    # の代表 sample。全 29 名の順序は G1JointIndex enum に従う (SDK 側との contract)。
    assert G1_JOINT_NAMES[0] == "left_hip_pitch_joint"
    assert G1_JOINT_NAMES[3] == "left_knee_joint"
    assert G1_JOINT_NAMES[12] == "waist_yaw_joint"  # G1JointIndex.WaistYaw = 12
    assert G1_JOINT_NAMES[15] == "left_shoulder_pitch_joint"  # LeftShoulderPitch = 15
    assert G1_JOINT_NAMES[28] == "right_wrist_yaw_joint"  # RightWristYaw = 28


def test_extract_positions_returns_29_floats_in_order():
    # SDK motor_state を模した fake list (実 SDK は固定長 array だが indexable + .q を
    # 持てば OK)。長さは 35 (SDK 側の余分な要素を模擬)、先頭 29 個だけ抽出される。
    motor_state = [_FakeMotorState(q=i * 0.01) for i in range(35)]
    positions = extract_positions(motor_state)
    assert len(positions) == G1_NUM_MOTOR
    for i, value in enumerate(positions):
        assert value == float(i) * 0.01


def test_extract_positions_returns_python_floats():
    # numpy.float32 等が混じっても Python の float に cast される (ROS2
    # sensor_msgs/JointState の position は float64 seq)。
    class _NumpyLikeQ:
        def __init__(self, value):
            self._value = value

        def __float__(self):
            return float(self._value)

    class _MS:
        def __init__(self, i):
            self.q = _NumpyLikeQ(i * 0.001)

    motor_state = [_MS(i) for i in range(G1_NUM_MOTOR)]
    positions = extract_positions(motor_state)
    assert all(isinstance(v, float) for v in positions)
    assert positions[5] == 5 * 0.001
