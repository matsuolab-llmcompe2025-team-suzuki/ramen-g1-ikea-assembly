import json
from typing import Dict, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32MultiArray, String
from trajectory_msgs.msg import JointTrajectory


DEFAULT_JOINTS = [
    f"g1_joint_{index:02d}" for index in range(36)
]
DEFAULT_IMAGE_TOPICS = {
    "observation.images.cam_0": "/g1/cameras/cam_0/image_raw",
    "observation.images.cam_1": "/g1/cameras/cam_1/image_raw",
    "observation.images.cam_2": "/g1/cameras/cam_2/image_raw",
    "observation.images.cam_3": "/g1/cameras/cam_3/image_raw",
}


class MockHardwareBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("g1_mock_hw_bridge")

        self.declare_parameter("joint_names", DEFAULT_JOINTS)
        self.declare_parameter("hand_names", ["left_hand", "right_hand"])
        self.declare_parameter("hand_dim", 2)
        self.declare_parameter("ee_state_dim", 12)
        self.declare_parameter("joint_target_topic", "/g1/joint_target")
        self.declare_parameter("hand_command_topic", "/g1/hand/command")
        self.declare_parameter("publish_rate_hz", 25.0)
        self.declare_parameter("publish_images", True)
        self.declare_parameter("image_topics_json", json.dumps(DEFAULT_IMAGE_TOPICS))
        self.declare_parameter("image_height", 480)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_publish_rate_hz", 5.0)
        self.declare_parameter("log_commands", True)
        self.declare_parameter("motion_enabled_default", True)

        self._joint_names = list(self.get_parameter("joint_names").value)
        self._hand_names = list(self.get_parameter("hand_names").value)
        self._joint_positions: Dict[str, float] = {name: 0.0 for name in self._joint_names}
        self._hand_positions = [0.0 for _ in range(int(self.get_parameter("hand_dim").value))]
        self._ee_state = [0.0 for _ in range(int(self.get_parameter("ee_state_dim").value))]
        self._motion_enabled = bool(self.get_parameter("motion_enabled_default").value)
        self._log_commands = bool(self.get_parameter("log_commands").value)
        self._image_height = int(self.get_parameter("image_height").value)
        self._image_width = int(self.get_parameter("image_width").value)
        self._image_data = bytes(max(self._image_height, 1) * max(self._image_width, 1) * 3)

        self.create_subscription(
            JointTrajectory,
            str(self.get_parameter("joint_target_topic").value),
            self._on_joint_target,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("hand_command_topic").value),
            self._on_hand_command,
            10,
        )
        self.create_subscription(Bool, "/g1/motion_enable", self._on_motion_enable, 10)

        self._joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._robot_state_pub = self.create_publisher(String, "/g1/robot_state", 10)
        self._hand_state_pub = self.create_publisher(Float32MultiArray, "/g1/hand_state", 10)
        self._ee_state_pub = self.create_publisher(Float32MultiArray, "/g1/ee_state", 10)
        self._image_pubs = {
            key: self.create_publisher(Image, topic, 10)
            for key, topic in self._parse_image_topics().items()
            if topic
        }

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_state)
        if bool(self.get_parameter("publish_images").value) and self._image_pubs:
            image_rate = float(self.get_parameter("image_publish_rate_hz").value)
            self.create_timer(1.0 / max(image_rate, 0.1), self._publish_images)
        self.get_logger().info("mock hardware bridge ready")

    def _parse_image_topics(self) -> Dict[str, str]:
        topics = dict(DEFAULT_IMAGE_TOPICS)
        raw_json = str(self.get_parameter("image_topics_json").value)
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"invalid image_topics_json: {exc}")
            return topics
        if isinstance(parsed, dict):
            topics.update({str(key): str(value) for key, value in parsed.items()})
        return topics

    def _on_joint_target(self, msg: JointTrajectory) -> None:
        if not self._motion_enabled:
            self._log("ignored joint target while motion disabled")
            return
        if not msg.points:
            return
        point = msg.points[-1]
        for name, position in zip(msg.joint_names, point.positions):
            if name in self._joint_positions:
                self._joint_positions[name] = float(position)
        self._log(f"accepted joint target for {len(msg.joint_names)} joints")

    def _on_hand_command(self, msg: Float32MultiArray) -> None:
        if not self._motion_enabled:
            self._log("ignored hand command while motion disabled")
            return
        if msg.data:
            values = [float(value) for value in msg.data]
            if len(values) < len(self._hand_positions):
                values.extend([0.0] * (len(self._hand_positions) - len(values)))
            self._hand_positions = values[: len(self._hand_positions)]
        self._log(f"accepted hand command: {[round(value, 2) for value in self._hand_positions]}")

    def _on_motion_enable(self, msg: Bool) -> None:
        if msg.data != self._motion_enabled:
            self.get_logger().info(f"motion_enabled={msg.data}")
        self._motion_enabled = bool(msg.data)

    def _publish_state(self) -> None:
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = list(self._joint_names) + list(self._hand_names)
        joint_state.position = [self._joint_positions[name] for name in self._joint_names] + list(
            self._hand_positions
        )
        joint_state.velocity = [0.0 for _ in joint_state.name]
        self._joint_state_pub.publish(joint_state)

        hand = Float32MultiArray()
        hand.data = list(self._hand_positions)
        self._hand_state_pub.publish(hand)

        ee = Float32MultiArray()
        ee.data = list(self._ee_state)
        self._ee_state_pub.publish(ee)

        robot_state = String()
        robot_state.data = json.dumps(
            {
                "motion_enabled": self._motion_enabled,
                "hand_position": [round(value, 3) for value in self._hand_positions],
                "ee_state_dim": len(self._ee_state),
            },
            sort_keys=True,
        )
        self._robot_state_pub.publish(robot_state)

    def _publish_images(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for image_key, publisher in self._image_pubs.items():
            image = Image()
            image.header.stamp = stamp
            image.header.frame_id = image_key.removeprefix("observation.images.")
            image.height = self._image_height
            image.width = self._image_width
            image.encoding = "rgb8"
            image.is_bigendian = 0
            image.step = self._image_width * 3
            image.data = self._image_data
            publisher.publish(image)

    def _log(self, message: str) -> None:
        if self._log_commands:
            self.get_logger().info(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockHardwareBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
