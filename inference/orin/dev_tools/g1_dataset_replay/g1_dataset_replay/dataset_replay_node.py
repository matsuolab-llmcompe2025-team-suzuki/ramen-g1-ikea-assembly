from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String


DEFAULT_JOINTS = [f"g1_joint_{index:02d}" for index in range(36)]


def _flatten(values: Any) -> List[float]:
    if values is None:
        return []
    if hasattr(values, "as_py"):
        values = values.as_py()
    if isinstance(values, (int, float)):
        return [float(values)]
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, bytearray)):
        flattened: List[float] = []
        for item in values:
            flattened.extend(_flatten(item))
        return flattened
    return []


def _pad_or_trim(values: List[float], dim: int) -> List[float]:
    if len(values) < dim:
        values = list(values) + [0.0] * (dim - len(values))
    return values[:dim]


class DatasetReplayNode(Node):
    def __init__(self) -> None:
        super().__init__("g1_dataset_replay")

        self.declare_parameter("dataset_root", "/workspace/datasets/G1_WBT_Dex1_Building-Children-Table")
        self.declare_parameter("parquet_file", "")
        self.declare_parameter("episode_index", -1)
        self.declare_parameter("use_action", False)
        self.declare_parameter("loop", True)
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("frame_stride", 1)
        self.declare_parameter("max_frames", 0)
        self.declare_parameter("joint_names", DEFAULT_JOINTS)
        self.declare_parameter("hand_names", ["left_hand", "right_hand"])
        self.declare_parameter("robot_q_state_key", "observation.state.robot_q_current")
        self.declare_parameter("hand_state_key", "observation.state.hand_state")
        self.declare_parameter("ee_state_key", "observation.state.ee_state")
        self.declare_parameter("robot_q_action_key", "action.robot_q_desired")
        self.declare_parameter("hand_action_key", "action.hand_cmd")

        self._joint_names = list(self.get_parameter("joint_names").value)
        self._hand_names = list(self.get_parameter("hand_names").value)
        self._use_action = bool(self.get_parameter("use_action").value)
        self._loop = bool(self.get_parameter("loop").value)
        self._frame_stride = max(1, int(self.get_parameter("frame_stride").value))
        self._index = 0
        self._finished = False

        self._joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._hand_pub = self.create_publisher(Float32MultiArray, "/g1/hand_state", 10)
        self._ee_pub = self.create_publisher(Float32MultiArray, "/g1/ee_state", 10)
        self._status_pub = self.create_publisher(String, "/g1/replay/status", 10)

        self._frames = self._load_frames()
        if not self._frames:
            self.get_logger().error("no replay frames loaded")
            return

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_next)
        self._publish_status({"event": "replay_ready", "frames": len(self._frames), "use_action": self._use_action})

    def _load_frames(self) -> List[Dict[str, Any]]:
        parquet_path = self._resolve_parquet_file()
        if parquet_path is None:
            return []

        try:
            import pyarrow.parquet as pq
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"pyarrow is required for dataset replay: {exc}")
            return []

        state_keys = [
            str(self.get_parameter("robot_q_state_key").value),
            str(self.get_parameter("hand_state_key").value),
            str(self.get_parameter("ee_state_key").value),
        ]
        action_keys = [
            str(self.get_parameter("robot_q_action_key").value),
            str(self.get_parameter("hand_action_key").value),
        ]
        selected_keys = action_keys if self._use_action else state_keys
        metadata_keys = ["timestamp", "frame_index", "episode_index", "task_index"]

        try:
            parquet = pq.ParquetFile(str(parquet_path))
            available = set(parquet.schema_arrow.names)
            columns = [key for key in selected_keys + metadata_keys if key in available]
            missing = [key for key in selected_keys if key not in available]
            if missing:
                self.get_logger().warn(f"missing replay columns in {parquet_path}: {missing}")
            table = parquet.read(columns=columns)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"failed to read parquet file {parquet_path}: {exc}")
            return []

        data = {name: table[name].to_pylist() for name in table.column_names}
        row_count = table.num_rows
        indices = list(range(row_count))

        episode_index = int(self.get_parameter("episode_index").value)
        if episode_index >= 0 and "episode_index" in data:
            indices = [index for index in indices if int(data["episode_index"][index]) == episode_index]

        indices = indices[:: self._frame_stride]
        max_frames = int(self.get_parameter("max_frames").value)
        if max_frames > 0:
            indices = indices[:max_frames]

        frames: List[Dict[str, Any]] = []
        for index in indices:
            robot_q = _flatten(data.get(selected_keys[0], [])[index]) if selected_keys[0] in data else []
            hand = _flatten(data.get(selected_keys[1], [])[index]) if selected_keys[1] in data else []
            ee = (
                _flatten(data.get(state_keys[2], [])[index])
                if not self._use_action and state_keys[2] in data
                else []
            )
            frames.append(
                {
                    "robot_q": _pad_or_trim(robot_q, len(self._joint_names)),
                    "hand": _pad_or_trim(hand, len(self._hand_names)),
                    "ee": ee,
                    "timestamp": float(data["timestamp"][index]) if "timestamp" in data else 0.0,
                    "frame_index": int(data["frame_index"][index]) if "frame_index" in data else index,
                    "episode_index": int(data["episode_index"][index]) if "episode_index" in data else -1,
                    "task_index": int(data["task_index"][index]) if "task_index" in data else -1,
                }
            )

        self.get_logger().info(
            f"loaded {len(frames)} frames from {parquet_path} "
            f"source={'action' if self._use_action else 'observation'}"
        )
        return frames

    def _resolve_parquet_file(self) -> Optional[Path]:
        explicit = str(self.get_parameter("parquet_file").value).strip()
        if explicit:
            path = Path(explicit).expanduser()
            if path.is_file():
                return path
            self.get_logger().error(f"parquet_file does not exist: {path}")
            return None

        root = Path(str(self.get_parameter("dataset_root").value)).expanduser()
        candidates = sorted(root.glob("data/chunk-*/file-*.parquet"))
        if candidates:
            return candidates[0]
        self.get_logger().error(
            f"no parquet file found under {root}. "
            "Set parquet_file:=/path/to/file.parquet or dataset_root:=/path/to/LeRobot dataset."
        )
        return None

    def _publish_next(self) -> None:
        if self._index >= len(self._frames):
            if not self._loop:
                if not self._finished:
                    self._finished = True
                    self._publish_status({"event": "replay_finished"})
                return
            self._index = 0
            self._finished = False

        frame = self._frames[self._index]
        stamp = self.get_clock().now().to_msg()

        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = list(self._joint_names) + list(self._hand_names)
        joint_state.position = list(frame["robot_q"]) + list(frame["hand"])
        joint_state.velocity = [0.0 for _ in joint_state.name]
        self._joint_pub.publish(joint_state)

        hand = Float32MultiArray()
        hand.data = list(frame["hand"])
        self._hand_pub.publish(hand)

        ee = Float32MultiArray()
        ee.data = list(frame["ee"])
        self._ee_pub.publish(ee)

        status_interval = max(1, int(float(self.get_parameter("publish_rate_hz").value)))
        if self._index % status_interval == 0:
            self._publish_status(
                {
                    "event": "replay_frame",
                    "index": self._index,
                    "frame_index": frame["frame_index"],
                    "episode_index": frame["episode_index"],
                    "task_index": frame["task_index"],
                    "timestamp": round(frame["timestamp"], 3),
                }
            )
        self._index += 1

    def _publish_status(self, payload: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self._status_pub.publish(msg)
        self.get_logger().info(msg.data)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetReplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
