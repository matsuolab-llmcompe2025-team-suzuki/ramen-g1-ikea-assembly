from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


UNITREE_G1_DEX1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
]


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare("g1_bringup"),
        "config",
        "system.yaml",
    ])
    urdf_file = LaunchConfiguration("urdf_file")
    urdf = PathJoinSubstitution([
        FindPackageShare("g1_description"),
        "urdf",
        urdf_file,
    ])
    rviz_config = PathJoinSubstitution([
        FindPackageShare("g1_description"),
        "rviz",
        "dataset_replay.rviz",
    ])

    dataset_root = LaunchConfiguration("dataset_root")
    parquet_file = LaunchConfiguration("parquet_file")
    episode_index = LaunchConfiguration("episode_index")
    use_action = LaunchConfiguration("use_action")
    loop = LaunchConfiguration("loop")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    frame_stride = LaunchConfiguration("frame_stride")
    max_frames = LaunchConfiguration("max_frames")
    rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "dataset_root",
            default_value="/workspace/datasets/G1_WBT_Dex1_Building-Children-Table",
            description="Local LeRobot dataset root.",
        ),
        DeclareLaunchArgument(
            "parquet_file",
            default_value="",
            description="Specific LeRobot parquet file. Overrides dataset_root.",
        ),
        DeclareLaunchArgument(
            "urdf_file",
            default_value="unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf",
            description="URDF path under g1_description/urdf.",
        ),
        DeclareLaunchArgument("episode_index", default_value="-1"),
        DeclareLaunchArgument("use_action", default_value="false"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("publish_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("frame_stride", default_value="1"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": ParameterValue(Command(["cat ", urdf]), value_type=str),
            }],
        ),
        Node(
            package="g1_dataset_replay",
            executable="dataset_replay_node",
            name="g1_dataset_replay",
            output="screen",
            parameters=[
                config,
                {
                    "dataset_root": dataset_root,
                    "parquet_file": parquet_file,
                    "episode_index": ParameterValue(episode_index, value_type=int),
                    "use_action": ParameterValue(use_action, value_type=bool),
                    "loop": ParameterValue(loop, value_type=bool),
                    "publish_rate_hz": ParameterValue(publish_rate_hz, value_type=float),
                    "frame_stride": ParameterValue(frame_stride, value_type=int),
                    "max_frames": ParameterValue(max_frames, value_type=int),
                    "joint_names": UNITREE_G1_DEX1_JOINT_NAMES,
                },
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(rviz),
        ),
    ])
