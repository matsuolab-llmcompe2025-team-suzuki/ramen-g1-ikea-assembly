import os
from glob import glob

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


DEFAULT_HEAD_CAMERA_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-STYT_260320_K_USB_Camera_01.00.00-video-index0"
)

# Issue #75 実機疎通機体 (2026-07-19 作業員 report) の D405 serial。
# 左右 mapping は operator が実映像で確認後に決定 (現状は物理 bus 番号順の初期 guess:
# bus 2-2.1 = 128422271048 = 左、bus 2-2.2 = 128422271925 = 右)。
# 実映像で swap が必要なら WRIST_LEFT_SERIAL / WRIST_RIGHT_SERIAL 環境変数 or
# launch arg で override 可能。
# RealSense API が報告するD405固有serial。/dev/v4l/by-idの文字列とは異なる。
# 実機で確認: bus 2-2.1 = 128422271048、bus 2-2.2 = 128422271925。
DEFAULT_WRIST_LEFT_SERIAL = "128422271048"
DEFAULT_WRIST_RIGHT_SERIAL = "128422271925"
DEFAULT_WRIST_LEFT_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-Intel_R__RealSense_TM__Depth_Camera_405_"
    "Intel_R__RealSense_TM__Depth_Camera_405_128223071636-video-index4"
)
DEFAULT_WRIST_RIGHT_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-Intel_R__RealSense_TM__Depth_Camera_405_"
    "Intel_R__RealSense_TM__Depth_Camera_405_130523070154-video-index4"
)


def _launch_head_camera(context):
    configured_device = LaunchConfiguration("head_camera_device").perform(context)
    if not os.path.exists(configured_device):
        candidates = [
            path
            for path in glob("/dev/v4l/by-id/*-video-index0")
            if "RealSense" not in os.path.basename(path)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "head camera by-id path is missing and automatic discovery "
                f"is ambiguous: configured={configured_device!r}, "
                f"candidates={candidates}"
            )
        configured_device = candidates[0]

    # usb_cam 0.6.x mishandles relative /dev/v4l/by-id symlinks, so retain the
    # stable external configuration but hand the driver a canonical /dev/videoN.
    resolved_device = os.path.realpath(configured_device)
    if not os.path.exists(resolved_device):
        raise RuntimeError(
            f"head camera device does not exist: {configured_device} -> "
            f"{resolved_device}"
        )
    sysfs_name = f"/sys/class/video4linux/{os.path.basename(resolved_device)}/name"
    try:
        with open(sysfs_name, encoding="utf-8") as stream:
            device_name = stream.read().strip()
    except OSError:
        device_name = ""
    if "RealSense" in device_name:
        raise RuntimeError(
            "refusing to start the head usb_cam on a RealSense endpoint: "
            f"{configured_device} -> {resolved_device} ({device_name})"
        )

    return [
        LogInfo(msg=(
            f"Head camera device: {configured_device} -> {resolved_device}"
        )),
        Node(
            package="usb_cam",
            executable="usb_cam_node_exe",
            namespace="head/camera/color",
            parameters=[{
                "video_device": resolved_device,
                "pixel_format": "mjpeg2rgb",
                "image_width": 1280,
                "image_height": 480,
                "framerate": 30.0,
                "camera_name": "head_camera",
                "camera_frame_id": "head_camera_optical_frame",
                # usb_cam 0.6 declares brightness=50 when omitted, even when
                # the UVC device default is 0.  That made the black work mat
                # drive auto-exposure into severe white clipping.  The
                # Team-RAMEN flip-table RGB data is consistent with the
                # neutral UVC offset (0), auto exposure, and auto white
                # balance, so make that profile explicit and leave all
                # scene-specific software colour correction disabled.
                "brightness": 0,
                "auto_white_balance": True,
                "autoexposure": True,
                # 運営データ収集コードと圧縮条件を揃える (default 95 → 90)。
                "image_raw.compressed.jpeg_quality": 90,
            }],
            output="screen",
        ),
    ]


def _launch_wrist_camera(context, side: str):
    """Launch one D405 wrist camera while preserving the canonical ROS topic.

    Normal inference uses the RealSense driver selected by immutable camera
    serial.  The UVC path remains an explicit diagnostic fallback only.  Both paths publish
    `/wrist_{side}/camera/color/image_raw{,/compressed}`.

    CLAUDE.md Key Facts: **D405 IR projector is OFF** (passive stereo design)。
    depth_module.emitter_enabled=0 で projector 無効化 (dataset G1_WBT と整合)。

    Publish topics (namespace prefix `/wrist_{side}/camera/` 付き):
      - color/image_raw + color/image_raw/compressed (mono RGB、YOLO 学習入力形式)
      - infra1/image_raw + infra1/image_raw/compressed (left IR)
      - infra2/image_raw + infra2/image_raw/compressed (right IR、stereo pair)
    depth は enable_depth=false で無効 (compute cost 節約、必要になったら enable)。
    """
    driver = LaunchConfiguration("wrist_camera_driver").perform(context).strip().lower()
    if driver not in {"uvc", "realsense"}:
        raise RuntimeError(
            f"unsupported wrist_camera_driver={driver!r}; expected 'uvc' or 'realsense'"
        )

    serial = LaunchConfiguration(f"wrist_{side}_serial").perform(context)
    enable_infra = (
        LaunchConfiguration("enable_wrist_infra").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if driver == "uvc":
        if enable_infra:
            raise RuntimeError(
                "enable_wrist_infra:=true requires wrist_camera_driver:=realsense"
            )
        configured_device = LaunchConfiguration(f"wrist_{side}_device").perform(
            context
        )
        resolved_device = os.path.realpath(configured_device)
        if not os.path.exists(resolved_device):
            raise RuntimeError(
                f"wrist {side} UVC device does not exist: {configured_device}"
            )
        return [
            LogInfo(
                msg=(
                    f"Wrist {side} camera: driver=uvc, "
                    f"device={configured_device} -> {resolved_device}, infra=disabled"
                )
            ),
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                namespace=f"wrist_{side}/camera/color",
                parameters=[{
                    "video_device": resolved_device,
                    "io_method": "mmap",
                    "pixel_format": "yuyv2rgb",
                    # 640x480 is the canonical inference resolution and remains
                    # available at 30 Hz when a D405 temporarily negotiates USB 2.
                    "image_width": 640,
                    "image_height": 480,
                    "framerate": 30.0,
                    "camera_name": f"wrist_{side}_camera",
                    "camera_frame_id": f"wrist_{side}_camera_optical_frame",
                    # Match the D405 neutral hardware profile used by the
                    # source demonstrations.  In particular, never inherit
                    # usb_cam's non-neutral brightness=50 process default.
                    "brightness": 0,
                    "auto_white_balance": True,
                    "autoexposure": True,
                    "image_raw.compressed.jpeg_quality": 90,
                }],
                output="screen",
            ),
        ]

    return [
        LogInfo(
            msg=(
                f"Wrist {side} camera: driver=realsense, serial={serial}, "
                f"infra={'enabled' if enable_infra else 'disabled'}"
            )
        ),
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            # realsense2_camera自身がcamera階層を持つため、namespaceはwrist側まで。
            # これで /wrist_{side}/camera/color/image_raw になる。
            namespace=f"wrist_{side}",
            parameters=[{
                # realsense2_camera 慣例: serial 前に "_" prefix 付ける
                "serial_no": f"_{serial}",
                "enable_color": True,
                # RGB-only inference must not spend USB/CPU bandwidth on the
                # unused IR pair.  Dataset collection can opt in through the
                # launch argument without changing this file.
                "enable_infra1": enable_infra,
                "enable_infra2": enable_infra,
                "enable_depth": False,          # depth compute しない (YAGNI)
                "depth_module.emitter_enabled": 0,  # IR projector OFF (CLAUDE.md 確定)
                # D405 の許容範囲は [0, 2]。driver default の 3 はこの機種では
                # reject されるため、無効 (0) を明示して起動 warning を避ける。
                "depth_module.power_line_frequency": 0,
                # D405 + ROS driverが実際に開くnative color profile。
                # 640幅を指定してもこの機体では848幅へfallbackする。
                "rgb_camera.color_profile": "848x480x30",
                "depth_module.depth_profile": "848x480x30",  # IR stereo 用 profile
                "camera_name": f"wrist_{side}_camera",
                "camera_frame_id": f"wrist_{side}_camera_optical_frame",
                # JPEG compression 条件は head camera と揃える
                "color.image_raw.compressed.jpeg_quality": 90,
                "infra1.image_raw.compressed.jpeg_quality": 90,
                "infra2.image_raw.compressed.jpeg_quality": 90,
            }],
            output="screen",
        ),
    ]


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare("g1_bringup"),
        "config",
        "system.yaml",
    ])

    mock_hardware = LaunchConfiguration("mock_hardware")
    enable_camera = LaunchConfiguration("enable_camera")
    enable_wrist_cameras = LaunchConfiguration("enable_wrist_cameras")
    common_output = "screen"

    camera_watchdog = Node(
        package="g1_bringup",
        executable="camera_stream_watchdog.py",
        name="camera_stream_watchdog",
        output=common_output,
        parameters=[{
            # Camera drivers can take several seconds to enumerate after a
            # host/service restart.  Once all streams have appeared, two
            # seconds without a compressed frame indicates a true outage.
            "startup_grace_s": 20.0,
            "stale_timeout_s": 2.0,
        }],
        condition=IfCondition(enable_wrist_cameras),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "mock_hardware",
            default_value="true",
            description=(
                "true: safe mock hardware bridge (実機不要、zero joint publish)。"
                " false: real Unitree bridge (unitree_sdk2py 経由で rt/lowstate 購読)。"
            ),
        ),
        DeclareLaunchArgument(
            "enable_camera",
            default_value="true",
            description=(
                "true: head USB camera を起動。false: camera 無しで hardware bridge のみ起動。"
            ),
        ),
        DeclareLaunchArgument(
            "head_camera_device",
            default_value=EnvironmentVariable(
                "HEAD_CAMERA_DEVICE",
                default_value=DEFAULT_HEAD_CAMERA_DEVICE,
            ),
            description=(
                "Head camera capture device. Prefer /dev/v4l/by-id; override "
                "for competition hardware with HEAD_CAMERA_DEVICE or this argument."
            ),
        ),
        # Issue #75 wrist camera (D405 x 2) launch args
        DeclareLaunchArgument(
            "enable_wrist_cameras",
            default_value="false",
            description=(
                "true: 両手 wrist RealSense D405 を起動 (RGB + IR stereo)。"
                " false (default): 起動しない (head camera 単体で pipeline verify したい時)。"
                " Issue #75 実機 verify Step 1 で true に切り替えて疎通確認。"
            ),
        ),
        DeclareLaunchArgument(
            "enable_wrist_infra",
            default_value="false",
            description=(
                "true: D405 infra1/infra2も起動する（RGB+IR dataset収集用）。"
                " false (default): RGBのみ。推論・smokeでUSB/CPU帯域を節約する。"
            ),
        ),
        DeclareLaunchArgument(
            "wrist_camera_driver",
            default_value="realsense",
            description=(
                "realsense (default): select each D405 by immutable RealSense "
                "serial. uvc: explicit RGB-only diagnostic fallback whose "
                "/dev/videoN endpoint is resolved immediately before launch."
            ),
        ),
        DeclareLaunchArgument(
            "wrist_left_device",
            default_value=EnvironmentVariable(
                "WRIST_LEFT_DEVICE", default_value=DEFAULT_WRIST_LEFT_DEVICE
            ),
            description="Left D405 UVC color endpoint (/dev/v4l/by-id/*-video-index4).",
        ),
        DeclareLaunchArgument(
            "wrist_right_device",
            default_value=EnvironmentVariable(
                "WRIST_RIGHT_DEVICE", default_value=DEFAULT_WRIST_RIGHT_DEVICE
            ),
            description="Right D405 UVC color endpoint (/dev/v4l/by-id/*-video-index4).",
        ),
        DeclareLaunchArgument(
            "wrist_left_serial",
            default_value=EnvironmentVariable(
                "WRIST_LEFT_SERIAL", default_value=DEFAULT_WRIST_LEFT_SERIAL
            ),
            description=(
                "Wrist left camera serial (D405)。実映像で左右対応付け確認後、"
                " 間違ってたらこの arg (or WRIST_LEFT_SERIAL env) で swap 可能。"
                " default は Issue #75 疎通機体 (2026-07-19) の bus 2-2.1 側 serial。"
            ),
        ),
        DeclareLaunchArgument(
            "wrist_right_serial",
            default_value=EnvironmentVariable(
                "WRIST_RIGHT_SERIAL", default_value=DEFAULT_WRIST_RIGHT_SERIAL
            ),
            description=(
                "Wrist right camera serial (D405)。default は Issue #75 疎通機体の"
                " bus 2-2.2 側 serial。左右間違ってたら wrist_left_serial と swap。"
            ),
        ),
        # mock bridge (`mock_hardware:=true` の default)。実機無しで /joint_states 等を zero publish。
        Node(
            package="g1_hw_bridge",
            executable="mock_hw_bridge_node",
            name="g1_mock_hw_bridge",
            output=common_output,
            parameters=[config],
            condition=IfCondition(mock_hardware),
        ),
        # real bridge (`mock_hardware:=false`)。G1 body 電源 ON + EtherCAT 接続後に使用。
        # motor 命令は発行しない (subscribe only) ので Damp state で安全に joint 位置を
        # Desktop 側に流せる。walk 動作テストは Issue #64 (Phase 2 #C) で。
        Node(
            package="g1_hw_bridge",
            executable="real_hw_bridge_node",
            name="g1_real_hw_bridge",
            output=common_output,
            parameters=[config],
            condition=UnlessCondition(mock_hardware),
        ),
        # 実 HBVCAM (USB2.0 Camera RGB) を usb_cam で grab (MJPG decode → raw Image)、
        # image_transport-plugins が /image_raw/compressed (CompressedImage) を自動生成する。
        # namespace=head/camera/color を付けて Desktop 側 Ros2FrameSource が subscribe する
        # /head/camera/color/image_raw/compressed に一致する topic 名にする。
        # v4l2_camera (humble binary 0.6.2) は MJPG 未対応で crash するため usb_cam に切替。
        OpaqueFunction(
            function=_launch_head_camera,
            condition=IfCondition(enable_camera),
        ),
        # Wrist camera 2 台 (D405)。通常は再起動で変わらないRealSense
        # serialで選択する。UVCは明示的な診断時だけ使う。
        # `enable_wrist_cameras:=true` でopt-in。IR projectorは常にOFF。
        OpaqueFunction(
            function=lambda ctx: _launch_wrist_camera(ctx, "left"),
            condition=IfCondition(enable_wrist_cameras),
        ),
        OpaqueFunction(
            function=lambda ctx: _launch_wrist_camera(ctx, "right"),
            condition=IfCondition(enable_wrist_cameras),
        ),
        camera_watchdog,
        RegisterEventHandler(
            OnProcessExit(
                target_action=camera_watchdog,
                on_exit=[
                    LogInfo(
                        msg=(
                            "Camera watchdog exited; shutting down the read-only "
                            "camera bringup so systemd can recover it."
                        )
                    ),
                    EmitEvent(
                        event=Shutdown(
                            reason="required camera stream watchdog exited"
                        )
                    ),
                ],
            ),
            condition=IfCondition(enable_wrist_cameras),
        ),
    ])
