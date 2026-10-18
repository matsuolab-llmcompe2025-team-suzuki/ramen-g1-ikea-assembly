# inference/orin/ — Orin 側 ROS2 Humble deploy

topology β における Orin 側の deploy guide。Orin は I/O layer 専任 (camera publish / hand SDK / motor bridge)、上位 orchestrator は Desktop 側 (`inference/desktop/`) で走る。

## Status

- **topology β 確定**: Desktop (RTX 5090) = pixi runtime + cyclonedds direct、Orin = ROS2 I/O layer
- **Docker deploy 実装済**、apt install (host native) 選択肢は Phase 2 #A で Orin 実物見て評価
- 詳細な ROS2 system 設計 (mermaid diagram / topic contract / GR00T observation) は [`ROS2_SYSTEM_REFERENCE.md`](ROS2_SYSTEM_REFERENCE.md) 参照 (元 topology γ 前提だが Orin 側 package 詳細は流用可)

## Active packages

### Production (`ros2_ws/src/`)

| package | 責務 |
|---|---|
| `g1_bringup` | launch entry (`launch/system.launch.py`, `launch/dataset_replay_rviz.launch.py`) + `system.yaml` |
| `g1_hw_bridge` | Unitree G1 hardware I/O bridge (`mock_hw_bridge_node` = 実機無し zero publish / `real_hw_bridge_node` = unitree_sdk2py 経由で rt/lowstate 購読)。`mock_hardware:=false` で real に切替 |
| `g1_description` | Unitree G1 + Dex1 hand URDF / mesh / RViz config |
| `g1_msgs` | 共有 msg 定義 (`PolicyOutput.msg`) |

### Dev tools (`dev_tools/`、実運用 deploy に不要)

| package | 責務 |
|---|---|
| `g1_dataset_replay` | LeRobot parquet を `/joint_states` 等に replay → RViz で recorded episode 可視化 (mock G1、実機不要) |

### Deprecated (`_deprecated/`、topology β では未使用)

- `g1_task_runner` — planner + subtask executor、Desktop orchestrator と重複。去就は Phase 2 #B-2 で判断予定
- `g1_motion_adapter` — PolicyOutput → joint_target 変換 + safety filter。safety filter は [`desktop/lower_policy/actuators/safety.py:SafeActuator`](../desktop/lower_policy/actuators/safety.py) に純 Python として migration 済

## Deploy 手順 (Docker)

現状 Orin 側 deploy は Docker approach で実装済。apt install (host native) 選択肢は Phase 2 #A で Orin 実物見て trade-off を評価する。

### 起動 (Orin 実機で、G1 body は OFF でも OK)

```bash
cd inference/orin
docker compose build
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py
```

- `ros:humble-ros-base` (ARM64 対応) base で ROS2 Humble + cyclonedds + camera driver + `unitree_sdk2py` を container 内に入れる (`Dockerfile` 参照、SDK は pinned commit で `git clone` される)
- `g1_hw_bridge` は default で mock 起動。`mock_hardware:=false` で real bridge (`real_hw_bridge_node`) に切替 → G1 body の `rt/lowstate` を購読して `/joint_states` + `/g1/robot_state` に publish (motor 命令は一切発行しない = Damp state 安全性を bridge レベルで保証)
- **実 HBVCAM camera** (USB2.0 Camera RGB) を `usb_cam` が MJPG grab (`pixel_format: mjpeg2rgb`)、`image_transport-plugins` が `/head/camera/color/image_raw/compressed` (CompressedImage) を自動生成 (Desktop 側 `Ros2FrameSource` が subscribe する topic 名と一致)
- 詳細な起動 flag / 動作は [`ROS2_SYSTEM_REFERENCE.md`](ROS2_SYSTEM_REFERENCE.md#quick-start) 参照 (**注: topology γ 前提の記述、`auto_start_task` などは廃止済**)

#### ヘッドカメラの安定指定（実機・評価時に確認必須）

`/dev/videoN` は USB の接続順、接続ポート、同時に接続されるカメラによって変わるため、launch や Compose へ固定しない。固定番号を使うと、再起動後にhead用`usb_cam`がD405を同時openしてRealSense XU errorとUSB resetを誘発し得る。開発機のdefaultは、現在のhead capture nodeを指す次のby-id pathとする。

```text
/dev/v4l/by-id/usb-STYT_260320_K_USB_Camera_01.00.00-video-index0
```

`docker-compose.yml` はこの値を `HEAD_CAMERA_DEVICE` として container へ渡し、`system.launch.py` の `head_camera_device` launch argument が参照する。launch 時に by-id symlink を正規化してから `usb_cam` へ渡す。`/dev/v4l` は container に bind mount 済みで、数値 device の割当が変わっても by-id symlink が現在の capture node を解決する。

コンペ本番を含む別環境では製品名、VID/PID、シリアル番号が開発機と異なる可能性があるため、default path の存在だけで判断せず、起動前に必ず実デバイスを確認すること。

```bash
ls -l /dev/v4l/by-id
v4l2-ctl --list-devices
v4l2-ctl -d /dev/v4l/by-id/<capture-node> --list-formats-ext
```

ヘッドカメラの capture node（metadata node ではない方）であり、使用する `MJPG`・`1280x480`・`30 fps` を提供できることを確認する。本番機の by-id path が異なる場合は、リポジトリを編集せず起動時に環境変数で上書きする。

```bash
HEAD_CAMERA_DEVICE=/dev/v4l/by-id/<capture-node> \
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py mock_hardware:=false
```

一度だけ別deviceを試す場合は launch argumentでも上書きできる。

```bash
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py mock_hardware:=false \
    head_camera_device:=/dev/video0
```

device path の上書きだけなら image rebuild は不要。対応解像度やpixel formatも異なる機種へ交換した場合は、camera parameterも実機仕様に合わせる。

#### RGB color profile（学習データ整合）

`usb_cam` 0.6は`brightness`未指定時に50を設定する。head UVC cameraで50のまま
では黒い作業マットを背景にした白い部材が白飛びし、彩度も低く見える。
`system.launch.py`はhead nodeへ次を明示する。

- `brightness=0`（neutral UVC offset）
- `auto_white_balance=true`
- `autoexposure=true`
- JPEG quality 90。左右D405はRealSense driverのhardware auto exposure/white
  balanceを使用し、JPEG qualityだけ90へ揃える。

基準は4 RGB cameraを30 Hzで保持する
`Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_4@b402c17d2b5475b3a5438b95b61c7194fa23159a`
の実映像。現在の作業環境ではheadの白飛び率が約33%から4～9%へ下がり、RGB平均
輝度も学習データと同程度になった。固定exposure、固定white balance、Desktop側
の後処理は、照明変化への追従とpolicy inputの一貫性を損なうため追加しない。

launch後の実値は次で確認できる。

```bash
ros2 param get /head/camera/color/usb_cam brightness
ros2 param get /head/camera/color/usb_cam auto_white_balance
ros2 param get /head/camera/color/usb_cam autoexposure
```

### 実機 deploy (real bridge)

G1 body 電源 ON + EtherCAT 接続後に real bridge で lowstate を購読する。

```bash
cd inference/orin
docker compose build   # 初回のみ (unitree_sdk2py を container 内に clone)
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py mock_hardware:=false
```

ヘッドUSBカメラが未接続の状態でbridgeだけを検証する場合は、camera node を起動せずに次を使う。

```bash
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py mock_hardware:=false enable_camera:=false
```

- `mock_hardware:=false` で `g1_real_hw_bridge` が起動 (mock は起動しない)。
- G1 と Orin を繋ぐ NIC 名は `system.yaml` の `g1_real_hw_bridge.ros__parameters.interface` で指定する。
  - 現在の Orin 実機の default は `eth0`。本番機 / 別開発機では別名になるので、起動前に必ず `ip a` で NIC 名を確認して書き換えること。書き換え忘れ時は lowstate を受け取れず bridge が publish しないので、`ros2 topic hz /joint_states` で ~50 Hz を必ず確認する。
- **motor 命令は一切発行しない** ので、G1 が Damp state / walking FSM いずれでも安全に joint 位置を Desktop に流せる。
- 疎通確認: `ros2 topic hz /joint_states` で ~50 Hz を確かめる。

### Wrist camera 起動 (D405 × 2)

両手 wrist RealSense D405 を起動する。**カメラはdefault off**なので明示的に
`enable_wrist_cameras:=true`を渡す。推論・smokeでは再起動後も変わらない
RealSense serialで左右を選択する。D405の`/dev/videoN`を固定したUVC起動は、
再列挙後に別camera nodeを開く危険があるため診断専用であり既定ではない。

```bash
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py \
    mock_hardware:=false \
    enable_wrist_cameras:=true
```

Publish topics (namespace `/wrist_{side}/camera/` prefix):
- `color/image_raw` + `color/image_raw/compressed` (mono RGB、YOLO 学習入力形式)

IR stereoも必要なdataset収集時だけ、次のようにopt-inする。

```bash
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py \
    mock_hardware:=false \
    enable_wrist_cameras:=true \
    enable_wrist_infra:=true
```

この場合はさらに次をpublishする。

- `infra1/image_raw` + `infra1/image_raw/compressed` (left IR)
- `infra2/image_raw` + `infra2/image_raw/compressed` (right IR、stereo pair)

診断用RGB-only UVC経路では次のby-id deviceを使う。別機体では
`WRIST_LEFT_DEVICE` / `WRIST_RIGHT_DEVICE`で上書きする。

- left: `...128223071636-video-index4`
- right: `...130523070154-video-index4`

IR用RealSense経路の**serial → 左右 mapping**は次のとおり。
- `wrist_left_serial` = `128422271048` (bus 2-2.1)
- `wrist_right_serial` = `128422271925` (bus 2-2.2)

`/dev/v4l/by-id`に見える文字列はRealSense driverが要求する固有serialと
異なることがある。ここには`realsense2_camera`起動ログで確認したserialを設定する。

この実機ではROS driverがRGBを`848x480x30`で開く。`640x480x30`を指定しても
848幅へfallbackするため、launchは実際のprofileを明示する。Desktop側policyは
native 848幅を受け取り、model preprocessで一度だけ入力解像度へresizeする。

実映像で左右対応付け確認後、間違っていたら **環境変数でdeviceをswap可能**
(image rebuild不要):

```bash
# 例: RGB-only UVC映像で左右逆と判明した場合
WRIST_LEFT_DEVICE=/dev/v4l/by-id/...130523070154-video-index4 \
WRIST_RIGHT_DEVICE=/dev/v4l/by-id/...128223071636-video-index4 \
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py \
    mock_hardware:=false enable_wrist_cameras:=true
```

**CLAUDE.md Key Facts (確定事項)**:
- **IR projector OFF** (`depth_module.emitter_enabled=0`)。dataset G1_WBT の passive stereo と整合、active stereo は使わない
- Depth stream は `enable_depth=false` で未 enable (YAGNI、必要になったら enable)

**疎通確認** (Desktop 側):
```bash
pixi run -e runtime python -m evaluate.perception.subscribe_smoke_check \
    --topic /wrist_left/camera/color/image_raw/compressed --count 30
pixi run -e runtime python -m evaluate.perception.subscribe_smoke_check \
    --topic /wrist_right/camera/color/image_raw/compressed --count 30
```

IR用RealSense経路で両D405が同時に`UVCIOC_CTRL_QUERY: Protocol error`となる場合は、
共通USB root/hubのruntime autosuspendを疑う。手動診断ではOrin hostで
次を実行し、両D405と親hubを`power/control=on`へ固定する。この処理はカメラの
電源管理だけを変更し、G1へ関節指令を送らない。

```bash
sudo bash inference/orin/scripts/keep_realsense_usb_awake.sh
```

起動後はDesktopで3物理カメラ（ヘッドstereo + 左右D405）を同時に最低60秒
検査する。1系統でも0.5秒停止、または25 Hz未満なら実機評価を開始しない。

```bash
pixi run -e runtime python -m \
  evaluate.model_evaluation.tools.check_camera_streams \
  --interface <G1_DDS_INTERFACE> \
  --duration-seconds 60 \
  --minimum-hz 25
```

### 再起動後も固定番号問題を再発させない設定

Orinで一度だけ次を実行する。D405のautosuspend無効化をudevへ登録し、
`avp_teleimager.service`との二重openを禁止したcanonical bringupを有効化する。

```bash
cd /home/unitree/iros_2026_ramen
sudo inference/orin/scripts/install_real_camera_bringup.sh
sudo systemctl start iros-ramen-real-camera.service
```

サービスは起動のたびにheadをby-idから解決し、D405が2台ともUSB 3/config=1で
なければfail closedする。さらにread-only watchdogがhead・左右D405の軽量な
`camera_info`を監視する（JPEGを二重購読しない）。起動後にいずれかが2秒停止した
場合はROS launchを失敗終了させ、
systemdがカメラ専用containerを再起動する。復旧してもDesktop側の実機制御は自動
再開しない。Desktop側の4-camera preflightが成功するまで実機評価を開始しない。
AVP image serverを使う場合は、先にこのserviceを停止して明示的に切り替える。

## Mock G1 (実機不要、RViz で動作確認)

LeRobot parquet を replay して RViz で G1 の動きを再現 (`g1_dataset_replay` 経由):

```bash
cd inference/orin
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup dataset_replay_rviz.launch.py \
    parquet_file:=/path/to/episode.parquet
```

## Desktop 側との接続 (topology β)

- **Camera stream**: Orin 側 `g1_bringup` が camera driver 起動 → ROS2 topic publish → Desktop 側 orchestrator (`inference/desktop/perception/frame_source.py:Ros2FrameSource`) が cyclonedds direct で subscribe
- **SDK 命令 (walk / manipulate)**: Desktop 側 SDK direct call (`inference/desktop/lower_policy/actuators/g1_sdk.py`) → cyclonedds 経由で G1 SDK。**Orin 側の `task_runner` / `motion_adapter` chain は経由しない** (topology γ 時代の chain は `_deprecated/` へ)
- **Hand I/O**: Orin 側 hand SDK が担当
- Desktop 側 entry: `inference/desktop/entrypoint.py`、詳細は [`inference/README.md`](../README.md) の "実 G1 orchestrator run" section

## Reference

- [`ROS2_SYSTEM_REFERENCE.md`](ROS2_SYSTEM_REFERENCE.md) — 元 ROS2 system 設計 doc (topology γ 前提だが Orin 側 package 詳細 / topic 一覧 / dataset contract は流用可)
- [`../README.md`](../README.md) — `inference/` 全体 overview
