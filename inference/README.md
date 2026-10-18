# inference/ — Streaming state-machine Orchestrator

> **履歴資料:** 以下には初期の設計・未実装 TODO が残る。
> 会場の起動方法には使用しない。

YOLO-OBB による perception → skill 遷移判定 → lower policy への dispatch を 1 frame ずつ streaming で回すパイプライン。real-time (ROS2 実 G1) と replay (LeRobot ep mp4) を同じ code path で処理する。

## Layout

- **`desktop/`** — Desktop (RTX 5090) 側 pipeline (本 README の主対象)
- **`orin/`** — Orin (ROS2 Humble) 側 I/O 資産、Docker / apt deploy。詳細は [`orin/README.md`](orin/README.md)
- **`_deprecated/`** — topology β では未使用、削除は別 Issue で consensus 後 (`_old_vit_planner` / `g1_task_runner` / `g1_motion_adapter`)

**以下 path は特記無き限り `inference/desktop/` 配下**。

## 責務境界

```
frame (rgb, t)
   │
   ├── perception/ ─── YOLO 推論 + detection cleaning + FrameSource 抽象
   │                    (yolo_obb.py, cleaner.py, stream.py, frame_source.py)
   │
   ├── skill_planner/ ─ 履歴 flag / 遷移 rule (state.py + enter_conditions.py + geometry.py)
   │
   ├── lower_policy/ ── Skill lifecycle (start/step/stop) + Actuator abstract
   │                    (dispatcher.py, skills/, actuators/)
   │                    + 腕 kinematics (kinematics/) と把持ポーズ算出 (obb_grasp.py)
   │
   ├── orchestrator.py ─ tick pipeline 本体 (TRANSITIONS graph + fire dispatch)
   │
   └── entrypoint.py   ── 実行 entry point (実 G1 / sim 等、旧 `run/g1_orchestrator.py`)
```

### lower_policy/ の内訳

| path | 役割 |
|---|---|
| `dispatcher.py` | skill 名 → Skill instance の routing |
| `skills/` | 個別 skill。`pick_table_leg.py` は CV + Motion Planning + ルールベースの 8-stage サブFSM |
| `actuators/` | `g1_sdk.py` (walk) / `g1_arm_sdk.py` (腕) / `dex1.py` (グリッパ抽象 + 把持判定) / `dex1_dds.py` (Dex1 の DDS 指令・状態経路) |
| `kinematics/` | 片腕 7-DoF の FK / IK。`g1_arm.py` が URDF 由来の実装、`mock.py` が test 用 |
| `obb_grasp.py` | YOLO-OBB 検出 → root_link 基準の把持ポーズ。到達可能な手首角の選択を含む |
| `grasp_pose.py` | 把持ポーズ供給の Protocol と固定値実装 |
| `pose_utils.py` | 14-D 腕 pose の共通定義、`/joint_states` からの抽出 |

`pick_table_leg` は具象実装ではなく 3 つの Protocol (`ArmKinematics` / `Dex1Gripper` /
`GraspPoseProvider`) に依存する。default env では mock を挿して SDK 無しに全 stage を
通せる。Dex1 の I/O は skill が自前で保持するので `Orchestrator._build_obs` は無改変。

## Data flow (per tick)

```
frame.rgb → YoloObbPerception.predict(rgb) → raw dets
          ↓
        DetectionStream.push(raw) → Optional[cleaned]  (2 frame delay if median enabled)
          ↓ (cleaned が None なら skip)
        SkillState.update(cleaned)                     (履歴 flag latch)
          ↓
        for cand in TRANSITIONS[current]:
            if enter_check[cand](cleaned, state):
                state.transition(cand, ctx)             (atomic 副作用)
                dispatcher.start(cand, params)
                break                                    (1 tick 1 transition)
          ↓
        dispatcher.step(obs) → Optional[np.ndarray]     (action tensor or None)
          ↓
        actuator_send_fn(action)                        (Type B のみ)
        JSONL log
```

## Skill pattern (Type A / Type B)

Skill の per-tick behavior は 2 pattern 混在を許容 (`Skill.step` の default None がその設計):

| Type | 適した skill | 書き方 |
|---|---|---|
| **A: 1 発発行完結** | walk / stand 系 (SDK servo が持続保持) | `_on_start` で actuator に 1 発発行、`step` は default None、`_on_stop` で停止指令 |
| **B: per-tick 推論** | VLA / GR00T / ACT / diffusion | `_on_start` で model.reset、`step(obs)` で毎 tick 推論、action tensor を返す |

現状の実装例:
- Type A: `skills/move_to_table.py` (`MoveToTable`)、SDK `Move(continous_move=True)` で 10 日持続
- Type B: 未実装 (別 Epic で `Gr00tSkill` 等を追加予定)

詳細は `lower_policy/skills/base.py` の `Skill.step` docstring 参照。

## 現状の実行 mode

### 1. Offline smoke (LeRobot mp4 replay + MockSkill)

```
pixi run -e runtime python -m evaluate.orchestrator.smoke_run \
    --video data/vit_phase1/hf_cache/videos/observation.images.cam_0/chunk-000/file-000.mp4 \
    --out outputs/orchestrator/smoke.jsonl \
    --max-frames 500
```

- mp4 を cv2 で 1 frame ずつ read → `LerobotFrameSource` で pull → Orchestrator に流す
- 6 skill 全て `MockSkill` で埋める (hardware / VLA 依存無し)
- fire event の順序を JSONL log から目視で検証

### 2. 実 G1 orchestrator run (runtime env、`inference/desktop/entrypoint.py`)

```
pixi run -e runtime python -m inference.desktop.entrypoint \
    --interface eth0 \
    --topic /head/camera/color/image_raw/compressed \
    --vx 0.15 \
    --log outputs/orch_run.jsonl
```

- **前提** (operator 責任): G1 が walking FSM state に既に入っている、ROS2 camera driver が publish 中、ハーネス/E-stop/clearance 確保
- Ros2FrameSource (latest-only) で subscribe、Orchestrator tick を 30Hz で回す
- walk skill (`move_to_table`) のみ実 SDK、他 5 skill は MockSkill (log のみ) — VLA 実装は別 Epic
- Ctrl+Cで`actuator.set_velocity(0)`。Dampには入れずwalking balancerを維持

### 3. 単体 walk test (実 G1)

```
pixi run -e runtime python -m inference.desktop.lower_policy.scripts.run_g1_walk_to_table \
    --interface eth0 --duration 1 --vx 0.15
```

- Orchestrator経由せず、preflight後に`MoveToTable`を1秒動かすdry-run対応test tool

## 依存 env

| env | Python | 主な依存 | 用途 |
|---|---|---|---|
| **main (default)** | 3.12 | ultralytics, opencv, numpy | unit test、offline smoke、YOLO 学習/推論 |
| **runtime** | 3.10 | + `unitree_sdk2py`, `cyclonedds` | 実 G1 動作 (`inference/desktop/entrypoint.py`) |

- **rclpy は install しない方針**。ROS2 camera topic の subscribe は cyclonedds を直接使う (SDK が既に使ってる DDS layer に相乗り)。
- `Ros2FrameSource` / `G1SDKWalkActuator` は module top-level では cyclonedds / SDK を import せず lazy import → main env でも import 可能 (test collection / `--help` が通る)

## 実機投入で今後追加が必要な作業

### 実機 G1 の環境と Orchestrator の実機調整（当時の TODO）

- [x] cyclonedds 直接 subscribe 方式の実装 (2026-07-14 完了):
  - `inference/desktop/perception/sensor_msgs_idl/` に `CompressedImage_` の cyclonedds IDL 定義 (`Header_` / `Time_` は `unitree_sdk2py.idl` の既存を流用、typename は ROS2 canonical と一致 = wire 互換)
  - `Ros2FrameSource` を rclpy 前提から SDK ChannelFactory 経由の cyclonedds subscribe に書き換え、default QoS を SensorDataQoS 相当 (BestEffort / KeepLast(1) / Volatile) で明示
- [x] runtime env に pytest 追加 (IDL test 実行のため、2026-07-14 完了)
- [ ] `inference/desktop/entrypoint.py` を実 G1 で動作確認 (walk skill のみ、他 5 は Mock のまま) — **Phase 2 (別 Issue) で対応**
- [ ] 実測 tuning (camera topic 名 / QoS / Orchestrator tick 実速度 / `MoveToTable` の vx 妥当性 / fire event 挙動)
- [ ] このREADMEに実測結果を反映

**なぜ rclpy を諦めたか**: G1 = Humble + cyclonedds 0.10.2 固定 (Unitree 公式)、SDK が cyclonedds==0.10.2 hard pin → runtime env は Python 3.10 縛り、robostack-humble に py310 build 不在で詰み。SDK fork や Python 3.9 落としより、rclpy を使わず cyclonedds 直接方式 (SDK と同じ DDS layer で subscribe) が最小変更。TF2 / ROS2 service / rosbag2 が使えなくなるが、VLA skill 想定 (image + joint state → action の end-to-end) では TF2 不要と確認済

**Topology β 前提**: Desktop (RTX 5090) で orchestrator を動かし、Orin
(G1 onboard) から publish された ROS2 camera topic を Ros2FrameSource で subscribe
する構成に固定。`Ros2FrameSource` (cyclonedds direct) が primary source。

### Head camera の runtime view（2026-07-18 実機確認済み）

Orin の `/head/camera/color/image_raw/compressed` は head stereo packed RGB
`1280x480` を publishする。YOLO-OBBの学習入力はLeRobot `cam_0`、すなわちpacked
画像の**左眼 `640x480`**である。

`inference.desktop.entrypoint` は `HEAD_CAMERA_STEREO_VIEW = "left"` に固定し、
`Ros2FrameSource(..., stereo_view="left")` でdecode直後に左半分だけを取り出す。
右眼とpacked全幅は上位オーケストレータのYOLO入力には使用しない。汎用の
`Ros2FrameSource` は他用途との互換性のため `stereo_view="packed"` をdefaultに
保つが、実機entrypointからは必ずleftを明示する。

実機＋Mock actuator試験では入力shape `(480, 640, 3)`、YOLO cleanup後の
`move_to_table → move_table_base` 遷移、有限速度指令に続くゼロ速度停止を確認済み。

### さらに先 (別 Epic)

- [ ] VLA skill 実装 (Type B: `Gr00tSkill` / `ActSkill` / diffusion 等、5 skill 分の model)
- [x] runtime monitor: `desktop/lower_policy/actuators/safety.py:SafeActuator` に wrapper 実装、wrap 対象の manipulation actuator 実装で active 化
- [ ] Safety layer 残り: joint velocity check / E-stop hook / physical fail-safe

## Test

```
# lower_policy 単体 (SDK-lazy、default env で軽量)
pixi run test-lower-policy

# perception tests (cyclonedds subscribe / IDL テスト、runtime env)
pixi run -e runtime python -m pytest inference/desktop/perception/tests/test_frame_source.py inference/desktop/perception/tests/test_sensor_msgs_idl.py -v
```

- `inference/desktop/tests/`, `inference/desktop/*/tests/`: pure unit test (mock 依存、CI で自動走行、fast)
- 実機・実データでの確かめは `evaluate/model_evaluation/` の runner (`run_skill` など、手動実行)
- `inference/desktop/` 全体 (ultralytics/opencv 込み) を走らせるには model/yolo_obb
  sub-workspace env、または runtime env が必要

## 実機起動 前チェック (最小)

**前提**: 環境構築 (`third_party/unitree_sdk2_python` clone / `pixi install -e runtime` /
実 G1 setup) は事前完了、詳細は [root `README.md` の「Setup」](../README.md#setup) 参照。
物理準備 (ハーネス / E-stop / clearance / バッテリ / G1 walking FSM 状態) は operator
事前確保の前提。

上記が揃っている前提で、orchestrator 起動者側は起動直前に以下だけ確認:

- [ ] `--interface` が Desktop ↔ G1 の DDS 疎通する NIC 名になっている (`ip a` で確認)
- [ ] `--topic` が Orin 側の camera driver が publish する topic 名と一致
- [ ] 単体testで`--vx 0.15 --duration 1`を確認する。`0.10`以下はRPC成功でも
  この実機のgait開始閾値未満だったため、成功確認値として使わない
- [ ] `--log` 出力先が意図通り (YOLO の重みは `policy_config.yaml` の `yolo.ckpt_ref`)

### 起動コマンド

```
pixi run -e runtime python -m inference.desktop.entrypoint \
    --interface eth0 \
    --topic /head/camera/color/image_raw/compressed \
    --vx 0.15 \
    --log outputs/orch_run.jsonl
```

**Ctrl+C 停止動作の事前確認** (初回のみ、上記コマンドを 3-5 秒で `Ctrl+C` 押す):

```
[shutdown] set_velocity(0); keeping walking FSM
```

がstderrに出て、実機が停止後もRegular Modeで自立を維持すればOK。Dampへは
切り替えない。--interface /
その他 flag 詳細は `pixi run -e runtime python -m inference.desktop.entrypoint --help` 参照。

