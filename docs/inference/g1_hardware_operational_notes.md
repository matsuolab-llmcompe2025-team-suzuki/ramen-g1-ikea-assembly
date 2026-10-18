# G1 Hardware Operational Notes

G1 実機統合で **最初にハマる罠 / 実測で判明した hardware 特性** をまとめたリファレンス。新規に G1 向け Skill / Perception / Actuator を実装する engineer 向け。

対象: Unitree G1 (29 dof body) + Dex1-1 hand + RealSense D405 wrist cameras
検証環境: 2026-07-19 の manipulation action loop の実機 verify セッション

---

## 1. Arm 制御は `rt/arm_sdk` 必須 (`rt/lowcmd` は walk と非併存)

### 罠

G1 の arm 制御 topic は 2 系統あり、**Regular Mode で walk と併存可能なのは `rt/arm_sdk` のみ**。`rt/lowcmd` は低レベル完全制御用で `MotionSwitcher.ReleaseMode` で walking balancer を停止させる前提 → walk と併存不可。

初回実装で `rt/lowcmd` に publish しても firmware が受け付けず**腕が動かない**症状で気付く (初回の実装の失敗で発覚)。

### 選び方

| 用途 | topic | walking balancer | 備考 |
|---|---|---|---|
| VLA policy (walk + arm 併用) | **`rt/arm_sdk`** | 維持 (Regular Mode 継続) | body 29 joints を hold + arm 14 joints を override |
| 全身低レベル制御 (walk なし) | `rt/lowcmd` | 停止 (`ReleaseMode` 必須) | 落下リスクあり、単体 arm task 用 |

### `rt/arm_sdk` publish の handshake

`rt/arm_sdk` publish 時は **body 全 29 joints を lock target で埋める** + arm 14 joints (slot 15..28) を desired 値で override + **motion-mode enable slot 29** に `.q = 1.0` を書き込む (§2 参照)。実装は [`inference/desktop/lower_policy/actuators/g1_arm_sdk.py`](../../inference/desktop/lower_policy/actuators/g1_arm_sdk.py) を参照。

---

## 2. Motion-mode enable slot 29 (magic value の由来)

`LowCmd_.motor_cmd` は array 長 35 (SDK padding)。実 body joint は 0..28 の 29 dof、**slot 29 は特別 slot `G1JointIndex.kNotUsedJoint`** で motion-mode enable/disable フラグ用。

```python
lowcmd.motor_cmd[29].q = 1.0  # 1: enable arm_sdk, 0: disable
```

### 出典 (protocol 変更時はここを再確認)

- [`third_party/unitree_sdk2_python/example/g1/high_level/g1_arm5_sdk_dds_example.py`](https://github.com/unitreerobotics/unitree_sdk2_python) L64, L133
  - L64: `class G1JointIndex: kNotUsedJoint = 29  # NOTE: Weight`
  - L133: `self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1  # 1:Enable arm_sdk, 0:Disable arm_sdk`
- H2 系 (兄弟モデル) は `h2_loco_client.SetArmSdkStatus(bool)` service を持つが、**G1 はこの service を持たない**。slot 29 magic write が唯一の公式 enable/disable 経路。

コード内 comment にも同 reference を記載済 ([`g1_arm_sdk.py:275-282`](../../inference/desktop/lower_policy/actuators/g1_arm_sdk.py))。

---

## 3. `mode_machine` は `lowstate` から dynamic 取得 (hardcode NG)

`LowCmd_.mode_machine` は body 状態を示す firmware side field。**個体・firmware version により値が異なる**ため hardcode 禁止。

### 実装 pattern

```python
# 起動時に lowstate を 1 サンプル受信 → mode_machine を capture
first_state = self._wait_for_lowstate()
self._mode_machine = first_state.mode_machine

# publish 時に毎回 forward
lowcmd.mode_machine = self._mode_machine
```

`ArmCmd_` を使う path でも同様。実装は [`g1_arm_sdk.py`](../../inference/desktop/lower_policy/actuators/g1_arm_sdk.py) の `_wait_for_lowstate` + publish loop を参照。

### 罠

hardcode (例: `mode_machine = 5`) すると firmware が cmd を silently reject → 「publish は通っているが腕が動かない」症状。lowstate 受信前に publish 開始しない (subscriber init + first sample wait を entrypoint で確実にやる)。

---

## 4. RealSense D405 wrist camera: USB 3.0 帯域上限

### 実測

Orin に D405 wrist 2 台を USB 3.0 接続 (`realsense2_camera` Node 2 個で並行 stream)、公称 30fps 設定に対し **実測 21〜24Hz に低下**。USB 3.0 の実効帯域が 848×480×2 cam の同時 stream に対して上限に達している。

### profile

- Resolution: `848x480x30` (D405 の default profile、`640x480x30` は driver が silently fall back するので明示指定推奨)
- Color: RGB (JPEG quality 90)
- IR projector: **OFF** (`emitter_enabled = 0`) — 競技環境で active stereo は使わない前提

実装は [`inference/orin/ros2_ws/src/g1_bringup/launch/system.launch.py`](../../inference/orin/ros2_ws/src/g1_bringup/launch/system.launch.py) の `_launch_wrist_camera` を参照。

### 影響

- perception pipeline は **21-24Hz 前提**で設計 (VLA inference / logging / rate limiter)
- 追加で他 USB 3.0 device (深度カメラ増設等) を足す時は帯域再測定必須
- USB 3.0 hub 経由での増設は避け、直挿し port を分ける

### serial の管理

D405 内部 serial number は `/dev/v4l/by-id/` 表示の sequence number **とは別**。`rs-enumerate-devices` で実 serial を取得して launch 側の env var (`WRIST_LEFT_D405_SERIAL` / `WRIST_RIGHT_D405_SERIAL`) に明示する。

---

## 5. 追加を待つ知見 (今後の実機セッション後に加筆)

- Dex1-1 hand SDK の polling issue workaround (現状 `check_dex1_state.py:44` で daemon thread + queue 回避、SDK 側 fix 待ち)
- lowstate `Read()` の SDK semantics 詳細 (現状は non-blocking cached read 前提で 250Hz publish loop から呼んでいる、`g1_arm_sdk.py:161-172` docstring 参照)
- Sim2Real gap (安全 pose envelope、joint velocity 上限の実測)
