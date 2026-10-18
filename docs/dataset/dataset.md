# Datasets

IROS 2026 Humanoid IKEA Assembly Challenge で利用しうる **BitRobot 組織公開の 4 dataset** を整理する。

---

## 1. 全体俯瞰

| # | repo | 形式 | episodes | frames | hours | シーン | 役割 |
|---|---|---|---:|---:|---:|---|---|
| A | [`BitRobot/2026-humanoid-ikea-assembly-challenge`](https://huggingface.co/datasets/BitRobot/2026-humanoid-ikea-assembly-challenge) | Raw MCAP | **533** | — | — | 家庭環境のみ (in-the-wild) | 競技公式 raw |
| B | [`BitRobot/G1_WBT_Dex1_Building-Children-Table`](https://huggingface.co/datasets/BitRobot/G1_WBT_Dex1_Building-Children-Table) | LeRobot v3.0 | **533** | **6.27M** | **58** | ラボのみ (黒マット + 白家具) | **競技 fine-tune ← 学習はこれ** |
| C | [`BitRobot/HIW-500`](https://huggingface.co/datasets/BitRobot/HIW-500) | Raw MCAP (タスク別 10 フォルダ) | ~23,074 mcap | — | — | 家事 10 タスク (組立含まず) | 家事系汎化 raw |
| D | [`BitRobot/HIW-500-LeRobot`](https://huggingface.co/datasets/BitRobot/HIW-500-LeRobot) | LeRobot v3.0 | **23,743** | **40.84M** | **378** | 家事 10 + 組立 1 = 11 タスク (混在) | **G1 用 VLA pretrain 候補** |

> **更新履歴**: 2026-06-25 に B の HF 側更新で `Home1/2/3` 分割が消失し flat 構造 (533 ep) に変更された。旧情報 (746 ep / 11.6M frames) は無効。

---

## 2. Dataset A: `2026-humanoid-ikea-assembly-challenge` (Raw MCAP)

競技公式の Raw bag。**全エピソードが家庭環境**で収録 (ピンク絨毯、人物、犬、家具背景。15 サンプル目視確認済)。

### 規模 / 構造

- **533 セッション**、1 episode ≈ 平均 419 秒 (≈7 分)、最短 7 秒 (中断 5%)
- 撮影日分布: 2026-05 = 399 件 (5/5〜5/14, 5/29)、2026-06 = 134 件 (6/17〜6/19)
- 表示は [Foxglove Studio](https://foxglove.dev/) / [Rerun](https://rerun.io/) 推奨。1 ファイル ~3〜5.5 GB

```
<root>/
└── episode_<YYYY-MM-DD_HH-MM-SS>/   ← 533 セッション
    └── episode_0001/
        ├── episode_0001.mcap        ← ~3〜5.5 GB
        ├── info.json                ← サブタスク時刻ラベル付き
        └── calibration/             ← カメラ等のキャリブ
```

### 取得ストリーム (1 episode = 17 channels)

| topic | schema | 周波数 |
|---|---|---:|
| `/stamped/lowstate` / `/stamped/lowcmd` | LowStateStamped / LowCmdStamped | **100 Hz** 全身関節状態・指令 |
| `/stamped/dex1/{left,right}/{state,cmd}` | MotorStateStamped / MotorCmdStamped | 100 Hz ハンド状態・指令 |
| `/stamped/secondary_imu` | IMUStateStamped | 100 Hz IMU |
| `/wbc_lerobot` | std_msgs/String | **50 Hz** WBC → LeRobot 用シリアライズ |
| `/camera/head/image/compressed` | CompressedImage | **30 Hz** 頭部 stereo packed RGB |
| `/camera/{left,right}_wrist/image/compressed` | CompressedImage | 30 Hz 手首 RGB ×2 |
| `/camera/{left,right}_wrist/ir{1,2}/compressed` | CompressedImage | 30 Hz 手首 IR ×4 |
| `/lf/odommodestate` | SportModeState | **20 Hz** locomotion state |
| `/annotation` | std_msgs/String | サブタスク境界 (info.json と整合) |

- カメラ計 **7 ストリーム** (1 head RGB stereo packed + 2 wrist RGB + 4 wrist IR)
- カスタム msg 型 `homies/msg/*` (Unitree homies SDK) のフィールドレベルデコードには公式 msg 定義が必要

### サブタスクラベル (`info.json`)

- **境界点のみ疎ラベル** (1 episode 内の 20 個の subtask 境界が映像と 20/20 一致を確認)
- 最大 229 秒の無ラベル区間あり → 訓練用スキルラベルには不適、サニティチェック用
- per-frame ラベルが欲しい場合は Dataset B 側の `task_index` を使う

### 役割と注意点

- **学習主体としては使わない** (LeRobot 変換コストが高い)。Foxglove での人間組立手順の観察、100 Hz 制御を直接学習したい場合の参照源
- 一部の古い episode (4 月収録分) は**現 Raw repo に存在しない**。HF 側で除外済

---

## 3. Dataset B: `G1_WBT_Dex1_Building-Children-Table` (LeRobot v3.0) — **競技 fine-tune 主軸**

**学習はこれを使う**。LeRobot v3.0 flat 構造、**全エピソードがラボ環境** (黒マット + 白家具、15 サンプル目視で家庭混在無しを確認)。

### 規模 / 構造

- **533 episodes / 6.27M frames / ~58 時間 @ 30 fps**
- `meta/info.json`: `total_episodes=533`, `splits={"train":"0:533"}`、`robot_type=unitree_g1`、`total_tasks=8`
- 1 episode = 1 セッション全体 (6〜8 個の subtask を含む list)。完成サンプル (8 task 全部含む) は 67% (358/533)

```
<root>/
├── data/chunk-000/*.parquet     ← state/action/index 等
├── videos/{cam_key}/chunk-???/  ← MP4 (h264) × 8 cam
└── meta/
    ├── info.json                ← schema, fps, splits=train 0:533
    ├── stats.json
    ├── tasks.parquet            ← 8 タスク辞書
    └── episodes/                ← per-episode メタ (動画区間 timestamp 完備)
```

### 観測スキーマ (`observation`)

| key | shape | 内容 |
|---|---|---|
| `images.cam_0` / `cam_1` | 480×640×3, h264 @ 30 fps | RGB 頭部 stereo (左眼 / 右眼 split) |
| `images.cam_2` / `cam_3` | 480×640×3, h264 @ 30 fps | RGB 両手首 |
| `images.cam_0_ir` … `cam_3_ir` | 480×640×3 | RealSense D405 IR ×4 (両手首 stereo) |
| `state.ee_state` | float32 [12] | 両手 6-DoF pose (= 6×2) |
| `state.hand_state` | float32 [2] | Dex1-1 ハンド 1-DoF ×2 |
| `state.robot_q_current` | float32 [36] | 全身関節角 (whole-body) |

→ state 計 **50 次元**

### 行動スキーマ (`action`)

| key | shape | 内容 |
|---|---|---|
| `action.ee_action` | float32 [12] | 両手 EE 指令 (IK 用) |
| `action.hand_cmd` | float32 [2] | ハンド開閉指令 |
| `action.robot_q_desired` | float32 [36] | 全身関節目標角 (WBC 用) |

→ action 計 **50 次元、EE-space + whole-body joint の dual labeling**。EE-only な VLA でも、whole-body joint policy でもどちらも学習可能。

### サブタスクラベル (`task_index`、密、学習用主軸)

| task_index | task | 出現率 (episodes) |
|---:|---|---:|
| 0 | insert table leg to table base | 99.6% |
| 1 | move to table | 94.7% |
| 2 | flip table | 99.6% |
| 3 | rotate leg to tighten | 99.6% |
| 4 | pick table leg | 99.6% |
| 5 | rotate table base | 99.6% |
| 6 | building children table | 100% (intro 用包括ラベル、frame=0 にしか出ない) |
| 7 | move table base | **69.0%** (一部 episode で欠落) |

**8 classes** (うち 6 = intro 用包括なので動作系は実質 **7**)。

### 時間配分 (episode 0 = 816 秒の例)

| label | 割合 |
|---|---:|
| insert table leg to table base | **61.1%** |
| rotate leg to tighten | **25.5%** |
| pick table leg | 6.4% |
| flip table | 3.5% |
| rotate table base | 2.3% |
| move to table | 1.1% |
| building children table | 0.1% (intro のみ) |

→ **insertion + tightening が時間の 86%**。SOTA π0.5 が DexJoCo で 52.5% で stall することを踏まえ、ここに投資を集中。

### 役割

- **競技 fine-tune の主軸**
- ラボ環境のみのため、家庭シーン汎化は Dataset A 由来 or Dataset D / task10 を補助で投入が必要
- per-frame `task_index` 完備なので、最初から階層化 (hi-level skill planner + low-level policy) を前提に設計可

### 推奨 DL パターン

1. `meta/info.json` + `meta/tasks.parquet` + `meta/episodes/chunk-000/file-000.parquet` だけ DL (数 MB)
2. `data/chunk-000/file-000.parquet` + 興味ある cam の `videos/.../chunk-000/file-000.mp4` (各 ~200MB) で先頭数 episode を Rerun で表示
3. Raw MCAP (Dataset A) は 1 エピソードだけ DL して Foxglove 確認 (**1 ファイル ~5 GB 注意**)

---

## 4. Dataset C: `HIW-500` (Raw MCAP, 家事 10 タスク汎化用)

**HIW = Humanoids In-the-Wild**。Unitree G1 / bimanual / teleop の汎用家事データセット (raw)。

- ~23,074 mcap、タスク別 10 フォルダ構造
- `building children table` (IKEA) は**含まれない** — 純粋に家事 10 タスク
- Dataset D (LeRobot 化) の元データの一部 (8/10 タスクは episode 数完全一致、2 タスクで計 91 episode が変換時除外)

### 役割

- 直接学習には Dataset D の LeRobot 版を使えば良いので raw を触る優先度は低い
- 100 Hz 制御や HIW 独自トピックを直接読みたい場合のみ参照

---

## 5. Dataset D: `HIW-500-LeRobot` (LeRobot v3.0, **G1 用 VLA pretrain 候補**)

家事 10 + 組立 1 = 11 タスク、**378 時間**の大規模 LeRobot dataset。

### 規模 / 内訳

- **23,743 episodes / 40.84M frames / 378.1 時間 @ 30 fps**
- license: CC-BY-4.0、pretty_name: "HIW500: Humanoids In-the-Wild Dataset"

| task_index | task | episodes | hours |
|---:|---|---:|---:|
| 0 | clean up the room | 1,537 | 46.1 |
| 1 | clothes washing | 2,423 | 14.7 |
| 2 | hang hanger | 1,492 | 50.5 |
| 3 | hang keys on a hook | 2,381 | 26.5 |
| 4 | kitchen organization | 1,832 | 36.1 |
| 5 | move the pillow to the sofa from floor | 2,039 | 18.4 |
| 6 | picking trash to rubbish bin | 1,780 | 16.0 |
| 7 | restocking fridge | 1,796 | 33.2 |
| 8 | setting the table | 5,526 | 34.0 |
| 9 | sweep floor | 2,177 | 18.6 |
| 10 | **building children table** | **760** | **84.1** |

### 観測 / 行動スキーマ (Dataset B と**非互換**)

| key | shape | 備考 |
|---|---|---|
| `observation.images.head` | 480×**1280**×3 | **head packed stereo (split されていない)** |
| `observation.images.left_wrist` / `right_wrist` | 480×640×3 | RGB mono、**IR チャンネル無し** |
| `observation.state` | float32 [29] | — |
| `observation.state.wbc` | float32 [23] | WBC 全身 |
| `action` | float32 [23] | WBC action |
| `language_persistent`, `language_events` | language [1] | 言語ラベル付き |

→ **Dataset B の 50 次元 (EE[12]+hand[2]+robot_q[36]) と非互換**。WBC 23 次元のみ、IR 無し、head は packed。

### task 10 (building children table) の中身は混在

- ep22983 (先頭) は家庭環境 (ピンク絨毯) で Dataset A と同一ソース
- ep23236 以降はラボ環境で Dataset B と同一シーン
- → HIW/task10 = 「Raw 家庭撮影分の再パッケージ」+「G1_WBT ラボ分の観測縮小再パッケージ」の合成と推定
- 数字: Dataset A 533 + B 533 = 1066 のうち 760 (71%) が含まれる

### 役割と注意

- **task 0〜9 (家事 10)** は IKEA と無関係 → **純粋な汎化 / 事前学習データ**として使える
- **task 10** は IKEA と同タスクだが**観測スキーマが Dataset B と非互換** (IR 無し、state 削減) なので、競技 fine-tune には Dataset B を主軸にすべき
- 想定利用方針: **visual encoder + LLM head の pretrain** に Dataset D、**行動 head は Dataset B で fine-tune**

---

## 6. 重要な前提 (locked-in facts)

学習・推論設計時に踏まえるべき検証済みの前提:

- **D405 IR projector は OFF** (10 タイムスタンプ + 視覚で確認) → active stereo 不可、**passive 前提で設計**
- **力センサ topic 無し** → wrist depth を接触検出 / 滑り検出 / insertion 完了判定の proxy として使う
- 頭部カメラは stereo RGB (baseline 63 mm)、手首は mono RGB + stereo IR (baseline 18 mm)
- info.json subtask 境界は 20/20 で映像一致、ただし**疎** (229 秒の無ラベル区間) → 訓練ラベルは LeRobot 側 per-frame `task_index` を使う
- L2 (SGBM+WLS) depth pipeline: wrist IR valid **80%** / head RGB valid **61%**。L4 (Deep stereo) は当面不要
- Insertion が総タスク時間の **61%**、tightening 25% (episode 0 sample 基準、全 533 ep での分布計測は未完)。SOTA π0.5 が DexJoCo で 52.5% で stall → ここに peak investment
- Dataset B (ラボ) と Dataset A (家庭) はセッションが完全分離。**本番シーン (Pittsburgh カーペット) はいずれとも異なる**前提で、未知シーン汎化を意識

---

## 7. データ取り扱い tips

学習を回す際に直接効く前処理・サンプリングの注意点:

### エピソードフィルタ

- **`duration_sec < 30` or `len(subtasks) == 0` の episode は除外**
  - Dataset A の `info.json` 検証で 20 件中 1 件 (5%) が該当 (`episode_2026-05-08_10-45-40`、duration 7.2s)
  - アノテーターが「内容なし」とみなして subtasks=[] にした**中断エピソード**と判明
  - Dataset B 側は per-frame 付与なので、収録段階で既に除外されている可能性が高い

### クラス不均衡対策

- Dataset B の `task_index` 分布は **insert 61% / tighten 25% / 残り 14%** と極端 (※ 数字は episode 0 sample 基準。全 533 ep での分布計測は未完なので、本番の class balancing 係数を決める前に計測スクリプトで確認推奨)
- 訓練時に **class balancing** か **segment 単位 sampling** が必要 (素朴な frame uniform sampling だと残りラベルが学習されない)
- insertion + tightening への投資集中とは別軸で、minority class の学習機会を確保する設計

### 失敗-リトライラベルの活用

- Dataset A の `info.json` で**同一サブタスクラベルが連続する区間** (例: insert → insert) は**失敗してやり直した**可能性が高い
- → **失敗検出ラベルとして利用可能**。リカバリ policy (失敗検出 + 部分再試行) の supervision に流せる
- 競技ルール上「介入は減点だが run 継続」なのでリカバリ価値は高い

