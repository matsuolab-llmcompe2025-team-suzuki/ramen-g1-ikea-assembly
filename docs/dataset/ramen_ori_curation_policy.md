# RAMEN-Ori curation policy (per-skill)

`data/curation_tool/workspace/labels.parquet` (schema v4、15,929 segment、by HARA) を
使って RAMEN-Ori 学習用 curated LeRobot v3 dataset を build する際の、skill 別
verdict 選択方針。

方針は user (HARA) が curation 実施時の意図を踏まえて 2026-08-09 に決定。

## Skill 別 policy

### skip 系

| task | skill | 判定 | 理由 |
|---:|---|---|---|
| 1 | move_to_table | **skip** (今回は使わない) | SDK walk で代替、RAMEN-Ori の学習対象外 |
| 6 | (未定義) | **skip** | `LEROBOT_TASK_INDEX_TO_SKILL_NAME` に定義無し、38 seg 中 36 failure で実質使えない |

### Overlap resolution policy (build 時強制、全 task)

`build_curated_task_dataset.py` は `select_segments` の直後、任意 task の build で
**同 episode 内で frame range が重なる seg のうち reviewed_at が古い方を drop** する。

**根拠**: curation UI で「修正時に旧 label を削除し忘れた」pattern が isolated に存在
(実例: ep 99 task 3 に `[5267,6446]` (20:30:24) と `[5361,6446]` (20:30:43) の 2 seg、
20 秒差で HARA が start を修正、直前の task 0 optimal `[5076,5360]` と隣接 5360→5361 に
接続する後者が正版、前者は削除忘れ)。この局所 curation ミスを build 側で自動救済し、
validate_segments が overlap で reject するのを避ける。

**動作**:
- 対象: build 対象 task 内 (`select_segments` 後の segments) の同 episode
- 判定: `a.frame_end >= b.frame_start` (curation schema = 閉区間 inclusive)
- Drop 基準: `reviewed_at` (ISO 8601 文字列比較で新旧判定)、新しい方を残す
- 3 個以上の連鎖 overlap は反復適用で非 overlap 集合に収束
- log: `[overlap resolved] ep X: kept [a,b] reviewed=..., dropped [c,d] reviewed=...` を stderr に出力 (curation 側改善のヒント)

**実装**: `build_curated_task_dataset.py::resolve_overlaps_by_reviewed_at`

**注記**: この logic は curation データの品質改善を代替するものではない。log で drop
発生を可視化し、HARA が curation UI 側で本来削除すべき旧 label を後日整理する運用と併用。

### Cross-task filter (build 時強制)

`build_curated_task_dataset.py` に **task 別の cross-task filter** を追加実装
(hard-code、CLI option ではなく該当 task 指定時に自動発動)。

**目的**: HARA curation labels の label mistake 可能性
(座標的にほぼ同じで **順序でのみ区別** される skill 間 : task 7 (move_table_base)
と task 5 (rotate_table_base) は pick 前後の順序で区別されるが label ミスあり得る)
を **build 時に構造的にカバー**する。

**適用 task と条件**: 各 task の Skill 別 policy 内 "Cross-task filter" 項で
1 つずつ確定 (task 別議論と同時に実装、CLI option 化しない)。

**未適用 task** (0 / 3 / 4 / 2): cross-task filter 発動せず、`--verdicts` 指定と
per-episode filter (merge stage) のみで抽出。

### 基本 build 系 (Phase 2 で使用)

#### task 7: move_table_base

- pick より **絶対前** で 一番序盤の skill (以降にあれば `rotate_table_base` として区別済)
- **Cross-task filter (build 時強制)**: 同 episode 内で **pick (task 4) の最小 frame_start より frame_start が小さい** task 7 seg のみ採用。pick 無し ep は制約発動せず全 seg 素通し
  - 実装: `build_curated_task_dataset.py::apply_move_table_base_cross_task_filter`
  - 目的: HARA curation の task 7 ⇄ task 5 label mistake 可能性 (座標的にほぼ同じ、順序でのみ区別) を構造的にカバー
- **基本**: `optimal` のみ (387 seg、cross-task filter 適用後の実 seg 数は smoke 実測で確認)
- optional (カリキュラム後半、精度出なければ使わない): `success` 追加

#### task 5: rotate_table_base

- pick **以降** にある table_base 系
- **Cross-task filter (build 時強制)**: 同 episode 内の判定基準を以下の順で決定
  1. pick (task 4) 有: **1 回目 pick.frame_start** を基準
  2. pick 無: **insert (task 0) or rotate (task 3) の 1 回目のうち早い方 (min)** を基準
  3. pick/insert/rotate 全部無: 判定基準無し、素通し
  - 採用: task 5 seg の `frame_start >= 基準`
  - 実装: `build_curated_task_dataset.py::apply_rotate_table_base_cross_task_filter`
  - 目的: task 7 ⇄ task 5 label mistake 可能性 (座標的にほぼ同じ、順序でのみ区別) を構造的にカバー
- **基本**: `optimal` のみ (1430 seg、cross-task filter 適用後の実 seg 数は smoke 実測で確認)
- optional (カリキュラム後半、精度出なければ使わない): `success` 追加

#### task 4: pick_table_leg

- **基本**: `optimal` のみ (786 seg)
- optional (汎化狙い): `success` = 2 回調整 / 微妙に完全ではないもの、追加検討

#### task 0: insert_table_leg

- **基本**: `optimal` のみ (1664 seg)
- optional (もし data 不足なら):
  - `success` を追加 (450 seg)
  - Episode 内で `success + failure + success` パターンで insert が **連続でラベル付けされている success を合体** させて 1 chunk にする (post-process 追加 script 必要)

#### task 3: rotate_leg_to_tighten

- **基本**: `success` (291 seg)
  - 理由: 壊れない程度の締めでかつ短い時間、これが実運用として望ましい挙動
- optional (実機運用時に締めすぎで壊れたら切替): `optimal` のみ (439 seg、少ないが可)

#### task 2: flip_table

Episode 内に flip_table seg が複数ある場合、**frame_start の順で意味が異なる**:

- **1 番最後 (frame_start 最大)**: **実際の flip_table 動作** (完成した机をひっくり返す) ← **これが学習させたい本命**
- **1 個手前 (frame_start 2 番目)**: 机を回す動き (rotate 系、実際の flip ではない)
- それ以前: 練習/やり直しの flip 試行

- **基本**: `success` から **Episode 内で frame_start 最大の 1 seg のみ保持** filter を適用 (= 1 番最後の実 flip 動作のみ)
- **Per-episode filter (build 時強制)**: `build_curated_task_dataset.py` に task 2 用 hard-code、945 seg → ~500 seg
  - 実装: `build_curated_task_dataset.py::apply_flip_table_per_episode_filter`
  - 目的: G 案 (6 個別 repo upload) では merge stage が無いため build 時に filter しないと individual repo 単独で「1 番最後 = 実 flip」の状態にならない
  - MIX 統合時 (merge script) は build 側で filter 済みのため `--task-filter 2:last_per_source_episode` の再指定は不要 (指定しても再度 filter 適用で最大 1 seg のみなので冪等)
- optional (机を回す動きも別 skill として学習したい場合): filter を "最後 2 seg" に緩める
  (merge script option 変更のみ、rotate 動きが 2 番目に混じる)

## Video features 保存方針

RAMEN-Ori curation は **frame 区間の絞りのみ** を行う。source dataset の
全 video feature (RGB×4 + IR×4 = **8 個**) を **必ず全部保存** する
(camera を落とすことは curation ではなく別処理)。

- **build 時**: `--video-keys` を **省略** して実行
  (`build_curated_task_dataset.py` の default 挙動で source の全 video feature
  を自動保存、[README L114-115](../../data/bitrobot_lerobot_subtask_datasets/README.md#L114))
- **学習側 (`model/ramen_ori/configs/*.yaml`) で使う feature 選択は独立**、
  build 段階では情報損失させない

### 実 source features (公式 `BitRobot/G1_WBT_Dex1_Building-Children-Table`、2026-08-09 fetch)

| feature | shape | fps | 種別 |
|---|---|---:|---|
| `observation.images.cam_0` ~ `cam_3` | [480,640,3] | 30 | RGB × 4 (head stereo L/R + 両手首) |
| `observation.images.cam_0_ir` ~ `cam_3_ir` | [480,640,3] | 30 | IR × 4 (D405) |

## Chunk 分割 (build 時 disk peak 対策)

WSL2 の vhdx が sparse 化されて Windows host 側 disk が圧迫される環境で、
1 task 分の build peak (30-266 GB 想定) が Windows 空きを食い尽くすため、
**各 task を N seg ずつの chunk に分割** して build → upload → local delete
のループで peak を抑える運用。

### 実装

- `build_curated_task_dataset.py::main` に `--seg-offset N` (default 0) を追加
- `cross-task filter` / `overlap resolution` 適用後の segments を `[seg-offset : seg-offset + max-segments]` で range 抽出
- Shell loop で chunk 毎に build → HF upload → local delete → 次 chunk

### Chunk size 目安 (Windows 26 GB 空き想定、peak 15 GB 以内)

| task | frame/seg (smoke ratio) | chunk_size | 予測 peak |
|---:|---:|---:|---:|
| 5 (rotate_table_base) | ~250 | 200 | ~13 GB |
| 4 (pick_table_leg) | ~200 | 200 | ~12 GB |
| 0 (insert_table_leg) | ~500 | 50 | ~13 GB |
| 3 (rotate_leg_to_tighten) | ~810 | 50 | ~13 GB |
| 2 (flip_table, filter 後) | ~200 | 200 | ~12 GB |

### 命名

各 chunk = 独立 HF repo に upload:

`Team-RAMEN/IROS2026_RAMEN_HARA_task_{task_id}_{verdict}_chunk{i}_of{total}_v1`

例: `HARA_task_5_optimal_chunk0_of7_v1`, `HARA_task_5_optimal_chunk1_of7_v1`, ...

### 後日 MIX merge

- 全 chunks (全 task ✕ chunk 数、~35-40 repo 想定) を local に download
- `merge_curated_lerobot_datasets.py` で統合 (task 内 chunk union + task 間 union)
- MIX 1 個 (`Team-RAMEN/IROS2026_RAMEN_HARA_mixed_v1`) に upload
- disk 十分な環境 (別マシン or Windows compact 済み環境) で実施

## HF repo 方針 (2026-08-11 方針変更: MIX 統合 → On-the-fly merge)

### 変更理由

`merge_curated_lerobot_datasets.py` で MIX 1 個作る当初計画は、WSL2 sparse .vhdx
の Windows host disk 圧迫 + task 0 の 34 chunks (合計 ~500 GB local peak) で実行
不能。代わりに **各 chunk を独立 HF repo に upload、learner 側で multi-dataset
loader を使って on-the-fly merge** する方針に変更。

### Chunk repos (計 56 個、全 upload 済 2026-08-11)

| task | skill | verdict | 実 seg (filter/preflight 後) | chunks | repo prefix |
|---:|---|---|---:|---:|---|
| 7 | move_table_base | optimal | 384 (cross-task filter -3) | 1 (単体) | `HARA_task_7_optimal_v1` |
| 5 | rotate_table_base | optimal | 1401 (cross-task filter -29) | 8 → **merged 2026-08-19** | `HARA_task_5_optimal_chunk{i}_of8_v1` (削除済、下記 note 参照) |
| 4 | pick_table_leg | optimal | 786 | 4 → **merged 2026-09-19** | `HARA_task_4_optimal_chunk{i}_of4_v1` (削除済、下記 note 参照) |
| 0 | insert_table_leg | optimal | 1664 (preflight drop 0) | 34 | `HARA_task_0_optimal_chunk{i}_of34_v1` |
| 3 | rotate_leg_to_tighten | success | 290 (overlap resolution -1) | 6 → **merged 2026-09-19** | `HARA_task_3_success_chunk{i}_of6_v1` (削除済、下記 note 参照) |
| 2 | flip_table | success | ~497 (per-episode filter -444, preflight -4) | 3 → **merged 2026-09-19** | `HARA_task_2_success_chunk{i}_of3_v1` (削除済、下記 note 参照) |

**namespace**: `Team-RAMEN/IROS2026_RAMEN_`
**合計**: **56 chunks / ~5,022 seg** (upload 時点、下記 note で task 5 は削減)

> **note (2026-08-19)**: task 5 の 8 chunks は毎起動 pull ceremony が
> 重い (~1h) ため、`merge_curated_lerobot_datasets.py` で単一 dataset に統合し
> `Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base` (private、8.7GB、
> 1401 episodes、146,873 frames) として upload。**元の 8 chunks は削除済** (2026-08-19)、
> reproducibility 検証 (metadata + LeRobot load smoke) 完了後に `HfApi.delete_repo`
> で執行。上記表の repo 名は歴史的 provenance として残す (build_curated 経路の記録)。
> Post-merge の全体は **48 chunks + 1 merged = 49 repos / ~5,022 seg (無変化)**。

> **note (2026-09-20)**: task 4 / 3 / 2 の chunks (計 13) も `merge_curated_lerobot_datasets.py
> --video-mode passthrough` (動画は作り直さず元の mp4 をそのまま使う) で skill ごとに統合し、
> `HARA_task_4_optimal_merged_v2` (786 episodes、232,564 frames)、`HARA_task_3_success_merged_v1` (290 episodes、
> 232,634 frames)、`HARA_task_2_success_merged_v1` (429 episodes、213,417 frames) として upload (private、統合 overlay
> cache 込み)。chunk と merge 後で episode・frame の数と抜き出した frame の画素・数値が一致し、upload 後の全 file の
> sha256 が HF と一致。chunk ごとの curation 記録 (segments) は merge 後の `meta/team_ramen_curation.json` の
> `sources[].build_provenance` に残る。**元の 13 chunks は削除済** (2026-09-20)。上記表の repo 名は provenance として残す。

### 学習側の Multi-dataset loader (Phase 2 実装対象)

Phase 2 で `model/ramen_ori/data.py` の `DummyRamenOriDataset` を実 dataset 対応に
差し替える際、以下の approach で 56 chunk repos を on-the-fly merge:

```python
from lerobot.common.datasets.multi_dataset import MultiLeRobotDataset
# or torch.utils.data.ConcatDataset

REPO_LISTS = {
    7: ["Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-move-table-base"],
    5: ["Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base"],  # 8 chunks merged
    4: ["Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-pick-table-leg"],  # 4 chunks merged
    0: [f"Team-RAMEN/IROS2026_RAMEN_HARA_task_0_optimal_chunk{i}_of34_v1" for i in range(34)],
    3: ["Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-leg-to-tighten"],  # 6 chunks merged
    2: ["Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-flip-table"],  # 3 chunks merged
}
all_repos = [r for lst in REPO_LISTS.values() for r in lst]
dataset = MultiLeRobotDataset(all_repos, ...)  # HF stream で on-demand video decode
```

- **task_index** は source dataset のまま保持 → learner 側の skill embedding
  (F-3、`nn.Embedding(6, 512)`) が task_index を key に skill 判別可能
- **Ep 番号衝突**: `MultiLeRobotDataset` が global ep index を自動再割当 (要検証、
  代替は `torch.utils.data.ConcatDataset` + 手動 offset)
- **Init cost**: 56 個 metadata load = 初回 数十秒-数分、以降 cache
- **Disk**: 全 chunks を local 保持不要、HF cache + on-demand fetch で minimal

### MIX 統合が必要になった場合 (後日の option)

もし後段で 1 個の MIX repo が必要になった場合:
- Sakura H100 サーバー (`/nvme` 7 TB) で全 56 chunks download → `merge_curated_lerobot_datasets.py` で統合 → `Team-RAMEN/IROS2026_RAMEN_HARA_mixed_v1` に upload
- 所要 ~2 h (GPU 1 枚)

## build 対象 一覧

| task | skill | verdict | seg 数 | merge stage filter | 予定 build 時間 |
|---:|---|---|---:|---|---:|
| 7 | move_table_base | optimal | 387 (filter 前) | build 側 cross-task filter (pick 前のみ、必須) | ~10 min |
| 5 | rotate_table_base | optimal | 1430 (filter 前) | build 側 cross-task filter (pick 以降 / fallback: insert or rotate 以降、必須) | ~25 min |
| 4 | pick_table_leg | optimal | 786 | (無し) | ~15 min |
| 0 | insert_table_leg | optimal | 1664 | (無し) | ~30 min |
| 3 | rotate_leg_to_tighten | success | 291 | (無し) | ~10 min |
| 2 | flip_table | success | 945 (filter 前) | build 側 per-episode filter (Episode 内 最後の 1 seg = 実 flip 動作、必須) | ~15 min |
| — | (mix merge + upload) | — | ~4700 seg 弱 (flip filter で 500 弱に減) | ~15 min |

**合計 walltime**: ~1.75-2.25 時間 (逐次)、GPU 不使用

> 注: 上記 walltime は video-keys 数 3-4 想定の旧見積もり。`--video-keys` **省略 (8 保存)** に方針変更後は、ffmpeg 処理時間が cam 数に概ね比例するため **1.5-2 倍程度に伸びる可能性**。task 7 smoke 実測後に見直す。

## 後続処理 (Phase 2 で必要になり次第)

- **task 0 insert 連続合体**: `success + failure + success` パターンで insert が連続の success 群を 1 chunk に merge、追加 post-process script、build 後の episode merge で対応
- **各 task の success 追加**: カリキュラム後半で精度出ない場合の切替用、build option だけ用意 (`--verdicts optimal success` に切替)
- **task 3 optimal に切替**: 実機運用時に締めすぎで壊れたら `--verdicts optimal` (439 seg) に切替

## References

- Source dataset: `BitRobot/G1_WBT_Dex1_Building-Children-Table` (公式 LeRobot v3、`build_curated_task_dataset.py` の default source)
  - 注: `Team-RAMEN/IROS2026_RAMEN_suzuki_{subtask}_1` は `build_subtask_datasets.py` (timestamp 切り単体 slice) の output、subtask_policy_training / flip_table_RL 用。RAMEN-Ori では非経由
- Curation labels: `Team-RAMEN/ramen-g1-ikea-assembly-curation` (private、HARA が UI で label 付け)
- Build script (single-task): `data/bitrobot_lerobot_subtask_datasets/scripts/build_curated_task_dataset.py` (single task_id 専用)
- CLI: `pixi run build-curated-task --task-id X --verdicts ... --upload-repo-id ...`
- Merge script (N → 1、本 doc 対応で新規、Step B 実装済): `data/bitrobot_lerobot_subtask_datasets/scripts/merge_curated_lerobot_datasets.py`
- CLI: `pixi run merge-curated --source-dir ... (repeatable) --task-filter TASK:FILTER --output-dir ... --force`
  - 実装機能: filter (last_per_source_episode) / data parquet reindex / video ffmpeg extract+repack / meta rebuild (tasks/episodes/info/stats/provenance)
  - Unit test 32 個 pass (fake source e2e + main CLI 動作 verify 済)
- Upload 手順: `data/bitrobot_lerobot_subtask_datasets/CURATED_DATASET_UPLOAD.md`
- Curation UI: `data/curation_tool/ui/`

## Step 実装ステータス

- ✅ Step A: policy doc (本 doc)
- ✅ Step B: merge script + tests 実装完了
  - B-2 skeleton / B-3 filter / B-4 data merge / B-5 meta rebuild / B-6 video / B-7 integration test
- ⏳ Step C (次 session): 6 task build → user 逐次確認 → MIX merge → HF upload
  - 見積 walltime: 1.75-2.25 時間 + upload 15 分
