# Curation Tool

IROS 2026 G1_WBT LeRobot dataset per-segment quality curation tool。

Schema v4 (9 cols) の per-segment label を UI 経由で収集し、LeRobot × labels の
join helper (loader) まで含む一式。annotation_tool と同じ self-contained pattern
(独自 pixi env、独自 workspace)。

## Layout

- `ui/` — FastAPI backend + vanilla JS 画面 (localhost:8765)
- `loader/` — LeRobot × curation labels join (analysis + training 用)
- `tests/` — pytest
- `workspace/` — 出力 data (`.gitignore`, HF 側で履歴管理)

## Setup / Usage

Phase B 実装完了時に埋める予定。予定 command:

```bash
cd data/curation_tool
pixi run serve         # UI 起動 (localhost:8765)
pixi run sync-hf       # HF 手動 sync CLI
pixi run test          # pytest
pixi run build-curated-task --task-id 2 --dry-run
```

## HF Repo

`Team-RAMEN/ramen-g1-ikea-assembly-curation` (private dataset、labels + registry の履歴管理用)

## Curated LeRobot dataset build

特定の task ID かつ `verdict != failure` の区間だけを元のBitRobot dataset
から切り出す場合:

```bash
pixi run build-curated-task \
  --task-id 2 \
  --video-keys \
    observation.images.cam_0 \
    observation.images.cam_1 \
    observation.images.cam_2 \
    observation.images.cam_3
```

既定では `optimal`, `success`, `ambiguous` を採用する。`ambiguous` も除く場合は
`--verdicts optimal success` を指定する。詳細は
`../bitrobot_lerobot_subtask_datasets/README.md` を参照。

生成後に検証済みdatasetを新しいprivate Hugging Face repoへアップロードする場合:

```bash
pixi run hf auth login
pixi run build-curated-task \
  --task-id 2 \
  --video-keys \
    observation.images.cam_0 \
    observation.images.cam_2 \
    observation.images.cam_3 \
  --upload-repo-id Team-RAMEN/IROS2026_RAMEN_HARA_task_2_non_failure
```

既存repoの更新には `--replace-remote` が必要。公開repoを新規作成する場合だけ
`--public` を指定する。詳しい認証、dry-run、安全動作、更新方法は
`../bitrobot_lerobot_subtask_datasets/CURATED_DATASET_UPLOAD.md` を参照。

## Schema v4

primary keyは `(episode_id, frame_start)`。`frame_end` はinclusiveで、各rowには
最終確定済みの `task_index`, `verdict`, `failure_category`, reviewer情報を保存する。

## References

- Sibling: `data/annotation_tool/` (同じ self-contained pattern)
