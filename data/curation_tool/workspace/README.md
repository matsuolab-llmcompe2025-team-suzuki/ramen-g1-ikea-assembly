# Curation Workspace (Output)

curation tool が作る per-segment quality labels の出力領域。
中身は全て `.gitignore` (HF 側で履歴管理、`Team-RAMEN/ramen-g1-ikea-assembly-curation`)。

## 出力ファイル一覧

- `labels.parquet` — 主保存 (Schema v3)
- `labels.jsonl` — append-only backup (crash 対策)
- `failure_categories.json` — 動的 failure category registry (UI から追加可)
- `metadata.json` — reviewer state (last_ep, last_segment)
- `labels_backup/` — 週次 snapshot
- `videos_cache/` — segment tile mp4 cache (4-tile、all-keyframe)

## HF Repo

`Team-RAMEN/ramen-g1-ikea-assembly-curation` (private dataset)。
UI の HF sync button か `pixi run sync-hf` で push。
