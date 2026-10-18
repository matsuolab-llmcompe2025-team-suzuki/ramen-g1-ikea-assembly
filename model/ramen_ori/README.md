# model/ramen_ori — RAMEN-Ori VLA (LingBot + OBB + Skill-lang)

IROS 2026 competition の自作 VLA、`ramen_ori` (旧称: Custom C)。
LingBot-B backbone (frozen) + Frame delta + YOLO-OBB coord token + Flow Matching DiT
で `pick_table_leg` 等 subtask 別 BC training を行う。

設計 doc:
- [../../docs/model/ramen_ori_vla_design.md](../../docs/model/ramen_ori_vla_design.md) — 全体設計

## Phase 1 default 構成 (design doc §7)

| Axis | 選定 | 備考 |
|---|---|---|
| A-1 | LingBot-Vision ViT-B/16 (frozen) | `robbyant/lingbot-vision-vit-base` (86M) |
| B-7 | Flow Matching DiT (chunk=16, action_dim=19) | 150M |
| C-2 | OBB coord token (11-dim × 12 token) | `inference/desktop/perception/yolo_obb.py` 利用 |
| D-3 | Frame delta (Δ encoder ~20M) | small ViT |
| E-δ | State 71D (joint 19 + tracking 19 + vel 19 + hand 2 + EE 12) | Dataset B に torque 系無し前提 |
| F-3 | Skill embedding `nn.Embedding(6, 512)` | 3K params |
| G-1 | Standard self-attn Fusion (d=512, 6 layer) | baseline から start |
| H-4 + H-6 | Joint multi-task + BC only | |
| I-1 | aux 無し | I-6 depth 追加は Phase 2 |

## Layout (Phase 0 時点)

```
model/ramen_ori/
├── README.md    # 本ファイル
├── pixi.toml    # sub-workspace env (Phase 0 は minimal、Phase 1 で拡張)
├── __init__.py  # 空
└── tests/
    └── __init__.py
```

Phase 1 で追加予定 (design doc §7):
- `data.py` — Subtask dataloader (LeRobot v3 format)
- `vision.py` — LingBot-B wrapper
- `temporal.py` — Δ encoder (D-3)
- `obb.py` — OBB coord token projection (C-2)
- `state.py` — State augmentation MLP (E-δ 71D → 1 token)
- `skill.py` — Skill embedding (F-3)
- `fusion.py` — Fusion transformer (G-1)
- `action_expert.py` — Flow Matching DiT (B-7)
- `model.py` — 全 module 統合の RAMEN-Ori policy
- `train.py` — BC training loop
- `configs/base.yaml` — Hydra config
- `scripts/precompute_yolo_obb_detections.py` — YOLO offline detection (Task 0-3、subtask dataset curate 後)

## Env

`model/yolo_obb/pixi.toml` と同 pattern の sub-workspace。root workspace の
runtime feature (Unitree SDK) や yolo_obb (ultralytics) とは独立。

### Install と確かめ

```bash
cd model/ramen_ori
pixi install     # 学習の環境 (lerobot は fork の commit に固定)
pixi run test    # unit test (会場の腕の test は運営の package が無ければ skip)
pixi run train   # 既定 (configs/base.yaml) は合成データ (dummy) で 100 step の smoke。GPU で数十秒
```

### Phase 1 で追加予定の deps (実装時に `/install` 経由で pin)

- `transformers` — LingBot-B backbone load
- `hydra-core` — training config
- `wandb` — training log (entity `ken05-matuo-llm-88_llm_2025_suzuki` fixed)
- `lerobot` — fork (`matsuolab-llmcompe2025-team-suzuki/lerobot`) の ramen branch の `7431fb1` に固定 (`pixi.toml`、2026-10-09)
- `ultralytics` — offline YOLO-OBB detection precompute (subtask dataset curate 後)

## Skill vocab (F-3 embedding index)

`nn.Embedding(6, 512)`。Phase 1 は `pick_table_leg` 1 skill だけ使用、embedding index は
Phase 3 で全 skill 並列化する際に安定させる。詳細 index は Phase 1 で `data.py` 実装時に fix。

## HF weights (Phase 0 で access 確認済)

| Weight | 取得先 | 用途 |
|---|---|---|
| LingBot-B | `robbyant/lingbot-vision-vit-base` (86M、public、not gated、3 file: `README.md` + `model.pt` + `.gitattributes`) | A-1 vision backbone (frozen) |
| YOLO-OBB | local: `model/yolo_obb/runs/m_lowaug_v11b/weights/best.pt` (42MB) / HF: `Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb/runs/m_lowaug_v11b/weights/best_20260818.pt` (public) | C-2 detection token 生成 |

### YOLO-OBB weight 選定根拠

`m_lowaug_v11b` を Phase 1 default とする (2026-08-19):
- v10 real 956 + augment 3510 + head_left cycle1 101 + cycle2 623 = 5190 train (v10_hsv07 に head_left 新環境 724 追加)
- val mAP50-95 = **0.7986** (v10_hsv07 = 0.7942、+0.44pt)、mAP50 = **0.9746** (v10 = 0.9682、+0.64pt)
- 未見 head_left 101 frame QC で hand 検出率 v10 44% → v11b 100%、mean conf 0.72 → 0.90 (+0.18)
- bootstrap active learning 2-cycle: cycle1 (100 frame user 100% 修正 → v11a) → v11a で cycle2 pseudo (623 frame、user 195 手修正 + v11a pseudo 428 全面採用) → v11b
- 学習設定は v10_hsv07_imgsz800 と完全同一 (hsv_v=0.7、fliplr=0/erasing=0、他 aug 継承)、dataset のみ差替

### Fresh clone / 新 host での setup 手順

```bash
# 1. LingBot-B pull (public、token 不要)
cd model/ramen_ori
pixi run hf download robbyant/lingbot-vision-vit-base --local-dir ../lingbot_vision_vit_base

# 2. YOLO-OBB weight pull (public、token 不要)
pixi run hf download Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb \
    runs/m_lowaug_v11b/weights/best_20260818.pt --local-dir ../yolo_obb/
```

Subtask dataset は user 側で curate 中、確定後に Task 0-3 の offline detection を実施予定。

### 学習データ・出力の置き場所 (`/nvme`)

学習の設定 (`configs/data/unified_*.yaml`・`configs/retrain/*.yaml`) は、学習に使った server の
`/nvme` (`/nvme/unified_cache/<skill>/...`・`/nvme/init/...`・`/nvme/train_outputs/...`) を前提にしている。
自分の環境では、`/nvme` を自分の置き場所への symlink にするか、Hydra の上書きで path を渡す
(例: `pixi run train --config-name=retrain/venue_rotate_all6 training.init_from=<path> training.ckpt_dir=<path>`)。
学習データは公開した dataset (`Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-<skill>`、各 card の Usage) から、
同じ構成 (`<skill>/src`・`<skill>/v2/overlay` など) に戻す。overlay の cache を作り直すときは
`data/bitrobot_lerobot_subtask_datasets/scripts/run_unified_cache_pipeline.py --work-root <path>`。
