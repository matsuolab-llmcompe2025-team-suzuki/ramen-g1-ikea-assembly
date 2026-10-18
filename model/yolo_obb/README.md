# model/yolo_obb — YOLO-OBB Training + Predict

v1、v2/v3 (aspect fix)、v4 (background aug ablation)、v5-v10 (augment + annotation iter)、v11 (head_left mp4 で新環境 domain gap 埋め、bootstrap active learning 2-cycle)。

`data/yolo_obb/dataset` (100 枚 v1 → 400 枚 v2/v3 → 966 枚 v10 → v10 + head_left 724 枚 v11) を使って
ultralytics YOLO11-obb を fine-tune、best.pt を出力。未見 ep で predict + 目視評価。

## 現行採用: **m_lowaug_v11b** (v11 cycle2、2026-08-19)

- weight: `runs/m_lowaug_v11b/weights/best.pt` (42MB、best epoch 25、patience 20 で 45ep early stop)
- val mAP50-95 = **0.7986** / mAP50 = **0.9746** (v10_hsv07 比 mAP50-95 +0.44pt、mAP50 +0.64pt)
- HF backup: `Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb/runs/m_lowaug_v11b/weights/best_20260818.pt`
- 学習: v10 real 956 + augment 3510 + head_left cycle1 101 + cycle2 623 = **5190 train**
- config: yolo11m + batch 4 + imgsz 800 + mosaic=0.5 + hsv_v=0.7 + fliplr=0.0 + erasing=0.0 + degrees=10 + translate=0.2 + scale=0.6 + perspective=0.001 (v10_hsv07 と同一設定、dataset のみ差替)

判定根拠 (未見 head_left 101 frame での目視 QC):
- **hand 検出率 v10 44% → v11b 100%** (両端の gripper 定着)
- **mean conf v10 0.72 → v11b 0.90** (+0.18)
- **workspace / table_top / hand が exact 1/frame** (v11a 過検出を cycle2 で修正)
- lab val でも v10 baseline 更新 (mAP50-95 +0.44pt) → 新環境追加が既存性能を壊さない

**履歴 (旧採用)**:
- **v10_hsv07 (2026-08-04)**: `runs/m_lowaug_v10_hsv07_imgsz800/weights/best.pt`、val mAP50-95=0.7942 / mAP50=0.9682。v10 real 956 + augment 3510 = 4466 train。head_left mp4 の新環境で hand 44% / conf 0.72 と domain gap あり → v11 で解決。詳細は本 README 下部 [v5-v10 セクション](#v5-v10) 参照。
- **v3**: fixed val mAP@0.5=0.9547、`runs/m_lowaug_v3/weights/best.pt`。詳細は [v3 セクション](#v2--v3-cycle--semi-supervised-active-learning--aspect-fix) 参照。
- **v4**: 家庭 domain gap 対策の背景 copy-paste augmentation ablation、実機評価待ちのまま v10 に統合されて役目終了。詳細は [v4 セクション](#v4--background-copy-paste-augmentation-ablation) 参照。

## Layout

```
model/yolo_obb/
├── scripts/
│   ├── train.py     # ultralytics wrapper、--model {n,s,m,l,x} / config yaml
│   └── predict.py   # 学習済 best.pt で未見 ep frame に inference + 描画
├── configs/         # 12 config (3 model size × 4 variant)
│   ├── {s,m,l}_baseline.yaml
│   ├── {s,m,l}_freeze10.yaml    # backbone freeze
│   ├── {s,m,l}_imgsz800.yaml    # 高解像
│   └── {s,m,l}_lowaug.yaml      # mosaic 抑制
└── runs/                        # .gitignore、train 出力 (weights + plots + summary.json)
    └── <run_name>/
```

## Usage

学習系タスクは `model/yolo_obb/pixi.toml` sub-workspace で管理 (root workspace の
Unitree SDK path dep とは独立、`third_party/unitree_sdk2_python` 不要)。初回のみ
`cd model/yolo_obb && pixi install` で env 準備。

```bash
cd model/yolo_obb

# train (config yaml 指定)
pixi run train-yolo --config model/yolo_obb/configs/m_lowaug.yaml

# predict (best.pt + 未見 ep から N 枚 sample、box overlay 描画)
pixi run predict-yolo --run-name m_lowaug --frames-ep 64 --n 30

# dataset build (HF pull → skill stratified split → ultralytics layout)
pixi run build-yolo-dataset --out data/yolo_obb/dataset --val-ratio 0.10 --seed 42
```

## 全 12 run 比較 (3 model × 4 variant)

100 枚 ep 60 dataset で 12 config を fine-tune。**35 分で完走** (RTX 3060 Ti 8GB)。

### val mAP@0.5 ランキング

| Rank | Run | mAP@0.5 | mAP@0.5-0.95 |
|---|---|---:|---:|
| **1** | l_freeze10 | **0.9906** | 0.7857 |
| 2 | l_lowaug | 0.9760 | 0.7854 |
| **3** | **m_lowaug** | **0.9732** | 0.7827 |
| 4 | m_baseline | 0.9691 | 0.7549 |
| 5 | s_baseline | 0.9663 | 0.7506 |
| 6 | s_imgsz800 | 0.9647 | 0.7637 |
| 7 | l_baseline | 0.9615 | 0.7411 |
| 8 | m_freeze10 | 0.9581 | 0.7423 |
| 9 | s_lowaug | 0.9557 | 0.7273 |
| 10 | l_imgsz800 | 0.9454 | 0.7132 |
| 11 | m_imgsz800 | 0.9427 | 0.7127 |
| 12 | s_freeze10 | 0.9289 | 0.6521 |

### Variant 別マトリクス (val mAP@0.5)

|         | baseline | freeze10 | imgsz800 | lowaug |
|---------|---------:|---------:|---------:|-------:|
| **s**   | 0.9663 | 0.9289 ↓ | 0.9647 ~ | 0.9557 ↓ |
| **m**   | 0.9691 | 0.9581 ↓ | 0.9427 ↓ | **0.9732 ↑** |
| **l**   | 0.9615 | **0.9906 ↑** | 0.9454 ↓ | 0.9760 ↑ |

### 学び (12 run 総合)

- **model size で単調じゃない**: m/l で lowaug が効くが s では逆に低下
- **freeze は l で最強、s で最弱**: capacity 大 (l) は backbone freeze で overfit 抑制、
  capacity 小 (s) は head-only では表現力不足
- **lowaug (mosaic=0.5) は m/l で有効、s で害**: 100 枚 dataset では強い aug が
  m/l の generalization を助ける、s は元々 aug 頼り
- **imgsz800 は全 size で悪化**: batch size 削減 (s: 8→6, m: 4→3, l: 2→1) の
  悪影響が精度低下を上回る、100 枚では効果薄
- **s_freeze10 が最下位、l_freeze10 が top**: same technique が正反対の効果

## Claude 目視評価 (top 3 model × ep 64 未見 3 frame)

val mAP は僅差 (0.9906 vs 0.9760 vs 0.9732) なので、実 predict の質を目視で比較。

| Frame (skill) | l_freeze10 (mAP top) | l_lowaug | m_lowaug |
|---|---|---|---|
| 01510 (pick 中) | 誤 hole 3、table_top OK | 誤 hole 3、hand_right 抜け | 誤 hole 3、全 class OK |
| 07394 (締め) | leg 3、hole 抜け、label 重複 | conf 低め、全 class ラベル | **leg_tip 高 conf、全 class 検出** |
| 10997 (組立後) | 3 leg + 2 leg_tip 抜け、hand_right 抜け | 3 leg + 3 leg_tip 混雑 | **4 leg + 4 leg_tip 完全検出** |

### 数値 vs 実 predict の乖離

- val の 10 枚が **train と同 ep 60 由来** → 真の未見データより「近い」perf を測定
- l_freeze10 は val で top だが、未見 predict では hand_right 抜け頻発
- m_lowaug は数値 3 位だが未見 predict で最も安定・完全

**教訓**: val は代表性が弱い、実 predict を目視する Phase C は必須。

## v1 暫定採用: **m_lowaug** (v3 で m_lowaug_v3 に更新済、履歴として残す)

| 指標 | 値 | 判定 |
|---|---|---|
| val mAP@0.5 | 0.9732 | ○ top と 0.017pt 差 (僅差) |
| 未見 predict 完全性 | ★★★ | frame 10997 で **4 leg + 4 leg_tip 完全** |
| model size | 20.9M | l (26.2M) より軽量、s (9.7M) より capacity 高 |
| VRAM | batch 4 imgsz 640 で 3GB 前後 | 8GB VRAM で余裕 |
| 推論速度 | ~5ms/frame | 30fps 楽勝 |
| 実機移植 | Jetson Orin Nano 8GB 想定 OK | l より軽量で有利 |

---

# v2 / v3 cycle — semi-supervised active learning + aspect fix

v1 の課題:
- val 代表性弱い (train と同 ep 60)
- 未見 predict で background hole 誤検出 + hand_right 抜け頻発
- 100 枚では汎化限界

に対して、**ep 62 に 300 枚追加 annotation** + **座標変換バグ fix** で改善サイクルを回した。

## v2: semi-supervised active learning (300 枚追加、まだバグあり)

**手法**:
1. m_lowaug (v1 winner) で ep 62 の 300 frame を pre-predict
2. LS project に predictions 付き task として明示 import (`data/annotation_tool/scripts/predict_and_prelabel.py`)
3. user は **描くのではなく修正のみ** → 工数 1/3~1/5 削減
4. 400 枚 (v1 ep 60 100 + v2 ep 62 300、val は v1 val 維持) で 12 config 再学習

**結果 (v1 と同 val で測定)**: 12 config 中 10 個で改善 (mean +1.0pt)、l_imgsz800_v2 が top 0.9838。

## v3: aspect distortion fix

v2 の annotation を目視検証中、**leg/hand_right/table_top の box が微妙に下右方向へ 5px ずれる**現象を user が発見。

**原因**: LS/Konva は **pixel-space** で rotation を適用、当初実装は **%-space** で cos/sin を掛けていた。
非正方 image (640×480、aspect 4:3) で x/y 軸の独立正規化が rotation math に混入して系統誤差 → 小回転で +5px ズレ。

**Fix** (`data/annotation_tool/scripts/ls_export_to_yolo_obb.py`):
- `ls_rect_to_yolo_obb_verts(value, img_w, img_h)` に image dims を渡す
- pixel-space で 4 頂点を rotate、%-value に戻す時に各軸独立正規化
- 逆変換 `yolo_obb_verts_to_ls_rect` も対称化
- test で 640×480 / 500×500 / 1280×720 の 3 aspect で round-trip bit-perfect 確認

**再変換 + 再学習**: HF push (ep 60/62 両方 fix 版) → dataset rebuild → v3 12 config retrain。

## 「真の」val の必要性 — fixed val で全 36 モデル re-val

v3 学習後、**当初は v3-v2 の平均 mAP が -0.005 (悪化に見えた)**。しかし調査で:
- v3 build 時、ep 60 の HF txt は v1 push 時の **buggy 版のまま** (fix 前)
- val (ep 60 の 10 frame) の GT が buggy 版のため、fixed 予測 (v3) が減点され、buggy 予測 (v2) が加点される **偽 val 現象**

**修正**: ep 60 も fix 版で HF push → dataset rebuild → 全 36 model を新 val で re-val (`scripts/reval_all_yolo_runs.py`)。

## 真の 3-way 比較 (fixed val 上、同一 val で測定)

| config | v1 | v2 | v3 | v3-v2 | v3-v1 |
|---|---:|---:|---:|---:|---:|
| s_baseline | 0.8788 | 0.9076 | 0.9454 | +0.038 | +0.067 |
| m_baseline | 0.8669 | 0.9447 | 0.9554 | +0.011 | +0.089 |
| l_baseline | 0.8868 | 0.9088 | 0.9451 | +0.036 | +0.058 |
| s_freeze10 | 0.8407 | 0.9319 | 0.9276 | -0.004 | +0.087 |
| s_imgsz800 | 0.8924 | 0.9176 | 0.9427 | +0.025 | +0.050 |
| s_lowaug | 0.8881 | 0.8996 | 0.9384 | +0.039 | +0.050 |
| m_freeze10 | 0.8751 | 0.9355 | 0.9480 | +0.012 | +0.073 |
| m_imgsz800 | 0.8973 | 0.9054 | 0.9426 | +0.037 | +0.045 |
| **m_lowaug** | 0.8921 | 0.9358 | **0.9547** | +0.019 | +0.063 |
| **l_freeze10** | 0.8959 | 0.9138 | **0.9638** | +0.050 | +0.068 |
| l_imgsz800 | 0.8907 | 0.9400 | 0.9541 | +0.014 | +0.064 |
| l_lowaug | 0.9039 | 0.9344 | 0.9434 | +0.009 | +0.040 |
| **mean** | 0.8842 | 0.9229 | **0.9468** | +0.024 | +0.063 |

- **v3-v2 mean +0.024、11/12 positive** ← aspect fix 明確な improvement
- **v3-v1 mean +0.063、12/12 positive** ← v3 は v1 全 config を上回る
- v1 で top だった l_freeze10 (**0.9906**) は fixed val 上で **0.8959** に急落 → buggy val の artifact だった

## v3 top 3 目視比較 (ep 67)

`scripts/compose_predict_comparison.py` で 3 モデル × 元画像を 2×2 grid 化、`outputs/yolo_obb/predict/comparison_v3_top3/` に 18 枚。

| 指標 | l_freeze10_v3 | m_baseline_v3 | m_lowaug_v3 |
|---|---:|---:|---:|
| fixed val mAP@0.5 | **0.9638** | 0.9554 | 0.9547 |
| ep 67 total boxes | 220 | 217 | 216 |
| avg conf | 0.822 | 0.826 | 0.827 |
| workspace (期待 18) | 18 ✓ | 18 ✓ | 18 ✓ |
| leg (期待 ≤72) | 72 | 70 | 71 |
| leg_tip | 36 | 40 | 35 |
| hand_right | 19 (+1) | 18 ✓ | 18 ✓ |
| hand_left | 17 | 17 | 18 ✓ |

**採用: m_lowaug_v3** — 数値 top ではないが目視で **かぶりなく適切に推論**、hand/leg 完全検出。

## 教訓 (v2 / v3)

- **val 品質が数値評価の全て**: val 側 GT にバグがあるとモデルは「bug 再現力」で採点される
- **座標変換は image 描画で verify 必須**: unit test の round-trip では検出不能 (自己一貫バグ)
- **aspect distortion は非正方 image + 回転で発生**: 640×480 で 5-10px、大回転で 15-20px
- **semi-supervised active learning は工数 1/3-1/5 削減の実効果あり**: predict + LS UI 修正
- **fixed val で v3 > v2 > v1 が明確** → aspect fix + semi-supervised 両方効いた

---

# v4 — background copy-paste augmentation ablation

家庭 domain gap 対策: 学習画像 (lab 黒マット背景) の OBB 内部を保持したまま、
周囲 pixel を Places365 の家庭シーン背景 (office / living_room) に copy-paste 合成。
アノテーション追加ゼロ (OBB 座標は元のまま copy)。

## 実験設計

- **背景プール**: Places365 val_large から office + living_room を DL、user 手動 filter 後 **39 枚** (人物・透かし除外)
- **合成 script**: `data/yolo_obb/augmentation/scripts/copy_paste_bbox.py`
  - OBB polygon rasterize + feather blur (5px) + alpha 合成
  - **strict subset 保証** (hashlib.md5 で per-source deterministic bg ordering、r40 ⊂ r45 ⊂ r50)
- **base hyperparams**: v3 と bit-perfect 同一 (m + lowaug + batch 4 + imgsz 640 + patience 20 + seed 42)、data だけ切替
- **val**: v3 と同一 fully-fixed val (10 imgs、lab domain)

## 4 run 構成

| run | 対象 skill | ×N bg | 合成枚数 | 総 train | move_to_table 割合 |
|---|---|---:|---:|---:|---:|
| **v4_flat** | 全 skill | ×3 | 1170 | 1560 | 17% (control、balance 保持) |
| **v4_r40** | move_to_table のみ | ×2 | 136 | 526 | 39% |
| **v4_r45** | move_to_table のみ | ×3 | 204 | 594 | 46% |
| **v4_r50** | move_to_table のみ | ×4 | 272 | 662 | 51% |

## 結果 (training-time val = 10 imgs、v3 と全 v4 で同一 val)

| run | best ep | mAP@0.5 | Δ vs v3 | mAP@0.5-0.95 | Δ vs v3 | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 baseline | 40 | **0.9780** | — | 0.7567 | — | 0.9195 | 0.9589 |
| v4_r40 | 27 | 0.9708 | -0.7% | 0.7490 | -1.0% | 0.9492 | 0.9234 |
| v4_r45 | 11 | 0.9742 | -0.4% | 0.7744 | +2.3% | 0.9591 | 0.9193 |
| **v4_r50** | 21 | 0.9720 | -0.6% | **0.7922** | **+3.5%** | 0.9622 | 0.9246 |
| **v4_flat** | 13 | 0.9711 | -0.7% | 0.7853 | +2.9% | 0.9639 | **0.9694** |

**観察**:
- **mAP@0.5 は全 v4 で微減 (-0.4〜-0.7%)**: val 10 枚で誤差範囲、有意差判定は限定的
- **mAP@0.5-0.95 は augment 有りが優位**: 高 IoU 精度改善 = 境界 localization 向上
- **r40 → r45 → r50 で単調改善** (mAP@0.5-0.95)、合成枚数増で伸びる
- **flat vs r45 (両方 N=3)**: flat 優位 → move 集中より全 skill 均等 augment の方が効率的だった
- **best epoch が早い** (11-21) → augment data で早期収束、100 epoch は過剰

**caveat**:
- val は lab domain のまま、家庭シーン混入なし → augment の本来の目的 (家庭 domain 汎化) は val で測定不能
- val 10 枚の統計信頼性は限定的、mAP@0.5 微減 vs mAP@0.5-0.95 +3.5% の解釈も noise 込みで慎重に
- **正式判定は実機評価待ち** (別メンバー担当)

## HF weights

repo: `Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb` (private)

| run | HF path (upload 2026-07-18) |
|---|---|
| m_lowaug_v4_flat | `runs/m_lowaug_v4_flat/weights/m_lowaug_v4_flat_20260718.pt` |
| m_lowaug_v4_r40 | `runs/m_lowaug_v4_r40/weights/m_lowaug_v4_r40_20260718.pt` |
| m_lowaug_v4_r45 | `runs/m_lowaug_v4_r45/weights/m_lowaug_v4_r45_20260718.pt` |
| m_lowaug_v4_r50 | `runs/m_lowaug_v4_r50/weights/m_lowaug_v4_r50_20260718.pt` |

## Roadmap (v5 以降)

**Phase 2 候補 (v4 結果依存)**:
- 背景プール拡大 (Home1/2/3 raw MCAP + Places365 全カテゴリ)
- SAM2 mask 方式 (pixel-perfect 切り抜き、bbox 内背景漏れ解消)
- 別 architecture (RF-DETR / D-FINE)

**val の代表性 (残課題)**:
- 別 ep で val set 独立化 (現状も train ep 60 + val ep 60 の依存が残る)
- **家庭 domain val set の整備** (v4 の本来目的が測れるように)
- Multi-seed 検証

**モデル改善**:
- Backbone 変更 (RTDETR, DINO 等)
- Curriculum learning (easy → hard skill)
- HIW-500-LeRobot pretrain (別 Epic)

**Out of scope (本 PR)**:
- ultralytics native visualizer (labels 重複問題) を差替
- ONNX export → Jetson runtime
- YOLO output → WorldState
- Perception wrapper

---

# v5-v10

annotation +145% / bg augment 3 種フル / hsv tune / augment override bug fix。

## 主要改善点

- **annotation 拡張**: v3 baseline 390 image → **v10 966 image (+145%)**、10 ep 対応
  - `push_annotations.py` に `--fallback-to-predictions` (LS 未 review task の predictions を annotation として扱う) 追加
  - LS export の predictions field は ID list なので実 dict を別 API で fetch + inline 修正
  - HF の注釈の repo (今は `Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb`) に total 966 txt 揃った状態
- **bg augment 3 種フル**: v5_material + v8_zoomout + v4_flat の全部盛りで total 4466 train image
- **hsv_v tune**: 0.7 vs 0.4 default で mAP50-95 +0.86pt
- **train.py bug fix**: fliplr / erasing / flipud の argparse override が未通線だった
  - 過去 v8_bg_clean / v9 系 全 run は default aug (fliplr=0.5 / erasing=0.4) で train されていた事実、v10 以降で修正反映

## Best 結果 (v10_hsv07)

| category | value |
|---|---|
| model | yolo11m-obb.pt |
| dataset | v10 (10 ep = 956 train + 10 val、augment 3 dir merge で total 4466 train) |
| batch / imgsz | 4 / 800 |
| override | fliplr=**0.0** / erasing=**0.0** / hsv_v=**0.7** / degrees=10 / translate=0.2 / scale=0.6 / perspective=0.001 |
| default 継続 | mosaic=0.5 / mixup=0 / hsv_s=0.7 / hsv_h=0.015 / flipud=0 / auto_augment=randaugment |
| best epoch | 32 (patience 20 で 52ep early stop) |
| **val mAP50-95** | **0.7942** (v3 = 0.7567、+3.75pt) |
| **val mAP50** | **0.9682** (v3 = 0.9547、+1.35pt) |

## Ablation

### hsv_v A/B (v10 dataset 上)

| config | hsv_v | mAP50-95 | mAP50 |
|---|---:|---:|---:|
| **v10_hsv07** | **0.7** | **0.7942** | 0.9682 |
| v10_baseline | 0.4 (default) | 0.7856 | 0.9709 |

### bg augment 3-way (v10 dataset + bug fix 後)

| config | v5_material | v8_zoomout | v4_flat | mAP50-95 | mAP50 |
|---|:---:|:---:|:---:|---:|---:|
| **v10_hsv07 (3種フル)** | ✓ | ✓ | ✓ | **0.7942** | 0.9682 |
| v10_no_flat | ✓ | ✓ | | 0.7885 | 0.9624 |
| v10_no_zoomout | ✓ | | ✓ | 0.7880 | 0.9556 |

## Video compare (unseen ep = yolo_test_head_left.mp4)

- 3-model compare (v10_hsv07 / v10_baseline / v8_bg_clean): `outputs/yolo_obb/predict/video_compare_v10/`
- 3-way bg ablation compare (v10_hsv07 / v10_no_flat / v10_no_zoomout): `outputs/yolo_obb/predict/video_compare_v10_bg_ablation/`
- **目視**: v10_hsv07 が全 frame で安定、bg augment 片方抜きは workspace / hand の一部で崩れ

## 次アクション (v10 の範囲外)

- 追加 hsv sweep (0.5〜0.8 の細粒度)、annotation 更拡張は別 issue
- v10 weight の HF upload (まだしていない)

---

# 検討外モデル: RF-DETR OBB (v5 の検証で試して見送り)

v5 の検証で **roboflow/rf-detr (PR #921 feat/obb-support、farukalamai fork 版)** を代替
architecture として検証した結果、**採用見送り**。実装 sub-workspace (`model/rf_detr_obb/`) と
DOTA 変換 pipeline は削除済み。ただし DOTA converter script
[data/yolo_obb/scripts/yolo_obb_to_dota.py](../../data/yolo_obb/scripts/yolo_obb_to_dota.py)
は MMRotate も同 format を使うため残置。

## 検証結果

- pixi feature-env で install → `rfdetr` import 成功、`ModelConfig(oriented=True)` 起動確認
- v5_material DOTA dataset (base 400 + augmentation 1170 = 1560 train + 10 val) で smoke train 通過
- **本 train (30 epoch × 1.5min/epoch = 46min) 実施 → val mAP 全 epoch 0.0** = 実質学習失敗

## 失敗の直接原因

roboflow 公式配布の pretrain `rf-detr-small.pth` 読み込み時に以下 warning:

```
Pretrained weights loaded only partially — 7 model parameter(s) not in checkpoint
(left at random init): [_kp_active_mask, angle_embed.layers.0.{weight,bias},
angle_embed.layers.1.{weight,bias}, ...]
Using a different number of positional encodings than DINOv2, ... not loading DINOv2 backbone weights
Using patch size 16 instead of 14, ... not loading DINOv2 backbone weights
```

- **公式 pretrain は AABB (axis-aligned) 版のみ、OBB 用 angle head は含まれない**
- PR #921 は未マージ・作者 fork の実装で、OBB pretrain は未配布
- 30 epoch fine-tune では angle head が random init のまま収束せず

## 見送り判断根拠

代替 MMRotate/RTMDet-R (DOTA-v1.0 benchmark) と YOLO11-OBB を比較:

| Model | Params(M) | Latency (TRT-FP16) | mAP50 (RR) |
|---|---:|---:|---:|
| **YOLO11m-obb** (現行) | 20.9 | 7.6ms | 80.9 |
| RTMDet-m (MMRotate) | 24.67 | 7.82ms | 80.26 |
| YOLO11l-obb | 26.2 | 12.0ms | 81.3 |
| RTMDet-l | 52.27 | 10.82ms | 80.54 |

- **YOLO11-OBB と RTMDet-R は精度・速度で ±1% 内、latency 同 order**
- **RF-DETR OBB は現状 setup で使い物にならない** (angle_embed pretrain 欠損)
- upstream (roboflow) が OBB pretrain 公式配布するまで待ち、または from-scratch 数百 epoch 必要
- **YOLO11-OBB (現行) を継続採用**、次に検証するなら MMRotate RTMDet-m を選ぶ

## 次アクション

- RF-DETR OBB は upstream 状況変化 (公式 OBB pretrain 配布) を待って再検討

---

# 検討外モデル: MMRotate (RTMDet-R, Oriented R-CNN 等、v5 の検証で試して見送り)

RF-DETR OBB 見送り後、代替として MMRotate/RTMDet-R (DOTA benchmark SOTA 級) の検証を試みたが、
**MMRotate 自体が事実上 abandoned で採用見送り**。

## 実態

- 最新 tag = **v1.0.0rc1 (2023-11)**、stable release **なし** (2 年経過)
- `dev-1.x` branch も **最終 commit 2024-09-13** (~1 年更新なし)
- mmcv 2.0.x pre-built wheel は openmmlab index に**無い** (`torch2.4/index.html` は 2.1.0/2.2.0 のみ、`torch2.1/index.html` は 2.1.0/2.2.0 のみ)
- mmcv 2.0.x を source build しようとすると `pkg_resources` build isolation 問題で fail (mim install 経由でも同じ)

## Version 依存の硬直

mmrotate 1.0.0rc1 を軸に compat matrix を辿ると **必然的に古い stack** に固定される:

- mmrotate 1.0.0rc1 → mmdet **>=3.0.0rc6, <3.1.0**
- mmdet 3.0.0 → mmcv **>=2.0.0rc4, <2.1.0**
- mmcv 2.0.1 → source build 必須 (pre-built wheel 無し) → build 失敗の連鎖

**「2023-05 相当の古い環境を無理やり組む」以外に install path なし**。

## DETR 系代替 (O2-RT-DETR / AO2-DETR) も検討

- **O2-RT-DETR** (wokaikaixinxin/O2-RT-DETR, IEEE TGRS 2026): **ai4rs framework 依存 = mmrotate と同種の install 難関**、DOTA AP50=78.45 (R50, YOLO11m 80.9 に微劣)
- **AO2-DETR** (Ixiaohuihuihui/AO2-DETR): 作者「サーバー水没でコード完全保存できず、baseline のみ」= incomplete
- **Oriented-DETR / RQFormer / s2anet**: 独立実装だが小 stars、maintain 不安定、精度不明

## 総合結論

**maintain 中 + install 現実的 + 明確に YOLO を上回る OBB Model は現時点で無い**:

| 評価軸 | YOLO11-OBB | MMRotate/RTMDet-R | RF-DETR OBB | O2-RT-DETR |
|---|---|---|---|---|
| maintain | ⭐ Ultralytics active | ❌ abandoned | ⚠️ PR unmerged | ⭐ 2026-03 update |
| install | ⭐ 簡単 | ❌ 不可 | ⭐ 成功 | ❌ ai4rs 依存 |
| pretrain | ⭐ DOTAv1 済 | ⭐ 済 | ❌ OBB head 欠損 | ⭐ modelscope 済 |
| DOTA AP50 | 80.9 (m) | 80.26 (m) | — | 78.45 (R50) |

**→ YOLO11-OBB (現行) 継続採用**。

## RF-DETR / MMRotate / O2-RT-DETR いずれも upstream 側の変化が必要:

- RF-DETR: roboflow が OBB pretrain を公式配布 (PR #921 マージ)
- MMRotate: OpenMMLab が新 stable release + mmdet 3.x 対応 (現状停止中)
- O2-RT-DETR: 独立 install pipeline (mmcv/mmdet 系脱却)
