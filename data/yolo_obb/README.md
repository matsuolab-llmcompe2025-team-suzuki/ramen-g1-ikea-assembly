# data/yolo_obb — YOLO-OBB Training Dataset Build

v1 (100 枚固定) → v10 (966 枚) → v11b (v10 + head_left 724 枚、2026-08-19)。

`data/annotation_tool/` の LS annotations と HF frames を pull して、
ultralytics YOLO-OBB 学習用 dataset を組む pipeline。

## Layout

```
data/yolo_obb/
├── configs/                          # ultralytics dataset yaml (v10, v11a, v11b, ...)
├── scripts/
│   ├── build_dataset.py              # v10 real: HF pull → skill stratified split
│   ├── extract_frames_for_pseudo.py  # mp4 → png 抽出 (v11 head_left 用)
│   ├── pseudo_label_and_mosaic.py    # YOLO 推論 + overlay + mosaic QC
│   ├── materialize_head_left.py      # LS export → YOLO OBB txt → dataset 配置
│   └── rebuild_from_hf.py            # cleanup 後の完全復元 (v10 + augment + head_left)
├── dataset/                          # .gitignore、run で生成 (v10 real 966)
├── augmentation/                     # .gitignore、synthetic 生成物
│   ├── synthetic/{v4_flat,v5_material_random,v8_zoomout_random}/{images,labels}/
│   └── backgrounds/places365/        # v4_flat の bg copy-paste source
├── head_left/                        # .gitignore、v11 新環境
│   ├── cycle1/{images,labels}/       # bootstrap 100 frame (100% user 修正)
│   └── cycle2/{images,labels}/       # 残り 623 frame (user 195 修正 + v11a pseudo 428)
├── pseudo_source/                    # .gitignore、pseudo-label 中間物 (削除可)
├── test_mp4/                         # source video (head_left mp4 等)
└── .hf_cache/                        # .gitignore、HF DL cache
```

## Usage

### v10 real dataset build (従来の flow)

```bash
# Root workspace で pixi run -e train (torch/cuda + ultralytics)
pixi run -e train python -m data.yolo_obb.scripts.build_dataset
```

optional flags:
- `--out data/yolo_obb/dataset` (出力 root)
- `--yaml data/yolo_obb/configs/dataset.yaml` (ultralytics config)
- `--val-ratio 0.10` (default 90/10)
- `--seed 42` (train/val 割り振り再現性)
- `--train-only-eps 62,120,510,513,515,517,518,519,531` (v10 で追加された ep は全部 train)

### 既知の挙動: 床のマットの枠がはみ出した画像は学習に使われない

床のマット (`workspace`) は画面より大きく写ることが多く、その斜めの枠は画像の外にはみ出す (label の座標が 0〜1 の外)。
ultralytics (8.4.80) は枠を軸に沿った箱に直して幅・高さが 1.01 を超えると、その画像を `corrupt` として**画像ごと**捨てる
(同じ画像のほかの枠も使われない)。v10 の 966 枚では 321 枚 (train 284・val 37) がこれに当たる。公開した YOLO (v11b) も
同じ版の ultralytics で学習していて、直していない。直すなら、データを作る段ではみ出した枠を見えている部分に
切り詰める (斜めの枠と画像の四角の重なり → それを囲む最小の斜めの四角)。

### 完全 rebuild (cleanup 後、v11b 学習 dataset 一式を HF から復元)

local dataset を全消しした後、以下 1 コマンドで **v10 real + augmentation 3 dir + head_left 724** を
HF から pull し、`dataset_v11b.yaml` が期待する layout に再構成する:

```bash
cd data/annotation_tool
pixi run python ../yolo_obb/scripts/rebuild_from_hf.py

# 完了後、通常通り train できる
cd ../../model/yolo_obb
pixi run train-yolo --config model/yolo_obb/configs/m_lowaug_v11b.yaml
```

optional flags:
- `--skip-real` (v10 real dataset build を skip)
- `--skip-aug` (augmentation 3 dir pull を skip)
- `--skip-head-left` (head_left cycle1/2 pull + split を skip)

処理:
1. **v10 real** (966 image + label): `build_dataset.py` を subprocess 呼び出し (HF pull + skill stratified split)
2. **augmentation 3 dir** (v4_flat / v5_material_random / v8_zoomout_random、3510 image + 3510 label):
   `snapshot_download` → `augmentation/synthetic/{name}/{images,labels}/{skill}/{ep}/` に配置
3. **head_left** (724 flat 上の HF layout): `stride=32` で cycle1 (0,32,...,3200) / cycle2 (rest) に自動 split →
   `head_left/{cycle1,cycle2}/{images,labels}/frame_XXXXXX.{png,txt}` に配置

### v11 head_left 新環境の bootstrap active learning (2 cycle)

新環境 (別 lab 照明 / home domain 等) の mp4 を YOLO 学習に取り込む一連の flow。
Cycle 1 で少数 frame を user 修正 → 学習 → その model で残り frame を pseudo-label →
Cycle 2 で軽微修正 → 最終学習、の 2 段構え:

```bash
# Cycle 1: 100 frame pilot (stride=32 均等抽出)
pixi run python data/yolo_obb/scripts/extract_frames_for_pseudo.py \
    --video data/yolo_obb/test_mp4/head_left.mp4 \
    --out data/yolo_obb/pseudo_source/head_left/frames --n 100
pixi run python data/annotation_tool/scripts/prelabel_head_left.py \
    --frames-dir data/yolo_obb/pseudo_source/head_left/frames \
    --weight model/yolo_obb/runs/m_lowaug_v10_hsv07_imgsz800/weights/best.pt \
    --project-title v11_cycle1 --ls-subdir head_left_v11_cycle1
# → LS UI で修正 → materialize
pixi run python data/yolo_obb/scripts/materialize_head_left.py \
    --project-id <LS id> --cycle cycle1 \
    --frames-dir data/yolo_obb/pseudo_source/head_left/frames
# → train v11a (m_lowaug_v11a_cycle1.yaml)

# Cycle 2: 6fps downsample、cycle1 と重複回避
pixi run python data/yolo_obb/scripts/extract_frames_for_pseudo.py \
    --video data/yolo_obb/test_mp4/head_left.mp4 \
    --out data/yolo_obb/pseudo_source/head_left_cycle2/frames --fps 6 \
    --exclude-dir data/yolo_obb/pseudo_source/head_left/frames
pixi run python data/annotation_tool/scripts/prelabel_head_left.py \
    --frames-dir data/yolo_obb/pseudo_source/head_left_cycle2/frames \
    --weight model/yolo_obb/runs/m_lowaug_v11a_cycle1/weights/best.pt \
    --project-title v11_cycle2 --ls-subdir head_left_v11_cycle2
# → LS UI で修正 (v11a pseudo は cycle1 より高精度なので軽微修正で済む) → materialize
pixi run python data/yolo_obb/scripts/materialize_head_left.py \
    --project-id <LS id> --cycle cycle2 --fallback-to-predictions \
    --frames-dir data/yolo_obb/pseudo_source/head_left_cycle2/frames
# → train v11b (m_lowaug_v11b.yaml)
```

`--fallback-to-predictions` は「user が review しなかった task に v11a の pseudo-label を
そのまま採用する」オプション (v11a の推論精度が高いことを利用して user 工数削減)。

## HF backup を source of truth として使う (local cleanup 後の retrieve 手順)

`annotation_tool/workspace/` (LS 内部 DB + image dump) を消した後でも、以下の 3 情報は HF の `Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb` に永続:
- **labels** (`yolo_obb/labels/**/*.txt`): frame ごとの annotation 結果
- **LS raw exports** (`ls_exports/*.json`): annotation の履歴 (誰が / いつ / どう修正) の source of truth
- **frames**: 注釈を付けた画像だけ (候補の pool は HF に無い。下の「annotate 状況の逆引き」)

### annotate 状況の逆引き (「どの ep のどの frame が未 annotate か」)

注釈を付けていない候補の画像は HF に置いていない (公開する dataset repo には注釈済みの画像だけを移した)。
候補は `data/annotation_tool/scripts/dump_head_frames.py` (`--hf-upload-repo` を付けなければ手元に抽出するだけ) で
元データから取り直し、手元の候補と `yolo_obb/labels/` を突き合わせる。

**pick 戦略の候補** (次に annotate する frame を選ぶ):
- **skill/ep 均等**: 各 skill/ep から N 枚ずつ pick (dataset の分布均衡)
- **uniform stride**: ep 内で等間隔サンプル (動作カバレッジ)
- **末尾優先**: 各 ep の後半 frame (assembly の insertion 段など難所)
- pick 結果を `prelabel_head_left.py` に食わせる形にする → LS import → 追加 annotate
  (ただし現行の prelabel_head_left は flat frame dir 前提、skill/ep 階層 pick 用に軽い adapter が要る)

### head_left (mp4 由来) の追加 annotate

HF 側の `head_left/frames/*.png` は現在 mp4 から抽出した 724 枚が全て = 全部 annotate 済 → HF diff 上は「未 annotate 0」。
更に増やすには mp4 から**別 stride / offset** で追加抽出:

```bash
# 例: 3fps 相当 (stride=10) で追加 322 frame、既存 724 は skip
pixi run python data/yolo_obb/scripts/extract_frames_for_pseudo.py \
    --video data/yolo_obb/test_mp4/head_left.mp4 \
    --out data/yolo_obb/pseudo_source/head_left_cycle3/frames \
    --fps 3 --exclude-dir <既存 cycle1+cycle2 の frame dir>
# → 以降 v11b で prelabel → LS 修正 → materialize → v11c train の cycle
```

### LS export JSON の pull (annotation 履歴を復元したい場合)

`ls_exports/*.json` は生 annotation の source of truth。**LS DB (`workspace/ls_data/`) を消した後でも annotation 中身は完全復元可**:

```bash
# 単一 file
huggingface-cli download Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb \
    ls_exports/head_left_cycle1_20260818T054514Z.json --repo-type dataset \
    --local-dir data/annotation_tool/workspace/exports/

# 全 file (20 file)
huggingface-cli download Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb \
    --include "ls_exports/*.json" --repo-type dataset \
    --local-dir data/annotation_tool/workspace/exports/
```

→ 以降 `ls_export_to_yolo_obb.py` で YOLO txt 変換、あるいは LS API `import` endpoint で
`workspace/ls_data/` に再 import して project 26/27 相当を再現、も可能。

## Data source

- HF `Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb` (画像とラベル。元の 2 repo から要る物だけを移した。CC-BY-4.0、出どころは dataset card)
  - `frames/{skill}/{ep}/*.png` と `yolo_obb/labels/{skill}/{ep}/*.txt` (v10 real)
  - `augmentation/synthetic/v5_material_random/...` と `yolo_obb/labels/augmentation/synthetic/v5_material_random/...`
  - `head_left/frames/*.png` と `yolo_obb/labels/head_left/*.txt` (v11 新環境、flat 724)
  - `ls_exports/*.json` (LS 生 export、reproducibility 用)
- **Places365 の背景の合成 (`v4_flat`・`v8_zoomout_random`) は HF に無い** (Places365 は再配布できない)。v11b の学習には要るので、
  Places365 を自分で取ってきて `data/yolo_obb/augmentation/scripts/` で作る (背景の置き場所は `augmentation/backgrounds/places365/`)
- v10 real dataset: 10 ep (60, 62, 120, 510, 513, 515, 517, 518, 519, 531) / 966 image / 6 skill
- head_left: 30fps × 107 秒の mp4 から stride=32 (cycle1 101) + 6fps (cycle2 623) = 724 image

## Split strategy

**skill stratified random split** (v1):
- 各 skill 内で `val_ratio` 分を val に割り振り
- 各 skill 最低 1 枚 val (少なすぎ回避)
- seed 固定で再現可能

100 枚 × val_ratio=0.10:
- 各 skill 20 枚 → 2 枚 val = 5 skill × 2 = 10 val (train 90 val 10)
- 完全 random だと val に居ない skill が出得る → stratified 採用

## Class balance report (100 枚 / seed 42 / val_ratio 0.10 実測)

```
train: 90 frames, 1009 boxes
val:   10 frames,  105 boxes

class          train  val    axis-aligned 外接矩形の平均 size (normalized)
─────────────  ─────  ───    ─────────────────────────────────────────────
workspace       92    10     w=1.00 h=0.72 (画面全体)
leg            347    40     w=0.17 h=0.32 (縦長)
leg_tip        165    13     w=0.08 h=0.08 (小)
hole           137    13     w=0.07 h=0.07 (小)
table_top       86     9     w=0.55 h=0.53
hand_right      92    10     w=0.22 h=0.25
hand_left       90    10     w=0.20 h=0.24
```

- **全 7 class が train/val 両方に存在** ✓
- **min class の box 数 (train)**: table_top = 86 (>> 30 の acceptance criteria クリア)
- leg 系 (leg, leg_tip) が上位 → 予想通り (工程で最も現れる)
- workspace は 1 frame につき 1 box、size 大 (画面全体) → 学習容易だが情報量少

## Ultralytics load 検証

```python
from ultralytics.data.utils import check_det_dataset
data = check_det_dataset('data/yolo_obb/configs/dataset.yaml')
# → pass、nc=7 / names 完全一致
```

`YOLO('yolo11n-obb.pt').train(data='data/yolo_obb/configs/dataset.yaml', ...)` の直投入 OK。

## Acceptance (v1)

- [x] `dataset.yaml` が `ultralytics YOLO('yolov11n-obb.pt').train(data=dataset.yaml, ...)` の load を通す
- [x] train/val split が seed 固定で再現可能 (split_manifest.json)
- [x] class balance report 出力
- [x] min class の box 数 > 30 (avg 11 boxes × 20 枚 = 200 期待、少ない class でも十分)

## Out of scope (v2)

- 追加 annotation → train 結果 (val mAP) 見てから判断 (v2)
- Test split の準備 (v1 は train + val のみ、test は Sakura 本走で判断)
- HIW-500 汎化 pretrain
