# PNGフレームキャッシュ

動画を一度だけ連続デコードし、学習時には必要なPNG画像だけを読みます。
JPEGへの再圧縮を避け、PyAVが動画から復元したRGB画素を保持するための機能です。
動画の元の圧縮で失われた情報を復元するものではありません。
解像度、カメラ数、関節データ、データ分割、学習時のaugmentationは変更しません。

既存JPEGの生成・読み込みは引き続き利用できます。PNGは明示的な選択が必要で、
新しい既定保存先は `frame_cache_png/` です。既存の `frame_cache/` と共存できます。
PNG生成は元MP4を削除しません。JPEG生成の既定には元MP4削除があるため、
形式にかかわらず元動画を残したい場合は `--keep-source-mp4` を指定してください。

## 生成

リポジトリのルートで、既存のLeRobot学習環境を有効化して実行します。
PNG生成にはPyAV・OpenCV・NumPyが必要です。`gr00t-lerobot.sif` の既存環境でも
実行でき、GPUは不要です。以下のパスは自分の保存先に置き換えてください。

```bash
DATASET_ROOT=/path/to/lerobot_dataset
PNG_CACHE_ROOT=/path/to/frame_cache_png

python data/bitrobot_lerobot_subtask_datasets/scripts/precompute_frame_cache.py \
  --lerobot-root "$DATASET_ROOT" \
  --cache-root "$PNG_CACHE_ROOT" \
  --image-format png \
  --png-compression 1 \
  --video-keys observation.images.cam_0,observation.images.cam_2,observation.images.cam_3 \
  --jobs 32 \
  --keep-source-mp4
```

`--jobs` は割り当て済みCPU数と、同時に動く学習の負荷に合わせてください。
上の `--video-keys` はBitRobotソースの頭部左・左右手首に対応します。
既に学習用viewを作成済みなら、view内の
`observation.images.head_left`、`observation.images.left_wrist`、
`observation.images.right_wrist` の実際のキーを指定するか、
3カメラだけのviewに対して `--video-keys` を省略できます。
未指定ではデータセット内の全video featureが対象になり、RGB以外も含まれ得るため、
この機能では8-bit RGBカメラだけを選択してください。depthのPNG化は対象外です。

`--png-compression` は0〜9、既定1です。数値は圧縮処理量・容量の調整であり、
画質の設定ではありません。まずは既定値で速度と容量を確認してください。
`--quality` は従来どおりJPEG専用です。

生成は動画全体が対象で、train/validation/testの分割では絞りません。
必要容量には元動画、PNG、既存の学習チェックポイントをすべて含めてください。
フレームごとにファイルを作るため、空きバイト数だけでなくinode数も確認します。

## 完全性と再実行

- `_cache_meta.json` に形式、圧縮設定、FPS、デコーダ、生成完了状態を記録します。
- PNGはPyAVで連続デコードし、ゼロ始まりの固定FPSとフレーム時刻を検証します。
  時刻の欠落・重複・ずれは、誤ったフレームを返す代わりに生成エラーにします。
- 途中終了や `--max-frames-per-mp4` による試験生成を、完成済みとして再利用しません。
- 完了していないPNGキャッシュは学習側でも拒否します。
- 再生成コマンドを実行すると、完成済みでもPNGの読み取り・整合性確認を行います。
  既に検証済みのキャッシュを次の学習で使うだけなら、下記の
  `FRAME_CACHE_PRECOMPUTE=false` で生成前処理を省略できます。
- 圧縮設定や形式を変更する場合は別の `--cache-root` を使うことを推奨します。
  `--force` は指定キャッシュの再生成を許可する操作です。他の実験が利用中の
  キャッシュには指定しないでください。
- 完成したキャッシュは学習中に上書きせず、別ディレクトリで準備してから
  次の学習プロセスで選択してください。

## 学習で利用する

生成済みのPNGキャッシュを共通学習スクリプトで利用する例です。
既存のデータセット・モデル・出力先の設定は、自分の実験設定を引き継ぎます。

```bash
cd model/subtask_policy_training

LEROBOT_FRAME_CACHE_ENABLE=true \
FRAME_CACHE_IMAGE_FORMAT=png \
FRAME_CACHE_ROOT_OVERRIDE=/path/to/frame_cache_png \
FRAME_CACHE_PRECOMPUTE=false \
FRAME_CACHE_STRICT=true \
FRAME_CACHE_NUM_VARIANTS=1 \
POLICY_TYPE=diffusion \
bash scripts/train_lerobot.sh
```

`FRAME_CACHE_STRICT=true` は、キャッシュ欠損・未完了・形式不一致のときに
動画デコードへ黙って戻ることを防ぎます。falseなら、当該要求全体を元の
デコーダへ戻します。PNGが欠けた場合に別のJPEGキャッシュを選ぶことはしません。
古いJPEGメタデータは形式フィールドがなくても引き続きJPEGとして扱います。

共通スクリプトに生成も任せる場合は、次の設定を利用できます。

```bash
LEROBOT_FRAME_CACHE_ENABLE=true \
FRAME_CACHE_IMAGE_FORMAT=png \
FRAME_CACHE_PNG_COMPRESSION=1 \
FRAME_CACHE_JOBS=32 \
FRAME_CACHE_PRECOMPUTE=true \
FRAME_CACHE_STRICT=true \
FRAME_CACHE_NUM_VARIANTS=1 \
POLICY_TYPE=diffusion \
bash scripts/train_lerobot.sh
```

この場合、単一データセットでは学習用view、複数データセットでは結合ソースの
`frame_cache_png/` が既定保存先です。`FRAME_CACHE_ROOT_OVERRIDE` を指定すれば、
生成側と読み込み側がともにそのディレクトリを使用します。
実行前に `DRY_RUN=true` を追加すると、生成コマンドと選択先を確認できます。
PNG自動生成では元MP4を保持します。

## 対応範囲・注意

- PNG対応はrawフレームの生成と、共通LeRobotキャッシュローダーです。
  RAMEN-Ori等で既存PNGを選ぶ場合は、その実験のJPEG自動生成を止めてください
  （`FRAME_CACHE_PRECOMPUTE=false`）。
- 焼込みaugmentation/OBBのvariantキャッシュは従来のJPEGのままです。
  PNG rawキャッシュでは `FRAME_CACHE_NUM_VARIANTS=1` を使用します。
  読み込み後の通常のオンラインaugmentationは引き続き適用できます。
- JPEG専用のunified cache生成・検証・tar packing・HF管理ワークフローを
  PNGへ切り替える機能は、この変更には含みません。
- 既に起動している学習ジョブや、別ブランチのSlurmラッパーを変更する機能では
  ありません。独自ラッパーがcacheをfalseに固定している場合は、別途切り替えが必要です。
- PNGの画像展開速度と、学習全体の高速化率は別です。導入前後の実際の
  ステップ時刻差で測定してください。生成時間と追加容量も考慮します。
