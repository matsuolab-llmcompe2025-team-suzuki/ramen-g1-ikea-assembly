# Curation済みLeRobot datasetの生成とHugging Faceへのアップロード

`build_curated_task_dataset.py` は、HARA curation metadataから指定task IDの
non-failure区間を抽出し、LeRobot v3 datasetを生成します。生成後の構造検証に
成功した場合だけ、Hugging Face Dataset repoへアップロードできます。

## 1. 環境と認証

```bash
cd "$REPO_ROOT"/data/curation_tool  # REPO_ROOT = this repository's root
pixi install
pixi run hf auth login
```

対話ログインの代わりに、write権限を持つtokenを環境変数で渡すこともできます。
tokenをコマンドライン引数に直接記載しないでください。

```bash
export HF_TOKEN="<write token>"
```

`Team-RAMEN` 以下に作成する場合は、tokenのユーザーにOrganizationのdataset
作成・書き込み権限が必要です。

## 2. 抽出内容の確認

`--dry-run` はsource/curationのrevisionと抽出件数を表示します。データ生成、
repo作成、アップロードは行いません。

```bash
pixi run build-curated-task \
  --task-id 2 \
  --video-keys \
    observation.images.cam_0 \
    observation.images.cam_2 \
    observation.images.cam_3 \
  --upload-repo-id Team-RAMEN/IROS2026_RAMEN_HARA_task_2_non_failure \
  --dry-run
```

GR00Tの既存3-camera構成では `cam_0`, `cam_2`, `cam_3` を使用します。
4枚のRGBを保存する場合は `cam_1` を加えてください。`--video-keys` 自体を
省略するとRGB/IRを含むsourceの全video featureを保存します。

動画は既定で `--video-mode lossless`（H.264 CRF 0）を使用します。指定した
frame境界を維持し、decode後の画素をsourceと一致させる可逆圧縮です。rawの
非圧縮動画ではありません。容量を優先して非可逆CRF 20を許容する場合だけ
`--video-mode reencode` を指定します。`--video-mode copy` は再encodeしませんが、
GOP境界の前後frameが混入し得るため、frameとstateの厳密な対応が必要な学習用
datasetには推奨しません。

出力frameの `task_index` と `meta/tasks.parquet` は常に0へ再採番せず、抽出に
指定したcuration task IDで統一します。

## 3. 新しいprivate repoへの生成・アップロード

```bash
pixi run build-curated-task \
  --task-id 2 \
  --video-keys \
    observation.images.cam_0 \
    observation.images.cam_2 \
    observation.images.cam_3 \
  --output-dir \
    ../bitrobot_lerobot_subtask_datasets/outputs/curated_task_2_non_failure \
  --upload-repo-id Team-RAMEN/IROS2026_RAMEN_HARA_task_2_non_failure
```

新規repoは既定でprivateです。公開datasetとして新規作成する場合だけ
`--public` を追加します。

処理順序は次のとおりです。

1. curation metadataを検証して区間を選択
2. 必要なsource shardだけをダウンロード
3. LeRobot v3 datasetをローカル生成
4. episode/frame index、Parquet、参照動画を検証
5. Hugging Faceへアップロード

アップロードには `HfApi.upload_large_folder` を使用します。転送状態はoutput
directoryの `.cache/.huggingface/` に保存され、大きな動画datasetでも再実行時に
転送済みファイルを利用できます。並列数は既定で4です。変更する場合は、例えば
`--upload-workers 2` を指定します。

### CPUを使った動画生成の並列化

動画clipのFFmpeg処理は `--video-workers` で並列化できます。
`--ffmpeg-threads` は各FFmpeg processが使用するthread数です。CPU使用thread数の
目安は両者の積になるため、32 CPU thread以内に抑える場合は次のようにします。

```bash
--video-workers 16 --ffmpeg-threads 2
```

あるいは、短いclipを多数処理する今回のdatasetでは次も利用できます。

```bash
--video-workers 32 --ffmpeg-threads 1
```

source動画のHugging Face downloadは、過剰な同時接続を避けるため最大8並列です。
現在のcameraのsource取得が完了すると、FFmpegでclipを生成している間に次のcameraの
sourceをbackgroundで先読みします。これによりnetwork I/OとCPU処理を重ね、camera
切り替え時の待ち時間を減らします。download workerはI/O用で、上記のFFmpeg CPU
thread数の積には含まれません。

clipの生成結果は並列実行の完了順ではなく、出力episode順に回収・連結されます。
したがって、worker数を増やしたり次cameraを先読みしたりしてもLeRobotの
camera/episode/frame順序は変わりません。

各clipの実frame数は対応するdata episodeの `length` と照合されます。packed動画の
`from_timestamp` / `to_timestamp` はcontainer durationの累積ではなく、file内の
整数frame offsetをFPSで割って生成します。これによりMP4 durationのミリ秒丸めが
積み重なってTorchCodecのframe indexがずれることを防ぎます。

`--video-workers` の既定値は1、`--ffmpeg-threads` の既定値は0（FFmpeg自動選択）
です。`--upload-workers` は生成完了後のHugging Face転送だけに使用され、動画生成
速度には影響しません。

## 4. 既存repoの更新

既存repoを誤って変更しないよう、同名repoが存在すると既定では停止します。
内容を更新する場合は、対象repoを確認してから `--replace-remote` を明示します。

```bash
pixi run build-curated-task \
  --task-id 2 \
  --video-keys \
    observation.images.cam_0 \
    observation.images.cam_2 \
    observation.images.cam_3 \
  --output-dir \
    ../bitrobot_lerobot_subtask_datasets/outputs/curated_task_2_non_failure \
  --upload-repo-id Team-RAMEN/IROS2026_RAMEN_HARA_task_2_non_failure \
  --replace-remote \
  --force
```

`--force` はローカルoutputを作り直します。`--replace-remote` は既存remote repo
へのアップロードを許可し、アップロード成功後に、ローカルdatasetに存在しない
古い `data/`, `meta/`, `videos/`, `README.md` だけを削除します。
`.gitattributes` や `LICENSE` など、dataset builderの管理外ファイルは削除しません。

## 5. verdictの指定

既定ではschema v4におけるFailure以外の全verdictを採用します。

```text
optimal
success
ambiguous
```

`ambiguous` も除外する場合:

```bash
--verdicts optimal success
```

`failure` を `--verdicts` に指定すると、安全のためエラーになります。

## 6. 出力とprovenance

アップロードされる主なファイル:

```text
README.md
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-*/file-*.parquet
meta/team_ramen_curation.json
data/chunk-*/file-*.parquet
videos/<video-key>/chunk-*/file-*.mp4
```

`meta/team_ramen_curation.json` には、source/curationのcommit SHA、採用verdict、
元episode ID、inclusiveな元frame範囲、出力episode IDが記録されます。

アップロード完了時には次のURLが表示されます。

```text
https://huggingface.co/datasets/<namespace>/<repo>
```

ローカル生成に成功した後でアップロードだけが失敗した場合、ローカルdatasetは
削除されません。認証やネットワークを修正した後、datasetを再生成せずに検証と
アップロードを再開できます。

```bash
pixi run build-curated-task \
  --upload-existing \
  --output-dir \
    ../bitrobot_lerobot_subtask_datasets/outputs/curated_task_2_non_failure \
  --upload-repo-id Team-RAMEN/IROS2026_RAMEN_HARA_task_2_non_failure \
  --replace-remote
```

最初のアップロードがrepo作成前に失敗した場合は `--replace-remote` は不要です。
repo作成後に失敗した場合は、既存repoへの再開を許可するため必要です。
`--upload-existing` でも、ローカルdatasetの全構造検証に成功しなければ
アップロードしません。
