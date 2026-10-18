# BitRobot LeRobot Subtask Datasets

This folder builds Team RAMEN subtask datasets directly from the official
BitRobot LeRobotDataset v3 dataset:

```text
BitRobot/G1_WBT_Dex1_Building-Children-Table
```

The output repos are:

```text
Team-RAMEN/IROS2026_RAMEN_suzuki_{subtask}_1
```

The conversion keeps the official LeRobot v3 feature names and data layout.
It only slices each full-table-building episode into subtask intervals using
timestamps from:

```text
BitRobot/2026-humanoid-ikea-assembly-challenge
```

No camera renaming, state/action concatenation, GR00T-specific layout, or
camera dropping is performed here. Training and inference scripts should select
and map model-specific inputs.

Parquet files are written with the LeRobot v3 `data_files_size_in_mb` metadata
set to 100 MB. Videos are clipped per subtask and then merged per camera into
LeRobot v3 video files with `video_files_size_in_mb` set to 500 MB.

## Usage

For an optional lossless PNG cache that avoids repeated video decoding during
training, see [PNG frame cache](docs/png_frame_cache.md). JPEG remains the
default; PNG caches are stored separately and retain the source videos.

Smoke check without videos:

```bash
model/subtask_policy_training/.venv/bin/python \
  data/bitrobot_lerobot_subtask_datasets/scripts/build_subtask_datasets.py \
  --subtasks flip_table \
  --max-episodes 2 \
  --skip-videos \
  --force
```

Build one full subtask locally with videos:

```bash
model/subtask_policy_training/.venv/bin/python \
  data/bitrobot_lerobot_subtask_datasets/scripts/build_subtask_datasets.py \
  --subtasks flip_table \
  --force
```

Upload after a successful build:

```bash
model/subtask_policy_training/.venv/bin/python \
  data/bitrobot_lerobot_subtask_datasets/scripts/build_subtask_datasets.py \
  --subtasks flip_table \
  --upload \
  --private \
  --video-files-size-mb 500 \
  --force
```

Full all-subtask conversion clips many videos and can take a long time. Use
`--video-keys` to limit cameras when intentionally making a model-specific
derived dataset; omit it to preserve all official RGB and IR cameras.

## Build from the HARA curation metadata

`build_curated_task_dataset.py` reads
`Team-RAMEN/ramen-g1-ikea-assembly-curation/labels.parquet`, selects one corrected
`task_index`, excludes `verdict=failure`, and materializes each remaining
curation segment as one output episode. Frame ranges are interpreted exactly as
the curation schema defines them: episode-local, zero-based, and inclusive.

Preview task ID 2 without downloading source data or videos:

```bash
cd data/curation_tool
pixi run build-curated-task --task-id 2 --dry-run
```

Build task ID 2 with the four RGB cameras:

```bash
cd data/curation_tool
pixi run build-curated-task \
  --task-id 2 \
  --video-keys \
    observation.images.cam_0 \
    observation.images.cam_1 \
    observation.images.cam_2 \
    observation.images.cam_3 \
  --video-workers 16 \
  --ffmpeg-threads 2 \
  --output-dir ../bitrobot_lerobot_subtask_datasets/outputs/curated_task_2_non_failure
```

The default accepted verdicts are `optimal`, `success`, and `ambiguous`, which
is the literal complement of `failure` in curation schema v4. To remove
ambiguous segments as well, pass:

```bash
--verdicts optimal success
```

Only source parquet/video shards referenced by selected episodes are
downloaded. Videos default to frame-accurate, pixel-lossless H.264 CRF 0
encoding. This preserves decoded source pixels; it is lossless compression,
not raw uncompressed video. `--video-mode reencode` opts into smaller, lossy
CRF 20 output. Stream-copy mode avoids encoding but may include neighboring
GOP frames and is therefore not frame-accurate. Omit `--video-keys` to preserve
all eight RGB/IR video features, or use `--skip-videos` for a metadata/data-only
build. The resolved source and curation commit SHAs plus every source
episode/frame range are recorded in
`meta/team_ramen_curation.json`.

### Build one continuous episode from a task sequence

`--task-sequence` joins accepted curation intervals in the requested order.
By default, the next interval must start on the frame immediately after the
previous interval ends (`--max-sequence-gap-frames 0`). This avoids silently
including an unlabelled recovery or unrelated action between subtasks. The
output uses a derived task ID, while every component task ID and frame range is
retained in `meta/team_ramen_curation.json` and the episode metadata.

For example, the following command selects only contiguous, all-`optimal`
insert (task 0) and tighten (task 3) pairs and stores them as derived task 8:

```bash
pixi run build-curated-task \
  --task-sequence 0 3 \
  --output-task-id 8 \
  --task-name "insert table leg to table base and rotate leg to tighten" \
  --verdicts optimal \
  --video-workers 32 \
  --ffmpeg-threads 1 \
  --output-dir ../bitrobot_lerobot_subtask_datasets/outputs/insert_and_tighten_optimal \
  --upload-repo-id Team-RAMEN/IROS2026_RAMEN_takada_insert_and_tighten_curated_optimal
```

Omitting `--output-task-id` chooses one greater than the largest source task
ID. A derived task ID is rejected if it collides with an existing source task.

Video clipping can run concurrently. For a 32-CPU-thread budget, use for
example:

```bash
--video-workers 16 --ffmpeg-threads 2
```

For many short clips, `--video-workers 32 --ffmpeg-threads 1` is another
32-thread configuration. `--video-workers` controls concurrent FFmpeg clip
processes, while `--ffmpeg-threads` controls the threads assigned to each
process; their product is the approximate encoding CPU budget. The defaults
remain one video worker and FFmpeg-managed per-process threading.

Clips are collected and concatenated in output episode order even when FFmpeg
jobs finish out of order. Source video downloads use at most eight I/O workers.
After the current camera's sources are ready, the next camera is prefetched in
the background while FFmpeg clips the current camera. This one-camera lookahead
keeps network transfers and CPU-heavy video processing overlapped without
changing the deterministic camera or episode output order.

Each encoded clip's frame count is checked against its data episode length.
Packed-video `from_timestamp` and `to_timestamp` values are derived from the
integer frame offset divided by FPS, rather than accumulated container
durations. This avoids millisecond rounding drift at episode boundaries in
TorchCodec.

### Upload the curated dataset to Hugging Face

After the local dataset passes structural validation, it can be uploaded
directly to a Hugging Face Dataset repo:

```bash
cd data/curation_tool
pixi run hf auth login

pixi run build-curated-task \
  --task-id 2 \
  --video-keys \
    observation.images.cam_0 \
    observation.images.cam_2 \
    observation.images.cam_3 \
  --video-workers 16 \
  --ffmpeg-threads 2 \
  --upload-repo-id Team-RAMEN/IROS2026_RAMEN_HARA_task_2_non_failure
```

New repos are private by default. Add `--public` only when intentionally
creating a public repo. Uploads to an existing repo are rejected unless
`--replace-remote` is passed. The large-folder uploader keeps resumable state
under the output directory and defaults to four upload workers.

Authentication is read from `HF_TOKEN` or the token saved by `hf auth login`;
tokens are not accepted as command-line arguments. For the full Japanese
workflow, safety behavior, and update command, see
[CURATED_DATASET_UPLOAD.md](CURATED_DATASET_UPLOAD.md).

If only the upload fails, resume from the already-built output without
re-downloading or re-encoding:

```bash
pixi run build-curated-task \
  --upload-existing \
  --output-dir ../bitrobot_lerobot_subtask_datasets/outputs/curated_task_2_non_failure \
  --upload-repo-id Team-RAMEN/IROS2026_RAMEN_HARA_task_2_non_failure \
  --replace-remote
```
