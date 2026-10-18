#!/usr/bin/env bash
# Issue #122 D-1: skill 単位 merged HF repo を作成する orchestrator。
#
# 動作:
#   1. subtask_training.json の {SUBTASK}.dataset_repos (chunk list) を snapshot_download
#   2. merge_curated_lerobot_datasets.py で 1 個の LeRobot v3 dataset に merge
#      (--allow-duplicate-task-indices)、mp4 は crf=18 再 encode (video-mode=reencode default)
#   3. upload_lerobot_dataset.py で merged_repo_template ({SUBTASK}_merged_v1) に push
#
# 5 Run pipeline とは独立 (chunk-level flow は無傷、これは skill-merged HF repo を
# 新規作成する片方向 job)。
#
# Usage:
#   SUBTASK=rotate_table_base bash prepare_merged_skill_dataset.sh
#   SUBTASK=move_table_base   bash prepare_merged_skill_dataset.sh
#   SUBTASK=pick_leg          bash prepare_merged_skill_dataset.sh   # 他 skill も同じ script
#
# Env:
#   SUBTASK              (required) subtask key in subtask_training.json
#   MERGED_LOCAL_DIR     (optional) merged output dir (default: outputs/merged_sources/${SUBTASK}_for_upload)
#   MERGED_VIDEO_MODE    (optional) ffmpeg mode (default: reencode = crf 18 visually lossless)
#   PRIVATE              (optional) HF repo private flag (default: true)
#   DRY_RUN              (optional) if true, print commands instead of running (default: false)
#   UPLOAD_LARGE_FOLDER  (optional) upload_lerobot_dataset --upload-large-folder (default: true、~50GB 向け)
#   FORCE_MERGE          (optional) delete existing MERGED_LOCAL_DIR before merge (default: false)
#
# 前提: HF_TOKEN が set 済 + Team-RAMEN org write 権限あり
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FEATURE_ROOT="$REPO_ROOT/model/subtask_policy_training"

if [[ -z "${SUBTASK:-}" ]]; then
  echo "ERROR: SUBTASK env is required (e.g. SUBTASK=rotate_table_base)" >&2
  exit 2
fi

VENV_DIR="${VENV_DIR:-$FEATURE_ROOT/.venv}"
if [[ ! -d "$VENV_DIR" && -d "$FEATURE_ROOT/.venv_lerobot060" ]]; then
  VENV_DIR="$FEATURE_ROOT/.venv_lerobot060"
fi
source "$VENV_DIR/bin/activate"

TRAIN_CONFIG="${TRAIN_CONFIG:-$FEATURE_ROOT/configs/subtask_training.json}"
# resolve_training_config を skill-merged 側の name (MERGED_DATASET_REPO_ID) 取得目的で呼ぶ。
# USE_MERGED_HF_REPO は explicit false を渡して chunk-level DATASET_REPO_IDS を取得
# (merge 元 = raw chunks なので)。
eval "$(USE_MERGED_HF_REPO=false SUBTASK="$SUBTASK" \
  python "$FEATURE_ROOT/scripts/resolve_training_config.py" \
  --config "$TRAIN_CONFIG" --format shell)"

if [[ -z "$MERGED_DATASET_REPO_ID" ]]; then
  echo "ERROR: MERGED_DATASET_REPO_ID is empty; subtask_training.json needs 'merged_repo_template'" >&2
  exit 2
fi

MERGED_LOCAL_DIR="${MERGED_LOCAL_DIR:-outputs/merged_sources/${SUBTASK}_for_upload}"
MERGED_VIDEO_MODE="${MERGED_VIDEO_MODE:-reencode}"
PRIVATE="${PRIVATE:-true}"
DRY_RUN="${DRY_RUN:-false}"
UPLOAD_LARGE_FOLDER="${UPLOAD_LARGE_FOLDER:-true}"
FORCE_MERGE="${FORCE_MERGE:-false}"

echo "=== prepare_merged_skill_dataset ==="
echo "subtask:               $SUBTASK"
echo "source chunks:         $DATASET_REPO_IDS"
echo "merged local dir:      $MERGED_LOCAL_DIR"
echo "merged HF repo target: $MERGED_DATASET_REPO_ID"
echo "video mode:            $MERGED_VIDEO_MODE (crf=18 if reencode)"
echo "private:               $PRIVATE"
echo "upload_large_folder:   $UPLOAD_LARGE_FOLDER"
echo "dry_run:               $DRY_RUN"

merge_cmd=(
  python "$FEATURE_ROOT/scripts/prepare_multi_repo_source.py"
  --repo-ids "$DATASET_REPO_IDS"
  --output-dir "$MERGED_LOCAL_DIR"
)
if [[ "$FORCE_MERGE" == "true" ]]; then
  merge_cmd+=(--force)
fi

upload_cmd=(
  python "$FEATURE_ROOT/scripts/upload_lerobot_dataset.py"
  --config "$TRAIN_CONFIG"
  --subtask "$SUBTASK"
  --repo-id "$MERGED_DATASET_REPO_ID"
  --root "$MERGED_LOCAL_DIR"
)
if [[ "$PRIVATE" == "true" ]]; then
  upload_cmd+=(--private)
fi
if [[ "$UPLOAD_LARGE_FOLDER" == "true" ]]; then
  upload_cmd+=(--upload-large-folder)
fi

if [[ "$DRY_RUN" == "true" ]]; then
  printf "merge:"
  printf " %q" "${merge_cmd[@]}"
  printf "\n"
  printf "upload:"
  printf " %q" "${upload_cmd[@]}"
  printf "\n"
  exit 0
fi

"${merge_cmd[@]}"
"${upload_cmd[@]}"

echo "[done] skill-merged repo published: https://huggingface.co/datasets/${MERGED_DATASET_REPO_ID}"
