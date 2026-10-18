#!/bin/bash
# Issue #108 v5 sequential train queue.
# 3 検証を順に GPU で train:
#   1. v5_material     (YOLO m_lowaug + workspace chroma-key material aug)
#   2. v5_downsample   (YOLO m_lowaug + move_to_table 50% downsample、bg augment なし)
#   3. rfdetr_v5_material (RF-DETR OBB small、oriented=True、DOTA v5_material_dota)
#
# 1 phase 失敗しても次へ進む (set +e)。各 phase の stdout/stderr は phase 別 log に、
# 全体の進捗は progress.log に記録。tmux/nohup で detach 起動想定。
#
# Usage (iros_2026_ramen root で):
#     bash model/yolo_obb/scripts/run_v5_train_queue.sh                        # foreground
#     tmux new -d -s v5_queue 'bash model/yolo_obb/scripts/run_v5_train_queue.sh'  # detach
#     nohup bash model/yolo_obb/scripts/run_v5_train_queue.sh > /tmp/v5_queue_stdout.log 2>&1 &

set +e  # 1 phase 失敗しても queue 継続

# repo root: script は model/yolo_obb/scripts/ 配下想定
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT" || { echo "[fatal] cd $REPO_ROOT failed"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/tmp/v5_queue_$TS
mkdir -p "$LOG_DIR"
PROGRESS="$LOG_DIR/progress.log"

log() {
    local msg="$*"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$PROGRESS"
}

run_phase() {
    local name="$1"
    local dir="$2"
    local cmd="$3"
    local phase_log="$LOG_DIR/${name}.log"
    log "=== BEGIN $name ==="
    log "cwd=$dir"
    log "cmd=$cmd"
    log "log=$phase_log"
    local start_epoch=$(date +%s)
    (cd "$dir" && eval "$cmd") > "$phase_log" 2>&1
    local rc=$?
    local end_epoch=$(date +%s)
    local dur=$((end_epoch - start_epoch))
    log "=== END $name exit=$rc duration=${dur}s ==="
    return $rc
}

log "V5 queue starting; repo_root=$REPO_ROOT log_dir=$LOG_DIR"
log "phases: v5_material -> v5_downsample -> rfdetr_v5_material"

run_phase "v5_material" "model/yolo_obb" \
    "pixi run train-yolo --config model/yolo_obb/configs/m_lowaug_v5_material.yaml --force"

run_phase "v5_downsample" "model/yolo_obb" \
    "pixi run train-yolo --config model/yolo_obb/configs/m_lowaug_v5_downsample.yaml --force"

run_phase "rfdetr_v5_material" "model/rf_detr_obb" \
    "pixi run train-rfdetr --config model/rf_detr_obb/configs/small_v5_material.yaml --force"

log "V5 queue done. progress=$PROGRESS"
log "個別 log: $LOG_DIR/{v5_material,v5_downsample,rfdetr_v5_material}.log"
