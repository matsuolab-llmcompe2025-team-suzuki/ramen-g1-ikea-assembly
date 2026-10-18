#!/bin/bash
# Issue #108 v9 hsv range sequential train queue.
# v8_bg (mAP50/recall top) を base に hsv_v 強化 2 pattern を順に回す:
#   1. v9_hsv06 (hsv_v 0.6, sweet spot、object 保持)
#   2. v9_hsv08 (hsv_v 0.8, 境界域、白飛び hard sample)
#
# Usage: tmux new -d -s v9_queue 'bash model/yolo_obb/scripts/run_v9_hsv_queue.sh'

set +e
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT" || { echo "[fatal] cd $REPO_ROOT failed"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/tmp/v9_queue_$TS
mkdir -p "$LOG_DIR"
PROGRESS="$LOG_DIR/progress.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

run_phase() {
    local name="$1"; local config="$2"
    local phase_log="$LOG_DIR/${name}.log"
    log "=== BEGIN $name ==="
    log "config=$config log=$phase_log"
    local start_epoch=$(date +%s)
    (cd model/yolo_obb && pixi run train-yolo --config "$config" --force) > "$phase_log" 2>&1
    local rc=$?
    local end_epoch=$(date +%s)
    log "=== END $name exit=$rc duration=$((end_epoch - start_epoch))s ==="
    return $rc
}

log "V9 hsv queue starting; log_dir=$LOG_DIR"
log "phases: v9_hsv06 -> v9_hsv08"

run_phase "v9_hsv06"   "model/yolo_obb/configs/m_lowaug_v9_hsv06_imgsz800.yaml"
run_phase "v9_hsv08"   "model/yolo_obb/configs/m_lowaug_v9_hsv08_imgsz800.yaml"

log "V9 hsv queue done. progress=$PROGRESS"
