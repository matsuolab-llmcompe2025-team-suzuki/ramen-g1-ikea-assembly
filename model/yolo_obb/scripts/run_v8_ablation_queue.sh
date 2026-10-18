#!/bin/bash
# Issue #108 v8 ablation sequential train queue.
# 3 train を順に GPU で回す (v8 base -> v8_move51 -> v8_bg):
#   1. v8              (base、workspace 改善確認)
#   2. v8_move51       (v8 + move 51% で move 割合の追加効果検証)
#   3. v8_bg           (v8 + v4_flat bg augment 追加で bg 効果検証)
#
# ablation: v8 と 1 軸 diff × 2 pattern で move と bg の単独効果を独立分解。
# 1 phase 失敗しても次へ進む (set +e)。各 phase の stdout/stderr は phase 別 log に。
#
# Usage (iros_2026_ramen root で):
#     tmux new -d -s v8_queue 'bash model/yolo_obb/scripts/run_v8_ablation_queue.sh'
#     # 監視: tmux a -t v8_queue、progress: tail -f /tmp/v8_queue_*/progress.log

set +e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT" || { echo "[fatal] cd $REPO_ROOT failed"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/tmp/v8_queue_$TS
mkdir -p "$LOG_DIR"
PROGRESS="$LOG_DIR/progress.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"
}

run_phase() {
    local name="$1"
    local config="$2"
    local phase_log="$LOG_DIR/${name}.log"
    log "=== BEGIN $name ==="
    log "config=$config log=$phase_log"
    local start_epoch=$(date +%s)
    (cd model/yolo_obb && pixi run train-yolo --config "$config" --force) > "$phase_log" 2>&1
    local rc=$?
    local end_epoch=$(date +%s)
    local dur=$((end_epoch - start_epoch))
    log "=== END $name exit=$rc duration=${dur}s ==="
    return $rc
}

log "V8 ablation queue starting; repo_root=$REPO_ROOT log_dir=$LOG_DIR"
log "phases: v8 -> v8_move51 -> v8_bg"

run_phase "v8"          "model/yolo_obb/configs/m_lowaug_v8_zoomout_imgsz800.yaml"
run_phase "v8_move51"   "model/yolo_obb/configs/m_lowaug_v8_move51_imgsz800.yaml"
run_phase "v8_bg"       "model/yolo_obb/configs/m_lowaug_v8_bg_imgsz800.yaml"

log "V8 ablation queue done. progress=$PROGRESS"
log "個別 log: $LOG_DIR/{v8,v8_move51,v8_bg}.log"
