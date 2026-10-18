#!/usr/bin/env bash
# Issue #46 batch trainer: 全 config を順次実行、失敗しても次に進む。
# 使い方:
#   v1 (default):  nohup bash scripts/yolo_batch_runs.sh > logs/yolo_batch.log 2>&1 &
#   v2 (Issue #53): YOLO_SUFFIX=_v2 nohup bash scripts/yolo_batch_runs.sh > logs/yolo_batch_v2.log 2>&1 &
#
# YOLO_SUFFIX: 各 run_name の末尾に append される (yaml の run_name は上書きしない)。
# 既存 v1 runs (m_lowaug 等) を非破壊で残しつつ、v2 dataset で {m_lowaug_v2, ...} を並行生成する。

set -o pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

SUFFIX="${YOLO_SUFFIX:-}"
if [ -n "$SUFFIX" ]; then
  echo "[$(date -Iseconds)] YOLO_SUFFIX='${SUFFIX}' → run_name append + separate runs dir"
fi

# 順番: baseline 3 → s 実験 3 → m 実験 3 → l 実験 3 (3 × 4 マトリクス)
CONFIGS=(
  "s_baseline"
  "m_baseline"
  "l_baseline"
  "s_freeze10"
  "s_imgsz800"
  "s_lowaug"
  "m_freeze10"
  "m_imgsz800"
  "m_lowaug"
  "l_freeze10"
  "l_imgsz800"
  "l_lowaug"
)

for cfg in "${CONFIGS[@]}"; do
  RUN_NAME="${cfg}${SUFFIX}"
  RUN_LOG="logs/train_${RUN_NAME}.log"
  RUN_DIR="model/yolo_obb/runs/${RUN_NAME}"

  # 既に best.pt があれば skip (resume 対応、途中 kill 再起動時に既存 skip)
  if [ -f "${RUN_DIR}/weights/best.pt" ] && [ -f "${RUN_DIR}/summary.json" ]; then
    echo "[$(date -Iseconds)] SKIP ${RUN_NAME} (already done, summary.json exists)"
    continue
  fi

  echo "[$(date -Iseconds)] START ${RUN_NAME}"
  # 個別 log に 出力、pipeline log にも state 出力
  # YOLO training は model/yolo_obb sub-workspace 経由 (root から切り出し、SDK 依存無し env)
  # pixi task 内で cwd="../.." (repo root) に降りるので、--config は repo root 起点で渡す
  (cd model/yolo_obb && pixi run train-yolo --config "model/yolo_obb/configs/${cfg}.yaml" \
    ${SUFFIX:+--run-name-suffix "${SUFFIX}"}) \
    > "${RUN_LOG}" 2>&1
  ec=$?
  if [ $ec -eq 0 ]; then
    echo "[$(date -Iseconds)] DONE ${RUN_NAME}"
  else
    echo "[$(date -Iseconds)] FAILED ${RUN_NAME} (exit=$ec) — see ${RUN_LOG}"
  fi
done

echo
echo "[$(date -Iseconds)] ALL RUNS FINISHED"
echo "Summaries:"
for cfg in "${CONFIGS[@]}"; do
  RUN_NAME="${cfg}${SUFFIX}"
  SUM="model/yolo_obb/runs/${RUN_NAME}/summary.json"
  if [ -f "$SUM" ]; then
    map50=$(python3 -c "import json; d=json.load(open('$SUM')); print(d.get('metrics',{}).get('metrics/mAP50(B)', 'N/A'))")
    echo "  ${RUN_NAME}: val mAP@0.5 = ${map50}"
  else
    echo "  ${RUN_NAME}: (no summary)"
  fi
done
