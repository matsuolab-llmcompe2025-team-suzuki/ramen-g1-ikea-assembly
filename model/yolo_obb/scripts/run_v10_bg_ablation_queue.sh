#!/usr/bin/env bash
# Issue #108 v10 bg augment ablation queue:
# v10_no_flat (v4_flat 抜き) → v10_no_zoomout (v8_zoomout 抜き) を sequential 実行
# 各 ~2.7h、合計 ~5.4h 想定。
set -eux
cd /home/ubuntu/work_dir/iros/iros_2026_ramen/model/yolo_obb

pixi run train-yolo --config model/yolo_obb/configs/m_lowaug_v10_no_flat_imgsz800.yaml 2>&1 | tee /tmp/v10_no_flat.log
pixi run train-yolo --config model/yolo_obb/configs/m_lowaug_v10_no_zoomout_imgsz800.yaml 2>&1 | tee /tmp/v10_no_zoomout.log
