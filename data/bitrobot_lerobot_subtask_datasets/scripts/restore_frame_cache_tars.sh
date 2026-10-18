#!/usr/bin/env bash
# frame cache の tar を展開する (Issue #139)。HF の frame_cache_tars/ と frame_cache_v2_tars/ に
# restore.sh として同梱し、置かれた dir の tar を展開する。
#
# Usage: bash restore.sh <out_dir> [jobs]
#
#   frame_cache_tars/    (v0、デコードのみ)
#     → <out_dir>/<cam>/<chunk>/<file>/frame_XXXXXX.jpg
#   frame_cache_v2_tars/ (overlay / token / common)
#     → <out_dir>/{overlay,token,common}/<cam>/<chunk>/<file>/frame_XXXXXX_vN.jpg
#       - common/ の cam (token と overlay で同一の wrist) を overlay/ と token/ から symlink
#       - overlay/ と token/ に学習 view 名の別名 (head_left → cam_0 等) を張る (GR00T / ACT 用)
#       - _cache_meta.json を各 root にコピー (lerobot_frame_cache_patch が fps を読む)
#
# GR00T / ACT / DP は FRAME_CACHE_ROOT_OVERRIDE=<out_dir>/overlay、RAMEN-Ori は
# frame_cache_root_baked=<out_dir>/overlay (token mode は <out_dir>/token) を指す。
set -euo pipefail

OUT_DIR="${1:?usage: restore.sh <out_dir> [jobs]}"
JOBS="${2:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 学習 view 名 → source 名 (subtask_training.json:source_dataset.camera_map と同じ対応)
VIEW_ALIASES="observation.images.head_left:observation.images.cam_0
observation.images.left_wrist:observation.images.cam_2
observation.images.right_wrist:observation.images.cam_3"

# <tar_dir> の *.tar (<cam>_<chunk>_<file>.tar) を <out_root>/<cam>/<chunk>/ に並列展開
extract_dir() {
  local tar_dir="$1" out_root="$2"
  mkdir -p "$out_root"
  find "$tar_dir" -maxdepth 1 -name '*.tar' -print0 |
    xargs -0 -P "$JOBS" -I{} bash -c '
      set -euo pipefail
      tar_path="$1"; out_root="$2"
      base=$(basename "$tar_path" .tar)   # observation.images.cam_0_chunk-000_file-000
      file_seg=${base##*_}                # file-000
      rest=${base%_"$file_seg"}           # observation.images.cam_0_chunk-000
      chunk_seg=${rest##*_}               # chunk-000
      cam=${rest%_"$chunk_seg"}           # observation.images.cam_0
      mkdir -p "$out_root/$cam/$chunk_seg"
      tar -xf "$tar_path" -C "$out_root/$cam/$chunk_seg"
      echo "extracted: $(basename "$tar_path") -> $out_root/$cam/$chunk_seg/$file_seg/"
    ' _ {} "$out_root"
}

copy_meta() {
  if [ -f "$SCRIPT_DIR/_cache_meta.json" ]; then
    cp -f "$SCRIPT_DIR/_cache_meta.json" "$1/"
  fi
}

if [ -d "$SCRIPT_DIR/overlay" ] || [ -d "$SCRIPT_DIR/token" ] || [ -d "$SCRIPT_DIR/common" ]; then
  for kind in common overlay token; do
    if [ -d "$SCRIPT_DIR/$kind" ]; then
      extract_dir "$SCRIPT_DIR/$kind" "$OUT_DIR/$kind"
      copy_meta "$OUT_DIR/$kind"
    fi
  done
  # common の cam を overlay / token から参照 (token と overlay で同一の cam)
  if [ -d "$OUT_DIR/common" ]; then
    for cam_dir in "$OUT_DIR"/common/observation.images.*; do
      [ -d "$cam_dir" ] || continue
      cam=$(basename "$cam_dir")
      for kind in overlay token; do
        if [ -d "$OUT_DIR/$kind" ] && [ ! -e "$OUT_DIR/$kind/$cam" ] && [ ! -L "$OUT_DIR/$kind/$cam" ]; then
          ln -s "../common/$cam" "$OUT_DIR/$kind/$cam"
        fi
      done
    done
  fi
  # 学習 view 名の別名 (GR00T / ACT の view は head_left 等の名前で cache を探す)
  for kind in overlay token; do
    [ -d "$OUT_DIR/$kind" ] || continue
    echo "$VIEW_ALIASES" | while IFS=: read -r alias_name target; do
      if [ -e "$OUT_DIR/$kind/$target" ] && [ ! -e "$OUT_DIR/$kind/$alias_name" ] && [ ! -L "$OUT_DIR/$kind/$alias_name" ]; then
        ln -s "$target" "$OUT_DIR/$kind/$alias_name"
      fi
    done
  done
else
  extract_dir "$SCRIPT_DIR" "$OUT_DIR"
  copy_meta "$OUT_DIR"
fi
echo "done. restored frame cache in: $OUT_DIR"
