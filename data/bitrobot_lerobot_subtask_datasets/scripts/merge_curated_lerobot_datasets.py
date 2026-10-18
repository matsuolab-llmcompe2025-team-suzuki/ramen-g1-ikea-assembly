#!/usr/bin/env python3
"""Merge N built LeRobot v3 curated task datasets into 1 combined dataset (Issue #115、Phase 2 prep)。

Input: N 個 の local built dataset directory (`build_curated_task_dataset.py` の output)。
Output: 1 個 の merged LeRobot v3 dataset directory。

- Data parquet: 各 source の episode を global episode_index に再採番、frame index も
  global に振り直し、`data_files_size_in_mb` に沿って merged chunk file に書き出し
- Videos: 各 camera について、source video file から生存 episode 分の clip を ffmpeg で
  extract → merged chunk file に concat (`video_files_size_in_mb` 遵守)。
  `--video-mode passthrough` (Issue #146) は切り出さず、元の mp4 を file ごとそのまま置いて
  (同じ filesystem なら hardlink) episode meta の chunk / file 番号だけ書き換える。
  全ての元 file が生存 episode で頭から最後まで隙間なく埋まっている時だけ使える (filter で
  episode を落とす merge は不可)。画素は元と同一、ffmpeg の encode が無いので数分で終わる
- Meta: tasks.parquet を union、episodes.parquet を rebuild、info.json / stats.json を rebuild
- Per-task filter (episode-level): flip_table (task 2) は Episode 内で frame_start 最大の
  1 seg のみ残す等の filter を config で指定、A1 の "data hygiene" 方針で merge stage で apply

CLI (data/curation_tool/ 側 pixi env で):

    cd data/curation_tool
    pixi run merge-curated \\
      --source-dir ../../outputs/curated_task_7_optimal \\
      --source-dir ../../outputs/curated_task_5_optimal \\
      --source-dir ../../outputs/curated_task_4_optimal \\
      --source-dir ../../outputs/curated_task_0_optimal \\
      --source-dir ../../outputs/curated_task_3_success \\
      --source-dir ../../outputs/curated_task_2_success \\
      --task-filter 2:last_per_source_episode \\
      --output-dir ../../outputs/mixed_v1 \\
      --force
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 既存 helper を再利用 (build_curated_task_dataset.py と同 pattern の import)
from build_subtask_datasets import (  # noqa: E402
    BYTES_PER_MB,
    DEFAULT_CHUNKS_SIZE,
    DEFAULT_DATA_FILE_SIZE_MB,
    DEFAULT_VIDEO_FILE_SIZE_MB,
    FileLocation,
    StatsAccumulator,
    clip_video,
    concat_video_files,
    data_path_from_episode,
    estimate_rows_size_bytes,
    get_video_frame_count,
    imagenet_video_stats,
    next_file_location,
    require_pyarrow,
    write_data_file,
    write_episode_meta,
    write_info,
    write_json,
)


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class SourceDataset:
    """1 個の source built dataset の meta を保持。"""

    root: Path
    info: dict[str, Any]
    tasks: list[dict[str, Any]]        # tasks.parquet rows
    episodes: list[dict[str, Any]]     # episodes.parquet rows (concat if multi file)
    video_keys: list[str]              # ['observation.images.cam_0', ...]
    fps: float
    features: dict[str, Any]


@dataclass
class FilterSpec:
    task_index: int
    filter_name: str


@dataclass
class MergePlan:
    """Merge の global schedule (episode global index + filter 適用後)。"""

    # 生存 episode の list、要素は {source_idx: int, src_ep_row: dict, output_episode_index: int}
    episodes: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Filters (per-task episode-level)
# =============================================================================


def _filter_last_per_source_episode(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Episode 内 (source_episode_index 別) で source_frame_start 最大の 1 seg のみ残す。

    flip_table (task 2) 用: 1 Episode に複数 flip segment がある中、最後の 1 個 (時系列的に
    後ろの seg) のみ学習に使う。
    """
    by_source_ep: dict[int, dict[str, Any]] = {}
    for ep in episodes:
        src_ep_idx = int(ep["source_episode_index"])
        cur = by_source_ep.get(src_ep_idx)
        if cur is None or int(ep["source_frame_start"]) > int(cur["source_frame_start"]):
            by_source_ep[src_ep_idx] = ep
    return [by_source_ep[k] for k in sorted(by_source_ep)]


def _filter_all(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(episodes)


FILTER_REGISTRY: dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = {
    "all": _filter_all,
    "last_per_source_episode": _filter_last_per_source_episode,
}


# =============================================================================
# Loading
# =============================================================================


def load_source(root: Path) -> SourceDataset:
    """1 個の source built dataset directory を読み込む。"""
    pa, pq = require_pyarrow()
    info_path = root / "meta" / "info.json"
    with info_path.open() as f:
        info = json.load(f)

    tasks_path = root / "meta" / "tasks.parquet"
    tasks = pq.read_table(tasks_path).to_pylist()

    # episodes.parquet は meta/episodes/chunk-000/file-XXX.parquet の複数 file
    episodes: list[dict[str, Any]] = []
    episodes_dir = root / "meta" / "episodes"
    if episodes_dir.exists():
        for chunk_dir in sorted(episodes_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for ep_file in sorted(chunk_dir.glob("file-*.parquet")):
                episodes.extend(pq.read_table(ep_file).to_pylist())

    features = info.get("features", {})
    video_keys = [
        k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"
    ]
    fps = float(info.get("fps", 30.0))

    return SourceDataset(
        root=root,
        info=info,
        tasks=tasks,
        episodes=episodes,
        video_keys=sorted(video_keys),
        fps=fps,
        features=features,
    )


# =============================================================================
# Validation
# =============================================================================


def _video_signature(feature: dict[str, Any], strict: bool) -> tuple[Any, ...]:
    """動画 feature のうち source 間で揃っている必要がある部分。

    merge 後の info.json は先頭 source の features をそのまま書くので、ここがずれた source を
    通すと宣言と中身が食い違う。解像度と fps は常に、codec / pix_fmt は元 file をそのまま置く
    passthrough のときだけ見る (作り直す側は libx264 / yuv420p に揃えるので違っていてよい)。
    """
    info = feature.get("info") or {}
    sig: tuple[Any, ...] = (tuple(feature.get("shape") or ()), info.get("video.fps"))
    if strict:
        sig += (info.get("video.codec"), info.get("video.pix_fmt"))
    return sig


def validate_schemas(
    sources: list[SourceDataset],
    allow_duplicate_task_indices: bool = False,
    video_mode: str = "reencode",
) -> None:
    """全 source が同 fps / features / video_keys か確認、mismatch なら error。

    Args:
        sources: source dataset list
        video_mode: passthrough なら動画の codec / pix_fmt も揃っていることを要求する
            (元 file をそのまま置くので、作り直しのように揃えられない)
        allow_duplicate_task_indices: True で同 task_id を持つ複数 source を許容
            (Issue #122: task 5 は 8 chunks 全て task_id=5 で concat 対象)。
            merge_tasks() は元々 dedup 済 (line 647)、build_merge_plan は per-source
            filter が同 task_id で一貫すれば安全 (同じ filter が全 chunks に適用される)。
            False (default) は従来通り「各 source は unique な 1 task_id 所有」を強制。
    """
    if not sources:
        raise ValueError("no source datasets provided")

    ref = sources[0]
    ref_fps = ref.fps
    ref_video_keys = tuple(ref.video_keys)
    ref_features_names = set(ref.features.keys())

    for s in sources[1:]:
        if s.fps != ref_fps:
            raise ValueError(
                f"fps mismatch: {s.root} has {s.fps}, ref {ref.root} has {ref_fps}"
            )
        if tuple(s.video_keys) != ref_video_keys:
            raise ValueError(
                f"video_keys mismatch: {s.root} has {s.video_keys}, ref {ref.video_keys}"
            )
        if set(s.features.keys()) != ref_features_names:
            diff_add = set(s.features.keys()) - ref_features_names
            diff_rem = ref_features_names - set(s.features.keys())
            raise ValueError(
                f"features mismatch: {s.root} added={diff_add}, missing={diff_rem}"
            )
        for key in ref_video_keys:
            got = _video_signature(s.features[key], video_mode == "passthrough")
            want = _video_signature(ref.features[key], video_mode == "passthrough")
            if got != want:
                raise ValueError(
                    f"video feature mismatch ({key}): {s.root} has {got}, ref {ref.root} has {want} "
                    f"(shape, fps" + (", codec, pix_fmt)" if video_mode == "passthrough" else ")")
                )

    # task_index 一意性チェック (allow_duplicate_task_indices=True で skip)。
    # Issue #122 で multi-chunk 同 task concat 対応のため flag 化。
    if not allow_duplicate_task_indices:
        all_task_indices: list[int] = []
        for s in sources:
            for t in s.tasks:
                all_task_indices.append(int(t["task_index"]))
        counts = Counter(all_task_indices)
        dup = [ti for ti, c in counts.items() if c > 1]
        if dup:
            raise ValueError(
                f"task_index duplicated across sources: {dup} "
                f"(each source must own 1 unique task_id、または --allow-duplicate-task-indices を渡す)"
            )


# =============================================================================
# Merge planning
# =============================================================================


def parse_task_filters(specs: list[str]) -> dict[int, str]:
    """`--task-filter 2:last_per_source_episode` の複数指定を dict に変換。"""
    result: dict[int, str] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(
                f"--task-filter must be 'TASK_ID:FILTER_NAME', got {spec!r}"
            )
        task_str, filter_name = spec.split(":", 1)
        task_id = int(task_str)
        if filter_name not in FILTER_REGISTRY:
            valid = ", ".join(FILTER_REGISTRY.keys())
            raise ValueError(
                f"unknown filter {filter_name!r} for task {task_id}; valid: {valid}"
            )
        if task_id in result:
            raise ValueError(f"task {task_id} filter specified more than once")
        result[task_id] = filter_name
    return result


def build_merge_plan(
    sources: list[SourceDataset],
    filters_by_task: dict[int, str],
) -> MergePlan:
    """Filter を適用し global episode_index を確定。"""
    plan = MergePlan()
    output_ep_idx = 0
    for src_idx, s in enumerate(sources):
        # Source は 1 task_id を持つ、その task_id の filter を適用
        task_id = int(s.tasks[0]["task_index"]) if s.tasks else None
        filter_name = filters_by_task.get(task_id, "all")
        filter_fn = FILTER_REGISTRY[filter_name]
        filtered = filter_fn(s.episodes)
        for src_ep_row in filtered:
            plan.episodes.append(
                {
                    "source_idx": src_idx,
                    "source_task_id": task_id,
                    "source_episode_row": src_ep_row,
                    "output_episode_index": output_ep_idx,
                }
            )
            output_ep_idx += 1
    return plan


# =============================================================================
# Data merge (parquet reindex + concat)
# =============================================================================


def merge_data(
    sources: list[SourceDataset],
    plan: MergePlan,
    output_dir: Path,
    data_files_size_mb: int,
    chunks_size: int,
) -> tuple[list[dict[str, Any]], int, dict[str, StatsAccumulator]]:
    """全 source の data parquet を読み、episode/frame index を再割り振り + concat。

    - 各 output episode の rows: source の frame 順を維持、episode_index/index は global に採番
    - `data_files_size_mb` に応じて output data files に batch 書き出し
    - 非-video feature の StatsAccumulator を並行 update

    Returns:
        output_episode_meta: 各生存 episode の meta (episode_index / data location / length / etc.)、
            後続 B-5 (meta rebuild) と B-6 (video re-encode) で使う
        total_frames: 全 output frame 数 (grand total)
        stats: 非-video feature の accumulator dict (B-5 で stats.json 化)
    """
    pa, pq = require_pyarrow()

    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    # LeRobot iloc 互換のため source task_index を 0-based contiguous に remap
    # (merge_tasks 側と同じ deterministic remap、build_task_index_remap 参照、Issue #122)
    task_index_remap = build_task_index_remap(sources)

    # 非-video feature に対して StatsAccumulator を用意 (video は data parquet に無い)
    stats: dict[str, StatsAccumulator] = {}
    ref_features = sources[0].features
    for name, feat in ref_features.items():
        if not (isinstance(feat, dict) and feat.get("dtype") == "video"):
            stats[name] = StatsAccumulator()

    # source data parquet の in-memory cache (1 file 分だけ持つ、次の file 読む時に破棄)
    cache_key: tuple[int, str] | None = None
    cache_rows_by_ep: dict[int, list[dict[str, Any]]] = {}

    output_episode_meta: list[dict[str, Any]] = []
    global_index = 0
    data_target_bytes = data_files_size_mb * BYTES_PER_MB
    data_buffer: list[dict[str, Any]] = []
    data_buffer_bytes = 0
    location = FileLocation(chunk_index=0, file_index=0)

    for plan_entry in plan.episodes:
        src_idx = plan_entry["source_idx"]
        source = sources[src_idx]
        src_ep_row = plan_entry["source_episode_row"]
        output_ep_idx = plan_entry["output_episode_index"]

        # source data file を lazy load (同 file 連続なら cache hit)
        source_data_path = data_path_from_episode(src_ep_row)
        cur_key = (src_idx, source_data_path)
        if cur_key != cache_key:
            table = pq.read_table(source.root / source_data_path)
            cache_rows_by_ep.clear()
            for row in table.to_pylist():
                cache_rows_by_ep.setdefault(int(row["episode_index"]), []).append(row)
            cache_key = cur_key

        src_ep_local_idx = int(src_ep_row["episode_index"])
        source_rows = cache_rows_by_ep.get(src_ep_local_idx, [])
        if not source_rows:
            raise RuntimeError(
                f"no data rows for source episode {src_ep_local_idx} "
                f"in {source.root / source_data_path}"
            )

        # Rewrite each row with new episode/global index
        output_rows: list[dict[str, Any]] = []
        for frame_index, source_row in enumerate(source_rows):
            output_row = {
                k: v for k, v in source_row.items() if k in source.features
            }
            output_row["timestamp"] = frame_index / source.fps
            output_row["frame_index"] = frame_index
            output_row["episode_index"] = output_ep_idx
            output_row["index"] = global_index + frame_index
            # task_index は 0-based contiguous に remap (Issue #122 fix)。
            # LeRobot reader は iloc[task_idx] で task 名を引くので、tasks.parquet の
            # row order と data の task_index を揃える必要がある。
            if "task_index" in source_row:
                output_row["task_index"] = task_index_remap[int(source_row["task_index"])]
            output_rows.append(output_row)

        # Flush buffer if would exceed target size
        segment_bytes = estimate_rows_size_bytes(pa, output_rows)
        if data_buffer and data_buffer_bytes + segment_bytes > data_target_bytes:
            write_data_file(pa, pq, output_dir, location, data_buffer)
            location = next_file_location(location, chunks_size)
            data_buffer = []
            data_buffer_bytes = 0
        segment_location = location

        # Update stats
        for out_row in output_rows:
            for feat_name, acc in stats.items():
                if feat_name in out_row:
                    acc.update(out_row[feat_name])

        data_buffer.extend(output_rows)
        data_buffer_bytes += segment_bytes

        length = len(output_rows)
        dataset_from = global_index
        global_index += length

        # task_name (source tasks.parquet から取得、通常 1 row)
        task_name = None
        if source.tasks:
            task_name = source.tasks[0].get("__index_level_0__")
        if task_name is None:
            task_name = f"task_{plan_entry['source_task_id']}"

        # Assemble output episode meta。B-6 (video merge) で videos/{key}/... field を追加、
        # B-5 (episodes.parquet write) で最終形になる
        out_meta = {
            "episode_index": output_ep_idx,
            "tasks": [task_name],
            "length": length,
            "data/chunk_index": segment_location.chunk_index,
            "data/file_index": segment_location.file_index,
            "dataset_from_index": dataset_from,
            "dataset_to_index": global_index,
            "source_dataset_root": str(source.root),
            "source_task_id": plan_entry["source_task_id"],
        }
        # source 側の provenance / video path 系 fields を preserve (B-6 で video 再抽出時に使う)
        for k in [
            "source_episode_index",
            "source_frame_start",
            "source_frame_end",
            "source_frame_start_sec",
            "source_frame_end_sec",
            "source_task_index",
            "curation_verdict",
            "curation_failure_category",
            "curation_reviewer",
            "curation_reviewed_at",
            "curation_schema_version",
        ]:
            if k in src_ep_row:
                out_meta[k] = src_ep_row[k]
        # videos/{key}/... fields は B-6 で置換される、参照用に source 側の値を一旦持っておく
        for k in src_ep_row:
            if k.startswith("videos/"):
                out_meta[f"__source_{k}"] = src_ep_row[k]

        output_episode_meta.append(out_meta)

    if data_buffer:
        write_data_file(pa, pq, output_dir, location, data_buffer)

    return output_episode_meta, global_index, stats


# =============================================================================
# Video re-encode (per-episode extract + repack)
# =============================================================================


def _source_video_path(
    source_root: Path,
    video_key: str,
    src_chunk_index: int,
    src_file_index: int,
) -> Path:
    return (
        source_root
        / "videos"
        / video_key
        / f"chunk-{src_chunk_index:03d}"
        / f"file-{src_file_index:03d}.mp4"
    )


def _output_video_path(
    output_dir: Path, video_key: str, location: FileLocation
) -> Path:
    return (
        output_dir
        / "videos"
        / video_key
        / f"chunk-{location.chunk_index:03d}"
        / f"file-{location.file_index:03d}.mp4"
    )


def _plan_clips_for_video_key(
    output_meta: list[dict[str, Any]],
    video_key: str,
) -> list[dict[str, Any]]:
    """各 output episode に対して source video 位置 + 抽出時刻を求める plan を作る。

    output_meta には B-4 で `__source_videos/{video_key}/chunk_index`, `file_index`,
    `from_timestamp`, `to_timestamp` が入っている前提。plan の要素:
      - episode_index (output global)
      - source_video_path
      - clip_start_sec, clip_end_sec
      - length (frame count)
    """
    plans = []
    for ep in output_meta:
        src_root = Path(ep["source_dataset_root"])
        src_chunk = int(ep[f"__source_videos/{video_key}/chunk_index"])
        src_file = int(ep[f"__source_videos/{video_key}/file_index"])
        from_ts = float(ep[f"__source_videos/{video_key}/from_timestamp"])
        to_ts = float(ep[f"__source_videos/{video_key}/to_timestamp"])
        plans.append(
            {
                "episode_index": int(ep["episode_index"]),
                "source_video_path": _source_video_path(
                    src_root, video_key, src_chunk, src_file
                ),
                "clip_start_sec": from_ts,
                "clip_end_sec": to_ts,
                "length": int(ep["length"]),
            }
        )
    return plans


def _batch_clips_to_output_files(
    clip_plans: list[dict[str, Any]],
    clip_sizes: list[int],
    clip_frame_counts: list[int],
    video_files_size_mb: int,
    chunks_size: int,
) -> list[dict[str, Any]]:
    """各 clip を どの output chunk file に振り分けるかの batching plan を返す。

    Pure logic (no IO)。unit test しやすくするため logic 分離。
    Returns list of dict with:
      - episode_index (output global)
      - output_chunk_index, output_file_index
      - from_timestamp_frames, to_timestamp_frames (batch 内での frame offset)
    """
    if len(clip_plans) != len(clip_sizes) or len(clip_plans) != len(clip_frame_counts):
        raise ValueError("clip_plans / clip_sizes / clip_frame_counts length mismatch")

    target_bytes = video_files_size_mb * BYTES_PER_MB
    location = FileLocation(chunk_index=0, file_index=0)
    result = []
    batch_size = 0
    batch_frame_count = 0
    for plan, size, frame_count in zip(clip_plans, clip_sizes, clip_frame_counts):
        if batch_size > 0 and batch_size + size > target_bytes:
            location = next_file_location(location, chunks_size)
            batch_size = 0
            batch_frame_count = 0
        result.append(
            {
                "episode_index": plan["episode_index"],
                "output_chunk_index": location.chunk_index,
                "output_file_index": location.file_index,
                "from_frame_count": batch_frame_count,
                "to_frame_count": batch_frame_count + frame_count,
            }
        )
        batch_size += size
        batch_frame_count += frame_count
    return result


# =============================================================================
# Video passthrough (元 file をそのまま置き、episode meta の位置だけ書き換える、Issue #146)
# =============================================================================


def _source_file_key(ep: dict[str, Any], video_key: str) -> tuple[str, int, int]:
    """episode が参照する元 file の識別子 (source root, chunk_index, file_index)。"""
    return (
        str(ep["source_dataset_root"]),
        int(ep[f"__source_videos/{video_key}/chunk_index"]),
        int(ep[f"__source_videos/{video_key}/file_index"]),
    )


def _timestamp_to_frame(timestamp: float, fps: float) -> int | None:
    """timestamp を frame 番号にする。整数から 1e-3 frame 以上ずれていれば None。"""
    frame = timestamp * fps
    rounded = round(frame)
    return rounded if abs(frame - rounded) < 1e-3 else None


def _plan_passthrough_files(
    output_meta: list[dict[str, Any]],
    video_key: str,
    file_frame_counts: dict[tuple[str, int, int], int],
    fps: float,
    chunks_size: int,
) -> dict[tuple[str, int, int], FileLocation]:
    """元 file → 出力の位置 (file は触らず、確認と番号付けだけ)。

    各元 file が生存 episode で frame 0 から最後まで隙間・重なりなく埋まり、各 episode の区間が
    length と一致する (上流由来の 1 frame 差は警告して許容) ことを確かめる。満たさない file が
    1 つでもあれば全部を列挙して ValueError (filter で episode を落とした merge や、区間の壊れた
    source を passthrough で通さない)。番号は出力 episode の順に、初めて出てきた file から振る。
    """
    by_file: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for ep in output_meta:
        by_file.setdefault(_source_file_key(ep, video_key), []).append(ep)

    problems: list[str] = []
    for key, eps in by_file.items():
        spans: list[tuple[int, int]] = []
        for ep in eps:
            start = _timestamp_to_frame(float(ep[f"__source_videos/{video_key}/from_timestamp"]), fps)
            end = _timestamp_to_frame(float(ep[f"__source_videos/{video_key}/to_timestamp"]), fps)
            # BitRobot 上流の meta / container の 1 frame 差は chunk を作る側 (build_subtask_datasets.py) も
            # 作り直す側 (下の materialize_clip) も許容している。timestamp は動画の実際の区切りなので、
            # ここで弾くと merge 全体が止まるだけで得るものがない (区間の連続は下の cursor で別に確かめる)
            if start is None or end is None or abs((end - start) - int(ep["length"])) > 1:
                problems.append(
                    f"{key}: episode {ep['episode_index']} の区間 "
                    f"[{ep[f'__source_videos/{video_key}/from_timestamp']}, "
                    f"{ep[f'__source_videos/{video_key}/to_timestamp']}) s が length {ep['length']} と合わない"
                )
                break
            if end - start != int(ep["length"]):
                print(
                    f"  [video] {video_key} episode {ep['episode_index']}: 動画 {end - start} frame / "
                    f"meta length {ep['length']} (1 frame 差を許容、BitRobot 上流の不整合)",
                    flush=True,
                )
            spans.append((start, end))
        else:
            cursor = 0
            for start, end in sorted(spans):
                if start != cursor:
                    problems.append(f"{key}: frame {cursor} の次の episode が frame {start} から始まる (隙間 / 重なり)")
                    break
                cursor = end
            else:
                if cursor != file_frame_counts[key]:
                    problems.append(f"{key}: episode は frame {cursor} まで、file は {file_frame_counts[key]} frame")
    if problems:
        raise ValueError(
            f"passthrough できない元 file が {len(problems)} 個 ({video_key})。"
            "episode を落とす merge や区間の合わない source は --video-mode reencode を使う:\n  "
            + "\n  ".join(problems)
        )

    mapping: dict[tuple[str, int, int], FileLocation] = {}
    location = FileLocation(chunk_index=0, file_index=0)
    for key in by_file:
        mapping[key] = location
        location = next_file_location(location, chunks_size)
    return mapping


def _link_or_copy(src: Path, dst: Path) -> bool:
    """dst に src を置く。同じ filesystem なら hardlink (True)、できなければ copy (False)。

    dst が既に在る (EEXIST) のは呼び出し側の間違いなので、黙って上書きせずそのまま投げる。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    real = src.resolve()  # HF cache の symlink は実体へ
    try:
        os.link(real, dst)
        return True
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.EOPNOTSUPP):
            raise
        shutil.copy2(real, dst)
        return False


def _write_passthrough_videos(
    *,
    output_meta: list[dict[str, Any]],
    output_dir: Path,
    video_keys: list[str],
    fps: float,
    chunks_size: int,
    max_workers: int,
) -> None:
    """全 camera の plan (= 前提の確認) を済ませてから file を置き、output_meta を in-place 更新。

    1 camera でも前提を満たさなければ、何も置かずに止まる。timestamp は元 file の中の位置のまま。
    """
    plans: dict[str, tuple[dict[tuple[str, int, int], Path], dict[tuple[str, int, int], FileLocation]]] = {}
    for video_key in video_keys:
        keys = list(dict.fromkeys(_source_file_key(ep, video_key) for ep in output_meta))
        paths = {k: _source_video_path(Path(k[0]), video_key, k[1], k[2]) for k in keys}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            counts = dict(zip(keys, ex.map(get_video_frame_count, [paths[k] for k in keys])))
        plans[video_key] = (paths, _plan_passthrough_files(output_meta, video_key, counts, fps, chunks_size))

    linked = copied = 0
    for video_key, (paths, mapping) in plans.items():
        for key, location in mapping.items():
            if _link_or_copy(paths[key], _output_video_path(output_dir, video_key, location)):
                linked += 1
            else:
                copied += 1
        for ep in output_meta:
            location = mapping[_source_file_key(ep, video_key)]
            ep[f"videos/{video_key}/chunk_index"] = location.chunk_index
            ep[f"videos/{video_key}/file_index"] = location.file_index
            ep[f"videos/{video_key}/from_timestamp"] = ep[f"__source_videos/{video_key}/from_timestamp"]
            ep[f"videos/{video_key}/to_timestamp"] = ep[f"__source_videos/{video_key}/to_timestamp"]
        print(f"  [video] {video_key} passthrough ({len(mapping)} files)", flush=True)
    print(f"  [video] passthrough: hardlink {linked} / copy {copied}", flush=True)


def write_merged_videos(
    *,
    sources: list[SourceDataset],
    output_meta: list[dict[str, Any]],
    output_dir: Path,
    video_keys: list[str],
    video_mode: str,
    video_files_size_mb: int,
    chunks_size: int,
    fps: float,
    ffmpeg_threads: int | None = None,
    max_video_workers: int = 4,
    max_clip_workers: int = 4,
) -> None:
    """各 camera について source video から生存 episode 分を extract + concat + repack。
    video_mode="passthrough" は切り出さずに元 file をそのまま置く (`_write_passthrough_videos`)。

    output_meta を in-place 更新: `videos/{video_key}/chunk_index`, `file_index`,
    `from_timestamp`, `to_timestamp` の new pack 位置。

    fake source (video 無し) の場合は video_keys=[] で全 loop skip。

    max_video_workers (Issue #122): camera 単位の並列度 (default 4)。camera loop の
    body は per-video-key isolated (temp_dir / output_path / output_meta の書き込み
    column が全 camera で disjoint) なので ThreadPool で並列化しても race せず、
    24-core H100 環境で ~15min → ~4min (実測見込み) に短縮。1 で従来 sequential。

    ffmpeg_threads (Issue #122): 各 ffmpeg process の thread 数。None (default) で
    auto (parallel 時 `cpu_count // (max_video_workers * max_clip_workers)`、
    sequential 時 0)。over-subscription 回避のため合計 concurrency ×
    thread/process = cpu_count に合わせる。0 を明示 override すると旧動作 (全 core
    per ffmpeg) に戻せる。

    max_clip_workers (Issue #122): camera 内の clip 抽出並列度 (default 4)。camera loop
    body の `for plan in plans:` を ThreadPool 化、camera 並列 (max_video_workers) と
    掛け合わせて合計 max_video_workers × max_clip_workers 個の ffmpeg concurrent。
    default 4×4 = 16 で 24-core 上に spread、期待 ~15min → ~5-6min (~2.5x 追加短縮)。
    ThreadPool.map で clip 順序保証、tmp file path も per-plan で unique なので race
    なし (episode_index を key に file name 生成)。
    """
    _ = sources  # sources は provenance にのみ使う、ここでは output_meta 経由で参照
    if not video_keys:
        return

    if video_mode == "passthrough":
        _write_passthrough_videos(
            output_meta=output_meta,
            output_dir=output_dir,
            video_keys=video_keys,
            fps=fps,
            chunks_size=chunks_size,
            max_workers=max(1, max_video_workers * max_clip_workers),
        )
        return

    import shutil as _shutil  # local import: script 冒頭でも import 済だが明示

    # ffmpeg_threads auto-compute: 合計 concurrency (video × clip workers) で cpu を割り、
    # over-subscription 回避。sequential (=1x1) 時は 0 (=全 core per ffmpeg) 維持。
    if ffmpeg_threads is None:
        import os as _os
        total_conc = max_video_workers * max_clip_workers
        if total_conc > 1:
            ffmpeg_threads = max(1, (_os.cpu_count() or 4) // total_conc)
            print(
                f"  [video] auto ffmpeg_threads={ffmpeg_threads} "
                f"(cpu={_os.cpu_count()} / (video={max_video_workers} × clip={max_clip_workers}) = {total_conc} concurrent)",
                flush=True,
            )
        else:
            ffmpeg_threads = 0  # 0 = ffmpeg default (all cores)

    temp_root = output_dir / ".tmp_video_clips"
    if temp_root.exists():
        _shutil.rmtree(temp_root)

    # output_meta の episode_index → dict 変換は camera loop の外で 1 回だけ作れば
    # 十分 (loop 内で毎回 build してた元 code を最適化、並列時の重複コスト回避)。
    meta_by_ep = {int(ep["episode_index"]): ep for ep in output_meta}

    def _process_video_key(video_key: str) -> None:
        safe_key = video_key.replace("/", "__")
        temp_dir = temp_root / safe_key
        temp_dir.mkdir(parents=True, exist_ok=True)

        plans = _plan_clips_for_video_key(output_meta, video_key)

        # 1) 各 clip を temp dir に抽出、sizes/frame_counts を計測。
        # Issue #122: camera 内の clip extract を max_clip_workers 並列化、
        # camera 並列と掛け合わせて 4×4=16 concurrent ffmpeg (default) で -~2.5x 追加短縮。
        # ThreadPool.map で clip 順序保証、tmp file path は episode_index で unique = race なし。
        def _extract_one(plan: dict[str, Any]) -> tuple[Path, int, int]:
            clip_path = temp_dir / f"clip-{plan['episode_index']:06d}.mp4"
            clip_video(
                plan["source_video_path"],
                clip_path,
                plan["clip_start_sec"],
                plan["clip_end_sec"],
                video_mode,
                ffmpeg_threads=ffmpeg_threads,
            )
            size = clip_path.stat().st_size
            frame_count = get_video_frame_count(clip_path)
            # Issue #122: ffmpeg cut boundary の GOP alignment で ±1 frame drift が起きる
            # ケースを許容 (特に既 re-encode 済 mp4 を source にした二重 encode パス、
            # 実測: task_5_merged を combined merge の入力にしたとき ep 452 で 70 vs 69)。
            # 余剰 frame は concat 末尾に置き去りになるだけで、meta の from_timestamp が
            # cumulative actual frame_count で recompute されるため次 episode の boundary
            # に影響しない (無害、disk 少しロス)。lack (actual < planned) は data 消失で
            # 危険なので tolerance 外は従来通り raise。
            drift = frame_count - plan["length"]
            if abs(drift) > 1:
                raise RuntimeError(
                    f"clip frame count {frame_count} mismatch "
                    f"episode length {plan['length']} for {video_key} "
                    f"episode {plan['episode_index']} "
                    f"(drift={drift}, tolerance=±1)"
                )
            if drift != 0:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    f"clip frame drift tolerated: actual={frame_count} planned={plan['length']} "
                    f"(diff={drift}) for {video_key} ep {plan['episode_index']}"
                )
            return clip_path, size, frame_count

        if max_clip_workers > 1:
            from concurrent.futures import ThreadPoolExecutor as _TPE
            with _TPE(max_workers=max_clip_workers) as _ex:
                results = list(_ex.map(_extract_one, plans))  # map で順序保証
        else:
            results = [_extract_one(p) for p in plans]

        clip_paths = [r[0] for r in results]
        clip_sizes = [r[1] for r in results]
        clip_frame_counts = [r[2] for r in results]

        # 2) Batching: 各 clip を output chunk file に振り分け (pure logic)
        batches = _batch_clips_to_output_files(
            plans, clip_sizes, clip_frame_counts, video_files_size_mb, chunks_size
        )

        # 3) 同 (chunk_index, file_index) の clip 群を concat → output に書き出し
        groups: dict[tuple[int, int], list[tuple[Path, dict]]] = {}
        for clip_path, batch in zip(clip_paths, batches):
            key = (batch["output_chunk_index"], batch["output_file_index"])
            groups.setdefault(key, []).append((clip_path, batch))

        for (chunk_idx, file_idx), items in sorted(groups.items()):
            output_path = _output_video_path(
                output_dir, video_key, FileLocation(chunk_idx, file_idx)
            )
            paths = [p for p, _ in items]
            concat_video_files(
                paths, output_path, fps, video_mode=video_mode,
                ffmpeg_threads=ffmpeg_threads,
            )

        # 4) output_meta を in-place update (video_key 毎に別 column なので camera
        # 並列でも write race しない、CPython GIL 下で dict[key]=value は atomic)
        for batch in batches:
            ep = meta_by_ep[batch["episode_index"]]
            ep[f"videos/{video_key}/chunk_index"] = batch["output_chunk_index"]
            ep[f"videos/{video_key}/file_index"] = batch["output_file_index"]
            ep[f"videos/{video_key}/from_timestamp"] = batch["from_frame_count"] / fps
            ep[f"videos/{video_key}/to_timestamp"] = batch["to_frame_count"] / fps

        print(f"  [video] {video_key} done ({len(batches)} clips)", flush=True)

    n_workers = min(max_video_workers, len(video_keys)) if max_video_workers > 1 else 1
    if n_workers <= 1:
        # sequential fallback (debug / ffmpeg over-subscription 回避時)
        for video_key in video_keys:
            _process_video_key(video_key)
    else:
        from concurrent.futures import ThreadPoolExecutor
        print(
            f"  [video] parallel processing {len(video_keys)} cameras with "
            f"{n_workers} workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # list() で全 future の完了を待ち、内部例外があれば再 raise (fail-fast)
            list(ex.map(_process_video_key, video_keys))

    _shutil.rmtree(temp_root, ignore_errors=True)


# =============================================================================
# Meta rebuild (tasks / episodes / info / stats / provenance)
# =============================================================================


# info.json の source からコピーする既知 key (write_info の known_info_keys 準拠)
_KNOWN_INFO_KEYS = {
    "codebase_version",
    "fps",
    "features",
    "robot_type",
    "tools",
}


def build_task_index_remap(sources: list[SourceDataset]) -> dict[int, int]:
    """Union of source task_indices → 0-based contiguous remap (sorted by original)。

    LeRobot の dataset reader は ``self._meta.tasks.iloc[task_idx]`` (positional
    indexer) で task 名を引く。task 5+7 のような sparse な source index をそのまま
    残すと merged tasks.parquet (2 rows) に対し ``iloc[5]`` / ``iloc[7]`` で
    ``IndexError: single positional indexer is out-of-bounds`` になり train が
    step 0 で落ちる (Issue #122 2 件目の pipeline bug)。

    対処: merge 時に source task_index を 0-based contiguous に remap し、
    tasks.parquet の row order と data parquet の task_index 列を揃える。
    (例) source task 5, 7 → {5:0, 7:1}、tasks.parquet 0 行目 = task 5 の name、
    1 行目 = task 7 の name、data task_index も 0 or 1 に書き換わる。

    Determinism: sorted(original) 順で新 index を採番するため、同じ sources set に
    対して常に同じ remap を返す (merge_tasks と merge_data が独立に compute しても
    一致)。
    """
    original: set[int] = set()
    for s in sources:
        for row in s.tasks:
            original.add(int(row["task_index"]))
    return {orig: new for new, orig in enumerate(sorted(original))}


def merge_tasks(sources: list[SourceDataset]) -> list[dict[str, Any]]:
    """全 source の tasks を union し、task_index を 0-based contiguous に remap。

    LeRobot iloc 互換のため remap を必ず適用する (build_task_index_remap 参照)。
    ``source_task_index`` などの provenance field は episode meta 側で別途 preserve
    されているので、tasks.parquet 側で source 値を保持する必要は無い。

    Column name の正規化: curation_tool (LeRobot 0.6.0) build の chunks は
    task 名 column が `__index_level_0__` になっているが、LeRobot v3.0 canonical
    schema は `task` column を要求 (lerobot/datasets/utils.py:344 が
    `if "task" in tasks.columns: sort + set_index("task")` する)。renameしないと
    load_tasks が RangeIndex のまま残し、`iloc[i].name` が int (RangeIndex value)
    を返してしまう。canonical 化することで item["task"] が正しく task 名 string
    になる。
    """
    remap = build_task_index_remap(sources)
    seen: dict[int, dict[str, Any]] = {}
    for s in sources:
        for row in s.tasks:
            ti = int(row["task_index"])
            if ti in seen:
                continue
            seen[ti] = dict(row)
    result: list[dict[str, Any]] = []
    for orig in sorted(seen):
        row = dict(seen[orig])
        row["task_index"] = remap[orig]
        # __index_level_0__ (curation_tool 0.6.0 legacy) → canonical "task" column に rename
        if "__index_level_0__" in row and "task" not in row:
            row["task"] = row.pop("__index_level_0__")
        result.append(row)
    return result


def _sanitize_episode_row(row: dict[str, Any]) -> dict[str, Any]:
    """internal-only fields (`__source_*`, `source_dataset_root`, `source_task_id`) を除外。"""
    return {
        k: v
        for k, v in row.items()
        if not k.startswith("__") and k not in {"source_dataset_root", "source_task_id"}
    }


def write_tasks_parquet(
    output_dir: Path, tasks: list[dict[str, Any]]
) -> None:
    pa, pq = require_pyarrow()
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(tasks), output_dir / "meta" / "tasks.parquet"
    )


def write_merged_info_json(
    *,
    output_dir: Path,
    source_info: dict[str, Any],
    features: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    chunks_size: int,
    data_files_size_mb: int,
    video_files_size_mb: int,
) -> None:
    """multi-task 対応の info.json 書き出し (build_subtask_datasets.write_info は
    total_tasks=1 hardcode なので、本 merger は独自版を持つ)。
    """
    info = {k: v for k, v in source_info.items() if k in _KNOWN_INFO_KEYS}
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = total_tasks
    info["chunks_size"] = chunks_size
    info["data_files_size_in_mb"] = data_files_size_mb
    info["video_files_size_in_mb"] = video_files_size_mb
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    info["features"] = features
    info["splits"] = {"train": f"0:{total_episodes}"}
    write_json(output_dir / "meta" / "info.json", info)


def write_stats_json(
    output_dir: Path,
    stats: dict[str, StatsAccumulator],
    video_keys: list[str],
) -> None:
    stats_json = {name: acc.to_dict() for name, acc in stats.items()}
    for video_key in video_keys:
        stats_json[video_key] = imagenet_video_stats()
    write_json(output_dir / "meta" / "stats.json", stats_json)


def write_provenance(
    output_dir: Path,
    sources: list[SourceDataset],
    plan: MergePlan,
    filters_by_task: dict[int, str],
    total_frames: int,
    video_mode: str,
) -> None:
    """team_ramen_curation.json: どの source dataset を merge したかの provenance を記録。
    source 側の build 時 team_ramen_curation.json (source SHA / curation SHA / verdicts /
    segments 詳細) を `sources[i].build_provenance` に nest することで、MIX 単独から
    元 dataset の traceback を可能にする。"""
    per_task_counts: dict[int, int] = {}
    for ep in plan.episodes:
        per_task_counts[int(ep["source_task_id"])] = (
            per_task_counts.get(int(ep["source_task_id"]), 0) + 1
        )

    def _load_source_build_provenance(root: Path) -> dict[str, Any] | None:
        p = root / "meta" / "team_ramen_curation.json"
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    provenance = {
        "merged_by": "merge_curated_lerobot_datasets.py",
        "total_episodes": len(plan.episodes),
        "total_frames": total_frames,
        "sources": [
            {
                "root": str(s.root),
                "task_indices": [int(t["task_index"]) for t in s.tasks],
                "episode_count_input": len(s.episodes),
                "build_provenance": _load_source_build_provenance(s.root),
            }
            for s in sources
        ],
        "filters_applied": {str(k): v for k, v in filters_by_task.items()},
        "episodes_per_task_after_filter": {str(k): v for k, v in per_task_counts.items()},
        # passthrough = 元の mp4 のまま (画素が元と同一)、reencode / lossless / copy = ffmpeg で切り出し
        "video_mode": video_mode,
    }
    write_json(output_dir / "meta" / "team_ramen_curation.json", provenance)


def rebuild_meta(
    *,
    sources: list[SourceDataset],
    output_meta: list[dict[str, Any]],
    output_dir: Path,
    total_frames: int,
    stats: dict[str, StatsAccumulator],
    video_keys: list[str],
    episodes_per_meta_file: int,
    data_files_size_mb: int,
    video_files_size_mb: int,
    chunks_size: int,
    plan: MergePlan,
    filters_by_task: dict[int, str],
    video_mode: str,
) -> None:
    """B-5: tasks.parquet + episodes.parquet + info.json + stats.json + provenance を書く。"""
    pa, pq = require_pyarrow()

    # 1. tasks.parquet
    tasks = merge_tasks(sources)
    write_tasks_parquet(output_dir, tasks)

    # 2. episodes.parquet (internal fields を除外して write_episode_meta へ)
    sanitized = [_sanitize_episode_row(row) for row in output_meta]
    write_episode_meta(pa, pq, output_dir, sanitized, episodes_per_meta_file)

    # 3. info.json (source[0] を base に、multi-task 対応)
    write_merged_info_json(
        output_dir=output_dir,
        source_info=sources[0].info,
        features=sources[0].features,
        total_episodes=len(output_meta),
        total_frames=total_frames,
        total_tasks=len(tasks),
        chunks_size=chunks_size,
        data_files_size_mb=data_files_size_mb,
        video_files_size_mb=video_files_size_mb,
    )

    # 4. stats.json
    write_stats_json(output_dir, stats, video_keys)

    # 5. provenance
    write_provenance(output_dir, sources, plan, filters_by_task, total_frames, video_mode)


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir",
        action="append",
        required=True,
        help="local built dataset directory (repeatable)",
    )
    p.add_argument(
        "--task-filter",
        action="append",
        default=[],
        help="'TASK_ID:FILTER_NAME' pair (repeatable). "
        f"filters: {', '.join(FILTER_REGISTRY.keys())}",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="output merged dataset directory",
    )
    p.add_argument(
        "--data-files-size-mb",
        type=int,
        default=DEFAULT_DATA_FILE_SIZE_MB,
    )
    p.add_argument(
        "--video-files-size-mb",
        type=int,
        default=DEFAULT_VIDEO_FILE_SIZE_MB,
    )
    p.add_argument(
        "--episodes-per-meta-file",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--video-mode",
        choices=["copy", "reencode", "lossless", "passthrough"],
        default="reencode",
        help=(
            "ffmpeg clip mode; default 'reencode' = H.264 CRF 18 (visually lossless, "
            "SSIM>0.986 vs source). "
            "'lossless' = CRF 0 (source が既に crf~16 lossy なため実質過剰品質、decode 4x 重い)、"
            "'copy' = re-encode 無し (GOP 境界で frame ズレの可能性、学習用非推奨)、"
            "'passthrough' = 切り出さず元 mp4 を file ごと置く (画素は元と同一、Issue #146)。"
            "元 file が生存 episode で隙間なく埋まっている時だけ (filter で落とす merge は不可)"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing --output-dir",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + print plan only, no output write",
    )
    p.add_argument(
        "--allow-duplicate-task-indices",
        action="store_true",
        help=(
            "同 task_id を持つ複数 source の concat を許可 (Issue #122: task 5 の 8 chunks 等)。"
            "default では 'each source must own 1 unique task_id' を強制、複数 chunks を"
            "束ねるときのみ このフラグを渡す。"
        ),
    )
    # Issue #122: camera 並列 (詳細は write_merged_videos docstring)
    p.add_argument(
        "--max-video-workers",
        type=int,
        default=4,
        help="camera 単位の並列度 (default 4、1 で sequential fallback)",
    )
    p.add_argument(
        "--max-clip-workers",
        type=int,
        default=4,
        help=(
            "camera 内の clip extract 並列度 (default 4)。max_video_workers との積で "
            "合計 concurrent ffmpeg 数決定 (default 4×4=16)。1 で従来 sequential。"
        ),
    )
    p.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=None,
        help=(
            "各 ffmpeg process の thread 数。default None で auto "
            "(cpu_count // (video_workers × clip_workers)、sequential 時 0=全 core)。"
            "明示 0 で旧動作 (over-subscription) に戻せる。"
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    source_dirs = [Path(d) for d in args.source_dir]
    for d in source_dirs:
        if not d.exists():
            raise FileNotFoundError(f"source dir not found: {d}")

    print(f"[load] {len(source_dirs)} source datasets ...")
    sources = [load_source(d) for d in source_dirs]

    print("[validate] schema consistency ...")
    validate_schemas(sources, allow_duplicate_task_indices=args.allow_duplicate_task_indices,
                     video_mode=args.video_mode)
    print(f"  fps={sources[0].fps}, video_keys={sources[0].video_keys}")
    print(f"  features={sorted(sources[0].features.keys())}")

    filters_by_task = parse_task_filters(args.task_filter)
    print(f"[filters] {filters_by_task}")

    print("[plan] apply filters + assign global episode_index ...")
    plan = build_merge_plan(sources, filters_by_task)
    total_input_eps = sum(len(s.episodes) for s in sources)
    print(f"  input episodes: {total_input_eps}")
    print(f"  output episodes (after filter): {len(plan.episodes)}")
    per_task_out = Counter(ep["source_task_id"] for ep in plan.episodes)
    for task_id in sorted(per_task_out):
        print(f"    task {task_id}: {per_task_out[task_id]} episodes")

    if args.dry_run:
        print("[dry-run] no output written")
        return 0

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_dir} exists; pass --force to replace"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    print(f"[output] {output_dir}")

    print("[data] merging parquet files ...")
    chunks_size = int(sources[0].info.get("chunks_size", DEFAULT_CHUNKS_SIZE))
    output_meta, total_frames, stats = merge_data(
        sources=sources,
        plan=plan,
        output_dir=output_dir,
        data_files_size_mb=args.data_files_size_mb,
        chunks_size=chunks_size,
    )
    print(f"  data: {len(output_meta)} episodes, {total_frames} frames")

    print(f"[video] video_mode={args.video_mode} ...")
    fps = float(sources[0].fps)
    write_merged_videos(
        sources=sources,
        output_meta=output_meta,
        output_dir=output_dir,
        video_keys=sources[0].video_keys,
        video_mode=args.video_mode,
        video_files_size_mb=args.video_files_size_mb,
        chunks_size=chunks_size,
        fps=fps,
        ffmpeg_threads=args.ffmpeg_threads,
        max_video_workers=args.max_video_workers,
        max_clip_workers=args.max_clip_workers,
    )
    print(f"  video: {len(sources[0].video_keys)} cameras processed")

    print("[meta] rebuilding tasks / episodes / info / stats / provenance ...")
    rebuild_meta(
        sources=sources,
        output_meta=output_meta,
        output_dir=output_dir,
        total_frames=total_frames,
        stats=stats,
        video_keys=sources[0].video_keys,
        episodes_per_meta_file=args.episodes_per_meta_file,
        data_files_size_mb=args.data_files_size_mb,
        video_files_size_mb=args.video_files_size_mb,
        chunks_size=chunks_size,
        plan=plan,
        filters_by_task=filters_by_task,
        video_mode=args.video_mode,
    )
    print(f"  meta: total_tasks={len(merge_tasks(sources))}")

    print(f"[done] merged dataset ready at {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
