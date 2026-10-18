#!/usr/bin/env python3
"""Build a LeRobot v3 dataset from non-failure curation segments.

Only the curation metadata and the source parquet/video shards required by the
selected segments are downloaded. Curation frame ranges are episode-local,
zero-based, and inclusive at both ends. A task sequence can also be materialized
as one output episode when its curated component ranges are contiguous.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_subtask_datasets import (  # noqa: E402
    BYTES_PER_MB,
    DEFAULT_CHUNKS_SIZE,
    DEFAULT_DATA_FILE_SIZE_MB,
    DEFAULT_VIDEO_FILE_SIZE_MB,
    FileLocation,
    StatsAccumulator,
    data_path_from_episode,
    estimate_rows_size_bytes,
    imagenet_video_stats,
    next_file_location,
    require_pyarrow,
    write_data_file,
    write_episode_meta,
    write_info,
    write_json,
    write_segment_videos,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REPO_ID = "BitRobot/G1_WBT_Dex1_Building-Children-Table"
DEFAULT_CURATION_REPO_ID = "Team-RAMEN/ramen-g1-ikea-assembly-curation"

# Cross-task filter / per-episode filter constants (RAMEN-Ori curation policy)
# docs/dataset/ramen_ori_curation_policy.md task 2 / task 5 / task 7 sections
TASK_ID_INSERT_TABLE_LEG = 0
TASK_ID_FLIP_TABLE = 2
TASK_ID_ROTATE_LEG_TO_TIGHTEN = 3
TASK_ID_PICK_TABLE_LEG = 4
TASK_ID_ROTATE_TABLE_BASE = 5
TASK_ID_MOVE_TABLE_BASE = 7


def _non_negative_int(value: str) -> int:
    """argparse type: non-negative int (argparse level で reject、SystemExit 経由)。"""
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value}")
    return n


DEFAULT_ALLOWED_VERDICTS = ("optimal", "success", "ambiguous")
REQUIRED_CURATION_COLUMNS = {
    "episode_id",
    "frame_start",
    "frame_end",
    "task_index",
    "verdict",
    "failure_category",
    "reviewer",
    "reviewed_at",
    "schema_version",
}
SUPPORTED_CURATION_SCHEMA = 4
REMOTE_DATASET_ROOTS = ("data/", "meta/", "videos/")


@dataclass(frozen=True)
class HubDataset:
    repo_id: str
    revision: str


@dataclass(frozen=True)
class CuratedTaskComponent:
    source_task_index: int
    source_frame_start: int
    source_frame_end: int
    verdict: str
    failure_category: str | None
    reviewer: str
    reviewed_at: str
    schema_version: int


@dataclass(frozen=True)
class CuratedSegment:
    source_episode_index: int
    source_frame_start: int
    source_frame_end: int
    source_task_index: int
    verdict: str
    failure_category: str | None
    reviewer: str
    reviewed_at: str
    schema_version: int
    components: tuple[CuratedTaskComponent, ...] = ()

    @property
    def frame_count(self) -> int:
        return self.source_frame_end - self.source_frame_start + 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one curation task ID or a contiguous sequence of task IDs, "
            "exclude failure verdicts, and build a standalone LeRobot v3 dataset "
            "from the required source shards."
        )
    )
    parser.add_argument(
        "--task-id",
        type=int,
        help=(
            "source/curation task ID; mutually exclusive with --task-sequence "
            "(one selector is required unless --upload-existing is used)"
        ),
    )
    parser.add_argument(
        "--task-sequence",
        nargs="+",
        type=_non_negative_int,
        metavar="TASK_ID",
        help=(
            "ordered source/curation task IDs to join into one output episode; "
            "each component must have an accepted verdict"
        ),
    )
    parser.add_argument(
        "--output-task-id",
        type=_non_negative_int,
        help=(
            "derived task ID for --task-sequence output; defaults to one greater "
            "than the largest source task ID"
        ),
    )
    parser.add_argument(
        "--max-sequence-gap-frames",
        type=_non_negative_int,
        default=0,
        help=(
            "maximum unlabeled frames allowed between sequence components "
            "(default: 0, strictly contiguous)"
        ),
    )
    parser.add_argument(
        "--task-name",
        help="output task text; defaults to the source meta/tasks.parquet text",
    )
    parser.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--curation-repo-id", default=DEFAULT_CURATION_REPO_ID)
    parser.add_argument("--curation-revision", default="main")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "output dataset root; defaults to "
            "outputs/curated_task_<task-id>_non_failure"
        ),
    )
    parser.add_argument(
        "--verdicts",
        nargs="+",
        default=list(DEFAULT_ALLOWED_VERDICTS),
        help=(
            "accepted verdicts (default: optimal success ambiguous, i.e. every "
            "schema-v4 verdict except failure)"
        ),
    )
    parser.add_argument(
        "--video-keys",
        nargs="*",
        default=None,
        help="video features to preserve; omit to preserve every source video feature",
    )
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument(
        "--video-mode",
        choices=("copy", "reencode", "lossless"),
        default="lossless",
        help=(
            "reencode keeps precise frame boundaries with H.264 CRF 20; "
            "lossless (default) uses H.264 CRF 0; copy avoids re-encoding but "
            "can include neighboring GOP frames"
        ),
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        default=1,
        help="parallel FFmpeg clip processes per camera (default: 1)",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help=(
            "threads per FFmpeg process; 0 lets FFmpeg decide (default: 0). "
            "Peak clip encoding threads are approximately video-workers x this value"
        ),
    )
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument(
        "--seg-offset",
        type=_non_negative_int,
        default=0,
        help=(
            "skip the first N segments (after cross-task filter / overlap "
            "resolution). Used with --max-segments to build a chunk range "
            "[seg-offset : seg-offset + max-segments]. Default 0 = no offset."
        ),
    )
    parser.add_argument(
        "--data-files-size-mb", type=int, default=DEFAULT_DATA_FILE_SIZE_MB
    )
    parser.add_argument(
        "--video-files-size-mb", type=int, default=DEFAULT_VIDEO_FILE_SIZE_MB
    )
    parser.add_argument("--episodes-per-meta-file", type=int, default=1_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve revisions and print the selection plan without downloading data/video shards",
    )
    parser.add_argument(
        "--upload-repo-id",
        metavar="NAMESPACE/NAME",
        help=(
            "upload the validated output to this Hugging Face dataset repo; "
            "authentication is read from HF_TOKEN or `hf auth login`"
        ),
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="create a public upload repo (new repos are private by default)",
    )
    parser.add_argument(
        "--replace-remote",
        action="store_true",
        help=(
            "allow upload to an existing dataset repo and remove stale remote "
            "data/meta/video files after a successful upload"
        ),
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=4,
        help="parallel workers for the resumable Hugging Face upload (default: 4)",
    )
    parser.add_argument(
        "--upload-existing",
        action="store_true",
        help=(
            "skip dataset generation, validate --output-dir, and upload/resume it; "
            "requires --output-dir and --upload-repo-id"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_segments < 0:
        raise ValueError("--max-segments must be non-negative")
    for name in (
        "data_files_size_mb",
        "video_files_size_mb",
        "episodes_per_meta_file",
        "upload_workers",
        "video_workers",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.ffmpeg_threads < 0:
        raise ValueError("--ffmpeg-threads must be non-negative")
    validate_upload_options(args)
    if args.upload_existing:
        output_dir = args.output_dir.expanduser().resolve()
        validate_existing_output_for_upload(output_dir)
        api = HfApi()
        print_hf_identity(api)
        upload_curated_dataset(
            api,
            output_dir=output_dir,
            repo_id=args.upload_repo_id,
            private=not args.public,
            replace_remote=args.replace_remote,
            num_workers=args.upload_workers,
        )
        return 0
    task_sequence = tuple(args.task_sequence or ())
    if args.task_id is not None and task_sequence:
        raise ValueError("--task-id and --task-sequence are mutually exclusive")
    if args.task_id is None and not task_sequence:
        raise ValueError(
            "--task-id or --task-sequence is required unless --upload-existing is used"
        )
    if args.task_id is not None and args.task_id < 0:
        raise ValueError("--task-id must be non-negative")
    if task_sequence and len(task_sequence) < 2:
        raise ValueError("--task-sequence requires at least two task IDs")
    if not task_sequence and args.output_task_id is not None:
        raise ValueError("--output-task-id is only valid with --task-sequence")

    allowed_verdicts = normalize_verdicts(args.verdicts)
    api = HfApi()
    source = resolve_dataset(api, args.source_repo_id, args.source_revision)
    curation = resolve_dataset(api, args.curation_repo_id, args.curation_revision)
    source_info = load_json_from_hub(source, "meta/info.json")
    source_tasks = load_source_tasks(source)
    episode_rows = load_source_episode_rows(api, source)
    labels = load_curation_rows(curation)

    if task_sequence:
        missing_task_ids = sorted(set(task_sequence) - set(source_tasks))
        if missing_task_ids:
            raise ValueError(
                f"task IDs {missing_task_ids} are absent from "
                f"{source.repo_id}/meta/tasks.parquet"
            )
        output_task_id = (
            args.output_task_id
            if args.output_task_id is not None
            else max(source_tasks) + 1
        )
        if output_task_id in source_tasks:
            raise ValueError(
                f"derived --output-task-id {output_task_id} collides with a source task ID"
            )
        task_name = args.task_name or " then ".join(
            source_tasks[task_id] for task_id in task_sequence
        )
        segments, overlap_resolutions = select_task_sequence_segments(
            labels,
            task_sequence=task_sequence,
            allowed_verdicts=allowed_verdicts,
            max_gap_frames=args.max_sequence_gap_frames,
        )
    else:
        output_task_id = args.task_id
        if output_task_id not in source_tasks and not args.task_name:
            raise ValueError(
                f"task ID {output_task_id} is absent from "
                f"{source.repo_id}/meta/tasks.parquet; pass --task-name only if "
                "this is an intentional corrected curation task"
            )
        task_name = args.task_name or source_tasks[output_task_id]
        segments = select_segments(
            labels,
            task_id=output_task_id,
            allowed_verdicts=allowed_verdicts,
        )
        segments, overlap_resolutions = resolve_overlaps_by_reviewed_at(segments)

    for kept, dropped in overlap_resolutions:
        print(
            f"[overlap resolved] ep {kept.source_episode_index}: "
            f"kept [{kept.source_frame_start},{kept.source_frame_end}] "
            f"reviewed={kept.reviewed_at}, "
            f"dropped [{dropped.source_frame_start},{dropped.source_frame_end}] "
            f"reviewed={dropped.reviewed_at}",
            file=sys.stderr,
        )
    if not task_sequence and output_task_id == TASK_ID_MOVE_TABLE_BASE:
        before_count = len(segments)
        segments = apply_move_table_base_cross_task_filter(segments, labels)
        print(
            f"[cross-task filter] task {output_task_id} (move_table_base): "
            f"{before_count} → {len(segments)} seg (pick より絶対前のみ採用)",
            file=sys.stderr,
        )
    elif not task_sequence and output_task_id == TASK_ID_ROTATE_TABLE_BASE:
        before_count = len(segments)
        segments = apply_rotate_table_base_cross_task_filter(segments, labels)
        print(
            f"[cross-task filter] task {output_task_id} (rotate_table_base): "
            f"{before_count} → {len(segments)} seg "
            f"(pick 以降 / fallback: insert or rotate 以降)",
            file=sys.stderr,
        )
    elif not task_sequence and output_task_id == TASK_ID_FLIP_TABLE:
        before_count = len(segments)
        segments = apply_flip_table_per_episode_filter(segments)
        print(
            f"[per-episode filter] task {output_task_id} (flip_table): "
            f"{before_count} → {len(segments)} seg "
            f"(Episode 内 frame_start 最大の 1 seg のみ = 実 flip 動作)",
            file=sys.stderr,
        )
    validate_segments(segments, episode_rows)
    if args.seg_offset:
        segments = segments[args.seg_offset :]
    if args.max_segments:
        segments = segments[: args.max_segments]
    if not segments:
        selector = (
            f"task sequence {list(task_sequence)}"
            if task_sequence
            else f"task ID {output_task_id}"
        )
        raise ValueError(
            f"no curation rows matched {selector} and verdicts "
            f"{sorted(allowed_verdicts)}"
        )

    features, video_keys = select_features(
        source_info["features"],
        requested_video_keys=args.video_keys,
        skip_videos=args.skip_videos,
    )
    # Preflight: curation label frame_end が container 実長 (from/to timestamp × fps)
    # を超える seg を drop。BitRobot 上流 length/container 不整合対応。
    if video_keys:
        before_preflight = len(segments)
        segments, preflight_drops = clamp_segments_to_container_length(
            segments, episode_rows, video_keys, float(source_info["fps"])
        )
        for seg, cont_len in preflight_drops:
            print(
                f"[preflight drop] ep {seg.source_episode_index} seg "
                f"[{seg.source_frame_start},{seg.source_frame_end}] "
                f"exceeds container len={cont_len} (BitRobot 上流不整合)",
                file=sys.stderr,
            )
        if preflight_drops:
            print(
                f"[preflight] {len(preflight_drops)} / {before_preflight} segs "
                f"dropped due to container length mismatch",
                file=sys.stderr,
            )
        if not segments:
            raise ValueError("all segments dropped by container length preflight")
    output_dir = (
        (
            args.output_dir
            if args.output_dir is not None
            else ROOT
            / "outputs"
            / (
                "curated_task_sequence_"
                + "_".join(str(task_id) for task_id in task_sequence)
                if task_sequence
                else f"curated_task_{output_task_id}_non_failure"
            )
        )
        .expanduser()
        .resolve()
    )
    summary = selection_summary(
        segments=segments,
        source=source,
        curation=curation,
        task_id=output_task_id,
        task_name=task_name,
        allowed_verdicts=allowed_verdicts,
        video_keys=video_keys,
        episode_rows=episode_rows,
        output_dir=output_dir,
    )
    if task_sequence:
        summary["source_task_sequence"] = list(task_sequence)
        summary["max_sequence_gap_frames"] = args.max_sequence_gap_frames
    summary["video_workers"] = args.video_workers
    summary["video_mode"] = args.video_mode
    summary["ffmpeg_threads_per_worker"] = (
        args.ffmpeg_threads if args.ffmpeg_threads else "auto"
    )
    summary["video_cpu_thread_budget"] = (
        args.video_workers * args.ffmpeg_threads
        if args.ffmpeg_threads
        else "FFmpeg auto"
    )
    if args.upload_repo_id:
        summary["upload_repo_id"] = args.upload_repo_id
        summary["upload_visibility"] = "public" if args.public else "private"
        summary["replace_remote"] = args.replace_remote
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    if args.upload_repo_id:
        print_hf_identity(api)
        validate_remote_upload_target(
            api,
            repo_id=args.upload_repo_id,
            replace_remote=args.replace_remote,
        )

    source_stats = load_json_from_hub(source, "meta/stats.json")
    build_dataset(
        source=source,
        curation=curation,
        source_info=source_info,
        source_stats=source_stats,
        episode_rows=episode_rows,
        segments=segments,
        task_id=output_task_id,
        task_name=task_name,
        allowed_verdicts=allowed_verdicts,
        features=features,
        video_keys=video_keys,
        output_dir=output_dir,
        video_mode=args.video_mode,
        data_files_size_mb=args.data_files_size_mb,
        video_files_size_mb=args.video_files_size_mb,
        episodes_per_meta_file=args.episodes_per_meta_file,
        force=args.force,
        video_workers=args.video_workers,
        ffmpeg_threads=args.ffmpeg_threads,
        source_task_sequence=task_sequence or None,
        max_sequence_gap_frames=args.max_sequence_gap_frames,
    )
    validate_output_dataset(
        output_dir,
        expected_episodes=len(segments),
        expected_frames=sum(segment.frame_count for segment in segments),
        expected_task_id=output_task_id,
        video_keys=video_keys,
    )
    print(f"local build complete: {output_dir}", flush=True)
    if args.upload_repo_id:
        upload_curated_dataset(
            api,
            output_dir=output_dir,
            repo_id=args.upload_repo_id,
            private=not args.public,
            replace_remote=args.replace_remote,
            num_workers=args.upload_workers,
        )
    return 0


def validate_upload_options(args: argparse.Namespace) -> None:
    if args.public and not args.upload_repo_id:
        raise ValueError("--public requires --upload-repo-id")
    if args.replace_remote and not args.upload_repo_id:
        raise ValueError("--replace-remote requires --upload-repo-id")
    if args.upload_existing:
        if not args.upload_repo_id:
            raise ValueError("--upload-existing requires --upload-repo-id")
        if args.output_dir is None:
            raise ValueError("--upload-existing requires --output-dir")
        if args.dry_run:
            raise ValueError("--upload-existing cannot be combined with --dry-run")


def print_hf_identity(api: HfApi) -> None:
    identity = api.whoami()
    identity_name = (
        str(identity.get("name", "authenticated"))
        if isinstance(identity, dict)
        else "authenticated"
    )
    print(f"huggingface_identity: {identity_name}", flush=True)


def validate_existing_output_for_upload(output_dir: Path) -> None:
    provenance_path = output_dir / "meta" / "team_ramen_curation.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"curation provenance is missing from existing output: {provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != "team_ramen_curated_lerobot_v1":
        raise ValueError(
            f"unsupported output provenance schema: {provenance.get('schema_version')!r}"
        )
    validate_output_dataset(
        output_dir,
        expected_episodes=int(provenance["episode_count"]),
        expected_frames=int(provenance["frame_count"]),
        expected_task_id=int(provenance["task_id"]),
        video_keys=[str(key) for key in provenance["video_keys"]],
    )
    print(f"existing local dataset validated: {output_dir}", flush=True)


def is_managed_dataset_path(path: str) -> bool:
    return path == "README.md" or path.startswith(REMOTE_DATASET_ROOTS)


def local_upload_files(output_dir: Path) -> set[str]:
    return {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(output_dir).parts
    }


def validate_remote_upload_target(
    api: HfApi,
    *,
    repo_id: str,
    replace_remote: bool,
) -> bool:
    exists = api.repo_exists(repo_id=repo_id, repo_type="dataset")
    if exists and not replace_remote:
        raise FileExistsError(
            f"Hugging Face dataset repo {repo_id} already exists; "
            "pass --replace-remote to update it"
        )
    return exists


def upload_curated_dataset(
    api: HfApi,
    *,
    output_dir: Path,
    repo_id: str,
    private: bool,
    replace_remote: bool,
    num_workers: int,
) -> None:
    """Upload a validated dataset with resumable large-folder transfers."""
    if not output_dir.is_dir():
        raise FileNotFoundError(f"upload directory does not exist: {output_dir}")
    required = ("README.md", "meta/info.json", "meta/team_ramen_curation.json")
    missing = [path for path in required if not (output_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"refusing to upload incomplete dataset; missing {missing}")

    exists = validate_remote_upload_target(
        api,
        repo_id=repo_id,
        replace_remote=replace_remote,
    )
    if not exists:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=False,
        )
        print(
            f"created {'private' if private else 'public'} dataset repo: {repo_id}",
            flush=True,
        )
    else:
        print(f"updating existing dataset repo: {repo_id}", flush=True)

    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_dir,
        private=private if not exists else None,
        ignore_patterns=[".cache/**"],
        num_workers=num_workers,
    )

    if exists:
        local_files = local_upload_files(output_dir)
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        stale_files = sorted(
            path for path in remote_files - local_files if is_managed_dataset_path(path)
        )
        if stale_files:
            api.delete_files(
                repo_id=repo_id,
                repo_type="dataset",
                delete_patterns=stale_files,
                commit_message="Remove stale files from curated LeRobot dataset",
            )
            print(f"removed {len(stale_files)} stale remote file(s)", flush=True)

    print(
        f"upload complete: https://huggingface.co/datasets/{repo_id}",
        flush=True,
    )


def resolve_dataset(api: HfApi, repo_id: str, revision: str) -> HubDataset:
    info = api.dataset_info(repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"could not resolve {repo_id}@{revision}")
    return HubDataset(repo_id=repo_id, revision=info.sha)


def download(hub: HubDataset, filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=hub.repo_id,
            repo_type="dataset",
            revision=hub.revision,
            filename=filename,
        )
    )


def load_json_from_hub(hub: HubDataset, filename: str) -> dict[str, Any]:
    return json.loads(download(hub, filename).read_text(encoding="utf-8"))


def list_repo_files(api: HfApi, hub: HubDataset, prefix: str, suffix: str) -> list[str]:
    return sorted(
        filename
        for filename in api.list_repo_files(
            repo_id=hub.repo_id,
            repo_type="dataset",
            revision=hub.revision,
        )
        if filename.startswith(prefix) and filename.endswith(suffix)
    )


def load_source_tasks(source: HubDataset) -> dict[int, str]:
    _, pq = require_pyarrow()
    rows = pq.read_table(download(source, "meta/tasks.parquet")).to_pylist()
    text_columns = (
        [column for column in rows[0] if column != "task_index"] if rows else []
    )
    if len(text_columns) != 1:
        raise ValueError(
            f"expected one task text column in {source.repo_id}/meta/tasks.parquet, "
            f"found {text_columns}"
        )
    text_column = text_columns[0]
    tasks = {int(row["task_index"]): str(row[text_column]) for row in rows}
    if len(tasks) != len(rows):
        raise ValueError("source tasks metadata contains duplicate task_index values")
    return tasks


def load_source_episode_rows(api: HfApi, source: HubDataset) -> list[dict[str, Any]]:
    _, pq = require_pyarrow()
    filenames = list_repo_files(api, source, "meta/episodes/", ".parquet")
    if not filenames:
        raise FileNotFoundError(f"no episode metadata in {source.repo_id}")
    rows: list[dict[str, Any]] = []
    for filename in filenames:
        rows.extend(pq.read_table(download(source, filename)).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    for expected, row in enumerate(rows):
        actual = int(row["episode_index"])
        if actual != expected:
            raise ValueError(
                "source episode metadata must be contiguous because video lookup "
                f"uses SDK episode IDs; expected {expected}, found {actual}"
            )
    return rows


def load_curation_rows(curation: HubDataset) -> list[dict[str, Any]]:
    _, pq = require_pyarrow()
    table = pq.read_table(download(curation, "labels.parquet"))
    missing = sorted(REQUIRED_CURATION_COLUMNS - set(table.column_names))
    if missing:
        raise ValueError(f"curation labels are missing columns: {missing}")
    rows = table.select(sorted(REQUIRED_CURATION_COLUMNS)).to_pylist()
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if int(row["schema_version"]) != SUPPORTED_CURATION_SCHEMA:
            raise ValueError(
                f"unsupported curation schema {row['schema_version']}; "
                f"expected {SUPPORTED_CURATION_SCHEMA}"
            )
        key = (int(row["episode_id"]), int(row["frame_start"]))
        if key in seen:
            raise ValueError(f"duplicate curation primary key {key}")
        seen.add(key)
    return rows


def normalize_verdicts(verdicts: Iterable[str]) -> frozenset[str]:
    normalized = frozenset(str(value).strip().lower() for value in verdicts)
    if not normalized or "" in normalized:
        raise ValueError("--verdicts must contain at least one non-empty value")
    known = {"optimal", "success", "ambiguous", "failure"}
    unknown = sorted(normalized - known)
    if unknown:
        raise ValueError(
            f"unknown verdict(s): {unknown}; expected values from {sorted(known)}"
        )
    if "failure" in normalized:
        raise ValueError(
            "this builder is intentionally non-failure-only; remove 'failure' from --verdicts"
        )
    return normalized


def select_segments(
    rows: Iterable[dict[str, Any]],
    *,
    task_id: int,
    allowed_verdicts: frozenset[str],
) -> list[CuratedSegment]:
    selected: list[CuratedSegment] = []
    for row in rows:
        verdict = str(row["verdict"]).strip().lower()
        if int(row["task_index"]) != task_id or verdict not in allowed_verdicts:
            continue
        selected.append(
            CuratedSegment(
                source_episode_index=int(row["episode_id"]),
                source_frame_start=int(row["frame_start"]),
                source_frame_end=int(row["frame_end"]),
                source_task_index=int(row["task_index"]),
                verdict=verdict,
                failure_category=(
                    None
                    if row["failure_category"] is None
                    else str(row["failure_category"])
                ),
                reviewer=str(row["reviewer"]),
                reviewed_at=str(row["reviewed_at"]),
                schema_version=int(row["schema_version"]),
            )
        )
    selected.sort(
        key=lambda segment: (
            segment.source_episode_index,
            segment.source_frame_start,
            segment.source_frame_end,
        )
    )
    return selected


def resolve_overlaps_by_reviewed_at(
    segments: list[CuratedSegment],
) -> tuple[list[CuratedSegment], list[tuple[CuratedSegment, CuratedSegment]]]:
    """同 episode 内で frame range が重なる seg のうち reviewed_at が古い方を drop。
    curation UI の "修正時に旧 label 削除忘れ" pattern を自動救済する。
    3 個以上の連鎖 overlap は反復適用で非 overlap 集合に収束。
    Returns: (kept segments, [(kept, dropped), ...] resolution log entries)
    詳細: docs/dataset/ramen_ori_curation_policy.md "Overlap resolution policy"."""
    by_ep: dict[int, list[CuratedSegment]] = defaultdict(list)
    for seg in segments:
        by_ep[seg.source_episode_index].append(seg)

    kept_all: list[CuratedSegment] = []
    resolutions: list[tuple[CuratedSegment, CuratedSegment]] = []
    for _ep, ep_segs in by_ep.items():
        current = list(ep_segs)
        while True:
            current.sort(key=lambda s: (s.source_frame_start, s.source_frame_end))
            found_overlap = False
            for i in range(len(current) - 1):
                a = current[i]
                b = current[i + 1]
                if a.source_frame_end >= b.source_frame_start:
                    if a.reviewed_at >= b.reviewed_at:
                        kept, dropped = a, b
                    else:
                        kept, dropped = b, a
                    resolutions.append((kept, dropped))
                    current = [s for s in current if s is not dropped]
                    found_overlap = True
                    break
            if not found_overlap:
                break
        kept_all.extend(current)

    kept_all.sort(
        key=lambda s: (
            s.source_episode_index,
            s.source_frame_start,
            s.source_frame_end,
        )
    )
    return kept_all, resolutions


def task_component(segment: CuratedSegment) -> CuratedTaskComponent:
    return CuratedTaskComponent(
        source_task_index=segment.source_task_index,
        source_frame_start=segment.source_frame_start,
        source_frame_end=segment.source_frame_end,
        verdict=segment.verdict,
        failure_category=segment.failure_category,
        reviewer=segment.reviewer,
        reviewed_at=segment.reviewed_at,
        schema_version=segment.schema_version,
    )


def select_task_sequence_segments(
    rows: Iterable[dict[str, Any]],
    *,
    task_sequence: tuple[int, ...],
    allowed_verdicts: frozenset[str],
    max_gap_frames: int = 0,
) -> tuple[
    list[CuratedSegment],
    list[tuple[CuratedSegment, CuratedSegment]],
]:
    """Join ordered curated tasks into episode-local composite segments.

    Each component first passes the same verdict selection and newest-review
    overlap resolution as single-task generation. The next component must start
    between one and ``max_gap_frames + 1`` frames after the previous component.
    A component is consumed by at most one output sequence.
    """
    if len(task_sequence) < 2:
        raise ValueError("task_sequence requires at least two task IDs")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")

    rows = list(rows)
    selected_by_task: dict[int, list[CuratedSegment]] = {}
    overlap_resolutions: list[tuple[CuratedSegment, CuratedSegment]] = []
    for task_id in dict.fromkeys(task_sequence):
        selected = select_segments(
            rows,
            task_id=task_id,
            allowed_verdicts=allowed_verdicts,
        )
        selected, resolutions = resolve_overlaps_by_reviewed_at(selected)
        selected_by_task[task_id] = selected
        overlap_resolutions.extend(resolutions)

    by_episode_and_task: dict[int, dict[int, list[CuratedSegment]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task_id, selected in selected_by_task.items():
        for segment in selected:
            by_episode_and_task[segment.source_episode_index][task_id].append(segment)

    output: list[CuratedSegment] = []
    used_components: set[CuratedSegment] = set()
    first_task_id = task_sequence[0]
    for episode_index in sorted(by_episode_and_task):
        task_segments = by_episode_and_task[episode_index]
        for first in task_segments.get(first_task_id, []):
            if first in used_components:
                continue
            components = [first]
            for next_task_id in task_sequence[1:]:
                previous = components[-1]
                min_start = previous.source_frame_end + 1
                max_start = min_start + max_gap_frames
                candidate = next(
                    (
                        segment
                        for segment in task_segments.get(next_task_id, [])
                        if segment not in used_components
                        and min_start <= segment.source_frame_start <= max_start
                    ),
                    None,
                )
                if candidate is None:
                    break
                components.append(candidate)
            if len(components) != len(task_sequence):
                continue

            used_components.update(components)
            verdicts = {component.verdict for component in components}
            reviewers = list(dict.fromkeys(component.reviewer for component in components))
            output.append(
                CuratedSegment(
                    source_episode_index=episode_index,
                    source_frame_start=components[0].source_frame_start,
                    source_frame_end=components[-1].source_frame_end,
                    source_task_index=components[0].source_task_index,
                    verdict=(next(iter(verdicts)) if len(verdicts) == 1 else "mixed"),
                    failure_category=None,
                    reviewer="; ".join(reviewers),
                    reviewed_at=max(component.reviewed_at for component in components),
                    schema_version=max(component.schema_version for component in components),
                    components=tuple(task_component(component) for component in components),
                )
            )

    output.sort(
        key=lambda segment: (
            segment.source_episode_index,
            segment.source_frame_start,
            segment.source_frame_end,
        )
    )
    return output, overlap_resolutions


def apply_move_table_base_cross_task_filter(
    task7_segments: list[CuratedSegment],
    all_curation_rows: Iterable[dict[str, Any]],
) -> list[CuratedSegment]:
    """task 7 (move_table_base) build 時、同 episode 内で pick (task 4) より
    絶対前 (frame_start が pick 最小 frame_start より小さい) の task 7 seg
    のみ採用。pick 無し ep は制約発動せず task 7 seg 全部通す。
    詳細は docs/dataset/ramen_ori_curation_policy.md task 7 section。"""
    pick_starts_by_ep: dict[int, list[int]] = defaultdict(list)
    for row in all_curation_rows:
        if int(row["task_index"]) == TASK_ID_PICK_TABLE_LEG:
            pick_starts_by_ep[int(row["episode_id"])].append(int(row["frame_start"]))
    kept: list[CuratedSegment] = []
    for seg in task7_segments:
        pick_starts = pick_starts_by_ep.get(seg.source_episode_index)
        if pick_starts and seg.source_frame_start >= min(pick_starts):
            continue
        kept.append(seg)
    return kept


def clamp_segments_to_container_length(
    segments: list[CuratedSegment],
    episode_rows: list[dict[str, Any]],
    video_keys: list[str],
    fps: float,
) -> tuple[list[CuratedSegment], list[tuple[CuratedSegment, int]]]:
    """各 seg で source video container の実 frame 数 (from/to_timestamp × fps) と
    比較、curation label の frame_end が container 実長を超える seg を drop する。
    BitRobot 上流の meta/episodes.length と video container の 1 frame 不整合対応
    (実例: ep 43 length=13727 vs cam_2 container=13726、curation label frame_end=13726
    → 実 clip 抽出時 947 frame vs expected 948、downstream validate reject)。
    Returns: (kept segments, [(dropped seg, container_len), ...])"""
    if not video_keys:
        return list(segments), []
    kept: list[CuratedSegment] = []
    dropped: list[tuple[CuratedSegment, int]] = []
    for seg in segments:
        if seg.source_episode_index < 0 or seg.source_episode_index >= len(episode_rows):
            kept.append(seg)
            continue
        row = episode_rows[seg.source_episode_index]
        container_lengths: list[int] = []
        for vk in video_keys:
            from_key = f"videos/{vk}/from_timestamp"
            to_key = f"videos/{vk}/to_timestamp"
            if from_key not in row or to_key not in row:
                continue
            from_ts = float(row[from_key])
            to_ts = float(row[to_key])
            container_lengths.append(round((to_ts - from_ts) * fps))
        if not container_lengths:
            kept.append(seg)
            continue
        min_container = min(container_lengths)
        if seg.source_frame_end >= min_container:
            dropped.append((seg, min_container))
        else:
            kept.append(seg)
    return kept, dropped


def apply_flip_table_per_episode_filter(
    task2_segments: list[CuratedSegment],
) -> list[CuratedSegment]:
    """task 2 (flip_table) build 時、同 episode 内で frame_start 最大の 1 seg のみ採用。
    「1 番最後 = 実 flip_table 動作」を残し、rotate 系や練習 seg を drop する。
    詳細は docs/dataset/ramen_ori_curation_policy.md task 2 section。"""
    latest_by_ep: dict[int, CuratedSegment] = {}
    for seg in task2_segments:
        cur = latest_by_ep.get(seg.source_episode_index)
        if cur is None or seg.source_frame_start > cur.source_frame_start:
            latest_by_ep[seg.source_episode_index] = seg
    kept = list(latest_by_ep.values())
    kept.sort(
        key=lambda s: (
            s.source_episode_index,
            s.source_frame_start,
            s.source_frame_end,
        )
    )
    return kept


def apply_rotate_table_base_cross_task_filter(
    task5_segments: list[CuratedSegment],
    all_curation_rows: Iterable[dict[str, Any]],
) -> list[CuratedSegment]:
    """task 5 (rotate_table_base) build 時、同 episode 内の判定基準:
      1. pick (task 4) 有: 1 回目 pick.frame_start を基準
      2. pick 無: insert (task 0) / rotate (task 3) の 1 回目のうち早い方 (min) を基準
      3. pick/insert/rotate 全部無: 素通し (基準決められず)
    採用: task 5 seg の frame_start >= 基準。
    詳細は docs/dataset/ramen_ori_curation_policy.md task 5 section。"""
    firsts_by_ep: dict[int, dict[str, int]] = defaultdict(dict)
    key_by_tid = {
        TASK_ID_PICK_TABLE_LEG: "pick",
        TASK_ID_INSERT_TABLE_LEG: "insert",
        TASK_ID_ROTATE_LEG_TO_TIGHTEN: "rotate",
    }
    for row in all_curation_rows:
        key = key_by_tid.get(int(row["task_index"]))
        if key is None:
            continue
        ep = int(row["episode_id"])
        fs = int(row["frame_start"])
        cur = firsts_by_ep[ep].get(key)
        if cur is None or fs < cur:
            firsts_by_ep[ep][key] = fs

    kept: list[CuratedSegment] = []
    for seg in task5_segments:
        firsts = firsts_by_ep.get(seg.source_episode_index, {})
        if "pick" in firsts:
            threshold = firsts["pick"]
        else:
            fallbacks = [firsts[k] for k in ("insert", "rotate") if k in firsts]
            if not fallbacks:
                kept.append(seg)
                continue
            threshold = min(fallbacks)
        if seg.source_frame_start >= threshold:
            kept.append(seg)
    return kept


def validate_segments(
    segments: list[CuratedSegment],
    episode_rows: list[dict[str, Any]],
) -> None:
    previous_by_episode: dict[int, CuratedSegment] = {}
    for segment in segments:
        if segment.source_episode_index < 0 or segment.source_episode_index >= len(
            episode_rows
        ):
            raise ValueError(
                f"curation episode {segment.source_episode_index} is absent from source metadata"
            )
        if (
            segment.source_frame_start < 0
            or segment.source_frame_end < segment.source_frame_start
        ):
            raise ValueError(f"invalid curation frame range: {segment}")
        source_length = int(episode_rows[segment.source_episode_index]["length"])
        if segment.source_frame_end >= source_length:
            raise ValueError(
                f"curation range {segment.source_frame_start}:{segment.source_frame_end} "
                f"exceeds source episode {segment.source_episode_index} length {source_length}"
            )
        previous = previous_by_episode.get(segment.source_episode_index)
        if (
            previous is not None
            and segment.source_frame_start <= previous.source_frame_end
        ):
            raise ValueError(
                "selected curation segments overlap in source episode "
                f"{segment.source_episode_index}: "
                f"{previous.source_frame_start}:{previous.source_frame_end} and "
                f"{segment.source_frame_start}:{segment.source_frame_end}"
            )
        previous_by_episode[segment.source_episode_index] = segment


def select_features(
    source_features: dict[str, Any],
    *,
    requested_video_keys: list[str] | None,
    skip_videos: bool,
) -> tuple[dict[str, Any], list[str]]:
    all_video_keys = [
        key
        for key, feature in source_features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    video_keys = (
        list(requested_video_keys)
        if requested_video_keys is not None
        else all_video_keys
    )
    duplicates = [key for key, count in Counter(video_keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate --video-keys: {sorted(duplicates)}")
    unknown = sorted(set(video_keys) - set(all_video_keys))
    if unknown:
        raise ValueError(f"unknown video feature(s): {unknown}")
    if skip_videos:
        video_keys = []
    selected = {
        key: value
        for key, value in source_features.items()
        if not (
            isinstance(value, dict)
            and value.get("dtype") == "video"
            and key not in set(video_keys)
        )
    }
    return selected, video_keys


def selection_summary(
    *,
    segments: list[CuratedSegment],
    source: HubDataset,
    curation: HubDataset,
    task_id: int,
    task_name: str,
    allowed_verdicts: frozenset[str],
    video_keys: list[str],
    episode_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    data_files = {
        data_path_from_episode(episode_rows[segment.source_episode_index])
        for segment in segments
    }
    source_video_files = {
        (
            video_key,
            int(
                episode_rows[segment.source_episode_index][
                    f"videos/{video_key}/chunk_index"
                ]
            ),
            int(
                episode_rows[segment.source_episode_index][
                    f"videos/{video_key}/file_index"
                ]
            ),
        )
        for segment in segments
        for video_key in video_keys
    }
    return {
        "source_repo_id": source.repo_id,
        "source_revision": source.revision,
        "curation_repo_id": curation.repo_id,
        "curation_revision": curation.revision,
        "task_id": task_id,
        "task_name": task_name,
        "allowed_verdicts": sorted(allowed_verdicts),
        "segment_count": len(segments),
        "source_episode_count": len(
            {segment.source_episode_index for segment in segments}
        ),
        "frame_count": sum(segment.frame_count for segment in segments),
        "verdict_counts": dict(sorted(Counter(s.verdict for s in segments).items())),
        "source_data_shards": len(data_files),
        "source_video_shards": len(source_video_files),
        "video_keys": video_keys,
        "output_dir": str(output_dir),
    }


def slice_source_rows(
    source_rows: list[dict[str, Any]],
    segment: CuratedSegment,
) -> list[dict[str, Any]]:
    by_frame = {int(row["frame_index"]): row for row in source_rows}
    expected_frames = list(
        range(segment.source_frame_start, segment.source_frame_end + 1)
    )
    missing = [frame for frame in expected_frames if frame not in by_frame]
    if missing:
        raise ValueError(
            f"source episode {segment.source_episode_index} is missing curated "
            f"frames {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    return [by_frame[frame] for frame in expected_frames]


def build_dataset(
    *,
    source: HubDataset,
    curation: HubDataset,
    source_info: dict[str, Any],
    source_stats: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    segments: list[CuratedSegment],
    task_id: int,
    task_name: str,
    allowed_verdicts: frozenset[str],
    features: dict[str, Any],
    video_keys: list[str],
    output_dir: Path,
    video_mode: str,
    data_files_size_mb: int,
    video_files_size_mb: int,
    episodes_per_meta_file: int,
    force: bool,
    video_workers: int = 1,
    ffmpeg_threads: int = 0,
    source_task_sequence: tuple[int, ...] | None = None,
    max_sequence_gap_frames: int = 0,
) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} exists; pass --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    pa, pq = require_pyarrow()
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([{"task_index": task_id, "__index_level_0__": task_name}]),
        output_dir / "meta" / "tasks.parquet",
    )

    source_file_segments: dict[str, list[CuratedSegment]] = defaultdict(list)
    for segment in segments:
        source_file_segments[
            data_path_from_episode(episode_rows[segment.source_episode_index])
        ].append(segment)

    stats = {
        key: StatsAccumulator()
        for key, feature in features.items()
        if not (isinstance(feature, dict) and feature.get("dtype") == "video")
    }
    episode_meta: list[dict[str, Any]] = []
    data_buffer: list[dict[str, Any]] = []
    data_location = FileLocation(0, 0)
    data_buffer_bytes = 0
    data_target_bytes = data_files_size_mb * BYTES_PER_MB
    global_index = 0
    fps = float(source_info.get("fps", 30.0))
    chunks_size = int(source_info.get("chunks_size", DEFAULT_CHUNKS_SIZE))

    for source_path in sorted(source_file_segments):
        table = pq.read_table(download(source, source_path))
        rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in table.to_pylist():
            rows_by_episode[int(row["episode_index"])].append(row)
        for segment in source_file_segments[source_path]:
            source_rows = rows_by_episode.get(segment.source_episode_index, [])
            selected_rows = slice_source_rows(source_rows, segment)
            output_episode_index = len(episode_meta)
            source_frame_start_sec = float(selected_rows[0]["timestamp"])
            source_frame_end_sec = float(selected_rows[-1]["timestamp"]) + 1.0 / fps
            output_rows: list[dict[str, Any]] = []
            for frame_index, source_row in enumerate(selected_rows):
                output_row = {
                    key: value for key, value in source_row.items() if key in features
                }
                output_row["timestamp"] = frame_index / fps
                output_row["frame_index"] = frame_index
                output_row["episode_index"] = output_episode_index
                output_row["index"] = global_index + frame_index
                output_row["task_index"] = task_id
                output_rows.append(output_row)

            segment_bytes = estimate_rows_size_bytes(pa, output_rows)
            if data_buffer and data_buffer_bytes + segment_bytes > data_target_bytes:
                write_data_file(pa, pq, output_dir, data_location, data_buffer)
                data_location = next_file_location(data_location, chunks_size)
                data_buffer = []
                data_buffer_bytes = 0
            segment_data_location = data_location
            for output_row in output_rows:
                for key, accumulator in stats.items():
                    if key in output_row:
                        accumulator.update(output_row[key])
            data_buffer.extend(output_rows)
            data_buffer_bytes += segment_bytes

            dataset_from_index = global_index
            global_index += len(output_rows)
            episode_meta.append(
                {
                    "episode_index": output_episode_index,
                    "tasks": [task_name],
                    "length": len(output_rows),
                    "data/chunk_index": segment_data_location.chunk_index,
                    "data/file_index": segment_data_location.file_index,
                    "dataset_from_index": dataset_from_index,
                    "dataset_to_index": global_index,
                    "source_episode_index": segment.source_episode_index,
                    "source_frame_start": segment.source_frame_start,
                    "source_frame_end": segment.source_frame_end,
                    "source_frame_start_sec": source_frame_start_sec,
                    "source_frame_end_sec": source_frame_end_sec,
                    "source_task_index": segment.source_task_index,
                    "curation_verdict": segment.verdict,
                    "curation_failure_category": segment.failure_category,
                    "curation_reviewer": segment.reviewer,
                    "curation_reviewed_at": segment.reviewed_at,
                    "curation_schema_version": segment.schema_version,
                    "curation_components_json": json.dumps(
                        [asdict(component) for component in segment.components],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    if data_buffer:
        write_data_file(pa, pq, output_dir, data_location, data_buffer)
    print(
        f"  data: {len(episode_meta)} episodes, {global_index} frames",
        flush=True,
    )
    if len(episode_meta) != len(segments):
        raise RuntimeError(
            f"materialized {len(episode_meta)} episodes for {len(segments)} segments"
        )

    if video_keys:
        write_segment_videos(
            source_repo_id=source.repo_id,
            source_revision=source.revision,
            output_dir=output_dir,
            episode_meta=episode_meta,
            episode_rows=episode_rows,
            video_keys=video_keys,
            video_mode=video_mode,
            video_files_size_mb=video_files_size_mb,
            chunks_size=chunks_size,
            fps=fps,
            video_workers=video_workers,
            ffmpeg_threads=ffmpeg_threads,
        )

    write_episode_meta(
        pa,
        pq,
        output_dir,
        episode_meta,
        episodes_per_meta_file,
    )
    stats_json = {key: accumulator.to_dict() for key, accumulator in stats.items()}
    for video_key in video_keys:
        stats_json[video_key] = source_stats.get(
            video_key,
            imagenet_video_stats(),
        )
    write_json(output_dir / "meta" / "stats.json", stats_json)
    write_info(
        output_dir=output_dir,
        source_info=source_info,
        features=features,
        total_episodes=len(episode_meta),
        total_frames=global_index,
        chunks_size=chunks_size,
        data_files_size_mb=data_files_size_mb,
        video_files_size_mb=video_files_size_mb,
    )
    write_provenance(
        output_dir=output_dir,
        source=source,
        curation=curation,
        task_id=task_id,
        task_name=task_name,
        allowed_verdicts=allowed_verdicts,
        segments=segments,
        video_keys=video_keys,
        video_mode=video_mode,
        total_frames=global_index,
        video_workers=video_workers,
        ffmpeg_threads=ffmpeg_threads,
        source_task_sequence=source_task_sequence,
        max_sequence_gap_frames=max_sequence_gap_frames,
    )
    write_dataset_card(
        output_dir=output_dir,
        source=source,
        curation=curation,
        task_id=task_id,
        task_name=task_name,
        allowed_verdicts=allowed_verdicts,
        episode_count=len(episode_meta),
        frame_count=global_index,
        video_keys=video_keys,
        video_mode=video_mode,
        source_task_sequence=source_task_sequence,
        max_sequence_gap_frames=max_sequence_gap_frames,
    )


def write_provenance(
    *,
    output_dir: Path,
    source: HubDataset,
    curation: HubDataset,
    task_id: int,
    task_name: str,
    allowed_verdicts: frozenset[str],
    segments: list[CuratedSegment],
    video_keys: list[str],
    video_mode: str,
    total_frames: int,
    video_workers: int,
    ffmpeg_threads: int,
    source_task_sequence: tuple[int, ...] | None = None,
    max_sequence_gap_frames: int = 0,
) -> None:
    write_json(
        output_dir / "meta" / "team_ramen_curation.json",
        {
            "schema_version": "team_ramen_curated_lerobot_v1",
            "source_repo_id": source.repo_id,
            "source_revision": source.revision,
            "curation_repo_id": curation.repo_id,
            "curation_revision": curation.revision,
            "curation_schema_version": SUPPORTED_CURATION_SCHEMA,
            "task_id": task_id,
            "task_name": task_name,
            "selection_mode": (
                "task_sequence" if source_task_sequence else "single_task"
            ),
            "source_task_sequence": (
                list(source_task_sequence) if source_task_sequence else None
            ),
            "max_sequence_gap_frames": (
                max_sequence_gap_frames if source_task_sequence else None
            ),
            "allowed_verdicts": sorted(allowed_verdicts),
            "episode_count": len(segments),
            "frame_count": total_frames,
            "video_keys": video_keys,
            "video_mode": video_mode,
            "video_timestamp_basis": "packed_frame_offset_over_fps",
            "video_workers": video_workers,
            "ffmpeg_threads_per_worker": ffmpeg_threads,
            "segments": [
                {"output_episode_index": index, **asdict(segment)}
                for index, segment in enumerate(segments)
            ],
        },
    )


def write_dataset_card(
    *,
    output_dir: Path,
    source: HubDataset,
    curation: HubDataset,
    task_id: int,
    task_name: str,
    allowed_verdicts: frozenset[str],
    episode_count: int,
    frame_count: int,
    video_keys: list[str],
    video_mode: str,
    source_task_sequence: tuple[int, ...] | None = None,
    max_sequence_gap_frames: int = 0,
) -> None:
    sequence_details = ""
    episode_unit = "curated segment"
    if source_task_sequence:
        episode_unit = "matched task sequence"
        sequence_details = (
            f"- Source task sequence: `{' -> '.join(map(str, source_task_sequence))}`\n"
            f"- Maximum inter-component gap: `{max_sequence_gap_frames}` frame(s)\n"
        )
    text = f"""---
datasets:
- {source.repo_id}
tags:
- robotics
- lerobot
- unitree-g1
- curated
---

# Curated task {task_id}: {task_name}

LeRobot v3 segments selected from
[`{source.repo_id}`](https://huggingface.co/datasets/{source.repo_id}) using
[`{curation.repo_id}`](https://huggingface.co/datasets/{curation.repo_id}).

- Source revision: `{source.revision}`
- Curation revision: `{curation.revision}`
- Task ID: `{task_id}`
{sequence_details}- Accepted verdicts: `{", ".join(sorted(allowed_verdicts))}`
- Episodes (one per {episode_unit}): `{episode_count}`
- Frames: `{frame_count}`
- Video keys: `{", ".join(video_keys) if video_keys else "none"}`
- Video mode: `{video_mode}` (H.264 CRF 0 is pixel-lossless)
- Video timestamp basis: packed-video frame offset divided by FPS

Rows with `verdict=failure` are excluded. Original episode/frame provenance is
stored in `meta/episodes/*.parquet` and `meta/team_ramen_curation.json`.
Frame ranges in the curation source are zero-based and inclusive. Output
`task_index` values are set to the task ID listed above. For sequence datasets,
the original component task IDs and ranges are retained in the provenance.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def validate_output_dataset(
    output_dir: Path,
    *,
    expected_episodes: int,
    expected_frames: int,
    expected_task_id: int,
    video_keys: list[str],
) -> None:
    _, pq = require_pyarrow()
    data_files = sorted((output_dir / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise RuntimeError("output contains no parquet data files")
    rows: list[dict[str, Any]] = []
    columns = ["episode_index", "frame_index", "index", "task_index", "timestamp"]
    for path in data_files:
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    if len(rows) != expected_frames:
        raise RuntimeError(f"output has {len(rows)} frames, expected {expected_frames}")
    if [int(row["index"]) for row in rows] != list(range(expected_frames)):
        raise RuntimeError("output global index is not contiguous")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_index"])].append(row)
        if int(row["task_index"]) != expected_task_id:
            raise RuntimeError(
                f"output task_index must equal configured task ID {expected_task_id}"
            )
    if sorted(grouped) != list(range(expected_episodes)):
        raise RuntimeError("output episode_index is not contiguous")
    for episode_index, episode_rows in grouped.items():
        frames = [int(row["frame_index"]) for row in episode_rows]
        if frames != list(range(len(episode_rows))):
            raise RuntimeError(
                f"output episode {episode_index} frame_index is not contiguous"
            )

    episode_meta_files = sorted(
        (output_dir / "meta" / "episodes").glob("chunk-*/*.parquet")
    )
    episode_meta = [
        row for path in episode_meta_files for row in pq.read_table(path).to_pylist()
    ]
    if len(episode_meta) != expected_episodes:
        raise RuntimeError(
            f"output has {len(episode_meta)} metadata episodes, "
            f"expected {expected_episodes}"
        )
    info = json.loads((output_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    for video_key in video_keys:
        frame_offsets: dict[tuple[int, int], int] = defaultdict(int)
        for row in episode_meta:
            location = (
                int(row[f"videos/{video_key}/chunk_index"]),
                int(row[f"videos/{video_key}/file_index"]),
            )
            frame_offset = frame_offsets[location]
            expected_from = frame_offset / fps
            actual_from = float(row[f"videos/{video_key}/from_timestamp"])
            actual_to = float(row[f"videos/{video_key}/to_timestamp"])
            actual_frame_count = round((actual_to - actual_from) * fps)
            expected_to = (frame_offset + actual_frame_count) / fps
            if (
                actual_frame_count <= 0
                or abs(actual_frame_count - int(row["length"])) > 1
                or not math.isclose(actual_from, expected_from, abs_tol=1e-9)
                or not math.isclose(actual_to, expected_to, abs_tol=1e-9)
            ):
                raise RuntimeError(
                    f"{video_key} episode {row['episode_index']} timestamps are not "
                    "aligned to packed-video frame offsets"
                )
            # write_segment_videos accepts the known BitRobot upstream mismatch of
            # one frame between episode metadata and the source video container.
            # Packed-video offsets therefore advance by the materialized clip's
            # frame count, which is encoded by from/to_timestamp, rather than by
            # the tabular episode length.
            frame_offsets[location] += actual_frame_count
        referenced = {
            (
                int(row[f"videos/{video_key}/chunk_index"]),
                int(row[f"videos/{video_key}/file_index"]),
            )
            for row in episode_meta
        }
        for chunk_index, file_index in referenced:
            path = (
                output_dir
                / "videos"
                / video_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"missing output video: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
