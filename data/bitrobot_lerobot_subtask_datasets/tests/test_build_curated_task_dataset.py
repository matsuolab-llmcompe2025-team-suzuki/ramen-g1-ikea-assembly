from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "build_curated_task_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "build_curated_task_dataset",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(
    *,
    episode_id: int = 0,
    frame_start: int = 0,
    frame_end: int = 9,
    task_index: int = 2,
    verdict: str = "success",
) -> dict:
    return {
        "episode_id": episode_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "task_index": task_index,
        "verdict": verdict,
        "failure_category": "slip" if verdict == "failure" else None,
        "reviewer": "hara",
        "reviewed_at": "2026-07-20T00:00:00+09:00",
        "schema_version": 4,
    }


def test_default_video_mode_is_lossless() -> None:
    module = load_builder()
    args = module.parse_args(["--task-id", "0"])
    assert args.video_mode == "lossless"


def test_seg_offset_default_zero() -> None:
    module = load_builder()
    args = module.parse_args(["--task-id", "0"])
    assert args.seg_offset == 0


def test_task_sequence_cli_defaults_to_strict_contiguity() -> None:
    module = load_builder()
    args = module.parse_args(
        ["--task-sequence", "0", "3", "--output-task-id", "8"]
    )
    assert args.task_sequence == [0, 3]
    assert args.output_task_id == 8
    assert args.max_sequence_gap_frames == 0


def test_select_task_sequence_joins_only_contiguous_accepted_components() -> None:
    module = load_builder()
    rows = [
        row(episode_id=0, frame_start=10, frame_end=19, task_index=0, verdict="optimal"),
        row(episode_id=0, frame_start=20, frame_end=39, task_index=3, verdict="optimal"),
        row(episode_id=0, frame_start=50, frame_end=59, task_index=0, verdict="optimal"),
        row(episode_id=0, frame_start=61, frame_end=79, task_index=3, verdict="optimal"),
        row(episode_id=1, frame_start=0, frame_end=9, task_index=0, verdict="optimal"),
        row(episode_id=1, frame_start=10, frame_end=29, task_index=3, verdict="success"),
    ]

    segments, resolutions = module.select_task_sequence_segments(
        rows,
        task_sequence=(0, 3),
        allowed_verdicts=module.normalize_verdicts(["optimal"]),
    )

    assert resolutions == []
    assert len(segments) == 1
    segment = segments[0]
    assert (segment.source_episode_index, segment.source_frame_start, segment.source_frame_end) == (
        0,
        10,
        39,
    )
    assert [component.source_task_index for component in segment.components] == [0, 3]
    assert [component.verdict for component in segment.components] == ["optimal", "optimal"]


def test_seg_offset_with_max_segments_extracts_range() -> None:
    """--seg-offset N --max-segments M で segments[N : N+M] だけ取る (chunk 分割用)。"""
    module = load_builder()
    # 10 seg (task_index=2, verdict=success) を用意
    rows = [row(frame_start=i * 10, frame_end=i * 10 + 9, task_index=2, verdict="success") for i in range(10)]
    # select_segments はそのまま 10 seg 返す
    allowed = module.normalize_verdicts(["success"])
    segments = module.select_segments(rows, task_id=2, allowed_verdicts=allowed)
    assert len(segments) == 10
    # seg_offset=3 max_segments=4 で segments[3:7] を模倣
    offset, limit = 3, 4
    sliced = segments[offset:]
    sliced = sliced[:limit]
    assert len(sliced) == 4
    assert [s.source_frame_start for s in sliced] == [30, 40, 50, 60]


def test_seg_offset_negative_raises() -> None:
    """--seg-offset に負の値を argparse level で reject する (SystemExit 経由)。
    argparse type=_non_negative_int が check、実装の raise を integration で verify。"""
    module = load_builder()
    with pytest.raises(SystemExit):
        module.parse_args(["--task-id", "0", "--seg-offset", "-1"])


def test_default_selection_excludes_only_failure() -> None:
    module = load_builder()
    allowed = module.normalize_verdicts(module.DEFAULT_ALLOWED_VERDICTS)
    segments = module.select_segments(
        [
            row(frame_start=0, verdict="optimal"),
            row(frame_start=10, frame_end=19, verdict="success"),
            row(frame_start=20, frame_end=29, verdict="ambiguous"),
            row(frame_start=30, frame_end=39, verdict="failure"),
            row(frame_start=40, frame_end=49, task_index=3, verdict="success"),
        ],
        task_id=2,
        allowed_verdicts=allowed,
    )

    assert [segment.verdict for segment in segments] == [
        "optimal",
        "success",
        "ambiguous",
    ]
    assert [
        (segment.source_frame_start, segment.source_frame_end) for segment in segments
    ] == [
        (0, 9),
        (10, 19),
        (20, 29),
    ]


def test_failure_cannot_be_explicitly_enabled() -> None:
    module = load_builder()
    with pytest.raises(ValueError, match="non-failure-only"):
        module.normalize_verdicts(["success", "failure"])


def test_segment_validation_checks_overlap_and_source_bounds() -> None:
    module = load_builder()
    episode_rows = [{"episode_index": 0, "length": 100}]

    valid = [
        module.CuratedSegment(0, 0, 9, 2, "success", None, "hara", "now", 4),
        module.CuratedSegment(0, 10, 20, 2, "optimal", None, "hara", "now", 4),
    ]
    module.validate_segments(valid, episode_rows)

    overlapping = [
        module.CuratedSegment(0, 0, 10, 2, "success", None, "hara", "now", 4),
        module.CuratedSegment(0, 10, 20, 2, "success", None, "hara", "now", 4),
    ]
    with pytest.raises(ValueError, match="overlap"):
        module.validate_segments(overlapping, episode_rows)

    outside = [module.CuratedSegment(0, 90, 100, 2, "success", None, "hara", "now", 4)]
    with pytest.raises(ValueError, match="exceeds"):
        module.validate_segments(outside, episode_rows)


def test_slice_source_rows_uses_inclusive_frame_range() -> None:
    module = load_builder()
    source_rows = [{"frame_index": index, "value": index} for index in range(5)]
    segment = module.CuratedSegment(
        0,
        1,
        3,
        2,
        "success",
        None,
        "hara",
        "now",
        4,
    )
    selected = module.slice_source_rows(source_rows, segment)
    assert [item["value"] for item in selected] == [1, 2, 3]


def test_builds_valid_metadata_only_lerobot_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_builder()
    source_parquet = tmp_path / "source.parquet"
    source_rows = [
        {
            "observation.state": [float(index), float(index + 1)],
            "action": [float(index + 2)],
            "timestamp": index / 30.0,
            "frame_index": index,
            "episode_index": 0,
            "index": index,
            "task_index": 2,
        }
        for index in range(4)
    ]
    pq.write_table(pa.Table.from_pylist(source_rows), source_parquet)
    monkeypatch.setattr(module, "download", lambda hub, filename: source_parquet)

    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    source = module.HubDataset("source/repo", "source-sha")
    curation = module.HubDataset("curation/repo", "curation-sha")
    episode_rows = [
        {
            "episode_index": 0,
            "length": 4,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
    ]
    segments = [
        module.CuratedSegment(
            0,
            1,
            3,
            2,
            "success",
            None,
            "hara",
            "now",
            4,
        )
    ]
    output = tmp_path / "output"
    module.build_dataset(
        source=source,
        curation=curation,
        source_info={
            "codebase_version": "v3.0",
            "robot_type": "unitree_g1",
            "fps": 30,
            "features": features,
            "chunks_size": 1000,
        },
        source_stats={},
        episode_rows=episode_rows,
        segments=segments,
        task_id=2,
        task_name="flip table",
        allowed_verdicts=frozenset({"success"}),
        features=features,
        video_keys=[],
        output_dir=output,
        video_mode="reencode",
        data_files_size_mb=100,
        video_files_size_mb=500,
        episodes_per_meta_file=1000,
        force=False,
    )
    module.validate_output_dataset(
        output,
        expected_episodes=1,
        expected_frames=3,
        expected_task_id=2,
        video_keys=[],
    )
    module.validate_existing_output_for_upload(output)

    output_rows = pq.read_table(output / "data/chunk-000/file-000.parquet").to_pylist()
    assert [item["frame_index"] for item in output_rows] == [0, 1, 2]
    assert [item["index"] for item in output_rows] == [0, 1, 2]
    assert [item["task_index"] for item in output_rows] == [2, 2, 2]
    output_tasks = pq.read_table(output / "meta/tasks.parquet").to_pylist()
    assert output_tasks == [{"task_index": 2, "__index_level_0__": "flip table"}]
    assert [item["timestamp"] for item in output_rows] == pytest.approx(
        [0.0, 1 / 30, 2 / 30]
    )

    provenance = json.loads((output / "meta/team_ramen_curation.json").read_text())
    assert provenance["source_revision"] == "source-sha"
    assert provenance["curation_revision"] == "curation-sha"
    assert provenance["segments"][0]["source_frame_start"] == 1
    assert provenance["segments"][0]["source_frame_end"] == 3


def test_validation_accepts_one_extra_packed_video_frame(
    tmp_path: Path,
) -> None:
    module = load_builder()
    output = tmp_path / "output"
    (output / "data/chunk-000").mkdir(parents=True)
    (output / "meta/episodes/chunk-000").mkdir(parents=True)
    video = output / "videos/observation.images.cam_0/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "index": episode_index * 2 + frame_index,
                    "task_index": 3,
                    "timestamp": frame_index / 30,
                }
                for episode_index in range(2)
                for frame_index in range(2)
            ]
        ),
        output / "data/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "length": 2,
                    "videos/observation.images.cam_0/chunk_index": 0,
                    "videos/observation.images.cam_0/file_index": 0,
                    "videos/observation.images.cam_0/from_timestamp": 0.0,
                    "videos/observation.images.cam_0/to_timestamp": 3 / 30,
                },
                {
                    "episode_index": 1,
                    "length": 2,
                    "videos/observation.images.cam_0/chunk_index": 0,
                    "videos/observation.images.cam_0/file_index": 0,
                    "videos/observation.images.cam_0/from_timestamp": 3 / 30,
                    "videos/observation.images.cam_0/to_timestamp": 5 / 30,
                },
            ]
        ),
        output / "meta/episodes/chunk-000/file-000.parquet",
    )
    (output / "meta/info.json").write_text('{"fps": 30}\n')

    module.validate_output_dataset(
        output,
        expected_episodes=2,
        expected_frames=4,
        expected_task_id=3,
        video_keys=["observation.images.cam_0"],
    )


class FakeHubApi:
    def __init__(self, *, exists: bool, remote_files: list[str] | None = None) -> None:
        self.exists = exists
        self.remote_files = remote_files or []
        self.create_calls: list[dict] = []
        self.upload_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    def repo_exists(self, **kwargs):
        assert kwargs == {
            "repo_id": "Team-RAMEN/curated-task-2",
            "repo_type": "dataset",
        }
        return self.exists

    def create_repo(self, **kwargs):
        self.create_calls.append(kwargs)

    def upload_large_folder(self, **kwargs):
        self.upload_calls.append(kwargs)

    def list_repo_files(self, **kwargs):
        assert kwargs == {
            "repo_id": "Team-RAMEN/curated-task-2",
            "repo_type": "dataset",
        }
        return self.remote_files

    def delete_files(self, **kwargs):
        self.delete_calls.append(kwargs)


def make_upload_directory(root: Path) -> Path:
    output = root / "dataset"
    (output / "meta").mkdir(parents=True)
    (output / "data/chunk-000").mkdir(parents=True)
    (output / "README.md").write_text("# Dataset\n")
    (output / "meta/info.json").write_text("{}\n")
    (output / "meta/team_ramen_curation.json").write_text("{}\n")
    (output / "data/chunk-000/file-000.parquet").write_bytes(b"parquet")
    return output


def test_upload_creates_private_dataset_repo_and_uses_resumable_upload(
    tmp_path: Path,
) -> None:
    module = load_builder()
    output = make_upload_directory(tmp_path)
    api = FakeHubApi(exists=False)

    module.upload_curated_dataset(
        api,
        output_dir=output,
        repo_id="Team-RAMEN/curated-task-2",
        private=True,
        replace_remote=False,
        num_workers=3,
    )

    assert api.create_calls == [
        {
            "repo_id": "Team-RAMEN/curated-task-2",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": False,
        }
    ]
    assert api.upload_calls == [
        {
            "repo_id": "Team-RAMEN/curated-task-2",
            "repo_type": "dataset",
            "folder_path": output,
            "private": True,
            "ignore_patterns": [".cache/**"],
            "num_workers": 3,
        }
    ]
    assert api.delete_calls == []


def test_upload_refuses_existing_repo_without_explicit_replace(
    tmp_path: Path,
) -> None:
    module = load_builder()
    output = make_upload_directory(tmp_path)
    api = FakeHubApi(exists=True)

    with pytest.raises(FileExistsError, match="--replace-remote"):
        module.upload_curated_dataset(
            api,
            output_dir=output,
            repo_id="Team-RAMEN/curated-task-2",
            private=True,
            replace_remote=False,
            num_workers=4,
        )

    assert api.upload_calls == []
    assert api.delete_calls == []


def test_replace_remote_removes_only_stale_dataset_files_after_upload(
    tmp_path: Path,
) -> None:
    module = load_builder()
    output = make_upload_directory(tmp_path)
    api = FakeHubApi(
        exists=True,
        remote_files=[
            "README.md",
            "meta/info.json",
            "meta/team_ramen_curation.json",
            "data/chunk-000/file-000.parquet",
            "data/chunk-999/old.parquet",
            "videos/observation.images.cam_0/old.mp4",
            "LICENSE",
            ".gitattributes",
        ],
    )

    module.upload_curated_dataset(
        api,
        output_dir=output,
        repo_id="Team-RAMEN/curated-task-2",
        private=True,
        replace_remote=True,
        num_workers=4,
    )

    assert api.create_calls == []
    assert len(api.upload_calls) == 1
    assert api.delete_calls == [
        {
            "repo_id": "Team-RAMEN/curated-task-2",
            "repo_type": "dataset",
            "delete_patterns": [
                "data/chunk-999/old.parquet",
                "videos/observation.images.cam_0/old.mp4",
            ],
            "commit_message": "Remove stale files from curated LeRobot dataset",
        }
    ]


def test_upload_cli_safety_options_require_a_target_repo() -> None:
    module = load_builder()
    with pytest.raises(ValueError, match="--public"):
        module.validate_upload_options(
            Namespace(
                upload_repo_id=None,
                public=True,
                replace_remote=False,
                upload_existing=False,
            )
        )
    with pytest.raises(ValueError, match="--replace-remote"):
        module.validate_upload_options(
            Namespace(
                upload_repo_id=None,
                public=False,
                replace_remote=True,
                upload_existing=False,
            )
        )


def test_upload_existing_requires_output_and_repo() -> None:
    module = load_builder()
    with pytest.raises(ValueError, match="--upload-repo-id"):
        module.validate_upload_options(
            Namespace(
                upload_repo_id=None,
                public=False,
                replace_remote=False,
                upload_existing=True,
                output_dir=Path("/tmp/dataset"),
                dry_run=False,
            )
        )
    with pytest.raises(ValueError, match="--output-dir"):
        module.validate_upload_options(
            Namespace(
                upload_repo_id="Team-RAMEN/curated-task-2",
                public=False,
                replace_remote=False,
                upload_existing=True,
                output_dir=None,
                dry_run=False,
            )
        )


def _seg(module, ep: int, fs: int, fe: int, tid: int = 7) -> object:
    """Helper: CuratedSegment shortcut for cross-task filter tests."""
    return module.CuratedSegment(ep, fs, fe, tid, "optimal", None, "hara", "now", 4)


def test_move_table_base_filter_keeps_only_pre_pick_segments() -> None:
    """task 7 seg のうち pick より frame_start が小さいものだけ残る。"""
    module = load_builder()
    task7 = [_seg(module, 0, 0, 9), _seg(module, 0, 50, 59)]
    rows = [
        row(episode_id=0, frame_start=0, frame_end=9, task_index=7, verdict="optimal"),
        row(episode_id=0, frame_start=30, frame_end=39, task_index=4, verdict="optimal"),
        row(episode_id=0, frame_start=50, frame_end=59, task_index=7, verdict="optimal"),
    ]
    kept = module.apply_move_table_base_cross_task_filter(task7, rows)
    assert [(s.source_frame_start, s.source_frame_end) for s in kept] == [(0, 9)]


def test_move_table_base_filter_passes_all_when_no_pick_in_episode() -> None:
    """pick 無し ep は制約発動せず task 7 seg 全部通す。"""
    module = load_builder()
    task7 = [_seg(module, 0, 0, 9), _seg(module, 0, 50, 59)]
    rows = [
        row(episode_id=0, frame_start=0, frame_end=9, task_index=7, verdict="optimal"),
        row(episode_id=0, frame_start=50, frame_end=59, task_index=7, verdict="optimal"),
    ]
    kept = module.apply_move_table_base_cross_task_filter(task7, rows)
    assert len(kept) == 2


def test_move_table_base_filter_uses_min_frame_start_when_multiple_picks() -> None:
    """pick が複数ある場合、min(frame_start) を比較基準にする。"""
    module = load_builder()
    task7 = [_seg(module, 0, 0, 9), _seg(module, 0, 15, 19), _seg(module, 0, 25, 29)]
    rows = [
        row(episode_id=0, frame_start=0, frame_end=9, task_index=7, verdict="optimal"),
        row(episode_id=0, frame_start=15, frame_end=19, task_index=7, verdict="optimal"),
        row(episode_id=0, frame_start=20, frame_end=24, task_index=4, verdict="optimal"),
        row(episode_id=0, frame_start=25, frame_end=29, task_index=7, verdict="optimal"),
        row(episode_id=0, frame_start=40, frame_end=44, task_index=4, verdict="optimal"),
    ]
    kept = module.apply_move_table_base_cross_task_filter(task7, rows)
    assert [(s.source_frame_start, s.source_frame_end) for s in kept] == [(0, 9), (15, 19)]


def test_move_table_base_filter_isolates_per_episode() -> None:
    """ep 別に独立して処理される (ep 0 は filter 発動、ep 1 は pick 無しで素通し)。"""
    module = load_builder()
    task7 = [
        _seg(module, 0, 0, 9),
        _seg(module, 0, 30, 39),
        _seg(module, 1, 5, 14),
        _seg(module, 1, 60, 69),
    ]
    rows = [
        row(episode_id=0, frame_start=0, frame_end=9, task_index=7, verdict="optimal"),
        row(episode_id=0, frame_start=20, frame_end=24, task_index=4, verdict="optimal"),
        row(episode_id=0, frame_start=30, frame_end=39, task_index=7, verdict="optimal"),
        row(episode_id=1, frame_start=5, frame_end=14, task_index=7, verdict="optimal"),
        row(episode_id=1, frame_start=60, frame_end=69, task_index=7, verdict="optimal"),
    ]
    kept = module.apply_move_table_base_cross_task_filter(task7, rows)
    assert [(s.source_episode_index, s.source_frame_start) for s in kept] == [
        (0, 0),
        (1, 5),
        (1, 60),
    ]


def test_clamp_segments_drops_when_frame_end_exceeds_container_length() -> None:
    """curation label frame_end が container 実長を超える seg を drop (BitRobot 上流 不整合)。"""
    module = load_builder()
    seg_ok = module.CuratedSegment(0, 0, 100, 2, "success", None, "hara", "now", 4)
    seg_over = module.CuratedSegment(1, 0, 1000, 2, "success", None, "hara", "now", 4)
    episode_rows = [
        {
            "episode_index": 0, "length": 200,
            "videos/cam_0/from_timestamp": 0.0, "videos/cam_0/to_timestamp": 200 / 30,
        },
        {
            "episode_index": 1, "length": 500,
            "videos/cam_0/from_timestamp": 0.0, "videos/cam_0/to_timestamp": 500 / 30,
        },
    ]
    kept, dropped = module.clamp_segments_to_container_length(
        [seg_ok, seg_over], episode_rows, ["cam_0"], 30.0
    )
    assert [s.source_episode_index for s in kept] == [0]
    assert len(dropped) == 1
    assert dropped[0][0].source_episode_index == 1
    assert dropped[0][1] == 500


def test_clamp_segments_uses_min_container_length_across_cams() -> None:
    """複数 cam で container 実長が異なる場合、最短を採用 (安全側)。"""
    module = load_builder()
    seg = module.CuratedSegment(0, 0, 199, 2, "success", None, "hara", "now", 4)
    episode_rows = [{
        "episode_index": 0, "length": 200,
        "videos/cam_0/from_timestamp": 0.0, "videos/cam_0/to_timestamp": 200 / 30,
        "videos/cam_1/from_timestamp": 0.0, "videos/cam_1/to_timestamp": 199 / 30,
    }]
    kept, dropped = module.clamp_segments_to_container_length(
        [seg], episode_rows, ["cam_0", "cam_1"], 30.0
    )
    assert len(kept) == 0
    assert len(dropped) == 1
    assert dropped[0][1] == 199


def test_clamp_segments_passthrough_when_video_keys_empty() -> None:
    """--skip-videos の場合 (video_keys 空) は preflight skip、全 seg 保持。"""
    module = load_builder()
    seg = module.CuratedSegment(0, 0, 100, 2, "success", None, "hara", "now", 4)
    kept, dropped = module.clamp_segments_to_container_length(
        [seg], [{"episode_index": 0, "length": 200}], [], 30.0
    )
    assert len(kept) == 1
    assert len(dropped) == 0


def _seg2(module, ep: int, fs: int, fe: int) -> object:
    """Helper: task 2 (flip_table) CuratedSegment shortcut."""
    return module.CuratedSegment(ep, fs, fe, 2, "success", None, "hara", "now", 4)


def test_flip_table_filter_keeps_only_latest_frame_start_per_episode() -> None:
    """ep 内 3 seg: frame_start 最大の 1 seg のみ残る (= 実 flip 動作)。"""
    module = load_builder()
    segs = [_seg2(module, 0, 100, 199), _seg2(module, 0, 500, 599), _seg2(module, 0, 300, 399)]
    kept = module.apply_flip_table_per_episode_filter(segs)
    assert len(kept) == 1
    assert kept[0].source_frame_start == 500


def test_flip_table_filter_single_segment_passthrough() -> None:
    """ep 内 1 seg のみ: そのまま残る。"""
    module = load_builder()
    segs = [_seg2(module, 0, 500, 599)]
    kept = module.apply_flip_table_per_episode_filter(segs)
    assert len(kept) == 1
    assert kept[0].source_frame_start == 500


def test_flip_table_filter_isolates_per_episode() -> None:
    """ep 別に独立、各 ep で frame_start 最大の 1 seg のみ残る。"""
    module = load_builder()
    segs = [
        _seg2(module, 0, 100, 199),
        _seg2(module, 0, 500, 599),
        _seg2(module, 1, 200, 299),
        _seg2(module, 1, 800, 899),
        _seg2(module, 2, 50, 149),
    ]
    kept = module.apply_flip_table_per_episode_filter(segs)
    kept_pairs = sorted((s.source_episode_index, s.source_frame_start) for s in kept)
    assert kept_pairs == [(0, 500), (1, 800), (2, 50)]


def _seg5(module, ep: int, fs: int, fe: int) -> object:
    """Helper: task 5 (rotate_table_base) CuratedSegment shortcut."""
    return module.CuratedSegment(ep, fs, fe, 5, "optimal", None, "hara", "now", 4)


def test_rotate_table_base_filter_keeps_only_post_pick_segments() -> None:
    """pick 有 ep: pick より frame_start 以降の task 5 seg のみ残る。"""
    module = load_builder()
    task5 = [_seg5(module, 0, 5, 9), _seg5(module, 0, 60, 69)]
    rows = [
        row(episode_id=0, frame_start=5, frame_end=9, task_index=5, verdict="optimal"),
        row(episode_id=0, frame_start=30, frame_end=39, task_index=4, verdict="optimal"),
        row(episode_id=0, frame_start=60, frame_end=69, task_index=5, verdict="optimal"),
    ]
    kept = module.apply_rotate_table_base_cross_task_filter(task5, rows)
    assert [(s.source_frame_start, s.source_frame_end) for s in kept] == [(60, 69)]


def test_rotate_table_base_filter_falls_back_to_insert_when_no_pick() -> None:
    """pick 無し + insert 有: 1 回目 insert 以降の task 5 seg のみ残る。"""
    module = load_builder()
    task5 = [_seg5(module, 0, 5, 9), _seg5(module, 0, 60, 69)]
    rows = [
        row(episode_id=0, frame_start=5, frame_end=9, task_index=5, verdict="optimal"),
        row(episode_id=0, frame_start=40, frame_end=49, task_index=0, verdict="optimal"),
        row(episode_id=0, frame_start=60, frame_end=69, task_index=5, verdict="optimal"),
    ]
    kept = module.apply_rotate_table_base_cross_task_filter(task5, rows)
    assert [(s.source_frame_start, s.source_frame_end) for s in kept] == [(60, 69)]


def test_rotate_table_base_filter_falls_back_to_rotate_when_no_pick_no_insert() -> None:
    """pick 無し + insert 無し + rotate 有: 1 回目 rotate 以降のみ残る。"""
    module = load_builder()
    task5 = [_seg5(module, 0, 5, 9), _seg5(module, 0, 60, 69)]
    rows = [
        row(episode_id=0, frame_start=5, frame_end=9, task_index=5, verdict="optimal"),
        row(episode_id=0, frame_start=45, frame_end=54, task_index=3, verdict="optimal"),
        row(episode_id=0, frame_start=60, frame_end=69, task_index=5, verdict="optimal"),
    ]
    kept = module.apply_rotate_table_base_cross_task_filter(task5, rows)
    assert [(s.source_frame_start, s.source_frame_end) for s in kept] == [(60, 69)]


def test_rotate_table_base_filter_prefers_earlier_between_insert_and_rotate() -> None:
    """pick 無し + insert + rotate 両有: 早い方 (min) を基準にする。"""
    module = load_builder()
    task5 = [_seg5(module, 0, 30, 39), _seg5(module, 0, 55, 64)]
    rows = [
        # ep 0: insert 1 回目 = 40, rotate 1 回目 = 50 → min=40 を基準、40 以降のみ残る
        row(episode_id=0, frame_start=30, frame_end=39, task_index=5, verdict="optimal"),
        row(episode_id=0, frame_start=40, frame_end=44, task_index=0, verdict="optimal"),
        row(episode_id=0, frame_start=50, frame_end=54, task_index=3, verdict="optimal"),
        row(episode_id=0, frame_start=55, frame_end=64, task_index=5, verdict="optimal"),
    ]
    kept = module.apply_rotate_table_base_cross_task_filter(task5, rows)
    assert [(s.source_frame_start, s.source_frame_end) for s in kept] == [(55, 64)]


def test_rotate_table_base_filter_passes_all_when_no_reference() -> None:
    """pick / insert / rotate 全部無し ep は素通し。"""
    module = load_builder()
    task5 = [_seg5(module, 0, 5, 9), _seg5(module, 0, 60, 69)]
    rows = [
        row(episode_id=0, frame_start=5, frame_end=9, task_index=5, verdict="optimal"),
        row(episode_id=0, frame_start=20, frame_end=29, task_index=7, verdict="optimal"),
        row(episode_id=0, frame_start=60, frame_end=69, task_index=5, verdict="optimal"),
    ]
    kept = module.apply_rotate_table_base_cross_task_filter(task5, rows)
    assert len(kept) == 2


def test_resolve_overlaps_keeps_newer_reviewed_at() -> None:
    """overlap ペア: reviewed_at 新しい方が残り、古い方 drop (ep 99 の HARA 操作ミス実例)。"""
    module = load_builder()
    older = module.CuratedSegment(
        99, 5267, 6446, 3, "success", None, "hara",
        "2026-07-23T20:30:24+09:00", 4,
    )
    newer = module.CuratedSegment(
        99, 5361, 6446, 3, "success", None, "hara",
        "2026-07-23T20:30:43+09:00", 4,
    )
    kept, resolutions = module.resolve_overlaps_by_reviewed_at([older, newer])
    assert len(kept) == 1
    assert kept[0].source_frame_start == 5361
    assert len(resolutions) == 1
    assert resolutions[0][0].source_frame_start == 5361  # kept
    assert resolutions[0][1].source_frame_start == 5267  # dropped


def test_resolve_overlaps_no_change_when_no_overlap() -> None:
    """overlap 無し (隣接含む): 全部残り、resolution 0 個。"""
    module = load_builder()
    a = module.CuratedSegment(0, 0, 9, 3, "success", None, "hara", "2026-07-20", 4)
    b = module.CuratedSegment(0, 10, 19, 3, "success", None, "hara", "2026-07-21", 4)
    kept, resolutions = module.resolve_overlaps_by_reviewed_at([a, b])
    assert len(kept) == 2
    assert resolutions == []


def test_resolve_overlaps_chain_three_reduces_to_newest() -> None:
    """3 個連鎖 overlap: 反復適用で最新の 1 個に収束。"""
    module = load_builder()
    old = module.CuratedSegment(0, 0, 30, 3, "success", None, "hara", "2026-07-20", 4)
    mid = module.CuratedSegment(0, 5, 40, 3, "success", None, "hara", "2026-07-21", 4)
    new = module.CuratedSegment(0, 10, 50, 3, "success", None, "hara", "2026-07-22", 4)
    kept, resolutions = module.resolve_overlaps_by_reviewed_at([old, mid, new])
    assert len(kept) == 1
    assert kept[0].reviewed_at == "2026-07-22"
    assert len(resolutions) == 2


def test_resolve_overlaps_isolates_per_episode() -> None:
    """ep 別に独立して resolve される。"""
    module = load_builder()
    a1 = module.CuratedSegment(0, 0, 30, 3, "success", None, "hara", "2026-07-20", 4)
    a2 = module.CuratedSegment(0, 15, 40, 3, "success", None, "hara", "2026-07-21", 4)
    b = module.CuratedSegment(1, 0, 30, 3, "success", None, "hara", "2026-07-22", 4)
    kept, _ = module.resolve_overlaps_by_reviewed_at([a1, a2, b])
    kept_pairs = sorted((s.source_episode_index, s.source_frame_start) for s in kept)
    assert kept_pairs == [(0, 15), (1, 0)]


def test_rotate_table_base_filter_uses_first_occurrence_of_each() -> None:
    """pick / insert / rotate 各々複数ある時、1 回目 (最小 frame_start) を採る。"""
    module = load_builder()
    task5 = [_seg5(module, 0, 15, 19), _seg5(module, 0, 35, 39)]
    rows = [
        # ep 0: pick 1 回目 = 20 (2 回目 50 は無視), 20 以降のみ残る
        row(episode_id=0, frame_start=15, frame_end=19, task_index=5, verdict="optimal"),
        row(episode_id=0, frame_start=20, frame_end=24, task_index=4, verdict="optimal"),
        row(episode_id=0, frame_start=35, frame_end=39, task_index=5, verdict="optimal"),
        row(episode_id=0, frame_start=50, frame_end=54, task_index=4, verdict="optimal"),
    ]
    kept = module.apply_rotate_table_base_cross_task_filter(task5, rows)
    assert [(s.source_frame_start, s.source_frame_end) for s in kept] == [(35, 39)]
