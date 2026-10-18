from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "scripts" / "build_subtask_datasets.py"
    spec = importlib.util.spec_from_file_location("build_subtask_datasets", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_subtask_config_contract() -> None:
    module = load_builder()
    config = module.load_json(ROOT / "configs" / "subtasks.json")
    specs = module.load_subtask_specs(config)

    assert config["source_repo_id"] == "BitRobot/G1_WBT_Dex1_Building-Children-Table"
    assert config["raw_repo_id"] == "BitRobot/2026-humanoid-ikea-assembly-challenge"
    assert (
        config["target_repo_template"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_{subtask}_1"
    )
    assert [spec.name for spec in specs] == [
        "move_to_work_pose",
        "pick_leg",
        "coarse_insert",
        "final_insert_contact",
        "tighten",
        "rotate_table_base",
        "recover_or_regrasp",
        "flip_table",
    ]


def test_plan_segments_direct_and_split_labels(monkeypatch) -> None:
    module = load_builder()
    specs = module.load_subtask_specs(
        module.load_json(ROOT / "configs" / "subtasks.json")
    )
    raw_info = {
        "episode_name": "episode_test",
        "start_timestamp_ns": 1_000_000_000,
        "end_timestamp_ns": 11_000_000_000,
        "subtasks": [
            {"task": "pick table leg", "timestamp_ns": 2_000_000_000},
            {"task": "insert table leg to table base", "timestamp_ns": 5_000_000_000},
            {"task": "flip table", "timestamp_ns": 9_000_000_000},
        ],
    }

    def fake_hf_download(*, repo_id: str, repo_type: str, filename: str) -> str:
        assert filename == "episode_test/episode_0001/info.json"
        return "/tmp/fake-info.json"

    def fake_load_json(path: Path):
        assert path == Path("/tmp/fake-info.json")
        return raw_info

    monkeypatch.setattr(module, "hf_hub_download", fake_hf_download)
    monkeypatch.setattr(module, "load_json", fake_load_json)
    segments = module.plan_segments(
        raw_repo_id="raw",
        raw_info_files=["episode_test/episode_0001/info.json"],
        episode_rows=[{"episode_index": 0}],
        specs=specs,
        min_duration_sec=0.1,
    )

    assert [(seg.start_sec, seg.end_sec) for seg in segments["pick_leg"]] == [
        (1.0, 4.0)
    ]
    assert [
        (round(seg.start_sec, 3), round(seg.end_sec, 3))
        for seg in segments["coarse_insert"]
    ] == [(4.0, 6.6)]
    assert [
        (round(seg.start_sec, 3), round(seg.end_sec, 3))
        for seg in segments["final_insert_contact"]
    ] == [(6.6, 8.0)]
    assert [(seg.start_sec, seg.end_sec) for seg in segments["flip_table"]] == [
        (8.0, 10.0)
    ]


def test_data_path_from_episode() -> None:
    module = load_builder()
    assert (
        module.data_path_from_episode({"data/chunk_index": 2, "data/file_index": 7})
        == "data/chunk-002/file-007.parquet"
    )


def test_storage_defaults_and_chunk_rollover() -> None:
    module = load_builder()

    assert module.DEFAULT_DATA_FILE_SIZE_MB == 100
    assert module.DEFAULT_VIDEO_FILE_SIZE_MB == 500
    assert module.next_file_location(
        module.FileLocation(0, 998), 1000
    ) == module.FileLocation(0, 999)
    assert module.next_file_location(
        module.FileLocation(0, 999), 1000
    ) == module.FileLocation(1, 0)


def test_write_info_keeps_lerobot_v3_storage_fields(tmp_path: Path) -> None:
    module = load_builder()
    features = {
        "observation.state.robot_q_current": {
            "dtype": "float32",
            "shape": [32],
            "names": None,
        },
        "action.robot_q_desired": {"dtype": "float32", "shape": [32], "names": None},
    }
    module.write_info(
        output_dir=tmp_path,
        source_info={
            "codebase_version": "v3.0",
            "robot_type": "unitree_g1",
            "fps": 30,
            "features": {},
            "dataset_name": "should_not_be_copied",
            "team_ramen": {"subtask": "old"},
        },
        features=features,
        total_episodes=2,
        total_frames=120,
        chunks_size=1000,
        data_files_size_mb=100,
        video_files_size_mb=500,
    )

    info = module.load_json(tmp_path / "meta" / "info.json")
    assert info["codebase_version"] == "v3.0"
    assert info["robot_type"] == "unitree_g1"
    assert info["data_files_size_in_mb"] == 100
    assert info["video_files_size_in_mb"] == 500
    assert info["features"] == features
    assert "dataset_name" not in info
    assert "team_ramen" not in info


def test_parallel_video_clipping_preserves_episode_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_builder()
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"source")
    download_calls: list[str] = []
    clip_threads: list[int] = []
    concat_orders: list[list[str]] = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_download(*, repo_id, repo_type, revision, filename):
        assert repo_id == "source/repo"
        assert repo_type == "dataset"
        assert revision == "source-sha"
        download_calls.append(filename)
        return str(source_video)

    def fake_clip(
        input_path,
        output_path,
        start_sec,
        end_sec,
        mode,
        *,
        ffmpeg_threads=0,
    ):
        nonlocal active, max_active
        assert input_path == source_video
        assert end_sec > start_sec
        assert mode == "reencode"
        clip_threads.append(ffmpeg_threads)
        episode_index = int(output_path.stem.split("-")[-1])
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04 if episode_index == 0 else 0.01)
        output_path.write_bytes(bytes([episode_index + 1]))
        with lock:
            active -= 1

    def fake_concat(
        paths,
        output_path,
        fps,
        *,
        video_mode="reencode",
        ffmpeg_threads=0,
    ):
        assert fps == 30.0
        assert video_mode == "reencode"
        assert ffmpeg_threads == 8
        concat_orders.append([path.name for path in paths])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"".join(path.read_bytes() for path in paths))

    monkeypatch.setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setattr(module, "clip_video", fake_clip)
    monkeypatch.setattr(module, "get_video_frame_count", lambda path: 30)
    monkeypatch.setattr(module, "concat_video_files", fake_concat)

    video_key = "observation.images.cam_0"
    episode_rows = [
        {
            "episode_index": 0,
            f"videos/{video_key}/chunk_index": 0,
            f"videos/{video_key}/file_index": 0,
            f"videos/{video_key}/from_timestamp": 0.0,
            f"videos/{video_key}/to_timestamp": 10.0,
        }
    ]
    episode_meta = [
        {
            "episode_index": index,
            "source_episode_index": 0,
            "source_frame_start_sec": float(index),
            "source_frame_end_sec": float(index) + 0.5,
            "length": 30,
        }
        for index in range(4)
    ]

    files = module.write_segment_videos(
        source_repo_id="source/repo",
        source_revision="source-sha",
        output_dir=tmp_path / "output",
        episode_meta=episode_meta,
        episode_rows=episode_rows,
        video_keys=[video_key],
        video_mode="reencode",
        video_files_size_mb=100,
        chunks_size=1000,
        fps=30.0,
        video_workers=4,
        ffmpeg_threads=2,
    )

    assert files == 1
    assert max_active > 1
    assert clip_threads == [2, 2, 2, 2]
    assert download_calls == [f"videos/{video_key}/chunk-000/file-000.mp4"]
    assert concat_orders == [
        [
            "clip-000000.mp4",
            "clip-000001.mp4",
            "clip-000002.mp4",
            "clip-000003.mp4",
        ]
    ]
    assert [meta[f"videos/{video_key}/from_timestamp"] for meta in episode_meta] == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]


def test_next_camera_download_is_prefetched_while_current_camera_clips(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_builder()
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"source")
    next_download_started = threading.Event()
    current_clip_started = threading.Event()

    def fake_download(*, repo_id, repo_type, revision, filename):
        if "cam_1" in filename:
            next_download_started.set()
            assert current_clip_started.wait(timeout=1.0)
        return str(source_video)

    def fake_clip(
        input_path,
        output_path,
        start_sec,
        end_sec,
        mode,
        *,
        ffmpeg_threads=0,
    ):
        if "cam_0" in str(output_path.parent):
            current_clip_started.set()
            assert next_download_started.wait(timeout=1.0)
        output_path.write_bytes(b"clip")

    def fake_concat(
        paths,
        output_path,
        fps,
        *,
        video_mode="reencode",
        ffmpeg_threads=0,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")

    monkeypatch.setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setattr(module, "clip_video", fake_clip)
    monkeypatch.setattr(module, "get_video_frame_count", lambda path: 15)
    monkeypatch.setattr(module, "concat_video_files", fake_concat)

    video_keys = ["observation.images.cam_0", "observation.images.cam_1"]
    source_episode = {"episode_index": 0}
    for video_key in video_keys:
        source_episode.update(
            {
                f"videos/{video_key}/chunk_index": 0,
                f"videos/{video_key}/file_index": 0,
                f"videos/{video_key}/from_timestamp": 0.0,
                f"videos/{video_key}/to_timestamp": 10.0,
            }
        )
    episode_meta = [
        {
            "episode_index": 0,
            "source_episode_index": 0,
            "source_frame_start_sec": 0.0,
            "source_frame_end_sec": 0.5,
            "length": 15,
        }
    ]

    files = module.write_segment_videos(
        source_repo_id="source/repo",
        source_revision="source-sha",
        output_dir=tmp_path / "output",
        episode_meta=episode_meta,
        episode_rows=[source_episode],
        video_keys=video_keys,
        video_mode="reencode",
        video_files_size_mb=100,
        chunks_size=1000,
        fps=30.0,
        video_workers=1,
        ffmpeg_threads=1,
    )

    assert files == 2
    assert next_download_started.is_set()
    assert current_clip_started.is_set()


def test_lossless_video_mode_uses_x264_crf_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_builder()
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.clip_video(
        tmp_path / "source.mp4",
        tmp_path / "clip.mp4",
        1.0,
        2.0,
        "lossless",
        ffmpeg_threads=2,
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-crf") + 1] == "0"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"


def test_reencode_video_mode_uses_x264_crf_18(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Issue #122: reencode mode は visually-lossless の crf=18 で encode する。

    Source 動画は既に crf~16 の h264 lossy なので crf=0 に上塗りする意味薄く、
    decode コストだけ 4x 重くなる。crf=18 は SSIM>0.986 (visually lossless)、
    decode 4x 高速で GR00T train ETA 48h→~19h に短縮。
    """
    module = load_builder()
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.clip_video(
        tmp_path / "source.mp4",
        tmp_path / "clip.mp4",
        1.0,
        2.0,
        "reencode",
        ffmpeg_threads=2,
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-crf") + 1] == "18"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-preset") + 1] == "veryfast"


def test_lossless_concat_normalizes_timestamps_without_reencoding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_builder()
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is True
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.concat_video_files(
        [tmp_path / "clip-0.mp4", tmp_path / "clip-1.mp4"],
        tmp_path / "packed.mp4",
        30.0,
        video_mode="lossless",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-c") + 1] == "copy"
    assert command[command.index("-bsf:v") + 1] == ("setts=pts=N/(30*TB):dts=N/(30*TB)")


def test_lossless_concat_fallback_is_crf_zero_and_constant_frame_rate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_builder()
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is True
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.concat_video_files_reencode_once(
        [tmp_path / "clip-0.mp4", tmp_path / "clip-1.mp4"],
        tmp_path / "packed.mp4",
        30.0,
        video_mode="lossless",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-crf") + 1] == "0"
    assert "setpts=N/(30*TB)[outv]" in command[command.index("-filter_complex") + 1]


def test_reencode_concat_fallback_is_crf_18(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Issue #122: concat fallback path も crf=18 で統一 (clip_video と揃える)。"""
    module = load_builder()
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is True
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.concat_video_files_reencode_once(
        [tmp_path / "clip-0.mp4", tmp_path / "clip-1.mp4"],
        tmp_path / "packed.mp4",
        30.0,
        video_mode="reencode",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-crf") + 1] == "18"
