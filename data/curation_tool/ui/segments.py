"""
segments.py — LeRobot v3 dataset の episode → segment 分解 + 4-tile mp4 生成。

- load_metadata / list_episodes: LeRobot meta の薄い wrapper
- get_task_index_per_frame + detect_segments (pure) + get_segments:
  per-frame task_index の連続 run を Segment 化
- get_episode_video_paths: HF hub から 3 camera mp4 を cache
- build_tile_mp4: ffmpeg で 4-tile 2x2 (head_L/R + wrist_L/R) を all-keyframe 出力
  → HTML5 <video> で frame-precise seek 可
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_REPO_ID = "BitRobot/G1_WBT_Dex1_Building-Children-Table"

# G1_WBT_Dex1 の camera key (dogfood で確定、research/02 の推定マッピングと一致):
#   cam_0 = head_L (stereo 左眼), cam_1 = head_R (stereo 右眼)
#   cam_2 = left_wrist RGB,     cam_3 = right_wrist RGB
# いずれも独立 640x480 mono (私が当初推測した「head stereo split 1280x480」ではない)。
CAMERA_HEAD_L = "observation.images.cam_0"
CAMERA_HEAD_R = "observation.images.cam_1"
CAMERA_WRIST_L = "observation.images.cam_2"
CAMERA_WRIST_R = "observation.images.cam_3"

# LeRobot task_index → curation UI 表示用 skill 名 (snake_case)。
# inference/desktop/skill_planner/enter_conditions.py の canonical 6-skill list を
# ベースに命名だけ揃え、tid=5 (rotate_table_base) と tid=7 (move_table_base) は
# curation では個別 label として維持 (inference は 2 つを "move_table_base" に merge
# するが、curation の粒度では区別する)。
LEROBOT_TASK_INDEX_TO_SKILL_NAME: dict[int, str] = {
    0: "insert_table_leg",
    1: "move_to_table",
    2: "flip_table",
    3: "rotate_leg_to_tighten",
    4: "pick_table_leg",
    5: "rotate_table_base",       # curation では個別維持 (inference の move_table_base merge に含めない)
    6: "building_children_table",  # LeRobot 側 intro ラベル
    7: "move_table_base",          # curation では個別維持
}


def get_task_name(task_index: int) -> str:
    """LeRobot task_index → curation 表示用 skill name (unknown は "unknown_N")。"""
    return LEROBOT_TASK_INDEX_TO_SKILL_NAME.get(task_index, f"unknown_{task_index}")

# 4-tile 用 camera key の順序 (build_tile_mp4 で使う)
_TILE_CAMERAS = [
    ("head_L", CAMERA_HEAD_L),
    ("head_R", CAMERA_HEAD_R),
    ("wrist_L", CAMERA_WRIST_L),
    ("wrist_R", CAMERA_WRIST_R),
]

_LEROBOT_PATCHED = False


def _patch_lerobot_version_check() -> None:
    """
    BitRobot/G1_WBT_Dex1 dataset は git tag が無く (branch=main のみ)、
    lerobot 0.6.0 の get_safe_version が RevisionNotFoundError を raise しようとして
    hf_hub 1.23.0 との API 互換 issue で TypeError に化ける。

    使用元 module (dataset_metadata.py) の import 済み参照を直接置換して、
    main branch を revision として返すようにする (dataset は main を直接指定できる)。
    idempotent。
    """
    global _LEROBOT_PATCHED
    if _LEROBOT_PATCHED:
        return
    from lerobot.datasets import dataset_metadata

    dataset_metadata.get_safe_version = lambda repo_id, version: "main"
    _LEROBOT_PATCHED = True


@dataclass
class Segment:
    """1 segment = 同 task_index が連続する frame 範囲 (ep 内)。"""

    episode_id: int
    segment_id: int  # ep 内 0-based
    task_index: int
    frame_start: int  # ep 内 0-based
    frame_end: int  # inclusive
    frame_count: int  # frame_end - frame_start + 1
    task_name: str = ""  # meta から解決した人間可読名 (例: "rotate leg to tighten")


# --- 1. Meta / episodes ---

def load_metadata(repo_id: str = DEFAULT_REPO_ID, root: Optional[Path] = None) -> Any:
    """LeRobot dataset の meta を load (meta files のみ download)。

    BitRobot dataset の git tag 無し issue を回避するため、事前に
    _patch_lerobot_version_check() を呼ぶ。

    Return: LeRobotDatasetMetadata (import は関数内で行い、test 側の依存を最小化)
    """
    _patch_lerobot_version_check()
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id=repo_id, root=root)


def list_episodes(meta: Any) -> list[int]:
    """dataset 内の episode index list (0-based)。"""
    return list(range(meta.total_episodes))


# --- 2. Per-ep task_index → segments ---

def get_task_index_per_frame(meta: Any, episode_id: int) -> list[int]:
    """指定 episode の per-frame task_index (shared parquet を直読 + episode_index filter)。

    LeRobot v3 は複数 ep を 1 parquet file に packed する。
    dataset_from_index/to_index は dataset 全体での位置 (file-局所ではない) なので、
    file-局所 df に対する iloc slice で使うと file-004 等の後方 ep で空になる。
    → episode_index column を filter する (file-局所でも dataset-wide でも正しい)。
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd

    rel = meta.get_data_file_path(episode_id)
    abs_path = hf_hub_download(
        repo_id=meta.repo_id,
        filename=str(rel),
        repo_type="dataset",
    )
    df = pd.read_parquet(abs_path, columns=["episode_index", "task_index"])
    df_ep = df[df["episode_index"] == episode_id]
    if len(df_ep) == 0:
        raise ValueError(
            f"no rows for episode_index={episode_id} in {abs_path}. "
            f"file may have wrong ep coverage or index type mismatch."
        )
    return df_ep["task_index"].tolist()


def detect_segments(task_indices: list[int]) -> list[tuple[int, int, int]]:
    """連続する同 task_index の run を検出 (pure function)。

    Return: list of (task_index, frame_start, frame_end) — frame_end inclusive。
    """
    if not task_indices:
        return []
    result: list[tuple[int, int, int]] = []
    start = 0
    prev = task_indices[0]
    for i in range(1, len(task_indices)):
        if task_indices[i] != prev:
            result.append((prev, start, i - 1))
            start = i
            prev = task_indices[i]
    result.append((prev, start, len(task_indices) - 1))
    return result


def get_segments(meta: Any, episode_id: int) -> list[Segment]:
    """episode の segments を返す (task_index 連続区間 + inference-style task_name)。"""
    task_indices = get_task_index_per_frame(meta, episode_id)
    runs = detect_segments(task_indices)
    return [
        Segment(
            episode_id=episode_id,
            segment_id=idx,
            task_index=t,
            frame_start=s,
            frame_end=e,
            frame_count=e - s + 1,
            task_name=get_task_name(t),
        )
        for idx, (t, s, e) in enumerate(runs)
    ]


# --- 3. Video tile 生成 ---

def get_episode_video_paths(meta: Any, episode_id: int) -> dict[str, Path]:
    """episode の 4 camera mp4 絶対 path (HF cache 済 or download)。

    Return: {"head_L": Path, "head_R": Path, "wrist_L": Path, "wrist_R": Path}
    """
    from huggingface_hub import hf_hub_download

    result: dict[str, Path] = {}
    for label, cam_key in _TILE_CAMERAS:
        rel = meta.get_video_file_path(episode_id, cam_key)
        abs_path = hf_hub_download(
            repo_id=meta.repo_id,
            filename=str(rel),
            repo_type="dataset",
        )
        result[label] = Path(abs_path)
    return result


def get_episode_video_offsets(meta: Any, episode_id: int) -> dict[str, float]:
    """
    episode の 4 camera video 内 from_timestamp (秒) を返す。

    LeRobot v3 は複数 ep を 1 mp4 に concat する場合あり、その ep の video 内開始秒を
    seek offset として ffmpeg に渡す必要がある。
    Return: {"head_L": from_ts_sec, "head_R": ..., "wrist_L": ..., "wrist_R": ...}
    """
    ep_info = meta.episodes[episode_id]
    return {
        label: ep_info[f"videos/{cam_key}/from_timestamp"]
        for label, cam_key in _TILE_CAMERAS
    }


def build_tile_mp4(
    meta: Any,
    segment: Segment,
    output_path: Path,
    fps: Optional[int] = None,
    force: bool = False,
) -> Path:
    """4-tile 2x2 mp4 (head_L + head_R + wrist_L + wrist_R) を build。

    - head は 1280x480 stereo split → 左右 crop で 2 面に分割
    - 2x2 grid = 1280x960、all-keyframe (-g 1) + libx264 で書き出し
    - HTML5 <video> で frame-precise seek 可能
    - cache 済 (output_path.exists()) で force=False なら skip

    Raises:
        FileNotFoundError: ffmpeg が PATH に無い
        subprocess.CalledProcessError: ffmpeg 実行失敗
    """
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError(
            "ffmpeg not found on PATH; required for video tile generation"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        return output_path

    fps_val = fps if fps is not None else meta.fps
    videos = get_episode_video_paths(meta, segment.episode_id)
    offsets = get_episode_video_offsets(meta, segment.episode_id)
    seg_offset_sec = segment.frame_start / fps_val
    duration_sec = segment.frame_count / fps_val

    # filter graph: 4 videos (all 640x480) → 2x2 (1280x960)
    #   [0:head_L][1:head_R] hstack → top row
    #   [2:wrist_L][3:wrist_R] hstack → bottom row
    #   [top][bot] vstack → out
    filter_complex = (
        "[0:v][1:v]hstack[top];"
        "[2:v][3:v]hstack[bot];"
        "[top][bot]vstack[out]"
    )

    cmd = ["ffmpeg", "-y"]
    for label in ("head_L", "head_R", "wrist_L", "wrist_R"):
        # ep 内 offset + segment 内 offset の絶対 seek 時刻
        seek_sec = offsets[label] + seg_offset_sec
        cmd += [
            "-ss", str(seek_sec),
            "-t", str(duration_sec),
            "-i", str(videos[label]),
        ]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-g", "1",  # all-keyframe = HTML5 seek 精度
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # Fix #3 for PR #83 review: ffmpeg 失敗時に stderr を error に載せる
        # (前実装は capture_output=True で stderr 拾っても error msg に含めず debug 不能)
        stderr_tail = (e.stderr or b"").decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(
            f"ffmpeg failed (returncode={e.returncode}) for ep{segment.episode_id} "
            f"seg{segment.segment_id}: {stderr_tail}"
        ) from e
    return output_path


def segment_tile_cache_path(
    workspace: Path,
    episode_id: int,
    segment_id: int,
) -> Path:
    """UI の tile mp4 cache 標準パス。

    Layout: workspace/videos_cache/ep{NNN}/seg{MM}_tile.mp4
    """
    return (
        Path(workspace)
        / "videos_cache"
        / f"ep{episode_id:03d}"
        / f"seg{segment_id:02d}_tile.mp4"
    )
