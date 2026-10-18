"""Unit tests for model/ramen_ori/obb_cache.py (Issue #122 D-2)。

precompute_yolo_obb.py の出力 (per-mp4 parquet + manifest) を dummy で作り、
ObbPrecomputedCache の manifest 検証 / ep mapping / lookup / wrist zero fill を検証。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from model.ramen_ori.obb_cache import (
    CACHE_MANIFEST_NAME,
    CACHE_SCHEMA_VERSION,
    ObbPrecomputedCache,
)


TOP_K = 4
NUM_CLASSES = 7
FPS = 30
CAMERAS_ALL = [
    "observation.images.cam_0",
    "observation.images.cam_1",
    "observation.images.cam_2",
    "observation.images.cam_3",
]
COVERED_CAMERAS = ["observation.images.cam_0", "observation.images.cam_1"]


def _write_manifest(
    cache_dir: Path,
    ckpt_hash: str,
    covered_cams: list[str] = COVERED_CAMERAS,
    top_k: int = TOP_K,
    num_classes: int = NUM_CLASSES,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": schema_version,
        "yolo_ckpt_hash": ckpt_hash,
        "yolo_ckpt_ref": f"hf:dummy@{ckpt_hash}",
        "yolo_class_names": {str(i): f"c{i}" for i in range(num_classes)},
        "cameras": list(covered_cams),
        "top_k": top_k,
        "imgsz": 800,
        "conf_threshold": 0.15,
        "num_classes": num_classes,
        "generated_at": "2026-08-18T00:00:00Z",
    }
    (cache_dir / CACHE_MANIFEST_NAME).write_text(json.dumps(manifest))


def _write_parquet(
    parquet_path: Path,
    n_frames: int,
    top_k: int = TOP_K,
    dets_per_frame: int = 2,
    class_start: int = 0,
) -> None:
    """decorative parquet: 各 frame に dets_per_frame 個の valid det、残りは padding。

    verts[i][0] = frame_index を埋め込んで dataloader 側 lookup の検証用。
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list] = {
        "frame_index": [],
        "det_idx": [],
        "verts": [],
        "conf": [],
        "class_id": [],
        "valid": [],
    }
    for f in range(n_frames):
        for d in range(top_k):
            rows["frame_index"].append(f)
            rows["det_idx"].append(d)
            if d < dets_per_frame:
                verts = [float(f)] + [float(d)] * 7  # verts[0] = frame, verts[1..7] = det_idx
                rows["verts"].append(verts)
                rows["conf"].append(0.9 - d * 0.1)
                rows["class_id"].append((class_start + d) % NUM_CLASSES)
                rows["valid"].append(True)
            else:
                rows["verts"].append([0.0] * 8)
                rows["conf"].append(0.0)
                rows["class_id"].append(-1)
                rows["valid"].append(False)
    schema = pa.schema(
        [
            pa.field("frame_index", pa.int32(), nullable=False),
            pa.field("det_idx", pa.int8(), nullable=False),
            pa.field("verts", pa.list_(pa.float32(), 8), nullable=False),
            pa.field("conf", pa.float32(), nullable=False),
            pa.field("class_id", pa.int8(), nullable=False),
            pa.field("valid", pa.bool_(), nullable=False),
        ]
    )
    pq.write_table(pa.table(rows, schema=schema), parquet_path, compression="snappy")


def _make_full_cache(tmp_path: Path, ckpt_hash: str = "abc123def456") -> Path:
    """1 mp4 (chunk-000/file-000)、2 cam (cam_0, cam_1)、20 frames の cache を作る。"""
    root = tmp_path / "yolo_obb_cache"
    cache_dir = root / ckpt_hash
    _write_manifest(cache_dir, ckpt_hash)
    for cam in COVERED_CAMERAS:
        _write_parquet(cache_dir / cam / "chunk-000" / "file-000.parquet", n_frames=20)
    return root


class _FakeEpisodesRow(dict):
    """dict-like row (LeRobot meta.episodes iteration の要素)。"""


def _make_ep_meta(episodes: list[dict]) -> list[_FakeEpisodesRow]:
    return [_FakeEpisodesRow(ep) for ep in episodes]


# ============================================================================
# Init / manifest validation
# ============================================================================


def test_init_auto_picks_single_hash_dir(tmp_path: Path) -> None:
    root = _make_full_cache(tmp_path)
    cache = ObbPrecomputedCache(
        root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
    )
    assert cache.ckpt_hash == "abc123def456"
    assert cache.covered_cameras == set(COVERED_CAMERAS)


def test_init_multiple_hash_dirs_requires_explicit(tmp_path: Path) -> None:
    root = tmp_path / "yolo_obb_cache"
    for h in ["hash1abc12345", "hash2xyz98765"]:
        _write_manifest(root / h, ckpt_hash=h)
    with pytest.raises(ValueError, match="multiple hash dirs"):
        ObbPrecomputedCache(
            root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
        )


def test_init_no_manifest_raises(tmp_path: Path) -> None:
    root = tmp_path / "yolo_obb_cache"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="no <hash>/manifest.json"):
        ObbPrecomputedCache(
            root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
        )


def test_init_top_k_mismatch_raises(tmp_path: Path) -> None:
    root = _make_full_cache(tmp_path)
    with pytest.raises(ValueError, match="top_K mismatch"):
        ObbPrecomputedCache(
            root=root, ckpt_hash="abc123def456", camera_keys=CAMERAS_ALL, top_K=8, num_classes=NUM_CLASSES, fps=FPS
        )


def test_init_num_classes_mismatch_raises(tmp_path: Path) -> None:
    root = _make_full_cache(tmp_path)
    with pytest.raises(ValueError, match="num_classes mismatch"):
        ObbPrecomputedCache(
            root=root, ckpt_hash="abc123def456", camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=8, fps=FPS
        )


def test_init_schema_version_mismatch_raises(tmp_path: Path) -> None:
    root = tmp_path / "yolo_obb_cache"
    ckpt_hash = "xyz"
    _write_manifest(root / ckpt_hash, ckpt_hash=ckpt_hash, schema_version="team_ramen_yolo_obb_cache_v0")
    with pytest.raises(ValueError, match="unsupported cache schema_version"):
        ObbPrecomputedCache(
            root=root, ckpt_hash=ckpt_hash, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
        )


def test_init_cache_disjoint_from_dataloader_raises(tmp_path: Path) -> None:
    """Issue #129 Phase K: cache と dataloader の camera set が完全 disjoint なら raise。

    元 test (`test_init_cache_camera_not_in_dataloader_raises`) は「cache に dataloader
    に無い cam があれば raise」だったが、Phase A で 4→3 cam に縮小した場合の superset
    cache を silent 拒否する false positive を招くため、check を「disjoint (intersect
    非空でない)」に緩和。superset / subset は許容。
    """
    root = tmp_path / "yolo_obb_cache"
    ckpt_hash = "xyz"
    _write_manifest(
        root / ckpt_hash,
        ckpt_hash=ckpt_hash,
        covered_cams=["observation.images.unknown"],  # 完全に無関係な cam
    )
    with pytest.raises(ValueError, match="disjoint"):
        ObbPrecomputedCache(
            root=root, ckpt_hash=ckpt_hash, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
        )


def test_init_cache_superset_of_dataloader_is_accepted(tmp_path: Path) -> None:
    """Issue #129 Phase K: cache が dataloader より広い cam 集合を持つ場合 (Phase A
    以前の 4 cam cache に対し dataloader が 3 cam の場合等) は許容、log warn のみ。"""
    root = tmp_path / "yolo_obb_cache"
    ckpt_hash = "xyz"
    dataloader_cams = [c for c in CAMERAS_ALL if not c.endswith(".cam_1")]
    _write_manifest(
        root / ckpt_hash,
        ckpt_hash=ckpt_hash,
        covered_cams=CAMERAS_ALL,
    )
    cache = ObbPrecomputedCache(
        root=root, ckpt_hash=ckpt_hash, camera_keys=dataloader_cams,
        top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS,
    )
    assert set(cache.covered_cameras) == set(CAMERAS_ALL)
    assert cache.camera_keys == dataloader_cams


# ============================================================================
# Episode mapping
# ============================================================================


def test_build_episode_mapping_derives_start_frame_from_timestamp(tmp_path: Path) -> None:
    root = _make_full_cache(tmp_path)
    cache = ObbPrecomputedCache(
        root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
    )
    # ep 0 は mp4 の先頭 (from_ts=0)、ep 1 は 100 frame 目から始まる (from_ts=100/30)
    eps = _make_ep_meta(
        [
            {
                "episode_index": 0,
                "videos/observation.images.cam_0/chunk_index": 0,
                "videos/observation.images.cam_0/file_index": 0,
                "videos/observation.images.cam_0/from_timestamp": 0.0,
                "videos/observation.images.cam_1/chunk_index": 0,
                "videos/observation.images.cam_1/file_index": 0,
                "videos/observation.images.cam_1/from_timestamp": 0.0,
            },
            {
                "episode_index": 1,
                "videos/observation.images.cam_0/chunk_index": 0,
                "videos/observation.images.cam_0/file_index": 0,
                "videos/observation.images.cam_0/from_timestamp": 100.0 / FPS,
                "videos/observation.images.cam_1/chunk_index": 0,
                "videos/observation.images.cam_1/file_index": 0,
                "videos/observation.images.cam_1/from_timestamp": 100.0 / FPS,
            },
        ]
    )
    cache.build_episode_mapping(eps)
    assert cache._ep_mapping[0]["observation.images.cam_0"] == ("chunk-000/file-000", 0)
    assert cache._ep_mapping[1]["observation.images.cam_0"] == ("chunk-000/file-000", 100)


def test_build_episode_mapping_missing_video_column_raises(tmp_path: Path) -> None:
    root = _make_full_cache(tmp_path)
    cache = ObbPrecomputedCache(
        root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
    )
    eps = _make_ep_meta([{"episode_index": 0}])
    with pytest.raises(KeyError, match="missing video column"):
        cache.build_episode_mapping(eps)


# ============================================================================
# lookup_frame / make_obb_tensors
# ============================================================================


def _cache_with_mapping(tmp_path: Path) -> ObbPrecomputedCache:
    root = _make_full_cache(tmp_path)
    cache = ObbPrecomputedCache(
        root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
    )
    eps = _make_ep_meta(
        [
            {
                "episode_index": 0,
                "videos/observation.images.cam_0/chunk_index": 0,
                "videos/observation.images.cam_0/file_index": 0,
                "videos/observation.images.cam_0/from_timestamp": 0.0,
                "videos/observation.images.cam_1/chunk_index": 0,
                "videos/observation.images.cam_1/file_index": 0,
                "videos/observation.images.cam_1/from_timestamp": 0.0,
            },
            {
                "episode_index": 1,
                "videos/observation.images.cam_0/chunk_index": 0,
                "videos/observation.images.cam_0/file_index": 0,
                "videos/observation.images.cam_0/from_timestamp": 5.0 / FPS,
                "videos/observation.images.cam_1/chunk_index": 0,
                "videos/observation.images.cam_1/file_index": 0,
                "videos/observation.images.cam_1/from_timestamp": 5.0 / FPS,
            },
        ]
    )
    cache.build_episode_mapping(eps)
    return cache


def test_lookup_frame_head_cam_returns_precomputed_dets(tmp_path: Path) -> None:
    cache = _cache_with_mapping(tmp_path)
    # ep 0 frame 3, cam_0 → parquet frame 3
    per = cache.lookup_frame(0, 3, "observation.images.cam_0")
    assert per is not None
    # verts[0][0] = 3 (frame), verts[0][1..7] = 0 (det_idx 0)
    np.testing.assert_allclose(per["verts"][0], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert per["valid"][0] is np.True_
    assert per["valid"][1] is np.True_
    # det_idx >= 2 は padding
    assert per["valid"][2] is np.False_
    assert per["valid"][3] is np.False_
    assert per["class_id"][2] == -1


def test_lookup_frame_wrist_cam_returns_none(tmp_path: Path) -> None:
    cache = _cache_with_mapping(tmp_path)
    per = cache.lookup_frame(0, 3, "observation.images.cam_2")
    assert per is None


def test_lookup_frame_start_frame_offset_applied(tmp_path: Path) -> None:
    cache = _cache_with_mapping(tmp_path)
    # ep 1 は from_ts=5/30 → start_frame=5、ep_frame 3 → mp4 frame 8
    per = cache.lookup_frame(1, 3, "observation.images.cam_0")
    assert per is not None
    np.testing.assert_allclose(per["verts"][0][0], 8.0)


def test_lookup_frame_out_of_range_raises(tmp_path: Path) -> None:
    cache = _cache_with_mapping(tmp_path)
    # parquet は 20 frames、ep 1 は start=5 なので ep_frame 15 = mp4 frame 20 (out)
    with pytest.raises(IndexError, match="out of range"):
        cache.lookup_frame(1, 15, "observation.images.cam_0")


def test_lookup_frame_unknown_episode_raises(tmp_path: Path) -> None:
    cache = _cache_with_mapping(tmp_path)
    with pytest.raises(KeyError, match="not in mapping"):
        cache.lookup_frame(99, 0, "observation.images.cam_0")


def test_make_obb_tensors_shape_matches_dummy(tmp_path: Path) -> None:
    cache = _cache_with_mapping(tmp_path)
    cam_ids = torch.arange(len(CAMERAS_ALL), dtype=torch.long)
    out = cache.make_obb_tensors(0, 3, cam_ids)
    assert out["verts"].shape == (len(CAMERAS_ALL), TOP_K, 8)
    assert out["conf"].shape == (len(CAMERAS_ALL), TOP_K, 1)
    assert out["class_id"].shape == (len(CAMERAS_ALL), TOP_K)
    assert out["cam_id"].shape == (len(CAMERAS_ALL), TOP_K)
    assert out["valid_mask"].shape == (len(CAMERAS_ALL), TOP_K)
    assert out["verts"].dtype == torch.float32
    assert out["class_id"].dtype == torch.long
    assert out["valid_mask"].dtype == torch.bool


def test_make_obb_tensors_wrist_zeros_head_precomputed(tmp_path: Path) -> None:
    cache = _cache_with_mapping(tmp_path)
    cam_ids = torch.arange(len(CAMERAS_ALL), dtype=torch.long)
    out = cache.make_obb_tensors(0, 3, cam_ids)
    # cam_0, cam_1 (head): valid det 2 個 + padding 2 個
    for cam_idx in [0, 1]:
        assert out["valid_mask"][cam_idx].sum().item() == 2
        # verts[0][0] は frame_index=3 が入る
        assert out["verts"][cam_idx, 0, 0].item() == pytest.approx(3.0)
    # cam_2, cam_3 (wrist): 全 padding、all zeros
    for cam_idx in [2, 3]:
        assert out["valid_mask"][cam_idx].sum().item() == 0
        assert torch.all(out["verts"][cam_idx] == 0)
        assert torch.all(out["conf"][cam_idx] == 0)
    # cam_id は arange broadcast (top_K 分)
    for cam_idx in range(len(CAMERAS_ALL)):
        assert torch.all(out["cam_id"][cam_idx] == cam_idx)


def test_make_obb_tensors_padding_class_id_clamped_to_zero(tmp_path: Path) -> None:
    """padding row の class_id=-1 は 0 (workspace) にクランプされる (embedding OOB 回避)。

    workspace collision は問題ない: valid_mask=False で Fusion attention が
    完全に filter するので class_embed[0] への gradient は 0。
    """
    cache = _cache_with_mapping(tmp_path)
    cam_ids = torch.arange(len(CAMERAS_ALL), dtype=torch.long)
    out = cache.make_obb_tensors(0, 3, cam_ids)
    # 全 class_id が [0, num_classes) にクランプされている
    assert out["class_id"].min().item() >= 0
    assert out["class_id"].max().item() < NUM_CLASSES


def test_parquet_cache_reused_across_lookups(tmp_path: Path) -> None:
    """同 mp4 の 2 回目 lookup は re-read しない (in-memory cache 効いている)。"""
    cache = _cache_with_mapping(tmp_path)
    cache.lookup_frame(0, 3, "observation.images.cam_0")
    key = ("observation.images.cam_0", "chunk-000/file-000")
    assert key in cache._parquet_cache
    # cache dict をわざと壊しても 2 回目 lookup は cache 経由なので影響なし
    original = cache._parquet_cache[key]
    cache.lookup_frame(0, 5, "observation.images.cam_0")
    assert cache._parquet_cache[key] is original


# ============================================================================
# lookup_mp4_frame (D-3 overlay hook 用 direct entry)
# ============================================================================


def test_lookup_mp4_frame_direct_by_cam_mp4_relative(tmp_path: Path) -> None:
    """mp4 semantic (cam, mp4_relative, frame_in_mp4) から直接 lookup、ep mapping 不要。"""
    root = _make_full_cache(tmp_path)
    cache = ObbPrecomputedCache(
        root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
    )
    # build_episode_mapping せずに直接呼べる
    per = cache.lookup_mp4_frame("observation.images.cam_0", "chunk-000/file-000", 3)
    assert per is not None
    # verts[0][0] = frame_index=3 (dummy parquet marker)
    np.testing.assert_allclose(per["verts"][0][0], 3.0)


def test_lookup_mp4_frame_wrist_cam_returns_none(tmp_path: Path) -> None:
    root = _make_full_cache(tmp_path)
    cache = ObbPrecomputedCache(
        root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
    )
    assert cache.lookup_mp4_frame("observation.images.cam_2", "chunk-000/file-000", 3) is None


def test_lookup_mp4_frame_out_of_range_returns_none(tmp_path: Path) -> None:
    """overlay hook 用: frame 範囲外は None を返す (error にしない、hook が skip)。"""
    root = _make_full_cache(tmp_path)
    cache = ObbPrecomputedCache(
        root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
    )
    # parquet は 20 frames、20 は範囲外
    assert cache.lookup_mp4_frame("observation.images.cam_0", "chunk-000/file-000", 20) is None


def test_lookup_mp4_frame_missing_parquet_returns_none(tmp_path: Path) -> None:
    """指定 mp4 の parquet が無い場合 (未 precompute mp4) → None、hook で skip。"""
    root = _make_full_cache(tmp_path)
    cache = ObbPrecomputedCache(
        root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=TOP_K, num_classes=NUM_CLASSES, fps=FPS
    )
    # 存在しない mp4 relative
    assert cache.lookup_mp4_frame("observation.images.cam_0", "chunk-999/file-999", 0) is None


def test_top_k_none_uses_manifest_and_episode_arrays(tmp_path: Path) -> None:
    """Issue #141 Phase 7: memory は cache にある検出を全部読む (top_K=None)、episode 1 本分をまとめて引く。"""
    root = _make_full_cache(tmp_path)
    cache = ObbPrecomputedCache(
        root=root, ckpt_hash=None, camera_keys=CAMERAS_ALL, top_K=None, num_classes=NUM_CLASSES, fps=FPS
    )
    assert cache.top_K == TOP_K
    cache.build_episode_mapping(_make_ep_meta([
        {"episode_index": 0, **{f"videos/{c}/{k}": v for c in COVERED_CAMERAS
                                for k, v in (("chunk_index", 0), ("file_index", 0), ("from_timestamp", 0.0))}},
        {"episode_index": 1, **{f"videos/{c}/{k}": v for c in COVERED_CAMERAS
                                for k, v in (("chunk_index", 0), ("file_index", 0), ("from_timestamp", 5 / FPS))}},
    ]))
    arr = cache.episode_arrays(1, 10, "observation.images.cam_0")
    assert arr["verts"].shape == (10, TOP_K, 8)
    np.testing.assert_array_equal(arr["verts"][:, 0, 0], np.arange(5, 15))   # verts[0] = parquet の frame 番号
    with pytest.raises(IndexError):
        cache.episode_arrays(1, 16, "observation.images.cam_0")
