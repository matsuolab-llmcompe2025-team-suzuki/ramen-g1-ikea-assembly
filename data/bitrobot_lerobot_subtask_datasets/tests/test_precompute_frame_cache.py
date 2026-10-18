"""precompute_frame_cache.py の unit test (Issue #122)。

# 対象
- _list_video_keys / _iter_mp4_files / cache_dir_for_mp4: pure helpers
- build_tasks: LeRobot v3 structure discovery
- _decode_one_mp4: 実 mp4 (cv2 で write) → JPG per-frame 出力、resume-safe

# 依存
opencv-python (VideoWriter で fake mp4 生成 + VideoCapture で decode)、
pyarrow は使わない (info.json は plain JSON)。
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_precompute():
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "precompute_frame_cache", scripts / "precompute_frame_cache.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


precompute = load_precompute()


def _make_fake_mp4(path: Path, num_frames: int = 10, size: tuple[int, int] = (32, 32)) -> None:
    """opencv で小さい mp4 を書く (test fixture)。

    Frame content: frame i は 全画素 (i, i, i) の gray → 復元時に frame idx を pixel から取れる。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    if not writer.isOpened():
        pytest.skip(f"cv2 VideoWriter cannot write mp4 (no ffmpeg?): {path}")
    for i in range(num_frames):
        img = np.full((size[1], size[0], 3), fill_value=i, dtype=np.uint8)
        writer.write(img)
    writer.release()


def _make_fake_lerobot_root(
    tmp: Path, video_keys: list[str], n_mp4_per_cam: int = 1, frames_per_mp4: int = 5
) -> Path:
    root = tmp / "fake_lerobot"
    (root / "meta").mkdir(parents=True)
    info = {
        "fps": 30.0,
        "features": {vk: {"dtype": "video", "shape": [32, 32, 3]} for vk in video_keys},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    for vk in video_keys:
        for f_idx in range(n_mp4_per_cam):
            mp4_path = root / "videos" / vk / "chunk-000" / f"file-{f_idx:03d}.mp4"
            _make_fake_mp4(mp4_path, num_frames=frames_per_mp4)
    return root


# ============================================================================
# _list_video_keys / _iter_mp4_files / cache_dir_for_mp4 (pure)
# ============================================================================


class TestListVideoKeys:
    def test_picks_video_dtype_only(self):
        info = {
            "features": {
                "observation.images.cam_0": {"dtype": "video", "shape": [480, 640, 3]},
                "observation.state.joint": {"dtype": "float32", "shape": [19]},
                "observation.images.cam_1": {"dtype": "video", "shape": [480, 640, 3]},
                "action.joint": {"dtype": "float32", "shape": [19]},
            }
        }
        assert precompute._list_video_keys(info) == [
            "observation.images.cam_0",
            "observation.images.cam_1",
        ]

    def test_returns_sorted(self):
        info = {
            "features": {
                f"observation.images.cam_{i}": {"dtype": "video"}
                for i in [3, 0, 2, 1]
            }
        }
        assert precompute._list_video_keys(info) == [
            f"observation.images.cam_{i}" for i in range(4)
        ]

    def test_no_video_returns_empty(self):
        info = {"features": {"x": {"dtype": "float32"}}}
        assert precompute._list_video_keys(info) == []


class TestCacheDirLayout:
    def test_layout_mirrors_mp4_path(self):
        cache_root = Path("/tmp/cache")
        # mp4 relative to <lerobot>/videos/<vk>/
        cd = precompute.cache_dir_for_mp4(
            cache_root, "observation.images.cam_0", Path("chunk-000/file-002.mp4")
        )
        assert cd == cache_root / "observation.images.cam_0" / "chunk-000" / "file-002"

    def test_multi_chunk(self):
        cache_root = Path("/tmp/c")
        cd = precompute.cache_dir_for_mp4(
            cache_root, "cam_ir", Path("chunk-003/file-010.mp4")
        )
        assert cd == cache_root / "cam_ir" / "chunk-003" / "file-010"


class TestIterMp4Files:
    def test_finds_all_cams_and_files(self, tmp_path):
        root = _make_fake_lerobot_root(
            tmp_path,
            video_keys=["observation.images.cam_0", "observation.images.cam_1"],
            n_mp4_per_cam=2,
        )
        result = precompute._iter_mp4_files(
            root, ["observation.images.cam_0", "observation.images.cam_1"]
        )
        assert len(result) == 4  # 2 cams x 2 mp4s
        # 各 cam 分ずつ
        cams = {vk for vk, _ in result}
        assert cams == {"observation.images.cam_0", "observation.images.cam_1"}

    def test_missing_video_dir_returns_empty_for_that_key(self, tmp_path):
        root = _make_fake_lerobot_root(
            tmp_path, video_keys=["observation.images.cam_0"], n_mp4_per_cam=1
        )
        # cam_1 の dir 存在しないが引数で指定
        result = precompute._iter_mp4_files(
            root, ["observation.images.cam_0", "observation.images.cam_1"]
        )
        cams = {vk for vk, _ in result}
        assert cams == {"observation.images.cam_0"}  # cam_1 は無視


# ============================================================================
# build_tasks (LeRobot v3 structure discovery)
# ============================================================================


class TestBuildTasks:
    def test_returns_tasks_and_fps(self, tmp_path):
        root = _make_fake_lerobot_root(
            tmp_path,
            video_keys=["observation.images.cam_0", "observation.images.cam_1"],
            n_mp4_per_cam=1,
        )
        cache_root = tmp_path / "cache"
        tasks, fps, video_keys = precompute.build_tasks(
            root, cache_root, quality=90, force=False
        )
        assert fps == 30.0
        assert video_keys == ["observation.images.cam_0", "observation.images.cam_1"]
        assert len(tasks) == 2
        for mp4_path, cd, q, force in tasks:
            assert mp4_path.suffix == ".mp4"
            assert cd.parent.parent.name in video_keys
            assert q == 90
            assert force is False


# ============================================================================
# _decode_one_mp4 (real cv2 decode + JPG write)
# ============================================================================


class TestDecodeOneMp4:
    def test_writes_expected_jpg_files(self, tmp_path):
        mp4_path = tmp_path / "src.mp4"
        _make_fake_mp4(mp4_path, num_frames=5)
        cache_dir = tmp_path / "cache"
        cd, saved, skipped, total = precompute._decode_one_mp4(
            (mp4_path, cache_dir, 90, False)
        )
        assert cd == cache_dir
        assert total == 5
        assert saved == 5
        assert skipped == 0
        for i in range(5):
            assert (cache_dir / f"frame_{i:06d}.jpg").is_file()
        assert (cache_dir / precompute.FRAME_COUNT_NAME).read_text() == "5"

    def test_resume_skips_existing(self, tmp_path):
        """既 JPG がある場合は force=False で skip、再走で saved=0。"""
        mp4_path = tmp_path / "src.mp4"
        _make_fake_mp4(mp4_path, num_frames=4)
        cache_dir = tmp_path / "cache"
        # 1 回目
        precompute._decode_one_mp4((mp4_path, cache_dir, 90, False))
        # 2 回目 (再走)
        cd, saved, skipped, total = precompute._decode_one_mp4(
            (mp4_path, cache_dir, 90, False)
        )
        assert saved == 0
        assert skipped == 4
        assert total == 4

    def test_force_overwrites(self, tmp_path):
        mp4_path = tmp_path / "src.mp4"
        _make_fake_mp4(mp4_path, num_frames=3)
        cache_dir = tmp_path / "cache"
        precompute._decode_one_mp4((mp4_path, cache_dir, 90, False))
        cd, saved, skipped, total = precompute._decode_one_mp4(
            (mp4_path, cache_dir, 90, True)
        )
        assert saved == 3
        assert skipped == 0

    def test_frames_are_readable_back_as_jpg(self, tmp_path):
        mp4_path = tmp_path / "src.mp4"
        _make_fake_mp4(mp4_path, num_frames=3)
        cache_dir = tmp_path / "cache"
        precompute._decode_one_mp4((mp4_path, cache_dir, 90, False))
        for i in range(3):
            img = cv2.imread(str(cache_dir / f"frame_{i:06d}.jpg"))
            assert img is not None
            assert img.shape == (32, 32, 3)

    def test_missing_mp4_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="failed to open"):
            precompute._decode_one_mp4(
                (tmp_path / "nonexistent.mp4", tmp_path / "cache", 90, False)
            )


# ============================================================================
# write_cache_meta (cache metadata)
# ============================================================================


class TestWriteCacheMeta:
    def test_writes_meta_json(self, tmp_path):
        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        h = precompute.compute_settings_hash(
            fps=30.0, video_keys=["observation.images.cam_0"], jpg_quality=90
        )
        meta_path = precompute.write_cache_meta(
            cache_root,
            lerobot_root=tmp_path / "src",
            fps=30.0,
            video_keys=["observation.images.cam_0"],
            jpg_quality=90,
            settings_hash=h,
        )
        assert meta_path == cache_root / precompute.CACHE_META_NAME
        meta = json.loads(meta_path.read_text())
        assert meta["schema_version"] == precompute.CACHE_SCHEMA_VERSION
        assert meta["fps"] == 30.0
        assert meta["video_keys"] == ["observation.images.cam_0"]
        assert meta["jpg_quality"] == 90
        assert meta["source_root"] == str(tmp_path / "src")
        assert meta["settings_hash"] == h


# ============================================================================
# compute_settings_hash / read_existing_cache_hash / wipe_cache (Issue #122 rev)
# ============================================================================


class TestComputeSettingsHash:
    def test_deterministic(self):
        h1 = precompute.compute_settings_hash(fps=30.0, video_keys=["a", "b"], jpg_quality=90)
        h2 = precompute.compute_settings_hash(fps=30.0, video_keys=["a", "b"], jpg_quality=90)
        assert h1 == h2

    def test_video_keys_order_matters(self):
        h1 = precompute.compute_settings_hash(fps=30.0, video_keys=["a", "b"], jpg_quality=90)
        h2 = precompute.compute_settings_hash(fps=30.0, video_keys=["b", "a"], jpg_quality=90)
        assert h1 != h2  # order 変化 = 別 cache 認識

    def test_fps_change_invalidates(self):
        h1 = precompute.compute_settings_hash(fps=30.0, video_keys=["a"], jpg_quality=90)
        h2 = precompute.compute_settings_hash(fps=15.0, video_keys=["a"], jpg_quality=90)
        assert h1 != h2

    def test_quality_change_invalidates(self):
        h1 = precompute.compute_settings_hash(fps=30.0, video_keys=["a"], jpg_quality=90)
        h2 = precompute.compute_settings_hash(fps=30.0, video_keys=["a"], jpg_quality=85)
        assert h1 != h2


class TestReadExistingCacheHash:
    def test_returns_stored_hash(self, tmp_path):
        (tmp_path / precompute.CACHE_META_NAME).write_text(
            json.dumps({"settings_hash": "abc123", "fps": 30.0})
        )
        assert precompute.read_existing_cache_hash(tmp_path) == "abc123"

    def test_missing_meta_returns_none(self, tmp_path):
        assert precompute.read_existing_cache_hash(tmp_path) is None

    def test_malformed_meta_returns_none(self, tmp_path):
        (tmp_path / precompute.CACHE_META_NAME).write_text("{malformed")
        assert precompute.read_existing_cache_hash(tmp_path) is None


class TestWipeCache:
    def test_removes_content_keeps_root(self, tmp_path):
        # 事前に少しファイル作る
        (tmp_path / precompute.CACHE_META_NAME).write_text("{}")
        (tmp_path / "cam_0").mkdir()
        (tmp_path / "cam_0" / "chunk-000").mkdir()
        (tmp_path / "cam_0" / "chunk-000" / "file-000").mkdir()
        (tmp_path / "cam_0" / "chunk-000" / "file-000" / "frame_000000.jpg").write_bytes(b"x")
        precompute.wipe_cache(tmp_path)
        assert tmp_path.is_dir()  # root 自体は残す
        assert list(tmp_path.iterdir()) == []  # 中身 空


# ============================================================================
# verify_mp4_cache_complete (per-mp4 verify、Issue #122 rev)
# ============================================================================


class TestVerifyMp4CacheComplete:
    def test_passes_when_complete(self, tmp_path):
        mp4_path = tmp_path / "src.mp4"
        _make_fake_mp4(mp4_path, num_frames=4)
        cache_dir = tmp_path / "cache"
        precompute._decode_one_mp4((mp4_path, cache_dir, 90, False))
        ok, reason = precompute.verify_mp4_cache_complete(mp4_path, cache_dir)
        assert ok, f"expected pass but got: {reason}"

    def test_fails_when_marker_missing(self, tmp_path):
        mp4_path = tmp_path / "src.mp4"
        _make_fake_mp4(mp4_path, num_frames=3)
        cache_dir = tmp_path / "cache"
        precompute._decode_one_mp4((mp4_path, cache_dir, 90, False))
        (cache_dir / precompute.FRAME_COUNT_NAME).unlink()
        ok, reason = precompute.verify_mp4_cache_complete(mp4_path, cache_dir)
        assert not ok
        assert "欠" in reason

    def test_fails_when_jpg_deleted(self, tmp_path):
        mp4_path = tmp_path / "src.mp4"
        _make_fake_mp4(mp4_path, num_frames=4)
        cache_dir = tmp_path / "cache"
        precompute._decode_one_mp4((mp4_path, cache_dir, 90, False))
        # 中 1 個消す → JPG count mismatch or hole
        (cache_dir / "frame_000002.jpg").unlink()
        ok, reason = precompute.verify_mp4_cache_complete(mp4_path, cache_dir)
        assert not ok

    def test_fails_when_marker_count_wrong(self, tmp_path):
        mp4_path = tmp_path / "src.mp4"
        _make_fake_mp4(mp4_path, num_frames=3)
        cache_dir = tmp_path / "cache"
        precompute._decode_one_mp4((mp4_path, cache_dir, 90, False))
        (cache_dir / precompute.FRAME_COUNT_NAME).write_text("999")
        ok, reason = precompute.verify_mp4_cache_complete(mp4_path, cache_dir)
        assert not ok


# ============================================================================
# drop_source_mp4s (auto mp4 drop、Issue #122 rev)
# ============================================================================


class TestDropSourceMp4s:
    def test_drops_real_files(self, tmp_path):
        mp4_1 = tmp_path / "v1.mp4"
        mp4_2 = tmp_path / "v2.mp4"
        _make_fake_mp4(mp4_1, num_frames=2)
        _make_fake_mp4(mp4_2, num_frames=2)
        assert mp4_1.exists() and mp4_2.exists()
        tasks = [(mp4_1, tmp_path / "c1", 90, False), (mp4_2, tmp_path / "c2", 90, False)]
        dropped, bytes_freed = precompute.drop_source_mp4s(tasks)
        assert dropped == 2
        assert bytes_freed > 0
        assert not mp4_1.exists()
        assert not mp4_2.exists()

    def test_dedupes_via_resolve(self, tmp_path):
        """同じ real file を指す symlink 2 個 → drop は 1 回だけ (real の 1 unlink)。"""
        real = tmp_path / "real.mp4"
        _make_fake_mp4(real, num_frames=2)
        sym1 = tmp_path / "sym1.mp4"
        sym2 = tmp_path / "sym2.mp4"
        sym1.symlink_to(real)
        sym2.symlink_to(real)
        tasks = [
            (sym1, tmp_path / "c1", 90, False),
            (sym2, tmp_path / "c2", 90, False),
        ]
        dropped, _ = precompute.drop_source_mp4s(tasks)
        assert dropped == 1  # real 1 個削除
        assert not real.exists()

    def test_missing_file_skipped(self, tmp_path):
        mp4 = tmp_path / "missing.mp4"
        tasks = [(mp4, tmp_path / "c1", 90, False)]
        dropped, bytes_freed = precompute.drop_source_mp4s(tasks)
        assert dropped == 0
        assert bytes_freed == 0


# ============================================================================
# --video-keys / --keep-source-mp4 (Issue #139)
# ============================================================================


class TestOnlyVideoKeys:
    def test_filters_to_requested_keys(self, tmp_path):
        keys = ["observation.images.cam_0", "observation.images.cam_1", "observation.images.cam_2"]
        root = _make_fake_lerobot_root(tmp_path, video_keys=keys)
        tasks, _, video_keys = precompute.build_tasks(
            root,
            tmp_path / "cache",
            quality=90,
            force=False,
            only_video_keys=["observation.images.cam_2", "observation.images.cam_0"],
        )
        assert video_keys == ["observation.images.cam_0", "observation.images.cam_2"]
        assert {t[1].parts[-3] for t in tasks} == set(video_keys)

    def test_unknown_key_raises(self, tmp_path):
        root = _make_fake_lerobot_root(tmp_path, video_keys=["observation.images.cam_0"])
        with pytest.raises(ValueError, match="cam_9"):
            precompute.build_tasks(
                root,
                tmp_path / "cache",
                quality=90,
                force=False,
                only_video_keys=["observation.images.cam_9"],
            )


class TestMainKeepSourceMp4:
    def _run(self, monkeypatch, root, cache, *extra):
        monkeypatch.setattr(
            sys,
            "argv",
            ["precompute_frame_cache.py", "--lerobot-root", str(root), "--cache-root", str(cache),
             "--jobs", "1", *extra],
        )
        return precompute.main()

    def test_keep_source_mp4_leaves_videos(self, tmp_path, monkeypatch):
        keys = ["observation.images.cam_0", "observation.images.cam_1"]
        root = _make_fake_lerobot_root(tmp_path, video_keys=keys, frames_per_mp4=3)
        cache = tmp_path / "cache"
        rc = self._run(
            monkeypatch, root, cache, "--video-keys", "observation.images.cam_0", "--keep-source-mp4"
        )
        assert rc == 0
        assert not list(cache.rglob("*.tmp"))  # atomic write の tmp が残らない
        assert (root / "videos" / keys[0] / "chunk-000" / "file-000.mp4").exists()
        assert (cache / keys[0] / "chunk-000" / "file-000" / "_frame_count.txt").is_file()
        # 指定していない cam は展開しない
        assert not (cache / keys[1]).exists()
        meta = json.loads((cache / "_cache_meta.json").read_text(encoding="utf-8"))
        assert meta["video_keys"] == [keys[0]]

    def test_default_still_drops_source_mp4(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, video_keys=["observation.images.cam_0"])
        rc = self._run(monkeypatch, root, tmp_path / "cache")
        assert rc == 0
        assert not (root / "videos" / "observation.images.cam_0" / "chunk-000" / "file-000.mp4").exists()


class TestMaxFramesPerMp4:
    def test_decodes_prefix_only_and_keeps_mp4(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, video_keys=["observation.images.cam_0"], frames_per_mp4=6)
        cache = tmp_path / "cache"
        monkeypatch.setattr(
            sys, "argv",
            ["precompute_frame_cache.py", "--lerobot-root", str(root), "--cache-root", str(cache),
             "--jobs", "1", "--max-frames-per-mp4", "2"],
        )
        assert precompute.main() == 0
        d = cache / "observation.images.cam_0" / "chunk-000" / "file-000"
        assert sorted(p.name for p in d.glob("*.jpg")) == ["frame_000000.jpg", "frame_000001.jpg"]
        assert (d / "_frame_count.txt").read_text() == "2"
        assert (root / "videos" / "observation.images.cam_0" / "chunk-000" / "file-000.mp4").exists()


def _run_main(monkeypatch, root, *extra):
    monkeypatch.setattr(sys, "argv", [
        "precompute_frame_cache.py", "--lerobot-root", str(root), "--jobs", "1", *extra,
    ])
    return precompute.main()


def _png_task(mp4_path, cache_dir, *, max_frames=None, compression=1):
    return (mp4_path, cache_dir, 90, False, max_frames, "png", compression, 30.0)


class TestPngSettings:
    def test_jpeg_hash_is_byte_identical_to_v1(self):
        payload = {"schema_version": "team_ramen_frame_cache_v1", "fps": 30.0,
                   "video_keys": ["cam"], "jpg_quality": 90}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert precompute.compute_settings_hash(fps=30, video_keys=["cam"], jpg_quality=90) == expected
        assert precompute.compute_settings_hash(fps=30, video_keys=["cam"], jpg_quality=90,
                                                image_format="jpg", png_compression=9) == expected

    def test_png_hash_includes_format_compression_decoder_but_not_jpeg_quality(self):
        settings = dict(fps=30.0, video_keys=["cam"], image_format="png")
        png = precompute.compute_settings_hash(**settings)
        assert png != precompute.compute_settings_hash(fps=30, video_keys=["cam"])
        assert png != precompute.compute_settings_hash(**settings, png_compression=9)
        assert png != precompute.compute_settings_hash(**settings, decoder="other")
        assert png == precompute.compute_settings_hash(**settings, jpg_quality=10)

    @pytest.mark.parametrize("compression", [-1, 10, 1.5])
    def test_invalid_compression_rejected(self, compression):
        with pytest.raises(ValueError, match="0..9"):
            precompute.compute_settings_hash(fps=30, video_keys=["cam"], image_format="png",
                                             png_compression=compression)

    @pytest.mark.parametrize("compression", ["-1", "10", "1.5"])
    def test_cli_invalid_compression_rejected_before_io(self, tmp_path, monkeypatch, compression):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, tmp_path, "--image-format", "png", "--png-compression", compression)
        assert exc.value.code == 2
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
    def test_invalid_png_fps_rejected(self, fps):
        with pytest.raises(ValueError, match="fps"):
            precompute.compute_settings_hash(fps=fps, video_keys=["cam"], image_format="png")

    def test_png_metadata_defaults_to_incomplete(self, tmp_path):
        precompute.write_cache_meta(tmp_path, lerobot_root=tmp_path / "src", fps=30,
                                    video_keys=["cam"], jpg_quality=90, settings_hash="abc",
                                    image_format="png")
        meta = json.loads((tmp_path / precompute.CACHE_META_NAME).read_text())
        assert meta["schema_version"] == "team_ramen_frame_cache_v2"
        assert meta["image_format"] == "png"
        assert meta["png_compression"] == 1
        assert meta["decoder"] == "pyav"
        assert meta["complete"] is False
        assert meta["pending_settings_hash"] == "abc"
        assert "settings_hash" not in meta
        assert precompute.read_existing_cache_hash(tmp_path) is None

    @pytest.mark.parametrize("complete", [None, False, 0, 1, "true", "missing"])
    def test_png_hash_requires_literal_complete_true(self, tmp_path, complete):
        meta = {"image_format": "png", "settings_hash": "abc"}
        if complete != "missing":
            meta["complete"] = complete
        (tmp_path / precompute.CACHE_META_NAME).write_text(json.dumps(meta))
        assert precompute.read_existing_cache_hash(tmp_path) is None


class TestPngDecode:
    @pytest.fixture(autouse=True)
    def _require_pyav(self):
        pytest.importorskip("av")

    def test_decoder_uses_one_thread_per_worker(self, tmp_path, monkeypatch):
        import av

        mp4_path = tmp_path / "mock.mp4"
        mp4_path.write_bytes(b"mock video")
        stream = SimpleNamespace(frames=1, average_rate=Fraction(30), thread_count=0)
        frame = SimpleNamespace(
            pts=0, time_base=Fraction(1, 30),
            to_ndarray=lambda format: np.zeros((4, 4, 3), dtype=np.uint8),
        )

        def decode(selected_stream):
            assert selected_stream is stream
            assert stream.thread_count == 1
            return iter([frame])

        container = SimpleNamespace(streams=SimpleNamespace(video=[stream]), decode=decode)
        monkeypatch.setattr(av, "open", lambda _: nullcontext(container))
        _, saved, skipped, total = precompute._decode_one_mp4(
            _png_task(mp4_path, tmp_path / "png")
        )
        assert (saved, skipped, total) == (1, 0, 1)

    @pytest.mark.parametrize("compression", [0, 1, 9])
    def test_rgb_round_trip_matches_pyav_bit_exactly(self, tmp_path, compression):
        import av

        mp4_path = tmp_path / "color.mp4"
        writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32))
        if not writer.isOpened():
            pytest.skip("MP4 encoder unavailable")
        rng = np.random.default_rng(168)
        for _ in range(5):
            writer.write(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        writer.release()
        cache_dir = tmp_path / "png"
        _, saved, skipped, total = precompute._decode_one_mp4(_png_task(mp4_path, cache_dir, compression=compression))
        assert (saved, skipped, total) == (5, 0, 5)
        with av.open(str(mp4_path)) as container:
            expected = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        for index, rgb in enumerate(expected):
            actual = cv2.cvtColor(cv2.imread(str(cache_dir / f"frame_{index:06d}.png")), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(actual, rgb)
        assert not list(cache_dir.glob("*.jpg"))
        assert precompute.verify_mp4_cache_complete(mp4_path, cache_dir, image_format="png",
                                                    fps=30, png_compression=compression) == (True, "ok")

    def test_png_resume_repairs_invalid_existing_file(self, tmp_path):
        mp4_path = tmp_path / "source.mp4"
        _make_fake_mp4(mp4_path, num_frames=3)
        cache_dir = tmp_path / "png"
        task = _png_task(mp4_path, cache_dir)
        precompute._decode_one_mp4(task)
        (cache_dir / "frame_000001.png").write_bytes(b"broken")
        _, saved, skipped, total = precompute._decode_one_mp4(task)
        assert (saved, skipped, total) == (1, 2, 3)

    @pytest.mark.parametrize("pts,declared_count,rate,error", [
        ([0, 2, 3], 3, 30, "timestamp"),
        ([1, 2, 3], 3, 30, "timestamp"),
        ([0, 0, 1], 3, 30, "timestamp"),
        ([0, None, 2], 3, 30, "timestamp"),
        ([0, 1], 3, 30, "frame count"),
        ([0, 1, 2], 0, 30, "frame count"),
        ([0, 1, 2], 3, 25, "fps"),
    ])
    def test_rejects_unmappable_or_incomplete_video(self, tmp_path, monkeypatch, pts, declared_count, rate, error):
        import av

        mp4_path = tmp_path / "mock.mp4"
        mp4_path.write_bytes(b"mock video")
        cache_dir = tmp_path / "png"
        cache_dir.mkdir()
        (cache_dir / precompute.FRAME_COUNT_NAME).write_text("999")
        (cache_dir / precompute.DECODE_COMPLETE_NAME).write_text("{}")
        frames = [SimpleNamespace(pts=pt, time_base=Fraction(1, 30),
                                  to_ndarray=lambda format: np.zeros((4, 4, 3), dtype=np.uint8)) for pt in pts]
        stream = SimpleNamespace(frames=declared_count, average_rate=Fraction(rate))
        container = SimpleNamespace(streams=SimpleNamespace(video=[stream]), decode=lambda _: iter(frames))
        monkeypatch.setattr(av, "open", lambda _: nullcontext(container))
        with pytest.raises(RuntimeError, match=error):
            precompute._decode_one_mp4(_png_task(mp4_path, cache_dir))
        assert not (cache_dir / precompute.FRAME_COUNT_NAME).exists()
        assert not (cache_dir / precompute.DECODE_COMPLETE_NAME).exists()

    def test_wrong_extension_is_not_accepted(self, tmp_path):
        mp4_path = tmp_path / "source.mp4"
        _make_fake_mp4(mp4_path, num_frames=2)
        cache_dir = tmp_path / "png"
        precompute._decode_one_mp4(_png_task(mp4_path, cache_dir))
        (cache_dir / "frame_000000.jpg").write_bytes(b"jpeg")
        ok, reason = precompute.verify_mp4_cache_complete(mp4_path, cache_dir, image_format="png", fps=30)
        assert not ok
        assert "unexpected jpg" in reason


class TestPngMain:
    @pytest.fixture(autouse=True)
    def _require_pyav(self):
        pytest.importorskip("av")

    def test_default_png_root_keeps_sources_and_other_cache(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=3)
        legacy = root / "frame_cache"
        legacy.mkdir()
        (legacy / "keep.txt").write_text("unchanged")
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        assert len(list((root / "videos").rglob("*.mp4"))) == 1
        assert (legacy / "keep.txt").read_text() == "unchanged"
        cache = root / "frame_cache_png"
        meta = json.loads((cache / precompute.CACHE_META_NAME).read_text())
        assert meta["complete"] is True
        assert "settings_hash" in meta and "pending_settings_hash" not in meta
        assert len(list(cache.rglob("*.png"))) == 3
        before = (cache / precompute.CACHE_META_NAME).stat().st_mtime_ns
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        assert (cache / precompute.CACHE_META_NAME).stat().st_mtime_ns == before

    def test_smoke_then_full_resumes_without_false_hash_hit(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=5)
        cache = root / "frame_cache_png"
        assert _run_main(monkeypatch, root, "--image-format", "png", "--max-frames-per-mp4", "2") == 0
        assert len(list(cache.rglob("*.png"))) == 2
        assert precompute.read_existing_cache_hash(cache) is None
        assert not list(cache.rglob(precompute.FRAME_COUNT_NAME))
        first = next(cache.rglob("frame_000000.png"))
        before = first.stat().st_mtime_ns
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        assert len(list(cache.rglob("*.png"))) == 5
        assert first.stat().st_mtime_ns == before
        assert precompute.read_existing_cache_hash(cache)

    def test_interrupted_write_can_resume(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=4)
        original = cv2.imencode
        calls = 0

        def interrupt(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return original(*args)

        monkeypatch.setattr(cv2, "imencode", interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            _run_main(monkeypatch, root, "--image-format", "png")
        cache = root / "frame_cache_png"
        assert precompute.read_existing_cache_hash(cache) is None
        assert len(list(cache.rglob("*.png"))) == 1
        monkeypatch.setattr(cv2, "imencode", original)
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        assert len(list(cache.rglob("*.png"))) == 4

    def test_complete_hash_does_not_hide_missing_frames(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=3)
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        cache = root / "frame_cache_png"
        missing = next(cache.rglob("frame_000001.png"))
        missing.unlink()
        assert precompute.read_existing_cache_hash(cache)
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        assert missing.is_file()

    def test_missing_png_completion_flag_is_republished(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=2)
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        meta_path = root / "frame_cache_png" / precompute.CACHE_META_NAME
        meta = json.loads(meta_path.read_text())
        del meta["complete"]
        meta_path.write_text(json.dumps(meta))
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        assert json.loads(meta_path.read_text())["complete"] is True

    @pytest.mark.parametrize("remaining_bytes", [8, 40])
    def test_complete_hash_does_not_hide_truncated_png(self, tmp_path, monkeypatch, remaining_bytes):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=3)
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        cache = root / "frame_cache_png"
        frame = next(cache.rglob("frame_000001.png"))
        original = frame.read_bytes()
        frame.write_bytes(original[:remaining_bytes])
        assert precompute.read_existing_cache_hash(cache)
        source = next((root / "videos").rglob("*.mp4"))
        ok, reason = precompute.verify_mp4_cache_complete(source, frame.parent, image_format="png", fps=30)
        assert not ok and "decode failed" in reason
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        assert frame.read_bytes() == original

    def test_png_verification_uses_bounded_worker_pool(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], n_mp4_per_cam=2, frames_per_mp4=2)
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        tasks, fps, _ = precompute.build_tasks(root, root / "frame_cache_png", quality=90,
                                                force=False, image_format="png")
        worker_counts = []

        class FakePool:
            def __init__(self, workers):
                worker_counts.append(workers)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def imap_unordered(self, function, items):
                return map(function, items)

        monkeypatch.setattr(precompute.mp, "Pool", FakePool)
        assert precompute._verify_tasks(tasks, image_format="png", fps=fps, png_compression=1, jobs=8) == []
        assert worker_counts == [2]

    def test_parallel_cli_generation_and_hash_hit(self, tmp_path):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], n_mp4_per_cam=2, frames_per_mp4=3)
        command = [sys.executable, str(ROOT / "scripts" / "precompute_frame_cache.py"),
                   "--lerobot-root", str(root), "--image-format", "png", "--jobs", "2"]
        first = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert first.returncode == 0, first.stdout + first.stderr
        assert "verify passed (2 mp4 all match)" in first.stdout
        cache = root / "frame_cache_png"
        assert len(list(cache.rglob("*.png"))) == 6
        meta_path = cache / precompute.CACHE_META_NAME
        before = meta_path.stat().st_mtime_ns
        second = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert second.returncode == 0, second.stdout + second.stderr
        assert "cache is up-to-date" in second.stdout
        assert meta_path.stat().st_mtime_ns == before
        assert len(list((root / "videos").rglob("*.mp4"))) == 2

    def test_settings_mismatch_requires_force(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=2)
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        meta_path = root / "frame_cache_png" / precompute.CACHE_META_NAME
        before = meta_path.read_bytes()
        with pytest.raises(RuntimeError, match="settings differ"):
            _run_main(monkeypatch, root, "--image-format", "png", "--png-compression", "9")
        assert meta_path.read_bytes() == before
        assert _run_main(monkeypatch, root, "--image-format", "png", "--png-compression", "9", "--force") == 0
        assert json.loads(meta_path.read_text())["png_compression"] == 9
        assert list((root / "videos").rglob("*.mp4"))

    def test_source_signature_change_revalidates_cache(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=2)
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        source = next((root / "videos").rglob("*.mp4"))
        cache_dir = root / "frame_cache_png" / "cam" / "chunk-000" / "file-000"
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        ok, reason = precompute.verify_mp4_cache_complete(source, cache_dir, image_format="png", fps=30)
        assert not ok and "source/settings" in reason
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        assert precompute.verify_mp4_cache_complete(source, cache_dir, image_format="png", fps=30) == (True, "ok")

    def test_source_inventory_change_requires_explicit_rebuild(self, tmp_path, monkeypatch):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=2)
        assert _run_main(monkeypatch, root, "--image-format", "png") == 0
        _make_fake_mp4(root / "videos" / "cam" / "chunk-000" / "file-001.mp4", num_frames=2)
        with pytest.raises(RuntimeError, match="inventory changed"):
            _run_main(monkeypatch, root, "--image-format", "png")

    @pytest.mark.parametrize("initial,target", [("jpg", "png"), ("png", "jpg")])
    def test_cross_format_reuse_requires_force(self, tmp_path, monkeypatch, initial, target):
        root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=2)
        cache = tmp_path / "shared"
        options = ("--cache-root", str(cache), "--keep-source-mp4")
        assert _run_main(monkeypatch, root, *options, "--image-format", initial) == 0
        before = (cache / precompute.CACHE_META_NAME).read_bytes()
        with pytest.raises(RuntimeError, match="image format differs"):
            _run_main(monkeypatch, root, *options, "--image-format", target)
        assert (cache / precompute.CACHE_META_NAME).read_bytes() == before
        assert _run_main(monkeypatch, root, *options, "--image-format", target, "--force") == 0
        assert list(cache.rglob(f"*.{target}"))
        assert not list(cache.rglob(f"*.{initial}"))


def test_jpeg_smoke_then_full_does_not_skip(monkeypatch, tmp_path):
    root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=5)
    assert _run_main(monkeypatch, root, "--max-frames-per-mp4", "2", "--keep-source-mp4") == 0
    cache = root / "frame_cache"
    assert precompute.read_existing_cache_hash(cache) is None
    assert _run_main(monkeypatch, root, "--keep-source-mp4") == 0
    assert len(list(cache.rglob("*.jpg"))) == 5


def test_legacy_early_published_jpeg_hash_does_not_hide_smoke_prefix(monkeypatch, tmp_path):
    root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=5)
    cache = root / "frame_cache"
    tasks, fps, keys = precompute.build_tasks(root, cache, quality=90, force=False)
    precompute._decode_one_mp4((*tasks[0], 2))
    legacy = {"fps": fps, "video_keys": keys, "jpg_quality": 90,
              "schema_version": precompute.CACHE_SCHEMA_VERSION,
              "settings_hash": precompute.compute_settings_hash(fps=fps, video_keys=keys, jpg_quality=90)}
    (cache / precompute.CACHE_META_NAME).write_text(json.dumps(legacy))
    assert _run_main(monkeypatch, root, "--keep-source-mp4") == 0
    assert len(list(cache.rglob("*.jpg"))) == 5


def test_legacy_jpeg_hash_hit_with_dropped_symlink_sources(monkeypatch, tmp_path):
    root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=2)
    source = next((root / "videos").rglob("*.mp4"))
    real = tmp_path / "real.mp4"
    source.rename(real)
    source.symlink_to(real)
    assert _run_main(monkeypatch, root) == 0
    assert source.is_symlink() and not source.exists()
    assert not real.exists()
    assert _run_main(monkeypatch, root) == 0


def test_legacy_jpeg_baked_variants_do_not_break_raw_cache_hit(monkeypatch, tmp_path):
    root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=2)
    assert _run_main(monkeypatch, root, "--keep-source-mp4") == 0
    cache = root / "frame_cache"
    cache_dir = cache / "cam" / "chunk-000" / "file-000"
    for frame in list(cache_dir.glob("frame_*.jpg")):
        for variant in (1, 2):
            frame.with_name(f"{frame.stem}_v{variant}.jpg").write_bytes(frame.read_bytes())
    before = (cache / precompute.CACHE_META_NAME).read_bytes()

    def unexpected_decode(_):
        raise AssertionError("a complete JPEG cache with baked variants must remain a cache hit")

    monkeypatch.setattr(precompute, "_decode_one_mp4", unexpected_decode)
    assert _run_main(monkeypatch, root, "--keep-source-mp4") == 0
    assert (cache / precompute.CACHE_META_NAME).read_bytes() == before
    source = next((root / "videos").rglob("*.mp4"))
    (cache_dir / "frame_000001.jpg").unlink()
    assert not precompute.verify_mp4_cache_complete(source, cache_dir)[0]


@pytest.mark.parametrize("image_format", ["jpg", "png"])
def test_force_cannot_wipe_resolved_symlink_source(monkeypatch, tmp_path, image_format):
    root = _make_fake_lerobot_root(tmp_path, ["cam"], frames_per_mp4=2)
    cache = tmp_path / "cache_with_source"
    cache.mkdir()
    source = next((root / "videos").rglob("*.mp4"))
    real_source = cache / "source.mp4"
    source.rename(real_source)
    source.symlink_to(real_source)
    before = real_source.read_bytes()
    with pytest.raises(ValueError, match="must not contain source"):
        _run_main(monkeypatch, root, "--cache-root", str(cache), "--image-format", image_format, "--force")
    assert source.is_symlink() and source.is_file()
    assert real_source.read_bytes() == before
    assert list(cache.iterdir()) == [real_source]


@pytest.mark.parametrize("symlink_camera", [False, True])
def test_force_cannot_wipe_unselected_camera_source(monkeypatch, tmp_path, symlink_camera):
    root = _make_fake_lerobot_root(tmp_path, ["cam0", "cam1"], frames_per_mp4=2)
    selected = root / "videos" / "cam0" / "chunk-000" / "file-000.mp4"
    camera = root / "videos" / "cam1"
    if symlink_camera:
        real_camera = tmp_path / "real_cam1"
        camera.rename(real_camera)
        camera.symlink_to(real_camera, target_is_directory=True)
        cache = real_camera
    else:
        cache = camera
    unselected = camera / "chunk-000" / "file-000.mp4"
    selected_before = selected.read_bytes()
    unselected_before = unselected.read_bytes()
    with pytest.raises(ValueError, match="must not contain source"):
        _run_main(monkeypatch, root, "--cache-root", str(cache), "--image-format", "png",
                  "--video-keys", "cam0", "--force")
    assert selected.read_bytes() == selected_before
    assert unselected.read_bytes() == unselected_before
    assert not (cache / precompute.CACHE_META_NAME).exists()
