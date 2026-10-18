"""Unit test for JPG frame cache patch (Issue #122)。

- _cache_dir_for_video_path / _cache_fps / _read_jpg_as_frame_tensor: pure helpers
- cached_decode_video_frames: env gate、cache hit round-trip、fallback 4 段階
- apply_patch: idempotent monkey-patch、両 signature (0.5.1 4-arg / 0.6.0 6-arg) 対応

CI 移植性のため fallback は `_original_decode` を stub 化 (torchcodec/pyav 依存除去)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from model.subtask_policy_training.scripts import lerobot_frame_cache_patch as fcp


# ============================================================================
# _cache_dir_for_video_path
# ============================================================================


class TestCacheDirForVideoPath:
    def test_positive_case_returns_expected_dir(self, tmp_path):
        cache_dir = tmp_path / "frame_cache" / "cam_0" / "chunk-000" / "file-000"
        cache_dir.mkdir(parents=True)
        vid = tmp_path / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        assert fcp._cache_dir_for_video_path(vid) == cache_dir

    def test_missing_cache_dir_returns_none(self, tmp_path):
        vid = tmp_path / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        assert fcp._cache_dir_for_video_path(vid) is None

    def test_path_without_videos_returns_none(self, tmp_path):
        vid = tmp_path / "other" / "cam_0" / "chunk-000" / "file-000.mp4"
        assert fcp._cache_dir_for_video_path(vid) is None

    def test_too_short_path_returns_none(self):
        assert fcp._cache_dir_for_video_path(Path("short.mp4")) is None

    def test_symlink_video_finds_cache_at_target_root(self, tmp_path):
        """Issue #122: training view の symlink → merged_source real path → cache 発見。

        LeRobot は training view path で decode 呼ぶ、cache は共有可能な merged_source
        側に置かれる。symlink resolve で cache 発見できないと GR00T ↔ ACT 越しの共有
        が動かない (今回の核心 fix)。
        """
        # merged_source 側: real videos + cache
        merged = tmp_path / "merged_sources" / "combined_task5_7"
        (merged / "videos" / "cam_0" / "chunk-000").mkdir(parents=True)
        real_mp4 = merged / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        real_mp4.touch()  # dummy real file
        expected_cache = merged / "frame_cache" / "cam_0" / "chunk-000" / "file-000"
        expected_cache.mkdir(parents=True)

        # training view 側: symlink from videos/cam_0 → merged/videos/cam_0
        view = tmp_path / "training_views" / "groot_combined_task5_7"
        (view / "videos").mkdir(parents=True)
        (view / "videos" / "cam_0").symlink_to(merged / "videos" / "cam_0")

        # LeRobot が渡す symlink 経由の video path
        symlinked_mp4 = view / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"

        # cache は training view 側には無い (merged 側にある) が、resolve() で発見できる
        result = fcp._cache_dir_for_video_path(symlinked_mp4)
        assert result == expected_cache

    def test_in_place_cache_priority_over_resolved(self, tmp_path):
        """Given path 側に cache が既に有るなら resolve せずそちら優先 (single-repo case)。"""
        vid_dir = tmp_path / "somewhere" / "videos" / "cam_0" / "chunk-000"
        vid_dir.mkdir(parents=True)
        vid = vid_dir / "file-000.mp4"
        vid.touch()
        # given path 側の cache
        cache_here = tmp_path / "somewhere" / "frame_cache" / "cam_0" / "chunk-000" / "file-000"
        cache_here.mkdir(parents=True)

        result = fcp._cache_dir_for_video_path(vid)
        assert result == cache_here


# ============================================================================
# _cache_fps (LRU-cached JSON reader)
# ============================================================================


class TestCacheFps:
    def setup_method(self):
        fcp._cache_fps.cache_clear()

    def test_reads_fps_from_meta(self, tmp_path):
        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        (cache_root / fcp.CACHE_META_NAME).write_text(
            json.dumps({"fps": 30.0, "video_keys": ["cam_0"]})
        )
        assert fcp._cache_fps(cache_root) == 30.0

    def test_missing_meta_returns_none(self, tmp_path):
        assert fcp._cache_fps(tmp_path) is None

    def test_malformed_meta_returns_none(self, tmp_path):
        cache_root = tmp_path / "c2"
        cache_root.mkdir()
        (cache_root / fcp.CACHE_META_NAME).write_text("{not: json}")
        assert fcp._cache_fps(cache_root) is None


# ============================================================================
# _read_jpg_as_frame_tensor
# ============================================================================


class TestReadJpgAsFrameTensor:
    def test_returns_chw_float32_normalized(self, tmp_path):
        # RGB=(200, 100, 50) の均一画像 → JPG → read 後 [0,1] 化 + CHW + RGB order
        img_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        img_rgb[..., 0] = 200
        img_rgb[..., 1] = 100
        img_rgb[..., 2] = 50
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        jpg_path = tmp_path / "f.jpg"
        cv2.imwrite(str(jpg_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        tensor = fcp._read_jpg_as_frame_tensor(jpg_path)
        assert tensor.dtype == torch.float32
        assert tensor.shape == (3, 16, 16)  # CHW
        assert 0.0 <= tensor.min().item() and tensor.max().item() <= 1.0
        # 中央画素の RGB (JPEG 圧縮誤差込み)
        r = tensor[0, 8, 8].item()
        g = tensor[1, 8, 8].item()
        b = tensor[2, 8, 8].item()
        assert abs(r - 200 / 255) < 0.05
        assert abs(g - 100 / 255) < 0.05
        assert abs(b - 50 / 255) < 0.05

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="cv2.imread failed"):
            fcp._read_jpg_as_frame_tensor(tmp_path / "no.jpg")

    def test_post_decode_hook_applied_before_tensor_conversion(self, tmp_path):
        """Issue #122 D-3: hook_context を渡すと registered hook が uint8 HWC BGR 段階で走る。

        hook が全画素を (100, 200, 50) BGR (= RGB (50, 200, 100)) に上書きし、
        その結果が tensor 化されて返ることで hook の前段挿入を verify。
        """
        img_bgr = np.full((16, 16, 3), fill_value=(10, 20, 30), dtype=np.uint8)
        jpg_path = tmp_path / "f.jpg"
        cv2.imwrite(str(jpg_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])

        captured: dict = {}

        def hook(img, video_path, frame_idx):
            captured["video_path"] = video_path
            captured["frame_idx"] = frame_idx
            captured["input_shape"] = img.shape
            captured["input_dtype"] = str(img.dtype)
            # hook 効果を可視化するため全画素を新色に置換
            img[:] = (100, 200, 50)  # BGR
            return img

        try:
            fcp.register_post_decode_hook(hook)
            tensor = fcp._read_jpg_as_frame_tensor(
                jpg_path,
                hook_context=(Path("/root/videos/cam_0/chunk-000/file-000.mp4"), 42),
            )
        finally:
            fcp.clear_post_decode_hooks()

        # hook が正しい context を受け取った
        assert captured["frame_idx"] == 42
        assert captured["video_path"] == Path("/root/videos/cam_0/chunk-000/file-000.mp4")
        assert captured["input_shape"] == (16, 16, 3)
        assert captured["input_dtype"] == "uint8"
        # hook 後の全画素は (RGB 50, 200, 100) → tensor CHW float32 で確認
        assert tensor.shape == (3, 16, 16)
        # center pixel の RGB
        r, g, b = (tensor[c, 8, 8].item() for c in range(3))
        assert abs(r - 50 / 255) < 0.05
        assert abs(g - 200 / 255) < 0.05
        assert abs(b - 100 / 255) < 0.05

    def test_no_hook_context_skips_hook_pathway(self, tmp_path):
        """hook_context=None (default) は hook を走らせない (backward compat)。"""
        img_bgr = np.full((16, 16, 3), fill_value=(10, 20, 30), dtype=np.uint8)
        jpg_path = tmp_path / "f.jpg"
        cv2.imwrite(str(jpg_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])

        called = {"count": 0}

        def hook(img, video_path, frame_idx):
            called["count"] += 1
            return img

        try:
            fcp.register_post_decode_hook(hook)
            # hook_context を渡さない → hook path 走らない
            _ = fcp._read_jpg_as_frame_tensor(jpg_path)
        finally:
            fcp.clear_post_decode_hooks()

        assert called["count"] == 0

    def test_register_and_clear_hooks_updates_registry(self):
        """register_post_decode_hook + clear_post_decode_hooks の basic contract。"""
        try:
            assert len(fcp._post_decode_hooks) == 0
            fcp.register_post_decode_hook(lambda i, v, f: i)
            fcp.register_post_decode_hook(lambda i, v, f: i)
            assert len(fcp._post_decode_hooks) == 2
        finally:
            fcp.clear_post_decode_hooks()
        assert len(fcp._post_decode_hooks) == 0

    def test_returns_uint8_chw_when_flag_set(self, tmp_path):
        """`return_uint8=True` で uint8 CHW をそのまま返す (0.6.0 GR00T factory 常用 mode)。"""
        img_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        img_rgb[..., 0] = 200
        img_rgb[..., 1] = 100
        img_rgb[..., 2] = 50
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        jpg_path = tmp_path / "f.jpg"
        cv2.imwrite(str(jpg_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        tensor = fcp._read_jpg_as_frame_tensor(jpg_path, return_uint8=True)
        assert tensor.dtype == torch.uint8
        assert tensor.shape == (3, 16, 16)  # CHW
        assert 0 <= tensor.min().item() and tensor.max().item() <= 255
        # 中央画素の RGB (JPEG 圧縮誤差込み、uint8 range で ±13 tol)
        r = tensor[0, 8, 8].item()
        g = tensor[1, 8, 8].item()
        b = tensor[2, 8, 8].item()
        assert abs(r - 200) < 13
        assert abs(g - 100) < 13
        assert abs(b - 50) < 13


# ============================================================================
# cached_decode_video_frames (env gate + cache lookup + fallback)
# ============================================================================


def _make_lerobot_with_cache(tmp_path: Path, num_frames: int = 5):
    """`<root>/videos/cam_0/chunk-000/file-000.mp4` (dummy path、実 mp4 は不要) と、
    平行して `<root>/frame_cache/.../frame_XXXXXX.jpg` per-frame + _cache_meta.json を作る。
    """
    root = tmp_path / "lerobot"
    vid_path = root / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
    vid_path.parent.mkdir(parents=True)
    vid_path.touch()  # ファイル本体は空でも path parsing だけ通ればよい
    cache_dir = root / "frame_cache" / "cam_0" / "chunk-000" / "file-000"
    cache_dir.mkdir(parents=True)
    for i in range(num_frames):
        img = np.full((16, 16, 3), fill_value=(i * 25) % 256, dtype=np.uint8)
        cv2.imwrite(
            str(cache_dir / f"frame_{i:06d}.jpg"),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    (root / "frame_cache" / fcp.CACHE_META_NAME).write_text(
        json.dumps({"fps": 30.0, "video_keys": ["cam_0"], "jpg_quality": 95})
    )
    return root, vid_path


class TestCachedDecodeVideoFrames:
    """Fallback paths は `_original_decode` を stub に差し替えて呼び出しだけ verify。"""

    def setup_method(self):
        fcp._cache_fps.cache_clear()
        # patch install の前に _original_decode に stub を仕込む (env 依存除去のため
        # 実 lerobot.datasets.video_utils.decode_video_frames は使わない)
        fcp._patch_applied = True  # apply_patch() を skip
        fcp._original_decode = self._make_stub()

    def teardown_method(self):
        fcp._patch_applied = False
        fcp._original_decode = None

    def _make_stub(self):
        self._calls: list[tuple] = []

        def _stub(video_path, timestamps, tolerance_s, backend=None, **extra):
            self._calls.append((video_path, list(timestamps), tolerance_s, backend, dict(extra)))
            return torch.zeros(len(timestamps), 3, 16, 16, dtype=torch.float32)

        return _stub

    def test_env_disabled_falls_back(self, tmp_path, monkeypatch):
        root, vid_path = _make_lerobot_with_cache(tmp_path)
        monkeypatch.delenv(fcp.ENV_ENABLE, raising=False)
        out = fcp.cached_decode_video_frames(vid_path, [0.0], tolerance_s=0.1)
        assert len(self._calls) == 1
        assert out.shape == (1, 3, 16, 16)

    def test_cache_hit_returns_cached_frame(self, tmp_path, monkeypatch):
        root, vid_path = _make_lerobot_with_cache(tmp_path, num_frames=5)
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        ts = [0.0, 1.0 / 30, 2.0 / 30]  # frame_idx [0, 1, 2]
        out = fcp.cached_decode_video_frames(vid_path, ts, tolerance_s=1e-4)
        assert len(self._calls) == 0  # fallback は呼ばれない
        assert out.shape == (3, 3, 16, 16)
        assert out.dtype == torch.float32
        for i in range(3):
            expected = ((i * 25) % 256) / 255.0
            actual = out[i, 0, 8, 8].item()
            assert abs(actual - expected) < 0.05, f"frame {i}: expected~{expected}, got {actual}"

    def test_cache_miss_beyond_range_falls_back(self, tmp_path, monkeypatch):
        root, vid_path = _make_lerobot_with_cache(tmp_path, num_frames=5)
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        # frame_idx=100 相当 (100/30 = 3.33s) は cache に無い → fallback
        out = fcp.cached_decode_video_frames(vid_path, [3.33], tolerance_s=1.0)
        assert len(self._calls) == 1
        assert out.shape == (1, 3, 16, 16)

    def test_no_cache_dir_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        vid_path = tmp_path / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        # cache dir 無し
        out = fcp.cached_decode_video_frames(vid_path, [0.0], tolerance_s=0.1)
        assert len(self._calls) == 1
        assert out.shape == (1, 3, 16, 16)

    def test_partial_cache_miss_within_call_falls_back(self, tmp_path, monkeypatch):
        """1 call 内で 1 frame でも miss したら whole call fallback (mixed 結果を返さない)。"""
        root, vid_path = _make_lerobot_with_cache(tmp_path, num_frames=3)
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        ts = [0.0, 100.0 / 30]  # frame 0 hit + frame 100 miss
        out = fcp.cached_decode_video_frames(vid_path, ts, tolerance_s=1.0)
        assert len(self._calls) == 1
        assert out.shape == (2, 3, 16, 16)

    # --- Issue #139: FRAME_CACHE_STRICT (video を読まない保証) ---

    def test_strict_cache_hit_is_served_without_decoding(self, tmp_path, monkeypatch):
        root, vid_path = _make_lerobot_with_cache(tmp_path, num_frames=5)
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        monkeypatch.setenv(fcp.ENV_STRICT, "true")
        out = fcp.cached_decode_video_frames(vid_path, [0.0, 1.0 / 30], tolerance_s=1e-4)
        assert len(self._calls) == 0
        assert out.shape == (2, 3, 16, 16)

    def test_strict_partial_miss_raises_instead_of_decoding(self, tmp_path, monkeypatch):
        root, vid_path = _make_lerobot_with_cache(tmp_path, num_frames=3)
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        monkeypatch.setenv(fcp.ENV_STRICT, "true")
        with pytest.raises(RuntimeError, match="missing .*frame_000100"):
            fcp.cached_decode_video_frames(vid_path, [0.0, 100.0 / 30], tolerance_s=1.0)
        assert len(self._calls) == 0  # mp4 decode (元関数) は一度も呼ばれない

    def test_strict_without_cache_dir_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        monkeypatch.setenv(fcp.ENV_STRICT, "true")
        vid_path = tmp_path / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        with pytest.raises(RuntimeError, match="no cache dir"):
            fcp.cached_decode_video_frames(vid_path, [0.0], tolerance_s=0.1)
        assert len(self._calls) == 0

    def test_strict_with_cache_disabled_raises(self, tmp_path, monkeypatch):
        """strict なのに cache 自体が off = 設定ミス。decode せずに止める。"""
        root, vid_path = _make_lerobot_with_cache(tmp_path)
        monkeypatch.delenv(fcp.ENV_ENABLE, raising=False)
        monkeypatch.setenv(fcp.ENV_STRICT, "true")
        with pytest.raises(RuntimeError, match="cache disabled"):
            fcp.cached_decode_video_frames(vid_path, [0.0], tolerance_s=0.1)
        assert len(self._calls) == 0

    def test_missing_cache_meta_falls_back(self, tmp_path, monkeypatch):
        """_cache_meta.json 欠 → fps 不明 → fallback。"""
        root, vid_path = _make_lerobot_with_cache(tmp_path, num_frames=3)
        (root / "frame_cache" / fcp.CACHE_META_NAME).unlink()
        fcp._cache_fps.cache_clear()
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        out = fcp.cached_decode_video_frames(vid_path, [0.0], tolerance_s=0.1)
        assert len(self._calls) == 1

    def test_return_uint8_uses_cache_and_returns_uint8(self, tmp_path, monkeypatch):
        """Issue #122: return_uint8=True でも cache hit する (JPG は元々 uint8 で対応可)。

        以前は fallback していたが、GR00T factory が全 datasets を return_uint8=True で
        構築するため fallback だと cache 一切効かず。JPG から uint8 CHW をそのまま返す
        (torchcodec return_uint8=True 出力形式に一致) 事で cache hit + 高速化を実現。
        """
        root, vid_path = _make_lerobot_with_cache(tmp_path, num_frames=3)
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        out = fcp.cached_decode_video_frames(
            vid_path, [0.0], tolerance_s=0.1, return_uint8=True
        )
        # fallback せず cache から返す
        assert len(self._calls) == 0
        # uint8 CHW (T, C, H, W)、値域 [0, 255]
        assert out.dtype == torch.uint8
        assert out.shape == (1, 3, 16, 16)
        assert 0 <= out.min().item() and out.max().item() <= 255

    def test_is_depth_flag_falls_back(self, tmp_path, monkeypatch):
        """0.6.0 の is_depth=True は cache 未対応 → fallback。"""
        root, vid_path = _make_lerobot_with_cache(tmp_path, num_frames=3)
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        out = fcp.cached_decode_video_frames(
            vid_path, [0.0], tolerance_s=0.1, is_depth=True
        )
        assert len(self._calls) == 1
        _, _, _, _, extras = self._calls[0]
        assert extras.get("is_depth") is True

    def test_051_signature_4arg_positional(self, tmp_path, monkeypatch):
        """0.5.1 signature (4 positional args) を patched 関数が受け付ける。"""
        root, vid_path = _make_lerobot_with_cache(tmp_path, num_frames=2)
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        # cache hit path で backend=None (0.5.1 の一般呼出し pattern)
        out = fcp.cached_decode_video_frames(vid_path, [0.0], 1e-4, None)
        assert out.shape == (1, 3, 16, 16)
        assert len(self._calls) == 0  # cache hit


# ============================================================================
# apply_patch
# ============================================================================


class TestApplyPatch:
    """apply_patch は実 lerobot import に依存する。lerobot import 失敗時は False 返す。"""

    def teardown_method(self):
        # 万一 patch が installed した状態でも reset
        try:
            fcp.reset_for_test()
        except Exception:
            pass

    def test_import_failure_returns_false(self, monkeypatch):
        """lerobot が installable でない env でも apply_patch() が例外なく False 返す。"""
        # sys.modules から lerobot を消して import 失敗を再現
        removed = {}
        for k in list(sys.modules):
            if k == "lerobot" or k.startswith("lerobot."):
                removed[k] = sys.modules.pop(k)
        # import lerobot が失敗するよう path を汚す
        monkeypatch.setitem(sys.modules, "lerobot", None)
        monkeypatch.setitem(sys.modules, "lerobot.datasets", None)
        monkeypatch.setitem(sys.modules, "lerobot.datasets.video_utils", None)
        try:
            fcp._patch_applied = False
            fcp._original_decode = None
            result = fcp.apply_patch()
            assert result is False
        finally:
            for k, v in removed.items():
                sys.modules[k] = v


class TestIsEnabled:
    def test_default_false(self, monkeypatch):
        monkeypatch.delenv(fcp.ENV_ENABLE, raising=False)
        assert fcp.is_enabled() is False

    @pytest.mark.parametrize("val", ["true", "True", "1", "yes", "TRUE"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv(fcp.ENV_ENABLE, val)
        assert fcp.is_enabled() is True

    @pytest.mark.parametrize("val", ["false", "0", "no", ""])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv(fcp.ENV_ENABLE, val)
        assert fcp.is_enabled() is False


# ============================================================================
# _make_patched_check (Issue #122: _check_cached_episodes_sufficient 差替)
# ============================================================================


class _FakeHFDataset:
    """`_check_cached_episodes_sufficient` が触る hf_dataset interface のみ模倣。

    - `len(self)` = 全 row 数
    - `self.unique("episode_index")` = episode_index の unique 値 list
    """

    def __init__(self, episode_indices: list[int]):
        self._episodes = episode_indices

    def __len__(self) -> int:
        return len(self._episodes)

    def unique(self, column: str) -> list[int]:
        assert column == "episode_index"
        # dedupe 順序保持
        seen: list[int] = []
        for e in self._episodes:
            if e not in seen:
                seen.append(e)
        return seen


class _FakeMeta:
    def __init__(self, total_episodes: int):
        self.total_episodes = total_episodes


class _FakeReader:
    """0.6.0 (DatasetReader) style: `self._meta` + `self.episodes` + `self.hf_dataset`。"""

    def __init__(
        self,
        hf_dataset: _FakeHFDataset | None,
        total_episodes: int,
        episodes: list[int] | None = None,
    ):
        self.hf_dataset = hf_dataset
        self._meta = _FakeMeta(total_episodes)
        self.episodes = episodes


class _FakeDataset051:
    """0.5.1 (LeRobotDataset) style: `self.meta` (underscore なし)。"""

    def __init__(
        self,
        hf_dataset: _FakeHFDataset | None,
        total_episodes: int,
        episodes: list[int] | None = None,
    ):
        self.hf_dataset = hf_dataset
        self.meta = _FakeMeta(total_episodes)
        self.episodes = episodes


def _original_that_returns(value: bool):
    """Original method の stub。呼び出し回数を記録する。"""
    calls: list[Any] = []

    def _f(self):
        calls.append(self)
        return value

    _f.calls = calls  # type: ignore[attr-defined]
    return _f


class TestMakePatchedCheck:
    """`_make_patched_check` の returns True / False / fall through 挙動。"""

    def test_env_off_falls_through_to_original(self, monkeypatch):
        monkeypatch.delenv(fcp.ENV_ENABLE, raising=False)
        original = _original_that_returns(True)
        patched = fcp._make_patched_check(original)
        reader = _FakeReader(_FakeHFDataset([0, 1]), total_episodes=2)
        assert patched(reader) is True
        assert len(original.calls) == 1  # original 呼ばれた

    def test_hf_dataset_none_returns_false(self, monkeypatch):
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        original = _original_that_returns(True)
        patched = fcp._make_patched_check(original)
        reader = _FakeReader(None, total_episodes=2)
        assert patched(reader) is False
        assert len(original.calls) == 0  # 呼ばれない (parquet 側で失敗)

    def test_hf_dataset_empty_returns_false(self, monkeypatch):
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        original = _original_that_returns(True)
        patched = fcp._make_patched_check(original)
        reader = _FakeReader(_FakeHFDataset([]), total_episodes=2)
        assert patched(reader) is False

    def test_all_episodes_present_returns_true_even_without_videos(self, monkeypatch):
        """本命: mp4 不在でも episodes 揃っていれば True (video path check を skip)。"""
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        original = _original_that_returns(False)  # original なら False (video 不在で)
        patched = fcp._make_patched_check(original)
        reader = _FakeReader(_FakeHFDataset([0, 1, 2]), total_episodes=3)
        assert patched(reader) is True
        assert len(original.calls) == 0  # original に fall through してない

    def test_missing_episode_returns_false(self, monkeypatch):
        """parquet 側 check は残っている: 要求 episode の一部 unavailable なら False。"""
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        original = _original_that_returns(True)
        patched = fcp._make_patched_check(original)
        # 要求 ep=[0,1,2] だが hf_dataset には [0,1] しか無い
        reader = _FakeReader(_FakeHFDataset([0, 1]), total_episodes=3)
        assert patched(reader) is False

    def test_explicit_episodes_subset(self, monkeypatch):
        """`self.episodes` 明示指定時、指定 subset のみ完備で True。"""
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        original = _original_that_returns(False)
        patched = fcp._make_patched_check(original)
        reader = _FakeReader(
            _FakeHFDataset([0, 1, 2, 3, 4]),
            total_episodes=10,
            episodes=[1, 3],
        )
        # 全 total_episodes=10 は unavailable だが、指定 [1,3] は available
        assert patched(reader) is True

    def test_051_meta_attribute_works(self, monkeypatch):
        """0.5.1 の `self.meta` (underscore なし) も duck typing で扱える。"""
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        original = _original_that_returns(False)
        patched = fcp._make_patched_check(original)
        dataset = _FakeDataset051(_FakeHFDataset([0, 1]), total_episodes=2)
        assert patched(dataset) is True

    def test_meta_missing_falls_back_to_original(self, monkeypatch):
        """unexpected shape (meta 不在) は silent bypass せず original に fall back。"""
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        original = _original_that_returns(True)
        patched = fcp._make_patched_check(original)

        class _NoMeta:
            hf_dataset = _FakeHFDataset([0, 1])
            episodes = None
            # _meta も meta も無い

        assert patched(_NoMeta()) is True  # original の戻り値
        assert len(original.calls) == 1

    def test_tensor_episode_index_handled(self, monkeypatch):
        """`unique("episode_index")` が torch.Tensor scalar list を返しても handle。"""
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        original = _original_that_returns(False)
        patched = fcp._make_patched_check(original)
        tensor_eps = [torch.tensor(0), torch.tensor(1)]
        reader = _FakeReader(_FakeHFDataset(tensor_eps), total_episodes=2)  # type: ignore[arg-type]
        assert patched(reader) is True


# ============================================================================
# _install_check_patch (両版 duck typing install + idempotency + rollback)
# ============================================================================


class TestInstallCheckPatch:
    """実 lerobot への binding は最小限、代わりに dummy module + class を用意して verify。"""

    def teardown_method(self):
        # `_original_checks` に registered 済 fake class の rollback
        try:
            fcp.reset_for_test()
        except Exception:
            pass
        # dummy module を sys.modules から掃除
        for k in list(sys.modules):
            if k.startswith("_fake_lerobot_test_"):
                sys.modules.pop(k, None)
        # apply_patch 状態も reset
        fcp._patch_applied = False
        fcp._original_decode = None

    def _install_dummy_target(self, module_name: str, class_name: str, return_value: bool = False):
        """`_check_patch_targets` に inject する dummy module + class を作る。"""
        import types

        mod = types.ModuleType(module_name)

        class DummyClass:
            def _check_cached_episodes_sufficient(self) -> bool:
                return return_value

        setattr(mod, class_name, DummyClass)
        sys.modules[module_name] = mod
        return DummyClass

    def test_replaces_target_method(self, monkeypatch):
        """Register 済 target module の method を patched 版に差替える。"""
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        cls = self._install_dummy_target(
            "_fake_lerobot_test_a", "Dummy", return_value=False
        )
        monkeypatch.setattr(
            fcp,
            "_check_patch_targets",
            [("_fake_lerobot_test_a", "Dummy")],
        )
        fcp._install_check_patch()

        # 差替後: hf_dataset + episodes 完備なら True (原挙動 False を上書き)
        instance = cls()
        instance.hf_dataset = _FakeHFDataset([0, 1])
        instance._meta = _FakeMeta(2)
        instance.episodes = None
        assert instance._check_cached_episodes_sufficient() is True

    def test_idempotent_second_call_noop(self, monkeypatch):
        """2 回目 call は既存 patch を上書きしない (元関数 loss しない)。"""
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        cls = self._install_dummy_target(
            "_fake_lerobot_test_b", "Dummy", return_value=False
        )
        monkeypatch.setattr(
            fcp,
            "_check_patch_targets",
            [("_fake_lerobot_test_b", "Dummy")],
        )
        fcp._install_check_patch()
        first_patched = cls._check_cached_episodes_sufficient
        fcp._install_check_patch()
        assert cls._check_cached_episodes_sufficient is first_patched
        # `_original_checks` にも 1 entry のみ
        assert cls in fcp._original_checks

    def test_import_failure_silent_skip(self, monkeypatch):
        """target module が import 不可でも例外を投げず skip する。"""
        monkeypatch.setattr(
            fcp,
            "_check_patch_targets",
            [("_nonexistent_module_xyz", "Foo")],
        )
        # 例外投げずに戻る
        fcp._install_check_patch()
        assert len(fcp._original_checks) == 0

    def test_reset_for_test_restores_original(self, monkeypatch):
        """`reset_for_test()` で method が元に戻る (test 分離のため)。"""
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        cls = self._install_dummy_target(
            "_fake_lerobot_test_c", "Dummy", return_value=False
        )
        original_method = cls._check_cached_episodes_sufficient
        monkeypatch.setattr(
            fcp,
            "_check_patch_targets",
            [("_fake_lerobot_test_c", "Dummy")],
        )
        fcp._install_check_patch()
        assert cls._check_cached_episodes_sufficient is not original_method
        fcp.reset_for_test()
        assert cls._check_cached_episodes_sufficient is original_method
        assert len(fcp._original_checks) == 0


# ============================================================================
# _install_decode_bind_patch (Issue #122: local binding 差替)
# ============================================================================


class TestInstallDecodeBindPatch:
    """`from .video_utils import decode_video_frames` local binding の差替を verify。

    実 lerobot ではなく dummy module を `_decode_bind_targets` に inject する。
    """

    def teardown_method(self):
        try:
            fcp.reset_for_test()
        except Exception:
            pass
        for k in list(sys.modules):
            if k.startswith("_fake_bind_test_"):
                sys.modules.pop(k, None)
        fcp._patch_applied = False
        fcp._original_decode = None
        fcp._patched_bind_modules.clear()

    def _install_dummy_bind_module(self, module_name: str, bound_fn):
        """`_decode_bind_targets` に inject する dummy module (decode_video_frames を bind 済)。"""
        import types

        mod = types.ModuleType(module_name)
        mod.decode_video_frames = bound_fn
        sys.modules[module_name] = mod
        return mod

    def test_replaces_bind_when_identity_matches(self, monkeypatch):
        """`_original_decode` と identity 一致 → 差替、`cached_decode_video_frames` に。"""

        def orig(*a, **k):
            return None

        fcp._original_decode = orig
        mod = self._install_dummy_bind_module("_fake_bind_test_a", orig)
        monkeypatch.setattr(fcp, "_decode_bind_targets", ["_fake_bind_test_a"])
        fcp._install_decode_bind_patch()
        assert mod.decode_video_frames is fcp.cached_decode_video_frames
        assert "_fake_bind_test_a" in fcp._patched_bind_modules

    def test_preserves_user_override_when_identity_differs(self, monkeypatch):
        """既に別関数 (user override 等) に置換済 → touch しない (identity mismatch)。"""

        def orig(*a, **k):
            return None

        def user_wrapper(*a, **k):
            return orig(*a, **k)

        fcp._original_decode = orig
        mod = self._install_dummy_bind_module("_fake_bind_test_b", user_wrapper)
        monkeypatch.setattr(fcp, "_decode_bind_targets", ["_fake_bind_test_b"])
        fcp._install_decode_bind_patch()
        # user_wrapper が保護される
        assert mod.decode_video_frames is user_wrapper
        assert "_fake_bind_test_b" not in fcp._patched_bind_modules

    def test_idempotent_second_call_noop(self, monkeypatch):
        """2 回目 call は既に差替済 module を skip (identity は cached、_original ではない)。"""

        def orig(*a, **k):
            return None

        fcp._original_decode = orig
        mod = self._install_dummy_bind_module("_fake_bind_test_c", orig)
        monkeypatch.setattr(fcp, "_decode_bind_targets", ["_fake_bind_test_c"])
        fcp._install_decode_bind_patch()
        assert mod.decode_video_frames is fcp.cached_decode_video_frames
        # 2 回目呼び出し: _patched_bind_modules skip で例外なし
        fcp._install_decode_bind_patch()
        assert mod.decode_video_frames is fcp.cached_decode_video_frames
        assert fcp._patched_bind_modules.count("_fake_bind_test_c") == 1

    def test_import_failure_silent_skip(self, monkeypatch):
        """target module が import 不可でも例外を投げず skip する。"""
        fcp._original_decode = lambda *a, **k: None
        monkeypatch.setattr(fcp, "_decode_bind_targets", ["_nonexistent_bind_module_xyz"])
        fcp._install_decode_bind_patch()
        assert len(fcp._patched_bind_modules) == 0

    def test_reset_for_test_restores_bind(self, monkeypatch):
        """`reset_for_test()` で bind 差替も rollback される (test 分離のため)。"""

        def orig(*a, **k):
            return None

        fcp._original_decode = orig
        mod = self._install_dummy_bind_module("_fake_bind_test_d", orig)
        monkeypatch.setattr(fcp, "_decode_bind_targets", ["_fake_bind_test_d"])
        fcp._install_decode_bind_patch()
        assert mod.decode_video_frames is fcp.cached_decode_video_frames
        # reset は video_utils import 経路も試すので `_patch_applied=True` に設定
        fcp._patch_applied = True
        fcp.reset_for_test()
        assert mod.decode_video_frames is orig
        assert len(fcp._patched_bind_modules) == 0

    def test_no_op_when_original_decode_none(self, monkeypatch):
        """`_original_decode` 未設定 (apply_patch 前) 呼び出しは何もせず戻る。"""
        fcp._original_decode = None
        mod = self._install_dummy_bind_module("_fake_bind_test_e", lambda *a, **k: None)
        original_fn = mod.decode_video_frames
        monkeypatch.setattr(fcp, "_decode_bind_targets", ["_fake_bind_test_e"])
        fcp._install_decode_bind_patch()
        # 差替されない、track もされない
        assert mod.decode_video_frames is original_fn
        assert len(fcp._patched_bind_modules) == 0


# ============================================================================
# Issue #129: offline aug variant 選択 (FRAME_CACHE_NUM_VARIANTS)
# ============================================================================


class TestGetNumVariants:
    """`_get_num_variants` env parse。"""

    def test_default_is_1_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(fcp.ENV_NUM_VARIANTS, raising=False)
        assert fcp._get_num_variants() == 1

    def test_env_2_returns_2(self, monkeypatch):
        monkeypatch.setenv(fcp.ENV_NUM_VARIANTS, "2")
        assert fcp._get_num_variants() == 2

    def test_env_10_returns_10(self, monkeypatch):
        monkeypatch.setenv(fcp.ENV_NUM_VARIANTS, "10")
        assert fcp._get_num_variants() == 10

    def test_env_invalid_returns_1(self, monkeypatch):
        monkeypatch.setenv(fcp.ENV_NUM_VARIANTS, "not-an-int")
        assert fcp._get_num_variants() == 1

    def test_env_zero_clamped_to_1(self, monkeypatch):
        monkeypatch.setenv(fcp.ENV_NUM_VARIANTS, "0")
        assert fcp._get_num_variants() == 1

    def test_env_negative_clamped_to_1(self, monkeypatch):
        monkeypatch.setenv(fcp.ENV_NUM_VARIANTS, "-5")
        assert fcp._get_num_variants() == 1


class TestResolveVariantJpgPath:
    """`_resolve_variant_jpg_path` variant idx pick + path 構築 (Issue #129 case F)。

    N>=2 の時は v0 (raw) を skip して v_1..v_(N-1) から uniform pick する。
    v0 は precompute で無変更で保持されており (rerun source として)、training には使わない。
    """

    def test_num_variants_1_returns_v0_path(self, tmp_path):
        """N=1 = 従来 backward compat、v0 のみ (offline aug 未使用時)。"""
        p = fcp._resolve_variant_jpg_path(tmp_path, 42, num_variants=1)
        assert p == tmp_path / "frame_000042.jpg"

    def test_num_variants_2_always_returns_v1(self, tmp_path):
        """N=2 は v1 のみ選択、v0 は skip。"""
        import random as _r
        seen = set()
        for seed in range(20):
            _r.seed(seed)
            p = fcp._resolve_variant_jpg_path(tmp_path, 42, num_variants=2)
            seen.add(p.name)
        assert seen == {"frame_000042_v1.jpg"}

    def test_num_variants_10_covers_v1_to_v9_only(self, tmp_path):
        """N=10 は v1..v9 全部 hit、v0 は 1 度も出ない。"""
        import random as _r
        seen = set()
        for seed in range(200):
            _r.seed(seed)
            p = fcp._resolve_variant_jpg_path(tmp_path, 0, num_variants=10)
            seen.add(p.name)
        expected = {f"frame_000000_v{i}.jpg" for i in range(1, 10)}
        assert seen == expected
        assert "frame_000000.jpg" not in seen  # v0 は明示的に排除

    def test_v0_never_picked_when_num_variants_ge_2(self, tmp_path):
        """N>=2 で 500 seed 試行しても v0 は 1 度も選ばれない。"""
        import random as _r
        for seed in range(500):
            _r.seed(seed)
            p = fcp._resolve_variant_jpg_path(tmp_path, 0, num_variants=5)
            assert p.name != "frame_000000.jpg", f"v0 picked at seed={seed}"

    def test_v_positive_uses_v_suffix(self, tmp_path):
        """N>=2 では常に `_v{n}` サフィックス (n>=1)。"""
        import random as _r
        for seed in range(50):
            _r.seed(seed)
            p = fcp._resolve_variant_jpg_path(tmp_path, 7, num_variants=5)
            assert p.name.startswith("frame_000007_v"), p.name
            assert p.name.endswith(".jpg")


# ----------------------------------------------------------------------------
# Issue #129 Phase K (2026-09-01): Frame cache override registry (multi-dataset safe)
# ----------------------------------------------------------------------------
class TestFrameCacheOverrideRegistry:
    """Registry の per-source-root 独立性 + env fallback + prefix match。

    RamenOriMultiDataset が複数 sub-dataset を build する時、env は global で衝突するため
    registry で分離する必要がある。ここでは registry の基本 API + prefix match 挙動を検証。
    """

    def setup_method(self):
        """各 test 前 registry を clear。"""
        fcp.clear_frame_cache_registry()

    def teardown_method(self):
        """test 後も clear (他 test への leak 防止)。"""
        fcp.clear_frame_cache_registry()

    def test_register_and_lookup(self, tmp_path):
        """register 後、その source_root 配下の video_path で lookup できる。"""
        source_root = tmp_path / "task5"
        source_root.mkdir()
        override_root = tmp_path / "task5" / "frame_cache_v2" / "overlay"
        fcp.register_frame_cache_override(source_root, override_root, num_variants=10)
        # video_path = <source_root>/videos/cam_0/chunk-000/file-000.mp4
        video_path = source_root / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        result = fcp._lookup_registry_for(video_path)
        assert result is not None
        override, nv = result
        assert override == override_root
        assert nv == 10

    def test_lookup_returns_none_for_unregistered_path(self, tmp_path):
        """未登録 source_root 配下の video_path は None を返す。"""
        source_root = tmp_path / "task5"
        source_root.mkdir()
        fcp.register_frame_cache_override(source_root, tmp_path / "overlay", num_variants=5)
        # 全く別の path
        other_path = tmp_path / "task7" / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        (tmp_path / "task7").mkdir()
        assert fcp._lookup_registry_for(other_path) is None

    def test_multi_source_root_isolation(self, tmp_path):
        """複数 source_root を登録した時、video_path の prefix で正しい override が引ける。"""
        task5_root = tmp_path / "task5"
        task5_root.mkdir()
        task7_root = tmp_path / "task7"
        task7_root.mkdir()
        task5_override = tmp_path / "task5_overlay"
        task7_override = tmp_path / "task7_overlay"
        fcp.register_frame_cache_override(task5_root, task5_override, num_variants=10)
        fcp.register_frame_cache_override(task7_root, task7_override, num_variants=3)

        task5_video = task5_root / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        task7_video = task7_root / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"

        r5 = fcp._lookup_registry_for(task5_video)
        r7 = fcp._lookup_registry_for(task7_video)
        assert r5 == (task5_override, 10)
        assert r7 == (task7_override, 3)

    def test_registry_takes_priority_over_env(self, tmp_path, monkeypatch):
        """env が set されていても registry hit の方が優先される。"""
        source_root = tmp_path / "task5"
        source_root.mkdir()
        env_override = tmp_path / "env_overlay"
        reg_override = tmp_path / "reg_overlay"
        monkeypatch.setenv(fcp.ENV_ROOT_OVERRIDE, str(env_override))
        fcp.register_frame_cache_override(source_root, reg_override, num_variants=7)

        video_path = source_root / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        # Registry hit で env 無視
        assert fcp._lookup_registry_for(video_path) == (reg_override, 7)

    def test_env_fallback_when_no_registry_entry(self, tmp_path, monkeypatch):
        """Registry 未登録の path で env override が引ける (backward compat)。"""
        env_override = tmp_path / "env_overlay"
        monkeypatch.setenv(fcp.ENV_ROOT_OVERRIDE, str(env_override))
        # Registry は空
        video_path = tmp_path / "some_task" / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        # Registry は None
        assert fcp._lookup_registry_for(video_path) is None
        # env は生きてる (別関数)
        assert fcp._get_root_override() == env_override

    def test_unregister_removes_entry(self, tmp_path):
        """unregister_frame_cache_override が entry を削除する。"""
        source_root = tmp_path / "task5"
        source_root.mkdir()
        fcp.register_frame_cache_override(source_root, tmp_path / "overlay", num_variants=5)
        video_path = source_root / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        assert fcp._lookup_registry_for(video_path) is not None
        fcp.unregister_frame_cache_override(source_root)
        assert fcp._lookup_registry_for(video_path) is None

    def test_clear_registry_removes_all_entries(self, tmp_path):
        """clear_frame_cache_registry が全 entry を消す。"""
        r1 = tmp_path / "task5"
        r2 = tmp_path / "task7"
        r1.mkdir()
        r2.mkdir()
        fcp.register_frame_cache_override(r1, tmp_path / "o1", num_variants=5)
        fcp.register_frame_cache_override(r2, tmp_path / "o2", num_variants=3)
        fcp.clear_frame_cache_registry()
        v1 = r1 / "videos" / "cam" / "chunk" / "file.mp4"
        v2 = r2 / "videos" / "cam" / "chunk" / "file.mp4"
        assert fcp._lookup_registry_for(v1) is None
        assert fcp._lookup_registry_for(v2) is None

    def test_reregister_updates_entry(self, tmp_path):
        """同一 source_root を再 register すると値が更新される (multi run で config 変更対応)。"""
        source_root = tmp_path / "task5"
        source_root.mkdir()
        fcp.register_frame_cache_override(source_root, tmp_path / "old", num_variants=1)
        fcp.register_frame_cache_override(source_root, tmp_path / "new", num_variants=10)
        video_path = source_root / "videos" / "cam" / "chunk" / "file.mp4"
        result = fcp._lookup_registry_for(video_path)
        assert result == (tmp_path / "new", 10)

    def test_nested_source_roots_longest_prefix_wins(self, tmp_path):
        """Nested source_root (parent + child 両登録) では child (深い方) が勝つ。"""
        parent = tmp_path / "root"
        child = tmp_path / "root" / "sub"
        parent.mkdir()
        child.mkdir()
        fcp.register_frame_cache_override(parent, tmp_path / "parent_ov", num_variants=1)
        fcp.register_frame_cache_override(child, tmp_path / "child_ov", num_variants=5)
        # child 配下の video → child override
        video = child / "videos" / "cam" / "chunk" / "file.mp4"
        assert fcp._lookup_registry_for(video) == (tmp_path / "child_ov", 5)
        # child 外 (parent のみ配下) → parent override
        video_parent_only = parent / "videos" / "cam" / "chunk" / "file.mp4"
        assert fcp._lookup_registry_for(video_parent_only) == (tmp_path / "parent_ov", 1)
