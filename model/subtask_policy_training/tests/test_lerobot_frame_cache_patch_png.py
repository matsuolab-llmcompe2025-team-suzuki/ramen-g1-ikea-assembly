"""Issue #168: lossless raw PNG cache selection, compatibility, and fallback."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from model.subtask_policy_training.scripts import lerobot_frame_cache_patch as fcp
from model.subtask_policy_training.tests.test_lerobot_frame_cache_patch import _make_lerobot_with_cache


def _make_png_cache(tmp_path: Path, num_frames: int = 4):
    root = tmp_path / "lerobot"
    video = root / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.touch()
    cache_root = root / "frame_cache_png"
    cache_dir = cache_root / "cam_0" / "chunk-000" / "file-000"
    cache_dir.mkdir(parents=True)
    frames_rgb = []
    for index in range(num_frames):
        # Distinct channels and sharp edges detect RGB swaps and lossy reads.
        rgb = ((np.arange(768).reshape(16, 16, 3) * 37 + index * 51) % 256).astype(np.uint8)
        assert cv2.imwrite(
            str(cache_dir / f"frame_{index:06d}.png"),
            rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
        frames_rgb.append(rgb)
    (cache_root / fcp.CACHE_META_NAME).write_text(json.dumps({
        "schema_version": "team_ramen_frame_cache_v2", "image_format": "png",
        "png_compression": 1, "decoder": "pyav", "complete": True,
        "fps": 30.0, "video_keys": ["cam_0"],
    }))
    return video, cache_root, cache_dir, torch.from_numpy(np.stack(frames_rgb)).permute(0, 3, 1, 2)


class TestPngFrameCache:
    @pytest.fixture(autouse=True)
    def _isolated_cache_state(self, monkeypatch):
        fcp.clear_frame_cache_registry()
        fcp.clear_post_decode_hooks()
        fcp._cache_fps.cache_clear()
        fcp._cache_metadata.cache_clear()
        for env_name in (
            fcp.ENV_ROOT_OVERRIDE, fcp.ENV_NUM_VARIANTS, fcp.ENV_STRICT, fcp.ENV_IMAGE_FORMAT,
        ):
            monkeypatch.delenv(env_name, raising=False)
        monkeypatch.setenv(fcp.ENV_ENABLE, "true")
        self.calls = []
        self.fallback_result = torch.full((1, 3, 16, 16), -1.0)

        def original(video, timestamps, tolerance_s, backend=None, **kwargs):
            self.calls.append((video, timestamps, tolerance_s, backend, kwargs))
            return self.fallback_result

        monkeypatch.setattr(fcp, "_original_decode", original)
        yield
        fcp.clear_frame_cache_registry()
        fcp.clear_post_decode_hooks()
        fcp._cache_fps.cache_clear()
        fcp._cache_metadata.cache_clear()

    def _select(self, tmp_path, monkeypatch):
        result = _make_png_cache(tmp_path)
        monkeypatch.setenv(fcp.ENV_ROOT_OVERRIDE, str(result[1]))
        return result

    @pytest.mark.parametrize("return_uint8", [False, True])
    def test_exact_rgb_order_and_repeated_timestamps(self, tmp_path, monkeypatch, return_uint8):
        video, _, _, expected = self._select(tmp_path, monkeypatch)
        indices = [2, 0, 2, 3, 1]
        output = fcp.cached_decode_video_frames(
            video, [index / 30 for index in indices], 1e-6, "pyav", return_uint8=return_uint8,
        )
        expected = expected[indices]
        if not return_uint8:
            expected = expected.to(torch.float32) / 255.0
        assert output.dtype == expected.dtype
        assert output.shape == (5, 3, 16, 16)
        assert torch.equal(output, expected)
        assert self.calls == []

    def test_strict_hit_and_minimal_metadata(self, tmp_path, monkeypatch):
        video, cache_root, _, expected = self._select(tmp_path, monkeypatch)
        # Provenance fields are not required for externally generated raw caches.
        (cache_root / fcp.CACHE_META_NAME).write_text(json.dumps({
            "fps": 30, "image_format": "png", "complete": True,
        }))
        monkeypatch.setenv(fcp.ENV_STRICT, "true")
        output = fcp.cached_decode_video_frames(video, [0.0], 0.0, return_uint8=True)
        assert torch.equal(output, expected[:1])
        assert self.calls == []

    @pytest.mark.parametrize("strict", [False, True])
    def test_partial_missing_png_never_uses_jpeg(self, tmp_path, monkeypatch, strict):
        _make_lerobot_with_cache(tmp_path)
        video, _, cache_dir, _ = self._select(tmp_path, monkeypatch)
        (cache_dir / "frame_000001.png").unlink()
        assert cv2.imwrite(str(cache_dir / "frame_000001.jpg"), np.zeros((16, 16, 3), np.uint8))
        monkeypatch.setenv(fcp.ENV_STRICT, str(strict))
        timestamps = [0.0, 1.0 / 30]
        if strict:
            with pytest.raises(RuntimeError, match=r"missing .*frame_000001\.png"):
                fcp.cached_decode_video_frames(video, timestamps, 1e-6, "pyav", return_uint8=True)
            assert self.calls == []
        else:
            output = fcp.cached_decode_video_frames(video, timestamps, 1e-6, "pyav", return_uint8=True)
            assert output is self.fallback_result
            assert self.calls == [(video, timestamps, 1e-6, "pyav", {"return_uint8": True})]

    @pytest.mark.parametrize("strict", [False, True])
    @pytest.mark.parametrize("image_format", [None, "jpeg", "webp", "PNG", ["png"], {"png": True}])
    def test_invalid_metadata_format_fails_safely(self, tmp_path, monkeypatch, strict, image_format):
        video, cache_root, _, _ = self._select(tmp_path, monkeypatch)
        (cache_root / fcp.CACHE_META_NAME).write_text(json.dumps({"fps": 30, "image_format": image_format}))
        monkeypatch.setenv(fcp.ENV_STRICT, str(strict))
        if strict:
            with pytest.raises(RuntimeError, match="unsupported cache image_format"):
                fcp.cached_decode_video_frames(video, [0.0], 0.0)
            assert self.calls == []
        else:
            assert fcp.cached_decode_video_frames(video, [0.0], 0.0) is self.fallback_result
            assert len(self.calls) == 1

    @pytest.mark.parametrize("meta_contents", [None, "{broken", "[]"])
    def test_missing_or_invalid_metadata_does_not_guess_from_files(self, tmp_path, monkeypatch, meta_contents):
        video, cache_root, _, _ = self._select(tmp_path, monkeypatch)
        meta_path = cache_root / fcp.CACHE_META_NAME
        if meta_contents is None:
            meta_path.unlink()
        else:
            meta_path.write_text(meta_contents)
        assert fcp.cached_decode_video_frames(video, [0.0], 0.0) is self.fallback_result
        assert len(self.calls) == 1

    @pytest.mark.parametrize("strict", [False, True])
    @pytest.mark.parametrize("complete", ["absent", False, None, 0, 1, "true", [], {}])
    def test_incomplete_png_refused(self, tmp_path, monkeypatch, strict, complete):
        video, cache_root, _, _ = self._select(tmp_path, monkeypatch)
        metadata = {"fps": 30, "image_format": "png"}
        if complete != "absent":
            metadata["complete"] = complete
        (cache_root / fcp.CACHE_META_NAME).write_text(json.dumps(metadata))
        monkeypatch.setenv(fcp.ENV_STRICT, str(strict))
        if strict:
            with pytest.raises(RuntimeError, match="PNG cache is incomplete"):
                fcp.cached_decode_video_frames(video, [0.0], 0.0)
            assert self.calls == []
        else:
            assert fcp.cached_decode_video_frames(video, [0.0], 0.0) is self.fallback_result

    @pytest.mark.parametrize("strict", [False, True])
    @pytest.mark.parametrize("use_registry", [False, True])
    def test_png_variants_are_rejected(self, tmp_path, monkeypatch, strict, use_registry):
        video, cache_root, cache_dir, _ = self._select(tmp_path, monkeypatch)
        assert cv2.imwrite(str(cache_dir / "frame_000000_v1.jpg"), np.zeros((16, 16, 3), np.uint8))
        if use_registry:
            fcp.register_frame_cache_override(video.parents[3], cache_root, num_variants=2)
        else:
            monkeypatch.setenv(fcp.ENV_NUM_VARIANTS, "2")
        monkeypatch.setenv(fcp.ENV_STRICT, str(strict))
        if strict:
            with pytest.raises(RuntimeError, match="PNG cache supports raw frames only"):
                fcp.cached_decode_video_frames(video, [0.0], 0.0)
            assert self.calls == []
        else:
            assert fcp.cached_decode_video_frames(video, [0.0], 0.0) is self.fallback_result

    @pytest.mark.parametrize("strict", [False, True])
    def test_off_grid_timestamp_is_whole_call_miss(self, tmp_path, monkeypatch, strict):
        video, _, _, _ = self._select(tmp_path, monkeypatch)
        monkeypatch.setenv(fcp.ENV_STRICT, str(strict))
        timestamps = [0.0, 1.0 / 30 + 0.002]
        if strict:
            with pytest.raises(RuntimeError, match="PNG timestamp .* outside cache grid tolerance"):
                fcp.cached_decode_video_frames(video, timestamps, 0.001)
            assert self.calls == []
        else:
            assert fcp.cached_decode_video_frames(video, timestamps, 0.001) is self.fallback_result
            assert self.calls[0][1] == timestamps

    def test_timestamp_within_tolerance_is_accepted(self, tmp_path, monkeypatch):
        video, _, _, expected = self._select(tmp_path, monkeypatch)
        output = fcp.cached_decode_video_frames(video, [1.0 / 30 + 0.0001], 0.001, return_uint8=True)
        assert torch.equal(output, expected[1:2])
        assert self.calls == []

    @pytest.mark.parametrize("fps", [0, -30, float("nan"), float("inf"), True, "30"])
    def test_invalid_fps_is_cache_miss(self, tmp_path, monkeypatch, fps):
        video, cache_root, _, _ = self._select(tmp_path, monkeypatch)
        (cache_root / fcp.CACHE_META_NAME).write_text(json.dumps({
            "fps": fps, "image_format": "png", "complete": True,
        }))
        assert fcp.cached_decode_video_frames(video, [0.0], 0.0) is self.fallback_result

    @pytest.mark.parametrize("strict", [False, True])
    @pytest.mark.parametrize("failure_mode", ["corrupt", "cv2_error"])
    def test_png_decode_failure_is_whole_call_miss(self, tmp_path, monkeypatch, strict, failure_mode):
        video, _, cache_dir, _ = self._select(tmp_path, monkeypatch)
        broken_frame = cache_dir / "frame_000001.png"
        if failure_mode == "corrupt":
            broken_frame.write_bytes(b"not a valid PNG")
        else:
            original_imread = cv2.imread

            def failing_imread(path, flags):
                if Path(path) == broken_frame:
                    raise cv2.error("simulated image decoder failure")
                return original_imread(path, flags)

            monkeypatch.setattr(cv2, "imread", failing_imread)
        monkeypatch.setenv(fcp.ENV_STRICT, str(strict))
        timestamps = [0.0, 1 / 30]
        if strict:
            with pytest.raises(RuntimeError, match="PNG decode failure: cv2.imread failed"):
                fcp.cached_decode_video_frames(video, timestamps, 0.0, "pyav", return_uint8=True)
            assert self.calls == []
        else:
            output = fcp.cached_decode_video_frames(video, timestamps, 0.0, "pyav", return_uint8=True)
            assert output is self.fallback_result
            assert self.calls == [(video, timestamps, 0.0, "pyav", {"return_uint8": True})]

    @pytest.mark.parametrize("error_type", [RuntimeError, cv2.error])
    def test_png_hook_errors_are_not_swallowed(self, tmp_path, monkeypatch, error_type):
        video, _, _, _ = self._select(tmp_path, monkeypatch)

        def broken_hook(*args):
            raise error_type("user hook failed")

        fcp.register_post_decode_hook(broken_hook)
        with pytest.raises(error_type, match="user hook failed"):
            fcp.cached_decode_video_frames(video, [0.0], 0.0)
        assert self.calls == []

    @pytest.mark.parametrize("failure_mode", ["corrupt", "cv2_error"])
    def test_jpeg_decode_failure_behavior_unchanged(self, tmp_path, monkeypatch, failure_mode):
        root, video = _make_lerobot_with_cache(tmp_path)
        frame = root / "frame_cache" / "cam_0" / "chunk-000" / "file-000" / "frame_000000.jpg"
        if failure_mode == "corrupt":
            frame.write_bytes(b"not a valid JPEG")
            error_type = RuntimeError
            message = "cv2.imread failed"
        else:
            error_type = cv2.error
            message = "legacy decoder failed"

            def failing_imread(*args):
                raise cv2.error(message)

            monkeypatch.setattr(cv2, "imread", failing_imread)
        with pytest.raises(error_type, match=message):
            fcp.cached_decode_video_frames(video, [0.0], 0.0)
        assert self.calls == []

    @pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_timestamp_is_cache_miss(self, tmp_path, monkeypatch, timestamp):
        video, _, _, _ = self._select(tmp_path, monkeypatch)
        assert fcp.cached_decode_video_frames(video, [timestamp], 0.001) is self.fallback_result

    @pytest.mark.parametrize("expected_format", [None, "jpg"])
    def test_png_presence_does_not_change_legacy_default(self, tmp_path, monkeypatch, expected_format):
        root, video = _make_lerobot_with_cache(tmp_path)
        _make_png_cache(tmp_path)
        if expected_format:
            monkeypatch.setenv(fcp.ENV_IMAGE_FORMAT, expected_format)
        output = fcp.cached_decode_video_frames(video, [0.0], 0.0, return_uint8=True)
        legacy = fcp._read_jpg_as_frame_tensor(
            root / "frame_cache" / "cam_0" / "chunk-000" / "file-000" / "frame_000000.jpg",
            return_uint8=True,
        )
        assert torch.equal(output[0], legacy)
        assert self.calls == []

    @pytest.mark.parametrize("metadata_present", [False, True])
    def test_missing_png_subtree_cannot_fall_through_to_jpeg(self, tmp_path, monkeypatch, metadata_present):
        root, video = _make_lerobot_with_cache(tmp_path)
        cache_root = root / "frame_cache_png"
        if metadata_present:
            cache_root.mkdir()
            (cache_root / fcp.CACHE_META_NAME).write_text(json.dumps({"fps": 30, "image_format": "png"}))
        else:
            monkeypatch.setenv(fcp.ENV_IMAGE_FORMAT, "png")
        monkeypatch.setenv(fcp.ENV_ROOT_OVERRIDE, str(cache_root))
        monkeypatch.setenv(fcp.ENV_STRICT, "true")
        with pytest.raises(RuntimeError, match="no cache dir"):
            fcp.cached_decode_video_frames(video, [0.0], 0.0)
        assert self.calls == []

    def test_missing_legacy_override_preserves_previous_fallback(self, tmp_path, monkeypatch):
        root, video = _make_lerobot_with_cache(tmp_path)
        monkeypatch.setenv(fcp.ENV_ROOT_OVERRIDE, str(root / "absent_legacy_cache"))
        assert fcp.cached_decode_video_frames(video, [0.0], 0.0).shape == (1, 3, 16, 16)
        assert self.calls == []

    @pytest.mark.parametrize("expected_format", ["jpg", "webp"])
    def test_explicit_format_mismatch_or_invalid_is_refused(self, tmp_path, monkeypatch, expected_format):
        video, _, _, _ = self._select(tmp_path, monkeypatch)
        monkeypatch.setenv(fcp.ENV_IMAGE_FORMAT, expected_format)
        monkeypatch.setenv(fcp.ENV_STRICT, "true")
        with pytest.raises(RuntimeError, match="does not match requested|unsupported FRAME_CACHE_IMAGE_FORMAT"):
            fcp.cached_decode_video_frames(video, [0.0], 0.0)
        assert self.calls == []

    def test_explicit_png_cannot_use_legacy_jpeg_metadata(self, tmp_path, monkeypatch):
        _, video = _make_lerobot_with_cache(tmp_path)
        monkeypatch.setenv(fcp.ENV_IMAGE_FORMAT, "png")
        monkeypatch.setenv(fcp.ENV_STRICT, "true")
        with pytest.raises(RuntimeError, match="does not match requested 'png'"):
            fcp.cached_decode_video_frames(video, [0.0], 0.0)
        assert self.calls == []

    def test_registry_and_symlink_select_png_over_global_jpeg(self, tmp_path, monkeypatch):
        root, _ = _make_lerobot_with_cache(tmp_path)
        _, cache_root, _, expected = _make_png_cache(tmp_path)
        view = tmp_path / "view"
        view.mkdir()
        (view / "videos").symlink_to(root / "videos", target_is_directory=True)
        view_video = view / "videos" / "cam_0" / "chunk-000" / "file-000.mp4"
        monkeypatch.setenv(fcp.ENV_ROOT_OVERRIDE, str(root / "frame_cache"))
        monkeypatch.setenv(fcp.ENV_NUM_VARIANTS, "10")
        fcp.register_frame_cache_override(root, cache_root, num_variants=1)
        output = fcp.cached_decode_video_frames(view_video, [0.0], 0.0, return_uint8=True)
        assert torch.equal(output, expected[:1])
        assert self.calls == []

    @pytest.mark.parametrize("return_uint8", [False, True])
    def test_png_and_jpeg_share_hook_pipeline(self, tmp_path, monkeypatch, return_uint8):
        root, video = _make_lerobot_with_cache(tmp_path)
        _, cache_root, _, expected = _make_png_cache(tmp_path)
        captured = []

        def paint(img_bgr, hook_video, frame_index):
            captured.append((img_bgr.copy(), hook_video, frame_index))
            img_bgr[:] = (21, 89, 203)
            return img_bgr

        def next_hook(img_bgr, hook_video, frame_index):
            assert np.all(img_bgr == (21, 89, 203))
            img_bgr[..., 1] += 1
            return img_bgr

        fcp.register_post_decode_hook(paint)
        fcp.register_post_decode_hook(next_hook)
        monkeypatch.setenv(fcp.ENV_ROOT_OVERRIDE, str(cache_root))
        png_output = fcp.cached_decode_video_frames(video, [1 / 30], 0.0, return_uint8=return_uint8)
        assert np.array_equal(captured[0][0], expected[1].permute(1, 2, 0).numpy()[..., ::-1])
        assert captured[0][1:] == (video, 1)
        assert captured[0][0].dtype == np.uint8
        monkeypatch.setenv(fcp.ENV_ROOT_OVERRIDE, str(root / "frame_cache"))
        jpeg_output = fcp.cached_decode_video_frames(video, [1 / 30], 0.0, return_uint8=return_uint8)
        assert torch.equal(png_output, jpeg_output)
        expected_pixel = torch.tensor([203, 90, 21], dtype=torch.uint8)
        if not return_uint8:
            expected_pixel = expected_pixel.float() / 255.0
        assert torch.equal(png_output[0, :, 0, 0], expected_pixel)
        assert captured[1][1:] == (video, 1)
        assert self.calls == []
