"""precompute_augmented_frame_cache.py の unit test (Issue #129 case F)。

# カバー範囲

- `_derive_frame_info`: v0 path → (training_cam, mp4_relative, frame_idx) 抽出
- `_load_camera_map`: marker load / fallback
- `_parse_class_filter`: JSON list-of-int parse
- `_sample_augs`: pool から 0..max_num 個 sample
- `_draw_overlay`: BGR image に polylines 描画 (色 / thickness 検証)
- `_worker_init` + `_process_one_frame`: end-to-end (v0 → aug + overlay bake、
  fake OBB cache + fake camera map、head cam vs wrist cam の分岐)
- Runtime 側 `_resolve_variant_jpg_path` は v0 skip する事の cross-check
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# cv2 が使えるか (Sakura の venv は健全、一部 pixi env では壊れた namespace-only shim
# だったり、そもそも installed でなかったり)。cv2-dependent tests は skip 判定。
try:
    import cv2  # noqa: PLC0415

    _CV2_HEALTHY = hasattr(cv2, "polylines") and hasattr(cv2, "imread")
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _CV2_HEALTHY = False
skip_if_cv2_broken = pytest.mark.skipif(
    not _CV2_HEALTHY, reason="cv2 unavailable or broken in this env (namespace-only shim / missing polylines)"
)


ROOT = Path(__file__).resolve().parents[1]


def load_precompute():
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "precompute_augmented_frame_cache", scripts / "precompute_augmented_frame_cache.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


precompute = load_precompute()


# ---------------------------------------------------------------------------
# _derive_frame_info
# ---------------------------------------------------------------------------


class TestDeriveFrameInfo:
    """v0 jpg path から (training_cam, mp4_relative, frame_idx) 抽出。"""

    def test_head_left_frame_42(self, tmp_path):
        v0 = (
            tmp_path
            / "frame_cache"
            / "observation.images.head_left"
            / "chunk-000"
            / "file-005"
            / "frame_000042.jpg"
        )
        cam, mp4, idx = precompute._derive_frame_info(v0)
        assert cam == "observation.images.head_left"
        assert mp4 == "chunk-000/file-005"
        assert idx == 42

    def test_wrist_frame_zero(self, tmp_path):
        v0 = (
            tmp_path
            / "frame_cache"
            / "observation.images.left_wrist"
            / "chunk-003"
            / "file-010"
            / "frame_000000.jpg"
        )
        cam, mp4, idx = precompute._derive_frame_info(v0)
        assert cam == "observation.images.left_wrist"
        assert mp4 == "chunk-003/file-010"
        assert idx == 0

    def test_rejects_non_frame_stem(self, tmp_path):
        bad = tmp_path / "cam" / "chunk-000" / "file-000" / "notframe.jpg"
        with pytest.raises(ValueError, match="unexpected jpg name"):
            precompute._derive_frame_info(bad)


# ---------------------------------------------------------------------------
# _load_camera_map
# ---------------------------------------------------------------------------


class TestLoadCameraMap:
    """marker file 読取 / fallback。"""

    def test_marker_present_returns_camera_map(self, tmp_path):
        cache_root = tmp_path / "frame_cache"
        cache_root.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        marker = {
            "schema_version": "team_ramen_training_view_v4",
            "camera_map": {
                "observation.images.head_left": "observation.images.cam_0",
                "observation.images.left_wrist": "observation.images.cam_2",
            },
        }
        (meta_dir / "team_ramen_training_view.json").write_text(json.dumps(marker))
        result = precompute._load_camera_map(cache_root)
        assert result == marker["camera_map"]

    def test_marker_missing_returns_fallback(self, tmp_path):
        cache_root = tmp_path / "frame_cache"
        cache_root.mkdir()
        result = precompute._load_camera_map(cache_root)
        # Fallback = subtask_training.json hardcoded
        assert result["observation.images.head_left"] == "observation.images.cam_0"
        assert result["observation.images.right_wrist"] == "observation.images.cam_3"

    def test_marker_empty_camera_map_returns_fallback(self, tmp_path):
        """marker の camera_map が空 dict でも fallback に落ちる (誤った marker 対策)。"""
        cache_root = tmp_path / "frame_cache"
        cache_root.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "team_ramen_training_view.json").write_text(
            json.dumps({"camera_map": {}})
        )
        result = precompute._load_camera_map(cache_root)
        assert "observation.images.head_left" in result


# ---------------------------------------------------------------------------
# _parse_class_filter
# ---------------------------------------------------------------------------


class TestParseClassFilter:
    def test_none_returns_none(self):
        assert precompute._parse_class_filter(None) is None

    def test_empty_returns_none(self):
        assert precompute._parse_class_filter("") is None
        assert precompute._parse_class_filter("   ") is None

    def test_valid_json_list(self):
        assert precompute._parse_class_filter("[3,4]") == [3, 4]

    def test_invalid_non_int_rejected(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError, match="JSON list of int"):
            precompute._parse_class_filter('["hole"]')


# ---------------------------------------------------------------------------
# _sample_augs
# ---------------------------------------------------------------------------


class TestSampleAugs:
    def test_returns_empty_when_zero_sampled(self):
        # rng.randint(0, X) が 0 になる seed を探す
        pool = [(f"t{i}", MagicMock()) for i in range(5)]
        rng = random.Random(0)
        # 複数試行、少なくとも 1 回は empty が返る事を確認
        empties = 0
        for seed in range(50):
            rng = random.Random(seed)
            got = precompute._sample_augs(pool, max_num=3, rng=rng)
            if got == []:
                empties += 1
        assert empties > 0, "0 個 sample の case が 50 seed で 1 度も出ない (randint 実装疑い)"

    def test_never_exceeds_max(self):
        pool = [(f"t{i}", MagicMock()) for i in range(5)]
        for seed in range(30):
            rng = random.Random(seed)
            got = precompute._sample_augs(pool, max_num=2, rng=rng)
            assert 0 <= len(got) <= 2

    def test_never_exceeds_pool_size(self):
        pool = [(f"t{i}", MagicMock()) for i in range(2)]
        for seed in range(30):
            rng = random.Random(seed)
            got = precompute._sample_augs(pool, max_num=10, rng=rng)
            assert 0 <= len(got) <= 2  # pool cap


# ---------------------------------------------------------------------------
# _draw_overlay (色 / thickness 検証)
# ---------------------------------------------------------------------------


@skip_if_cv2_broken
class TestDrawOverlay:
    """cv2.polylines で色分け枠線を BGR image に描画する部分の検証。"""

    def _fake_per_frame(
        self,
        n_valid: int = 2,
        confs: list[float] | None = None,
        class_ids: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """`ObbPrecomputedCache.lookup_mp4_frame` 相当の shape/dtype の per-frame dict を作る。

        verts: (top_K=4, 8) normalized [0,1] xyxyxyxy
        conf : (top_K,) float32
        class_id: (top_K,) int8
        valid: (top_K,) bool
        """
        top_K = 4
        verts = np.zeros((top_K, 8), dtype=np.float32)
        # 4 頂点、画像の中央付近に小さい四角
        square = np.array([0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6], dtype=np.float32)
        for k in range(top_K):
            verts[k] = square
        conf = np.array(confs if confs is not None else [0.9] * top_K, dtype=np.float32)
        cid = np.array(class_ids if class_ids is not None else [3, 4, 0, 0], dtype=np.int8)
        valid = np.array([True] * n_valid + [False] * (top_K - n_valid), dtype=bool)
        return {"verts": verts, "conf": conf, "class_id": cid, "valid": valid}

    def test_draws_polylines_for_valid_dets(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)  # 中間 gray
        pf = self._fake_per_frame(n_valid=1, class_ids=[3, 0, 0, 0])  # hole = red BGR (0,0,255)
        out = precompute._draw_overlay(img, pf, conf_threshold=0.1, line_thickness=2, class_filter=None)
        # 中央付近に red pixel が出現する事を検出 (BGR: R=255, G=0, B=0 → cv2 stores as (0,0,255))
        red_mask = (out[:, :, 0] == 0) & (out[:, :, 1] == 0) & (out[:, :, 2] == 255)
        assert red_mask.any(), "hole (red) の polyline が描画されていない"

    def test_skips_low_conf(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        pf = self._fake_per_frame(n_valid=1, confs=[0.1, 0.0, 0.0, 0.0], class_ids=[3, 0, 0, 0])
        out = precompute._draw_overlay(img, pf, conf_threshold=0.3, line_thickness=2, class_filter=None)
        # conf < 0.3 なので何も描画されないはず、image が unchanged
        assert (out == 128).all()

    def test_skips_invalid(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        pf = self._fake_per_frame(n_valid=0)  # 全 valid=False
        out = precompute._draw_overlay(img, pf, conf_threshold=0.1, line_thickness=2, class_filter=None)
        assert (out == 128).all()

    def test_class_filter_excludes_others(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        pf = self._fake_per_frame(n_valid=2, class_ids=[3, 4, 0, 0])  # hole + table_top
        out = precompute._draw_overlay(img, pf, conf_threshold=0.1, line_thickness=2, class_filter={3})
        # red (hole) は描画されるが blue (table_top) は描画されない
        red = (out[:, :, 0] == 0) & (out[:, :, 1] == 0) & (out[:, :, 2] == 255)
        blue = (out[:, :, 0] == 255) & (out[:, :, 1] == 0) & (out[:, :, 2] == 0)
        assert red.any()
        assert not blue.any()

    def test_uses_class_colors_bgr_palette(self):
        """model/ramen_ori/overlay.py の CLASS_COLORS_BGR と一致する事の cross-check。"""
        from model.ramen_ori.overlay import CLASS_COLORS_BGR as external
        assert precompute.CLASS_COLORS_BGR == external


# ---------------------------------------------------------------------------
# End-to-end (fake v0 + fake cache → process_one_frame → verify v_i)
# ---------------------------------------------------------------------------


@skip_if_cv2_broken
class TestProcessOneFrameEndToEnd:
    """`_process_one_frame` を worker global を手動 populate して単体 test。

    - head_left cam: overlay 描画 (fake per_frame) が v_i に反映される
    - wrist cam: overlay skip (aug のみ)、per_frame lookup が None
    - v0 は無変更 (rerun safety)
    """

    def _write_v0(self, tmp_path: Path, training_cam: str, frame_idx: int = 0) -> Path:
        """<tmp>/frame_cache/<cam>/chunk-000/file-000/frame_XXXXXX.jpg に fake v0 を書く。"""
        d = tmp_path / "frame_cache" / training_cam / "chunk-000" / "file-000"
        d.mkdir(parents=True)
        img = np.full((64, 64, 3), 200, dtype=np.uint8)  # light gray canvas
        # PIL は RGB、cv2 は BGR、簡単化のため全 pixel 200 (color invariant)
        Image.fromarray(img).save(d / f"frame_{frame_idx:06d}.jpg", format="JPEG", quality=90)
        return d / f"frame_{frame_idx:06d}.jpg"

    def _populate_worker_globals(
        self,
        cache: MagicMock,
        camera_map: dict[str, str],
        overlay_conf: float = 0.1,
        overlay_thickness: int = 2,
        class_filter: set[int] | None = None,
        jpg_quality: int = 90,
        max_num_photo: int = 3,
        max_num_geo: int = 1,
    ) -> None:
        """`_worker_init` の side effect を手動再現 (mp.Pool 経由せずに単体 test)。"""
        precompute._worker_cache = cache
        precompute._worker_camera_map = dict(camera_map)
        precompute._worker_photometric_pool, precompute._worker_geometric_pool = (
            precompute._build_aug_pools()
        )
        precompute._worker_config = {
            "overlay_conf": overlay_conf,
            "overlay_thickness": overlay_thickness,
            "overlay_class_filter": class_filter,
            "jpg_quality": jpg_quality,
            "max_num_photometric": max_num_photo,
            "max_num_geometric": max_num_geo,
        }

    def _fake_cache_with_lookup(
        self,
        want_cam: str,
        per_frame: dict[str, np.ndarray] | None,
    ) -> MagicMock:
        """`lookup_mp4_frame` が want_cam のみ per_frame を返す fake ObbPrecomputedCache。"""
        cache = MagicMock()

        def lookup(source_cam, mp4_rel, frame_idx):
            if source_cam == want_cam:
                return per_frame
            return None

        cache.lookup_mp4_frame.side_effect = lookup
        return cache

    def test_head_left_generates_v_with_overlay(self, tmp_path):
        v0 = self._write_v0(tmp_path, "observation.images.head_left", frame_idx=0)
        v0_bytes_before = v0.read_bytes()

        # OBB fake: hole (class 3 = red BGR (0,0,255)) 1 個描画
        top_K = 4
        verts = np.tile(
            np.array([0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7], dtype=np.float32),
            (top_K, 1),
        )
        pf = {
            "verts": verts,
            "conf": np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32),
            "class_id": np.array([3, 0, 0, 0], dtype=np.int8),
            "valid": np.array([True, False, False, False], dtype=bool),
        }
        cache = self._fake_cache_with_lookup("observation.images.cam_0", pf)
        camera_map = {"observation.images.head_left": "observation.images.cam_0"}
        self._populate_worker_globals(cache, camera_map, overlay_conf=0.1)

        # num_variants=3 → v1, v2 が生成
        _, saved, _ = precompute._process_one_frame((v0, 3, False))
        assert saved == 2
        assert (v0.parent / "frame_000000_v1.jpg").is_file()
        assert (v0.parent / "frame_000000_v2.jpg").is_file()

        # v0 は無変更 (rerun safety)
        assert v0.read_bytes() == v0_bytes_before

        # v1 に red pixel が入っている事 (overlay 描画 confirm)
        v1_img = cv2.imread(str(v0.parent / "frame_000000_v1.jpg"), cv2.IMREAD_COLOR)
        # 完全 red pixel は jpg 圧縮で微妙にズレる、緩めに R>150 B<100 G<100
        red_ish = (v1_img[:, :, 2] > 150) & (v1_img[:, :, 1] < 100) & (v1_img[:, :, 0] < 100)
        assert red_ish.any(), "v1 に overlay の red 枠線が確認できない"

    def test_wrist_cam_skips_overlay_but_generates_aug(self, tmp_path):
        v0 = self._write_v0(tmp_path, "observation.images.left_wrist", frame_idx=0)
        cache = self._fake_cache_with_lookup("observation.images.cam_0", None)  # cam_2 は covered_cameras 外
        camera_map = {"observation.images.left_wrist": "observation.images.cam_2"}
        # cache は cam_2 に対して None を返すよう mock (lookup_mp4_frame が None)
        cache.lookup_mp4_frame.side_effect = lambda *a, **kw: None
        self._populate_worker_globals(cache, camera_map)

        _, saved, _ = precompute._process_one_frame((v0, 3, False))
        assert saved == 2
        v1_path = v0.parent / "frame_000000_v1.jpg"
        assert v1_path.is_file()

        # wrist v1 は overlay 描画されない、red pixel 出ないはず (元 gray canvas + photometric/geometric aug のみ)
        v1_img = cv2.imread(str(v1_path), cv2.IMREAD_COLOR)
        red_strong = (v1_img[:, :, 2] > 200) & (v1_img[:, :, 1] < 50) & (v1_img[:, :, 0] < 50)
        assert not red_strong.any(), "wrist なのに overlay 描画されている"

    def test_skip_existing_variants(self, tmp_path):
        """既存 v_i があれば --force 無しで skip。"""
        v0 = self._write_v0(tmp_path, "observation.images.head_left", frame_idx=0)
        # v1 を事前 write
        pre_v1 = v0.parent / "frame_000000_v1.jpg"
        Image.fromarray(np.full((64, 64, 3), 111, dtype=np.uint8)).save(pre_v1, format="JPEG")
        pre_v1_bytes = pre_v1.read_bytes()

        cache = self._fake_cache_with_lookup("observation.images.cam_0", None)
        cache.lookup_mp4_frame.side_effect = lambda *a, **kw: None
        self._populate_worker_globals(cache, {"observation.images.head_left": "observation.images.cam_0"})

        _, saved, skipped = precompute._process_one_frame((v0, 3, False))
        # v1 skip、v2 新規 → saved=1, skipped=1
        assert saved == 1
        assert skipped == 1
        assert pre_v1.read_bytes() == pre_v1_bytes  # 上書きされない

    def test_force_regenerates_existing(self, tmp_path):
        v0 = self._write_v0(tmp_path, "observation.images.head_left", frame_idx=0)
        pre_v1 = v0.parent / "frame_000000_v1.jpg"
        Image.fromarray(np.full((64, 64, 3), 111, dtype=np.uint8)).save(pre_v1, format="JPEG")
        pre_v1_bytes = pre_v1.read_bytes()

        cache = self._fake_cache_with_lookup("observation.images.cam_0", None)
        cache.lookup_mp4_frame.side_effect = lambda *a, **kw: None
        self._populate_worker_globals(cache, {"observation.images.head_left": "observation.images.cam_0"})

        _, saved, skipped = precompute._process_one_frame((v0, 3, True))
        assert saved == 2
        assert skipped == 0
        assert pre_v1.read_bytes() != pre_v1_bytes  # 上書きされている
