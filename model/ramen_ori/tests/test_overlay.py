"""Unit tests for model/ramen_ori/overlay.py (Issue #122 D-3)。

OverlayRenderer: mock ObbPrecomputedCache を渡して cv2.polylines 呼び出しを検証。
実 cv2 呼び出しは MagicMock で patch し、渡された引数 (points, color, thickness) 確認。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from model.ramen_ori.overlay import (
    CLASS_COLORS_BGR,
    OverlayRenderer,
    _derive_cam_and_mp4_from_video_path,
)


class _FakeCache:
    """ObbPrecomputedCache の最小 stub。lookup_mp4_frame のみ実装。"""

    def __init__(self, covered_cameras: set[str], per_frame_dets: dict | None = None):
        self.covered_cameras = covered_cameras
        # per_frame_dets: {(cam, mp4_rel, frame): {"verts","conf","class_id","valid"}}
        self._store = per_frame_dets or {}

    def lookup_mp4_frame(self, cam_key: str, mp4_relative: str, frame_idx: int):
        if cam_key not in self.covered_cameras:
            return None
        return self._store.get((cam_key, mp4_relative, frame_idx))


# ============================================================================
# _derive_cam_and_mp4_from_video_path
# ============================================================================


def test_derive_cam_mp4_from_standard_path() -> None:
    p = Path("/root/dataset/videos/observation.images.cam_0/chunk-000/file-000.mp4")
    result = _derive_cam_and_mp4_from_video_path(p)
    assert result == ("observation.images.cam_0", "chunk-000/file-000")


def test_derive_cam_mp4_from_non_video_path_returns_none() -> None:
    p = Path("/tmp/random/file.mp4")
    assert _derive_cam_and_mp4_from_video_path(p) is None


def test_derive_cam_mp4_from_too_short_path_returns_none() -> None:
    p = Path("file.mp4")
    assert _derive_cam_and_mp4_from_video_path(p) is None


# ============================================================================
# OverlayRenderer init validation
# ============================================================================


def test_init_conf_threshold_out_of_range_raises() -> None:
    cache = _FakeCache({"cam_0"})
    with pytest.raises(ValueError, match="conf_threshold"):
        OverlayRenderer(cache=cache, conf_threshold=1.5)


def test_init_line_thickness_zero_raises() -> None:
    cache = _FakeCache({"cam_0"})
    with pytest.raises(ValueError, match="line_thickness"):
        OverlayRenderer(cache=cache, line_thickness=0)


def test_init_class_filter_converts_to_set() -> None:
    cache = _FakeCache({"cam_0"})
    r = OverlayRenderer(cache=cache, class_filter=[1, 3, 4])
    assert r.class_filter == {1, 3, 4}


def test_init_class_colors_defaults_to_module_palette() -> None:
    cache = _FakeCache({"cam_0"})
    r = OverlayRenderer(cache=cache)
    assert r.class_colors_bgr[1] == CLASS_COLORS_BGR[1]
    assert r.class_colors_bgr[2] == CLASS_COLORS_BGR[2]  # yellow (0, 255, 255)


# ============================================================================
# __call__: 描画 skip cases
# ============================================================================


def _make_img(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_call_non_video_path_returns_unchanged() -> None:
    cache = _FakeCache({"observation.images.cam_0"})
    r = OverlayRenderer(cache=cache)
    img = _make_img()
    out = r(img, Path("/tmp/random.mp4"), 0)
    assert out is img  # 描画 skip、そのまま


def test_call_wrist_cam_returns_unchanged() -> None:
    """wrist cam は cache.covered_cameras に無い → lookup が None → 描画 skip。"""
    cache = _FakeCache({"observation.images.cam_0"})  # cam_0 のみ covered
    r = OverlayRenderer(cache=cache)
    img = _make_img()
    video_path = Path("/root/videos/observation.images.cam_2/chunk-000/file-000.mp4")
    out = r(img, video_path, 0)
    assert out is img
    assert (out == 0).all()  # 未描画のまま


def test_call_frame_out_of_range_returns_unchanged() -> None:
    """cache.lookup_mp4_frame が None を返す (frame 範囲外) → 描画 skip。"""
    cache = _FakeCache({"observation.images.cam_0"}, per_frame_dets={})
    r = OverlayRenderer(cache=cache)
    img = _make_img()
    video_path = Path("/root/videos/observation.images.cam_0/chunk-000/file-000.mp4")
    out = r(img, video_path, 999)
    assert (out == 0).all()


# ============================================================================
# __call__: 描画 execute (cv2.polylines mock)
# ============================================================================


def _make_dets(valid_dets: list[tuple[list[float], float, int]], top_K: int = 4) -> dict:
    """top_K の valid + padding 混在 det array を作る。valid_dets = [(verts_flat_8, conf, class_id)]。"""
    verts = np.zeros((top_K, 8), dtype=np.float32)
    conf = np.zeros(top_K, dtype=np.float32)
    class_id = np.zeros(top_K, dtype=np.int8)
    valid = np.zeros(top_K, dtype=bool)
    for i, (v, c, cid) in enumerate(valid_dets[:top_K]):
        verts[i] = v
        conf[i] = c
        class_id[i] = cid
        valid[i] = True
    return {"verts": verts, "conf": conf, "class_id": class_id, "valid": valid}


def test_call_draws_valid_det_with_correct_color_and_thickness() -> None:
    """valid + conf 十分な det を color palette に沿って描画。"""
    cache_key = ("observation.images.cam_0", "chunk-000/file-000", 5)
    # normalized [0..1] verts: 中央 100x100 square (640x480 で pixel (270, 190) - (370, 290))
    verts_flat = [0.5, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6]  # xyxyxyxy
    dets = _make_dets([(verts_flat, 0.9, 1)])  # class_id=1 (leg green)
    cache = _FakeCache({"observation.images.cam_0"}, per_frame_dets={cache_key: dets})
    r = OverlayRenderer(cache=cache, conf_threshold=0.30, line_thickness=2)
    img = _make_img()
    video_path = Path("/root/videos/observation.images.cam_0/chunk-000/file-000.mp4")

    with patch("cv2.polylines") as mock_pl:
        r(img, video_path, 5)
        assert mock_pl.call_count == 1
        _, kwargs = mock_pl.call_args
        assert kwargs["color"] == CLASS_COLORS_BGR[1]  # green
        assert kwargs["thickness"] == 2
        assert kwargs["isClosed"] is True


def test_call_skips_det_below_conf_threshold() -> None:
    cache_key = ("observation.images.cam_0", "chunk-000/file-000", 5)
    dets = _make_dets(
        [
            ([0.1] * 8, 0.90, 1),  # 描画対象
            ([0.2] * 8, 0.25, 3),  # conf 0.25 < 0.30 → skip
            ([0.3] * 8, 0.50, 4),  # 描画対象
        ]
    )
    cache = _FakeCache({"observation.images.cam_0"}, per_frame_dets={cache_key: dets})
    r = OverlayRenderer(cache=cache, conf_threshold=0.30)
    with patch("cv2.polylines") as mock_pl:
        r(_make_img(), Path("/r/videos/observation.images.cam_0/chunk-000/file-000.mp4"), 5)
        # 2 個描画 (0.9 と 0.5)
        assert mock_pl.call_count == 2


def test_call_skips_invalid_det() -> None:
    """valid=False の row は描画しない (padding row 想定)。"""
    cache_key = ("observation.images.cam_0", "chunk-000/file-000", 5)
    # 手動で 4 row 中 1 個だけ valid=True
    verts = np.random.rand(4, 8).astype(np.float32)
    conf = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    class_id = np.array([1, 2, 3, 4], dtype=np.int8)
    valid = np.array([True, False, False, False])
    dets = {"verts": verts, "conf": conf, "class_id": class_id, "valid": valid}
    cache = _FakeCache({"observation.images.cam_0"}, per_frame_dets={cache_key: dets})
    r = OverlayRenderer(cache=cache, conf_threshold=0.30)
    with patch("cv2.polylines") as mock_pl:
        r(_make_img(), Path("/r/videos/observation.images.cam_0/chunk-000/file-000.mp4"), 5)
        assert mock_pl.call_count == 1


def test_call_class_filter_restricts_drawing() -> None:
    """class_filter に含まれる class_id のみ描画。"""
    cache_key = ("observation.images.cam_0", "chunk-000/file-000", 5)
    dets = _make_dets(
        [
            ([0.1] * 8, 0.9, 1),  # leg
            ([0.2] * 8, 0.9, 3),  # hole (fitler 内)
            ([0.3] * 8, 0.9, 4),  # table_top (filter 内)
            ([0.4] * 8, 0.9, 6),  # hand_left
        ]
    )
    cache = _FakeCache({"observation.images.cam_0"}, per_frame_dets={cache_key: dets})
    r = OverlayRenderer(cache=cache, class_filter=[3, 4])  # hole + table_top のみ
    with patch("cv2.polylines") as mock_pl:
        r(_make_img(), Path("/r/videos/observation.images.cam_0/chunk-000/file-000.mp4"), 5)
        assert mock_pl.call_count == 2
        drawn_colors = {kwargs["color"] for _, kwargs in mock_pl.call_args_list}
        assert drawn_colors == {CLASS_COLORS_BGR[3], CLASS_COLORS_BGR[4]}


def test_call_verts_scaled_to_image_pixel_dims() -> None:
    """normalized verts [0,1] を image (W, H) pixel に scale。"""
    cache_key = ("observation.images.cam_0", "chunk-000/file-000", 5)
    # verts = 全 (0.5, 0.5)、640x480 image で pixel (320, 240) に集約
    verts_flat = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    dets = _make_dets([(verts_flat, 0.9, 1)])
    cache = _FakeCache({"observation.images.cam_0"}, per_frame_dets={cache_key: dets})
    r = OverlayRenderer(cache=cache)
    with patch("cv2.polylines") as mock_pl:
        r(_make_img(480, 640), Path("/r/videos/observation.images.cam_0/chunk-000/file-000.mp4"), 5)
        args, _ = mock_pl.call_args
        pts = args[1][0]  # list of arrays, 最初 [0]
        # pts shape (4, 1, 2)、全て (320, 240) 付近
        for pt in pts.reshape(-1, 2):
            assert pt[0] == 320  # x
            assert pt[1] == 240  # y


def test_default_leg_tip_color_is_yellow() -> None:
    """leg_tip = yellow (0, 255, 255) BGR、preview mosaic 決定通り。"""
    assert CLASS_COLORS_BGR[2] == (0, 255, 255)


def test_all_seven_classes_have_colors() -> None:
    for cls_id in range(7):
        assert cls_id in CLASS_COLORS_BGR


# ============================================================================
# Issue #129 Phase D (2026-08-31): draw_overlay_on_tensor (inline 経路)
# ============================================================================


class TestDrawOverlayOnTensor:
    """RAMEN-Ori dataloader inline 経路の tensor helper (Phase D)。

    - Tensor input (3, H, W) float [0,1] RGB → 描画 → 同 shape/dtype で return
    - all valid=False (wrist cam scenario) → no-op (image 変化なし)
    - conf_threshold / class_filter 適用
    - dtype / device 保存
    """

    def _make_tensor(self, h: int = 224, w: int = 224):
        import torch
        # 0.5 でフラット埋め (描画があれば cv2 pixel が overwrite)
        return torch.full((3, h, w), 0.5, dtype=torch.float32)

    def _obb_single_box(self, class_id: int = 3, conf: float = 0.9):
        """Center 付近に 1 個 valid 箱を置く OBB tensor (verts, conf, class_id, valid)。"""
        import numpy as np
        verts = np.zeros((4, 8), dtype=np.float32)
        # 中心 (0.4, 0.4) ~ (0.6, 0.6) の正方形 (normalized [0,1])
        verts[0] = [0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6]
        conf_arr = np.zeros(4, dtype=np.float32)
        conf_arr[0] = conf
        class_id_arr = np.zeros(4, dtype=np.int64)
        class_id_arr[0] = class_id
        valid = np.zeros(4, dtype=bool)
        valid[0] = True
        return verts, conf_arr, class_id_arr, valid

    def test_shape_and_dtype_preserved(self) -> None:
        from model.ramen_ori.overlay import draw_overlay_on_tensor
        img = self._make_tensor()
        v, c, k, val = self._obb_single_box()
        out = draw_overlay_on_tensor(img, v, c, k, val)
        assert out.shape == img.shape
        assert out.dtype == img.dtype

    def test_all_invalid_yields_noop(self) -> None:
        """Wrist cam scenario: 全 detection invalid → 画像はほぼ不変 (小 rounding のみ)。"""
        import numpy as np
        import torch
        from model.ramen_ori.overlay import draw_overlay_on_tensor
        img = self._make_tensor()
        verts = np.zeros((4, 8), dtype=np.float32)
        conf = np.zeros(4, dtype=np.float32)
        class_id = np.zeros(4, dtype=np.int64)
        valid = np.zeros(4, dtype=bool)  # 全 False
        out = draw_overlay_on_tensor(img, verts, conf, class_id, valid)
        # float32 → uint8 → float32 round-trip で ~1/255 の差、それ以下なら実質不変
        torch.testing.assert_close(out, img, atol=2.5 / 255.0, rtol=0.0)

    def test_conf_below_threshold_skipped(self) -> None:
        """conf_threshold 未満は描画 skip = no-op と同じ effect。"""
        import torch
        from model.ramen_ori.overlay import draw_overlay_on_tensor
        img = self._make_tensor()
        v, c, k, val = self._obb_single_box(conf=0.20)
        out = draw_overlay_on_tensor(img, v, c, k, val, conf_threshold=0.30)
        torch.testing.assert_close(out, img, atol=2.5 / 255.0, rtol=0.0)

    def test_class_filter_excludes(self) -> None:
        """class_filter に含まれない class は skip。"""
        import torch
        from model.ramen_ori.overlay import draw_overlay_on_tensor
        img = self._make_tensor()
        v, c, k, val = self._obb_single_box(class_id=3)  # hole
        # {1, 4} filter → 3 (hole) 除外
        out = draw_overlay_on_tensor(img, v, c, k, val, class_filter={1, 4})
        torch.testing.assert_close(out, img, atol=2.5 / 255.0, rtol=0.0)

    def test_drawing_actually_modifies_image_when_valid(self) -> None:
        """valid + conf >= threshold + class 通過なら実描画され、pixel が変化。"""
        import torch
        from model.ramen_ori.overlay import draw_overlay_on_tensor
        img = self._make_tensor()
        v, c, k, val = self._obb_single_box(class_id=3, conf=0.99)
        out = draw_overlay_on_tensor(img, v, c, k, val, conf_threshold=0.30, line_thickness=2)
        # 描画されたら pixel が変化するはず
        diff = (out - img).abs().sum().item()
        assert diff > 0.5, f"expected drawing to change pixels, got diff={diff}"

    def test_wrong_shape_raises(self) -> None:
        import torch
        from model.ramen_ori.overlay import draw_overlay_on_tensor
        img_bad = torch.zeros(224, 224)  # (H, W) 2D
        v, c, k, val = self._obb_single_box()
        with pytest.raises(ValueError, match=r"\(3, H, W\)"):
            draw_overlay_on_tensor(img_bad, v, c, k, val)

    def test_accepts_torch_tensor_obb_data(self) -> None:
        """OBB data が torch.Tensor でも動く (numpy に変換される)。"""
        import torch
        from model.ramen_ori.overlay import draw_overlay_on_tensor
        img = self._make_tensor()
        v_np, c_np, k_np, val_np = self._obb_single_box()
        out = draw_overlay_on_tensor(
            img,
            torch.from_numpy(v_np),
            torch.from_numpy(c_np),
            torch.from_numpy(k_np),
            torch.from_numpy(val_np),
        )
        assert out.shape == img.shape
