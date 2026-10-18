"""predict_and_prelabel.py の unit test。

主に verts_px → LS RectangleLabels の座標変換が round-trip で spec 通り復元できるかを確認。
ultralytics は runtime 依存 (`from ultralytics import YOLO` は run_yolo_predict 内で lazy import)
なので、これらの test は annotation_tool env でも動く。
"""

from __future__ import annotations

import math

import pytest

from scripts.ls_export_to_yolo_obb import ls_rect_to_yolo_obb_verts
from scripts.predict_and_prelabel import (
    _strip_images_prefix,
    build_ls_task,
    verts_px_to_ls_result,
)


# ------------------------------------------------------------
# verts_px_to_ls_result
# ------------------------------------------------------------
class TestVertsPxToLsResult:
    def test_axis_aligned_round_trip(self) -> None:
        """回転なしの axis-aligned rect は %-space に落としても round-trip 一致する。"""
        img_w, img_h = 640, 480
        verts_px = [(100, 100), (200, 100), (200, 150), (100, 150)]
        result = verts_px_to_ls_result(
            class_id=1, verts_px=verts_px, img_w=img_w, img_h=img_h
        )
        v_reconstructed = ls_rect_to_yolo_obb_verts(result["value"], img_w, img_h)
        expected = [(px[0] / img_w * 100, px[1] / img_h * 100) for px in verts_px]
        for got, exp in zip(v_reconstructed, expected):
            assert abs(got[0] - exp[0]) < 1e-6
            assert abs(got[1] - exp[1]) < 1e-6

    def test_non_square_45deg_round_trip(self) -> None:
        """Theory C fix 後は 640x480 の非正方 image でも round-trip 一致 (aspect fix regression)。"""
        img_w, img_h = 640, 480
        cx, cy = 320, 240
        w, h = 100, 50
        theta = math.radians(45)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # center から local (TL, TR, BR, BL) を rotate + translate (all in pixel space)
        local = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        verts_px = []
        for lx, ly in local:
            rx = cos_t * lx - sin_t * ly
            ry = sin_t * lx + cos_t * ly
            verts_px.append((cx + rx, cy + ry))
        result = verts_px_to_ls_result(
            class_id=2, verts_px=verts_px, img_w=img_w, img_h=img_h
        )
        v_reconstructed = ls_rect_to_yolo_obb_verts(result["value"], img_w, img_h)
        expected = [(px[0] / img_w * 100, px[1] / img_h * 100) for px in verts_px]
        for got, exp in zip(v_reconstructed, expected):
            assert abs(got[0] - exp[0]) < 1e-4
            assert abs(got[1] - exp[1]) < 1e-4

    @pytest.mark.parametrize(
        "class_id,expected_name",
        [
            (0, "workspace"),
            (1, "leg"),
            (2, "leg_tip"),
            (3, "hole"),
            (4, "table_top"),
            (5, "hand_right"),
            (6, "hand_left"),
        ],
    )
    def test_class_id_to_name(self, class_id: int, expected_name: str) -> None:
        result = verts_px_to_ls_result(
            class_id=class_id,
            verts_px=[(0, 0), (1, 0), (1, 1), (0, 1)],
            img_w=640,
            img_h=480,
        )
        assert result["value"]["rectanglelabels"] == [expected_name]

    def test_from_name_to_name_type(self) -> None:
        """from_name/to_name/type が label_config.xml と一致する。"""
        result = verts_px_to_ls_result(
            class_id=0,
            verts_px=[(0, 0), (1, 0), (1, 1), (0, 1)],
            img_w=640,
            img_h=480,
        )
        assert result["from_name"] == "label"
        assert result["to_name"] == "image"
        assert result["type"] == "rectanglelabels"

    def test_original_dims_preserved(self) -> None:
        result = verts_px_to_ls_result(
            class_id=0,
            verts_px=[(0, 0), (1, 0), (1, 1), (0, 1)],
            img_w=640,
            img_h=480,
        )
        assert result["original_width"] == 640
        assert result["original_height"] == 480


# ------------------------------------------------------------
# build_ls_task
# ------------------------------------------------------------
class TestBuildLsTask:
    def test_task_with_boxes_has_predictions(self) -> None:
        boxes = [
            {
                "class_id": 1,
                "verts_px": [(10, 10), (20, 10), (20, 20), (10, 20)],
                "conf": 0.9,
            }
        ]
        task = build_ls_task(
            "insert_table_leg/ep0062/frame_00001.png", boxes, 640, 480, "m_lowaug"
        )
        assert "predictions" in task
        assert task["predictions"][0]["model_version"] == "m_lowaug"
        assert task["predictions"][0]["score"] == pytest.approx(0.9)
        assert len(task["predictions"][0]["result"]) == 1

    def test_task_without_boxes_has_no_predictions(self) -> None:
        """検出なしでも task は成立 (predictions field は含めない = LS で空 annotation)。"""
        task = build_ls_task(
            "insert_table_leg/ep0062/frame_00001.png", [], 640, 480, "m_lowaug"
        )
        assert "predictions" not in task
        assert (
            task["data"]["image"]
            == "/data/local-files/?d=images/insert_table_leg/ep0062/frame_00001.png"
        )

    def test_score_averaged_across_boxes(self) -> None:
        boxes = [
            {"class_id": 1, "verts_px": [(0, 0), (1, 0), (1, 1), (0, 1)], "conf": 0.8},
            {"class_id": 2, "verts_px": [(0, 0), (1, 0), (1, 1), (0, 1)], "conf": 0.6},
        ]
        task = build_ls_task("x/y/z.png", boxes, 640, 480, "m_lowaug")
        assert task["predictions"][0]["score"] == pytest.approx(0.7)

    def test_image_url_format(self) -> None:
        task = build_ls_task(
            "finalize_leg/ep0062/frame_02345.png", [], 640, 480, "m_lowaug"
        )
        assert (
            task["data"]["image"]
            == "/data/local-files/?d=images/finalize_leg/ep0062/frame_02345.png"
        )


# ------------------------------------------------------------
# _strip_images_prefix (defensive helper)
# ------------------------------------------------------------
def test_strip_images_prefix_noop_when_no_prefix() -> None:
    assert (
        _strip_images_prefix("insert_table_leg/ep0062/x.png")
        == "insert_table_leg/ep0062/x.png"
    )


def test_strip_images_prefix_strips_when_present() -> None:
    assert (
        _strip_images_prefix("images/insert_table_leg/ep0062/x.png")
        == "insert_table_leg/ep0062/x.png"
    )
