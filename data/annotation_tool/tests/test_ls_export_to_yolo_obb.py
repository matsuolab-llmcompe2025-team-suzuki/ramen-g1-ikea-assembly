"""ls_export_to_yolo_obb.py の round-trip test。

LS format → YOLO OBB verts → LS format で bit-perfect (float tolerance 内) 一致を確認。
Issue #53 の aspect distortion fix 対応: 全 conversion 関数に img_w/img_h を渡す。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ls_export_to_yolo_obb import (
    CLASS_NAME_TO_ID,
    ls_rect_to_yolo_obb_verts,
    parse_ls_export,
    write_yolo_obb_txts,
    yolo_obb_verts_to_ls_rect,
)


# ------------------------------------------------------------
# unit: 座標変換 round-trip
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [
        # 回転なし: 素直な axis-aligned box
        {"x": 10, "y": 20, "width": 30, "height": 40, "rotation": 0.0,
         "rectanglelabels": ["leg"]},
        # 45 度回転
        {"x": 25, "y": 25, "width": 20, "height": 10, "rotation": 45.0,
         "rectanglelabels": ["hand_right"]},
        # 90 度: top-right が top-left の真下に来る
        {"x": 10, "y": 10, "width": 20, "height": 15, "rotation": 90.0,
         "rectanglelabels": ["hole"]},
        # 負の rotation (反時計回り)
        {"x": 50, "y": 50, "width": 20, "height": 10, "rotation": -30.0,
         "rectanglelabels": ["leg_tip"]},
        # 微小 rotation
        {"x": 5, "y": 5, "width": 40, "height": 25, "rotation": 0.5,
         "rectanglelabels": ["table_top"]},
    ],
)
@pytest.mark.parametrize("img_dims", [(640, 480), (500, 500), (1280, 720)])
def test_round_trip_ls_to_verts_to_ls(value: dict, img_dims: tuple[int, int]) -> None:
    """LS value → verts → LS value で数値が一致することを検証。

    Theory C (pixel-space rotation) fix 後は non-square (4:3, 16:9) でも
    round-trip が bit-perfect になる。
    """
    img_w, img_h = img_dims
    verts = ls_rect_to_yolo_obb_verts(value, img_w, img_h)
    assert len(verts) == 4
    reconstructed = yolo_obb_verts_to_ls_rect(verts, img_w, img_h)
    tol = 1e-6
    assert abs(reconstructed["x"] - value["x"]) < tol
    assert abs(reconstructed["y"] - value["y"]) < tol
    assert abs(reconstructed["width"] - value["width"]) < tol
    assert abs(reconstructed["height"] - value["height"]) < tol
    # rotation は atan2 由来なので modulo 360 で比較 (-180..180 に normalize)
    def _norm(a: float) -> float:
        return (a + 180) % 360 - 180
    assert abs(_norm(reconstructed["rotation"]) - _norm(value["rotation"])) < tol


def test_zero_rotation_is_axis_aligned() -> None:
    """rotation=0 は 4 頂点が axis-aligned rectangle になるはず (non-square でも)。"""
    img_w, img_h = 640, 480
    value = {
        "x": 10, "y": 20, "width": 30, "height": 40, "rotation": 0.0,
        "rectanglelabels": ["leg"],
    }
    verts = ls_rect_to_yolo_obb_verts(value, img_w, img_h)
    # rotation=0 なので pixel rotation 適用しても値は同じ、%-value に戻すと元通り
    expected = [(10.0, 20.0), (40.0, 20.0), (40.0, 60.0), (10.0, 60.0)]
    for got, want in zip(verts, expected):
        assert abs(got[0] - want[0]) < 1e-9
        assert abs(got[1] - want[1]) < 1e-9


def test_90deg_rotation_on_square_image() -> None:
    """square image (aspect 1:1) では 90° 回転で TL→TR ベクトルが下向きになる。"""
    img_w = img_h = 100
    value = {
        "x": 0, "y": 0, "width": 10, "height": 5, "rotation": 90.0,
        "rectanglelabels": ["leg"],
    }
    verts = ls_rect_to_yolo_obb_verts(value, img_w, img_h)
    # pixel space: TL (0,0), TR (10, 0) → rotated 90° CW → (0, 10)
    # BR (10, 5) → (-5, 10), BL (0, 5) → (-5, 0)
    # square image なので %-value と pixel value は同じ (100 で割ると同値)
    expected = [(0.0, 0.0), (0.0, 10.0), (-5.0, 10.0), (-5.0, 0.0)]
    for got, want in zip(verts, expected):
        assert abs(got[0] - want[0]) < 1e-9
        assert abs(got[1] - want[1]) < 1e-9


def test_non_square_pixel_rotation_matches_ls_konva() -> None:
    """640x480 に -7.32° 回転を掛けた時、pixel space で計算した頂点と一致する。

    Issue #53 で発覚したバグ (%-space rotation) の regression prevention。
    hand_right ケース (x=54.86%, y=36.84%, w=21.59%, h=25.24%, rot=352.68°=-7.32°)。
    """
    import math
    img_w, img_h = 640, 480
    value = {
        "x": 54.86, "y": 36.84, "width": 21.59, "height": 25.24,
        "rotation": 352.68,
        "rectanglelabels": ["hand_right"],
    }
    verts_pct = ls_rect_to_yolo_obb_verts(value, img_w, img_h)

    # 手計算 (Theory C = LS/Konva 準拠) の期待値
    theta = math.radians(352.68)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    tl_x_px = 54.86 / 100 * img_w  # 351.104
    tl_y_px = 36.84 / 100 * img_h  # 176.832
    pw = 21.59 / 100 * img_w        # 138.176
    ph = 25.24 / 100 * img_h        # 121.152
    corners = [(0, 0), (pw, 0), (pw, ph), (0, ph)]
    expected = []
    for cx, cy in corners:
        rx = cos_t * cx - sin_t * cy
        ry = sin_t * cx + cos_t * cy
        expected.append(
            ((tl_x_px + rx) / img_w * 100, (tl_y_px + ry) / img_h * 100)
        )
    for got, want in zip(verts_pct, expected):
        assert abs(got[0] - want[0]) < 1e-9
        assert abs(got[1] - want[1]) < 1e-9


# ------------------------------------------------------------
# integration: LS export → YOLO OBB txt round-trip
# ------------------------------------------------------------
def _make_ls_export(tmp_path: Path) -> Path:
    """テスト用の LS export JSON を作る。frame_00060 に 2 boxes。"""
    export = [
        {
            "id": 1,
            "data": {
                "image": "/data/local-files/?d=insert_table_leg/ep0062/frame_00060.png",
            },
            "annotations": [
                {
                    "result": [
                        {
                            "type": "rectanglelabels",
                            "original_width": 640,
                            "original_height": 480,
                            "value": {
                                "x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0,
                                "rotation": 15.0,
                                "rectanglelabels": ["leg"],
                            },
                        },
                        {
                            "type": "rectanglelabels",
                            "original_width": 640,
                            "original_height": 480,
                            "value": {
                                "x": 50.0, "y": 50.0, "width": 20.0, "height": 10.0,
                                "rotation": -30.0,
                                "rectanglelabels": ["hand_right"],
                            },
                        },
                    ],
                },
            ],
        },
    ]
    p = tmp_path / "export.json"
    p.write_text(json.dumps(export))
    return p


def test_parse_ls_export_rel_path_and_class_id(tmp_path: Path) -> None:
    export_path = _make_ls_export(tmp_path)
    parsed = parse_ls_export(export_path)
    assert len(parsed) == 1
    # `images/` prefix は strip、階層は `/` 区切りで保持
    rel_path = "insert_table_leg/ep0062/frame_00060"
    assert rel_path in parsed
    boxes = parsed[rel_path]
    assert len(boxes) == 2
    # class_id チェック (leg=1, hand_right=5)
    assert boxes[0][0] == CLASS_NAME_TO_ID["leg"] == 1
    assert boxes[1][0] == CLASS_NAME_TO_ID["hand_right"] == 5
    # verts 数
    assert len(boxes[0][1]) == 4
    assert len(boxes[1][1]) == 4


def test_write_yolo_obb_txt_format(tmp_path: Path) -> None:
    export_path = _make_ls_export(tmp_path)
    out_dir = tmp_path / "labels"
    parsed = parse_ls_export(export_path)
    n = write_yolo_obb_txts(parsed, out_dir)
    assert n == 1
    # hierarchical output: labels/insert_table_leg/ep0062/frame_00060.txt
    txt_path = out_dir / "insert_table_leg" / "ep0062" / "frame_00060.txt"
    assert txt_path.exists()
    txt = txt_path.read_text().strip().splitlines()
    assert len(txt) == 2
    # 1 line = class_id + 8 float
    for line in txt:
        parts = line.split()
        assert len(parts) == 9
        assert parts[0] in {"1", "5"}
        for v in parts[1:]:
            f = float(v)
            # % ÷ 100 = 0..1 の正規化。負値もあり得る (rotation により image 外に出る場合)
            assert -1.5 <= f <= 1.5, f"unexpected normalized value: {v}"


def test_full_round_trip_from_ls_export(tmp_path: Path) -> None:
    """LS export → txt 書き出し → verts 再構築 → LS value 復元 で数値一致。"""
    export_path = _make_ls_export(tmp_path)
    out_dir = tmp_path / "labels"
    parsed = parse_ls_export(export_path)
    write_yolo_obb_txts(parsed, out_dir)

    # 復元 (hierarchical path、img_dims は fixture と同じ 640x480)
    img_w, img_h = 640, 480
    txt = (out_dir / "insert_table_leg" / "ep0062" / "frame_00060.txt").read_text().strip().splitlines()
    original_values = [
        {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0, "rotation": 15.0},
        {"x": 50.0, "y": 50.0, "width": 20.0, "height": 10.0, "rotation": -30.0},
    ]
    tol = 1e-3  # txt は %.6f で保存、% 単位に戻すため精度 loss あり (pixel scale 分やや大きめ)
    for line, want in zip(txt, original_values):
        parts = line.split()
        # 0..1 → % に戻す
        verts_pct = [
            (float(parts[1 + i * 2]) * 100.0, float(parts[2 + i * 2]) * 100.0)
            for i in range(4)
        ]
        got = yolo_obb_verts_to_ls_rect(verts_pct, img_w, img_h)
        assert abs(got["x"] - want["x"]) < tol
        assert abs(got["y"] - want["y"]) < tol
        assert abs(got["width"] - want["width"]) < tol
        assert abs(got["height"] - want["height"]) < tol
        def _norm(a: float) -> float:
            return (a + 180) % 360 - 180
        assert abs(_norm(got["rotation"]) - _norm(want["rotation"])) < tol
