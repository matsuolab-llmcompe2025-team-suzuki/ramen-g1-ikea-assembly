"""Label Studio export JSON → YOLO OBB txt に変換する。

Epic #43 / Issue #44、Issue #53 で aspect distortion fix。

Label Studio の `RectangleLabels canRotate=true` 出力形式:
    result: [
      {
        "value": {"x": %, "y": %, "width": %, "height": %,
                  "rotation": deg, "rectanglelabels": ["leg"]},
        "original_width": 640, "original_height": 480,
        "type": "rectanglelabels",
      },
      ...
    ]
    (x, y) は **top-left 座標**、rotation は top-left を軸に時計回り (LS 慣習)。

**重要: 回転は pixel-space で適用する** (LS/Konva の実挙動):
    LS UI は Konva.js で描画し、rotation を pixel 座標系で適用する。
    座標 (x, y, w, h) は image dims の % だが、rotation を掛ける時は
    pixel_w = w/100 * img_w, pixel_h = h/100 * img_h に一度戻してから回転する。
    %-space で直接回転すると 4:3 の image で aspect が cos/sin に混ざり、
    5px 前後の系統誤差 (小回転で右下方向にずれ) が入るため NG (Issue #53 で発覚)。

YOLO OBB (ultralytics):
    class_id x1 y1 x2 y2 x3 y3 x4 y4    # 0-1 正規化、4 頂点
    頂点順は top-left → top-right → bottom-right → bottom-left。

Usage:
    pixi run ls-to-yolo --export workspace/exports/project-1.json \\
                        --out-dir datasets/v1_train/labels/

出力:
    <out-dir>/<image_stem>.txt
    (LS の task.data.image が `/data/local-files/?d=insert_table_leg/ep0062/frame_00060.png`
     の形式の場合、image_stem = insert_table_leg__ep0062__frame_00060 に flatten される)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import TypedDict


# ---- class map (class_map.json と同期。読むより hard-code で確実性重視) ----
CLASS_NAME_TO_ID: dict[str, int] = {
    "workspace": 0,
    "leg": 1,
    "leg_tip": 2,
    "hole": 3,
    "table_top": 4,
    "hand_right": 5,
    "hand_left": 6,
}
CLASS_ID_TO_NAME: dict[int, str] = {v: k for k, v in CLASS_NAME_TO_ID.items()}


class LSValue(TypedDict, total=False):
    """LS RectangleLabels+rotation の value 部 (LS 1.23 export 形式)。"""

    x: float          # % (0..100)
    y: float          # % (0..100)
    width: float      # % (0..100)
    height: float     # % (0..100)
    rotation: float   # 度、top-left 中心に時計回り (LS 慣習)
    rectanglelabels: list[str]


# ------------------------------------------------------------
# 座標変換 (LS → YOLO OBB)
# ------------------------------------------------------------
def ls_rect_to_yolo_obb_verts(
    value: LSValue, img_w: int, img_h: int
) -> list[tuple[float, float]]:
    """LS RectangleLabels+rotation → 4 vertices in **% units** (0..100)。

    LS の rotation は **top-left 頂点を軸に時計回り** (y-down 座標系)、かつ
    Konva.js の実挙動として **pixel 座標系で回転を適用**する (Issue #53 で確認)。

    処理手順:
    1. (x, y, w, h) の %-value を pixel に戻す (img_w, img_h でスケール)
    2. TL 起点の local 矩形 (pixel) に rotation を適用
    3. 4 corner を pixel から %-value に戻す (各軸独立に正規化)

    Args:
        value: LS の `result[].value` (x, y, w, h, rotation 全て %)
        img_w: 元画像の pixel width (task の original_width、必須)
        img_h: 元画像の pixel height (task の original_height、必須)

    Returns:
        [(x1_pct, y1_pct), (x2_pct, y2_pct), (x3_pct, y3_pct), (x4_pct, y4_pct)]
        top-left → top-right → bottom-right → bottom-left の順。
    """
    x = float(value["x"])
    y = float(value["y"])
    w = float(value["width"])
    h = float(value["height"])
    rot_deg = float(value.get("rotation", 0.0))
    theta = math.radians(rot_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # %-value を pixel に戻す (LS/Konva は pixel space で rotate する)
    tl_x_px = x / 100.0 * img_w
    tl_y_px = y / 100.0 * img_h
    pixel_w = w / 100.0 * img_w
    pixel_h = h / 100.0 * img_h

    # local pixel corners: TL, TR, BR, BL
    corners_local_px = [
        (0.0, 0.0),
        (pixel_w, 0.0),
        (pixel_w, pixel_h),
        (0.0, pixel_h),
    ]

    # rotate around (0, 0) in pixel space, translate to (tl_x_px, tl_y_px)
    # 各 corner を pixel 位置に置いてから、各軸独立に %-value へ戻す
    verts: list[tuple[float, float]] = []
    for cx, cy in corners_local_px:
        rx = cos_t * cx - sin_t * cy
        ry = sin_t * cx + cos_t * cy
        world_x_px = tl_x_px + rx
        world_y_px = tl_y_px + ry
        verts.append(
            (world_x_px / img_w * 100.0, world_y_px / img_h * 100.0)
        )
    return verts


def yolo_obb_verts_to_ls_rect(
    verts_pct: list[tuple[float, float]], img_w: int, img_h: int
) -> LSValue:
    """Inverse of `ls_rect_to_yolo_obb_verts`。round-trip test 用。

    4 vertices (%-units) から LS の (x, y, w, h, rotation) を復元。
    pixel-space で辺長と angle を計算し、w/h は各軸のスケールで %-value に戻す。
    """
    if len(verts_pct) != 4:
        raise ValueError(f"expected 4 verts, got {len(verts_pct)}")
    v0, v1, _v2, v3 = verts_pct
    x_pct, y_pct = v0

    # pixel 座標に戻す
    v0_px = (v0[0] / 100.0 * img_w, v0[1] / 100.0 * img_h)
    v1_px = (v1[0] / 100.0 * img_w, v1[1] / 100.0 * img_h)
    v3_px = (v3[0] / 100.0 * img_w, v3[1] / 100.0 * img_h)

    # pixel-space の TL→TR ベクトルで w、TL→BL で h
    dx_w_px = v1_px[0] - v0_px[0]
    dy_w_px = v1_px[1] - v0_px[1]
    dx_h_px = v3_px[0] - v0_px[0]
    dy_h_px = v3_px[1] - v0_px[1]
    w_px = math.hypot(dx_w_px, dy_w_px)
    h_px = math.hypot(dx_h_px, dy_h_px)

    # %-value: 幅は img_w、高さは img_h でそれぞれ割る (LS 慣習)
    w_pct = w_px / img_w * 100.0
    h_pct = h_px / img_h * 100.0

    # rotation は pixel-space の TL→TR 方向 (LS/Konva と同じ)
    rotation = math.degrees(math.atan2(dy_w_px, dx_w_px))

    return {"x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct, "rotation": rotation}


# ------------------------------------------------------------
# LS export JSON パース
# ------------------------------------------------------------
def _image_rel_path_from_task(task: dict) -> str:
    """LS task の data.image URL から image の相対 path (拡張子除く) を組み立てる。

    LS Local storage の URL 例:
      `/data/local-files/?d=images/insert_table_leg/ep0062/frame_00060.png`
    → 出力: `insert_table_leg/ep0062/frame_00060`
      (`images/` の先頭 prefix は LS DOCUMENT_ROOT からの相対 path で意味を持たないので strip、
       YOLO OBB dataset の 画像↔label ペアは basename 一致で成立するため hierarchical に出力する)
    """
    image_ref = (task.get("data") or {}).get("image", "")
    m = re.search(r"[?&]d=([^&]+)", image_ref)
    path = m.group(1) if m else image_ref
    p = Path(path)
    parts = list(p.parent.parts)
    # LS DOCUMENT_ROOT からの相対 path 先頭が `images/` の場合 (bootstrap の DOCUMENT_ROOT
    # が `workspace/` で storage が `workspace/images/{skill}/ep{XX}/`) は strip する
    if parts and parts[0] == "images":
        parts = parts[1:]
    return "/".join([*parts, p.stem])


def parse_ls_export(
    export_path: Path,
    fallback_to_predictions: bool = False,
) -> dict[str, list[tuple[int, list[tuple[float, float]]]]]:
    """LS export JSON を「image rel_path → [(class_id, verts_pct), ...]」の dict に変換。

    Args:
        export_path: LS export の JSON path
        fallback_to_predictions: True にすると task の annotations が空の時、
            predictions フィールド (prelabel 結果) を fallback として使う。
            user 未 review frame も prelabel のまま採用したい時に使用。

    Returns:
        {rel_path: [(class_id, [(x1_pct, y1_pct), ...4 vertices]), ...]}
        rel_path は拡張子除いた相対 path (e.g. `insert_table_leg/ep0062/frame_00060`)。
        `write_yolo_obb_txts` が `out_dir/{rel_path}.txt` に書く。

    img dims は各 result item の `original_width` / `original_height` から取る。
    """
    with open(export_path) as f:
        data = json.load(f)
    result: dict[str, list[tuple[int, list[tuple[float, float]]]]] = {}
    for task in data:
        rel_path = _image_rel_path_from_task(task)
        result.setdefault(rel_path, [])
        anns = task.get("annotations", [])
        if not anns and fallback_to_predictions:
            # user 未 review frame は predictions を annotation として採用
            anns = task.get("predictions", [])
        for ann in anns:
            for r in ann.get("result", []):
                if r.get("type") != "rectanglelabels":
                    continue
                value = r["value"]
                labels = value.get("rectanglelabels", [])
                if not labels:
                    continue
                name = labels[0]
                if name not in CLASS_NAME_TO_ID:
                    print(f"[warn] unknown class '{name}' — skipping", file=sys.stderr)
                    continue
                class_id = CLASS_NAME_TO_ID[name]
                img_w = int(r.get("original_width", 0))
                img_h = int(r.get("original_height", 0))
                if img_w <= 0 or img_h <= 0:
                    print(
                        f"[warn] missing original_width/height for {rel_path} — skipping box",
                        file=sys.stderr,
                    )
                    continue
                verts = ls_rect_to_yolo_obb_verts(value, img_w, img_h)
                result[rel_path].append((class_id, verts))
    return result


# ------------------------------------------------------------
# YOLO OBB txt 出力
# ------------------------------------------------------------
def write_yolo_obb_txts(
    parsed: dict[str, list[tuple[int, list[tuple[float, float]]]]],
    out_dir: Path,
) -> int:
    """0-1 正規化した YOLO OBB txt を image ごとに書く。返り値は書いた txt 数。

    parsed の key は rel_path (拡張子除く、e.g. `insert_table_leg/ep0062/frame_00060`)、
    出力は `out_dir/{rel_path}.txt` で階層を維持する (image 側と basename ペアが取れる)。
    """
    n = 0
    for rel_path, boxes in parsed.items():
        lines = []
        for class_id, verts in boxes:
            # verts は %-units、YOLO は 0..1 正規化 (単に /100)
            flat = " ".join(f"{v / 100.0:.6f}" for pair in verts for v in pair)
            lines.append(f"{class_id} {flat}")
        out_path = out_dir / f"{rel_path}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    return n


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--export", type=Path, required=True, help="LS export JSON (task list)")
    p.add_argument("--out-dir", type=Path, required=True, help="YOLO OBB txt 出力 dir")
    args = p.parse_args()

    if not args.export.exists():
        raise SystemExit(f"export not found: {args.export}")

    parsed = parse_ls_export(args.export)
    n = write_yolo_obb_txts(parsed, args.out_dir)
    total_boxes = sum(len(v) for v in parsed.values())
    print(f"[done] {n} images, {total_boxes} boxes → {args.out_dir}")


if __name__ == "__main__":
    main()
