"""既存 frame cache (v0 jpg) から N variant の aug + OBB overlay baked jpg を生成 (Issue #129, case F)。

# 目的

Online data augmentation は train iter で data_s bottleneck を作る (実測: aug on 1.6s/step
vs aug off 0.9s/step、GR00T 100k step で ~19h 差)。加えて runtime overlay hook は cam name
mismatch (training view = head_left / OBB cache = cam_0) で silent skip されており事実上機能
していなかった。両方を offline precompute で bake する:

- v0: **無変更で保持** (raw source として rerun 時の re-generation base、training では使わない)
- v_i (i = 1..N-1): v0 → photometric aug → overlay draw → geometric aug → save

Runtime `_resolve_variant_jpg_path` は v_1..v_(N-1) から uniform random 選択 (v0 skip、実装
は `lerobot_frame_cache_patch.py` 側で対応)。この設計により v0 を上書きせず、rerun 時の
overlay 二重描画リスクを排除。`OBB_OVERLAY_ENABLE=false` に切替えて runtime hook 廃止。

# Aug pipeline order (photometric → overlay → geometric)

- photometric (ColorJitter × 4 + Sharpness) を **overlay 前** に適用 = overlay canonical color
  (red=hole, green=leg 等) が jitter で崩れない
- geometric (RandomAffine) を **overlay 後** に適用 = overlay 枠と object が同じ affine で
  warp、幾何一致 (元 OBB 座標のまま geometric aug をかけると overlay が最大 ±5° / ±5% translate
  ずれる)

# Overlay 対応 cam

- head_left (source cam_0): OBB cache から verts 引いて cv2.polylines
- wrist (source cam_2 / cam_3): OBB cache 未収録 → 描画 skip (aug のみ apply)
- head_right: 学習 view で drop 済 (materialize_lerobot_training_view.py:699)

# Cam name mapping

Training view frame_cache の cam 名 (`observation.images.head_left` 等) は source dataset で
rename されている (materialize view 時)。OBB precompute cache は source cam 名
(`observation.images.cam_0` 等) を使う。
- 第一候補: training view meta の `team_ramen_training_view.json:camera_map` から読取
- Fallback: subtask_training.json の hardcoded map (head_left→cam_0 等)

# Cache layout (backward compat)

    <cache_root>/{cam}/{chunk}/{file}/
        _frame_count.txt
        frame_000000.jpg           # v0 (raw、無変更、rerun 用 source)
        frame_000000_v1.jpg        # aug variant 1 (photometric + overlay + geometric)
        ...
        frame_000000_v{N-1}.jpg

Runtime `_resolve_variant_jpg_path` を修正して v_1..v_(N-1) から uniform random 選択
(v0 は skip、training には使わない)。`OBB_OVERLAY_ENABLE=false` に切替えて hook 廃止。

# Usage

    python precompute_augmented_frame_cache.py \\
        --cache-root <path>/frame_cache \\
        --obb-root <path>/obb_yolo \\
        --num-variants 10 \\
        --jobs 24 \\
        [--obb-hash <hash>]                # 未指定なら root 直下唯一の hash dir auto-pick
        [--overlay-conf 0.30]
        [--overlay-thickness 2]
        [--overlay-class-filter '[3,4]']   # JSON list of int、未指定 = 全 class
        [--force]                          # 既存 v_i を無視して再生成
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torchvision.transforms import v2


# Repo root を sys.path に注入 (mp.Pool fork でも継承される)。
# `<repo>/data/bitrobot_lerobot_subtask_datasets/scripts/<this>` から辿って repo root を導出。
# worker init が `from model.ramen_ori.obb_cache import ObbPrecomputedCache` するため必要。
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_JPG_QUALITY = 90
DEFAULT_MAX_NUM_PHOTOMETRIC = 3
DEFAULT_MAX_NUM_GEOMETRIC = 1
DEFAULT_OVERLAY_CONF = 0.30
DEFAULT_OVERLAY_THICKNESS = 2

MARKER_REL_PATH = Path("meta") / "team_ramen_training_view.json"

# subtask_training.json:source_dataset.camera_map の hardcoded 版 (marker 不在時 fallback)
# training view で使う cam 名 → source dataset cam 名 (OBB cache の key)
FALLBACK_CAMERA_MAP: dict[str, str] = {
    "observation.images.head_left": "observation.images.cam_0",
    "observation.images.head_right": "observation.images.cam_1",
    "observation.images.left_wrist": "observation.images.cam_2",
    "observation.images.right_wrist": "observation.images.cam_3",
}


def _build_aug_pools() -> tuple[
    list[tuple[str, v2.Transform]], list[tuple[str, v2.Transform]]
]:
    """Photometric / geometric に分けた aug transform pool。

    Photometric (色/シャープ、幾何変換なし): overlay の色を崩さないため overlay 前に適用。
    Geometric (幾何変換): overlay と object を一緒に warp するため overlay 後に適用。

    Weights は per-transform 1.0 均等 (LeRobot config と一致)。
    """
    photometric: list[tuple[str, v2.Transform]] = [
        ("brightness", v2.ColorJitter(brightness=(0.8, 1.2))),
        ("contrast", v2.ColorJitter(contrast=(0.8, 1.2))),
        ("hue", v2.ColorJitter(hue=(-0.05, 0.05))),
        ("saturation", v2.ColorJitter(saturation=(0.5, 1.5))),
        ("sharpness", v2.RandomAdjustSharpness(sharpness_factor=1.5, p=1.0)),
    ]
    geometric: list[tuple[str, v2.Transform]] = [
        ("affine", v2.RandomAffine(degrees=(-5.0, 5.0), translate=(0.05, 0.05))),
    ]
    return photometric, geometric


def _sample_augs(
    pool: list[tuple[str, v2.Transform]], max_num: int, rng: random.Random
) -> list[v2.Transform]:
    """Pool から 0..max_num 個を random sample。0 個も可 (=aug skip)。"""
    n = rng.randint(0, min(len(pool), max_num))
    if n == 0:
        return []
    chosen = rng.sample(pool, n)
    return [t for _, t in chosen]


def _apply_augs(img: Image.Image, transforms: list[v2.Transform]) -> Image.Image:
    """Transform list を順に apply (empty list は no-op)。"""
    for t in transforms:
        img = t(img)
    return img


def _derive_frame_info(v0_path: Path) -> tuple[str, str, int]:
    """`<cache_root>/{cam}/{chunk-XXX}/{file-YYY}/frame_XXXXXX.jpg` から
    (training_cam_key, mp4_relative, frame_idx_in_mp4) を抽出。

    mp4_relative は "chunk-XXX/file-YYY" (OBB cache lookup 用)。
    """
    parts = v0_path.parts
    if len(parts) < 4:
        raise ValueError(f"v0 path too short: {v0_path}")
    training_cam = parts[-4]  # observation.images.head_left
    chunk = parts[-3]         # chunk-000
    file_stem = parts[-2]     # file-000
    stem = v0_path.stem       # frame_000042
    if not stem.startswith("frame_"):
        raise ValueError(f"unexpected jpg name: {v0_path.name}")
    frame_idx = int(stem[len("frame_") :])
    return training_cam, f"{chunk}/{file_stem}", frame_idx


def _load_camera_map(cache_root: Path) -> dict[str, str]:
    """Training view marker から camera_map を読取、無ければ hardcoded fallback。

    cache_root = <view>/frame_cache 前提、marker = <view>/meta/team_ramen_training_view.json。
    marker が存在すれば `camera_map` field (training_cam → source_cam の dict) を返す。
    無ければ FALLBACK_CAMERA_MAP をそのまま返す。
    """
    marker_path = cache_root.parent / MARKER_REL_PATH
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        camera_map = marker.get("camera_map")
        if isinstance(camera_map, dict) and camera_map:
            return dict(camera_map)
    return dict(FALLBACK_CAMERA_MAP)


# ---------------------------------------------------------------------------
# OBB overlay drawing (overlay.py の kernel を duplicate、YAGNI: 追加 refactor 避)
# ---------------------------------------------------------------------------

# BGR (cv2 convention)、model/ramen_ori/overlay.py の CLASS_COLORS_BGR と一致
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (128, 128, 128),  # workspace  = gray
    1: (0, 255, 0),      # leg        = green
    2: (0, 255, 255),    # leg_tip    = yellow
    3: (0, 0, 255),      # hole       = red
    4: (255, 0, 0),      # table_top  = blue
    5: (0, 128, 255),    # hand_right = orange
    6: (255, 0, 255),    # hand_left  = magenta
}


def _draw_overlay(
    img_bgr: np.ndarray,
    per_frame: dict[str, np.ndarray],
    conf_threshold: float,
    line_thickness: int,
    class_filter: set[int] | None,
) -> np.ndarray:
    """OBB cache の per-frame lookup 結果を img_bgr (uint8 HWC BGR) に描画。in-place。

    verts は normalized [0, 1] xyxyxyxy (4 vert × 2)、valid=False や conf<threshold は skip。
    """
    import cv2  # noqa: PLC0415

    H, W = img_bgr.shape[:2]
    verts = per_frame["verts"]      # (top_K, 8)
    conf = per_frame["conf"]         # (top_K,)
    class_id = per_frame["class_id"] # (top_K,) int8
    valid = per_frame["valid"]       # (top_K,) bool

    for k in range(len(valid)):
        if not bool(valid[k]):
            continue
        if float(conf[k]) < conf_threshold:
            continue
        cid = int(class_id[k])
        if class_filter is not None and cid not in class_filter:
            continue
        color = CLASS_COLORS_BGR.get(cid, (255, 255, 255))
        v = verts[k].reshape(4, 2) * np.array([W, H], dtype=np.float32)
        pts = v.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=line_thickness)
    return img_bgr


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# Worker-global (mp.Pool initializer で set、per-worker 一度だけ生成)
_worker_cache: Any = None
_worker_camera_map: dict[str, str] = {}
_worker_photometric_pool: list[tuple[str, v2.Transform]] = []
_worker_geometric_pool: list[tuple[str, v2.Transform]] = []
_worker_config: dict[str, Any] = {}


def _worker_init(
    obb_root: str,
    obb_hash: str | None,
    obb_camera_keys: list[str],
    obb_top_k: int,
    obb_num_classes: int,
    obb_fps: int,
    camera_map: dict[str, str],
    overlay_conf: float,
    overlay_thickness: int,
    overlay_class_filter_list: list[int] | None,
    jpg_quality: int,
    max_num_photometric: int,
    max_num_geometric: int,
) -> None:
    """Pool worker 起動時に 1 回だけ呼ばれる。OBB cache instance + aug pool を作って
    module global に格納 (multiprocess fork なので per-process で独立)。"""
    global _worker_cache, _worker_camera_map
    global _worker_photometric_pool, _worker_geometric_pool, _worker_config

    # lazy import: ObbPrecomputedCache は numpy/pyarrow 依存、main import 時に他 worker
    # 側の env と conflict しないよう worker 側でだけ import する。
    from model.ramen_ori.obb_cache import ObbPrecomputedCache  # noqa: PLC0415

    _worker_cache = ObbPrecomputedCache(
        root=Path(obb_root),
        ckpt_hash=obb_hash,
        camera_keys=obb_camera_keys,
        top_K=obb_top_k,
        num_classes=obb_num_classes,
        fps=obb_fps,
    )
    _worker_camera_map = dict(camera_map)
    _worker_photometric_pool, _worker_geometric_pool = _build_aug_pools()
    _worker_config = {
        "overlay_conf": float(overlay_conf),
        "overlay_thickness": int(overlay_thickness),
        "overlay_class_filter": (
            set(overlay_class_filter_list) if overlay_class_filter_list is not None else None
        ),
        "jpg_quality": int(jpg_quality),
        "max_num_photometric": int(max_num_photometric),
        "max_num_geometric": int(max_num_geometric),
    }


def _lookup_overlay_for_frame(
    training_cam: str, mp4_relative: str, frame_idx: int
) -> dict[str, np.ndarray] | None:
    """training cam → source cam に翻訳して OBB cache 引く。cam or frame 対象外なら None。"""
    source_cam = _worker_camera_map.get(training_cam)
    if source_cam is None:
        return None
    # ObbPrecomputedCache.lookup_mp4_frame は covered_cameras に無い cam なら None を返す
    return _worker_cache.lookup_mp4_frame(source_cam, mp4_relative, frame_idx)


def _process_one_frame(
    args_tuple: tuple[Path, int, bool],
) -> tuple[Path, int, int]:
    """1 frame について v1..v(N-1) の augmented+overlay jpg を生成。v0 は変更しない。

    Args:
        args_tuple: (v0_jpg_path, num_variants, force)

    Returns:
        (v0_jpg_path, saved, skipped) — saved は新規 write 数、skipped は既存 skip 数。
    """
    v0_path, num_variants, force = args_tuple

    if not v0_path.is_file():
        return v0_path, 0, 0

    training_cam, mp4_relative, frame_idx = _derive_frame_info(v0_path)
    per_frame = _lookup_overlay_for_frame(training_cam, mp4_relative, frame_idx)
    conf_threshold = _worker_config["overlay_conf"]
    line_thickness = _worker_config["overlay_thickness"]
    class_filter = _worker_config["overlay_class_filter"]
    jpg_quality = _worker_config["jpg_quality"]
    max_num_photo = _worker_config["max_num_photometric"]
    max_num_geo = _worker_config["max_num_geometric"]

    # v0 は無変更で保持 (rerun 時の raw source として使う、二重 overlay bake 回避)
    base_img_pil = Image.open(v0_path).convert("RGB")

    # Deterministic seed (path + variant idx で hash) = 再現性 + variant 独立
    frame_seed_base = hash(str(v0_path)) & 0xFFFFFFFF

    saved = 0
    skipped = 0

    for v_idx in range(1, num_variants):
        out_path = v0_path.parent / f"{v0_path.stem}_v{v_idx}.jpg"
        if out_path.exists() and not force:
            skipped += 1
            continue

        rng = random.Random(frame_seed_base ^ v_idx)
        photo_augs = _sample_augs(_worker_photometric_pool, max_num_photo, rng)
        geo_augs = _sample_augs(_worker_geometric_pool, max_num_geo, rng)

        # Step 1: photometric aug (色/シャープ) — overlay 前 = overlay の canonical color を守る
        img = base_img_pil.copy()
        img = _apply_augs(img, photo_augs)

        # Step 2: overlay draw (per_frame が None = wrist なら skip)
        if per_frame is not None:
            img_np = np.asarray(img)  # RGB HWC uint8
            img_bgr = img_np[:, :, ::-1].copy()  # PIL RGB → cv2 BGR
            img_bgr = _draw_overlay(img_bgr, per_frame, conf_threshold, line_thickness, class_filter)
            img = Image.fromarray(img_bgr[:, :, ::-1])  # BGR → RGB → PIL

        # Step 3: geometric aug (affine) — overlay 後 = overlay と object 一緒に warp
        img = _apply_augs(img, geo_augs)

        img.save(out_path, format="JPEG", quality=jpg_quality)
        saved += 1

    base_img_pil.close()
    return v0_path, saved, skipped


def _collect_all_v0_frames(cache_root: Path) -> list[Path]:
    """`<cache_root>/{cam}/{chunk}/{file}/frame_*.jpg` (v0 のみ、_v サフィックス除く) を列挙。"""
    frames: list[Path] = []
    for jpg in cache_root.rglob("frame_*.jpg"):
        if "_v" in jpg.stem.split("frame_")[-1]:
            continue
        frames.append(jpg)
    return frames


def _parse_class_filter(spec: str | None) -> list[int] | None:
    """`--overlay-class-filter '[3,4]'` の JSON list-of-int を parse。無指定なら None。"""
    if spec is None or spec.strip() == "":
        return None
    parsed = json.loads(spec)
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise argparse.ArgumentTypeError(
            f"--overlay-class-filter must be JSON list of int, got {spec!r}"
        )
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="既存 frame cache root (例: outputs/training_views/groot_rotate_table_base/frame_cache)",
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=10,
        help="Total variants per frame (v0 + v1..v_(N-1))。Default 10",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=DEFAULT_JPG_QUALITY,
        help="JPG encoding quality (0-100)。Default 90 (元 cache と一致)",
    )
    parser.add_argument(
        "--max-num-photometric",
        type=int,
        default=DEFAULT_MAX_NUM_PHOTOMETRIC,
        help="Per-variant で apply する photometric aug 上限 (0..この値)。Default 3",
    )
    parser.add_argument(
        "--max-num-geometric",
        type=int,
        default=DEFAULT_MAX_NUM_GEOMETRIC,
        help="Per-variant で apply する geometric aug 上限 (0..この値、現在 pool 中 1 種)。Default 1",
    )
    parser.add_argument("--jobs", type=int, default=24, help="Multiprocessing worker 数")
    parser.add_argument(
        "--force", action="store_true", help="既存 v_i / v0 baked を無視して再生成"
    )
    # === OBB overlay params ===
    parser.add_argument(
        "--obb-root",
        type=Path,
        required=True,
        help="OBB precompute cache root (例: /nvme/rotate_table_base/obb_yolo)",
    )
    parser.add_argument(
        "--obb-hash",
        type=str,
        default=None,
        help="OBB cache hash dir 名。未指定なら root 直下唯一の hash dir auto-pick",
    )
    parser.add_argument(
        "--obb-cameras",
        type=str,
        default="observation.images.cam_0,observation.images.cam_1",
        help="OBB cache の cam key list (comma-sep)、manifest.cameras の superset である必要あり",
    )
    parser.add_argument("--obb-top-k", type=int, default=4)
    parser.add_argument("--obb-num-classes", type=int, default=7)
    parser.add_argument("--obb-fps", type=int, default=30)
    parser.add_argument(
        "--overlay-conf", type=float, default=DEFAULT_OVERLAY_CONF, help="Overlay conf threshold"
    )
    parser.add_argument(
        "--overlay-thickness",
        type=int,
        default=DEFAULT_OVERLAY_THICKNESS,
        help="cv2.polylines thickness",
    )
    parser.add_argument(
        "--overlay-class-filter",
        type=str,
        default=None,
        help="JSON list of int (例 '[3,4]' = hole+table_top のみ描画)、未指定 = 全 class",
    )
    args = parser.parse_args()

    if not args.cache_root.is_dir():
        print(f"ERROR: cache_root not found: {args.cache_root}", file=sys.stderr)
        return 2
    if args.num_variants < 2:
        print(
            f"ERROR: num-variants must be >= 2 (got {args.num_variants}); "
            f"1 = v0 のみ、この script 不要",
            file=sys.stderr,
        )
        return 2
    if not args.obb_root.is_dir():
        print(f"ERROR: obb-root not found: {args.obb_root}", file=sys.stderr)
        return 2

    camera_map = _load_camera_map(args.cache_root)
    print(f"[precompute_aug] camera_map: {camera_map}", flush=True)

    class_filter_list = _parse_class_filter(args.overlay_class_filter)
    obb_cameras = [c.strip() for c in args.obb_cameras.split(",") if c.strip()]

    print(f"[precompute_aug] scanning {args.cache_root} for v0 frames...", flush=True)
    v0_frames = _collect_all_v0_frames(args.cache_root)
    n_frames = len(v0_frames)
    if n_frames == 0:
        print(f"ERROR: no v0 frames found under {args.cache_root}", file=sys.stderr)
        return 2

    n_variants_to_gen = args.num_variants - 1  # v0 は無変更、v1..v_(N-1) を新規生成
    total_jpg = n_frames * n_variants_to_gen
    print(
        f"[precompute_aug] {n_frames:,} v0 frames × {n_variants_to_gen} augmented variants "
        f"= {total_jpg:,} jpg to generate "
        f"(jobs={args.jobs}, max_num_photometric={args.max_num_photometric}, "
        f"max_num_geometric={args.max_num_geometric}, "
        f"overlay conf={args.overlay_conf}, thickness={args.overlay_thickness}, "
        f"class_filter={class_filter_list})",
        flush=True,
    )

    tasks = [(v0, args.num_variants, args.force) for v0 in v0_frames]

    initargs = (
        str(args.obb_root),
        args.obb_hash,
        obb_cameras,
        args.obb_top_k,
        args.obb_num_classes,
        args.obb_fps,
        camera_map,
        args.overlay_conf,
        args.overlay_thickness,
        class_filter_list,
        args.jpg_quality,
        args.max_num_photometric,
        args.max_num_geometric,
    )

    t0 = time.monotonic()
    total_saved = 0
    total_skipped = 0
    with mp.Pool(
        processes=args.jobs, initializer=_worker_init, initargs=initargs
    ) as pool:
        for i, (_, saved, skipped) in enumerate(
            pool.imap_unordered(_process_one_frame, tasks), 1
        ):
            total_saved += saved
            total_skipped += skipped
            if i % 500 == 0 or i == n_frames:
                elapsed = time.monotonic() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (n_frames - i) / rate if rate > 0 else float("inf")
                print(
                    f"[precompute_aug] {i:,}/{n_frames:,} frames processed, "
                    f"saved={total_saved:,} skipped={total_skipped:,}, "
                    f"rate={rate:.1f} frame/s, eta={eta/60:.1f} min",
                    flush=True,
                )

    elapsed = time.monotonic() - t0
    print(
        f"[precompute_aug] DONE in {elapsed/60:.1f} min. "
        f"total saved: {total_saved:,}, skipped (already existed): {total_skipped:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
