"""LeRobot v3 dataset の mp4 を per-frame JPG / lossless PNG に事前展開。

# 目的
H.264 の随時 decode は 1 frame 30-100ms の seek+decode cost が train iter で
律速になる (data_s 78%)。事前に全 frame を JPG (self-contained、seek 不要) に
展開しておくことで per-frame decode を 10-30x 高速化する。

# Cache 構造
    <cache_root>/
        _cache_meta.json                              # settings hash + fps / video_keys
        <cam_key>/
            <chunk-000>/
                <file-000>/                           # mp4 basename
                    _frame_count.txt                  # total decoded frames
                    frame_000000.jpg
                    frame_000001.jpg
                    ...

# 挙動 (Issue #122 revised)
1. Settings hash 検証: `_cache_meta.json` に (fps, jpg_quality, video_keys, ...)
   の deterministic hash を保存。次回起動時に expected と比較、mismatch なら
   cache を wipe + 再生成 (auto)。
2. Per-mp4 verify: 各 mp4 について cache dir 内 JPG 数 == mp4 frame 数 を確認。
3. Auto mp4 drop: verify 全 pass 後、source mp4 (resolved real files) を削除。
   Cache が唯一の source of truth になり disk が節約される。

# Frame index の対応
LeRobot dataset は timestamp で query してくる ({from_timestamp + local_ts, ...})。
patch 側 (lerobot_frame_cache_patch.py) が受け取った timestamp を fps で正規化:
    frame_idx = round(timestamp * fps)
となり、cache 上の frame_{frame_idx:06d}.jpg を読む。

# Usage
    python precompute_frame_cache.py \\
        --lerobot-root <merged_source>          # 例: outputs/merged_sources/combined_task5_7
        [--cache-root <path>]                    # default: <lerobot_root>/frame_cache
        [--jobs 24] [--quality 90]
        [--image-format {jpg,png}] [--png-compression 1]
        [--force]                                # 既 cache を無視して強制再生成
        [--video-keys observation.images.cam_0,observation.images.cam_2]  # 展開する cam を限定
        [--keep-source-mp4]                      # 3. の auto mp4 drop をしない

PNG uses PyAV RGB decoding (the training decoder), validates zero-start CFR
timestamps, defaults to frame_cache_png, and ALWAYS retains source MP4s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

CACHE_META_NAME = "_cache_meta.json"
FRAME_COUNT_NAME = "_frame_count.txt"
CACHE_SCHEMA_VERSION = "team_ramen_frame_cache_v1"
PNG_CACHE_SCHEMA_VERSION = "team_ramen_frame_cache_v2"
DECODE_COMPLETE_NAME = "_decode_complete.json"
PNG_TIMESTAMP_TOLERANCE_S = 0.001


def _validate_format(image_format: str, png_compression: int) -> None:
    if image_format not in {"jpg", "png"}:
        raise ValueError(f"unsupported image format: {image_format!r}")
    if not isinstance(png_compression, int) or not 0 <= png_compression <= 9:
        raise ValueError("png_compression must be an integer in 0..9")


def _validate_fps(fps: float) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _list_video_keys(info: dict[str, Any]) -> list[str]:
    """meta/info.json の features から video feature key を昇順 list で返す。"""
    features = info.get("features", {})
    return sorted(
        k
        for k, v in features.items()
        if isinstance(v, dict) and v.get("dtype") == "video"
    )


def _iter_mp4_files(
    lerobot_root: Path, video_keys: list[str]
) -> list[tuple[str, Path]]:
    """各 video_key の videos/ 配下の mp4 files を (video_key, path) の list で返す。"""
    result: list[tuple[str, Path]] = []
    for vk in video_keys:
        vdir = lerobot_root / "videos" / vk
        if not vdir.exists():
            continue
        for mp4 in sorted(vdir.rglob("*.mp4")):
            result.append((vk, mp4))
    return result


def cache_dir_for_mp4(cache_root: Path, video_key: str, mp4_relative: Path) -> Path:
    """mp4 の relative path (chunk-XXX/file-YYY.mp4) → cache subdir (chunk-XXX/file-YYY/)。

    Patch 側と共有される layout。変更するときは lerobot_frame_cache_patch.py の
    ``_derive_cache_dir_from_path`` も同時に直すこと。
    """
    return cache_root / video_key / mp4_relative.parent / mp4_relative.stem


def compute_settings_hash(
    *,
    fps: float,
    video_keys: list[str],
    jpg_quality: int = 90,
    schema_version: str | None = None,
    image_format: str = "jpg",
    png_compression: int = 1,
    decoder: str = "pyav",
) -> str:
    """Cache 有効性を決める settings の deterministic hash。

    JPEG v1 includes fps / video_keys (ordered) / jpg_quality / schema_version.
    PNG v2 replaces jpg_quality with image_format / png_compression / decoder.
    含めない: source_root (path 移動しても cache 有効なはず)、jobs (並列度は結果に無影響)。
    PNG settings changes require explicit --force; legacy JPEG keeps auto-rebuild.
    """
    _validate_format(image_format, png_compression)
    payload = {
        "schema_version": schema_version or (
            PNG_CACHE_SCHEMA_VERSION if image_format == "png" else CACHE_SCHEMA_VERSION
        ),
        "fps": float(fps),
        "video_keys": list(video_keys),
    }
    if image_format == "png":
        _validate_fps(fps)
        payload.update(image_format="png", png_compression=png_compression, decoder=decoder)
    else:
        # Preserve the exact v1 payload (including omission of image_format).
        payload["jpg_quality"] = int(jpg_quality)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()


def _read_cache_meta(cache_root: Path) -> dict[str, Any]:
    meta_path = cache_root / CACHE_META_NAME
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def read_existing_cache_hash(cache_root: Path) -> str | None:
    """Read a published settings hash; an incomplete run is never a cache hit."""
    meta = _read_cache_meta(cache_root)
    if meta.get("image_format") == "png" and meta.get("complete") is not True:
        return None
    if meta.get("complete") is False:
        return None
    value = meta.get("settings_hash")
    return value if isinstance(value, str) else None


def wipe_cache(cache_root: Path) -> None:
    """Cache dir 配下 (frame JPG + marker) を全消し。cache_root 自体は残す。"""
    if not cache_root.is_dir():
        return
    for child in cache_root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _decode_one_mp4(
    args_tuple: tuple,
) -> tuple[Path, int, int, int]:
    """Sequentially decode one MP4 into independent JPG or PNG frames.

    Legacy: (mp4_path, cache_dir, jpg_quality, force[, max_frames]).
    PNG appends image_format, png_compression, fps after max_frames.
    A smoke prefix never publishes PNG completion markers.

    Returns:
        (cache_dir, saved, skipped, total)
    """
    mp4_path, cache_dir, jpg_quality, force = args_tuple[:4]
    max_frames = args_tuple[4] if len(args_tuple) > 4 else None
    image_format = args_tuple[5] if len(args_tuple) > 5 else "jpg"
    png_compression = args_tuple[6] if len(args_tuple) > 6 else 1
    _validate_format(image_format, png_compression)
    if image_format == "png":
        fps = args_tuple[7] if len(args_tuple) > 7 else None
        if fps is None:
            raise ValueError("PNG decoding requires the dataset metadata fps")
        return _decode_one_png_mp4(
            mp4_path, cache_dir, force=force, max_frames=max_frames,
            png_compression=png_compression, fps=fps,
        )
    # cv2 の import は worker process で lazy (main 側の fork 前に import すると
    # 一部 platform で libgl race するため)
    import cv2  # noqa: PLC0415

    cv2.setNumThreads(1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {mp4_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, int(max_frames))
    saved = 0
    skipped = 0
    try:
        for frame_idx in range(total):
            out_path = cache_dir / f"frame_{frame_idx:06d}.jpg"
            if out_path.exists() and not force:
                # sequential decode 維持のため advance だけ
                cap.grab()
                skipped += 1
                continue
            ok, img = cap.read()
            if not ok:
                break
            # tmp に書いてから rename: 途中で止まっても半端な jpg を残さない
            # (resume 時は既存 file を skip するので、半端な file がそのまま残ってしまう)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            if not ok:
                raise RuntimeError(f"jpeg encode failed: {out_path}")
            tmp_path = out_path.with_name(out_path.name + ".tmp")
            tmp_path.write_bytes(buf.tobytes())
            os.replace(tmp_path, out_path)
            saved += 1
    finally:
        cap.release()
    # frame count marker
    (cache_dir / FRAME_COUNT_NAME).write_text(str(total), encoding="utf-8")
    return cache_dir, saved, skipped, total


def _source_signature(mp4_path: Path) -> dict[str, int]:
    stat = mp4_path.stat()
    return {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}


def _decode_one_png_mp4(
    mp4_path: Path, cache_dir: Path, *, force: bool,
    max_frames: int | None, png_compression: int, fps: float,
) -> tuple[Path, int, int, int]:
    """Losslessly store exactly the RGB24 pixels produced by training's PyAV path.

    Index-based lookup is only safe for zero-start CFR streams. Reject missing,
    duplicate, shifted or variable timestamps instead of silently renumbering them.
    Completion markers are published only after a validated, full decode.
    """
    import av  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    cv2.setNumThreads(1)
    _validate_fps(fps)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in (FRAME_COUNT_NAME, DECODE_COMPLETE_NAME):
        (cache_dir / name).unlink(missing_ok=True)
    source_signature = _source_signature(mp4_path)
    saved = skipped = total = 0
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        # --jobs is the CPU budget; do not multiply it by libav auto threads.
        stream.thread_count = 1
        declared_count = int(stream.frames)
        if declared_count <= 0:
            raise RuntimeError(f"invalid/unknown MP4 frame count: {mp4_path}")
        if stream.average_rate is None or not math.isclose(
            float(stream.average_rate), fps, rel_tol=0, abs_tol=0.001
        ):
            raise RuntimeError(f"MP4 fps {stream.average_rate} != metadata fps {fps}: {mp4_path}")
        for frame_idx, frame in enumerate(container.decode(stream)):
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame.pts is None or frame.time_base is None:
                raise RuntimeError(f"missing frame timestamp at index {frame_idx}: {mp4_path}")
            timestamp = float(frame.pts * frame.time_base)
            expected_timestamp = frame_idx / fps
            if not math.isfinite(timestamp) or abs(timestamp - expected_timestamp) > PNG_TIMESTAMP_TOLERANCE_S:
                raise RuntimeError(
                    f"non-zero-start/non-CFR timestamp at frame {frame_idx}: "
                    f"{timestamp} != {expected_timestamp} ({mp4_path})"
                )
            rgb = frame.to_ndarray(format="rgb24")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out_path = cache_dir / f"frame_{frame_idx:06d}.png"
            # Validate resumed pixels too: interruption never blesses corrupt PNGs.
            existing = cv2.imread(str(out_path), cv2.IMREAD_COLOR) if out_path.exists() and not force else None
            if existing is not None and existing.shape == bgr.shape and (existing == bgr).all():
                skipped += 1
            else:
                ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
                if not ok:
                    raise RuntimeError(f"png encode failed: {out_path}")
                tmp_path = out_path.with_name(out_path.name + ".tmp")
                tmp_path.write_bytes(buf.tobytes())
                os.replace(tmp_path, out_path)
                saved += 1
            total += 1
    if max_frames is not None:
        return cache_dir, saved, skipped, total
    if total != declared_count:
        raise RuntimeError(f"decoded frame count {total} != MP4 frame count {declared_count}: {mp4_path}")
    if _source_signature(mp4_path) != source_signature:
        raise RuntimeError(f"source MP4 changed while decoding: {mp4_path}")
    (cache_dir / FRAME_COUNT_NAME).write_text(str(total), encoding="utf-8")
    _write_json_atomic(cache_dir / DECODE_COMPLETE_NAME, {
        "image_format": "png", "decoder": "pyav", "fps": fps,
        "png_compression": png_compression, "frame_count": total,
        **source_signature,
    })
    return cache_dir, saved, skipped, total


def verify_mp4_cache_complete(
    mp4_path: Path, cache_dir: Path, *, image_format: str = "jpg",
    fps: float | None = None, png_compression: int = 1,
) -> tuple[bool, str]:
    """Cache dir が mp4 に対し完全か verify (drop 前必須)。

    Checks:
      1. `_frame_count.txt` marker 存在 + 値が mp4 frame 数と一致
      2. JPG file 数 == expected frame 数
      3. 各 frame index に対応 JPG 存在 (連番 hole なし)

    Returns:
        (ok, reason): ok=True なら pass、False なら reason に失敗理由。
    """
    _validate_format(image_format, png_compression)
    marker = cache_dir / FRAME_COUNT_NAME
    if not marker.is_file():
        return False, f"{FRAME_COUNT_NAME} 欠"
    try:
        expected = int(marker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as e:
        return False, f"{FRAME_COUNT_NAME} 読み取り失敗: {e}"

    if image_format == "png":
        try:
            completion = json.loads((cache_dir / DECODE_COMPLETE_NAME).read_text(encoding="utf-8"))
            signature = _source_signature(mp4_path)
        except (OSError, ValueError) as e:
            return False, f"PNG full-decode marker/source missing or invalid: {e}"
        if not isinstance(completion, dict) or any(
            completion.get(key) != value for key, value in {
                "image_format": "png", "decoder": "pyav", "fps": fps,
                "png_compression": png_compression, "frame_count": expected,
                **signature,
            }.items()
        ):
            return False, "PNG full-decode marker does not match source/settings"
        if expected <= 0:
            return False, "PNG frame count must be positive"
        return _verify_frame_files(cache_dir, expected, image_format)

    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return False, f"mp4 open 失敗: {mp4_path}"
    mp4_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if expected != mp4_frames:
        return False, (
            f"{FRAME_COUNT_NAME} value {expected} != mp4 frame count {mp4_frames}"
        )

    return _verify_frame_files(cache_dir, expected, image_format)


def _verify_frame_files(cache_dir: Path, expected: int, image_format: str) -> tuple[bool, str]:
    opposite = "jpg" if image_format == "png" else "png"
    if next(cache_dir.glob(f"frame_*.{opposite}"), None) is not None:
        return False, f"unexpected {opposite} frames in {image_format} cache"
    files = sorted(cache_dir.glob(f"frame_*.{image_format}"))
    if image_format == "jpg":
        # Augmentation bakers keep raw frames and _vN variants in the same dir.
        # Only known variant names are ignored; malformed raw names still fail.
        files = [frame for frame in files if not re.fullmatch(r"frame_\d{6,}_v\d+\.jpg", frame.name)]
    if len(files) != expected:
        return False, f"{image_format.upper()} count {len(files)} != expected {expected}"

    # 連番 check (hole 検知)
    for i, frame in enumerate(files):
        if frame.name != f"frame_{i:06d}.{image_format}":
            return False, f"hole at frame_{i:06d}.{image_format} (got {frame.name})"
        if image_format == "png":
            try:
                with frame.open("rb") as stream:
                    if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                        return False, f"not a PNG: {frame}"
            except OSError as e:
                return False, f"cannot read PNG: {frame}: {e}"
            import cv2  # noqa: PLC0415

            # A signature alone accepts truncated/corrupt IDAT payloads. Full
            # PNG decoding is deliberate; main parallelizes this per video.
            decoded = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3 or decoded.dtype.name != "uint8":
                return False, f"PNG decode failed or not RGB8: {frame}"

    return True, "ok"


def _verify_one_task(task: tuple) -> tuple[Path, bool, str]:
    mp4_path, cache_dir, image_format, fps, png_compression = task
    if image_format == "png":
        import cv2  # noqa: PLC0415

        cv2.setNumThreads(1)
    ok, reason = verify_mp4_cache_complete(
        mp4_path, cache_dir, image_format=image_format,
        fps=fps, png_compression=png_compression,
    )
    return mp4_path, ok, reason


def _verify_tasks(
    tasks: list[tuple], *, image_format: str, fps: float,
    png_compression: int, jobs: int,
) -> list[tuple[Path, str]]:
    """Verify all selected videos; PNG payload checks share the CLI worker bound."""
    verify_tasks = [(task[0], task[1], image_format, fps, png_compression) for task in tasks]
    if image_format == "png" and jobs > 1 and len(verify_tasks) > 1:
        with mp.Pool(min(jobs, len(verify_tasks))) as pool:
            return [(path, reason) for path, ok, reason in pool.imap_unordered(_verify_one_task, verify_tasks) if not ok]
    return [(path, reason) for path, ok, reason in map(_verify_one_task, verify_tasks) if not ok]


def drop_source_mp4s(
    tasks: list[tuple[Path, Path, int, bool]],
) -> tuple[int, int]:
    """Verify pass した mp4 の resolved real path を削除 (auto mp4 drop、Issue #122)。

    Symlink の場合は resolve() で real file を探し、それを削除。原 symlink は残す
    (patch 側は cache_dir 発見時に mp4 file 触らないので dangling symlink 問題なし)。
    Real file が既に無ければ skip。

    Returns:
        (dropped, bytes_freed): 削除した file 数と解放 byte 数。
    """
    dropped = 0
    bytes_freed = 0
    seen: set[Path] = set()
    for mp4_path, _, _, _ in tasks:
        real = mp4_path.resolve(strict=False)
        if real in seen:
            continue
        seen.add(real)
        if not real.exists():
            continue
        try:
            size = real.stat().st_size
        except OSError:
            size = 0
        try:
            real.unlink()
            dropped += 1
            bytes_freed += size
        except OSError as e:
            print(f"[drop] warn: {real} unlink failed: {e}", file=sys.stderr)
    return dropped, bytes_freed


def write_cache_meta(
    cache_root: Path,
    *,
    lerobot_root: Path,
    fps: float,
    video_keys: list[str],
    jpg_quality: int,
    settings_hash: str,
    image_format: str = "jpg",
    png_compression: int = 1,
    complete: bool | None = None,
    source_videos: list[str] | None = None,
) -> Path:
    """Cache root に metadata を書く (patch 側 fps 読み取り + hash 検証 用)。"""
    meta_path = cache_root / CACHE_META_NAME
    _validate_format(image_format, png_compression)
    if complete is None:
        complete = image_format == "jpg"
    meta = {
        "schema_version": PNG_CACHE_SCHEMA_VERSION if image_format == "png" else CACHE_SCHEMA_VERSION,
        "image_format": image_format,
        "fps": float(fps),
        "video_keys": list(video_keys),
        "source_root": str(lerobot_root),
        "complete": complete,
    }
    meta["settings_hash" if complete else "pending_settings_hash"] = settings_hash
    if image_format == "png":
        _validate_fps(fps)
        meta.update(png_compression=png_compression, decoder="pyav")
    else:
        meta["jpg_quality"] = int(jpg_quality)
    if source_videos is not None:
        meta["source_videos"] = source_videos
    cache_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(meta_path, meta)
    return meta_path


def build_tasks(
    lerobot_root: Path,
    cache_root: Path,
    *,
    quality: int,
    force: bool,
    only_video_keys: list[str] | None = None,
    image_format: str = "jpg",
    png_compression: int = 1,
) -> tuple[list[tuple], float, list[str]]:
    """Precompute tasks を構築 + fps/video_keys を返す。

    only_video_keys を渡すとその cam だけ展開する (Issue #139、IR / head_right を省く)。
    """
    info_path = lerobot_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    _validate_format(image_format, png_compression)
    if image_format == "png":
        if "fps" not in info:
            raise ValueError(f"PNG requires metadata fps in {info_path}")
        fps = float(info["fps"])
        _validate_fps(fps)
    else:
        fps = float(info.get("fps") or 30.0)
    video_keys = _list_video_keys(info)
    if not video_keys:
        raise RuntimeError(f"no video features found in {info_path}")
    if only_video_keys is not None:
        unknown = sorted(set(only_video_keys) - set(video_keys))
        if unknown:
            raise ValueError(f"--video-keys に info.json に無い key がある: {unknown}")
        video_keys = sorted(set(only_video_keys))

    tasks: list[tuple] = []
    for vk, mp4 in _iter_mp4_files(lerobot_root, video_keys):
        rel = mp4.relative_to(lerobot_root / "videos" / vk)
        cache_dir = cache_dir_for_mp4(cache_root, vk, rel)
        task = (mp4, cache_dir, quality, force)
        if image_format == "png":
            task += (None, image_format, png_compression, fps)
        tasks.append(task)
    return tasks, fps, video_keys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lerobot-root",
        type=Path,
        required=True,
        help="LeRobot v3 dataset root (has meta/, data/, videos/)",
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="output cache dir (default: <lerobot_root>/frame_cache or frame_cache_png)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="parallel workers (default: os.cpu_count()-2)",
    )
    p.add_argument("--quality", type=int, default=90, help="JPEG quality (default 90)")
    p.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    p.add_argument("--png-compression", type=int, choices=range(10), default=1,
                   help="PNG compression level 0..9 (default 1; lossless at every level)")
    p.add_argument(
        "--force",
        action="store_true",
        help="hash 一致でも wipe + 再生成、既存 JPG も上書き",
    )
    p.add_argument(
        "--video-keys",
        default=None,
        help="展開する video key (comma 区切り)。未指定なら info.json の全 video key",
    )
    p.add_argument(
        "--keep-source-mp4",
        action="store_true",
        help="verify 後の source mp4 削除 (auto mp4 drop) をしない",
    )
    p.add_argument(
        "--max-frames-per-mp4",
        type=int,
        default=None,
        help="smoke 用: mp4 ごとに先頭 N frame だけ展開 (verify / mp4 drop はしない)",
    )
    args = p.parse_args()
    if args.jobs is not None and args.jobs <= 0:
        p.error("--jobs must be positive")
    if args.max_frames_per_mp4 is not None and args.max_frames_per_mp4 <= 0:
        p.error("--max-frames-per-mp4 must be positive")

    lerobot_root = args.lerobot_root.resolve()
    default_cache = "frame_cache_png" if args.image_format == "png" else "frame_cache"
    cache_root = (args.cache_root or lerobot_root / default_cache).resolve()
    if cache_root == lerobot_root or cache_root in lerobot_root.parents:
        raise ValueError("cache root must not be the dataset root or its ancestor")
    cache_root.mkdir(parents=True, exist_ok=True)

    only_video_keys = (
        [k.strip() for k in args.video_keys.split(",") if k.strip()]
        if args.video_keys
        else None
    )
    tasks, fps, video_keys = build_tasks(
        lerobot_root,
        cache_root,
        quality=args.quality,
        force=args.force,
        only_video_keys=only_video_keys,
        image_format=args.image_format,
        png_compression=args.png_compression,
    )
    # A selected-camera rebuild must also protect unselected source cameras.
    # Enumerating each camera directory preserves discovery through camera-dir
    # symlinks (Path.rglob on the videos root alone does not follow those).
    videos_root = lerobot_root / "videos"
    source_keys = sorted(path.name for path in videos_root.iterdir() if path.is_dir()) if videos_root.is_dir() else []
    all_sources = _iter_mp4_files(lerobot_root, source_keys)
    if any(source.is_relative_to(cache_root) or source.resolve().is_relative_to(cache_root)
           for _, source in all_sources):
        raise ValueError("cache root must not contain source MP4 files")

    # Settings hash: 一致 → skip、mismatch → wipe + 再生成
    expected_hash = compute_settings_hash(
        fps=fps, video_keys=video_keys, jpg_quality=args.quality,
        image_format=args.image_format, png_compression=args.png_compression,
    )
    existing_meta = _read_cache_meta(cache_root)
    existing_hash = read_existing_cache_hash(cache_root)
    previous_hash = existing_meta.get("pending_settings_hash") or existing_meta.get("settings_hash")
    if not isinstance(previous_hash, str):
        previous_hash = None
    hash_matches = existing_hash == expected_hash
    source_videos = [str(t[0].relative_to(lerobot_root / "videos")) for t in tasks]

    # A format change must not silently wipe the user's other cache.
    existing_format = existing_meta.get("image_format", "jpg") if existing_meta else None
    opposite_format = "jpg" if args.image_format == "png" else "png"
    opposite_files = next(cache_root.rglob(f"frame_*.{opposite_format}"), None)
    if not args.force and (
        existing_format not in {None, args.image_format} or opposite_files is not None
    ):
        raise RuntimeError("cache image format differs; choose a separate --cache-root or explicitly use --force")
    if args.image_format == "png" and not args.force:
        if previous_hash is not None and previous_hash != expected_hash:
            raise RuntimeError("PNG cache settings differ; choose a separate --cache-root or explicitly use --force")
        if existing_meta.get("source_videos", source_videos) != source_videos:
            raise RuntimeError("PNG source video inventory changed; restore sources or explicitly use --force")
        if previous_hash is None and any(cache_root.iterdir()):
            raise RuntimeError("unrecognized PNG cache metadata; choose a separate --cache-root or explicitly use --force")

    jobs = args.jobs if args.jobs is not None else max(1, (os.cpu_count() or 4) - 2)

    def verification_failures() -> list[tuple[Path, str]]:
        return _verify_tasks(
            tasks, image_format=args.image_format, fps=fps,
            png_compression=args.png_compression, jobs=jobs,
        )

    if args.force:
        if not tasks or not all(t[0].is_file() for t in tasks):
            raise RuntimeError("cannot regenerate cache: source MP4s are missing")
        print(f"[precompute] --force 指定 → wipe + 再生成")
        wipe_cache(cache_root)
    elif hash_matches and (
        # Legacy source symlinks may remain dangling after MP4 drop.
        (args.image_format == "jpg" and not any(t[0].is_file() for t in tasks))
        or (bool(tasks) and not verification_failures())
    ):
        print(
            f"[precompute] cache is up-to-date (settings_hash={expected_hash[:12]}...) → skip"
        )
        return 0
    elif previous_hash is not None and previous_hash != expected_hash:
        print(
            f"[precompute] settings changed (old_hash={previous_hash[:12]}... "
            f"new_hash={expected_hash[:12]}...) → wipe + 再生成"
        )
        # Wipe 前に source mp4 存在 check、無ければ再生成不可
        any_mp4_exists = any(task[0].exists() for task in tasks)
        if not any_mp4_exists:
            raise RuntimeError(
                "cache settings 変更で再生成必要だが source mp4 が既に drop 済 = 再生成不可。"
                f"HF から re-download → re-merge が必要 ({lerobot_root} を消して再走)。"
            )
        wipe_cache(cache_root)

    if not tasks:
        raise RuntimeError("no source MP4 files found; cannot build a complete frame cache")

    write_cache_meta(
        cache_root,
        lerobot_root=lerobot_root,
        fps=fps,
        video_keys=video_keys,
        jpg_quality=args.quality,
        settings_hash=expected_hash,
        image_format=args.image_format,
        png_compression=args.png_compression,
        complete=False,
        source_videos=source_videos,
    )

    print(f"[precompute] lerobot_root: {lerobot_root}")
    print(f"[precompute] cache_root:   {cache_root}")
    print(f"[precompute] fps={fps}, video_keys={len(video_keys)}: {video_keys}")
    print(f"[precompute] settings_hash={expected_hash}")
    print(f"[precompute] {len(tasks)} mp4 files to process")

    print(f"[precompute] parallel jobs: {jobs}")

    t0 = time.time()
    total_saved = 0
    total_skipped = 0
    total_frames = 0
    if args.max_frames_per_mp4 is not None:
        tasks = [(*t[:4], args.max_frames_per_mp4, *t[5:]) for t in tasks]
    if jobs <= 1:
        iterator = (_decode_one_mp4(t) for t in tasks)
    else:
        pool = mp.Pool(jobs)
        iterator = pool.imap_unordered(_decode_one_mp4, tasks)
    try:
        for i, (cache_dir, saved, skipped, total) in enumerate(iterator):
            total_saved += saved
            total_skipped += skipped
            total_frames += total
            if (i + 1) % max(1, len(tasks) // 20) == 0 or i + 1 == len(tasks):
                elapsed = time.time() - t0
                print(
                    f"[precompute] {i + 1}/{len(tasks)} mp4s, "
                    f"{total_saved} saved / {total_skipped} skipped / "
                    f"{total_frames} frames total ({elapsed:.1f}s)",
                    flush=True,
                )
    finally:
        if jobs > 1:
            pool.close()
            pool.join()

    elapsed = time.time() - t0
    print(
        f"[precompute] decode done in {elapsed:.1f}s: {total_saved} saved / "
        f"{total_skipped} skipped / {total_frames} frames total"
    )

    if args.max_frames_per_mp4 is not None:
        print(f"[precompute] smoke: first {args.max_frames_per_mp4} frames per mp4 (verify skipped)")
        return 0

    # Per-mp4 verify (drop 前必須)
    print(f"[precompute] verifying {len(tasks)} mp4 caches ...")
    fails = verification_failures()
    if fails:
        print(f"[precompute] verify FAILED for {len(fails)} mp4:", file=sys.stderr)
        for m, r in fails[:10]:
            print(f"  {m}: {r}", file=sys.stderr)
        return 1
    print(f"[precompute] verify passed ({len(tasks)} mp4 all match)")

    write_cache_meta(
        cache_root, lerobot_root=lerobot_root, fps=fps, video_keys=video_keys,
        jpg_quality=args.quality, settings_hash=expected_hash,
        image_format=args.image_format, png_compression=args.png_compression,
        complete=True, source_videos=source_videos,
    )

    if args.keep_source_mp4 or args.image_format == "png":
        print(f"[precompute] cache_root={cache_root} (source mp4 kept)")
        return 0

    # Auto mp4 drop (Issue #122、--keep-source-mp4 で無効化): lerobot_frame_cache_patch の
    # _check_cached_episodes_sufficient 差替で mp4 不在でも train 成立する。
    dropped, bytes_freed = drop_source_mp4s(tasks)
    print(
        f"[precompute] auto mp4 drop: {dropped} files removed, "
        f"{bytes_freed / (1024 * 1024 * 1024):.2f} GB freed"
    )
    print(f"[precompute] cache_root={cache_root} (source mp4 dropped)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
