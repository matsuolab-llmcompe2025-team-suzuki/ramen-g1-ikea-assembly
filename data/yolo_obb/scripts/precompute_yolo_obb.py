"""YOLO-OBB precompute (Issue #122 D-1'): JPG frame cache → per-mp4 parquet + manifest。

Head Left/Right の 2 cam のみ YOLO 推論、per-mp4 parquet を JPG cache と同 layout で
出力。yolo_ckpt_hash で cache dir を namespace 分離 (v10 vs v11a など ckpt 差替時に
互換維持、旧 hash も残す運用)。学習時はこの cache を D-2 dataloader が read する。

# Cache layout (JPG cache の layout をミラー)

    <output_root>/<yolo_ckpt_hash>/
        manifest.json                          # yolo_ckpt ref, cameras, class_names, ...
        <cam_key>/                              # e.g. "observation.images.cam_0"
            <chunk-000>/
                <file-000>.parquet             # 1 parquet per mp4 file
                <file-001>.parquet
                ...

# Parquet schema (1 row = 1 (frame, det_idx) pair, top_K 分は padding で埋める)

    frame_index:  int32     # mp4 file 内 frame index (JPG cache と同じ semantic)
    det_idx:      int8      # 0..top_K-1、confidence 降順
    verts:        list<float32>[8]  # xyxyxyxyn flat (xy0, xy1, xy2, xy3)
    conf:         float32
    class_id:     int8      # padding row は -1
    valid:        bool      # padding row は False (dataloader で mask 掛ける用)

# Usage (H100, model/yolo_obb pixi env)

    cd model/yolo_obb
    pixi run python ../../data/yolo_obb/scripts/precompute_yolo_obb.py \\
        --frame-cache-root ../../outputs/merged_sources/rotate_table_base_from_hf/frame_cache \\
        --yolo-ckpt Team-RAMEN/IROS2026_RAMEN_Hara_yoloobb_upperpolicy \\
        --yolo-revision <sha> \\
        --output-root ../../outputs/yolo_obb_cache \\
        --cameras observation.images.cam_0,observation.images.cam_1

    # dry-run で plan だけ表示
    python ... --dry-run

    # 全件保存 (Issue #139): top_k を十分大きく取り、超えたら止める。1 GPU に 4 process 並べる
    for i in 0 1 2 3; do
      python ... --cameras observation.images.cam_0 --top-k 32 --fail-on-truncation \\
          --num-shards 4 --shard-index $i &
    done; wait
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

# lazy: ultralytics / torch / huggingface_hub は model/yolo_obb pixi env のみ install。
# ここでは import せず、load_ultralytics_predict_fn 内で lazy import する。

CACHE_SCHEMA_VERSION = "team_ramen_yolo_obb_cache_v1"
CACHE_MANIFEST_NAME = "manifest.json"
FRAME_COUNT_NAME = "_frame_count.txt"
FRAME_CACHE_META_NAME = "_cache_meta.json"
DEFAULT_CAMERAS = ("observation.images.cam_0", "observation.images.cam_1")


# ============================================================================
# YOLO ckpt reference + hash 化 (HF repo+revision or local .pt file)
# ============================================================================


@dataclass(frozen=True)
class YoloCkptRef:
    """YOLO ckpt reference。cache dir 名 (hash) + manifest 記録に使う。

    HF repo は revision 必須 (deterministic hash 保証)、local file は content SHA。
    """

    kind: str  # "hf" or "file"
    repo_id: str | None = None
    revision: str | None = None
    file_path: Path | None = None
    file_sha256: str | None = None

    def to_ref_string(self) -> str:
        """Manifest 記録用の人間可読 ref (hash 化の入力にもなる)。"""
        if self.kind == "hf":
            return f"hf:{self.repo_id}@{self.revision}"
        return f"file:sha256:{self.file_sha256}"

    def to_hash(self) -> str:
        """Cache dir 名に使う 12 char hex hash (SHA256 の先頭)。"""
        return hashlib.sha256(self.to_ref_string().encode("utf-8")).hexdigest()[:12]


def resolve_ckpt(ckpt: str, revision: str | None) -> YoloCkptRef:
    """CLI --yolo-ckpt (HF repo id or local file path) を YoloCkptRef に解決。

    - local file が実在すれば file mode、content SHA256 を計算
    - そうでなければ HF repo id 扱い、revision 必須 (未指定なら error)
    """
    p = Path(ckpt)
    if p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return YoloCkptRef(
            kind="file", file_path=p.resolve(), file_sha256=h.hexdigest()
        )
    if revision is None:
        raise ValueError(
            f"--yolo-revision is required when --yolo-ckpt is an HF repo id "
            f"(deterministic hash 化のため必須): got --yolo-ckpt={ckpt!r}"
        )
    return YoloCkptRef(kind="hf", repo_id=ckpt, revision=revision)


# ============================================================================
# JPG frame cache 走査
# ============================================================================


def iter_mp4_cache_dirs(
    frame_cache_root: Path, cameras: list[str]
) -> Iterator[tuple[str, Path, Path]]:
    """(cam_key, mp4_rel_path (chunk-XXX/file-YYY), mp4_cache_dir) を iterate。

    mp4_rel_path は cam dir 内 relative path 部分。parquet 出力先の naming に使う。
    """
    for cam in cameras:
        cam_dir = frame_cache_root / cam
        if not cam_dir.is_dir():
            raise FileNotFoundError(
                f"cam cache dir not found: {cam_dir} "
                f"(precompute_frame_cache.py を先に実行済か確認)"
            )
        for chunk_dir in sorted(cam_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for file_dir in sorted(chunk_dir.iterdir()):
                if not file_dir.is_dir():
                    continue
                rel = Path(chunk_dir.name) / file_dir.name
                yield cam, rel, file_dir


def list_frames(mp4_cache_dir: Path) -> list[Path]:
    """mp4 cache dir から frame_XXXXXX.jpg を昇順で列挙。

    _frame_count.txt marker で total frames を確定、hole (欠 index) は error。
    """
    marker = mp4_cache_dir / FRAME_COUNT_NAME
    if not marker.is_file():
        raise FileNotFoundError(f"missing {FRAME_COUNT_NAME}: {mp4_cache_dir}")
    n = int(marker.read_text(encoding="utf-8").strip())
    frames = [mp4_cache_dir / f"frame_{i:06d}.jpg" for i in range(n)]
    missing = [p.name for p in frames if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"frame cache incomplete in {mp4_cache_dir}: "
            f"first missing {missing[:3]} ({len(missing)} total)"
        )
    return frames


# ============================================================================
# Parquet schema + padding + write
# ============================================================================


def make_parquet_schema(top_k: int) -> pa.Schema:
    """fixed shape schema (verts は length-8 fixed list、dataloader 側で to_numpy 一発)。"""
    return pa.schema(
        [
            pa.field("frame_index", pa.int32(), nullable=False),
            pa.field("det_idx", pa.int8(), nullable=False),
            pa.field("verts", pa.list_(pa.float32(), 8), nullable=False),
            pa.field("conf", pa.float32(), nullable=False),
            pa.field("class_id", pa.int8(), nullable=False),
            pa.field("valid", pa.bool_(), nullable=False),
        ],
        metadata={b"top_k": str(top_k).encode()},
    )


def pad_dets_to_top_k(
    dets: list[tuple[list[float], float, int]], top_k: int
) -> list[dict]:
    """YOLO 出力 (verts flat 8, conf, class_id) を top_K に pad。

    dets は confidence 降順 (top-K 選別済) を想定。< K は zeros + valid=False で埋める。
    """
    result: list[dict] = []
    for i in range(top_k):
        if i < len(dets):
            verts, conf, cls_id = dets[i]
            if len(verts) != 8:
                raise ValueError(f"verts must be length 8, got {len(verts)}")
            result.append(
                {
                    "det_idx": i,
                    "verts": [float(v) for v in verts],
                    "conf": float(conf),
                    "class_id": int(cls_id),
                    "valid": True,
                }
            )
        else:
            result.append(
                {
                    "det_idx": i,
                    "verts": [0.0] * 8,
                    "conf": 0.0,
                    "class_id": -1,
                    "valid": False,
                }
            )
    return result


def write_mp4_parquet(
    output_path: Path,
    frame_dets: list[list[dict]],
    schema: pa.Schema,
) -> None:
    """per-frame padded det list を per-mp4 parquet (snappy 圧縮) に書く。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list] = {
        "frame_index": [],
        "det_idx": [],
        "verts": [],
        "conf": [],
        "class_id": [],
        "valid": [],
    }
    for frame_idx, dets in enumerate(frame_dets):
        for d in dets:
            rows["frame_index"].append(frame_idx)
            rows["det_idx"].append(d["det_idx"])
            rows["verts"].append(d["verts"])
            rows["conf"].append(d["conf"])
            rows["class_id"].append(d["class_id"])
            rows["valid"].append(d["valid"])
    table = pa.table(rows, schema=schema)
    pq.write_table(table, output_path, compression="snappy")


# ============================================================================
# YOLO 推論 (batch)
# ============================================================================


# 型: list[frame_path] → list[per-frame dets]、per-frame dets = list[(verts8, conf, class_id)]
PredictFn = Callable[[list[Path]], list[list[tuple[list[float], float, int]]]]


class TruncationError(RuntimeError):
    """1 frame の検出数が top_k を超えた (`--fail-on-truncation` 時)。"""


def run_yolo_on_frames(
    predict_fn: PredictFn,
    frames: list[Path],
    batch_size: int,
    top_k: int,
    stats: Counter | None = None,
    fail_on_truncation: bool = False,
) -> list[list[dict]]:
    """全 frame paths を batch 単位で予測、per-frame padded det list を返す。

    Args:
        stats: 渡すと 1 frame あたり検出数の分布 (key = 検出数) を加算する (Issue #139)。
        fail_on_truncation: True なら top_k を超える検出が出た時点で TruncationError。
            全件保存 (top_k を十分大きく取る) の前提が崩れていないかを検出する。
    """
    all_dets: list[list[dict]] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        batch_dets = predict_fn(batch)
        if len(batch_dets) != len(batch):
            raise RuntimeError(
                f"predict_fn returned {len(batch_dets)} results for batch of {len(batch)}"
            )
        for frame_path, dets in zip(batch, batch_dets):
            if stats is not None:
                stats[len(dets)] += 1
            if fail_on_truncation and len(dets) > top_k:
                # cache の hash は ckpt だけで決まり、既存の parquet は skip される。hash dir を
                # 残したまま top_k だけ上げると、他 shard の書き済み parquet (旧 top_k) が混ざる
                raise TruncationError(
                    f"{frame_path}: {len(dets)} detections > top_k={top_k} "
                    "(--top-k を上げ、この hash の出力 dir を消してから再実行)"
                )
            # confidence 降順 sort + top_K 切り (念のため、ultralytics も既に降順)
            sorted_dets = sorted(dets, key=lambda d: -d[1])[:top_k]
            all_dets.append(pad_dets_to_top_k(sorted_dets, top_k))
    return all_dets


def assign_shards(weights: list[int], num_shards: int) -> list[int]:
    """item ごとの重み (frame 数) から shard 番号を決める (重い順に一番軽い shard へ)。

    全 shard process が同じ入力から同じ割当を計算するので、協調なしで分担できる。
    Returns: item index と同順の shard 番号 list。
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    load = [0] * num_shards
    assignment = [0] * len(weights)
    for i in sorted(range(len(weights)), key=lambda j: (-weights[j], j)):
        shard = min(range(num_shards), key=lambda s: (load[s], s))
        assignment[i] = shard
        load[shard] += weights[i]
    return assignment


def load_ultralytics_predict_fn(
    ckpt_ref: YoloCkptRef,
    imgsz: int,
    conf: float,
    device: str,
) -> tuple[PredictFn, dict[int, str]]:
    """ultralytics YOLO を load、predict_fn + class_names を返す。lazy import。"""
    # lazy: model/yolo_obb pixi env のみ install (default env は unit test 用に軽量維持)
    from ultralytics import YOLO  # noqa: PLC0415

    if ckpt_ref.kind == "file":
        weight_path = ckpt_ref.file_path
    else:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        local = Path(
            snapshot_download(
                repo_id=ckpt_ref.repo_id,
                revision=ckpt_ref.revision,
                repo_type="model",
            )
        )
        pt_files = list(local.rglob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"no .pt weight found under {local}")
        # best.pt を優先 (ultralytics 慣例)、無ければ最初の .pt
        best = [p for p in pt_files if p.name == "best.pt"]
        weight_path = best[0] if best else pt_files[0]
    model = YOLO(str(weight_path))
    class_names: dict[int, str] = dict(model.names)

    def predict_fn(
        paths: list[Path],
    ) -> list[list[tuple[list[float], float, int]]]:
        results = model(
            [str(p) for p in paths],
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        out: list[list[tuple[list[float], float, int]]] = []
        for r in results:
            if r.obb is None or r.obb.cls is None or len(r.obb.cls) == 0:
                out.append([])
                continue
            cls_arr = r.obb.cls.cpu().numpy()
            conf_arr = r.obb.conf.cpu().numpy()
            verts_arr = r.obb.xyxyxyxyn.cpu().numpy()  # (N, 4, 2)
            dets: list[tuple[list[float], float, int]] = []
            for i in range(len(cls_arr)):
                dets.append(
                    (
                        verts_arr[i].reshape(-1).tolist(),
                        float(conf_arr[i]),
                        int(cls_arr[i]),
                    )
                )
            out.append(dets)
        return out

    return predict_fn, class_names


# ============================================================================
# Manifest
# ============================================================================


def write_manifest(
    output_root: Path,
    ckpt_ref: YoloCkptRef,
    class_names: dict[int, str],
    cameras: list[str],
    top_k: int,
    imgsz: int,
    conf_threshold: float,
    num_classes: int,
) -> None:
    """cache dir 直下に manifest.json を書く (dataloader が read してこの hash が意図通りか判定)。"""
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "yolo_ckpt_hash": ckpt_ref.to_hash(),
        "yolo_ckpt_ref": ckpt_ref.to_ref_string(),
        "yolo_class_names": {str(k): v for k, v in sorted(class_names.items())},
        "cameras": list(cameras),
        "top_k": int(top_k),
        "imgsz": int(imgsz),
        "conf_threshold": float(conf_threshold),
        "num_classes": int(num_classes),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_root / CACHE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ============================================================================
# CLI
# ============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--frame-cache-root",
        type=Path,
        required=True,
        help="precompute_frame_cache.py の出力 dir (<merged_source>/frame_cache)",
    )
    p.add_argument(
        "--yolo-ckpt",
        required=True,
        help="HF repo id (e.g. Team-RAMEN/...) or local .pt file path",
    )
    p.add_argument(
        "--yolo-revision",
        default=None,
        help="HF revision sha (--yolo-ckpt が HF repo id の場合 required)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/yolo_obb_cache"),
        help="cache 出力 root (<output_root>/<hash>/ に書く)",
    )
    p.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERAS),
        help="comma-separated cam keys (default: head Left/Right = cam_0, cam_1)",
    )
    p.add_argument("--top-k", type=int, default=4, help="per-frame max detections")
    p.add_argument("--num-classes", type=int, default=7, help="expected class 数 (ckpt と一致 verify)")
    p.add_argument("--imgsz", type=int, default=800, help="YOLO 推論 image size")
    p.add_argument("--conf", type=float, default=0.15, help="YOLO conf threshold")
    p.add_argument("--batch-size", type=int, default=64, help="batch size (H100 で 64-128 が efficient)")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    p.add_argument(
        "--force",
        action="store_true",
        help="既存 per-mp4 parquet を上書き (default は skip)",
    )
    p.add_argument(
        "--fail-on-truncation",
        action="store_true",
        help="1 frame の検出数が --top-k を超えたら止める (全件保存の確認、Issue #139)",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="mp4 を frame 数で均した N 個に分け、1 GPU に複数 process を並べて回す",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="この process が担当する shard (0..num_shards-1)。manifest は shard 0 だけが書く",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="plan のみ表示、YOLO load や推論はしない",
    )
    args = p.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        p.error(f"--shard-index {args.shard_index} は 0..{args.num_shards - 1} の範囲外")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not cameras:
        raise ValueError("--cameras must not be empty")

    ckpt_ref = resolve_ckpt(args.yolo_ckpt, args.yolo_revision)
    cache_dir = args.output_root / ckpt_ref.to_hash()

    all_mp4_dirs = list(iter_mp4_cache_dirs(args.frame_cache_root, cameras))
    frame_counts = []
    for _cam, _rel, mp4_cache_dir in all_mp4_dirs:
        marker = mp4_cache_dir / FRAME_COUNT_NAME
        frame_counts.append(
            int(marker.read_text(encoding="utf-8").strip()) if marker.is_file() else 0
        )
    shard_of = assign_shards(frame_counts, args.num_shards)
    mp4_dirs = [d for d, s in zip(all_mp4_dirs, shard_of) if s == args.shard_index]
    total_frames = sum(n for n, s in zip(frame_counts, shard_of) if s == args.shard_index)

    print(f"[plan] cache dir:      {cache_dir}")
    print(f"[plan] ckpt ref:       {ckpt_ref.to_ref_string()}")
    print(f"[plan] ckpt hash:      {ckpt_ref.to_hash()}")
    print(f"[plan] cameras:        {cameras}")
    print(f"[plan] shard:          {args.shard_index}/{args.num_shards}")
    print(f"[plan] mp4 files:      {len(mp4_dirs)} / {len(all_mp4_dirs)} (per-cam × chunk × file)")
    print(f"[plan] total frames:   {total_frames}")
    print(f"[plan] top_k={args.top_k}, imgsz={args.imgsz}, conf={args.conf}, batch={args.batch_size}, device={args.device}")

    if args.dry_run:
        print("[dry-run] YOLO load / inference skipped")
        return 0

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[load] initializing YOLO ...", flush=True)
    predict_fn, class_names = load_ultralytics_predict_fn(
        ckpt_ref, imgsz=args.imgsz, conf=args.conf, device=args.device
    )
    if len(class_names) != args.num_classes:
        raise ValueError(
            f"YOLO ckpt has {len(class_names)} classes, expected {args.num_classes} "
            f"(--num-classes と ckpt が不一致、config を確認)"
        )
    print(f"[load] class_names = {class_names}")

    if args.shard_index == 0:
        write_manifest(
            cache_dir,
            ckpt_ref,
            class_names,
            cameras,
            top_k=args.top_k,
            imgsz=args.imgsz,
            conf_threshold=args.conf,
            num_classes=args.num_classes,
        )
    schema = make_parquet_schema(args.top_k)

    det_hist: Counter = Counter()
    for i, (cam, rel, mp4_cache_dir) in enumerate(mp4_dirs, 1):
        out_parquet = cache_dir / cam / f"{rel}.parquet"
        if out_parquet.exists() and not args.force:
            print(f"[{i}/{len(mp4_dirs)}] skip (exists): {cam}/{rel}.parquet", flush=True)
            continue
        t0 = time.time()
        frames = list_frames(mp4_cache_dir)
        mp4_hist: Counter = Counter()
        frame_dets = run_yolo_on_frames(
            predict_fn,
            frames,
            batch_size=args.batch_size,
            top_k=args.top_k,
            stats=mp4_hist,
            fail_on_truncation=args.fail_on_truncation,
        )
        # 途中で止まっても半端な parquet を残さないよう tmp に書いてから rename
        tmp_parquet = out_parquet.with_name(out_parquet.name + ".tmp")
        write_mp4_parquet(tmp_parquet, frame_dets, schema)
        tmp_parquet.replace(out_parquet)
        det_hist.update(mp4_hist)
        dt = time.time() - t0
        fps = len(frames) / max(dt, 0.01)
        print(
            f"[{i}/{len(mp4_dirs)}] {cam}/{rel}.parquet "
            f"({len(frames)} frames, {dt:.1f}s = {fps:.1f} fps, "
            f"max dets/frame = {max(mp4_hist) if mp4_hist else 0})",
            flush=True,
        )

    if det_hist:
        n = sum(det_hist.values())
        mean = sum(k * v for k, v in det_hist.items()) / n
        over = sum(v for k, v in det_hist.items() if k > args.top_k)
        print(
            f"[summary] shard {args.shard_index}: {n} frames, dets/frame mean={mean:.2f} "
            f"max={max(det_hist)}, frames over top_k={over}, hist={dict(sorted(det_hist.items()))}",
            flush=True,
        )
    print(f"[done] cache ready at {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
