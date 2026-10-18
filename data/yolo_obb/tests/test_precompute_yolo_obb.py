"""Unit tests for data/yolo_obb/scripts/precompute_yolo_obb.py (Issue #122 D-1')。

ultralytics / torch / huggingface_hub は model/yolo_obb pixi env のみ install なので
default env test では load_ultralytics_predict_fn は触らない (mock predict_fn を渡す)。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[3]  # iros_2026_ramen/
SCRIPT_PATH = ROOT / "data" / "yolo_obb" / "scripts" / "precompute_yolo_obb.py"


def load_module():
    import sys

    spec = importlib.util.spec_from_file_location("precompute_yolo_obb", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass の annotation 解決 (typing.get_type_hints) が sys.modules を参照する
    # ため、exec_module 前に登録しておく (from __future__ import annotations 使用時必須)。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================================
# YoloCkptRef + resolve_ckpt
# ============================================================================


def test_resolve_ckpt_local_file_uses_content_sha(tmp_path: Path) -> None:
    m = load_module()
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"fake weight content")
    ref = m.resolve_ckpt(str(pt), revision=None)
    assert ref.kind == "file"
    assert ref.file_path == pt.resolve()
    assert ref.file_sha256 is not None
    assert len(ref.file_sha256) == 64
    # hash 反映で ref_string / to_hash が変わる
    ref2 = m.resolve_ckpt(str(pt), revision=None)
    assert ref.file_sha256 == ref2.file_sha256


def test_resolve_ckpt_hf_repo_requires_revision(tmp_path: Path) -> None:
    m = load_module()
    with pytest.raises(ValueError, match="--yolo-revision is required"):
        m.resolve_ckpt("Team-RAMEN/dummy", revision=None)


def test_resolve_ckpt_hf_repo_with_revision(tmp_path: Path) -> None:
    m = load_module()
    ref = m.resolve_ckpt("Team-RAMEN/dummy", revision="abcdef12")
    assert ref.kind == "hf"
    assert ref.repo_id == "Team-RAMEN/dummy"
    assert ref.revision == "abcdef12"
    assert ref.to_ref_string() == "hf:Team-RAMEN/dummy@abcdef12"


def test_yolo_ckpt_hash_deterministic_and_differs_by_revision() -> None:
    m = load_module()
    a = m.YoloCkptRef(kind="hf", repo_id="X/Y", revision="sha1")
    b = m.YoloCkptRef(kind="hf", repo_id="X/Y", revision="sha1")
    c = m.YoloCkptRef(kind="hf", repo_id="X/Y", revision="sha2")
    assert a.to_hash() == b.to_hash()
    assert a.to_hash() != c.to_hash()
    assert len(a.to_hash()) == 12


# ============================================================================
# iter_mp4_cache_dirs / list_frames (JPG cache 走査)
# ============================================================================


def _make_dummy_frame_cache(root: Path, cameras: list[str], mp4_layout: dict[str, list[int]]) -> None:
    """dummy JPG frame cache を作る (JPG は空 file、_frame_count.txt marker のみ有意)。

    mp4_layout: {"chunk-000/file-000": 3, "chunk-000/file-001": 2} → 3+2 frames
    """
    for cam in cameras:
        for rel, n_frames in mp4_layout.items():
            d = root / cam / rel
            d.mkdir(parents=True, exist_ok=True)
            (d / "_frame_count.txt").write_text(str(n_frames))
            for i in range(n_frames):
                (d / f"frame_{i:06d}.jpg").write_bytes(b"")


def test_iter_mp4_cache_dirs_returns_all_cam_chunk_file(tmp_path: Path) -> None:
    m = load_module()
    _make_dummy_frame_cache(
        tmp_path,
        cameras=["observation.images.cam_0", "observation.images.cam_1"],
        mp4_layout={"chunk-000/file-000": 3, "chunk-000/file-001": 2},
    )
    results = list(
        m.iter_mp4_cache_dirs(
            tmp_path, ["observation.images.cam_0", "observation.images.cam_1"]
        )
    )
    # 2 cam × 2 mp4 = 4 entries
    assert len(results) == 4
    cams = {r[0] for r in results}
    assert cams == {"observation.images.cam_0", "observation.images.cam_1"}
    rels = {str(r[1]) for r in results}
    assert rels == {"chunk-000/file-000", "chunk-000/file-001"}


def test_iter_mp4_cache_dirs_missing_cam_raises(tmp_path: Path) -> None:
    m = load_module()
    _make_dummy_frame_cache(
        tmp_path,
        cameras=["observation.images.cam_0"],
        mp4_layout={"chunk-000/file-000": 3},
    )
    with pytest.raises(FileNotFoundError, match="cam cache dir not found"):
        list(m.iter_mp4_cache_dirs(tmp_path, ["observation.images.cam_1"]))


def test_list_frames_incomplete_cache_raises(tmp_path: Path) -> None:
    m = load_module()
    d = tmp_path / "cam" / "chunk-000" / "file-000"
    d.mkdir(parents=True)
    (d / "_frame_count.txt").write_text("5")
    # 3 frames しか置かない (5 のうち index 3, 4 が hole)
    for i in range(3):
        (d / f"frame_{i:06d}.jpg").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="frame cache incomplete"):
        m.list_frames(d)


def test_list_frames_missing_marker_raises(tmp_path: Path) -> None:
    m = load_module()
    d = tmp_path / "cam" / "chunk-000" / "file-000"
    d.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="_frame_count.txt"):
        m.list_frames(d)


# ============================================================================
# pad_dets_to_top_k + make_parquet_schema + write_mp4_parquet
# ============================================================================


def test_pad_dets_shorter_than_k_padded_with_invalid() -> None:
    m = load_module()
    dets = [([0.1] * 8, 0.9, 3), ([0.2] * 8, 0.8, 4)]
    padded = m.pad_dets_to_top_k(dets, top_k=4)
    assert len(padded) == 4
    assert padded[0]["valid"] is True and padded[0]["class_id"] == 3
    assert padded[1]["valid"] is True and padded[1]["class_id"] == 4
    assert padded[2]["valid"] is False and padded[2]["class_id"] == -1
    assert padded[3]["valid"] is False and padded[3]["conf"] == 0.0


def test_pad_dets_exact_k_no_padding() -> None:
    m = load_module()
    dets = [([float(i)] * 8, 0.5, i) for i in range(4)]
    padded = m.pad_dets_to_top_k(dets, top_k=4)
    assert len(padded) == 4
    assert all(p["valid"] for p in padded)


def test_pad_dets_empty_all_invalid() -> None:
    m = load_module()
    padded = m.pad_dets_to_top_k([], top_k=4)
    assert len(padded) == 4
    assert all(not p["valid"] for p in padded)
    assert all(p["class_id"] == -1 for p in padded)


def test_pad_dets_wrong_verts_length_raises() -> None:
    m = load_module()
    with pytest.raises(ValueError, match="verts must be length 8"):
        m.pad_dets_to_top_k([([0.1] * 6, 0.9, 3)], top_k=4)


def test_make_parquet_schema_has_expected_fields() -> None:
    m = load_module()
    schema = m.make_parquet_schema(top_k=4)
    field_names = {f.name for f in schema}
    assert field_names == {"frame_index", "det_idx", "verts", "conf", "class_id", "valid"}
    assert schema.metadata[b"top_k"] == b"4"


def test_write_mp4_parquet_roundtrip(tmp_path: Path) -> None:
    m = load_module()
    top_k = 4
    schema = m.make_parquet_schema(top_k)
    # 2 frames: frame 0 に 2 det、frame 1 に 0 det (all padding)
    frame_dets = [
        m.pad_dets_to_top_k([([0.1] * 8, 0.9, 3), ([0.2] * 8, 0.8, 4)], top_k),
        m.pad_dets_to_top_k([], top_k),
    ]
    out = tmp_path / "cam_0" / "chunk-000" / "file-000.parquet"
    m.write_mp4_parquet(out, frame_dets, schema)
    assert out.is_file()
    table = pq.read_table(out)
    # 2 frames × 4 top_K = 8 rows
    assert table.num_rows == 8
    # frame_index の分布
    frame_indices = table.column("frame_index").to_pylist()
    assert frame_indices.count(0) == 4
    assert frame_indices.count(1) == 4
    # frame 0 の valid=True は 2 個、frame 1 は 0 個
    valids = table.column("valid").to_pylist()
    frame_0_valid = sum(1 for fi, v in zip(frame_indices, valids) if fi == 0 and v)
    frame_1_valid = sum(1 for fi, v in zip(frame_indices, valids) if fi == 1 and v)
    assert frame_0_valid == 2
    assert frame_1_valid == 0


# ============================================================================
# run_yolo_on_frames (mock predict_fn)
# ============================================================================


def test_run_yolo_on_frames_with_mock_predict_fn_batches_correctly(tmp_path: Path) -> None:
    m = load_module()
    frames = [tmp_path / f"frame_{i}.jpg" for i in range(10)]
    calls: list[int] = []

    def mock_predict(paths):
        calls.append(len(paths))
        # 全 frame に 2 detection を返す (confidence 順)
        return [
            [([0.1] * 8, 0.9, 1), ([0.2] * 8, 0.7, 2)]
            for _ in paths
        ]

    result = m.run_yolo_on_frames(mock_predict, frames, batch_size=4, top_k=4)
    # batch 分割: 4, 4, 2
    assert calls == [4, 4, 2]
    assert len(result) == 10
    # 全 frame で 4 padded det (2 valid + 2 padding)
    for per_frame in result:
        assert len(per_frame) == 4
        valid_count = sum(1 for d in per_frame if d["valid"])
        assert valid_count == 2


def test_run_yolo_on_frames_sorts_by_conf_descending(tmp_path: Path) -> None:
    m = load_module()
    frames = [tmp_path / "frame_0.jpg"]

    def mock_predict(paths):
        # わざと降順でない det を返す
        return [
            [([0.0] * 8, 0.3, 0), ([0.1] * 8, 0.9, 1), ([0.2] * 8, 0.6, 2)]
        ]

    result = m.run_yolo_on_frames(mock_predict, frames, batch_size=1, top_k=4)
    per_frame = result[0]
    valid_dets = [d for d in per_frame if d["valid"]]
    confs = [d["conf"] for d in valid_dets]
    assert confs == sorted(confs, reverse=True)
    assert confs[0] == pytest.approx(0.9)


def test_run_yolo_on_frames_truncates_to_top_k(tmp_path: Path) -> None:
    m = load_module()
    frames = [tmp_path / "frame_0.jpg"]

    def mock_predict(paths):
        # 6 dets を返す (top_K=4 で 4 に切られる)
        return [
            [([float(i)] * 8, 1.0 - i * 0.1, i) for i in range(6)]
        ]

    result = m.run_yolo_on_frames(mock_predict, frames, batch_size=1, top_k=4)
    per_frame = result[0]
    assert len(per_frame) == 4
    # top 4 は class 0-3 (conf 1.0, 0.9, 0.8, 0.7)
    class_ids = [d["class_id"] for d in per_frame if d["valid"]]
    assert class_ids == [0, 1, 2, 3]


def test_run_yolo_on_frames_wrong_count_raises(tmp_path: Path) -> None:
    m = load_module()
    frames = [tmp_path / f"f_{i}.jpg" for i in range(4)]

    def bad_predict(paths):
        return [[] for _ in range(len(paths) - 1)]  # 1 個足りない

    with pytest.raises(RuntimeError, match="predict_fn returned"):
        m.run_yolo_on_frames(bad_predict, frames, batch_size=4, top_k=4)


# ============================================================================
# write_manifest
# ============================================================================


def test_write_manifest_contains_hash_ref_class_names(tmp_path: Path) -> None:
    m = load_module()
    ckpt_ref = m.YoloCkptRef(kind="hf", repo_id="X/Y", revision="sha1")
    m.write_manifest(
        tmp_path,
        ckpt_ref,
        class_names={0: "workspace", 1: "leg", 2: "leg_tip"},
        cameras=["observation.images.cam_0", "observation.images.cam_1"],
        top_k=4,
        imgsz=800,
        conf_threshold=0.15,
        num_classes=7,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == "team_ramen_yolo_obb_cache_v1"
    assert manifest["yolo_ckpt_hash"] == ckpt_ref.to_hash()
    assert manifest["yolo_ckpt_ref"] == "hf:X/Y@sha1"
    assert manifest["yolo_class_names"] == {"0": "workspace", "1": "leg", "2": "leg_tip"}
    assert manifest["cameras"] == ["observation.images.cam_0", "observation.images.cam_1"]
    assert manifest["top_k"] == 4
    assert "generated_at" in manifest


# ============================================================================
# parse_args
# ============================================================================


def test_parse_args_defaults_head_l_r_cameras() -> None:
    m = load_module()
    args = m.parse_args(
        ["--frame-cache-root", "/tmp/fc", "--yolo-ckpt", "/tmp/best.pt"]
    )
    # cam_0 (head_left) + cam_1 (head_right) が default
    assert args.cameras == "observation.images.cam_0,observation.images.cam_1"
    assert args.top_k == 4
    assert args.imgsz == 800
    assert args.conf == pytest.approx(0.15)


# ============================================================================
# 全件保存の確認 / shard 分割 (Issue #139)
# ============================================================================

mod = load_module()


def _dets(n: int) -> list[tuple[list[float], float, int]]:
    return [([0.1] * 8, 0.9 - 0.01 * i, i % 7) for i in range(n)]


def test_run_yolo_on_frames_records_detection_histogram(tmp_path: Path) -> None:
    from collections import Counter

    counts = [0, 3, 3, 5]
    frames = [tmp_path / f"f{i}.jpg" for i in range(len(counts))]
    batches = iter([[_dets(c) for c in counts[:2]], [_dets(c) for c in counts[2:]]])
    hist: Counter = Counter()
    out = mod.run_yolo_on_frames(lambda b: next(batches), frames, batch_size=2, top_k=8, stats=hist)
    assert hist == Counter({0: 1, 3: 2, 5: 1})
    assert [sum(d["valid"] for d in f) for f in out] == counts


def test_run_yolo_on_frames_fail_on_truncation(tmp_path: Path) -> None:
    frames = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    with pytest.raises(mod.TruncationError, match="b.jpg"):
        mod.run_yolo_on_frames(
            lambda b: [_dets(4), _dets(5)], frames, batch_size=2, top_k=4, fail_on_truncation=True
        )


def test_run_yolo_on_frames_exact_top_k_is_not_truncation(tmp_path: Path) -> None:
    out = mod.run_yolo_on_frames(
        lambda b: [_dets(4)], [tmp_path / "a.jpg"], batch_size=1, top_k=4, fail_on_truncation=True
    )
    assert all(d["valid"] for d in out[0])


def test_assign_shards_balances_by_weight_and_covers_all() -> None:
    weights = [50, 10, 40, 30, 20, 25]
    shards = mod.assign_shards(weights, 3)
    assert len(shards) == len(weights)
    loads = [sum(w for w, s in zip(weights, shards) if s == k) for k in range(3)]
    assert max(loads) - min(loads) <= 15
    assert shards == mod.assign_shards(weights, 3)  # 全 process で同じ割当になる


def test_assign_shards_single_shard() -> None:
    assert mod.assign_shards([3, 1, 2], 1) == [0, 0, 0]


def test_parse_args_rejects_shard_index_out_of_range() -> None:
    with pytest.raises(SystemExit):
        mod.parse_args(
            ["--frame-cache-root", "x", "--yolo-ckpt", "a/b", "--num-shards", "2", "--shard-index", "2"]
        )


def _fake_frame_cache(root: Path, cam: str, files: dict[str, int]) -> None:
    for name, n in files.items():
        d = root / cam / "chunk-000" / name
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"frame_{i:06d}.jpg").write_bytes(b"x")
        (d / "_frame_count.txt").write_text(str(n), encoding="utf-8")


def test_main_shards_split_mp4s_and_only_shard0_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    cam = "observation.images.cam_0"
    _fake_frame_cache(tmp_path / "fc", cam, {"file-000": 5, "file-001": 3, "file-002": 2})
    weight = tmp_path / "w.pt"
    weight.write_bytes(b"weights")
    monkeypatch.setattr(
        mod,
        "load_ultralytics_predict_fn",
        lambda ref, imgsz, conf, device: (
            lambda paths: [_dets(2) for _ in paths],
            {i: f"c{i}" for i in range(7)},
        ),
    )
    common = ["--frame-cache-root", str(tmp_path / "fc"), "--yolo-ckpt", str(weight),
              "--output-root", str(tmp_path / "out"), "--cameras", cam, "--top-k", "8",
              "--fail-on-truncation", "--num-shards", "2", "--device", "cpu"]
    assert mod.main([*common, "--shard-index", "1"]) == 0
    cache_dir = next((tmp_path / "out").iterdir())
    assert not (cache_dir / "manifest.json").exists()  # shard 1 は manifest を書かない
    assert mod.main([*common, "--shard-index", "0"]) == 0
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["top_k"] == 8 and manifest["cameras"] == [cam]
    parquets = sorted(p.name for p in (cache_dir / cam / "chunk-000").glob("*.parquet"))
    assert parquets == ["file-000.parquet", "file-001.parquet", "file-002.parquet"]
    assert not list(cache_dir.rglob("*.tmp"))
    table = pq.read_table(cache_dir / cam / "chunk-000" / "file-000.parquet")
    assert table.num_rows == 5 * 8


def test_main_stops_on_truncation(tmp_path: Path, monkeypatch) -> None:
    cam = "observation.images.cam_0"
    _fake_frame_cache(tmp_path / "fc", cam, {"file-000": 2})
    weight = tmp_path / "w.pt"
    weight.write_bytes(b"weights")
    monkeypatch.setattr(
        mod,
        "load_ultralytics_predict_fn",
        lambda ref, imgsz, conf, device: (lambda paths: [_dets(9) for _ in paths], {i: "c" for i in range(7)}),
    )
    with pytest.raises(mod.TruncationError):
        mod.main(["--frame-cache-root", str(tmp_path / "fc"), "--yolo-ckpt", str(weight),
                  "--output-root", str(tmp_path / "out"), "--cameras", cam, "--top-k", "8",
                  "--fail-on-truncation", "--device", "cpu"])
    assert not list((tmp_path / "out").rglob("*.parquet"))
