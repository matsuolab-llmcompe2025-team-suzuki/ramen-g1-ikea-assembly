"""check_unified_cache.py の unit test (Issue #146)。

小さな合成データ (v0 の jpg と YOLO の parquet) に本物の焼き込み script を CPU で流し、
- aug: そのままの出力は全項目 OK、枠を消す / ずらす / variant を消す / wrist を幾何変換する と該当項目が NG。
  GPU worker で焼いた出力は --device cuda で OK、CPU の基準では NG (CUDA がある環境のみ)
- yolo: 合成した検出の数字が手計算と一致し、一覧画像が出る
- roundtrip: smoke の範囲の index の選び方
を確かめる。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CONFIG = ROOT / "configs" / "unified_frame_cache.yaml"
HEAD = "observation.images.cam_0"
WRISTS = ("observation.images.cam_2", "observation.images.cam_3")
MP4 = "chunk-000/file-000"
MP4_2 = "chunk-000/file-001"
N_FRAMES = 3
NUM_VARIANTS = 4
SKILL = "flip_table"  # config で kinds = [overlay, common]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check = _load("check_unified_cache", ROOT / "scripts" / "check_unified_cache.py")
yolo = _load("precompute_yolo_obb", REPO / "data" / "yolo_obb" / "scripts" / "precompute_yolo_obb.py")
bake = check._load_bake_module()
CFG = check.pipe.load_config(CONFIG)
OBB_HASH = check.pipe.yolo_cache_hash(CFG)
TOP_K = CFG["yolo"]["top_k"]
BOXES = [  # (verts, conf, class_id)。0.2 は overlay の閾値 0.30 未満で描かれない
    ([0.15, 0.15, 0.55, 0.15, 0.55, 0.50, 0.15, 0.50], 0.9, 3),
    ([0.40, 0.30, 0.85, 0.30, 0.85, 0.80, 0.40, 0.80], 0.8, 4),
    ([0.20, 0.55, 0.45, 0.55, 0.45, 0.85, 0.20, 0.85], 0.2, 1),
]


@pytest.fixture(autouse=True)
def _restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def _write_v0(v0: Path, cam: str, seed: int, mp4: str = MP4, n: int = N_FRAMES) -> None:
    d = v0 / cam / mp4
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, size=(72, 96, 3), dtype=np.uint8)).save(
            d / f"frame_{i:06d}.jpg", quality=90
        )
    (d / "_frame_count.txt").write_text(str(n), encoding="utf-8")


def _write_obb(obb_root: Path, frames: list[list[tuple[list[float], float, int]]], mp4: str = MP4) -> None:
    cache = obb_root / OBB_HASH
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "manifest.json").write_text(json.dumps({
        "schema_version": yolo.CACHE_SCHEMA_VERSION, "yolo_ckpt_hash": OBB_HASH, "yolo_ckpt_ref": "test",
        "yolo_class_names": {str(i): f"c{i}" for i in range(7)}, "cameras": [HEAD], "top_k": TOP_K,
        "imgsz": 800, "conf_threshold": 0.15, "num_classes": 7, "generated_at": "2026-09-19T00:00:00Z",
    }), encoding="utf-8")
    yolo.write_mp4_parquet(
        cache / HEAD / f"{mp4}.parquet", [yolo.pad_dets_to_top_k(f, TOP_K) for f in frames],
        yolo.make_parquet_schema(TOP_K),
    )


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["bake"]["num_variants"] = NUM_VARIANTS
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _bake_work(tmp_path: Path, cfg_path: Path, monkeypatch, cuda_workers: int) -> Path:
    """pipeline の作業 dir と同じ形 (v0 / obb_yolo / v2) に、本物の焼き込みを流したもの (1 worker、cuda_workers=1 で GPU)。"""
    w = tmp_path / SKILL
    v0 = w / "v0"
    for mp4 in (MP4, MP4_2):  # 2 本目は標本に当たらない dir (全 dir の確認が効くか見るため)
        _write_v0(v0, HEAD, seed=0, mp4=mp4)
        for k, cam in enumerate(WRISTS):
            _write_v0(v0, cam, seed=1 + k, mp4=mp4)
        _write_obb(w / "obb_yolo", [BOXES] * N_FRAMES, mp4=mp4)
    (v0 / "_cache_meta.json").write_text(json.dumps({"fps": 30.0}), encoding="utf-8")
    b = CFG["bake"]
    monkeypatch.setattr(sys, "argv", [
        "bake", "--input-cache-root", str(v0),
        "--output-overlay-root", str(w / "v2" / "overlay"), "--output-common-root", str(w / "v2" / "common"),
        "--aug-config-yaml", str(cfg_path), "--obb-root", str(w / "obb_yolo"), "--obb-hash", OBB_HASH,
        "--obb-cameras", HEAD, "--obb-top-k", str(TOP_K), "--num-variants", str(NUM_VARIANTS), "--jobs", "1",
        "--cuda-workers", str(cuda_workers),
        "--overlay-conf", str(b["overlay_conf"]), "--overlay-thickness", str(b["overlay_thickness"]),
        "--gpu-jpeg-encode",  # CPU でも 4:4:4 で保存 (本番の cache と同じ色差)
    ])
    assert bake.main() == 0
    return w


@pytest.fixture
def work(tmp_path: Path, cfg_path: Path, monkeypatch) -> Path:
    return _bake_work(tmp_path, cfg_path, monkeypatch, cuda_workers=0)


def _run_aug(cfg_path: Path, work: Path, out: Path, device: str = "cpu",
             frames: int = N_FRAMES, wrist_frames: int = N_FRAMES) -> tuple[int, dict]:
    rc = check.main(["aug", "--config", str(cfg_path), "--skill", f"{SKILL}={work}", "--out", str(out),
                     "--frames", str(frames), "--wrist-frames", str(wrist_frames), "--device", device])
    report = json.loads((out / "aug.json").read_text(encoding="utf-8"))[SKILL]["verdict"]
    return rc, {k: v["ok"] for k, v in report.items()}


def _aug_pipe():
    from model.ramen_ori.image_aug import ImageAugPipeline

    return ImageAugPipeline(CFG["augmentation"])


def _replace(path: Path, tensor: torch.Tensor) -> None:
    path.unlink()
    bake._save_tensor_as_jpg(tensor, path, 90, gpu_encode=True)


def _idx(cam: str, i: int, v: int) -> str:
    return f"{cam}:chunk-000:file-000:{i:06d}:v{v}"


def test_aug_passes_on_real_bake_output(work: Path, cfg_path: Path, tmp_path: Path) -> None:
    rc, ok = _run_aug(cfg_path, work, tmp_path / "out")
    assert rc == 0
    assert all(ok.values()), ok
    assert "token と基準の差" not in ok  # flip_table は token を焼かない
    for sheet in ("head", "wrist"):
        assert (tmp_path / "out" / f"aug_{SKILL}_{sheet}.jpg").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が無い")
def test_aug_reference_must_use_bake_device(tmp_path: Path, cfg_path: Path, monkeypatch) -> None:
    """GPU worker の焼き込みは --device cuda の基準と一致し、CPU の基準とはずれる (ノイズ以降の乱数が device で変わる)。"""
    w = _bake_work(tmp_path, cfg_path, monkeypatch, cuda_workers=1)
    rc, ok = _run_aug(cfg_path, w, tmp_path / "cuda", device="cuda")
    assert rc == 0
    assert all(ok.values()), ok
    rc, ok = _run_aug(cfg_path, w, tmp_path / "cpu", device="cpu")
    assert rc == 1
    assert ok["枠が検出どおり"]  # 枠の位置 (幾何変換) は numpy の乱数なので device によらない


def test_aug_fails_when_nothing_was_checked(tmp_path: Path, cfg_path: Path) -> None:
    """v0 も焼いた jpg も無い dir (--skill の指定ミスなど) を「合格」にしない。"""
    w = tmp_path / SKILL
    _write_obb(w / "obb_yolo", [BOXES] * N_FRAMES)
    rc, ok = _run_aug(cfg_path, w, tmp_path / "out")
    assert rc == 1
    assert not ok["確認できた枚数"]


def test_aug_detects_dir_with_missing_jpgs(work: Path, cfg_path: Path, tmp_path: Path) -> None:
    """標本に当たらない mp4 dir で jpg が欠けていても、全 dir の枚数の突き合わせで気づく。"""
    for jpg in sorted((work / "v2" / "common" / WRISTS[1] / MP4_2).glob("frame_*_v*.jpg"))[:2]:
        jpg.unlink()
    rc, ok = _run_aug(cfg_path, work, tmp_path / "out", frames=1, wrist_frames=1)
    assert rc == 1
    assert not ok["jpg がそろう"]
    report = json.loads((tmp_path / "out" / "aug.json").read_text(encoding="utf-8"))[SKILL]
    assert report["missing"] == []  # 標本では踏んでいない
    assert len(report["bad_dirs"]) == 1


def test_aug_detects_dir_without_done_marker(work: Path, cfg_path: Path, tmp_path: Path) -> None:
    """標本に当たらない mp4 dir の完了マーカーが無ければ NG。"""
    (work / "v2" / "overlay" / HEAD / MP4_2 / bake.DONE_MARKER_NAME).unlink()
    rc, ok = _run_aug(cfg_path, work, tmp_path / "out", frames=1, wrist_frames=1)
    assert rc == 1
    assert not ok["完了マーカー"]


def test_aug_rejects_mixed_cpu_gpu_bake(work: Path, cfg_path: Path, tmp_path: Path) -> None:
    """CPU / GPU 混成で焼いた cache は基準を 1 つの device で作れないので、判定せず止まる。"""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["bake"]["jobs"], cfg["bake"]["cuda_workers"] = 4, 2
    mixed = tmp_path / "mixed.yaml"
    mixed.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(SystemExit, match="混成"):
        _run_aug(mixed, work, tmp_path / "out")


def test_aug_detects_boxes_outside_image() -> None:
    """検出はあるのに線を測れた枠が少ない (画像外・低コントラスト) 場合も数字で出す。"""
    r = {"kinds": ["overlay", "common"], "missing": [], "bad_sampling": 0, "bad_size": 0,
         "boxes_drawable": 10, "boxes_visible": 2, "presence": [1.0, 1.0], "spurious": [0.0],
         "head_mean_diff": [0.01], "token_mean_diff": [], "wrist_mean_diff": [0.01],
         "bad_done_marker": [], "dirs": 3, "bad_dirs": []}
    args = argparse.Namespace(presence=0.8, visible_ratio=0.8, spurious_rate=0.002, max_mean_diff=0.03)
    verdict = check.aug_verdict(r, args)
    assert not verdict["枠が画像内にある"][0]
    assert verdict["枠が検出どおり"][0]


def test_aug_detects_missing_lines(work: Path, cfg_path: Path, tmp_path: Path) -> None:
    """枠を描かずに色の変化と幾何変換だけを掛けた画像に差し替えると、枠が検出どおりでなくなる。"""
    pipe_aug = _aug_pipe()
    v0 = bake._load_v0_as_tensor(work / "v0" / HEAD / MP4 / "frame_000000.jpg")[0]
    idx = _idx(HEAD, 0, 1)
    no_lines = pipe_aug.apply_geometric(pipe_aug.photometric(v0, HEAD, idx), pipe_aug.geometric_params(HEAD, idx))
    _replace(work / "v2" / "overlay" / HEAD / MP4 / "frame_000000_v1.jpg", no_lines)
    rc, ok = _run_aug(cfg_path, work, tmp_path / "out")
    assert rc == 1
    assert not ok["枠が検出どおり"]
    assert ok["jpg がそろう"]


def test_aug_detects_shifted_overlay(work: Path, cfg_path: Path, tmp_path: Path) -> None:
    """焼いた画像が 3 px ずれていると (幾何変換の食い違い)、枠の外に差が出る。"""
    path = work / "v2" / "overlay" / HEAD / MP4 / "frame_000001_v2.jpg"
    img = check._load_jpg(path)[0]
    _replace(path, torch.roll(img, shifts=3, dims=2))
    rc, ok = _run_aug(cfg_path, work, tmp_path / "out", frames=2 * N_FRAMES)
    assert rc == 1
    assert not ok["枠の外に余計な差なし"]


def test_aug_detects_missing_variant(work: Path, cfg_path: Path, tmp_path: Path) -> None:
    (work / "v2" / "common" / WRISTS[0] / MP4 / "frame_000002_v3.jpg").unlink()
    rc, ok = _run_aug(cfg_path, work, tmp_path / "out")
    assert rc == 1
    assert not ok["jpg がそろう"]


def test_aug_detects_geometric_on_wrist(work: Path, cfg_path: Path, tmp_path: Path) -> None:
    """wrist に head の幾何変換を掛けた画像に差し替えると、wrist の基準との差が大きくなる。"""
    pipe_aug = _aug_pipe()
    cam = WRISTS[1]
    v0 = bake._load_v0_as_tensor(work / "v0" / cam / MP4 / "frame_000000.jpg")[0]
    idx = _idx(cam, 0, 1)
    warped = pipe_aug.apply_geometric(pipe_aug.photometric(v0, cam, idx), pipe_aug.geometric_params(HEAD, idx))
    _replace(work / "v2" / "common" / cam / MP4 / "frame_000000_v1.jpg", warped)
    rc, ok = _run_aug(cfg_path, work, tmp_path / "out")
    assert rc == 1
    assert not ok["wrist は幾何変換なし"]


def test_yolo_stats_match_hand_count(tmp_path: Path) -> None:
    """file-000 の 4 frame に 2 件ずつ (conf 0.9 の class 3、0.2 の class 4)、file-001 の 2 frame は検出なし。"""
    w = tmp_path / SKILL
    frames = [[(BOXES[0][0], 0.9, 3), (BOXES[1][0], 0.2, 4)]] * 4
    _write_obb(w / "obb_yolo", frames)
    _write_obb(w / "obb_yolo", [[]] * 2, mp4="chunk-000/file-001")
    _write_v0(w / "v0", HEAD, seed=0, n=4)
    out = tmp_path / "out"
    assert check.main(["yolo", "--config", str(CONFIG), "--skill", f"{SKILL}={w}", "--out", str(out),
                       "--samples", "3"]) == 0
    s = json.loads((out / "yolo.json").read_text(encoding="utf-8"))[SKILL]
    assert s["frames"] == 6
    assert s["dets_per_frame_mean"] == pytest.approx(8 / 6)
    assert (s["dets_per_frame_max"], s["frames_without_det"], s["top_k"]) == (2, 2, TOP_K)
    c3, c4 = s["per_class"]["c3"], s["per_class"]["c4"]
    assert (c3["frame_rate"], c3["per_frame"]) == (pytest.approx(4 / 6), pytest.approx(4 / 6))
    assert (c4["frame_rate"], c4["per_frame"]) == (0.0, 0.0)  # conf 0.2 は overlay の閾値未満
    assert c4["conf_p50"] == pytest.approx(0.2)
    assert s["per_class"]["c0"]["conf_p50"] is None
    assert (out / f"yolo_{SKILL}.jpg").is_file()
    assert "| c3 |" in (out / "yolo.md").read_text(encoding="utf-8")


def _write_episodes(src: Path, eps: list[dict]) -> None:
    d = src / "meta" / "episodes" / "chunk-000"
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(eps), d / "file-000.parquet")


def _ep(i: int, start: int, pos: dict[str, tuple[int, int]], length: int = 100) -> dict:
    ep = {"episode_index": i, "length": length, "dataset_from_index": start, "dataset_to_index": start + length}
    for cam, (file_index, frame) in pos.items():
        ep[f"videos/{cam}/chunk_index"] = 0
        ep[f"videos/{cam}/file_index"] = file_index
        ep[f"videos/{cam}/from_timestamp"] = frame / 30
        ep[f"videos/{cam}/to_timestamp"] = (frame + length) / 30
    return ep


def test_eligible_indices_follow_smoke_limits(tmp_path: Path) -> None:
    """smoke は mp4 ごとに先頭 max_frame だけ焼くので、全 cam でその範囲に入る frame だけを選ぶ。"""
    cams = [HEAD, WRISTS[0]]
    _write_episodes(tmp_path, [
        _ep(0, 0, {HEAD: (0, 0), WRISTS[0]: (0, 0)}),
        _ep(1, 100, {HEAD: (0, 100), WRISTS[0]: (1, 0)}),  # head は file の 100 frame 目から、wrist は次の file
    ])
    got = check.eligible_indices(tmp_path, cams, 30.0, max_frame=150, max_files=None)
    assert got == list(range(0, 100)) + list(range(100, 150))
    assert check.eligible_indices(tmp_path, cams, 30.0, max_frame=150, max_files=1) == list(range(0, 100))
    assert check.eligible_indices(tmp_path, cams, 30.0, max_frame=None, max_files=None) == list(range(200))


def test_roundtrip_uses_training_data_config_and_checks_skill_id(tmp_path: Path, monkeypatch) -> None:
    """学習の data config の共通設定のまま sub だけ差し替えて instantiate し、skill_id を確かめる
    (dataset の組み立ては偽物。hydra は RAMEN-Ori env にだけあるので、main env では skip)。"""
    hydra = pytest.importorskip("hydra")
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    src = tmp_path / "src"
    _write_episodes(src, [_ep(0, 0, {HEAD: (0, 0), WRISTS[0]: (0, 0), WRISTS[1]: (0, 0)}, length=10)])
    (src / "meta" / "info.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
    overlay = tmp_path / "restore" / "overlay"
    data_cfg = REPO / "model" / "ramen_ori" / "configs" / "data" / "unified_beta.yaml"
    argv = ["roundtrip", "--data-config", str(data_cfg), "--sub", f"{SKILL}={src},{overlay}", "--max-frame", "5"]
    captured: dict = {}

    def fake_instantiate(skill_id: int):
        class Sub:
            def __getitem__(self, i: int) -> dict:
                captured.setdefault("indices", []).append(i)
                return {"skill_id": torch.tensor(skill_id), "images": torch.zeros(3, 3, 4, 4)}

        def instantiate(cfg, **kw):
            captured["cfg"], captured["kw"] = cfg, kw
            return SimpleNamespace(subs=[Sub()])

        return instantiate

    monkeypatch.setattr(hydra.utils, "instantiate", fake_instantiate(1))  # flip_table = 1
    assert check.main(argv) == 0
    cfg = captured["cfg"]
    assert (cfg.chunk_len, cfg.memory, cfg.frame_cache_mode) == (32, False, "baked_overlay")
    assert OmegaConf.to_container(cfg.sub_datasets) == [
        {"merged_source_root": str(src), "repo_id": f"local/{SKILL}", "frame_cache_root_baked": str(overlay)}
    ]
    assert captured["kw"] == {"augmentation_cfg": None}
    assert set(captured["indices"]) <= set(range(5))  # smoke の範囲 (先頭 5 frame) だけを読む

    monkeypatch.setattr(hydra.utils, "instantiate", fake_instantiate(3))
    assert check.main(argv) == 1


def test_merge_picks_cover_chunk_edges_with_cumulative_numbering() -> None:
    picks = check.merge_picks([50, 40], 3)
    assert {(0, 0, 0), (49, 0, 49), (50, 1, 0), (89, 1, 39)} <= set(picks)
    assert all(e == (0 if ci == 0 else 50) + ce for e, ci, ce in picks)  # merge 後の番号 = chunk の累積 + chunk 内
    for ci in (0, 1):
        assert 2 <= sum(p[1] == ci for p in picks) <= 5
    assert picks == check.merge_picks([50, 40], 3)  # 同じ seed なら同じ
    assert check.merge_picks([1], 3) == [(0, 0, 0)]


def test_pad_tasks_lets_iloc_reach_original_task_index() -> None:
    import pandas as pd
    from types import SimpleNamespace

    ds = SimpleNamespace(meta=SimpleNamespace(
        tasks=pd.DataFrame({"task_index": [3]}, index=pd.Index(["rotate leg to tighten"], name="task"))
    ))
    check._pad_tasks(ds)
    assert len(ds.meta.tasks) == 4
    assert ds.meta.tasks.iloc[3].name == "rotate leg to tighten"


def test_named_args_require_name_and_value() -> None:
    with pytest.raises(SystemExit):
        check._parse_named(["flip_table"], "--skill")
    assert check._parse_named(["a=b=c"], "--x") == {"a": "b=c"}
