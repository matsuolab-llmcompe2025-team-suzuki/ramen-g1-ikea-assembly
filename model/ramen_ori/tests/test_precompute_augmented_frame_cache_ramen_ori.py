"""precompute_augmented_frame_cache_ramen_ori.py の unit test (Issue #139)。

小さな v0 cache (head cam_0 + wrist cam_2) と OBB cache を作って main を回し、
- token / overlay / common の出力と symlink、完了マーカー
- 焼いた jpg が `ImageAugPipeline.apply` を token / overlay で呼んだ参照とバイト一致すること
- source 名 (cam_0) の v0 dir でも OBB を引けること、外れたら止まること
を確認する。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
BAKE_SCRIPT = REPO / "model" / "ramen_ori" / "scripts" / "precompute_augmented_frame_cache_ramen_ori.py"
YOLO_SCRIPT = REPO / "data" / "yolo_obb" / "scripts" / "precompute_yolo_obb.py"
UNIFIED_CFG = REPO / "data" / "bitrobot_lerobot_subtask_datasets" / "configs" / "unified_frame_cache.yaml"
HEAD = "observation.images.cam_0"
WRIST = "observation.images.cam_2"
MP4 = "chunk-000/file-000"
N_FRAMES = 3
NUM_VARIANTS = 4
OBB_HASH = "abc123def456"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bake = _load("precompute_augmented_frame_cache_ramen_ori", BAKE_SCRIPT)
yolo = _load("precompute_yolo_obb", YOLO_SCRIPT)


@pytest.fixture(autouse=True)
def _restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def _write_v0(root: Path, cam: str, n_frames: int, seed: int) -> None:
    d = root / cam / MP4
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n_frames):
        arr = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / f"frame_{i:06d}.jpg", format="JPEG", quality=90)
    (d / "_frame_count.txt").write_text(str(n_frames), encoding="utf-8")


def _write_obb(obb_root: Path, n_frames: int, mp4: str = MP4) -> None:
    cache_dir = obb_root / OBB_HASH
    cache_dir.mkdir(parents=True)
    manifest = {
        "schema_version": yolo.CACHE_SCHEMA_VERSION,
        "yolo_ckpt_hash": OBB_HASH,
        "yolo_ckpt_ref": "file:sha256:test",
        "yolo_class_names": {str(i): f"c{i}" for i in range(7)},
        "cameras": [HEAD],
        "top_k": 4,
        "imgsz": 800,
        "conf_threshold": 0.15,
        "num_classes": 7,
        "generated_at": "2026-09-12T00:00:00Z",
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    box = [0.1, 0.1, 0.6, 0.1, 0.6, 0.6, 0.1, 0.6]
    frame_dets = [yolo.pad_dets_to_top_k([(box, 0.9, 3), (box, 0.2, 1)], 4) for _ in range(n_frames)]
    yolo.write_mp4_parquet(
        cache_dir / HEAD / f"{mp4}.parquet", frame_dets, yolo.make_parquet_schema(4)
    )


def _write_aug_yaml(path: Path) -> dict:
    aug = yaml.safe_load(UNIFIED_CFG.read_text(encoding="utf-8"))["augmentation"]
    path.write_text(yaml.safe_dump({"augmentation": aug}), encoding="utf-8")
    return aug


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    v0 = tmp_path / "v0"
    _write_v0(v0, HEAD, N_FRAMES, seed=0)
    _write_v0(v0, WRIST, N_FRAMES, seed=1)
    (v0 / "_cache_meta.json").write_text(json.dumps({"fps": 30.0}), encoding="utf-8")
    _write_obb(tmp_path / "obb", N_FRAMES)
    _write_aug_yaml(tmp_path / "aug.yaml")
    return tmp_path


def _run(monkeypatch, wd: Path, *extra: str, common: bool = True) -> int:
    argv = [
        "bake",
        "--input-cache-root", str(wd / "v0"),
        "--output-token-root", str(wd / "v2" / "token"),
        "--output-overlay-root", str(wd / "v2" / "overlay"),
        "--aug-config-yaml", str(wd / "aug.yaml"),
        "--obb-root", str(wd / "obb"),
        "--obb-cameras", HEAD,
        "--num-variants", str(NUM_VARIANTS),
        "--jobs", "1",
        *extra,
    ]
    if common:
        argv += ["--output-common-root", str(wd / "v2" / "common")]
    monkeypatch.setattr(sys, "argv", argv)
    return bake.main()


def _names() -> list[str]:
    return [f"frame_{i:06d}_v{v}.jpg" for i in range(N_FRAMES) for v in range(1, NUM_VARIANTS)]


def test_bake_writes_overlay_token_common_and_links(workdir: Path, monkeypatch) -> None:
    assert _run(monkeypatch, workdir) == 0
    v2 = workdir / "v2"
    for kind in ("overlay", "token"):
        head_dir = v2 / kind / HEAD / MP4
        assert sorted(p.name for p in head_dir.glob("*.jpg")) == sorted(_names())
        done = json.loads((head_dir / "_bake_done.json").read_text(encoding="utf-8"))
        assert done == {
            "frames": N_FRAMES,
            "num_variants": NUM_VARIANTS,
            "jpg_subsampling": "4:2:0",  # --gpu-jpeg-encode 無し = PIL の既定
        }
        # wrist は common を指す symlink
        link = v2 / kind / WRIST
        assert link.is_symlink()
        assert os.readlink(link) == os.path.join("..", "common", WRIST)
        assert (v2 / kind / "_cache_meta.json").is_file()
    common_dir = v2 / "common" / WRIST / MP4
    assert sorted(p.name for p in common_dir.glob("*.jpg")) == sorted(_names())
    assert (common_dir / "_bake_done.json").is_file()
    # head は common に書かない
    assert not (v2 / "common" / HEAD).exists()


def test_bake_outputs_match_apply_reference(workdir: Path, monkeypatch, tmp_path: Path) -> None:
    """高速経路の jpg が、apply を token / overlay で呼んだ参照とバイト一致する。"""
    from model.ramen_ori.image_aug import ImageAugPipeline
    from model.ramen_ori.obb_cache import ObbPrecomputedCache
    from model.ramen_ori.overlay import draw_overlay_on_tensor

    assert _run(monkeypatch, workdir) == 0
    aug = yaml.safe_load((workdir / "aug.yaml").read_text(encoding="utf-8"))["augmentation"]
    pipe = ImageAugPipeline(aug)
    cache = ObbPrecomputedCache(
        root=workdir / "obb", ckpt_hash=OBB_HASH, camera_keys=[HEAD], top_K=4, num_classes=7, fps=30
    )
    ref_dir = tmp_path / "ref"
    for cam in (HEAD, WRIST):
        for i in range(N_FRAMES):
            v0 = bake._load_v0_as_tensor(workdir / "v0" / cam / MP4 / f"frame_{i:06d}.jpg")[0]
            img_2 = v0.unsqueeze(0).repeat(2, 1, 1, 1)
            per_frame = cache.lookup_mp4_frame(cam, MP4, i)
            for v in range(1, NUM_VARIANTS):
                idx = f"{cam}:chunk-000:file-000:{i:06d}:v{v}"
                name = f"frame_{i:06d}_v{v}.jpg"
                if per_frame is None:
                    cb = None
                else:
                    def cb(img, _fi, pf=per_frame):
                        return draw_overlay_on_tensor(
                            img, verts=pf["verts"], conf=pf["conf"], class_id=pf["class_id"],
                            valid=pf["valid"], conf_threshold=0.30, line_thickness=2,
                        )
                token_ref = pipe.apply(img_2, cam, idx, True, skip_normalize=True, skip_geometric=True)[0][1]
                overlay_ref = pipe.apply(img_2, cam, idx, True, overlay_callback=cb, skip_normalize=True)[0][1]
                bake._save_tensor_as_jpg(token_ref, ref_dir / "token" / name, 90)
                bake._save_tensor_as_jpg(overlay_ref, ref_dir / "overlay" / name, 90)
                token_got = (workdir / "v2" / "token" / cam / MP4 / name).read_bytes()
                overlay_got = (workdir / "v2" / "overlay" / cam / MP4 / name).read_bytes()
                assert token_got == (ref_dir / "token" / name).read_bytes()
                assert overlay_got == (ref_dir / "overlay" / name).read_bytes()
                if cam == HEAD:
                    assert overlay_got != token_got  # 枠と回転・平行移動が入っている


def test_obb_frame_mismatch_skips_marker_and_fails(tmp_path: Path, monkeypatch) -> None:
    v0 = tmp_path / "v0"
    _write_v0(v0, HEAD, N_FRAMES, seed=0)
    _write_obb(tmp_path / "obb", N_FRAMES - 1)  # 最終 frame の OBB が無い
    _write_aug_yaml(tmp_path / "aug.yaml")
    assert _run(monkeypatch, tmp_path, common=False) == 4
    assert not (tmp_path / "v2" / "overlay" / HEAD / MP4 / "_bake_done.json").exists()


def test_zero_obb_hits_aborts_early(tmp_path: Path, monkeypatch) -> None:
    v0 = tmp_path / "v0"
    _write_v0(v0, HEAD, N_FRAMES, seed=0)
    _write_obb(tmp_path / "obb", N_FRAMES, mp4="chunk-000/file-009")  # 別の mp4 の OBB しか無い
    _write_aug_yaml(tmp_path / "aug.yaml")
    assert _run(monkeypatch, tmp_path, "--guard-min-frames", "1", common=False) == 3


def test_without_common_root_writes_wrist_to_both_roots(workdir: Path, monkeypatch) -> None:
    assert _run(monkeypatch, workdir, common=False) == 0
    token = workdir / "v2" / "token" / WRIST / MP4
    overlay = workdir / "v2" / "overlay" / WRIST / MP4
    assert not (workdir / "v2" / "token" / WRIST).is_symlink()
    for name in _names():
        assert (token / name).read_bytes() == (overlay / name).read_bytes()


def test_real_dir_conflicts_with_shared_link(workdir: Path, monkeypatch) -> None:
    (workdir / "v2" / "token" / WRIST).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        _run(monkeypatch, workdir)


def test_build_cam_plans_resolves_view_and_source_names() -> None:
    covered = {HEAD}
    plans = bake.build_cam_plans(
        [
            "observation.images.head_left",
            HEAD,
            "observation.images.cam_1",
            WRIST,
        ],
        camera_map=dict(bake.FALLBACK_CAMERA_MAP),
        covered_cameras=covered,
        uses_geometric=lambda cam: cam not in {WRIST},
        share_enabled=True,
    )
    assert plans["observation.images.head_left"] == bake.CamPlan(source_cam=HEAD, shared=False)
    assert plans[HEAD] == bake.CamPlan(source_cam=HEAD, shared=False)  # source 名は同名で引く
    # OBB なしでも geometric が掛かる cam は token と overlay が別物なので共有しない
    assert plans["observation.images.cam_1"] == bake.CamPlan(source_cam=None, shared=False)
    assert plans[WRIST] == bake.CamPlan(source_cam=None, shared=True)


def test_max_frames_processes_prefix_without_marker(workdir: Path, monkeypatch) -> None:
    """速度計測用 --max-frames: 途中までの mp4 dir には完了マーカーを書かない。"""
    assert _run(monkeypatch, workdir, "--max-frames", "2", common=False) == 0
    head_dir = workdir / "v2" / "overlay" / HEAD / MP4
    assert (head_dir / "frame_000001_v1.jpg").is_file()
    assert not (head_dir / "frame_000002_v1.jpg").exists()
    assert not (head_dir / "_bake_done.json").exists()


def test_no_tmp_files_left_after_bake(workdir: Path, monkeypatch) -> None:
    assert _run(monkeypatch, workdir) == 0
    assert not list((workdir / "v2").rglob("*.tmp"))


class _FakeCounter:
    """mp.Value の代わり (get_lock / value だけ持つ)。"""

    def __init__(self) -> None:
        import threading

        self.value = 0
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock


def test_assign_device_gives_first_k_workers_cuda() -> None:
    counter = _FakeCounter()
    devices = [bake._assign_device(counter, 3) for _ in range(8)]
    assert devices == ["cuda"] * 3 + ["cpu"] * 5
    assert bake._assign_device(None, 0) == "cpu"
    assert bake._assign_device(None, 2) == "cuda"


def test_cuda_workers_must_not_exceed_jobs(workdir: Path, monkeypatch) -> None:
    assert _run(monkeypatch, workdir, "--jobs", "4", "--cuda-workers", "5") == 2


def test_gpu_jpeg_encode_cache_is_444_even_when_saved_on_cpu(workdir: Path, monkeypatch) -> None:
    """--gpu-jpeg-encode の cache は、CPU worker (や nvJPEG 失敗後) が保存した jpg も 4:4:4。

    1 つの cache に色差が混ざると、推論 slot の overlay_jpeg_subsampling (1 つ) で合わせきれない。
    """
    from PIL import JpegImagePlugin

    assert _run(monkeypatch, workdir, "--gpu-jpeg-encode") == 0  # --jobs 1、CUDA worker 無し
    head_dir = workdir / "v2" / "overlay" / HEAD / MP4
    done = json.loads((head_dir / "_bake_done.json").read_text(encoding="utf-8"))
    assert done["jpg_subsampling"] == "4:4:4"
    saved = [head_dir / name for name in _names()]
    saved.append(workdir / "v2" / "common" / WRIST / MP4 / _names()[0])
    assert {JpegImagePlugin.get_sampling(Image.open(path)) for path in saved} == {0}
