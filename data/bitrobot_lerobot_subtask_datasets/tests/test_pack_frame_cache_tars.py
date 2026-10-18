"""pack_frame_cache_tars.py + restore_frame_cache_tars.sh の unit test (Issue #139)。

- v0 / variants の tar 化、欠け / 0 byte の検出、watch mode の終了条件
- restore.sh で展開した v2 cache を、学習側 (lerobot_frame_cache_patch) の path 解決で
  学習 view 名 (head_left 等) と source 名 (cam_0 等) の両方から読めること
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SCRIPTS = ROOT / "scripts"
RESTORE = SCRIPTS / "restore_frame_cache_tars.sh"
HEAD = "observation.images.cam_0"
WRISTS = ["observation.images.cam_2", "observation.images.cam_3"]
MP4 = Path("chunk-000") / "file-000"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pack_frame_cache_tars", SCRIPTS / "pack_frame_cache_tars.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pack = _load()


def _make_v0(root: Path, cam: str, n: int = 3) -> Path:
    d = root / cam / MP4
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"frame_{i:06d}.jpg").write_bytes(b"jpg-v0-%d" % i)
    (d / "_frame_count.txt").write_text(str(n), encoding="utf-8")
    return d


def _make_variants(root: Path, cam: str, n: int = 3, num_variants: int = 3, done: bool = True) -> Path:
    d = root / cam / MP4
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        for v in range(1, num_variants):
            (d / f"frame_{i:06d}_v{v}.jpg").write_bytes(b"jpg-%s-%d-%d" % (cam.encode(), i, v))
    if done:
        (d / "_bake_done.json").write_text(
            json.dumps({"frames": n, "num_variants": num_variants}), encoding="utf-8"
        )
    return d


def _run(**kw) -> int:
    defaults = dict(
        cameras=None, expect_from=None, watch=False, done_flag=None, jobs=1, poll_interval=0.01
    )
    defaults.update(kw)
    return pack.run(**defaults)


# ---------------------------------------------------------------------------
# tar 化
# ---------------------------------------------------------------------------


def test_pack_v0_members_and_name(tmp_path: Path) -> None:
    _make_v0(tmp_path / "v0", HEAD)
    assert _run(src_root=tmp_path / "v0", dest_dir=tmp_path / "out", mode="v0") == 0
    tar_path = tmp_path / "out" / f"{HEAD}_chunk-000_file-000.tar"
    with tarfile.open(tar_path) as tf:
        names = sorted(tf.getnames())
    assert names == sorted(
        [f"file-000/frame_{i:06d}.jpg" for i in range(3)] + ["file-000/_frame_count.txt"]
    )
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_pack_skips_existing_tar(tmp_path: Path) -> None:
    _make_v0(tmp_path / "v0", HEAD)
    dest = tmp_path / "out"
    dest.mkdir()
    existing = dest / f"{HEAD}_chunk-000_file-000.tar"
    existing.write_bytes(b"already")
    assert _run(src_root=tmp_path / "v0", dest_dir=dest, mode="v0") == 0
    assert existing.read_bytes() == b"already"


def test_pack_rejects_zero_byte_jpg(tmp_path: Path) -> None:
    d = _make_v0(tmp_path / "v0", HEAD)
    (d / "frame_000001.jpg").write_bytes(b"")
    with pytest.raises(pack.PackError, match="0 byte"):
        _run(src_root=tmp_path / "v0", dest_dir=tmp_path / "out", mode="v0")
    assert not list((tmp_path / "out").glob("*"))  # 半端な tar を残さない


def test_pack_rejects_missing_variant(tmp_path: Path) -> None:
    d = _make_variants(tmp_path / "overlay", HEAD)
    (d / "frame_000002_v2.jpg").unlink()
    with pytest.raises(pack.PackError, match="missing"):
        _run(src_root=tmp_path / "overlay", dest_dir=tmp_path / "out", mode="variants")


def test_pack_variants_skips_symlinked_cam(tmp_path: Path) -> None:
    common = tmp_path / "common"
    overlay = tmp_path / "overlay"
    _make_variants(common, WRISTS[0])
    _make_variants(overlay, HEAD)
    (overlay / WRISTS[0]).symlink_to(Path("..") / "common" / WRISTS[0], target_is_directory=True)
    assert _run(src_root=overlay, dest_dir=tmp_path / "out", mode="variants") == 0
    assert [p.name for p in (tmp_path / "out").glob("*.tar")] == [f"{HEAD}_chunk-000_file-000.tar"]


def test_watch_waits_for_marker_then_fails_if_bake_finished_without_it(tmp_path: Path) -> None:
    v0 = tmp_path / "v0"
    _make_v0(v0, HEAD)
    overlay = tmp_path / "overlay"
    _make_variants(overlay, HEAD, done=False)  # marker 無し
    flag = tmp_path / ".bake.done"
    flag.touch()
    rc = _run(
        src_root=overlay, dest_dir=tmp_path / "out", mode="variants",
        cameras=[HEAD], expect_from=v0, watch=True, done_flag=flag,
    )
    assert rc == 3


def test_watch_packs_expected_dirs_then_exits(tmp_path: Path) -> None:
    v0 = tmp_path / "v0"
    _make_v0(v0, HEAD)
    overlay = tmp_path / "overlay"
    _make_variants(overlay, HEAD)
    flag = tmp_path / ".bake.done"
    flag.touch()
    rc = _run(
        src_root=overlay, dest_dir=tmp_path / "out", mode="variants",
        cameras=[HEAD], expect_from=v0, watch=True, done_flag=flag,
    )
    assert rc == 0
    assert (tmp_path / "out" / f"{HEAD}_chunk-000_file-000.tar").is_file()


# ---------------------------------------------------------------------------
# restore.sh と学習側の読み込み
# ---------------------------------------------------------------------------


def _stage_restore(tars_dir: Path) -> Path:
    shutil.copy2(RESTORE, tars_dir / "restore.sh")
    (tars_dir / "_cache_meta.json").write_text(json.dumps({"fps": 30.0}), encoding="utf-8")
    return tars_dir / "restore.sh"


def test_restore_flat_layout(tmp_path: Path) -> None:
    _make_v0(tmp_path / "v0", HEAD)
    tars = tmp_path / "frame_cache_tars"
    assert _run(src_root=tmp_path / "v0", dest_dir=tars, mode="v0") == 0
    script = _stage_restore(tars)
    out = tmp_path / "restored"
    subprocess.run(["bash", str(script), str(out), "2"], check=True, capture_output=True)
    assert (out / HEAD / MP4 / "frame_000002.jpg").read_bytes() == b"jpg-v0-2"
    assert (out / "_cache_meta.json").is_file()


def _build_v2_tars(tmp_path: Path) -> Path:
    v2 = tmp_path / "v2"
    for kind in ("overlay", "token"):
        _make_variants(v2 / kind, HEAD)
    for cam in WRISTS:
        _make_variants(v2 / "common", cam)
    tars = tmp_path / "frame_cache_v2_tars"
    for kind in ("overlay", "token", "common"):
        assert _run(src_root=v2 / kind, dest_dir=tars / kind, mode="variants") == 0
    _stage_restore(tars)
    return tars


def test_restore_v2_layout_links_common_and_view_aliases(tmp_path: Path) -> None:
    tars = _build_v2_tars(tmp_path)
    out = tmp_path / "restored"
    subprocess.run(["bash", str(tars / "restore.sh"), str(out), "3"], check=True, capture_output=True)
    for kind in ("overlay", "token"):
        root = out / kind
        assert (root / "_cache_meta.json").is_file()
        for cam in WRISTS:
            assert (root / cam).is_symlink()
            assert (root / cam / MP4 / "frame_000000_v1.jpg").read_bytes() == b"jpg-%s-0-1" % cam.encode()
        # 学習 view 名の別名
        assert (root / "observation.images.head_left" / MP4 / "frame_000001_v2.jpg").is_file()
        assert (root / "observation.images.left_wrist" / MP4 / "frame_000001_v2.jpg").is_file()
        assert (root / "observation.images.right_wrist" / MP4 / "frame_000001_v2.jpg").is_file()
    # 2 回目の restore でも壊れない (symlink 済みを飛ばす)
    subprocess.run(["bash", str(tars / "restore.sh"), str(out), "3"], check=True, capture_output=True)


@pytest.mark.parametrize(
    "video_cam",
    [
        "observation.images.head_left",   # GR00T / ACT / DP の学習 view
        "observation.images.cam_0",       # RAMEN-Ori (source 名)
        "observation.images.left_wrist",
        "observation.images.cam_3",
    ],
)
def test_restored_overlay_root_resolves_like_training_loader(
    tmp_path: Path, monkeypatch, video_cam: str
) -> None:
    """lerobot_frame_cache_patch と同じ path 解決で、どちらの cam 名からも jpg に届く。"""
    from model.subtask_policy_training.scripts import lerobot_frame_cache_patch as patch

    tars = _build_v2_tars(tmp_path)
    out = tmp_path / "restored"
    subprocess.run(["bash", str(tars / "restore.sh"), str(out)], check=True, capture_output=True)

    monkeypatch.setenv("FRAME_CACHE_ROOT_OVERRIDE", str(out / "overlay"))
    video_path = tmp_path / "view" / "videos" / video_cam / "chunk-000" / "file-000.mp4"
    cache_dir = patch._cache_dir_for_video_path(video_path)
    assert cache_dir is not None
    assert patch._cache_fps(patch._resolve_cache_root(cache_dir)) == 30.0
    for _ in range(5):
        jpg = patch._resolve_variant_jpg_path(cache_dir, 2, 3)
        assert jpg.is_file()
