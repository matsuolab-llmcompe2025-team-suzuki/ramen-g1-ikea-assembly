"""page_cache の unit test (Issue #188 ② 段 2)。

読み終わった重みの file を正しく見つけ、`posix_fadvise(DONTNEED)` を file ごとに 1 回呼ぶ。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference.desktop.lower_policy.policies import page_cache
from inference.desktop.lower_policy.policies.page_cache import (
    ckpt_ref_files,
    drop_page_cache,
    hf_snapshot_files,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


def _hf_repo(
    cache: Path, repo_id: str, files: dict[str, bytes], *, ref: str = "main"
) -> Path:
    """HF の cache と同じ形 (blobs + snapshots の symlink + refs) を作る。"""
    root = cache / f"models--{repo_id.replace('/', '--')}"
    snapshot = root / "snapshots" / SHA
    for index, (name, data) in enumerate(files.items()):
        blob = root / "blobs" / f"blob{index}"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / ref).write_text(SHA, encoding="utf-8")
    return root


def test_a_commit_revision_resolves_to_the_blobs_behind_the_snapshot(tmp_path):
    root = _hf_repo(
        tmp_path,
        "Team-RAMEN/pick",
        {"model.safetensors": b"x" * 10, "sub/config.json": b"{}"},
    )
    files = hf_snapshot_files("Team-RAMEN/pick", SHA, cache_dir=tmp_path)
    assert sorted(path.name for path in files) == ["blob0", "blob1"]
    assert all(path.parent == (root / "blobs").resolve() for path in files)


def test_a_branch_or_no_revision_reads_the_ref(tmp_path):
    _hf_repo(tmp_path, "Qwen/Qwen3-VL-8B-Instruct", {"a.safetensors": b"a"})
    assert len(hf_snapshot_files("Qwen/Qwen3-VL-8B-Instruct", cache_dir=tmp_path)) == 1
    assert (
        hf_snapshot_files("Qwen/Qwen3-VL-8B-Instruct", "dev", cache_dir=tmp_path) == []
    )


def test_a_repo_not_in_the_cache_has_no_files(tmp_path):
    assert hf_snapshot_files("nobody/none", SHA, cache_dir=tmp_path) == []
    assert ckpt_ref_files(f"nobody/none@{SHA}", cache_dir=tmp_path) == []


def test_a_ckpt_ref_is_a_repo_at_revision_or_a_local_path(tmp_path):
    _hf_repo(tmp_path / "hub", "Team-RAMEN/rotate", {"ckpt.pt": b"w"})
    assert [
        p.name
        for p in ckpt_ref_files(f"Team-RAMEN/rotate@{SHA}", cache_dir=tmp_path / "hub")
    ] == ["blob0"]
    local = tmp_path / "local"
    (local / "step").mkdir(parents=True)
    (local / "step" / "ckpt.pt").write_bytes(b"w")
    assert [p.name for p in ckpt_ref_files(str(local))] == ["ckpt.pt"]
    assert [p.name for p in ckpt_ref_files(str(local / "step" / "ckpt.pt"))] == [
        "ckpt.pt"
    ]


def test_every_file_is_advised_once_with_dontneed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        os,
        "posix_fadvise",
        lambda fd, offset, length, advice: calls.append((offset, length, advice)),
        raising=False,
    )
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    first = tmp_path / "a.bin"
    first.write_bytes(b"1234")
    second = tmp_path / "b.bin"
    second.write_bytes(b"12")
    count, size = drop_page_cache([first, second, first, tmp_path / "missing.bin"])
    assert (count, size) == (2, 6)
    assert calls == [(0, 0, 4), (0, 0, 4)]


def test_an_os_without_fadvise_does_nothing(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "posix_fadvise", raising=False)
    path = tmp_path / "a.bin"
    path.write_bytes(b"1")
    assert drop_page_cache([path]) == (0, 0)


@pytest.mark.skipif(
    not hasattr(os, "posix_fadvise"), reason="posix_fadvise is Linux-only"
)
def test_the_real_fadvise_accepts_a_file(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"1" * 4096)
    assert drop_page_cache([path]) == (1, 4096)


def test_the_cache_dir_follows_hf_hub_cache_without_huggingface_hub(
    monkeypatch, tmp_path
):
    import builtins

    real_import = builtins.__import__

    def _no_hf(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_hf)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert page_cache.hf_hub_cache_dir() == tmp_path / "hub"
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert page_cache.hf_hub_cache_dir() == tmp_path / "home" / "hub"
