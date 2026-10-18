"""data/yolo_obb/scripts/build_dataset.py の HF からの取得の test (Issue #196)。

HF には繋がない。`_dl` (1 file の取得) と `HfApi` を差し替えて、並列に取っても
結果の順番・失敗の扱いが 1 つずつ取っていたときと同じかを確かめる。
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]  # iros_2026_ramen/
SCRIPT_PATH = ROOT / "data" / "yolo_obb" / "scripts" / "build_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_dataset", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _CountingDownload:
    """同時に走っている取得の数の最大を数える偽の `_dl`。"""

    def __init__(self, fail: set[str] = frozenset()) -> None:
        self.fail = fail
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(
        self, repo: str, filename: str, cache: Path, token: str | None
    ) -> Path:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            # 後ろの file ほど早く終わらせ、完了の順と入力の順をずらす
            time.sleep(0.02 if filename.endswith("_00.png") else 0.005)
            if filename in self.fail:
                raise OSError(f"404 {filename}")
            return cache / filename
        finally:
            with self.lock:
                self.active -= 1


def test_pull_images_keeps_input_order_and_drops_only_failures(
    monkeypatch, tmp_path, capsys
) -> None:
    mod = load_module()
    names = [f"frames/leg/ep0001/frame_{i:02d}.png" for i in range(30)]
    fake = _CountingDownload(fail={names[3]})
    monkeypatch.setattr(mod, "_dl", fake)
    entries = [{"img_hf_path": n, "frame": i} for i, n in enumerate(names)]

    ok = mod.pull_images(entries, tmp_path, token=None)

    assert [e["frame"] for e in ok] == [i for i in range(30) if i != 3]
    assert all(e["img_path"] == tmp_path / "frames" / e["img_hf_path"] for e in ok)
    assert fake.max_active > 1  # 本当に並列に取っている
    assert "1 image DL failed" in capsys.readouterr().err


def test_pull_annotations_keeps_sorted_order_and_skips_unexpected_paths(
    monkeypatch, tmp_path
) -> None:
    mod = load_module()
    files = [
        "yolo_obb/labels/leg/ep0002/frame_00010.txt",
        "yolo_obb/labels/head_left/frame_00005.txt",  # 想定外の形 (階層が足りない) は外す
        "yolo_obb/labels/hole/ep0001/frame_00007.txt",
        "frames/leg/ep0002/frame_00010.png",  # label でない物は数えない
    ]

    class _FakeApi:
        def __init__(self, token=None) -> None:
            pass

        def list_repo_files(self, repo_id: str, repo_type: str) -> list[str]:
            return files

    fake = _CountingDownload()
    monkeypatch.setattr(mod, "HfApi", _FakeApi)
    monkeypatch.setattr(mod, "_dl", fake)

    result = mod.pull_annotations(tmp_path, token=None)

    assert [(e["skill"], e["ep"], e["frame"]) for e in result] == [
        ("hole", 1, 7),
        ("leg", 2, 10),
    ]
    assert (
        result[0]["txt_path"]
        == tmp_path / "annotations" / "yolo_obb/labels/hole/ep0001/frame_00007.txt"
    )
    assert result[1]["img_hf_path"] == "frames/leg/ep0002/frame_00010.png"


def test_dl_retries_a_transient_failure_and_gives_up_after_the_last_attempt(
    monkeypatch, tmp_path
) -> None:
    """並列に繋ぐと一時的な切断が起きる。1 回の失敗では画像を落とさず、試し直す。"""
    mod = load_module()
    calls: list[str] = []
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    def flaky(*, filename: str, local_dir: str, **_kwargs) -> str:
        calls.append(filename)
        if filename == "always.png" or calls.count(filename) == 1:
            raise OSError("Server disconnected without sending a response.")
        return f"{local_dir}/{filename}"

    monkeypatch.setattr(mod, "hf_hub_download", flaky)

    assert mod._dl("repo", "once.png", tmp_path, None) == tmp_path / "once.png"
    assert calls.count("once.png") == 2
    with pytest.raises(OSError):
        mod._dl("repo", "always.png", tmp_path, None)
    assert calls.count("always.png") == mod.DOWNLOAD_ATTEMPTS
