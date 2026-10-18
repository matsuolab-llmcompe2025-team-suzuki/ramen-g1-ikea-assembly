"""YOLO の重みの取り方 (resolve_yolo_ckpt_ref) の test (B7-01)。

重みそのものも ultralytics も要らないので、実 weight が無いと skip する test_yolo_obb.py とは分ける
(そちらに置くと、重みの無い環境ではこの test が一度も走らない)。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

#: HF の実物と同じ入れ子の置き場所と、commit hash の revision (B7-01)。
_REPO = "owner/repo"
_SHA = "8221d0aef45d63666537c1f030810fb9b56dd1ed"
_FILE = "runs/m_lowaug_v11b/weights/best_20260818.pt"


def test_resolve_yolo_ckpt_ref_downloads_only_the_pinned_file(monkeypatch) -> None:
    """repo の一覧を問い合わせず、決め打ちの path を 1 回取るだけ。"""
    from inference.desktop.perception.yolo_obb import resolve_yolo_ckpt_ref

    calls = []
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = lambda repo_id, filename, revision=None: (
        calls.append((repo_id, filename, revision)) or f"/cache/{filename}"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    assert resolve_yolo_ckpt_ref(f"{_REPO}@{_SHA}", _FILE) == Path(f"/cache/{_FILE}")
    assert calls == [(_REPO, _FILE, _SHA)]


def test_resolve_yolo_ckpt_ref_reads_a_cached_weight_without_the_network(
    monkeypatch, tmp_path: Path
) -> None:
    """会場 (実行時オフライン): cache に入れ子で置かれた重みを、通信せずにそのまま読む。"""
    pytest.importorskip("huggingface_hub")
    import socket

    from huggingface_hub import constants

    from inference.desktop.perception.yolo_obb import resolve_yolo_ckpt_ref

    weight = tmp_path / "models--owner--repo" / "snapshots" / _SHA / _FILE
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weights")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    def _no_network(*_args, **_kwargs):
        raise AssertionError("resolve_yolo_ckpt_ref must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)

    assert resolve_yolo_ckpt_ref(f"{_REPO}@{_SHA}", _FILE) == weight


def test_resolve_yolo_ckpt_ref_rejects_a_ref_without_revision() -> None:
    from inference.desktop.perception.yolo_obb import resolve_yolo_ckpt_ref

    with pytest.raises(ValueError, match="repo@revision"):
        resolve_yolo_ckpt_ref(_REPO, _FILE)
