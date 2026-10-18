"""Unit tests for data/yolo_obb/scripts/upload_yolo_obb_to_hf.py (Issue #122 D-1')。

HF push は integration test 領域なので、ここでは manifest 読み込み / path 解決 /
argparse / dry-run flow のみ検証 (HfApi 実 call は mock)。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]  # iros_2026_ramen/
SCRIPT_PATH = ROOT / "data" / "yolo_obb" / "scripts" / "upload_yolo_obb_to_hf.py"


def load_module():
    import sys

    spec = importlib.util.spec_from_file_location("upload_yolo_obb_to_hf", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_valid_manifest(cache_dir: Path, ckpt_hash: str = "abc123def456") -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "team_ramen_yolo_obb_cache_v1",
        "yolo_ckpt_hash": ckpt_hash,
        "yolo_ckpt_ref": f"hf:Team-RAMEN/dummy@{ckpt_hash}",
        "yolo_class_names": {"0": "workspace", "1": "leg"},
        "cameras": ["observation.images.cam_0", "observation.images.cam_1"],
        "top_k": 4,
        "imgsz": 800,
        "conf_threshold": 0.15,
        "num_classes": 7,
        "generated_at": "2026-08-18T00:00:00Z",
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))


def test_load_manifest_missing_raises(tmp_path: Path) -> None:
    m = load_module()
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        m.load_manifest(tmp_path)


def test_load_manifest_wrong_schema_raises(tmp_path: Path) -> None:
    m = load_module()
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": "wrong_v0"}))
    with pytest.raises(ValueError, match="unsupported manifest schema_version"):
        m.load_manifest(tmp_path)


def test_load_manifest_missing_required_field_raises(tmp_path: Path) -> None:
    m = load_module()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "team_ramen_yolo_obb_cache_v1",
                "yolo_ckpt_hash": "abc",
                # yolo_ckpt_ref, cameras 欠
            }
        )
    )
    with pytest.raises(ValueError, match="manifest missing required field"):
        m.load_manifest(tmp_path)


def test_resolve_path_in_repo_default_from_hash(tmp_path: Path) -> None:
    m = load_module()
    _write_valid_manifest(tmp_path, ckpt_hash="deadbeef1234")
    manifest = m.load_manifest(tmp_path)
    assert m.resolve_path_in_repo(manifest, override=None) == "obb_yolo/deadbeef1234"


def test_resolve_path_in_repo_override_wins(tmp_path: Path) -> None:
    m = load_module()
    _write_valid_manifest(tmp_path)
    manifest = m.load_manifest(tmp_path)
    assert m.resolve_path_in_repo(manifest, override="custom/dir") == "custom/dir"


def test_build_commit_message_default_contains_hash_and_ref(tmp_path: Path) -> None:
    m = load_module()
    _write_valid_manifest(tmp_path, ckpt_hash="face1234")
    manifest = m.load_manifest(tmp_path)
    msg = m.build_commit_message(manifest, override=None)
    assert "hash=face1234" in msg
    assert "ckpt=hf:Team-RAMEN/dummy@face1234" in msg


def test_build_commit_message_override_wins(tmp_path: Path) -> None:
    m = load_module()
    _write_valid_manifest(tmp_path)
    manifest = m.load_manifest(tmp_path)
    assert m.build_commit_message(manifest, override="my message") == "my message"


def test_main_dry_run_does_not_call_hf(tmp_path: Path, capsys) -> None:
    """--dry-run で HfApi を import / instantiate せずに 0 return。"""
    m = load_module()
    _write_valid_manifest(tmp_path, ckpt_hash="dryrun12hash")
    rc = m.main(
        [
            "--cache-dir",
            str(tmp_path),
            "--repo-id",
            "Team-RAMEN/some_merged_v1",
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Team-RAMEN/some_merged_v1" in captured
    assert "obb_yolo/dryrun12hash" in captured
    assert "[dry-run]" in captured
