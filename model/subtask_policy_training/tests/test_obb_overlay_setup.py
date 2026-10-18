"""Unit tests for model/subtask_policy_training/scripts/obb_overlay_setup.py (Issue #122 D-4-S1)。

env-driven setup が (a) enable flag off で no-op、(b) 必要 env 未設定で ValueError、
(c) 正常系で ObbPrecomputedCache + OverlayRenderer 作成 + hook register する事を検証。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# setup module 側と同じ import 経路を使う (別 instance を作らない、pytest.ini の
# pythonpath=. で resolvable)
from model.subtask_policy_training.scripts import (  # noqa: E402
    lerobot_frame_cache_patch as fcp,
)
from model.subtask_policy_training.scripts import (  # noqa: E402
    obb_overlay_setup as setup,
)


def _write_dummy_cache(tmp_path: Path, ckpt_hash: str = "abc123def456"):
    """`lookup_mp4_frame` を通せる最小 cache dir + parquet を作る。"""
    root = tmp_path / "yolo_obb_cache"
    cache_dir = root / ckpt_hash
    cache_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "team_ramen_yolo_obb_cache_v1",
        "yolo_ckpt_hash": ckpt_hash,
        "yolo_ckpt_ref": f"hf:dummy@{ckpt_hash}",
        "yolo_class_names": {str(i): f"c{i}" for i in range(7)},
        "cameras": ["observation.images.cam_0", "observation.images.cam_1"],
        "top_k": 4,
        "imgsz": 800,
        "conf_threshold": 0.15,
        "num_classes": 7,
        "generated_at": "2026-08-18T00:00:00Z",
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))

    schema = pa.schema(
        [
            pa.field("frame_index", pa.int32(), nullable=False),
            pa.field("det_idx", pa.int8(), nullable=False),
            pa.field("verts", pa.list_(pa.float32(), 8), nullable=False),
            pa.field("conf", pa.float32(), nullable=False),
            pa.field("class_id", pa.int8(), nullable=False),
            pa.field("valid", pa.bool_(), nullable=False),
        ]
    )
    for cam in manifest["cameras"]:
        p = cache_dir / cam / "chunk-000" / "file-000.parquet"
        p.parent.mkdir(parents=True)
        rows: dict[str, list] = {
            "frame_index": [], "det_idx": [], "verts": [], "conf": [],
            "class_id": [], "valid": [],
        }
        for f in range(5):  # 5 frames
            for d in range(4):  # top_K=4
                rows["frame_index"].append(f)
                rows["det_idx"].append(d)
                rows["verts"].append([0.0] * 8)
                rows["conf"].append(0.0)
                rows["class_id"].append(-1)
                rows["valid"].append(False)
        pq.write_table(pa.table(rows, schema=schema), p, compression="snappy")
    return root, ckpt_hash


def _clear_all_obb_env(monkeypatch):
    for name in [
        "OBB_OVERLAY_ENABLE",
        "OBB_PRECOMPUTED_ROOT",
        "OBB_PRECOMPUTED_HASH",
        "OBB_OVERLAY_CAMERAS",
        "OBB_OVERLAY_TOP_K",
        "OBB_OVERLAY_NUM_CLASSES",
        "OBB_OVERLAY_FPS",
        "OBB_OVERLAY_CONF",
        "OBB_OVERLAY_THICKNESS",
        "OBB_OVERLAY_CLASS_FILTER",
    ]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _reset_hook_registry():
    """test 毎に patch module の hook registry を clear (test 分離)。"""
    fcp.clear_post_decode_hooks()
    yield
    fcp.clear_post_decode_hooks()


# ============================================================================
# is_enabled
# ============================================================================


def test_is_enabled_default_false(monkeypatch) -> None:
    monkeypatch.delenv("OBB_OVERLAY_ENABLE", raising=False)
    assert setup.is_enabled() is False


@pytest.mark.parametrize("val", ["true", "TRUE", "1", "yes", "on"])
def test_is_enabled_truthy_values(monkeypatch, val: str) -> None:
    monkeypatch.setenv("OBB_OVERLAY_ENABLE", val)
    assert setup.is_enabled() is True


@pytest.mark.parametrize("val", ["false", "0", "no", "off", "", " "])
def test_is_enabled_falsy_values(monkeypatch, val: str) -> None:
    monkeypatch.setenv("OBB_OVERLAY_ENABLE", val)
    assert setup.is_enabled() is False


# ============================================================================
# setup_from_env
# ============================================================================


def test_setup_disabled_returns_none_no_hook_registered(monkeypatch) -> None:
    _clear_all_obb_env(monkeypatch)
    result = setup.setup_from_env()
    assert result is None
    # hook registry も空のまま
    assert len(fcp._post_decode_hooks) == 0


def test_setup_enabled_without_root_raises(monkeypatch) -> None:
    _clear_all_obb_env(monkeypatch)
    monkeypatch.setenv("OBB_OVERLAY_ENABLE", "true")
    with pytest.raises(ValueError, match="OBB_PRECOMPUTED_ROOT"):
        setup.setup_from_env()


def test_setup_enabled_registers_hook(monkeypatch, tmp_path) -> None:
    _clear_all_obb_env(monkeypatch)
    root, hash_ = _write_dummy_cache(tmp_path)
    monkeypatch.setenv("OBB_OVERLAY_ENABLE", "true")
    monkeypatch.setenv("OBB_PRECOMPUTED_ROOT", str(root))
    renderer = setup.setup_from_env()

    assert renderer is not None
    # hook registered
    assert len(fcp._post_decode_hooks) == 1
    # renderer が hook registry の値と一致 (単一 renderer 保証)
    assert fcp._post_decode_hooks[0] is renderer


def test_setup_clears_prior_hooks_before_register(monkeypatch, tmp_path) -> None:
    """setup は clear_post_decode_hooks() を先に呼ぶ (単一 hook invariant)。"""
    _clear_all_obb_env(monkeypatch)
    root, _ = _write_dummy_cache(tmp_path)
    fcp.register_post_decode_hook(lambda i, v, f: i)  # dummy prior hook
    fcp.register_post_decode_hook(lambda i, v, f: i)
    assert len(fcp._post_decode_hooks) == 2

    monkeypatch.setenv("OBB_OVERLAY_ENABLE", "true")
    monkeypatch.setenv("OBB_PRECOMPUTED_ROOT", str(root))
    setup.setup_from_env()

    # 1 hook のみ残る (setup 側の overlay renderer)
    assert len(fcp._post_decode_hooks) == 1


def test_setup_reads_conf_thickness_class_filter_from_env(monkeypatch, tmp_path) -> None:
    _clear_all_obb_env(monkeypatch)
    root, _ = _write_dummy_cache(tmp_path)
    monkeypatch.setenv("OBB_OVERLAY_ENABLE", "true")
    monkeypatch.setenv("OBB_PRECOMPUTED_ROOT", str(root))
    monkeypatch.setenv("OBB_OVERLAY_CONF", "0.45")
    monkeypatch.setenv("OBB_OVERLAY_THICKNESS", "3")
    monkeypatch.setenv("OBB_OVERLAY_CLASS_FILTER", "[3, 4]")
    renderer = setup.setup_from_env()

    assert renderer.conf_threshold == pytest.approx(0.45)
    assert renderer.line_thickness == 3
    assert renderer.class_filter == {3, 4}


def test_setup_default_conf_and_thickness_match_d3_decision(monkeypatch, tmp_path) -> None:
    """D-3 で決定した preview-based default 値 (conf=0.30, thickness=2) が env 未指定時に効く。"""
    _clear_all_obb_env(monkeypatch)
    root, _ = _write_dummy_cache(tmp_path)
    monkeypatch.setenv("OBB_OVERLAY_ENABLE", "true")
    monkeypatch.setenv("OBB_PRECOMPUTED_ROOT", str(root))
    renderer = setup.setup_from_env()

    assert renderer.conf_threshold == pytest.approx(0.30)
    assert renderer.line_thickness == 2
    assert renderer.class_filter is None


def test_setup_default_cameras_are_head_l_r(monkeypatch, tmp_path) -> None:
    _clear_all_obb_env(monkeypatch)
    root, _ = _write_dummy_cache(tmp_path)
    monkeypatch.setenv("OBB_OVERLAY_ENABLE", "true")
    monkeypatch.setenv("OBB_PRECOMPUTED_ROOT", str(root))
    renderer = setup.setup_from_env()

    assert renderer.cache.covered_cameras == {
        "observation.images.cam_0",
        "observation.images.cam_1",
    }


def test_setup_invalid_class_filter_json_raises(monkeypatch, tmp_path) -> None:
    _clear_all_obb_env(monkeypatch)
    root, _ = _write_dummy_cache(tmp_path)
    monkeypatch.setenv("OBB_OVERLAY_ENABLE", "true")
    monkeypatch.setenv("OBB_PRECOMPUTED_ROOT", str(root))
    monkeypatch.setenv("OBB_OVERLAY_CLASS_FILTER", '{"not": "a list"}')
    with pytest.raises(ValueError, match="must be a JSON list of int"):
        setup.setup_from_env()


def test_setup_explicit_hash_wins_over_auto_pick(monkeypatch, tmp_path) -> None:
    """OBB_PRECOMPUTED_HASH が指定されていれば auto-pick より優先。"""
    _clear_all_obb_env(monkeypatch)
    root, hash_ = _write_dummy_cache(tmp_path, ckpt_hash="mine12345678")
    monkeypatch.setenv("OBB_OVERLAY_ENABLE", "true")
    monkeypatch.setenv("OBB_PRECOMPUTED_ROOT", str(root))
    monkeypatch.setenv("OBB_PRECOMPUTED_HASH", "mine12345678")
    renderer = setup.setup_from_env()
    assert renderer.cache.ckpt_hash == "mine12345678"


def test_module_level_renderer_reference_retained(monkeypatch, tmp_path) -> None:
    """setup 後 module-level `_registered_renderer` が dead reference にならない。"""
    _clear_all_obb_env(monkeypatch)
    root, _ = _write_dummy_cache(tmp_path)
    monkeypatch.setenv("OBB_OVERLAY_ENABLE", "true")
    monkeypatch.setenv("OBB_PRECOMPUTED_ROOT", str(root))
    renderer = setup.setup_from_env()
    assert setup._registered_renderer is renderer
