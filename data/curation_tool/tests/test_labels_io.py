"""labels_io.py の unit test (Schema v4 前提)。

Schema v4 変更点:
- primary key = (episode_id, frame_start) — segment_id / sub_segment_id 廃止
- 保存 field は最終値のみ (task_index / frame_start / frame_end)、_original / _corrected
  の delta 概念廃止
- delete_label 追加 (merge 操作で consumed unit を消す用途)
"""

import json

import pandas as pd
import pytest

from ui.labels_io import (
    SCHEMA_VERSION,
    add_category,
    delete_label,
    load_categories,
    load_labels,
    load_metadata,
    make_row,
    restore_parquet_from_jsonl,
    save_metadata,
    upsert_label,
)


def _sample_row(**overrides) -> dict:
    row = make_row(
        episode_id=0,
        frame_start=100,
        frame_end=250,
        task_index=2,
        verdict="success",
    )
    row.update(overrides)
    return row


# --- load_labels ---

def test_load_labels_empty_when_no_file(tmp_workspace):
    df = load_labels(tmp_workspace)
    assert len(df) == 0
    assert "verdict" in df.columns
    assert "frame_start" in df.columns
    assert "schema_version" in df.columns
    # v4 で廃止された field は無いこと
    assert "segment_id" not in df.columns
    assert "task_index_original" not in df.columns
    assert "frame_start_corrected" not in df.columns


def test_load_labels_sorted_by_frame_start(tmp_workspace):
    upsert_label(_sample_row(frame_start=500, frame_end=600), tmp_workspace)
    upsert_label(_sample_row(frame_start=100, frame_end=200), tmp_workspace)
    upsert_label(_sample_row(frame_start=300, frame_end=400), tmp_workspace)
    df = load_labels(tmp_workspace)
    assert list(df["frame_start"]) == [100, 300, 500]


# --- upsert_label ---

def test_upsert_new_and_load_roundtrip(tmp_workspace):
    upsert_label(_sample_row(), tmp_workspace)
    df = load_labels(tmp_workspace)
    assert len(df) == 1
    assert df.iloc[0]["episode_id"] == 0
    assert df.iloc[0]["frame_start"] == 100
    assert df.iloc[0]["frame_end"] == 250
    assert df.iloc[0]["task_index"] == 2
    assert df.iloc[0]["verdict"] == "success"
    assert df.iloc[0]["schema_version"] == SCHEMA_VERSION


def test_upsert_writes_jsonl(tmp_workspace):
    upsert_label(_sample_row(), tmp_workspace)
    jsonl = tmp_workspace / "labels.jsonl"
    assert jsonl.exists()
    with jsonl.open() as f:
        lines = f.readlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["verdict"] == "success"
    assert obj["frame_start"] == 100


def test_upsert_overwrites_existing_by_frame_start(tmp_workspace):
    upsert_label(_sample_row(), tmp_workspace)
    upsert_label(_sample_row(verdict="optimal", frame_end=280), tmp_workspace)

    df = load_labels(tmp_workspace)
    assert len(df) == 1
    assert df.iloc[0]["verdict"] == "optimal"
    assert df.iloc[0]["frame_end"] == 280


def test_upsert_different_frame_start_inserts_new(tmp_workspace):
    upsert_label(_sample_row(frame_start=100, frame_end=200), tmp_workspace)
    upsert_label(_sample_row(frame_start=300, frame_end=400), tmp_workspace)

    df = load_labels(tmp_workspace)
    assert len(df) == 2


def test_upsert_multiple_episodes(tmp_workspace):
    upsert_label(_sample_row(episode_id=0, frame_start=100), tmp_workspace)
    upsert_label(_sample_row(episode_id=1, frame_start=100), tmp_workspace)

    df = load_labels(tmp_workspace)
    assert len(df) == 2
    assert set(df["episode_id"]) == {0, 1}


def test_upsert_jsonl_preserves_history(tmp_workspace):
    upsert_label(_sample_row(), tmp_workspace)
    upsert_label(_sample_row(verdict="ambiguous"), tmp_workspace)

    with (tmp_workspace / "labels.jsonl").open() as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["verdict"] == "success"
    assert lines[1]["verdict"] == "ambiguous"


# --- Validation ---

def test_verdict_failure_requires_failure_category(tmp_workspace):
    row = _sample_row(verdict="failure", failure_category=None)
    with pytest.raises(ValueError, match="failure_category is required"):
        upsert_label(row, tmp_workspace)


def test_verdict_failure_with_category_ok(tmp_workspace):
    row = _sample_row(verdict="failure", failure_category="slip")
    upsert_label(row, tmp_workspace)
    df = load_labels(tmp_workspace)
    assert df.iloc[0]["failure_category"] == "slip"


def test_invalid_verdict_raises(tmp_workspace):
    row = _sample_row(verdict="bad_value")
    with pytest.raises(ValueError, match="invalid verdict"):
        upsert_label(row, tmp_workspace)


def test_schema_version_mismatch_on_write_raises(tmp_workspace):
    row = _sample_row()
    row["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        upsert_label(row, tmp_workspace)


def test_missing_field_raises(tmp_workspace):
    row = _sample_row()
    del row["reviewer"]
    with pytest.raises(ValueError, match="missing fields"):
        upsert_label(row, tmp_workspace)


def test_frame_start_greater_than_end_raises(tmp_workspace):
    row = _sample_row(frame_start=300, frame_end=200)
    with pytest.raises(ValueError, match="frame_start"):
        upsert_label(row, tmp_workspace)


def test_frame_start_equals_end_ok(tmp_workspace):
    """1 frame の segment (境界含む)。"""
    row = _sample_row(frame_start=100, frame_end=100)
    upsert_label(row, tmp_workspace)
    df = load_labels(tmp_workspace)
    assert len(df) == 1


# --- delete_label ---

def test_delete_existing_returns_true(tmp_workspace):
    upsert_label(_sample_row(frame_start=100, frame_end=250), tmp_workspace)
    upsert_label(_sample_row(frame_start=300, frame_end=400), tmp_workspace)

    assert delete_label(0, 100, tmp_workspace) is True

    df = load_labels(tmp_workspace)
    assert len(df) == 1
    assert df.iloc[0]["frame_start"] == 300


def test_delete_nonexistent_returns_false(tmp_workspace):
    assert delete_label(0, 999, tmp_workspace) is False


def test_delete_writes_marker_to_jsonl(tmp_workspace):
    upsert_label(_sample_row(frame_start=100), tmp_workspace)
    delete_label(0, 100, tmp_workspace)

    with (tmp_workspace / "labels.jsonl").open() as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[1].get("__delete__") is True
    assert lines[1]["frame_start"] == 100


# --- restore_parquet_from_jsonl ---

def test_restore_after_upsert_and_delete(tmp_workspace):
    """upsert → upsert → delete の履歴が正しく restore される。"""
    upsert_label(_sample_row(frame_start=100, frame_end=250), tmp_workspace)
    upsert_label(_sample_row(frame_start=300, frame_end=400), tmp_workspace)
    delete_label(0, 100, tmp_workspace)

    (tmp_workspace / "labels.parquet").unlink()
    n = restore_parquet_from_jsonl(tmp_workspace)
    assert n == 1
    df = load_labels(tmp_workspace)
    assert list(df["frame_start"]) == [300]


def test_restore_no_jsonl_yields_empty(tmp_workspace):
    n = restore_parquet_from_jsonl(tmp_workspace)
    assert n == 0
    assert len(load_labels(tmp_workspace)) == 0


def test_restore_delete_then_recreate(tmp_workspace):
    """delete 後に同 key を再挿入 → 最新の row が生き残る。"""
    upsert_label(_sample_row(frame_start=100, verdict="success"), tmp_workspace)
    delete_label(0, 100, tmp_workspace)
    upsert_label(_sample_row(frame_start=100, verdict="optimal"), tmp_workspace)

    (tmp_workspace / "labels.parquet").unlink()
    restore_parquet_from_jsonl(tmp_workspace)
    df = load_labels(tmp_workspace)
    assert len(df) == 1
    assert df.iloc[0]["verdict"] == "optimal"


# --- Categories registry ---

def test_categories_empty_when_no_file(tmp_workspace):
    assert load_categories(tmp_workspace) == []


def test_categories_add_and_load(tmp_workspace):
    add_category("slip", "滑り落ち", added_ep=0, workspace=tmp_workspace)
    add_category("flip", "leg 反転", added_ep=5, workspace=tmp_workspace)

    cats = load_categories(tmp_workspace)
    assert len(cats) == 2
    keys = {c["key"] for c in cats}
    assert keys == {"slip", "flip"}


def test_categories_duplicate_key_raises(tmp_workspace):
    add_category("slip", "滑り落ち", added_ep=0, workspace=tmp_workspace)
    with pytest.raises(ValueError, match="already exists"):
        add_category("slip", "別 label", added_ep=1, workspace=tmp_workspace)


# --- Metadata ---

def test_metadata_default_when_no_file(tmp_workspace):
    meta = load_metadata(tmp_workspace)
    assert meta["last_ep"] is None
    assert meta["last_frame_start"] is None
    assert meta["reviewer"] == "hara"
    assert meta["schema_version"] == SCHEMA_VERSION


def test_metadata_save_and_load_roundtrip(tmp_workspace):
    save_metadata(
        {"last_ep": 3, "last_frame_start": 5000, "reviewer": "hara",
         "schema_version": SCHEMA_VERSION},
        tmp_workspace,
    )
    meta = load_metadata(tmp_workspace)
    assert meta["last_ep"] == 3
    assert meta["last_frame_start"] == 5000


def test_metadata_missing_fields_filled_by_default(tmp_workspace):
    save_metadata({"last_ep": 7}, tmp_workspace)
    meta = load_metadata(tmp_workspace)
    assert meta["last_ep"] == 7
    assert meta["reviewer"] == "hara"
    assert meta["schema_version"] == SCHEMA_VERSION
