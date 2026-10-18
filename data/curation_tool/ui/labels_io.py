"""
labels_io.py — parquet + jsonl + registry の R/W (Schema v4)。

Schema v4 の設計 (前 v3 から大幅簡略化):
- primary key = (episode_id, frame_start)
- 各 row = 1 curation unit (= 1 skill、user が確定させた最終 view)
- 保存する field は「最終値」のみ (task_index_original / _corrected の delta 概念廃止)
- unit_id は UI 側で frame_start sort により導出 (parquet には持たない)
- split: 2 row 挿入 (1 元 seg を 2 unit に分割)
- merge: 1 row の frame 範囲を拡張 + 統合対象 row を delete

- labels.jsonl: primary (append-only)、delete op も `__delete__` marker で記録
- labels.parquet: 毎回 rebuild
- metadata.json: reviewer state
- failure_categories.json: 動的 category registry
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


SCHEMA_VERSION = 4

VERDICTS = ("optimal", "success", "ambiguous", "failure")

# Schema v4: 9 data cols (primary key = ep + frame_start)
_SCHEMA: dict[str, str] = {
    "episode_id": "int64",
    "frame_start": "int64",
    "frame_end": "int64",
    "task_index": "int64",
    "verdict": "string",
    "failure_category": "string",
    "reviewer": "string",
    "reviewed_at": "string",
    "schema_version": "int64",
}

_JST = timezone(timedelta(hours=9))


# --- Path helpers ---

def _default_workspace() -> Path:
    return Path(__file__).resolve().parent.parent / "workspace"


def workspace_dir(workspace: Optional[Path] = None) -> Path:
    ws = Path(workspace) if workspace is not None else _default_workspace()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


# --- Empty df ---

def _empty_labels_df() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in _SCHEMA.items()})


# --- Validation ---

def _validate_row(row: dict) -> None:
    missing = [k for k in _SCHEMA if k not in row]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if row["verdict"] not in VERDICTS:
        raise ValueError(
            f"invalid verdict: {row['verdict']!r}, must be one of {VERDICTS}"
        )
    if row["verdict"] == "failure" and not row.get("failure_category"):
        raise ValueError("failure_category is required when verdict == 'failure'")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {row['schema_version']} != current {SCHEMA_VERSION}"
        )
    if row["frame_start"] > row["frame_end"]:
        raise ValueError(
            f"frame_start ({row['frame_start']}) > frame_end ({row['frame_end']})"
        )


# --- I/O helpers ---

def _write_row_to_jsonl(row: dict, ws: Path) -> None:
    with (ws / "labels.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_df_to_parquet(df: pd.DataFrame, ws: Path) -> None:
    df.astype(_SCHEMA).to_parquet(ws / "labels.parquet", index=False)


# --- Load ---

def load_labels(workspace: Optional[Path] = None) -> pd.DataFrame:
    """labels.parquet を読む。無ければ empty df。frame_start で sort 済で返す。"""
    ws = workspace_dir(workspace)
    path = ws / "labels.parquet"
    if not path.exists():
        return _empty_labels_df()
    df = pd.read_parquet(path)
    if "schema_version" in df.columns:
        bad = df[df["schema_version"] != SCHEMA_VERSION]
        if len(bad) > 0:
            warnings.warn(
                f"{len(bad)} rows have schema_version != {SCHEMA_VERSION}, "
                "consider migration"
            )
    return df.sort_values(["episode_id", "frame_start"]).reset_index(drop=True)


# --- Save (upsert / delete) ---

def upsert_label(row: dict, workspace: Optional[Path] = None) -> None:
    """
    (episode_id, frame_start) primary key で upsert。
    frame_start が既存と一致すれば更新、無ければ挿入。
    jsonl は append (履歴)、parquet は最新 dedup 状態を保存。
    """
    _validate_row(row)
    ws = workspace_dir(workspace)

    _write_row_to_jsonl(row, ws)

    df = load_labels(ws)
    key_mask = (df["episode_id"] == row["episode_id"]) & (
        df["frame_start"] == row["frame_start"]
    )
    if key_mask.any():
        df = df[~key_mask]
    combined = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _write_df_to_parquet(combined, ws)


def reset_ep_labels(episode_id: int, workspace: Optional[Path] = None) -> int:
    """指定 episode の全 label を削除 (検証用の LeRobot 元状態リセット)。

    jsonl に __delete__ marker を row 数分書き込み、parquet から該当 row 除外。
    Return: 削除した row 数。
    """
    ws = workspace_dir(workspace)
    df = load_labels(ws)
    mask = df["episode_id"] == episode_id
    to_delete = df[mask]
    if len(to_delete) == 0:
        return 0

    now = datetime.now(_JST).isoformat(timespec="seconds")
    for _, row in to_delete.iterrows():
        marker = {
            "__delete__": True,
            "episode_id": int(row["episode_id"]),
            "frame_start": int(row["frame_start"]),
            "deleted_at": now,
            "schema_version": SCHEMA_VERSION,
        }
        _write_row_to_jsonl(marker, ws)

    df = df[~mask]
    _write_df_to_parquet(df, ws)
    return len(to_delete)


def delete_label(
    episode_id: int, frame_start: int, workspace: Optional[Path] = None
) -> bool:
    """(episode_id, frame_start) の row を削除。存在時 True。

    jsonl には delete marker (`__delete__: true`) を append (履歴保持)、
    parquet からは row を除外。merge 操作等で不要になった unit を消す用途。
    """
    ws = workspace_dir(workspace)
    df = load_labels(ws)
    key_mask = (df["episode_id"] == episode_id) & (df["frame_start"] == frame_start)
    if not key_mask.any():
        return False

    marker = {
        "__delete__": True,
        "episode_id": episode_id,
        "frame_start": frame_start,
        "deleted_at": datetime.now(_JST).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
    }
    _write_row_to_jsonl(marker, ws)

    df = df[~key_mask]
    _write_df_to_parquet(df, ws)
    return True


def restore_parquet_from_jsonl(workspace: Optional[Path] = None) -> int:
    """
    jsonl から parquet を rebuild (crash recovery)。
    __delete__ marker があれば その key を latest から除外、upsert は上書き。
    Return: parquet に含めた row 数。
    """
    ws = workspace_dir(workspace)
    jsonl = ws / "labels.jsonl"
    if not jsonl.exists():
        _write_df_to_parquet(_empty_labels_df(), ws)
        return 0

    rows: list[dict] = []
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    latest: dict[tuple[int, int], dict] = {}
    for row in rows:
        key = (row["episode_id"], row["frame_start"])
        if row.get("__delete__"):
            latest.pop(key, None)
        else:
            latest[key] = row

    if not latest:
        _write_df_to_parquet(_empty_labels_df(), ws)
        return 0

    df = pd.DataFrame(list(latest.values()))
    _write_df_to_parquet(df, ws)
    return len(df)


# --- Metadata (reviewer state) ---

_DEFAULT_METADATA: dict = {
    "last_ep": None,
    "last_frame_start": None,  # frame_start of last edited unit (was last_segment)
    "reviewer": "hara",
    "schema_version": SCHEMA_VERSION,
}


def load_metadata(workspace: Optional[Path] = None) -> dict:
    ws = workspace_dir(workspace)
    path = ws / "metadata.json"
    if not path.exists():
        return dict(_DEFAULT_METADATA)
    with path.open(encoding="utf-8") as f:
        meta = json.load(f)
    for k, v in _DEFAULT_METADATA.items():
        meta.setdefault(k, v)
    return meta


def save_metadata(meta: dict, workspace: Optional[Path] = None) -> None:
    ws = workspace_dir(workspace)
    with (ws / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# --- Failure category registry ---

def load_categories(workspace: Optional[Path] = None) -> list[dict]:
    ws = workspace_dir(workspace)
    path = ws / "failure_categories.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("categories", []))


def add_category(
    key: str,
    label: str,
    added_ep: int,
    workspace: Optional[Path] = None,
) -> None:
    ws = workspace_dir(workspace)
    cats = load_categories(ws)
    if any(c["key"] == key for c in cats):
        raise ValueError(f"category key {key!r} already exists in registry")
    now = datetime.now(_JST).isoformat(timespec="seconds")
    cats.append({
        "key": key,
        "label": label,
        "added_at": now,
        "added_ep": added_ep,
    })
    with (ws / "failure_categories.json").open("w", encoding="utf-8") as f:
        json.dump({"categories": cats}, f, ensure_ascii=False, indent=2)


# --- Row helper ---

def make_row(
    episode_id: int,
    frame_start: int,
    frame_end: int,
    task_index: int,
    verdict: str,
    failure_category: Optional[str] = None,
    reviewer: str = "hara",
    reviewed_at: Optional[str] = None,
) -> dict:
    """Schema v4 row dict を組み立てる helper。"""
    if reviewed_at is None:
        reviewed_at = datetime.now(_JST).isoformat(timespec="seconds")
    return {
        "episode_id": episode_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "task_index": task_index,
        "verdict": verdict,
        "failure_category": failure_category,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "schema_version": SCHEMA_VERSION,
    }
