"""hf_sync.py の unit test (network mock ベース)。

実 HF push は Phase B ep0 dogfood で smoke。ここでは resolve_token と
compute_preview / sync_to_hf の I/O ロジックを monkeypatch で検証する。
"""

from unittest.mock import MagicMock

import pytest

from ui import hf_sync
from ui.hf_sync import (
    SyncPreview,
    compute_preview,
    resolve_token,
    sync_to_hf,
)
from ui.labels_io import add_category, make_row, upsert_label


# --- resolve_token ---

def test_resolve_token_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env_token")
    assert resolve_token("explicit_token") == "explicit_token"


def test_resolve_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env_token_value")
    assert resolve_token() == "env_token_value"


def test_resolve_token_no_source_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hf_sync, "DEFAULT_ENV_PATH", tmp_path / "nonexistent.env")
    with pytest.raises(RuntimeError, match="HF_TOKEN not found"):
        resolve_token()


def test_resolve_token_from_dotenv_file(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=dotenv_token_value\n")
    monkeypatch.setattr(hf_sync, "DEFAULT_ENV_PATH", env_file)
    assert resolve_token() == "dotenv_token_value"


# --- SyncPreview ---

def test_sync_preview_default_fields():
    p = SyncPreview(
        remote_exists=False,
        labels_new_rows=0,
        labels_total_rows_local=0,
        jsonl_new_lines=0,
    )
    assert p.categories_added == []
    assert p.files_to_upload == []


# --- compute_preview ---

def _seed_workspace(ws):
    """labels 1 row + category 1 個 を初期化 (Schema v4)。"""
    upsert_label(
        make_row(
            episode_id=0, frame_start=100, frame_end=250,
            task_index=2, verdict="success",
        ),
        workspace=ws,
    )
    add_category("slip", "滑り落ち", added_ep=0, workspace=ws)


def test_compute_preview_initial_no_remote(monkeypatch, tmp_workspace):
    _seed_workspace(tmp_workspace)
    monkeypatch.setenv("HF_TOKEN", "fake_token")
    monkeypatch.setattr(
        hf_sync, "_fetch_remote_labels_row_count",
        lambda repo_id, token: None,
    )
    monkeypatch.setattr(
        hf_sync, "_fetch_remote_categories",
        lambda repo_id, token: None,
    )

    preview = compute_preview(tmp_workspace)
    assert preview.remote_exists is False
    assert preview.labels_new_rows == 1
    assert preview.labels_total_rows_local == 1
    assert preview.jsonl_new_lines == 1
    assert len(preview.categories_added) == 1
    assert preview.categories_added[0]["key"] == "slip"
    assert "labels.parquet" in preview.files_to_upload
    assert "labels.jsonl" in preview.files_to_upload
    assert "failure_categories.json" in preview.files_to_upload


def test_compute_preview_partial_remote_new_rows_only(monkeypatch, tmp_workspace):
    for i in range(5):
        upsert_label(
            make_row(
                episode_id=0, frame_start=i * 100, frame_end=(i + 1) * 100,
                task_index=2, verdict="success",
            ),
            workspace=tmp_workspace,
        )
    monkeypatch.setenv("HF_TOKEN", "fake_token")
    monkeypatch.setattr(
        hf_sync, "_fetch_remote_labels_row_count",
        lambda repo_id, token: 3,
    )
    monkeypatch.setattr(
        hf_sync, "_fetch_remote_categories",
        lambda repo_id, token: [],
    )

    preview = compute_preview(tmp_workspace)
    assert preview.remote_exists is True
    assert preview.labels_new_rows == 2
    assert preview.labels_total_rows_local == 5


def test_compute_preview_categories_diff(monkeypatch, tmp_workspace):
    for k, l in [("slip", "滑り"), ("flip", "反転"), ("drop", "落下")]:
        add_category(k, l, added_ep=0, workspace=tmp_workspace)
    monkeypatch.setenv("HF_TOKEN", "fake_token")
    monkeypatch.setattr(
        hf_sync, "_fetch_remote_labels_row_count",
        lambda repo_id, token: 0,
    )
    monkeypatch.setattr(
        hf_sync, "_fetch_remote_categories",
        lambda repo_id, token: [{"key": "slip", "label": "滑り"}],
    )

    preview = compute_preview(tmp_workspace)
    added_keys = {c["key"] for c in preview.categories_added}
    assert added_keys == {"flip", "drop"}


def test_compute_preview_backup_dir_included(monkeypatch, tmp_workspace):
    _seed_workspace(tmp_workspace)
    backup_dir = tmp_workspace / "labels_backup"
    backup_dir.mkdir()
    (backup_dir / "20260716_2200_labels.parquet").write_bytes(b"fake")
    (backup_dir / "20260717_2200_labels.parquet").write_bytes(b"fake")

    monkeypatch.setenv("HF_TOKEN", "fake_token")
    monkeypatch.setattr(
        hf_sync, "_fetch_remote_labels_row_count",
        lambda repo_id, token: None,
    )
    monkeypatch.setattr(
        hf_sync, "_fetch_remote_categories",
        lambda repo_id, token: None,
    )

    preview = compute_preview(tmp_workspace)
    backup_files = [f for f in preview.files_to_upload if f.startswith("labels_backup/")]
    assert len(backup_files) == 2


# --- sync_to_hf ---

def test_sync_to_hf_calls_create_commit(monkeypatch, tmp_workspace):
    _seed_workspace(tmp_workspace)
    monkeypatch.setenv("HF_TOKEN", "fake_token")

    mock_commit_info = MagicMock()
    mock_commit_info.commit_url = "https://huggingface.co/datasets/foo/bar/commit/abc"
    mock_api = MagicMock()
    mock_api.create_commit.return_value = mock_commit_info

    def mock_hfapi_ctor(token):
        assert token == "fake_token"
        return mock_api

    monkeypatch.setattr(hf_sync, "HfApi", mock_hfapi_ctor)

    url = sync_to_hf(tmp_workspace, commit_message="test sync")
    assert "commit" in url
    mock_api.create_commit.assert_called_once()

    call_kwargs = mock_api.create_commit.call_args.kwargs
    assert call_kwargs["repo_id"] == hf_sync.DEFAULT_REPO_ID
    assert call_kwargs["repo_type"] == "dataset"
    assert call_kwargs["commit_message"] == "test sync"
    assert len(call_kwargs["operations"]) == 3  # 3 SYNC_FILES


def test_sync_to_hf_default_commit_message(monkeypatch, tmp_workspace):
    _seed_workspace(tmp_workspace)
    monkeypatch.setenv("HF_TOKEN", "fake_token")

    mock_commit_info = MagicMock()
    mock_commit_info.commit_url = "https://foo/commit/abc"
    mock_api = MagicMock()
    mock_api.create_commit.return_value = mock_commit_info
    monkeypatch.setattr(hf_sync, "HfApi", lambda token: mock_api)

    sync_to_hf(tmp_workspace)

    msg = mock_api.create_commit.call_args.kwargs["commit_message"]
    assert msg.startswith("curation: sync at ")


def test_sync_to_hf_no_files_raises(monkeypatch, tmp_workspace):
    monkeypatch.setenv("HF_TOKEN", "fake_token")
    with pytest.raises(RuntimeError, match="no files to upload"):
        sync_to_hf(tmp_workspace)


def test_sync_to_hf_includes_backup_files(monkeypatch, tmp_workspace):
    _seed_workspace(tmp_workspace)
    backup_dir = tmp_workspace / "labels_backup"
    backup_dir.mkdir()
    (backup_dir / "snap1.parquet").write_bytes(b"fake1")
    (backup_dir / "snap2.parquet").write_bytes(b"fake2")

    monkeypatch.setenv("HF_TOKEN", "fake_token")
    mock_commit_info = MagicMock()
    mock_commit_info.commit_url = "url"
    mock_api = MagicMock()
    mock_api.create_commit.return_value = mock_commit_info
    monkeypatch.setattr(hf_sync, "HfApi", lambda token: mock_api)

    sync_to_hf(tmp_workspace)

    ops = mock_api.create_commit.call_args.kwargs["operations"]
    paths = [op.path_in_repo for op in ops]
    assert "labels_backup/snap1.parquet" in paths
    assert "labels_backup/snap2.parquet" in paths
    assert len(ops) == 5  # 3 SYNC_FILES + 2 backups
