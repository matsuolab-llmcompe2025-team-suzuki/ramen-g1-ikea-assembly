"""app.py endpoint test (Schema v4)。"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ui import hf_sync, segments
from ui.app import create_app
from ui.hf_sync import SyncPreview
from ui.segments import Segment


def _mock_meta() -> MagicMock:
    m = MagicMock()
    m.total_episodes = 3
    m.fps = 30
    return m


def _mock_segments(ep_id: int) -> list[Segment]:
    return [
        Segment(episode_id=ep_id, segment_id=0, task_index=2,
                frame_start=0, frame_end=99, frame_count=100,
                task_name="flip_table"),
        Segment(episode_id=ep_id, segment_id=1, task_index=3,
                frame_start=100, frame_end=249, frame_count=150,
                task_name="rotate_leg_to_tighten"),
    ]


@pytest.fixture
def client(monkeypatch, tmp_workspace):
    monkeypatch.setattr(
        segments, "get_segments",
        lambda meta, ep_id: (
            _mock_segments(ep_id) if ep_id < 3
            else (_ for _ in ()).throw(IndexError(f"ep {ep_id} out of range"))
        ),
    )
    def fake_build(meta, seg, output_path, fps=None, force=False):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake mp4 content")
        return output_path
    monkeypatch.setattr(segments, "build_tile_mp4", fake_build)

    app = create_app(
        workspace=tmp_workspace,
        meta_loader=_mock_meta,
        load_env=False,
    )
    with TestClient(app) as c:
        yield c


def _label_payload(**overrides) -> dict:
    d = {
        "episode_id": 0,
        "frame_start": 0,
        "frame_end": 99,
        "task_index": 2,
        "verdict": "success",
    }
    d.update(overrides)
    return d


# --- /api/episodes ---

def test_get_episodes(client):
    r = client.get("/api/episodes")
    assert r.status_code == 200
    assert r.json() == {"episodes": [0, 1, 2]}


def test_get_task_names(client):
    r = client.get("/api/task_names")
    assert r.status_code == 200
    body = r.json()
    assert "task_names" in body
    assert body["task_names"]["0"] == "insert_table_leg"


# --- /api/episodes/{ep_id}/segments ---

def test_get_segments_ok(client):
    r = client.get("/api/episodes/0/segments")
    assert r.status_code == 200
    body = r.json()
    assert len(body["segments"]) == 2
    seg0 = body["segments"][0]
    assert seg0["episode_id"] == 0
    assert seg0["segment_id"] == 0
    assert seg0["task_index"] == 2
    assert seg0["task_name"] == "flip_table"


def test_get_segments_ep_out_of_range(client):
    r = client.get("/api/episodes/999/segments")
    assert r.status_code == 404


# --- /api/tile.mp4 ---

def test_get_tile_builds_cache(client, tmp_workspace):
    r = client.get("/api/episodes/0/segments/0/tile.mp4")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    cache = tmp_workspace / "videos_cache" / "ep000" / "seg00_tile.mp4"
    assert cache.exists()


# --- /api/labels ---

def test_get_labels_empty(client):
    r = client.get("/api/labels")
    assert r.status_code == 200
    assert r.json() == {"labels": []}


def test_post_label_ok(client):
    r = client.post("/api/labels", json=_label_payload())
    assert r.status_code == 200
    got = client.get("/api/labels").json()
    assert len(got["labels"]) == 1
    assert got["labels"][0]["verdict"] == "success"
    assert got["labels"][0]["frame_start"] == 0


def test_post_label_invalid_verdict(client):
    r = client.post("/api/labels", json=_label_payload(verdict="bad"))
    assert r.status_code == 400


def test_post_label_failure_without_category(client):
    r = client.post("/api/labels", json=_label_payload(verdict="failure"))
    assert r.status_code == 400


def test_post_label_failure_with_category(client):
    r = client.post(
        "/api/labels",
        json=_label_payload(verdict="failure", failure_category="slip"),
    )
    assert r.status_code == 200


def test_post_label_upsert_by_frame_start(client):
    """同 (ep, frame_start) は自動 upsert (v4 は明示 upsert flag 廃止)。"""
    client.post("/api/labels", json=_label_payload())
    r = client.post("/api/labels", json=_label_payload(verdict="optimal"))
    assert r.status_code == 200
    got = client.get("/api/labels").json()
    assert len(got["labels"]) == 1
    assert got["labels"][0]["verdict"] == "optimal"


def test_post_label_different_frame_starts_insert_both(client):
    client.post("/api/labels", json=_label_payload(frame_start=0, frame_end=99))
    client.post("/api/labels", json=_label_payload(frame_start=100, frame_end=200))
    got = client.get("/api/labels").json()
    assert len(got["labels"]) == 2


def test_get_labels_filter_by_ep(client):
    for ep in [0, 1, 2]:
        client.post(
            "/api/labels",
            json=_label_payload(
                episode_id=ep, frame_start=ep * 100, frame_end=ep * 100 + 50,
            ),
        )
    r = client.get("/api/labels?ep_id=1")
    assert r.status_code == 200
    labels = r.json()["labels"]
    assert len(labels) == 1
    assert labels[0]["episode_id"] == 1


# --- DELETE /api/labels ---

def test_delete_label_ok(client):
    client.post("/api/labels", json=_label_payload(frame_start=0))
    client.post("/api/labels", json=_label_payload(frame_start=100, frame_end=200))
    r = client.delete("/api/labels?ep_id=0&frame_start=0")
    assert r.status_code == 200
    got = client.get("/api/labels").json()
    assert len(got["labels"]) == 1
    assert got["labels"][0]["frame_start"] == 100


def test_delete_label_not_found(client):
    r = client.delete("/api/labels?ep_id=0&frame_start=999")
    assert r.status_code == 404


# --- /api/categories ---

def test_get_categories_empty(client):
    r = client.get("/api/categories")
    assert r.status_code == 200
    assert r.json() == {"categories": []}


def test_post_category_ok(client):
    r = client.post("/api/categories", json={
        "key": "slip", "label": "滑り落ち", "added_ep": 0,
    })
    assert r.status_code == 200


def test_post_category_duplicate(client):
    client.post("/api/categories", json={"key": "slip", "label": "滑り", "added_ep": 0})
    r = client.post("/api/categories", json={"key": "slip", "label": "他", "added_ep": 1})
    assert r.status_code == 400


# --- /api/metadata ---

def test_get_metadata_default(client):
    r = client.get("/api/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["reviewer"] == "hara"
    assert body["schema_version"] == 4


def test_post_metadata_roundtrip(client):
    r = client.post("/api/metadata", json={
        "last_ep": 5, "last_frame_start": 3000, "reviewer": "hara",
        "schema_version": 4,
    })
    assert r.status_code == 200
    got = client.get("/api/metadata").json()
    assert got["last_ep"] == 5
    assert got["last_frame_start"] == 3000


# --- /api/hf ---

def test_hf_preview(client, monkeypatch):
    fake_preview = SyncPreview(
        remote_exists=False, labels_new_rows=3, labels_total_rows_local=3,
        jsonl_new_lines=3, categories_added=[], files_to_upload=["labels.parquet"],
    )
    monkeypatch.setattr(hf_sync, "compute_preview", lambda ws: fake_preview)
    r = client.post("/api/hf/preview")
    assert r.status_code == 200
    body = r.json()
    assert body["labels_new_rows"] == 3


def test_hf_sync_endpoint(client, monkeypatch):
    monkeypatch.setattr(
        hf_sync, "sync_to_hf",
        lambda ws, commit_message=None: "https://hf.co/datasets/foo/commit/abc",
    )
    r = client.post("/api/hf/sync", json={"commit_message": "test"})
    assert r.status_code == 200


def test_hf_preview_token_missing(client, monkeypatch):
    def bad_preview(ws):
        raise RuntimeError("HF_TOKEN not found")
    monkeypatch.setattr(hf_sync, "compute_preview", bad_preview)
    r = client.post("/api/hf/preview")
    assert r.status_code == 401
