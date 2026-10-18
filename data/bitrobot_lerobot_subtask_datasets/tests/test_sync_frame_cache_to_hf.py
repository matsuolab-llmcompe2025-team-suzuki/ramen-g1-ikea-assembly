"""sync_frame_cache_to_hf.py の unit test (Issue #139)。HfApi は fake に差し替える。"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "sync_frame_cache_to_hf", SCRIPTS / "sync_frame_cache_to_hf.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sync = _load()


@dataclass
class Entry:
    path: str
    size: int | None = None  # None = folder


class FakeApi:
    """repo を {path: size} で持つ fake。"""

    def __init__(self, files: dict[str, int]) -> None:
        self.files = dict(files)
        self.upload_calls = 0
        self.commits: list = []
        self.on_upload = None

    def upload_large_folder(self, **kw) -> None:
        self.upload_calls += 1
        assert kw["repo_type"] == "dataset"
        assert "*.tmp" in kw["ignore_patterns"]
        if self.on_upload:
            self.on_upload()

    def get_paths_info(self, repo_id, paths, repo_type):
        return [Entry(p, self.files[p]) for p in paths if p in self.files]

    def list_repo_tree(self, repo_id, path_in_repo=None, repo_type=None, recursive=False):
        prefix = "" if path_in_repo is None else path_in_repo + "/"
        seen: dict[str, Entry] = {}
        for path, size in self.files.items():
            if not path.startswith(prefix):
                continue
            head = path[len(prefix):].split("/", 1)
            full = prefix + head[0]
            seen[full] = Entry(full, size) if len(head) == 1 else Entry(full, None)
        return list(seen.values())

    def create_commit(self, **kw) -> None:
        self.commits.append(kw)


def _is_folder(entry: Entry) -> bool:
    return entry.size is None


def _staging(tmp_path: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return tmp_path


def test_list_local_files_skips_resume_metadata_and_tmp(tmp_path: Path) -> None:
    st = _staging(
        tmp_path,
        {
            "frame_cache_tars/a.tar": b"12",
            "frame_cache_tars/b.tar.tmp": b"x",
            ".cache/huggingface/upload/a.tar.metadata": b"m",
        },
    )
    assert sync.list_local_files(st) == {"frame_cache_tars/a.tar": 2}


def test_verify_reports_missing_and_size_mismatch(tmp_path: Path) -> None:
    st = _staging(tmp_path, {"a/x.tar": b"123", "a/y.tar": b"12", "a/z.tar": b"1"})
    api = FakeApi({"a/x.tar": 3, "a/y.tar": 99})
    problems = sync.verify(api, repo_id="r", staging=st)
    assert problems == ["size mismatch: a/y.tar (local 2 / hub 99)", "missing on hub: a/z.tar"]


def test_upload_repeats_until_flag_then_final_round(tmp_path: Path) -> None:
    st = _staging(tmp_path / "st", {"a/x.tar": b"1"})
    flag = tmp_path / ".pack.done"
    api = FakeApi({})

    def set_flag_after_second() -> None:
        if api.upload_calls == 2:
            flag.touch()

    api.on_upload = set_flag_after_second
    sync.upload(api, repo_id="r", staging=st, num_workers=1, until_flag=flag, interval=0)
    # 2 回目の後に flag が立つ → flag を見てからの最終回 (3 回目) で終わる
    assert api.upload_calls == 3


def test_compute_stale_deletes_superseded_paths_only(tmp_path: Path) -> None:
    remote = {
        "data/chunk-000/file-000.parquet": 1,             # 管理対象外
        "obb_yolo/ce877e8237be/manifest.json": 1,
        "obb_yolo/ce877e8237be/observation.images.cam_0/chunk-000/file-000.parquet": 1,
        "obb_yolo/ce877e8237be/observation.images.cam_1/chunk-000/file-000.parquet": 1,
        "obb_yolo/3bbb5f67c017/manifest.json": 1,
        "frame_cache_tars/observation.images.head_left_chunk-000_file-000.tar": 1,
        "frame_cache_tars/observation.images.cam_0_chunk-000_file-000.tar": 1,
        "frame_cache_tars/restore.sh": 1,
        "frame_cache_v2/overlay/observation.images.cam_0/chunk-000/file-000/frame_000000_v1.jpg": 1,
        "frame_cache_v2_tars/overlay_cam_0.tar": 1,
        "frame_cache_v2_tars/overlay/observation.images.cam_0_chunk-000_file-000.tar": 1,
    }
    staged = {
        "obb_yolo/ce877e8237be/manifest.json",
        "obb_yolo/ce877e8237be/observation.images.cam_0/chunk-000/file-000.parquet",
        "frame_cache_tars/observation.images.cam_0_chunk-000_file-000.tar",
        "frame_cache_tars/restore.sh",
        "frame_cache_v2_tars/overlay/observation.images.cam_0_chunk-000_file-000.tar",
    }
    stale = sync.compute_stale(
        FakeApi(remote),
        repo_id="r",
        staged_paths=staged,
        managed_prefixes=sync.DEFAULT_MANAGED_PREFIXES,
        is_folder=_is_folder,
    )
    assert stale == [
        ("frame_cache_tars/observation.images.head_left_chunk-000_file-000.tar", False),
        ("frame_cache_v2", True),
        ("frame_cache_v2_tars/overlay_cam_0.tar", False),
        ("obb_yolo/3bbb5f67c017", True),
        ("obb_yolo/ce877e8237be/observation.images.cam_1", True),
    ]


def test_cleanup_refuses_when_verify_fails(tmp_path: Path, capsys) -> None:
    st = _staging(tmp_path, {"frame_cache_tars/a.tar": b"12"})
    api = FakeApi({})
    rc = sync.cleanup(api, repo_id="r", staging=st, managed_prefixes=("frame_cache_tars",), execute=True)
    assert rc == 1
    assert api.commits == []


def test_cleanup_dry_run_then_execute(tmp_path: Path, monkeypatch) -> None:
    st = _staging(tmp_path, {"frame_cache_tars/new.tar": b"12"})
    api = FakeApi({"frame_cache_tars/new.tar": 2, "frame_cache_tars/old.tar": 5})
    monkeypatch.setattr(sync, "_is_folder", _is_folder)
    assert sync.cleanup(api, repo_id="r", staging=st, managed_prefixes=("frame_cache_tars",), execute=False) == 0
    assert api.commits == []
    assert sync.cleanup(api, repo_id="r", staging=st, managed_prefixes=("frame_cache_tars",), execute=True) == 0
    ops = api.commits[0]["operations"]
    assert [(op.path_in_repo, op.is_folder) for op in ops] == [("frame_cache_tars/old.tar", False)]


def test_main_rejects_missing_staging(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        sync.main(["verify"])
    assert sync.main(["verify", "--repo-id", "r", "--staging", str(tmp_path / "nope")]) == 2
