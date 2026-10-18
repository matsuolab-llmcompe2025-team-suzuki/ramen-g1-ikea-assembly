"""act_diffusion/ckpt_uploader.py の同期計画 (Issue #139)。network は使わない。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from model.subtask_policy_training.act_diffusion import ckpt_uploader as up


def make_checkpoint(checkpoints_dir: Path, step: str) -> Path:
    """validate_model_dir を通る最小の LeRobot checkpoint を作る。"""
    model_dir = checkpoints_dir / step / "pretrained_model"
    model_dir.mkdir(parents=True)
    for name in ("config.json", "train_config.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (model_dir / name).write_text(
            json.dumps({"steps": [{"registry_name": "device_processor", "config": {}}]}),
            encoding="utf-8",
        )
    state_dir = checkpoints_dir / step / "training_state"
    state_dir.mkdir()
    (state_dir / "optimizer_state.safetensors").write_bytes(b"adam")
    (state_dir / "training_step.json").write_text(json.dumps({"step": int(step)}), encoding="utf-8")
    return model_dir


def point_last(checkpoints_dir: Path, step: str) -> None:
    link = checkpoints_dir / "last"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(step)  # LeRobot の update_last_checkpoint と同じく相対 symlink


def repo_paths(plan: up.Plan) -> list[str]:
    return [repo_path for _, repo_path in plan.adds]


def test_no_checkpoint_until_last_link_exists(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, "010000")
    # save_checkpoint 直後、update_last_checkpoint 前の状態 = まだ完了扱いにしない
    assert up.completed_steps(tmp_path) == []
    assert up.plan_sync(tmp_path, up.SyncState()).is_empty()


def test_in_progress_step_newer_than_last_is_skipped(tmp_path: Path) -> None:
    for step in ("010000", "020000", "030000"):
        make_checkpoint(tmp_path, step)
    point_last(tmp_path, "020000")  # 030000 は書き込み中

    plan = up.plan_sync(tmp_path, up.SyncState())

    paths = repo_paths(plan)
    assert not any("030000" in p for p in paths)
    # 全完了 step の weights、training_state は最新 (020000) だけ
    assert "checkpoints/010000/pretrained_model/model.safetensors" in paths
    assert "checkpoints/020000/pretrained_model/model.safetensors" in paths
    assert "checkpoints/020000/training_state/optimizer_state.safetensors" in paths
    assert not any(p.startswith("checkpoints/010000/training_state/") for p in paths)
    assert plan.deletes == []
    assert plan.state.uploaded_steps == ["010000", "020000"]
    assert plan.state.training_state_step == "020000"


def test_new_latest_moves_training_state_and_deletes_old_one(tmp_path: Path) -> None:
    for step in ("010000", "020000"):
        make_checkpoint(tmp_path, step)
    point_last(tmp_path, "020000")
    first = up.plan_sync(tmp_path, up.SyncState())

    make_checkpoint(tmp_path, "030000")
    point_last(tmp_path, "030000")
    second = up.plan_sync(tmp_path, first.state)

    paths = repo_paths(second)
    # 既に上げた step の weights は再送しない
    assert not any(p.startswith(("checkpoints/010000/", "checkpoints/020000/")) for p in paths)
    assert "checkpoints/030000/pretrained_model/model.safetensors" in paths
    assert "checkpoints/030000/training_state/training_step.json" in paths
    assert second.deletes == ["checkpoints/020000/training_state/"]
    assert second.state.training_state_step == "030000"


def test_synced_state_produces_empty_plan(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, "010000")
    point_last(tmp_path, "010000")
    first = up.plan_sync(tmp_path, up.SyncState())

    assert up.plan_sync(tmp_path, first.state).is_empty()


def test_invalid_checkpoint_is_rejected_before_upload(tmp_path: Path) -> None:
    model_dir = make_checkpoint(tmp_path, "010000")
    (model_dir / "model.safetensors").unlink()
    point_last(tmp_path, "010000")

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        up.plan_sync(tmp_path, up.SyncState())


def test_final_root_replaces_only_root_model_files(tmp_path: Path) -> None:
    for step in ("010000", "020000"):
        make_checkpoint(tmp_path, step)
    point_last(tmp_path, "020000")
    remote = {
        ".gitattributes",
        "README.md",
        "model.safetensors",          # 前回の root model → 上書き
        "stale_processor.json",       # 最新 model に無い root file → 削除
        "checkpoints/010000/pretrained_model/model.safetensors",
        "checkpoints/020000/training_state/training_step.json",
    }

    plan = up.plan_final_root(tmp_path, remote)

    local = {p.relative_to(tmp_path / "020000" / "pretrained_model").as_posix() for p, _ in plan.adds}
    assert set(repo_paths(plan)) == local
    # checkpoints/ 配下と README / .gitattributes は残す
    assert plan.deletes == ["stale_processor.json"]


def test_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "hf_upload_state.json"
    up.SyncState(["020000", "010000"], "020000").save(path)

    loaded = up.SyncState.load(path)

    assert loaded.uploaded_steps == ["010000", "020000"]
    assert loaded.training_state_step == "020000"
    assert not path.with_suffix(".tmp").exists()


def test_last_link_is_relative_like_lerobot(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, "010000")
    point_last(tmp_path, "010000")

    assert os.readlink(tmp_path / "last") == "010000"
    assert up.completed_steps(tmp_path) == ["010000"]


def test_existing_deletes_keeps_only_paths_present_on_hub() -> None:
    # 前回の commit が Hub に届いた後の再試行では、消し済みの training_state がもう無い
    remote = {
        "checkpoints/010000/training_state/optimizer_state.safetensors",
        "checkpoints/010000/pretrained_model/model.safetensors",
        "old_processor.json",
    }
    deletes = [
        "checkpoints/010000/training_state/",
        "checkpoints/020000/training_state/",
        "old_processor.json",
        "gone.json",
    ]

    assert up.existing_deletes(deletes, remote) == [
        "checkpoints/010000/training_state/",
        "old_processor.json",
    ]


class _FakeHfApi:
    """HfApi の代わり。最初の create_repo だけ一時的な network 失敗にする。"""

    def __init__(self) -> None:
        self.create_repo_calls = 0
        self.commits: list[list] = []
        self.files = ["README.md", "checkpoints/010000/training_state/training_step.json"]

    def create_repo(self, *args, **kwargs) -> None:
        self.create_repo_calls += 1
        if self.create_repo_calls == 1:
            raise ConnectionError("temporary network failure")

    def list_repo_files(self, *args, **kwargs) -> list[str]:
        return list(self.files)

    def create_commit(self, *, operations, **kwargs) -> None:
        self.commits.append(operations)


def test_repo_is_created_on_first_hub_access_and_retried(monkeypatch) -> None:
    """起動時の一時的な失敗で uploader ごと落ちず、次の周回で作り直す。"""
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)
    uploader = up.Uploader("org/repo", private=True, dry_run=False)
    assert uploader._api.create_repo_calls == 0

    with pytest.raises(ConnectionError):
        uploader.remote_files()
    assert uploader.remote_files() == set(_FakeHfApi().files)
    assert uploader._api.create_repo_calls == 2


class _NewRepoHfApi(_FakeHfApi):
    """まだ repo が無い Hub。create_repo の前の create_commit は 404 で失敗する。"""

    def __init__(self) -> None:
        super().__init__()
        self.files = []
        self.created = False

    def create_repo(self, *args, **kwargs) -> None:
        self.create_repo_calls += 1
        self.created = True

    def create_commit(self, *, operations, **kwargs) -> None:
        if not self.created:
            raise RuntimeError("404 Repository Not Found")
        super().create_commit(operations=operations, **kwargs)


def test_first_sync_to_a_new_repo_creates_the_repo(monkeypatch, tmp_path: Path) -> None:
    """新規 repo の最初の plan (delete 無し) でも repo を作ってから commit する。"""
    monkeypatch.setattr("huggingface_hub.HfApi", _NewRepoHfApi)
    checkpoints = tmp_path / "checkpoints"
    make_checkpoint(checkpoints, "010000")
    point_last(checkpoints, "010000")
    uploader = up.Uploader("org/new_repo", private=True, dry_run=False)

    state = up.sync_once(uploader, checkpoints, None, up.SyncState())

    assert state.uploaded_steps == ["010000"]
    assert uploader._api.create_repo_calls == 1
    assert len(uploader._api.commits) == 1


def test_commit_drops_deletes_that_are_already_gone(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)
    uploader = up.Uploader("org/repo", private=True, dry_run=False)
    uploader._repo_ready = True
    added = tmp_path / "training_step.json"
    added.write_text("{}", encoding="utf-8")
    plan = up.Plan(
        adds=[(added, "checkpoints/020000/training_state/training_step.json")],
        deletes=["checkpoints/010000/training_state/", "checkpoints/005000/training_state/"],
    )

    uploader.commit(plan, "sync")

    deleted = [op.path_in_repo for op in uploader._api.commits[0] if hasattr(op, "is_folder")]
    assert deleted == ["checkpoints/010000/training_state/"]


def test_final_publishes_root_even_if_checkpoint_sync_fails(monkeypatch, tmp_path: Path) -> None:
    """途中 ckpt の同期に失敗しても、repo root の最新 model は置きにいく (status 1 で知らせる)。"""
    checkpoints = tmp_path / "checkpoints"
    make_checkpoint(checkpoints, "010000")
    point_last(checkpoints, "010000")
    messages: list[str] = []

    class _FlakyUploader:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def remote_files(self) -> set[str]:
            return set()

        def commit(self, plan: up.Plan, message: str) -> None:
            messages.append(message)
            if message.startswith("sync checkpoints"):
                raise RuntimeError("commit failed")

    monkeypatch.setattr(up, "Uploader", _FlakyUploader)

    status = up.main(["--checkpoints-dir", str(checkpoints), "--repo-id", "org/repo", "--final"])

    assert status == 1
    assert messages == ["sync checkpoints up to 010000", "publish latest checkpoint at repo root"]
