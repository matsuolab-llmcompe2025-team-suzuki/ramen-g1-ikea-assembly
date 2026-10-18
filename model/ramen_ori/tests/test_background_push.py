"""background_push.py と scripts/push_ckpt_to_hf.py の unit test (Issue #141)。

- BackgroundCkptPusher (本物の子 process を、HF に触らない fake の script で起動する):
  submit は push の終わりを待たない / 次の submit は前の push を待ち、順番を保つ / wait は最後の push を待つ /
  子 process の環境変数で HF の進捗バーを止める / 失敗は WARN と failed に残し、例外は投げない / private の指定 /
  script が無ければ作るときに止める
- push_ckpt_to_hf.py: HfApi に渡す引数、成功で exit 0・失敗で WARN と exit 1 (HfApi は fake に差し替える)
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

from model.ramen_ori.background_push import PUSH_SCRIPT, BackgroundCkptPusher
from model.ramen_ori.scripts import push_ckpt_to_hf

# 引数と環境変数を 1 行で記録する fake の push script (FAKE_PUSH_SLEEP 秒待ってから書き、FAKE_PUSH_RC で終わる)
_FAKE_PUSH = """
import argparse, os, sys, time
p = argparse.ArgumentParser()
p.add_argument("--ckpt"); p.add_argument("--repo-id"); p.add_argument("--step"); p.add_argument("--public", action="store_true")
a = p.parse_args()
time.sleep(float(os.environ.get("FAKE_PUSH_SLEEP", "0")))
with open(os.environ["FAKE_PUSH_LOG"], "a") as f:
    f.write(f"{a.step} {a.repo_id} {a.ckpt} public={a.public} bars_off={os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS')}\\n")
sys.exit(int(os.environ.get("FAKE_PUSH_RC", "0")))
"""


@pytest.fixture
def fake_push(tmp_path, monkeypatch):
    script = tmp_path / "fake_push.py"
    script.write_text(_FAKE_PUSH)
    log = tmp_path / "push.log"
    monkeypatch.setenv("FAKE_PUSH_LOG", str(log))
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    return script, log


def _lines(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


class TestBackgroundCkptPusher:
    def test_submit_does_not_wait_for_push(self, fake_push, monkeypatch, tmp_path) -> None:
        script, log = fake_push
        monkeypatch.setenv("FAKE_PUSH_SLEEP", "1.5")
        pusher = BackgroundCkptPusher("org/repo", script=script)
        t0 = time.time()
        pusher.submit(tmp_path / "ckpt_step_010000.pt", step=10000)
        assert time.time() - t0 < 1.0
        assert _lines(log) == []  # まだ push の途中
        pusher.wait()
        assert _lines(log) == [f"10000 org/repo {tmp_path / 'ckpt_step_010000.pt'} public=False bars_off=1"]
        assert pusher.failed == []

    def test_next_submit_waits_for_previous_and_keeps_order(self, fake_push, monkeypatch, tmp_path) -> None:
        script, log = fake_push
        monkeypatch.setenv("FAKE_PUSH_SLEEP", "0.8")
        pusher = BackgroundCkptPusher("org/repo", script=script)
        pusher.submit(tmp_path / "ckpt_step_010000.pt", step=10000)
        t0 = time.time()
        pusher.submit(tmp_path / "ckpt_step_020000.pt", step=20000)
        # 2 本目の submit は 1 本目の終わりを待つ。1 本目は書き終わり、2 本目はまだ
        assert time.time() - t0 >= 0.5
        assert [line.split()[0] for line in _lines(log)] == ["10000"]
        pusher.wait()
        assert [line.split()[0] for line in _lines(log)] == ["10000", "20000"]

    def test_wait_without_submit_is_noop(self, fake_push, capsys) -> None:
        script, _ = fake_push
        BackgroundCkptPusher("org/repo", script=script).wait()
        assert "all pushes finished" in capsys.readouterr().out

    def test_failed_push_is_reported_not_raised(self, fake_push, monkeypatch, tmp_path, capsys) -> None:
        script, _ = fake_push
        monkeypatch.setenv("FAKE_PUSH_RC", "1")
        pusher = BackgroundCkptPusher("org/repo", script=script)
        pusher.submit(tmp_path / "ckpt_step_010000.pt", step=10000)
        pusher.submit(tmp_path / "ckpt_step_020000.pt", step=20000)
        pusher.wait()
        assert pusher.failed == ["ckpt_step_010000.pt", "ckpt_step_020000.pt"]
        out = capsys.readouterr().out
        assert "WARN: push of ckpt_step_010000.pt exited with 1" in out
        assert "failed 2" in out

    def test_public_repo_flag(self, fake_push, tmp_path) -> None:
        script, log = fake_push
        pusher = BackgroundCkptPusher("org/repo", private=False, script=script)
        pusher.submit(tmp_path / "ckpt_step_010000.pt", step=10000)
        pusher.wait()
        assert "public=True" in _lines(log)[0]

    def test_missing_script_fails_before_training(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="push script not found"):
            BackgroundCkptPusher("org/repo", script=tmp_path / "no_such_script.py")

    def test_default_script_is_the_push_script(self) -> None:
        assert PUSH_SCRIPT == Path(push_ckpt_to_hf.__file__).resolve()


class _FakeHfApi:
    calls: list[tuple[str, dict]] = []
    fail_upload = False

    def create_repo(self, **kwargs) -> None:
        self.calls.append(("create_repo", kwargs))

    def upload_file(self, **kwargs) -> None:
        if self.fail_upload:
            raise RuntimeError("boom")
        self.calls.append(("upload_file", kwargs))


@pytest.fixture
def fake_hf_api(monkeypatch):
    _FakeHfApi.calls = []
    _FakeHfApi.fail_upload = False
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_FakeHfApi))
    return _FakeHfApi


class TestPushCkptToHfScript:
    def test_pushes_with_expected_arguments(self, fake_hf_api, tmp_path, capsys) -> None:
        ckpt = tmp_path / "ckpt_step_030000.pt"
        rc = push_ckpt_to_hf.main(["--ckpt", str(ckpt), "--repo-id", "org/repo", "--step", "30000"])
        assert rc == 0
        assert fake_hf_api.calls == [
            ("create_repo", {"repo_id": "org/repo", "repo_type": "model", "exist_ok": True, "private": True}),
            ("upload_file", {
                "path_or_fileobj": str(ckpt),
                "path_in_repo": "ckpt_step_030000.pt",
                "repo_id": "org/repo",
                "repo_type": "model",
                "commit_message": "autopush ckpt at step 30000",
            }),
        ]
        assert "[hf_autopush] pushed ckpt_step_030000.pt -> org/repo" in capsys.readouterr().out

    def test_public_flag(self, fake_hf_api, tmp_path) -> None:
        push_ckpt_to_hf.main(["--ckpt", str(tmp_path / "c.pt"), "--repo-id", "org/repo", "--step", "1", "--public"])
        assert fake_hf_api.calls[0][1]["private"] is False

    def test_failure_warns_and_exits_1(self, fake_hf_api, tmp_path, capsys) -> None:
        fake_hf_api.fail_upload = True
        rc = push_ckpt_to_hf.main(["--ckpt", str(tmp_path / "c.pt"), "--repo-id", "org/repo", "--step", "1"])
        assert rc == 1
        assert "[hf_autopush] WARN: push failed (RuntimeError: boom)" in capsys.readouterr().out
