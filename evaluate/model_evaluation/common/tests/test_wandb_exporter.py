"""wandb_exporter の eligibility gate unit test (upload 実行は含まない)。"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from evaluate.model_evaluation.common.wandb_exporter import (
    UploadDecision,
    evaluate_eligibility,
    upload_run,
    write_upload_status,
)


class TestEvaluateEligibility:
    def test_all_conditions_met(self):
        d = evaluate_eligibility(
            actuated=True, policy_seconds=15.0, tick_count=200, exit_code=0
        )
        assert d.eligible is True
        assert d.reasons == ()

    def test_not_actuated_blocks(self):
        d = evaluate_eligibility(
            actuated=False, policy_seconds=15.0, tick_count=200, exit_code=0
        )
        assert d.eligible is False
        assert any("not actuated" in r for r in d.reasons)

    def test_too_short_policy_blocks(self):
        d = evaluate_eligibility(
            actuated=True, policy_seconds=3.0, tick_count=90, exit_code=0
        )
        assert d.eligible is False
        assert any("policy ran only" in r for r in d.reasons)

    def test_too_few_ticks_blocks(self):
        d = evaluate_eligibility(
            actuated=True, policy_seconds=15.0, tick_count=5, exit_code=0
        )
        assert d.eligible is False
        assert any("only 5 tick" in r for r in d.reasons)

    def test_nonzero_exit_is_uploadable_for_long_physical_run(self):
        d = evaluate_eligibility(
            actuated=True,
            policy_seconds=2.0,
            tick_count=20,
            exit_code=1,
            requested_seconds=30.0,
        )
        assert d.eligible is True
        assert d.reasons == ()

    def test_multiple_reasons_collected(self):
        d = evaluate_eligibility(
            actuated=False,
            policy_seconds=0.0,
            tick_count=0,
            exit_code=1,
            requested_seconds=1.0,
        )
        assert d.eligible is False
        assert len(d.reasons) == 3

    def test_intentional_one_second_smoke_is_not_uploaded_when_it_fails(self):
        d = evaluate_eligibility(
            actuated=True,
            policy_seconds=0.5,
            tick_count=10,
            exit_code=1,
            requested_seconds=1.0,
        )
        assert d.eligible is False
        assert any("short smoke" in reason for reason in d.reasons)

    def test_failure_before_first_policy_command_is_not_uploaded(self):
        d = evaluate_eligibility(
            actuated=True,
            policy_seconds=0.0,
            tick_count=0,
            exit_code=1,
            requested_seconds=30.0,
        )
        assert d.eligible is False
        assert any("before any policy command" in reason for reason in d.reasons)


class TestWriteUploadStatus:
    def test_writes_eligible_true_with_url(self, tmp_path):
        write_upload_status(
            tmp_path,
            decision=UploadDecision(eligible=True),
            url="https://wandb.ai/example/run/abc123",
        )
        payload = json.loads((tmp_path / "wandb_upload.json").read_text())
        assert payload["eligible"] is True
        assert payload["uploaded"] is True
        assert payload["status"] == "uploaded"
        assert payload["reasons"] == []
        assert payload["wandb_url"] == "https://wandb.ai/example/run/abc123"

    def test_writes_ineligible_with_reasons(self, tmp_path):
        write_upload_status(
            tmp_path,
            decision=UploadDecision(
                eligible=False, reasons=("not actuated", "too short")
            ),
            url=None,
        )
        payload = json.loads((tmp_path / "wandb_upload.json").read_text())
        assert payload["eligible"] is False
        assert payload["uploaded"] is False
        assert payload["status"] == "ineligible"
        assert payload["reasons"] == ["not actuated", "too short"]
        assert payload["wandb_url"] is None


def test_failed_long_run_uploads_partial_video_and_state_action(monkeypatch, tmp_path):
    (tmp_path / "capture/videos").mkdir(parents=True)
    video = tmp_path / "capture/videos/head_left.mp4"
    video.write_bytes(b"diagnostic video")
    (tmp_path / "capture/video_manifest.json").write_text(
        json.dumps(
            {
                "complete": False,
                "videos": {
                    "head_left": {
                        "relative_path": "capture/videos/head_left.mp4"
                    }
                },
                "errors": ["right_wrist: fewer than two frames"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "states.jsonl").write_text(
        json.dumps(
            {
                "t": 1,
                "joint_position": [0.0] * 29,
                "hand_position_rad": [1.0, 2.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "actions.jsonl").write_text(
        json.dumps({"t": 1, "chunk_0": [0.1] * 19, "latency_ms": 4.0})
        + "\n",
        encoding="utf-8",
    )

    class FakeRun:
        def __init__(self):
            self.logged = []
            self.summary = {}

        def log(self, value):
            self.logged.append(value)

        def get_url(self):
            return "https://wandb.ai/fake/run/1"

    run = FakeRun()
    initialized = {}
    finished = []
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: initialized.update(kwargs) or run,
        save=lambda *args, **kwargs: None,
        Video=lambda path, format: (path, format),
        Table=lambda columns, data: {"columns": columns, "data": data},
        finish=lambda **kwargs: finished.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    url = upload_run(
        tmp_path,
        metadata={
            "runner_exit_code": 1,
            "runner_status": "failed",
            "policy_seconds": 20.0,
            "tick_count": 300,
        },
        run_name="failed-diagnostic",
    )

    assert url == "https://wandb.ai/fake/run/1"
    assert "runner-failed" in initialized["tags"]
    assert any("camera/head_left" in item for item in run.logged)
    assert any("telemetry/state_action_trace" in item for item in run.logged)
    assert run.summary["runner/exit_code"] == 1
    assert finished == [{"exit_code": 1}]


def test_namespace_package_named_wandb_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(__file__=None))

    assert upload_run(tmp_path, metadata={}, run_name="invalid-sdk") is None
