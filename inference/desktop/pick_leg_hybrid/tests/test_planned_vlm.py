"""PlannedVlm の unit test (Issue #188 ② 段 2)。

`--gpu-models plan` で、VLM を読む順番の 1 本として起動する。手順は今の起動時と同じで、
途中で失敗したら server を止めてから例外を返す (pick の前の保持が止める)。
"""

from __future__ import annotations

import pytest

from inference.desktop.pick_leg_hybrid.planned_vlm import PlannedVlm


class _Server:
    def __init__(self, calls: list, *, fail_at: str | None = None) -> None:
        self.calls = calls
        self.fail_at = fail_at

    def start(self) -> None:
        self.calls.append("start")
        if self.fail_at == "start":
            raise RuntimeError("port in use")

    def wait_until_ready(self) -> float:
        self.calls.append("wait")
        if self.fail_at == "wait":
            raise RuntimeError("the VLM server exited while loading")
        return 1.0

    def close(self) -> None:
        self.calls.append("close")


def _vlm(calls: list, *, fail_at: str | None = None) -> PlannedVlm:
    def warm_up(cfg, images):
        calls.append(("warm_up", images))
        if fail_at == "warm_up":
            raise RuntimeError("warm-up failed")

    def probe(cfg, references):
        calls.append("probe")
        if fail_at == "probe":
            raise RuntimeError("answered wrong")
        return {"model": "qwen", "multimodal_latency_sec": 0.8}

    return PlannedVlm(
        _Server(calls, fail_at=fail_at),
        cfg="cfg",
        references=["a", "b"],
        warm_up=warm_up,
        probe=probe,
        self_check_images=lambda cfg, references: f"images({len(references)})",
    )


def test_prepare_runs_the_same_steps_as_the_startup_did(capsys):
    calls = []
    vlm = _vlm(calls)
    assert not vlm.is_loaded
    vlm.prepare()
    assert calls == ["start", "wait", ("warm_up", "images(2)"), "probe"]
    assert vlm.is_loaded
    assert vlm.endpoint == {"model": "qwen", "multimodal_latency_sec": 0.8}
    assert "VLM ready: model=qwen" in capsys.readouterr().err
    vlm.prepare()  # 2 回目は何もしない
    assert calls.count("start") == 1


@pytest.mark.parametrize("fail_at", ["start", "wait", "warm_up", "probe"])
def test_a_failure_stops_the_server_and_raises(fail_at):
    calls = []
    vlm = _vlm(calls, fail_at=fail_at)
    with pytest.raises(RuntimeError):
        vlm.prepare()
    assert calls[-1] == "close"
    assert not vlm.is_loaded


def test_close_stops_the_server_and_can_be_called_twice():
    calls = []
    vlm = _vlm(calls)
    vlm.prepare()
    vlm.close()
    vlm.close()
    assert not vlm.is_loaded
    assert calls.count("close") == 2
