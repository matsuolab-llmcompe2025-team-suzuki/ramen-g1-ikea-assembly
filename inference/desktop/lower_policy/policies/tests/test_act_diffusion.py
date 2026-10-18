"""ActDiffusionPolicy (Issue #139) の unit test。model は偽 backend で置き換える。

実 ckpt の読み込み・forward は `test_act_diffusion_integration.py` (integration mark)。
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from inference.desktop.lower_policy.policies import act_diffusion as act_diffusion_module
from inference.desktop.lower_policy.policies.act_diffusion import (
    CAMERAS,
    ActDiffusionPolicy,
    build_state_from_raw,
)
from inference.desktop.lower_policy.policies.act_diffusion_worker import serve
from inference.desktop.lower_policy.policies.groot_worker_protocol import (
    receive_archive,
    send_archive,
)
from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    Observation,
    PolicyConfig,
    RawRobotState,
)
from inference.desktop.lower_policy.policies.config_loader import resolve_policy_class
from inference.desktop.perception.yolo_obb import OBBDetection


class _FakeBackend:
    """ActDiffusionModel / worker client と同じ属性と predict_chunk を持つ偽物。"""

    def __init__(self, n_obs_steps: int = 2, chunk_len: int = 31) -> None:
        self.policy_kind = "diffusion"
        self.n_obs_steps = n_obs_steps
        self.chunk_len = chunk_len
        self.calls: list[tuple[list[dict], list[np.ndarray]]] = []
        self.closed = False

    def predict_chunk(self, frames_seq, states_seq) -> tuple[np.ndarray, float]:
        self.calls.append(([dict(f) for f in frames_seq], [s.copy() for s in states_seq]))
        # 最新観測の state[0] で埋めた chunk (どの観測で推論したかを出力から追える)
        return np.full((self.chunk_len, 19), states_seq[-1][0], dtype=np.float32), 1.0

    def close(self) -> None:
        self.closed = True


def _policy(
    mode: str = "none", backend: _FakeBackend | None = None, **cfg_overrides
) -> ActDiffusionPolicy:
    cfg = PolicyConfig(mode=mode, ckpt_ref="x", cams=CAMERAS, replan_family=None, **cfg_overrides)
    return ActDiffusionPolicy(cfg, backend or _FakeBackend())


def _obs(state_value: float, detections: list | None = None) -> Observation:
    return Observation(
        frames_bgr={cam: np.zeros((480, 640, 3), dtype=np.uint8) for cam in CAMERAS},
        frames_bgr_prev=None,
        state=np.full(19, state_value, dtype=np.float32),
        skill_id=None,
        language=None,
        obb_detections=None if detections is None else {CameraKey.HEAD_LEFT: detections},
        timestamp_ns=0,
    )


def test_policy_type_resolves_to_act_diffusion_policy() -> None:
    assert resolve_policy_class("act_diffusion") is ActDiffusionPolicy


def test_state_is_upper_body_joints_then_grippers() -> None:
    raw = RawRobotState(
        joint_positions=np.arange(29, dtype=np.float32),
        hand_state=np.array([100.0, 101.0], dtype=np.float32),
        ee_state=np.zeros(12, dtype=np.float32),
    )
    state = build_state_from_raw(raw)
    # waist 12-14 → 左腕 15-21 → 右腕 22-28 → gripper 左右 (学習の state.names 順)
    np.testing.assert_array_equal(state, [*range(12, 29), 100.0, 101.0])
    assert state.dtype == np.float32


def test_state_rejects_wrong_shapes() -> None:
    raw = RawRobotState(
        joint_positions=np.zeros(17, dtype=np.float32),
        hand_state=np.zeros(2, dtype=np.float32),
        ee_state=np.zeros(12, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="joint_positions"):
        build_state_from_raw(raw)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "precomputed_token"}, "mode"),
        ({"dtype": "bf16"}, "fp32"),
        ({"cams": tuple(CameraKey)}, "cams"),
    ],
)
def test_config_rejects_unsupported_settings(kwargs: dict, match: str) -> None:
    cfg = PolicyConfig(**{"mode": "none", "ckpt_ref": "x", "cams": CAMERAS, **kwargs})
    with pytest.raises(ValueError, match=match):
        ActDiffusionPolicy(cfg, _FakeBackend())


def test_first_tick_fills_history_with_the_first_observation() -> None:
    backend = _FakeBackend(n_obs_steps=2)
    _policy(backend=backend).predict(_obs(1.0))
    _, states = backend.calls[0]
    np.testing.assert_array_equal(states, [np.full(19, 1.0), np.full(19, 1.0)])


def test_history_is_oldest_first_and_reset_clears_it() -> None:
    backend = _FakeBackend(n_obs_steps=2)
    policy = _policy(backend=backend)
    policy.predict(_obs(1.0))
    policy.predict(_obs(2.0))
    assert [s[0] for s in backend.calls[1][1]] == [1.0, 2.0]

    policy.reset()
    policy.predict(_obs(3.0))
    assert [s[0] for s in backend.calls[2][1]] == [3.0, 3.0]


def test_predict_returns_one_blended_row() -> None:
    action = _policy(backend=_FakeBackend(chunk_len=31)).predict(_obs(0.5))
    assert action.action_chunk.shape == (1, 19)
    assert action.action_chunk.dtype == np.float32
    np.testing.assert_allclose(action.action_chunk, 0.5)
    assert action.metadata["policy_kind"] == "diffusion"
    assert action.metadata["model_chunk_len"] == 31
    assert action.metadata["chunk_source"] == "sync"


def test_camera_buffers_are_copied() -> None:
    policy = _policy()
    obs = _obs(0.0)
    policy.predict(obs)
    obs.frames_bgr[CameraKey.HEAD_LEFT][:] = 255  # camera 側が buffer を使い回す想定
    assert np.count_nonzero(policy._history[-1].frames[CameraKey.HEAD_LEFT]) == 0


def test_rejects_frames_that_differ_from_training_resolution() -> None:
    obs = _obs(0.0)
    obs.frames_bgr[CameraKey.WRIST_LEFT] = np.zeros((480, 848, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="wrist_left"):
        _policy().predict(obs)


def test_overlay_requires_live_detections() -> None:
    with pytest.raises(RuntimeError, match="YOLO-OBB"):
        _policy(mode="overlay").predict(_obs(0.0, detections=None))


@pytest.mark.parametrize("subsampling", [None, "4:4:4", "4:2:0"])
def test_overlay_head_matches_training_cache_encoding(subsampling: str | None) -> None:
    """head_left だけに overlay を描き、学習 cache と同じ jpg 保存を 1 回通す。

    None = slot で未指定 → 既定の 4:4:4 (Issue #139 の統合 cache、再学習した ACT / DP)。
    """
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    from inference.desktop.lower_policy.policies.ramen_ori import (
        match_training_jpeg,
        overlay_obb_on_frame,
    )

    detection = OBBDetection(
        class_id=4,
        class_name="table_top",
        confidence=0.9,
        verts=np.asarray([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], dtype=np.float32),
    )
    backend = _FakeBackend(n_obs_steps=1, chunk_len=100)
    obs = _obs(0.0, detections=[detection])
    overrides = {} if subsampling is None else {"overlay_jpeg_subsampling": subsampling}
    action = _policy(mode="overlay", backend=backend, **overrides).predict(obs)

    head = backend.calls[0][0][0][CameraKey.HEAD_LEFT]
    drawn = overlay_obb_on_frame(np.zeros((480, 640, 3), dtype=np.uint8), [detection])
    np.testing.assert_array_equal(head, match_training_jpeg(drawn, subsampling or "4:4:4"))
    # 4:2:0 は色差の間引きで線の青が滲む (推論時にそのまま描いた 255 より下がる)。4:4:4 は保つ
    line = drawn[..., 0] == 255
    if subsampling == "4:2:0":
        assert head[..., 0][line].mean() < 250
    else:
        assert head[..., 0][line].mean() > 250
    frames = backend.calls[0][0][0]
    assert np.count_nonzero(frames[CameraKey.WRIST_LEFT]) == 0
    assert np.count_nonzero(frames[CameraKey.WRIST_RIGHT]) == 0
    assert np.count_nonzero(obs.frames_bgr[CameraKey.HEAD_LEFT]) == 0
    assert action.metadata["overlay_detection_count"] == 1


def test_warmup_runs_the_model_then_clears_history() -> None:
    backend = _FakeBackend(n_obs_steps=2)
    policy = _policy(backend=backend)
    policy.warmup(n_iter=2)
    assert len(backend.calls) == 2
    assert len(backend.calls[0][1]) == 2
    assert len(policy._history) == 0


def test_close_closes_backend() -> None:
    backend = _FakeBackend()
    _policy(backend=backend).close()
    assert backend.closed


# ---- worker process の lifecycle ---- #


def _socket_path() -> Path:
    # AF_UNIX の path 長 (macOS 104 byte) に収まるよう /tmp 直下に作る
    return Path("/tmp") / f"actdp_test_{uuid.uuid4().hex[:8]}.sock"


def test_worker_exits_when_no_runtime_connects() -> None:
    """接続前に runtime が落ちても、GPU を掴んだ worker を残さない。"""
    model = _FakeBackend()
    path = _socket_path()
    serve(model, path, accept_timeout_s=0.2)
    assert model.closed
    assert not path.exists()


def test_worker_waits_for_the_next_request_and_ends_when_runtime_disconnects() -> None:
    """要求の間が空いても worker は落ちず (operator の gate 待ち)、接続が切れたら終わる。"""
    model = _FakeBackend(n_obs_steps=1, chunk_len=4)
    path = _socket_path()
    thread = threading.Thread(
        target=serve, args=(model, path), kwargs={"accept_timeout_s": 5.0}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 5.0
    while not path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as runtime:
        runtime.connect(str(path))
        time.sleep(0.3)
        send_archive(runtime, kind=np.asarray("describe"))
        reply = receive_archive(runtime)
        assert int(reply["ok"][0]) == 1
        assert int(reply["chunk_len"][0]) == 4
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert model.closed


class _FakeWorkerProcess:
    """`pixi run` の Popen の代わり。terminate されるまで生きている。"""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("pixi", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def test_interrupt_during_worker_startup_terminates_the_worker(monkeypatch) -> None:
    """起動待ち中の Ctrl+C でも worker を落とす (別 session なので SIGINT は届かない)。"""
    process = _FakeWorkerProcess()
    monkeypatch.setattr(act_diffusion_module.shutil, "which", lambda _: sys.executable)
    monkeypatch.setattr(act_diffusion_module.subprocess, "Popen", lambda *a, **k: process)

    def interrupted(self, timeout_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(act_diffusion_module._ActDiffusionWorkerClient, "_connect", interrupted)
    cfg = PolicyConfig(mode="overlay", ckpt_ref="x", cams=CAMERAS, replan_family=None)
    with pytest.raises(KeyboardInterrupt):
        act_diffusion_module._ActDiffusionWorkerClient(cfg)
    assert process.terminated


def test_worker_starts_with_the_pixi_env_as_is(monkeypatch) -> None:
    """worker は image に入れた環境をそのまま使う (会場はオフライン。pixi に install させない)。"""
    started: list[list[str]] = []
    process = _FakeWorkerProcess()

    def fake_popen(command, **_kwargs):
        started.append(list(command))
        return process

    monkeypatch.setattr(act_diffusion_module.shutil, "which", lambda _: sys.executable)
    monkeypatch.setattr(act_diffusion_module.subprocess, "Popen", fake_popen)

    def interrupted(self, timeout_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(act_diffusion_module._ActDiffusionWorkerClient, "_connect", interrupted)
    cfg = PolicyConfig(mode="overlay", ckpt_ref="x", cams=CAMERAS, replan_family=None)
    with pytest.raises(KeyboardInterrupt):
        act_diffusion_module._ActDiffusionWorkerClient(cfg)
    (command,) = started
    assert command[1:3] == ["run", "--as-is"]


@pytest.mark.parametrize(
    ("offline", "expected_extra"),
    [(False, {}), (True, {"local_files_only": True})],
)
def test_checkpoint_is_read_from_the_cache_only_when_offline(
    monkeypatch, tmp_path, offline: bool, expected_extra: dict
) -> None:
    """会場 (HF_HUB_OFFLINE=1) では cache だけを見る。

    worker の desktop env (huggingface_hub 1.28) は commit hash 指定でも file 一覧の記録が
    無いとネットへ取りに行く (GR00T 53D で GB10 で起きた、2026-09-25)。
    """
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", offline)
    resolved = act_diffusion_module._resolve_checkpoint_root("Team-RAMEN/model@deadbeef", None)
    assert resolved == snapshot.resolve()
    assert calls == [{
        "repo_id": "Team-RAMEN/model",
        "revision": "deadbeef",
        "ignore_patterns": ["checkpoints/**"],
        **expected_extra,
    }]


def test_backbone_is_not_downloaded_when_the_checkpoint_is_built(monkeypatch, tmp_path) -> None:
    """ckpt の config は ImageNet の backbone 重みを指すが、会場はオフラインなので取りに行かせない。

    lerobot は policy を作る時点で torchvision に `weights=config.pretrained_backbone_weights`
    を渡す。その後 state dict (strict=True) が backbone ごと上書きする。
    """
    import types

    built_with: list[object] = []

    class _Config:
        type = "diffusion"
        device = "cpu"
        horizon = 32
        n_obs_steps = 2
        n_action_steps = 8
        pretrained_backbone_weights = "ResNet18_Weights.IMAGENET1K_V1"

    class _PolicyClass:
        @classmethod
        def from_pretrained(cls, _root, *, config, strict):
            assert strict is True
            built_with.append(config.pretrained_backbone_weights)
            policy = cls()
            policy.config = config
            return policy

        def eval(self):
            return self

    configs = types.ModuleType("lerobot.configs.policies")
    configs.PreTrainedConfig = types.SimpleNamespace(from_pretrained=lambda _root: _Config())
    factory = types.ModuleType("lerobot.policies.factory")
    factory.get_policy_class = lambda _kind: _PolicyClass
    factory.make_pre_post_processors = lambda *_a, **_k: (None, None)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", configs)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory)
    monkeypatch.setattr(act_diffusion_module, "_resolve_checkpoint_root", lambda *_a: tmp_path)
    monkeypatch.setattr(act_diffusion_module, "_validate_checkpoint_config", lambda _c: None)

    act_diffusion_module.ActDiffusionModel.from_checkpoint("x", device="cpu")
    assert built_with == [None]


def test_unanswered_request_times_out_and_drops_the_connection() -> None:
    """worker が hang したら数秒で例外にし、応答とずれうる接続は使わない。"""
    client = object.__new__(act_diffusion_module._ActDiffusionWorkerClient)
    runtime_end, worker_end = socket.socketpair()
    client._connection = runtime_end
    client._request_lock = threading.Lock()
    try:
        with pytest.raises(TimeoutError, match="did not answer"):
            client._request(timeout_s=0.1, kind=np.asarray("predict"))
        assert client._connection is None
    finally:
        worker_end.close()
