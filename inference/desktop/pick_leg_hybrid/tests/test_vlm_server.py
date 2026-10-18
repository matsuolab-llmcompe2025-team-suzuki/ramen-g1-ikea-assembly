"""会場で VLM server を run ごとに立てる部分 (vlm_server.py) の test。

本物の vLLM の代わりに、engine のように孫 process を作る小さな HTTP server を起動する。
"""

from __future__ import annotations

import http.server
import json
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from inference.desktop import entrypoint
from inference.desktop.pick_leg_hybrid.config import DEFAULT_CONFIG_PATH
from inference.desktop.pick_leg_hybrid.real_skill import (
    build_vlm_self_check_images,
    load_reference_images,
)
from inference.desktop.pick_leg_hybrid.vlm import GRASP_PHASE_TEXT
from inference.desktop.pick_leg_hybrid.vlm_server import (
    VENUE_VLM_HOST,
    VENUE_VLM_PORT,
    VENUE_VLM_SCRIPT,
    VenueVlmServer,
    check_venue_endpoint,
    warm_up_vlm,
)

# 引数: port / 待ち秒 / "serve" か "crash" / 孫 process の pid を書く file
FAKE_VLM_SERVER = r"""
import http.server, subprocess, sys, time
port, delay, mode, pid_file = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
print("fake vlm: loading", flush=True)
engine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
open(pid_file, "w").write(str(engine.pid))
if mode == "crash":
    print("fake vlm: CUDA error during load", flush=True)
    sys.exit(3)
time.sleep(delay)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass

http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _gone(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _fake_server(tmp_path: Path, *, delay: float = 0.0, mode: str = "serve"):
    script = tmp_path / "fake_vlm.py"
    script.write_text(FAKE_VLM_SERVER)
    pid_file = tmp_path / "engine.pid"
    port = _free_port()
    server = VenueVlmServer(
        tmp_path / "logs" / "vlm.log",
        command=[
            sys.executable,
            str(script),
            str(port),
            str(delay),
            mode,
            str(pid_file),
        ],
        port=port,
        poll_interval_s=0.05,
        progress_interval_s=0.3,
        stop_timeout_s=2.0,
    )
    return server, pid_file


def _engine_pid(pid_file: Path) -> int:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if pid_file.exists() and pid_file.read_text():
            return int(pid_file.read_text())
        time.sleep(0.02)
    raise AssertionError("fake engine pid was never written")


def test_waits_for_a_slow_load_without_a_time_limit(tmp_path, capsys) -> None:
    server, _ = _fake_server(tmp_path, delay=1.0)
    try:
        server.start()
        elapsed = server.wait_until_ready()
    finally:
        server.close()
    assert elapsed >= 1.0
    err = capsys.readouterr().err
    assert "loading..." in err
    assert "server ready" in err
    assert str(tmp_path / "logs" / "vlm.log") in err


def test_a_server_that_exits_while_loading_stops_the_run_with_its_log(tmp_path) -> None:
    server, _ = _fake_server(tmp_path, mode="crash")
    errors: list[RuntimeError] = []

    def wait() -> None:
        try:
            server.wait_until_ready()
        except RuntimeError as exc:
            errors.append(exc)

    # 待ちには時間制限が無いので、見落とすと test が止まったままになる。別 thread で見張る
    waiter = threading.Thread(target=wait, daemon=True)
    try:
        server.start()
        waiter.start()
        waiter.join(10.0)
    finally:
        server.close()
    assert not waiter.is_alive(), "kept waiting after the server exited"
    (error,) = errors
    assert "exit_code=3" in str(error)
    assert "CUDA error during load" in str(error)


def test_refuses_to_start_when_the_port_is_already_taken(tmp_path) -> None:
    server, pid_file = _fake_server(tmp_path)
    with socket.socket() as other:
        other.bind(("127.0.0.1", server.port))
        other.listen(1)
        with pytest.raises(RuntimeError, match="already in use"):
            server.start()
    server.close()
    assert not pid_file.exists()


def test_close_stops_the_server_and_its_engine_process(tmp_path) -> None:
    server, pid_file = _fake_server(tmp_path)
    server.start()
    server.wait_until_ready()
    leader = server._process.pid
    engine = _engine_pid(pid_file)
    server.close()
    assert _gone(leader)
    assert _gone(engine)
    server.close()  # 2 回目は何もしない


def test_close_after_a_crash_still_stops_the_leftover_engine(tmp_path) -> None:
    server, pid_file = _fake_server(tmp_path, mode="crash")
    server.start()
    with pytest.raises(RuntimeError):
        server.wait_until_ready()
    engine = _engine_pid(pid_file)
    assert not _gone(engine, timeout_s=0.2)
    server.close()
    assert _gone(engine)


def test_the_venue_endpoint_matches_the_script_and_the_yaml() -> None:
    text = VENUE_VLM_SCRIPT.read_text()
    assert re.search(r"--host (\S+)", text).group(1) == VENUE_VLM_HOST
    assert int(re.search(r"--port (\d+)", text).group(1)) == VENUE_VLM_PORT
    cfg, _ = load_reference_images(DEFAULT_CONFIG_PATH)
    check_venue_endpoint(cfg.vlm.endpoint)
    for other in (
        "http://127.0.0.1:8001/v1/chat/completions",
        "http://10.0.0.5:8000/v1/chat/completions",
        "https://127.0.0.1:8000/v1/chat/completions",
    ):
        with pytest.raises(ValueError, match="--spawn-vlm-server"):
            check_venue_endpoint(other)


class _ChatServer:
    """chat completions を 1 つ受ける server。delay 秒待ってから status を返す。"""

    def __init__(self, *, delay: float, status: int = 200, body: str = "") -> None:
        self.requests: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(json.loads(self.rfile.read(length)))
                time.sleep(delay)
                reply = body or json.dumps(
                    {"choices": [{"message": {"content": '{"answer": 1}'}}]}
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(reply.encode("utf-8"))

            def log_message(self, *args):
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = (
            f"http://127.0.0.1:{self._httpd.server_port}/v1/chat/completions"
        )
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _cfg_for(endpoint: str, timeout_sec: float):
    cfg, references = load_reference_images(
        DEFAULT_CONFIG_PATH, endpoint_override=endpoint
    )
    return replace(cfg, vlm=replace(cfg.vlm, timeout_sec=timeout_sec)), references


def test_warm_up_sends_the_production_request_and_outlasts_the_request_timeout() -> (
    None
):
    chat = _ChatServer(delay=0.6)
    try:
        cfg, references = _cfg_for(chat.endpoint, timeout_sec=0.2)
        images = build_vlm_self_check_images(cfg, references)
        elapsed = warm_up_vlm(cfg, images)
    finally:
        chat.close()
    assert elapsed >= 0.6 > cfg.vlm.timeout_sec
    (request,) = chat.requests
    assert request["model"] == cfg.vlm.model
    assert request["max_tokens"] == cfg.vlm.max_tokens
    user = request["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in user) == len(images) == 5
    assert GRASP_PHASE_TEXT in user[-1]["text"]


def test_warm_up_reports_progress_while_it_waits_and_stops_after(capsys) -> None:
    """時間制限なしの間も経過を出す (詰まったときに待つか止めるかを決められるように)。"""
    chat = _ChatServer(delay=0.35)
    try:
        cfg, references = _cfg_for(chat.endpoint, timeout_sec=5.0)
        warm_up_vlm(
            cfg, build_vlm_self_check_images(cfg, references), progress_interval_s=0.1
        )
        after_return = capsys.readouterr().err
        time.sleep(0.3)
    finally:
        chat.close()
    assert "[vlm] warm-up still running..." in after_return
    assert "warm-up still running" not in capsys.readouterr().err  # 終わったら止まる


def test_warm_up_failure_is_raised_with_the_server_message() -> None:
    chat = _ChatServer(
        delay=0.0, status=400, body='{"error": "At most 4 image(s) may be provided"}'
    )
    try:
        cfg, references = _cfg_for(chat.endpoint, timeout_sec=5.0)
        with pytest.raises(RuntimeError, match="At most 4 image"):
            warm_up_vlm(cfg, build_vlm_self_check_images(cfg, references))
    finally:
        chat.close()


def test_the_entrypoint_spawns_the_vlm_only_when_asked(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["entrypoint", "--stage", "2"])
    assert entrypoint.parse_args().spawn_vlm_server is False
    monkeypatch.setattr(
        sys, "argv", ["entrypoint", "--stage", "2", "--spawn-vlm-server"]
    )
    assert entrypoint.parse_args().spawn_vlm_server is True


#: 会場の VLM がネットに出ないための環境変数 (vLLM 0.29.0 の serve 経路の出口)。
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "FLASHINFER_CUBINS_REPOSITORY": "http://127.0.0.1:9/",
    "VLLM_HOST_IP": "127.0.0.1",
    "HTTP_PROXY": "http://127.0.0.1:9",
    "HTTPS_PROXY": "http://127.0.0.1:9",
    "ALL_PROXY": "http://127.0.0.1:9",
    "http_proxy": "http://127.0.0.1:9",
    "https_proxy": "http://127.0.0.1:9",
    "all_proxy": "http://127.0.0.1:9",
    "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost",
}


def _run_venue_script(tmp_path: Path, **extra_env: str) -> dict:
    """偽の pixi / vllm で本物の script を走らせ、vllm が受け取った env と引数を返す。"""
    pixi_bin = tmp_path / "pixi_bin"
    vlm_env_bin = tmp_path / "vlm_env_bin"
    pixi_bin.mkdir()
    vlm_env_bin.mkdir()
    pixi_args = tmp_path / "pixi_args.txt"
    fake_pixi = pixi_bin / "pixi"
    fake_pixi.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" > {pixi_args}\n'
        f"echo 'export PATH=\"{vlm_env_bin}:$PATH\"'\n"
    )
    fake_vllm = vlm_env_bin / "vllm"
    fake_vllm.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        'print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "env": dict(os.environ)}))\n'
    )
    for path in (fake_pixi, fake_vllm):
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    env = {"PATH": f"{pixi_bin}:/usr/bin:/bin", "HOME": str(tmp_path), **extra_env}
    result = subprocess.run(
        ["bash", str(VENUE_VLM_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return {"pixi_args": pixi_args.read_text().split(), **json.loads(result.stdout)}


def test_the_venue_script_starts_vllm_offline_with_the_venue_settings(tmp_path) -> None:
    ran = _run_venue_script(tmp_path)
    env, argv = ran["env"], ran["argv"]
    for name, value in OFFLINE_ENV.items():
        assert env.get(name) == value, name
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    # vlm 環境を install も lock の更新もせずに有効にする
    assert ran["pixi_args"][:1] == ["shell-hook"]
    assert "--as-is" in ran["pixi_args"]
    assert ran["pixi_args"][ran["pixi_args"].index("-e") + 1] == "vlm"
    assert Path(ran["cwd"]) == VENUE_VLM_SCRIPT.parent.parent

    def option(name: str) -> str:
        return argv[argv.index(name) + 1]

    assert argv[:2] == ["serve", "Qwen/Qwen3-VL-8B-Instruct"]
    assert option("--host") == VENUE_VLM_HOST
    assert int(option("--port")) == VENUE_VLM_PORT
    assert option("--allowed-media-domains") == "127.0.0.1"
    assert option("--kv-cache-memory-bytes") == "1G"
    assert option("--gpu-memory-utilization") == "0.01"
    assert option("--attention-backend") == "FLASHINFER"
    assert "--enforce-eager" in argv


def test_the_venue_script_keeps_compile_caches_under_one_mountable_dir(tmp_path) -> None:
    cache = tmp_path / "host_cache"
    env = _run_venue_script(tmp_path, RAMEN_VLM_CACHE_DIR=str(cache))["env"]
    assert env["TRITON_CACHE_DIR"] == str(cache / "triton")
    assert env["VLLM_CACHE_ROOT"] == str(cache / "vllm")
    assert env["FLASHINFER_WORKSPACE_BASE"] == str(cache)
