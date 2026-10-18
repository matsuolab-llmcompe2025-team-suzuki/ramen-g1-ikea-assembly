from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "camera_stream_watchdog.py"
)
SPEC = spec_from_file_location("camera_stream_watchdog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
CameraLiveness = MODULE.CameraLiveness


def test_startup_grace_allows_cameras_to_appear_independently():
    health = CameraLiveness(("head", "left", "right"), 20.0, 2.0, 100.0)
    health.observe("head", 100.1)
    health.observe("left", 101.0)
    assert health.failure(119.9) is None
    assert "right" in health.failure(120.0)


def test_one_stale_stream_is_named_even_when_others_continue():
    health = CameraLiveness(("head", "left", "right"), 20.0, 2.0, 100.0)
    for role in health.roles:
        health.observe(role, 101.0)
    health.observe("head", 103.1)
    health.observe("left", 103.1)
    failure = health.failure(103.1)
    assert failure is not None
    assert "right=2.100s" in failure
    assert "head" not in failure
    assert "left" not in failure


def test_continuous_updates_remain_healthy():
    health = CameraLiveness(("head", "left", "right"), 20.0, 2.0, 100.0)
    for now_s in (100.1, 101.0, 102.0, 103.0):
        for role in health.roles:
            health.observe(role, now_s)
        assert health.failure(now_s + 0.5) is None
