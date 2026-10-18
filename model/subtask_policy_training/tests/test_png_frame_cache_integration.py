"""Issue #168: actual LeRobot/PyAV parity and launch wiring (CPU only)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
FEATURE_ROOT = ROOT / "model/subtask_policy_training"
GENERATOR = ROOT / "data/bitrobot_lerobot_subtask_datasets/scripts/precompute_frame_cache.py"


def test_png_cache_matches_real_lerobot_pyav_decoder(tmp_path, monkeypatch):
    import json

    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("av")
    pytest.importorskip("lerobot.datasets.video_utils")

    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    key = "observation.images.head_left"
    (dataset / "meta/info.json").write_text(json.dumps({
        "fps": 30, "features": {key: {"dtype": "video", "shape": [24, 32, 3]}},
    }))
    video = dataset / "videos" / key / "chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 24))
    if not writer.isOpened():
        pytest.skip("This environment cannot encode an MP4 test fixture")
    rng = np.random.default_rng(168)
    for _ in range(6):
        writer.write(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))
    writer.release()

    spec = importlib.util.spec_from_file_location("png_cache_generator_e2e", GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, generator)
    spec.loader.exec_module(generator)
    monkeypatch.setattr(sys, "argv", [
        str(GENERATOR), "--lerobot-root", str(dataset), "--image-format", "png",
        "--jobs", "1", "--keep-source-mp4",
    ])
    assert generator.main() == 0
    assert video.is_file()

    monkeypatch.setenv("LEROBOT_FRAME_CACHE_ENABLE", "true")
    monkeypatch.setenv("FRAME_CACHE_IMAGE_FORMAT", "png")
    monkeypatch.setenv("FRAME_CACHE_ROOT_OVERRIDE", str(dataset / "frame_cache_png"))
    monkeypatch.setenv("FRAME_CACHE_NUM_VARIANTS", "1")
    monkeypatch.setenv("FRAME_CACHE_STRICT", "true")
    # A fresh process is intentional: nearby unit tests emulate LeRobot modules
    # and patch module-local aliases. This check must use the installed decoder.
    result = subprocess.run([sys.executable, "-c", '''
import importlib
import sys
import torch
from lerobot.datasets import video_utils
from model.subtask_policy_training.scripts import lerobot_frame_cache_patch as cache

video = sys.argv[1]
times = [2 / 30, 1 / 30, 0.0, 2 / 30, 5 / 30]
references = {
    uint8: video_utils.decode_video_frames(
        video, times, 0.001, backend="pyav", return_uint8=uint8,
    ) for uint8 in (False, True)
}
assert cache.apply_patch()
def unexpected_fallback(*args, **kwargs):
    raise AssertionError("PNG integration test fell back to video decoding")
cache._original_decode = unexpected_fallback
for uint8, expected in references.items():
    actual = video_utils.decode_video_frames(
        video, times, 0.001, backend="pyav", return_uint8=uint8,
    )
    assert torch.equal(actual, expected)
checked_aliases = 0
for name in ("lerobot.datasets.dataset_reader", "lerobot.datasets.lerobot_dataset"):
    try:
        module = importlib.import_module(name)
    except ImportError:
        continue
    if hasattr(module, "decode_video_frames"):
        assert module.decode_video_frames is cache.cached_decode_video_frames
        checked_aliases += 1
assert checked_aliases > 0
''', str(video)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _training_dry_run(tmp_path, overrides):
    pytest.importorskip("lerobot")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("FRAME_CACHE_", "LEROBOT_FRAME_CACHE_"))
    }
    environment.update({
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "VENV_DIR": str(tmp_path / "no_venv"),
        "DRY_RUN": "true", "POLICY_TYPE": "diffusion", "SUBTASK": "coarse_insert",
        "CONTROL_SCOPE": "upper_body", "USE_MERGED_HF_REPO": "false",
        "MATERIALIZE_TRAINING_VIEW": "true", "OBB_OVERLAY_ENABLE": "false",
        "WANDB_ENABLE": "false", "UPLOAD_AFTER_TRAIN": "false", "PUSH_TO_HUB": "false",
        "OUTPUT_DIR": str(tmp_path / "training_output"),
        "TRAINING_VIEW_ROOT": str(tmp_path / "view with spaces"),
        "LEROBOT_FRAME_CACHE_ENABLE": "true",
        **overrides,
    })
    return subprocess.run(
        ["bash", str(FEATURE_ROOT / "scripts/train_lerobot.sh")],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )


def _precompute_command(result):
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("precompute_frame_cache:"))
    return shlex.split(line.split(":", 1)[1])


def test_png_training_dry_run_selects_safe_matching_root(tmp_path):
    command = _precompute_command(_training_dry_run(tmp_path, {
        "FRAME_CACHE_IMAGE_FORMAT": "png", "FRAME_CACHE_PNG_COMPRESSION": "3",
        "FRAME_CACHE_JOBS": "2",
    }))
    assert command[command.index("--image-format") + 1] == "png"
    assert command[command.index("--png-compression") + 1] == "3"
    assert command[command.index("--cache-root") + 1] == str(tmp_path / "view with spaces/frame_cache_png")
    assert command[command.index("--jobs") + 1] == "2"
    assert "--keep-source-mp4" in command
    assert not (tmp_path / "view with spaces").exists()


def test_png_training_dry_run_preserves_explicit_root(tmp_path):
    selected = tmp_path / "selected cache"
    command = _precompute_command(_training_dry_run(tmp_path, {
        "FRAME_CACHE_IMAGE_FORMAT": "png", "FRAME_CACHE_ROOT_OVERRIDE": str(selected),
    }))
    assert command[command.index("--cache-root") + 1] == str(selected)


def test_jpeg_training_dry_run_preserves_legacy_defaults(tmp_path):
    command = _precompute_command(_training_dry_run(tmp_path, {}))
    assert "--image-format" not in command
    assert "--cache-root" not in command
    assert "--keep-source-mp4" not in command


def test_png_training_dry_run_can_reuse_without_precompute(tmp_path):
    result = _training_dry_run(tmp_path, {
        "FRAME_CACHE_IMAGE_FORMAT": "png", "FRAME_CACHE_PRECOMPUTE": "false",
    })
    assert result.returncode == 0, result.stderr
    assert "precompute_frame_cache:" not in result.stdout
    assert "frame_cache_root_override:" in result.stdout


@pytest.mark.parametrize("overrides", [
    {"FRAME_CACHE_IMAGE_FORMAT": "gif"},
    {"FRAME_CACHE_IMAGE_FORMAT": "png", "FRAME_CACHE_PNG_COMPRESSION": "10"},
    {"FRAME_CACHE_JOBS": "0"},
])
def test_training_dry_run_rejects_invalid_cache_options(tmp_path, overrides):
    result = _training_dry_run(tmp_path, overrides)
    assert result.returncode == 2
    assert "FRAME_CACHE_" in result.stderr
    assert not (tmp_path / "view with spaces").exists()
