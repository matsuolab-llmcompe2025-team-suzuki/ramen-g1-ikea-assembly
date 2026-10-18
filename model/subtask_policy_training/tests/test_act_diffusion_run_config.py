"""act_diffusion/run_config.py の run 設定検証 (Issue #139)。"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from model.subtask_policy_training.act_diffusion import run_config as rc

CONFIG_DIR = Path(__file__).resolve().parents[1] / "act_diffusion" / "configs"


@pytest.mark.parametrize("name", ["act_rotate_table_base.yaml", "diffusion_rotate_table_base.yaml"])
def test_shipped_configs_resolve(name: str) -> None:
    env = rc.resolve_env(rc.load_config(CONFIG_DIR / name), [])

    assert env["SUBTASK"] == "rotate_table_base"
    assert env["UPLOAD_AFTER_TRAIN"] == "false"
    assert env["PUSH_TO_HUB"] == "false"
    assert env["TRAIN_IMAGE_TRANSFORMS_ENABLE"] == "false"
    assert env["FRAME_CACHE_PRECOMPUTE"] == "false"
    assert env["FRAME_CACHE_STRICT"] == "true"
    assert env["POLICY_REPO_ID"].endswith(f"_rotate_table_base_{env['POLICY_TYPE']}_v1")


def test_overrides_win_and_bools_are_lowercase() -> None:
    cfg = {"env": {"SUBTASK": "rotate_table_base", "POLICY_TYPE": "act", "POLICY_REPO_ID": "o/r",
                   "TRAIN_BATCH_SIZE": 64, "TRAIN_STEPS": 100000, "TRAIN_SAVE_FREQ": 10000,
                   "WANDB_ENABLE": True}}

    env = rc.resolve_env(cfg, ["TRAIN_STEPS=40", "TRAIN_BATCH_SIZE=2"])

    assert env["TRAIN_STEPS"] == "40"
    assert env["TRAIN_BATCH_SIZE"] == "2"
    assert env["WANDB_ENABLE"] == "true"


@pytest.mark.parametrize("key", ["UPLOAD_AFTER_TRAIN", "PUSH_TO_HUB"])
def test_final_uploads_that_would_delete_checkpoints_are_rejected(key: str) -> None:
    cfg = rc.load_config(CONFIG_DIR / "act_rotate_table_base.yaml")

    with pytest.raises(ValueError, match="must be false"):
        rc.resolve_env(cfg, [f"{key}=true"])


def test_rejects_missing_keys_unknown_policy_and_bad_override() -> None:
    cfg = rc.load_config(CONFIG_DIR / "act_rotate_table_base.yaml")
    with pytest.raises(ValueError, match="missing required keys"):
        rc.resolve_env({"env": {"POLICY_TYPE": "act"}}, [])
    with pytest.raises(ValueError, match="POLICY_TYPE"):
        rc.resolve_env(cfg, ["POLICY_TYPE=groot"])
    with pytest.raises(ValueError, match="KEY=VALUE"):
        rc.resolve_env(cfg, ["TRAIN_STEPS"])
    with pytest.raises(ValueError, match="invalid env names"):
        rc.resolve_env(cfg, ["train_steps=1"])


def test_shell_output_is_quoted() -> None:
    line = rc.format_shell({"DESCRIPTION": "a b; rm -rf /"})

    assert shlex.split(line) == ["export", "DESCRIPTION=a b; rm -rf /"]
