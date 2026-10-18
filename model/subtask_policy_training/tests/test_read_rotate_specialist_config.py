"""Issue #129 rotate specialist YAML config → shell env 変換の test。

Coverage:
- YAML load + structure validation (required keys 欠落で raise)
- shell 変換 (shlex escape、bool/float の str 化)
- 実 3 YAML file (run1/2/3) の load + 変換
- env 名の invalid char reject
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.subtask_policy_training.scripts.rotate_specialist import config as m

CONFIG_DIR = REPO_ROOT / "model" / "subtask_policy_training" / "configs" / "rotate_specialist"


def test_load_run1_yaml_ok() -> None:
    """Run 1 YAML が load + validate 通過。"""
    cfg = m.load_config(CONFIG_DIR / "run1_t01_only.yaml")
    assert cfg["run_id"] == "run1_t01"
    assert cfg["env"]["GROOT_LEFT_HAND_LOSS_WEIGHT"] == 2.0
    assert cfg["env"]["GROOT_BILATERAL_LOSS_WEIGHT"] == 0.0
    assert cfg["env"]["GROOT_FK_ANCHOR_LOSS_WEIGHT"] == 0.0
    assert cfg["env"]["SUBTASK"] == "rotate_table_base"


def test_load_run2_yaml_ok() -> None:
    """Run 2 YAML が load + validate 通過。"""
    cfg = m.load_config(CONFIG_DIR / "run2_l1l4_only.yaml")
    assert cfg["run_id"] == "run2_l1l4"
    assert cfg["env"]["GROOT_LEFT_HAND_LOSS_WEIGHT"] == 1.0
    assert cfg["env"]["GROOT_BILATERAL_LOSS_WEIGHT"] == 0.5
    assert cfg["env"]["GROOT_FK_ANCHOR_LOSS_WEIGHT"] == 0.3


def test_load_run3_yaml_ok() -> None:
    """Run 3 YAML が load + validate 通過。"""
    cfg = m.load_config(CONFIG_DIR / "run3_all.yaml")
    assert cfg["run_id"] == "run3_all"
    assert cfg["env"]["GROOT_LEFT_HAND_LOSS_WEIGHT"] == 2.0
    assert cfg["env"]["GROOT_BILATERAL_LOSS_WEIGHT"] == 0.5
    assert cfg["env"]["GROOT_FK_ANCHOR_LOSS_WEIGHT"] == 0.3


def test_load_missing_run_id_raises(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("env:\n  FOO: 1\n")
    with pytest.raises(ValueError, match="missing required key: run_id"):
        m.load_config(p)


def test_load_missing_env_raises(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("run_id: x\n")
    with pytest.raises(ValueError, match="missing required key: env"):
        m.load_config(p)


def test_load_missing_required_env_key_raises(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("run_id: x\nenv:\n  GROOT_LEFT_HAND_LOSS_WEIGHT: 2.0\n")
    with pytest.raises(ValueError, match="env missing required keys"):
        m.load_config(p)


def test_load_non_dict_root_raises(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- item\n")
    with pytest.raises(ValueError, match="root must be a dict"):
        m.load_config(p)


def test_env_to_shell_escaping() -> None:
    """shell escape が特殊文字入りの値でも安全。"""
    env = {"FOO": "hello world", "BAR": 2.0, "BAZ": True, "QUUX": "with $var"}
    out = m.env_to_shell(env)
    lines = out.split("\n")
    # 各 line は export で始まる
    for line in lines:
        assert line.startswith("export ")
    # 特殊文字 (space, $) を含む値は quote される、literal $ が escape 済
    assert any("'hello world'" in line for line in lines)
    assert any("'with $var'" in line for line in lines)
    # bool は "True" 文字列に (shlex.quote は simple な True を quote しない、値だけ確認)
    assert any("BAZ=True" in line for line in lines)
    # float は "2.0" 文字列に
    assert any("BAR=2.0" in line for line in lines)


def test_env_to_shell_invalid_var_name_raises() -> None:
    """env 変数名に invalid char あると raise。"""
    with pytest.raises(ValueError, match="invalid env var name"):
        m.env_to_shell({"BAD NAME": "1"})
    with pytest.raises(ValueError, match="invalid env var name"):
        m.env_to_shell({"$INJECTION": "1"})


def test_cli_shell_output_run1() -> None:
    """CLI 経由で run1 YAML → shell export が生成される (subprocess で実行)。"""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "model" / "subtask_policy_training" / "scripts" / "rotate_specialist" / "config.py"),
            "--config",
            str(CONFIG_DIR / "run1_t01_only.yaml"),
            "--format",
            "shell",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    # shlex.quote は simple value ('2.0' 等) を quote しない、値さえ含まれれば OK
    assert "GROOT_LEFT_HAND_LOSS_WEIGHT=" in out and "2.0" in out
    assert "SUBTASK=" in out and "rotate_table_base" in out
    assert "HF_AUTOPUSH_REPO_ID=" in out
    # 各 line は export で始まる
    for line in out.strip().split("\n"):
        assert line.startswith("export ")


def test_cli_summary_output_run1() -> None:
    """CLI 経由で summary format も動く。"""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "model" / "subtask_policy_training" / "scripts" / "rotate_specialist" / "config.py"),
            "--config",
            str(CONFIG_DIR / "run1_t01_only.yaml"),
            "--format",
            "summary",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    assert "run1_t01" in out
    assert "GROOT_LEFT_HAND_LOSS_WEIGHT=2.0" in out
