"""Smoke test for lerobot_train_with_frame_cache wrapper (Issue #122 D-4-S2)。

wrapper.main() が
  1. apply_patch() を呼ぶ
  2. obb_overlay_setup.setup_from_env() を呼ぶ
  3. lerobot.scripts.lerobot_train.main を呼ぶ
の順序で連鎖する事を、実 lerobot import を bypass しつつ検証。
"""

from __future__ import annotations

import sys
import types

import pytest

from model.subtask_policy_training.scripts import (
    lerobot_train_with_frame_cache as wrapper,
)


def _make_fake_lerobot(call_order: list[str]) -> types.ModuleType:
    """`lerobot.scripts.lerobot_train.main` の代わりに呼ばれる fake module。"""
    lerobot_train_mod = types.ModuleType("lerobot.scripts.lerobot_train")

    def fake_main() -> None:
        call_order.append("lerobot_train.main")

    lerobot_train_mod.main = fake_main
    return lerobot_train_mod


def test_wrapper_main_invokes_patch_then_setup_then_lerobot_train(monkeypatch) -> None:
    call_order: list[str] = []

    # apply_patch を差替 (実 lerobot 依存を回避)
    def fake_apply_patch() -> bool:
        call_order.append("apply_patch")
        return True

    monkeypatch.setattr(
        "model.subtask_policy_training.scripts.lerobot_frame_cache_patch.apply_patch",
        fake_apply_patch,
    )

    # setup_from_env を差替 (env 未設定でも side-effect を明示 record)
    def fake_setup_from_env():
        call_order.append("setup_from_env")
        return None

    monkeypatch.setattr(
        "model.subtask_policy_training.scripts.obb_overlay_setup.setup_from_env",
        fake_setup_from_env,
    )

    # lerobot.scripts.lerobot_train を fake module に差替 (import 前に inject)
    fake_module = _make_fake_lerobot(call_order)
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.scripts", types.ModuleType("lerobot.scripts"))
    monkeypatch.setitem(sys.modules, "lerobot.scripts.lerobot_train", fake_module)

    wrapper.main()

    # 順序が期待通り
    assert call_order == ["apply_patch", "setup_from_env", "lerobot_train.main"]


def test_wrapper_main_warns_when_apply_patch_fails(monkeypatch, capsys) -> None:
    """apply_patch が False を返した時、warn を stderr に出しつつ処理続行。"""
    call_order: list[str] = []

    def fake_apply_patch() -> bool:
        return False

    def fake_setup_from_env():
        call_order.append("setup_from_env")
        return None

    monkeypatch.setattr(
        "model.subtask_policy_training.scripts.lerobot_frame_cache_patch.apply_patch",
        fake_apply_patch,
    )
    monkeypatch.setattr(
        "model.subtask_policy_training.scripts.obb_overlay_setup.setup_from_env",
        fake_setup_from_env,
    )
    fake_module = _make_fake_lerobot(call_order)
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.scripts", types.ModuleType("lerobot.scripts"))
    monkeypatch.setitem(sys.modules, "lerobot.scripts.lerobot_train", fake_module)

    wrapper.main()

    err = capsys.readouterr().err
    assert "apply_patch()" in err
    # patch fail でも setup + lerobot_train は呼ばれる (frame cache 無効で継続)
    assert call_order == ["setup_from_env", "lerobot_train.main"]
