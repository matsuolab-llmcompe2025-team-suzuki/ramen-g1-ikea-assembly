"""Dex1 把持判定 / mock グリッパの unit test (Issue #123)。

しきい値の妥当性は **dataset 実測値をそのまま流し込んで** 検証する
(`Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1`、975,291 frame の集計値)。
"""

from __future__ import annotations

import pytest

from inference.desktop.lower_policy.actuators.dex1 import (
    Dex1Gripper,
    GraspState,
    GraspThresholds,
    MockDex1Gripper,
    classify_grasp,
)
from inference.desktop.lower_policy.kinematics.types import Side


# ------------------------------------------------ dataset 実測値による検証

# (command, state, 期待, 説明) — dataset の帯別 median から構成。
_DATASET_CASES = [
    (1.849, 2.199, GraspState.HOLDING, "左: HOLD帯 指令 + 追従誤差 +0.350"),
    (1.888, 2.238, GraspState.HOLDING, "右: HOLD帯 指令 + 追従誤差 +0.350"),
    (4.500, 4.458, GraspState.OPEN, "左: 開指令、追従誤差 -0.006"),
    (4.379, 4.379, GraspState.OPEN, "右: 開指令、追従"),
    (0.000, 0.028, GraspState.EMPTY_CLOSED, "右: 全閉指令で実測 0.028 = 空振り"),
    (0.000, 0.217, GraspState.EMPTY_CLOSED, "左: 全閉指令で実測 0.217 = 空振り"),
    (2.943, 2.937, GraspState.TRANSIT, "中間帯、追従している = 動作中"),
    (3.029, 3.042, GraspState.TRANSIT, "中間帯、追従している = 動作中"),
]


@pytest.mark.parametrize("command,state,expected,desc", _DATASET_CASES)
def test_classify_grasp_on_dataset_medians(command, state, expected, desc):
    assert classify_grasp(command, state) is expected, desc


def test_holding_requires_both_tracking_error_and_diameter_band():
    """追従誤差だけ、径だけ では HOLDING と判定しない。"""
    th = GraspThresholds()
    # 追従誤差は大きいが、止まった位置が脚の径帯の外 (太すぎる物を噛んだ)
    assert classify_grasp(1.85, 3.20, th) is GraspState.TRANSIT
    # 位置は径帯だが追従している (単にそこへ動かしただけ)
    assert classify_grasp(2.20, 2.20, th) is GraspState.TRANSIT


def test_empty_closed_takes_priority_over_tracking_error():
    """全閉まで到達したら、多少の追従誤差があっても空振り扱い。"""
    assert classify_grasp(0.0, 0.30, GraspThresholds()) is GraspState.EMPTY_CLOSED


def test_classify_grasp_respects_custom_thresholds():
    th = GraspThresholds(hold_state_min=2.15, hold_state_max=2.25)
    assert classify_grasp(1.85, 2.20, th) is GraspState.HOLDING
    assert classify_grasp(1.85, 2.30, th) is GraspState.TRANSIT


# ------------------------------------------------ GraspThresholds


def test_thresholds_reject_inconsistent_ordering():
    with pytest.raises(ValueError, match="満たすこと"):
        GraspThresholds(fist_max=3.0)
    with pytest.raises(ValueError, match="満たすこと"):
        GraspThresholds(hold_state_min=2.5, hold_state_max=2.4)


def test_thresholds_reject_non_positive_tracking_error():
    with pytest.raises(ValueError, match="tracking_err_min"):
        GraspThresholds(tracking_err_min=0.0)


def test_thresholds_from_config_defaults_and_override():
    assert GraspThresholds.from_config(None) == GraspThresholds()
    got = GraspThresholds.from_config({"tracking_err_min": 0.25})
    assert got.tracking_err_min == 0.25
    assert got.open_min == GraspThresholds().open_min


def test_thresholds_from_config_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown key"):
        GraspThresholds.from_config({"grip_force": 1.0})
    with pytest.raises(ValueError, match="mapping"):
        GraspThresholds.from_config([1, 2])


# ------------------------------------------------ MockDex1Gripper


def test_mock_tracks_command_when_no_obstacle():
    g = MockDex1Gripper()
    g.command(Side.RIGHT, 4.5)
    assert g.read(Side.RIGHT) == 4.5
    g.command(Side.RIGHT, 0.0)
    assert g.read(Side.RIGHT) == 0.0
    assert classify_grasp(0.0, g.read(Side.RIGHT)) is GraspState.EMPTY_CLOSED


def test_mock_stalls_at_obstacle_when_closing():
    g = MockDex1Gripper(obstacle_at={Side.RIGHT: 2.2})
    g.command(Side.RIGHT, 1.85)
    state = g.read(Side.RIGHT)
    assert state == 2.2
    assert classify_grasp(1.85, state) is GraspState.HOLDING


def test_mock_obstacle_does_not_block_opening():
    g = MockDex1Gripper(obstacle_at={Side.RIGHT: 2.2})
    g.command(Side.RIGHT, 4.5)
    assert g.read(Side.RIGHT) == 4.5


def test_mock_set_obstacle_switches_mid_run():
    """掴む前は空振り、掴んだ後は保持、に切り替えられること (全閉を指令)。"""
    g = MockDex1Gripper()
    g.command(Side.RIGHT, 0.0)
    assert classify_grasp(0.0, g.read(Side.RIGHT)) is GraspState.EMPTY_CLOSED
    g.set_obstacle(Side.RIGHT, 2.2)
    assert classify_grasp(0.0, g.read(Side.RIGHT)) is GraspState.HOLDING


def test_full_close_command_separates_success_from_failure():
    """把持指令は全閉。成否が HOLDING / EMPTY_CLOSED に排他的に分かれること。

    dataset の指令値 1.85 をそのまま送ると、空振り時にグリッパが 1.85 に到達して
    TRANSIT に張り付き失敗を検出できない。この test がその差を固定する。
    """
    # 全閉指令: 脚あり / 空振り が排他的に分かれる
    held = MockDex1Gripper(obstacle_at={Side.RIGHT: 2.2})
    held.command(Side.RIGHT, 0.0)
    assert classify_grasp(0.0, held.read(Side.RIGHT)) is GraspState.HOLDING

    missed = MockDex1Gripper()
    missed.command(Side.RIGHT, 0.0)
    assert classify_grasp(0.0, missed.read(Side.RIGHT)) is GraspState.EMPTY_CLOSED

    # 1.85 指令: 空振りが TRANSIT に落ちてしまい失敗を検出できない (アンチパターン)
    naive = MockDex1Gripper()
    naive.command(Side.RIGHT, 1.85)
    assert classify_grasp(1.85, naive.read(Side.RIGHT)) is GraspState.TRANSIT


def test_mock_read_none_before_command_and_when_unavailable():
    g = MockDex1Gripper(unavailable={Side.LEFT})
    assert g.read(Side.RIGHT) is None          # 未指令
    g.command(Side.LEFT, 4.5)
    assert g.read(Side.LEFT) is None            # 未受信の模擬
    assert g.last_command(Side.LEFT) == 4.5


def test_mock_logs_commands_and_closes():
    g = MockDex1Gripper()
    g.command(Side.RIGHT, 4.5)
    g.command(Side.LEFT, 1.85)
    assert g.command_log == [(Side.RIGHT, 4.5), (Side.LEFT, 1.85)]
    assert not g.closed
    g.close()
    assert g.closed


def test_mock_satisfies_protocol():
    assert isinstance(MockDex1Gripper(), Dex1Gripper)
