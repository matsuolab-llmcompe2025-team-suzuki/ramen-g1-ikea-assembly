"""実 PickTableLegSkill を orchestrator / dispatcher 越しに動かす test (Issue #123)。

これまで `pick_table_leg` は orchestrator の test では **名前の文字列**でしかなく、
中身は MockSkill だった。実 skill を挿した状態の dispatch・観測の受け渡し・
`max_dwell_sec` の露出が未検証だったのでここで埋める。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import yaml

from inference.desktop.entrypoint import build_pick_table_leg
from inference.desktop.lower_policy.actuators.dex1 import MockDex1Gripper
from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
from inference.desktop.lower_policy.kinematics.types import Side
from inference.desktop.lower_policy.obb_grasp import make_obb
from inference.desktop.lower_policy.pose_utils import (
    NUM_ARM_JOINTS,
    ROS_ARM_JOINT_NAMES,
)
from inference.desktop.lower_policy.skills.mock import MockSkill
from inference.desktop.lower_policy.skills.pick_table_leg import SkillStatus
from inference.desktop.orchestrator import DEFAULT_TRANSITIONS

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "skill_config.yaml"
LEG_STALL = 2.2
# OBB corners are on the physical support plane; the provider separately adds
# the Dex1-base command height.
LEG_CENTER = np.array([0.3662, -0.1632, 0.0])
LEG_YAW, LEG_LEN, LEG_WIDTH = 0.3047, 0.36, 0.045
ROS_JOINT_NAMES = tuple(f"leg_{i}" for i in range(12)) + (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
) + ROS_ARM_JOINT_NAMES


@dataclass
class FakeJointState:
    name: tuple[str, ...]
    position: np.ndarray


def skills_section() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["skills"]


def leg_pixels(provider, waist=np.zeros(3)) -> np.ndarray:
    u = np.array([np.cos(LEG_YAW), np.sin(LEG_YAW), 0.0])
    v = np.array([-np.sin(LEG_YAW), np.cos(LEG_YAW), 0.0])
    hl, hw = LEG_LEN / 2, LEG_WIDTH / 2
    pts = np.array([LEG_CENTER - u * hl - v * hw, LEG_CENTER + u * hl - v * hw,
                    LEG_CENTER + u * hl + v * hw, LEG_CENTER - u * hl + v * hw])
    origin, R = provider.camera_pose_in_root(waist)
    cam = (pts - origin) @ R
    uu = provider.camera.fx * cam[:, 0] / cam[:, 2] + provider.camera.cx
    vv = provider.camera.fy * cam[:, 1] / cam[:, 2] + provider.camera.cy
    return np.stack([uu / provider.camera.width, vv / provider.camera.height], axis=1)


class FakeClock:
    """stage の進行は経過時間で駆動されるので、tick と時間を対応させる。"""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class Harness:
    """dispatcher 越しに実 skill を回す。orchestrator が渡す obs 形式を再現する。"""

    def __init__(self) -> None:
        self.gripper = MockDex1Gripper(
            obstacle_at={Side.LEFT: LEG_STALL, Side.RIGHT: LEG_STALL}
        )
        self.clock = FakeClock()
        self.skill = build_pick_table_leg(
            skills_section(), self.gripper, time_fn=self.clock
        )
        registry: dict[str, object] = {
            name: MockSkill(name)
            for name in DEFAULT_TRANSITIONS
            if name != "pick_table_leg"
        }
        registry["pick_table_leg"] = self.skill
        self.dispatcher = SkillDispatchLowerPolicy(registry)
        self.arm = np.zeros(NUM_ARM_JOINTS)
        self.arm[1], self.arm[8] = 1.2, -1.2
        self.px = leg_pixels(self.skill._grasp_provider)  # noqa: SLF001

    def obs(self) -> dict:
        pos = np.concatenate([np.zeros(12), np.zeros(3), self.arm])
        return {
            "head_rgb": np.zeros((1, 1, 3), dtype=np.uint8),
            "t": 0,
            "cleaned": [make_obb(self.px)],
            "joint_state": FakeJointState(name=ROS_JOINT_NAMES, position=pos),
        }

    def step(self, n: int = 1) -> Optional[np.ndarray]:
        out = None
        for _ in range(n):
            out = self.dispatcher.step(self.obs())
            if out is not None:
                self.arm = out.copy()
            self.clock.advance(1.0 / 30)   # 30Hz の tick
        return out


# ---------------------------------------------------------------- dispatch

def test_dispatcher_starts_and_forwards_to_the_real_skill():
    h = Harness()
    assert h.dispatcher.active_skill_name is None
    h.dispatcher.start("pick_table_leg", {})
    assert h.dispatcher.active_skill_name == "pick_table_leg"
    assert h.dispatcher.active_skill is h.skill
    action = h.step()
    assert action is not None and action.shape == (NUM_ARM_JOINTS,)
    assert h.skill.status is SkillStatus.RUNNING


def test_action_history_records_the_skill_output():
    """dispatcher の ring buffer に実 skill の指令が積まれること (resume 経路が依存)。"""
    h = Harness()
    h.dispatcher.start("pick_table_leg", {})
    h.step(5)
    history = h.dispatcher.action_history
    assert len(history) == 5
    assert all(a.shape == (NUM_ARM_JOINTS,) for a in history)
    np.testing.assert_allclose(history[-1], h.arm)


def test_runs_to_completion_through_the_dispatcher():
    h = Harness()
    h.dispatcher.start("pick_table_leg", {})
    for _ in range(6000):
        if h.skill.status is not SkillStatus.RUNNING:
            break
        h.step()
    assert h.skill.status is SkillStatus.DONE, h.skill.failure_reason


def test_max_dwell_sec_is_visible_to_the_orchestrator():
    """orchestrator は `dispatcher.active_skill.max_dwell_sec` を fail-safe に読む。"""
    h = Harness()
    h.dispatcher.start("pick_table_leg", {})
    dwell = h.dispatcher.active_skill.max_dwell_sec
    assert dwell is not None and dwell > 0
    # 全 stage の timeout 合計 + retry ぶんを覆っていること
    base = sum(s.timeout_sec for s in h.skill.stages)
    assert dwell >= base


def test_stopping_through_the_dispatcher_closes_the_gripper():
    """skill 交代でグリッパの後始末が走ること (native thread を残さない)。"""
    h = Harness()
    h.dispatcher.start("pick_table_leg", {})
    h.step(3)
    h.dispatcher.start("insert_table_leg", {})
    assert h.gripper.closed
    assert h.dispatcher.active_skill_name == "insert_table_leg"


def test_restart_after_stop_resets_progress():
    h = Harness()
    h.dispatcher.start("pick_table_leg", {})
    h.step(90)   # pregrasp は 2.0s = 60 tick なので、これで次 stage へ進む
    assert h.skill.stage_index > 0
    h.dispatcher.stop()
    h.dispatcher.start("pick_table_leg", {})
    assert h.skill.stage_index == 0
    assert h.skill.status is SkillStatus.RUNNING


def test_missing_joint_state_yields_no_action_but_keeps_running():
    """観測が揃わない間は指令を出さない (0 を送ると腕が落ちる)。"""
    h = Harness()
    h.dispatcher.start("pick_table_leg", {})
    assert h.dispatcher.step({"head_rgb": None, "t": 0, "cleaned": []}) is None
    assert h.skill.status is SkillStatus.RUNNING
    assert h.dispatcher.action_history == []


def test_transition_graph_lists_pick_table_leg_between_base_and_insert():
    """遷移グラフ上の位置が変わっていないこと (本 Issue で触っていない確認)。"""
    assert DEFAULT_TRANSITIONS["move_table_base"] == ["pick_table_leg"]
    assert DEFAULT_TRANSITIONS["pick_table_leg"] == ["insert_table_leg"]


def test_unknown_skill_name_does_not_stop_the_active_one():
    """registry ミスで動作中の skill が中途半端に止まらないこと (atomic 性)。"""
    h = Harness()
    h.dispatcher.start("pick_table_leg", {})
    h.step(3)
    with pytest.raises(KeyError):
        h.dispatcher.start("does_not_exist", {})
    assert h.dispatcher.active_skill_name == "pick_table_leg"
    assert not h.gripper.closed
