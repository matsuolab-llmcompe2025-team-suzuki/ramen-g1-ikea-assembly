"""pick_table_leg の統合 test (Issue #123)。

mock を kinematics と perception から外し、**実 IK (G1ArmKinematics) と
実 OBB 変換 (ObbGraspPoseProvider) を出荷 YAML そのままで**組んで 8 stage を通す。
実機を要するのはグリッパの DDS だけなので、そこだけ mock に差し替える。

entrypoint が行う組み立てと同じ経路を通るので、config の配線ミスもここで落ちる。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from inference.desktop.lower_policy.actuators.dex1 import MockDex1Gripper
from inference.desktop.lower_policy.kinematics.g1_arm import G1ArmKinematics
from inference.desktop.lower_policy.kinematics.types import Side
from inference.desktop.lower_policy.obb_grasp import ObbGraspPoseProvider, make_obb
from inference.desktop.lower_policy.pose_utils import (
    NUM_ARM_JOINTS,
    ROS_ARM_JOINT_NAMES,
)
from inference.desktop.lower_policy.skills.pick_table_leg import (
    PickTableLegSkill,
    SkillStatus,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "skill_config.yaml"
LEG_STALL = 2.2
# The physical leg lies on the support plane.  The provider adds the shipped
# Dex1-base height offset (0.101 m) only after ray intersection.
LEG_CENTER = np.array([0.3662, -0.1632, 0.0])
LEG_YAW = 0.3047
LEG_LEN = 0.36
LEG_WIDTH = 0.045
# ROS /joint_states の並び: 脚12 → 腰3 → 腕14 (G1_JOINT_NAMES と同順)
ROS_JOINT_NAMES = tuple(f"leg_{i}" for i in range(12)) + (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
) + ROS_ARM_JOINT_NAMES


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["skills"]["pick_table_leg"]


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class FakeJointState:
    name: tuple[str, ...]
    position: np.ndarray


def leg_corners(center=LEG_CENTER, yaw=LEG_YAW) -> np.ndarray:
    u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    v = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    hl, hw = LEG_LEN / 2, LEG_WIDTH / 2
    return np.array([center - u * hl - v * hw, center + u * hl - v * hw,
                     center + u * hl + v * hw, center - u * hl + v * hw])


def leg_corners_len(length: float) -> np.ndarray:
    """長さを変えた脚 (持ち替え不能な短い脚の試験用)。"""
    u = np.array([np.cos(LEG_YAW), np.sin(LEG_YAW), 0.0])
    v = np.array([-np.sin(LEG_YAW), np.cos(LEG_YAW), 0.0])
    hl, hw = length / 2, LEG_WIDTH / 2
    return np.array([LEG_CENTER - u * hl - v * hw, LEG_CENTER + u * hl - v * hw,
                     LEG_CENTER + u * hl + v * hw, LEG_CENTER - u * hl + v * hw])


def project(provider: ObbGraspPoseProvider, pts: np.ndarray, waist) -> np.ndarray:
    origin, R = provider.camera_pose_in_root(waist)
    cam = (np.asarray(pts) - origin) @ R
    u = provider.camera.fx * cam[:, 0] / cam[:, 2] + provider.camera.cx
    v = provider.camera.fy * cam[:, 1] / cam[:, 2] + provider.camera.cy
    return np.stack([u / provider.camera.width, v / provider.camera.height], axis=1)


class Rig:
    """実 IK + 実 OBB 変換で skill を駆動する。完全追従の actuator を仮定。"""

    def __init__(self, *, waist=np.zeros(3), gripper=None, cfg=None):
        self.cfg = cfg or load_cfg()
        self.kin = G1ArmKinematics()
        # entrypoint と同じく IK を渡す (到達可能な手首姿勢を選ばせる)
        self.provider = ObbGraspPoseProvider.from_config(
            self.cfg.get("obb_grasp"), kinematics=self.kin
        )
        self.gripper = gripper or MockDex1Gripper(
            obstacle_at={Side.LEFT: LEG_STALL, Side.RIGHT: LEG_STALL}
        )
        self.clock = FakeClock()
        self.skill = PickTableLegSkill.from_config(
            self.cfg,
            kinematics=self.kin,
            gripper=self.gripper,
            grasp_provider=self.provider,
            time_fn=self.clock,
        )
        self.waist = np.asarray(waist, dtype=np.float64)
        # 腕は SetupSkill の clearance 姿勢近傍から始める
        self.arm = np.zeros(NUM_ARM_JOINTS)
        self.arm[1], self.arm[8] = 1.2, -1.2
        self.px = project(self.provider, leg_corners(), self.waist)

    def obs(self) -> dict:
        pos = np.concatenate([np.zeros(12), self.waist, self.arm])
        return {
            "joint_state": FakeJointState(name=ROS_JOINT_NAMES, position=pos),
            "cleaned": [make_obb(self.px)],
        }

    def run(self, ticks: int = 1) -> None:
        for _ in range(ticks):
            out = self.skill.step(self.obs())
            if out is not None:
                self.arm = out.copy()
            self.clock.advance(1.0 / 30)

    def run_until(self, pred, max_ticks: int = 6000) -> bool:
        for _ in range(max_ticks):
            if pred(self.skill):
                return True
            self.run()
        return pred(self.skill)

    def ee(self, side: Side):
        sl = slice(0, 7) if side is Side.LEFT else slice(7, 14)
        return self.kin.fk(self.arm[sl], side, self.waist)


# ---------------------------------------------------------------- 通し


def test_shipped_config_builds_with_real_kinematics_and_perception():
    """出荷 YAML が実 IK / 実 OBB 変換でそのまま組めること (entrypoint と同じ経路)。"""
    rig = Rig()
    assert [s.name for s in rig.skill.stages] == [
        "pregrasp", "approach_grasp", "close_grasp", "to_presentation",
        "approach_handover", "close_handover", "release", "retract",
    ]
    assert rig.provider.leg_class == "leg"


def test_full_sequence_completes_with_real_ik():
    rig = Rig()
    rig.skill.start({})
    assert rig.run_until(lambda s: s.status is SkillStatus.DONE), (
        f"stage={rig.skill.stage_name!r} reason={rig.skill.failure_reason}"
    )
    assert rig.skill.retries == 0


def test_reaches_the_detected_leg_within_tolerance():
    """CV が出した把持ポーズに実 IK で到達できること。"""
    rig = Rig()
    rig.skill.start({})
    target = rig.provider.grasp_pose(rig.obs())
    assert target is not None
    # 把持点は脚の中心ではなく、持ち替え距離から逆算した比率の位置。
    # 近い側の端から fraction の点に来る。
    u = np.array([np.cos(LEG_YAW), np.sin(LEG_YAW), 0.0])
    near_end = LEG_CENTER - u * (LEG_LEN / 2)
    expected = near_end + u * (LEG_LEN * rig.provider.last_grasp_fraction)
    expected[2] += rig.provider.grasp_height_offset_m
    np.testing.assert_allclose(target.position, expected, atol=1e-6)
    assert 0.0 < rig.provider.last_grasp_fraction < 0.5, "中心より手前を掴む"

    assert rig.run_until(lambda s: s.stage_name == "close_grasp")
    reached = rig.ee(Side.RIGHT)
    assert reached.position_error(target) <= 0.02
    assert reached.rotation_error(target) <= 0.12


def test_pregrasp_then_vertical_descent_with_real_ik():
    rig = Rig()
    rig.skill.start({})
    rig.run_until(lambda s: s.stage_name == "approach_grasp")
    above = rig.ee(Side.RIGHT).position.copy()
    grasp_z = LEG_CENTER[2] + rig.provider.grasp_height_offset_m
    assert above[2] > grasp_z + 0.07, "把持点の真上に来ていない"
    rig.run_until(lambda s: s.stage_name == "close_grasp")
    down = rig.ee(Side.RIGHT).position
    dz, dxy = above[2] - down[2], float(np.linalg.norm(down[:2] - above[:2]))
    assert dz > 0.07 and dz > 5 * dxy, f"降下が鉛直でない dz={dz:.4f} dxy={dxy:.4f}"


def test_ik_solutions_stay_inside_the_actuator_safety_clamp():
    """全 tick の指令が ±1.5 rad に収まること (actuator の silent clamp を避ける)。"""
    rig = Rig()
    rig.skill.start({})
    peak = 0.0
    for _ in range(6000):
        if rig.skill.status is not SkillStatus.RUNNING:
            break
        rig.run()
        peak = max(peak, float(np.abs(rig.arm).max()))
    assert rig.skill.status is SkillStatus.DONE
    assert peak <= 1.5 + 1e-9, f"安全枠超過 {peak:.4f} rad"


def test_handover_target_follows_the_actual_right_hand():
    """左手の目標が右手の実測 FK から解決されること。"""
    rig = Rig()
    rig.skill.start({})
    rig.run_until(lambda s: s.stage_name == "close_handover")
    right, left = rig.ee(Side.RIGHT), rig.ee(Side.LEFT)
    offset_local = right.matrix.T @ (left.position - right.position)
    expected = np.array(rig.cfg["stages"][4]["goal"]["offset"])
    np.testing.assert_allclose(offset_local, expected, atol=0.03)


def test_left_grasps_higher_than_right():
    """dataset の構造 (左手が右手より上を掴む) が再現されること。"""
    rig = Rig()
    rig.skill.start({})
    rig.run_until(lambda s: s.stage_name == "close_grasp")
    right_grasp_z = rig.ee(Side.RIGHT).position[2]
    rig.run_until(lambda s: s.stage_name == "close_handover")
    assert rig.ee(Side.LEFT).position[2] > right_grasp_z + 0.05


@pytest.mark.parametrize("waist", [
    np.zeros(3),
    np.array([0.10, 0.0, 0.0]),
    np.array([0.0, 0.05, -0.04]),
])
def test_completes_under_waist_offsets(waist):
    """腰が傾いていても、CV と IK の双方に反映されて通ること。"""
    rig = Rig(waist=waist)
    rig.skill.start({})
    assert rig.run_until(lambda s: s.status is SkillStatus.DONE), (
        f"waist={waist} stage={rig.skill.stage_name!r} {rig.skill.failure_reason}"
    )


@pytest.mark.parametrize("dx,dy", [
    (0.0, 0.0), (-0.04, 0.05), (-0.04, -0.06), (-0.08, 0.0), (0.0, 0.06),
])
def test_completes_when_the_leg_moves_within_the_reachable_envelope(dx, dy):
    """脚が到達域内で動いても通ること。

    到達域は把持点中央値を中心に **前方 +x が 0.010m、外 -y 0.085m、内 +y 0.095m、
    後方 -x 0.205m** (手首姿勢の候補選択あり、実測)。+x に余裕が無いのは固定ベースの
    構造的な限界。
    """
    rig = Rig()
    rig.px = project(rig.provider, leg_corners(LEG_CENTER + np.array([dx, dy, 0.0])), rig.waist)
    rig.skill.start({})
    assert rig.run_until(lambda s: s.status is SkillStatus.DONE), (
        f"offset=({dx},{dy}) stage={rig.skill.stage_name!r} {rig.skill.failure_reason}"
    )


def test_unreachable_leg_is_rejected_instead_of_aborting_mid_motion():
    """腕が届かない脚は provider が候補から外し、skill は待機する。

    途中まで動いてから IK 失敗で abort するより安全。前方 +x は余裕が
    0.010m しかないので、この経路は実機で現実に起きる。
    """
    rig = Rig()
    rig.px = project(
        rig.provider, leg_corners(LEG_CENTER + np.array([0.10, 0.0, 0.0])), rig.waist
    )
    assert rig.provider.grasp_pose(rig.obs()) is None, "届かない脚を返している"
    rig.skill.start({})
    start_arm = rig.arm.copy()
    rig.run(30)
    np.testing.assert_allclose(rig.arm, start_arm, atol=1e-12)
    assert rig.skill.status is SkillStatus.RUNNING


def test_orientation_is_chosen_for_reachability():
    """姿勢候補の選択が実際に効いていること。

    第一候補で解ける位置ではそれを使い、解けない位置では別候補に切り替わる。
    到達域の縁 (前方 +x 側) を掃くと切り替わりが現れる。
    """
    rig = Rig()
    first = (rig.provider.roll_candidates[0], rig.provider.pitch_candidates[0])
    used_first = switched = False
    for dx in np.arange(-0.10, 0.09, 0.01):
        for dy in (-0.06, 0.0, 0.06):
            rig.px = project(
                rig.provider,
                leg_corners(LEG_CENTER + np.array([dx, dy, 0.0])),
                rig.waist,
            )
            if rig.provider.grasp_pose(rig.obs()) is None:
                continue
            if rig.provider.last_orientation == first:
                used_first = True
            else:
                switched = True
    assert used_first, "第一候補が一度も使われていない"
    assert switched, "全位置で第一候補 = 候補選択が効いていない"


def test_grasp_miss_retries_and_recovers_with_real_ik():
    gripper = MockDex1Gripper(obstacle_at={Side.LEFT: LEG_STALL})   # 右は空振り
    rig = Rig(gripper=gripper)
    rig.skill.start({})
    rig.run_until(lambda s: s.retries >= 1)
    assert rig.skill.stage_name == "pregrasp"
    gripper.set_obstacle(Side.RIGHT, LEG_STALL)
    assert rig.run_until(lambda s: s.status is SkillStatus.DONE)


def test_no_detection_holds_position_without_failing_early():
    rig = Rig()
    rig.skill.start({})
    start_arm = rig.arm.copy()
    for _ in range(20):
        out = rig.skill.step({
            "joint_state": FakeJointState(
                name=ROS_JOINT_NAMES,
                position=np.concatenate([np.zeros(12), rig.waist, rig.arm]),
            ),
            "cleaned": [],
        })
        if out is not None:
            rig.arm = out.copy()
        rig.clock.advance(1.0 / 30)
    np.testing.assert_allclose(rig.arm, start_arm, atol=1e-12)
    assert rig.skill.status is SkillStatus.RUNNING


def test_completes_within_the_declared_dwell_budget():
    """max_dwell_sec (orchestrator の fail-safe) 内に終わること。"""
    rig = Rig()
    rig.skill.start({})
    rig.run_until(lambda s: s.status is SkillStatus.DONE)
    assert rig.clock.t < rig.skill.max_dwell_sec, (
        f"所要 {rig.clock.t:.1f}s が max_dwell_sec {rig.skill.max_dwell_sec:.1f}s を超えた"
    )


def test_left_hand_lands_within_the_protruding_part_of_the_leg():
    """左手の把持点が、右手より先に出ている脚の範囲に収まること。

    右手は脚長から逆算した比率で掴むので、把持点より先に
    clearance + margin ぶんが出る。左手はそこから clearance の位置を掴むので、
    実際の両手間距離がその突き出し量を超えていたら空を掴んでいる。
    """
    rig = Rig()
    og = rig.cfg["obb_grasp"]
    clearance = float(og["handover_clearance_m"])
    margin = float(og["handover_margin_m"])

    rig.skill.start({})
    rig.run_until(lambda s: s.stage_name == "close_grasp")
    fraction = rig.provider.last_grasp_fraction
    length = rig.provider.last_leg_length
    protrusion = (1.0 - fraction) * length
    assert protrusion == pytest.approx(clearance + margin, abs=1e-6)

    rig.run_until(lambda s: s.stage_name == "close_handover")
    span = rig.ee(Side.RIGHT).position_error(rig.ee(Side.LEFT))
    assert span <= protrusion, (
        f"両手間距離 {span:.4f} m が突き出し量 {protrusion:.4f} m を超えている"
        " = 左手が脚の先の空を掴んでいる"
    )
    assert span == pytest.approx(clearance, abs=0.03)


def test_leg_shorter_than_the_handover_distance_is_never_grasped():
    """持ち替えできない短い脚は、掴む前に候補から外れること。"""
    rig = Rig()
    rig.px = project(rig.provider, leg_corners_len(0.15), rig.waist)
    assert rig.provider.grasp_pose(rig.obs()) is None
    assert rig.provider.last_reject_reason == "leg_too_short"
    rig.skill.start({})
    start_arm = rig.arm.copy()
    rig.run(30)
    np.testing.assert_allclose(rig.arm, start_arm, atol=1e-12)
    assert rig.skill.status is SkillStatus.RUNNING
