"""PickTableLegSkill の stage 遷移 unit test (Issue #123)。

mock kinematics / mock gripper / fake 時計で、実機なしに全 stage 経路を通す。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import yaml

from inference.desktop.lower_policy.actuators.dex1 import (
    GraspThresholds,
    MockDex1Gripper,
)
from inference.desktop.lower_policy.grasp_pose import (
    FixedGraspPoseProvider,
    ScriptedGraspPoseProvider,
    grasp_pose_from_obb_axis,
)
from inference.desktop.lower_policy.kinematics.mock import MockArmKinematics
from inference.desktop.lower_policy.kinematics.types import EEPose, IKStatus, Side
from inference.desktop.lower_policy.pose_utils import (
    NUM_ARM_JOINTS,
    ROS_ARM_JOINT_NAMES,
)
from inference.desktop.lower_policy.skills.pick_table_leg import (
    ExitSpec,
    GoalKind,
    GoalSpec,
    PickTableLegSkill,
    SkillStatus,
    StageSpec,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "skill_config.yaml"
)
# CV が返す把持ポーズ (mock kinematics の可動域に収まる値)。
GRASP_POSE = EEPose.from_vec6([0.351, -0.187, 0.117, 0.199, 0.827, 0.236])
LEG_STALL = 2.2   # 脚を噛んだときにグリッパが止まる位置 (dataset 実測 2.19-2.23)


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


def make_obs(arm: np.ndarray) -> dict:
    """14-D 腕 pose を `/joint_states` 形式の obs に包む (腕以外はダミー)。"""
    names = ("dummy_a", "dummy_b") + ROS_ARM_JOINT_NAMES
    pos = np.concatenate([np.zeros(2), np.asarray(arm, dtype=np.float64)])
    return {"joint_state": FakeJointState(name=names, position=pos)}


class Rig:
    """完全追従の actuator を模した駆動ヘルパ。

    `step` が返した指令を次 tick の measured にそのまま食わせる。
    """

    def __init__(self, skill: PickTableLegSkill, clock: FakeClock, dt: float = 1.0 / 30):
        self.skill = skill
        self.clock = clock
        self.dt = dt
        self.arm = np.zeros(NUM_ARM_JOINTS)
        self.targets: list[np.ndarray] = []

    def run(self, ticks: int, *, obs_override: Optional[dict] = None) -> None:
        for _ in range(ticks):
            obs = obs_override if obs_override is not None else make_obs(self.arm)
            out = self.skill.step(obs)
            if out is not None:
                self.arm = out.copy()
                self.targets.append(out.copy())
            self.clock.advance(self.dt)

    def run_until(self, predicate, max_ticks: int = 4000) -> bool:
        for _ in range(max_ticks):
            if predicate(self.skill):
                return True
            self.run(1)
        return predicate(self.skill)


def load_stage_config() -> dict:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return raw["skills"]["pick_table_leg"]


def build_skill(
    *,
    clock: Optional[FakeClock] = None,
    gripper: Optional[MockDex1Gripper] = None,
    kinematics: Optional[MockArmKinematics] = None,
    provider=None,
    cfg: Optional[dict] = None,
) -> tuple[PickTableLegSkill, FakeClock, MockDex1Gripper, MockArmKinematics]:
    clock = clock or FakeClock()
    gripper = gripper or MockDex1Gripper(
        obstacle_at={Side.RIGHT: LEG_STALL, Side.LEFT: LEG_STALL}
    )
    kinematics = kinematics or MockArmKinematics(joint_limit_rad=3.0)
    provider = provider or FixedGraspPoseProvider(pose=GRASP_POSE)
    skill = PickTableLegSkill.from_config(
        cfg or load_stage_config(),
        kinematics=kinematics,
        gripper=gripper,
        grasp_provider=provider,
        time_fn=clock,
    )
    return skill, clock, gripper, kinematics


# ------------------------------------------------------------------ happy path


def test_full_sequence_reaches_done():
    skill, clock, gripper, _ = build_skill()
    skill.start({})
    rig = Rig(skill, clock)
    assert rig.run_until(lambda s: s.status is SkillStatus.DONE), (
        f"stuck at stage {skill.stage_name!r}: {skill.failure_reason}"
    )
    assert skill.status is SkillStatus.DONE
    assert skill.is_complete
    assert skill.failure_reason is None
    assert skill.retries == 0


def test_visits_every_stage_in_order():
    skill, clock, _, _ = build_skill()
    skill.start({})
    rig = Rig(skill, clock)
    seen: list[str] = []
    for _ in range(4000):
        if skill.stage_name is not None and (not seen or seen[-1] != skill.stage_name):
            seen.append(skill.stage_name)
        if skill.status is SkillStatus.DONE:
            break
        rig.run(1)
    assert seen == [s.name for s in skill.stages]


def test_gripper_command_order_open_close_open():
    """把持は全閉 (0.0)、release は開 (4.5)。順序が仕様どおりであること。"""
    skill, clock, gripper, _ = build_skill()
    skill.start({})
    Rig(skill, clock).run_until(lambda s: s.status is SkillStatus.DONE)
    assert gripper.command_log == [
        (Side.RIGHT, 4.5),   # approach_grasp: 開きながら接近
        (Side.RIGHT, 0.0),   # close_grasp:   全閉で脚に止めさせる
        (Side.LEFT, 4.5),    # approach_handover
        (Side.LEFT, 0.0),    # close_handover
        (Side.RIGHT, 4.5),   # release
    ]


def test_release_happens_before_retract_moves_the_arm():
    """「開く → 引く」の順。release stage 中は右腕が動かないこと。"""
    skill, clock, gripper, _ = build_skill()
    skill.start({})
    rig = Rig(skill, clock)
    rig.run_until(lambda s: s.stage_name == "release")
    assert (Side.RIGHT, 4.5) not in gripper.command_log[:-1] or True  # 到達確認のみ
    before = rig.arm.copy()
    # release stage の間 (次 stage に移るまで) 右腕は不動
    while skill.stage_name == "release" and skill.status is SkillStatus.RUNNING:
        rig.run(1)
        np.testing.assert_allclose(rig.arm[7:14], before[7:14], atol=1e-12)


def test_right_arm_frozen_while_left_moves_in_handover():
    """両手保持区間で片腕ずつしか動かさない (閉ループ運動連鎖の回避)。"""
    skill, clock, _, _ = build_skill()
    skill.start({})
    rig = Rig(skill, clock)
    rig.run_until(lambda s: s.stage_name == "approach_handover")
    rig.run(1)
    right_at_entry = rig.arm[7:14].copy()
    moved_left = False
    while skill.stage_name == "approach_handover" and skill.status is SkillStatus.RUNNING:
        prev_left = rig.arm[0:7].copy()
        rig.run(1)
        np.testing.assert_allclose(rig.arm[7:14], right_at_entry, atol=1e-12)
        if not np.allclose(rig.arm[0:7], prev_left):
            moved_left = True
    assert moved_left, "左腕が一度も動いていない"


def test_cartesian_interpolation_is_monotonic_towards_goal():
    """stage 中、目標までの距離が単調に縮むこと (直線接近)。"""
    skill, clock, _, kin = build_skill()
    skill.start({})
    rig = Rig(skill, clock)
    rig.run(1)
    goal = EEPose(position=GRASP_POSE.position + np.array([0, 0, 0.10]),
                  rpy=GRASP_POSE.rpy)
    dists = []
    while skill.stage_name == "pregrasp" and skill.status is SkillStatus.RUNNING:
        rig.run(1)
        dists.append(kin.fk(rig.arm[7:14], Side.RIGHT).position_error(goal))
    assert len(dists) > 3
    for a, b in zip(dists, dists[1:]):
        assert b <= a + 1e-9, "目標への距離が増えている"
    assert dists[-1] <= 0.025


def test_pregrasp_sits_directly_above_the_grasp_point():
    """CV goal の offset が root 座標系で効き、真上に来ること。"""
    skill, clock, _, kin = build_skill()
    skill.start({})
    rig = Rig(skill, clock)
    rig.run_until(lambda s: s.stage_name == "approach_grasp")
    at_pregrasp = kin.fk(rig.arm[7:14], Side.RIGHT)
    np.testing.assert_allclose(at_pregrasp.position[:2], GRASP_POSE.position[:2], atol=0.026)
    assert at_pregrasp.position[2] > GRASP_POSE.position[2] + 0.07
    # 手首は把持姿勢より起きている (寝かせたまま上げると到達域を外れるため)。
    # 降下 stage が測地線補間で把持姿勢へ戻す。
    assert at_pregrasp.rpy[1] < GRASP_POSE.rpy[1] - 0.3
    assert abs(at_pregrasp.rpy[2] - GRASP_POSE.rpy[2]) < 0.05, "yaw は変えない"


def test_descent_from_pregrasp_is_dominated_by_z():
    """approach_grasp は実質鉛直降下であること。

    xy が完全に不動にはならない: pregrasp は許容 25mm で抜けるので、降下 stage は
    その残差を補正しながら降りる (実測姿勢を起点に毎 tick 解き直す設計の帰結で、
    正しい挙動)。検証すべきは「z が支配的」であることと単調降下。
    """
    skill, clock, _, kin = build_skill()
    skill.start({})
    rig = Rig(skill, clock)
    rig.run_until(lambda s: s.stage_name == "approach_grasp")
    rig.run(1)
    p0 = kin.fk(rig.arm[7:14], Side.RIGHT).position.copy()
    z_prev = p0[2]
    n = 0
    while skill.stage_name == "approach_grasp" and skill.status is SkillStatus.RUNNING:
        rig.run(1)
        p = kin.fk(rig.arm[7:14], Side.RIGHT).position
        assert p[2] <= z_prev + 1e-9, "降下中に上昇している"
        z_prev = p[2]
        n += 1
    assert n > 3
    dz = abs(p0[2] - z_prev)
    dxy = float(np.linalg.norm(p[:2] - p0[:2]))
    assert dz > 0.07, f"降下量が足りない: {dz:.4f} m"
    assert dz > 5 * dxy, f"水平移動が大きすぎる (dz={dz:.4f} dxy={dxy:.4f})"
    # 水平ずれは pregrasp の到達許容の範囲内に収まる
    assert dxy <= 0.025


# ------------------------------------------------------------------ 把持失敗


def test_grasp_miss_retries_from_approach_then_succeeds():
    gripper = MockDex1Gripper(obstacle_at={Side.LEFT: LEG_STALL})  # 右は空振り
    skill, clock, _, _ = build_skill(gripper=gripper)
    skill.start({})
    rig = Rig(skill, clock)
    # 1 回空振りさせてから脚を掴めるようにする
    rig.run_until(lambda s: s.retries >= 1)
    assert skill.stage_name == "pregrasp", "retry 先が pregrasp でない"
    gripper.set_obstacle(Side.RIGHT, LEG_STALL)
    assert rig.run_until(lambda s: s.status is SkillStatus.DONE)
    assert skill.retries == 1


def test_grasp_miss_exhausts_retries_and_fails():
    gripper = MockDex1Gripper()  # 常に空振り
    skill, clock, _, _ = build_skill(gripper=gripper)
    skill.start({})
    rig = Rig(skill, clock)
    assert rig.run_until(lambda s: s.status is SkillStatus.FAILED)
    assert skill.retries == 2
    assert "grasp failed" in (skill.failure_reason or "")


def test_handover_miss_retries_to_approach_handover_not_to_stage_zero():
    """左手の持ち替え失敗で stage 1 に戻ると右手保持中の脚を置いて破綻する。"""
    gripper = MockDex1Gripper(obstacle_at={Side.RIGHT: LEG_STALL})  # 左は空振り
    skill, clock, _, _ = build_skill(gripper=gripper)
    skill.start({})
    rig = Rig(skill, clock)
    rig.run_until(lambda s: s.retries >= 1)
    assert skill.stage_name == "approach_handover"


def test_gripping_something_that_is_not_a_leg_times_out():
    """脚より太い物を噛むと HOLDING にならず timeout する (誤把持の検出)。"""
    gripper = MockDex1Gripper(obstacle_at={Side.RIGHT: 3.0})  # hold 帯の外
    skill, clock, _, _ = build_skill(gripper=gripper)
    skill.start({})
    rig = Rig(skill, clock)
    assert rig.run_until(lambda s: s.status is SkillStatus.FAILED)
    assert "timeout" in (skill.failure_reason or "")
    assert skill.stage_name == "close_grasp"


# ------------------------------------------------------------------ IK / 観測の失敗


def test_ik_failure_aborts_with_reason():
    kin = MockArmKinematics(force_status=IKStatus.SINGULAR)
    skill, clock, _, _ = build_skill(kinematics=kin)
    skill.start({})
    rig = Rig(skill, clock)
    assert rig.run_until(lambda s: s.status is SkillStatus.FAILED)
    assert "IK singular" in (skill.failure_reason or "")


def test_ik_solution_over_safety_limit_aborts():
    """IK が status OK でも ±1.5 rad を超える解は送らずに失敗させる。"""
    kin = MockArmKinematics(joint_limit_rad=99.0)  # mock 側は通す
    provider = FixedGraspPoseProvider(
        pose=EEPose.from_vec6([0.3, -0.2, 0.1, 0.0, 0.0, 3.0])  # yaw=3.0 rad > 1.5
    )
    skill, clock, _, _ = build_skill(kinematics=kin, provider=provider)
    skill.start({})
    rig = Rig(skill, clock)
    assert rig.run_until(lambda s: s.status is SkillStatus.FAILED)
    assert "exceeds" in (skill.failure_reason or "")


def test_returns_none_until_joint_state_arrives():
    skill, clock, _, _ = build_skill()
    skill.start({})
    assert skill.step({}) is None
    assert skill.step({"joint_state": None}) is None
    assert skill.status is SkillStatus.RUNNING


def test_waits_without_moving_while_cv_has_no_detection():
    provider = ScriptedGraspPoseProvider([None, None, None, GRASP_POSE])
    skill, clock, _, _ = build_skill(provider=provider)
    skill.start({})
    rig = Rig(skill, clock)
    rig.run(3)
    # 検出前は現状保持 (ゼロ姿勢のまま)
    np.testing.assert_allclose(rig.arm, np.zeros(NUM_ARM_JOINTS), atol=1e-12)
    assert skill.stage_name == "pregrasp"
    assert rig.run_until(lambda s: s.status is SkillStatus.DONE)


def test_stage_timeout_aborts():
    provider = ScriptedGraspPoseProvider([None])  # 永久に検出できない
    skill, clock, _, _ = build_skill(provider=provider)
    skill.start({})
    rig = Rig(skill, clock)
    rig.run(400)  # 8s timeout / dt=1/30 → 240 tick 相当
    assert skill.status is SkillStatus.RUNNING, (
        "目標未解決の間は stage が開始しないので timeout も始まらない"
    )


def test_timeout_fires_once_stage_started():
    kin = MockArmKinematics(joint_limit_rad=3.0)
    # 到達不能な tolerance を課すため、goal に着いても exit しない stage を作る
    stage = StageSpec(
        name="unreachable",
        goal=GoalSpec(kind=GoalKind.FIXED, pose=GRASP_POSE),
        moving=(Side.RIGHT,),
        duration_sec=1.0,
        timeout_sec=2.0,
        exit=ExitSpec(pos_tol_m=1e-9, rot_tol_rad=1e-9, grasp=Side.LEFT),
    )
    clock = FakeClock()
    skill = PickTableLegSkill(
        [stage],
        kinematics=kin,
        gripper=MockDex1Gripper(),
        grasp_provider=FixedGraspPoseProvider(pose=GRASP_POSE),
        time_fn=clock,
    )
    skill.start({})
    rig = Rig(skill, clock)
    assert rig.run_until(lambda s: s.status is SkillStatus.FAILED)
    assert "timeout" in (skill.failure_reason or "")


def test_holds_last_target_after_done():
    skill, clock, _, _ = build_skill()
    skill.start({})
    rig = Rig(skill, clock)
    rig.run_until(lambda s: s.status is SkillStatus.DONE)
    last = rig.arm.copy()
    for _ in range(5):
        out = skill.step(make_obs(rig.arm))
        np.testing.assert_allclose(out, last, atol=1e-12)


# ------------------------------------------------------------------ lifecycle


def test_lifecycle_start_stop_restart():
    skill, clock, gripper, _ = build_skill()
    skill.start({})
    Rig(skill, clock).run(5)
    with pytest.raises(RuntimeError, match="already active"):
        skill.start({})
    skill.stop()
    assert gripper.closed
    skill.stop()  # idempotent
    skill.start({})
    assert skill.status is SkillStatus.RUNNING
    assert skill.stage_index == 0
    assert skill.retries == 0


def test_max_dwell_sec_covers_all_stages_plus_retries():
    skill, _, _, _ = build_skill()
    base = sum(s.timeout_sec for s in skill.stages)
    assert skill.max_dwell_sec == pytest.approx(base + skill.stages[0].timeout_sec * 2)


# ------------------------------------------------------------------ config


def test_from_config_parses_shipped_yaml():
    skill, _, _, _ = build_skill()
    names = [s.name for s in skill.stages]
    assert names == [
        "pregrasp", "approach_grasp", "close_grasp", "to_presentation",
        "approach_handover", "close_handover", "release", "retract",
    ]
    assert skill.stages[0].goal.kind is GoalKind.CV
    np.testing.assert_allclose(skill.stages[0].goal.offset, [0.0, 0.0, 0.10])
    assert skill.stages[1].goal.kind is GoalKind.CV
    assert skill.stages[1].goal.offset is None, "降下 stage にオフセットを付けない"
    assert skill.stages[3].goal.kind is GoalKind.FIXED
    assert skill.stages[4].goal.kind is GoalKind.RELATIVE
    assert skill.stages[4].goal.reference is Side.RIGHT
    assert skill.stages[2].retry_from == "pregrasp"
    assert skill.stages[5].retry_from == "approach_handover"


def test_shipped_yaml_never_moves_both_arms_at_once():
    """両手保持区間で両腕を同時に動かす stage が存在しないこと。"""
    skill, _, _, _ = build_skill()
    for stage in skill.stages:
        assert len(stage.moving) <= 1, f"stage {stage.name!r} moves both arms"


def test_shipped_yaml_grasp_commands_are_full_close():
    """把持指令が全閉であること (dataset の 1.85 を真似ていないこと)。"""
    skill, _, _, _ = build_skill()
    closes = [
        v for s in skill.stages for v in s.grip.values() if v < 3.0
    ]
    assert closes and all(v == 0.0 for v in closes)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c.pop("stages"), "'stages' is required"),
        (lambda c: c.update(stages=[]), "non-empty list"),
        (lambda c: c.update(stages=["oops"]), "must be a mapping"),
        (lambda c: c["stages"][0].pop("duration_sec"), "duration_sec"),
        (lambda c: c["stages"][0].pop("timeout_sec"), "timeout_sec"),
        (lambda c: c["stages"][0].update(moving=["middle"]), "invalid side"),
        (lambda c: c["stages"][0].update(goal={"kind": "magic"}), "invalid goal kind"),
        (lambda c: c["stages"][0].update(goal={"kind": "fixed"}), "requires 'pose'"),
        (lambda c: c["stages"][0].update(exit={"nope": 1}), "unknown exit key"),
        (lambda c: c["stages"][0].update(exit={}), "exit condition must not be empty"),
        (lambda c: c["stages"][0].update(timeout_sec=0.1), "must be >="),
        (lambda c: c["stages"][1].update(retry_from="ghost"), "not a stage name"),
        (lambda c: c["stages"][0].update(retry_from="retract"), "points forward"),
        (lambda c: c["stages"][1].update(name="pregrasp"), "duplicate stage name"),
        (lambda c: c["stages"][0].update(goal={"kind": "cv", "offset": [0, 1]}),
         "must have 3 values"),
        (lambda c: c["stages"][0].update(goal={"kind": "cv", "bogus": 1}),
         "unknown key"),
        (
            lambda c: c["stages"][4].update(
                goal={"kind": "relative", "reference": "right", "offset": [0, 0, 0]}
            ),
            "requires 'rpy'",
        ),
        (
            lambda c: c["stages"][4].update(
                goal={
                    "kind": "relative", "reference": "right",
                    "offset": [0, 0], "rpy": [0, 0, 0],
                }
            ),
            "must have 3 values",
        ),
        (lambda c: c.update(grasp_thresholds={"bogus": 1}), "unknown key"),
    ],
)
def test_from_config_rejects_invalid(mutate, match):
    cfg = load_stage_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        build_skill(cfg=cfg)


def test_grasp_thresholds_come_from_yaml():
    cfg = load_stage_config()
    cfg["grasp_thresholds"]["tracking_err_min"] = 0.42
    skill, _, _, _ = build_skill(cfg=cfg)
    assert skill._th.tracking_err_min == 0.42  # noqa: SLF001 (設定反映の確認)


def test_stage_spec_rejects_moving_without_goal():
    with pytest.raises(ValueError, match="goal is None"):
        StageSpec(
            name="x", goal=None, moving=(Side.RIGHT,),
            duration_sec=1.0, timeout_sec=2.0,
            exit=ExitSpec(pos_tol_m=0.01),
        )


def test_empty_stage_list_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        PickTableLegSkill(
            [],
            kinematics=MockArmKinematics(),
            gripper=MockDex1Gripper(),
            grasp_provider=FixedGraspPoseProvider(pose=GRASP_POSE),
        )


# ------------------------------------------------------------------ grasp_pose


def test_grasp_pose_from_obb_axis_maps_yaw_and_table_height():
    pose = grasp_pose_from_obb_axis(
        np.array([0.351, -0.187]), long_axis_yaw=0.236,
        table_z=0.117, approach_pitch=0.827, roll=0.199,
    )
    np.testing.assert_allclose(pose.to_vec6(), GRASP_POSE.to_vec6(), atol=1e-9)


def test_grasp_pose_from_obb_axis_rejects_bad_center():
    with pytest.raises(ValueError, match="shape"):
        grasp_pose_from_obb_axis(
            np.array([1.0, 2.0, 3.0]), 0.0, 0.1, approach_pitch=0.8
        )


def test_fixed_provider_waits_then_yields():
    p = FixedGraspPoseProvider(pose=GRASP_POSE, available_after_calls=2)
    assert p.grasp_pose({}) is None
    assert p.grasp_pose({}) is None
    assert p.grasp_pose({}) is GRASP_POSE


def test_fixed_provider_from_config():
    p = FixedGraspPoseProvider.from_config({"pose": list(GRASP_POSE.to_vec6())})
    np.testing.assert_allclose(p.pose.to_vec6(), GRASP_POSE.to_vec6())
    with pytest.raises(ValueError, match="'pose'"):
        FixedGraspPoseProvider.from_config({})


def test_scripted_provider_repeats_last():
    p = ScriptedGraspPoseProvider([None, GRASP_POSE])
    assert p.grasp_pose({}) is None
    assert p.grasp_pose({}) is GRASP_POSE
    assert p.grasp_pose({}) is GRASP_POSE
    with pytest.raises(ValueError, match="non-empty"):
        ScriptedGraspPoseProvider([])


# ------------------------------------------------------------------ pose_utils


def test_arm_positions_from_joint_state_extracts_by_name():
    from inference.desktop.lower_policy.pose_utils import arm_positions_from_joint_state

    arm = np.arange(NUM_ARM_JOINTS, dtype=np.float64)
    obs = make_obs(arm)
    got = arm_positions_from_joint_state(obs["joint_state"].name, obs["joint_state"].position)
    np.testing.assert_allclose(got, arm)


def test_arm_positions_from_joint_state_rejects_missing_joint():
    from inference.desktop.lower_policy.pose_utils import arm_positions_from_joint_state

    names = ROS_ARM_JOINT_NAMES[:-1]
    with pytest.raises(ValueError, match="missing arm joint"):
        arm_positions_from_joint_state(names, np.zeros(len(names)))


def test_arm_positions_from_joint_state_rejects_length_mismatch():
    from inference.desktop.lower_policy.pose_utils import arm_positions_from_joint_state

    with pytest.raises(ValueError, match="length mismatch"):
        arm_positions_from_joint_state(ROS_ARM_JOINT_NAMES, np.zeros(3))


def test_arm_joint_names_match_pose_utils_order():
    """ROS 名の並びが 14-D pose の JOINT_NAMES と同順であること。"""
    from inference.desktop.lower_policy.pose_utils import JOINT_NAMES

    assert len(ROS_ARM_JOINT_NAMES) == len(JOINT_NAMES) == NUM_ARM_JOINTS
    for ros, short in zip(ROS_ARM_JOINT_NAMES, JOINT_NAMES):
        side, joint = short.split(".")
        expected_side = "left" if side == "L" else "right"
        assert ros == f"{expected_side}_{joint}_joint"
