"""LeftHandZCeiling の unit test (Issue #137 Phase 3)。"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from inference.desktop.lower_policy.skills.z_ceiling import (
    LEFT_ARM_19D,
    LeftHandZCeiling,
)
from inference.desktop.perception.g1_urdf_fk import G1WristFK


# rotate_table_base の skill_config wrist_tool_offset (runtime と同じ FK 設定)
ROTATE_LEFT_OFFSET = np.array([0.091, 0.010, 0.085])
ROTATE_RIGHT_OFFSET = np.array([0.119, -0.078, -0.023])
HARD_FLOOR_M = 0.1121  # 教師 1401 セグメント 最下点 p5

# 実機 run の開始姿勢 (skill_config.yaml rotate_table_base.initial_pose)
INITIAL_ARM_14 = np.array([
    -0.4830605686, 0.4441821277, -0.0769655555, 0.2223441899,
    0.0807874203, 0.4568341970, -0.5851719379,
    -0.3696676791, -0.1454999149, 0.0941407904, 0.2638106048,
    0.1134036779, 0.3263158500, 0.4706455171,
])


@pytest.fixture(scope="module")
def fk() -> G1WristFK:
    return G1WristFK.from_urdf(
        left_tool_offset=ROTATE_LEFT_OFFSET, right_tool_offset=ROTATE_RIGHT_OFFSET
    )


@pytest.fixture(scope="module")
def initial_target() -> np.ndarray:
    target = np.zeros(19, dtype=np.float64)
    target[3:17] = INITIAL_ARM_14
    return target


def _ceiling(fk: G1WristFK, **kwargs) -> LeftHandZCeiling:
    return LeftHandZCeiling(fk, hard_floor_m=HARD_FLOOR_M, **kwargs)


def test_no_ceiling_is_a_noop(fk, initial_target) -> None:
    clamp = _ceiling(fk)
    result = clamp.apply(initial_target, None)
    assert not result.bind
    assert np.array_equal(result.target_19d, initial_target)


def test_target_below_ceiling_is_untouched(fk, initial_target) -> None:
    clamp = _ceiling(fk)
    z = clamp.left_tool_z(initial_target)
    result = clamp.apply(initial_target, z + 0.05)
    assert not result.bind
    assert result.correction_m == 0.0
    assert np.array_equal(result.target_19d, initial_target)


def test_input_array_is_not_mutated(fk, initial_target) -> None:
    clamp = _ceiling(fk)
    snapshot = initial_target.copy()
    clamp.apply(initial_target, HARD_FLOOR_M + 0.005)
    assert np.array_equal(initial_target, snapshot)


def test_correction_is_capped_per_tick(fk, initial_target) -> None:
    clamp = _ceiling(fk, max_correction_m=0.005)
    z0 = clamp.left_tool_z(initial_target)
    result = clamp.apply(initial_target, HARD_FLOOR_M)  # 大きく超過している
    assert result.bind
    # 1 tick の下げ幅は cap 付近 (DLS の残差ぶんだけ僅かに足りない)
    assert -0.0055 < result.correction_m < -0.0045
    assert result.z_after_m < z0


def test_repeated_ticks_converge_to_ceiling(fk, initial_target) -> None:
    """毎 tick 5mm ずつ下がって天井に漸近し、天井を割り込まない。"""
    clamp = _ceiling(fk, max_correction_m=0.005)
    ceiling = HARD_FLOOR_M + 0.02
    target = initial_target
    for _ in range(60):
        target = clamp.apply(target, ceiling).target_19d
    z = clamp.left_tool_z(target)
    assert z == pytest.approx(ceiling, abs=0.002)


def test_never_goes_below_hard_floor(fk, initial_target) -> None:
    clamp = _ceiling(fk, max_correction_m=0.05)
    target = initial_target
    for _ in range(80):
        result = clamp.apply(target, HARD_FLOOR_M)
        target = result.target_19d
    assert clamp.left_tool_z(target) >= HARD_FLOOR_M - 1e-3


def test_ceiling_below_hard_floor_is_rejected(fk, initial_target) -> None:
    clamp = _ceiling(fk)
    with pytest.raises(ValueError, match="below hard_floor_m"):
        clamp.apply(initial_target, HARD_FLOOR_M - 0.001)


def test_only_left_arm_joints_are_modified(fk, initial_target) -> None:
    clamp = _ceiling(fk)
    result = clamp.apply(initial_target, HARD_FLOOR_M + 0.01)
    assert result.bind
    changed = ~np.isclose(result.target_19d, initial_target)
    touched = set(np.flatnonzero(changed).tolist())
    assert touched <= set(range(LEFT_ARM_19D.start, LEFT_ARM_19D.stop))


def test_joint_step_is_bounded(fk, initial_target) -> None:
    clamp = _ceiling(fk, max_correction_m=0.5, max_joint_step_rad=0.01)
    result = clamp.apply(initial_target, HARD_FLOOR_M)
    assert result.max_joint_step_rad <= 0.01 + 1e-9


def test_single_tick_lands_on_the_ceiling_when_no_cap_binds(fk, initial_target) -> None:
    """cap を外し、関節上限にも当たらない下げ幅なら 1 tick で天井に乗る。"""
    clamp = _ceiling(fk, max_correction_m=1.0)
    ceiling = clamp.left_tool_z(initial_target) - 0.02
    result = clamp.apply(initial_target, ceiling)
    assert result.bind
    assert result.z_after_m == pytest.approx(ceiling, abs=0.001)
    assert result.max_joint_step_rad < 0.05


def test_large_gap_is_limited_by_joint_step_cap(fk, initial_target) -> None:
    """1 tick の下げ幅が大きすぎる場合、関節上限が効いて途中までしか下がらない。"""
    clamp = _ceiling(fk, max_correction_m=1.0, max_joint_step_rad=0.05)
    ceiling = HARD_FLOOR_M + 0.03  # 約 64mm の要求
    result = clamp.apply(initial_target, ceiling)
    assert result.bind
    assert result.max_joint_step_rad == pytest.approx(0.05, abs=1e-9)
    assert ceiling < result.z_after_m < result.z_before_m


def test_numeric_jacobian_matches_finite_difference(fk, initial_target) -> None:
    """Jacobian が実際の z 変化を予測できていること。"""
    clamp = _ceiling(fk)
    jz = clamp._numeric_dz_jacobian(initial_target)
    dq = np.full(7, 1e-3)
    moved = initial_target.copy()
    moved[LEFT_ARM_19D] += dq
    predicted = float(jz @ dq)
    actual = clamp.left_tool_z(moved) - clamp.left_tool_z(initial_target)
    assert predicted == pytest.approx(actual, abs=1e-5)


@pytest.mark.parametrize("bad", [
    {"hard_floor_m": 0.0}, {"hard_floor_m": 1.5},
    {"max_correction_m": 0.0}, {"damping": 0.0},
    {"max_joint_step_rad": -1.0}, {"fd_step_rad": 0.0},
])
def test_invalid_config_is_rejected(fk, bad: dict) -> None:
    kwargs = {"hard_floor_m": HARD_FLOOR_M}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        LeftHandZCeiling(fk, **kwargs)


def test_requires_fk_with_left_tool_position() -> None:
    with pytest.raises(TypeError):
        LeftHandZCeiling(object(), hard_floor_m=HARD_FLOOR_M)


def test_initial_pose_z_is_pinned(fk, initial_target) -> None:
    """回帰: skill_config の initial_pose (指令値) の左手先 z を固定する。

    この値がズレたら corridor (教師 最下点 median 122.8 / p5 112.1mm) の前提が
    崩れるので、閾値の出所と一緒に固定しておく。実機の policy 開始時点の実測は
    201.6mm で、指令値との 4.6mm 差は pre-motion の残差 (実測
    max_measured_error_rad 0.014)。
    """
    z_mm = fk.left_tool_position(
        np.concatenate([np.zeros(12), np.zeros(3), INITIAL_ARM_14])
    )[2] * 1000
    assert z_mm == pytest.approx(206.2, abs=0.5)


@pytest.mark.integration
def test_clamp_on_real_teacher_trajectory_stays_bounded(fk) -> None:
    """教師 1401 セグメントの実姿勢に天井をかけても発散しないこと。

    `TestFkCrossVerifyTrainingData` と同じ方針で、HF cache に
    `Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base` の parquet が
    あれば走り、無ければ skip。marker gate なので default pytest では collect のみ。
    """
    import glob

    pq = pytest.importorskip("pyarrow.parquet")
    matches = sorted(glob.glob(
        str(pathlib.Path.home()
            / ".cache/huggingface/hub"
            / "datasets--Team-RAMEN--ramen-g1-ikea-assembly-ramen-ori-rotate-table-base"
            / "snapshots/*/data/chunk-000/*.parquet")
    ))
    if not matches:
        pytest.skip("rotate_table_base merged dataset not in HF cache")

    column = "observation.state.robot_q_current"
    table = pq.read_table(matches[0], columns=[column])
    q36 = np.stack(table[column].to_pylist()).astype(np.float64)
    sample = q36[np.linspace(0, len(q36) - 1, 300).astype(int)]

    ceiling = 0.1228  # 教師 最下点 median
    clamp = _ceiling(fk, max_correction_m=0.005)
    binds = 0
    for row in sample:
        target = np.zeros(19, dtype=np.float64)
        target[0:3] = row[7 + 12: 7 + 15]     # waist
        target[3:17] = row[7 + 15: 7 + 29]    # arms 14
        result = clamp.apply(target, ceiling)
        assert result.max_joint_step_rad <= 0.05 + 1e-9
        if result.bind:
            binds += 1
            assert result.z_after_m < result.z_before_m
            # 押し下げた結果が下限を割らないこと。教師自身は 5% のストロークで
            # p5 より深くまで降りている (実測 最小 80.8mm) が、それは
            # `bind=False` の素通り経路なので下限の対象外。
            assert result.z_after_m >= HARD_FLOOR_M - 1e-3
    # 教師は上げ下げするので一部は天井を超える。全frame binding も 0 も異常。
    assert 0 < binds < len(sample)


def _build_skill(z_ceiling, ceiling_m=None):
    import time

    from inference.desktop.lower_policy.actuators.hand import MockHandActuator
    from inference.desktop.lower_policy.actuators.waist import MockWaistActuator
    from inference.desktop.lower_policy.skills.tests.test_vla_skill import _StubPolicy
    from inference.desktop.lower_policy.skills.vla_skill import MoveTableBaseVlaSkill

    skill = MoveTableBaseVlaSkill(
        policy=_StubPolicy(chunk_value=0.0),
        waist_actuator=MockWaistActuator(),
        hand_actuator=MockHandActuator(),
        z_ceiling=z_ceiling,
    )
    skill._z_ceiling_target_m = ceiling_m
    skill.start({})
    obs = {
        "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_left_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_right_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "joint_state": None,
        "cleaned": [],
        "t": time.monotonic_ns(),
    }
    return skill, obs


def test_vla_skill_without_ceiling_is_unchanged(fk) -> None:
    skill_off, obs = _build_skill(None)
    skill_on, _ = _build_skill(_ceiling(fk), ceiling_m=None)
    assert np.allclose(skill_off.step(dict(obs)), skill_on.step(dict(obs)))
    assert "z_ceiling_bind" not in skill_off.last_action.metadata
    assert skill_on.last_action.metadata["z_ceiling_bind"] is False


def test_vla_skill_applies_ceiling_before_motion_limiter(fk) -> None:
    clamp = _ceiling(fk, max_correction_m=0.005)
    skill_off, obs = _build_skill(None)
    arm_off = skill_off.step(dict(obs))
    # StubPolicy の chunk[0] が実現する z より低い天井を置く
    target = np.zeros(19, dtype=np.float64)
    target[3:17] = arm_off
    ceiling = clamp.left_tool_z(target) - 0.004
    skill_on, _ = _build_skill(clamp, ceiling_m=ceiling)
    arm_on = skill_on.step(dict(obs))

    meta = skill_on.last_action.metadata
    assert meta["z_ceiling_bind"] is True
    assert meta["z_after_mm"] < meta["z_before_mm"]
    assert meta["z_ceiling_target_mm"] == pytest.approx(ceiling * 1000.0)
    # 左腕 7 joint だけが動く (14D arm target の前半)
    assert not np.allclose(arm_on[:7], arm_off[:7])
    assert np.allclose(arm_on[7:], arm_off[7:])


def test_apply_discards_correction_when_solution_undershoots_hard_floor(
    fk, initial_target
) -> None:
    """PR #138 review: DLS の解が hard_floor を割ったら補正を破棄する。

    `_solve_dq` を「行き過ぎる」ものに差し替え、出力側の防御だけを見る。
    入口の `ceiling_m >= hard_floor_m` は満たしたままにする。
    """
    clamp = _ceiling(fk)
    z0 = clamp.left_tool_z(initial_target)
    jz = clamp._numeric_dz_jacobian(initial_target)
    overshoot = z0 - HARD_FLOOR_M + 0.05  # hard_floor を確実に割る量
    clamp._solve_dq = lambda t19, dz: jz * (-overshoot / float(jz @ jz))

    result = clamp.apply(initial_target, z0 - 0.001)

    assert result.floor_rejected is True
    assert result.bind is False
    assert result.correction_m == 0.0
    np.testing.assert_allclose(result.target_19d, initial_target)
    assert clamp.left_tool_z(result.target_19d) >= clamp.hard_floor_m


def test_apply_keeps_correction_when_solution_stays_above_hard_floor(
    fk, initial_target
) -> None:
    """通常経路は従来どおり (回帰防止)。"""
    clamp = _ceiling(fk)
    z0 = clamp.left_tool_z(initial_target)

    result = clamp.apply(initial_target, z0 - 0.002)

    assert result.floor_rejected is False
    assert result.bind is True
    assert result.z_after_m >= clamp.hard_floor_m
