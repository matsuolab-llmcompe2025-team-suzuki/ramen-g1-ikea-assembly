"""RotateRetryController の unit test (Issue #137 Phase 4)。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from inference.desktop.lower_policy.skills.rotate_retry import (
    ReturnMode,
    RetryPhase,
    RotateRetryController,
    arm_return_complete,
    load_retry_controller_for_skill,
)


RUNGS = [None, 0.1228, 0.1121]
STEP_NS = int(1e9 / 30)


@dataclass(frozen=True)
class _Progress:
    usable: bool = True
    sustained: bool = False


def _controller(**kwargs) -> RotateRetryController:
    kwargs.setdefault("rungs", RUNGS)
    kwargs.setdefault("attempt_seconds", 3.0)
    return RotateRetryController(**kwargs)


def _run(controller, *, ticks, t0=0, progress=_Progress(), return_complete=False):
    out = None
    for i in range(ticks):
        out = controller.update(
            t_ns=t0 + i * STEP_NS, progress=progress, return_complete=return_complete
        )
    return out


def test_starts_running_on_the_bare_model() -> None:
    controller = _controller()
    decision = controller.update(t_ns=0, progress=_Progress(), return_complete=False)
    assert decision.phase is RetryPhase.RUNNING
    assert decision.ceiling_m is None      # 段 1 = 素のモデル
    assert decision.attempt_index == 0
    assert not decision.flush_execution_state


def test_attempt_timeout_starts_returning() -> None:
    controller = _controller(attempt_seconds=1.0)
    decision = _run(controller, ticks=40)  # 40 tick = 1.30s
    assert decision.phase is RetryPhase.RETURNING
    assert decision.ceiling_m is None      # RETURNING 中は天井を適用しない
    assert decision.reason == "attempt_timeout"


def test_return_completion_advances_the_rung_and_flushes() -> None:
    controller = _controller(attempt_seconds=1.0)
    _run(controller, ticks=40)
    decision = controller.update(
        t_ns=41 * STEP_NS, progress=_Progress(), return_complete=True
    )
    assert decision.phase is RetryPhase.RUNNING
    assert decision.attempt_index == 1
    assert decision.ceiling_m == pytest.approx(0.1228)
    assert decision.flush_execution_state
    assert decision.reason == "return_complete"


def test_flush_is_raised_only_on_the_boundary_tick() -> None:
    controller = _controller(attempt_seconds=1.0)
    _run(controller, ticks=40)
    controller.update(t_ns=41 * STEP_NS, progress=_Progress(), return_complete=True)
    following = controller.update(
        t_ns=42 * STEP_NS, progress=_Progress(), return_complete=False
    )
    assert not following.flush_execution_state


def test_return_timeout_advances_even_without_completion() -> None:
    controller = _controller(attempt_seconds=1.0, return_timeout_s=0.5)
    _run(controller, ticks=40)
    decision = _run(controller, ticks=30, t0=41 * STEP_NS)  # 戻り 1.0s、未到達
    assert decision.phase is RetryPhase.RUNNING
    assert decision.attempt_index == 1
    assert decision.reason == "return_timeout"


def test_rungs_descend_through_all_stages() -> None:
    controller = _controller(attempt_seconds=1.0)
    seen = []
    t = 0
    for _ in range(3):
        _run(controller, ticks=40, t0=t)
        seen.append(controller.update(
            t_ns=t + 41 * STEP_NS, progress=_Progress(), return_complete=True
        ).ceiling_m)
        t += 42 * STEP_NS
    assert seen[0] == pytest.approx(0.1228)
    assert seen[1] == pytest.approx(0.1121)
    # 最終段では戻らずそのまま回し続ける
    assert seen[2] == pytest.approx(0.1121)


def test_last_rung_keeps_running_instead_of_stopping() -> None:
    controller = _controller(rungs=[None], attempt_seconds=1.0)
    decision = _run(controller, ticks=60)
    assert decision.phase is RetryPhase.RUNNING
    assert decision.reason == "last_rung_continues"


def test_success_holds_and_never_returns_on_its_own() -> None:
    """成功後は上位 policy の遷移待ち。自分から RUNNING に戻らない。"""
    controller = _controller(attempt_seconds=1.0)
    decision = controller.update(
        t_ns=0, progress=_Progress(sustained=True), return_complete=False
    )
    assert decision.phase is RetryPhase.HOLDING
    assert decision.reason == "rotation_detected"
    later = _run(controller, ticks=200, t0=STEP_NS)
    assert later.phase is RetryPhase.HOLDING


def test_unusable_progress_is_not_treated_as_success() -> None:
    controller = _controller(attempt_seconds=1.0)
    decision = controller.update(
        t_ns=0,
        progress=_Progress(usable=False, sustained=True),
        return_complete=False,
    )
    assert decision.phase is RetryPhase.RUNNING


def test_missing_progress_is_not_success() -> None:
    controller = _controller(attempt_seconds=1.0)
    decision = controller.update(t_ns=0, progress=None, return_complete=False)
    assert decision.phase is RetryPhase.RUNNING


def test_reset_returns_to_first_rung() -> None:
    controller = _controller(attempt_seconds=1.0)
    _run(controller, ticks=40)
    controller.update(t_ns=41 * STEP_NS, progress=_Progress(), return_complete=True)
    controller.reset()
    decision = controller.update(t_ns=0, progress=_Progress(), return_complete=False)
    assert decision.attempt_index == 0
    assert decision.ceiling_m is None


@pytest.mark.parametrize("bad", [
    {"rungs": []},
    {"rungs": [None, 0.1121, 0.1228]},   # 昇順 = 段が進むほど浅い
    {"rungs": [None, -0.1]},
    {"attempt_seconds": 0.0},
    {"return_timeout_s": -1.0},
])
def test_invalid_config_is_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        _controller(**bad)


def test_arm_return_completion_uses_worst_joint() -> None:
    target = np.zeros(14)
    near = np.full(14, 0.04)
    far = near.copy()
    far[7] = 0.20
    assert arm_return_complete(near, target, 0.05)
    assert not arm_return_complete(far, target, 0.05)


def test_loader_returns_none_without_retry_block() -> None:
    controller, options = load_retry_controller_for_skill(
        {"skills": {"rotate_table_base": {}}}, "rotate_table_base"
    )
    assert controller is None
    assert options == {}


def test_loader_converts_mm_to_metres() -> None:
    cfg = {"skills": {"rotate_table_base": {"retry": {
        "rungs_mm": [None, 122.8, 112.1],
        "attempt_seconds": 3.0,
        "return_mode": "lift_and_reapproach",
        "lift_margin_mm": 10.0,
        "max_seconds_hard": 15,
    }}}}
    controller, options = load_retry_controller_for_skill(cfg, "rotate_table_base")
    assert controller.rungs == [None, pytest.approx(0.1228), pytest.approx(0.1121)]
    assert options["return_mode"] is ReturnMode.LIFT_AND_REAPPROACH
    assert options["lift_margin_m"] == pytest.approx(0.010)
    assert options["max_seconds_hard"] == 15


def test_loader_rejects_unknown_keys() -> None:
    cfg = {"skills": {"rotate_table_base": {"retry": {"rung_mm": [None]}}}}
    with pytest.raises(ValueError, match="unknown keys"):
        load_retry_controller_for_skill(cfg, "rotate_table_base")


# =========================================================================
# VlaSkill 統合 (FSM が実際に policy 呼び出しと腕目標を切り替えること)
# =========================================================================
def _skill_with_retry(**kwargs):
    import time

    from inference.desktop.lower_policy.actuators.hand import MockHandActuator
    from inference.desktop.lower_policy.actuators.waist import MockWaistActuator
    from inference.desktop.lower_policy.skills.tests.test_vla_skill import _StubPolicy
    from inference.desktop.lower_policy.skills.vla_skill import MoveTableBaseVlaSkill

    class CountingPolicy(_StubPolicy):
        def __init__(self) -> None:
            super().__init__(chunk_value=0.0)
            self.predicts = 0
            self.resets = 0

        def predict(self, observation):  # noqa: ANN001, ANN201
            self.predicts += 1
            return super().predict(observation)

        def reset(self) -> None:
            self.resets += 1

    policy = CountingPolicy()
    initial_arm = np.full(14, 0.25)
    skill = MoveTableBaseVlaSkill(
        policy=policy,
        waist_actuator=MockWaistActuator(),
        hand_actuator=MockHandActuator(),
        retry_controller=RotateRetryController(rungs=RUNGS, attempt_seconds=1.0),
        retry_options={
            "return_mode": ReturnMode.FULL_RETURN,
            "return_tolerance_rad": 0.05,
            "initial_arm_14": initial_arm,
        },
        **kwargs,
    )
    skill.start({})

    def obs_at(tick: int, joints: np.ndarray | None = None) -> dict:
        class _Joints:
            position = np.zeros(29) if joints is None else joints

        return {
            "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_left_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_right_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "joint_state": _Joints(),
            "hand_state": None,
            "cleaned": [],
            "t": tick * STEP_NS + int(time.monotonic_ns()),
        }

    return skill, policy, obs_at, initial_arm


def test_returning_stops_calling_policy_and_drives_to_initial_pose() -> None:
    skill, policy, obs_at, initial_arm = _skill_with_retry()
    for tick in range(40):           # 1.30s > attempt_seconds 1.0s
        arm = skill.step(obs_at(tick))
    meta = skill.last_action.metadata
    assert meta["retry_phase"] == RetryPhase.RETURNING.value
    predicts_at_return = policy.predicts
    # RETURNING 中は policy を呼ばず、腕は初期姿勢へ向かう
    for tick in range(40, 50):
        arm = skill.step(obs_at(tick))
    assert policy.predicts == predicts_at_return
    assert np.all(np.abs(arm[:14] - initial_arm) < np.abs(0.0 - initial_arm))


def test_reaching_initial_pose_advances_rung_and_flushes_policy() -> None:
    skill, policy, obs_at, initial_arm = _skill_with_retry()
    for tick in range(40):
        skill.step(obs_at(tick))
    resets_before = policy.resets
    measured = np.zeros(29)
    measured[15:29] = initial_arm     # 戻り完了状態を measured で与える
    skill.step(obs_at(40, measured))
    meta = skill.last_action.metadata
    assert meta["retry_phase"] == RetryPhase.RUNNING.value
    assert meta["retry_attempt_index"] == 1
    assert meta["retry_reason"] == "return_complete"
    assert policy.resets == resets_before + 1


def test_no_retry_controller_keeps_calling_policy_every_tick() -> None:
    import time

    from inference.desktop.lower_policy.actuators.hand import MockHandActuator
    from inference.desktop.lower_policy.actuators.waist import MockWaistActuator
    from inference.desktop.lower_policy.skills.tests.test_vla_skill import _StubPolicy
    from inference.desktop.lower_policy.skills.vla_skill import MoveTableBaseVlaSkill

    skill = MoveTableBaseVlaSkill(
        policy=_StubPolicy(chunk_value=0.0),
        waist_actuator=MockWaistActuator(),
        hand_actuator=MockHandActuator(),
    )
    skill.start({})
    obs = {
        "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_left_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_right_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "joint_state": None,
        "cleaned": [],
        "t": time.monotonic_ns(),
    }
    for _ in range(50):
        skill.step(dict(obs))
    assert "retry_phase" not in skill.last_action.metadata


def test_repo_skill_config_retry_block_is_disabled_by_default() -> None:
    """既定は現状動作。有効化は yaml のコメントを外すだけ。"""
    import pathlib as _pathlib

    import yaml

    path = (
        _pathlib.Path(__file__).resolve().parents[2]
        / "configs" / "skill_config.yaml"
    )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    controller, options = load_retry_controller_for_skill(cfg, "rotate_table_base")
    assert controller is None
    assert options == {}


def test_holding_keeps_the_successful_rung_ceiling() -> None:
    """PR #138 review: HOLDING は拘束を解く状態ではない。

    成功検知で HOLDING に入った後も、成功した段の天井を返し続けること。
    評価経路 (上位 policy 無し) では max_seconds まで HOLDING が続くので、
    ここで None を返すとその間モデルが自由に潜れてしまう。
    """
    controller = _controller()
    # 1 段目を時間切れ → 戻り完了で 2 段目 (0.1228) へ
    _run(controller, ticks=95)
    decision = controller.update(
        t_ns=96 * STEP_NS, progress=_Progress(), return_complete=True
    )
    assert decision.phase is RetryPhase.RUNNING
    assert decision.ceiling_m == RUNGS[1]

    success = _Progress(usable=True, sustained=True)
    decision = controller.update(
        t_ns=97 * STEP_NS, progress=success, return_complete=False
    )
    assert decision.phase is RetryPhase.HOLDING
    assert decision.ceiling_m == RUNGS[1]

    decision = controller.update(
        t_ns=300 * STEP_NS, progress=success, return_complete=False
    )
    assert decision.phase is RetryPhase.HOLDING
    assert decision.ceiling_m == RUNGS[1]


def test_returning_still_releases_the_ceiling() -> None:
    """RETURNING は lift_to で持ち上げる局面なので天井なしのまま。"""
    controller = _controller()
    decision = _run(controller, ticks=95)
    assert decision.phase is RetryPhase.RETURNING
    assert decision.ceiling_m is None
