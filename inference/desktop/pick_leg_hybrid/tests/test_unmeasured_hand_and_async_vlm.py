"""手の実測が無い経路 (Codex #2) と VLM の非同期化 (Codex #8)。"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from inference.desktop.perception.g1_urdf_fk import G1WristFK
from inference.desktop.pick_leg_hybrid.boundary import (
    BoundaryConfig,
    GraspBoundaryDetector,
)
from inference.desktop.pick_leg_hybrid.config import DEFAULT_CONFIG_PATH, load_yaml
from inference.desktop.pick_leg_hybrid.interlock import (
    LEFT,
    RIGHT,
    GraspInterlockConfig,
    UnmeasuredHandConfig,
)
from inference.desktop.pick_leg_hybrid.phase2 import EePose
from inference.desktop.pick_leg_hybrid.phase3_rule import (
    Phase3RuleController,
    Phase3RuleStage,
)
from inference.desktop.pick_leg_hybrid.phases import Phase
from inference.desktop.pick_leg_hybrid.real_skill import RealPickLegHybridVlaSkill
from inference.desktop.pick_leg_hybrid.tests.test_phase3_rule import _config
from inference.desktop.pick_leg_hybrid.tests.test_real_skill import (
    _Actuator,
    _observation,
    _Policy,
)
from inference.desktop.pick_leg_hybrid.vlm import VlmAnswer


# ---------------------------------------------------------------- async VLM
class _GatedVlm:
    """応答を gate が開くまで返さない VLM (HTTP の遅延を模す)。"""

    def __init__(self, value: int = 1) -> None:
        self.gate = threading.Event()
        self.calls = 0
        self.value = value

    def ask(self, *_args, **_kwargs) -> VlmAnswer:
        self.calls += 1
        self.gate.wait(5.0)
        return VlmAnswer(value=self.value)


def _detector(vlm, **cfg) -> GraspBoundaryDetector:
    params = dict(confirm_count=1, min_interval_sec=0.0, max_answer_age_sec=6.0)
    params.update(cfg)
    return GraspBoundaryDetector(
        cfg=BoundaryConfig(**params),
        vlm_client=vlm,
        reference_images_b64=["A", "B"],
        async_vlm=True,
    )


def _update(detector, t, *, interlock=True, images=None):
    return detector.update(
        t=t,
        hand_state=[4.5, 2.3],
        hand_cmd=[4.5, 1.8],
        images_b64=images if images is not None else ["img"],
        interlock=interlock,
    )


def _wait_pending_done(detector) -> None:
    deadline = time.monotonic() + 2.0
    while detector._pending is not None and not detector._pending.done:
        assert time.monotonic() < deadline
        time.sleep(0.001)


def test_async_vlm_never_blocks_the_control_tick():
    vlm = _GatedVlm()
    detector = _detector(vlm)
    started = time.monotonic()
    first = _update(detector, 0.0)
    assert time.monotonic() - started < 0.1  # HTTP を待たない
    assert first.vlm_submitted and first.vlm_pending and not first.fire
    second = _update(detector, 0.033)
    assert not second.vlm_submitted  # 1 本だけ
    vlm.gate.set()
    _wait_pending_done(detector)
    third = _update(detector, 0.066)
    assert third.vlm_called and third.fire


def test_answer_from_before_an_interlock_drop_is_discarded():
    vlm = _GatedVlm()
    detector = _detector(vlm)
    _update(detector, 0.0)
    _update(detector, 0.1, interlock=False)  # 手が外れた
    vlm.gate.set()
    time.sleep(0.05)
    out = _update(detector, 0.2)
    assert not out.fire
    assert out.vlm_submitted  # 新しい把持について問い直す


def test_stale_answer_is_discarded():
    vlm = _GatedVlm()
    detector = _detector(vlm, max_answer_age_sec=1.0)
    _update(detector, 0.0)
    vlm.gate.set()
    _wait_pending_done(detector)
    out = _update(detector, 5.0)
    assert out.vlm_discarded.startswith("stale")
    assert not out.fire


def test_images_are_encoded_only_when_a_query_is_submitted():
    vlm = _GatedVlm()
    detector = _detector(vlm)
    encoded = []

    def images():
        encoded.append(1)
        return ["img"]

    _update(detector, 0.0, images=images)
    _update(detector, 0.033, images=images)
    _update(detector, 0.066, images=images)
    assert len(encoded) == 1
    vlm.gate.set()


def test_reset_invalidates_the_running_query():
    vlm = _GatedVlm()
    detector = _detector(vlm)
    _update(detector, 0.0)
    detector.reset()
    vlm.gate.set()
    time.sleep(0.05)
    out = _update(detector, 0.1)
    assert not out.fire and out.vlm_submitted


# ------------------------------------------------ Phase 1 (real skill) unmeasured
class _YesVlm:
    def ask(self, *_args, **_kwargs) -> VlmAnswer:
        return VlmAnswer(value=1)


def _skill(**kwargs) -> RealPickLegHybridVlaSkill:
    kwargs.setdefault("next_initial_arm_target", np.zeros(14))
    kwargs.setdefault("next_initial_hand_target", np.array([3.0, 1.85]))
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(),
        waist_actuator=_Actuator(),
        hand_actuator=_Actuator(),
        fk=G1WristFK.from_urdf(),
        dispatch_waist=False,
        hybrid_config_path=DEFAULT_CONFIG_PATH,
        **kwargs,
    )
    skill._boundary = GraspBoundaryDetector(
        cfg=BoundaryConfig(confirm_count=1, min_interval_sec=0.0),
        interlock_cfg=skill._hybrid_cfg.interlock,
        vlm_client=_YesVlm(),
        reference_images_b64=["A", "B"],
    )
    return skill


def _echo_obs(t_ns: int, *, measured: bool) -> dict:
    """合成 state = 指令エコー (policy の右手 1.8 がそのまま state に返る)。"""
    obs = _observation(t_ns)
    obs["hand_state"] = SimpleNamespace(
        position_rad=np.array([4.5, 1.8], dtype=np.float64), measured=measured
    )
    return obs


def test_production_detector_is_asynchronous():
    skill = RealPickLegHybridVlaSkill(
        policy=_Policy(),
        waist_actuator=_Actuator(),
        hand_actuator=_Actuator(),
        fk=G1WristFK.from_urdf(),
        dispatch_waist=False,
        hybrid_config_path=DEFAULT_CONFIG_PATH,
        next_initial_arm_target=np.zeros(14),
        next_initial_hand_target=np.array([3.0, 1.85]),
    )
    assert skill._boundary.async_vlm is True


def test_echoed_state_never_passes_the_measured_interlock():
    """再現: 合成 state を実測扱いすると差が 0 で Phase 1 から進まない。"""
    skill = _skill()
    skill.start({})
    for i in range(1, 60):
        skill.step(_echo_obs(i * 100_000_000, measured=True))
    assert skill.hybrid_phase is Phase.APPROACH_GRASP
    skill.stop()


def test_unmeasured_hand_uses_close_command_settle_then_vlm():
    pytest.importorskip("cv2")  # VLM へ渡す JPEG を実際に作る
    skill = _skill()
    skill.start({})
    settle = skill._hybrid_cfg.unmeasured_hand.phase1_command_settle_sec
    skill.step(_echo_obs(1_000_000_000, measured=False))
    assert skill.hybrid_phase is Phase.APPROACH_GRASP  # まだ待ち時間
    t_ns = int((1.0 + settle + 0.05) * 1e9)
    skill.step(_echo_obs(t_ns, measured=False))
    assert skill.hybrid_phase is Phase.CARRY_TO_LEFT
    skill.stop()


def test_unmeasured_hand_without_config_fails_closed():
    skill = _skill()
    object.__setattr__(skill._hybrid_cfg, "unmeasured_hand", None)
    skill.start({})
    skill.step(_echo_obs(1_000_000_000, measured=False))
    skill.step(_echo_obs(3_000_000_000, measured=False))
    assert skill.hybrid_phase is Phase.APPROACH_GRASP
    assert "unmeasured_hand" in (skill.failure_reason or "")
    skill.stop()


# ------------------------------------------------------------- Phase 3 unmeasured
_UNMEASURED = UnmeasuredHandConfig(
    phase1_command_settle_sec=1.0,
    phase3_grasp_command_rad=1.85,
    phase3_hand_settle_sec=0.5,
)


def _unmeasured_controller() -> Phase3RuleController:
    return Phase3RuleController(
        _config(),
        GraspInterlockConfig(),
        insert_arm_target=np.linspace(-0.35, 0.35, 14),
        insert_hand_target=np.array([3.0, 1.85]),
        unmeasured_cfg=_UNMEASURED,
    )


def test_unmeasured_phase3_commands_the_grasp_value_not_the_full_close_probe():
    ctl = _unmeasured_controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    arm = np.zeros(14)
    ctl.start(0.0, left, right, arm, hand_measured=False)
    out = ctl.step(
        t=0.1, left=left, right=right,
        hand_state=np.array([1.85, 1.85]), previous_hand_command=np.array([1.85, 1.85]),
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.CLOSE_LEFT  # 待ち時間の前は進まない
    assert out.hand[LEFT] == pytest.approx(1.85)  # 全閉 0.0 ではない
    out = ctl.step(
        t=0.6, left=left, right=right,
        hand_state=out.hand, previous_hand_command=out.hand, arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.OPEN_RIGHT
    assert out.failure_reason is None  # 支えの喪失を誤検出しない


def test_unmeasured_phase3_open_waits_for_the_settle_time():
    ctl = _unmeasured_controller()
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    right = EePose.of([0.3, -0.1, 0.1], [1, 0, 0, 0])
    arm = np.zeros(14)
    ctl.start(0.0, left, right, arm, hand_measured=False)
    out = ctl.step(
        t=0.6, left=left, right=right,
        hand_state=np.array([1.85, 1.85]), previous_hand_command=np.array([1.85, 1.85]),
        arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.OPEN_RIGHT
    out = ctl.step(
        t=0.7, left=left, right=right,
        hand_state=out.hand, previous_hand_command=out.hand, arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.OPEN_RIGHT
    assert out.hand[RIGHT] == pytest.approx(4.5)
    out = ctl.step(
        t=1.2, left=left, right=right,
        hand_state=out.hand, previous_hand_command=out.hand, arm_state=arm,
    )
    assert out.stage is Phase3RuleStage.MOVE_LEFT_PRESENT


def test_unmeasured_phase3_without_config_refuses_to_start():
    ctl = Phase3RuleController(
        _config(),
        GraspInterlockConfig(),
        insert_arm_target=np.zeros(14),
        insert_hand_target=np.array([3.0, 1.85]),
    )
    left = EePose.of([0.2, 0.1, 0.2], [1, 0, 0, 0])
    with pytest.raises(ValueError, match="unmeasured_hand"):
        ctl.start(0.0, left, left, np.zeros(14), hand_measured=False)


def test_settle_time_longer_than_a_stage_timeout_is_rejected():
    with pytest.raises(ValueError, match="phase3_hand_settle_sec"):
        Phase3RuleController(
            _config(),
            GraspInterlockConfig(),
            insert_arm_target=np.zeros(14),
            insert_hand_target=np.array([3.0, 1.85]),
            unmeasured_cfg=UnmeasuredHandConfig(1.0, 1.85, 5.0),
        )


def test_production_yaml_unmeasured_values_fit_the_phase3_timeouts():
    cfg = load_yaml(DEFAULT_CONFIG_PATH)
    assert cfg.unmeasured_hand is not None
    assert cfg.boundary.max_answer_age_sec > cfg.vlm.timeout_sec
    Phase3RuleController(
        cfg.phase3_rule_based,
        cfg.interlock,
        insert_arm_target=np.zeros(14),
        insert_hand_target=np.array([3.0, 1.85]),
        unmeasured_cfg=cfg.unmeasured_hand,
    )
