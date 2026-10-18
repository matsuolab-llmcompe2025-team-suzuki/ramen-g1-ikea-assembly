"""実G1 entrypointの停止処理に対する副作用なしの契約テスト。"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from inference.desktop import entrypoint as entrypoint_module
from inference.desktop.lower_policy.policies.load_plan import PlannedModel
from inference.desktop.entrypoint import (
    HEAD_CAMERA_STEREO_VIEW,
    HEAD_PERCEPTION_VIEW,
    PHASE1_11_POLICY_VARIANT,
    PRODUCTION_HYBRID_PICK_VARIANT,
    _DEFAULT_LOG_DIR,
    _POLICY_SLOT_ATTRS,
    _default_log_path,
    _positive_float,
    _insert_policy_start_gate_transition,
    _selected_policy_start_gate_stage,
    apply_default_policy_variants,
    configure_official_endpoints,
    resolve_policy_alternatives,
    resolve_production_pick_mode,
    required_policy_slots,
    _require_strict_regular,
    _safe_shutdown,
    _start_hand_during_registry_build,
    _validate_phase1_11_config,
    _validate_phase1_12_config,
    _validate_phase1_13_config,
    _validate_phase3_config,
    _validate_rule_pick_geometry_snapshot,
    _wait_for_phase1_11_sources,
    load_verified_rule_pick_calibration,
)


def test_boundary_endpoints_come_from_environment_not_a_robot_constant(monkeypatch):
    monkeypatch.setenv("IROS_ORIN_HOST", "10.20.30.40")
    args = argparse.Namespace(
        action_sink="boundary",
        head_source="ros2",
        joint_source="lowstate",
        zmq_endpoint=None,
        boundary_state_host=None,
    )
    configure_official_endpoints(args)
    assert args.head_source == "zmq"
    assert args.joint_source == "boundary"
    assert args.zmq_endpoint == "tcp://10.20.30.40:5555"
    assert args.boundary_state_host == "10.20.30.40"


def test_boundary_endpoint_is_required_when_environment_is_absent(monkeypatch):
    monkeypatch.delenv("IROS_ORIN_HOST", raising=False)
    monkeypatch.delenv("G1_IMAGE_SERVER_IP", raising=False)
    args = argparse.Namespace(
        action_sink="boundary",
        head_source="ros2",
        joint_source="lowstate",
        zmq_endpoint=None,
        boundary_state_host=None,
    )
    with pytest.raises(ValueError, match="requires --zmq-endpoint"):
        configure_official_endpoints(args)


def test_boundary_state_host_is_inferred_from_explicit_camera_endpoint(monkeypatch):
    monkeypatch.delenv("IROS_ORIN_HOST", raising=False)
    monkeypatch.delenv("G1_IMAGE_SERVER_IP", raising=False)
    args = argparse.Namespace(
        action_sink="boundary",
        head_source="ros2",
        joint_source="lowstate",
        zmq_endpoint="tcp://event-orin.local:5555",
        boundary_state_host=None,
    )
    configure_official_endpoints(args)
    assert args.head_source == "zmq"
    assert args.joint_source == "boundary"
    assert args.boundary_state_host == "event-orin.local"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (argparse.Namespace(stage=0, phase3_full=False), None),
        (argparse.Namespace(stage=1, phase3_full=False), 1),
        (argparse.Namespace(stage=5, phase3_full=False), 5),
        (
            argparse.Namespace(
                stage=None,
                phase3_full=True,
                phase3_start_stage=2,
                phase3_end_stage=2,
            ),
            2,
        ),
        (
            argparse.Namespace(
                stage=None,
                phase3_full=True,
                phase3_start_stage=1,
                phase3_end_stage=5,
            ),
            None,
        ),
    ],
)
def test_policy_start_gate_is_only_for_an_independently_selected_policy_stage(
    args, expected
):
    assert _selected_policy_start_gate_stage(args) == expected


def test_policy_start_gate_is_inserted_only_before_first_policy():
    transitions = {
        "hold_pose_for_rotate_table_base": ["rotate_table_base"],
        "rotate_table_base": ["arm_transition_rotate_table_base_to_pick_table_leg"],
    }
    gate = _insert_policy_start_gate_transition(
        transitions, first_policy="rotate_table_base"
    )
    assert gate == "operator_gate_for_rotate_table_base"
    assert transitions["hold_pose_for_rotate_table_base"] == [gate]
    assert transitions[gate] == ["rotate_table_base"]
    assert transitions["rotate_table_base"] == [
        "arm_transition_rotate_table_base_to_pick_table_leg"
    ]


def test_rule_pick_geometry_preflight_uses_head_left_and_reports_safe_pose():
    frame = SimpleNamespace(rgb=np.zeros((480, 1280, 3), dtype=np.uint8))
    perception = MagicMock()
    leg = SimpleNamespace(class_name="leg", confidence=0.9)
    perception.predict.return_value = [leg]
    provider = MagicMock()
    provider.leg_class = "leg"
    provider.min_confidence = 0.5
    provider.last_leg_length = 0.36
    provider.last_grasp_fraction = 0.39
    pose = SimpleNamespace(
        position=np.array([0.36, -0.16, 0.10]),
        rpy=np.array([0.18, 0.87, 0.30]),
    )
    provider.grasp_pose.return_value = pose
    joint_state = object()

    assert (
        _validate_rule_pick_geometry_snapshot(
            head_frame=frame,
            joint_state=joint_state,
            perception=perception,
            grasp_provider=provider,
        )
        is pose
    )
    assert perception.predict.call_args.args[0].shape == (480, 640, 3)
    provider.grasp_pose.assert_called_once_with(
        {"cleaned": [leg], "joint_state": joint_state}
    )


def test_rule_pick_geometry_preflight_fails_closed_without_safe_pose():
    frame = SimpleNamespace(rgb=np.zeros((480, 1280, 3), dtype=np.uint8))
    perception = MagicMock()
    perception.predict.return_value = [
        SimpleNamespace(class_name="leg", confidence=0.9)
    ]
    provider = MagicMock()
    provider.leg_class = "leg"
    provider.min_confidence = 0.5
    provider.grasp_pose.return_value = None
    provider.last_reject_reason = "leg_too_short"
    provider.last_leg_length = 0.22

    with pytest.raises(RuntimeError, match="no safe reachable grasp"):
        _validate_rule_pick_geometry_snapshot(
            head_frame=frame,
            joint_state=object(),
            perception=perception,
            grasp_provider=provider,
        )


def test_phase3_never_starts_dex1_during_registry_build():
    assert not _start_hand_during_registry_build(
        phase1_profile=None, phase3_active=True
    )
    assert not _start_hand_during_registry_build(
        phase1_profile="1.13", phase3_active=True
    )
    assert _start_hand_during_registry_build(phase1_profile=None, phase3_active=False)


def test_policy_resources_close_once_in_reverse_registration_order():
    entrypoint_module._OPEN_POLICY_RESOURCES.clear()
    first = MagicMock()
    second = MagicMock()
    order: list[str] = []
    first.close.side_effect = lambda: order.append("first")
    second.close.side_effect = lambda: order.append("second")

    entrypoint_module._register_policy_resource(first)
    entrypoint_module._register_policy_resource(second)
    entrypoint_module._register_policy_resource(first)

    assert entrypoint_module._close_policy_resources() == []
    assert order == ["first", "second"]
    assert entrypoint_module._OPEN_POLICY_RESOURCES == []


def test_safe_shutdown_stops_velocity_without_entering_damp():
    """arm_actuator 無し (`--no-arm-actuator` 相当 or 起動失敗ケース) の shutdown。"""
    actuator = MagicMock()
    log_sink = MagicMock()
    source = MagicMock()

    _safe_shutdown(actuator, log_sink, source=source)

    actuator.set_velocity.assert_called_once_with(0.0, 0.0, 0.0, duration=1.0)
    actuator.damp.assert_not_called()
    actuator.stand_up.assert_not_called()
    source.close.assert_called_once_with()
    log_sink.close.assert_called_once_with()


def test_safe_shutdown_stops_walk_then_controlled_releases_arm():
    """Base stops immediately while arm holds, then arm_sdk ramps to zero."""
    walk_actuator = MagicMock()
    arm_actuator = MagicMock()
    log_sink = MagicMock()
    call_order: list[str] = []
    arm_actuator.controlled_release.side_effect = lambda **kwargs: call_order.append(
        "arm_release"
    )
    arm_actuator.stop.side_effect = lambda: call_order.append("arm_stop")
    walk_actuator.set_velocity.side_effect = lambda *a, **k: call_order.append(
        "walk_stop"
    )

    _safe_shutdown(
        walk_actuator,
        log_sink,
        arm_actuator=arm_actuator,
        arm_actuator_started=True,
    )

    assert call_order == ["walk_stop", "arm_release", "arm_stop"]
    arm_actuator.controlled_release.assert_called_once_with(duration_s=2.0)
    arm_actuator.stop.assert_called_once_with()
    walk_actuator.set_velocity.assert_called_once_with(0.0, 0.0, 0.0, duration=1.0)
    log_sink.close.assert_called_once_with()


def test_safe_shutdown_swallows_arm_stop_exception():
    """arm.stop() が raise しても walk 停止 / log close は続行 (二段構え保護)。"""
    walk_actuator = MagicMock()
    arm_actuator = MagicMock()
    arm_actuator.stop.side_effect = RuntimeError("simulated arm stop failure")
    log_sink = MagicMock()

    _safe_shutdown(
        walk_actuator,
        log_sink,
        arm_actuator=arm_actuator,
        arm_actuator_started=True,
    )

    walk_actuator.set_velocity.assert_called_once()
    log_sink.close.assert_called_once()


def test_safe_shutdown_does_not_touch_unstarted_arm():
    walk_actuator = MagicMock()
    arm_actuator = MagicMock()

    _safe_shutdown(walk_actuator, None, arm_actuator=arm_actuator)

    arm_actuator.controlled_release.assert_not_called()
    arm_actuator.stop.assert_not_called()
    walk_actuator.set_velocity.assert_not_called()


def _phase_args(**overrides):
    values = {
        "policy_variant": PHASE1_11_POLICY_VARIANT,
        "use_real_waist": False,
        "use_real_hand": False,
        "synthetic_hand_state": False,
        "action_sink": "sdk",
        "no_wrist_cameras": False,
        "policy_variant_pick": None,
        "policy_variant_insert": None,
        "policy_variant_rotate_leg": None,
        "policy_variant_flip": None,
        "pick_leg_hybrid": False,
        "pick_leg_phase3_executor": "rule_based",
        "rule_based_pick_table_leg": False,
        "policy_variant_rotate_table_base": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _phase_skills():
    return {
        "setup": {"stages": [{"dwell_sec": 2.0}, {"dwell_sec": 2.0}]},
        "move_to_table": {"vx": 0.185, "max_dwell_sec": 1.0},
    }


def test_phase1_11_contract_accepts_exact_arm_only_flow():
    _validate_phase1_11_config(_phase_args(), _phase_skills())


@pytest.mark.parametrize(
    "override,match",
    [
        ({"policy_variant": None}, "requires"),
        ({"use_real_waist": True}, "arm-only"),
        ({"use_real_hand": True}, "arm-only"),
        ({"no_wrist_cameras": True}, "wrist-camera"),
        ({"policy_variant_pick": "x"}, "only the move_table_base"),
    ],
)
def test_phase1_11_contract_rejects_scope_expansion(override, match):
    with pytest.raises(ValueError, match=match):
        _validate_phase1_11_config(_phase_args(**override), _phase_skills())


def test_phase1_11_contract_rejects_distance_change():
    skills = _phase_skills()
    skills["move_to_table"]["max_dwell_sec"] = 1.1
    with pytest.raises(ValueError, match="walk dwell mismatch"):
        _validate_phase1_11_config(_phase_args(), skills)


def test_phase1_12_contract_accepts_exact_arm_waist_flow():
    _validate_phase1_12_config(_phase_args(use_real_waist=True), _phase_skills())


@pytest.mark.parametrize(
    "override,match",
    [
        ({"policy_variant": None}, "requires"),
        ({"use_real_waist": False}, "real waist"),
        ({"use_real_waist": True, "use_real_hand": True}, "Dex1 Mock"),
        ({"use_real_waist": True, "no_wrist_cameras": True}, "wrist-camera"),
        (
            {"use_real_waist": True, "policy_variant_pick": "x"},
            "only the move_table_base",
        ),
    ],
)
def test_phase1_12_contract_rejects_scope_change(override, match):
    with pytest.raises(ValueError, match=match):
        _validate_phase1_12_config(_phase_args(**override), _phase_skills())


def test_phase1_13_contract_accepts_exact_full_upper_body_flow():
    _validate_phase1_13_config(
        _phase_args(use_real_waist=True, use_real_hand=True), _phase_skills()
    )


@pytest.mark.parametrize(
    "override,match",
    [
        ({"policy_variant": None}, "requires"),
        ({"use_real_waist": False, "use_real_hand": True}, "real waist"),
        ({"use_real_waist": True, "use_real_hand": False}, "real waist"),
        (
            {"use_real_waist": True, "use_real_hand": True, "no_wrist_cameras": True},
            "wrist-camera",
        ),
        (
            {"use_real_waist": True, "use_real_hand": True, "policy_variant_pick": "x"},
            "only the move_table_base",
        ),
    ],
)
def test_phase1_13_contract_rejects_scope_change(override, match):
    with pytest.raises(ValueError, match=match):
        _validate_phase1_13_config(_phase_args(**override), _phase_skills())


def test_require_strict_regular_is_read_only_and_fail_closed():
    actuator = MagicMock()
    actuator.get_loco_status.return_value = argparse.Namespace(fsm_id=501, fsm_mode=0)
    _require_strict_regular(actuator)
    actuator.get_loco_status.assert_called_once_with()
    actuator.set_velocity.assert_not_called()

    actuator.get_loco_status.return_value = argparse.Namespace(fsm_id=501, fsm_mode=1)
    with pytest.raises(RuntimeError, match="strict Regular"):
        _require_strict_regular(actuator)


def test_phase1_11_source_gate_waits_for_all_inputs(monkeypatch):
    monkeypatch.setattr("inference.desktop.entrypoint.time.sleep", lambda _: None)
    sources = [MagicMock() for _ in range(5)]
    for source in sources:
        source.get.return_value = object()

    _wait_for_phase1_11_sources(
        head_source=sources[0],
        wrist_left_source=sources[1],
        wrist_right_source=sources[2],
        joint_state_source=sources[3],
        dex1_state_source=sources[4],
        timeout_s=0.1,
    )
    for source in sources:
        source.get.assert_called()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_positive_float_rejects_unsafe_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_float(value)


def test_positive_float_accepts_valid_value():
    assert _positive_float("0.5") == 0.5


def _phase3_args(stage: int | None, **overrides):
    values = {
        "stage": stage,
        "phase3_full": False,
        "phase3_start_stage": 0,
        "phase3_end_stage": 5,
        "actuate": False,
        "no_wrist_cameras": False,
        "use_real_waist": False,
        "use_real_hand": False,
        "policy_variant": None,
        "policy_variant_rotate_table_base": None,
        "policy_variant_pick": None,
        "policy_variant_insert": None,
        "policy_variant_rotate_leg": None,
        "policy_variant_flip": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


REAL_POLICY_CONFIG = Path("inference/desktop/lower_policy/configs/policy_config.yaml")


def _fill(args):
    """本番 config の既定で埋める (Issue #148)。"""
    return apply_default_policy_variants(args, config_path=REAL_POLICY_CONFIG)


def test_default_variants_are_not_applied_to_stage_zero():
    # stage 0 は歩行と腕の pre-motion だけ。1 つでも載ると検査が弾く。
    args = _phase3_args(0)
    assert _fill(args) == {}
    assert args.policy_variant_pick is None
    _validate_phase3_config(args)


def test_default_variants_fill_only_the_slots_the_stage_needs():
    args = _phase3_args(1, use_real_waist=True, use_real_hand=True)
    source = _fill(args)
    assert source == {"pick": "config", "insert": "config", "rotate_leg": "config"}
    assert args.policy_variant_pick == "pick_table_leg_ramen_ori_venue_all6_20k"
    # stage 1 は 1 本目なので卓を回さない。埋めると「使わない policy」で弾かれる。
    assert args.policy_variant_rotate_table_base is None
    assert args.policy_variant_flip is None
    _validate_phase3_config(args)


def test_sdk_leg_stage_defaults_to_the_learned_pick_of_the_config():
    """本番の config は pick_leg_hybrid: false (Issue #188)。hybrid は R の候補に残す。"""
    args = _phase3_args(
        2,
        pick_leg_hybrid=None,
        use_real_waist=True,
        use_real_hand=True,
    )
    assert resolve_production_pick_mode(args) is False
    assert args.pick_leg_hybrid is False
    _fill(args)
    assert args.policy_variant_pick == "pick_table_leg_ramen_ori_venue_all6_20k"
    _validate_phase3_config(args)


def test_explicit_learned_only_pick_is_not_silently_overridden():
    args = _phase3_args(2, pick_leg_hybrid=False)
    assert resolve_production_pick_mode(args) is False
    assert args.pick_leg_hybrid is False


def test_non_pick_stage_does_not_enable_hybrid_pick():
    args = _phase3_args(5, pick_leg_hybrid=None)
    assert resolve_production_pick_mode(args) is False
    assert args.pick_leg_hybrid is False


def test_boundary_path_follows_the_same_pick_mode_as_the_sdk_path():
    # 会場の経路だからといって pick のやり方は変わらない (どちらも config の pick_leg_hybrid)
    args = _phase3_args(
        2,
        pick_leg_hybrid=None,
        action_sink="boundary",
        synthetic_hand_state=True,
    )
    assert resolve_production_pick_mode(args) is False
    assert args.pick_leg_hybrid is False


def test_default_variants_add_rotate_table_from_stage_two():
    args = _phase3_args(2, use_real_waist=True, use_real_hand=True)
    assert _fill(args)["rotate_table"] == "config"
    assert (
        args.policy_variant_rotate_table_base
        == "rotate_table_base_ramen_ori_venue_all6_20k"
    )
    _validate_phase3_config(args)


def test_default_variants_fill_only_flip_for_stage_five():
    args = _phase3_args(5, use_real_hand=True)
    assert _fill(args) == {"flip": "config"}
    assert args.policy_variant_flip == "flip_table_ramen_ori_flip_sel_a_n15_20k"
    assert args.policy_variant_pick is None
    _validate_phase3_config(args)


def test_cli_variant_wins_over_the_config_default():
    args = _phase3_args(
        2,
        use_real_waist=True,
        use_real_hand=True,
        policy_variant_rotate_table_base="rotate_table_base_ramen_ori_all6_400k",
    )
    source = _fill(args)
    assert source["rotate_table"] == "cli"
    assert source["pick"] == "config"
    assert (
        args.policy_variant_rotate_table_base == "rotate_table_base_ramen_ori_all6_400k"
    )
    _validate_phase3_config(args)


def test_continuous_run_fills_every_slot_in_the_stage_range():
    args = _phase3_args(
        None,
        phase3_full=True,
        phase3_start_stage=0,
        phase3_end_stage=5,
        use_real_waist=True,
        use_real_hand=True,
    )
    assert set(_fill(args)) == {"pick", "insert", "rotate_leg", "rotate_table", "flip"}
    _validate_phase3_config(args)


def test_continuous_run_without_flip_stage_does_not_fill_flip():
    args = _phase3_args(
        None,
        phase3_full=True,
        phase3_start_stage=0,
        phase3_end_stage=4,
        use_real_waist=True,
        use_real_hand=True,
    )
    assert "flip" not in _fill(args)
    assert args.policy_variant_flip is None
    _validate_phase3_config(args)


def test_required_slots_match_the_validated_stage_rule():
    assert required_policy_slots(_phase3_args(0)) == ()
    assert required_policy_slots(_phase3_args(1)) == ("pick", "insert", "rotate_leg")
    assert required_policy_slots(_phase3_args(3)) == (
        "pick",
        "insert",
        "rotate_leg",
        "rotate_table",
    )
    assert required_policy_slots(_phase3_args(5)) == ("flip",)
    assert required_policy_slots(_phase3_args(None)) == ()


def test_unknown_skill_in_the_default_section_is_rejected(tmp_path):
    config = tmp_path / "policy_config.yaml"
    config.write_text(
        "default_variant_by_skill:\n"
        "  rotate_table: groot_pick_legs_v1\n"   # slot 名であって skill 名ではない
        "policies:\n"
        "  groot_pick_legs_v1:\n"
        "    policy_type: groot_pick_legs\n"
        "    mode: none\n"
        "    ckpt_ref: 'repo@sha'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown skills"):
        apply_default_policy_variants(_phase3_args(1), config_path=config)


def test_missing_default_for_a_required_slot_is_rejected(tmp_path):
    config = tmp_path / "policy_config.yaml"
    config.write_text(
        "default_variant_by_skill:\n"
        "  pick_table_leg: groot_pick_legs_v1\n"
        "policies:\n"
        "  groot_pick_legs_v1:\n"
        "    policy_type: groot_pick_legs\n"
        "    mode: none\n"
        "    ckpt_ref: 'repo@sha'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing 'insert_table_leg'"):
        apply_default_policy_variants(_phase3_args(1), config_path=config)


def test_phase3_stage0_accepts_command_free_preflight_and_actuation():
    _validate_phase3_config(_phase3_args(0))
    _validate_phase3_config(_phase3_args(0, actuate=True))


def test_phase3_rejects_actuate_without_stage():
    with pytest.raises(ValueError, match="only supported with"):
        _validate_phase3_config(_phase3_args(None, actuate=True))


def test_rule_based_pick_only_accepts_read_only_and_actuation():
    _validate_phase3_config(_phase3_args(None, rule_based_pick_table_leg=True))
    _validate_phase3_config(
        _phase3_args(None, rule_based_pick_table_leg=True, actuate=True)
    )


def test_rule_pick_calibration_is_rejected_outside_dedicated_mode(tmp_path):
    with pytest.raises(ValueError, match="requires --rule-based"):
        _validate_phase3_config(
            _phase3_args(None, rule_based_pick_calibration=tmp_path / "calib.yaml")
        )


def test_verified_rule_pick_calibration_loader_is_fail_closed(tmp_path):
    path = tmp_path / "calib.yaml"
    path.write_text(
        """
calibration_verified: true
calibration_id: head-mount-20260907
camera: {fx: 510.0, fy: 510.0, cx: 320.0, cy: 240.0, width: 640, height: 480}
extrinsic: {xyz: [0.1, 0.0, 0.4], rpy: [0.0, 0.8, 0.0]}
support_plane: {normal: [0.0, 0.0, 1.0], offset: 0.1}
""",
        encoding="utf-8",
    )
    result = load_verified_rule_pick_calibration(path)
    assert set(result) == {"camera", "extrinsic", "support_plane"}

    path.write_text(path.read_text().replace("true", "false"), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly true"):
        load_verified_rule_pick_calibration(path)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"stage": 1}, "mutually exclusive"),
        ({"policy_variant_pick": "learned"}, "must not load"),
        ({"use_real_waist": True}, "forbidden"),
        ({"use_real_hand": True}, "forbidden"),
        ({"no_wrist_cameras": True}, "requires both wrist"),
    ],
)
def test_rule_based_pick_only_rejects_scope_expansion(override, match):
    stage = override.get("stage")
    extra = {key: value for key, value in override.items() if key != "stage"}
    with pytest.raises(ValueError, match=match):
        _validate_phase3_config(
            _phase3_args(
                stage,
                rule_based_pick_table_leg=True,
                **extra,
            )
        )


def test_phase3_full_requires_all_policies_and_real_upper_body():
    valid = _phase3_args(
        None,
        phase3_full=True,
        actuate=True,
        use_real_waist=True,
        use_real_hand=True,
        policy_variant_rotate_table_base="rotate-table",
        policy_variant_pick="pick",
        policy_variant_insert="insert",
        policy_variant_rotate_leg="rotate-leg",
        policy_variant_flip="flip",
    )
    _validate_phase3_config(valid)
    valid.policy_variant_flip = None
    with pytest.raises(ValueError, match="missing policy variants: flip"):
        _validate_phase3_config(valid)


def test_phase3_full_stage_five_uses_flip_hand_only():
    valid = _phase3_args(
        None,
        phase3_full=True,
        phase3_start_stage=5,
        phase3_end_stage=5,
        use_real_hand=True,
        policy_variant_flip="flip",
    )
    _validate_phase3_config(valid)
    valid.use_real_waist = True
    with pytest.raises(ValueError, match="Regular Mode"):
        _validate_phase3_config(valid)


def test_phase3_full_stage_zero_rejects_models_and_extra_ownership():
    valid = _phase3_args(
        None,
        phase3_full=True,
        phase3_start_stage=0,
        phase3_end_stage=0,
    )
    _validate_phase3_config(valid)
    valid.policy_variant_flip = "unused"
    with pytest.raises(ValueError, match="unused policy variants"):
        _validate_phase3_config(valid)


def test_phase3_full_stage_one_does_not_require_rotate_table():
    valid = _phase3_args(
        None,
        phase3_full=True,
        phase3_start_stage=1,
        phase3_end_stage=1,
        use_real_waist=True,
        use_real_hand=True,
        policy_variant_pick="pick",
        policy_variant_insert="insert",
        policy_variant_rotate_leg="rotate-leg",
    )
    _validate_phase3_config(valid)


def test_phase3_leg_stage_requires_exact_live_scope():
    valid = _phase3_args(
        1,
        use_real_waist=True,
        use_real_hand=True,
        policy_variant_pick="pick",
        policy_variant_insert="insert",
        policy_variant_rotate_leg="rotate-leg",
    )
    _validate_phase3_config(valid)
    valid.policy_variant_insert = None
    with pytest.raises(ValueError, match="missing policy variants: insert"):
        _validate_phase3_config(valid)


@pytest.mark.parametrize(
    "pick_variant",
    [PRODUCTION_HYBRID_PICK_VARIANT, "pick_table_leg_ramen_ori_venue_all6_20k"],
)
def test_phase3_leg_stage_accepts_validated_hybrid_pick(pick_variant):
    """hybrid の区間 1 は既定の GR00T か、会場の腕の RAMEN-Ori (Issue #188) のどちらでもよい。"""
    valid = _phase3_args(
        2,
        pick_leg_hybrid=True,
        use_real_waist=True,
        use_real_hand=True,
        policy_variant_rotate_table_base="rotate-table",
        policy_variant_pick=pick_variant,
        policy_variant_insert="insert",
        policy_variant_rotate_leg="rotate-leg",
    )
    _validate_phase3_config(valid)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"stage": 0}, "selected Stage in 1..4"),
        ({"use_real_hand": False}, "requires --use-real-hand"),
        ({"policy_variant_pick": None}, "requires --policy-variant-pick"),
        ({"no_wrist_cameras": True}, "requires both wrist"),
        ({"pick_leg_phase3_executor": "vla"}, "requires.*rule_based"),
    ],
)
def test_phase3_hybrid_pick_rejects_unsafe_scope(override, match):
    values = {
        "stage": 2,
        "pick_leg_hybrid": True,
        "use_real_waist": True,
        "use_real_hand": True,
        "policy_variant_rotate_table_base": "rotate-table",
        "policy_variant_pick": PRODUCTION_HYBRID_PICK_VARIANT,
        "policy_variant_insert": "insert",
        "policy_variant_rotate_leg": "rotate-leg",
    }
    values.update(override)
    with pytest.raises(ValueError, match=match):
        _validate_phase3_config(_phase3_args(**values))


@pytest.mark.parametrize("pick_variant", ["groot_pick_legs_v2", "pick_table_leg_ramen_ori_all6_400k"])
def test_phase3_hybrid_pick_rejects_a_different_checkpoint_variant(pick_variant):
    with pytest.raises(ValueError, match="HYBRID_PHASE1_VARIANTS"):
        _validate_phase3_config(
            _phase3_args(
                2,
                pick_leg_hybrid=True,
                use_real_waist=True,
                use_real_hand=True,
                policy_variant_rotate_table_base="rotate-table",
                policy_variant_pick=pick_variant,
                policy_variant_insert="insert",
                policy_variant_rotate_leg="rotate-leg",
            )
        )


def test_phase3_flip_forbids_waist_ownership():
    valid = _phase3_args(
        5,
        use_real_hand=True,
        policy_variant_flip="flip",
    )
    _validate_phase3_config(valid)
    valid.use_real_waist = True
    with pytest.raises(ValueError, match="waist/legs under Regular Mode"):
        _validate_phase3_config(valid)


def test_entrypoint_preserves_stereo_for_policy_and_uses_left_for_detector():
    assert HEAD_CAMERA_STEREO_VIEW == "packed"
    assert HEAD_PERCEPTION_VIEW == "left"


def test_default_log_path_uses_timestamped_filename_under_default_dir():
    """--log 未指定時の default path。

    outputs/orch_logs/orch_<yyyy-mm-ddTHH-MM-SS>.jsonl 形式で 1 run 1 file、
    resume 経路が「最新 log 自動検出」で拾える stem 順序性を持つ。
    """
    path = _default_log_path()

    assert path.parent == _DEFAULT_LOG_DIR
    assert path.suffix == ".jsonl"
    # stem = "orch_<yyyy-mm-ddTHH-MM-SS>"
    assert path.stem.startswith("orch_")
    timestamp_part = path.stem[len("orch_") :]
    # ISO 8601 T 区切り: 4-2-2T2-2-2 = 19 chars
    assert len(timestamp_part) == 19
    # 桁数チェック (数値 group)
    date_part, time_part = timestamp_part.split("T")
    y, m, d = date_part.split("-")
    hh, mm, ss = time_part.split("-")
    assert len(y) == 4 and len(m) == 2 and len(d) == 2
    assert len(hh) == 2 and len(mm) == 2 and len(ss) == 2
    # 数値化できる (int() で例外出ない)
    int(y), int(m), int(d), int(hh), int(mm), int(ss)


def _rule_based_pick_skills_section() -> dict:
    import yaml

    path = (
        Path(__file__).resolve().parents[1]
        / "lower_policy"
        / "configs"
        / "skill_config.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))["skills"]


def test_build_rule_based_pick_uses_shipped_config_and_shared_kinematics():
    from inference.desktop.entrypoint import build_pick_table_leg
    from inference.desktop.lower_policy.actuators.dex1 import MockDex1Gripper

    skill = build_pick_table_leg(_rule_based_pick_skills_section(), MockDex1Gripper())

    assert skill.name == "pick_table_leg"
    assert [stage.name for stage in skill.stages] == [
        "pregrasp",
        "approach_grasp",
        "close_grasp",
        "to_presentation",
        "approach_handover",
        "close_handover",
        "release",
        "retract",
    ]
    assert skill._grasp_provider.kinematics is skill._kin  # noqa: SLF001


def test_build_rule_based_pick_rejects_missing_or_malformed_config():
    from inference.desktop.entrypoint import build_pick_table_leg
    from inference.desktop.lower_policy.actuators.dex1 import MockDex1Gripper

    gripper = MockDex1Gripper()
    with pytest.raises(ValueError, match="missing"):
        build_pick_table_leg({}, gripper)
    with pytest.raises(ValueError, match="must be a mapping"):
        build_pick_table_leg({"pick_table_leg": []}, gripper)


# =========================================================================
# 起動時の変数の束縛 (Issue #141 の review 指摘 1〜3)
# =========================================================================


def _main_ast() -> ast.FunctionDef:
    tree = ast.parse(inspect.getsource(entrypoint_module))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


class _TopLevelNames(ast.NodeVisitor):
    """main の直下で実行される code だけを見る (入れ子の関数は定義時には走らない)。"""

    def __init__(self) -> None:
        self.first_store: dict[str, int] = {}
        self.first_load: dict[str, int] = {}
        # 内包表記の変数は独自スコープで main のローカルではない
        self.comprehension_locals: set[str] = set()

    def visit_FunctionDef(self, node):  # noqa: N802 - ast の約束
        self.first_store.setdefault(node.name, node.lineno)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):  # noqa: N802
        return

    def _visit_comprehension(self, node):
        for generator in node.generators:
            for target in ast.walk(generator.target):
                if isinstance(target, ast.Name):
                    self.comprehension_locals.add(target.id)
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        children = (
            [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        )
        for child in children:
            self.visit(child)

    visit_ListComp = _visit_comprehension  # noqa: N815 - ast の約束
    visit_SetComp = _visit_comprehension  # noqa: N815
    visit_DictComp = _visit_comprehension  # noqa: N815
    visit_GeneratorExp = _visit_comprehension  # noqa: N815

    def visit_Name(self, node):  # noqa: N802
        seen = self.first_store if isinstance(node.ctx, ast.Store) else self.first_load
        seen.setdefault(node.id, node.lineno)
        self.generic_visit(node)


def test_main_binds_every_local_before_it_reads_it():
    """main が自分の変数を「代入より前の行」で読んでいないこと。

    Python は関数内で後から代入する名前を先に読むと UnboundLocalError になる。
    main は 1,000 行あって起動を通しで走らせる test が無いので、静的に見る。
    """
    scan = _TopLevelNames()
    for statement in _main_ast().body:
        scan.visit(statement)
    used_before_binding = sorted(
        (name, scan.first_load[name], scan.first_store[name])
        for name in scan.first_store.keys() - scan.comprehension_locals
        if name in scan.first_load and scan.first_load[name] < scan.first_store[name]
    )
    assert used_before_binding == []


def test_main_initializes_every_name_its_finally_reads_before_the_try():
    """finally が読む名前は try の前で束縛しておくこと。

    センサや記録の構築で落ちたときに finally が UnboundLocalError に化けると、
    元の例外が消えるうえ `_safe_shutdown` (腕の解放と log の後始末) も呼ばれない。
    """
    main_fn = _main_ast()
    tries = [
        node for node in main_fn.body if isinstance(node, ast.Try) and node.finalbody
    ]
    assert len(tries) == 1, "main の直下の try は 1 つの前提"
    run_try = tries[0]

    bound_before_try: set[str] = set()
    for statement in main_fn.body:
        if statement is run_try:
            break
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound_before_try.add(node.id)

    assigned_in_main = {
        node.id
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    read_in_finally = {
        node.id
        for statement in run_try.finalbody
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert sorted((read_in_finally & assigned_in_main) - bound_before_try) == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (
            dict(phase3_full=True, phase3_start_stage=0, phase3_end_stage=5),
            (0, (0, 1, 2, 3, 4, 5)),
        ),
        (dict(phase3_full=True, phase3_start_stage=2, phase3_end_stage=3), (2, (2, 3))),
        (dict(stage=2), (2, (2,))),
        ({}, (None, ())),  # --rule-based-pick-table-leg のように stage を指定しない run
    ],
)
def test_resolve_stage_selection_covers_every_launch_mode(overrides, expected):
    args = SimpleNamespace(
        phase3_full=False, phase3_start_stage=0, phase3_end_stage=5, stage=None
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    assert entrypoint_module.resolve_stage_selection(args) == expected


@pytest.mark.parametrize("start,end", [(0, None), (None, 5), (2, 4)])
def test_phase3_range_requires_full_mode(start, end):
    args = SimpleNamespace(
        phase3_full=False,
        phase3_start_stage=start,
        phase3_end_stage=end,
    )
    with pytest.raises(ValueError, match="require --phase3-full"):
        entrypoint_module.normalize_phase3_range_args(args)


def test_phase3_full_range_defaults_are_normalized():
    args = SimpleNamespace(
        phase3_full=True,
        phase3_start_stage=None,
        phase3_end_stage=None,
    )
    entrypoint_module.normalize_phase3_range_args(args)
    assert (args.phase3_start_stage, args.phase3_end_stage) == (0, 5)


def test_omitted_non_full_range_keeps_legacy_or_stage_launch_compatible():
    args = SimpleNamespace(
        phase3_full=False,
        phase3_start_stage=None,
        phase3_end_stage=None,
    )
    entrypoint_module.normalize_phase3_range_args(args)
    assert (args.phase3_start_stage, args.phase3_end_stage) == (0, 5)


# ---- --synthetic-hand-state (会場のリグは Dex1 state を配信しない) ----
def _stage1_variants(**overrides):
    """Stage 1 は 1 本目なので卓を回さない (`STAGE_SKILL_SEQUENCES[1]`)。

    `rotate_table_base` を渡すと「使わない policy を GPU に載せている」として
    弾かれる (Issue #148)。Thor は unified memory なので、この無駄は他の
    expert の常駐枠を直接削る。
    """
    values = {
        "policy_variant_pick": "v",
        "policy_variant_insert": "v",
        "policy_variant_rotate_leg": "v",
    }
    values.update(overrides)
    return values


def _full_run_variants(**overrides):
    """--phase3-full (既定は stage 0..5) は脚 3 つ + 卓回し + flip を要求する。"""
    return _stage1_variants(
        policy_variant_rotate_table_base="v",
        policy_variant_flip="v",
        **overrides,
    )


def test_stage1_still_requires_real_hand_by_default():
    with pytest.raises(ValueError, match="requires --use-real-hand"):
        _validate_phase3_config(
            _phase3_args(1, use_real_waist=True, **_stage1_variants())
        )


def test_stage1_accepts_synthetic_hand_state_instead_of_a_real_hand():
    """stateを配信しないリグでもstageを走らせる（SDK laneではmock固定）。"""
    _validate_phase3_config(
        _phase3_args(
            1,
            use_real_waist=True,
            synthetic_hand_state=True,
            **_stage1_variants(),
        )
    )


def test_synthetic_hand_state_does_not_excuse_a_missing_real_waist():
    with pytest.raises(ValueError, match="requires --use-real-waist"):
        _validate_phase3_config(
            _phase3_args(1, synthetic_hand_state=True, **_stage1_variants())
        )


def test_synthetic_hand_state_conflicts_with_a_real_hand():
    """合成 state で実グリッパを駆動すると prime_hold が推測値から始まる。"""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_phase3_config(
            _phase3_args(
                1,
                use_real_waist=True,
                use_real_hand=True,
                synthetic_hand_state=True,
                **_stage1_variants(),
            )
        )


# ---- --action-sink boundary (会場でグリッパを動かせる唯一の経路) ----
def test_boundary_sink_rejects_a_real_hand_actuator():
    """グリッパはboundaryのhand列で動かす。SDK直と併用すると二重指令になる。"""
    with pytest.raises(ValueError, match="boundary and --use-real-hand"):
        _validate_phase3_config(
            _phase3_args(
                1,
                action_sink="boundary",
                use_real_hand=True,
                use_real_waist=True,
                **_stage1_variants(),
            )
        )


def test_boundary_sink_rejects_a_real_waist_actuator():
    with pytest.raises(ValueError, match="boundary and --use-real-waist"):
        _validate_phase3_config(
            _phase3_args(
                1, action_sink="boundary", use_real_waist=True, **_stage1_variants()
            )
        )


def test_sdk_sink_is_unaffected_by_the_new_check():
    """既定 (sdk) では従来どおり実 waist/hand を要求する。"""
    _validate_phase3_config(
        _phase3_args(1, use_real_waist=True, use_real_hand=True, **_stage1_variants())
    )


def test_boundary_sink_starts_stage1_without_any_sdk_actuator():
    """**会場でグリッパを動かせる唯一の経路**。ここが通らないと stage 1-4 が丸ごと 0 点。

    腕・腰・手はまとめてboundaryへ渡すのでSDK直の実actuatorは要らない
    (併用はむしろ上の 2 test のとおり禁止)。腰は指令が無ければ実測値で埋まる。
    """
    _validate_phase3_config(
        _phase3_args(
            1,
            action_sink="boundary",
            synthetic_hand_state=True,
            **_stage1_variants(),
        )
    )


def test_boundary_sink_starts_a_full_run_without_any_sdk_actuator():
    _validate_phase3_config(
        _phase3_args(
            None,
            phase3_full=True,
            action_sink="boundary",
            synthetic_hand_state=True,
            **_full_run_variants(),
        )
    )


def test_boundary_sink_requires_synthetic_hand_state():
    """付け忘れると**グリッパ指令が黙って落ちる**ので、起動時に弾く。

    `_build_hand_actuator` が mock を共有するのは `--synthetic-hand-state` のときだけ。
    共有されないと skill ごとに別 instance になり、skill が書いた `.latest` が
    boundary sinkに届かない = hand列がframe-0の開度のまま動かない。
    エラーは出ないので、現地では「掴まない」としか見えない。
    """
    with pytest.raises(ValueError, match="requires --synthetic-hand-state"):
        _validate_phase3_config(
            _phase3_args(1, action_sink="boundary", **_stage1_variants())
        )


# =========================================================================
# stage 5 (flip) を boundary で起動できること
# =========================================================================
def _stage5_boundary_args(**overrides):
    values = {
        "policy_variant_flip": "groot_flip_table_n17_2_baseline",
        "action_sink": "boundary",
        "synthetic_hand_state": True,
    }
    values.update(overrides)
    return _phase3_args(5, **values)


def test_phase3_stage5_accepts_the_boundary_sink():
    """`--stage 5 --action-sink boundary --synthetic-hand-state` が通ること。

    boundary は `--use-real-hand` を禁止する一方で stage 5 はそれを要求していたため、
    この組み合わせが成立せず **会場で flip だけをやり直す手段が無かった**
    (`--phase3-full` 経由は `_require_real_waist_and_hand` の boundary 例外を通るので
    動いていた = 単独 stage だけが落ちていた)。実 image を載せた pod で
    `Phase 3 stage 5 requires --use-real-hand` を実測 (2026-09-21)。
    """
    _validate_phase3_config(_stage5_boundary_args())


def test_phase3_stage5_boundary_requires_synthetic_hand_state():
    """`--synthetic-hand-state` が無いとグリッパ指令が黙って落ちるので弾く。"""
    with pytest.raises(ValueError, match="synthetic-hand-state"):
        _validate_phase3_config(_stage5_boundary_args(synthetic_hand_state=False))


def test_phase3_stage5_still_requires_a_real_hand_on_the_sdk_sink():
    """SDK 経路の要求は変えない (グリッパは実 actuator で動かす)。"""
    with pytest.raises(ValueError, match="use-real-hand"):
        _validate_phase3_config(
            _stage5_boundary_args(action_sink="sdk", synthetic_hand_state=False)
        )


def test_a_variant_set_replaces_only_the_skills_it_names():
    """`--policy-variant-set all6_400k` で stage 1 の insert だけ替わり、set に無い pick は既定のまま (Issue #159)。"""
    args = _phase3_args(1, use_real_waist=True, use_real_hand=True, policy_variant_set="all6_400k")
    source = _fill(args)
    assert args.policy_variant_insert == "insert_table_leg_ramen_ori_all6_400k"
    assert source["insert"] == "set:all6_400k"
    assert args.policy_variant_pick == "pick_table_leg_ramen_ori_venue_all6_20k"
    assert source["pick"] == "config"
    _validate_phase3_config(args)


@pytest.mark.parametrize("stage", [1, 2, 3, 4, 5])
def test_the_all6_set_switches_every_slot_but_pick(stage):
    """会場の候補 `all6_400k` (Issue #161): pick 以外は 6 skill 統合 RAMEN-Ori、pick は既定のまま。

    既定 (default_variant_by_skill) は変えずに、起動の引数だけで切り替えられること。
    main() と同じ順 (hybrid の判定 → 既定で埋める → 検査) で、会場の起動口の引数で通す。
    """

    def venue_args(**overrides):
        args = _phase3_args(
            stage,
            pick_leg_hybrid=None,
            action_sink="boundary",
            synthetic_hand_state=True,
            **overrides,
        )
        resolve_production_pick_mode(args)
        return args

    defaults = _fill(venue_args())
    args = venue_args(policy_variant_set="all6_400k")
    source = _fill(args)
    assert set(source) == set(defaults)
    for slot in source:
        variant = getattr(args, _POLICY_SLOT_ATTRS[slot])
        if slot == "pick":
            assert variant == "pick_table_leg_ramen_ori_venue_all6_20k"  # set は pick を変えない
            assert source[slot] == "config"
        else:
            assert variant.endswith("_ramen_ori_all6_400k"), (slot, variant)
            assert source[slot] == "set:all6_400k"
    _validate_phase3_config(args)


def test_a_slot_flag_still_wins_over_the_variant_set():
    args = _phase3_args(
        1,
        use_real_waist=True,
        use_real_hand=True,
        policy_variant_set="all6_400k",
        policy_variant_insert="groot_insert_leg_200k",
    )
    source = _fill(args)
    assert args.policy_variant_insert == "groot_insert_leg_200k"
    assert source["insert"] == "cli"


class TestReturnBeforeRelease:
    """解放の直前に「手を開いて腕を下ろす」(Issue #152)。

    正常終了・Ctrl+C・安全停止のどれでも同じ処理を通す。腕を上げたまま arm_sdk を
    手放すと、次の run が同じ状態から始められない (stage 0 は腕が下がっていることを
    求める) ので、run の終わりを毎回同じ姿勢に揃える。
    """

    class _Arm:
        def __init__(self, initial=None) -> None:
            self.commands: list[np.ndarray] = []
            self.initial = np.zeros(14) if initial is None else np.asarray(initial).copy()

        def send_action(self, target) -> None:
            self.commands.append(np.asarray(target, dtype=np.float64))

        def read_last_published_targets(self):
            return (self.commands[-1].copy() if self.commands else self.initial.copy(), None)

    class _JointSource:
        """指令をそのまま追従する腕 (実測 = 直前の指令)。"""

        def __init__(self, arm, start: np.ndarray) -> None:
            self._arm = arm
            self._measured = start.copy()

        def get(self):
            if self._arm.commands:
                target = self._arm.commands[-1]
                self._measured += np.clip(target - self._measured, -0.05, 0.05)
            position = np.zeros(29)
            position[15:29] = self._measured
            return SimpleNamespace(position=position, velocity=np.zeros(29))

    class _Hand:
        def __init__(self) -> None:
            self.targets: list[list[float]] = []

        def send_action(self, target) -> None:
            self.targets.append(list(target))

    def _skill_config(self) -> dict:
        return {
            "skills": {"walk_lowered_pose": {"pose_rad": {}}},
            "arm_pre_motion": {
                "velocity_limit_rad_s": 50.0,
                "acceleration_limit_rad_s2": 500.0,
                "measured_tolerance_rad": 0.1,
                "measured_velocity_tolerance_rad_s": 0.05,
                "stage_timeout_s": 15.0,
                "boundary_lift_rad": 0.15,
            },
        }

    @pytest.mark.parametrize("opening", [5.4, 5.3])
    def test_opens_the_hand_and_lowers_the_arms(self, opening) -> None:
        raised = np.zeros(14)
        raised[0] = -0.5
        raised[1] = 1.0
        arm = self._Arm()
        hand = self._Hand()
        config = self._skill_config()
        config["hand_pre_motion"] = {"open_rad": opening}
        entrypoint_module._return_before_release(
            skill_config=config,
            arm_actuator=arm,
            joint_state_source=self._JointSource(arm, raised),
            hand_actuator=hand,
            dex1_state_source=SimpleNamespace(
                get=lambda: SimpleNamespace(position_rad=np.array([0.0, 0.0]))
            ),
        )
        assert hand.targets, "手を開く指令が出ていない"
        assert hand.targets[-1] == pytest.approx([opening, opening])
        # 最後の指令は下がった姿勢 (全関節 0)
        np.testing.assert_allclose(arm.commands[-1], np.zeros(14), atol=1e-2)

    def test_missing_lowered_pose_skips_the_return(self) -> None:
        arm = self._Arm()
        entrypoint_module._return_before_release(
            skill_config={"skills": {}},
            arm_actuator=arm,
            joint_state_source=self._JointSource(arm, np.zeros(14)),
        )
        assert arm.commands == []

    def test_ctrl_c_restores_current_policy_initial_before_opening_hand(self) -> None:
        raised = np.zeros(14)
        raised[0] = -0.4
        initial = np.zeros(14)
        initial[0] = 0.2
        arm = self._Arm(raised)
        hand = self._Hand()
        hand.arm_count_at_first_open = None
        original_send = hand.send_action

        def record_open(target):
            if hand.arm_count_at_first_open is None:
                hand.arm_count_at_first_open = len(arm.commands)
            original_send(target)

        hand.send_action = record_open
        entrypoint_module._return_before_release(
            skill_config=self._skill_config(),
            arm_actuator=arm,
            joint_state_source=self._JointSource(arm, raised),
            hand_actuator=hand,
            dex1_state_source=SimpleNamespace(
                get=lambda: SimpleNamespace(position_rad=np.array([0.0, 0.0]))
            ),
            policy_initial_arm=initial,
        )
        assert hand.arm_count_at_first_open is not None
        assert hand.arm_count_at_first_open > 0
        np.testing.assert_allclose(
            arm.commands[hand.arm_count_at_first_open - 1], initial, atol=0.03
        )
        np.testing.assert_allclose(arm.commands[-1], np.zeros(14), atol=0.03)

    def test_startup_abort_opens_hand_without_arm_return_motion(self) -> None:
        arm = self._Arm(np.full(14, 0.25))
        hand = self._Hand()
        entrypoint_module._return_before_release(
            skill_config=self._skill_config(),
            arm_actuator=arm,
            joint_state_source=self._JointSource(arm, np.full(14, 0.25)),
            hand_actuator=hand,
            dex1_state_source=SimpleNamespace(
                get=lambda: SimpleNamespace(position_rad=np.array([1.0, 1.0]))
            ),
            open_only=True,
        )
        assert hand.targets[-1] == pytest.approx([5.4, 5.4])
        assert arm.commands  # boundary atomically resends the held arm row
        for command in arm.commands:
            np.testing.assert_allclose(command, np.full(14, 0.25))

    def test_partial_startup_abort_reverses_only_completed_route(self) -> None:
        partial = np.zeros(14)
        partial[0] = 0.2
        arm = self._Arm(partial)
        entrypoint_module._return_before_release(
            skill_config=self._skill_config(),
            arm_actuator=arm,
            joint_state_source=self._JointSource(arm, partial),
            retreat_waypoints=(np.zeros(14),),
        )
        np.testing.assert_allclose(arm.commands[-1], np.zeros(14), atol=0.03)
        assert max(abs(command[1]) for command in arm.commands) < 0.1

    def test_a_second_ctrl_c_stops_the_return(self) -> None:
        arm = self._Arm()

        class _InterruptingSource:
            def get(self):
                raise KeyboardInterrupt

        # 例外を外に出さない (解放は必ず続ける)
        entrypoint_module._return_before_release(
            skill_config=self._skill_config(),
            arm_actuator=arm,
            joint_state_source=_InterruptingSource(),
        )


def test_base_is_stopped_before_the_arms_are_returned():
    """boundary の row は navigate_cmd を毎回運ぶ。return の前に速度 0 を送る。"""
    walk = MagicMock()
    entrypoint_module._stop_base_before_return(walk)
    walk.set_velocity.assert_called_once_with(0.0, 0.0, 0.0, duration=1.0)


def test_base_stop_failure_does_not_abort_the_return():
    walk = MagicMock()
    walk.set_velocity.side_effect = RuntimeError("rpc down")
    entrypoint_module._stop_base_before_return(walk)  # must not raise


def test_finally_stops_the_base_before_returning_the_arms():
    """main の finally で return より前に walk を止めていること (順序の回帰防止)。"""
    source = inspect.getsource(entrypoint_module.main)
    stop = source.index("_stop_base_before_return(actuator)")
    ret = source.index("_return_before_release(")
    assert stop < ret


def test_boundary_lane_requires_synthetic_hand_state_in_main():
    source = inspect.getsource(entrypoint_module.main)
    assert 'args.action_sink == "boundary" and not args.synthetic_hand_state' in source


@pytest.mark.parametrize(
    "action_sink, use_real_hand, synthetic, stages, expected",
    [
        # Codex #3: Stage 0 開始の通しでも後続 Stage の手順に手が入る
        ("boundary", False, True, (0, 1, 2, 3, 4, 5), True),
        ("boundary", False, True, (2,), True),
        ("boundary", False, True, (0,), False),
        ("boundary", False, False, (1,), False),
        ("sdk", True, False, (0, 1), True),
        ("sdk", False, True, (1,), False),
    ],
)
def test_hand_scaffold_follows_the_hand_command_path(
    action_sink, use_real_hand, synthetic, stages, expected
):
    args = argparse.Namespace(
        action_sink=action_sink,
        use_real_hand=use_real_hand,
        synthetic_hand_state=synthetic,
    )
    assert entrypoint_module.resolve_include_hand_in_head(args, stages) is expected


def test_boundary_waist_is_always_the_measured_angle():
    """Codex #4: skill 側と別の mock を読む死んだ配線を残さない。腰は実測値。"""
    source = inspect.getsource(entrypoint_module.main)
    assert "_boundary_waist" not in source
    assert "BOUNDARY_WAIST_NOTE" in source


def _endpoint_args(**overrides):
    values = dict(
        action_sink="boundary",
        head_source="ros2",
        joint_source="lowstate",
        zmq_endpoint=None,
        boundary_state_host=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_explicit_camera_endpoint_beats_a_leftover_environment_host(monkeypatch):
    """Codex #10: --zmq-endpoint を明示したら state もその host に繋ぐ。"""
    monkeypatch.setenv("G1_IMAGE_SERVER_IP", "192.0.2.10")
    monkeypatch.delenv("IROS_ORIN_HOST", raising=False)
    args = _endpoint_args(zmq_endpoint="tcp://198.51.100.7:5555")
    configure_official_endpoints(args)
    assert args.boundary_state_host == "198.51.100.7"


def test_camera_and_state_hosts_must_agree(monkeypatch):
    monkeypatch.delenv("IROS_ORIN_HOST", raising=False)
    monkeypatch.delenv("G1_IMAGE_SERVER_IP", raising=False)
    args = _endpoint_args(
        zmq_endpoint="tcp://198.51.100.7:5555", boundary_state_host="192.0.2.10"
    )
    with pytest.raises(ValueError, match="differ"):
        configure_official_endpoints(args)


def test_conflicting_environment_hosts_are_rejected(monkeypatch):
    monkeypatch.setenv("IROS_ORIN_HOST", "192.0.2.1")
    monkeypatch.setenv("G1_IMAGE_SERVER_IP", "192.0.2.2")
    with pytest.raises(ValueError, match="disagree"):
        configure_official_endpoints(_endpoint_args())


@pytest.mark.parametrize(
    "overrides",
    [
        {"zmq_endpoint": "tcp://192.0.2.9:5555"},
        {"boundary_state_host": "192.0.2.9"},
        {
            "zmq_endpoint": "tcp://192.0.2.9:5555",
            "boundary_state_host": "192.0.2.9",
        },
    ],
)
def test_explicit_orin_host_overrides_conflicting_environment(monkeypatch, overrides):
    monkeypatch.setenv("IROS_ORIN_HOST", "192.0.2.1")
    monkeypatch.setenv("G1_IMAGE_SERVER_IP", "192.0.2.2")
    args = _endpoint_args(**overrides)
    configure_official_endpoints(args)
    assert args.zmq_endpoint == "tcp://192.0.2.9:5555"
    assert args.boundary_state_host == "192.0.2.9"


def test_explicit_camera_and_state_hosts_still_must_agree_with_conflicting_env(monkeypatch):
    monkeypatch.setenv("IROS_ORIN_HOST", "192.0.2.1")
    monkeypatch.setenv("G1_IMAGE_SERVER_IP", "192.0.2.2")
    args = _endpoint_args(
        zmq_endpoint="tcp://192.0.2.9:5555", boundary_state_host="192.0.2.10"
    )
    with pytest.raises(ValueError, match="differ"):
        configure_official_endpoints(args)


def test_explicit_state_host_builds_the_camera_endpoint(monkeypatch):
    monkeypatch.setenv("IROS_ORIN_HOST", "192.0.2.1")
    monkeypatch.delenv("G1_IMAGE_SERVER_IP", raising=False)
    args = _endpoint_args(boundary_state_host="192.0.2.9")
    configure_official_endpoints(args)
    assert args.zmq_endpoint == "tcp://192.0.2.9:5555"
    assert args.boundary_state_host == "192.0.2.9"


def test_bounded_cleanup_does_not_wait_forever_or_raise():
    """Codex #7: 後始末が止まっても・例外を出しても先へ進む。"""
    import threading
    import time as _time

    never = threading.Event()
    started = _time.monotonic()
    assert not entrypoint_module._bounded_cleanup(
        "hang", lambda: never.wait(10.0), timeout_s=0.05
    )
    assert _time.monotonic() - started < 2.0

    def boom():
        raise RuntimeError("recorder broke")

    assert not entrypoint_module._bounded_cleanup("boom", boom, timeout_s=1.0)
    assert entrypoint_module._bounded_cleanup("ok", lambda: None, timeout_s=1.0)


def test_robot_release_runs_before_model_and_recorder_cleanup():
    """解放は model 解放・録画の終了より先、独立した finally で必ず通る。"""
    source = inspect.getsource(entrypoint_module.main)
    release = source.index("_release_robot_outputs(")
    assert release < source.index("single_stage_residency.close")
    assert release < source.index("recorder.__exit__")
    assert "single_stage_residency.close()" not in source  # 無期限の直呼びを残さない


def test_release_robot_outputs_never_raises():
    walk = MagicMock()
    walk.set_velocity.side_effect = RuntimeError("rpc")
    arm = MagicMock()
    arm.controlled_release.side_effect = RuntimeError("release")
    arm.stop.side_effect = RuntimeError("stop")
    hand = MagicMock()
    entrypoint_module._release_robot_outputs(
        walk, arm_actuator=arm, arm_actuator_started=True, hand_actuator=hand
    )
    hand.stop.assert_called_once()


# ---------------------------------------------------------------- boundary lane (2026-09-25)
#
# 運営が :5556 に joint lane ((T,22) の腕の関節角、運営 IK を通らない) を足した。
# joint-space policy は既定で joint lane。pose は明示した互換経路だけで使う。


def test_the_boundary_lane_defaults_to_joint(monkeypatch):
    monkeypatch.setattr("sys.argv", ["entrypoint", "--action-sink", "boundary"])
    assert entrypoint_module.parse_args().boundary_lane == "joint"
    monkeypatch.setattr(
        "sys.argv", ["entrypoint", "--action-sink", "boundary", "--boundary-lane", "joint"]
    )
    assert entrypoint_module.parse_args().boundary_lane == "joint"


def test_the_boundary_lane_setting_is_harmless_on_the_sdk_path():
    _validate_phase3_config(_phase3_args(None, boundary_lane="joint"))


def test_arrival_is_compared_in_task_space_only_on_the_pose_lane():
    """pose lane は運営 IK が関節角を選ぶので手先で、joint lane と sdk は関節角で比べる。"""
    arrival = entrypoint_module._boundary_taskspace_arrival
    assert arrival(argparse.Namespace(action_sink="boundary", boundary_lane="pose"))
    assert not arrival(argparse.Namespace(action_sink="boundary", boundary_lane="joint"))
    assert not arrival(argparse.Namespace(action_sink="sdk", boundary_lane="pose"))


@pytest.mark.parametrize("stage", [0, 1, 2, 3, 4, 5])
def test_every_stage_accepts_the_joint_lane_with_the_venue_arguments(stage):
    args = _phase3_args(
        stage,
        pick_leg_hybrid=None,
        action_sink="boundary",
        synthetic_hand_state=True,
        boundary_lane="joint",
    )
    resolve_production_pick_mode(args)
    _fill(args)
    _validate_phase3_config(args)


def test_main_switches_every_task_space_check_with_the_lane():
    """main() の中の配線 (closure なので source で見る): sink に lane を渡し、準備動作の判定と
    hybrid pick の区間 3 の到達判定が同じ helper で pose / joint を切り替える。"""
    import inspect

    source = inspect.getsource(entrypoint_module.main)
    assert "lane=args.boundary_lane" in source
    assert "if not _boundary_taskspace_arrival(args):" in source
    assert '"boundary_taskspace_arrival": _boundary_taskspace_arrival(args)' in source
    assert '"boundary_taskspace_arrival": args.action_sink == "boundary"' not in source


# ------------------------------------------------ policy 開始の gate の到達判定 (2026-09-25)


def test_the_gate_compares_joint_angles_without_a_task_space_checker():
    check = entrypoint_module._gate_arrival_check(np.zeros(14), None, 0.10)
    measured = np.zeros(14)
    measured[10] = 0.5
    reached, detail = check(measured)
    assert not reached and "R.elbow" in detail
    assert check(np.full(14, 0.05))[0]


def test_the_gate_uses_the_task_space_checker_on_the_pose_lane():
    calls = []

    def checker(goal, measured, velocity):
        calls.append((np.asarray(goal).copy(), np.asarray(measured).copy(), np.asarray(velocity)))
        return False, "ee_pos_error=0.2303m"

    target = np.linspace(-0.3, 0.3, 14)
    check = entrypoint_module._gate_arrival_check(target, checker, 0.10)
    assert check(np.zeros(14)) == (False, "ee_pos_error=0.2303m")
    np.testing.assert_allclose(calls[0][0], target)
    np.testing.assert_allclose(calls[0][2], 0.0)  # 保持中なので速さは見ない


def test_main_builds_the_gate_with_the_arrival_check():
    source = inspect.getsource(entrypoint_module.main)
    assert "arrival = _gate_arrival_check(" in source
    assert "arrival_check=arrival" in source
    assert "require_arrival=True" in source
    assert "_boundary_convergence_checker(policy)" in source


def test_manual_stage_graph_requires_enter_at_every_policy_boundary():
    from inference.desktop.orchestrator import build_stage_transitions

    graph = build_stage_transitions(2)
    initial_gate = entrypoint_module._install_operator_policy_gates(
        graph,
        ["rotate_table_base", "pick_table_leg", "insert_table_leg",
         "rotate_leg_to_tighten"],
    )
    assert initial_gate is None
    assert any(
        next_skills == ["operator_gate_for_rotate_table_base"]
        for next_skills in graph.values()
    )
    for skill in (
        "rotate_table_base", "pick_table_leg", "insert_table_leg",
        "rotate_leg_to_tighten",
    ):
        assert graph[f"operator_gate_for_{skill}"] == [skill]
    assert graph["hold_transition_pick_table_leg_to_insert_table_leg"] == [
        "operator_gate_for_insert_table_leg"
    ]


def test_continuous_stage_one_begins_with_pick_enter_gate():
    from inference.desktop.orchestrator import build_stage_transitions

    graph = build_stage_transitions(1, is_start_stage=False)
    gate = entrypoint_module._install_operator_policy_gates(
        graph, ["pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"]
    )
    assert gate == "operator_gate_for_pick_table_leg"
    assert graph[gate] == ["pick_table_leg"]


# ------------------------------------------------ 手首 roll の clamp の切り替え (接続テスト用)


def test_the_wrist_roll_clamp_defaults_to_off(monkeypatch):
    monkeypatch.setattr("sys.argv", ["entrypoint", "--action-sink", "boundary"])
    assert entrypoint_module.parse_args().wrist_roll_clamp == "off"


@pytest.mark.parametrize(
    "overrides",
    [
        {"action_sink": "boundary", "boundary_lane": "pose"},
        {"action_sink": "boundary", "boundary_lane": "joint"},
        {"action_sink": "sdk", "boundary_lane": "joint"},
    ],
)
def test_the_wrist_roll_clamp_is_harmless_outside_pose_lane(overrides):
    args = _phase3_args(
        5,
        synthetic_hand_state=True,
        wrist_roll_clamp="off",
        policy_variant_flip="groot_flip_table_n17_2_baseline",
        **overrides,
    )
    if overrides["action_sink"] == "sdk":
        args.synthetic_hand_state = False
        args.use_real_hand = True
    _validate_phase3_config(args)


def test_main_passes_the_wrist_roll_clamp_to_the_sink():
    source = inspect.getsource(entrypoint_module.main)
    assert 'clamp_wrist_roll=args.wrist_roll_clamp == "on"' in source



# ------------------------------------------------ Stage 0 の下ろした腕の確かめ方 (接続テスト用)


def test_the_walk_latch_check_comes_from_yaml_unless_overridden():
    import yaml

    config = yaml.safe_load(
        Path("inference/desktop/lower_policy/configs/skill_config.yaml").read_text()
    )
    assert entrypoint_module._walk_latch_check(config) == "joint"  # 既定は YAML
    assert entrypoint_module._walk_latch_check(config, "converged") == "converged"
    assert entrypoint_module._walk_latch_check({}) == "joint"


def test_the_walk_lowering_check_flag_defaults_to_the_yaml(monkeypatch):
    monkeypatch.setattr("sys.argv", ["entrypoint", "--stage", "0"])
    assert entrypoint_module.parse_args().walk_lowering_check is None
    monkeypatch.setattr(
        "sys.argv", ["entrypoint", "--stage", "0", "--walk-lowering-check", "converged"]
    )
    assert entrypoint_module.parse_args().walk_lowering_check == "converged"


def test_main_passes_the_walk_latch_check_to_stage_0():
    source = inspect.getsource(entrypoint_module.main)
    assert "latch_check=_walk_latch_check(skill_cfg_raw, args.walk_lowering_check)" in source


def test_every_hold_in_main_keeps_the_last_command_not_the_measurement():
    """開始待ち・手・保持の skill は、最後に送った腕 target を保持する (Issue #172)。

    実測を保持 target にすると、重力で遅れて止まる実測がさらに目標になり、段ごとに腕が
    下がる。模擬では遅れ 0.03 rad でも開始待ちで 0.12 rad になり、Enter が通らなかった。
    """
    main_fn = _main_ast()
    missing = []
    for node in ast.walk(main_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id if isinstance(func, ast.Name)
            else f"{getattr(func.value, 'id', '')}.{func.attr}"
            if isinstance(func, ast.Attribute) else ""
        )
        keywords = {kw.arg for kw in node.keywords}
        if name in (
            "OperatorConfirmationHoldSkill", "HoldPoseSkill", "HandPreMotionSkill.from_config"
        ) and "hold_arm_target_provider" not in keywords:
            missing.append((name, node.lineno))
        if name in (
            "assembly.build_head_procedure", "assembly.build_model_transition_procedure"
        ) and "published_arm_target_provider" not in keywords:
            missing.append((name, node.lineno))
    assert missing == []


def test_boundary_sender_publishes_the_stop_even_when_the_state_is_stale():
    """state が古くても「歩行ゼロ + 最後の腕 target の保持」は送る (Issue #172)。

    送らないと、運営 adapter の keepalive が最後の歩行指令 (vx 等) を送り続ける。
    新しい腕の目標は、state が古い間は従来どおり出さない。
    """
    now = {"ns": 10_000_000_000}
    fresh = SimpleNamespace(
        position=np.arange(29, dtype=np.float64) * 0.01,
        received_monotonic_ns=now["ns"],
    )
    joint = {"state": fresh}
    sent = []

    class Sink:
        def send_action(self, action19, body_q29, *, navigate_cmd, state_stale):
            sent.append((np.asarray(action19).copy(), np.asarray(body_q29).copy(),
                         np.asarray(navigate_cmd).copy(), state_stale))
            return True

    walk = SimpleNamespace(latest=np.array([0.3, 0.0, 0.0]))
    last = {"arm": None}
    sender = entrypoint_module.make_boundary_sender(
        joint_source=SimpleNamespace(get=lambda: joint["state"]),
        sink=Sink(),
        hand_actuator=SimpleNamespace(latest=[4.5, 4.5]),
        walk_actuator=walk,
        last_published_arm=lambda: last["arm"],
        monotonic_ns=lambda: now["ns"],
    )
    arm = np.full(14, 0.2)
    assert sender(arm)  # walking with a fresh state
    last["arm"] = arm.copy()
    assert sent[-1][3] is False

    now["ns"] += 600_000_000  # the :5557 state stops (> 0.5 s)
    assert not sender(np.full(14, 0.3))  # no new arm target on a stale state
    walk.latest = np.zeros(3)
    assert not sender(np.full(14, 0.3))  # still not a hold of the last command
    assert sender(arm)  # the stop: zero walk + last arm target
    action19, body_q29, navigate, stale = sent[-1]
    np.testing.assert_allclose(navigate, 0.0)
    np.testing.assert_allclose(action19[3:17], arm)
    np.testing.assert_allclose(body_q29, fresh.position)  # last good state
    assert stale is True


def test_boundary_sender_without_any_state_sends_nothing():
    sender = entrypoint_module.make_boundary_sender(
        joint_source=SimpleNamespace(get=lambda: None),
        sink=SimpleNamespace(send_action=lambda *a, **k: pytest.fail("published")),
        hand_actuator=SimpleNamespace(latest=None),
        walk_actuator=SimpleNamespace(latest=np.zeros(3)),
        last_published_arm=lambda: np.zeros(14),
    )
    assert not sender(np.zeros(14))


def test_pacing_rejection_does_not_update_the_arm_hold_record():
    import time

    from inference.desktop.lower_policy.actuators.boundary_sink import BoundaryArmActuator

    arm = BoundaryArmActuator()
    state = SimpleNamespace(position=np.zeros(29), received_monotonic_ns=time.monotonic_ns())
    sent = iter((True, False))
    sender = entrypoint_module.make_boundary_sender(
        joint_source=SimpleNamespace(get=lambda: state),
        sink=SimpleNamespace(send_action=lambda *a, **kw: next(sent)),
        hand_actuator=SimpleNamespace(latest=[4.5, 4.5]),
        walk_actuator=SimpleNamespace(latest=np.zeros(3)),
        last_published_arm=lambda: arm.read_last_published_targets()[0],
    )
    arm.configure(sender=sender, state_getter=lambda: np.zeros(14))
    arm.start()
    arm.send_action(np.full(14, 0.1))
    arm.send_action(np.full(14, 0.4))
    np.testing.assert_allclose(arm.read_last_published_targets()[0], 0.1)


class _DecisionConsole:
    def __init__(self, outcome, running=True):
        self.outcome = outcome
        self.running = running
        self.waits = []

    def wait_for(self, key, *, stage, phase):
        self.waits.append((key, stage, phase))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return key


@pytest.mark.parametrize(
    "outcome, boundary, expected, decision",
    [
        ("enter", True, True, "return"),
        (KeyboardInterrupt(), True, False, "hold_in_place"),
        (KeyboardInterrupt(), False, False, "hold_in_place"),
        (EOFError(), True, False, "hold_in_place_terminal_closed"),
        (EOFError(), False, True, "return_terminal_closed"),
    ],
)
def test_safety_stop_waits_for_the_operator_before_any_return_motion(
    outcome, boundary, expected, decision
):
    """安全停止の後は、操作者が Enter を押すまで腕も手も動かさない (Issue #172)。"""
    events = []
    console = _DecisionConsole(outcome)
    assert entrypoint_module._await_return_decision(
        console, stage=2, reason="camera head is stale", boundary=boundary,
        event_fn=events.append,
    ) is expected
    assert console.waits and console.waits[0][0] == "enter"
    assert console.waits[0][2].startswith("安全停止")
    assert [e["decision"] for e in events] == [decision]
    assert events[0]["reason"] == "camera head is stale"


@pytest.mark.parametrize("boundary, expected", [(True, False), (False, True)])
def test_safety_stop_without_a_terminal_never_moves_on_the_boundary(boundary, expected):
    console = _DecisionConsole("enter", running=False)
    assert entrypoint_module._await_return_decision(
        console, stage=2, reason="joint state is stale", boundary=boundary
    ) is expected
    assert console.waits == []


def test_main_gates_the_return_motion_on_the_operator_after_a_fault():
    source = inspect.getsource(entrypoint_module.main)
    assert 'run_end["kind"] not in ("completed", "operator_stop")' in source
    assert "_await_return_decision(" in source
    assert "if run_return:" in source
    assert "operator_ack_before_controlled_release" not in source


@pytest.mark.parametrize(
    "overrides",
    [
        dict(action_sink="sdk"),
        dict(boundary_gravity_offset="off"),
        dict(boundary_lane="pose"),
    ],
)
def test_gravity_offset_is_built_only_for_the_boundary_joint_lane(overrides):
    args = argparse.Namespace(
        action_sink="boundary", boundary_gravity_offset="on",
        boundary_lane="joint", boundary_gravity_offset_scale=1.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    assert entrypoint_module.build_boundary_gravity_offset(args, {}) is None


def test_gravity_offset_on_the_joint_lane_fails_loudly_without_its_config():
    args = argparse.Namespace(
        action_sink="boundary", boundary_gravity_offset="on",
        boundary_lane="joint", boundary_gravity_offset_scale=1.0,
    )
    with pytest.raises(ValueError, match="boundary_gravity_offset"):
        entrypoint_module.build_boundary_gravity_offset(args, {})


@pytest.mark.parametrize("scale, ok", [(0.0, True), (1.0, True), (2.0, True), (-0.1, False), (2.5, False)])
def test_gravity_offset_scale_is_validated_before_start(scale, ok):
    args = argparse.Namespace(
        boundary_publish_period_s=0.1, boundary_chunk_rows=16, boundary_hold_refresh_s=1.0,
        boundary_gravity_offset_scale=scale,
    )
    if ok:
        entrypoint_module.validate_boundary_publish_args(args)
    else:
        with pytest.raises(ValueError, match="gravity-offset-scale"):
            entrypoint_module.validate_boundary_publish_args(args)


def test_the_venue_default_turns_the_gravity_offset_on_and_wires_it_to_the_sink():
    import sys

    argv = sys.argv
    try:
        sys.argv = ["entrypoint", "--stage", "2", "--action-sink", "boundary"]
        args = entrypoint_module.parse_args()
    finally:
        sys.argv = argv
    assert args.boundary_gravity_offset == "on"
    assert args.boundary_gravity_offset_scale == 1.0
    source = inspect.getsource(entrypoint_module.main)
    assert "gravity_offset=boundary_gravity_offset," in source
    assert "build_boundary_gravity_offset(args, skill_cfg_raw)" in source


# ---- R で選べる候補 (Issue #188) ----

_ALTERNATIVES = {
    "rotate_table_base": ["rotate_table_base_ramen_ori_all6_400k", "rotate_table_base_ramen_ori_all6_300k"],
    "pick_table_leg": ["pick_table_leg_ramen_ori_all6_400k"],
    "insert_table_leg": ["groot_insert_leg_200k"],
    "rotate_leg_to_tighten": ["rotate_leg_to_tighten_ramen_ori_all6_400k"],
    "flip_table": ["groot_flip_table_n17_2_legacy_fk"],
}


def _resolve_alternatives(tmp_path, args, alternatives=_ALTERNATIVES):
    """既定で埋めてから候補を決める (main() と同じ順)。"""
    data = yaml.safe_load(REAL_POLICY_CONFIG.read_text(encoding="utf-8"))
    data["alternatives_by_skill"] = alternatives
    config = tmp_path / "policy_config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    apply_default_policy_variants(args, config_path=config)
    return resolve_policy_alternatives(args, config_path=config)


def _names(resolved):
    """slot → 候補の variant 名の tuple (`PolicyAlternative` の名前だけを見る)。"""
    return {slot: tuple(c.variant for c in candidates) for slot, candidates in resolved.items()}


def test_a_stage_gets_the_alternatives_of_its_own_skills_only(tmp_path):
    args = _phase3_args(2)
    resolved = _resolve_alternatives(tmp_path, args)
    assert _names(resolved) == {
        "rotate_table": ("rotate_table_base_ramen_ori_all6_400k", "rotate_table_base_ramen_ori_all6_300k"),
        "pick": ("pick_table_leg_ramen_ori_all6_400k",),
        "insert": ("groot_insert_leg_200k",),
        "rotate_leg": ("rotate_leg_to_tighten_ramen_ori_all6_400k",),
    }
    assert args.policy_alternatives == resolved
    assert _names(_resolve_alternatives(tmp_path, _phase3_args(5))) == {
        "flip": ("groot_flip_table_n17_2_legacy_fk",),
    }


def test_stage_zero_has_no_alternatives(tmp_path):
    args = _phase3_args(0)
    assert _resolve_alternatives(tmp_path, args) == {}
    assert args.policy_alternatives == {}


def test_the_chosen_main_is_not_offered_again_as_an_alternative(tmp_path):
    # slot の引数で既定を候補の 1 つに替えたら、その名前は候補から外れる
    args = _phase3_args(2, policy_variant_rotate_table_base="rotate_table_base_ramen_ori_all6_400k")
    assert _names(_resolve_alternatives(tmp_path, args))["rotate_table"] == ("rotate_table_base_ramen_ori_all6_300k",)
    # 束で既定が候補と同じになった slot は、候補が無くなるので載らない
    args = _phase3_args(2, policy_variant_set="all6_400k")
    assert "rotate_leg" not in _resolve_alternatives(tmp_path, args)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        (0, {}),
        (
            1,
            {
                "pick": (
                    "groot_pick_legs_v1（hybrid）",
                    "pick_table_leg_ramen_ori_venue_all6_20k（hybrid）",
                ),
                "insert": ("insert_table_leg_diffusion_optimal_100k",),
            },
        ),
        (
            2,
            {
                "pick": (
                    "groot_pick_legs_v1（hybrid）",
                    "pick_table_leg_ramen_ori_venue_all6_20k（hybrid）",
                ),
                "insert": ("insert_table_leg_diffusion_optimal_100k",),
            },
        ),
        (
            5,
            {
                "flip": ("groot_flip_table_n17_2_legacy_fk",),
            },
        ),
    ],
)
def test_the_real_config_offers_the_chosen_candidates(stage, expected):
    """選んだ候補 (2026-09-30): pick の既定は hybrid なしの RAMEN-Ori、hybrid の 2 通りを候補に残す。

    insert は前の既定 (Diffusion 100k)、flip も前の既定 (GR00T。2026-10-10 に既定を a_n15 に替えた)。台を回すのは前の既定 (141) の model を
    2026-10-09 (Issue #190) に消したので、締め付けと同じく候補なし。
    main() と同じ順 (pick のやり方 → 既定で埋める → 候補) で、会場の経路の引数で決める。
    """
    args = _phase3_args(
        stage, pick_leg_hybrid=None, action_sink="boundary", synthetic_hand_state=True
    )
    resolve_production_pick_mode(args)
    _fill(args)
    resolved = resolve_policy_alternatives(args, config_path=REAL_POLICY_CONFIG)
    assert {
        slot: tuple(
            entrypoint_module.describe_policy_alternative(
                c, slot=slot, hybrid_run=bool(args.pick_leg_hybrid)
            )
            for c in candidates
        )
        for slot, candidates in resolved.items()
    } == expected


# ---- --gpu-models plan: 決めた順に 1 本ずつ読む (Issue #188 ② 段 2) ----


def test_the_plan_lists_each_skill_main_first_then_its_candidates():
    sequence = [
        "hold_pose_for_rotate_table_base", "rotate_table_base",
        "hold_transition_rotate_table_base_to_pick_table_leg", "pick_table_leg",
    ]
    mains = {"pick_table_leg": "pick-main", "rotate_table_base": "rotate-main"}
    models = entrypoint_module.plan_models_for_stage(
        sequence,
        mains,
        main_variants={"rotate_table_base": "r141", "pick_table_leg": "groot_pick_legs_v1"},
        main_kinds={"rotate_table_base": "ramen_ori", "pick_table_leg": "groot_pick_legs"},
        candidates={
            "rotate_table_base": [("r_a", "ramen_ori", "ra"), ("r_b", "ramen_ori", "rb")],
            "pick_table_leg": [("p_a", "ramen_ori", "pa")],
        },
    )
    assert [(m.skill, m.variant, m.kind, m.policy, m.required) for m in models] == [
        ("rotate_table_base", "r141", "ramen_ori", "rotate-main", True),
        ("rotate_table_base", "r_a", "ramen_ori", "ra", False),
        ("rotate_table_base", "r_b", "ramen_ori", "rb", False),
        ("pick_table_leg", "groot_pick_legs_v1", "groot_pick_legs", "pick-main", True),
        ("pick_table_leg", "p_a", "ramen_ori", "pa", False),
    ]


def test_every_gate_that_starts_motion_waits_for_the_load():
    class _Gate:
        def __init__(self):
            self.barrier = None
            self.cancel = None

        def set_leave_barrier(self, ready_fn, *, description, cancel_fn=None):
            self.barrier = ready_fn
            self.cancel = cancel_fn

    names = [
        "operator_gate_for_pick_table_leg", "retry_wait_pick_table_leg",
        "retry_hand_gate_pick_table_leg", "retry_hold_pick_table_leg",
    ]
    registry = {name: _Gate() for name in names}

    def may_leave():
        return True

    def stay():
        return None

    installed = entrypoint_module.install_leave_barriers(
        registry, ["pick_table_leg", "insert_table_leg"], may_leave, stay
    )
    assert installed == names[:3]
    assert all(registry[name].barrier is may_leave for name in names[:3])
    assert all(registry[name].cancel is stay for name in names[:3])  # 確認の取り消しで読み込みを続ける
    assert registry["retry_hold_pick_table_leg"].barrier is None  # 次も止まっている区間


def test_plan_is_only_for_a_single_stage_and_reads_model_loading_first():
    source = inspect.getsource(entrypoint_module.main)
    parse = source.index('load_plan_active = str(args.gpu_models).lower() == "plan"')
    assert source.index("--gpu-models plan needs a single --stage N run") > parse
    assert source.index("load_model_loading(args.policy_config)") > parse
    # 候補を読むのは plan のときだけ (他の読み方で黙って使わない)
    assert "policy candidates are loaded only with" in source
    # 動かす run で、動き出す関所が読み込みを待たなければ起動で止める
    assert "these gates cannot wait for a model load" in source


def test_the_hybrid_vlm_is_listed_before_the_pick_main_and_waited_for():
    models = entrypoint_module.plan_models_for_stage(
        ["rotate_table_base", "pick_table_leg"],
        {"rotate_table_base": "r", "pick_table_leg": "p"},
        main_variants={"rotate_table_base": "r141", "pick_table_leg": "groot_pick_legs_v1"},
        main_kinds={"rotate_table_base": "ramen_ori", "pick_table_leg": "groot_pick_legs"},
        candidates={"pick_table_leg": [("p_a", "ramen_ori", "pa")]},
        before_main={
            "pick_table_leg": [
                PlannedModel(skill="pick_table_leg", label="pick_table_leg VLM", kind="vlm", policy="vlm")
            ]
        },
    )
    assert [(m.label, m.kind, m.required) for m in models] == [
        ("rotate_table_base main r141", "ramen_ori", True),
        ("pick_table_leg VLM", "vlm", True),
        ("pick_table_leg main groot_pick_legs_v1", "groot_pick_legs", True),
        ("pick_table_leg candidate p_a", "ramen_ori", False),
    ]


def test_plan_starts_the_vlm_in_the_load_order_and_other_modes_at_startup():
    source = inspect.getsource(entrypoint_module.main)
    plan_branch = source.index("planned_vlm = PlannedVlm(vlm_server, hybrid_pick_cfg, references)")
    startup = source.index("vlm_server.wait_until_ready()")
    assert source.index("if load_plan_active:", plan_branch - 400) < plan_branch < startup
    # plan では起動時の確かめ (probe) を PlannedVlm に任せる
    assert "if planned_vlm is not None:" in source
    assert 'label="pick_table_leg VLM",' in source and "policy=planned_vlm," in source


def test_each_planned_model_knows_its_weight_files_from_the_ckpt_ref(monkeypatch):
    from inference.desktop.lower_policy.policies import page_cache

    monkeypatch.setattr(page_cache, "ckpt_ref_files", lambda ref: [f"files of {ref}"])
    models = entrypoint_module.plan_models_for_stage(
        ["rotate_table_base"],
        {"rotate_table_base": "r"},
        main_variants={"rotate_table_base": "r141"},
        main_kinds={"rotate_table_base": "ramen_ori"},
        candidates={"rotate_table_base": [("r_a", "ramen_ori", "ra"), ("r_b", "ramen_ori", "rb")]},
        ckpt_refs={"r141": "Team-RAMEN/r@1", "r_a": "Team-RAMEN/a@2"},
    )
    assert models[0].weight_files() == ["files of Team-RAMEN/r@1"]
    assert models[1].weight_files() == ["files of Team-RAMEN/a@2"]
    assert models[2].weight_files is None  # ckpt_ref が分からない物は落とさない


def test_the_load_records_go_to_the_orch_log():
    source = inspect.getsource(entrypoint_module.main)
    assert "event_fn=_log_load_event," in source
    writer = source[source.index("def _log_load_event") :]
    writer = writer[: writer.index("def _build_load_plan")]
    assert "log_sink.write(json.dumps(record" in writer and "log_sink.flush()" in writer


def test_the_dry_run_also_checks_the_candidates_and_the_vlm_under_plan():
    source = inspect.getsource(entrypoint_module.main)
    mains = source.index("validated.add(id(policy_resource))")
    extras = source.index("single_stage_residency.check_each(skip_ids=validated)")
    done = source.index("[preflight] Phase 3 model/config/4-camera/")
    assert mains < extras < done


# ---- R の後に候補の model を選ぶ (Issue #188 ② 段 3) ----

_SKILL_YAML = Path("inference/desktop/lower_policy/configs/skill_config.yaml")


def _main_skill_config(flip_variant="groot_flip_table_n17_2_legacy_fk"):
    """entrypoint と同じ順に作った main の設定と、読んだままの yaml。"""
    from inference.desktop.lower_policy.initial_pose import apply_policy_variant_profile

    raw = yaml.safe_load(_SKILL_YAML.read_text(encoding="utf-8"))
    main = apply_policy_variant_profile(raw, "flip_table", flip_variant)
    main = entrypoint_module.apply_venue_start_poses(main, "boundary")
    main["hand_pre_motion"] = {**(main.get("hand_pre_motion") or {}), "open_rad": 4.2}
    main["skills"]["pick_table_leg"]["dispatch_waist"] = False  # hybrid が外す
    return main, raw


def test_a_flip_candidate_does_not_inherit_the_legacy_fk_offset():
    main, raw = _main_skill_config()
    assert main["skills"]["flip_table"]["wrist_tool_offset"]["left"] == [0.05, 0.0, 0.0]
    candidate = entrypoint_module.candidate_skill_config(
        main, raw, "flip_table", "flip_table_ramen_ori_flip_sel_a_n15_20k", "boundary"
    )
    assert candidate["skills"]["flip_table"].get("wrist_tool_offset") == raw["skills"][
        "flip_table"
    ].get("wrist_tool_offset")
    assert (
        candidate["skills"]["flip_table"].get("wrist_tool_offset")
        != main["skills"]["flip_table"]["wrist_tool_offset"]
    )
    # 全体の節と他の skill はこの run の設定のまま
    assert candidate["hand_pre_motion"]["open_rad"] == 4.2
    assert (
        candidate["skills"]["rotate_table_base"] == main["skills"]["rotate_table_base"]
    )


def test_a_groot_flip_candidate_brings_its_own_legacy_fk_offset():
    """既定が a_n15 (offset は yaml のまま) のとき、候補の GR00T は自分の offset (0.05, 0, 0) を持つ。"""
    main, raw = _main_skill_config("flip_table_ramen_ori_flip_sel_a_n15_20k")
    assert main["skills"]["flip_table"].get("wrist_tool_offset") == raw["skills"]["flip_table"].get(
        "wrist_tool_offset"
    )
    candidate = entrypoint_module.candidate_skill_config(
        main, raw, "flip_table", "groot_flip_table_n17_2_legacy_fk", "boundary"
    )
    assert candidate["skills"]["flip_table"]["wrist_tool_offset"] == {
        "left": [0.05, 0.0, 0.0],
        "right": [0.05, 0.0, 0.0],
    }


def test_a_candidate_gets_the_venue_start_pose_and_the_yaml_waist_setting():
    from inference.desktop.lower_policy.initial_pose import initial_pose_from_config

    main, raw = _main_skill_config()
    for skill in ("flip_table", "pick_table_leg", "rotate_table_base"):
        candidate = entrypoint_module.candidate_skill_config(
            main, raw, skill, f"{skill}_ramen_ori_venue_all6_20k", "boundary"
        )
        assert (
            entrypoint_module.start_pose_difference(
                initial_pose_from_config(main, skill),
                initial_pose_from_config(candidate, skill),
            )
            is None
        ), skill
    pick = entrypoint_module.candidate_skill_config(
        main,
        raw,
        "pick_table_leg",
        "pick_table_leg_ramen_ori_venue_all6_20k",
        "boundary",
    )
    assert pick["skills"]["pick_table_leg"]["dispatch_waist"] == raw["skills"][
        "pick_table_leg"
    ].get("dispatch_waist", True)


def test_a_candidate_with_another_start_pose_is_detected():
    from inference.desktop.lower_policy.initial_pose import initial_pose_from_config

    main, raw = _main_skill_config()
    n17_4 = entrypoint_module.candidate_skill_config(
        main, raw, "flip_table", "groot_flip_table_n17_4", "boundary"
    )
    difference = entrypoint_module.start_pose_difference(
        initial_pose_from_config(main, "flip_table"),
        initial_pose_from_config(n17_4, "flip_table"),
    )
    assert difference is not None and "arm joint" in difference


def test_the_r_screen_offers_the_choice_and_the_plan_binds_it():
    source = inspect.getsource(entrypoint_module.main)
    assert (
        "_build_policy_choices() if load_plan_active and policy_alternatives else {}"
        in source
    )
    # R の後の待ちは、候補があれば数字キーも受ける
    wait = source[
        source.index("skill_registry[wait_name] = OperatorConfirmationHoldSkill(") :
    ]
    wait = wait[: wait.index("skill_registry[arm_name]")]
    assert '("r",) + choice.keys' in wait and "labels=choice.labels()" in wait
    assert "choice=choice," in wait
    # 候補の policy は候補の skill の中の物を LoadPlan が読む。選ぶ口は LoadPlan の状態と dispatcher
    plan = source[source.index("def _build_load_plan") :]
    assert "option.skill._policy" in plan
    assert (
        "manager.model_state(" in plan and "_replace_policy_skill(skill, new)" in plan
    )
    replace = source[source.index("def _replace_policy_skill") :]
    assert "dispatcher.replace_skill(skill, new_skill)" in replace[:400]
    # 開始姿勢の違う候補は起動時に止める
    assert "starts from another " in source


# ---- pick の候補: hybrid あり・なし (Issue #188、候補の選定) ----

_VENUE_PICK = "pick_table_leg_ramen_ori_venue_all6_20k"


def test_a_hybrid_run_offers_the_venue_pick_with_and_without_the_hybrid(tmp_path):
    from inference.desktop.lower_policy.policies.config_loader import PolicyAlternative

    args = _phase3_args(2, pick_leg_hybrid=True, policy_variant_pick="groot_pick_legs_v1")
    resolved = _resolve_alternatives(
        tmp_path,
        args,
        {
            "pick_table_leg": [
                _VENUE_PICK,
                {"variant": _VENUE_PICK, "pick_leg_hybrid": False},
                # 既定と同じ GR00T でも hybrid なしなら別のやり方なので残る
                {"variant": "groot_pick_legs_v1", "pick_leg_hybrid": False},
                # 既定そのもの (hybrid の GR00T) は除く
                "groot_pick_legs_v1",
            ]
        },
    )
    assert resolved["pick"] == (
        PolicyAlternative(_VENUE_PICK),
        PolicyAlternative(_VENUE_PICK, False),
        PolicyAlternative("groot_pick_legs_v1", False),
    )
    labels = [
        entrypoint_module.describe_policy_alternative(c, slot="pick", hybrid_run=True)
        for c in resolved["pick"]
    ]
    assert labels == [
        f"{_VENUE_PICK}（hybrid）",
        f"{_VENUE_PICK}（hybrid なし）",
        "groot_pick_legs_v1（hybrid なし）",
    ]


def test_a_hybrid_run_rejects_a_pick_the_hybrid_cannot_run(tmp_path):
    args = _phase3_args(2, pick_leg_hybrid=True)
    with pytest.raises(ValueError, match="cannot run in Phase 1"):
        _resolve_alternatives(
            tmp_path, args, {"pick_table_leg": ["pick_table_leg_ramen_ori_all6_400k"]}
        )
    # hybrid なしと書けば通る
    ok = _resolve_alternatives(
        tmp_path,
        _phase3_args(2, pick_leg_hybrid=True),
        {"pick_table_leg": [{"variant": "pick_table_leg_ramen_ori_all6_400k", "pick_leg_hybrid": False}]},
    )
    assert _names(ok) == {"pick": ("pick_table_leg_ramen_ori_all6_400k",)}


def test_without_the_hybrid_both_forms_are_the_same_learned_pick(tmp_path):
    args = _phase3_args(2, pick_leg_hybrid=False, policy_variant_pick=_VENUE_PICK)
    resolved = _resolve_alternatives(
        tmp_path,
        args,
        {"pick_table_leg": [{"variant": _VENUE_PICK, "pick_leg_hybrid": False}, "groot_pick_legs_v2"]},
    )
    assert _names(resolved) == {"pick": ("groot_pick_legs_v2",)}  # 既定と同じ学習の pick は除く
    assert (
        entrypoint_module.describe_policy_alternative(
            resolved["pick"][0], slot="pick", hybrid_run=False
        )
        == "groot_pick_legs_v2"
    )


def test_pick_candidates_share_one_policy_and_the_hybrid_one_keeps_the_hybrid_setup():
    source = inspect.getsource(entrypoint_module.main)
    choices = source[source.index("def _build_policy_choices") :]
    choices = choices[: choices.index("policy_choices = (")]
    # 同じ variant の policy は 1 つだけ作る (既定の policy も共有の対象)
    assert "shared_policies = {main_variant: main_skill._policy}" in choices
    assert "policy=shared_policies[name]," in choices
    # hybrid の候補は既定と同じ引数・腰を送らない
    assert "RealPickLegHybridVlaSkill, dict(hybrid_pick_kwargs)" in choices
    assert 'cfg["skills"][skill_name]["dispatch_waist"] = False' in choices
    assert "pick_extra_kwargs = hybrid_pick_kwargs" in source
    # LoadPlan は共有した policy を 1 本だけ読む
    plan = source[source.index("def _build_load_plan") :]
    assert "seen = {id(policies[skill])}" in plan


def test_the_plan_wires_stay_and_a_bad_candidate_config_exits_cleanly():
    source = inspect.getsource(entrypoint_module.main)
    assert "skill_registry, list(policies), manager.may_leave, manager.stay" in source
    choices = source[source.index("def _build_policy_choices") :]
    assert "skill config invalid for policy candidate" in choices[: choices.index("policy_choices = (")]


# ---- 既定が hybrid でない pick と、hybrid の R の候補 (Issue #188) ----


def _config_with(tmp_path, **sections):
    data = yaml.safe_load(REAL_POLICY_CONFIG.read_text(encoding="utf-8"))
    data.update(sections)
    config = tmp_path / "policy_config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config


@pytest.mark.parametrize(("config_value", "expected"), [(True, True), (False, False)])
def test_the_pick_mode_default_follows_the_config(tmp_path, config_value, expected):
    config = _config_with(tmp_path, pick_leg_hybrid=config_value)
    args = _phase3_args(2, pick_leg_hybrid=None, policy_config=config)
    assert resolve_production_pick_mode(args) is expected
    assert args.pick_leg_hybrid is expected
    # 引数で指定したら config より優先
    args = _phase3_args(2, pick_leg_hybrid=True, policy_config=config)
    assert resolve_production_pick_mode(args) is False and args.pick_leg_hybrid is True


def _learned_pick_args(**overrides):
    values = dict(
        pick_leg_hybrid=False,
        policy_variant_pick=_VENUE_PICK,
        action_sink="boundary",
        synthetic_hand_state=True,
    )
    values.update(overrides)
    return _phase3_args(2, **values)


def test_a_learned_pick_run_offers_the_hybrid_candidates(tmp_path):
    from inference.desktop.lower_policy.policies.config_loader import PolicyAlternative

    args = _learned_pick_args()
    resolved = _resolve_alternatives(
        tmp_path,
        args,
        {
            "pick_table_leg": [
                {"variant": "groot_pick_legs_v1", "pick_leg_hybrid": True},
                {"variant": _VENUE_PICK, "pick_leg_hybrid": True},
                {"variant": _VENUE_PICK, "pick_leg_hybrid": False},  # 既定そのものは除く
            ]
        },
    )
    assert resolved["pick"] == (
        PolicyAlternative("groot_pick_legs_v1", True),
        PolicyAlternative(_VENUE_PICK, True),
    )
    assert [
        entrypoint_module.describe_policy_alternative(c, slot="pick", hybrid_run=False)
        for c in resolved["pick"]
    ] == ["groot_pick_legs_v1（hybrid）", f"{_VENUE_PICK}（hybrid）"]


def test_a_run_that_cannot_do_the_hybrid_drops_only_the_hybrid_candidates(tmp_path, capsys):
    """既定の pick は動けるので止めない (PR #189 の review)。hybrid の候補だけ WARNING を出して外す。"""
    args = _learned_pick_args(no_wrist_cameras=True)
    resolved = _resolve_alternatives(
        tmp_path,
        args,
        {
            "pick_table_leg": [
                {"variant": "groot_pick_legs_v1", "pick_leg_hybrid": True},
                "groot_pick_legs_v2",  # hybrid なしの候補は残る
            ]
        },
    )
    assert _names(resolved) == {"pick": ("groot_pick_legs_v2",)}
    assert "hybrid pick candidate 'groot_pick_legs_v1' is dropped" in capsys.readouterr().err
    assert entrypoint_module.hybrid_pick_unavailable(_learned_pick_args()) is None


def test_a_wrongly_written_hybrid_candidate_still_stops_the_run(tmp_path):
    # 条件が足りなくても、hybrid の区間 1 に使えない variant を書いたのは書き方の誤り
    with pytest.raises(ValueError, match="cannot run in Phase 1"):
        _resolve_alternatives(
            tmp_path,
            _learned_pick_args(no_wrist_cameras=True),
            {"pick_table_leg": [{"variant": "pick_table_leg_ramen_ori_all6_400k", "pick_leg_hybrid": True}]},
        )


def test_the_hybrid_pick_candidates_can_be_dropped_together():
    from inference.desktop.lower_policy.policies.config_loader import PolicyAlternative

    alternatives = {
        "pick": (PolicyAlternative("groot_pick_legs_v1", True), PolicyAlternative("groot_pick_legs_v2")),
        "insert": (PolicyAlternative("insert_x"),),
    }
    assert entrypoint_module.without_hybrid_pick_alternatives(alternatives, False) == {
        "pick": (PolicyAlternative("groot_pick_legs_v2"),),
        "insert": (PolicyAlternative("insert_x"),),
    }
    only_hybrid = {"pick": (PolicyAlternative("groot_pick_legs_v1", True),)}
    assert entrypoint_module.without_hybrid_pick_alternatives(only_hybrid, False) == {}


def test_a_learned_pick_run_keeps_going_when_the_hybrid_cannot_be_prepared():
    source = inspect.getsource(entrypoint_module.main)
    block = source[source.index("    if needs_hybrid_pick:\n        try:") :]
    block = block[: block.index("    if args.phase1_11_arm_only:")]
    # 参照画像・外の VLM の確かめが失敗したら: 既定が hybrid なら今までどおり止め、そうでなければ候補を外す
    handler = block[block.index("        except Exception as exc:") :]
    assert "            if args.pick_leg_hybrid:\n                raise" in handler
    assert "without_hybrid_pick_alternatives(policy_alternatives, False)" in handler
    assert "planned_vlm = None" in handler


def test_the_vlm_is_a_candidate_part_when_the_default_pick_is_learned():
    vlm = PlannedModel(skill="pick_table_leg", label="vlm", kind="vlm", policy="vlm", required=False, variant="vlm")
    models = entrypoint_module.plan_models_for_stage(
        ["pick_table_leg"],
        {"pick_table_leg": "venue"},
        main_variants={"pick_table_leg": _VENUE_PICK},
        main_kinds={"pick_table_leg": "ramen_ori"},
        candidates={"pick_table_leg": [("groot_pick_legs_v1", "groot_pick_legs", "groot")]},
        before_candidates={"pick_table_leg": [vlm]},
    )
    assert [(m.label, m.required) for m in models] == [
        (f"pick_table_leg main {_VENUE_PICK}", True),
        ("vlm", False),  # 保持は待たない。hybrid の候補の一部
        ("pick_table_leg candidate groot_pick_legs_v1", False),
    ]


def test_a_learned_pick_run_prepares_the_hybrid_for_its_candidates():
    source = inspect.getsource(entrypoint_module.main)
    assert "needs_hybrid_pick = bool(args.pick_leg_hybrid or hybrid_pick_candidates)" in source
    assert "    if needs_hybrid_pick:\n" in source  # hybrid の設定・VLM
    choices = source[source.index("def _build_policy_choices") :]
    choices = choices[: choices.index("policy_choices = (")]
    assert 'requires = (\n                    ("vlm",) if hybrid and not hybrid_run' in choices
    plan = source[source.index("def _build_load_plan") :]
    assert "required=bool(args.pick_leg_hybrid)," in plan
    assert "before_candidates = {} if args.pick_leg_hybrid else vlm_models" in plan


def _walk_arm(left_elbow: float, right_elbow: float) -> np.ndarray:
    """腕 14 関節 (左 7・右 7)。肩 roll だけ下ろした姿勢の値、肘は引数。"""
    arm = np.zeros(14)
    arm[1], arm[8] = 0.19, -0.19
    arm[3], arm[10] = left_elbow, right_elbow
    return arm


_LOWERED_POSE = _walk_arm(0.9, 0.9)


def test_stage0_preflight_lets_setup_lower_bent_arms():
    # 会場 09-30: 曲がった腕のまま起動 (肘 左 1.550・右 1.278)。setup が下ろして 0.83 で歩いた
    message = entrypoint_module._stage0_walk_arm_preflight(
        _walk_arm(1.550, 1.278), _LOWERED_POSE
    )

    assert "outside the lowered walk envelope" in message
    assert "setup lowers them" in message


def test_stage0_preflight_blocks_bent_arms_without_lowered_pose():
    with pytest.raises(RuntimeError, match="not in the lowered walk envelope"):
        entrypoint_module._stage0_walk_arm_preflight(_walk_arm(1.550, 1.278), None)


@pytest.mark.parametrize("lowered_pose", [None, _LOWERED_POSE])
def test_stage0_preflight_passes_lowered_arms(lowered_pose):
    message = entrypoint_module._stage0_walk_arm_preflight(
        _walk_arm(0.832, 0.847), lowered_pose
    )

    assert "lowered walk-arm envelope passed" in message


def test_stage0_preflight_rejects_invalid_arm_even_with_lowered_pose():
    arm = _walk_arm(1.550, 1.278)
    arm[0] = np.nan

    with pytest.raises(ValueError, match="finite 14-D"):
        entrypoint_module._stage0_walk_arm_preflight(arm, _LOWERED_POSE)


def test_stage0_preflight_uses_the_configured_lowered_pose():
    source = " ".join(inspect.getsource(entrypoint_module.main).split())

    assert (
        "_stage0_walk_arm_preflight( arm_actuator.read_arm_positions(), "
        "_walk_lowered_pose(skill_cfg_raw), )"
    ) in source
