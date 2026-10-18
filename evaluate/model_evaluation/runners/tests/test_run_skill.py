from pathlib import Path

import numpy as np
import pytest
import yaml

from evaluate.model_evaluation.runners.run_skill import (
    CAMERA_PREFLIGHT_STALE_TIMEOUT_S,
    CAMERA_STALE_TIMEOUT_S,
    TICK_HZ,
    _camera_stale_roles,
    _collect_obs,
    _load_evaluation_initial_pose,
    _measured_arm_from_observation,
    _missing_observation_inputs,
    _require_strict_regular,
    _resolve_hard_timeout_from_skill_config,
    _sleep_to_next_tick,
)


def test_collect_obs_runs_overlay_detector_on_head_left_only() -> None:
    packed = np.zeros((4, 12, 3), dtype=np.uint8)
    packed[:, :6] = 17
    packed[:, 6:] = 99

    class Source:
        def __init__(self, value) -> None:  # noqa: ANN001
            self.value = value

        def get(self):  # noqa: ANN201
            return self.value

    class Perception:
        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []

        def predict(self, frame: np.ndarray) -> list[str]:
            self.frames.append(frame.copy())
            return ["detected"]

    perception = Perception()
    obs = _collect_obs(
        None,
        None,
        Source(packed),
        Source(np.zeros((4, 6, 3), dtype=np.uint8)),
        Source(np.zeros((4, 6, 3), dtype=np.uint8)),
        overlay_perception=perception,
    )

    assert obs["cleaned"] == ["detected"]
    assert len(perception.frames) == 1
    assert perception.frames[0].shape == (4, 6, 3)
    assert np.all(perception.frames[0] == 17)


def test_missed_control_deadline_does_not_trigger_catch_up_burst(monkeypatch) -> None:
    now = 10.0
    sleeps: list[float] = []
    monkeypatch.setattr(
        "evaluate.model_evaluation.runners.run_skill.time.monotonic", lambda: now
    )
    monkeypatch.setattr(
        "evaluate.model_evaluation.runners.run_skill.time.sleep", sleeps.append
    )

    following = _sleep_to_next_tick(next_deadline=now - 0.050)

    assert sleeps == []
    assert following == pytest.approx(now + 1.0 / TICK_HZ)


def test_on_time_control_deadline_preserves_periodic_phase(monkeypatch) -> None:
    now = 10.0
    sleeps: list[float] = []
    monkeypatch.setattr(
        "evaluate.model_evaluation.runners.run_skill.time.monotonic", lambda: now
    )
    monkeypatch.setattr(
        "evaluate.model_evaluation.runners.run_skill.time.sleep", sleeps.append
    )

    following = _sleep_to_next_tick(next_deadline=now + 0.010)

    assert sleeps == pytest.approx([0.010])
    assert following == pytest.approx(now + 0.010 + 1.0 / TICK_HZ)


def test_camera_staleness_uses_receive_time_not_cached_image_presence() -> None:
    now_ns = 10_000_000_000
    obs = {
        "_camera_frames": {
            role: np.zeros((4, 4, 3), dtype=np.uint8)
            for role in ("head_left", "head_right", "left_wrist", "right_wrist")
        },
        "_camera_received_monotonic_ns": {
            "head_left": now_ns - 10_000_000,
            "head_right": now_ns - 10_000_000,
            "left_wrist": now_ns - 750_000_000,
            "right_wrist": now_ns - 20_000_000,
        },
    }

    stale = _camera_stale_roles(obs, require_wrist=True, now_ns=now_ns)

    assert set(stale) == {"left_wrist"}
    assert stale["left_wrist"] == pytest.approx(0.75)


def test_camera_staleness_requires_bilateral_wrist_timestamps() -> None:
    now_ns = 10_000_000_000
    obs = {
        "_camera_received_monotonic_ns": {
            "head_left": now_ns,
            "head_right": now_ns,
            "left_wrist": now_ns,
        }
    }

    stale = _camera_stale_roles(obs, require_wrist=True, now_ns=now_ns)

    assert set(stale) == {"right_wrist"}
    assert np.isinf(stale["right_wrist"])


def test_preflight_allows_small_transient_without_weakening_runtime_limit() -> None:
    """Regression: one 0.534 s DDS gap must not fail a healthy rate window."""

    now_ns = 10_000_000_000
    obs = {
        "_camera_received_monotonic_ns": {
            role: now_ns - 534_000_000
            for role in ("head_left", "head_right", "left_wrist", "right_wrist")
        }
    }

    runtime_stale = _camera_stale_roles(
        obs,
        require_wrist=True,
        max_age_s=CAMERA_STALE_TIMEOUT_S,
        now_ns=now_ns,
    )
    preflight_stale = _camera_stale_roles(
        obs,
        require_wrist=True,
        max_age_s=CAMERA_PREFLIGHT_STALE_TIMEOUT_S,
        now_ns=now_ns,
    )

    assert set(runtime_stale) == {
        "head_left",
        "head_right",
        "left_wrist",
        "right_wrist",
    }
    assert preflight_stale == {}


def test_preflight_still_rejects_a_real_camera_outage() -> None:
    now_ns = 10_000_000_000
    obs = {
        "_camera_received_monotonic_ns": {
            role: now_ns - 751_000_000
            for role in ("head_left", "head_right", "left_wrist", "right_wrist")
        }
    }

    stale = _camera_stale_roles(
        obs,
        require_wrist=True,
        max_age_s=CAMERA_PREFLIGHT_STALE_TIMEOUT_S,
        now_ns=now_ns,
    )

    assert set(stale) == {
        "head_left",
        "head_right",
        "left_wrist",
        "right_wrist",
    }


def test_move_table_base_uses_final_setup_stage(tmp_path: Path) -> None:
    """Old configs without frame-zero metadata retain a safe fallback."""
    config = {
        "skills": {
            "setup": {
                "stages": [
                    {
                        "name": "lateral",
                        "dwell_sec": 1.0,
                        "pose_rad": {"L.shoulder_roll": 0.8},
                    },
                    {
                        "name": "forward",
                        "dwell_sec": 1.0,
                        "pose_rad": {
                            "L.shoulder_pitch": -0.2,
                            "L.shoulder_roll": 1.2,
                            "R.shoulder_pitch": -0.2,
                            "R.shoulder_roll": -1.2,
                        },
                    },
                ]
            },
            "move_table_base": {"dispatch_waist": True},
        }
    }
    path = tmp_path / "skills.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    pose = _load_evaluation_initial_pose(path, "move_table_base")

    assert pose.statistic == "skill_config.setup.final_stage"
    assert pose.arm_position_rad.shape == (14,)
    np.testing.assert_allclose(
        pose.arm_position_rad[[0, 1, 7, 8]], [-0.2, 1.2, -0.2, -1.2]
    )
    assert pose.requires_separate_hand_initialization is False


def test_repo_move_table_base_uses_dataset_frame0_instead_of_clearance() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    path = repo_root / "inference/desktop/lower_policy/configs/skill_config.yaml"

    pose = _load_evaluation_initial_pose(path, "move_table_base")

    assert pose.statistic.startswith("median_action_robot_q_desired_frame0")
    assert pose.training_episode_count == 368
    assert pose.dataset_revision == "4b5961c6f8b97ececa75ee73041b30bcbf463ae2"
    # The old setup clearance was shoulder roll +/-1.2.  Policy must wait at
    # the dataset pose instead.
    np.testing.assert_allclose(
        pose.arm_position_rad[[1, 8]], [0.5789217949, -0.5614076257]
    )


def test_object_in_hand_skills_require_separate_hand_initialization() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    path = repo_root / "inference/desktop/lower_policy/configs/skill_config.yaml"

    insert = _load_evaluation_initial_pose(path, "insert_table_leg")
    tighten = _load_evaluation_initial_pose(path, "rotate_leg_to_tighten")
    combined = _load_evaluation_initial_pose(path, "insert_and_tighten")

    assert insert.requires_separate_hand_initialization is True
    assert tighten.requires_separate_hand_initialization is True
    assert combined.requires_separate_hand_initialization is True
    np.testing.assert_allclose(
        insert.dex1_target_rad,
        np.asarray(insert.dex1_opening_fraction) * 4.5,
    )
    np.testing.assert_allclose(
        tighten.dex1_target_rad,
        np.asarray(tighten.dex1_opening_fraction) * 4.5,
    )
    np.testing.assert_allclose(
        combined.dex1_target_rad,
        np.asarray(combined.dex1_opening_fraction) * 4.5,
    )


def test_missing_pose_for_other_skill_still_fails(tmp_path: Path) -> None:
    path = tmp_path / "skills.yaml"
    path.write_text(
        yaml.safe_dump({"skills": {"pick_table_leg": {}}}), encoding="utf-8"
    )

    with pytest.raises(KeyError, match="no initial_pose"):
        _load_evaluation_initial_pose(path, "pick_table_leg")


def test_policy_warmup_requires_atomic_live_inputs() -> None:
    class Joint:
        position = np.zeros(29)

    class Hand:
        position_rad = np.zeros(2)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    complete = {
        "joint_state": Joint(),
        "hand_state": Hand(),
        "head_rgb": np.zeros((480, 1280, 3), dtype=np.uint8),
        "wrist_left_rgb": frame,
        "wrist_right_rgb": frame,
    }
    assert _missing_observation_inputs(complete, require_wrist=True) == []

    incomplete = dict(complete)
    incomplete["wrist_right_rgb"] = None
    assert _missing_observation_inputs(incomplete, require_wrist=True) == [
        "wrist_right"
    ]
    assert _missing_observation_inputs(incomplete, require_wrist=False) == []


def test_strict_regular_gate_rejects_other_fsm() -> None:
    class Loco:
        def get_loco_status(self):
            return type("Status", (), {"fsm_id": 501, "fsm_mode": 1})()

    with pytest.raises(RuntimeError, match="strict Regular Mode"):
        _require_strict_regular(Loco())


def test_measured_arm_extracts_only_arm_14d() -> None:
    class Joint:
        position = np.arange(29, dtype=np.float64)

    measured = _measured_arm_from_observation({"joint_state": Joint()})
    np.testing.assert_array_equal(measured, np.arange(15, 29, dtype=np.float64))


def test_resolve_hard_timeout_reads_positive_yaml_value() -> None:
    cfg = {"skills": {"move_table_base": {"max_seconds_hard": 9}}}
    assert _resolve_hard_timeout_from_skill_config(cfg, "move_table_base") == 9.0


def test_resolve_hard_timeout_raises_when_field_missing() -> None:
    # 新 skill 追加時の書き忘れを検知するため fail-closed。
    cfg = {"skills": {"unknown_skill": {"dispatch_waist": True}}}
    with pytest.raises(KeyError, match="max_seconds_hard"):
        _resolve_hard_timeout_from_skill_config(cfg, "unknown_skill")


def test_resolve_hard_timeout_rejects_non_positive_value() -> None:
    cfg = {"skills": {"broken_skill": {"max_seconds_hard": 0}}}
    with pytest.raises(ValueError, match="must be > 0"):
        _resolve_hard_timeout_from_skill_config(cfg, "broken_skill")


def test_resolve_hard_timeout_raises_when_skill_missing() -> None:
    cfg = {"skills": {"other": {"max_seconds_hard": 5}}}
    with pytest.raises(KeyError, match="skills.missing"):
        _resolve_hard_timeout_from_skill_config(cfg, "missing")


def test_shipped_skill_config_has_max_seconds_hard_for_all_evaluated_skills() -> None:
    # skill_config.yaml と SKILL_REGISTRY (argparse --skill 選択肢) が同期して
    # いることの回帰 guard。新 skill が SKILL_REGISTRY に増えたら max_seconds_hard
    # 追加を強制する。
    from evaluate.model_evaluation.common.skill_registry import SKILL_REGISTRY
    from evaluate.model_evaluation.runners.run_skill import DEFAULT_SKILL_CONFIG

    with open(DEFAULT_SKILL_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    for skill in sorted(SKILL_REGISTRY.keys()):
        _resolve_hard_timeout_from_skill_config(cfg, skill)


def test_phase3_configured_timeout_progression_and_competition_budgets() -> None:
    from evaluate.model_evaluation.runners.run_skill import DEFAULT_SKILL_CONFIG

    with open(DEFAULT_SKILL_CONFIG) as fh:
        skills = yaml.safe_load(fh)["skills"]
    for name in (
        "rotate_table_base",
        "pick_table_leg",
        "insert_table_leg",
        "rotate_leg_to_tighten",
        "flip_table",
    ):
        # Current competition config deliberately advances on learned-policy
        # timeout (2026-09-22 decision). Hybrid pick has its own 65 s deadline;
        # failed preparation and grasp gates never count as successful arrival.
        assert skills[name]["on_timeout"] == "advance"
    assert skills["rotate_table_base"]["max_seconds_hard"] == 30
    flip = skills["flip_table"]
    assert flip["max_seconds_hard"] == 180
    assert flip["initial_pose"]["dataset_repo_id"].endswith("flip_table_2")
    assert flip["initial_pose"]["training_episode_count"] == 156
    assert flip["teacher_joint_range"]["source"]["frame_count"] == 71650


def test_flip_v4_explicit_variant_restores_its_own_dataset_contract() -> None:
    from evaluate.model_evaluation.runners.run_skill import DEFAULT_SKILL_CONFIG
    from inference.desktop.lower_policy.initial_pose import (
        apply_policy_variant_profile,
    )

    with open(DEFAULT_SKILL_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    profiled = apply_policy_variant_profile(
        cfg, "flip_table", "groot_flip_table_n17_4"
    )["skills"]["flip_table"]
    assert profiled["max_seconds_hard"] == 180
    assert profiled["on_timeout"] == "advance"
    assert profiled["initial_pose"]["dataset_repo_id"].endswith("flip_table_4")
    assert profiled["initial_pose"]["training_episode_count"] == 268
    assert profiled["teacher_joint_range"]["source"]["frame_count"] == 326556


def test_flip_legacy_fk_variant_changes_only_the_wrist_tool_offset() -> None:
    from evaluate.model_evaluation.runners.run_skill import DEFAULT_SKILL_CONFIG
    from inference.desktop.lower_policy.initial_pose import (
        apply_policy_variant_profile,
    )
    from inference.desktop.lower_policy.policies.config_loader import (
        load_policy_variant,
    )

    with open(DEFAULT_SKILL_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    baseline_skill = apply_policy_variant_profile(
        cfg, "flip_table", "groot_flip_table_n17_2_baseline"
    )["skills"]["flip_table"]
    legacy_skill = apply_policy_variant_profile(
        cfg, "flip_table", "groot_flip_table_n17_2_legacy_fk"
    )["skills"]["flip_table"]
    assert "wrist_tool_offset" not in baseline_skill
    offset = legacy_skill.pop("wrist_tool_offset")
    assert offset == {
        "left": [0.05, 0.0, 0.0],
        "right": [0.05, 0.0, 0.0],
    }
    assert legacy_skill == baseline_skill

    baseline = load_policy_variant(
        "inference/desktop/lower_policy/configs/policy_config.yaml",
        "groot_flip_table_n17_2_baseline",
    ).policy_config
    legacy = load_policy_variant(
        "inference/desktop/lower_policy/configs/policy_config.yaml",
        "groot_flip_table_n17_2_legacy_fk",
    ).policy_config
    assert legacy == baseline


def test_flip_full_run_default_is_ramen_ori_and_run_skill_keeps_the_groot_baseline() -> None:
    """通しの flip の既定は RAMEN-Ori の a_n15 に替えた (Issue #190、2026-10-10)。前の既定の GR00T v2 baseline
    (近い成功の手先の FK) は R で選べる候補に残し、1 skill の評価 (run_skill) の既定はそのまま。"""
    from evaluate.model_evaluation.common.skill_registry import SKILL_REGISTRY
    from inference.desktop.lower_policy.policies.config_loader import (
        PolicyAlternative,
        load_alternatives_by_skill,
        load_default_variant_by_skill,
    )

    legacy = "groot_flip_table_n17_2_legacy_fk"
    assert SKILL_REGISTRY["flip_table"].default_variant == legacy
    assert load_default_variant_by_skill()["flip_table"] == "flip_table_ramen_ori_flip_sel_a_n15_20k"
    assert PolicyAlternative(legacy) in load_alternatives_by_skill()["flip_table"]


def test_run_metadata_records_the_execution_layer_knobs() -> None:
    """Issue #137: 実機評価 run が「どの設定で走ったか」を残すこと。

    run_skill の metadata dump は PolicyConfig から直接読むので、field 名が
    変わると silent に記録が欠ける。09-07 の実機 run が実際にそうなっていて
    (execution_steps / temporal_lambda が残っておらず後追いできなかった)、
    RTC / async / TE を実機で振る今回は致命的になる。
    """
    from inference.desktop.lower_policy.policies.base import PolicyConfig
    from inference.desktop.lower_policy.rtc import RtcConfig

    cfg = PolicyConfig(mode="none", ckpt_ref="x")
    for name in ("action_space", "replan_family", "execution_steps", "temporal_lambda"):
        assert hasattr(cfg, name), f"run metadata reads PolicyConfig.{name}"
    assert isinstance(cfg.rtc, RtcConfig)
    for name in ("enabled", "frozen_steps", "overlap_steps", "ramp_rate"):
        assert hasattr(cfg.rtc, name), f"run metadata reads PolicyConfig.rtc.{name}"
