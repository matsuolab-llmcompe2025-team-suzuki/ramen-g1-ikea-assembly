"""assembly.py の unit test (Issue #141 束 1-8)。

主眼: 評価と本番が同じ部品を受け取ること。
    - policy は読み込んだ直後に必ず warmup → reset を通る (INF-13)
    - VlaSkill に prompt / skill_id / 手首 FK の offset / 教師の範囲 / progress monitor が入る
      (INF-8 / INF-3 / INF-9 / 1-4)
    - 位置の限界 (URDF − 0.03 rad) の 19D limiter が両経路で付く (INF-1)
    - 手首 FK は offset ごとに 1 個だけ作って使い回す
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import yaml

from inference.desktop import assembly
from inference.desktop.lower_policy.policies.base import PolicyConfig
from inference.desktop.lower_policy.policies.config_loader import VariantEntry


class _StubPolicy:
    """from_ckpt → warmup → reset の呼ばれ方を記録する。"""

    instances: list["_StubPolicy"] = []

    def __init__(self) -> None:
        self.calls: list[str] = []
        _StubPolicy.instances.append(self)

    @classmethod
    def from_ckpt(cls, policy_config):  # noqa: ANN001
        policy = cls()
        policy.calls.append("from_ckpt")
        return policy

    def warmup(self, n_iter: int = 5) -> None:
        self.calls.append(f"warmup({n_iter})")

    def reset(self) -> None:
        self.calls.append("reset")


class _StubVlaSkill:
    """VlaSkill の代わりに、渡された引数をそのまま持つ。"""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _variant(**policy_config_kwargs) -> VariantEntry:
    defaults = dict(mode="none", ckpt_ref="Team-RAMEN/stub@sha")
    defaults.update(policy_config_kwargs)
    return VariantEntry(
        name="stub_variant",
        policy_type="ramen_ori",
        policy_config=PolicyConfig(**defaults),
        yolo_ckpt_ref=None,
    )


def _skill_config(**skill_overrides) -> dict:
    skill: dict = {"dispatch_waist": False}
    skill.update(skill_overrides)
    return {
        "motion_limits_default": {
            "arm_velocity_rad_s": 1.0,
            "arm_acceleration_rad_s2": 10.0,
            "hand_velocity_rad_s": 2.0,
            "hand_acceleration_rad_s2": 20.0,
        },
        "skills": {"demo": skill},
    }


@pytest.fixture(autouse=True)
def _stub_policy_class(monkeypatch):
    """`resolve_policy_class` を stub に差し替える (torch 無しで回すため)。"""
    _StubPolicy.instances = []
    monkeypatch.setattr(assembly, "resolve_policy_class", lambda policy_type: _StubPolicy)


class TestLoadPolicy:
    def test_warms_up_once_and_resets_after_loading(self):
        policy = assembly.load_policy(_variant())
        assert policy.calls == ["from_ckpt", "warmup(1)", "reset"]

    def test_deferred_does_not_load_until_used(self):
        policy = assembly.load_policy(_variant(), deferred=True)
        assert _StubPolicy.instances == []
        policy.reset()
        assert _StubPolicy.instances[0].calls == ["from_ckpt", "warmup(1)", "reset", "reset"]

    def test_deferred_reports_loading_on_first_reset(self):
        policy = assembly.load_policy(_variant(), deferred=True)
        policy.reset()
        assert policy.loaded_during_last_reset
        policy.reset()
        assert not policy.loaded_during_last_reset

    def test_policy_without_reset_is_loaded_without_error(self, monkeypatch):
        """reset を持たない policy (Gr00tPolicyPickLegs) でも読み込めること。

        reset は policy の約束の任意の部分で、`VlaSkill._on_start` も持っている
        ものだけ呼ぶ。ここで無条件に呼ぶと、13 GiB の ckpt を読んだ後に
        AttributeError で落ちる (`groot_pick_legs_v2` は実在の slot)。
        """

        class _NoResetPolicy:
            def __init__(self):
                self.calls: list[str] = []

            @classmethod
            def from_ckpt(cls, cfg):
                policy = cls()
                policy.calls.append("from_ckpt")
                return policy

            def warmup(self, n_iter: int = 5):
                self.calls.append(f"warmup({n_iter})")

        monkeypatch.setattr(assembly, "resolve_policy_class", lambda _t: _NoResetPolicy)
        policy = assembly.load_policy(_variant())
        assert policy.calls == ["from_ckpt", "warmup(1)"]


class TestDispatchWaist:
    def test_reads_explicit_value(self):
        assert assembly.skill_dispatch_waist(_skill_config(dispatch_waist=True), "demo")

    def test_missing_is_rejected(self):
        config = _skill_config()
        del config["skills"]["demo"]["dispatch_waist"]
        with pytest.raises(ValueError, match="dispatch_waist must be explicit"):
            assembly.skill_dispatch_waist(config, "demo")

    def test_non_boolean_is_rejected(self):
        with pytest.raises(ValueError, match="must be boolean"):
            assembly.skill_dispatch_waist(_skill_config(dispatch_waist="yes"), "demo")


class _FakeFK:
    created: list[dict] = []

    @classmethod
    def from_urdf(cls, path, left_tool_offset=None, right_tool_offset=None):  # noqa: ANN001
        fk = cls()
        cls.created.append({"left": left_tool_offset, "right": right_tool_offset})
        return fk

    def left_tool_position(self, joint_positions):  # noqa: ANN001 - z 天井が要求する
        return np.zeros(3)


@pytest.fixture
def fake_fk(monkeypatch):
    _FakeFK.created = []
    monkeypatch.setattr(
        "inference.desktop.perception.g1_urdf_fk.G1WristFK", _FakeFK
    )
    return _FakeFK


class TestFkFactory:
    def test_default_offset_is_shared_between_skills(self, fake_fk):
        factory = assembly.FkFactory()
        config = {"skills": {"a": {}, "b": {}}}
        assert factory.for_skill(config, "a") is factory.for_skill(config, "b")
        assert len(fake_fk.created) == 1

    def test_per_skill_offset_builds_its_own_fk(self, fake_fk):
        config = {
            "skills": {
                "a": {"wrist_tool_offset": {"left": [0.0, 0.0, 0.1], "right": [0.0, 0.0, 0.2]}},
                "b": {},
            }
        }
        factory = assembly.FkFactory()
        assert factory.for_skill(config, "a") is not factory.for_skill(config, "b")
        assert len(fake_fk.created) == 2
        assert np.allclose(fake_fk.created[0]["left"], [0.0, 0.0, 0.1])

    def test_same_offset_twice_is_built_once(self, fake_fk):
        offset = {"left": [0.0, 0.0, 0.1], "right": [0.0, 0.0, 0.2]}
        config = {"skills": {"a": {"wrist_tool_offset": offset}, "b": {"wrist_tool_offset": dict(offset)}}}
        factory = assembly.FkFactory()
        assert factory.for_skill(config, "a") is factory.for_skill(config, "b")
        assert len(fake_fk.created) == 1

    def test_missing_urdf_disables_fk(self, tmp_path):
        factory = assembly.FkFactory(urdf_path=tmp_path / "absent.urdf")
        assert factory.urdf_path is None
        assert factory.for_skill({"skills": {"a": {}}}, "a") is None

    def test_malformed_offset_is_rejected(self, fake_fk):
        config = {"skills": {"a": {"wrist_tool_offset": {"left": [0.0, 0.1], "right": [0.0, 0.0, 0.2]}}}}
        with pytest.raises(ValueError, match="length 3"):
            assembly.FkFactory().for_skill(config, "a")


class TestBuildVlaSkill:
    def _build(self, *, variant=None, skill_config=None, fake_fk=None):
        return assembly.build_vla_skill(
            skill_name="demo",
            vla_skill_cls=_StubVlaSkill,
            variant=variant or _variant(),
            skill_config=skill_config or _skill_config(),
            waist_actuator="waist",
            hand_actuator="hand",
            fk_factory=assembly.FkFactory(),
        )

    def test_passes_prompt_and_skill_id_from_the_slot(self, fake_fk):
        built = self._build(
            variant=_variant(language_prompt="rotate table base", skill_id=0)
        )
        assert built.skill.kwargs["language_override"] == "rotate table base"
        assert built.skill.kwargs["skill_id_override"] == 0

    def test_slot_without_overrides_leaves_the_class_values(self, fake_fk):
        built = self._build()
        assert built.skill.kwargs["language_override"] is None
        assert built.skill.kwargs["skill_id_override"] is None

    def test_policy_is_loaded_and_returned(self, fake_fk):
        built = self._build()
        assert built.policy.calls == ["from_ckpt", "warmup(1)", "reset"]
        assert built.skill.kwargs["policy"] is built.policy

    def test_retry_block_attaches_the_controller_and_the_z_ceiling(self, fake_fk):
        """#137 の空振り retry と左手先の z 天井が両経路に付くこと (review 指摘 11)。

        共通化で渡し忘れていたので、`retry:` の節を外しても何も起きない状態だった。
        """
        from inference.desktop.lower_policy.initial_pose import ARM_JOINT_ORDER

        arm = {name: 0.1 for name in ARM_JOINT_ORDER}
        config = _skill_config(
            retry={"rungs_mm": [None, 122.8, 112.1], "attempt_seconds": 3.0},
            initial_pose={
                "arm_position_rad": arm,
                "dex1_opening_fraction": {"left": 1.0, "right": 1.0},
            },
        )
        built = self._build(skill_config=config, fake_fk=fake_fk)
        kwargs = built.skill.kwargs
        assert kwargs["retry_controller"] is not None
        assert kwargs["z_ceiling"] is not None
        assert kwargs["retry_options"]["initial_arm_14"] == [0.1] * 14
        # 一番深い段 (p5 = 112.1mm) が天井の下限になる
        assert kwargs["z_ceiling"].hard_floor_m == pytest.approx(0.1121)

    def test_without_a_retry_block_nothing_is_attached(self, fake_fk):
        kwargs = self._build(fake_fk=fake_fk).skill.kwargs
        assert kwargs["retry_controller"] is None
        assert kwargs["z_ceiling"] is None
        assert kwargs["retry_options"] == {}

    def test_retry_without_a_rung_is_rejected(self, fake_fk):
        config = _skill_config(retry={"rungs_mm": [None]})
        with pytest.raises(ValueError, match="天井が無い"):
            self._build(skill_config=config, fake_fk=fake_fk)

    def test_motion_limiter_holds_targets_outside_the_urdf_limits(self, fake_fk):
        """評価経路にしか入っていなかった位置の限界 (INF-1) が両経路で付く。"""
        limiter = self._build(fake_fk=fake_fk).skill.kwargs["motion_limiter"]
        measured = np.zeros(19)
        limiter.apply(target=np.zeros(19), measured=measured)
        assert not limiter.last_target_held
        limiter.apply(target=np.full(19, 100.0), measured=measured)
        assert limiter.last_target_held

    def test_teacher_range_is_attached_when_configured(self, fake_fk):
        from inference.desktop.lower_policy.pose_utils import JOINT_NAMES

        config = _skill_config(
            teacher_joint_range={
                "margin_rad": 0.02,
                "arm_position_rad": {name: [-1.0, 1.0] for name in JOINT_NAMES},
            }
        )
        built = self._build(skill_config=config)
        assert built.skill.kwargs["teacher_range"] is not None

    def test_teacher_range_is_absent_without_config(self, fake_fk):
        assert self._build(fake_fk=fake_fk).skill.kwargs["teacher_range"] is None

    def test_dispatch_waist_reaches_both_the_skill_and_the_result(self, fake_fk):
        built = self._build(skill_config=_skill_config(dispatch_waist=True))
        assert built.dispatch_waist is True
        assert built.skill.kwargs["dispatch_waist"] is True


class TestSensorSources:
    def test_unknown_joint_source_is_rejected(self):
        with pytest.raises(ValueError, match="lowstate"):
            assembly.build_sensor_sources(head_topic="/head", joint_source="ros")

    def test_boundary_without_synthetic_hand_never_touches_dds(self, monkeypatch):
        """公式 boundary に Dex1 state は無い。rt/dex1 を購読せず、source も作らずに落ちる。"""
        import inference.desktop.perception.boundary_state_source as bss
        import inference.desktop.perception.dex1_state_source as dss

        def _forbidden(*_args, **_kwargs):
            raise AssertionError("must not be constructed")

        monkeypatch.setattr(bss, "BoundaryJointStateSource", _forbidden)
        monkeypatch.setattr(dss, "Dex1StateSource", _forbidden)
        with pytest.raises(ValueError, match="synthetic_hand_initial_rad"):
            assembly.build_sensor_sources(
                head_topic="/head",
                joint_source="boundary",
                head_source="zmq",
                zmq_endpoint="tcp://orin:5555",
                boundary_state_host="orin",
            )

    def test_close_closes_every_source(self):
        class _Closable:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        sources = [_Closable() for _ in range(4)]
        bundle = assembly.SensorSources(
            joint=sources[0], dex1=sources[1], head=sources[2],
            wrist_left=sources[3], wrist_right=None,
        )
        bundle.close()
        assert all(s.closed for s in sources)


class _Frame:
    def __init__(self, *, width: int = 8, t: int = 5, age_s: float = 0.0) -> None:
        import time as _time

        self.rgb = np.zeros((4, width, 3), dtype=np.uint8)
        self.rgb[:, : width // 2] = 11
        self.rgb[:, width // 2 :] = 99
        self.t = t
        self.received_monotonic_ns = _time.monotonic_ns() - int(age_s * 1e9)


class _Source:
    def __init__(self, value) -> None:
        self.value = value

    def get(self):
        return self.value


class TestBuildObservation:
    """観測の組み立ては評価と本番で 1 つ (Issue #141 D3)。"""

    def test_host_clock_is_used_for_the_tick_time(self):
        obs = assembly.build_observation(head=_Frame(t=42))
        assert obs["t"] > 1_000_000          # monotonic ns。header.stamp (42) ではない
        assert obs["_camera_generations"]["head_left"] == 42

    def test_head_is_kept_packed_and_split_only_for_the_camera_record(self):
        frame = _Frame(width=8)
        obs = assembly.build_observation(head=frame)
        assert obs["head_rgb"].shape == (4, 8, 3)
        assert obs["_camera_frames"]["head_left"].shape == (4, 4, 3)
        assert int(obs["_camera_frames"]["head_left"][0, 0, 0]) == 11
        assert int(obs["_camera_frames"]["head_right"][0, 0, 0]) == 99

    def test_sources_that_are_absent_add_no_field(self):
        obs = assembly.build_observation(head=_Frame())
        assert "joint_state" not in obs and "wrist_left_rgb" not in obs

    def test_source_returning_none_still_adds_the_field(self):
        obs = assembly.build_observation(
            head=_Frame(), joint_state_source=_Source(None), wrist_left_source=_Source(None)
        )
        assert obs["joint_state"] is None
        assert obs["wrist_left_rgb"] is None
        assert "left_wrist" not in obs["_camera_received_monotonic_ns"]

    def test_detections_are_passed_through_as_cleaned(self):
        assert assembly.build_observation(head=_Frame(), detections=[])["cleaned"] == []
        assert assembly.build_observation(head=_Frame())["cleaned"] is None

    def test_missing_head_leaves_no_camera_entry(self):
        obs = assembly.build_observation(head=None)
        assert "head_rgb" not in obs
        assert obs["_camera_received_monotonic_ns"] == {}


class TestCameraStaleRoles:
    def test_fresh_cameras_are_not_reported(self):
        obs = assembly.build_observation(
            head=_Frame(), wrist_left_source=_Source(_Frame()),
            wrist_right_source=_Source(_Frame()),
        )
        assert assembly.camera_stale_roles(obs, require_wrist=True) == {}

    def test_old_receive_time_is_reported_with_its_age(self):
        obs = assembly.build_observation(head=_Frame(age_s=2.0))
        stale = assembly.camera_stale_roles(obs, require_wrist=False)
        assert set(stale) == {"head_left", "head_right"}
        assert stale["head_left"] == pytest.approx(2.0, abs=0.2)

    def test_a_camera_that_never_arrived_is_infinitely_old(self):
        obs = assembly.build_observation(head=_Frame())
        stale = assembly.camera_stale_roles(obs, require_wrist=True)
        assert stale["left_wrist"] == float("inf")

    def test_wrist_is_ignored_when_not_required(self):
        obs = assembly.build_observation(head=_Frame())
        assert assembly.camera_stale_roles(obs, require_wrist=False) == {}


class TestSelectHeadView:
    def test_packed_returns_the_whole_frame(self):
        rgb = np.zeros((4, 8, 3), dtype=np.uint8)
        assert assembly.select_head_view(rgb, "packed") is rgb

    def test_left_and_right_split_the_width(self):
        rgb = np.zeros((4, 8, 3), dtype=np.uint8)
        rgb[:, 4:] = 7
        assert int(assembly.select_head_view(rgb, "left")[0, 0, 0]) == 0
        assert int(assembly.select_head_view(rgb, "right")[0, 0, 0]) == 7

    def test_odd_width_is_rejected(self):
        with pytest.raises(ValueError, match="even width"):
            assembly.select_head_view(np.zeros((4, 7, 3), dtype=np.uint8), "left")

    def test_unknown_view_is_rejected(self):
        with pytest.raises(ValueError, match="head view"):
            assembly.select_head_view(np.zeros((4, 8, 3), dtype=np.uint8), "middle")


SKILL_CONFIG_PATH = "inference/desktop/lower_policy/configs/skill_config.yaml"


class TestBuildHeadProcedure:
    def _config(self) -> dict:
        with open(SKILL_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _pose(self, *, requires_hand_init: bool):
        from inference.desktop.lower_policy.initial_pose import initial_pose_from_config

        pose = initial_pose_from_config(self._config(), "rotate_table_base")
        return replace(pose, requires_separate_hand_initialization=requires_hand_init)

    def _names(self, **kwargs) -> list[str]:
        from inference.desktop.lower_policy.actuators.hand import MockHandActuator

        skills = assembly.build_head_procedure(
            skill_config=self._config(),
            skill_name="rotate_table_base",
            initial_pose=kwargs.pop("initial_pose", self._pose(requires_hand_init=False)),
            hand_actuator=MockHandActuator(),
            hold_sec=kwargs.pop("hold_sec", 0.0),
            include_hand=kwargs.pop("include_hand", True),
        )
        return [s.name for s in skills]

    def test_order_is_open_then_arm_then_width_then_hold(self):
        assert self._names() == [
            "wait_for_go_live_rotate_table_base",
            "hand_open_rotate_table_base",
            "arm_pre_motion_for_rotate_table_base",
            "hand_pose_rotate_table_base",
            "hold_pose_for_rotate_table_base",
        ]

    def test_object_in_hand_skill_ends_with_the_grasp_move(self):
        names = self._names(initial_pose=self._pose(requires_hand_init=True))
        assert names[3] == "hand_grasp_rotate_table_base"

    def test_without_a_real_hand_only_the_arm_and_hold_remain(self):
        assert self._names(include_hand=False) == [
            "wait_for_go_live_rotate_table_base",
            "arm_pre_motion_for_rotate_table_base",
            "hold_pose_for_rotate_table_base",
        ]

    def test_model_boundary_preserves_hand_until_arm_path_finishes(self):
        from inference.desktop.lower_policy.actuators.hand import MockHandActuator
        from inference.desktop.lower_policy.initial_pose import initial_pose_from_config

        config = self._config()
        skills = assembly.build_model_transition_procedure(
            skill_config=config,
            previous_skill="rotate_table_base",
            next_skill="pick_table_leg",
            initial_pose=initial_pose_from_config(config, "pick_table_leg"),
            hand_actuator=MockHandActuator(),
            hold_sec=0.0,
            include_hand=True,
        )
        assert [skill.name for skill in skills] == [
            "hand_release_rotate_table_base_to_pick_table_leg",
            "arm_transition_rotate_table_base_to_pick_table_leg",
            "hand_transition_rotate_table_base_to_pick_table_leg",
            "hold_transition_rotate_table_base_to_pick_table_leg",
        ]
        assert not any("open" in skill.name for skill in skills)
        assert skills[0]._release

    def test_pick_to_insert_carries_the_leg_without_releasing(self):
        """pick -> insert だけは脚を持ったまま運ぶ (Issue #159 T3)。"""
        from inference.desktop.lower_policy.actuators.hand import MockHandActuator
        from inference.desktop.lower_policy.initial_pose import initial_pose_from_config

        config = self._config()
        skills = assembly.build_model_transition_procedure(
            skill_config=config,
            previous_skill="pick_table_leg",
            next_skill="insert_table_leg",
            initial_pose=initial_pose_from_config(config, "insert_table_leg"),
            hand_actuator=MockHandActuator(),
            hold_sec=0.0,
            include_hand=True,
        )
        assert [skill.name for skill in skills] == [
            "arm_transition_pick_table_leg_to_insert_table_leg",
            "hand_transition_pick_table_leg_to_insert_table_leg",
            "hold_transition_pick_table_leg_to_insert_table_leg",
        ]

    def test_hand_and_hold_steps_keep_the_last_command_not_the_measurement(self):
        """手の段と保持の段は最後に送った腕 target を保持する (Issue #172)。

        実測を保持すると、重力で遅れて止まる実測が次の目標になり、段ごとに腕が下がって
        開始待ちの到達判定 (0.10 rad) を満たせなくなる。
        """
        from types import SimpleNamespace

        from inference.desktop.lower_policy.actuators.hand import MockHandActuator
        from inference.desktop.lower_policy.initial_pose import initial_pose_from_config
        from inference.desktop.lower_policy.pose_utils import ROS_ARM_JOINT_NAMES

        commanded = np.full(14, 0.25)
        sagged = commanded - 0.06
        names = tuple(f"unused_{i}" for i in range(15)) + tuple(ROS_ARM_JOINT_NAMES)
        obs = {
            "joint_state": SimpleNamespace(
                name=names, position=tuple(np.zeros(15)) + tuple(sagged)
            ),
            "hand_state": None,
        }
        config = self._config()
        head = assembly.build_head_procedure(
            skill_config=config,
            skill_name="rotate_table_base",
            initial_pose=self._pose(requires_hand_init=False),
            hand_actuator=MockHandActuator(),
            hold_sec=1.0,
            include_hand=True,
            published_arm_target_provider=lambda: commanded,
        )
        transition = assembly.build_model_transition_procedure(
            skill_config=config,
            previous_skill="rotate_table_base",
            next_skill="pick_table_leg",
            initial_pose=initial_pose_from_config(config, "pick_table_leg"),
            hand_actuator=MockHandActuator(),
            hold_sec=1.0,
            include_hand=True,
            published_arm_target_provider=lambda: commanded,
        )
        holds = [
            skill for skill in [*head, *transition]
            if skill.name.startswith(("hand_", "hold_"))
        ]
        assert [s.name for s in holds] == [
            "hand_open_rotate_table_base",
            "hand_pose_rotate_table_base",
            "hold_pose_for_rotate_table_base",
            "hand_release_rotate_table_base_to_pick_table_leg",
            "hand_transition_rotate_table_base_to_pick_table_leg",
            "hold_transition_rotate_table_base_to_pick_table_leg",
        ]
        for skill in holds:
            assert skill._hold_arm_target_provider is not None, skill.name
            np.testing.assert_allclose(skill._hold_arm_target_provider(), commanded)
        hold = holds[2]
        hold.start({})
        np.testing.assert_allclose(hold.step(obs), commanded)


class TestHoldSecFromConfig:
    """頭の手順・境界の最後に保持する秒数は YAML が正本 (Issue #159 T9、B4a-14)。

    以前は自前経路の CLI (--setup-hold-sec、既定 3.0) にしか無く、大会経路の driver
    には値の出どころが無かった。
    """

    def test_the_shipped_value_is_three_seconds(self):
        import yaml

        with open(
            "inference/desktop/lower_policy/configs/skill_config.yaml", encoding="utf-8"
        ) as f:
            config = yaml.safe_load(f)
        assert assembly.hold_sec_from_config(config) == pytest.approx(3.0)

    def test_a_negative_hold_is_rejected(self):
        with pytest.raises(ValueError, match="hold_sec"):
            assembly.hold_sec_from_config({"hold_pose": {"hold_sec": -1.0}})

    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValueError, match="unknown"):
            assembly.hold_sec_from_config({"hold_pose": {"hold_sec": 3.0, "typo": 1}})


class TestTeacherRangeWithinTheLimiter:
    """教師の範囲は MotionLimiter の範囲 (URDF − 0.03) と交わらせる (Issue #159 B4b-01)。

    教師が URDF の限界まで捻った関節は、教師の端 = 限界そのもの (手首 ±1.6144 等)。
    そのままだと (URDF − 0.03, 教師の端] の帯で teacher_range は補正せず MotionLimiter が
    拒否し、範囲を出ても引き戻し先 (端 − 0.02) がまた拒否される。拒否されると上半身の
    19D が丸ごと前の指令で止まる (rotate_leg は教師の 10% の episode がこの帯に入る)。
    """

    @staticmethod
    def _config() -> dict:
        import yaml

        with open(
            "inference/desktop/lower_policy/configs/skill_config.yaml", encoding="utf-8"
        ) as f:
            return yaml.safe_load(f)

    def test_every_shipped_pull_target_is_accepted_by_the_limiter(self):
        from inference.desktop.lower_policy.skills.teacher_range import (
            load_teacher_joint_range_for_skill,
        )

        config = self._config()
        limiter = assembly._build_motion_limiter(
            config, "rotate_leg_to_tighten", dispatch_waist=True
        )
        lower, upper = limiter._lower[3:17], limiter._upper[3:17]
        checked = 0
        for name in config["skills"]:
            loaded = load_teacher_joint_range_for_skill(config, name)
            if loaded is None:
                continue
            narrowed = assembly.teacher_range_within_limits(loaded, name)
            assert np.all(narrowed._pull_high <= upper + 1e-12), name
            assert np.all(narrowed._pull_low >= lower - 1e-12), name
            checked += 1
        assert checked >= 4

    def test_a_wrist_at_the_teacher_limit_does_not_freeze_the_upper_body(self):
        """手首 yaw 1.60 (教師の範囲内、MotionLimiter の外) で 19D を保持しない。"""
        from inference.desktop.lower_policy.skills.teacher_range import (
            load_teacher_joint_range_for_skill,
        )

        config = self._config()
        name = "rotate_leg_to_tighten"
        teacher = assembly.teacher_range_within_limits(
            load_teacher_joint_range_for_skill(config, name), name
        )
        limiter = assembly._build_motion_limiter(config, name, dispatch_waist=True)
        measured = np.zeros(19)
        measured[17:19] = 3.0
        target = measured.copy()
        target[16] = 1.60  # R.wrist_yaw (19D の 3 + 13)
        corrected = teacher.apply(target).target_19d
        limiter.apply(target=corrected, measured=measured)
        assert limiter.last_target_held is False
