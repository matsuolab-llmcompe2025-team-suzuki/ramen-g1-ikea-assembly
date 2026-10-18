"""腕の遅れ・押し負けの補い (先読み・押している間の送る間隔、Issue #188)。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import yaml

from inference.desktop.lower_policy.actuators.arm_tracking_assist import (
    ARM_TUNING_PRESETS,
    ArmTrackingAssist,
    ArmTrackingConfig,
)
from inference.desktop.lower_policy.actuators.boundary_sink import (
    BODY_DOF,
    BoundaryActionSink,
)
from inference.desktop.lower_policy.actuators.g1_arm_sdk import (
    G1_ARM_POSITION_UPPER_RAD,
)

SKILL_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "skill_config.yaml"


def _skill_config() -> dict:
    return yaml.safe_load(SKILL_CONFIG.read_text())


def _config(**overrides) -> ArmTrackingConfig:
    values = dict(
        lead_horizon_s=0.15,
        lead_window_s=0.2,
        lead_max_rad=0.10,
        blocked_period_s=0.3,
        blocked_gap_rad=0.08,
        blocked_still_rad_s=0.05,
        push_gain=0.0,
        push_max_rad=0.0,
    )
    values.update(overrides)
    return ArmTrackingConfig(**values)


class TestConfig:
    def test_yaml_has_every_preset_and_the_default_is_standard(self):
        cfg = _skill_config()
        config, name = ArmTrackingConfig.from_config(cfg)
        assert name == "standard"
        # 既定は押している間の間隔と押し込みだけ (先読みは pick のずれを増やしたので入れない)
        assert not config.uses_lead
        assert config.blocked_period_s == pytest.approx(0.3)
        assert config.push_gain == pytest.approx(0.7)
        assert config.push_max_rad == pytest.approx(0.10)
        for preset in ARM_TUNING_PRESETS:
            ArmTrackingConfig.from_config(cfg, preset)
        lead, _ = ArmTrackingConfig.from_config(cfg, "lead")
        assert lead.uses_lead and lead.uses_push and lead.uses_blocked_period

    def test_today_sends_as_on_09_29(self):
        config, _ = ArmTrackingConfig.from_config(_skill_config(), "today")
        assert not config.uses_lead
        assert not config.uses_blocked_period
        assert not config.uses_push

    def test_cli_overrides_replace_only_their_value(self):
        config, name = ArmTrackingConfig.from_config(
            _skill_config(),
            "lead",
            lead_horizon_s=0.1,
            blocked_period_s=0.0,
            push_max_rad=0.05,
        )
        assert name == "lead"
        assert config.lead_horizon_s == pytest.approx(0.1)
        assert config.blocked_period_s == 0.0
        assert config.push_max_rad == pytest.approx(0.05)
        assert config.lead_max_rad == pytest.approx(0.10)

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda s: s.pop("presets"), "presets"),
            (lambda s: s.update(extra=1), "unknown keys"),
            (lambda s: s["presets"].pop("lead"), "missing"),
            (
                lambda s: s["presets"]["standard"].update(lead_horizon_s=0.6),
                "lead_horizon_s",
            ),
            (
                lambda s: s["presets"]["standard"].update(blocked_period_s=1.5),
                "blocked_period_s",
            ),
            (lambda s: s["presets"]["standard"].pop("lead_max_rad"), "missing/unknown"),
            (lambda s: s["presets"]["standard"].update(typo=1), "missing/unknown"),
        ],
    )
    def test_bad_yaml_is_rejected(self, mutate, match):
        cfg = _skill_config()
        mutate(cfg["boundary_arm_tracking"])
        with pytest.raises(ValueError, match=match):
            ArmTrackingConfig.from_config(cfg)

    def test_unknown_preset_is_rejected(self):
        with pytest.raises(ValueError, match="unknown arm tracking preset"):
            ArmTrackingConfig.from_config(_skill_config(), "turbo")


def _arms(value: float = 0.2) -> np.ndarray:
    return np.full(14, value, dtype=np.float64)


class TestLead:
    def test_constant_speed_leads_by_speed_times_horizon(self):
        assist = ArmTrackingAssist(_config())
        speed = 0.4  # rad/s (会場の指令の p95 は 0.5〜0.6)
        led = None
        for k in range(16):
            t = k / 30.0
            led = assist.lead(_arms(0.2 + speed * t), t)
        t = 15 / 30.0
        np.testing.assert_allclose(
            led - _arms(0.2 + speed * t), speed * 0.15, atol=1e-6
        )

    def test_lead_is_capped_per_joint(self):
        assist = ArmTrackingAssist(_config(lead_max_rad=0.05))
        for k in range(16):
            led = assist.lead(_arms(0.2 + 2.0 * k / 30.0), k / 30.0)
        np.testing.assert_allclose(led - _arms(0.2 + 2.0 * 15 / 30.0), 0.05, atol=1e-9)

    def test_still_intent_gets_no_lead(self):
        assist = ArmTrackingAssist(_config())
        for k in range(10):
            led = assist.lead(_arms(), k / 30.0)
        np.testing.assert_array_equal(led, _arms())

    def test_a_gap_in_commands_forgets_the_old_speed(self):
        assist = ArmTrackingAssist(_config())
        for k in range(10):
            assist.lead(_arms(0.1 * k), k / 30.0)
        # 0.3 s を超えて指令が来なかった (skill の切り替え) → 最初の数 tick は先読みしない
        led = assist.lead(_arms(0.5), 2.0)
        np.testing.assert_array_equal(led, _arms(0.5))

    def test_a_sudden_reversal_does_not_flip_the_lead_in_one_tick(self):
        assist = ArmTrackingAssist(_config())
        for k in range(10):
            assist.lead(_arms(0.2 + 0.5 * k / 30.0), k / 30.0)
        top = 0.2 + 0.5 * 9 / 30.0
        led = assist.lead(_arms(top - 0.5 / 30.0), 10 / 30.0)  # 1 tick だけ逆向き
        assert np.all(led - _arms(top - 0.5 / 30.0) > 0.0)  # ならした速さはまだ前向き

    def test_the_lead_stops_at_the_urdf_limit(self):
        assist = ArmTrackingAssist(_config())
        upper = np.asarray(G1_ARM_POSITION_UPPER_RAD, dtype=np.float64)
        for k in range(16):
            led = assist.lead(upper - 0.02 + 0.3 * k / 30.0 - 0.3 * 15 / 30.0, k / 30.0)
        assert np.all(led <= upper - 0.005 + 1e-12)

    def test_today_is_the_identity(self):
        config, _ = ArmTrackingConfig.from_config(_skill_config(), "today")
        assist = ArmTrackingAssist(config, name="today")
        for k in range(10):
            intent = _arms(0.1 + 0.3 * k / 30.0)
            np.testing.assert_array_equal(assist.lead(intent, k / 30.0), intent)


class TestBlocked:
    def test_a_stuck_joint_far_from_its_target_is_pushing(self):
        assist = ArmTrackingAssist(_config())
        sent = _arms(0.3)
        measured = sent.copy()
        measured[10] -= 0.15  # 右肘が押し返されて止まっている (09-29 の会場)
        for k in range(10):
            blocked = assist.update_blocked(sent, measured, k / 30.0)
        assert blocked
        assert assist.publish_period(0.1) == pytest.approx(0.3)

    def test_a_joint_still_catching_up_is_not_pushing(self):
        assist = ArmTrackingAssist(_config())
        sent = _arms(0.3)
        for k in range(10):
            measured = _arms(
                0.3 - 0.15 + 0.6 * k / 30.0
            )  # 0.6 rad/s で追いついている途中
            blocked = assist.update_blocked(sent, measured, k / 30.0)
        assert not blocked
        assert assist.publish_period(0.1) == pytest.approx(0.1)

    def test_small_gaps_are_not_pushing(self):
        assist = ArmTrackingAssist(_config())
        for k in range(10):
            blocked = assist.update_blocked(_arms(0.3), _arms(0.25), k / 30.0)
        assert not blocked

    def test_stale_state_keeps_the_last_decision(self):
        assist = ArmTrackingAssist(_config())
        sent, measured = _arms(0.3), _arms(0.1)
        for k in range(10):
            assist.update_blocked(sent, measured, k / 30.0)
        assert assist.update_blocked(sent, _arms(0.3), 0.5, stale=True)

    def test_nothing_sent_yet_is_not_pushing(self):
        assist = ArmTrackingAssist(_config())
        assert not assist.update_blocked(None, _arms(), 0.0)

    def test_reset_forgets_everything(self):
        assist = ArmTrackingAssist(_config())
        for k in range(10):
            assist.update_blocked(_arms(0.3), _arms(0.1), k / 30.0)
        assist.reset()
        assert not assist.blocked
        assert assist.publish_period(0.1) == pytest.approx(0.1)


def _push_config(**overrides) -> ArmTrackingConfig:
    values = dict(lead_horizon_s=0.0, lead_max_rad=0.0, push_gain=0.7, push_max_rad=0.1)
    values.update(overrides)
    return _config(**values)


def _pushed(sent, measured, intent, steps, dt=1 / 30.0, assist=None):
    assist = assist or ArmTrackingAssist(_push_config())
    for k in range(steps):
        assist.update_blocked(sent, measured, k * dt, intent14=intent)
    return assist


class TestPush:
    def test_a_stuck_joint_is_pushed_toward_the_intent(self):
        intent = _arms(0.3)
        measured = intent.copy()
        measured[10] = 0.45  # 右肘が指令より 0.15 伸びたまま (09-29 の会場の向き)
        assist = _pushed(intent, measured, intent, steps=60)
        push = assist.push
        # 0.7 x (0.3 - 0.45) = -0.105 → 上限 0.1 で -0.1。ほかの関節は 0
        assert push[10] == pytest.approx(-0.1, abs=1e-9)
        assert np.count_nonzero(push) == 1
        led = assist.lead(intent, 2.0)
        assert led[10] == pytest.approx(0.2, abs=1e-9)

    def test_the_push_grows_no_faster_than_its_rate_limit(self):
        intent = _arms(0.3)
        measured = intent.copy()
        measured[10] = 0.45
        assist = _pushed(intent, measured, intent, steps=4)  # 0.1 s
        assert abs(assist.push[10]) <= 0.5 * 0.1 + 1e-9

    def test_the_push_fades_once_the_joint_moves_again(self):
        intent = _arms(0.3)
        measured = intent.copy()
        measured[10] = 0.45
        assist = _pushed(intent, measured, intent, steps=60)
        # 台が動いた: 右肘が 1 rad/s で指令へ戻る → 押していない → 0 へ戻る
        t0 = 60 / 30.0
        for k in range(1, 30):
            m = measured.copy()
            m[10] = max(0.3, 0.45 - k / 30.0)
            assist.update_blocked(intent, m, t0 + k / 30.0, intent14=intent)
        assert assist.push[10] == pytest.approx(0.0, abs=1e-9)

    def test_no_push_without_the_intent_or_when_disabled(self):
        intent = _arms(0.3)
        measured = intent.copy()
        measured[10] = 0.45
        assist = ArmTrackingAssist(_push_config())
        for k in range(60):
            assist.update_blocked(intent, measured, k / 30.0)
        assert not assist.push.any()
        disabled = _pushed(
            intent, measured, intent, 60, assist=ArmTrackingAssist(_config())
        )
        assert not disabled.push.any()

    def test_bad_push_values_are_rejected(self):
        with pytest.raises(ValueError, match="push_gain"):
            _push_config(push_gain=3.0)
        with pytest.raises(ValueError, match="blocked_gap_rad"):
            _push_config(blocked_gap_rad=0.0, blocked_period_s=0.0)


# ---------------------------------------------------------------- sink との組み合わせ


class _FakeJointSink:
    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []

    def send_chunk(self, actions: np.ndarray, issued_at=None) -> None:
        self.chunks.append(np.asarray(actions))

    def close(self) -> None:
        pass


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _joint_sink(assist, clock, log=None):
    sink = BoundaryActionSink.__new__(BoundaryActionSink)
    fake = _FakeJointSink()
    sink._lane = "joint"
    sink._clamp_wrist_roll = False
    sink._fk = None
    sink._ee_frame_transform = None
    sink._log_fn = log
    sink._sender_clock_offset_fn = None
    sink._sink = fake
    sink._sent = 0
    sink._ee_waist = None
    sink._clamped = 0
    sink._publish_period_s = 0.1
    sink._chunk_rows = 16
    sink._hold_refresh_s = 1.0
    sink._clock = clock
    sink._last_row = None
    sink._last_publish_at = None
    sink._gravity_offset = None
    sink._arm_tracking = assist
    sink._last_expected_arms = None
    sink._preparation_owner = None
    return sink, fake


def _action(arms14) -> np.ndarray:
    action = np.zeros(19)
    action[3:17] = arms14
    action[17:19] = (4.5, 4.5)
    return action


def _body(arms14) -> np.ndarray:
    body = np.zeros(BODY_DOF)
    body[15:29] = arms14
    return body


def _run(sink, clock, intents, measured_fn, dt=1 / 30.0):
    for k, intent in enumerate(intents):
        clock.t = k * dt
        sink.send_action(_action(intent), _body(measured_fn(k, intent)))


class TestSinkWithArmTracking:
    @pytest.mark.requires_boundary
    def test_today_publishes_exactly_as_without_the_assist(self):
        intents = [_arms(0.2 + 0.4 * k / 30.0) for k in range(30)]
        clock_a, clock_b = _Clock(), _Clock()
        config, _ = ArmTrackingConfig.from_config(_skill_config(), "today")
        with_assist, fake_a = _joint_sink(
            ArmTrackingAssist(config, name="today"), clock_a
        )
        without, fake_b = _joint_sink(None, clock_b)
        _run(with_assist, clock_a, intents, lambda k, i: i)
        _run(without, clock_b, intents, lambda k, i: i)
        assert len(fake_a.chunks) == len(fake_b.chunks)
        for a, b in zip(fake_a.chunks, fake_b.chunks):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.requires_boundary
    def test_the_published_arms_lead_a_moving_intent(self):
        log = []
        clock = _Clock()
        sink, fake = _joint_sink(
            ArmTrackingAssist(_config(), name="standard"), clock, log.append
        )
        intents = [_arms(0.2 + 0.4 * k / 30.0) for k in range(30)]
        _run(sink, clock, intents, lambda k, i: i)
        last = log[-1]
        assert last["arm_tracking_preset"] == "standard"
        np.testing.assert_allclose(last["arm_lead_rad"], 0.4 * 0.15, atol=1e-6)
        np.testing.assert_allclose(
            fake.chunks[-1][0][4:18],
            np.asarray(last["arms_intent"]) + 0.4 * 0.15,
            atol=1e-6,
        )

    @pytest.mark.requires_boundary
    def test_pushing_stretches_the_publish_period(self):
        clock = _Clock()
        sink, fake = _joint_sink(ArmTrackingAssist(_config()), clock)
        intents = [
            _arms(0.3 + 0.01 * k / 30.0) for k in range(60)
        ]  # ゆっくり押し続ける

        # 右肘だけ 0.15 rad 手前で止まっている (台に押し返されている)
        def measured(k, intent):
            m = np.asarray(intent, dtype=np.float64).copy()
            m[10] = 0.15
            return m

        _run(sink, clock, intents, measured)
        # 2 s で 0.1 s おきなら約 20 回、押している間は 0.3 s おきなので 7〜8 回
        assert len(fake.chunks) <= 9

    @pytest.mark.requires_boundary
    def test_the_push_reaches_the_published_arms_and_the_log(self):
        log = []
        clock = _Clock()
        sink, fake = _joint_sink(
            ArmTrackingAssist(_push_config(), name="push"), clock, log.append
        )
        intents = [_arms(0.3) for _ in range(90)]

        def measured(k, intent):
            m = np.asarray(intent, dtype=np.float64).copy()
            m[10] = 0.45  # 右肘が押し返されたまま
            return m

        _run(sink, clock, intents, measured)
        assert log[-1]["arm_push_rad"][10] == pytest.approx(-0.1, abs=1e-6)
        assert fake.chunks[-1][0][4 + 10] == pytest.approx(0.2, abs=1e-6)

    @pytest.mark.requires_boundary
    def test_free_motion_keeps_the_normal_period(self):
        clock = _Clock()
        sink, fake = _joint_sink(ArmTrackingAssist(_config()), clock)
        intents = [_arms(0.2 + 0.4 * k / 30.0) for k in range(60)]
        _run(sink, clock, intents, lambda k, i: i)
        assert len(fake.chunks) >= 18

    def test_the_end_of_a_preparation_resets_the_assist(self):
        clock = _Clock()
        assist = ArmTrackingAssist(_config())
        sink, _ = _joint_sink(assist, clock)
        for k in range(10):
            assist.update_blocked(_arms(0.3), _arms(0.1), k / 30.0)
        owner = object()
        sink.acquire_preparation(owner)
        sink.release_preparation(owner)
        assert not assist.blocked
        assert sink._last_expected_arms is None

    def test_the_pose_lane_refuses_the_assist(self):
        with pytest.raises(ValueError, match="joint lane only"):
            BoundaryActionSink(lane="pose", arm_tracking=ArmTrackingAssist(_config()))

    def test_blocked_period_longer_than_the_hold_refresh_is_refused(self):
        with pytest.raises(ValueError, match="hold_refresh_s"):
            BoundaryActionSink(
                lane="joint",
                publish_period_s=0.1,
                hold_refresh_s=0.2,
                arm_tracking=ArmTrackingAssist(_config(blocked_period_s=0.3)),
            )


# ---------------------------------------------------------------- entrypoint


def _args(**overrides) -> argparse.Namespace:
    values = dict(
        action_sink="boundary",
        boundary_lane="joint",
        arm_tuning=None,
        arm_lead_horizon_s=None,
        arm_blocked_period_s=None,
        boundary_hold_refresh_s=1.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class TestEntrypointBuilder:
    def _build(self, **overrides):
        from inference.desktop import entrypoint

        return entrypoint.build_boundary_arm_tracking(
            _args(**overrides), _skill_config()
        )

    def test_default_is_the_yaml_default_preset(self):
        assist = self._build()
        assert assist.name == "standard"

    def test_today_and_overrides(self):
        assert not self._build(arm_tuning="today").config.uses_lead
        assist = self._build(arm_tuning="lead", arm_lead_horizon_s=0.1)
        assert assist.config.lead_horizon_s == pytest.approx(0.1)

    def test_only_the_boundary_joint_lane_uses_it(self):
        assert self._build(action_sink="sdk") is None
        assert self._build(boundary_lane="pose") is None

    def test_push_override(self):
        assist = self._build(arm_push_max_rad=0.05)
        assert assist.config.push_max_rad == pytest.approx(0.05)

    def test_blocked_period_must_fit_in_the_hold_refresh(self):
        with pytest.raises(ValueError, match="hold-refresh"):
            self._build(arm_blocked_period_s=0.8, boundary_hold_refresh_s=0.5)

    def test_preview_override(self):
        assert self._build().config.preview_steps(30.0) == 6
        assert self._build(arm_preview_s=0.0).config.preview_steps(30.0) == 0
        assert self._build(arm_preview_s=0.3).config.preview_s == pytest.approx(0.3)


class TestPreview:
    """腕だけ policy の予定の少し先を送る (Issue #188)。値を持つのはここ、差し替えは VlaSkill。"""

    def test_standard_and_lead_send_0_2_s_ahead_and_today_does_not(self):
        cfg = _skill_config()
        for preset, expected in (("standard", 0.2), ("lead", 0.2), ("today", 0.0)):
            config, _ = ArmTrackingConfig.from_config(cfg, preset)
            assert config.preview_s == pytest.approx(expected), preset
        standard, _ = ArmTrackingConfig.from_config(cfg)
        assert standard.preview_steps(30.0) == 6
        assert "preview 0.2s" in standard.describe()
        today, _ = ArmTrackingConfig.from_config(cfg, "today")
        assert today.preview_steps(30.0) == 0
        assert "preview off" in today.describe()

    def test_constructing_without_preview_keeps_the_old_behaviour(self):
        assert _config().preview_s == 0.0

    def test_override_and_range(self):
        config, _ = ArmTrackingConfig.from_config(_skill_config(), "today", preview_s=0.1)
        assert config.preview_steps(30.0) == 3
        with pytest.raises(ValueError, match="preview_s"):
            ArmTrackingConfig.from_config(_skill_config(), preview_s=0.6)
        with pytest.raises(ValueError, match="preview_s"):
            ArmTrackingConfig.from_config(_skill_config(), preview_s=-0.1)

    def test_a_preset_without_preview_is_rejected(self):
        cfg = _skill_config()
        cfg["boundary_arm_tracking"]["presets"]["standard"].pop("preview_s")
        with pytest.raises(ValueError, match="missing/unknown"):
            ArmTrackingConfig.from_config(cfg)


class TestApplyArmPreview:
    """entrypoint が学習済みの skill にだけ step 数を配る (Issue #188)。"""

    class _Skill:
        def __init__(self):
            self.steps = None

        def set_arm_preview_steps(self, steps):
            self.steps = steps

    def _registry(self):
        return {
            "rotate_table_base": self._Skill(),
            "pick_table_leg": self._Skill(),
            "setup": self._Skill(),          # 学習済みでない skill には配らない
            "flip_table": object(),          # 口の無い skill (rule-based など) は飛ばす
        }

    def test_the_venue_default_sets_six_ticks_on_learned_skills(self):
        from inference.desktop import entrypoint

        registry = self._registry()
        assist = entrypoint.build_boundary_arm_tracking(_args(), _skill_config())
        assert entrypoint.apply_arm_preview(registry, assist, 30.0) == 6
        assert registry["rotate_table_base"].steps == 6
        assert registry["pick_table_leg"].steps == 6
        assert registry["setup"].steps is None

    def test_no_arm_tracking_means_no_preview(self):
        from inference.desktop import entrypoint

        registry = self._registry()
        assert entrypoint.apply_arm_preview(registry, None, 30.0) == 0
        today = entrypoint.build_boundary_arm_tracking(_args(arm_tuning="today"), _skill_config())
        assert entrypoint.apply_arm_preview(registry, today, 30.0) == 0
        assert registry["rotate_table_base"].steps is None


class TestSinkWithGravityOffset:
    """重力の垂れ補正と一緒に使う (Issue #188 の review)。補正の分は「押している」差に数えない。

    肘の補正は 2.1 倍で開始姿勢 0.12〜0.14 rad あり、blocked_gap_rad (0.08) を超える。送った腕と
    実測の差で判定すると、補正が効いて止まっている肘まで押していることになり、腕の行全体の
    送る間隔が 0.3 s に延びていた。
    """

    @staticmethod
    def _gravity(elbow_rad: float = 0.12):
        from inference.desktop.lower_policy.actuators.boundary_sink import ArmGravitySagOffset

        kp = np.array([100, 100, 40, 40, 20, 20, 20] * 2, dtype=np.float64)
        offset = np.zeros(14)
        offset[[3, 10]] = elbow_rad
        return ArmGravitySagOffset(lambda arms: offset * kp, kp, max_offset_rad=0.18)

    @pytest.mark.requires_boundary
    def test_a_compensated_still_elbow_is_not_pushing(self):
        log = []
        clock = _Clock()
        sink, fake = _joint_sink(ArmTrackingAssist(_push_config()), clock, log.append)
        sink._gravity_offset = self._gravity()
        # 肩だけ動かす。肘は止まっていて、補正が効いて指令どおりの位置にある
        intents = []
        for k in range(60):
            arms = _arms(0.3)
            arms[[0, 7]] = 0.2 + 0.4 * k / 30.0
            intents.append(arms)
        _run(sink, clock, intents, lambda k, i: i)
        assert not any(record["arm_blocked"] for record in log)
        assert len(fake.chunks) >= 18  # 0.1 s おきのまま
        # 送った肘には補正が入っている (補正そのものは変えていない)
        assert fake.chunks[-1][0][4 + 3] == pytest.approx(0.3 + 0.12, abs=1e-6)

    @pytest.mark.requires_boundary
    def test_a_really_pushed_elbow_is_still_found_with_the_offset(self):
        log = []
        clock = _Clock()
        sink, fake = _joint_sink(ArmTrackingAssist(_config()), clock, log.append)
        sink._gravity_offset = self._gravity()
        intents = [_arms(0.3 + 0.01 * k / 30.0) for k in range(60)]

        def measured(k, intent):
            m = np.asarray(intent, dtype=np.float64).copy()
            m[10] = 0.15  # 右肘が押し返されて止まっている
            return m

        _run(sink, clock, intents, measured)
        assert log[-1]["arm_blocked"]
        assert len(fake.chunks) <= 9
