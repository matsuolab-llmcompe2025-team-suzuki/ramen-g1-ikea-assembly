"""BoundaryActionSink / assemble_action38 の unit test。

`boundary` は zmq / cv2 / msgpack を引くので default env には入らない。
`BoundaryActionSink.__init__` を bypass して `_sink` に fake を挿し、
publish する中身 (19-D → 38-D → (T,25)) だけを検証する。
`assemble_action38` は純 numpy なのでそのまま呼べる。
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.actuators.boundary_sink import (
    BoundaryArmActuator,
    BoundaryWalkActuator,
    BODY_DOF,
    BoundaryActionSink,
    assemble_action19,
    assemble_action38,
    waist_joints_to_torso_rpy,
)


def test_boundary_actuators_never_require_the_unitree_sdk():
    sent = []
    arm = BoundaryArmActuator()
    arm.configure(
        sender=lambda value: sent.append(np.asarray(value).copy()),
        state_getter=lambda: np.arange(14, dtype=np.float64),
    )
    arm.start()
    arm.send_action(np.ones(14))
    assert len(sent) == 1
    assert np.array_equal(arm.read_arm_positions(), np.arange(14))

    published = []
    walk = BoundaryWalkActuator()
    walk.set_publish_callback(lambda: published.append(walk.latest.copy()))
    walk.set_velocity(0.1, -0.2, 0.3)
    assert np.allclose(published[-1], [0.1, -0.2, 0.3])


class TestBoundaryArmTargetRecord:
    """「最後に送った target」は publish に成功したものだけ (Codex #1)。"""

    @staticmethod
    def _arm(results):
        sent = []

        def sender(value):
            sent.append(np.asarray(value).copy())
            return results.pop(0)

        arm = BoundaryArmActuator()
        arm.configure(sender=sender, state_getter=lambda: np.zeros(14))
        arm.start()
        return arm, sent

    def test_normal_send_updates_the_record_the_walk_callback_resends(self):
        arm, sent = self._arm([True, True])
        walk = BoundaryWalkActuator()
        walk.set_publish_callback(
            lambda: arm.send_action(arm.read_last_published_targets()[0])
        )
        arm.send_action(np.full(14, 0.3))
        walk.set_velocity(0.0, 0.0, 0.0)
        np.testing.assert_allclose(sent[-1], np.full(14, 0.3))
        np.testing.assert_allclose(arm.read_last_published_targets()[0], 0.3)

    def test_unpublished_target_is_not_recorded(self):
        arm, _ = self._arm([True, False])
        arm.send_action(np.full(14, 0.1))
        arm.send_action(np.full(14, 0.9))  # joint state 未受信で publish されない
        np.testing.assert_allclose(arm.read_last_published_targets()[0], 0.1)

    def test_rejected_target_is_not_recorded(self):
        arm = BoundaryArmActuator()

        def sender(_value):
            raise ValueError("ActionError")

        arm.configure(sender=sender, state_getter=lambda: np.full(14, 0.2))
        arm.start()
        with pytest.raises(ValueError):
            arm.send_action(np.full(14, 0.7))
        np.testing.assert_allclose(arm.read_last_published_targets()[0], 0.2)

    def test_nothing_is_published_before_start(self):
        sent = []
        arm = BoundaryArmActuator()
        arm.configure(
            sender=lambda v: sent.append(v) or True,
            state_getter=lambda: np.zeros(14),
        )
        arm.send_action(np.full(14, 0.4))
        assert sent == []

    def test_late_first_state_seeds_walk_stop_from_measurement(self):
        state = {"arms": None}
        sent = []
        arm = BoundaryArmActuator()
        arm.configure(
            sender=lambda value: sent.append(np.asarray(value).copy()) or True,
            state_getter=lambda: state["arms"],
        )
        arm.send_action(np.full(14, 0.8))  # not physically sent
        with pytest.raises(RuntimeError, match="unavailable"):
            arm.start()
        with pytest.raises(RuntimeError, match="unavailable"):
            arm.read_last_published_targets()
        state["arms"] = np.linspace(-0.5, 0.5, 14)
        arm.start()
        walk = BoundaryWalkActuator()
        walk.set_publish_callback(
            lambda: arm.send_action(arm.read_last_published_targets()[0])
        )
        walk.set_velocity(0.0, 0.0, 0.0)  # startup shutdown path
        np.testing.assert_allclose(sent[-1], state["arms"])

    def test_invalid_first_state_fails_closed(self):
        arm = BoundaryArmActuator()
        arm.configure(sender=lambda _: True, state_getter=lambda: np.full(14, np.nan))
        with pytest.raises(RuntimeError, match="finite measured"):
            arm.start()


def test_entrypoint_routes_every_boundary_publish_through_the_arm_actuator():
    """orchestrator と歩行の再送が _send_to_boundary を直に呼ばないこと。"""
    import inspect

    from inference.desktop import entrypoint

    source = inspect.getsource(entrypoint.main)
    assert "actuator_send_fn = _send_to_boundary" not in source
    assert "lambda: _send_to_boundary(" not in source


class TestAssembleAction38:
    def test_legs_come_from_measurement_and_the_rest_from_the_action(self):
        """脚 12 は実測、腰 3 と腕 14 と手 2 は action。policy は脚を出さない。"""
        action19 = np.arange(19, dtype=np.float64)
        body_q29 = np.arange(100, 100 + BODY_DOF, dtype=np.float64)

        out = assemble_action38(action19, body_q29)

        assert out.shape == (38,)
        np.testing.assert_allclose(out[0:7], [0, 0, 0, 1, 0, 0, 0])  # root = 単位 quat
        np.testing.assert_allclose(out[7:19], body_q29[:12])  # 脚は実測
        np.testing.assert_allclose(out[19:22], action19[0:3])  # 腰
        np.testing.assert_allclose(out[22:36], action19[3:17])  # 腕
        np.testing.assert_allclose(out[36:38], action19[17:19])  # 手

    def test_measured_arms_are_ignored(self):
        """実測の腕ではなく action の腕が入る (指令が観測に上書きされない)。"""
        action19 = np.full(19, 7.0)
        body_q29 = np.full(BODY_DOF, -1.0)

        out = assemble_action38(action19, body_q29)

        np.testing.assert_allclose(out[22:36], 7.0)
        np.testing.assert_allclose(out[7:19], -1.0)

    @pytest.mark.parametrize("bad_len", [18, 20])
    def test_rejects_wrong_action_width(self, bad_len):
        with pytest.raises(ValueError, match=r"action19 must be \(19,\)"):
            assemble_action38(np.zeros(bad_len), np.zeros(BODY_DOF))

    def test_rejects_wrong_body_width(self):
        with pytest.raises(ValueError, match=r"body_q29 must be \(29,\)"):
            assemble_action38(np.zeros(19), np.zeros(28))

    def test_rejects_non_finite_action(self):
        action = np.zeros(19)
        action[5] = np.nan
        with pytest.raises(ValueError, match="action19 must be finite"):
            assemble_action38(action, np.zeros(BODY_DOF))

    def test_rejects_non_finite_measured_legs(self):
        body = np.zeros(BODY_DOF)
        body[3] = np.inf
        with pytest.raises(ValueError, match="legs must be finite"):
            assemble_action38(np.zeros(19), body)

    def test_non_leg_measurements_may_be_non_finite(self):
        """腕・腰の実測は使わないので、そこが壊れていても組み立てられる。"""
        body = np.zeros(BODY_DOF)
        body[20] = np.nan  # 腕側 = 捨てられる
        assemble_action38(np.zeros(19), body)


class _FakeDecoupledSink:
    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []
        self.issued_at: list = []
        self.closed = 0

    def send_chunk(self, actions: np.ndarray, issued_at=None) -> None:
        self.chunks.append(np.asarray(actions))
        self.issued_at.append(issued_at)

    def close(self) -> None:
        self.closed += 1


class _FakeLog:
    """`log_fn` は dict を 1 件受け取る callable。"""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def __call__(self, record: dict) -> None:
        self.records.append(record)


class _FakeFk:
    """`groot_chunk_to_taskspace` が呼ぶ最小の FK 契約。

    実物 (`G1WristFK.compute_ee_transforms`) と同じく
    `(left_pos, left_R, right_pos, right_R)` の 4 値を返す。
    """

    def compute_ee_transforms(self, body_q29):  # noqa: ANN001
        left_pos = np.array([0.1, 0.2, 0.3])
        right_pos = np.array([0.4, 0.5, 0.6])
        return left_pos, np.eye(3), right_pos, np.eye(3)


def _make_sink(log_fn=None) -> tuple[BoundaryActionSink, _FakeDecoupledSink]:
    """__init__ を bypass して DecoupledSink を fake に差し替える。"""
    sink = BoundaryActionSink.__new__(BoundaryActionSink)
    fake = _FakeDecoupledSink()
    sink._lane = "pose"  # type: ignore[attr-defined]
    sink._clamp_wrist_roll = True  # type: ignore[attr-defined]
    sink._fk = _FakeFk()  # type: ignore[attr-defined]
    sink._ee_frame_transform = None  # type: ignore[attr-defined]
    sink._log_fn = log_fn  # type: ignore[attr-defined]
    sink._sender_clock_offset_fn = None  # type: ignore[attr-defined]
    sink._sink = fake  # type: ignore[attr-defined]
    sink._sent = 0  # type: ignore[attr-defined]
    sink._ee_waist = None  # type: ignore[attr-defined]
    sink._clamped = 0  # type: ignore[attr-defined]
    # 送り方は class 既定 (毎回 1 行) = __init__ と同じ
    sink._publish_period_s = 0.0  # type: ignore[attr-defined]
    sink._chunk_rows = 1  # type: ignore[attr-defined]
    sink._hold_refresh_s = 0.0  # type: ignore[attr-defined]
    sink._clock = lambda: 0.0  # type: ignore[attr-defined]
    sink._last_row = None  # type: ignore[attr-defined]
    sink._last_publish_at = None  # type: ignore[attr-defined]
    sink._gravity_offset = None  # type: ignore[attr-defined]
    sink._arm_tracking = None  # type: ignore[attr-defined]
    sink._last_expected_arms = None  # type: ignore[attr-defined]
    sink._preparation_owner = None
    return sink, fake


class TestBoundaryIssuedAtClock:
    """送信時刻を運営 adapter の時計 (PC2) に揃える (Issue #161、C2-01)。"""

    def test_issued_at_is_stamped_on_the_sender_clock(self):
        import time

        sink, fake = _make_sink()
        # PC2 の時計が 5 s 進んでいても、adapter から見て古くならない。
        sink._sender_clock_offset_fn = lambda: 5.0  # type: ignore[attr-defined]
        before = time.time()
        sink.send_action(np.zeros(19), np.zeros(BODY_DOF))
        after = time.time()
        assert before + 5.0 <= fake.issued_at[0] <= after + 5.0

    def test_local_clock_is_kept_until_the_offset_is_known(self):
        sink, fake = _make_sink()
        sink._sender_clock_offset_fn = lambda: None  # type: ignore[attr-defined]
        sink.send_action(np.zeros(19), np.zeros(BODY_DOF))
        # None = DecoupledSink の既定 (この host の time.time())。
        assert fake.issued_at == [None]

    def test_the_log_records_the_stamp_and_the_offset(self):
        log = _FakeLog()
        sink, _ = _make_sink(log_fn=log)
        sink._sender_clock_offset_fn = lambda: -2.0  # type: ignore[attr-defined]
        sink.send_action(np.zeros(19), np.zeros(BODY_DOF))
        record = log.records[-1]
        assert record["sender_clock_offset_s"] == -2.0
        assert record["issued_at"] is not None


class TestBoundaryActionSink:
    def test_publishes_one_taskspace_row_per_action(self):
        sink, fake = _make_sink()

        sink.send_action(np.zeros(19), np.zeros(BODY_DOF))

        assert len(fake.chunks) == 1
        assert fake.chunks[0].shape == (1, 25)
        assert sink.sent_count == 1

    def test_hand_columns_carry_the_commanded_gripper(self):
        """(T,25) の [0:2]/[2:4] が hand。boundary は -1 = 運営の全開 / +1 = 全閉。

        model の値は物理の rad なので、運営の写像で q = -rad になる値を渡す
        (Issue #159 B3a-04)。dataset の最大 4.5 rad は運営の全開 5.30 rad より手前。
        """
        action = np.zeros(19)
        action[17:19] = (4.5, 0.0)  # 左 = dataset の最大開度、右 = 全閉
        sink, fake = _make_sink()

        sink.send_action(action, np.zeros(BODY_DOF))

        row = fake.chunks[0][0]
        np.testing.assert_allclose(row[0:2], 1.0 - 2.0 * 4.5 / 5.30, rtol=1e-6)
        np.testing.assert_allclose(row[2:4], +1.0)  # 0.0 rad → +1 (closed)

    def test_logs_the_raw_taskspace_for_post_hoc_diagnosis(self):
        """WBC_RUNBOOK §5: グリッパが閉じたかは自分の publish 値で分かる。"""
        log = _FakeLog()
        sink, _ = _make_sink(log_fn=log)
        action = np.zeros(19)
        action[17:19] = (0.0, 0.0)

        sink.send_action(action, np.zeros(BODY_DOF))

        assert len(log.records) == 1
        record = log.records[0]
        assert record["event"] == "boundary_taskspace"
        assert record["seq"] == 1
        np.testing.assert_allclose(record["left_hand"], [1.0, 1.0])
        assert len(record["taskspace_25"]) == 25

    def test_works_without_a_log_sink(self):
        sink, fake = _make_sink(log_fn=None)
        sink.send_action(np.zeros(19), np.zeros(BODY_DOF))
        assert len(fake.chunks) == 1

    def test_navigation_columns_follow_the_walk_command(self):
        sink, fake = _make_sink()
        sink.send_action(
            np.zeros(19),
            np.zeros(BODY_DOF),
            navigate_cmd=(0.1, -0.2, 0.3),
        )
        np.testing.assert_allclose(fake.chunks[0][0, 18:21], [0.1, -0.2, 0.3])

    def test_waist_order_is_converted_to_contract_torso_rpy(self):
        sink, fake = _make_sink()
        action = np.zeros(19)
        action[:3] = [0.3, 0.1, -0.2]  # yaw, roll, pitch
        sink.send_action(action, np.zeros(BODY_DOF))
        np.testing.assert_allclose(
            fake.chunks[0][0, 22:25],
            waist_joints_to_torso_rpy(action[:3]),
        )

    def test_compound_waist_rotation_is_not_treated_as_a_plain_reorder(self):
        joints = np.asarray([0.3, 0.4, -0.35])
        rpy = waist_joints_to_torso_rpy(joints)
        assert not np.allclose(rpy, joints[[1, 2, 0]])

        # Reconstruct standard RPY and prove that it is the same orientation
        # as the G1 URDF's yaw -> roll -> pitch serial chain.
        roll, pitch, yaw = rpy
        cr, sr, cp, sp, cy, sy = (
            np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch),
            np.cos(yaw), np.sin(yaw),
        )
        standard = np.asarray(
            (
                (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                (-sp, cp * sr, cp * cr),
            )
        )
        jy, jr, jp = joints
        rz = np.asarray(
            ((np.cos(jy), -np.sin(jy), 0), (np.sin(jy), np.cos(jy), 0), (0, 0, 1))
        )
        rx = np.asarray(
            ((1, 0, 0), (0, np.cos(jr), -np.sin(jr)), (0, np.sin(jr), np.cos(jr)))
        )
        ry = np.asarray(
            ((np.cos(jp), 0, np.sin(jp)), (0, 1, 0), (-np.sin(jp), 0, np.cos(jp)))
        )
        np.testing.assert_allclose(standard, rz @ rx @ ry, atol=1e-12)

    def test_close_is_idempotent(self):
        sink, fake = _make_sink()
        sink.close()
        sink.close()
        assert fake.closed == 1

    def test_malformed_action_is_rejected_before_publishing(self):
        """壊れた action を boundary に流さない (検証はこちらでも止める)。"""
        sink, fake = _make_sink()
        with pytest.raises(ValueError):
            sink.send_action(np.zeros(18), np.zeros(BODY_DOF))
        assert fake.chunks == []
        assert sink.sent_count == 0


class _WaistDependentFk:
    """腰角で EE が動く FK (腰の揺れが publish に出るかを見るため)。"""

    def compute_ee_transforms(self, body_q29):  # noqa: ANN001
        waist = np.asarray(body_q29)[12:15]
        left = np.array([0.3, 0.2, 0.1]) + waist
        right = np.array([0.3, -0.2, 0.1]) + waist
        return left, np.eye(3), right, np.eye(3)


class TestStableTargetsForTheOrganizerIkCache:
    """腕の指令が同じなら、腰の実測が揺れても publish する行は bit 単位で同じ (Issue #164)。

    運営 `ik.py` は目標が前回と同じなら解き直さない。解き直すたびに 7 自由度の余りが
    流れ、模擬の閉ループでは歩行中の保持で肘が 1.06 → 1.25 rad 動いた。
    """

    @staticmethod
    def _sink():
        sink, fake = _make_sink()
        sink._fk = _WaistDependentFk()  # type: ignore[attr-defined]
        return sink, fake

    def test_waist_jitter_below_the_threshold_keeps_the_row_identical(self):
        from inference.desktop.lower_policy.actuators.boundary_sink import (
            EE_WAIST_REFRESH_RAD,
        )

        sink, fake = self._sink()
        rng = np.random.default_rng(0)
        base = np.zeros(19)
        base[:3] = (-0.15, -0.005, 0.12)
        for _ in range(50):
            action = base.copy()
            # 運営機の実測: tick ごとの揺れは 1e-5〜1.5e-4 rad
            action[:3] += rng.uniform(-1.5e-4, 1.5e-4, 3)
            sink.send_action(action, np.zeros(BODY_DOF))
        first = fake.chunks[0].tobytes()
        assert all(chunk.tobytes() == first for chunk in fake.chunks)
        assert EE_WAIST_REFRESH_RAD > 1.5e-4

    def test_a_real_waist_move_is_followed(self):
        from inference.desktop.lower_policy.actuators.boundary_sink import (
            EE_WAIST_REFRESH_RAD,
        )

        sink, fake = self._sink()
        action = np.zeros(19)
        sink.send_action(action, np.zeros(BODY_DOF))
        moved = action.copy()
        moved[0] = 2.0 * EE_WAIST_REFRESH_RAD
        sink.send_action(moved, np.zeros(BODY_DOF))
        np.testing.assert_allclose(
            fake.chunks[1][0, 4:7] - fake.chunks[0][0, 4:7],
            [2.0 * EE_WAIST_REFRESH_RAD, 0.0, 0.0],
            atol=1e-6,
        )

    def test_the_arm_command_still_moves_the_row(self):
        sink, fake = _make_sink()
        sink.send_action(np.zeros(19), np.zeros(BODY_DOF))
        action = np.zeros(19)
        action[17] = 2.0  # 手の指令だけ変える
        sink.send_action(action, np.zeros(BODY_DOF))
        assert fake.chunks[0].tobytes() != fake.chunks[1].tobytes()


def test_default_fk_is_the_organizer_ik_kinematics(monkeypatch):
    """FK を渡さない sink は運営 IK と同じ URDF で EE を作る (Issue #164)。"""
    import sys
    import types

    from inference.desktop.perception.g1_urdf_fk import (
        ORGANIZER_IK_URDF_PATH,
        G1WristFK,
    )

    fake_boundary = types.ModuleType("inference.desktop.boundary")
    fake_boundary.DecoupledSink = lambda port, host: _FakeDecoupledSink()
    monkeypatch.setitem(sys.modules, "inference.desktop.boundary", fake_boundary)

    sink = BoundaryActionSink()
    rng = np.random.default_rng(1)
    action = np.zeros(19)
    action[3:17] = rng.uniform(-0.5, 0.5, 14)
    sink.send_action(action, np.zeros(BODY_DOF))
    row = sink._sink.chunks[0][0]  # type: ignore[attr-defined]

    body = np.zeros(BODY_DOF)
    body[15:29] = action[3:17]
    left, _, right, _ = G1WristFK.from_urdf(ORGANIZER_IK_URDF_PATH).compute_ee_transforms(
        body
    )
    np.testing.assert_allclose(row[4:7], left, atol=1e-6)
    np.testing.assert_allclose(row[11:14], right, atol=1e-6)


def test_pose_sink_defaults_to_the_latest_organizer_wrist_roll_limits(monkeypatch):
    """Latest organizer IK removed the former hard +/-0.9 wrist-roll cap."""
    import sys
    import types

    fake_boundary = types.ModuleType("inference.desktop.boundary")
    fake_boundary.DecoupledSink = lambda port, host: _FakeDecoupledSink()
    monkeypatch.setitem(sys.modules, "inference.desktop.boundary", fake_boundary)

    sink = BoundaryActionSink(fk=_FakeFk())
    assert sink._clamp_wrist_roll is False  # type: ignore[attr-defined]


def test_entrypoint_main_binds_numpy_for_the_boundary_closures():
    """会場の closure が使う ``np`` を main() が import していること (Issue #164)。

    module の先頭では numpy を import していないので、main() の中に無いと
    ``--action-sink boundary --actuate`` が最初の関節読み取りで NameError になる。
    既存の試験は ``--actuate`` 無しの経路しか通らず、これを見逃していた。
    """
    import ast
    import inspect

    from inference.desktop import entrypoint

    tree = ast.parse(inspect.getsource(entrypoint.main).lstrip())
    main = tree.body[0]
    assert isinstance(main, ast.FunctionDef)
    binds_np = any(
        isinstance(node, ast.Import)
        and any(alias.name == "numpy" and alias.asname == "np" for alias in node.names)
        for node in main.body
    )
    uses_np = any(
        isinstance(node, ast.Name) and node.id == "np" for node in ast.walk(main)
    )
    assert uses_np
    assert binds_np or hasattr(entrypoint, "np")


class TestAssembleAction19:
    def test_uses_the_latest_targets_from_each_actuator(self):
        out = assemble_action19((0.1, 0.2, 0.3), np.arange(14.0), (4.5, 0.0))
        np.testing.assert_allclose(out[0:3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(out[3:17], np.arange(14.0))
        np.testing.assert_allclose(out[17:19], [4.5, 0.0])

    def test_falls_back_to_the_measured_waist_before_the_skill_commands_one(self):
        """skill が腰を送る前の tick でも、実測値で 19-D を組める。"""
        out = assemble_action19(
            None, np.zeros(14), (1.0, 1.0), measured_waist3=(0.7, 0.8, 0.9)
        )
        np.testing.assert_allclose(out[0:3], [0.7, 0.8, 0.9])

    def test_waist_defaults_to_zero_without_any_source(self):
        out = assemble_action19(None, np.zeros(14), (1.0, 1.0))
        np.testing.assert_allclose(out[0:3], 0.0)

    def test_hand_falls_back_when_the_skill_has_not_commanded_one(self):
        out = assemble_action19(
            (0.0, 0.0, 0.0), np.zeros(14), None, fallback_hand2=(4.5, 4.5)
        )
        np.testing.assert_allclose(out[17:19], [4.5, 4.5])

    def test_rejects_wrong_arm_width(self):
        with pytest.raises(ValueError, match=r"arms14 must be \(14,\)"):
            assemble_action19(None, np.zeros(13), None)


class TestOrganizerIkJointLimits:
    """publish する腕を運営 IK が解ける範囲に収める (Issue #164)。

    範囲の外を FK した EE は運営 IK が 1 mm 以内に解けず、運営 adapter がその腕の
    7 関節を丸ごと直前の姿勢で止める (模擬 Stage 2 の tighten で左腕 25% が該当)。
    """

    def test_limits_are_the_urdf_limits_with_the_organizer_overrides(self):
        import xml.etree.ElementTree as ET

        from inference.desktop.lower_policy.actuators.boundary_sink import (
            ORGANIZER_IK_ARM_LOWER_RAD,
            ORGANIZER_IK_ARM_UPPER_RAD,
            ORGANIZER_IK_LIMIT_MARGIN_RAD,
        )
        from inference.desktop.perception.g1_urdf_fk import ORGANIZER_IK_URDF_PATH

        root = ET.parse(ORGANIZER_IK_URDF_PATH).getroot()
        limits = {
            joint.get("name"): joint.find("limit")
            for joint in root.findall("joint")
            if joint.find("limit") is not None
        }
        names = [
            f"{side}_{name}_joint"
            for side in ("left", "right")
            for name in (
                "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                "wrist_roll", "wrist_pitch", "wrist_yaw",
            )
        ]
        lower = np.array([float(limits[n].get("lower")) for n in names])
        upper = np.array([float(limits[n].get("upper")) for n in names])
        # 運営 ik.py IKSettings: elbow_upper_limit_override=1.4, wrist_roll_limit_override=0.9
        upper[[3, 10]] = 1.4
        lower[[4, 11]] = -0.9
        upper[[4, 11]] = 0.9
        np.testing.assert_allclose(
            ORGANIZER_IK_ARM_LOWER_RAD, lower + ORGANIZER_IK_LIMIT_MARGIN_RAD
        )
        np.testing.assert_allclose(
            ORGANIZER_IK_ARM_UPPER_RAD, upper - ORGANIZER_IK_LIMIT_MARGIN_RAD
        )

    def test_out_of_range_joints_are_pulled_in_and_reported(self):
        from inference.desktop.lower_policy.actuators.boundary_sink import (
            ORGANIZER_IK_ARM_LOWER_RAD,
            ORGANIZER_IK_ARM_UPPER_RAD,
            clamp_arms_to_organizer_ik,
        )

        arms = np.zeros(14)
        arms[4] = -0.99  # 左 wrist_roll (教師の tighten が使う値)
        arms[10] = 1.39  # 右 elbow (教師の pick が使う値、1.4 - 余裕 の外)
        clamped, mask = clamp_arms_to_organizer_ik(arms)
        assert clamped[4] == pytest.approx(ORGANIZER_IK_ARM_LOWER_RAD[4])
        assert clamped[10] == pytest.approx(ORGANIZER_IK_ARM_UPPER_RAD[10])
        assert np.flatnonzero(mask).tolist() == [4, 10]
        np.testing.assert_array_equal(np.delete(clamped, [4, 10]), 0.0)

    def test_the_published_row_uses_the_clamped_arm(self):
        seen = []

        class _RecordingFk:
            def compute_ee_transforms(self, body_q29):  # noqa: ANN001
                seen.append(np.asarray(body_q29).copy())
                return np.zeros(3), np.eye(3), np.zeros(3), np.eye(3)

        sink, _ = _make_sink(log_fn=_FakeLog())
        sink._fk = _RecordingFk()  # type: ignore[attr-defined]
        action = np.zeros(19)
        action[3 + 4] = -1.2  # 左 wrist_roll
        sink.send_action(action, np.zeros(BODY_DOF))
        assert seen[-1][19] == pytest.approx(-0.88)
        assert sink._log_fn.records[-1]["organizer_ik_clamped_joints"] == [4]


# ---------------------------------------------------------------- joint lane (2026-09-25)
#
# 運営が :5556 に足した (T,22) の関節角の lane。腕の関節角をそのまま送り、運営 IK を通らない。
# 手と骨盤高さの列は pose lane と同じ変換 (向き -1=open / +1=closed、高さ 0.74 m) で作る。


def _make_joint_sink(log_fn=None) -> tuple[BoundaryActionSink, _FakeDecoupledSink]:
    sink, fake = _make_sink(log_fn=log_fn)
    sink._lane = "joint"  # type: ignore[attr-defined]
    sink._fk = None  # type: ignore[attr-defined]  # joint lane は FK を使わない
    return sink, fake


def _joint_action() -> np.ndarray:
    action = np.zeros(19)
    action[:3] = (0.3, -0.05, 0.1)  # 腰 (どちらの lane でも運営が実測で保持)
    action[3:17] = np.linspace(-0.9, 1.3, 14)
    action[3 + 3] = 1.8  # 左 elbow: 運営 IK の 1.4 より外 (joint lane では寄せない)
    action[3 + 4] = -1.2  # 左 wrist_roll: 0.9 より外
    action[17:19] = (4.5, 1.85)  # Dex1 の開き (model 空間 [rad])
    return action


class TestJointLane:
    @pytest.mark.requires_boundary
    def test_the_row_carries_the_policy_arms_as_they_are(self):
        sink, fake = _make_joint_sink(log_fn=_FakeLog())
        action = _joint_action()
        sink.send_action(action, np.zeros(BODY_DOF), navigate_cmd=(0.2, 0.0, -0.1))
        row = fake.chunks[0][0]
        assert fake.chunks[0].shape == (1, 22)
        # 腕は policy の関節角そのまま (FK・IK 用の clamp 無し)、左 7 → 右 7 = body_q[15:29] の順
        np.testing.assert_allclose(row[4:18], action[3:17], atol=1e-6)
        np.testing.assert_allclose(row[18:21], (0.2, 0.0, -0.1), atol=1e-6)
        assert row[21] == pytest.approx(0.74)  # 0 は「骨盤を床へ」
        record = sink._log_fn.records[-1]
        assert record["event"] == "boundary_joint"
        assert "organizer_ik_clamped_joints" not in record

    @pytest.mark.requires_boundary
    def test_hands_and_base_height_match_the_pose_lane(self):
        action = _joint_action()
        joint_sink, joint_fake = _make_joint_sink()
        pose_sink, pose_fake = _make_sink()
        joint_sink.send_action(action.copy(), np.zeros(BODY_DOF))
        pose_sink.send_action(action.copy(), np.zeros(BODY_DOF))
        joint_row, pose_row = joint_fake.chunks[0][0], pose_fake.chunks[0][0]
        np.testing.assert_allclose(joint_row[0:4], pose_row[0:4], atol=1e-6)
        assert joint_row[21] == pytest.approx(pose_row[21])

    @pytest.mark.requires_boundary
    def test_the_row_passes_the_organizer_joint_sink_validation(self):
        from inference.desktop.boundary import JointSink
        from inference.desktop.boundary.actions import JOINT_SLICES

        sink, fake = _make_joint_sink()
        action = _joint_action()
        sink.send_action(action, np.zeros(BODY_DOF))
        chunk = JointSink.validate_chunk(fake.chunks[0])
        np.testing.assert_allclose(chunk[0, JOINT_SLICES["left_arm"]], action[3:10], atol=1e-6)
        np.testing.assert_allclose(chunk[0, JOINT_SLICES["right_arm"]], action[10:17], atol=1e-6)

    @pytest.mark.requires_boundary
    def test_issued_at_is_stamped_on_the_sender_clock(self):
        import time

        sink, fake = _make_joint_sink()
        sink._sender_clock_offset_fn = lambda: 5.0  # type: ignore[attr-defined]
        before = time.time()
        sink.send_action(_joint_action(), np.zeros(BODY_DOF))
        assert before + 5.0 <= fake.issued_at[0] <= time.time() + 5.0

    def test_an_unknown_lane_is_rejected(self):
        with pytest.raises(ValueError, match="lane"):
            BoundaryActionSink(lane="taskspace")

    @pytest.mark.requires_boundary
    def test_the_organizer_adapter_receives_a_joint_message(self):
        """実 socket: 運営の JointSink が bind し、adapter と同じ形 (b"joint" + msgpack) で届く。"""
        import socket
        import time

        import msgpack
        import zmq

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        sink = BoundaryActionSink(lane="joint", port=port, host="127.0.0.1")
        context = zmq.Context.instance()
        sub = context.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"joint")
        sub.setsockopt(zmq.RCVTIMEO, 100)
        sub.connect(f"tcp://127.0.0.1:{port}")
        action = _joint_action()
        try:
            deadline = time.monotonic() + 5.0
            message = None
            while message is None and time.monotonic() < deadline:
                sink.send_action(action, np.zeros(BODY_DOF))  # PUB/SUB は接続直後を取りこぼす
                try:
                    message = sub.recv()
                except zmq.Again:
                    continue
        finally:
            sub.close(0)
            sink.close()
        assert message is not None and message.startswith(b"joint")
        payload = msgpack.unpackb(message[len(b"joint"):], raw=False)
        assert payload["shape"] == [1, 22] and payload["dtype"] == "f32"
        row = np.frombuffer(payload["actions"], dtype=np.float32).reshape(1, 22)[0]
        np.testing.assert_allclose(row[4:18], action[3:17], atol=1e-6)



# ------------------------------------------------ 手首 roll の clamp の切り替え (接続テスト用)
#
# 運営 IK の手首 roll の上限 0.9 は a1af470 で無くなった。PC2 の版を見て `--wrist-roll-clamp off`。


class TestWristRollClampSwitch:
    def test_off_keeps_the_wrist_roll_within_the_urdf_only(self):
        from inference.desktop.lower_policy.actuators.boundary_sink import (
            ORGANIZER_IK_ARM_UPPER_RAD,
            clamp_arms_to_organizer_ik,
        )

        arms = np.zeros(14)
        arms[4] = -1.2  # 左 wrist_roll (教師は -0.99 まで、端は -1.97)
        arms[11] = 2.5  # 右 wrist_roll: URDF の外
        arms[3] = 1.6  # 左 elbow: 運営 IK の 1.4 の外 (こちらは off でも寄せる)
        on, on_mask = clamp_arms_to_organizer_ik(arms)
        off, off_mask = clamp_arms_to_organizer_ik(arms, clamp_wrist_roll=False)
        assert on[4] == pytest.approx(-0.88)
        assert off[4] == pytest.approx(-1.2)  # 0.9 に寄せない
        assert off[11] == pytest.approx(1.972222054 - 0.02)  # URDF の範囲には寄せる
        assert off[3] == pytest.approx(ORGANIZER_IK_ARM_UPPER_RAD[3])
        assert np.flatnonzero(off_mask).tolist() == [3, 11]
        assert np.flatnonzero(on_mask).tolist() == [3, 4, 11]

    def test_the_sink_applies_the_switch_on_the_pose_lane(self):
        seen = []

        class _RecordingFk:
            def compute_ee_transforms(self, body_q29):  # noqa: ANN001
                seen.append(np.asarray(body_q29).copy())
                return np.zeros(3), np.eye(3), np.zeros(3), np.eye(3)

        sink, _ = _make_sink()
        sink._fk = _RecordingFk()  # type: ignore[attr-defined]
        sink._clamp_wrist_roll = False  # type: ignore[attr-defined]
        action = np.zeros(19)
        action[3 + 4] = -1.2  # 左 wrist_roll
        sink.send_action(action, np.zeros(BODY_DOF))
        assert seen[-1][19] == pytest.approx(-1.2)


# ------------------------------------------------ 送る間隔と行数 (Issue #172)
#
# 運営 adapter は chunk の 1 行目を実測から ±0.05 rad に切り詰める。30 Hz で 1 行ずつ送ると
# 重力で遅れる腕が下がり続けるので、会場の起動は「変わったときだけ、最短 0.1 s おきに、
# 同じ行を 16 行」送る。変わらない間は送らず (adapter が最後の行を保持)、1 s おきに送り直す。


def _paced_joint_sink(log_fn=None):
    clock = {"t": 100.0}
    sink, fake = _make_joint_sink(log_fn=log_fn)
    sink._publish_period_s = 0.1  # type: ignore[attr-defined]
    sink._chunk_rows = 16  # type: ignore[attr-defined]
    sink._hold_refresh_s = 1.0  # type: ignore[attr-defined]
    sink._clock = lambda: clock["t"]  # type: ignore[attr-defined]
    return sink, fake, clock


class TestPublishPacing:
    @pytest.mark.requires_boundary
    def test_the_class_default_still_publishes_every_call_as_one_row(self):
        sink, fake = _make_joint_sink()
        for _ in range(3):
            assert sink.send_action(_joint_action(), np.zeros(BODY_DOF))
        assert [c.shape for c in fake.chunks] == [(1, 22)] * 3

    @pytest.mark.requires_boundary
    def test_the_first_target_goes_out_as_sixteen_identical_rows(self):
        log = _FakeLog()
        sink, fake, _ = _paced_joint_sink(log_fn=log)
        assert sink.send_action(_joint_action(), np.zeros(BODY_DOF))
        assert fake.chunks[0].shape == (16, 22)
        np.testing.assert_array_equal(fake.chunks[0], np.repeat(fake.chunks[0][:1], 16, 0))
        assert log.records[-1]["chunk_rows"] == 16
        assert log.records[-1]["publish_reason"] == "first"

    @pytest.mark.requires_boundary
    def test_an_unchanged_target_is_held_by_the_adapter_until_the_refresh(self):
        sink, fake, clock = _paced_joint_sink()
        action = _joint_action()
        sink.send_action(action, np.zeros(BODY_DOF))
        for _ in range(29):  # 30 Hz で 1 s 弱
            clock["t"] += 1 / 30
            assert not sink.send_action(action, np.zeros(BODY_DOF))
        clock["t"] += 1 / 30
        assert sink.send_action(action, np.zeros(BODY_DOF))
        assert len(fake.chunks) == 2

    @pytest.mark.requires_boundary
    def test_a_changing_target_goes_out_at_most_every_period_with_the_latest_value(self):
        sink, fake, clock = _paced_joint_sink()
        action = _joint_action()
        sink.send_action(action, np.zeros(BODY_DOF))
        published = []
        for tick in range(1, 10):
            clock["t"] += 1 / 30
            action = action.copy()
            action[3] += 0.01  # 左 shoulder pitch が毎 tick 動く
            if sink.send_action(action, np.zeros(BODY_DOF)):
                published.append((tick, action[3]))
        assert [tick for tick, _ in published] == [3, 6, 9]  # 100 ms おき
        np.testing.assert_allclose(fake.chunks[-1][0][4], published[-1][1], atol=1e-6)

    @pytest.mark.requires_boundary
    def test_a_walking_command_change_goes_out_immediately(self):
        sink, fake, clock = _paced_joint_sink()
        action = _joint_action()
        sink.send_action(action, np.zeros(BODY_DOF), navigate_cmd=(0.3, 0.0, 0.0))
        clock["t"] += 1 / 30
        # 停止 (歩行ゼロ) は間隔を待たずに送る
        assert sink.send_action(action, np.zeros(BODY_DOF), navigate_cmd=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(fake.chunks[-1][:, 18:21], 0.0)

    @pytest.mark.requires_boundary
    def test_force_publishes_regardless_of_the_pacing(self):
        sink, fake, clock = _paced_joint_sink()
        action = _joint_action()
        sink.send_action(action, np.zeros(BODY_DOF))
        clock["t"] += 0.01
        assert sink.send_action(action, np.zeros(BODY_DOF), force=True)
        assert len(fake.chunks) == 2

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(publish_period_s=-0.1), "publish_period_s"),
            (dict(hold_refresh_s=float("nan")), "hold_refresh_s"),
            (dict(publish_period_s=0.5, hold_refresh_s=0.2), "hold_refresh_s"),
            (dict(chunk_rows=0), "chunk_rows"),
            (dict(chunk_rows=65), "chunk_rows"),
        ],
    )
    def test_invalid_pacing_is_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BoundaryActionSink(lane="joint", port=0, host="127.0.0.1", **kwargs)

    @pytest.mark.requires_boundary
    def test_the_organizer_wire_carries_the_sixteen_row_chunk(self):
        """実 socket: 運営 JointSink の検証を通った (16,22) が b"joint" + msgpack で届く。"""
        import socket
        import time

        import msgpack
        import zmq

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        sink = BoundaryActionSink(
            lane="joint", port=port, host="127.0.0.1",
            publish_period_s=0.1, chunk_rows=16, hold_refresh_s=1.0,
        )
        context = zmq.Context.instance()
        sub = context.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"joint")
        sub.setsockopt(zmq.RCVTIMEO, 100)
        sub.connect(f"tcp://127.0.0.1:{port}")
        action = _joint_action()
        try:
            deadline = time.monotonic() + 5.0
            message = None
            while message is None and time.monotonic() < deadline:
                sink.send_action(action, np.zeros(BODY_DOF), force=True)
                try:
                    message = sub.recv()
                except zmq.Again:
                    continue
        finally:
            sub.close(0)
            sink.close()
        assert message is not None
        payload = msgpack.unpackb(message[len(b"joint"):], raw=False)
        assert payload["shape"] == [16, 22]
        rows = np.frombuffer(payload["actions"], dtype=np.float32).reshape(16, 22)
        np.testing.assert_allclose(rows[:, 4:18], np.tile(action[3:17], (16, 1)), atol=1e-6)


# ------------------------------------------------ 重力の垂れ補正 (Issue #172)
#
# 運営 WBC は腕を重力補償なしの PD で動かす。送る関節角に 重力トルク / kp を足すと、実測が
# policy の目標に揃う (ラボの重力 FF と学習データの条件)。


def _offset(torque, *, kp=None, scale=1.0, max_offset_rad=0.12, joint_scale=None):
    from inference.desktop.lower_policy.actuators.boundary_sink import ArmGravitySagOffset

    kp = np.array([100, 100, 40, 40, 20, 20, 20] * 2, dtype=float) if kp is None else kp
    return ArmGravitySagOffset(
        lambda _arms: np.asarray(torque, dtype=float),
        kp, scale=scale, max_offset_rad=max_offset_rad, joint_scale=joint_scale,
    )


class TestGravitySagOffset:
    def test_the_offset_is_gravity_torque_over_the_wbc_kp(self):
        torque = np.zeros(14)
        torque[3] = -2.6  # 左 elbow: -2.6 Nm / kp 40 = -0.065 rad
        torque[7] = 5.0   # 右 shoulder pitch: 5 Nm / kp 100 = 0.05 rad
        command, offset = _offset(torque).apply(np.zeros(14))
        assert offset[3] == pytest.approx(-0.065)
        assert offset[7] == pytest.approx(0.05)
        np.testing.assert_allclose(command, offset)

    def test_scale_and_the_per_joint_bound(self):
        torque = np.full(14, 2.0)
        _, half = _offset(torque, scale=0.5).apply(np.zeros(14))
        assert half[0] == pytest.approx(0.01)  # 0.5 * 2 / 100
        _, capped = _offset(np.full(14, 50.0)).apply(np.zeros(14))
        assert np.max(np.abs(capped)) == pytest.approx(0.12)

    def test_the_joint_scale_multiplies_only_its_joints(self):
        """会場で肘だけ補正が足りなかった分 (Issue #188): 肘だけ 2.1 倍、他はそのまま。"""
        torque = np.zeros(14)
        torque[3] = -2.6   # 左 elbow: -0.065 rad
        torque[10] = -2.6  # 右 elbow
        torque[0] = 5.0    # 左 shoulder pitch: 0.05 rad
        joint = np.array([1.0, 1.0, 1.0, 2.1, 1.0, 1.0, 1.0] * 2)
        _, offset = _offset(torque, joint_scale=joint, max_offset_rad=0.18).apply(np.zeros(14))
        assert offset[3] == pytest.approx(-0.1365)
        assert offset[10] == pytest.approx(-0.1365)
        assert offset[0] == pytest.approx(0.05)
        _, capped = _offset(torque, joint_scale=joint, max_offset_rad=0.12).apply(np.zeros(14))
        assert capped[3] == pytest.approx(-0.12)  # 上限は倍率の後にかかる

    def test_a_wrong_joint_scale_in_the_config_is_rejected(self):
        from inference.desktop.lower_policy.actuators.boundary_sink import ArmGravitySagOffset

        config = {"boundary_gravity_offset": {"wbc_arm_kp": [100, 100, 40, 40, 20, 20, 20],
                                              "joint_scale": [1.0, 2.1]}}
        with pytest.raises(ValueError, match="joint_scale must list 7"):
            ArmGravitySagOffset.from_config(config)

    def test_the_offset_never_pushes_a_joint_past_its_urdf_limit(self):
        from inference.desktop.lower_policy.actuators.g1_arm_sdk import (
            G1_ARM_POSITION_UPPER_RAD,
        )

        arms = np.zeros(14)
        arms[3] = G1_ARM_POSITION_UPPER_RAD[3] - 0.01  # 左 elbow が上端の近く
        arms[10] = G1_ARM_POSITION_UPPER_RAD[10] + 0.2  # 右 elbow: policy が既に範囲外
        command, offset = _offset(np.full(14, 4.0)).apply(arms)
        assert command[3] == pytest.approx(G1_ARM_POSITION_UPPER_RAD[3] - 0.005)
        assert offset[10] == 0.0 and command[10] == arms[10]  # 範囲外の意図には足さない

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(kp=np.full(13, 40.0)), "wbc_arm_kp"),
            (dict(kp=np.zeros(14)), "wbc_arm_kp"),
            (dict(scale=-1.0), "scale"),
            (dict(max_offset_rad=0.0), "max_offset_rad"),
            (dict(joint_scale=np.ones(13)), "joint_scale"),
            (dict(joint_scale=np.zeros(14)), "joint_scale"),
            (dict(joint_scale=np.full(14, 3.5)), "joint_scale"),
        ],
    )
    def test_invalid_settings_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _offset(np.zeros(14), **kwargs)

    @pytest.mark.requires_boundary
    def test_the_joint_row_carries_intent_plus_offset_and_logs_both(self):
        log = _FakeLog()
        sink, fake = _make_joint_sink(log_fn=log)
        torque = np.zeros(14)
        torque[3] = -2.6
        sink._gravity_offset = _offset(torque)  # type: ignore[attr-defined]
        action = _joint_action()
        sink.send_action(action, np.zeros(BODY_DOF))
        row = fake.chunks[0][0]
        expected = action[3:17].copy()
        expected[3] -= 0.065
        np.testing.assert_allclose(row[4:18], expected, atol=1e-6)
        record = log.records[-1]
        np.testing.assert_allclose(record["arms_intent"], action[3:17])
        assert record["gravity_offset_rad"][3] == pytest.approx(-0.065)
        assert record["gravity_offset_scale"] == 1.0

    def test_a_failing_gravity_model_refuses_new_commands(self):
        from inference.desktop.lower_policy.actuators.boundary_sink import ArmGravitySagOffset

        def _broken(_arms):
            raise RuntimeError("gravity feed-forward exceeded the deployment bound")

        log = _FakeLog()
        sink, fake = _make_joint_sink(log_fn=log)
        sink._gravity_offset = ArmGravitySagOffset(  # type: ignore[attr-defined]
            _broken, np.full(14, 40.0)
        )
        action = _joint_action()
        with pytest.raises(RuntimeError, match="deployment bound"):
            sink.send_action(action, np.zeros(BODY_DOF))
        assert fake.chunks == []
        assert log.records == []

    def test_the_pose_lane_refuses_the_offset(self):
        with pytest.raises(ValueError, match="joint lane only"):
            BoundaryActionSink(
                lane="pose", port=0, host="127.0.0.1", gravity_offset=_offset(np.zeros(14))
            )

    def test_the_shipped_config_and_model_predict_the_start_pose_sag(self):
        """実物の model と YAML: 各 policy の開始姿勢で、model だけの補正は約 0.06 rad、肘は 2.1 倍。"""
        pytest.importorskip("pinocchio")
        import yaml

        from inference.desktop.lower_policy.actuators.boundary_sink import ArmGravitySagOffset
        from inference.desktop.lower_policy.initial_pose import initial_pose_from_config

        with open(
            "inference/desktop/lower_policy/configs/skill_config.yaml", encoding="utf-8"
        ) as f:
            config = yaml.safe_load(f)
        offset = ArmGravitySagOffset.from_config(config)
        plain_config = {
            "boundary_gravity_offset": {
                key: value for key, value in config["boundary_gravity_offset"].items()
                if key != "joint_scale"
            }
        }
        plain = ArmGravitySagOffset.from_config(plain_config)
        elbows = [3, 10]
        for skill in ("rotate_table_base", "pick_table_leg", "insert_table_leg",
                      "rotate_leg_to_tighten", "flip_table"):
            arms = initial_pose_from_config(config, skill).arm_position_rad
            _, model_only = plain.apply(arms)
            _, added = offset.apply(arms)
            assert 0.05 <= np.max(np.abs(model_only)) <= 0.08, skill
            np.testing.assert_allclose(added[elbows], 2.1 * model_only[elbows], atol=1e-9)
            others = [k for k in range(14) if k not in elbows]
            np.testing.assert_allclose(added[others], model_only[others], atol=1e-9)
            assert np.max(np.abs(added)) <= 0.18, skill
