"""Phase 9a smoke: VlaSkill wrapper + Waist/Hand actuator mock (Issue #125)。

- 19D action chunk[0] → waist(3) + arm(14) + hand(2) の dispatch
- per-skill_name identity hardcode (MoveTableBaseVlaSkill.SKILL_ID = 5)
- Skill lifecycle (start/stop/step) が既存 Skill ABC と compatible
- MockWaist/MockHand actuator の history / latest 保持
"""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pytest

from inference.desktop.lower_policy.actuators.hand import (
    G1_NUM_HAND_JOINTS,
    MockHandActuator,
)
from inference.desktop.lower_policy.actuators.waist import (
    G1_NUM_WAIST_JOINTS,
    G1_WAIST_JOINT_INDICES,
    MockWaistActuator,
)
from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    PolicyAction,
    PolicyConfig,
    RawRobotState,
)
from inference.desktop.lower_policy.skills.vla_skill import (
    ACTION_DIM_TOTAL,
    ARMS_SLICE,
    FlipTableVlaSkill,
    HAND_SLICE,
    InsertAndTightenVlaSkill,
    InsertTableLegVlaSkill,
    MoveTableBaseVlaSkill,
    PickTableLegVlaSkill,
    RotateLegToTightenVlaSkill,
    RotateTableBaseVlaSkill,
    VlaSkill,
    WAIST_SLICE,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


class _StubPolicy:
    """Test 用 stub policy: predict() が固定 chunk を返す。"""

    def __init__(self, chunk_value: float = 0.5, chunk_len: int = 16) -> None:
        self._chunk = (
            np.arange(chunk_len * ACTION_DIM_TOTAL, dtype=np.float32).reshape(
                chunk_len, ACTION_DIM_TOTAL
            )
            + chunk_value
        )
        self.predict_call_count = 0

    @classmethod
    def from_ckpt(cls, cfg):  # noqa: ANN001
        return cls()

    @staticmethod
    def build_state_from_raw(raw: RawRobotState) -> np.ndarray:
        return np.zeros(71, dtype=np.float32)

    def warmup(self, n_iter: int = 5) -> None:
        pass

    def predict(self, obs) -> PolicyAction:  # noqa: ANN001
        self.predict_call_count += 1
        return PolicyAction(
            action_chunk=self._chunk.copy(),
            latency_ms=1.0,
            metadata={"stub": True},
        )

    def close(self) -> None:
        pass


class _ChunkExecutingStubPolicy(_StubPolicy):
    """GR00T-like policy which opts in to sequential chunk execution."""

    EXECUTION_HORIZON = 3


class TestActionSlice:
    """19D action の 3-way slice (waist / arm / hand) が正しい範囲を選ぶ。"""

    def test_slices_cover_19d(self) -> None:
        total = (
            (WAIST_SLICE.stop - WAIST_SLICE.start)
            + (ARMS_SLICE.stop - ARMS_SLICE.start)
            + (HAND_SLICE.stop - HAND_SLICE.start)
        )
        assert total == ACTION_DIM_TOTAL == 19

    def test_slices_dims(self) -> None:
        assert WAIST_SLICE.stop - WAIST_SLICE.start == 3
        assert ARMS_SLICE.stop - ARMS_SLICE.start == 14
        assert HAND_SLICE.stop - HAND_SLICE.start == 2


class TestMockActuators:
    def test_waist_history(self) -> None:
        act = MockWaistActuator()
        act.send_action([0.1, 0.2, 0.3])
        act.send_action([0.4, 0.5, 0.6])
        assert len(act.history) == 2
        assert act.latest == (0.4, 0.5, 0.6)

    def test_waist_reject_wrong_length(self) -> None:
        act = MockWaistActuator()
        with pytest.raises(ValueError, match="length 3"):
            act.send_action([0.1, 0.2])

    def test_hand_history(self) -> None:
        act = MockHandActuator()
        act.send_action([0.1, 0.9])
        assert act.latest == (0.1, 0.9)

    def test_hand_reject_wrong_length(self) -> None:
        act = MockHandActuator()
        with pytest.raises(ValueError, match="length 2"):
            act.send_action([0.1])

    def test_waist_joint_indices(self) -> None:
        # G1 body layout: waist = 12, 13, 14
        assert G1_WAIST_JOINT_INDICES == (12, 13, 14)
        assert G1_NUM_WAIST_JOINTS == 3

    def test_hand_num_joints(self) -> None:
        assert G1_NUM_HAND_JOINTS == 2


class TestVlaSkillIdentity:
    def test_base_class_rejects_no_identity(self) -> None:
        # VlaSkill 単体は SKILL_ID/LANGUAGE 未設定 → construct 拒絶
        with pytest.raises(ValueError, match="SKILL_ID or LANGUAGE"):
            VlaSkill(
                policy=_StubPolicy(),
                waist_actuator=MockWaistActuator(),
                hand_actuator=MockHandActuator(),
            )

    def test_move_table_base_identity(self) -> None:
        assert MoveTableBaseVlaSkill.SKILL_ID == 5
        assert MoveTableBaseVlaSkill.LANGUAGE == "rotate and move table base (combined 5+7)"
        assert MoveTableBaseVlaSkill.name == "move_table_base"

    def test_rotate_table_base_identity(self) -> None:
        assert RotateTableBaseVlaSkill.SKILL_ID == 4
        assert RotateTableBaseVlaSkill.LANGUAGE == "rotate and move table base (combined 5+7)"
        assert RotateTableBaseVlaSkill.name == "rotate_table_base"

    def test_pick_table_leg_identity(self) -> None:
        # skill_mapping.py:SKILL_PICK_TABLE_LEG = 3
        # subtask_training.json:pick_leg.task = "pick table leg"
        assert PickTableLegVlaSkill.SKILL_ID == 3
        assert PickTableLegVlaSkill.LANGUAGE == "pick table leg"
        assert PickTableLegVlaSkill.name == "pick_table_leg"

    def test_insert_table_leg_identity(self) -> None:
        # skill_mapping.py:SKILL_INSERT_TABLE_LEG = 0
        # takada dataset tasks.parquet: "insert table leg to table base"
        assert InsertTableLegVlaSkill.SKILL_ID == 0
        assert (
            InsertTableLegVlaSkill.LANGUAGE == "insert table leg to table base"
        )
        assert InsertTableLegVlaSkill.name == "insert_table_leg"

    def test_rotate_leg_to_tighten_identity(self) -> None:
        # skill_mapping.py:SKILL_ROTATE_LEG_TO_TIGHTEN = 2
        # takada dataset tasks.parquet: "rotate leg to tighten"
        assert RotateLegToTightenVlaSkill.SKILL_ID == 2
        assert RotateLegToTightenVlaSkill.LANGUAGE == "rotate leg to tighten"
        assert RotateLegToTightenVlaSkill.name == "rotate_leg_to_tighten"

    def test_insert_and_tighten_identity_is_groot_only(self) -> None:
        assert InsertAndTightenVlaSkill.SKILL_ID is None
        assert InsertAndTightenVlaSkill.LANGUAGE == (
            "insert table leg to table base and rotate leg to tighten"
        )
        assert InsertAndTightenVlaSkill.name == "insert_and_tighten"

    def test_rotate_leg_vs_rotate_table_base_distinct(self) -> None:
        # skill_mapping.py docstring 記載: rotate_leg_to_tighten と rotate_table_base
        # は完全別 skill (skill_id 2 vs 4)、user 指摘の "別物" を回帰防止 test で担保
        assert (
            RotateLegToTightenVlaSkill.SKILL_ID != RotateTableBaseVlaSkill.SKILL_ID
        )
        assert (
            RotateLegToTightenVlaSkill.LANGUAGE != RotateTableBaseVlaSkill.LANGUAGE
        )

    def test_flip_table_identity(self) -> None:
        # skill_mapping.py:SKILL_FLIP_TABLE = 1
        # suzuki dataset tasks.parquet: "flip table"
        assert FlipTableVlaSkill.SKILL_ID == 1
        assert FlipTableVlaSkill.LANGUAGE == "flip table"
        assert FlipTableVlaSkill.name == "flip_table"

    def test_all_skill_ids_unique(self) -> None:
        # 6 skill (setup + walk 系除く) の skill_id 全 unique (skill_mapping.py canonical)
        ids = {
            MoveTableBaseVlaSkill.SKILL_ID,
            RotateTableBaseVlaSkill.SKILL_ID,
            PickTableLegVlaSkill.SKILL_ID,
            InsertTableLegVlaSkill.SKILL_ID,
            RotateLegToTightenVlaSkill.SKILL_ID,
            FlipTableVlaSkill.SKILL_ID,
        }
        assert ids == {0, 1, 2, 3, 4, 5}  # 6 skill 全 skill_id カバー


@pytest.mark.skipif(
    not HAS_TORCH, reason="torch required (VlaSkill._build_observation calls preprocess)"
)
class TestVlaSkillLifecycle:
    def _make_skill(self) -> tuple[
        MoveTableBaseVlaSkill, _StubPolicy, MockWaistActuator, MockHandActuator
    ]:
        policy = _StubPolicy(chunk_value=0.0)
        waist = MockWaistActuator()
        hand = MockHandActuator()
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=waist,
            hand_actuator=hand,
        )
        return skill, policy, waist, hand

    def _obs(self) -> dict:
        # orchestrator obs schema (SampleVLASkill docstring 準拠)
        return {
            "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),  # 単 cam (split 不要)
            "wrist_left_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_right_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "joint_state": None,
            "cleaned": [],
            "t": time.monotonic_ns(),
        }

    def test_start_stop_lifecycle(self) -> None:
        skill, _, _, _ = self._make_skill()
        skill.start({})
        skill.stop()
        # 再 start OK
        skill.start({})
        skill.stop()

    def test_step_returns_arm_14d(self) -> None:
        skill, policy, waist, hand = self._make_skill()
        skill.start({})
        arm = skill.step(self._obs())
        assert arm.shape == (14,)
        assert policy.predict_call_count == 1
        # last_action metadata 保持
        assert skill.last_action is not None
        assert skill.last_action.metadata["stub"] is True

    def test_pending_operator_key_suppresses_stale_hand_and_waist_command(self) -> None:
        skill, policy, waist, hand = self._make_skill()
        skill.set_operator_interrupt_pending(lambda: True)
        skill.start({})
        assert skill.step(self._obs()) is None
        assert policy.predict_call_count == 1
        assert hand.history == []
        assert waist.history == []

    def test_empty_detector_result_is_preserved_for_overlay_policy(self) -> None:
        skill, _, _, _ = self._make_skill()
        obs = self._obs()
        obs["cleaned"] = []

        observation = skill._build_observation(obs)

        assert observation.obb_detections == {CameraKey.HEAD_LEFT: []}

    def test_missing_detector_remains_distinct_from_zero_detections(self) -> None:
        skill, _, _, _ = self._make_skill()
        obs = self._obs()
        obs["cleaned"] = None

        observation = skill._build_observation(obs)

        assert observation.obb_detections is None

    def test_language_uses_class_default_when_override_not_provided(self) -> None:
        # Issue #132 Phase A: 他 skill 用に「override 未指定なら class hard-code」
        # の chain を担保 (user 心配「Default が入る失敗」に対する guard)。
        skill, _, _, _ = self._make_skill()
        observation = skill._build_observation(self._obs())
        assert observation.language == MoveTableBaseVlaSkill.LANGUAGE
        assert observation.language == "rotate and move table base (combined 5+7)"

    def test_language_override_takes_priority(self) -> None:
        # Issue #132 Phase A/B: 当時の groot_overlay slot (rotate_table_base default、2026-10-08 に外した)
        # の language_prompt が policy_config.yaml 経由で VlaSkill に渡された
        # 場合、class-level LANGUAGE を上書きする。default_prompt (module 定数)
        # には落ちない。
        policy = _StubPolicy(chunk_value=0.0)
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=MockWaistActuator(),
            hand_actuator=MockHandActuator(),
            language_override="rotate table base",
        )
        observation = skill._build_observation(self._obs())
        assert observation.language == "rotate table base"
        assert observation.language != MoveTableBaseVlaSkill.LANGUAGE

    def test_language_override_empty_string_rejected(self) -> None:
        # 空文字 override は「指定した意図と誤解釈される trap」= error に落とす
        # (PolicyConfig 側 guard と VlaSkill 側 guard で 2 段防御)。
        with pytest.raises(ValueError, match="language_override"):
            MoveTableBaseVlaSkill(
                policy=_StubPolicy(),
                waist_actuator=MockWaistActuator(),
                hand_actuator=MockHandActuator(),
                language_override="",
            )

    def test_step_dispatches_waist_and_hand(self) -> None:
        skill, _, waist, hand = self._make_skill()
        skill.start({})
        arm = skill.step(self._obs())
        # StubPolicy の chunk[0] = arange(19) + 0.0 = [0, 1, 2, ..., 18]
        # waist = chunk[0][0:3] = [0.0, 1.0, 2.0]
        # arm   = chunk[0][3:17] = [3.0..16.0]
        # hand  = chunk[0][17:19] = [17.0, 18.0] -> Dex1 rad safety clamp [5.4, 5.4]
        assert waist.latest == (0.0, 1.0, 2.0)
        assert hand.latest == (5.4, 5.4)

    def test_step_arm_positions_values(self) -> None:
        skill, _, _, _ = self._make_skill()
        skill.start({})
        arm = skill.step(self._obs())
        np.testing.assert_array_almost_equal(
            arm, np.arange(3, 17, dtype=np.float64)
        )

    def test_frame_buffer_updates_across_ticks(self) -> None:
        # 1 tick 目は frames_prev None、2 tick 目は前 tick の frames_bgr が保持されてる
        skill, _, _, _ = self._make_skill()
        skill.start({})
        assert skill._frames_prev is None
        skill.step(self._obs())
        assert skill._frames_prev is not None
        assert CameraKey.HEAD_LEFT in skill._frames_prev

    def test_frame_buffer_reset_on_start(self) -> None:
        skill, _, _, _ = self._make_skill()
        skill.start({})
        skill.step(self._obs())
        assert skill._frames_prev is not None
        skill.stop()
        skill.start({})
        assert skill._frames_prev is None

    def test_head_stereo_split(self) -> None:
        # 480x1280 packed head → split して HEAD_LEFT / HEAD_RIGHT 別々に埋まる
        skill, policy, _, _ = self._make_skill()
        skill.start({})
        obs = self._obs()
        # 左半 = 255、右半 = 100 の packed frame で識別
        packed = np.zeros((480, 1280, 3), dtype=np.uint8)
        packed[:, :640, :] = 255
        packed[:, 640:, :] = 100
        obs["head_rgb"] = packed
        skill.step(obs)
        # frame buffer には split 済 image が入る
        assert skill._frames_prev[CameraKey.HEAD_LEFT].shape == (480, 640, 3)
        assert skill._frames_prev[CameraKey.HEAD_LEFT][0, 0, 0] == 255
        assert skill._frames_prev[CameraKey.HEAD_RIGHT][0, 0, 0] == 100

    def test_multi_step_predict_count(self) -> None:
        # MVP は per-tick predict = N step で N 回 predict
        skill, policy, _, _ = self._make_skill()
        skill.start({})
        for _ in range(3):
            skill.step(self._obs())
        assert policy.predict_call_count == 3

    def test_opt_in_policy_executes_chunk_rows_before_replanning(self) -> None:
        policy = _ChunkExecutingStubPolicy(chunk_value=0.0, chunk_len=3)
        waist = MockWaistActuator()
        hand = MockHandActuator()
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=waist,
            hand_actuator=hand,
        )
        skill.start({})

        arms = [skill.step(self._obs()) for _ in range(7)]

        assert policy.predict_call_count == 3
        for tick, arm in enumerate(arms):
            chunk_row = tick % 3
            expected = policy._chunk[chunk_row, ARMS_SLICE]
            np.testing.assert_array_equal(arm, expected.astype(np.float64))
        assert skill.last_action is not None
        assert skill.last_action.metadata["chunk_step_index"] == 0
        assert skill.last_action.metadata["chunk_execution_horizon"] == 3
        assert skill.last_action.metadata["chunk_reused"] is False

    def test_action_queue_is_cleared_by_skill_restart(self) -> None:
        policy = _ChunkExecutingStubPolicy(chunk_value=0.0, chunk_len=3)
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=MockWaistActuator(),
            hand_actuator=MockHandActuator(),
        )
        skill.start({})
        skill.step(self._obs())
        assert len(skill._action_queue) == 2
        skill.stop()
        skill.start({})
        assert skill._action_queue == []
        assert skill._action_queue_next_index == 0

    def test_dispatch_waist_true_calls_waist_actuator(self) -> None:
        # default: dispatch_waist=True → waist_actuator.send_action が呼ばれる
        skill, _, waist, _ = self._make_skill()
        skill.start({})
        skill.step(self._obs())
        assert waist.latest is not None
        assert len(waist.history) == 1

    def test_dispatch_waist_false_skips_waist_actuator(self) -> None:
        # dispatch_waist=False → Model waist chunk は捨てられ、waist_actuator は
        # send_action 未呼出 = latest=None、history 空
        policy = _StubPolicy(chunk_value=0.0)
        waist = MockWaistActuator()
        hand = MockHandActuator()
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=waist,
            hand_actuator=hand,
            dispatch_waist=False,
        )
        skill.start({})
        skill.step(self._obs())
        assert waist.latest is None
        assert len(waist.history) == 0
        # hand は現行通り dispatch されている (waist だけ gate)
        assert hand.latest is not None

    def test_policy_without_waist_output_skips_waist_actuator(self) -> None:
        # Issue #144: skill が dispatch_waist=True でも、policy が
        # outputs_waist=False (RAMEN-Ori の 16D 学習 = 腰は推論側が 0 で pad) なら
        # 腰は送らない。送ると腰が 0 rad = 直立へ寄っていく。
        policy = _StubPolicy(chunk_value=0.0)
        policy.outputs_waist = False
        waist = MockWaistActuator()
        hand = MockHandActuator()
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=waist,
            hand_actuator=hand,
            dispatch_waist=True,
        )
        skill.start({})
        skill.step(self._obs())
        assert waist.latest is None
        assert len(waist.history) == 0
        # 腕と hand は通常どおり (gate は腰だけ)
        assert hand.latest is not None

    def test_policy_with_waist_output_still_dispatches(self) -> None:
        # outputs_waist=True を明示した policy は従来どおり流す (19D 学習の ckpt)。
        policy = _StubPolicy(chunk_value=0.0)
        policy.outputs_waist = True
        waist = MockWaistActuator()
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=waist,
            hand_actuator=MockHandActuator(),
            dispatch_waist=True,
        )
        skill.start({})
        skill.step(self._obs())
        assert waist.latest is not None

    def test_wrist_848_cropped_to_640(self) -> None:
        # D405 native 848x480 は _build_observation で 640x480 に center crop され、
        # frames_bgr / frames_prev には 640x480 として入る。訓練 aspect 整合。
        skill, _, _, _ = self._make_skill()
        skill.start({})
        obs = self._obs()
        wrist_848 = np.arange(480 * 848 * 3, dtype=np.uint8).reshape(480, 848, 3)
        obs["wrist_left_rgb"] = wrist_848
        obs["wrist_right_rgb"] = wrist_848
        skill.step(obs)
        assert skill._frames_prev[CameraKey.WRIST_LEFT].shape == (480, 640, 3)
        assert skill._frames_prev[CameraKey.WRIST_RIGHT].shape == (480, 640, 3)
        # 中央 640 列と一致 (crop 位置: 104..744)
        np.testing.assert_array_equal(
            skill._frames_prev[CameraKey.WRIST_LEFT], wrist_848[:, 104:744, :]
        )


# ---- Phase A-6 end-to-end smoke (Gap A 完成 verify) ---- #


class _JointStateStub:
    """Minimal stub mimicking JointStateData (perception.joint_state_source)。"""

    def __init__(self, position: np.ndarray) -> None:
        self.position = position


class _RamenOriPolicyStub:
    """RAMEN-Ori 相当の Policy stub。build_state_from_raw で実 71D state を返す
    = Gap A の全 A-1〜A-5 統合 verify に使う。"""

    STATE_DIM = 71

    def __init__(self) -> None:
        self.predict_call_count = 0
        self.received_states: list[np.ndarray] = []

    @classmethod
    def from_ckpt(cls, cfg):  # noqa: ANN001
        return cls()

    @staticmethod
    def build_state_from_raw(raw):  # noqa: ANN001
        # 実 ramen_ori.build_state_from_raw を叩いて 71D layout で state を組む
        from inference.desktop.lower_policy.policies.ramen_ori import (
            build_state_from_raw as _real_build,
        )

        return _real_build(raw)

    def warmup(self, n_iter: int = 5) -> None:
        pass

    def predict(self, obs) -> PolicyAction:  # noqa: ANN001
        self.predict_call_count += 1
        self.received_states.append(obs.state.copy())
        return PolicyAction(
            action_chunk=(
                np.arange(16 * ACTION_DIM_TOTAL, dtype=np.float32).reshape(
                    16, ACTION_DIM_TOTAL
                )
                / 100.0  # small values ~ [0, 0.5] rad range
            ),
            latency_ms=1.0,
            metadata={"stub": "ramen_ori"},
        )

    def close(self) -> None:
        pass


class TestPhaseA6EndToEndSmoke:
    """Gap A 全部 (A-1 schema + A-2/A-3 slice + A-4 buffer + A-5 FK) を 30-tick
    loop で end-to-end 検証。

    torch 依存無し (StubPolicy.predict は build_batch_dict 経由せず = preprocess_frame
    未呼び出し)、default env で走る。TestVlaSkillLifecycle が torch skip guard 付き
    なのは初期の conservative 判断で、本 class は skip 不要。"""

    def _make_skill_with_fk(self, use_fk: bool = True):
        from pathlib import Path

        from inference.desktop.perception.g1_urdf_fk import (
            DEFAULT_URDF_PATH,
            G1WristFK,
        )

        # tests → skills → lower_policy → desktop → inference → iros_2026_ramen (6 parents)
        repo_root = Path(__file__).parents[5]
        urdf_path = repo_root / DEFAULT_URDF_PATH
        fk = G1WristFK.from_urdf(str(urdf_path)) if (use_fk and urdf_path.exists()) else None

        policy = _RamenOriPolicyStub()
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=MockWaistActuator(),
            hand_actuator=MockHandActuator(),
            fk=fk,
        )
        return skill, policy, fk

    def _obs(self, tick: int, joint_positions: np.ndarray) -> dict:
        return {
            "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_left_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_right_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "joint_state": _JointStateStub(joint_positions),
            "hand_state": np.array([4.2, 1.1], dtype=np.float32),
            "cleaned": [],
            "t": time.monotonic_ns(),
        }

    def test_30_tick_loop_no_exceptions(self) -> None:
        # A-1〜A-5 全パイプが 30 tick 例外無しに動く (smoke as gap-A completion criterion)
        skill, policy, _ = self._make_skill_with_fk(use_fk=True)
        skill.start({})
        rng = np.random.default_rng(seed=123)
        for tick in range(30):
            jp = rng.uniform(-0.5, 0.5, size=29).astype(np.float32)
            arm = skill.step(self._obs(tick, jp))
            assert arm.shape == (14,)
        assert policy.predict_call_count == 30
        # State buffer が最終 tick まで更新されている
        assert skill._prev_joint_positions is not None
        assert skill._prev_action_19d is not None
        assert skill._prev_action_19d.shape == (19,)

    def test_state_slices_populated_after_first_tick(self) -> None:
        # 1 tick 目: last_action_19d = None → tracking_err zeros、
        # joint_positions_prev = None → velocity zeros
        # 2 tick 目以降: 両方 populated
        skill, policy, fk = self._make_skill_with_fk(use_fk=True)
        skill.start({})

        # tick 0
        jp0 = np.linspace(-0.3, 0.3, 29, dtype=np.float32)
        skill.step(self._obs(0, jp0))
        state0 = policy.received_states[0]
        # joint slice = joint_positions[UPPER_BODY] + hand
        # UPPER_BODY = [12:15] + [15:22] + [22:29] = 17
        np.testing.assert_allclose(state0[0:17], jp0[12:29], atol=1e-6)
        np.testing.assert_allclose(state0[17:19], [4.2, 1.1], atol=1e-6)
        np.testing.assert_allclose(state0[57:59], [4.2, 1.1], atol=1e-6)
        # tracking_err / velocity zeros
        np.testing.assert_array_equal(state0[19:38], np.zeros(19, dtype=np.float32))
        np.testing.assert_array_equal(state0[38:57], np.zeros(19, dtype=np.float32))
        # ee_state 埋まっている (FK 有効時)
        assert not np.all(state0[59:71] == 0), "ee_state should be populated by FK"

        # tick 1
        jp1 = np.linspace(0.1, 0.4, 29, dtype=np.float32)
        skill.step(self._obs(1, jp1))
        state1 = policy.received_states[1]
        # velocity = current UPPER - prev UPPER (hand 部分は current_hand - 0)
        expected_upper_vel = jp1[12:29] - jp0[12:29]
        np.testing.assert_allclose(state1[38:55], expected_upper_vel, atol=1e-6)
        np.testing.assert_allclose(state1[55:57], [0.0, 0.0], atol=1e-6)
        # tracking_err = last_action_19d - joint_slice (last_action = chunk[0]/100 for stub)
        assert not np.all(state1[19:38] == 0), "tracking_err should be populated after first tick"

    def test_ee_state_zero_when_fk_disabled(self) -> None:
        # FK 無し (fk=None) では ee_state = zeros
        skill, policy, _ = self._make_skill_with_fk(use_fk=False)
        skill.start({})
        jp = np.zeros(29, dtype=np.float32)
        skill.step(self._obs(0, jp))
        state = policy.received_states[0]
        np.testing.assert_array_equal(state[59:71], np.zeros(12, dtype=np.float32))

    def test_reset_on_start_clears_buffers(self) -> None:
        skill, policy, _ = self._make_skill_with_fk(use_fk=True)
        skill.start({})
        skill.step(self._obs(0, np.zeros(29, dtype=np.float32)))
        assert skill._prev_joint_positions is not None
        assert skill._prev_action_19d is not None
        # 再 start で buffer reset
        skill.stop()
        skill.start({})
        assert skill._prev_joint_positions is None
        assert skill._prev_hand_state is None
        assert skill._prev_action_19d is None
        assert skill._frames_prev is None

    def test_all_states_finite(self) -> None:
        # 30 tick 全 state が finite (NaN/Inf 混入無し)
        skill, policy, _ = self._make_skill_with_fk(use_fk=True)
        skill.start({})
        rng = np.random.default_rng(seed=7)
        for tick in range(30):
            jp = rng.uniform(-0.8, 0.8, size=29).astype(np.float32)
            skill.step(self._obs(tick, jp))
        for i, s in enumerate(policy.received_states):
            assert np.all(np.isfinite(s)), f"tick {i}: non-finite state"


# ---- Phase 2 (Issue #128): policy.reset() propagation from _on_start ---- #


class _ResettablePolicy:
    """policy.reset() を持つ stub。VlaSkill._on_start が reset を呼ぶこと verify 用。"""

    def __init__(self) -> None:
        self.reset_call_count = 0
        self._chunk = np.zeros((16, ACTION_DIM_TOTAL), dtype=np.float32)

    @staticmethod
    def build_state_from_raw(raw: RawRobotState) -> np.ndarray:
        return np.zeros(71, dtype=np.float32)

    def predict(self, obs) -> PolicyAction:  # noqa: ANN001
        return PolicyAction(
            action_chunk=self._chunk.copy(),
            latency_ms=0.5,
            metadata={},
        )

    def reset(self) -> None:
        self.reset_call_count += 1


class _NoResetPolicy:
    """reset method 無しの stub (protocol optional であることの verify 用)。"""

    @staticmethod
    def build_state_from_raw(raw: RawRobotState) -> np.ndarray:
        return np.zeros(71, dtype=np.float32)

    def predict(self, obs) -> PolicyAction:  # noqa: ANN001
        return PolicyAction(
            action_chunk=np.zeros((16, ACTION_DIM_TOTAL), dtype=np.float32),
            latency_ms=0.5,
            metadata={},
        )


class TestVlaSkillPolicyResetPropagation:
    def test_on_start_calls_policy_reset_when_available(self) -> None:
        policy = _ResettablePolicy()
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=MockWaistActuator(),
            hand_actuator=MockHandActuator(),
        )
        assert policy.reset_call_count == 0
        skill.start({})
        assert policy.reset_call_count == 1
        skill.stop()
        skill.start({})
        # 再 start でも reset される (skill 遷移時の candidate クリア pattern)
        assert policy.reset_call_count == 2

    def test_on_start_no_op_when_policy_has_no_reset(self) -> None:
        # protocol optional: reset を持たない policy でも VlaSkill が壊れないこと
        policy = _NoResetPolicy()
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=MockWaistActuator(),
            hand_actuator=MockHandActuator(),
        )
        skill.start({})  # AttributeError で落ちないこと
        skill.stop()


# ---- Phase 3 (Issue #128): MotionLimiter DI + apply propagation ---- #


class _RecordingLimiter:
    """apply / reset の呼出を記録する stub。VlaSkill が正しく wire しているか verify 用。"""

    def __init__(self) -> None:
        self.apply_call_count = 0
        self.reset_call_count = 0
        self.last_target = None
        self.last_measured = None
        # apply の戻り: target をそのまま返す (identity)
        # scale で明示差別化したい test は _apply_scale を設定
        self._apply_scale = 1.0

    def apply(self, *, target, measured):
        import numpy as np
        self.apply_call_count += 1
        self.last_target = np.asarray(target, dtype=np.float64).copy()
        self.last_measured = np.asarray(measured, dtype=np.float64).copy()
        return (np.asarray(target, dtype=np.float64) * self._apply_scale)

    def reset(self) -> None:
        self.reset_call_count += 1


class TestVlaSkillMotionLimiterIntegration:
    def _make_skill(self, limiter=None):
        policy = _StubPolicy(chunk_value=0.0)
        waist = MockWaistActuator()
        hand = MockHandActuator()
        skill = MoveTableBaseVlaSkill(
            policy=policy,
            waist_actuator=waist,
            hand_actuator=hand,
            motion_limiter=limiter,
        )
        return skill, policy, waist, hand

    def _obs(self, joint_state_val=0.0, hand_state_val=1.0):
        # joint_state = 29-dim numpy (leg 0..11 + waist 12..14 + arm 15..28)
        js = type("JS", (), {})()
        js.position = np.full(29, joint_state_val, dtype=np.float32)
        hs = type("HS", (), {})()
        hs.position_rad = np.full(2, hand_state_val, dtype=np.float32)
        return {
            "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_left_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_right_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "joint_state": js,
            "hand_state": hs,
            "cleaned": [],
            "t": 0,
        }

    def test_limiter_none_bypasses_apply(self) -> None:
        skill, _, _, _ = self._make_skill(limiter=None)
        skill.start({})
        arm = skill.step(self._obs())
        # limiter 無しでも normal に arm 14D 返る
        assert arm.shape == (14,)

    def test_limiter_apply_called_each_step(self) -> None:
        limiter = _RecordingLimiter()
        skill, _, _, _ = self._make_skill(limiter=limiter)
        skill.start({})
        for _ in range(3):
            skill.step(self._obs())
        assert limiter.apply_call_count == 3

    def test_limiter_reset_called_on_start(self) -> None:
        limiter = _RecordingLimiter()
        skill, _, _, _ = self._make_skill(limiter=limiter)
        assert limiter.reset_call_count == 0
        skill.start({})
        assert limiter.reset_call_count == 1
        skill.stop()
        skill.start({})
        assert limiter.reset_call_count == 2

    def test_measured_19d_extracted_correctly(self) -> None:
        # joint_state[12:15] → waist、joint_state[15:29] → arm、
        # hand_state → hand の順で measured 19D が組まれること
        limiter = _RecordingLimiter()
        skill, _, _, _ = self._make_skill(limiter=limiter)
        skill.start({})
        skill.step(self._obs(joint_state_val=0.7, hand_state_val=2.3))
        assert limiter.last_measured is not None
        assert limiter.last_measured.shape == (19,)
        # waist [0:3] と arm [3:17] は joint_state_val (=0.7) から埋まる
        assert np.allclose(limiter.last_measured[0:17], 0.7)
        # hand [17:19] は hand_state_val (=2.3)
        assert np.allclose(limiter.last_measured[17:19], 2.3)

    def test_limiter_output_used_for_dispatch(self) -> None:
        # limiter の apply 出力が waist/hand actuator に流れることを verify
        # (limiter が chunk[0] を単に×2 した値を返す設定にして dispatch 値と比較)
        limiter = _RecordingLimiter()
        limiter._apply_scale = 0.5  # apply 出力 = target * 0.5
        skill, _, waist, hand = self._make_skill(limiter=limiter)
        skill.start({})
        arm = skill.step(self._obs())
        # StubPolicy の chunk[0] = np.arange(19) + 0.0 = [0, 1, 2, ..., 18]
        # limiter で ×0.5 → [0, 0.5, 1.0, ..., 9.0]
        # waist = [0.0, 0.5, 1.0]
        assert waist.latest is not None
        assert abs(waist.latest[0] - 0.0) < 1e-6
        assert abs(waist.latest[1] - 0.5) < 1e-6
        assert abs(waist.latest[2] - 1.0) < 1e-6
        np.testing.assert_allclose(arm, np.arange(3, 17) * 0.5)
        assert skill.last_action is not None
        np.testing.assert_allclose(
            skill.last_action.action_chunk[0], np.arange(19) * 0.5
        )
        assert skill.last_action.metadata["motion_limiter_applied"] is True
        assert skill.last_action.metadata["raw_target_max_abs_delta"] == 9.0

    def test_measured_uses_prev_buffer_when_joint_state_missing(self) -> None:
        limiter = _RecordingLimiter()
        skill, _, _, _ = self._make_skill(limiter=limiter)
        skill.start({})
        # 1st step: joint_state OK
        skill.step(self._obs(joint_state_val=0.5, hand_state_val=1.5))
        # 2nd step: joint_state None → 前 buffer 使用
        obs = self._obs()
        obs["joint_state"] = None
        skill.step(obs)
        # 2nd measured の waist/arm slice は 0.5 (前 buffer)
        assert np.allclose(limiter.last_measured[0:17], 0.5)


class _ObservationRecordingPolicy(_StubPolicy):
    """predict() が受け取った Observation を残す。"""

    def __init__(self) -> None:
        super().__init__()
        self.observations: list = []

    def predict(self, obs):  # noqa: ANN001
        self.observations.append(obs)
        return super().predict(obs)


class _Obs:
    @staticmethod
    def make() -> dict:
        return {
            "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_left_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_right_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "joint_state": None,
            "cleaned": [],
            "t": time.monotonic_ns(),
        }


class TestSkillIdOverride:
    """slot 側で skill_id を明示できる (Issue #141 1-2 / INF-3)。"""

    def _skill(self, **kwargs):
        return RotateTableBaseVlaSkill(
            policy=_ObservationRecordingPolicy(),
            waist_actuator=MockWaistActuator(),
            hand_actuator=MockHandActuator(),
            **kwargs,
        )

    def test_class_value_is_used_by_default(self) -> None:
        skill = self._skill()
        skill.start({})
        skill.step(_Obs.make())
        assert skill._policy.observations[0].skill_id == RotateTableBaseVlaSkill.SKILL_ID == 4

    def test_override_replaces_the_class_value(self) -> None:
        """Phase K の ckpt は rotate を skill_id 0 で学習している。"""
        skill = self._skill(skill_id_override=0)
        skill.start({})
        skill.step(_Obs.make())
        assert skill._policy.observations[0].skill_id == 0

    def test_non_integer_override_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="skill_id_override must be int"):
            self._skill(skill_id_override="4")

    def test_boolean_override_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="skill_id_override must be int"):
            self._skill(skill_id_override=True)

    def test_negative_override_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            self._skill(skill_id_override=-1)


class TestTeacherRangeCorrection:
    """教師の範囲を出た関節だけ内側へ戻す (Issue #141 1-4 / D5-1)。"""

    def _skill(self, teacher_range, *, observe_only: bool = False):
        return RotateTableBaseVlaSkill(
            policy=_StubPolicy(),
            waist_actuator=MockWaistActuator(),
            hand_actuator=MockHandActuator(),
            teacher_range=teacher_range,
            teacher_range_observe_only=observe_only,
        )

    def _range(self, lower: float, upper: float):
        from inference.desktop.lower_policy.skills.teacher_range import TeacherJointRange

        return TeacherJointRange(
            np.full(14, lower), np.full(14, upper), margin_rad=0.02
        )

    def test_no_correction_without_a_range(self) -> None:
        skill = self._skill(None)
        skill.start({})
        skill.step(_Obs.make())
        assert "teacher_range_bind" not in skill.last_action.metadata

    def test_out_of_range_joint_is_pulled_inside(self) -> None:
        # _StubPolicy の chunk[0] は 0.5, 1.5, 2.5, ... なので腕は全部 1 を超える。
        skill = self._skill(self._range(-1.0, 1.0))
        skill.start({})
        arm = skill.step(_Obs.make())
        assert skill.last_action.metadata["teacher_range_bind"] is True
        assert len(skill.last_action.metadata["teacher_range_corrected_joints"]) == 14
        # limiter を付けていないので、補正後の目標がそのまま出る。
        assert float(np.max(arm)) == pytest.approx(0.98)

    def test_in_range_target_is_left_alone(self) -> None:
        skill = self._skill(self._range(-100.0, 100.0))
        skill.start({})
        skill.step(_Obs.make())
        assert skill.last_action.metadata["teacher_range_bind"] is False
        assert skill.last_action.metadata["teacher_range_corrected_joints"] == []
        assert skill.last_action.metadata["teacher_range_max_excursion_rad"] == 0.0

    def test_raw_target_keeps_the_uncorrected_model_output(self) -> None:
        """補正の前の生の指令は記録に残る (実機で drift を追えるように)。"""
        skill = self._skill(self._range(-1.0, 1.0))
        skill.start({})
        skill.step(_Obs.make())
        raw = skill.last_action.metadata["raw_target_19d"]
        assert max(raw) > 1.0

    def test_observe_only_records_range_violation_without_changing_target(self) -> None:
        skill = self._skill(self._range(-1.0, 1.0), observe_only=True)
        skill.start({})
        arm = skill.step(_Obs.make())
        metadata = skill.last_action.metadata
        assert metadata["teacher_range_bind"] is True
        assert metadata["teacher_range_observe_only"] is True
        assert metadata["teacher_range_applied"] is False
        assert len(metadata["teacher_range_would_correct_joints"]) == 14
        assert metadata["teacher_range_corrected_joints"] == []
        assert float(np.max(arm)) > 1.0
        np.testing.assert_allclose(
            arm, np.asarray(metadata["raw_target_19d"])[ARMS_SLICE]
        )

    def test_observe_only_needs_a_teacher_range(self) -> None:
        with pytest.raises(ValueError, match="requires teacher_range"):
            self._skill(None, observe_only=True)

    def test_observe_only_still_holds_a_physical_limit_violation(self) -> None:
        from inference.desktop.lower_policy.skills.motion_limiter import MotionLimiter

        lower = np.full(19, -100.0)
        upper = np.full(19, 100.0)
        upper[ARMS_SLICE.start] = 1.0
        skill = self._skill(self._range(-1.0, 1.0), observe_only=True)
        skill._motion_limiter = MotionLimiter(
            dim=19,
            arm_slice=slice(0, 17),
            hand_slice=HAND_SLICE,
            arm_velocity=100.0,
            arm_acceleration=100.0,
            hand_velocity=100.0,
            hand_acceleration=100.0,
            control_hz=30.0,
            lower=lower,
            upper=upper,
            reject_position_slice=ARMS_SLICE,
        )
        skill.start({})
        skill.step(_Obs.make())
        assert skill.last_action.metadata["teacher_range_bind"] is True
        assert skill.last_action.metadata["teacher_range_applied"] is False
        assert skill.last_action.metadata["unsafe_position_target_held"] is True


def test_boundary_base_quat_is_not_fed_to_the_model_root_pose():
    """公式 :5557 の IMU 姿勢 (yaw 任意) は model 入力に入れない。

    実機検証済みの pick_legs 入力は SDK 経路と同じ直立 proxy。経路で入力を変えない。
    """
    from types import SimpleNamespace

    from inference.desktop.lower_policy.skills.vla_skill import VlaSkill

    skill = VlaSkill.__new__(VlaSkill)
    skill._fk = None
    skill._prev_action_19d = None
    skill._prev_joint_positions = None
    skill._prev_hand_state = None
    joint_state = SimpleNamespace(
        position=np.zeros(29),
        base_quat_wxyz=np.asarray([0.7071, 0.0, 0.0, 0.7071]),
    )
    raw = skill._build_raw_state(joint_state)
    assert raw.root_pose is None


# ---- 先の予定 (Issue #188): 腕だけ予定の少し先を送る ----


class _SingleRowPreviewPolicy(_StubPolicy):
    """ensemble を中に持つ policy の形: 1 行だけ返し、先の予定は preview_target で答える。"""

    def __init__(self, preview=None) -> None:
        super().__init__(chunk_value=0.0, chunk_len=1)
        self.preview = preview
        self.asked: list[int] = []

    def preview_target(self, steps: int):
        self.asked.append(steps)
        return self.preview


class TestArmPreview:
    def _skill(self, policy, steps: int):
        # 行ごとに違う小さな値 (Mock の手は 5.4 rad で切り詰めるので、大きな値だと比べられない)
        rows, cols = policy._chunk.shape
        policy._chunk = (
            0.01 * np.arange(rows)[:, None] + 0.001 * np.arange(cols)[None, :]
        ).astype(np.float32)
        waist = MockWaistActuator()
        hand = MockHandActuator()
        skill = MoveTableBaseVlaSkill(
            policy=policy, waist_actuator=waist, hand_actuator=hand
        )
        skill.set_arm_preview_steps(steps)
        skill.start({})
        return skill, waist, hand

    @staticmethod
    def _obs() -> dict:
        return TestVlaSkillLifecycle()._obs()

    def test_zero_steps_sends_the_current_step(self) -> None:
        policy = _StubPolicy(chunk_value=0.0)
        skill, _, _ = self._skill(policy, 0)
        arm = skill.step(self._obs())
        np.testing.assert_allclose(arm, policy._chunk[0, ARMS_SLICE])
        assert skill.last_arm_preview_steps == 0

    def test_only_the_arms_come_from_k_rows_ahead(self) -> None:
        policy = _StubPolicy(chunk_value=0.0)
        skill, waist, hand = self._skill(policy, 6)
        arm = skill.step(self._obs())
        chunk = policy._chunk
        np.testing.assert_allclose(arm, chunk[6, ARMS_SLICE])
        np.testing.assert_allclose(hand.latest, chunk[0, HAND_SLICE])
        np.testing.assert_allclose(waist.latest, chunk[0, WAIST_SLICE])
        assert skill.last_arm_preview_steps == 6
        # 記録は実際に送った行、状態の入力 (前 tick の指令) は予定の今の値
        np.testing.assert_allclose(skill.last_action.action_chunk[0, ARMS_SLICE], chunk[6, ARMS_SLICE])
        np.testing.assert_allclose(skill._prev_action_19d[ARMS_SLICE], chunk[0, ARMS_SLICE])

    def test_a_short_chunk_uses_its_last_row(self) -> None:
        policy = _StubPolicy(chunk_value=0.0, chunk_len=4)
        skill, _, _ = self._skill(policy, 6)
        arm = skill.step(self._obs())
        np.testing.assert_allclose(arm, policy._chunk[3, ARMS_SLICE])
        assert skill.last_arm_preview_steps == 3

    def test_a_single_row_policy_is_asked_for_its_plan(self) -> None:
        future = np.full(ACTION_DIM_TOTAL, 7.0)
        policy = _SingleRowPreviewPolicy(preview=(future, 2))
        skill, _, hand = self._skill(policy, 6)
        arm = skill.step(self._obs())
        assert policy.asked == [6]
        np.testing.assert_allclose(arm, future[ARMS_SLICE])
        np.testing.assert_allclose(hand.latest, policy._chunk[0, HAND_SLICE])
        assert skill.last_arm_preview_steps == 2

    def test_no_plan_ahead_keeps_the_current_step(self) -> None:
        policy = _SingleRowPreviewPolicy(preview=None)
        skill, _, _ = self._skill(policy, 6)
        arm = skill.step(self._obs())
        np.testing.assert_allclose(arm, policy._chunk[0, ARMS_SLICE])
        assert skill.last_arm_preview_steps == 0
        np.testing.assert_allclose(skill._prev_action_19d[ARMS_SLICE], policy._chunk[0, ARMS_SLICE])

    def test_the_queue_path_peeks_past_the_rows_it_executes(self) -> None:
        """pick の GR00T (実行 3 行・chunk 16 行の stub): 実行する行数は変えずに先を覗く。"""
        policy = _ChunkExecutingStubPolicy(chunk_value=0.0)
        skill, _, hand = self._skill(policy, 2)
        chunk = policy._chunk
        for index in range(3):
            arm = skill.step(self._obs())
            np.testing.assert_allclose(arm, chunk[index + 2, ARMS_SLICE])
            np.testing.assert_allclose(hand.latest, chunk[index, HAND_SLICE])
        assert policy.predict_call_count == 1
        arm = skill.step(self._obs())                 # 3 行使い切って推論し直す
        assert policy.predict_call_count == 2
        np.testing.assert_allclose(arm, chunk[2, ARMS_SLICE])

    def test_the_queue_path_stops_at_the_end_of_the_chunk(self) -> None:
        policy = _ChunkExecutingStubPolicy(chunk_value=0.0)
        skill, _, _ = self._skill(policy, 20)
        arm = skill.step(self._obs())
        np.testing.assert_allclose(arm, policy._chunk[15, ARMS_SLICE])
        assert skill.last_arm_preview_steps == 15

    def test_the_plan_is_recorded_only_when_previewing(self) -> None:
        policy = _StubPolicy(chunk_value=0.0)
        skill, _, _ = self._skill(policy, 6)
        skill.step(self._obs())
        meta = skill.last_action.metadata
        assert meta["arm_preview_steps"] == 6
        assert meta["arm_preview_used_steps"] == 6
        np.testing.assert_allclose(meta["plan_target_arms"], policy._chunk[0, ARMS_SLICE], rtol=1e-6)
        plain, _, _ = self._skill(_StubPolicy(chunk_value=0.0), 0)
        plain.step(self._obs())
        assert "plan_target_arms" not in plain.last_action.metadata
        assert "arm_preview_steps" not in plain.last_action.metadata

    @pytest.mark.parametrize("bad", [-1, 1.5, True, "6"])
    def test_bad_steps_are_rejected(self, bad) -> None:
        skill, _, _ = self._skill(_StubPolicy(), 0)
        with pytest.raises(ValueError, match="arm preview steps"):
            skill.set_arm_preview_steps(bad)
