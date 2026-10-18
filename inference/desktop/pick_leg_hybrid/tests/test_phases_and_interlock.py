"""区間の状態機械と把持インターロックの unit test (Issue #136)。"""

from __future__ import annotations

import pytest

from inference.desktop.pick_leg_hybrid.interlock import (
    LEFT,
    RIGHT,
    GraspInterlockConfig,
    is_grasping,
)
from inference.desktop.pick_leg_hybrid.phases import (
    EXECUTOR,
    LAST_PHASE,
    Phase,
    PhaseStateMachine,
)


class TestPhaseStateMachine:
    def test_starts_at_phase_1_with_vla(self) -> None:
        sm = PhaseStateMachine()
        assert sm.phase is Phase.APPROACH_GRASP
        assert sm.executor == "vla"
        assert sm.advance_count == 0

    def test_advances_through_all_three_phases(self) -> None:
        sm = PhaseStateMachine()
        assert sm.advance() is True
        assert sm.phase is Phase.CARRY_TO_LEFT
        assert sm.executor == "mp"
        assert sm.advance() is True
        assert sm.phase is Phase.HANDOVER_ONWARD
        assert sm.executor == "vla"
        assert sm.advance_count == 2

    def test_advance_at_last_phase_is_a_no_op(self) -> None:
        """最終区間で発火し続けても無害に握り潰す (例外にしない)。"""
        sm = PhaseStateMachine(Phase.HANDOVER_ONWARD)
        assert sm.is_last is True
        for _ in range(5):
            assert sm.advance() is False
        assert sm.phase is LAST_PHASE
        assert sm.advance_count == 0

    def test_has_no_way_to_go_backwards(self) -> None:
        """単調性の保証はここ。後戻りする API を持たないこと (D-9)。"""
        sm = PhaseStateMachine()
        sm.advance()
        public = {n for n in dir(sm) if not n.startswith("_")}
        assert not (public & {"back", "rewind", "previous", "set_phase", "regress"})

    def test_reset_returns_to_phase_1(self) -> None:
        """reset はエピソード境界専用。実行中の後戻りには使わない。"""
        sm = PhaseStateMachine()
        sm.advance()
        sm.advance()
        sm.reset()
        assert sm.phase is Phase.APPROACH_GRASP
        assert sm.advance_count == 0

    def test_executor_map_is_vla_mp_vla(self) -> None:
        """VLA の流れに MP が 1 区間だけ挟まる (D-4)。"""
        assert [EXECUTOR[p] for p in Phase] == ["vla", "mp", "vla"]


class TestGraspInterlock:
    """実測 (2114 episode) に基づく閾値の検証。"""

    def test_open_hand_following_command_is_not_grasping(self) -> None:
        """開いていて追従している = 何も噛んでいない (追従誤差 median ±0.006)。"""
        assert is_grasping([4.49, 4.48], [4.50, 4.48], RIGHT) is False

    def test_holding_the_leg_is_detected(self) -> None:
        """保持中: 実測 hand_state ≈ 2.2、指令は閉じ側 → 誤差 +0.35。"""
        assert is_grasping([2.19, 2.24], [1.84, 1.89], RIGHT) is True
        assert is_grasping([2.19, 2.24], [1.84, 1.89], LEFT) is True

    def test_closing_transient_is_not_grasping(self) -> None:
        """**追従誤差だけでは足りない。**

        閉じ始めの過渡でも実測は指令に追いつかない。実測では err>0.2 が最初に
        立つ瞬間の hand_state は median 4.46 (ほぼ全開) で、HOLD 帯に入るのは
        0.6% のみ。値の帯を AND することでこれを弾く。
        """
        # 誤差は大きい (4.46 - 3.0 = 1.46 > 0.2) が、値が全開側なので掴んでいない
        assert is_grasping([4.46, 4.46], [3.00, 3.00], RIGHT) is False

    def test_value_inside_band_but_following_is_not_grasping(self) -> None:
        """帯の中でも追従していれば掴んでいない (空振りで閉じ切った状態)。"""
        assert is_grasping([2.00, 2.00], [1.99, 1.99], RIGHT) is False

    def test_sides_are_independent(self) -> None:
        assert is_grasping([2.19, 4.49], [1.84, 4.50], LEFT) is True
        assert is_grasping([2.19, 4.49], [1.84, 4.50], RIGHT) is False

    def test_band_edges(self) -> None:
        cfg = GraspInterlockConfig()
        assert is_grasping([cfg.hold_max, 0.0], [cfg.hold_max - 0.5, 0.0], LEFT) is True
        assert is_grasping(
            [cfg.hold_max + 0.01, 0.0], [cfg.hold_max - 0.5, 0.0], LEFT
        ) is False
        assert is_grasping([cfg.hold_min, 0.0], [cfg.hold_min - 0.5, 0.0], LEFT) is True
        assert is_grasping(
            [cfg.hold_min - 0.01, 0.0], [cfg.hold_min - 0.5, 0.0], LEFT
        ) is False

    def test_rejects_malformed_input(self) -> None:
        with pytest.raises(ValueError, match="length 2"):
            is_grasping([2.2], [1.8], RIGHT)
        with pytest.raises(ValueError, match="finite"):
            is_grasping([2.2, float("nan")], [1.8, 1.8], RIGHT)
        with pytest.raises(ValueError, match="side"):
            is_grasping([2.2, 2.2], [1.8, 1.8], 7)

    def test_config_validates_its_own_bounds(self) -> None:
        with pytest.raises(ValueError, match="hold_min"):
            GraspInterlockConfig(hold_min=3.0, hold_max=1.0)
        with pytest.raises(ValueError, match="follow_error_min"):
            GraspInterlockConfig(follow_error_min=0.0)
