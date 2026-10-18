"""GR00T 38D → decoupled (T,25) task-space adapter の unit test。

FK は実 URDF を使う (numpy-only、GPU/model 不要)。boundary の validate_chunk
契約 (単位 quat / hands∈[-1,1] / finite / (T,25)) を再現して検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.perception.g1_urdf_fk import G1WristFK, DEFAULT_URDF_PATH
from inference.desktop.lower_policy.policies.taskspace_adapter import (
    BASE_HEIGHT_MAX_M,
    BASE_HEIGHT_MIN_M,
    DEFAULT_BASE_HEIGHT_M,
    DEX1_OPEN_VALUE,
    check_base_height,
    GROOT_ACTION_DIM,
    TASKSPACE_DIM,
    dex1_model_to_taskspace,
    groot_action_to_taskspace,
    groot_chunk_to_taskspace,
    rotation_matrix_to_quat_wxyz,
)


@pytest.fixture(scope="module")
def fk() -> G1WristFK:
    return G1WristFK.from_urdf(DEFAULT_URDF_PATH)


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class TestRotationMatrixToQuat:
    def test_identity(self) -> None:
        q = rotation_matrix_to_quat_wxyz(np.eye(3))
        assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])

    def test_unit_norm_and_w_nonneg(self) -> None:
        for a in (0.1, 1.0, 2.0, 3.0, -2.5):
            for R in (_rot_x(a), _rot_z(a)):
                q = rotation_matrix_to_quat_wxyz(R)
                assert abs(np.linalg.norm(q) - 1.0) < 1e-9
                assert q[0] >= 0.0

    def test_roundtrip_matrix(self) -> None:
        # 各分岐 (trace>0 / x / y / z 最大) を通る回転で行列を復元できる
        for R in (
            _rot_z(0.3),
            _rot_x(np.pi),  # trace<0, x 分岐
            _rot_z(np.pi),  # z or y 分岐
            _rot_x(0.5) @ _rot_z(2.0),
        ):
            q = rotation_matrix_to_quat_wxyz(R)
            assert np.allclose(_quat_to_matrix(q), R, atol=1e-9)

    def test_bad_shape(self) -> None:
        with pytest.raises(ValueError):
            rotation_matrix_to_quat_wxyz(np.eye(4))


def _organizer_hand_norm_to_dex1_q(norm: float) -> float:
    """運営 `tools/run_wbc_with_dex1.py` の `hand_norm_to_dex1_q` の写し。

    この repo の定数を使わずに書く (同じ定数を使い回すと、定数が間違っていても
    自己無矛盾に通ってしまう)。
    """
    closed_q, open_q = 0.0, -5.30
    norm = float(np.clip(norm, -1.0, 1.0))
    return closed_q + (open_q - closed_q) * (1.0 - norm) / 2.0


class TestHandMapping:
    """Issue #159 B3a-04: 運営 relay を通った後の物理角が model 値と一致すること。

    学習データの手の値は Dex1 モータの物理角 [rad] で、上限 4.5 はハードウェアの
    全開 (5.4) ではない (ラボ実測: 脚を握ると指令 1.85 / 実測 2.15、学習 2.315)。
    以前は model 4.5 を boundary -1 (= 運営の全開 5.30) に写していたので、
    会場の物理角が 1.18 倍になり、握る指令が脚より外側に出ていた。
    """

    @pytest.mark.parametrize("model_rad", [0.0, 0.5, 1.85, 2.15, 3.0, 4.5])
    def test_the_relayed_motor_angle_equals_the_model_value(self, model_rad) -> None:
        q = _organizer_hand_norm_to_dex1_q(dex1_model_to_taskspace(model_rad))
        assert -q == pytest.approx(model_rad, abs=1e-9)

    def test_a_grasp_command_stays_inside_the_leg(self) -> None:
        """握る指令 (1.85) が脚の位置 (2.15) より内側に出ること = 締め付けが残る。"""
        q = _organizer_hand_norm_to_dex1_q(dex1_model_to_taskspace(1.85))
        assert -q < 2.15

    def test_closed_is_plus_one(self) -> None:
        assert dex1_model_to_taskspace(0.0) == pytest.approx(1.0)

    def test_training_maximum_keeps_its_physical_angle(self) -> None:
        norm = dex1_model_to_taskspace(DEX1_OPEN_VALUE)
        assert -1.0 < norm < 0.0
        assert -_organizer_hand_norm_to_dex1_q(norm) == pytest.approx(DEX1_OPEN_VALUE)

    def test_safety_scaffold_can_request_physical_full_open(self) -> None:
        assert dex1_model_to_taskspace(5.30) == pytest.approx(-1.0)

    def test_clip_out_of_range(self) -> None:
        assert dex1_model_to_taskspace(10.0) == pytest.approx(
            dex1_model_to_taskspace(5.30)
        )
        assert dex1_model_to_taskspace(-3.0) == pytest.approx(1.0)


def _make_action(body29: np.ndarray, hand: tuple[float, float]) -> np.ndarray:
    root = np.array([0, 0, 0.70, 1, 0, 0, 0], dtype=np.float64)  # proxy root
    return np.concatenate([root, body29, np.asarray(hand, dtype=np.float64)])


class TestGrootActionToTaskspace:
    def _body(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # 現実的な小さめ関節角 (arm/waist は ±0.5 rad)
        return rng.uniform(-0.5, 0.5, size=29).astype(np.float64)

    def test_shape_and_dtype(self, fk) -> None:
        out = groot_action_to_taskspace(_make_action(self._body(), (4.5, 0.0)), fk)
        assert out.shape == (TASKSPACE_DIM,)
        assert out.dtype == np.float32

    def test_boundary_contract(self, fk) -> None:
        # DecoupledSink.validate_chunk が課す不変条件を再現
        out = groot_action_to_taskspace(_make_action(self._body(3), (4.5, 0.0)), fk)
        assert np.all(np.isfinite(out))
        hands = np.concatenate([out[0:2], out[2:4]])
        assert np.max(np.abs(hands)) <= 1.0 + 1e-3
        for sl in (slice(7, 11), slice(14, 18)):
            assert abs(np.linalg.norm(out[sl]) - 1.0) < 1e-2

    def test_hand_slots(self, fk) -> None:
        out = groot_action_to_taskspace(_make_action(self._body(), (4.5, 0.0)), fk)
        assert np.allclose(out[0:2], dex1_model_to_taskspace(4.5))  # left 開 (2 列同値)
        assert np.all(out[0:2] < 0.0)
        assert np.allclose(out[2:4], 1.0)  # right 全閉

    def test_lower_body_defaults_zero(self, fk) -> None:
        out = groot_action_to_taskspace(_make_action(self._body(), (2.0, 2.0)), fk)
        assert np.allclose(out[18:21], 0.0)  # navigate
        assert np.allclose(out[22:25], 0.0)  # torso_rpy

    def test_base_height_defaults_to_neutral_not_zero(self, fk) -> None:
        """col [21] の既定は 0 ではなく 0.74 であること。

        運営の adapter は col [21] をそのまま WBC へ渡す
        (`wbc_driver.run_decoupled` の `base_height_command=[[float(r[21])] ...]`)。0 は「指定なし」
        ではなく **骨盤高さ 0 m** = 床まで沈む指令になる。運営側の既定も
        `wbc_adapter/wbc_goal.py:46 DEFAULT_BASE_HEIGHT = 0.74`。
        呼出側は誰も渡していないので、ここが実際に会場へ出る値になる。
        """
        out = groot_action_to_taskspace(_make_action(self._body(), (2.0, 2.0)), fk)
        assert out[21] == pytest.approx(DEFAULT_BASE_HEIGHT_M)
        assert out[21] == pytest.approx(0.74)

    def test_lower_body_override(self, fk) -> None:
        out = groot_action_to_taskspace(
            _make_action(self._body(), (2.0, 2.0)),
            fk,
            navigate_cmd=(0.1, 0.0, 0.2),
            base_height_cmd=0.75,
            torso_rpy_cmd=(0.0, 0.0, 0.3),
        )
        assert np.allclose(out[18:21], [0.1, 0.0, 0.2])
        assert out[21] == pytest.approx(0.75)
        assert np.allclose(out[22:25], [0.0, 0.0, 0.3])

    def test_ee_frame_transform_identity_noop(self, fk) -> None:
        act = _make_action(self._body(5), (1.0, 3.0))
        base = groot_action_to_taskspace(act, fk)
        with_id = groot_action_to_taskspace(act, fk, ee_frame_transform=np.eye(4))
        assert np.allclose(base, with_id, atol=1e-6)

    def test_ee_frame_transform_translation(self, fk) -> None:
        # +z に 1.0 m 平行移動する frame 変換で EE 位置が z+1 になる
        act = _make_action(self._body(5), (1.0, 3.0))
        base = groot_action_to_taskspace(act, fk)
        tf = np.eye(4)
        tf[2, 3] = 1.0
        moved = groot_action_to_taskspace(act, fk, ee_frame_transform=tf)
        assert moved[6] == pytest.approx(base[6] + 1.0, abs=1e-5)  # left z
        assert moved[13] == pytest.approx(base[13] + 1.0, abs=1e-5)  # right z
        # 純平行移動は姿勢 (quat) を変えない
        assert np.allclose(moved[7:11], base[7:11], atol=1e-5)

    def test_bad_shape(self, fk) -> None:
        with pytest.raises(ValueError):
            groot_action_to_taskspace(np.zeros(19), fk)


class TestFKConsistency:
    """compute_ee_transforms が compute_ee_state と同一 FK chain を使うことの確認。

    ⚠️ 両者は **原点が違う**。`compute_ee_state` は model の state 入力なので学習時と
    同じ tool offset を足すが、`compute_ee_transforms` は publish する `(T,25)` 用で
    **zero tool offset (wrist_yaw_link 原点)** を返す (運営 IK が zero offset 固定、
    `WBC_RUNBOOK` §3)。以前は両者一致を assert していたが、それは publish 側が
    tool offset を足していた頃の前提で、pick の IK 残差 60x の原因そのものだった。

    同一 chain であることは「offset を足し戻すと一致する」ことで確認する。
    """

    def test_position_matches_compute_ee_state_after_adding_the_tool_offset(
        self, fk
    ) -> None:
        rng = np.random.default_rng(11)
        jp = rng.uniform(-0.6, 0.6, size=29).astype(np.float64)
        ee = fk.compute_ee_state(jp).astype(np.float64)  # (12,) tool point
        lp, lR, rp, rR = fk.compute_ee_transforms(jp)  # wrist_yaw_link 原点
        assert np.allclose(lp + lR @ fk._left_offset, ee[0:3], atol=1e-6)
        assert np.allclose(rp + rR @ fk._right_offset, ee[6:9], atol=1e-6)
        # 返り R は直交行列 (det=+1)
        for R in (lR, rR):
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


class TestGrootChunkToTaskspace:
    def test_chunk_shape(self, fk) -> None:
        rng = np.random.default_rng(7)
        chunk = np.stack(
            [_make_action(rng.uniform(-0.5, 0.5, 29), (4.5, 0.0)) for _ in range(16)]
        )
        out = groot_chunk_to_taskspace(chunk, fk)
        assert out.shape == (16, TASKSPACE_DIM)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    def test_chunk_bad_dim(self, fk) -> None:
        with pytest.raises(ValueError):
            groot_chunk_to_taskspace(np.zeros((4, GROOT_ACTION_DIM + 1)), fk)


# ---------------------------------------------------------------- 範囲の歯止め
#
# col [21] は運営側でも boundary/actions.py でも範囲を検証されない
# (hand は [-1,1]、quat は unit norm を見るのに下半身列だけ素通し)。
# 実際 0.0 が数週間どこにも引っかからずに WBC へ届いていた。
class TestBaseHeightGuard:
    def test_the_neutral_value_passes(self) -> None:
        assert check_base_height(DEFAULT_BASE_HEIGHT_M) == pytest.approx(0.74)

    def test_zero_is_rejected_with_the_reason(self) -> None:
        """**これが今回の事故そのもの。** 0 は「未指定」ではなく 0 m 指令。"""
        with pytest.raises(ValueError, match="sink the pelvis to the floor"):
            check_base_height(0.0)

    @pytest.mark.parametrize("value", [-0.1, 0.29, 1.01, 74.0, float("nan")])
    def test_implausible_values_are_rejected(self, value) -> None:
        with pytest.raises(ValueError):
            check_base_height(value)

    def test_the_bounds_themselves_pass(self) -> None:
        assert check_base_height(BASE_HEIGHT_MIN_M) == pytest.approx(BASE_HEIGHT_MIN_M)
        assert check_base_height(BASE_HEIGHT_MAX_M) == pytest.approx(BASE_HEIGHT_MAX_M)

    def test_the_guard_reaches_the_published_row(self, fk) -> None:
        """adapter を通しても効くこと (helper だけ直っていても意味がない)。"""
        with pytest.raises(ValueError):
            groot_action_to_taskspace(
                _make_action(np.zeros(29), (2.0, 2.0)), fk, base_height_cmd=0.0
            )
