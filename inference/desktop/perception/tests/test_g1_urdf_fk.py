"""G1 URDF-based wrist FK の unit test (Issue #125 Phase A-5)。

- URDF parse chain 長 / joint 一貫性
- T-pose (全 zero joint) の期待 wrist position (URDF 定義から手計算)
- euler round-trip (matrix→euler→matrix identity)
- 90 度回転で expected な wrist 座標変化
- (integration) training LeRobot sample との cross-verify

Default env で pure numpy 実装のため 全 test 走る (env-gate 無し)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inference.desktop.perception.g1_urdf_fk import (
    DEFAULT_URDF_PATH,
    G1_JOINT_NAMES,
    G1WristFK,
    LEFT_WRIST_LINK,
    LEFT_WRIST_TOOL_OFFSET_M,
    RIGHT_WRIST_LINK,
    RIGHT_WRIST_TOOL_OFFSET_M,
    ROOT_LINK,
    _axis_angle_matrix,
    _euler_xyz_matrix,
    _matrix_to_euler_xyz,
)


# repo root からの relative path (test は repo root で走る想定)
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
URDF_PATH = REPO_ROOT / DEFAULT_URDF_PATH


@pytest.fixture(scope="module")
def fk() -> G1WristFK:
    if not URDF_PATH.exists():
        pytest.skip(f"URDF not found at {URDF_PATH}")
    return G1WristFK.from_urdf(str(URDF_PATH))


# ---- rotation math primitives ---- #


class TestEulerXyzMatrix:
    def test_identity(self) -> None:
        R = _euler_xyz_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_yaw_90deg(self) -> None:
        # yaw=π/2: Rz(π/2) = [[0,-1,0],[1,0,0],[0,0,1]]
        R = _euler_xyz_matrix(0.0, 0.0, np.pi / 2)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_orthogonal(self) -> None:
        R = _euler_xyz_matrix(0.5, -0.3, 1.2)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestMatrixToEulerXyz:
    @pytest.mark.parametrize(
        "roll,pitch,yaw",
        [
            (0.0, 0.0, 0.0),
            (0.5, -0.3, 1.2),
            (-1.5, 0.4, 0.7),
            (0.1, 0.0, -0.9),
        ],
    )
    def test_round_trip(self, roll: float, pitch: float, yaw: float) -> None:
        R = _euler_xyz_matrix(roll, pitch, yaw)
        rr, pp, yy = _matrix_to_euler_xyz(R)
        R2 = _euler_xyz_matrix(rr, pp, yy)
        np.testing.assert_allclose(R2, R, atol=1e-10)


class TestAxisAngleMatrix:
    def test_zero_axis(self) -> None:
        R = _axis_angle_matrix(np.array([0.0, 0.0, 0.0]), 1.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_z_axis_90deg(self) -> None:
        R = _axis_angle_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_x_axis_90deg(self) -> None:
        R = _axis_angle_matrix(np.array([1.0, 0.0, 0.0]), np.pi / 2)
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-10)


# ---- URDF parse ---- #


class TestUrdfChain:
    def test_urdf_file_exists(self) -> None:
        assert URDF_PATH.exists(), f"URDF not found at {URDF_PATH}"

    def test_left_chain_length(self, fk: G1WristFK) -> None:
        # waist 3 + left_arm 7 = 10 joints
        assert len(fk._left_chain) == 10

    def test_right_chain_length(self, fk: G1WristFK) -> None:
        assert len(fk._right_chain) == 10

    def test_left_chain_joint_names(self, fk: G1WristFK) -> None:
        # 順: waist_yaw, waist_roll, waist_pitch, left_shoulder_pitch,
        # left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll,
        # left_wrist_pitch, left_wrist_yaw
        names = [cj.name for cj in fk._left_chain]
        assert names == [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]

    def test_left_chain_joint_indices(self, fk: G1WristFK) -> None:
        # G1_JOINT_NAMES 内の SDK index と一致
        indices = [cj.joint_index for cj in fk._left_chain]
        assert indices == [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    def test_right_chain_joint_indices(self, fk: G1WristFK) -> None:
        indices = [cj.joint_index for cj in fk._right_chain]
        assert indices == [12, 13, 14, 22, 23, 24, 25, 26, 27, 28]


# ---- FK: T-pose expected values ---- #


class TestFkTpose:
    """全 zero joint (T-pose) で URDF の fixed origin を積んだ位置と一致することを検証。"""

    def test_shape_and_dtype(self, fk: G1WristFK) -> None:
        state = fk.compute_ee_state(np.zeros(29))
        assert state.shape == (12,)
        assert state.dtype == np.float32

    def test_reject_bad_shape(self, fk: G1WristFK) -> None:
        with pytest.raises(ValueError, match="joint_positions must be"):
            fk.compute_ee_state(np.zeros(28))

    def test_wrist_positions_reasonable(self, fk: G1WristFK) -> None:
        # URDF T-pose (全 zero joint) = 腕が下垂した姿勢。wrist は肩から下がる。
        state = fk.compute_ee_state(np.zeros(29))
        left_xyz = state[0:3]
        right_xyz = state[6:9]
        # Left wrist は Y+ (左)、Right wrist は Y- (右) の側にある
        assert left_xyz[1] > 0, f"left_xyz.y should be >0, got {left_xyz}"
        assert right_xyz[1] < 0, f"right_xyz.y should be <0, got {right_xyz}"
        # Issue #132 Phase A: LEFT/RIGHT 別 tool offset (BitRobot GT fit 由来、
        # 左右非対称) を導入したので URDF 左右対称性から少しズレる。fit 実測
        # asymmetry ~10cm を許容する閾値 (0.05 → 0.15) に緩和。原因は BitRobot
        # 側 SDK teleop record の Dex1 grasp point 定義が左右で微妙に違う可能性。
        assert abs(abs(left_xyz[1]) - abs(right_xyz[1])) < 0.15
        assert abs(left_xyz[0] - right_xyz[0]) < 0.15
        # z: 腕下垂 T-pose なので pelvis 高さ近傍 (肩は pelvis から +0.5m 前後、腕長 ~0.4m、
        # ただし G1 URDF は shoulder pitch rpy に前傾入っているので実測ベース)
        assert -0.3 < left_xyz[2] < 0.5, f"left_xyz.z out of range: {left_xyz}"
        assert -0.3 < right_xyz[2] < 0.5, f"right_xyz.z out of range: {right_xyz}"
        assert abs(left_xyz[2] - right_xyz[2]) < 0.15

    def test_reports_training_tool_point_not_wrist_link_origin(
        self, fk: G1WristFK
    ) -> None:
        # Issue #132 Phase A: LEFT/RIGHT 別 tool offset (BitRobot LeRobot dataset の
        # GT `ee_state` との 3-dataset fit 由来、左右非対称)。
        joints = np.zeros(29)
        reported = fk.compute_ee_state(joints)
        left_link = fk._fk_chain(fk._left_chain, joints)
        right_link = fk._fk_chain(fk._right_chain, joints)
        np.testing.assert_allclose(
            reported[:3],
            left_link[:3, 3] + left_link[:3, :3] @ LEFT_WRIST_TOOL_OFFSET_M,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            reported[6:9],
            right_link[:3, 3] + right_link[:3, :3] @ RIGHT_WRIST_TOOL_OFFSET_M,
            atol=1e-7,
        )


# ---- FK: Constructor arg (Issue #132 Phase A: skill 別 tool offset override) ---- #


class TestSkillWristToolOffsetOverride:
    """Constructor / from_urdf の left_tool_offset / right_tool_offset 引数の
    後方互換 + override 動作を verify (Issue #132 Phase A)。"""

    def test_default_matches_module_constants(self, fk: G1WristFK) -> None:
        # fixture の fk は from_urdf(...) だけで construct 済 (offset 未指定)。
        # 内部 attribute (`_left_offset` / `_right_offset`) が module default と一致。
        np.testing.assert_array_equal(fk._left_offset, LEFT_WRIST_TOOL_OFFSET_M)
        np.testing.assert_array_equal(fk._right_offset, RIGHT_WRIST_TOOL_OFFSET_M)

    def test_override_via_from_urdf(self) -> None:
        # rotate_table_base skill 用 offset (skill_config 由来値) を渡した instance
        # で compute_ee_state すると module default と違う位置が返る。
        rot_left = np.array([0.091, 0.010, 0.085], dtype=np.float64)
        rot_right = np.array([0.119, -0.078, -0.023], dtype=np.float64)
        override_fk = G1WristFK.from_urdf(
            urdf_path=str(URDF_PATH),
            left_tool_offset=rot_left,
            right_tool_offset=rot_right,
        )
        default_fk = G1WristFK.from_urdf(urdf_path=str(URDF_PATH))
        jp = np.zeros(29)
        s_override = override_fk.compute_ee_state(jp)
        s_default = default_fk.compute_ee_state(jp)
        # module default と rot 独自 offset の差は L2 で最低数 mm 以上 (rot と avg
        # が同一値でなければ)。等しくないことだけ確認 (数値は fit 由来なので不変)。
        assert not np.allclose(s_override[:3], s_default[:3])
        assert not np.allclose(s_override[6:9], s_default[6:9])
        # override_fk が保持する offset が渡した値と一致 (copy されて mutation 隔離)
        np.testing.assert_array_equal(override_fk._left_offset, rot_left)
        np.testing.assert_array_equal(override_fk._right_offset, rot_right)

    def test_override_reject_wrong_shape(self, fk: G1WristFK) -> None:
        with pytest.raises(ValueError, match="left_tool_offset must be"):
            G1WristFK.from_urdf(
                urdf_path=str(URDF_PATH),
                left_tool_offset=np.zeros(2),
            )
        with pytest.raises(ValueError, match="right_tool_offset must be"):
            G1WristFK.from_urdf(
                urdf_path=str(URDF_PATH),
                right_tool_offset=np.zeros((3, 3)),
            )

    def test_override_reject_non_finite(self) -> None:
        with pytest.raises(ValueError, match="left_tool_offset must be finite"):
            G1WristFK.from_urdf(
                urdf_path=str(URDF_PATH),
                left_tool_offset=np.array([np.nan, 0, 0]),
            )

    def test_override_copies_input(self) -> None:
        # 外部から mutation されても instance の offset は変わらない (copy 済確認)。
        src = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        fk_local = G1WristFK.from_urdf(urdf_path=str(URDF_PATH), left_tool_offset=src)
        src[0] = 99.0
        assert fk_local._left_offset[0] == 0.1


# ---- FK: known-motion sanity ---- #


class TestFkKnownMotion:
    """既知の joint config で expected な wrist 変化が起きることを sanity check。"""

    def test_waist_yaw_rotates_wrist_around_z(self, fk: G1WristFK) -> None:
        # waist_yaw (index 12) = 90° = 左手が前方 (+X) から左 (+Y) 側にほぼ回転
        jp = np.zeros(29)
        jp[12] = np.pi / 2  # waist_yaw
        state = fk.compute_ee_state(jp)
        # 元 T-pose の (approx) left_xyz は (small_x, positive_y, positive_z)
        # 90° yaw 回転後: 元の +Y が -X 方向に、元の +X が +Y 方向に → 新 x < 0 が期待
        assert state[0] < -0.1, (
            f"after waist_yaw +90°, left_x should be <-0.1, got {state[0:3]}"
        )

    def test_elbow_bend_reduces_wrist_reach(self, fk: G1WristFK) -> None:
        # left_elbow (index 18) を 90° 曲げる = wrist が肘に近付く = |wrist - shoulder| が短くなる
        jp0 = np.zeros(29)
        jp1 = np.zeros(29)
        jp1[18] = np.pi / 2  # left elbow

        state0 = fk.compute_ee_state(jp0)
        state1 = fk.compute_ee_state(jp1)
        # 単純に xyz の "変化" があること
        dxyz = np.linalg.norm(state1[0:3] - state0[0:3])
        assert dxyz > 0.1, f"elbow bend should move wrist noticeably, got diff={dxyz}"


# ---- FK: numerical stability / repeatability ---- #


class TestFkStability:
    def test_deterministic(self, fk: G1WristFK) -> None:
        rng = np.random.default_rng(seed=42)
        jp = rng.uniform(-0.5, 0.5, size=29)
        s1 = fk.compute_ee_state(jp)
        s2 = fk.compute_ee_state(jp)
        np.testing.assert_array_equal(s1, s2)

    def test_finite(self, fk: G1WristFK) -> None:
        rng = np.random.default_rng(seed=7)
        for _ in range(20):
            jp = rng.uniform(-1.5, 1.5, size=29)
            state = fk.compute_ee_state(jp)
            assert np.all(np.isfinite(state)), f"non-finite output for {jp}"


# ---- (integration) Training dataset cross-verify ---- #


@pytest.mark.integration
class TestFkCrossVerifyTrainingData:
    """Training LeRobot episode の GT `ee_state` と numpy FK の一致度を verify
    (Issue #132 Phase A、3-dataset fit 由来の LEFT/RIGHT_WRIST_TOOL_OFFSET_M
    hardcode の acceptance test)。

    データ path (Sakura 想定):
      - /nvme/rotate_table_base/data/chunk-000/file-000.parquet
      - /nvme/task7/data/chunk-000/file-000.parquet
      - /nvme/task0_chunks/chunk0/data/chunk-000/file-000.parquet

    存在する dataset だけ run、全部無ければ skip。Local で走らせるには HF hub
    から `Team-RAMEN/IROS2026_RAMEN_HARA_*_v1` を pull。Marker
    `@pytest.mark.integration` gate なので default pytest では collect のみ、
    `pytest -m integration ...` で run。

    合格 threshold (mean L2 error < 100mm): fit の std ~3-9cm + 3-dataset avg
    offset vs 各 dataset 独自 offset の差 ~1-4cm を許容。**mm 精度は本 test で
    は保証しない**、Orin `/g1/ee_state` topic 追加 (Issue #63 A-2 後) で完全一致。
    """

    DATASET_PATHS = (
        (
            "rotate_table_base",
            "/nvme/rotate_table_base/data/chunk-000/file-000.parquet",
        ),
        ("task_7", "/nvme/task7/data/chunk-000/file-000.parquet"),
        ("task_0", "/nvme/task0_chunks/chunk0/data/chunk-000/file-000.parquet"),
    )
    N_SAMPLE = 100
    # Threshold rationale (Issue #132 Phase A、real GT に対する empirical bound):
    # - DEFAULT (3-dataset avg offset): rotate right mean ~118mm、task_7 right ~74mm、
    #   task_0 right ~78mm。5cm hardcode 時代 (mean 137-187mm) より明確改善だが、
    #   fit std 3-9cm の inherent variability + avg-vs-個別 offset 差 = ~118mm 到達。
    #   閾値 150mm = 実測 max より少し余裕。
    # - ROTATE OVERRIDE (rotate 独自 fit offset): rotate Left std 3D L2 ≈ 99mm、
    #   Right std 3D L2 ≈ 124mm、mean 誤差期待は std に近い (Gaussian で 0.8σ 相当)。
    #   閾値 130mm = Right std upper bound を許容、Left でも余裕 (実測 mean ~93mm)。
    MAX_MEAN_ERROR_M_DEFAULT = 0.15
    MAX_MEAN_ERROR_M_ROTATE_OVERRIDE = 0.13

    def _compute_mean_errors(
        self, fk_instance: G1WristFK, path: str
    ) -> tuple[float, float]:
        pd = pytest.importorskip("pandas")
        df = pd.read_parquet(path)
        n_total = len(df)
        stride = max(1, n_total // self.N_SAMPLE)
        diffs_L, diffs_R = [], []
        for idx in range(0, n_total, stride):
            row = df.iloc[idx]
            q29 = np.asarray(
                row["observation.state.robot_q_current"], dtype=np.float32
            )[:29]
            gt = np.asarray(row["observation.state.ee_state"], dtype=np.float32)
            pred = fk_instance.compute_ee_state(q29)
            diffs_L.append(np.linalg.norm(pred[:3] - gt[:3]))
            diffs_R.append(np.linalg.norm(pred[6:9] - gt[6:9]))
        return float(np.mean(diffs_L)), float(np.mean(diffs_R))

    def _available_datasets(self) -> list[tuple[str, str]]:
        return [
            (name, path) for name, path in self.DATASET_PATHS if Path(path).exists()
        ]

    def test_default_offset_matches_lerobot_ee_state(self, fk: G1WristFK) -> None:
        """Default FK (module-level 3-dataset avg offset) が全 skill dataset で
        threshold 内に収まる (avg offset の全 skill 共通 fallback として妥当性)。"""
        available = self._available_datasets()
        if not available:
            pytest.skip(
                f"cross-verify data not available at any of "
                f"{[p for _, p in self.DATASET_PATHS]}; run on Sakura (/nvme/*) or"
                " pull LeRobot HF dataset locally"
            )
        for name, path in available:
            mean_L, mean_R = self._compute_mean_errors(fk, path)
            assert mean_L < self.MAX_MEAN_ERROR_M_DEFAULT, (
                f"{name} (default offset): mean left FK-GT L2 = {mean_L * 1000:.1f}mm "
                f"> {self.MAX_MEAN_ERROR_M_DEFAULT * 1000:.0f}mm threshold"
            )
            assert mean_R < self.MAX_MEAN_ERROR_M_DEFAULT, (
                f"{name} (default offset): mean right FK-GT L2 = {mean_R * 1000:.1f}mm "
                f"> {self.MAX_MEAN_ERROR_M_DEFAULT * 1000:.0f}mm threshold"
            )

    def test_rotate_override_improves_over_default(self) -> None:
        """rotate_table_base 独自 offset (skill_config.yaml 由来) で FK を構築すると、
        rotate_table_base_merged_v1 dataset での GT 誤差が default より小さくなる。
        (Issue #132 Phase A skill 別 offset framework の acceptance)"""
        rotate_path = "/nvme/rotate_table_base/data/chunk-000/file-000.parquet"
        if not Path(rotate_path).exists():
            pytest.skip(f"rotate_table_base parquet not available at {rotate_path}")
        # skill_config.yaml:skills.rotate_table_base.wrist_tool_offset と同値
        rotate_left = np.array([0.091, 0.010, 0.085], dtype=np.float64)
        rotate_right = np.array([0.119, -0.078, -0.023], dtype=np.float64)
        override_fk = G1WristFK.from_urdf(
            urdf_path=str(URDF_PATH),
            left_tool_offset=rotate_left,
            right_tool_offset=rotate_right,
        )
        default_fk = G1WristFK.from_urdf(urdf_path=str(URDF_PATH))
        mean_L_ov, mean_R_ov = self._compute_mean_errors(override_fk, rotate_path)
        mean_L_def, mean_R_def = self._compute_mean_errors(default_fk, rotate_path)
        # Override 独自 threshold 内
        assert mean_L_ov < self.MAX_MEAN_ERROR_M_ROTATE_OVERRIDE, (
            f"rotate override: mean left FK-GT L2 = {mean_L_ov * 1000:.1f}mm > "
            f"{self.MAX_MEAN_ERROR_M_ROTATE_OVERRIDE * 1000:.0f}mm threshold"
        )
        assert mean_R_ov < self.MAX_MEAN_ERROR_M_ROTATE_OVERRIDE, (
            f"rotate override: mean right FK-GT L2 = {mean_R_ov * 1000:.1f}mm > "
            f"{self.MAX_MEAN_ERROR_M_ROTATE_OVERRIDE * 1000:.0f}mm threshold"
        )
        # Override が default より確実に improve (framework の意義 verify)
        assert mean_L_ov < mean_L_def, (
            f"rotate override should improve over default: override L={mean_L_ov * 1000:.1f}mm "
            f">= default L={mean_L_def * 1000:.1f}mm"
        )
        assert mean_R_ov < mean_R_def, (
            f"rotate override should improve over default: override R={mean_R_ov * 1000:.1f}mm "
            f">= default R={mean_R_def * 1000:.1f}mm"
        )


class TestPublishedEeHasNoToolOffset:
    """`compute_ee_transforms` は **publish する (T,25) 用**なので tool offset を足さない。

    運営 WBC の decoupled IK は zero tool offset (生の wrist_yaw_link 原点) 固定で
    我々の `(T,25)` を解く (`WBC_RUNBOOK` §3)。5cm の tool offset を足して publish すると
    運営 IK 期待点との系統偏りになり、2026-09-14 dry-run の **pick IK 残差 60x (≈6cm)**
    がまさにこれだった。しかも **エラーは出ず腕が違う場所へ行くだけ**なので、
    ここを守るのは test しかない。

    一方 model への state 入力 (`compute_ee_state`) は学習時と同じ offset を**維持**する。
    両者で意味が違うので、片方を直すときにもう片方を巻き込まないこと。
    """

    def test_transforms_return_the_bare_wrist_yaw_link_origin(self, fk) -> None:
        jp = np.zeros(29, dtype=np.float64)
        left_pos, left_R, right_pos, right_R = fk.compute_ee_transforms(jp)

        # chain の素の原点 (offset を足す前) と一致すること。
        t_left = fk._fk_chain(fk._left_chain, jp)
        t_right = fk._fk_chain(fk._right_chain, jp)
        np.testing.assert_allclose(left_pos, t_left[:3, 3], atol=1e-12)
        np.testing.assert_allclose(right_pos, t_right[:3, 3], atol=1e-12)
        np.testing.assert_allclose(left_R, t_left[:3, :3], atol=1e-12)
        np.testing.assert_allclose(right_R, t_right[:3, :3], atol=1e-12)

    def test_state_input_still_carries_the_tool_offset(self, fk) -> None:
        """state 側まで zero にしてしまうと学習時と食い違う。分離を守る。"""
        jp = np.zeros(29, dtype=np.float64)
        left_pos, _, right_pos, _ = fk.compute_ee_transforms(jp)
        ee_state = fk.compute_ee_state(jp)  # (12,) left xyz+euler / right xyz+euler

        # offset が非ゼロである限り、publish 位置と state 位置は一致してはいけない。
        if np.linalg.norm(fk._left_offset) > 0:
            assert not np.allclose(ee_state[0:3], left_pos, atol=1e-9), (
                "state 入力が publish と同じ = tool offset が state 側からも消えている"
            )
        if np.linalg.norm(fk._right_offset) > 0:
            assert not np.allclose(ee_state[6:9], right_pos, atol=1e-9), (
                "state 入力が publish と同じ = tool offset が state 側からも消えている"
            )

    def test_the_offset_gap_is_the_configured_tool_offset(self, fk) -> None:
        """差分がちょうど設定された tool offset (回転込み) であること。

        `compute_ee_state` は float32 を返すので、許容は float32 相当 (1e-6) で見る。
        """
        jp = np.zeros(29, dtype=np.float64)
        left_pos, left_R, right_pos, right_R = fk.compute_ee_transforms(jp)
        ee_state = fk.compute_ee_state(jp).astype(np.float64)

        np.testing.assert_allclose(
            ee_state[0:3] - left_pos, left_R @ fk._left_offset, atol=1e-6
        )
        np.testing.assert_allclose(
            ee_state[6:9] - right_pos, right_R @ fk._right_offset, atol=1e-6
        )


class TestOrganizerIkKinematics:
    """publish 用の FK は運営 IK と同じ運動学 (Issue #164)。

    運営の IK / WBC は `g1_29dof_with_hand.urdf` で解く (運営機の実物 md5
    093c36ba... = NVIDIA GR00T-WholeBodyControl 版)。同梱は腰・腕の joint が一致する
    Unitree 版 (unitree_ros 9926cc2)。学習用の mode_15 とは腕の鎖が 4 か所違う。
    """

    def test_bundled_file_is_the_pinned_unitree_copy(self):
        import hashlib

        from inference.desktop.perception.g1_urdf_fk import ORGANIZER_IK_URDF_PATH

        digest = hashlib.md5(Path(ORGANIZER_IK_URDF_PATH).read_bytes()).hexdigest()
        assert digest == "2bad065c162186a19222ec81f99b91b6"

    def test_arm_chain_differs_from_the_training_urdf_exactly_where_expected(self):
        from inference.desktop.perception.g1_urdf_fk import (
            ORGANIZER_IK_URDF_PATH,
            _parse_urdf_chain,
        )

        index = {name: i for i, name in enumerate(G1_JOINT_NAMES)}
        for link, side in ((LEFT_WRIST_LINK, "left"), (RIGHT_WRIST_LINK, "right")):
            ours = _parse_urdf_chain(DEFAULT_URDF_PATH, link, index)
            organizer = _parse_urdf_chain(ORGANIZER_IK_URDF_PATH, link, index)
            assert [c.name for c in ours] == [c.name for c in organizer]
            differing = {
                a.name
                for a, b in zip(ours, organizer)
                if not np.allclose(a.fixed_T, b.fixed_T, atol=1e-9)
                or not np.allclose(a.axis, b.axis)
            }
            assert differing == {
                "waist_roll_joint",
                "waist_pitch_joint",
                f"{side}_shoulder_pitch_joint",
                f"{side}_wrist_yaw_joint",
            }

    def test_wrist_origin_differs_by_about_five_millimetres(self):
        from inference.desktop.perception.g1_urdf_fk import ORGANIZER_IK_URDF_PATH

        ours = G1WristFK.from_urdf(DEFAULT_URDF_PATH)
        organizer = G1WristFK.from_urdf(ORGANIZER_IK_URDF_PATH)
        zero = np.zeros(29)
        lo, lr, ro, rr = ours.compute_ee_transforms(zero)
        ln, lrn, rn, rrn = organizer.compute_ee_transforms(zero)
        # wrist_yaw_joint の origin x が 0.051 (mode_15) vs 0.046 (運営) = 前腕方向に 5 mm
        assert np.linalg.norm(lo - ln) == pytest.approx(0.005, abs=1e-4)
        assert np.linalg.norm(ro - rn) == pytest.approx(0.005, abs=1e-4)
        np.testing.assert_allclose(lr, lrn, atol=1e-12)
        np.testing.assert_allclose(rr, rrn, atol=1e-12)
