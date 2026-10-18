"""fk.py の unit test (Issue #129 Phase B、2026-08-31)。

- URDF parse OK (default path)
- Forward output shape / batch shape
- Zero action で pelvis 直下の合理的 wrist 位置
- Autograd 通過 (backward で loss.grad が 0 でない)
- 19D layout の gripper 部分 (index 17,18) は wrist chain に影響しない
- `assemble_action19` の shape / concat 検証
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from model.ramen_ori.fk import (
    ACTION19_DIM,
    DEFAULT_URDF_PATH,
    G1_JOINT_NAMES,
    LEFT_WRIST_LINK,
    LEFT_WRIST_TOOL_OFFSET_M,
    RIGHT_WRIST_LINK,
    RIGHT_WRIST_TOOL_OFFSET_M,
    G1WristFKTorch,
    _ACTION19_TO_SDK29_INDEX,
    _parse_urdf_chain,
    assemble_action19,
)


@pytest.fixture(scope="module")
def fk() -> G1WristFKTorch:
    """URDF parse cost が高いので module scope で share。"""
    return G1WristFKTorch.from_default_urdf()


class TestUrdfExists:
    def test_default_urdf_file_exists(self) -> None:
        p = Path(DEFAULT_URDF_PATH)
        assert p.is_file(), f"URDF not found at {p}"

    def test_default_urdf_is_local_to_ramen_ori(self) -> None:
        """URDF は model/ramen_ori/assets/urdf/ 配下 (inference/ 依存なし、Phase B copy 済)。"""
        p = Path(DEFAULT_URDF_PATH).resolve()
        assert "model/ramen_ori/assets/urdf" in str(p), (
            f"URDF path should be under model/ramen_ori/assets/urdf/, got {p}"
        )


class TestUrdfParse:
    def test_left_chain_ends_at_left_wrist(self) -> None:
        joint_map = {n: i for i, n in enumerate(G1_JOINT_NAMES)}
        chain = _parse_urdf_chain(DEFAULT_URDF_PATH, LEFT_WRIST_LINK, joint_map)
        assert len(chain) == 10, (
            f"left wrist chain expected 10 joints (waist 3 + arm 7), got {len(chain)}"
        )
        # chain 最初は waist、最後は left_wrist_yaw
        assert chain[0].name == "waist_yaw_joint"
        assert chain[-1].name == "left_wrist_yaw_joint"

    def test_right_chain_ends_at_right_wrist(self) -> None:
        joint_map = {n: i for i, n in enumerate(G1_JOINT_NAMES)}
        chain = _parse_urdf_chain(DEFAULT_URDF_PATH, RIGHT_WRIST_LINK, joint_map)
        assert len(chain) == 10
        assert chain[0].name == "waist_yaw_joint"
        assert chain[-1].name == "right_wrist_yaw_joint"


class TestForwardShape:
    def test_forward_zero_action_returns_two_3d_pos(self, fk: G1WristFKTorch) -> None:
        q = torch.zeros(ACTION19_DIM)
        left, right = fk(q)
        assert left.shape == (3,)
        assert right.shape == (3,)

    @pytest.mark.parametrize("batch_shape", [(2,), (3, 4), (1, 16)])
    def test_forward_batch_shapes(
        self, fk: G1WristFKTorch, batch_shape: tuple[int, ...]
    ) -> None:
        q = torch.zeros(*batch_shape, ACTION19_DIM)
        left, right = fk(q)
        assert left.shape == (*batch_shape, 3)
        assert right.shape == (*batch_shape, 3)

    def test_forward_raises_on_wrong_dim(self, fk: G1WristFKTorch) -> None:
        q_bad = torch.zeros(16)
        with pytest.raises(ValueError, match=str(ACTION19_DIM)):
            fk(q_bad)


class TestForwardValues:
    def test_zero_action_wrist_pos_is_reasonable(self, fk: G1WristFKTorch) -> None:
        """Zero action (T-pose に近い home) の wrist 位置が pelvis frame で
        妥当な range (arm 長 ~0.5m 級) にあることを sanity check。"""
        q = torch.zeros(ACTION19_DIM)
        left, right = fk(q)
        # pelvis frame で ~0.5m の距離、左右反転
        assert 0.05 < abs(left[0]) < 2.0 or 0.05 < abs(left[1]) < 2.0 or 0.05 < abs(left[2]) < 2.0
        assert 0.05 < abs(right[0]) < 2.0 or 0.05 < abs(right[1]) < 2.0 or 0.05 < abs(right[2]) < 2.0
        # 左右対称性 (waist=0 なら x 座標 or y 座標が符号反転する期待、URDF 構造依存)
        # 厳密検証は URDF trust だが少なくとも右手/左手は違う位置
        assert not torch.allclose(left, right, atol=1e-3), "left/right wrist should differ"

    def test_gripper_indices_do_not_affect_wrist_pos(self, fk: G1WristFKTorch) -> None:
        """19D の gripper 部分 (index 17,18) は wrist chain 不参加 → 変更しても
        wrist 3D 位置は変わらない。"""
        q1 = torch.zeros(ACTION19_DIM)
        q2 = q1.clone()
        q2[17] = 1.0
        q2[18] = -1.0
        left1, right1 = fk(q1)
        left2, right2 = fk(q2)
        torch.testing.assert_close(left1, left2)
        torch.testing.assert_close(right1, right2)


class TestAutograd:
    def test_gradient_flows_through_fk(self, fk: G1WristFKTorch) -> None:
        """FK output で計算した loss の gradient が q_action19 の非 gripper 部分に流れる。"""
        q = torch.zeros(ACTION19_DIM, requires_grad=True)
        left, right = fk(q)
        loss = (left ** 2).sum() + (right ** 2).sum()
        loss.backward()
        assert q.grad is not None
        # waist + arms 部分 (index 0..16) は gradient あり
        assert torch.any(q.grad[:17] != 0.0), "gradient should flow to waist/arms"

    def test_no_gradient_on_gripper_indices(self, fk: G1WristFKTorch) -> None:
        """Gripper 部分 (index 17,18) は wrist chain 不参加 → gradient は 0。"""
        q = torch.zeros(ACTION19_DIM, requires_grad=True)
        left, right = fk(q)
        loss = (left ** 2).sum() + (right ** 2).sum()
        loss.backward()
        assert q.grad is not None
        torch.testing.assert_close(q.grad[17:19], torch.zeros(2))


class TestAssembleAction19:
    def test_concat_shape(self) -> None:
        waist = torch.zeros(2, 3)
        arms_hand = torch.zeros(2, 16)
        out = assemble_action19(waist, arms_hand)
        assert out.shape == (2, 19)

    def test_concat_batch_shape(self) -> None:
        waist = torch.zeros(4, 16, 3)
        arms_hand = torch.zeros(4, 16, 16)
        out = assemble_action19(waist, arms_hand)
        assert out.shape == (4, 16, 19)

    def test_concat_values_preserved(self) -> None:
        waist = torch.tensor([[1.0, 2.0, 3.0]])
        arms_hand = torch.arange(16, dtype=torch.float32).unsqueeze(0)
        out = assemble_action19(waist, arms_hand)
        torch.testing.assert_close(out[0, :3], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(out[0, 3:], torch.arange(16, dtype=torch.float32))

    def test_raises_on_wrong_waist_dim(self) -> None:
        waist = torch.zeros(2, 2)  # 3 でない
        arms_hand = torch.zeros(2, 16)
        with pytest.raises(ValueError, match="waist_3 last dim must be 3"):
            assemble_action19(waist, arms_hand)

    def test_raises_on_wrong_arms_hand_dim(self) -> None:
        waist = torch.zeros(2, 3)
        arms_hand = torch.zeros(2, 15)  # 16 でない
        with pytest.raises(ValueError, match="arms_hand_16 last dim must be 16"):
            assemble_action19(waist, arms_hand)


class TestFKGradientEndToEnd:
    """L4 loss で想定される使い方: pred_arms_hand を最適化変数、waist を known で FK に渡す。"""

    def test_gradient_flows_only_through_arms_hand(self, fk: G1WristFKTorch) -> None:
        pred_arms_hand = torch.zeros(2, 16, requires_grad=True)
        waist_teacher = torch.zeros(2, 3)  # no grad
        q19 = assemble_action19(waist_teacher, pred_arms_hand)
        left, right = fk(q19)
        # L4 loss 相当 (teacher EE を 0 と仮定)
        loss = (left ** 2).sum() + (right ** 2).sum()
        loss.backward()
        assert pred_arms_hand.grad is not None
        # 少なくとも arms 部分 (index 0..13 in 16D、= 19D の 3..16) には gradient
        # gripper 部分 (index 14,15 in 16D、= 19D の 17,18) は wrist chain 不参加 → 0
        assert torch.any(pred_arms_hand.grad[..., :14] != 0.0), (
            "gradient should flow to arms in 16D action"
        )
        torch.testing.assert_close(pred_arms_hand.grad[..., 14:16], torch.zeros(2, 2))


class TestToolOffset:
    """Issue #134: tool_offset の init / default / override validation。"""

    def test_default_offset_matches_module_constants(self, fk: G1WristFKTorch) -> None:
        """引数無し init 時、buffer が module 定数と bit-equal。"""
        left_buf = fk._left_tool_offset.detach().cpu().numpy()
        right_buf = fk._right_tool_offset.detach().cpu().numpy()
        np.testing.assert_allclose(left_buf, LEFT_WRIST_TOOL_OFFSET_M, atol=1e-6)
        np.testing.assert_allclose(right_buf, RIGHT_WRIST_TOOL_OFFSET_M, atol=1e-6)

    def test_override_offset_tensor_reflected_in_buffer(self) -> None:
        """override 引数 (tensor) が buffer に反映される。"""
        custom_left = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        custom_right = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32)
        fk_override = G1WristFKTorch.from_default_urdf(
            left_tool_offset=custom_left, right_tool_offset=custom_right
        )
        torch.testing.assert_close(fk_override._left_tool_offset, custom_left)
        torch.testing.assert_close(fk_override._right_tool_offset, custom_right)

    def test_override_offset_accepts_numpy_ndarray(self) -> None:
        """override 引数が np.ndarray でも受理される (numpy → tensor 変換)。"""
        custom = np.array([0.05, -0.05, 0.1], dtype=np.float64)
        fk_override = G1WristFKTorch.from_default_urdf(left_tool_offset=custom)
        np.testing.assert_allclose(
            fk_override._left_tool_offset.detach().cpu().numpy(), custom, atol=1e-6
        )

    def test_wrong_shape_raises(self) -> None:
        bad = torch.zeros(4, dtype=torch.float32)  # (3,) でない
        with pytest.raises(ValueError, match="left_tool_offset must be shape"):
            G1WristFKTorch.from_default_urdf(left_tool_offset=bad)

    def test_non_finite_raises(self) -> None:
        bad = torch.tensor([float("inf"), 0.0, 0.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="right_tool_offset must be finite"):
            G1WristFKTorch.from_default_urdf(right_tool_offset=bad)

    def test_tool_offset_buffers_not_persistent(self, fk: G1WristFKTorch) -> None:
        """Issue #134: tool_offset buffer は persistent=False なので state_dict に
        含まれない。既存 Phase K R-6 ckpt (buffer 追加前学習) を新 fk 定義で load
        しても `_validate_architecture_state_dict_load` の missing_non_backbone に
        引っかからないことを保証する。"""
        state_keys = set(fk.state_dict().keys())
        assert "_left_tool_offset" not in state_keys, (
            f"_left_tool_offset must be persistent=False (regression: got in state_dict). "
            f"keys={sorted(state_keys)}"
        )
        assert "_right_tool_offset" not in state_keys, (
            f"_right_tool_offset must be persistent=False (regression: got in state_dict). "
            f"keys={sorted(state_keys)}"
        )
        # ただし instance attribute としては存在 (register_buffer は attr 登録もする)
        assert hasattr(fk, "_left_tool_offset")
        assert hasattr(fk, "_right_tool_offset")

    def test_zero_offset_shifts_wrist_position(self) -> None:
        """offset=[0,0,0] override 時と default (offset 有り) で forward 結果が
        offset magnitude order で乖離することを確認 (offset が実際に加算されている根拠)。"""
        fk_default = G1WristFKTorch.from_default_urdf()
        fk_zero = G1WristFKTorch.from_default_urdf(
            left_tool_offset=torch.zeros(3, dtype=torch.float32),
            right_tool_offset=torch.zeros(3, dtype=torch.float32),
        )
        q = torch.zeros(ACTION19_DIM, dtype=torch.float32)
        left_d, right_d = fk_default(q)
        left_z, right_z = fk_zero(q)
        # 差 = rot(zero_action) @ default_offset、rot は unit quaternion 相当なので
        # 差の magnitude は default_offset の magnitude と同じ order (~13cm)
        left_offset_mag = float(np.linalg.norm(LEFT_WRIST_TOOL_OFFSET_M))
        right_offset_mag = float(np.linalg.norm(RIGHT_WRIST_TOOL_OFFSET_M))
        diff_left = (left_d - left_z).norm().item()
        diff_right = (right_d - right_z).norm().item()
        # rot @ offset の norm ≈ offset の norm (rotation 保存)、tol 1mm
        np.testing.assert_allclose(diff_left, left_offset_mag, atol=1e-3)
        np.testing.assert_allclose(diff_right, right_offset_mag, atol=1e-3)


class TestConstantDriftFromInference:
    """Issue #134: `model/ramen_ori/fk.py` の tool offset + URDF file が `inference/desktop/
    perception/g1_urdf_fk.py` / `inference/orin/.../g1_description/` 側と drift しない
    ことを cross-verify。

    Self-contained 方針 (runtime `inference/` import 禁止、user 決定 2026-08-31) の例外
    として、test だけ inference 側 constant / file を参照して bit-equal を assert する。
    drift すれば CI で検出、`pixi run sync-model-assets` (root pixi.toml、Issue #134 Phase E)
    で URDF file は re-sync 可能。
    """

    def test_left_tool_offset_matches_inference_side(self) -> None:
        from inference.desktop.perception.g1_urdf_fk import (
            LEFT_WRIST_TOOL_OFFSET_M as INF_LEFT,
        )
        np.testing.assert_array_equal(LEFT_WRIST_TOOL_OFFSET_M, INF_LEFT)

    def test_right_tool_offset_matches_inference_side(self) -> None:
        from inference.desktop.perception.g1_urdf_fk import (
            RIGHT_WRIST_TOOL_OFFSET_M as INF_RIGHT,
        )
        np.testing.assert_array_equal(RIGHT_WRIST_TOOL_OFFSET_M, INF_RIGHT)

    def test_urdf_file_matches_inference_side(self) -> None:
        """URDF file が inference/orin/ 側 (single source of truth) と byte-equal。
        drift 検出時は `pixi run sync-model-assets` で re-sync。"""
        ramen_urdf = Path(DEFAULT_URDF_PATH).resolve()
        # test_fk.py 自身 (model/ramen_ori/tests/test_fk.py) 起点で repo root を
        # 導出、asset dir 階層が変わっても test 側は影響を受けない (PR #135 review)。
        repo_root = Path(__file__).resolve().parents[3]
        inf_urdf = repo_root / (
            "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/"
            "g1_29dof_mode_15_with_dex1_1.urdf"
        )
        assert inf_urdf.is_file(), (
            f"inference-side URDF not found (repo layout drift?): {inf_urdf}"
        )
        assert ramen_urdf.read_bytes() == inf_urdf.read_bytes(), (
            f"URDF drift detected:\n  ramen_ori: {ramen_urdf}\n"
            f"  inference: {inf_urdf}\n"
            f"Run `pixi run sync-model-assets` from repo root to re-sync."
        )


class TestForwardDetailed:
    """Issue #141 RO-16: 肘の位置・手先の位置・手の向き (同じ 1 回の chain の計算から取り出す)。"""

    @staticmethod
    def _random_q19(n: int, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randn(n, 19, generator=g, dtype=torch.float64) * 0.4

    def test_hand_position_and_rotation_match_inference_numpy_fk(self) -> None:
        """推論側の numpy FK と、手先の位置・回転が一致する。

        位置の比較相手は **`compute_ee_state`** (tool offset 込み、model の state 入力と
        同じ経路)。`compute_ee_transforms` は運営の WBC 向けに **tool offset を付けない**
        生の `wrist_yaw_link` 原点を返す仕様なので (付けると IK 残差が 60 倍、CLAUDE.md)、
        offset 込みの `forward_detailed` と直接比べてはいけない。
        回転は平行移動の offset に影響されないので `compute_ee_transforms` の R と比べる。
        """
        from inference.desktop.perception.g1_urdf_fk import G1WristFK

        fk64 = G1WristFKTorch.from_default_urdf(dtype=torch.float64)
        ref = G1WristFK.from_urdf(DEFAULT_URDF_PATH)
        q19 = self._random_q19(5)
        parts = fk64.forward_detailed(q19)
        for i in range(5):
            q29 = np.zeros(29)
            for j19, j29 in enumerate(_ACTION19_TO_SDK29_INDEX):
                if j29 >= 0:
                    q29[j29] = float(q19[i, j19])
            ee_state = ref.compute_ee_state(q29)
            np.testing.assert_allclose(parts["left_hand"][i].numpy(), ee_state[0:3], atol=1e-6)
            np.testing.assert_allclose(parts["right_hand"][i].numpy(), ee_state[6:9], atol=1e-6)

            left_pos, left_R, right_pos, right_R = ref.compute_ee_transforms(q29)
            np.testing.assert_allclose(parts["left_rot"][i].numpy(), left_R, atol=1e-9)
            np.testing.assert_allclose(parts["right_rot"][i].numpy(), right_R, atol=1e-9)

            # 2 つの推論 API の関係を固定する: 差はちょうど tool offset を手首の向きで回した分。
            # `compute_ee_transforms` に offset を足し戻す regression (IK 残差 60 倍) を捕まえる。
            np.testing.assert_allclose(
                left_pos + left_R @ LEFT_WRIST_TOOL_OFFSET_M, ee_state[0:3], atol=1e-6
            )
            np.testing.assert_allclose(
                right_pos + right_R @ RIGHT_WRIST_TOOL_OFFSET_M, ee_state[6:9], atol=1e-6
            )

    def test_elbow_matches_chain_ending_at_elbow_link(self) -> None:
        """肘の位置 = pelvis → elbow_link で終わる chain の原点 (別に parse した chain で照合)。"""
        name_to_idx = {n: i for i, n in enumerate(G1_JOINT_NAMES)}
        fk64 = G1WristFKTorch.from_default_urdf(dtype=torch.float64)
        elbow_fk = G1WristFKTorch(
            left_chain=_parse_urdf_chain(DEFAULT_URDF_PATH, "left_elbow_link", name_to_idx),
            right_chain=_parse_urdf_chain(DEFAULT_URDF_PATH, "right_elbow_link", name_to_idx),
            left_tool_offset=np.zeros(3), right_tool_offset=np.zeros(3), dtype=torch.float64,
        )
        q19 = self._random_q19(4, seed=1)
        parts = fk64.forward_detailed(q19)
        ref_left, ref_right = elbow_fk(q19)
        torch.testing.assert_close(parts["left_elbow"], ref_left)
        torch.testing.assert_close(parts["right_elbow"], ref_right)

    def test_forward_equals_detailed_hand(self, fk: G1WristFKTorch) -> None:
        q19 = self._random_q19(3, seed=2).float()
        left, right = fk(q19)
        parts = fk.forward_detailed(q19)
        torch.testing.assert_close(left, parts["left_hand"])
        torch.testing.assert_close(right, parts["right_hand"])

    def test_features_layout(self, fk: G1WristFKTorch) -> None:
        from model.ramen_ori.fk import FK_FEATURE_DIM, FK_FEATURE_SLICES

        q19 = self._random_q19(2, seed=3).float().reshape(1, 2, 19)
        feats = fk.features(q19)
        parts = fk.forward_detailed(q19)
        assert feats.shape == (1, 2, FK_FEATURE_DIM)
        torch.testing.assert_close(feats[..., FK_FEATURE_SLICES["left_elbow"]], parts["left_elbow"])
        torch.testing.assert_close(feats[..., FK_FEATURE_SLICES["right_hand"]], parts["right_hand"])
        # rot6d = 回転行列の最初の 2 列
        torch.testing.assert_close(
            feats[..., FK_FEATURE_SLICES["left_rot6d"]],
            torch.cat([parts["left_rot"][..., :, 0], parts["left_rot"][..., :, 1]], dim=-1),
        )
        torch.testing.assert_close(
            feats[..., FK_FEATURE_SLICES["hand_diff"]], parts["left_hand"] - parts["right_hand"]
        )

    def test_left_features_do_not_depend_on_right_arm(self, fk: G1WristFKTorch) -> None:
        from model.ramen_ori.fk import FK_FEATURE_SLICES

        q19 = self._random_q19(1, seed=4).float()
        q19b = q19.clone()
        q19b[:, 10:17] += 0.3   # 右腕
        a, b = fk.features(q19), fk.features(q19b)
        for name in ("left_elbow", "left_hand", "left_rot6d"):
            torch.testing.assert_close(a[:, FK_FEATURE_SLICES[name]], b[:, FK_FEATURE_SLICES[name]])
        assert not torch.allclose(a[:, FK_FEATURE_SLICES["right_hand"]], b[:, FK_FEATURE_SLICES["right_hand"]])

    def test_autocast_bf16_keeps_fk_precision(self, fk: G1WristFKTorch) -> None:
        """Issue #141: 学習は bf16 の autocast。FK は autocast の外 (buffer の精度 fp32) で計算し、bf16 の誤差を入れない。"""
        q19 = self._random_q19(64, seed=5).float()
        ref = fk.features(q19)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = fk.features(q19.bfloat16())
        assert out.dtype == torch.float32
        # 入力を bf16 に丸めた分だけの差 (bf16 で計算すると手先で mm 単位ずれる)
        ref_rounded = fk.features(q19.bfloat16().float())
        torch.testing.assert_close(out, ref_rounded)
        assert not torch.equal(ref, ref_rounded)
