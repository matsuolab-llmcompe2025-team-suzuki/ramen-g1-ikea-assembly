"""Phase 3 smoke: ramen_ori.py の preprocessing 契約 + config validation (Issue #125)。

- Preprocessing pipeline (frame resize / ImageNet normalize / batch build) は
  cv2 依存 = runtime env or model/ramen_ori env での実行前提。default env で
  cv2 が無い場合、preprocess_frame test は skip。他 test は default env で回る。
- 実 forward test は `@pytest.mark.integration` で skip (実 ckpt DL + torch 必要、
  Sakura env or local `pixi run -e model/ramen_ori` を想定)。
- Training-side spec: `model/ramen_ori/model.py` の batch key docstring を単一の
  真実として shape/dtype 検証。
"""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pytest

from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    Observation,
    PolicyConfig,
    RawRobotState,
)
from inference.desktop.lower_policy.policies.ramen_ori import (
    validate_contract,
    ACTION_DIM,
    CAMERAS,
    CAMERAS_3CAM_PHASE_K,
    CHUNK_LEN,
    CLASS_COLORS_BGR,
    DEFAULT_OVERLAY_CONF_THRESHOLD,
    DEFAULT_OVERLAY_LINE_THICKNESS,
    IMAGE_HW,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    NUM_SKILLS,
    OVERLAY_TARGET_CAMS,
    RamenOriPolicy,
    SKILL_MOVE_TABLE_BASE,
    SKILL_ROTATE_TABLE_BASE,
    STATE_DIM,
    TOP_K,
    VALID_CAM_LAYOUTS,
    build_batch_dict,
    build_state_from_raw,
    ema_state_dict_is_stale,
    strip_compile_prefix,
    match_training_jpeg,
    overlay_obb_on_frame,
    pack_obb_tokens,
    preprocess_frame,
    split_head_stereo,
    validate_ramen_ori_config,
)
from inference.desktop.perception.yolo_obb import OBBDetection


HAS_CV2 = importlib.util.find_spec("cv2") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


# ---- Constants match training config ---- #


class TestConstants:
    def test_state_dim_matches_e_delta(self) -> None:
        # state_derive.py:RAMEN_ORI_STATE_DIM
        assert STATE_DIM == 71

    def test_action_dim_matches_base_yaml(self) -> None:
        # base.yaml:model.action_dim
        assert ACTION_DIM == 19

    def test_chunk_len_matches_base_yaml(self) -> None:
        # base.yaml:model.chunk_len
        assert CHUNK_LEN == 16

    def test_image_size(self) -> None:
        assert IMAGE_HW == (224, 224)

    def test_cameras_is_4cam_layout(self) -> None:
        # data_lerobot.py:default_camera_keys() 順序と一致
        assert CAMERAS == (
            CameraKey.HEAD_LEFT,
            CameraKey.HEAD_RIGHT,
            CameraKey.WRIST_LEFT,
            CameraKey.WRIST_RIGHT,
        )

    def test_num_classes(self) -> None:
        # base.yaml:model.obb.num_classes
        assert NUM_CLASSES == 7

    def test_top_k(self) -> None:
        assert TOP_K == 4

    def test_num_skills(self) -> None:
        # skill_mapping.NUM_SKILLS
        assert NUM_SKILLS == 6

    def test_skill_id_canonical_map(self) -> None:
        # skill_mapping.py canonical
        assert SKILL_MOVE_TABLE_BASE == 5
        assert SKILL_ROTATE_TABLE_BASE == 4

    def test_imagenet_stats(self) -> None:
        # data_lerobot.py:IMAGENET_MEAN/STD (LingBot / DINOv2 標準)
        assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
        assert IMAGENET_STD == (0.229, 0.224, 0.225)


# ---- validate_ramen_ori_config ---- #


class TestValidateRamenOriConfig:
    def _cfg(self, **overrides) -> PolicyConfig:
        defaults = dict(
            mode="none",
            ckpt_ref="Team-RAMEN/foo@abc",
            device="cuda",
            dtype="fp32",
            cams=CAMERAS,
        )
        defaults.update(overrides)
        return PolicyConfig(**defaults)

    def test_valid_none(self) -> None:
        validate_ramen_ori_config(self._cfg())

    def test_valid_overlay(self) -> None:
        validate_ramen_ori_config(self._cfg(mode="overlay"))

    def test_valid_precomputed_token(self) -> None:
        validate_ramen_ori_config(self._cfg(mode="precomputed_token"))

    def test_accept_3cam_phase_k_layout(self) -> None:
        # Issue #132 Phase C: Phase K R-6 bench (Run 5 winner 含む) は
        # head_right 除外の 3 cam 学習 = CAMERAS_3CAM_PHASE_K も valid。
        validate_ramen_ori_config(self._cfg(cams=CAMERAS_3CAM_PHASE_K))
        assert CAMERAS_3CAM_PHASE_K == (
            CameraKey.HEAD_LEFT,
            CameraKey.WRIST_LEFT,
            CameraKey.WRIST_RIGHT,
        )

    def test_valid_cam_layouts_covers_4cam_and_3cam(self) -> None:
        assert CAMERAS in VALID_CAM_LAYOUTS
        assert CAMERAS_3CAM_PHASE_K in VALID_CAM_LAYOUTS
        # (HEAD_RIGHT 単独抜きの) その他組合せは reject
        with pytest.raises(ValueError, match="not supported"):
            validate_ramen_ori_config(
                self._cfg(cams=(CameraKey.WRIST_LEFT, CameraKey.WRIST_RIGHT))
            )

    def test_action_space_rel_accepted(self) -> None:
        # Issue #132 Phase C: action_space=rel は RAMEN-Ori で valid
        validate_ramen_ori_config(self._cfg(action_space="rel"))


# ---- build_state_from_raw ---- #


class TestBuildStateFromRaw:
    def _raw(
        self, joint_positions_prev: np.ndarray | None = None
    ) -> RawRobotState:
        return RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
            last_action_19d=None,
            joint_positions_prev=joint_positions_prev,
        )

    def test_output_shape_first_tick(self) -> None:
        # 先頭 tick (joint_positions_prev=None)、velocity は zeros
        state = build_state_from_raw(self._raw())
        assert state.shape == (STATE_DIM,)
        assert state.dtype == np.float32

    def test_output_shape_with_prev(self) -> None:
        prev = np.ones(29, dtype=np.float32) * 0.1
        state = build_state_from_raw(self._raw(joint_positions_prev=prev))
        assert state.shape == (STATE_DIM,)

    def test_reject_bad_shape(self) -> None:
        raw = RawRobotState(
            joint_positions=np.zeros(19, dtype=np.float32),  # wrong
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="joint_positions"):
            build_state_from_raw(raw)

    def test_joint_slice_populated_from_upper_body(self) -> None:
        # joint_positions[12:29] (waist3+arm14) + hand_state (2) → state[0:19]
        jp = np.arange(29, dtype=np.float32)
        hand = np.array([100.0, 200.0], dtype=np.float32)
        raw = RawRobotState(
            joint_positions=jp,
            hand_state=hand,
            ee_state=np.zeros(12, dtype=np.float32),
        )
        state = build_state_from_raw(raw)
        # UPPER_BODY: waist=[12,13,14], left_arm=[15..21], right_arm=[22..28]
        expected_upper = np.concatenate(
            [np.arange(12, 15), np.arange(15, 22), np.arange(22, 29)]
        ).astype(np.float32)
        np.testing.assert_array_equal(state[0:17], expected_upper)
        # 続く [17:19] = hand_state
        np.testing.assert_array_equal(state[17:19], hand)

    def test_hand_state_and_ee_pose_slices(self) -> None:
        # [57:59] = hand_state (explicit re-token)、[59:71] = ee_state
        hand = np.array([0.42, 0.73], dtype=np.float32)
        ee = np.linspace(1.0, 12.0, 12, dtype=np.float32)
        raw = RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=hand,
            ee_state=ee,
        )
        state = build_state_from_raw(raw)
        np.testing.assert_array_equal(state[57:59], hand)
        np.testing.assert_array_equal(state[59:71], ee)

    def test_tracking_err_first_tick_zeros(self) -> None:
        # last_action_19d=None → tracking_err zeros
        raw = RawRobotState(
            joint_positions=np.arange(29, dtype=np.float32),
            hand_state=np.array([0.5, 0.5], dtype=np.float32),
            ee_state=np.ones(12, dtype=np.float32),
            last_action_19d=None,
            joint_positions_prev=None,
        )
        state = build_state_from_raw(raw)
        np.testing.assert_array_equal(state[19:38], np.zeros(19, dtype=np.float32))

    def test_tracking_err_populated_from_last_action(self) -> None:
        # tracking_err = last_action_19d - joint_slice
        jp = np.zeros(29, dtype=np.float32)  # upper body 全 0
        hand = np.array([0.0, 0.0], dtype=np.float32)
        last_action = np.linspace(0.1, 1.9, 19, dtype=np.float32)
        raw = RawRobotState(
            joint_positions=jp,
            hand_state=hand,
            ee_state=np.zeros(12, dtype=np.float32),
            last_action_19d=last_action,
            joint_positions_prev=None,
        )
        state = build_state_from_raw(raw)
        # joint_slice = zeros(19)、tracking_err = last_action - 0 = last_action
        np.testing.assert_allclose(state[19:38], last_action, atol=1e-6)

    def test_velocity_first_tick_zeros(self) -> None:
        # joint_positions_prev=None → velocity zeros
        raw = RawRobotState(
            joint_positions=np.ones(29, dtype=np.float32),
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
            joint_positions_prev=None,
        )
        state = build_state_from_raw(raw)
        np.testing.assert_array_equal(state[38:57], np.zeros(19, dtype=np.float32))

    def test_velocity_populated_from_prev_joint_positions(self) -> None:
        # joint_slice - prev_joint_slice (Dex1も実測の前tickとの差分)
        jp = np.arange(29, dtype=np.float32)
        prev = np.arange(29, dtype=np.float32) - 0.5  # 全 slot 差分 0.5
        hand = np.array([0.9, 0.8], dtype=np.float32)
        hand_prev = np.array([0.7, 0.5], dtype=np.float32)
        raw = RawRobotState(
            joint_positions=jp,
            hand_state=hand,
            ee_state=np.zeros(12, dtype=np.float32),
            joint_positions_prev=prev,
            hand_state_prev=hand_prev,
        )
        state = build_state_from_raw(raw)
        # UPPER slice の差分 = 0.5 (17 entries)
        upper_expected = np.full(17, 0.5, dtype=np.float32)
        np.testing.assert_allclose(state[38:55], upper_expected, atol=1e-6)
        np.testing.assert_allclose(state[55:57], hand - hand_prev, atol=1e-6)

    def test_reject_bad_last_action_shape(self) -> None:
        raw = RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
            last_action_19d=np.zeros(20, dtype=np.float32),  # wrong
        )
        with pytest.raises(ValueError, match="last_action_19d"):
            build_state_from_raw(raw)

    def test_reject_bad_joint_positions_prev_shape(self) -> None:
        raw = RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
            joint_positions_prev=np.zeros(28, dtype=np.float32),  # wrong
        )
        with pytest.raises(ValueError, match="joint_positions_prev"):
            build_state_from_raw(raw)


# ---- preprocess_frame (cv2 依存、runtime env or model env) ---- #


@pytest.mark.skipif(
    not HAS_TORCH, reason="torch + torchvision required (runtime env or model env)"
)
class TestPreprocessFrame:
    def test_output_shape_dtype(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        chw = preprocess_frame(frame)
        assert chw.shape == (3, 224, 224)
        assert chw.dtype == np.float32

    def test_normalize_applied(self) -> None:
        # 全 pixel が 0 の frame は BGR → RGB (0,0,0) → 0/255=0 → (0 - mean) / std
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        chw = preprocess_frame(frame)
        # channel 0 (R): (0 - 0.485) / 0.229 ≈ -2.1179
        assert chw[0, 0, 0] == pytest.approx(-0.485 / 0.229, abs=1e-5)
        # channel 1 (G): (0 - 0.456) / 0.224
        assert chw[1, 0, 0] == pytest.approx(-0.456 / 0.224, abs=1e-5)
        # channel 2 (B): (0 - 0.406) / 0.225
        assert chw[2, 0, 0] == pytest.approx(-0.406 / 0.225, abs=1e-5)

    def test_bgr_to_rgb_ordering(self) -> None:
        # BGR=(255, 0, 0) = blue の frame は RGB=(0, 0, 255) → channel 2 (B) が
        # 一番大きな正の値になる
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # B channel in BGR
        chw = preprocess_frame(frame)
        # RGB 変換後、channel index 2 が B に対応 (H,W,C=BGR→RGB で index 2 が B)
        # (1 - 0.406) / 0.225 ≈ 2.6400
        assert chw[2, 100, 100] > 2.0

    def test_reject_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="uint8 BGR"):
            preprocess_frame(np.zeros((480, 640, 3), dtype=np.float32))

    def test_reject_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="uint8 BGR"):
            preprocess_frame(np.zeros((480, 640), dtype=np.uint8))


# ---- build_batch_dict ---- #


class TestPackObbTokens:
    """C-2 precomputed_token: per-cam det → RAMEN-Ori 5-tuple tensor pack。"""

    def test_output_shapes_and_dtypes(self) -> None:
        verts, conf, class_id, valid = pack_obb_tokens({}, cams=CAMERAS)
        assert verts.shape == (4, TOP_K, 8)
        assert verts.dtype == np.float32
        assert conf.shape == (4, TOP_K, 1)
        assert conf.dtype == np.float32
        assert class_id.shape == (4, TOP_K)
        assert class_id.dtype == np.int64
        assert valid.shape == (4, TOP_K)
        assert valid.dtype == bool
        assert not valid.any()  # 空 dict → 全 invalid

    def test_conf_sort_desc(self) -> None:
        det_low = OBBDetection(
            class_id=1, class_name="leg", confidence=0.3,
            verts=np.zeros((4, 2), dtype=np.float32),
        )
        det_high = OBBDetection(
            class_id=3, class_name="hole", confidence=0.9,
            verts=np.ones((4, 2), dtype=np.float32) * 0.5,
        )
        det_mid = OBBDetection(
            class_id=2, class_name="leg_tip", confidence=0.6,
            verts=np.ones((4, 2), dtype=np.float32) * 0.7,
        )
        _, conf, class_id, valid = pack_obb_tokens(
            {CameraKey.HEAD_LEFT: [det_low, det_high, det_mid]},
        )
        # cam 0 (head_left) の slot 0 が highest conf
        assert conf[0, 0, 0] == pytest.approx(0.9)
        assert conf[0, 1, 0] == pytest.approx(0.6)
        assert conf[0, 2, 0] == pytest.approx(0.3)
        assert valid[0, 0] and valid[0, 1] and valid[0, 2]
        assert not valid[0, 3]  # padding

    def test_top_k_truncation(self) -> None:
        # top_K=4 を超える det は落とす (低 conf 側)
        dets = [
            OBBDetection(
                class_id=1, class_name="leg", confidence=0.5 + 0.1 * i,
                verts=np.zeros((4, 2), dtype=np.float32),
            )
            for i in range(6)
        ]
        _, conf, _, valid = pack_obb_tokens({CameraKey.HEAD_LEFT: dets})
        assert valid[0].sum() == TOP_K  # 4 slot 全 valid
        # 上位 4 個 (conf 1.0, 0.9, 0.8, 0.7) が採用され、下位 2 個 (0.6, 0.5) は落ちる
        expected_confs = sorted([0.5 + 0.1 * i for i in range(6)], reverse=True)[:TOP_K]
        for i in range(TOP_K):
            assert conf[0, i, 0] == pytest.approx(expected_confs[i])

    def test_invalid_class_id_filtered(self) -> None:
        # class_id >= NUM_CLASSES=7 は skip される (embedding out of range 防止)
        bad = OBBDetection(
            class_id=10, class_name="unknown", confidence=0.9,
            verts=np.zeros((4, 2), dtype=np.float32),
        )
        _, _, _, valid = pack_obb_tokens({CameraKey.HEAD_LEFT: [bad]})
        assert not valid[0].any()

    def test_per_cam_dispatch(self) -> None:
        # head_left と head_right に別々の det を渡す → cam_id 順に pack
        det_l = OBBDetection(
            class_id=1, class_name="leg", confidence=0.8,
            verts=np.zeros((4, 2), dtype=np.float32),
        )
        det_r = OBBDetection(
            class_id=3, class_name="hole", confidence=0.7,
            verts=np.ones((4, 2), dtype=np.float32) * 0.5,
        )
        _, _, class_id, valid = pack_obb_tokens({
            CameraKey.HEAD_LEFT: [det_l],
            CameraKey.HEAD_RIGHT: [det_r],
        })
        # cam 0 (head_left) slot 0 = class 1 (leg)
        assert class_id[0, 0] == 1
        assert valid[0, 0]
        # cam 1 (head_right) slot 0 = class 3 (hole)
        assert class_id[1, 0] == 3
        assert valid[1, 0]
        # cam 2 / 3 (wrist) = 全 invalid
        assert not valid[2].any()
        assert not valid[3].any()

    def test_verts_flat_pack(self) -> None:
        # verts (4, 2) → flat (8,) xyxyxyxy
        det = OBBDetection(
            class_id=1, class_name="leg", confidence=0.9,
            verts=np.array(
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32
            ),
        )
        verts, _, _, _ = pack_obb_tokens({CameraKey.HEAD_LEFT: [det]})
        np.testing.assert_array_almost_equal(
            verts[0, 0], np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        )


class TestOverlayObbOnFrame:
    """OBB overlay 描画 (C-11、training OverlayRenderer と palette / thickness / conf 一致)。"""

    def test_class_palette_matches_training(self) -> None:
        # model/ramen_ori/overlay.py:CLASS_COLORS_BGR と厳密一致
        assert CLASS_COLORS_BGR == {
            0: (128, 128, 128),
            1: (0, 255, 0),
            2: (0, 255, 255),
            3: (0, 0, 255),
            4: (255, 0, 0),
            5: (0, 128, 255),
            6: (255, 0, 255),
        }

    def test_default_conf_threshold(self) -> None:
        # Issue #122 D-3 preview 決定値
        assert DEFAULT_OVERLAY_CONF_THRESHOLD == 0.30

    def test_default_line_thickness(self) -> None:
        assert DEFAULT_OVERLAY_LINE_THICKNESS == 2

    def test_overlay_target_cams(self) -> None:
        # precompute cache が head 2 cam のみ = wrist は overlay 対象外
        assert OVERLAY_TARGET_CAMS == (CameraKey.HEAD_LEFT, CameraKey.HEAD_RIGHT)

    @pytest.mark.skipif(not HAS_CV2, reason="cv2 required")
    def test_empty_detections_no_op(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = overlay_obb_on_frame(frame, detections=[])
        assert np.array_equal(out, np.zeros((480, 640, 3), dtype=np.uint8))

    @pytest.mark.skipif(not HAS_CV2, reason="cv2 required")
    def test_draws_leg_green(self) -> None:
        # class_id=1 (leg) の det を描画 → 該当 pixel が green (0, 255, 0) BGR
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = OBBDetection(
            class_id=1,
            class_name="leg",
            confidence=0.9,
            verts=np.array(
                [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]], dtype=np.float32
            ),
        )
        overlay_obb_on_frame(frame, detections=[det])
        # 描画 pixel が存在 (green channel が >0)
        assert (frame[:, :, 1] > 0).any(), "no green pixel drawn (leg overlay failed)"
        # 上辺 y = 0.1 * 480 = 48 付近に green line がある
        # 左端 x = 0.1 * 640 = 64 付近から右端 x = 0.5 * 640 = 320 まで
        assert frame[48, 100, 1] == 255 or frame[47, 100, 1] == 255 or frame[49, 100, 1] == 255

    @pytest.mark.skipif(not HAS_CV2, reason="cv2 required")
    def test_conf_threshold_filters(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        low_conf_det = OBBDetection(
            class_id=1,
            class_name="leg",
            confidence=0.1,  # threshold 0.30 未満 → skip
            verts=np.array(
                [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]], dtype=np.float32
            ),
        )
        overlay_obb_on_frame(frame, detections=[low_conf_det], conf_threshold=0.30)
        assert not (frame[:, :, 1] > 0).any(), "low-conf det should be skipped"

    @pytest.mark.skipif(not HAS_CV2, reason="cv2 required")
    def test_class_filter(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        leg = OBBDetection(
            class_id=1,
            class_name="leg",
            confidence=0.9,
            verts=np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]], dtype=np.float32),
        )
        hole = OBBDetection(
            class_id=3,
            class_name="hole",
            confidence=0.9,
            verts=np.array([[0.6, 0.6], [0.8, 0.6], [0.8, 0.8], [0.6, 0.8]], dtype=np.float32),
        )
        # class_filter = {3} (hole のみ)
        overlay_obb_on_frame(frame, detections=[leg, hole], class_filter={3})
        # green (leg) pixel 0、red (hole) pixel > 0
        # BGR: hole = (0, 0, 255) = channel 2 が 255
        assert not (frame[:, :, 1] > 0).any(), "leg should be filtered out"
        assert (frame[:, :, 2] > 0).any(), "hole should be drawn"

    @pytest.mark.skipif(not HAS_CV2, reason="cv2 required")
    def test_reject_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="uint8 BGR"):
            overlay_obb_on_frame(np.zeros((480, 640, 3), dtype=np.float32), detections=[])

    @pytest.mark.skipif(not HAS_CV2, reason="cv2 required")
    def test_reject_bad_conf_threshold(self) -> None:
        with pytest.raises(ValueError, match="conf_threshold"):
            overlay_obb_on_frame(
                np.zeros((480, 640, 3), dtype=np.uint8), detections=[], conf_threshold=1.5
            )


class TestMatchTrainingJpeg:
    """overlay 画像を学習 cache と同じ jpg に 1 回通す (Issue #139、色差は slot ごとに選ぶ)。"""

    @staticmethod
    def _table_top_box_frame() -> np.ndarray:
        # 灰色の背景に table_top (BGR 青) の 2px 線の box
        det = OBBDetection(
            class_id=4,
            class_name="table_top",
            confidence=0.9,
            verts=np.array([[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]], dtype=np.float32),
        )
        return overlay_obb_on_frame(np.full((480, 640, 3), 128, dtype=np.uint8), [det])

    @pytest.mark.skipif(not HAS_CV2, reason="cv2 required")
    def test_420_matches_pil_default_of_pre_139_bakes(self) -> None:
        # #139 より前の焼き込みは PIL の既定 (quality 90) で保存。"4:2:0" はそれと bit 一致
        import io

        import cv2
        from PIL import Image

        frame = self._table_top_box_frame()
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(frame[..., ::-1])).save(
            buffer, format="JPEG", quality=90
        )
        expected = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(match_training_jpeg(frame, "4:2:0"), expected)

    @pytest.mark.skipif(not HAS_CV2, reason="cv2 required")
    def test_444_keeps_line_color_that_420_smears(self) -> None:
        # 4:4:4 は色差を間引かないので、2px の色線が 4:2:0 よりずっと元の色に近い
        frame = self._table_top_box_frame()
        line = np.abs(frame.astype(np.int16) - 128).max(axis=2) > 0
        error = {
            subsampling: np.abs(
                match_training_jpeg(frame, subsampling).astype(np.int16) - frame
            )[line].mean()
            for subsampling in ("4:4:4", "4:2:0")
        }
        assert error["4:4:4"] < error["4:2:0"] / 3, error

    def test_rejects_unknown_subsampling(self) -> None:
        with pytest.raises(ValueError, match="subsampling"):
            match_training_jpeg(np.zeros((480, 640, 3), dtype=np.uint8), "4:2:2")


class TestSplitHeadStereo:
    """head 480x1280 packed stereo → (L, R) split helper。"""

    def test_output_shapes(self) -> None:
        packed = np.zeros((480, 1280, 3), dtype=np.uint8)
        left, right = split_head_stereo(packed)
        assert left.shape == (480, 640, 3)
        assert right.shape == (480, 640, 3)
        assert left.dtype == np.uint8
        assert right.dtype == np.uint8

    def test_left_right_content_disjoint(self) -> None:
        # 左半 = 255、右半 = 100 のフレームで split が正しく起こるか
        packed = np.zeros((480, 1280, 3), dtype=np.uint8)
        packed[:, :640, :] = 255
        packed[:, 640:, :] = 100
        left, right = split_head_stereo(packed)
        assert left.min() == 255 and left.max() == 255
        assert right.min() == 100 and right.max() == 100

    def test_reject_wrong_shape(self) -> None:
        # 480x640 (単 cam) を stereo split に渡すと reject
        with pytest.raises(ValueError, match=r"\(480, 1280, 3\)"):
            split_head_stereo(np.zeros((480, 640, 3), dtype=np.uint8))


@pytest.mark.skipif(
    not HAS_TORCH,
    reason="torch + torchvision required (build_batch_dict calls preprocess_frame)",
)
class TestBuildBatchDict:
    def _obs(
        self,
        skill_id: int | None = SKILL_MOVE_TABLE_BASE,
        state_dim: int = STATE_DIM,
        skip_cam: CameraKey | None = None,
        frames_bgr_prev: dict | None = None,
    ) -> Observation:
        H, W = 480, 640
        frames = {cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS}
        if skip_cam is not None:
            del frames[skip_cam]
        return Observation(
            frames_bgr=frames,
            frames_bgr_prev=frames_bgr_prev,
            state=np.zeros(state_dim, dtype=np.float32),
            skill_id=skill_id,
            language=None,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )

    def test_output_keys(self) -> None:
        batch = build_batch_dict(self._obs(), mode="none")
        # model.py 冒頭の batch format と厳密一致
        assert "images" in batch
        assert "images_prev" in batch
        assert "cam_id" in batch
        assert "obb_verts" in batch
        assert "obb_conf" in batch
        assert "obb_class_id" in batch
        assert "obb_cam_id" in batch
        assert "obb_valid_mask" in batch
        assert "state" in batch
        assert "skill_id" in batch

    def test_images_shape(self) -> None:
        batch = build_batch_dict(self._obs(), mode="none")
        # (N_cams=4, 3, 224, 224)
        assert batch["images"].shape == (4, 3, 224, 224)
        assert batch["images"].dtype == np.float32

    def test_images_prev_matches_now_when_none(self) -> None:
        # frames_bgr_prev=None → images_prev = images で差分 0 (Issue #141 束 1-3 / INF-4)。
        # 学習も区間の頭は images_prev = images (ep 境界の clamp)。
        batch = build_batch_dict(self._obs(frames_bgr_prev=None), mode="none")
        assert batch["images_prev"].shape == (4, 3, 224, 224)
        np.testing.assert_array_equal(batch["images_prev"], batch["images"])

    def test_cam_id_arange(self) -> None:
        # cam_id 0..N_cams-1 (data_lerobot.py:_cam_id = torch.arange)
        batch = build_batch_dict(self._obs(), mode="none")
        assert batch["cam_id"].shape == (4,)
        assert batch["cam_id"].dtype == np.int64
        np.testing.assert_array_equal(batch["cam_id"], np.arange(4))

    def test_3cam_phase_k_layout(self) -> None:
        """Issue #132 Phase C: Phase K 3 cam (head_left + wrist L/R) 経路。

        cams 引数を渡すと images / cam_id shape が 3 に縮む、obb_verts 系も 3 で
        揃う。head_right frame は Observation に含まれていても無視 (build_batch_dict
        は cams tuple を絶対的な source of truth として扱う)。
        """
        # Phase K 3 cam に必要な frame のみ含む Observation (head_right skip)
        H, W = 480, 640
        frames = {
            cam: np.zeros((H, W, 3), dtype=np.uint8)
            for cam in CAMERAS_3CAM_PHASE_K
        }
        obs = Observation(
            frames_bgr=frames,
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_ROTATE_TABLE_BASE,
            language=None,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )
        batch = build_batch_dict(obs, mode="none", cams=CAMERAS_3CAM_PHASE_K)
        assert batch["images"].shape == (3, 3, 224, 224)
        assert batch["images_prev"].shape == (3, 3, 224, 224)
        assert batch["cam_id"].shape == (3,)
        np.testing.assert_array_equal(batch["cam_id"], np.arange(3))
        assert batch["obb_verts"].shape == (3, TOP_K, 8)
        assert batch["obb_valid_mask"].shape == (3, TOP_K)

    def test_3cam_missing_frame_rejected(self) -> None:
        # cams が 3 cam でも、Observation に required frame が無ければ KeyError
        H, W = 480, 640
        frames = {
            CameraKey.HEAD_LEFT: np.zeros((H, W, 3), dtype=np.uint8),
            CameraKey.WRIST_LEFT: np.zeros((H, W, 3), dtype=np.uint8),
            # WRIST_RIGHT 欠落
        }
        obs = Observation(
            frames_bgr=frames,
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_ROTATE_TABLE_BASE,
            language=None,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )
        with pytest.raises(KeyError, match="wrist_right"):
            build_batch_dict(obs, mode="none", cams=CAMERAS_3CAM_PHASE_K)

    def test_obb_none_mode_all_invalid(self) -> None:
        batch = build_batch_dict(self._obs(), mode="none")
        assert batch["obb_verts"].shape == (4, TOP_K, 8)
        assert batch["obb_conf"].shape == (4, TOP_K, 1)
        assert batch["obb_class_id"].shape == (4, TOP_K)
        assert batch["obb_cam_id"].shape == (4, TOP_K)
        assert batch["obb_valid_mask"].shape == (4, TOP_K)
        assert batch["obb_valid_mask"].dtype == bool
        assert not batch["obb_valid_mask"].any()  # 全 invalid

    def test_obb_cam_id_broadcast(self) -> None:
        batch = build_batch_dict(self._obs(), mode="none")
        # obb_cam_id[c, k] = c (data_lerobot.py:cam_ids_2d)
        expected = np.broadcast_to(
            np.arange(4)[:, None], (4, TOP_K)
        )
        np.testing.assert_array_equal(batch["obb_cam_id"], expected)

    def test_state_shape(self) -> None:
        batch = build_batch_dict(self._obs(), mode="none")
        assert batch["state"].shape == (STATE_DIM,)
        assert batch["state"].dtype == np.float32

    def test_skill_id_scalar(self) -> None:
        batch = build_batch_dict(self._obs(skill_id=5), mode="none")
        assert batch["skill_id"] == 5
        assert batch["skill_id"].dtype == np.int64

    def test_reject_missing_skill_id(self) -> None:
        with pytest.raises(ValueError, match="requires obs.skill_id"):
            build_batch_dict(self._obs(skill_id=None), mode="none")

    def test_reject_skill_id_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"in \[0,"):
            build_batch_dict(self._obs(skill_id=NUM_SKILLS), mode="none")

    def test_reject_wrong_state_dim(self) -> None:
        with pytest.raises(ValueError, match=r"\(71,\)"):
            build_batch_dict(self._obs(state_dim=49), mode="none")

    def test_reject_missing_cam(self) -> None:
        with pytest.raises(KeyError, match="head_left"):
            build_batch_dict(self._obs(skip_cam=CameraKey.HEAD_LEFT), mode="none")

    def test_reject_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            build_batch_dict(self._obs(), mode="bogus")

    def test_overlay_mode_runs_end_to_end(self) -> None:
        # Phase 5 実装済: obb_detections が入っていても overlay 描画後 preprocess で完走
        H, W = 480, 640
        det = OBBDetection(
            class_id=1,  # leg (green)
            class_name="leg",
            confidence=0.9,
            verts=np.array(
                [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]], dtype=np.float32
            ),
        )
        # dict[Cam, list] で per-cam に det を渡す (YOLO は head 2 cam で走る前提)
        obs = Observation(
            frames_bgr={cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS},
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_MOVE_TABLE_BASE,
            language=None,
            obb_detections={CameraKey.HEAD_LEFT: [det], CameraKey.HEAD_RIGHT: [det]},
            timestamp_ns=time.monotonic_ns(),
        )
        batch = build_batch_dict(obs, mode="overlay", overlay_jpeg_subsampling="4:4:4")
        # Shape 保証
        assert batch["images"].shape == (4, 3, 224, 224)
        # head cam (0, 1) は overlay 描画があるので mean が wrist と異なる (green pixel 影響)
        # wrist cam (2, 3) は overlay 対象外で zero frame preprocess と同結果
        assert not np.allclose(batch["images"][0], batch["images"][2])

    @pytest.mark.parametrize("subsampling", ["4:4:4", "4:2:0"])
    def test_overlay_mode_no_detections_only_matches_training_jpeg(
        self, subsampling: str
    ) -> None:
        # box が 0 件でも学習 overlay cache の head 画像は jpg 保存を通っているので、
        # head cam は同じ jpg を 1 回通した画像になる (wrist は overlay 対象外で none と同じ)
        H, W = 480, 640
        rng = np.random.default_rng(0)
        frames = {cam: rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for cam in CAMERAS}
        obs_overlay = Observation(
            frames_bgr=frames,
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_MOVE_TABLE_BASE,
            language=None,
            obb_detections={},
            timestamp_ns=time.monotonic_ns(),
        )
        batch_overlay = build_batch_dict(
            obs_overlay, mode="overlay", overlay_jpeg_subsampling=subsampling
        )
        batch_none = build_batch_dict(
            Observation(
                frames_bgr={
                    cam: (
                        match_training_jpeg(frame, subsampling)
                        if cam in OVERLAY_TARGET_CAMS
                        else frame
                    )
                    for cam, frame in frames.items()
                },
                frames_bgr_prev=None,
                state=np.zeros(STATE_DIM, dtype=np.float32),
                skill_id=SKILL_MOVE_TABLE_BASE,
                language=None,
                obb_detections=None,
                timestamp_ns=time.monotonic_ns(),
            ),
            mode="none",
        )
        np.testing.assert_array_equal(batch_overlay["images"], batch_none["images"])

    def test_overlay_mode_requires_jpeg_subsampling(self) -> None:
        # 学習 cache の色差を知らないまま overlay 画像を作らない (焼き込みの時期で違う)
        with pytest.raises(ValueError, match="overlay_jpeg_subsampling"):
            build_batch_dict(self._obs(), mode="overlay")

    def test_precomputed_token_mode_packs_per_cam(self) -> None:
        """Phase 6: mode=precomputed_token で per-cam det が OBB tensor に pack される。"""
        H, W = 480, 640
        det_leg = OBBDetection(
            class_id=1,
            class_name="leg",
            confidence=0.9,
            verts=np.array(
                [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]], dtype=np.float32
            ),
        )
        det_hole = OBBDetection(
            class_id=3,
            class_name="hole",
            confidence=0.7,
            verts=np.array(
                [[0.5, 0.5], [0.7, 0.5], [0.7, 0.7], [0.5, 0.7]], dtype=np.float32
            ),
        )
        obs = Observation(
            frames_bgr={cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS},
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_MOVE_TABLE_BASE,
            language=None,
            obb_detections={
                CameraKey.HEAD_LEFT: [det_leg, det_hole],  # 2 det on head_left
                CameraKey.HEAD_RIGHT: [det_hole],           # 1 det on head_right
            },
            timestamp_ns=time.monotonic_ns(),
        )
        batch = build_batch_dict(obs, mode="precomputed_token")
        # head_left (cam_id=0): 2 valid det (conf desc: leg 0.9 → slot 0, hole 0.7 → slot 1)
        assert batch["obb_valid_mask"][0, 0] == True
        assert batch["obb_valid_mask"][0, 1] == True
        assert batch["obb_valid_mask"][0, 2] == False
        assert batch["obb_class_id"][0, 0] == 1  # leg (highest conf)
        assert batch["obb_class_id"][0, 1] == 3  # hole
        assert batch["obb_conf"][0, 0, 0] == pytest.approx(0.9)
        assert batch["obb_conf"][0, 1, 0] == pytest.approx(0.7)
        # head_right (cam_id=1): 1 valid det
        assert batch["obb_valid_mask"][1, 0] == True
        assert batch["obb_valid_mask"][1, 1] == False
        # wrist_left (cam_id=2): 全 invalid (dict に無い)
        assert not batch["obb_valid_mask"][2].any()
        assert not batch["obb_valid_mask"][3].any()

    def test_precomputed_token_mode_empty_all_invalid(self) -> None:
        """dict is None → obb signals 全 invalid (mode=none と同じ)。"""
        H, W = 480, 640
        obs = Observation(
            frames_bgr={cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS},
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_MOVE_TABLE_BASE,
            language=None,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )
        batch = build_batch_dict(obs, mode="precomputed_token")
        assert not batch["obb_valid_mask"].any()


# ---- RamenOriPolicy skeleton lifecycle ---- #


class TestRamenOriPolicyLifecycle:
    def _cfg(self, mode: str = "none") -> PolicyConfig:
        return PolicyConfig(
            mode=mode,
            ckpt_ref="Team-RAMEN/foo@abc",
            dtype="fp32",
            cams=CAMERAS,
        )

    def test_init_without_load_raises_on_predict(self) -> None:
        p = RamenOriPolicy(self._cfg())
        obs = Observation(
            frames_bgr={cam: np.zeros((480, 640, 3), dtype=np.uint8) for cam in CAMERAS},
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_MOVE_TABLE_BASE,
            language=None,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            p.predict(obs)

    def test_init_without_load_raises_on_warmup(self) -> None:
        p = RamenOriPolicy(self._cfg())
        with pytest.raises(RuntimeError, match="not loaded"):
            p.warmup(n_iter=1)

    def test_close_no_op_when_not_loaded(self) -> None:
        p = RamenOriPolicy(self._cfg())
        p.close()  # no raise

    @pytest.mark.integration
    def test_from_ckpt_real(self) -> None:
        # Real load 経路 (Hydra instantiate + HF DL + state_dict load)。
        # model/ramen_ori pixi env + local GPU (or CPU) で走る。
        pytest.skip(
            "integration: requires model/ramen_ori pixi env (hydra + torch 2.10 + "
            "lerobot fork + LingBot HF DL + Run 1 ckpt DL ~4GB)"
        )

    def test_class_constants_expose_contract(self) -> None:
        assert RamenOriPolicy.STATE_DIM == STATE_DIM
        assert RamenOriPolicy.ACTION_DIM == ACTION_DIM
        assert RamenOriPolicy.CHUNK_LEN == CHUNK_LEN
        assert RamenOriPolicy.CAMERAS == CAMERAS
        assert RamenOriPolicy.NUM_SKILLS == NUM_SKILLS

    def test_class_exposes_canonical_state_builder_for_vla_skill(self) -> None:
        raw = RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
        )
        # state の版は ckpt の約束で決まるので、instance 経由で呼ぶ (Issue #141 P8-4)。
        policy = RamenOriPolicy(cfg=self._cfg())
        np.testing.assert_array_equal(
            policy.build_state_from_raw(raw),
            build_state_from_raw(raw),
        )

    @pytest.mark.integration
    def test_forward_real_ckpt(self) -> None:
        # Sakura env or local model/ramen_ori env でのみ走る:
        # Run 1 HF ckpt を DL、mode=none で 1 tick forward → shape 検証
        pytest.skip(
            "integration: requires torch + lerobot fork + real ckpt DL, run in "
            "model/ramen_ori pixi env or on sakura H100"
        )


class TestPolicyProtocolCompatibility:
    def test_ramen_ori_matches_policy_protocol(self) -> None:
        from inference.desktop.lower_policy.policies.base import Policy

        p = RamenOriPolicy(PolicyConfig(mode="none", ckpt_ref="x", cams=CAMERAS))
        assert isinstance(p, Policy)


# ---- Issue #132 Phase C: predict() の 16→19 pad + rel 復元経路 ---- #


@pytest.mark.skipif(not HAS_TORCH, reason="predict requires torch tensors")
class TestPredictShapeAndRelReconstruction:
    """predict の pad/rel 変換経路を mock model で検証。

    `_model.predict_action(batch)` を mock して、実 model init なし (from_ckpt
    通さない) で pad / rel 復元ロジックだけ isolate して verify。
    """

    def _obs_3cam(self, state_arms_current: float = 0.0) -> Observation:
        H, W = 480, 640
        frames = {
            cam: np.zeros((H, W, 3), dtype=np.uint8)
            for cam in CAMERAS_3CAM_PHASE_K
        }
        state = np.zeros(STATE_DIM, dtype=np.float32)
        # state71 layout: [0:3] waist、[3:17] arms 14、[17:19] hand 2
        state[3:17] = state_arms_current
        return Observation(
            frames_bgr=frames,
            frames_bgr_prev=None,
            state=state,
            skill_id=SKILL_ROTATE_TABLE_BASE,
            language=None,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )

    def _cfg_3cam(self, action_space: str = "abs") -> PolicyConfig:
        return PolicyConfig(
            mode="none",
            ckpt_ref="dummy",
            device="cpu",
            dtype="fp32",
            cams=CAMERAS_3CAM_PHASE_K,
            action_space=action_space,
        )

    def test_outputs_waist_is_false_for_a_16d_ckpt(self) -> None:
        """Issue #144: 16D 学習の ckpt は腰を出さない (推論側が 0 を pad する)。

        VlaSkill はこれを見て腰を送るかを決める。True になると 0 rad の絶対指令が
        腰に流れ、直立方向へ寄っていく。
        """
        class Model16D:
            action_dim = 16

        policy = RamenOriPolicy(self._cfg_3cam(), _model=Model16D(), _device="cpu")
        assert policy.outputs_waist is False

    def test_outputs_waist_is_true_for_a_19d_ckpt(self) -> None:
        """19D 学習 (Phase 1/2 の variant) は腰も出すので、従来どおり流す。"""
        class Model19D:
            action_dim = 19

        policy = RamenOriPolicy(self._cfg_3cam(), _model=Model19D(), _device="cpu")
        assert policy.outputs_waist is True

    def test_outputs_waist_defaults_to_holding_when_unknown(self) -> None:
        """action_dim が分からない間は腰を止める側に倒す (hold は安全な状態)。"""
        class ModelWithoutActionDim:
            pass

        policy = RamenOriPolicy(
            self._cfg_3cam(), _model=ModelWithoutActionDim(), _device="cpu"
        )
        assert policy.outputs_waist is False

    def test_pad_16d_to_19d_zeros_waist(self) -> None:
        """action_dim=16 model → predict output は zeros waist prepend で (chunk, 19)。"""
        import torch

        chunk_len = 16

        class MockModel16D:
            """Phase K spec: (B, chunk, 16) [arms 14, hand 2] を返す。"""

            def predict_action(self, batch):
                # 一定 pattern を返して arms/hand 位置が正確に維持されるか確認
                out = torch.arange(
                    chunk_len * 16, dtype=torch.float32
                ).reshape(1, chunk_len, 16)
                return out

        policy = RamenOriPolicy(
            self._cfg_3cam(), _model=MockModel16D(), _device="cpu"
        )
        action = policy.predict(self._obs_3cam())
        assert action.action_chunk.shape == (chunk_len, 19)
        # waist [0:3] = zeros (pad)
        assert (action.action_chunk[:, 0:3] == 0.0).all()
        # arms + hand [3:19] = mock の 16D 出力そのまま
        expected = np.arange(chunk_len * 16, dtype=np.float32).reshape(
            chunk_len, 16
        )
        np.testing.assert_array_equal(action.action_chunk[:, 3:], expected)
        assert action.metadata["action_dim"] == 19

    def test_19d_pass_through_no_pad(self) -> None:
        """action_dim=19 model → predict output は pad 経路通らず 19D pass-through。"""
        import torch

        chunk_len = 16

        class MockModel19D:
            def predict_action(self, batch):
                return torch.arange(
                    chunk_len * 19, dtype=torch.float32
                ).reshape(1, chunk_len, 19)

        policy = RamenOriPolicy(
            PolicyConfig(mode="none", ckpt_ref="dummy", device="cpu", cams=CAMERAS),
            _model=MockModel19D(),
            _device="cpu",
        )
        # 4 cam Observation
        H, W = 480, 640
        obs = Observation(
            frames_bgr={cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS},
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_MOVE_TABLE_BASE,
            language=None,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )
        action = policy.predict(obs)
        assert action.action_chunk.shape == (chunk_len, 19)
        expected = np.arange(chunk_len * 19, dtype=np.float32).reshape(
            chunk_len, 19
        )
        np.testing.assert_array_equal(action.action_chunk, expected)

    def test_unexpected_action_dim_rejected(self) -> None:
        """action_dim が 16 でも 19 でもない → RuntimeError。"""
        import torch

        class MockModelBad:
            def predict_action(self, batch):
                return torch.zeros(1, 16, 42)  # 42D は Phase 1/2 / Phase K どちらでもない

        policy = RamenOriPolicy(
            self._cfg_3cam(), _model=MockModelBad(), _device="cpu"
        )
        with pytest.raises(RuntimeError, match="unexpected action_dim"):
            policy.predict(self._obs_3cam())

    def test_rel_reconstruction_zero_dq_gives_arms_current(self) -> None:
        """action_space=rel + Δq_norm=0 → arms_abs = arms_current (cumsum of zeros)。

        mean=0 / std=1 (identity normalize) を渡した状態で dq_norm=0 なら
        denorm も 0 = cumsum 0 = arms_abs = arms_current + 0 = arms_current。
        rel 復元 pipeline (extract → reconstruct → replace slice) の正当性を verify。
        """
        import torch

        arms_current_val = 0.5
        chunk_len = 16

        class MockRelModel16D:
            """rel 学習 spec: arms 14D は normalized Δq (= 0)、hand 2D は abs。"""

            def __init__(self):
                # register_buffer に相当 (nn.Module 継承せず attr 直付)
                self._relative_arms_mean = torch.zeros(14, dtype=torch.float32)
                self._relative_arms_std = torch.ones(14, dtype=torch.float32)

            def predict_action(self, batch):
                # Δq_norm = 0 (arms 14D)、hand = 7.0 (arbitrary abs value、pass-through)
                out = torch.zeros(1, chunk_len, 16, dtype=torch.float32)
                out[..., 14:16] = 7.0
                return out

        policy = RamenOriPolicy(
            self._cfg_3cam(action_space="rel"),
            _model=MockRelModel16D(),
            _device="cpu",
        )
        action = policy.predict(self._obs_3cam(state_arms_current=arms_current_val))
        assert action.action_chunk.shape == (chunk_len, 19)
        # waist [0:3] = zeros pad
        assert (action.action_chunk[:, 0:3] == 0.0).all()
        # arms [3:17] = arms_current (Δq_norm=0 → cumsum=0 → arms_current + 0)
        np.testing.assert_allclose(
            action.action_chunk[:, 3:17],
            np.full((chunk_len, 14), arms_current_val, dtype=np.float32),
        )
        # hand [17:19] = mock の 7.0 pass-through (abs、rel でも変換しない)
        np.testing.assert_allclose(
            action.action_chunk[:, 17:19], np.full((chunk_len, 2), 7.0)
        )

    def test_rel_reconstruction_nonzero_dq_cumsum(self) -> None:
        """Δq_norm=1 全部 + std=0.01 → arms は arms_current + 0.01, 0.02, 0.03, ...。

        cumsum の正しさ (chunk step k での arms_abs = arms_current + sum(Δq[0..k]))
        と normalize 逆変換 (dq_norm * std + mean) を joint verify。
        """
        import torch

        arms_current_val = 1.0
        chunk_len = 16
        step_delta = 0.01

        class MockRelModel16D:
            def __init__(self):
                self._relative_arms_mean = torch.zeros(14, dtype=torch.float32)
                self._relative_arms_std = torch.full(
                    (14,), step_delta, dtype=torch.float32
                )

            def predict_action(self, batch):
                # Δq_norm = 1 全部 → denorm = 1 * step_delta + 0 = step_delta
                return torch.ones(1, chunk_len, 16, dtype=torch.float32)

        policy = RamenOriPolicy(
            self._cfg_3cam(action_space="rel"),
            _model=MockRelModel16D(),
            _device="cpu",
        )
        action = policy.predict(self._obs_3cam(state_arms_current=arms_current_val))
        # arms[k] = arms_current + (k+1)*step_delta (step 0 は arms_current+step_delta)
        expected_arms = np.array(
            [
                [arms_current_val + (k + 1) * step_delta] * 14
                for k in range(chunk_len)
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            action.action_chunk[:, 3:17], expected_arms, rtol=1e-6
        )


class TestValidateArchitectureAllowList:
    """Issue #134 Phase H-4: `_validate_architecture_state_dict_load` の allow-list
    が意図通り動作することを verify (実 ckpt smoke に加えて continuous 検出用)。
    """

    def test_backbone_missing_allowed(self) -> None:
        """`vision.backbone.*` prefix は allow-list、raise なし。"""
        from inference.desktop.lower_policy.policies.ramen_ori import (
            _validate_architecture_state_dict_load,
        )
        _validate_architecture_state_dict_load(
            missing=["vision.backbone.cls_token", "vision.backbone.blocks.0.attn.qkv.weight"],
            unexpected=[],
            ckpt_path="test.pt",
        )

    def test_relative_arms_buffers_missing_allowed(self) -> None:
        """`_relative_arms_mean/std` は init 時に埋まる buffer、EMA path で missing
        判定されても raise なし (Phase K R-6 rel ckpt での from_ckpt を通すため)。"""
        from inference.desktop.lower_policy.policies.ramen_ori import (
            _validate_architecture_state_dict_load,
        )
        _validate_architecture_state_dict_load(
            missing=[
                "vision.backbone.cls_token",
                "_relative_arms_mean",
                "_relative_arms_std",
            ],
            unexpected=[],
            ckpt_path="test.pt",
        )

    def test_arbitrary_missing_key_rejected(self) -> None:
        """未 allow-list な missing key は raise (fail-closed 保護維持)。"""
        import pytest
        from inference.desktop.lower_policy.policies.ramen_ori import (
            _validate_architecture_state_dict_load,
        )
        with pytest.raises(RuntimeError, match="missing_non_backbone=1"):
            _validate_architecture_state_dict_load(
                missing=["action_expert.some_layer.weight"],
                unexpected=[],
                ckpt_path="test.pt",
            )

    def test_unexpected_key_rejected(self) -> None:
        """`unexpected` key は依然 raise (allow-list 拡張しても保護維持)。"""
        import pytest
        from inference.desktop.lower_policy.policies.ramen_ori import (
            _validate_architecture_state_dict_load,
        )
        with pytest.raises(RuntimeError, match="unexpected=1"):
            _validate_architecture_state_dict_load(
                missing=[],
                unexpected=["stale_buffer_from_old_arch"],
                ckpt_path="test.pt",
            )


class TestCompilePrefixStrip:
    """Issue #137: torch.compile 由来の key prefix 除去 (torch 不要)。"""

    def test_strip_removes_orig_mod_prefix(self) -> None:
        out = strip_compile_prefix(
            {"_orig_mod.action_expert.output_proj.weight": 1, "_orig_mod.vision.projection.bias": 2}
        )
        assert set(out) == {"action_expert.output_proj.weight", "vision.projection.bias"}

    def test_strip_is_noop_without_prefix(self) -> None:
        sd = {"action_expert.output_proj.weight": 1}
        # 正規化済 ckpt (Issue #137 Phase B 以降) は同一 object をそのまま返す
        assert strip_compile_prefix(sd) is sd

    def test_strip_leaves_unprefixed_keys_in_mixed_dict(self) -> None:
        out = strip_compile_prefix({"_orig_mod.a": 1, "b": 2})
        assert set(out) == {"a", "b"}


@pytest.mark.skipif(not HAS_TORCH, reason="torch tensors required for stale-EMA detection")
class TestStaleEmaDetection:
    """Issue #137: EMA shadow が初期化のままの ckpt を検出する。"""

    def test_stale_when_ema_zero_but_model_trained(self) -> None:
        # output_proj は zero-init。EMA が未更新なら 0 のまま、model 側は学習で非ゼロ。
        import torch

        ema = {"action_expert.output_proj.weight": torch.zeros(4)}
        model = {"action_expert.output_proj.weight": torch.full((4,), 0.007)}
        assert ema_state_dict_is_stale(ema, model) is True

    def test_not_stale_when_both_updated(self) -> None:
        import torch

        ema = {"action_expert.output_proj.weight": torch.full((4,), 0.006)}
        model = {"action_expert.output_proj.weight": torch.full((4,), 0.007)}
        assert ema_state_dict_is_stale(ema, model) is False

    def test_not_stale_when_both_zero(self) -> None:
        # 学習しても 0 のままの層は「未更新」の証拠にならない (片側判定を守る)
        import torch

        ema = {"x": torch.zeros(4)}
        model = {"x": torch.zeros(4)}
        assert ema_state_dict_is_stale(ema, model) is False

    def test_ignores_keys_absent_from_model(self) -> None:
        import torch

        assert ema_state_dict_is_stale({"only_in_ema": torch.zeros(4)}, {}) is False

    def test_ignores_shape_mismatch(self) -> None:
        import torch

        ema = {"x": torch.zeros(4)}
        model = {"x": torch.full((8,), 0.007)}
        assert ema_state_dict_is_stale(ema, model) is False


class TestRamenOriAsyncReplanning:
    """Issue #137 Phase C-1b: async chunk 実行。

    学習データは 30fps だが、同期経路は毎 tick 推論 (実測 50.8ms) が cadence を
    律速して 16.2Hz しか出ていない。async 化はそのズレを埋めるためのもの。
    **replan_family 未設定 (既定) では従来動作のまま**であることを併せて固定する。
    """

    def _policy(self, **overrides) -> RamenOriPolicy:
        defaults = dict(
            mode="none",
            ckpt_ref="Team-RAMEN/foo",
            dtype="fp32",
            cams=CAMERAS,
            execution_steps=8,
        )
        defaults.update(overrides)
        policy = RamenOriPolicy(PolicyConfig(**defaults))
        chunk = np.tile(
            np.arange(CHUNK_LEN, dtype=np.float32)[:, None], (1, ACTION_DIM)
        )
        policy._in_process_predict_chunk_19d = (
            lambda _obs, *, rtc_step=None: (chunk.copy(), 50.8)
        )
        return policy

    def _obs(self) -> Observation:
        return Observation(
            frames_bgr={cam: np.zeros((480, 640, 3), dtype=np.uint8) for cam in CAMERAS},
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_MOVE_TABLE_BASE,
            language=None,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )

    def test_execution_horizon_is_one(self) -> None:
        # 自前で chunk を合流させるので VlaSkill 側の queue 経路は使わない
        assert RamenOriPolicy.EXECUTION_HORIZON == 1

    def test_sync_default_returns_the_full_chunk(self) -> None:
        # replan_family 未設定 = 従来動作 (chunk 全体を返し VlaSkill が row 0 を使う)
        policy = self._policy()
        action = policy.predict(self._obs())
        assert action.action_chunk.shape == (CHUNK_LEN, ACTION_DIM)
        assert action.metadata["chunk_source"] == "sync"
        assert policy._pipeline is None

    def test_async_returns_a_single_blended_row(self) -> None:
        policy = self._policy(replan_family="ramen_ori_flow_v1")
        action = policy.predict(self._obs())
        assert action.action_chunk.shape == (1, ACTION_DIM)
        assert action.metadata["chunk_source"] == "sync"  # seed
        assert policy._pipeline is not None

    def test_pipeline_is_lazy(self) -> None:
        policy = self._policy(replan_family="ramen_ori_flow_v1")
        assert policy._pipeline is None
        policy.predict(self._obs())
        assert policy._pipeline is not None
        assert policy._pipeline_lead_steps == 8 - 5  # execution 8 - replan_after 5

    def test_step_counter_advances_and_blends(self) -> None:
        policy = self._policy(replan_family="ramen_ori_flow_v1")
        obs = self._obs()
        for _ in range(5):
            action = policy.predict(obs)
        assert policy._current_step == 5
        assert action.metadata["blended_from_n_candidates"] >= 1
        # seed chunk を消費している間は前進する (chunk 行が step に対応)
        assert action.action_chunk.shape == (1, ACTION_DIM)

    def test_reset_clears_async_state(self) -> None:
        policy = self._policy(replan_family="ramen_ori_flow_v1")
        policy.predict(self._obs())
        assert policy._pipeline is not None
        policy.reset()
        assert policy._pipeline is None
        assert policy._current_step == 0
        assert policy._pending_submit_step is None
        assert policy._last_seen_obs is None

    def test_reset_is_safe_without_pipeline(self) -> None:
        policy = self._policy()
        policy.reset()  # no raise
        assert policy._current_step == 0

    def test_execution_steps_at_or_above_chunk_len_warns(self, capsys) -> None:
        # 重なり 0 = temporal ensemble が一度も blend しない (GR00T で実測した落とし穴)
        policy = self._policy(
            replan_family="ramen_ori_flow_v1", execution_steps=CHUNK_LEN
        )
        policy.predict(self._obs())
        assert "never overlap" in capsys.readouterr().err

    def test_inference_is_serialized(self) -> None:
        # predict は stateless なので correctness ではなく VRAM ピーク対策
        import threading as _threading

        policy = self._policy()
        active = 0
        max_active = 0
        book = _threading.Lock()

        def stub(_obs, *, rtc_step=None):
            nonlocal active, max_active
            with book:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with book:
                active -= 1
            return np.zeros((CHUNK_LEN, ACTION_DIM), dtype=np.float32), 1.0

        policy._in_process_predict_chunk_19d = stub
        threads = [
            _threading.Thread(target=policy._sync_predict_chunk_19d, args=(None,))
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
        assert max_active == 1


class TestRamenOriReplanningProfile:
    def test_profile_values(self) -> None:
        from inference.desktop.lower_policy.async_replanning import (
            family_replanning_schedule,
        )

        # 実測 latency median 50.8ms / max 79.8ms、30Hz で 1.5-2.4 tick。
        # lead 2 (66ms) だと p90 で間に合わないので 3 (100ms)。
        replan_after, max_age = family_replanning_schedule("ramen_ori_flow_v1", 8)
        assert replan_after == 5  # execution_steps 8 - lead 3
        assert max_age == 0.25


class TestRamenOriRtcPrefix:
    """Issue #137 Phase C-3: RTC prefix の re-anchoring。

    最重要は **rel の prefix が `reconstruct_arms_abs_from_dq_norm` の逆になって
    いる**こと。ここが崩れると model には別の軌道が拘束として渡る。
    """

    MEAN = 0.01
    STD = 0.05

    def _policy(self, **overrides) -> RamenOriPolicy:
        torch = pytest.importorskip("torch")
        from types import SimpleNamespace

        defaults = dict(
            mode="none",
            ckpt_ref="Team-RAMEN/foo",
            dtype="fp32",
            device="cpu",
            cams=CAMERAS,
            execution_steps=8,
        )
        defaults.update(overrides)
        policy = RamenOriPolicy(PolicyConfig(**defaults))
        policy._device = "cpu"
        policy._model = SimpleNamespace(
            _relative_arms_mean=torch.full((14,), self.MEAN),
            _relative_arms_std=torch.full((14,), self.STD),
        )
        policy._model_action_dim = 16
        policy._model_chunk_len = CHUNK_LEN
        return policy

    def _torch_batch(self, arms_current):
        torch = pytest.importorskip("torch")
        state = torch.zeros(1, STATE_DIM)
        state[0, 3:17] = torch.as_tensor(arms_current)
        return {"state": state}

    def _absolute_chunk(self, rows: int = 8) -> np.ndarray:
        rng = np.random.default_rng(4)
        return rng.uniform(-0.5, 0.5, size=(rows, ACTION_DIM)).astype(np.float32)

    def test_abs_prefix_only_drops_the_waist_pad(self) -> None:
        # abs は model 出力が絶対関節角そのもの = 恒等変換
        policy = self._policy(action_space="abs")
        chunk = self._absolute_chunk()
        out = policy._encode_prefix_to_model_space(
            chunk, self._torch_batch(np.zeros(14))
        )
        assert tuple(out.shape) == (8, 16)
        np.testing.assert_allclose(out.numpy(), chunk[:, 3:], rtol=0, atol=0)

    def test_relative_rtc_is_rejected_before_model_load(self) -> None:
        from inference.desktop.lower_policy.rtc import RtcConfig

        with pytest.raises(ValueError, match="not safe with action_space='rel'"):
            self._policy(
                action_space="rel",
                replan_family="ramen_ori_flow_v1",
                rtc=RtcConfig(enabled=True),
            )

    def test_relative_rtc_requires_explicit_experimental_override(self) -> None:
        from inference.desktop.lower_policy.rtc import RtcConfig

        policy = self._policy(
            action_space="rel",
            replan_family="ramen_ori_flow_v1",
            rtc=RtcConfig(
                enabled=True,
                allow_experimental_relative_action=True,
            ),
        )
        assert policy.cfg.rtc.enabled is True
        assert policy.cfg.rtc.allow_experimental_relative_action is True

    def test_rel_prefix_is_the_inverse_of_reconstruction(self) -> None:
        torch = pytest.importorskip("torch")
        from model.ramen_ori.relative_action import (
            reconstruct_arms_abs_from_dq_norm,
        )

        policy = self._policy(action_space="rel")
        chunk = self._absolute_chunk()
        arms_current = np.linspace(-0.2, 0.2, 14).astype(np.float32)
        prefix = policy._encode_prefix_to_model_space(
            chunk, self._torch_batch(arms_current)
        )
        # 復元すると元の絶対 arms に戻る = re-anchor が正しい
        restored = reconstruct_arms_abs_from_dq_norm(
            prefix[:, 0:14],
            torch.as_tensor(arms_current),
            torch.full((14,), self.MEAN),
            torch.full((14,), self.STD),
        )
        np.testing.assert_allclose(
            restored.numpy(), chunk[:, 3:17], rtol=1e-4, atol=1e-5
        )

    def test_rel_prefix_depends_on_the_current_arms(self) -> None:
        # 起点が変われば prefix も変わる。ここが no-op だと re-anchor が効いていない。
        policy = self._policy(action_space="rel")
        chunk = self._absolute_chunk()
        first = policy._encode_prefix_to_model_space(
            chunk, self._torch_batch(np.zeros(14))
        ).numpy()
        second = policy._encode_prefix_to_model_space(
            chunk, self._torch_batch(np.full(14, 0.3))
        ).numpy()
        # 差分の起点だけが変わるので row 0 のみ変化する
        assert not np.allclose(first[0, 0:14], second[0, 0:14])
        np.testing.assert_allclose(first[1:, 0:14], second[1:, 0:14], atol=1e-6)
        # hand 2D は rel でも abs pass-through
        np.testing.assert_allclose(first[:, 14:16], second[:, 14:16], atol=1e-6)

    def test_disabled_by_default(self) -> None:
        policy = self._policy()
        assert policy._rtc_leftover is None
        kwargs, metadata = policy._build_rtc_prefix({}, 0)
        assert kwargs == {}
        assert metadata["rtc_enabled"] is False

    def test_enabled_builds_prefix_and_strength(self) -> None:
        from inference.desktop.lower_policy.rtc import RtcConfig

        policy = self._policy(
            action_space="abs",
            replan_family="ramen_ori_flow_v1",
            rtc=RtcConfig(enabled=True),
        )
        policy._rtc_leftover.store(
            self._absolute_chunk(rows=CHUNK_LEN), origin_step=0
        )
        policy._rtc_delay.add(0.0508)
        for _ in range(4):
            policy._rtc_tick_deltas.append(0.0333)
        kwargs, metadata = policy._build_rtc_prefix(
            self._torch_batch(np.zeros(14)), 8
        )
        assert set(kwargs) == {"prefix", "velocity_strength"}
        assert tuple(kwargs["prefix"].shape) == (8, 16)
        assert tuple(kwargs["velocity_strength"].shape) == (CHUNK_LEN,)
        # 50.8ms / 33.3ms = 1.53 → 切り上げ 2
        assert metadata["rtc_frozen_steps"] == 2
        strength = kwargs["velocity_strength"].numpy()
        assert np.all(strength[:2] == 0.0)
        assert np.all(strength[8:] == 1.0)

    def test_sync_execution_freezes_nothing(self) -> None:
        from inference.desktop.lower_policy.rtc import RtcConfig

        policy = self._policy(action_space="abs", rtc=RtcConfig(enabled=True))
        policy._rtc_leftover.store(
            self._absolute_chunk(rows=CHUNK_LEN), origin_step=0
        )
        policy._rtc_delay.add(0.0508)
        kwargs, metadata = policy._build_rtc_prefix(
            self._torch_batch(np.zeros(14)), 8
        )
        assert metadata["rtc_frozen_steps"] == 0
        assert metadata["rtc_async_execution"] is False

    def test_error_disables_rtc_for_the_instance(self) -> None:
        from inference.desktop.lower_policy.rtc import RtcConfig

        policy = self._policy(action_space="abs", rtc=RtcConfig(enabled=True))
        policy._rtc_leftover.store(
            self._absolute_chunk(rows=CHUNK_LEN), origin_step=0
        )
        policy._model_action_dim = 17  # 未知の action 空間
        kwargs, metadata = policy._build_rtc_prefix(
            self._torch_batch(np.zeros(14)), 8
        )
        assert kwargs == {}
        assert metadata["rtc_enabled"] is False
        assert policy._rtc_disabled_after_error is True


class TestRamenOriWarmupAndClose:
    """Issue #137: 実 ckpt smoke で見つかった 2 件の回帰ガード。"""

    def _policy(self, **overrides) -> RamenOriPolicy:
        from inference.desktop.lower_policy.rtc import RtcConfig

        defaults = dict(
            mode="none",
            ckpt_ref="Team-RAMEN/foo",
            dtype="fp32",
            cams=CAMERAS,
            execution_steps=8,
            rtc=RtcConfig(enabled=True),
        )
        defaults.update(overrides)
        policy = RamenOriPolicy(PolicyConfig(**defaults))
        policy._model = object()
        return policy

    def test_warmup_discards_its_own_rtc_state(self) -> None:
        # warmup は dummy observation + 初回 JIT (~500ms-2s)。これを遅延推定
        # (window 内 max) に残すと d が過大評価され、overlap 全域が凍結される
        # (実測: 966ms の warmup latency で frozen が overlap と同値になった)。
        policy = self._policy()
        calls = []
        policy._make_dummy_observation = lambda: "dummy"
        policy.predict = lambda obs: calls.append(obs)

        def dirty(_obs):
            policy._rtc_delay.add(0.966)
            policy._rtc_tick_deltas.append(0.033)
            policy._rtc_leftover.store(
                np.zeros((CHUNK_LEN, ACTION_DIM), dtype=np.float32), origin_step=0
            )
            policy._current_step += 1

        policy.predict = dirty
        policy.warmup(n_iter=2)

        assert policy._rtc_delay.max_latency_s is None
        assert len(policy._rtc_tick_deltas) == 0
        assert policy._rtc_leftover.remaining(0) is None
        assert policy._current_step == 0

    def test_close_shuts_down_the_async_pipeline(self) -> None:
        # daemon thread が生きたまま model を落とすとプロセス終了時に abort する
        # (実測: 実 ckpt smoke で core dump)。
        policy = self._policy(replan_family="ramen_ori_flow_v1")
        closed = []

        class FakePipeline:
            def close(self, **kwargs):
                closed.append(kwargs)

        policy._pipeline = FakePipeline()
        policy.close()
        assert closed, "close() must shut the async pipeline down"
        assert policy._pipeline is None

    def test_close_is_safe_without_pipeline(self) -> None:
        policy = self._policy()
        policy.close()  # no raise


# =========================================================================
# Issue #141 P8: 約束つき ckpt (再学習した RAMEN-Ori)
# =========================================================================
class TestValidateContract:
    """ckpt の約束と推論の設定が合っているかを、重みを読む前に確かめる (P8-3)。"""

    def _contract(self, **overrides) -> dict:
        contract = {
            "version": 1,
            "skills": [{"id": 4, "name": "rotate_table_base"}, {"id": 0}],
            "state": {"variant": "71d", "version": 2},
            "action": {"space": "abs", "dim": 16, "chunk_len": 32},
            "images": {
                "camera_keys": [
                    "observation.images.cam_0",
                    "observation.images.cam_2",
                    "observation.images.cam_3",
                ],
                "img_size": 224,
                "overlay": {"yolo_ckpt": "repo@sha", "conf": 0.30, "thickness": 2},
            },
        }
        contract.update(overrides)
        return contract

    def _cfg(self, **overrides) -> PolicyConfig:
        kwargs = dict(
            mode="overlay",
            ckpt_ref="Team-RAMEN/foo@abc",
            dtype="fp32",
            cams=CAMERAS_3CAM_PHASE_K,
            skill_id=4,
            yolo_ckpt_ref="repo@sha",
        )
        kwargs.update(overrides)
        return PolicyConfig(**kwargs)

    def test_matching_contract_passes(self):
        validate_contract(self._contract(), self._cfg())

    def test_unsupported_version_is_rejected(self):
        with pytest.raises(RuntimeError, match="contract version"):
            validate_contract(self._contract(version=2), self._cfg())

    def test_skill_id_outside_the_ckpt_is_rejected(self):
        with pytest.raises(RuntimeError, match="skill_id=5"):
            validate_contract(self._contract(), self._cfg(skill_id=5))

    def test_missing_skill_id_is_rejected(self):
        with pytest.raises(RuntimeError, match="does not set skill_id"):
            validate_contract(self._contract(), self._cfg(skill_id=None))

    def test_a_different_yolo_weight_is_rejected(self):
        with pytest.raises(RuntimeError, match="overlay YOLO weight"):
            validate_contract(self._contract(), self._cfg(yolo_ckpt_ref="repo@other"))

    def test_a_renamed_yolo_repo_at_the_same_commit_passes(self):
        """HF で repo を改名しても commit の sha は同じ。ckpt は改名の前の名前を持っている (Issue #190)。"""
        sha = "8221d0aef45d63666537c1f030810fb9b56dd1ed"
        contract = self._contract()
        contract["images"]["overlay"]["yolo_ckpt"] = f"repo@{sha}"
        validate_contract(contract, self._cfg(yolo_ckpt_ref=f"renamed@{sha}"))

    def test_a_branch_name_does_not_match_across_repos(self):
        """main などの branch 名は repo ごとに中身が違うので、名前まで同じでなければ通さない。"""
        contract = self._contract()
        contract["images"]["overlay"]["yolo_ckpt"] = "repo@main"
        validate_contract(contract, self._cfg(yolo_ckpt_ref="repo@main"))
        with pytest.raises(RuntimeError, match="overlay YOLO weight"):
            validate_contract(contract, self._cfg(yolo_ckpt_ref="renamed@main"))

    def test_an_unpinned_yolo_ref_must_match_as_a_whole(self):
        contract = self._contract()
        contract["images"]["overlay"]["yolo_ckpt"] = "repo"
        validate_contract(contract, self._cfg(yolo_ckpt_ref="repo"))
        with pytest.raises(RuntimeError, match="overlay YOLO weight"):
            validate_contract(contract, self._cfg(yolo_ckpt_ref="renamed"))

    def test_a_different_camera_set_is_rejected(self):
        with pytest.raises(RuntimeError, match="cameras"):
            validate_contract(self._contract(), self._cfg(cams=CAMERAS))

    def test_a_different_action_space_is_rejected(self):
        with pytest.raises(RuntimeError, match="action space"):
            validate_contract(self._contract(), self._cfg(action_space="rel"))

    def test_overlay_threshold_mismatch_is_rejected(self):
        contract = self._contract()
        contract["images"]["overlay"]["conf"] = 0.25
        with pytest.raises(RuntimeError, match="overlay conf"):
            validate_contract(contract, self._cfg())

    def test_every_problem_is_listed_at_once(self):
        with pytest.raises(RuntimeError) as excinfo:
            validate_contract(self._contract(), self._cfg(skill_id=9, cams=CAMERAS))
        message = str(excinfo.value)
        assert "skill_id=9" in message and "cameras" in message


class TestStateVariant2:
    """再学習の state (版 2) は腰の tracking_err を 0 にする (P8-4)。"""

    def _raw(self) -> RawRobotState:
        return RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
            last_action_19d=np.ones(19, dtype=np.float32),
        )

    def test_variant_1_keeps_the_waist_tracking_error(self):
        state = build_state_from_raw(self._raw(), state_variant=1)
        np.testing.assert_allclose(state[19:22], 1.0)

    def test_variant_2_zeroes_the_waist_tracking_error(self):
        state = build_state_from_raw(self._raw(), state_variant=2)
        np.testing.assert_allclose(state[19:22], 0.0)
        np.testing.assert_allclose(state[22:38], 1.0)   # 腕と hand はそのまま

    def test_the_first_tick_of_a_skill_has_no_tracking_error(self):
        raw = RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
        )
        np.testing.assert_allclose(build_state_from_raw(raw, state_variant=2)[19:38], 0.0)

    def test_a_policy_without_a_contract_uses_variant_1(self):
        policy = RamenOriPolicy(cfg=PolicyConfig(mode="none", ckpt_ref="x"))
        np.testing.assert_allclose(policy.build_state_from_raw(self._raw())[19:22], 1.0)

    def test_a_policy_with_a_contract_uses_its_variant(self):
        policy = RamenOriPolicy(
            cfg=PolicyConfig(mode="none", ckpt_ref="x"),
            _contract={"version": 1, "state": {"variant": "71d", "version": 2}},
        )
        np.testing.assert_allclose(policy.build_state_from_raw(self._raw())[19:22], 0.0)


class TestMemoryToken:
    """memory の ckpt は学習と同じ tracker を tick ごとに回す (P8-6)。"""

    class _StubTracker:
        def __init__(self) -> None:
            self.calls: list[tuple] = []
            self.resets = 0

        def update(self, class_id, conf, verts, hand_pos):
            self.calls.append((class_id, conf, verts, hand_pos))
            return np.arange(51, dtype=np.float32)

        def reset(self) -> None:
            self.resets += 1

    class _StubFk:
        def forward_detailed(self, command):
            import torch

            self.command = command
            return {
                "left_hand": torch.zeros(1, 3),
                "right_hand": torch.ones(1, 3),
            }

    class _StubModel:
        def __init__(self, *, memory=None, fk=None) -> None:
            self.memory = memory
            self.fk = fk

    def _policy(self, *, memory=True):
        policy = RamenOriPolicy(
            cfg=PolicyConfig(mode="overlay", ckpt_ref="x", device="cpu"),
            _model=self._StubModel(memory=object() if memory else None, fk=self._StubFk()),
            _device="cpu",
            _contract={"version": 1, "state": {"variant": "71d", "version": 2}, "memory": {"version": 1}},
        )
        policy._memory_tracker = self._StubTracker()
        return policy

    def _obs(self, state: np.ndarray, detections=None) -> Observation:
        return Observation(
            frames_bgr={},
            frames_bgr_prev=None,
            state=state,
            skill_id=4,
            language=None,
            obb_detections=detections,
            timestamp_ns=1,
        )

    def test_memory_is_off_without_a_memory_module(self):
        assert not self._policy(memory=False)._memory_enabled()

    def test_the_first_tick_has_no_previous_command(self):
        pytest.importorskip("torch")
        policy = self._policy()
        policy._memory_vector(self._obs(np.zeros(71, dtype=np.float32)))
        assert policy._memory_tracker.calls[0][3] is None

    def test_the_previous_command_is_rebuilt_from_the_state(self):
        """joint + tracking_err = 前の tick に送った指令 (Issue #141 P8-6 の案 a)。"""
        pytest.importorskip("torch")
        policy = self._policy()
        state = np.zeros(71, dtype=np.float32)
        state[0:19] = 0.25
        state[19:38] = 0.5
        policy._memory_vector(self._obs(state))       # 1 tick 目は学習の t=0 と同じで None
        policy._memory_vector(self._obs(state))       # 2 tick 目から FK に渡る
        command = policy._model.fk.command
        assert command.shape == (1, 19)
        np.testing.assert_allclose(command.numpy()[0], 0.75, rtol=1e-6)

    def test_detections_are_passed_through_for_the_memory_camera(self):
        pytest.importorskip("torch")
        policy = self._policy()
        detection = OBBDetection(
            class_id=3, class_name="leg", confidence=0.9,
            verts=np.zeros((4, 2), dtype=np.float32),
        )
        policy._memory_vector(
            self._obs(np.zeros(71, dtype=np.float32), {CameraKey.HEAD_LEFT: [detection]})
        )
        class_id, conf, verts, _ = policy._memory_tracker.calls[0]
        assert class_id.tolist() == [3]
        assert conf.tolist() == [0.9]
        assert verts.shape == (1, 4, 2)

    def test_detections_from_another_camera_are_ignored(self):
        pytest.importorskip("torch")
        policy = self._policy()
        detection = OBBDetection(
            class_id=3, class_name="leg", confidence=0.9,
            verts=np.zeros((4, 2), dtype=np.float32),
        )
        policy._memory_vector(
            self._obs(np.zeros(71, dtype=np.float32), {CameraKey.WRIST_LEFT: [detection]})
        )
        class_id, _, verts, _ = policy._memory_tracker.calls[0]
        assert class_id.size == 0 and verts.shape == (0, 4, 2)

    def test_the_command_comes_from_this_tick_not_the_previous_one(self):
        """FK に渡すのは「今の state が持つ前の指令」であること (review 指摘 7)。

        学習は frame t で `hand_pos[t-1]` = FK(action[t-1]) を渡す
        (model/ramen_ori/data_lerobot.py)。推論の state はその action を持っているので、
        控えて次の tick で使うと学習より 1 tick (33 ms) 古い手先が memory に入る。
        """
        pytest.importorskip("torch")
        policy = self._policy()
        first = np.zeros(71, dtype=np.float32)
        first[0:19] = 0.25
        first[19:38] = 0.25                           # 戻すと 0.5
        second = np.zeros(71, dtype=np.float32)
        second[0:19] = 0.5
        second[19:38] = 0.5                           # 戻すと 1.0
        policy._memory_vector(self._obs(first))
        policy._memory_vector(self._obs(second))
        np.testing.assert_allclose(
            policy._model.fk.command.numpy()[0], 1.0, rtol=1e-6
        )

    def test_reset_forgets_the_tracker_and_the_tick_count(self):
        pytest.importorskip("torch")
        policy = self._policy()
        policy._memory_vector(self._obs(np.zeros(71, dtype=np.float32)))
        assert policy._memory_ticks == 1
        policy.reset()
        assert policy._memory_tracker.resets == 1
        assert policy._memory_ticks == 0


class TestFirstTickPrevFrame:
    """skill の 1 tick 目は画像差分を 0 にする (Issue #141 束 1-3 / INF-4)。

    学習も区間の頭では `images_prev = images` (ep 境界の clamp) で差分 0。zeros を入れると
    「画像がまるごと現れた」差分になり、学習に無い入力で 1 tick 目の指令を出すことになる。
    """

    def _frames(self, value: int) -> dict:
        return {
            cam: np.full((480, 640, 3), value, dtype=np.uint8)
            for cam in CAMERAS_3CAM_PHASE_K
        }

    def _obs(self, prev=None) -> Observation:
        return Observation(
            frames_bgr=self._frames(30),
            frames_bgr_prev=prev,
            state=np.zeros(71, dtype=np.float32),
            skill_id=4,
            language=None,
            obb_detections=None,
            timestamp_ns=1,
        )

    def test_without_a_previous_frame_the_difference_is_zero(self):
        pytest.importorskip("torch")
        batch = build_batch_dict(self._obs(), mode="none", cams=CAMERAS_3CAM_PHASE_K)
        np.testing.assert_array_equal(batch["images_prev"], batch["images"])

    def test_a_previous_frame_is_used_when_present(self):
        pytest.importorskip("torch")
        batch = build_batch_dict(
            self._obs(prev=self._frames(200)), mode="none", cams=CAMERAS_3CAM_PHASE_K
        )
        assert not np.allclose(batch["images_prev"], batch["images"])

    def test_a_camera_missing_from_the_previous_frame_falls_back_to_now(self):
        pytest.importorskip("torch")
        prev = self._frames(200)
        del prev[CAMERAS_3CAM_PHASE_K[1]]
        batch = build_batch_dict(
            self._obs(prev=prev), mode="none", cams=CAMERAS_3CAM_PHASE_K
        )
        np.testing.assert_array_equal(batch["images_prev"][1], batch["images"][1])
        assert not np.allclose(batch["images_prev"][0], batch["images"][0])

    def test_a_ckpt_without_temporal_keeps_zeros(self):
        """画像差分を使わない ckpt では前処理を飛ばす (中身は model が読まない)。"""
        pytest.importorskip("torch")
        batch = build_batch_dict(
            self._obs(), mode="none", cams=CAMERAS_3CAM_PHASE_K, with_prev_frames=False
        )
        np.testing.assert_array_equal(batch["images_prev"], 0.0)
