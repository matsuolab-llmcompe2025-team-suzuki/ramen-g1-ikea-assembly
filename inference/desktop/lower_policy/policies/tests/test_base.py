"""Phase 1 smoke: base.py の dataclass 契約 (Issue #125)。

- CameraKey → cam_id / training_key mapping が training-side (`data_lerobot.py`
  の cam vocabulary) と一致するかは実 config 依存 (Phase 3 の RAMEN-Ori loader
  smoke で最終検証)。ここでは type/enum 側の invariant のみ確認。
- Observation / PolicyAction / PolicyConfig は frozen dataclass、mutation 拒絶を
  確認 (per-tick に new instance 作る運用を強制)。
- skill_id / language / obb_detections は None 許容 (Skill wrapper が埋める前提)。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    Observation,
    Policy,
    PolicyAction,
    PolicyConfig,
)
from inference.desktop.perception.yolo_obb import OBBDetection


class TestCameraKey:
    def test_cam_id_matches_training_order(self) -> None:
        # data_lerobot.py:default_camera_keys() の順序 (cam_0..3) を単一の真実として、
        # inference 側 enum の cam_id 割当が一致することを保証する。
        assert CameraKey.HEAD_LEFT.cam_id == 0
        assert CameraKey.HEAD_RIGHT.cam_id == 1
        assert CameraKey.WRIST_LEFT.cam_id == 2
        assert CameraKey.WRIST_RIGHT.cam_id == 3

    def test_training_key_format(self) -> None:
        # LeRobot column key の format は `observation.images.cam_{i}` (data_lerobot.py:70)
        assert CameraKey.HEAD_LEFT.training_key == "observation.images.cam_0"
        assert CameraKey.WRIST_RIGHT.training_key == "observation.images.cam_3"

    def test_is_wrist_matches_training_dispatch(self) -> None:
        # data_lerobot.py:is_wrist() は "wrist" substring 判定。enum 名の prefix で近似。
        assert not CameraKey.HEAD_LEFT.is_wrist
        assert not CameraKey.HEAD_RIGHT.is_wrist
        assert CameraKey.WRIST_LEFT.is_wrist
        assert CameraKey.WRIST_RIGHT.is_wrist


class TestObservation:
    def _make(
        self,
        skill_id: int | None = 0,
        language: str | None = None,
        obb_detections: list[OBBDetection] | None = None,
        frames_bgr_prev: dict | None = None,
    ) -> Observation:
        H, W = 480, 640
        return Observation(
            frames_bgr={
                CameraKey.HEAD_LEFT: np.zeros((H, W, 3), dtype=np.uint8),
                CameraKey.HEAD_RIGHT: np.zeros((H, W, 3), dtype=np.uint8),
            },
            frames_bgr_prev=frames_bgr_prev,
            state=np.zeros(71, dtype=np.float32),
            skill_id=skill_id,
            language=language,
            obb_detections=obb_detections,
            timestamp_ns=time.monotonic_ns(),
        )

    def test_construct_ramen_ori_style(self) -> None:
        # Skill wrapper が RAMEN-Ori 用に skill_id を埋め、language は None
        obs = self._make(skill_id=5, language=None)
        assert obs.state.shape == (71,)
        assert obs.frames_bgr[CameraKey.HEAD_LEFT].dtype == np.uint8
        assert obs.skill_id == 5
        assert obs.language is None

    def test_construct_groot_style(self) -> None:
        # Skill wrapper が GR00T 用に language を埋め、skill_id は None
        obs = self._make(skill_id=None, language="rotate and move table base")
        assert obs.skill_id is None
        assert obs.language == "rotate and move table base"

    def test_obb_detections_optional(self) -> None:
        # mode=none Policy は obb_detections を無視する前提、None 許容
        obs = self._make(obb_detections=None)
        assert obs.obb_detections is None
        # mode=overlay / precomputed_token では per-cam dict を渡す (YOLO は
        # head_left + head_right の 2 cam で走る前提、cam attribution 保持)
        det = OBBDetection(
            class_id=0,
            class_name="leg",
            confidence=0.9,
            verts=np.zeros((4, 2), dtype=np.float32),
        )
        obs2 = self._make(obb_detections={CameraKey.HEAD_LEFT: [det]})
        assert CameraKey.HEAD_LEFT in obs2.obb_detections
        assert len(obs2.obb_detections[CameraKey.HEAD_LEFT]) == 1

    def test_frames_bgr_prev_optional(self) -> None:
        # First tick は None 許容 (Policy 側で zeros に fill)
        obs = self._make(frames_bgr_prev=None)
        assert obs.frames_bgr_prev is None
        # 通常 tick は前 frame dict を渡す
        prev = {CameraKey.HEAD_LEFT: np.zeros((480, 640, 3), dtype=np.uint8)}
        obs2 = self._make(frames_bgr_prev=prev)
        assert CameraKey.HEAD_LEFT in obs2.frames_bgr_prev

    def test_frozen(self) -> None:
        obs = self._make()
        # dataclass(frozen=True) → set attribute 拒絶。per-tick に new instance 作る運用強制。
        with pytest.raises(Exception):  # noqa: BLE001 (FrozenInstanceError or AttributeError)
            obs.skill_id = 1  # type: ignore[misc]


class TestPolicyAction:
    def test_construct_defaults(self) -> None:
        act = PolicyAction(
            action_chunk=np.zeros((16, 19), dtype=np.float32),
            latency_ms=12.3,
        )
        assert act.action_chunk.shape == (16, 19)
        assert act.metadata == {}

    def test_metadata_is_dict(self) -> None:
        act = PolicyAction(
            action_chunk=np.zeros((16, 19), dtype=np.float32),
            latency_ms=1.0,
            metadata={"yolo_det_count": 4, "mode": "overlay"},
        )
        assert act.metadata["yolo_det_count"] == 4


class TestPolicyConfig:
    def test_valid_none_mode(self) -> None:
        cfg = PolicyConfig(mode="none", ckpt_ref="Team-RAMEN/foo@abc")
        assert cfg.mode == "none"
        assert cfg.dtype == "fp32"
        # cams default は 4 cam full set (RAMEN-Ori 想定)
        assert len(cfg.cams) == 4

    def test_valid_overlay_mode(self) -> None:
        # overlay mode は yolo_ckpt_ref を必要としない (YOLO は perception 側で走る)
        cfg = PolicyConfig(mode="overlay", ckpt_ref="Team-RAMEN/foo@abc")
        assert cfg.mode == "overlay"

    def test_valid_precomputed_token_mode(self) -> None:
        cfg = PolicyConfig(mode="precomputed_token", ckpt_ref="Team-RAMEN/foo@abc")
        assert cfg.mode == "precomputed_token"

    def test_groot_3cam_config(self) -> None:
        # GR00T は 3 cam (head_left + wrist_left + wrist_right、head_right ignore)
        cfg = PolicyConfig(
            mode="none",
            ckpt_ref="Team-RAMEN/foo@abc",
            cams=(
                CameraKey.HEAD_LEFT,
                CameraKey.WRIST_LEFT,
                CameraKey.WRIST_RIGHT,
            ),
        )
        assert len(cfg.cams) == 3
        assert CameraKey.HEAD_RIGHT not in cfg.cams

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            PolicyConfig(mode="unknown", ckpt_ref="Team-RAMEN/foo@abc")

    def test_invalid_dtype(self) -> None:
        with pytest.raises(ValueError, match="dtype must be"):
            PolicyConfig(mode="none", ckpt_ref="Team-RAMEN/foo@abc", dtype="int8")

    def test_action_space_default_abs(self) -> None:
        # Issue #132 Phase C: field 未指定なら "abs" (既存 slot 全部が backward compat)
        cfg = PolicyConfig(mode="none", ckpt_ref="Team-RAMEN/foo@abc")
        assert cfg.action_space == "abs"

    def test_action_space_rel_valid(self) -> None:
        cfg = PolicyConfig(
            mode="overlay",
            ckpt_ref="Team-RAMEN/foo@abc",
            action_space="rel",
        )
        assert cfg.action_space == "rel"

    def test_action_space_invalid_rejected(self) -> None:
        with pytest.raises(ValueError, match="'abs' or 'rel'"):
            PolicyConfig(
                mode="none",
                ckpt_ref="Team-RAMEN/foo@abc",
                action_space="delta_ee",
            )

    def test_checkpoint_subdir_must_be_safe_relative_path(self) -> None:
        cfg = PolicyConfig(
            mode="none",
            ckpt_ref="Team-RAMEN/foo@abc",
            checkpoint_subdir="checkpoints/020000/pretrained_model",
        )
        assert cfg.checkpoint_subdir == "checkpoints/020000/pretrained_model"
        for invalid in ("", "/absolute", "../escape", "checkpoints/../escape"):
            with pytest.raises(ValueError, match="checkpoint_subdir"):
                PolicyConfig(
                    mode="none",
                    ckpt_ref="Team-RAMEN/foo@abc",
                    checkpoint_subdir=invalid,
                )


class TestPolicyProtocol:
    def test_runtime_checkable(self) -> None:
        # runtime_checkable(Protocol) は @classmethod + method 名の存在で判定する。
        # 部分実装 (predict のみ) は Protocol match しないことを確認。
        class Incomplete:
            def predict(self, obs):  # noqa: ANN001
                pass

        assert not isinstance(Incomplete(), Policy)

    def test_full_impl_matches_protocol(self) -> None:
        class Full:
            @staticmethod
            def build_state_from_raw(raw):  # noqa: ANN001
                return np.zeros(71, dtype=np.float32)

            @classmethod
            def from_ckpt(cls, cfg):  # noqa: ANN001
                return cls()

            def warmup(self, n_iter=5):  # noqa: ANN001
                pass

            def predict(self, obs):  # noqa: ANN001
                return PolicyAction(
                    action_chunk=np.zeros((16, 19), dtype=np.float32),
                    latency_ms=0.0,
                )

            def close(self):
                pass

        assert isinstance(Full(), Policy)
