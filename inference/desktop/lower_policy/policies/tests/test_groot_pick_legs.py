"""pick_legs GR00T policy/worker contract tests (Issue #125)。

- 契約定数 (STATE_DIM=38, ACTION_DIM=38, CHUNK_LEN=16, CAMERAS=4, CKPT_SUBDIR)
- validate / build_state_from_raw / build_batch_dict の shape/key 検証
- from_ckpt はisolated workerへ委譲し、predictは38Dから共有19Dへ変換
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    Observation,
    PolicyConfig,
    RawRobotState,
)
from inference.desktop.lower_policy.policies.groot_pick_legs import (
    ACTION_DIM,
    CAMERAS,
    CHUNK_LEN,
    CKPT_SUBDIR,
    DEFAULT_LANGUAGE_PROMPT,
    EMBODIMENT_ID,
    Gr00tPolicyPickLegs,
    IMAGE_HW,
    MODEL_REVISION,
    RUNTIME_CHECKPOINT_FILES,
    STATE_DIM,
    _PickLegsWorkerClient,
    build_batch_dict,
    build_state_from_raw,
    validate_pick_legs_config,
)


class TestConstants:
    def test_state_dim(self) -> None:
        # g1_pick_leg_config.py: robot_q(36) + hand(2) = 38
        assert STATE_DIM == 38

    def test_action_dim(self) -> None:
        # g1_pick_leg_config.py: robot_q(36) + hand(2) = 38 (ABSOLUTE)
        assert ACTION_DIM == 38

    def test_chunk_len(self) -> None:
        # g1_pick_leg_config.py:action.delta_indices=list(range(0, 16))
        assert CHUNK_LEN == 16

    def test_cameras_4cam_layout(self) -> None:
        # g1_pick_leg_config.py:video.modality_keys=[cam_0, cam_1, cam_2, cam_3]
        assert CAMERAS == (
            CameraKey.HEAD_LEFT,   # cam_0
            CameraKey.HEAD_RIGHT,  # cam_1
            CameraKey.WRIST_LEFT,  # cam_2
            CameraKey.WRIST_RIGHT, # cam_3
        )

    def test_ckpt_subdir(self) -> None:
        # HF Trainer native output = checkpoint-<step>/ subdir
        assert CKPT_SUBDIR == "checkpoint-40000"

    def test_image_hw(self) -> None:
        assert IMAGE_HW == (256, 256)

    def test_default_language_prompt(self) -> None:
        assert DEFAULT_LANGUAGE_PROMPT == "pick table leg"

    def test_embodiment_id(self) -> None:
        # embodiment_id.json: unitree_g1_full_body_with_waist_height_nav_cmd = 25
        assert EMBODIMENT_ID == 25


class TestValidatePickLegsConfig:
    def _cfg(self, **overrides) -> PolicyConfig:
        defaults = dict(
            mode="none",
            ckpt_ref="Team-RAMEN/groot-n1.7-pick-legs-ver1",
            device="cuda",
            dtype="bf16",
            cams=CAMERAS,
        )
        defaults.update(overrides)
        return PolicyConfig(**defaults)

    def test_valid_default(self) -> None:
        validate_pick_legs_config(self._cfg())

    def test_valid_overlay(self) -> None:
        validate_pick_legs_config(self._cfg(mode="overlay"))

    def test_reject_precomputed_token(self) -> None:
        with pytest.raises(ValueError, match="precomputed_token"):
            validate_pick_legs_config(self._cfg(mode="precomputed_token"))

    def test_reject_3cam_layout(self) -> None:
        # task_5+7 GR00T の 3 cam layout は pick_legs では reject
        with pytest.raises(ValueError, match="4 cam layout"):
            validate_pick_legs_config(
                self._cfg(cams=(CameraKey.HEAD_LEFT, CameraKey.WRIST_LEFT, CameraKey.WRIST_RIGHT))
            )


class TestBuildBatchDict:
    def _obs(
        self,
        language: str | None = DEFAULT_LANGUAGE_PROMPT,
        state_dim: int = STATE_DIM,
        skip_cam: CameraKey | None = None,
    ) -> Observation:
        H, W = 480, 640
        frames = {cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS}
        if skip_cam is not None:
            del frames[skip_cam]
        return Observation(
            frames_bgr=frames,
            frames_bgr_prev=None,
            state=np.zeros(state_dim, dtype=np.float32),
            skill_id=None,
            language=language,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )

    def test_output_keys(self) -> None:
        batch = build_batch_dict(self._obs())
        # modality.json / g1_pick_leg_config.py の modality_keys と一致
        assert "robot_q" in batch
        assert "hand" in batch
        assert "annotation.human.task_description" in batch
        assert "cam_0" in batch
        assert "cam_1" in batch
        assert "cam_2" in batch
        assert "cam_3" in batch

    def test_state_split(self) -> None:
        obs = self._obs()
        obs.state[0:36] = 1.0  # robot_q part
        obs.state[36:38] = 0.5  # hand part
        batch = build_batch_dict(obs)
        assert batch["robot_q"].shape == (36,)
        assert batch["hand"].shape == (2,)
        assert batch["robot_q"].dtype == np.float32
        assert batch["hand"].dtype == np.float32
        np.testing.assert_array_equal(batch["robot_q"], np.ones(36, dtype=np.float32))
        np.testing.assert_array_equal(batch["hand"], np.array([0.5, 0.5], dtype=np.float32))

    def test_image_uint8_bgr(self) -> None:
        batch = build_batch_dict(self._obs())
        for i in range(4):
            img = batch[f"cam_{i}"]
            assert img.shape == (480, 640, 3)
            assert img.dtype == np.uint8

    def test_language_prompt(self) -> None:
        batch = build_batch_dict(self._obs())
        assert batch["annotation.human.task_description"] == "pick table leg"

    def test_reject_missing_language(self) -> None:
        with pytest.raises(ValueError, match="requires obs.language"):
            build_batch_dict(self._obs(language=None))

    def test_reject_wrong_state_dim(self) -> None:
        # 71D (RAMEN-Ori) や 49D (task_5+7 GR00T) を pick_legs に渡すと reject
        with pytest.raises(ValueError, match=r"\(38,\)"):
            build_batch_dict(self._obs(state_dim=49))

    def test_reject_missing_camera(self) -> None:
        with pytest.raises(KeyError, match="head_right"):
            build_batch_dict(self._obs(skip_cam=CameraKey.HEAD_RIGHT))


class TestBuildStateFromRaw:
    def _raw(self) -> RawRobotState:
        return RawRobotState(
            joint_positions=np.arange(29, dtype=np.float32),
            hand_state=np.array([0.3, 0.7], dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
            last_action_19d=None,
            joint_positions_prev=None,
        )

    def test_output_shape_and_content(self) -> None:
        state = build_state_from_raw(self._raw())
        assert state.shape == (STATE_DIM,)
        assert state.dtype == np.float32
        # stationary root proxy + joint_positions(29) + hand_state(2) = 38
        np.testing.assert_array_equal(
            state[:7],
            np.array([0, 0, 0.70, 1, 0, 0, 0], dtype=np.float32),
        )
        np.testing.assert_array_equal(state[7:36], np.arange(29, dtype=np.float32))
        np.testing.assert_array_equal(state[36:], np.array([0.3, 0.7], dtype=np.float32))

    def test_uses_official_boundary_base_orientation_when_available(self) -> None:
        raw = self._raw()
        raw = RawRobotState(
            joint_positions=raw.joint_positions,
            hand_state=raw.hand_state,
            ee_state=raw.ee_state,
            root_pose=np.asarray([0, 0, 0.70, 0.9, 0.1, 0.2, 0.3], dtype=np.float32),
        )
        np.testing.assert_allclose(build_state_from_raw(raw)[:7], raw.root_pose)

    def test_reject_bad_joint_positions(self) -> None:
        raw = RawRobotState(
            joint_positions=np.zeros(19, dtype=np.float32),  # 誤
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="joint_positions"):
            build_state_from_raw(raw)

    def test_reject_bad_hand(self) -> None:
        raw = RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=np.zeros(5, dtype=np.float32),  # 誤
            ee_state=np.zeros(12, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="hand_state"):
            build_state_from_raw(raw)


class TestGr00tPolicyPickLegsLifecycle:
    def _cfg(self) -> PolicyConfig:
        return PolicyConfig(
            mode="none",
            ckpt_ref="Team-RAMEN/groot-n1.7-pick-legs-ver1",
            checkpoint_subdir=CKPT_SUBDIR,
            dtype="bf16",
            cams=CAMERAS,
        )

    def test_init_validates_cfg(self) -> None:
        # cams validation は __init__ で走る
        Gr00tPolicyPickLegs(self._cfg())  # no raise
        with pytest.raises(ValueError, match="4 cam layout"):
            Gr00tPolicyPickLegs(
                PolicyConfig(
                    mode="none", ckpt_ref="x", cams=(CameraKey.HEAD_LEFT,)
                )
            )

    def test_from_ckpt_delegates_to_isolated_worker(self, monkeypatch) -> None:
        import inference.desktop.lower_policy.policies.groot_pick_legs as module

        worker = object()
        monkeypatch.setattr(module, "_PickLegsWorkerClient", lambda cfg: worker)
        policy = Gr00tPolicyPickLegs.from_ckpt(self._cfg())
        assert policy._worker_client is worker

    def test_checkpoint_download_excludes_optimizer_state(
        self, monkeypatch, tmp_path
    ) -> None:
        import huggingface_hub

        snapshot = tmp_path / "snapshot"
        checkpoint = snapshot / CKPT_SUBDIR
        checkpoint.mkdir(parents=True)
        for name in (
            "config.json",
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
            "model.safetensors.index.json",
        ):
            (checkpoint / name).write_text("{}", encoding="utf-8")
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return str(snapshot)

        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", fake_snapshot_download
        )
        cfg = self._cfg()

        resolved = _PickLegsWorkerClient._resolve_checkpoint(tmp_path, cfg)

        assert resolved == checkpoint.resolve()
        assert calls == [{
            "repo_id": cfg.ckpt_ref,
            "revision": MODEL_REVISION,
            "allow_patterns": tuple(
                f"{CKPT_SUBDIR}/{name}" for name in RUNTIME_CHECKPOINT_FILES
            ),
        }]
        assert all("optimizer" not in pattern for pattern in calls[0]["allow_patterns"])

    def test_checkpoint_root_layout_download(self, monkeypatch, tmp_path) -> None:
        # ver2-lora: checkpoint_subdir=null → root 直下から load。
        # @revision は cfg.ckpt_ref から剥がして snapshot_download に渡す。
        import huggingface_hub

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir(parents=True)
        for name in (
            "config.json",
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
            "model.safetensors.index.json",
        ):
            (snapshot / name).write_text("{}", encoding="utf-8")
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return str(snapshot)

        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", fake_snapshot_download
        )
        cfg = PolicyConfig(
            mode="none",
            ckpt_ref="Team-RAMEN/groot-n1.7-pick-legs-ver2-lora@deadbeef",
            checkpoint_subdir=None,
            dtype="bf16",
            cams=CAMERAS,
        )

        resolved = _PickLegsWorkerClient._resolve_checkpoint(tmp_path, cfg)

        assert resolved == snapshot.resolve()
        assert calls == [{
            "repo_id": "Team-RAMEN/groot-n1.7-pick-legs-ver2-lora",
            "revision": "deadbeef",
            "allow_patterns": tuple(RUNTIME_CHECKPOINT_FILES),
        }]

    def test_parse_ref_strips_revision(self) -> None:
        assert _PickLegsWorkerClient._parse_ref(
            PolicyConfig(
                mode="none",
                ckpt_ref="Team-RAMEN/repo@abc123",
                dtype="bf16",
                cams=CAMERAS,
            )
        ) == ("Team-RAMEN/repo", "abc123")
        # @revision 省略時は ver1 pinned MODEL_REVISION が default。
        assert _PickLegsWorkerClient._parse_ref(
            PolicyConfig(
                mode="none",
                ckpt_ref="Team-RAMEN/groot-n1.7-pick-legs-ver1",
                dtype="bf16",
                cams=CAMERAS,
            )
        ) == ("Team-RAMEN/groot-n1.7-pick-legs-ver1", MODEL_REVISION)

    def test_predict_without_load_raises(self) -> None:
        p = Gr00tPolicyPickLegs(self._cfg())
        obs = Observation(
            frames_bgr={cam: np.zeros((480, 640, 3), dtype=np.uint8) for cam in CAMERAS},
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=None,
            language=DEFAULT_LANGUAGE_PROMPT,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            p.predict(obs)

    def test_class_constants(self) -> None:
        assert Gr00tPolicyPickLegs.STATE_DIM == STATE_DIM
        assert callable(Gr00tPolicyPickLegs.build_state_from_raw)
        assert Gr00tPolicyPickLegs.CAMERAS == CAMERAS
        assert Gr00tPolicyPickLegs.CKPT_SUBDIR == CKPT_SUBDIR
        assert Gr00tPolicyPickLegs.EMBODIMENT_ID == EMBODIMENT_ID
