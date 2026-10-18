"""Phase 2 smoke: groot.py の preprocessing 契約 + config validation (Issue #125)。

- 実 lerobot / isaac-gr00t 依存 test は `@pytest.mark.integration` で skip、
  Sakura env で verify する。default env (torch 無し) では preprocessing helper
  の shape/key 検証と config validation のみ回る。
- build_batch_dict の key naming は subtask_training.json 準拠を保証する。
"""

from __future__ import annotations

import json
import threading
import time
import socket
import threading

import numpy as np
import pytest

from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    Observation,
    PolicyAction,
    PolicyConfig,
    RawRobotState,
)
from inference.desktop.lower_policy.rtc import (
    RtcConfig,
    validate_rtc_against_chunk_len,
)
from inference.desktop.lower_policy.policies.groot import (
    ACTION_DIM,
    CAMERAS,
    CHUNK_LEN,
    DEFAULT_LANGUAGE_PROMPT,
    EMBODIMENT_ID,
    Gr00tPolicy,
    GrootArtifactContract,
    STATE_DIM,
    _euler_xyz_to_rot6d,
    _image_bgr_hwc_to_rgb_chw,
    _prepare_gr00t_frames,
    _read_groot_artifact_contract,
    _gpu_free_bytes_for_load,
    _require_groot_load_headroom,
    _wait_for_groot_host_load_headroom,
    _resolve_groot_checkpoint_root,
    _validate_groot_checkpoint_keys,
    _wrist_pose_to_xyz_rot6d,
    build_batch_dict,
    build_state_from_raw,
    slice_53d_to_19d,
    validate_gr00t_config,
)
from inference.desktop.lower_policy.policies.ramen_ori import (
    match_training_jpeg,
    overlay_obb_on_frame,
)
from inference.desktop.perception.yolo_obb import OBBDetection
from inference.desktop.lower_policy.policies.groot_worker_protocol import (
    receive_archive,
    send_archive,
)


# ---- Constants match subtask_training.json ---- #


class TestConstants:
    def test_state_dim_matches_real_g1_relative_eef(self) -> None:
        # g1_full_body_mapping.REAL_G1_RELATIVE_EEF_STATE_DIM
        assert STATE_DIM == 49

    def test_action_dim_matches_real_g1_relative_eef(self) -> None:
        # g1_full_body_mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM
        assert ACTION_DIM == 53

    def test_chunk_len_matches_config(self) -> None:
        # subtask_training.json:groot.chunk_size / n_action_steps
        assert CHUNK_LEN == 16

    def test_cameras_is_3cam_layout(self) -> None:
        # subtask_training.json:cameras (head_right = cam_1 は ignored_video_keys で drop)
        assert CAMERAS == (
            CameraKey.HEAD_LEFT,
            CameraKey.WRIST_LEFT,
            CameraKey.WRIST_RIGHT,
        )

    def test_default_prompt_matches_config(self) -> None:
        # subtask_training.json:subtasks.combined_task5_7.task
        assert DEFAULT_LANGUAGE_PROMPT == "rotate and move table base (combined 5+7)"

    def test_embodiment_id_matches_mapping(self) -> None:
        # g1_full_body_mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_ID
        assert EMBODIMENT_ID == 25



class TestConstantDriftFromTrainingMapping:
    """Issue #137: 推論の 49D/53D 定数が学習 mapping と乖離したら落とす。

    従来 `TestConstants` は値を hard-code し、根拠は comment 参照だけだった。
    `g1_full_body_mapping` 側を変更しても推論側が追従せず silent に drift する
    ため、実際に import して突き合わせる (URDF asset drift guard と同じ位置づけ)。
    """

    @staticmethod
    def _mapping():
        from model.subtask_policy_training.gr00t import g1_full_body_mapping as m

        return m

    def test_state_dim_matches_mapping(self) -> None:
        assert STATE_DIM == self._mapping().REAL_G1_RELATIVE_EEF_STATE_DIM

    def test_action_dim_matches_mapping(self) -> None:
        assert ACTION_DIM == self._mapping().REAL_G1_RELATIVE_EEF_ACTION_DIM

    def test_embodiment_id_matches_mapping(self) -> None:
        assert EMBODIMENT_ID == self._mapping().REAL_G1_RELATIVE_EEF_EMBODIMENT_ID

    def test_action_slices_match_mapping(self) -> None:
        """53D action の部位 slice が学習 mapping と 1 対 1 で一致する。"""
        from inference.desktop.lower_policy.policies.groot import (
            _ACTION_53D_LEFT_ARM_SLICE,
            _ACTION_53D_LEFT_HAND_SLICE,
            _ACTION_53D_RIGHT_ARM_SLICE,
            _ACTION_53D_RIGHT_HAND_SLICE,
            _ACTION_53D_WAIST_SLICE,
        )

        expected = self._mapping().REAL_G1_RELATIVE_EEF_ACTION_SLICES
        actual = {
            "left_hand": _ACTION_53D_LEFT_HAND_SLICE,
            "right_hand": _ACTION_53D_RIGHT_HAND_SLICE,
            "left_arm": _ACTION_53D_LEFT_ARM_SLICE,
            "right_arm": _ACTION_53D_RIGHT_ARM_SLICE,
            "waist": _ACTION_53D_WAIST_SLICE,
        }
        for name, sl in actual.items():
            assert (sl.start, sl.stop) == tuple(expected[name]), name

    def test_state_slices_cover_dim_without_gap(self) -> None:
        """build_state_from_raw が書く範囲が state 全域を隙間なく覆う。"""
        slices = self._mapping().REAL_G1_RELATIVE_EEF_STATE_SLICES
        bounds = sorted(tuple(v) for v in slices.values())
        assert bounds[0][0] == 0
        assert bounds[-1][1] == STATE_DIM
        for (_, prev_stop), (next_start, _) in zip(bounds, bounds[1:]):
            assert prev_stop == next_start

# ---- memory-safe load guard ---- #


class TestGrootLoadHeadroom:
    _GIB = 1024**3

    def test_accepts_sufficient_host_and_gpu_memory(self) -> None:
        _require_groot_load_headroom(
            checkpoint_bytes=12 * self._GIB,
            host_available_bytes=24 * self._GIB,
            gpu_free_bytes=30 * self._GIB,
        )

    def test_rejects_insufficient_host_memory_before_load(self) -> None:
        with pytest.raises(MemoryError, match="host memory"):
            _require_groot_load_headroom(
                checkpoint_bytes=12 * self._GIB,
                host_available_bytes=17 * self._GIB,
                gpu_free_bytes=30 * self._GIB,
            )

    def test_rejects_insufficient_gpu_memory_before_load(self) -> None:
        with pytest.raises(MemoryError, match="GPU memory"):
            _require_groot_load_headroom(
                checkpoint_bytes=12 * self._GIB,
                host_available_bytes=24 * self._GIB,
                gpu_free_bytes=19 * self._GIB,
            )

    def test_unknown_metrics_do_not_block_non_linux_unit_tests(self) -> None:
        _require_groot_load_headroom(
            checkpoint_bytes=12 * self._GIB,
            host_available_bytes=None,
            gpu_free_bytes=None,
        )

    def test_wait_accepts_memory_released_after_hf_reconstruction(
        self, monkeypatch
    ) -> None:
        samples = iter((20 * self._GIB, 22 * self._GIB))
        monkeypatch.setattr(
            "inference.desktop.lower_policy.policies.groot._host_mem_available_bytes",
            lambda: next(samples),
        )
        monkeypatch.setattr(
            "inference.desktop.lower_policy.policies.groot.time.sleep", lambda _: None
        )
        available = _wait_for_groot_host_load_headroom(
            checkpoint_bytes=13 * self._GIB,
            timeout_s=1.0,
        )
        assert available == 22 * self._GIB

    def test_wait_does_not_relax_required_headroom(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "inference.desktop.lower_policy.policies.groot._host_mem_available_bytes",
            lambda: 20 * self._GIB,
        )
        monotonic = iter((0.0, 2.0))
        monkeypatch.setattr(
            "inference.desktop.lower_policy.policies.groot.time.monotonic",
            lambda: next(monotonic),
        )
        with pytest.raises(MemoryError, match="host memory"):
            _wait_for_groot_host_load_headroom(
                checkpoint_bytes=13 * self._GIB,
                timeout_s=1.0,
            )


class TestGpuFreeBytesForLoad:
    """統合 GPU (Thor) では MemAvailable で空きを測る (B3b-4)。"""

    _GIB = 1024**3

    @staticmethod
    def _torch(*, integrated: bool, free: int, total: int):
        from types import SimpleNamespace

        cuda = SimpleNamespace(
            mem_get_info=lambda: (free, total),
            current_device=lambda: 0,
            get_device_properties=lambda _device: SimpleNamespace(is_integrated=integrated),
        )
        return SimpleNamespace(cuda=cuda)

    def test_integrated_gpu_uses_mem_available_and_loads(self, monkeypatch, capsys) -> None:
        # page cache が溜まり cudaMemGetInfo は 10 GiB しか返さないが、実際は 60 GiB 使える
        monkeypatch.setattr(
            "inference.desktop.lower_policy.policies.groot._host_mem_available_bytes",
            lambda: 60 * self._GIB,
        )
        torch = self._torch(integrated=True, free=10 * self._GIB, total=122 * self._GIB)
        free = _gpu_free_bytes_for_load(torch)
        assert free == 60 * self._GIB
        _require_groot_load_headroom(
            checkpoint_bytes=12 * self._GIB, host_available_bytes=None, gpu_free_bytes=free
        )
        err = capsys.readouterr().err
        assert "MemAvailable=60.0GiB" in err and "cudaMemGetInfo free=10.0/122.0GiB" in err

    def test_discrete_gpu_keeps_cuda_free_and_still_rejects(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "inference.desktop.lower_policy.policies.groot._host_mem_available_bytes",
            lambda: 60 * self._GIB,
        )
        torch = self._torch(integrated=False, free=10 * self._GIB, total=32 * self._GIB)
        free = _gpu_free_bytes_for_load(torch)
        assert free == 10 * self._GIB
        with pytest.raises(MemoryError, match="GPU memory"):
            _require_groot_load_headroom(
                checkpoint_bytes=12 * self._GIB, host_available_bytes=None, gpu_free_bytes=free
            )

    def test_integrated_gpu_falls_back_to_cuda_free_without_meminfo(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "inference.desktop.lower_policy.policies.groot._host_mem_available_bytes",
            lambda: None,
        )
        torch = self._torch(integrated=True, free=30 * self._GIB, total=122 * self._GIB)
        assert _gpu_free_bytes_for_load(torch) == 30 * self._GIB


class TestGrootCheckpointKeyContract:
    _CANONICAL = "_groot_model.backbone.model.lm_head.weight"
    _ALIAS = (
        "_groot_model.backbone.model.model.language_model.embed_tokens.weight"
    )

    def test_accepts_only_byte_equal_tied_embedding_alias(self, tmp_path) -> None:
        import torch
        from safetensors.torch import save_file

        value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        checkpoint = tmp_path / "model.safetensors"
        save_file(
            {self._CANONICAL: value, self._ALIAS: value.clone()}, checkpoint
        )
        _validate_groot_checkpoint_keys(checkpoint, {self._CANONICAL})

    def test_rejects_tied_alias_with_different_values(self, tmp_path) -> None:
        import torch
        from safetensors.torch import save_file

        value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        checkpoint = tmp_path / "model.safetensors"
        save_file(
            {self._CANONICAL: value, self._ALIAS: value + 1}, checkpoint
        )
        with pytest.raises(RuntimeError, match="tied checkpoint alias differs"):
            _validate_groot_checkpoint_keys(checkpoint, {self._CANONICAL})

    def test_rejects_any_other_missing_or_unexpected_key(self, tmp_path) -> None:
        import torch
        from safetensors.torch import save_file

        checkpoint = tmp_path / "model.safetensors"
        save_file({"unrelated.weight": torch.ones(2)}, checkpoint)
        with pytest.raises(RuntimeError, match="checkpoint schema mismatch"):
            _validate_groot_checkpoint_keys(checkpoint, {self._CANONICAL})

    def test_allows_rotate_specialist_fk_singleton_buffers(self, tmp_path) -> None:
        """Issue #134: rotate specialist Run 2/3 の train patch が attach する
        `_tr_fk_singleton.*` buffer 6 個は Isaac-GR00T module 定義に無いが
        inference forward で使われない、allow-list prefix でスキップされる。"""
        import torch
        from safetensors.torch import save_file

        fk_buffers = {
            "_groot_model.action_head._tr_fk_singleton._left_fixed_Ts": torch.zeros(10, 4, 4),
            "_groot_model.action_head._tr_fk_singleton._right_fixed_Ts": torch.zeros(10, 4, 4),
            "_groot_model.action_head._tr_fk_singleton._left_axes": torch.zeros(10, 3),
            "_groot_model.action_head._tr_fk_singleton._right_axes": torch.zeros(10, 3),
            "_groot_model.action_head._tr_fk_singleton._left_groot_idx": torch.zeros(10, dtype=torch.long),
            "_groot_model.action_head._tr_fk_singleton._right_groot_idx": torch.zeros(10, dtype=torch.long),
        }
        checkpoint = tmp_path / "model.safetensors"
        save_file({self._CANONICAL: torch.ones(1), **fk_buffers}, checkpoint)
        # 6 個の fk singleton buffer が unexpected でも allow-list で raise 無し
        _validate_groot_checkpoint_keys(checkpoint, {self._CANONICAL})

    def test_rotate_specialist_allowlist_is_exactly_scoped(self) -> None:
        from inference.desktop.lower_policy.policies.groot import (
            _is_allowed_groot_unexpected_key,
        )

        assert _is_allowed_groot_unexpected_key(
            "_groot_model.action_head._tr_fk_singleton._left_axes"
        )
        assert not _is_allowed_groot_unexpected_key(
            "_groot_model.action_head._tr_other._left_axes"
        )
        assert not _is_allowed_groot_unexpected_key(
            "action_head._tr_fk_singleton._left_axes"
        )

    def test_rejects_non_fk_singleton_unexpected_prefix(self, tmp_path) -> None:
        """`_tr_fk_singleton.*` 以外の `_tr_*` prefix (別 train patch 由来 or
        unknown drift) は依然 reject (fail-closed 保護維持)。"""
        import torch
        from safetensors.torch import save_file

        checkpoint = tmp_path / "model.safetensors"
        save_file(
            {
                self._CANONICAL: torch.ones(1),
                "_groot_model.some_other._tr_random_buffer": torch.zeros(3),
            },
            checkpoint,
        )
        with pytest.raises(RuntimeError, match="checkpoint schema mismatch"):
            _validate_groot_checkpoint_keys(checkpoint, {self._CANONICAL})


class TestGrootCheckpointResolution:
    def test_local_directory_is_resolved_without_hf(self, tmp_path) -> None:
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        assert _resolve_groot_checkpoint_root(str(checkpoint)) == checkpoint.resolve()

    def test_repo_revision_is_passed_to_snapshot_download(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import huggingface_hub

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        calls: list[tuple[str, str | None]] = []

        def fake_snapshot_download(*, repo_id: str, revision: str | None) -> str:
            calls.append((repo_id, revision))
            return str(snapshot)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
        resolved = _resolve_groot_checkpoint_root("Team-RAMEN/model@deadbeef")
        assert resolved == snapshot.resolve()
        assert calls == [("Team-RAMEN/model", "deadbeef")]

    def test_nested_checkpoint_download_is_scoped_to_selected_subdir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import huggingface_hub

        snapshot = tmp_path / "snapshot"
        subdir = "checkpoints/020000/pretrained_model"
        checkpoint = snapshot / subdir
        checkpoint.mkdir(parents=True)
        calls: list[dict[str, object]] = []

        def fake_snapshot_download(**kwargs: object) -> str:
            calls.append(kwargs)
            return str(snapshot)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
        resolved = _resolve_groot_checkpoint_root(
            "Team-RAMEN/model@deadbeef", subdir
        )
        assert resolved == checkpoint.resolve()
        assert calls == [{
            "repo_id": "Team-RAMEN/model",
            "revision": "deadbeef",
            "allow_patterns": ["checkpoints/020000/pretrained_model/**"],
        }]

    def test_offline_reads_only_the_cached_snapshot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """会場 (HF_HUB_OFFLINE=1) では cache だけを見る。

        huggingface_hub 1.28 は commit hash 指定でも file 一覧の記録が無いとネットへ
        取りに行き、GR00T 53D の worker が起動中に落ちた (GB10、2026-09-25)。
        """
        import huggingface_hub

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        calls: list[dict[str, object]] = []

        def fake_snapshot_download(**kwargs: object) -> str:
            calls.append(kwargs)
            return str(snapshot)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
        assert _resolve_groot_checkpoint_root("Team-RAMEN/model@deadbeef") == (
            snapshot.resolve()
        )
        assert calls == [{
            "repo_id": "Team-RAMEN/model",
            "revision": "deadbeef",
            "local_files_only": True,
        }]

    def test_local_nested_checkpoint_must_exist(self, tmp_path) -> None:
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        with pytest.raises(FileNotFoundError, match="checkpoint directory"):
            _resolve_groot_checkpoint_root(str(snapshot), "missing/model")

    @pytest.mark.parametrize("checkpoint_ref", ["@sha", "Team-RAMEN/model@"])
    def test_rejects_malformed_pinned_reference(self, checkpoint_ref: str) -> None:
        with pytest.raises(ValueError, match="HF-repo@revision"):
            _resolve_groot_checkpoint_root(checkpoint_ref)


class TestGrootArtifactContract:
    def test_accepts_furniture_two_frame_contract(self, tmp_path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({
            "type": "furniture_groot",
            "input_features": {"observation.state": {"shape": [49]}},
            "output_features": {"action": {"shape": [53]}},
            "embodiment_tag": "real_g1_relative_eef_relative_joints",
            "use_relative_actions": True,
            "chunk_size": 40,
            "n_action_steps": 4,
            "max_state_dim": 132,
            "max_action_dim": 132,
            "valid_action_dim": 46,
            "expert_role": "insertion",
        }))
        (tmp_path / "policy_preprocessor.json").write_text(json.dumps({
            "steps": [{
                "registry_name": "groot_n1_7_pack_inputs_v1",
                "config": {"video_horizon": 2},
            }]
        }))
        contract = _read_groot_artifact_contract(tmp_path)
        assert contract.video_horizon == 2
        assert contract.history_offset_ticks == 20
        assert contract.n_action_steps == 4
        assert contract.expert_role == "insertion"

    def test_rejects_furniture_single_frame_contract(self, tmp_path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({
            "type": "furniture_groot",
            "input_features": {"observation.state": {"shape": [49]}},
            "output_features": {"action": {"shape": [53]}},
            "embodiment_tag": "real_g1_relative_eef_relative_joints",
            "use_relative_actions": True,
            "chunk_size": 40,
            "max_state_dim": 132,
            "max_action_dim": 132,
            "valid_action_dim": 46,
        }))
        (tmp_path / "policy_preprocessor.json").write_text(json.dumps({
            "steps": [{
                "registry_name": "groot_n1_7_pack_inputs_v1",
                "config": {"video_horizon": 1},
            }]
        }))
        with pytest.raises(ValueError, match="two-frame"):
            _read_groot_artifact_contract(tmp_path)

class TestGrootWorkerProtocol:
    def test_numpy_archive_round_trip(self) -> None:
        left, right = socket.socketpair()
        try:
            send_archive(
                left,
                kind=np.asarray("predict"),
                state=np.arange(STATE_DIM, dtype=np.float32),
            )
            received = receive_archive(right)
            assert received is not None
            assert str(received["kind"].item()) == "predict"
            np.testing.assert_array_equal(
                received["state"], np.arange(STATE_DIM, dtype=np.float32)
            )
        finally:
            left.close()
            right.close()

    def test_runtime_without_lerobot_constructs_worker_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import inference.desktop.lower_policy.policies.groot as groot_module

        class FakeWorker:
            closed = False

            def __init__(self, cfg: PolicyConfig) -> None:
                self.cfg = cfg

            def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(groot_module.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(groot_module, "_GrootWorkerClient", FakeWorker)
        cfg = PolicyConfig(
            mode="none",
            ckpt_ref="Team-RAMEN/test",
            device="cuda",
            dtype="bf16",
            cams=CAMERAS,
        )
        policy = Gr00tPolicy.from_ckpt(cfg)
        worker = policy._worker_client
        assert isinstance(worker, FakeWorker)
        policy.close()
        assert worker.closed is True


# ---- validate_gr00t_config ---- #


class TestValidateGr00tConfig:
    def _cfg(self, **overrides) -> PolicyConfig:
        defaults = dict(
            mode="none",
            ckpt_ref="Team-RAMEN/foo@abc",
            device="cuda",
            dtype="bf16",
            cams=CAMERAS,
        )
        defaults.update(overrides)
        return PolicyConfig(**defaults)

    def test_valid_default(self) -> None:
        validate_gr00t_config(self._cfg())  # no raise

    def test_valid_overlay_mode(self) -> None:
        validate_gr00t_config(self._cfg(mode="overlay"))

    def test_reject_precomputed_token_mode(self) -> None:
        # GR00T は explicit OBB token channel を持たない
        with pytest.raises(ValueError, match="does not support mode='precomputed_token'"):
            validate_gr00t_config(self._cfg(mode="precomputed_token"))

    def test_reject_4cam_layout(self) -> None:
        # RAMEN-Ori 用 4 cam を GR00T に渡すと reject
        with pytest.raises(ValueError, match="3 cam layout"):
            validate_gr00t_config(
                self._cfg(
                    cams=(
                        CameraKey.HEAD_LEFT,
                        CameraKey.HEAD_RIGHT,
                        CameraKey.WRIST_LEFT,
                        CameraKey.WRIST_RIGHT,
                    )
                )
            )


class TestOverlayFrames:
    @staticmethod
    def _obs(detections) -> Observation:  # noqa: ANN001
        frames = {
            cam: np.zeros((48, 64, 3), dtype=np.uint8) for cam in CAMERAS
        }
        return Observation(
            frames_bgr=frames,
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=4,
            language=DEFAULT_LANGUAGE_PROMPT,
            obb_detections=detections,
            timestamp_ns=time.monotonic_ns(),
        )

    def test_overlay_draws_head_left_only_without_mutating_input(self) -> None:
        pytest.importorskip("cv2")
        detection = OBBDetection(
            class_id=0,
            class_name="table",
            confidence=0.9,
            verts=np.asarray(
                [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                dtype=np.float32,
            ),
        )
        obs = self._obs({CameraKey.HEAD_LEFT: [detection]})

        frames, count = _prepare_gr00t_frames(obs, "overlay", "4:2:0")

        assert count == 1
        drawn = overlay_obb_on_frame(np.zeros((48, 64, 3), dtype=np.uint8), [detection])
        np.testing.assert_array_equal(
            frames[CameraKey.HEAD_LEFT], match_training_jpeg(drawn, "4:2:0")
        )
        assert np.count_nonzero(frames[CameraKey.HEAD_LEFT]) > 0
        assert np.count_nonzero(obs.frames_bgr[CameraKey.HEAD_LEFT]) == 0
        assert frames[CameraKey.WRIST_LEFT] is obs.frames_bgr[CameraKey.WRIST_LEFT]
        assert frames[CameraKey.WRIST_RIGHT] is obs.frames_bgr[CameraKey.WRIST_RIGHT]

    def test_overlay_empty_detections_still_match_training_jpeg(self) -> None:
        # 学習 cache は box が 0 件の frame も同じ jpg 保存を通っている
        pytest.importorskip("cv2")
        obs = self._obs({CameraKey.HEAD_LEFT: []})
        head = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        obs.frames_bgr[CameraKey.HEAD_LEFT] = head

        frames, count = _prepare_gr00t_frames(obs, "overlay", "4:2:0")

        assert count == 0
        np.testing.assert_array_equal(
            frames[CameraKey.HEAD_LEFT], match_training_jpeg(head, "4:2:0")
        )
        assert not np.array_equal(frames[CameraKey.HEAD_LEFT], head)

    def test_overlay_rejects_missing_detector(self) -> None:
        obs = self._obs(None)

        with pytest.raises(RuntimeError, match="requires live YOLO-OBB"):
            _prepare_gr00t_frames(obs, "overlay", "4:2:0")

    def test_overlay_requires_jpeg_subsampling(self) -> None:
        # 学習 cache の色差 (焼き込みの時期で違う) を slot から受け取らずに描かない
        obs = self._obs({CameraKey.HEAD_LEFT: []})

        with pytest.raises(ValueError, match="overlay_jpeg_subsampling"):
            _prepare_gr00t_frames(obs, "overlay")

    def test_non_overlay_keeps_all_frames_raw(self) -> None:
        obs = self._obs(None)

        frames, count = _prepare_gr00t_frames(obs, "none")

        assert count == 0
        assert all(frames[cam] is obs.frames_bgr[cam] for cam in CAMERAS)


# ---- build_batch_dict ---- #


class TestBuildBatchDict:
    def _obs(
        self,
        language: str | None = DEFAULT_LANGUAGE_PROMPT,
        state_dim: int = STATE_DIM,
        skip_cam: CameraKey | None = None,
        with_history: bool = False,
    ) -> Observation:
        H, W = 480, 640
        frames = {cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS}
        if skip_cam is not None:
            del frames[skip_cam]
        return Observation(
            frames_bgr=frames,
            frames_bgr_prev=(
                {cam: np.ones((H, W, 3), dtype=np.uint8) for cam in CAMERAS}
                if with_history
                else None
            ),
            state=np.zeros(state_dim, dtype=np.float32),
            skill_id=None,
            language=language,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )

    def test_output_keys(self) -> None:
        batch = build_batch_dict(self._obs())
        assert "observation.state" in batch
        assert "task" in batch
        assert "embodiment_id" in batch
        # subtask_training.json:cameras に厳密に一致する key で入る
        assert "observation.images.head_left" in batch
        assert "observation.images.left_wrist" in batch
        assert "observation.images.right_wrist" in batch

    def test_state_shape(self) -> None:
        batch = build_batch_dict(self._obs())
        assert batch["observation.state"].shape == (STATE_DIM,)
        assert batch["observation.state"].dtype == np.float32

    def test_image_uint8_bgr_shape(self) -> None:
        batch = build_batch_dict(self._obs())
        img = batch["observation.images.head_left"]
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8

    def test_two_frame_video_preserves_history_then_current(self) -> None:
        obs = self._obs(with_history=True)
        batch = build_batch_dict(obs, video_horizon=2)
        image = batch["observation.images.head_left"]
        assert image.shape == (2, 480, 640, 3)
        assert np.all(image[0] == 1)
        assert np.all(image[1] == 0)

    def test_two_frame_video_rejects_missing_history(self) -> None:
        with pytest.raises(ValueError, match="missing history"):
            build_batch_dict(self._obs(), video_horizon=2)

    def test_task_string(self) -> None:
        batch = build_batch_dict(self._obs())
        assert batch["task"] == DEFAULT_LANGUAGE_PROMPT

    def test_embodiment_id_int(self) -> None:
        batch = build_batch_dict(self._obs())
        assert batch["embodiment_id"] == EMBODIMENT_ID

    def test_reject_missing_language(self) -> None:
        with pytest.raises(ValueError, match="requires obs.language"):
            build_batch_dict(self._obs(language=None))

    def test_reject_wrong_state_dim(self) -> None:
        # 71D (RAMEN-Ori) を GR00T に渡すと reject
        with pytest.raises(ValueError, match=r"\(49,\)"):
            build_batch_dict(self._obs(state_dim=71))

    def test_reject_missing_camera(self) -> None:
        with pytest.raises(KeyError, match="head_left"):
            build_batch_dict(self._obs(skip_cam=CameraKey.HEAD_LEFT))

    def test_reject_wrong_frame_dtype(self) -> None:
        obs = self._obs()
        # BGR frame を float に置き換え
        obs.frames_bgr[CameraKey.HEAD_LEFT] = np.zeros(
            (480, 640, 3), dtype=np.float32
        )
        with pytest.raises(ValueError, match="uint8 BGR"):
            build_batch_dict(obs)

    def test_live_bgr_is_converted_to_unbatched_rgb_chw(self) -> None:
        frame = np.zeros((2, 4, 3), dtype=np.uint8)
        frame[0, 0] = [11, 22, 33]  # B, G, R
        converted = _image_bgr_hwc_to_rgb_chw(frame)
        assert converted.shape == (3, 2, 4)
        assert converted.dtype == np.uint8
        assert converted.flags.c_contiguous
        np.testing.assert_array_equal(converted[:, 0, 0], [33, 22, 11])

    def test_temporal_bgr_is_converted_to_rgb_tchw(self) -> None:
        frame = np.zeros((2, 2, 4, 3), dtype=np.uint8)
        frame[1, 0, 0] = [11, 22, 33]
        converted = _image_bgr_hwc_to_rgb_chw(frame)
        assert converted.shape == (2, 3, 2, 4)
        np.testing.assert_array_equal(converted[1, :, 0, 0], [33, 22, 11])


# ---- build_state_from_raw ---- #


class TestBuildStateFromRaw:
    def _raw(self) -> RawRobotState:
        return RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
            last_action_19d=None,
            joint_positions_prev=None,
        )

    def test_output_shape(self) -> None:
        state = build_state_from_raw(self._raw())
        assert state.shape == (STATE_DIM,)
        assert state.dtype == np.float32

    def test_reject_bad_joint_positions_shape(self) -> None:
        raw = self._raw()
        raw = RawRobotState(
            joint_positions=np.zeros(19, dtype=np.float32),  # 誤 shape
            hand_state=raw.hand_state,
            ee_state=raw.ee_state,
        )
        with pytest.raises(ValueError, match="joint_positions must be"):
            build_state_from_raw(raw)

    def test_arm_and_waist_slice_populated_from_joint_positions(self) -> None:
        jp = np.arange(29, dtype=np.float32)
        raw = RawRobotState(
            joint_positions=jp,
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
        )
        state = build_state_from_raw(raw)
        # [32:39] left_arm = joint[15:22]
        np.testing.assert_array_equal(state[32:39], jp[15:22])
        # [39:46] right_arm = joint[22:29]
        np.testing.assert_array_equal(state[39:46], jp[22:29])
        # [46:49] waist = joint[12:15]
        np.testing.assert_array_equal(state[46:49], jp[12:15])

    def test_hand_slice_matches_training_synergy(self) -> None:
        from model.subtask_policy_training.gr00t.dex1_hand_synergy import dex1_to_hand

        raw = RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=np.array([0.42, 0.73], dtype=np.float32),
            ee_state=np.zeros(12, dtype=np.float32),
        )
        state = build_state_from_raw(raw)
        np.testing.assert_allclose(
            state[18:25], dex1_to_hand(0.42, side="left", kind="state")
        )
        np.testing.assert_allclose(
            state[25:32], dex1_to_hand(0.73, side="right", kind="state")
        )

    def test_eef_9d_slice_from_ee_state(self) -> None:
        # ee_state = [left(xyz+euler), right(xyz+euler)] = 12D
        ee = np.array(
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, -1.0, -2.0, -3.0, -0.4, -0.5, -0.6],
            dtype=np.float32,
        )
        raw = RawRobotState(
            joint_positions=np.zeros(29, dtype=np.float32),
            hand_state=np.zeros(2, dtype=np.float32),
            ee_state=ee,
        )
        state = build_state_from_raw(raw)
        # XYZ 部分は直接 copy
        np.testing.assert_array_equal(state[0:3], ee[0:3])
        np.testing.assert_array_equal(state[9:12], ee[6:9])
        # ROT6D は helper と一致
        expected_left_rot6d = _euler_xyz_to_rot6d(0.1, 0.2, 0.3)
        expected_right_rot6d = _euler_xyz_to_rot6d(-0.4, -0.5, -0.6)
        np.testing.assert_allclose(state[3:9], expected_left_rot6d, atol=1e-6)
        np.testing.assert_allclose(state[12:18], expected_right_rot6d, atol=1e-6)


class TestEulerToRot6d:
    def test_identity_rotation(self) -> None:
        # Zero euler = identity → row0 = [1,0,0], row1 = [0,1,0]
        rot6d = _euler_xyz_to_rot6d(0.0, 0.0, 0.0)
        np.testing.assert_allclose(rot6d, np.array([1, 0, 0, 0, 1, 0]), atol=1e-6)

    def test_yaw_90deg(self) -> None:
        # yaw=90° = Rz(π/2)、row0 = [0,-1,0]、row1 = [1, 0, 0]
        rot6d = _euler_xyz_to_rot6d(0.0, 0.0, np.pi / 2)
        np.testing.assert_allclose(
            rot6d, np.array([0, -1, 0, 1, 0, 0], dtype=np.float32), atol=1e-6
        )

    def test_shape_and_dtype(self) -> None:
        rot6d = _euler_xyz_to_rot6d(0.1, 0.2, 0.3)
        assert rot6d.shape == (6,)
        assert rot6d.dtype == np.float32


@pytest.mark.integration
class TestRot6dMatchesTraining:
    """training-side source_euler_xyz_pose_to_xyz_rot6d と 1e-6 一致 (integration)."""

    def test_matches_training_helper(self) -> None:
        from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
            source_euler_xyz_pose_to_xyz_rot6d,
        )

        pose = [1.5, -2.3, 0.7, 0.5, -0.3, 1.2]
        inference_9d = _wrist_pose_to_xyz_rot6d(np.array(pose, dtype=np.float32))
        training_9d = np.array(source_euler_xyz_pose_to_xyz_rot6d(pose), dtype=np.float32)
        np.testing.assert_allclose(inference_9d, training_9d, atol=1e-6)


# ---- Gr00tPolicy skeleton (integration test は sakura で走る) ---- #


class TestGr00tPolicyLifecycle:
    def _cfg(self) -> PolicyConfig:
        return PolicyConfig(
            mode="none",
            ckpt_ref="Team-RAMEN/foo@abc",
            dtype="bf16",
            cams=CAMERAS,
        )

    def test_init_without_load_raises_on_predict(self) -> None:
        # _lerobot_policy=None で init → predict() は明示的エラー
        p = Gr00tPolicy(self._cfg())
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

    def test_init_without_load_raises_on_warmup(self) -> None:
        p = Gr00tPolicy(self._cfg())
        with pytest.raises(RuntimeError, match="not loaded"):
            p.warmup(n_iter=1)

    def test_close_no_op_when_not_loaded(self) -> None:
        # Not loaded 状態でも close() は idempotent
        p = Gr00tPolicy(self._cfg())
        p.close()  # no raise

    def test_async_diagnostics_are_initialized_before_first_prediction(self) -> None:
        # warmup uses the synchronous path before an async pipeline exists,
        # but predict metadata still exposes this field.
        p = Gr00tPolicy(self._cfg())
        assert p._pending_submit_step is None

    def test_class_constants_expose_contract(self) -> None:
        assert Gr00tPolicy.STATE_DIM == STATE_DIM
        assert callable(Gr00tPolicy.build_state_from_raw)
        assert Gr00tPolicy.ACTION_DIM == ACTION_DIM
        assert Gr00tPolicy.CHUNK_LEN == CHUNK_LEN
        assert Gr00tPolicy.CAMERAS == CAMERAS
        assert Gr00tPolicy.DEFAULT_LANGUAGE_PROMPT == DEFAULT_LANGUAGE_PROMPT

    def test_dummy_observation_has_valid_identity_wrist_rotations(self) -> None:
        policy = Gr00tPolicy(self._cfg())
        state = policy._make_dummy_observation().state
        identity = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(state[3:9], identity)
        np.testing.assert_array_equal(state[12:18], identity)

    @pytest.mark.integration
    def test_from_ckpt_real(self) -> None:
        # Sakura env でのみ走る: lerobot + isaac-gr00t + GPU 必要
        # (default env は import 段階で失敗するので integration marker で skip 前提)
        pytest.skip("integration: requires lerobot + gr00t deps + sakura GPU env")


class TestPolicyProtocolCompatibility:
    def test_gr00t_matches_policy_protocol(self) -> None:
        # runtime_checkable(Policy) は method 名で判定。Gr00tPolicy は
        # from_ckpt / warmup / predict / close 全て持つ。
        from inference.desktop.lower_policy.policies.base import Policy

        p = Gr00tPolicy(PolicyConfig(mode="none", ckpt_ref="x", cams=CAMERAS))
        assert isinstance(p, Policy)


# ---- 53D REAL_G1_RELATIVE_EEF → 19D VlaSkill action adapter ---- #
#
# 実 GR00T 出力 (chunk_len, 53) を VlaSkill.step の受け入れ layout
# (chunk_len, 19) = waist(3) + arms(14) + hand(2) に落とす slice の検証。
# 実 model 依存の RELATIVE→ABSOLUTE 復元 verify は integration test 側 (Sakura GPU)。


class TestSlice53dTo19d:
    def _make_53d(self, chunk_len: int = 4) -> np.ndarray:
        """各 53D index に自身の値 (0.0, 1.0, ..., 52.0) を入れた test fixture。
        slice が正しく mapping されれば 19D 側の値と 53D index が対応する。"""
        chunk = np.zeros((chunk_len, ACTION_DIM), dtype=np.float32)
        for t in range(chunk_len):
            for i in range(ACTION_DIM):
                # 各 tick で index * 100 + t のパターンで unique value を入れる。
                # (tick 毎の slice も同時に verify、broadcast bug の early detection)
                chunk[t, i] = i * 100 + t
        return chunk

    def test_output_shape_is_chunk_len_by_19(self) -> None:
        out = slice_53d_to_19d(self._make_53d(chunk_len=16))
        assert out.shape == (16, 19)

    def test_output_dtype_preserved(self) -> None:
        chunk_f32 = self._make_53d(chunk_len=2).astype(np.float32)
        chunk_f64 = chunk_f32.astype(np.float64)
        assert slice_53d_to_19d(chunk_f32).dtype == np.float32
        assert slice_53d_to_19d(chunk_f64).dtype == np.float64

    def test_waist_slice_maps_from_46_49(self) -> None:
        # 19D[0:3] は 53D[46:49] (waist yaw/roll/pitch)。
        out = slice_53d_to_19d(self._make_53d(chunk_len=3))
        for t in range(3):
            for j, src_idx in enumerate((46, 47, 48)):
                assert out[t, j] == src_idx * 100 + t

    def test_left_arm_slice_maps_from_32_39(self) -> None:
        # 19D[3:10] は 53D[32:39] (左 arm 7 joint、rep=RELATIVE)。
        out = slice_53d_to_19d(self._make_53d(chunk_len=2))
        for t in range(2):
            for j, src_idx in enumerate(range(32, 39)):
                assert out[t, 3 + j] == src_idx * 100 + t

    def test_right_arm_slice_maps_from_39_46(self) -> None:
        # 19D[10:17] は 53D[39:46] (右 arm 7 joint、rep=RELATIVE)。
        out = slice_53d_to_19d(self._make_53d(chunk_len=2))
        for t in range(2):
            for j, src_idx in enumerate(range(39, 46)):
                assert out[t, 10 + j] == src_idx * 100 + t

    def test_hand_slice_synergy_round_trip_open(self) -> None:
        # Dex1-1 は per side 1 DOF、GR00T layout は 7D synergy (index_0/1,
        # middle_0/1, thumb_0/1/2)。dex1_to_hand(DEX1_OPEN) で 7D synergy を
        # 埋めて adapter を通し、hand_to_dex1 が DEX1_OPEN (=4.5) を復元できる
        # ことを verify (round-trip closure)。
        from model.subtask_policy_training.gr00t.dex1_hand_synergy import (
            dex1_to_hand, DEX1_OPEN,
        )
        chunk = np.zeros((1, ACTION_DIM), dtype=np.float32)
        chunk[0, 18:25] = dex1_to_hand(DEX1_OPEN, side="left", kind="action")
        chunk[0, 25:32] = dex1_to_hand(DEX1_OPEN, side="right", kind="action")
        out = slice_53d_to_19d(chunk)
        assert abs(out[0, 17] - DEX1_OPEN) < 1e-5, f"left grip {out[0,17]} != {DEX1_OPEN}"
        assert abs(out[0, 18] - DEX1_OPEN) < 1e-5, f"right grip {out[0,18]} != {DEX1_OPEN}"

    def test_hand_slice_synergy_round_trip_closed(self) -> None:
        from model.subtask_policy_training.gr00t.dex1_hand_synergy import (
            dex1_to_hand, DEX1_CLOSED,
        )
        chunk = np.zeros((1, ACTION_DIM), dtype=np.float32)
        chunk[0, 18:25] = dex1_to_hand(DEX1_CLOSED, side="left", kind="action")
        chunk[0, 25:32] = dex1_to_hand(DEX1_CLOSED, side="right", kind="action")
        out = slice_53d_to_19d(chunk)
        assert abs(out[0, 17] - DEX1_CLOSED) < 1e-5
        assert abs(out[0, 18] - DEX1_CLOSED) < 1e-5

    def test_hand_slice_synergy_round_trip_intermediate(self) -> None:
        # 中間 opening でも round-trip が成立すること (Dex1 open=4.5 / closed=0.0
        # の間の連続値を確認、synergy が linear interpolation で成立するので
        # 中点でも closure がきく)。
        from model.subtask_policy_training.gr00t.dex1_hand_synergy import (
            dex1_to_hand,
        )
        for dex1_value in (1.0, 2.25, 3.5):
            chunk = np.zeros((1, ACTION_DIM), dtype=np.float32)
            chunk[0, 18:25] = dex1_to_hand(dex1_value, side="left", kind="action")
            chunk[0, 25:32] = dex1_to_hand(dex1_value, side="right", kind="action")
            out = slice_53d_to_19d(chunk)
            assert abs(out[0, 17] - dex1_value) < 1e-5, (
                f"left grip {out[0,17]} != {dex1_value} @ dex1={dex1_value}"
            )
            assert abs(out[0, 18] - dex1_value) < 1e-5

    def test_hand_slice_matches_direct_hand_to_dex1_call(self) -> None:
        # adapter の hand 変換が hand_to_dex1() の per-step 呼びと一致することを
        # verify (adapter が意図せず追加処理を挟んでいないことの防御)。
        from model.subtask_policy_training.gr00t.dex1_hand_synergy import (
            hand_to_dex1,
        )
        # 意図的に synergy から外れた任意 7D vector を注入 (least-squares projection
        # される想定、両方が同じ projection を返すこと)。
        chunk = np.zeros((3, ACTION_DIM), dtype=np.float32)
        chunk[0, 18:25] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        chunk[0, 25:32] = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]
        chunk[1, 18:25] = [-1.0, 0.5, 0.3, -0.7, 0.9, 0.1, 0.2]
        chunk[1, 25:32] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        # chunk[2] は zeros のまま
        out = slice_53d_to_19d(chunk)
        for t in range(3):
            expected_l = hand_to_dex1(
                chunk[t, 18:25].tolist(), side="left", kind="action"
            )
            expected_r = hand_to_dex1(
                chunk[t, 25:32].tolist(), side="right", kind="action"
            )
            # adapter は float32 chunk 上で計算するので precision は float32 レベル
            # (~1e-7)。runtime env (numpy 1.26) と default env (numpy 2.x) で
            # 微妙に丸め挙動が違うため、tolerance を float32 相当に緩める。
            assert abs(out[t, 17] - expected_l) < 1e-5
            assert abs(out[t, 18] - expected_r) < 1e-5

    def test_hand_slice_consumes_all_seven_dims_not_just_first(self) -> None:
        # 旧 naive 実装 (position [18] / [25] のみ採用) では検出できない bug
        # を刺すテスト:
        #   Dex1=2.0 の synergy 7D 全体を注入 (正しい synergy 上の point)、
        #   ただし index [18] のみ "open" pose での値に上書き。
        #   - naive adapter (index[18] 直接読み) → open 位置由来の value を返す
        #     (physical rad scale で、2.0 ではない)
        #   - synergy adapter (7D 全部を hand_to_dex1 に渡す) → tampered 7D の
        #     projection = ほぼ 2.0 (index [18] だけ変更しても残 6 dim が支配的)
        # baseline は "変更前の 7D を渡した場合の adapter 出力" と一致し、
        # naive adapter の場合は index [18] のみに反応するので differ する。
        from model.subtask_policy_training.gr00t.dex1_hand_synergy import (
            dex1_to_hand, hand_to_dex1, load_synergy_manifest,
        )
        manifest = load_synergy_manifest()
        opened_pos0_left = float(manifest["left"]["action_open"][0])
        opened_pos0_right = float(manifest["right"]["action_open"][0])

        # Dex1=2.0 の synergy 7D
        seven_left_2_0 = dex1_to_hand(2.0, side="left", kind="action")
        seven_right_2_0 = dex1_to_hand(2.0, side="right", kind="action")

        # index [18] / [25] だけ "open pose" の値に差し替え = naive adapter が
        # 誤って open 側の physical rad scale を返すはず
        tampered_left = list(seven_left_2_0)
        tampered_left[0] = opened_pos0_left
        tampered_right = list(seven_right_2_0)
        tampered_right[0] = opened_pos0_right

        chunk = np.zeros((1, ACTION_DIM), dtype=np.float32)
        chunk[0, 18:25] = tampered_left
        chunk[0, 25:32] = tampered_right
        out = slice_53d_to_19d(chunk)

        # naive adapter なら out[17] == opened_pos0_left (~-0.028 rad級)、
        # synergy adapter なら hand_to_dex1(tampered) = 2.0 に近い値
        expected_left = hand_to_dex1(tampered_left, side="left", kind="action")
        expected_right = hand_to_dex1(tampered_right, side="right", kind="action")
        assert abs(out[0, 17] - expected_left) < 1e-4, (
            f"synergy 適用済 = {expected_left:.4f} を期待、adapter 出力 "
            f"{out[0,17]:.4f} (naive 版なら {opened_pos0_left:.4f} が返る)"
        )
        assert abs(out[0, 18] - expected_right) < 1e-4
        # かつ naive 出力とは有意に離れていることも防御
        assert abs(out[0, 17] - opened_pos0_left) > 0.1, (
            "adapter が naive index[18] を返している疑い (opened_pos0 と一致)"
        )
        assert abs(out[0, 18] - opened_pos0_right) > 0.1

    def test_dropped_dims_do_not_leak(self) -> None:
        # 現在の drop 対象は EEF 18 + base 4 = 22 dim (hand slice は synergy 逆射影
        # で消費されるので drop リストから外す)。53D 側で EEF+base のみ +1e6 を
        # 注入して waist/arm 部分の 19D に leak しないことを verify。
        # (hand 部分は 7D synergy 混入で意味が変わるので別 test でカバー、
        #  ここでは arm/waist の 17 dim [0:17] だけ verify する)
        chunk = np.zeros((1, ACTION_DIM), dtype=np.float32)
        drop_indices = (
            list(range(0, 18))              # EEF L+R (18 dim)
            + list(range(49, 53))           # base_height + navigate (4 dim)
        )
        for idx in drop_indices:
            chunk[0, idx] = 1e6
        # 保持対象 waist/arm の 17 index には検出可能な小さい値
        for idx in (46, 47, 48,             # waist 3
                    32, 33, 34, 35, 36, 37, 38,  # left arm 7
                    39, 40, 41, 42, 43, 44, 45): # right arm 7
            chunk[0, idx] = 1.0
        out = slice_53d_to_19d(chunk)
        assert np.all(out[:, 0:17] == 1.0), (
            f"drop 対象の EEF/base index が waist/arm slice に leak (max={out[:,0:17].max()})"
        )

    def test_rejects_wrong_last_dim(self) -> None:
        # 52 or 54 は fail-close (VlaSkill 側の 19D 検証と同種の防御)。
        for bad_last in (32, 52, 54, 19):
            chunk = np.zeros((3, bad_last), dtype=np.float32)
            with pytest.raises(ValueError, match=str(ACTION_DIM)):
                slice_53d_to_19d(chunk)

    def test_rejects_wrong_rank(self) -> None:
        chunk_1d = np.zeros((ACTION_DIM,), dtype=np.float32)
        chunk_3d = np.zeros((1, 3, ACTION_DIM), dtype=np.float32)
        with pytest.raises(ValueError):
            slice_53d_to_19d(chunk_1d)
        with pytest.raises(ValueError):
            slice_53d_to_19d(chunk_3d)

    def test_output_is_independent_copy(self) -> None:
        # 上流 tensor の再利用 (denorm dict の in-place 更新) と衝突しないよう、
        # 返却 array は 53D buffer と別 storage であること (write でも 53D は破壊されない)。
        chunk = self._make_53d(chunk_len=2)
        original = chunk.copy()
        out = slice_53d_to_19d(chunk)
        out[:] = -999.0
        assert np.array_equal(chunk, original), (
            "slice_53d_to_19d が 53D 入力を view で返している (書き込みが逆流している)"
        )


class TestGr00tPolicyActionDimContract:
    """Policy が VlaSkill contract (19D) を守ることの meta-verify。"""

    def test_class_exposes_53d_constant_for_debug(self) -> None:
        # 内部 layout は 53D のまま (metadata や debug 目的で参照される)
        assert Gr00tPolicy.ACTION_DIM == 53
        # 実 predict() 出力は 19D なので、上位 VlaSkill が受け入れ可能
        # (実 forward 検証は integration marker、ここは shape 契約の意図明示のみ)


# ---- Phase 2 (Issue #128): temporal ensemble integration into Gr00tPolicy ---- #


class TestGr00tPolicyTemporalEnsembleIntegration:
    """Gr00tPolicy が TargetTemporalEnsembler を持ち、reset / step counter を
    正しく管理していることの verify (実 lerobot forward を stub して確認)。"""

    def _make_stubbed_policy(self, temporal_lambda=-0.1) -> Gr00tPolicy:
        cfg = PolicyConfig(
            mode="none",
            ckpt_ref="Team-RAMEN/foo",
            dtype="bf16",
            cams=CAMERAS,
            temporal_lambda=temporal_lambda,
        )
        p = Gr00tPolicy(cfg)
        # ensembler は __init__ で作られる、_lerobot_policy は None のまま = predict は
        # 実 forward を stub して差替える必要がある。ここでは ensembler の存在と
        # reset の効果だけ確認する。
        return p

    def test_init_creates_ensembler(self) -> None:
        p = self._make_stubbed_policy()
        # _ensembler は Gr00tPolicy 内部 attribute (private)、直接触るのは test 特権
        assert hasattr(p, "_ensembler")
        assert p._ensembler.dim == 19
        assert p._ensembler.decay_lambda == -0.1
        assert p._current_step == 0

    def test_reset_clears_ensembler_state(self) -> None:
        p = self._make_stubbed_policy()
        # 手動で candidate を入れて counter を進める (predict を stub できないので
        # ensembler の attr 経由で直接 populate、reset() の効果検証だけに絞る)
        p._ensembler.add_chunk(
            origin_step=0, absolute_targets=np.ones((5, 19)) * 42.0
        )
        p._current_step = 5
        assert p._ensembler.candidate_count(0) == 1
        p.reset()
        assert p._ensembler.candidate_count(0) == 0
        assert p._current_step == 0

    def test_config_defaults_temporal_lambda_neg_zero_one(self) -> None:
        # PolicyConfig の default が -0.1 (Phase 2 kickoff の user 判断)
        cfg = PolicyConfig(mode="none", ckpt_ref="x", cams=CAMERAS)
        assert cfg.temporal_lambda == -0.1

    def test_config_temporal_lambda_none_is_valid(self) -> None:
        # blend 無効化の明示指定
        cfg = PolicyConfig(
            mode="none", ckpt_ref="x", cams=CAMERAS, temporal_lambda=None
        )
        assert cfg.temporal_lambda is None

    def test_config_rejects_non_finite_temporal_lambda(self) -> None:
        with pytest.raises(ValueError, match="temporal_lambda"):
            PolicyConfig(
                mode="none",
                ckpt_ref="x",
                cams=CAMERAS,
                temporal_lambda=float("nan"),
            )
        with pytest.raises(ValueError, match="temporal_lambda"):
            PolicyConfig(
                mode="none",
                ckpt_ref="x",
                cams=CAMERAS,
                temporal_lambda=float("inf"),
            )


# ---- Phase 4 (Issue #128): async_replanning integration into Gr00tPolicy ---- #


class TestGr00tPolicyAsyncPipelineIntegration:
    """Gr00tPolicy が cfg.replan_family 経由で AsyncActionChunkPipeline を
    正しく組み立て、reset() で bounded shutdown することの verify。
    実 lerobot forward を持たず、_sync_predict_chunk_19d を monkey-patch する。"""

    def _make_stub_policy(
        self,
        replan_family: str | None = "groot_relative_eef_v1",
        execution_steps: int = 10,
    ) -> Gr00tPolicy:
        cfg = PolicyConfig(
            mode="none",
            ckpt_ref="Team-RAMEN/foo",
            dtype="bf16",
            cams=CAMERAS,
            temporal_lambda=-0.1,
            replan_family=replan_family,
            execution_steps=execution_steps,
        )
        return Gr00tPolicy(cfg)

    def test_pipeline_field_is_none_by_default(self) -> None:
        # replan_family=None (Phase 2 sync mode) → pipeline は None のまま
        p = self._make_stub_policy(replan_family=None)
        assert p._pipeline is None

    def test_pipeline_is_lazy_initialized_on_first_predict(self) -> None:
        # replan_family 設定時、__init__ 直後は pipeline=None (lazy init)
        p = self._make_stub_policy(replan_family="groot_relative_eef_v1")
        assert p._pipeline is None
        # 実 predict() 呼ぶには lerobot 依存があるので、helper 経由で lazy init を trigger
        # ここでは init_pipeline を直接叩いて 挙動 verify (実 predict は integration test)
        seed = np.zeros((16, 19), dtype=np.float32)
        p._init_pipeline(seed)
        assert p._pipeline is not None
        assert p._pipeline_lead_steps is not None
        assert p._pipeline_max_age_s is not None
        assert p._pipeline.action_index == 1

    def test_late_async_boundary_holds_without_sync_fallback(self) -> None:
        p = self._make_stub_policy(execution_steps=4)
        release = threading.Event()
        calls = 0

        def predict_chunk(_obs, *, rtc_step=None):
            nonlocal calls
            calls += 1
            if calls > 1:
                assert release.wait(timeout=2.0)
            values = np.full((16, 19), float(calls), dtype=np.float32)
            return values, 1.0, np.zeros((16, 53), dtype=np.float32)

        p._sync_predict_chunk_19d = predict_chunk
        obs = p._make_dummy_observation()
        try:
            first = p.predict(obs)
            np.testing.assert_allclose(first.action_chunk, 1.0)
            p.predict(obs)  # submits the only asynchronous request
            # execution_steps controls replanning cadence, while the complete
            # native H16 seed remains available to TE/safety fallback.
            for _ in range(14):
                p.predict(obs)
            started = time.monotonic()
            held = p.predict(obs)
            assert time.monotonic() - started < 0.05
            np.testing.assert_allclose(held.action_chunk, first.action_chunk)
            assert held.metadata["chunk_source"] == "async_hold"
            # One seed call + one in-flight async call.  There is no third,
            # command-thread synchronous fallback at the H4 boundary.
            assert calls == 2
        finally:
            release.set()
            p.reset()

    def test_reset_closes_active_pipeline_and_clears_state(self) -> None:
        p = self._make_stub_policy()
        seed = np.zeros((16, 19), dtype=np.float32)
        p._init_pipeline(seed)
        assert p._pipeline is not None
        p.reset()
        assert p._pipeline is None
        assert p._current_step == 0
        assert p._last_seen_obs is None

    def test_config_defaults_replan_family_none(self) -> None:
        # PolicyConfig default replan_family=None (async 無効)、後方互換
        cfg = PolicyConfig(mode="none", ckpt_ref="x", cams=CAMERAS)
        assert cfg.replan_family is None
        assert cfg.execution_steps == 10

    def test_config_rejects_non_positive_execution_steps(self) -> None:
        with pytest.raises(ValueError, match="execution_steps"):
            PolicyConfig(
                mode="none", ckpt_ref="x", cams=CAMERAS, execution_steps=0,
            )
        with pytest.raises(ValueError, match="execution_steps"):
            PolicyConfig(
                mode="none", ckpt_ref="x", cams=CAMERAS, execution_steps=-1,
            )

    def test_config_rejects_wrong_type_replan_family(self) -> None:
        with pytest.raises(TypeError, match="replan_family"):
            PolicyConfig(
                mode="none", ckpt_ref="x", cams=CAMERAS,
                replan_family=123,  # type: ignore[arg-type]
            )

    def test_pipeline_family_lookup_uses_correct_profile(self) -> None:
        # groot_relative_eef_v1 = (lead=4, max_age=0.35s) issue-70 verified
        p = self._make_stub_policy(
            replan_family="groot_relative_eef_v1", execution_steps=10
        )
        seed = np.zeros((16, 19), dtype=np.float32)
        p._init_pipeline(seed)
        assert p._pipeline_lead_steps == 4
        assert abs(p._pipeline_max_age_s - 0.35) < 1e-9


class TestGr00tPolicyInferenceSerialization:
    """Issue #137: in-process 推論区間が 1 スレッド限定であることの verify。

    GR00T の pre_processor (GrootN17PackInputsStep) は raw state をインスタンスに
    キャッシュし、post_processor の decode がそれを読んで relative → absolute を
    復元する。async worker と main の sync_fallback が同時にこの区間へ入ると、
    後から state を書いた側の基準で相手の chunk が復元され、error を出さずに
    誤った絶対 action になる。
    """

    def _make_policy(self) -> Gr00tPolicy:
        policy = Gr00tPolicy(
            PolicyConfig(mode="none", ckpt_ref="Team-RAMEN/foo", cams=CAMERAS)
        )
        # _lerobot_policy None チェックを通すためのダミー (forward は stub 側で差替え)
        policy._lerobot_policy = object()
        return policy

    def test_lock_exists(self) -> None:
        assert isinstance(self._make_policy()._inference_lock, threading.Lock().__class__)

    def test_concurrent_calls_do_not_overlap(self) -> None:
        policy = self._make_policy()
        active = 0
        max_active = 0
        bookkeeping = threading.Lock()

        def stub(_obs, *, rtc_step=None):
            nonlocal active, max_active
            with bookkeeping:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with bookkeeping:
                active -= 1
            return (
                np.zeros((16, 19), dtype=np.float32),
                1.0,
                np.zeros((16, ACTION_DIM), dtype=np.float32),
            )

        policy._in_process_predict_chunk_19d = stub

        threads = [
            threading.Thread(target=policy._sync_predict_chunk_19d, args=(None,))
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive()

        assert max_active == 1, f"inference section ran concurrently ({max_active})"

    def test_worker_client_path_does_not_take_the_lock(self) -> None:
        # worker process 経路は _GrootWorkerClient 側の _request_lock が守るので、
        # in-process lock を取らずに早期 return すること (二重 lock を避ける)。
        policy = self._make_policy()

        class FakeWorker:
            def predict(self, _obs, *, rtc_step=None, rtc_tick_period_s=None):
                assert not policy._inference_lock.locked()
                assert rtc_step == 7
                assert rtc_tick_period_s == pytest.approx(0.033)
                return PolicyAction(
                    action_chunk=np.zeros((16, 19), dtype=np.float32),
                    latency_ms=1.0,
                    metadata={"raw_action_shape_53d": (16, ACTION_DIM)},
                )

        policy._worker_client = FakeWorker()
        for _ in range(3):
            policy._rtc_tick_deltas.append(0.033)
        chunk, latency_ms, raw = policy._sync_predict_chunk_19d(None, rtc_step=7)
        assert chunk.shape == (16, 19)
        assert raw.shape == (16, ACTION_DIM)
        assert latency_ms == 1.0


def _contract(*, chunk_size: int) -> GrootArtifactContract:
    """RTC の検証に必要な chunk_size だけを持つ最小の artifact contract。"""
    return GrootArtifactContract(
        policy_type="groot",
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        video_horizon=1,
        history_offset_ticks=1,
        expert_role=None,
    )


class TestRtcChunkLenValidation:
    """Issue #137: overlap は chunk 長を超えられない。

    chunk_size は ckpt の config.json にしか無いので config_loader では検証
    できないが、ここは **重み load の前** なので設定ミスは即座に落ちる。
    """

    def _cfg(self, **rtc_kwargs) -> PolicyConfig:
        return PolicyConfig(
            mode="none",
            ckpt_ref="x",
            cams=CAMERAS,
            execution_steps=rtc_kwargs.pop("execution_steps", 8),
            rtc=RtcConfig(**rtc_kwargs),
        )

    def test_disabled_skips_validation(self) -> None:
        # RTC off なら overlap が chunk を超えていても既存 variant は動くべき
        validate_rtc_against_chunk_len(self._cfg(overlap_steps=64), chunk_len=16)

    def test_overlap_within_chunk_len_passes(self) -> None:
        validate_rtc_against_chunk_len(
            self._cfg(enabled=True, overlap_steps=8), chunk_len=16
        )

    def test_overlap_exceeding_chunk_len_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds the checkpoint chunk_size"):
            validate_rtc_against_chunk_len(
                self._cfg(enabled=True, overlap_steps=20), chunk_len=16
            )

    def test_null_overlap_falls_back_to_execution_steps(self) -> None:
        # overlap_steps: null は execution_steps を流用するので、そちらが
        # 超えていれば同じく落ちる
        with pytest.raises(ValueError, match="exceeds the checkpoint chunk_size"):
            validate_rtc_against_chunk_len(
                self._cfg(enabled=True, execution_steps=20), chunk_len=16
            )

    def test_frozen_exceeding_resolved_overlap_rejected(self) -> None:
        # overlap_steps: null のとき RtcConfig 単体では frozen と比較できない
        # (execution_steps を知らないため)。解決後の overlap との突き合わせは
        # ここが唯一の防波堤になる。
        with pytest.raises(ValueError, match="rtc.frozen_steps"):
            validate_rtc_against_chunk_len(
                self._cfg(enabled=True, frozen_steps=9, execution_steps=8),
                chunk_len=16,
            )

    def test_auto_frozen_is_not_compared_statically(self) -> None:
        validate_rtc_against_chunk_len(
            self._cfg(enabled=True, overlap_steps=16), chunk_len=16
        )


class TestGr00tPolicyRtcState:
    """RTC 有効/無効での Gr00tPolicy 側の状態管理。"""

    def _policy(self, **rtc_kwargs) -> Gr00tPolicy:
        # 実 config の GR00T variant は全て async (replan_family 設定済み)。
        # 同期実行では frozen=0 が確定するので tick 周期に依存しなくなり、
        # auto 推定側の経路を検証できない。
        replan_family = rtc_kwargs.pop("replan_family", "groot_relative_eef_v1")
        return Gr00tPolicy(
            PolicyConfig(
                mode="none",
                ckpt_ref="x",
                cams=CAMERAS,
                execution_steps=8,
                replan_family=replan_family,
                rtc=RtcConfig(**rtc_kwargs),
            )
        )

    def test_disabled_creates_no_leftover_buffer(self) -> None:
        # RTC off の間は prefix 経路に一切入らない = 従来動作と同一
        policy = self._policy()
        assert policy._rtc_leftover is None
        assert policy._build_rtc_prefix(0) == ({}, {"rtc_enabled": False})

    def test_enabled_creates_leftover_buffer(self) -> None:
        policy = self._policy(enabled=True)
        assert policy._rtc_leftover is not None
        assert policy._rtc_leftover.action_dim == ACTION_DIM

    def test_no_prefix_before_first_chunk(self) -> None:
        # 初回は前 chunk が無いので prefix 無しの通常推論
        policy = self._policy(enabled=True)
        kwargs, metadata = policy._build_rtc_prefix(0)
        assert kwargs == {}
        assert metadata == {"rtc_enabled": True, "rtc_prefix_rows": 0}

    def test_no_prefix_before_tick_period_is_measured(self) -> None:
        # frozen_steps: auto は tick 周期が測れるまで d を決められない。
        # 過小評価は送信済み step の書き換えになるので、その間は RTC を使わない。
        policy = self._policy(enabled=True)
        policy._artifact_contract = _contract(chunk_size=16)
        policy._rtc_leftover.store(
            np.zeros((16, ACTION_DIM), dtype=np.float32), origin_step=0
        )
        policy._rtc_delay.add(0.09)
        kwargs, metadata = policy._build_rtc_prefix(8)
        assert kwargs == {}
        assert metadata["rtc_prefix_rows"] == 0

    def test_external_tick_period_enables_worker_rtc_prefix(self) -> None:
        # DDS parent measures the 30 Hz control cadence.  The model worker only
        # sees sparse replan requests, so it must use the forwarded period.
        policy = self._policy(enabled=True)
        policy._artifact_contract = _contract(chunk_size=16)
        policy._rtc_leftover.store(
            np.zeros((16, ACTION_DIM), dtype=np.float32), origin_step=0
        )
        policy._rtc_delay.add(0.09)
        class FakePackStep:
            @staticmethod
            def get_cached_raw_state():
                return {"dummy": np.zeros((1, 1), dtype=np.float32)}

        class FakeDecodeStep:
            pack_step = FakePackStep()

        policy._find_action_decode_step = FakeDecodeStep

        import inference.desktop.lower_policy.policies.groot_rtc as rtc_module

        original = rtc_module.encode_absolute_chunk_to_model_space
        rtc_module.encode_absolute_chunk_to_model_space = lambda chunk, **_kwargs: chunk
        try:
            kwargs, metadata = policy._build_rtc_prefix(
                8, tick_period_s_override=0.033
            )
        finally:
            rtc_module.encode_absolute_chunk_to_model_space = original

        assert kwargs["inference_delay"] == 3
        assert metadata["rtc_prefix_rows"] == 8
        assert metadata["rtc_tick_period_ms"] == pytest.approx(33.0)

    def test_tick_period_uses_median(self) -> None:
        policy = self._policy(enabled=True)
        assert policy._rtc_tick_period_s() is None  # sample 不足
        for value in (0.033, 0.034, 0.500, 0.033):
            policy._rtc_tick_deltas.append(value)
        # 外れ値 0.5 に引きずられない
        assert policy._rtc_tick_period_s() == pytest.approx(0.0335, abs=1e-6)

    def test_reset_clears_rtc_state(self) -> None:
        # 前 skill の chunk を新 skill の prefix に混ぜない
        policy = self._policy(enabled=True)
        policy._rtc_leftover.store(
            np.zeros((16, ACTION_DIM), dtype=np.float32), origin_step=0
        )
        policy._rtc_delay.add(0.09)
        policy._rtc_tick_deltas.append(0.033)
        policy._rtc_last_tick_ns = 123
        policy.reset()
        assert policy._rtc_leftover.remaining(0) is None
        assert policy._rtc_delay.max_latency_s is None
        assert len(policy._rtc_tick_deltas) == 0
        assert policy._rtc_last_tick_ns is None

    def test_error_disables_rtc_for_the_instance(self) -> None:
        # 実機 slot を落とさず、かつ「時々効く」状態にして数字を濁らせない
        policy = self._policy(enabled=True)
        policy._artifact_contract = _contract(chunk_size=16)
        policy._rtc_leftover.store(
            np.zeros((16, ACTION_DIM), dtype=np.float32), origin_step=0
        )
        policy._rtc_delay.add(0.09)
        for _ in range(3):
            policy._rtc_tick_deltas.append(0.033)
        policy._find_action_decode_step = lambda: (_ for _ in ()).throw(
            RuntimeError("no decode step")
        )
        kwargs, metadata = policy._build_rtc_prefix(8)
        assert kwargs == {}
        assert metadata["rtc_enabled"] is False
        assert metadata["rtc_error"] == "RuntimeError"
        assert policy._rtc_disabled_after_error is True
        # 以降の tick も RTC を使わない
        assert policy._build_rtc_prefix(9) == ({}, {"rtc_enabled": False})

    def test_reset_does_not_re_enable_after_error(self) -> None:
        policy = self._policy(enabled=True)
        policy._rtc_disabled_after_error = True
        policy.reset()
        assert policy._rtc_disabled_after_error is True


class TestGr00tPolicyWarmupClearsRtcState:
    """Issue #137: warmup の dummy predict が RTC 状態を汚染しないこと。

    RAMEN-Ori の実 ckpt smoke で見つかった問題。GR00T も warmup が predict を
    叩く同じ構造なので、同じガードを張る。
    """

    def test_warmup_discards_its_own_rtc_state(self) -> None:
        policy = Gr00tPolicy(
            PolicyConfig(
                mode="none",
                ckpt_ref="x",
                cams=CAMERAS,
                execution_steps=8,
                replan_family="groot_relative_eef_v1",
                rtc=RtcConfig(enabled=True),
            )
        )
        policy._lerobot_policy = object()
        policy._make_dummy_observation = lambda: "dummy"

        def dirty(_obs):
            policy._rtc_delay.add(1.5)          # 初回 JIT の外れ値
            policy._rtc_tick_deltas.append(0.033)
            policy._rtc_leftover.store(
                np.zeros((16, ACTION_DIM), dtype=np.float32), origin_step=0
            )
            policy._current_step += 1

        policy.predict = dirty
        policy.warmup(n_iter=3)

        assert policy._rtc_delay.max_latency_s is None
        assert len(policy._rtc_tick_deltas) == 0
        assert policy._rtc_leftover.remaining(0) is None
        assert policy._current_step == 0


# ---- GR00T 53D の worker の待ち時間 (B3c-2) --------------------------------------------


class _FakeWarmupPolicy:
    def __init__(self) -> None:
        self.warmups: list[int] = []
        self.closed = False

    def warmup(self, n_iter: int) -> None:
        self.warmups.append(n_iter)

    def close(self) -> None:
        self.closed = True


def _worker_socket_path():
    import uuid
    from pathlib import Path

    # AF_UNIX の path 長 (macOS 104 byte) に収まるよう /tmp 直下に作る
    return Path("/tmp") / f"groot_test_{uuid.uuid4().hex[:8]}.sock"


def _start_worker(policy, path, accept_timeout_s: float) -> threading.Thread:
    from inference.desktop.lower_policy.policies import groot_worker

    thread = threading.Thread(
        target=groot_worker.serve,
        args=(policy, path),
        kwargs={"accept_timeout_s": accept_timeout_s},
        daemon=True,
    )
    thread.start()
    return thread


def test_worker_waits_for_requests_without_an_idle_limit(monkeypatch) -> None:
    """warmup の後は次の要求を時間制限なしで待ち (Enter 待ちで落ちない)、切断で終わる。"""
    timeouts: list[float | None] = []
    original = socket.socket.settimeout

    def record(self, value):
        timeouts.append(value)
        return original(self, value)

    monkeypatch.setattr(socket.socket, "settimeout", record)
    policy = _FakeWarmupPolicy()
    path = _worker_socket_path()
    thread = _start_worker(policy, path, accept_timeout_s=5.0)
    deadline = time.monotonic() + 5.0
    while not path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as runtime:
        runtime.connect(str(path))
        send_archive(runtime, kind=np.asarray("warmup"), count=np.asarray([1]))
        assert int(receive_archive(runtime)["ok"][0]) == 1
        time.sleep(0.2)
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert policy.warmups == [1] and policy.closed
    # accept だけ期限付き、つないだ後の要求待ちは期限なし (以前は 600 s で worker が落ちた)
    assert timeouts[:2] == [5.0, None]
    assert 600.0 not in timeouts


def test_worker_exits_when_no_runtime_connects() -> None:
    """接続前に runtime が落ちても、GPU を掴んだ worker を残さない。"""
    policy = _FakeWarmupPolicy()
    path = _worker_socket_path()
    thread = _start_worker(policy, path, accept_timeout_s=0.2)
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert policy.closed
    assert not path.exists()


def test_client_waits_for_a_slow_worker_load_without_a_time_limit(monkeypatch, capsys) -> None:
    """worker が 300 s を超えて model を読んでいても、生きている間は待ち続ける。"""
    import inference.desktop.lower_policy.policies.groot as groot_module

    class _AliveProcess:
        def poll(self):
            return None

    client = object.__new__(groot_module._GrootWorkerClient)
    path = _worker_socket_path()
    client._process = _AliveProcess()
    client._socket_path = path
    clock = {"now": 0.0}
    worker_socket: list[socket.socket] = []

    def fake_sleep(_seconds: float) -> None:
        clock["now"] += 50.0
        if clock["now"] >= 400.0 and not worker_socket:  # 400 s 後に読み終えて socket を開く
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)
            worker_socket.append(server)

    monkeypatch.setattr(groot_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(groot_module.time, "sleep", fake_sleep)
    try:
        connection = client._wait_for_worker_socket()
        connection.close()
    finally:
        for server in worker_socket:
            server.close()
        path.unlink(missing_ok=True)
    assert clock["now"] >= 400.0
    assert "waiting for the GR00T worker to load" in capsys.readouterr().err


def test_client_stops_the_worker_on_ctrl_c_during_startup(monkeypatch, tmp_path) -> None:
    """起動待ち中の Ctrl+C でも worker を落とす (別 session なので SIGINT は届かない)。"""
    import subprocess
    import sys
    from types import SimpleNamespace

    import inference.desktop.lower_policy.policies.groot as groot_module

    class _FakeWorkerProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("pixi", timeout)
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    process = _FakeWorkerProcess()
    monkeypatch.setattr(
        groot_module, "_resolve_groot_checkpoint_root", lambda *_args: tmp_path
    )
    monkeypatch.setattr(
        groot_module,
        "_read_groot_artifact_contract",
        lambda _root: SimpleNamespace(video_horizon=1, history_offset_ticks=0),
    )
    monkeypatch.setattr(groot_module.shutil, "which", lambda _name: sys.executable)
    monkeypatch.setattr(groot_module.subprocess, "Popen", lambda *a, **k: process)

    def interrupted(self, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        groot_module._GrootWorkerClient, "_wait_for_worker_socket", interrupted
    )
    with pytest.raises(KeyboardInterrupt):
        groot_module._GrootWorkerClient(
            PolicyConfig(mode="none", ckpt_ref="x", cams=CAMERAS)
        )
    assert process.terminated


def test_worker_starts_with_the_pixi_env_as_is(monkeypatch, tmp_path) -> None:
    """worker は image に入れた環境をそのまま使う (会場はオフライン。pixi に install させない)。"""
    import sys
    from types import SimpleNamespace

    import inference.desktop.lower_policy.policies.groot as groot_module

    started: list[list[str]] = []

    class _Process:
        returncode = -15

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def fake_popen(command, **_kwargs):
        started.append(list(command))
        return _Process()

    monkeypatch.setattr(groot_module, "_resolve_groot_checkpoint_root", lambda *_a: tmp_path)
    monkeypatch.setattr(
        groot_module,
        "_read_groot_artifact_contract",
        lambda _root: SimpleNamespace(video_horizon=1, history_offset_ticks=0),
    )
    monkeypatch.setattr(groot_module.shutil, "which", lambda _name: sys.executable)
    monkeypatch.setattr(groot_module.subprocess, "Popen", fake_popen)

    def interrupted(self, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(groot_module._GrootWorkerClient, "_wait_for_worker_socket", interrupted)
    with pytest.raises(KeyboardInterrupt):
        groot_module._GrootWorkerClient(PolicyConfig(mode="none", ckpt_ref="x", cams=CAMERAS))
    (command,) = started
    assert command[1:3] == ["run", "--as-is"]


def test_worker_without_lerobot_stops_instead_of_starting_another_worker(monkeypatch) -> None:
    """lerobot の無い Python で起動した worker は、worker を起こさずに案内を出して終わる (Issue #194)。

    inference/desktop の環境が無いと worker が外側の Python で動き、Gr00tPolicy.from_ckpt が
    また worker を起こし続けていた (RTX 5090 の PC で 20 分・1,126 回)。
    """
    import importlib.util
    import sys

    from inference.desktop.lower_policy.policies import groot_worker

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name == "lerobot" else real_find_spec(name, *args, **kwargs)
        ),
    )

    def must_not_load(_cfg):
        raise AssertionError("the worker must not load GR00T without lerobot")

    monkeypatch.setattr(groot_worker.Gr00tPolicy, "from_ckpt", must_not_load)
    monkeypatch.setattr(
        sys,
        "argv",
        ["groot_worker", "--socket", "/tmp/unused.sock", "--mode", "none", "--ckpt-ref", "x",
         "--device", "cpu", "--dtype", "float32"],
    )
    with pytest.raises(SystemExit) as excinfo:
        groot_worker.main()
    assert "cd inference/desktop && pixi install" in str(excinfo.value)
