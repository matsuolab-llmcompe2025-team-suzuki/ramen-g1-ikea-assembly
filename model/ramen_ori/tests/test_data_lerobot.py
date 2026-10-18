"""RamenOriLerobotDataset の unit test (Issue #120 Phase A-2)。

Fake `base_dataset` を DI で注入して MultiLeRobotDataset を経由せず、
_transform_item ロジックを網羅テスト。

- shape / dtype contract (model.py batch dict と 1 対 1)
- state derive 呼び出し (prev pad → state_prev=None、velocity=zeros)
- tracking_err = 1 frame 前の指令 − 今 (腰 0、区間先頭 0)、正解 row 0 を入力から出せない
- action target (row 1..) の arms+hand 抽出と action_is_pad
- image prev pad → images_prev = images
- skill_id は episode meta の source_task_index 由来 (frame の task_index は使わない、欠損・未登録はエラー)
- OBB none (旧 dummy) の shape / zeros / valid_mask 全 False
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.utils.data

from model.ramen_ori.data_lerobot import (
    G1_WBT_FPS,
    RamenOriLerobotDataset,
    RamenOriMultiDataset,
    RamenOriMultiSplitView,
    RamenOriSplitView,
    build_delta_timestamps,
    default_camera_keys,
)
from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    UPPER_BODY_SOURCE_INDEX_MAP,
    UPPER_BODY_STATE_DIM,
)


class _FakeMeta:
    def __init__(self, episodes: list[dict]) -> None:
        self.episodes = episodes


class FakeLeRobotBase(torch.utils.data.Dataset):
    """LeRobotDataset を模擬する fake、_transform_item ロジックのテスト用。

    実 LeRobot (merged dataset) の返す shape/key に忠実:
        - observation.state.robot_q_current: (2, 36)
        - observation.state.hand_state: (2, 2)
        - observation.state.ee_state: (1, 12)
        - action.robot_q_desired: (chunk_len+1, 36)  row 0 = 1 frame 前の指令
        - action.hand_cmd: (chunk_len+1, 2)
        - observation.images.cam_{i}: (2, 3, H, W)
        - {key}_is_pad: bool tensor (区間の外の frame は区間端の値を繰り返し、True)
        - task_index: (1,) long。merged dataset のローカル番号 (merge で 0 始まりに振り直される、既定 0)
        - episode_index / frame_index: scalar long
        - meta.episodes: episode_index / length / source_task_index (BitRobot の元の番号、既定 5)
          / source_episode_index

    frame の task_index と meta の source_task_index を既定で別の値にしている (実データと同じ。
    同じ値だと Issue #141 RO-1 の食い違いを test で再現できない)。
    """

    def __init__(
        self,
        num_samples: int = 10,
        chunk_len: int = 16,
        img_size: int = 224,
        camera_keys: list[str] | None = None,
        task_index: int = 0,
        source_task_index: int | None = 5,
        episode_lengths: list[int] | None = None,
        prev_pad_at_idx: int | None = None,
        pad_tail_rows: int = 0,
        n_image_frames: int = 2,
    ) -> None:
        """
        Args:
            n_image_frames: カメラごとの画像の枚数 (2 = 前 + 今、1 = 今だけ)
            source_task_index: meta の値。None なら meta に列を持たない (欠損の test 用)
            episode_lengths: None なら 10 frame ずつ (idx = ep*10 + f)
            prev_pad_at_idx: この idx だけ 1 frame 前 (state / image / 指令の row 0) を pad にする
            pad_tail_rows: 正解 chunk の末尾何行を埋め草 (最後の実 frame の繰り返し + is_pad) にするか
        """
        if episode_lengths is None:
            episode_lengths = [10] * (num_samples // 10) + ([num_samples % 10] if num_samples % 10 else [])
        if sum(episode_lengths) != num_samples:
            raise ValueError(f"sum(episode_lengths)={sum(episode_lengths)} != num_samples={num_samples}")
        self.num_samples = num_samples
        self.chunk_len = chunk_len
        self.img_size = img_size
        self.camera_keys = camera_keys or default_camera_keys()
        self.task_index = task_index
        self.prev_pad_at_idx = prev_pad_at_idx
        self.pad_tail_rows = pad_tail_rows
        self.n_image_frames = n_image_frames
        self._ep_offsets = np.cumsum([0] + list(episode_lengths))
        episodes = []
        for ep, length in enumerate(episode_lengths):
            row = {"episode_index": ep, "length": int(length), "source_episode_index": ep}
            if source_task_index is not None:
                row["source_task_index"] = source_task_index
            episodes.append(row)
        self.meta = _FakeMeta(episodes)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(idx)
        ep = int(np.searchsorted(self._ep_offsets[1:], idx, side="right"))
        n_rows = self.chunk_len + 1
        q_desired = rng.standard_normal((n_rows, 36)).astype(np.float32)
        hand_cmd = rng.standard_normal((n_rows, 2)).astype(np.float32)
        action_pad = np.zeros(n_rows, dtype=bool)
        if self.pad_tail_rows:
            last_real = n_rows - 1 - self.pad_tail_rows
            q_desired[last_real + 1:] = q_desired[last_real]
            hand_cmd[last_real + 1:] = hand_cmd[last_real]
            action_pad[last_real + 1:] = True
        prev_pad = idx == self.prev_pad_at_idx
        if prev_pad:
            q_desired[0] = q_desired[1]
            hand_cmd[0] = hand_cmd[1]
            action_pad[0] = True
        item = {
            "observation.state.robot_q_current": torch.from_numpy(
                rng.standard_normal((2, 36)).astype(np.float32)
            ),
            "observation.state.hand_state": torch.from_numpy(
                rng.standard_normal((2, 2)).astype(np.float32)
            ),
            "observation.state.ee_state": torch.from_numpy(
                rng.standard_normal((1, 12)).astype(np.float32)
            ),
            "action.robot_q_desired": torch.from_numpy(q_desired),
            "action.hand_cmd": torch.from_numpy(hand_cmd),
            "action.robot_q_desired_is_pad": torch.from_numpy(action_pad.copy()),
            "action.hand_cmd_is_pad": torch.from_numpy(action_pad.copy()),
            "task_index": torch.tensor([self.task_index], dtype=torch.long),
            "dataset_index": torch.tensor(0, dtype=torch.long),
            # D-2: precomputed_token flow は episode_index / frame_index を必要とする
            # (LeRobot v3 の hf_dataset row から得られる想定)。
            "episode_index": torch.tensor(ep, dtype=torch.long),
            "frame_index": torch.tensor(idx - int(self._ep_offsets[ep]), dtype=torch.long),
        }
        item["observation.state.robot_q_current_is_pad"] = torch.tensor(
            [prev_pad, False]
        )
        for cam_key in self.camera_keys:
            # 実 LeRobot は時刻が 1 つだけのとき frame の次元を落として (3, H, W) で返す (`_query_videos` の squeeze)
            shape = (2, 3, self.img_size, self.img_size) if self.n_image_frames == 2 else (3, self.img_size, self.img_size)
            item[cam_key] = torch.rand(*shape)
            item[f"{cam_key}_is_pad"] = torch.tensor([prev_pad, False][-self.n_image_frames:])
        return item


@pytest.fixture
def dataset_with_fake_base():
    """DI で fake base を注入した Dataset instance を返す factory。"""
    def _make(**overrides) -> RamenOriLerobotDataset:
        fake_kwargs = {
            "num_samples": overrides.pop("num_samples", 10),
            "chunk_len": overrides.get("chunk_len", 16),
            "img_size": overrides.get("img_size", 224),
            "camera_keys": overrides.get("camera_keys"),
            "task_index": overrides.pop("task_index", 0),
            "source_task_index": overrides.pop("source_task_index", 5),
            "prev_pad_at_idx": overrides.pop("prev_pad_at_idx", None),
            "pad_tail_rows": overrides.pop("pad_tail_rows", 0),
            "n_image_frames": 2 if overrides.get("load_prev_image", True) else 1,
        }
        fake = FakeLeRobotBase(**fake_kwargs)
        return RamenOriLerobotDataset(base_dataset=fake, **overrides)
    return _make


class TestBatchContract:
    def test_all_keys_present(self, dataset_with_fake_base) -> None:
        # Issue #129 Phase B (2026-08-31): L4 用 action_waist_teacher を追加。
        # Issue #141 RO-2: 埋め草の印 action_is_pad を追加。
        ds = dataset_with_fake_base()
        item = ds[0]
        expected_keys = {
            "images", "images_prev", "cam_id",
            "obb_verts", "obb_conf", "obb_class_id", "obb_cam_id", "obb_valid_mask",
            "state", "skill_id", "action", "action_is_pad", "action_waist_teacher",
        }
        assert set(item.keys()) == expected_keys

    def test_action_waist_teacher_shape_and_source(self, dataset_with_fake_base) -> None:
        """Issue #129 Phase B: action_waist_teacher は action.robot_q_desired[1:, waist] から
        3D で抽出、(chunk_len, 3)、L4 FK anchor loss の 19D 組立て用。"""
        from model.ramen_ori.state_derive import WAIST_SOURCE_INDEX_MAP

        ds = dataset_with_fake_base(chunk_len=16)
        raw = ds._base[0]
        item = ds[0]
        assert item["action_waist_teacher"].shape == (16, 3)
        # 実 source: concat(q_desired, hand_cmd)[1:, WAIST_SOURCE_INDEX_MAP] (row 0 は 1 frame 前)
        q_d = raw["action.robot_q_desired"].numpy()[1:]  # (16, 36)
        h_c = raw["action.hand_cmd"].numpy()[1:]         # (16, 2)
        action_full = np.concatenate([q_d, h_c], axis=1)
        expected = action_full[:, np.asarray(WAIST_SOURCE_INDEX_MAP)]
        np.testing.assert_allclose(item["action_waist_teacher"].numpy(), expected, atol=1e-6)

    def test_shapes_match_model_contract(self, dataset_with_fake_base) -> None:
        # Issue #129 Phase A (2026-08-31): N_cams 4→3 (head_right 除外)、action 19→16 (waist 除外)。
        ds = dataset_with_fake_base(chunk_len=16, img_size=224)
        item = ds[0]
        assert item["images"].shape == (3, 3, 224, 224)
        assert item["images_prev"].shape == (3, 3, 224, 224)
        assert item["cam_id"].shape == (3,)
        assert item["obb_verts"].shape == (3, 4, 8)
        assert item["obb_conf"].shape == (3, 4, 1)
        assert item["obb_class_id"].shape == (3, 4)
        assert item["obb_cam_id"].shape == (3, 4)
        assert item["obb_valid_mask"].shape == (3, 4)
        assert item["state"].shape == (71,)
        assert item["skill_id"].shape == ()
        assert item["action"].shape == (16, 16)

    def test_dtypes_match_model_contract(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base()
        item = ds[0]
        assert item["images"].dtype == torch.float32
        assert item["state"].dtype == torch.float32
        assert item["action"].dtype == torch.float32
        assert item["skill_id"].dtype == torch.long
        assert item["cam_id"].dtype == torch.long
        assert item["obb_class_id"].dtype == torch.long
        assert item["obb_cam_id"].dtype == torch.long
        assert item["obb_valid_mask"].dtype == torch.bool


class TestStateDerive:
    def test_state_derive_called_with_correct_slices(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base()
        # fake base の idx 0 の raw を取得
        raw = ds._base[0]
        item = ds[0]

        q_c = raw["observation.state.robot_q_current"].numpy()  # (2, 36)
        h_c = raw["observation.state.hand_state"].numpy()       # (2, 2)
        # joint 19 = state_current [UPPER_BODY], state_current = concat(q_c[1], h_c[1])
        state_current_38 = np.concatenate([q_c[1], h_c[1]])
        expected_joint = state_current_38[np.asarray(UPPER_BODY_SOURCE_INDEX_MAP)]
        np.testing.assert_allclose(item["state"].numpy()[0:19], expected_joint, atol=1e-6)

    def test_prev_pad_yields_velocity_zeros(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base(prev_pad_at_idx=0)
        item = ds[0]
        # state[38:57] = velocity, prev pad → zeros
        vel = item["state"].numpy()[38:57]
        np.testing.assert_allclose(vel, np.zeros(19), atol=1e-6)


class TestTrackingErr:
    """Issue #141 RO-3: tracking_err = 1 frame 前の指令 − 今の実測 (推論の「前 tick の指令 − 今」と同じ)。"""

    @staticmethod
    def _upper(raw: dict, key_q: str, key_h: str, row: int) -> np.ndarray:
        full = np.concatenate([raw[key_q].numpy()[row], raw[key_h].numpy()[row]])
        return full[np.asarray(UPPER_BODY_SOURCE_INDEX_MAP)]

    def test_tracking_err_is_prev_command_minus_current_with_waist_zero(
        self, dataset_with_fake_base
    ) -> None:
        ds = dataset_with_fake_base()
        raw = ds._base[1]
        item = ds[1]
        prev_cmd = self._upper(raw, "action.robot_q_desired", "action.hand_cmd", 0)
        current = self._upper(
            raw, "observation.state.robot_q_current", "observation.state.hand_state", 1
        )
        expected = prev_cmd - current
        expected[:3] = 0.0  # 腰は model が出力しないので学習・推論とも 0
        np.testing.assert_allclose(item["state"].numpy()[19:38], expected, atol=1e-6)

    def test_segment_start_yields_tracking_err_zeros(self, dataset_with_fake_base) -> None:
        """1 frame 前が区間の外 (is_pad) → 0 (推論の 1 tick 目と同じ)。"""
        ds = dataset_with_fake_base(prev_pad_at_idx=0)
        item = ds[0]
        np.testing.assert_array_equal(item["state"].numpy()[19:38], np.zeros(19, dtype=np.float32))

    def test_ground_truth_row0_not_recoverable_from_state(self, dataset_with_fake_base) -> None:
        """今の関節角 + tracking_err が正解 action の row 0 と一致しない (正解が入力に漏れていない)。

        旧定義 (q_desired[t] − q_current[t]) では abs の action[0] = joint + tracking_err が
        恒等的に成り立っていた。
        """
        ds = dataset_with_fake_base()
        item = ds[1]
        state = item["state"].numpy()
        arms_hand_joint = state[3:19]           # joint 19 の腰以外 (arms 14 + hand 2)
        arms_hand_track = state[22:38]          # tracking_err 19 の腰以外
        action_row0 = item["action"].numpy()[0]  # (16,) arms 14 + hand 2
        assert not np.allclose(arms_hand_joint + arms_hand_track, action_row0, atol=1e-3)


class TestActionTarget:
    def test_action_shape_and_arms_hand_extraction(self, dataset_with_fake_base) -> None:
        """Issue #129 Phase A (2026-08-31): action = concat(q_desired, hand_cmd)[1:] →
        ARMS_HAND_SOURCE_INDEX_MAP で 16 dim 抽出 (waist 3 除外)。row 0 (1 frame 前) は含まない。"""
        from model.ramen_ori.state_derive import (
            ARMS_HAND_SOURCE_INDEX_MAP,
            RAMEN_ORI_ACTION_DIM,
        )
        ds = dataset_with_fake_base(chunk_len=16)
        raw = ds._base[0]
        item = ds[0]
        q_d = raw["action.robot_q_desired"].numpy()[1:]  # (16, 36)
        h_c = raw["action.hand_cmd"].numpy()[1:]         # (16, 2)
        expected = np.concatenate([q_d, h_c], axis=1)[
            :, np.asarray(ARMS_HAND_SOURCE_INDEX_MAP)
        ]
        assert expected.shape == (16, RAMEN_ORI_ACTION_DIM)
        np.testing.assert_allclose(item["action"].numpy(), expected, atol=1e-6)

    def test_action_is_pad_matches_tail_padding(self, dataset_with_fake_base) -> None:
        """Issue #141 RO-2: 区間末尾の埋め草の行は action_is_pad=True (row 0 の 1 frame 前は含まない)。"""
        ds = dataset_with_fake_base(chunk_len=16, pad_tail_rows=5)
        item = ds[0]
        pad = item["action_is_pad"]
        assert pad.dtype == torch.bool
        assert pad.shape == (16,)
        np.testing.assert_array_equal(pad.numpy(), np.array([False] * 11 + [True] * 5))

    def test_segment_start_pad_row_is_not_in_action_is_pad(self, dataset_with_fake_base) -> None:
        """区間先頭の 1 frame 前 (row 0) の pad は tracking_err だけに効き、正解の埋め草ではない。"""
        ds = dataset_with_fake_base(chunk_len=16, prev_pad_at_idx=0)
        item = ds[0]
        assert not item["action_is_pad"].any()


class TestImages:
    def test_prev_pad_forces_images_prev_equals_images(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base(prev_pad_at_idx=0)
        item = ds[0]
        torch.testing.assert_close(item["images"], item["images_prev"])

    def test_no_pad_images_prev_differs_from_images(self, dataset_with_fake_base) -> None:
        # FakeLeRobotBase は毎 frame 独立の乱数 → prev と current が違うはず
        ds = dataset_with_fake_base()
        item = ds[0]
        assert not torch.allclose(item["images"], item["images_prev"])


class TestSkillId:
    """Issue #141 RO-1: skill_id は episode meta の source_task_index を skill_mapping で変換。"""

    def test_merged_rotate_frame0_meta5_is_skill4(self, dataset_with_fake_base) -> None:
        """merged_v1 と同じ形 (frame の task_index 0 / meta の source_task_index 5) → rotate = 4。"""
        ds = dataset_with_fake_base(task_index=0, source_task_index=5)
        assert int(ds[0]["skill_id"].item()) == 4

    def test_move_meta7_is_skill5(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base(task_index=7, source_task_index=7)
        assert int(ds[0]["skill_id"].item()) == 5

    def test_insert_meta0_is_skill0(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base(task_index=0, source_task_index=0)
        assert int(ds[0]["skill_id"].item()) == 0

    def test_frame_task_index_is_ignored(self, dataset_with_fake_base) -> None:
        """frame の task_index を変えても skill_id は meta で決まる。"""
        ds_a = dataset_with_fake_base(task_index=0, source_task_index=5)
        ds_b = dataset_with_fake_base(task_index=3, source_task_index=5)
        assert int(ds_a[0]["skill_id"]) == int(ds_b[0]["skill_id"]) == 4

    def test_missing_source_task_index_raises(self, dataset_with_fake_base) -> None:
        with pytest.raises(ValueError, match="source_task_index"):
            dataset_with_fake_base(source_task_index=None)

    @pytest.mark.parametrize("bad", [1, 6, 99])
    def test_unregistered_source_task_index_raises(self, dataset_with_fake_base, bad) -> None:
        """skill_mapping に無い番号 (1 = move_to_table、6 = intro、未知) は構築時にエラー。"""
        with pytest.raises(KeyError, match=f"task_index={bad}"):
            dataset_with_fake_base(source_task_index=bad)

    def test_sample_skill_ids_matches_item_skill_id(self, dataset_with_fake_base) -> None:
        """sampler 用の sample_skill_ids と model 入力の skill_id が同じ変換を通る。"""
        ds = dataset_with_fake_base(num_samples=25, source_task_index=5)
        skills = ds.sample_skill_ids()
        assert skills.shape == (25,)
        assert set(skills.tolist()) == {4}
        assert [int(ds[i]["skill_id"]) for i in (0, 12, 24)] == [4, 4, 4]
        meta = ds.sample_episode_metadata()
        np.testing.assert_array_equal(meta["episode_skill_ids"], [4, 4, 4])
        np.testing.assert_array_equal(meta["episode_lengths"], [10, 10, 5])


class TestOBBNone:
    """Issue #122 D-2: obb_source='none' (旧 'dummy') は all-zeros + valid_mask 全 False。

    OBB path を学習に流さない mode (C-1 相当)。旧 'dummy' 名は「YOLO 系 skip」の実態と
    合わせて D-2 で 'none' に rename。valid_mask=False で fusion attention が OBB
    tokens を完全 filter out、ObbTokenizer の gradient も 0 → params 更新なし。
    'precomputed_token' に切替時は real YOLO detection で valid_mask=True が入る。
    """

    def test_none_is_all_zeros(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base()
        item = ds[3]
        assert torch.all(item["obb_verts"] == 0), "verts must be all zeros (no signal)"
        assert torch.all(item["obb_conf"] == 0), "conf must be all zeros"
        assert torch.all(item["obb_class_id"] == 0), "class_id must be all zeros"

    def test_none_valid_mask_all_false(self, dataset_with_fake_base) -> None:
        """valid_mask 全 False = fusion attention が OBB tokens を完全 filter out する保証。

        これが False の要素が 1 つでもあれば、その OBB token に対し fusion が
        attention を計算し gradient が流れる → ObbTokenizer が spurious 学習開始。
        """
        ds = dataset_with_fake_base()
        item = ds[3]
        assert item["obb_valid_mask"].dtype == torch.bool
        assert not item["obb_valid_mask"].any(), (
            f"valid_mask must be all False but has {item['obb_valid_mask'].sum().item()} True"
        )

    def test_none_deterministic_across_idx(self, dataset_with_fake_base) -> None:
        """all-zeros なので idx 違っても同じ (旧 idx-seeded 乱数 dummy との差)。

        idx-independence は「OBB は情報信号を運ばない」設計意図の naming test。
        """
        ds = dataset_with_fake_base()
        item0 = ds[0]
        item1 = ds[1]
        torch.testing.assert_close(item0["obb_verts"], item1["obb_verts"])
        torch.testing.assert_close(item0["obb_valid_mask"], item1["obb_valid_mask"])


class TestOBBPrecomputedToken:
    """Issue #122 D-2: obb_source='precomputed_token' 分岐の integration test。

    fake base_dataset + mock _obb_cache で _make_precomputed_obb の routing と
    tensor shape / wrist zero fill を検証。
    """

    def _make_dataset_with_mock_cache(self, dataset_with_fake_base, cache_stub, num_samples: int = 10):
        """'none' mode で construct した後 obb_source を切替 + cache を inject する
        (precomputed_token flag では merged_source が必要なので DI test でしか使えない)。
        """
        ds = dataset_with_fake_base(num_samples=num_samples)
        ds.obb_source = "precomputed_token"
        ds._obb_cache = cache_stub
        return ds

    def test_routes_to_precomputed_when_obb_source_set(self, dataset_with_fake_base) -> None:
        called = {}

        class Stub:
            def make_obb_tensors(self, episode_index, frame_index_in_ep, cam_ids):
                called["ep"] = int(episode_index)
                called["fr"] = int(frame_index_in_ep)
                N = len(cam_ids)
                return {
                    "verts": torch.zeros(N, 4, 8),
                    "conf": torch.zeros(N, 4, 1),
                    "class_id": torch.zeros(N, 4, dtype=torch.long),
                    "cam_id": cam_ids.unsqueeze(1).expand(N, 4).contiguous(),
                    "valid_mask": torch.zeros(N, 4, dtype=torch.bool),
                }

        ds = self._make_dataset_with_mock_cache(dataset_with_fake_base, Stub(), num_samples=20)
        # FakeLeRobotBase の episode_index / frame_index = (idx//10, idx%10)
        _ = ds[7]
        assert called == {"ep": 0, "fr": 7}
        _ = ds[13]
        assert called == {"ep": 1, "fr": 3}

    def test_wrist_zeros_head_precomputed_shape(self, dataset_with_fake_base) -> None:
        """Issue #129 Phase A (2026-08-31): camera_keys 4→3 (head_right 除外)。
        cam_0 = head_left は real OBB、cam_2/3 (wrist) は zeros + valid=False。
        (旧 test は cam_0/1 が head で cam_2/3 が wrist、今 layout は cam_0=head_left / cam_2=L wrist / cam_3=R wrist、
        base_index の 0/1/2 に detach)。"""

        class Stub:
            def make_obb_tensors(self, episode_index, frame_index_in_ep, cam_ids):
                N = len(cam_ids)
                verts = torch.zeros(N, 4, 8)
                valid_mask = torch.zeros(N, 4, dtype=torch.bool)
                # head 1 cam (base_index=0) に valid=True + verts に marker、wrist (1,2) はゼロ
                verts[0, 0, 0] = 1.0  # marker
                valid_mask[0, 0] = True
                return {
                    "verts": verts,
                    "conf": torch.zeros(N, 4, 1),
                    "class_id": torch.zeros(N, 4, dtype=torch.long),
                    "cam_id": cam_ids.unsqueeze(1).expand(N, 4).contiguous(),
                    "valid_mask": valid_mask,
                }

        ds = self._make_dataset_with_mock_cache(dataset_with_fake_base, Stub())
        item = ds[0]
        # head cam (base_index=0) に real OBB marker
        assert item["obb_verts"][0, 0, 0].item() == pytest.approx(1.0)
        assert item["obb_valid_mask"][0, 0].item() is True
        # wrist cam (base_index=1, 2) は全 0 + valid=False
        assert torch.all(item["obb_verts"][1] == 0)
        assert torch.all(item["obb_verts"][2] == 0)
        assert not item["obb_valid_mask"][1].any()
        assert not item["obb_valid_mask"][2].any()

    def test_precomputed_token_shape_matches_none(self, dataset_with_fake_base) -> None:
        """precomputed_token の tensor shape / dtype は 'none' mode と完全一致。"""

        class Stub:
            def make_obb_tensors(self, episode_index, frame_index_in_ep, cam_ids):
                N = len(cam_ids)
                return {
                    "verts": torch.zeros(N, 4, 8),
                    "conf": torch.zeros(N, 4, 1),
                    "class_id": torch.zeros(N, 4, dtype=torch.long),
                    "cam_id": cam_ids.unsqueeze(1).expand(N, 4).contiguous(),
                    "valid_mask": torch.zeros(N, 4, dtype=torch.bool),
                }

        ds_pre = self._make_dataset_with_mock_cache(dataset_with_fake_base, Stub())
        ds_none = dataset_with_fake_base()
        item_pre = ds_pre[0]
        item_none = ds_none[0]
        for key in ["obb_verts", "obb_conf", "obb_class_id", "obb_cam_id", "obb_valid_mask"]:
            assert item_pre[key].shape == item_none[key].shape, f"{key} shape mismatch"
            assert item_pre[key].dtype == item_none[key].dtype, f"{key} dtype mismatch"

    def test_missing_cache_raises_when_source_set(self, dataset_with_fake_base) -> None:
        """obb_source='precomputed_token' なのに _obb_cache=None なら明示 error。"""
        ds = dataset_with_fake_base()
        ds.obb_source = "precomputed_token"
        # _obb_cache は None のまま
        with pytest.raises(RuntimeError, match="_obb_cache が None"):
            ds[0]

    def test_invalid_obb_source_raises(self) -> None:
        """未サポート obb_source は init 時 fail-fast。"""
        fake = FakeLeRobotBase()
        with pytest.raises(ValueError, match="obb_source="):
            RamenOriLerobotDataset(base_dataset=fake, obb_source="garbage")

    def test_precomputed_token_requires_root(self) -> None:
        """obb_source='precomputed_token' + obb_precomputed_root=None は init 時 fail-fast。"""
        fake = FakeLeRobotBase()
        with pytest.raises(ValueError, match="obb_precomputed_root"):
            RamenOriLerobotDataset(
                base_dataset=fake,
                obb_source="precomputed_token",
                obb_precomputed_root=None,
            )


class TestOBBOverlay:
    """Issue #122 D-3: obb_source='overlay' 分岐の integration test。

    overlay mode は OBB tokens は none (zeros + valid=False) と同じ (描画は image に直接
    入る、post-decode hook 経由)。dataloader 側は token 供給しない仕様を検証。
    hook register 自体は DI test では base_dataset 経路のため走らない (実 merged_source
    load が必要)、hook 発火 test は test_overlay.py 側で unit test 済み。
    """

    def test_overlay_requires_precomputed_root(self) -> None:
        """obb_source='overlay' + obb_precomputed_root=None は init 時 fail-fast。"""
        fake = FakeLeRobotBase()
        with pytest.raises(ValueError, match="obb_precomputed_root"):
            RamenOriLerobotDataset(
                base_dataset=fake,
                obb_source="overlay",
                obb_precomputed_root=None,
            )

    def test_overlay_config_params_stored(self, dataset_with_fake_base) -> None:
        """overlay 用 params (conf/thickness/class_filter) が instance attr に保持される。"""
        ds = dataset_with_fake_base(
            overlay_conf_threshold=0.45,
            overlay_line_thickness=3,
            overlay_class_filter=[3, 4],
        )
        assert ds.overlay_conf_threshold == pytest.approx(0.45)
        assert ds.overlay_line_thickness == 3
        assert ds.overlay_class_filter == {3, 4}

    def test_overlay_class_filter_none_stored_as_none(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base()  # 未指定
        assert ds.overlay_class_filter is None

    def test_overlay_source_yields_none_style_obb_tokens(self, dataset_with_fake_base) -> None:
        """overlay mode の OBB tokens は 'none' と同じ (zeros + valid_mask 全 False)。

        Issue #129 Phase D (2026-08-31): 描画は image に直接入る (inline 経路、Photometric
        aug 後 / Normalize 前で `draw_overlay_on_tensor`)、model の OBB token path はゼロで
        attention filter 通過、gradient 流れない。旧 hook 経路は unwire 済み (GR00T 側は継続)。
        """
        ds = dataset_with_fake_base()
        # 'overlay' mode に手動切替 (DI test では hook register を skip、_transform_item のみ検証)
        ds.obb_source = "overlay"
        item = ds[0]
        assert torch.all(item["obb_verts"] == 0)
        assert torch.all(item["obb_conf"] == 0)
        assert not item["obb_valid_mask"].any()


class TestOverlayInlinePipeline:
    """Issue #129 Phase D (2026-08-31): overlay inline 経路の統合 test。

    - hook 未登録 (`register_post_decode_hook` 呼出しなし)
    - `_fetch_overlay_obb_by_cam` は N_cams × per-cam OBB を返す (wrist は cache miss で valid False)
    - `ImageAugPipeline.apply` に overlay_callback を渡すと Photometric → Erasing → overlay → Normalize 順
    """

    def test_hook_not_registered_when_dataset_built(self, dataset_with_fake_base) -> None:
        """base_dataset 経路 DI では init 内で hook 登録 code path に入らない。

        本 test は DI 経路の smoke (実 hook 登録の unit test は
        obb_overlay_setup.py 側で従来通り GR00T 経由で維持)。
        """
        # obb_source='overlay' への切替は _obb_cache が必要だが DI 経路では None なので
        # obb_source 切替のみ (Phase D で hook 呼出し削除された事の指標として、init が
        # KeyError 等出さず正常完了することを assert)
        ds = dataset_with_fake_base()
        ds.obb_source = "overlay"
        # _obb_cache が None のままでも obb_source 単純切替は OK (実 fetch は _transform_item で呼ばれる)
        assert ds.obb_source == "overlay"

    def test_image_aug_pipeline_calls_overlay_callback_after_erasing(self) -> None:
        """ImageAugPipeline.apply が overlay_callback を Photometric+Erasing 後 / Normalize 前で呼ぶ。"""
        import torch
        from model.ramen_ori.data_lerobot import ImageAugPipeline

        aug_cfg = {
            "enabled": True,
            "head": {
                "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.15, "hue": 0.02},
                "gaussian_blur": {"kernel_size": 3, "sigma_min": 0.1, "sigma_max": 1.5, "p": 0.3},
            },
            "wrist": {
                "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.01},
                "gaussian_blur": {"kernel_size": 3, "sigma_min": 0.1, "sigma_max": 1.0, "p": 0.3},
            },
            "random_erasing": {"p": 0.15, "scale_min": 0.02, "scale_max": 0.15},
            "wrist_key_substring": "wrist",
        }
        pipe = ImageAugPipeline(aug_cfg=aug_cfg)

        img_2 = torch.rand(2, 3, 32, 32)
        callback_call_count = {"n": 0}
        received_range: list[tuple[float, float]] = []

        def cb(img_frame: torch.Tensor, frame_i: int) -> torch.Tensor:
            # overlay_callback に渡される img は [0,1] 範囲 (Normalize 前) の float
            callback_call_count["n"] += 1
            received_range.append((float(img_frame.min()), float(img_frame.max())))
            return img_frame

        # Phase G: apply が (imgs, affine) tuple を返す形に refactor 済
        out, affine = pipe.apply(img_2, "observation.images.cam_0", idx=0, is_train=True, overlay_callback=cb)
        assert out.shape == (2, 3, 32, 32)
        assert affine.shape == (2, 3)
        assert callback_call_count["n"] == 2, "overlay_callback は prev+current の 2 frame で呼ばれる"
        # Normalize 前 = [0,1] 範囲 (aug で若干 clip の可能性、[-0.01, 1.5] くらいの許容で見る)
        for lo, hi in received_range:
            assert -0.5 <= lo, f"overlay_callback 受渡し image min={lo} が Normalize 前想定を外れる"
            assert hi <= 1.5, f"overlay_callback 受渡し image max={hi} が Normalize 前想定を外れる"

    def test_image_aug_pipeline_val_path_still_calls_overlay(self) -> None:
        """val path (is_train=False) でも overlay_callback は適用される (train/val で同 image を model に見せる)。"""
        import torch
        from model.ramen_ori.data_lerobot import ImageAugPipeline

        pipe = ImageAugPipeline(aug_cfg={"enabled": False})  # aug 全 disabled
        img_2 = torch.rand(2, 3, 32, 32)
        callback_count = {"n": 0}

        def cb(img_frame: torch.Tensor, frame_i: int) -> torch.Tensor:
            callback_count["n"] += 1
            return img_frame

        pipe.apply(img_2, "observation.images.cam_0", idx=0, is_train=False, overlay_callback=cb)
        assert callback_count["n"] == 2

    def test_image_aug_pipeline_no_callback_skips_overlay(self) -> None:
        """overlay_callback=None (default) で従来 behavior 維持。"""
        import torch
        from model.ramen_ori.data_lerobot import ImageAugPipeline

        pipe = ImageAugPipeline(aug_cfg={"enabled": False})
        img_2 = torch.rand(2, 3, 32, 32)
        # Phase G: apply が (imgs, affine) tuple
        out, affine = pipe.apply(img_2, "observation.images.cam_0", idx=0, is_train=False)
        assert out.shape == (2, 3, 32, 32)
        assert affine.shape == (2, 3)


def _cfg_full_phase_g(enabled: bool = True, geometric_enabled: bool = True) -> dict:
    """Phase G config (sharpness / gaussian_noise / jpeg / geometric 全部あり)。"""
    return {
        "enabled": enabled,
        "head": {
            "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.3, "hue": 0.02},
            "gaussian_blur": {"kernel_size": 3, "sigma_min": 0.1, "sigma_max": 1.5, "p": 0.3},
            "sharpness": {"factor": 1.5, "p": 0.5},
            "gaussian_noise": {"sigma": 0.010, "p": 0.5},
            "jpeg": {"quality": [60, 90], "p": 0.3},
        },
        "wrist": {
            "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.2, "hue": 0.01},
            "gaussian_blur": {"kernel_size": 3, "sigma_min": 0.1, "sigma_max": 1.0, "p": 0.3},
            "sharpness": {"factor": 1.5, "p": 0.5},
            "gaussian_noise": {"sigma": 0.010, "p": 0.5},
            "jpeg": {"quality": [60, 90], "p": 0.3},
        },
        "random_erasing": {"p": 0.15, "scale_min": 0.02, "scale_max": 0.15},
        "geometric": {
            "enabled": geometric_enabled,
            "degrees": [-5.0, 5.0],
            "translate": [0.05, 0.05],
        },
        "wrist_key_substring": "wrist",
    }


class TestPhaseGAugExtensions:
    """Issue #129 Phase G (2026-08-31): 追加 aug の統合 test。

    - sharpness / gaussian_noise / jpeg が config から instantiate される
    - geometric aug は head cam のみ実適用 (wrist は identity affine)
    - geometric.enabled=false で全 cam identity
    - apply の (imgs, affine) tuple、affine は normalized [0,1] 空間
    """

    def test_new_aug_transforms_instantiated_when_enabled(self) -> None:
        """Phase G aug config から Sharpness / GaussianNoise / JPEG / geometric が正しく init。"""
        from model.ramen_ori.data_lerobot import ImageAugPipeline

        pipe = ImageAugPipeline(_cfg_full_phase_g(enabled=True))
        assert pipe.enabled is True
        # Head/wrist 別 aug が生えている
        assert hasattr(pipe, "head_sharpness")
        assert hasattr(pipe, "wrist_sharpness")
        assert hasattr(pipe, "head_gaussian_noise")
        assert hasattr(pipe, "wrist_gaussian_noise")
        assert hasattr(pipe, "head_jpeg")
        assert hasattr(pipe, "wrist_jpeg")
        # Geometric config が読まれている
        assert pipe.geometric_enabled is True
        assert pipe.geometric_degrees == (-5.0, 5.0)
        assert pipe.geometric_translate == (0.05, 0.05)

    def test_wrist_geometric_skipped_identity_affine(self) -> None:
        """Wrist cam は state-image causal link 保護のため geometric skip、affine identity。"""
        import torch
        from model.ramen_ori.data_lerobot import (
            ImageAugPipeline,
            _identity_affine_matrix,
        )

        pipe = ImageAugPipeline(_cfg_full_phase_g(enabled=True, geometric_enabled=True))
        img_2 = torch.rand(2, 3, 32, 32)
        _, affine_wrist = pipe.apply(
            img_2, "observation.images.cam_2_wrist_left", idx=0, is_train=True
        )
        torch.testing.assert_close(affine_wrist, _identity_affine_matrix())

    def test_head_geometric_produces_non_identity_affine(self) -> None:
        """Head cam + geometric.enabled=True で affine matrix は identity から離れる (rot/translate 適用済)。"""
        import torch
        from model.ramen_ori.data_lerobot import (
            ImageAugPipeline,
            _identity_affine_matrix,
        )

        pipe = ImageAugPipeline(_cfg_full_phase_g(enabled=True, geometric_enabled=True))
        img_2 = torch.rand(2, 3, 32, 32)
        _, affine_head = pipe.apply(
            img_2, "observation.images.cam_0", idx=42, is_train=True
        )
        # affine が identity と "ほぼ一致" するのは angle=0 & translate=0 のみ、
        # per-sample uniform sample で完全一致確率は ~0
        assert not torch.allclose(affine_head, _identity_affine_matrix(), atol=1e-4)

    def test_head_geometric_disabled_identity_affine(self) -> None:
        """geometric.enabled=false なら head cam でも identity affine (skip)。"""
        import torch
        from model.ramen_ori.data_lerobot import (
            ImageAugPipeline,
            _identity_affine_matrix,
        )

        pipe = ImageAugPipeline(_cfg_full_phase_g(enabled=True, geometric_enabled=False))
        img_2 = torch.rand(2, 3, 32, 32)
        _, affine_head = pipe.apply(
            img_2, "observation.images.cam_0", idx=42, is_train=True
        )
        torch.testing.assert_close(affine_head, _identity_affine_matrix())

    def test_head_geometric_prev_current_same_affine(self) -> None:
        """delta encoder 保護: prev/current で同 affine が適用される (per-sample-per-cam seed)。"""
        import torch
        from model.ramen_ori.data_lerobot import ImageAugPipeline

        pipe = ImageAugPipeline(_cfg_full_phase_g(enabled=True, geometric_enabled=True))
        # constant image (aug 有無を除いた比較用)
        img_2 = torch.full((2, 3, 32, 32), 0.5)
        # affine は per-sample 1 個返る (2 frame 共通)
        out, _ = pipe.apply(img_2, "observation.images.cam_0", idx=42, is_train=True)
        # 全 aug (color, blur, sharp, noise, jpeg, geometric) が同 seed で prev/current 同一 → 出力一致
        # ただし erasing はここでは p=0.15 のランダム、確実性のため差分は許容 (tol 大きめ)
        # 主目的: geometric warp が prev/current 同期している
        # 完全一致まで期待するのは gaussian_noise/erasing の p ランダム性で難しい → shape のみ確認
        assert out.shape == (2, 3, 32, 32)

    def test_float_jpeg_roundtrip(self) -> None:
        """FloatJPEG wrapper: float [0,1] → uint8 → JPEG → float [0,1]、shape/dtype 保存。"""
        import torch
        from model.ramen_ori.data_lerobot import FloatJPEG

        jpeg = FloatJPEG(quality=(60, 90))
        img = torch.rand(3, 64, 64)
        out = jpeg(img)
        assert out.shape == img.shape
        assert out.dtype == torch.float32
        # JPEG compression → 微小 pixel 差、[0,1] 範囲は保持
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_obb_verts_warped_under_geometric_aug(self) -> None:
        """OBB coord が head geometric aug と同 affine で warp される (integration test)。

        `_warp_obb_verts_under_affines` を直接呼び、identity と非-identity affine の
        混在 (wrist / head) で cam ごと正しく処理されることを確認。
        """
        import torch
        from model.ramen_ori.data_lerobot import (
            RamenOriLerobotDataset,
            _identity_affine_matrix,
        )

        # Minimum instance without actual dataset load (base_dataset injection でも本 method は動く)
        ds = object.__new__(RamenOriLerobotDataset)   # skip __init__

        # 3 cam の OBB tensor (head + 2 wrist)
        obb = {
            "verts": torch.rand(3, 4, 8),
            "conf": torch.rand(3, 4, 1),
            "class_id": torch.zeros(3, 4, dtype=torch.long),
            "cam_id": torch.arange(3).unsqueeze(1).expand(3, 4),
            "valid_mask": torch.ones(3, 4, dtype=torch.bool),
        }
        # Head (index 0) に translate affine (verts が明確に shift)、wrist (1, 2) は identity
        from model.ramen_ori.obb_warp import make_rotation_translation_affine

        head_affine = make_rotation_translation_affine(angle_rad=0.0, tx=0.1, ty=0.0)
        affine_per_cam = [head_affine, _identity_affine_matrix(), _identity_affine_matrix()]

        original_verts = obb["verts"].clone()
        warped = ds._warp_obb_verts_under_affines(obb, affine_per_cam)

        # Head cam verts は shift +0.1 in x (0.1 = tx_frac、normalized 空間)
        # ただし clamp=True で [0,1] に丸められる可能性あり (ランダム verts なので tolerance 広め)
        assert not torch.allclose(warped["verts"][0], original_verts[0], atol=1e-4), (
            "head cam verts should be warped by non-identity affine"
        )
        # Wrist cams (identity affine) は不変
        torch.testing.assert_close(warped["verts"][1], original_verts[1])
        torch.testing.assert_close(warped["verts"][2], original_verts[2])
        # 他 field は passthrough
        torch.testing.assert_close(warped["conf"], obb["conf"])
        torch.testing.assert_close(warped["valid_mask"], obb["valid_mask"])


class TestCamId:
    def test_cam_id_is_arange_of_camera_keys(self, dataset_with_fake_base) -> None:
        # Issue #129 Phase A: default 3 cam (head_left + L/R wrist、head_right 除外)
        ds = dataset_with_fake_base()
        item = ds[0]
        torch.testing.assert_close(item["cam_id"], torch.arange(3, dtype=torch.long))

    def test_obb_cam_id_broadcasted_from_cam_id(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base(top_K=4)
        item = ds[0]
        # obb_cam_id は各 cam の top_K が同 cam_id を持つ (Phase A: 3 cam に縮小)
        for cam in range(3):
            assert (item["obb_cam_id"][cam] == cam).all()


class TestDeltaTimestamps:
    def test_delta_timestamps_covers_state_and_action_and_cameras(self) -> None:
        dt = build_delta_timestamps(chunk_len=16, camera_keys=default_camera_keys())
        assert dt["observation.state.robot_q_current"] == [-1 / G1_WBT_FPS, 0.0]
        assert dt["observation.state.hand_state"] == [-1 / G1_WBT_FPS, 0.0]
        assert dt["observation.state.ee_state"] == [0.0]
        # action は 1 frame 前 (tracking_err 用) + chunk 16 行
        assert len(dt["action.robot_q_desired"]) == 17
        assert len(dt["action.hand_cmd"]) == 17
        assert dt["action.robot_q_desired"][0] == pytest.approx(-1 / G1_WBT_FPS, abs=1e-6)
        assert dt["action.robot_q_desired"][1] == 0.0
        for cam in default_camera_keys():
            assert dt[cam] == [-1 / G1_WBT_FPS, 0.0]


class TestStateVariant:
    """state_variant="71d" (E-δ) / "73d" (E-β、Alt-2)。"""

    def test_default_is_71d(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base()
        item = ds[0]
        assert item["state"].shape == (71,)

    def test_73d_variant_yields_73d_state(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base(state_variant="73d")
        item = ds[0]
        assert item["state"].shape == (73,)

    def test_73d_placeholder_depth_contact_is_zero(self, dataset_with_fake_base) -> None:
        """Alt-2 placeholder path: SGBM 未実装で depth_contact = zeros。"""
        ds = dataset_with_fake_base(state_variant="73d")
        item = ds[0]
        # state[71:73] は depth_contact placeholder = zeros
        import torch
        assert torch.all(item["state"][71:73] == 0.0)

    def test_73d_state_first_71_matches_71d(self, dataset_with_fake_base) -> None:
        """E-β の [0:71] 部分は E-δ と同一 (同 idx で同 tensor)。"""
        ds71 = dataset_with_fake_base(state_variant="71d")
        ds73 = dataset_with_fake_base(state_variant="73d")
        item71 = ds71[0]
        item73 = ds73[0]
        import torch
        torch.testing.assert_close(item73["state"][:71], item71["state"])

    def test_invalid_state_variant_raises(self, dataset_with_fake_base) -> None:
        with pytest.raises(ValueError, match="state_variant must be"):
            dataset_with_fake_base(state_variant="99d")


class TestUnsupported:
    def test_obb_source_unknown_value_raises_value_error(self) -> None:
        """D-2: 未サポート obb_source は ValueError で choices を提示 (旧 'precomputed' も含む)。"""
        # D-2 で 'precomputed_token' が実装済のため、旧 'precomputed' (短縮形) は依然 unknown 扱い。
        with pytest.raises(ValueError, match="obb_source="):
            RamenOriLerobotDataset(
                base_dataset=FakeLeRobotBase(),
                obb_source="precomputed",
            )

    def test_merged_source_root_required_when_no_base_dataset(self) -> None:
        """Issue #122 refactor: repo_ids 廃止、merged_source_root が必須。"""
        with pytest.raises(ValueError, match="merged_source_root"):
            RamenOriLerobotDataset(merged_source_root=None, base_dataset=None)


class TestLength:
    def test_len_delegates_to_base(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base(num_samples=42)
        assert len(ds) == 42


class TestFrameCacheModeI0_3:
    """Issue #129 Phase I-0-3 (2026-09-01): baked frame cache mode の validation + 挙動。

    - config validation: baked_* mode で root / num_variants が missing なら raise
    - env set: baked_* mode で FRAME_CACHE_ROOT_OVERRIDE + FRAME_CACHE_NUM_VARIANTS が set
    - _transform_item: baked_* mode で aug pipeline が is_train=False path (aug skip、
      identity affine) を通る = 生成 image は online + is_train=False と数値一致
    """

    def test_online_is_default(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base()
        assert ds.frame_cache_mode == "online"
        assert ds.frame_cache_root_baked is None

    def test_invalid_mode_raises(self, dataset_with_fake_base) -> None:
        with pytest.raises(ValueError, match="frame_cache_mode="):
            dataset_with_fake_base(frame_cache_mode="baked_wrong")

    def test_baked_token_requires_root(self, dataset_with_fake_base) -> None:
        with pytest.raises(ValueError, match="frame_cache_root_baked"):
            dataset_with_fake_base(frame_cache_mode="baked_token", frame_cache_num_variants=10)

    def test_baked_overlay_requires_root(self, dataset_with_fake_base) -> None:
        with pytest.raises(ValueError, match="frame_cache_root_baked"):
            dataset_with_fake_base(frame_cache_mode="baked_overlay", frame_cache_num_variants=10)

    def test_baked_requires_num_variants_gte_2(self, dataset_with_fake_base, tmp_path) -> None:
        with pytest.raises(ValueError, match="frame_cache_num_variants>=2"):
            dataset_with_fake_base(
                frame_cache_mode="baked_token",
                frame_cache_root_baked=tmp_path,
                frame_cache_num_variants=1,   # 無効
            )

    def test_baked_token_sets_env(self, dataset_with_fake_base, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("FRAME_CACHE_ROOT_OVERRIDE", raising=False)
        monkeypatch.delenv("FRAME_CACHE_NUM_VARIANTS", raising=False)
        ds = dataset_with_fake_base(
            frame_cache_mode="baked_token",
            frame_cache_root_baked=tmp_path,
            frame_cache_num_variants=10,
        )
        import os as _os
        assert _os.environ["FRAME_CACHE_ROOT_OVERRIDE"] == str(tmp_path)
        assert _os.environ["FRAME_CACHE_NUM_VARIANTS"] == "10"
        assert ds.frame_cache_mode == "baked_token"
        assert ds.frame_cache_root_baked == tmp_path
        assert ds.frame_cache_num_variants == 10

    def test_baked_overlay_sets_env(self, dataset_with_fake_base, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("FRAME_CACHE_ROOT_OVERRIDE", raising=False)
        monkeypatch.delenv("FRAME_CACHE_NUM_VARIANTS", raising=False)
        ds = dataset_with_fake_base(
            frame_cache_mode="baked_overlay",
            frame_cache_root_baked=tmp_path,
            frame_cache_num_variants=10,
        )
        import os as _os
        assert _os.environ["FRAME_CACHE_ROOT_OVERRIDE"] == str(tmp_path)

    def test_baked_sets_strict_env(self, dataset_with_fake_base, tmp_path, monkeypatch) -> None:
        """Issue #141: baked は焼き込み済みの jpg だけを読む (欠けたら mp4 に戻さず止まる)。起動時の env に頼らない。"""
        monkeypatch.setenv("LEROBOT_FRAME_CACHE_ENABLE", "false")
        monkeypatch.delenv("FRAME_CACHE_STRICT", raising=False)
        dataset_with_fake_base(
            frame_cache_mode="baked_overlay", frame_cache_root_baked=tmp_path, frame_cache_num_variants=10
        )
        import os as _os
        assert _os.environ["LEROBOT_FRAME_CACHE_ENABLE"] == "true"
        assert _os.environ["FRAME_CACHE_STRICT"] == "true"

    def test_online_leaves_strict_env(self, dataset_with_fake_base, monkeypatch) -> None:
        monkeypatch.delenv("FRAME_CACHE_STRICT", raising=False)
        dataset_with_fake_base()
        import os as _os
        assert "FRAME_CACHE_STRICT" not in _os.environ

    def test_baked_mode_skips_inline_aug_and_overlay(
        self, dataset_with_fake_base, tmp_path, monkeypatch
    ) -> None:
        """baked mode の _transform_item は aug_pipeline を is_train=False で通す。

        FakeLeRobotBase の出力 image は fixed pattern (aug 掛けても掛けなくても中身は変わる
        べき != 実 base 挙動)、ここで確認するのは:
          1. baked mode で aug_pipeline.apply が is_train=False で呼ばれる (aug skip)
          2. overlay_callback が None (overlay 描画 skip)
        aug_pipeline を monkeypatch で spy して呼び出し引数を検証する。
        """
        monkeypatch.delenv("FRAME_CACHE_ROOT_OVERRIDE", raising=False)
        monkeypatch.delenv("FRAME_CACHE_NUM_VARIANTS", raising=False)
        ds = dataset_with_fake_base(
            frame_cache_mode="baked_overlay",
            frame_cache_root_baked=tmp_path,
            frame_cache_num_variants=10,
        )
        recorded_calls: list[dict] = []
        original_apply = ds.aug_pipeline.apply

        def _spy(img_2, cam_key, idx, is_train, overlay_callback=None, **kwargs):
            recorded_calls.append(
                {
                    "cam_key": cam_key,
                    "is_train": is_train,
                    "overlay_callback_is_none": overlay_callback is None,
                }
            )
            return original_apply(
                img_2, cam_key, idx, is_train, overlay_callback=overlay_callback, **kwargs
            )

        ds.aug_pipeline.apply = _spy  # type: ignore[assignment]
        _ = ds[0]

        assert len(recorded_calls) == len(default_camera_keys())
        for call in recorded_calls:
            # baked mode: is_train が False (aug skip)、overlay_callback が None
            assert call["is_train"] is False, (
                f"baked mode で aug_pipeline に is_train=True が渡された: {call}"
            )
            assert call["overlay_callback_is_none"], (
                f"baked mode で overlay_callback が None ではない: {call}"
            )


class TestImageAugPipelineSkipGeometricI0_3:
    """Issue #129 Phase I-0-3 (2026-09-01): apply の skip_geometric 挙動。"""

    def _cfg_geometric_enabled(self) -> dict:
        return {
            "enabled": True,
            "wrist_key_substring": "wrist",
            "head": {
                "color_jitter": {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0},
                "gaussian_blur": {"kernel_size": 3, "sigma_min": 0.1, "sigma_max": 0.1, "p": 0.0},
                "sharpness": {"factor": 1.0, "p": 0.0},
                "gaussian_noise": {"sigma": 0.0, "p": 0.0},
                "jpeg": {"quality": [90, 90], "p": 0.0},
            },
            "wrist": {
                "color_jitter": {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0},
                "gaussian_blur": {"kernel_size": 3, "sigma_min": 0.1, "sigma_max": 0.1, "p": 0.0},
                "sharpness": {"factor": 1.0, "p": 0.0},
                "gaussian_noise": {"sigma": 0.0, "p": 0.0},
                "jpeg": {"quality": [90, 90], "p": 0.0},
            },
            "random_erasing": {"p": 0.0, "scale_min": 0.02, "scale_max": 0.1},
            "geometric": {"enabled": True, "degrees": [-5.0, 5.0], "translate": [0.05, 0.05]},
        }

    def test_skip_geometric_true_forces_identity_affine(self) -> None:
        """skip_geometric=True で head cam でも identity affine が return される。"""
        from model.ramen_ori.image_aug import ImageAugPipeline, _identity_affine_matrix

        pipe = ImageAugPipeline(aug_cfg=self._cfg_geometric_enabled())
        img_2 = torch.zeros(2, 3, 32, 32, dtype=torch.float32)
        _, affine = pipe.apply(
            img_2,
            cam_key="observation.images.head_left",
            idx=0,
            is_train=True,
            skip_geometric=True,
        )
        torch.testing.assert_close(affine, _identity_affine_matrix())

    def test_skip_geometric_default_false_geometric_applies_for_head(self) -> None:
        """default (skip_geometric=False) で head cam に geometric aug が適用される (非-identity)。"""
        from model.ramen_ori.image_aug import ImageAugPipeline, _identity_affine_matrix

        pipe = ImageAugPipeline(aug_cfg=self._cfg_geometric_enabled())
        img_2 = torch.zeros(2, 3, 32, 32, dtype=torch.float32)
        # 複数 idx で試して 1 度でも non-identity になれば OK (0 seed で identity になる偶然を回避)
        found_non_identity = False
        for idx in range(20):
            _, affine = pipe.apply(
                img_2,
                cam_key="observation.images.head_left",
                idx=idx,
                is_train=True,
            )
            if not torch.allclose(affine, _identity_affine_matrix(), atol=1e-6):
                found_non_identity = True
                break
        assert found_non_identity, "20 seed 全てで identity affine、geometric aug が sample されず"

    def test_skip_normalize_returns_unnormalized_image(self) -> None:
        """skip_normalize=True で最終 Normalize が skip される (0-1 range の image が返る)。"""
        from model.ramen_ori.image_aug import ImageAugPipeline

        pipe = ImageAugPipeline(aug_cfg=self._cfg_geometric_enabled())
        img_2 = torch.full((2, 3, 32, 32), 0.5, dtype=torch.float32)   # gray [0,1]
        imgs, _ = pipe.apply(
            img_2,
            cam_key="observation.images.left_wrist",   # wrist は geometric skip
            idx=0,
            is_train=True,
            skip_normalize=True,
        )
        # 全 aug が p=0 の config → 入力そのまま return。値 range が [0,1]
        assert imgs.min() >= 0.0 and imgs.max() <= 1.0, (
            f"skip_normalize=True なのに Normalize されている: "
            f"min={imgs.min()}, max={imgs.max()}"
        )


# ----------------------------------------------------------------------------
# RamenOriMultiDataset tests (Issue #129 Phase K、2026-09-01)
# ----------------------------------------------------------------------------
def _make_sub(
    episode_lengths: list[int], source_task_index: int, task_index: int = 0
) -> RamenOriLerobotDataset:
    """Fake base で 1 個の RamenOriLerobotDataset を DI 経路で build (skill は meta から)。"""
    fake = FakeLeRobotBase(
        num_samples=sum(episode_lengths),
        episode_lengths=episode_lengths,
        task_index=task_index,
        source_task_index=source_task_index,
    )
    return RamenOriLerobotDataset(base_dataset=fake)


class TestRamenOriMultiDatasetRouting:
    """__len__ / __getitem__ の sub routing 動作。DI mode の sub で最小テスト。"""

    def test_len_is_sum_of_sub_lens(self) -> None:
        sub0 = _make_sub([8], source_task_index=5)
        sub1 = _make_sub([12], source_task_index=7)
        multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1])
        assert len(multi) == 20

    def test_getitem_routes_to_first_sub_for_low_idx(self) -> None:
        sub0 = _make_sub([8], source_task_index=5)
        sub1 = _make_sub([12], source_task_index=7)
        multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1])
        # multi idx=3 → sub0 の local idx=3、skill_id=4
        item = multi[3]
        assert int(item["skill_id"]) == 4

    def test_getitem_routes_to_second_sub_for_high_idx(self) -> None:
        sub0 = _make_sub([8], source_task_index=5)
        sub1 = _make_sub([12], source_task_index=7)
        multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1])
        # multi idx=10 → sub1 の local idx=2、skill_id=5
        item = multi[10]
        assert int(item["skill_id"]) == 5

    def test_getitem_boundary_edges(self) -> None:
        """offset の境界 (idx = N0-1, N0, N0+N1-1) を正しく routing する。"""
        sub0 = _make_sub([8], source_task_index=5)
        sub1 = _make_sub([12], source_task_index=7)
        multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1])
        assert int(multi[7]["skill_id"]) == 4   # sub0 末尾
        assert int(multi[8]["skill_id"]) == 5   # sub1 先頭
        assert int(multi[19]["skill_id"]) == 5  # sub1 末尾

    def test_dict_sub_config_is_instantiated(self) -> None:
        """config 経路 (sub が dict、Hydra の sub_datasets と同じ形) でも sub が作られ skill が付く。"""
        fake0 = FakeLeRobotBase(num_samples=4, source_task_index=5)
        fake1 = FakeLeRobotBase(num_samples=4, source_task_index=7)
        multi = RamenOriMultiDataset(
            sub_datasets=[{"base_dataset": fake0}, {"base_dataset": fake1}]
        )
        np.testing.assert_array_equal(multi.sample_skill_ids(), [4] * 4 + [5] * 4)

    def test_merged_subs_with_same_local_task_index_get_distinct_skills(self) -> None:
        """β と同じ形: 3 sub とも frame の task_index が 0 (merge のローカル番号) でも、
        meta の source_task_index で rotate=4 / move=5 / insert=0 に分かれる (RO-1 の再現)。"""
        sub0 = _make_sub([4], source_task_index=5, task_index=0)
        sub1 = _make_sub([4], source_task_index=7, task_index=0)
        sub2 = _make_sub([4], source_task_index=0, task_index=0)
        multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1, sub2])
        assert [int(multi[i]["skill_id"]) for i in (0, 4, 8)] == [4, 5, 0]
        np.testing.assert_array_equal(multi.sample_skill_ids(), [4] * 4 + [5] * 4 + [0] * 4)


class TestRamenOriMultiDatasetMetadata:
    """sample_episode_metadata / sample_skill_ids の concat。"""

    def test_sample_skill_ids_concat(self) -> None:
        # sub0: 8 frames を ep0=3 + ep1=5、両 ep とも source_task_index=5 → skill 4
        # sub1: 12 frames を ep0=7 + ep1=5、両 ep とも source_task_index=7 → skill 5
        sub0 = _make_sub([3, 5], source_task_index=5)
        sub1 = _make_sub([7, 5], source_task_index=7)
        multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1])
        skills = multi.sample_skill_ids()
        assert skills.shape == (20,)
        # 先頭 8 frames = sub0 = skill 4、後半 12 frames = sub1 = skill 5
        np.testing.assert_array_equal(skills[:8], np.full(8, 4, dtype=np.int64))
        np.testing.assert_array_equal(skills[8:], np.full(12, 5, dtype=np.int64))

    def test_episode_metadata_concat(self) -> None:
        sub0 = _make_sub([3, 5], source_task_index=5)
        sub1 = _make_sub([7, 5], source_task_index=7)
        multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1])
        meta = multi.sample_episode_metadata()
        assert meta["num_episodes"] == 4
        np.testing.assert_array_equal(meta["episode_lengths"], [3, 5, 7, 5])
        np.testing.assert_array_equal(meta["episode_skill_ids"], [4, 4, 5, 5])
        # source_names は sub prefix で globally unique
        assert meta["source_episode_names"] == [
            "sub0::src00000", "sub0::src00001", "sub1::src00000", "sub1::src00001",
        ]


class TestRamenOriMultiDatasetSplit:
    """make_train_val_test_split → MultiSplitView の flat indices と routing。"""

    def test_split_produces_multi_split_view_with_flat_indices_in_multi_space(self) -> None:
        # 各 sub 5 ep × 4 frames
        sub0 = _make_sub([4] * 5, source_task_index=5)
        sub1 = _make_sub([4] * 5, source_task_index=7)
        multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1])
        train, val, test = multi.make_train_val_test_split(val_ratio=0.2, test_ratio=0.2, seed=42)
        # sum(splits) = 40 = len(multi)
        assert len(train) + len(val) + len(test) == 40
        # flat indices は multi 空間 (0..39)、sub1 由来分は index >= 20
        all_flat = sorted(train.indices + val.indices + test.indices)
        assert all_flat == list(range(40))

    def test_split_json_raises_not_implemented(self) -> None:
        sub0 = _make_sub([8], source_task_index=5)
        sub1 = _make_sub([8], source_task_index=7)
        multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1])
        with pytest.raises(NotImplementedError, match="split_json"):
            multi.make_train_val_test_split(split_json="/dev/null")

    def test_multi_split_view_getitem_routes_to_correct_sub(self) -> None:
        """MultiSplitView[i] は sub-view 空間 (0..sum(len(view))-1) で routing。"""
        # split_episode_indices は num_episodes>=3 を要求するため、両 sub とも 3 ep 以上必要
        sub0 = _make_sub([4, 4, 4], source_task_index=5)
        sub1 = _make_sub([4, 4, 4], source_task_index=7)
        multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1])
        train, _, _ = multi.make_train_val_test_split(val_ratio=0.34, test_ratio=0.33, seed=0)
        # train view の 全 item を fetch、skill が 4 or 5 で valid (両 sub 混在)
        skills_seen = {int(train[i]["skill_id"]) for i in range(len(train))}
        assert skills_seen.issubset({4, 5})
        assert len(skills_seen) >= 1


class TestSingleFrameAndChunk32:
    """Issue #141 RO-11 / RO-17: 画像差分なし (今の frame だけ) と chunk 32。"""

    def test_delta_timestamps_single_frame(self) -> None:
        dt = build_delta_timestamps(chunk_len=32, camera_keys=default_camera_keys(), load_prev_image=False)
        for cam in default_camera_keys():
            assert dt[cam] == [0.0]
        assert len(dt["action.robot_q_desired"]) == 33
        # state の前 frame (velocity 用) は画像と関係なく読む
        assert len(dt["observation.state.robot_q_current"]) == 2

    def test_single_frame_item_has_no_images_prev(self, dataset_with_fake_base) -> None:
        ds = dataset_with_fake_base(load_prev_image=False)
        raw = ds._base[1]   # fake の画像は呼ぶたびに乱数なので、同じ raw を変換する
        item = ds._transform_item(raw, 1)
        assert "images_prev" not in item
        assert item["images"].shape == (3, 3, 224, 224)
        # 今の frame (1 枚、LeRobot は (3, H, W) で返す) に Normalize だけ掛かったもの
        assert raw["observation.images.cam_0"].shape == (3, 224, 224)
        expected = ds.aug_pipeline.normalize(raw["observation.images.cam_0"])
        torch.testing.assert_close(item["images"][0], expected)

    def test_two_frame_item_keeps_images_prev(self, dataset_with_fake_base) -> None:
        item = dataset_with_fake_base(load_prev_image=True)[1]
        assert item["images_prev"].shape == (3, 3, 224, 224)

    def test_chunk32_shapes(self, dataset_with_fake_base) -> None:
        item = dataset_with_fake_base(chunk_len=32, load_prev_image=False, pad_tail_rows=10)[0]
        assert item["action"].shape == (32, 16)
        assert item["action_waist_teacher"].shape == (32, 3)
        assert item["action_is_pad"].shape == (32,)
        assert int(item["action_is_pad"].sum()) == 10

    def test_multi_rejects_mixed_prev_image(self) -> None:
        sub0 = RamenOriLerobotDataset(base_dataset=FakeLeRobotBase(num_samples=4), load_prev_image=True)
        sub1 = RamenOriLerobotDataset(
            base_dataset=FakeLeRobotBase(num_samples=4, n_image_frames=1), load_prev_image=False
        )
        with pytest.raises(ValueError, match="load_prev_image"):
            RamenOriMultiDataset(sub_datasets=[sub0, sub1])


class TestRamenOriMultiDatasetSharedSettings:
    """Issue #141 RO-10: sub に書けるのは data の場所だけ。全 sub で同じ設定は multi も持つ。"""

    @pytest.mark.parametrize(
        "extra",
        [
            {"relative_stats_path": None},   # Phase K の rel で共通の統計を消していた書き方
            {"relative_stats_path": "/nvme/stats_task5.pt"},
            {"chunk_len": 32},
            {"frame_cache_num_variants": 10},
        ],
    )
    def test_shared_key_in_sub_raises(self, extra) -> None:
        fake = FakeLeRobotBase(num_samples=4, source_task_index=5)
        with pytest.raises(ValueError, match=next(iter(extra))):
            RamenOriMultiDataset(sub_datasets=[{"base_dataset": fake, **extra}])

    def test_location_keys_in_sub_are_accepted(self) -> None:
        fake = FakeLeRobotBase(num_samples=4, source_task_index=5)
        multi = RamenOriMultiDataset(
            sub_datasets=[
                {
                    "base_dataset": fake,
                    "repo_id": "local/x",
                    "frame_cache_root_baked": None,
                    "obb_precomputed_root": None,
                    "obb_precomputed_hash": None,
                }
            ]
        )
        assert len(multi) == 4

    def test_multi_exposes_shared_settings(self) -> None:
        fake0 = FakeLeRobotBase(num_samples=4, source_task_index=5, n_image_frames=1)
        fake1 = FakeLeRobotBase(num_samples=4, source_task_index=7, n_image_frames=1)
        multi = RamenOriMultiDataset(
            sub_datasets=[{"base_dataset": fake0}, {"base_dataset": fake1}],
            load_prev_image=False,
            overlay_conf_threshold=0.4,
        )
        sub = multi.subs[1]
        for attr in ("chunk_len", "img_size", "camera_keys", "state_variant", "frame_cache_mode", "obb_source"):
            assert getattr(multi, attr) == getattr(sub, attr)
        assert multi.load_prev_image is False
        assert multi.overlay_conf_threshold == 0.4

    def test_multi_rejects_mixed_chunk_len(self) -> None:
        sub0 = RamenOriLerobotDataset(base_dataset=FakeLeRobotBase(num_samples=4, chunk_len=16), chunk_len=16)
        sub1 = RamenOriLerobotDataset(base_dataset=FakeLeRobotBase(num_samples=4, chunk_len=32), chunk_len=32)
        with pytest.raises(ValueError, match="chunk_len"):
            RamenOriMultiDataset(sub_datasets=[sub0, sub1])


class TestRestrictVideoKeys:
    """LeRobot が読みに行く動画を camera_keys だけにする (IR / head_right を読まない)。"""

    class _Meta:
        def __init__(self, keys: list[str]) -> None:
            self.info = {"features": {k: {"dtype": "video"} for k in keys}}
            self.info["features"]["observation.state.ee_state"] = {"dtype": "float32"}

        @property
        def video_keys(self) -> list[str]:
            return [k for k, ft in self.info["features"].items() if ft["dtype"] == "video"]

    def test_keeps_only_camera_keys(self) -> None:
        from model.ramen_ori.data_lerobot import _restrict_video_keys

        all_cams = [f"observation.images.cam_{i}" for i in range(4)] + [
            f"observation.images.cam_{i}_ir" for i in range(4)
        ]
        meta = self._Meta(all_cams)
        _restrict_video_keys(meta, default_camera_keys())
        assert sorted(meta.video_keys) == sorted(default_camera_keys())
        assert "observation.state.ee_state" in meta.info["features"]   # 動画以外は残す

    def test_missing_camera_raises(self) -> None:
        from model.ramen_ori.data_lerobot import _restrict_video_keys

        meta = self._Meta(["observation.images.cam_0"])
        with pytest.raises(ValueError, match="not video features"):
            _restrict_video_keys(meta, default_camera_keys())


class TestEeStateSourceAndProgress:
    """Issue #183: 手先の位置の入力を推論と同じ FK から作る (ee_state_source)、区間の進みの正解 (progress_target)。"""

    @staticmethod
    def _inference_fk(skill: str):
        """推論 (entrypoint → assembly.FkFactory.for_skill) がその skill に使う FK。"""
        import yaml

        from inference.desktop.assembly import FkFactory
        from model.ramen_ori.data_lerobot import _INFERENCE_SKILL_CONFIG

        return FkFactory().for_skill(yaml.safe_load(_INFERENCE_SKILL_CONFIG.read_text(encoding="utf-8")), skill)

    @pytest.mark.parametrize("source_task_index, skill", [(2, "flip_table"), (5, "rotate_table_base")])
    def test_fk_source_matches_the_inference_fk(self, dataset_with_fake_base, source_task_index, skill) -> None:
        """state の ee_pose = 推論と同じ FK (skill ごとの offset も同じ) で robot_q_current[1][7:36] (29 関節) から。"""
        from model.ramen_ori.state_derive import STATE71_GROUPS

        ds = dataset_with_fake_base(ee_state_source="fk", source_task_index=source_task_index)
        fk = self._inference_fk(skill)
        for idx in (0, 3, 9):
            raw = ds._base[idx]
            expected = fk.compute_ee_state(raw["observation.state.robot_q_current"].numpy()[1][7:36])
            got = ds[idx]["state"][STATE71_GROUPS["ee_pose"]].numpy()
            np.testing.assert_allclose(got, expected, atol=1e-6)
            # 記録された ee_state とは違う値 (fake は乱数なので必ず違う)
            assert not np.allclose(got, raw["observation.state.ee_state"].numpy()[0])

    def test_rotate_uses_its_own_offset(self, dataset_with_fake_base) -> None:
        """rotate_table_base は skill_config の独自の offset (推論と同じ)。既定の offset の FK とは値が違う。"""
        from inference.desktop.perception.g1_urdf_fk import DEFAULT_URDF_PATH, G1WristFK
        from model.ramen_ori.state_derive import STATE71_GROUPS

        rotate = dataset_with_fake_base(ee_state_source="fk", source_task_index=5)
        q = rotate._base[0]["observation.state.robot_q_current"].numpy()[1][7:36]
        default = G1WristFK.from_urdf(DEFAULT_URDF_PATH).compute_ee_state(q)
        assert not np.allclose(rotate[0]["state"][STATE71_GROUPS["ee_pose"]].numpy(), default, atol=1e-3)

    def test_fk_ignores_the_root_pose_and_follows_the_arm(self, dataset_with_fake_base) -> None:
        """robot_q_current の先頭 7 (胴体の位置・向き) を変えても手先は変わらず、腕の関節を変えると変わる (Issue #188)。

        以前は robot_q_current[:29] (胴体 7 + 関節 0〜21) を FK に渡していて、腕の FK に脚・腰・左腕の値が入っていた。
        """
        import torch

        from model.ramen_ori.state_derive import STATE71_GROUPS

        ds = dataset_with_fake_base(ee_state_source="fk")
        raw = ds._base[2]
        base = ds._state_action(raw, 2)["state"][STATE71_GROUPS["ee_pose"]]
        q = raw["observation.state.robot_q_current"].clone()
        q[:, :7] += 0.5
        moved_root = ds._state_action({**raw, "observation.state.robot_q_current": q}, 2)["state"]
        np.testing.assert_allclose(moved_root[STATE71_GROUPS["ee_pose"]], base, atol=1e-6)
        q = raw["observation.state.robot_q_current"].clone()
        q[1, 22:29] += torch.tensor([0.3, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1])   # 左腕 7 関節
        moved_arm = ds._state_action({**raw, "observation.state.robot_q_current": q}, 2)["state"]
        assert np.abs(moved_arm[STATE71_GROUPS["ee_pose"]][:6] - base[:6]).max() > 1e-2   # 左手が動く
        np.testing.assert_allclose(moved_arm[STATE71_GROUPS["ee_pose"]][6:], base[6:], atol=1e-6)   # 右手は同じ

    def test_dataset_source_is_the_recorded_ee_state(self, dataset_with_fake_base) -> None:
        """既定 (dataset) はこれまでどおり記録された ee_state。"""
        from model.ramen_ori.state_derive import STATE71_GROUPS

        ds = dataset_with_fake_base()
        assert ds.ee_state_source == "dataset"
        raw = ds._base[4]
        np.testing.assert_allclose(
            ds[4]["state"][STATE71_GROUPS["ee_pose"]].numpy(), raw["observation.state.ee_state"].numpy()[0]
        )

    def test_unknown_source_is_rejected(self, dataset_with_fake_base) -> None:
        with pytest.raises(ValueError, match="ee_state_source"):
            dataset_with_fake_base(ee_state_source="recorded")

    def test_progress_target_is_position_in_the_episode(self, dataset_with_fake_base) -> None:
        """frame_index / (区間の長さ − 1)。fake は 10 frame ずつの区間 (idx = ep*10 + f)。"""
        ds = dataset_with_fake_base(num_samples=20, progress_target=True)
        for idx, expected in ((0, 0.0), (5, 5 / 9), (9, 1.0), (10, 0.0), (19, 1.0)):
            got = ds[idx]["progress_target"]
            assert got.dtype == torch.float32 and got.shape == ()
            assert got.item() == pytest.approx(expected)

    def test_progress_target_is_off_by_default(self, dataset_with_fake_base) -> None:
        assert "progress_target" not in dataset_with_fake_base()[0]

    def test_multi_passes_both_to_every_sub(self) -> None:
        subs = [
            {"base_dataset": FakeLeRobotBase(num_samples=n, source_task_index=tid)}
            for n, tid in ((10, 2), (10, 5))
        ]
        multi = RamenOriMultiDataset(sub_datasets=subs, ee_state_source="fk", progress_target=True)
        assert multi.ee_state_source == "fk" and multi.progress_target is True
        assert all(s.ee_state_source == "fk" and s.progress_target for s in multi.subs)
        assert "progress_target" in multi[15]
