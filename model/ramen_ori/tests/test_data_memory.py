"""dataset の memory の表 (Issue #141 Phase 7) の test。

偽の OBB の cache (天板と左手が frame ごとに動く) と、指令の列を持つ偽の LeRobot で dataset を作り、
表が「MemoryTracker を区間ごとに手で回した結果」と一致するか、item に正しい frame の memory が入るかを見る。
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from model.ramen_ori.data_lerobot import RamenOriLerobotDataset, RamenOriMultiDataset
from model.ramen_ori.fk import G1WristFKTorch
from model.ramen_ori.memory_features import MEMORY_CAMERA, MEMORY_DIM, MEMORY_LAYOUT, MOTION_SLICE, MemoryTracker
from model.ramen_ori.tests.test_data_lerobot import FakeLeRobotBase
from model.ramen_ori.tests.test_obb_cache import _write_manifest
from model.subtask_policy_training.gr00t.g1_full_body_mapping import UPPER_BODY_SOURCE_INDEX_MAP

HASH = "ce877e8237be"
TOP_K = 4


class _Columns(dict):
    def with_format(self, _fmt):
        return self


class _FakeBaseWithColumns(FakeLeRobotBase):
    """FakeLeRobotBase + 指令の列 (hf_dataset) + meta の動画の列 (OBB の cache の対応づけ用)。"""

    def __init__(self, episode_lengths: list[int], source_task_index: int = 5, seed: int = 0) -> None:
        super().__init__(
            num_samples=sum(episode_lengths), episode_lengths=episode_lengths,
            source_task_index=source_task_index, n_image_frames=1,
        )
        rng = np.random.default_rng(seed)
        n = self.num_samples
        self.hf_dataset = _Columns(
            {
                "action.robot_q_desired": (rng.normal(size=(n, 36)) * 0.2).astype(np.float32),
                "action.hand_cmd": rng.uniform(0, 1, size=(n, 2)).astype(np.float32),
                "episode_index": np.repeat(np.arange(len(episode_lengths)), episode_lengths),
            }
        )
        offset = 0
        for row, length in zip(self.meta.episodes, episode_lengths):
            row.update(
                {
                    f"videos/{MEMORY_CAMERA}/chunk_index": 0,
                    f"videos/{MEMORY_CAMERA}/file_index": 0,
                    f"videos/{MEMORY_CAMERA}/from_timestamp": offset / 30.0,
                }
            )
            offset += length

    def _ensure_hf_dataset_loaded(self) -> None:
        pass


def _detections(n_frames: int) -> dict[str, np.ndarray]:
    """frame f: 天板 (class 4) が右へ動き、左手 (class 6) は偶数 frame だけ見える。残りは埋め草。"""
    verts = np.zeros((n_frames, TOP_K, 4, 2), dtype=np.float32)
    conf = np.zeros((n_frames, TOP_K), dtype=np.float32)
    cls = np.full((n_frames, TOP_K), -1, dtype=np.int8)
    valid = np.zeros((n_frames, TOP_K), dtype=bool)
    square = np.array([[0, 0], [0.2, 0.02], [0.18, 0.12], [-0.02, 0.1]], dtype=np.float32)
    for f in range(n_frames):
        verts[f, 0] = square + [0.3 + 0.01 * f, 0.4]
        conf[f, 0], cls[f, 0], valid[f, 0] = 0.9, 4, True
        if f % 2 == 0:
            verts[f, 1] = square * 0.5 + [0.1, 0.7 - 0.005 * f]
            conf[f, 1], cls[f, 1], valid[f, 1] = 0.6, 6, True
    return {"verts": verts, "conf": conf, "class_id": cls, "valid": valid}


def _write_cache(tmp_path, n_frames: int):
    root = tmp_path / "obb_yolo"
    cache_dir = root / HASH
    _write_manifest(cache_dir, HASH, covered_cams=[MEMORY_CAMERA], top_k=TOP_K)
    det = _detections(n_frames)
    rows = {
        "frame_index": np.repeat(np.arange(n_frames, dtype=np.int32), TOP_K),
        "det_idx": np.tile(np.arange(TOP_K, dtype=np.int8), n_frames),
        "verts": list(det["verts"].reshape(-1, 8)),
        "conf": det["conf"].reshape(-1),
        "class_id": det["class_id"].reshape(-1),
        "valid": det["valid"].reshape(-1),
    }
    schema = pa.schema(
        [
            pa.field("frame_index", pa.int32(), nullable=False),
            pa.field("det_idx", pa.int8(), nullable=False),
            pa.field("verts", pa.list_(pa.float32(), 8), nullable=False),
            pa.field("conf", pa.float32(), nullable=False),
            pa.field("class_id", pa.int8(), nullable=False),
            pa.field("valid", pa.bool_(), nullable=False),
        ]
    )
    path = cache_dir / MEMORY_CAMERA / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table(rows, schema=schema), path)
    return root, det


def _expected_table(base: _FakeBaseWithColumns, det: dict, lengths: list[int]) -> np.ndarray:
    command38 = np.concatenate([base.hf_dataset["action.robot_q_desired"], base.hf_dataset["action.hand_cmd"]], axis=1)
    parts = G1WristFKTorch.from_default_urdf().forward_detailed(
        torch.from_numpy(command38[:, np.asarray(UPPER_BODY_SOURCE_INDEX_MAP)])
    )
    hand = torch.cat([parts["left_hand"], parts["right_hand"]], dim=-1).detach().numpy()
    tracker, table, start = MemoryTracker(), [], 0
    for length in lengths:
        tracker.reset()
        for t in range(length):
            f, v = start + t, det["valid"][start + t]
            table.append(
                tracker.update(det["class_id"][f][v], det["conf"][f][v], det["verts"][f][v], hand[f - 1] if t else None)
            )
        start += length
    return np.stack(table)


def _dataset(tmp_path, lengths: list[int], **kwargs) -> tuple[RamenOriLerobotDataset, dict, _FakeBaseWithColumns]:
    root, det = _write_cache(tmp_path, sum(lengths))
    base = _FakeBaseWithColumns(lengths)
    ds = RamenOriLerobotDataset(
        base_dataset=base, load_prev_image=False, memory=True,
        obb_precomputed_root=root, obb_precomputed_hash=HASH, **kwargs,
    )
    return ds, det, base


def test_table_matches_tracker_run_per_episode(tmp_path) -> None:
    lengths = [7, 5, 9]
    ds, det, base = _dataset(tmp_path, lengths)
    table, skill_ids = ds.memory_arrays()
    assert table.shape == (21, MEMORY_DIM)
    np.testing.assert_allclose(table, _expected_table(base, det, lengths), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(skill_ids, [4] * 21)
    assert ds.memory_yolo_ckpt == f"dummy@{HASH}"   # manifest の "hf:" を外した形 (推論の policy_config.yaml と同じ)


def test_item_has_memory_of_its_frame(tmp_path) -> None:
    ds, _, _ = _dataset(tmp_path, [6, 6])
    table, _ = ds.memory_arrays()
    for idx in (0, 5, 6, 11):
        torch.testing.assert_close(ds[idx]["memory"], torch.from_numpy(table[idx]))


def test_episode_start_resets_references(tmp_path) -> None:
    """2 本目の区間の先頭は、天板の差 0 (区間ごとに最初に見えた frame が基準)、速度の移動平均も 0。"""
    ds, _, _ = _dataset(tmp_path, [6, 6])
    table, _ = ds.memory_arrays()
    dcx = MEMORY_LAYOUT.index("table_top.dcx")
    assert table[5, dcx] == pytest.approx(0.05, abs=1e-6)   # 1 本目の終わり: 天板は 5 frame で 0.05 右へ
    assert table[6, dcx] == 0                                  # 2 本目の先頭で基準を張り直す
    assert table[5, MOTION_SLICE].any()
    np.testing.assert_array_equal(table[6, MOTION_SLICE], 0)


def test_rows_out_of_episode_order_raise(tmp_path) -> None:
    root, _ = _write_cache(tmp_path, 12)
    base = _FakeBaseWithColumns([6, 6])
    base.hf_dataset["episode_index"] = base.hf_dataset["episode_index"][::-1].copy()
    with pytest.raises(ValueError, match="meta.episodes の順"):
        RamenOriLerobotDataset(
            base_dataset=base, load_prev_image=False, memory=True, obb_precomputed_root=root, obb_precomputed_hash=HASH
        )


def test_memory_requires_obb_root() -> None:
    with pytest.raises(ValueError, match="obb_precomputed_root"):
        RamenOriLerobotDataset(base_dataset=_FakeBaseWithColumns([4]), load_prev_image=False, memory=True)


def test_multi_concatenates_tables(tmp_path) -> None:
    sub0, _, _ = _dataset(tmp_path / "a", [5])
    sub1, _, _ = _dataset(tmp_path / "b", [4, 3])
    multi = RamenOriMultiDataset(sub_datasets=[sub0, sub1])
    table, skill_ids = multi.memory_arrays()
    assert multi.memory is True and multi.memory_yolo_ckpt == f"dummy@{HASH}"
    np.testing.assert_array_equal(table, np.concatenate([sub0.memory_arrays()[0], sub1.memory_arrays()[0]]))
    assert len(skill_ids) == 12
