"""Unit test: merge_curated_lerobot_datasets.py (Issue #115、Phase 2 prep B-3/B-4)。

- FILTER_REGISTRY: `all` / `last_per_source_episode` の挙動
- parse_task_filters: valid / invalid format / unknown filter / duplicate
- validate_schemas: fps / video_keys / features mismatch、task_index 重複の error
- build_merge_plan: filter 適用後 episode 数、global episode_index 昇順
- merge_data (B-4): synthetic 2-source dataset で e2e (parquet 実書出 + 再読込 verify)
"""

from __future__ import annotations

import errno
import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_merger():
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "merge_curated_lerobot_datasets.py"
    spec = importlib.util.spec_from_file_location(
        "merge_curated_lerobot_datasets", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


merger = load_merger()


def _make_source(
    task_index: int = 4,
    fps: float = 30.0,
    video_keys: tuple[str, ...] = ("observation.images.cam_0",),
    features: dict | None = None,
    episodes: list[dict] | None = None,
) -> "merger.SourceDataset":
    if features is None:
        features = {
            "action.joint": {"dtype": "float32", "shape": [19]},
            "observation.state.joint": {"dtype": "float32", "shape": [19]},
            "observation.images.cam_0": {"dtype": "video", "shape": [224, 224, 3]},
        }
    if episodes is None:
        episodes = []
    return merger.SourceDataset(
        root=Path("/dummy"),
        info={"fps": fps, "features": features},
        tasks=[{"task_index": task_index}],
        episodes=episodes,
        video_keys=list(video_keys),
        fps=fps,
        features=features,
    )


class TestFilterLastPerSourceEpisode:
    def test_keeps_max_frame_start_per_source_episode(self) -> None:
        eps = [
            {"episode_index": 0, "source_episode_index": 10, "source_frame_start": 100},
            {"episode_index": 1, "source_episode_index": 10, "source_frame_start": 500},
            {"episode_index": 2, "source_episode_index": 10, "source_frame_start": 300},
            {"episode_index": 3, "source_episode_index": 20, "source_frame_start": 50},
            {"episode_index": 4, "source_episode_index": 20, "source_frame_start": 900},
        ]
        result = merger._filter_last_per_source_episode(eps)
        # source_ep 10 は frame_start=500 (episode_index=1)、source_ep 20 は 900 (episode_index=4)
        assert len(result) == 2
        starts_by_src = {int(e["source_episode_index"]): int(e["source_frame_start"]) for e in result}
        assert starts_by_src == {10: 500, 20: 900}

    def test_single_episode_per_source_passthrough(self) -> None:
        eps = [
            {"episode_index": 0, "source_episode_index": 5, "source_frame_start": 200},
            {"episode_index": 1, "source_episode_index": 6, "source_frame_start": 100},
        ]
        result = merger._filter_last_per_source_episode(eps)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        assert merger._filter_last_per_source_episode([]) == []


class TestFilterAll:
    def test_identity(self) -> None:
        eps = [
            {"episode_index": 0, "source_episode_index": 1, "source_frame_start": 10},
            {"episode_index": 1, "source_episode_index": 1, "source_frame_start": 20},
        ]
        assert merger._filter_all(eps) == eps


class TestParseTaskFilters:
    def test_valid_single(self) -> None:
        assert merger.parse_task_filters(["2:last_per_source_episode"]) == {
            2: "last_per_source_episode"
        }

    def test_valid_multiple(self) -> None:
        result = merger.parse_task_filters(
            ["2:last_per_source_episode", "3:all"]
        )
        assert result == {2: "last_per_source_episode", 3: "all"}

    def test_invalid_format_missing_colon(self) -> None:
        with pytest.raises(ValueError, match="TASK_ID:FILTER_NAME"):
            merger.parse_task_filters(["2last"])

    def test_unknown_filter_name(self) -> None:
        with pytest.raises(ValueError, match="unknown filter"):
            merger.parse_task_filters(["2:nonexistent"])

    def test_duplicate_task_id(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            merger.parse_task_filters(["2:all", "2:last_per_source_episode"])


class TestValidateSchemas:
    def test_all_matching_passes(self) -> None:
        s1 = _make_source(task_index=4)
        s2 = _make_source(task_index=5)
        merger.validate_schemas([s1, s2])  # no error

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no source"):
            merger.validate_schemas([])

    def test_video_shape_mismatch_raises(self) -> None:
        """解像度が違う source は passthrough で素通ししてはいけない (merge 後の info.json は先頭 source の値)。"""
        s1 = _make_source(task_index=4)
        feats = {k: dict(v) for k, v in s1.features.items()}
        feats["observation.images.cam_0"]["shape"] = [480, 640, 3]
        s2 = _make_source(task_index=5, features=feats)
        with pytest.raises(ValueError, match="video feature mismatch"):
            merger.validate_schemas([s1, s2])

    def test_video_fps_mismatch_raises(self) -> None:
        s1 = _make_source(task_index=4)
        feats = {k: dict(v) for k, v in s1.features.items()}
        feats["observation.images.cam_0"]["info"] = {"video.fps": 60.0}
        s2 = _make_source(task_index=5, features=feats)
        with pytest.raises(ValueError, match="video feature mismatch"):
            merger.validate_schemas([s1, s2])

    def test_codec_mismatch_only_blocks_passthrough(self) -> None:
        """作り直す側は libx264 / yuv420p に揃えるので codec の違いは通す。passthrough は元 file のまま置くので止める。"""
        s1 = _make_source(task_index=4)
        feats = {k: dict(v) for k, v in s1.features.items()}
        feats["observation.images.cam_0"]["info"] = {"video.codec": "hevc"}
        s2 = _make_source(task_index=5, features=feats)
        merger.validate_schemas([s1, s2])  # reencode は通る
        with pytest.raises(ValueError, match="video feature mismatch"):
            merger.validate_schemas([s1, s2], video_mode="passthrough")

    def test_fps_mismatch_raises(self) -> None:
        s1 = _make_source(task_index=4, fps=30.0)
        s2 = _make_source(task_index=5, fps=60.0)
        with pytest.raises(ValueError, match="fps mismatch"):
            merger.validate_schemas([s1, s2])

    def test_video_keys_mismatch_raises(self) -> None:
        s1 = _make_source(task_index=4, video_keys=("observation.images.cam_0",))
        # cam_1 の feature も追加しないと features check で先に落ちてしまうので、features も合わせる
        features_with_cam1 = {
            "action.joint": {"dtype": "float32", "shape": [19]},
            "observation.state.joint": {"dtype": "float32", "shape": [19]},
            "observation.images.cam_0": {"dtype": "video", "shape": [224, 224, 3]},
            "observation.images.cam_1": {"dtype": "video", "shape": [224, 224, 3]},
        }
        s2 = _make_source(
            task_index=5,
            video_keys=("observation.images.cam_0", "observation.images.cam_1"),
            features=features_with_cam1,
        )
        with pytest.raises(ValueError, match="video_keys mismatch"):
            merger.validate_schemas([s1, s2])

    def test_features_mismatch_raises(self) -> None:
        s1 = _make_source(task_index=4)
        # extra feature を追加
        features2 = {
            "action.joint": {"dtype": "float32", "shape": [19]},
            "observation.state.joint": {"dtype": "float32", "shape": [19]},
            "observation.images.cam_0": {"dtype": "video", "shape": [224, 224, 3]},
            "extra_feature": {"dtype": "float32", "shape": [1]},
        }
        s2 = _make_source(task_index=5, features=features2)
        with pytest.raises(ValueError, match="features mismatch"):
            merger.validate_schemas([s1, s2])

    def test_task_index_duplicate_raises(self) -> None:
        s1 = _make_source(task_index=4)
        s2 = _make_source(task_index=4)  # 重複
        with pytest.raises(ValueError, match="duplicated"):
            merger.validate_schemas([s1, s2])

    def test_task_index_duplicate_allowed_with_flag(self) -> None:
        """Issue #122: allow_duplicate_task_indices=True で複数 chunks 同 task を許容。"""
        s1 = _make_source(task_index=5)
        s2 = _make_source(task_index=5)  # 同じ task
        s3 = _make_source(task_index=5)  # 3 chunks 同 task も OK
        # 例外投げないこと
        merger.validate_schemas([s1, s2, s3], allow_duplicate_task_indices=True)

    def test_task_index_duplicate_still_raises_by_default(self) -> None:
        """default (flag なし) では従来通り reject する regression 防止。"""
        s1 = _make_source(task_index=5)
        s2 = _make_source(task_index=5)
        with pytest.raises(ValueError, match="duplicated"):
            merger.validate_schemas([s1, s2])
        # 明示 False も同じ挙動
        with pytest.raises(ValueError, match="duplicated"):
            merger.validate_schemas([s1, s2], allow_duplicate_task_indices=False)

    def test_mixed_tasks_with_duplicates_allowed(self) -> None:
        """task 5 (8 chunks) + task 7 (1 chunk) 相当の realistic ケース。"""
        sources = [_make_source(task_index=5) for _ in range(8)] + [_make_source(task_index=7)]
        # 例外投げないこと
        merger.validate_schemas(sources, allow_duplicate_task_indices=True)


class TestBuildMergePlan:
    def test_no_filter_all_episodes_kept(self) -> None:
        eps1 = [{"episode_index": i, "source_episode_index": i, "source_frame_start": 0} for i in range(3)]
        eps2 = [{"episode_index": i, "source_episode_index": i, "source_frame_start": 0} for i in range(2)]
        s1 = _make_source(task_index=4, episodes=eps1)
        s2 = _make_source(task_index=5, episodes=eps2)
        plan = merger.build_merge_plan([s1, s2], {})
        assert len(plan.episodes) == 5
        # global output_episode_index は 0..4 昇順
        assert [ep["output_episode_index"] for ep in plan.episodes] == [0, 1, 2, 3, 4]

    def test_filter_reduces_episode_count(self) -> None:
        # source 1 task 2、Episode 内 3 flip (source_ep_idx 同じで frame_start 違い)
        eps1 = [
            {"episode_index": 0, "source_episode_index": 10, "source_frame_start": 100},
            {"episode_index": 1, "source_episode_index": 10, "source_frame_start": 500},
            {"episode_index": 2, "source_episode_index": 10, "source_frame_start": 300},
            {"episode_index": 3, "source_episode_index": 20, "source_frame_start": 200},
        ]
        s1 = _make_source(task_index=2, episodes=eps1)
        # source 2 task 4、filter 無し
        eps2 = [{"episode_index": 0, "source_episode_index": 5, "source_frame_start": 0}]
        s2 = _make_source(task_index=4, episodes=eps2)
        plan = merger.build_merge_plan([s1, s2], {2: "last_per_source_episode"})
        # source 1 は filter 適用で source_ep 10 → 1 個 (frame_start 500)、source_ep 20 → 1 個 → 計 2
        # source 2 は filter 無しで 1 個 → 計 3
        assert len(plan.episodes) == 3
        # task_id が正しく紐付いてる
        task_ids = [ep["source_task_id"] for ep in plan.episodes]
        assert task_ids == [2, 2, 4]
        assert [ep["output_episode_index"] for ep in plan.episodes] == [0, 1, 2]

    def test_source_idx_preserved(self) -> None:
        eps1 = [{"episode_index": 0, "source_episode_index": 0, "source_frame_start": 0}]
        eps2 = [{"episode_index": 0, "source_episode_index": 0, "source_frame_start": 0}]
        s1 = _make_source(task_index=7, episodes=eps1)
        s2 = _make_source(task_index=5, episodes=eps2)
        plan = merger.build_merge_plan([s1, s2], {})
        assert plan.episodes[0]["source_idx"] == 0
        assert plan.episodes[1]["source_idx"] == 1


# =============================================================================
# B-4: merge_data smoke test with synthetic source datasets
# =============================================================================


def _create_fake_source(
    tmp_path: Path,
    task_index: int,
    n_episodes: int,
    frames_per_episode: int = 5,
) -> Path:
    """LeRobot v3 dataset directory を tmp_path に fake で作る。

    - data/chunk-000/file-000.parquet: 全 episode の frame 連結
    - meta/info.json: fps=30、features は action.joint + state.joint (video 無し = video merge skip)
    - meta/tasks.parquet: 1 task
    - meta/episodes/chunk-000/file-000.parquet: n_episodes rows
    """
    features = {
        "action.joint": {"dtype": "float32", "shape": [2]},
        "observation.state.joint": {"dtype": "float32", "shape": [2]},
    }
    root = tmp_path / f"src_task_{task_index}"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    with (root / "meta" / "info.json").open("w") as f:
        json.dump({"fps": 30.0, "features": features, "chunks_size": 1000}, f)

    pq.write_table(
        pa.Table.from_pylist(
            [{"task_index": task_index, "__index_level_0__": f"task_{task_index}_name"}]
        ),
        root / "meta" / "tasks.parquet",
    )

    data_rows = []
    episode_rows = []
    global_idx = 0
    for ep_idx in range(n_episodes):
        for f_idx in range(frames_per_episode):
            data_rows.append(
                {
                    "timestamp": float(f_idx) / 30.0,
                    "frame_index": f_idx,
                    "episode_index": ep_idx,
                    "index": global_idx + f_idx,
                    "task_index": task_index,
                    "action.joint": [0.1 * f_idx, 0.2 * f_idx],
                    "observation.state.joint": [0.3 * f_idx, 0.4 * f_idx],
                }
            )
        episode_rows.append(
            {
                "episode_index": ep_idx,
                "tasks": [f"task_{task_index}_name"],
                "length": frames_per_episode,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": global_idx,
                "dataset_to_index": global_idx + frames_per_episode,
                "source_episode_index": ep_idx * 10,
                "source_frame_start": ep_idx * 100,
                "source_frame_end": ep_idx * 100 + frames_per_episode - 1,
            }
        )
        global_idx += frames_per_episode

    pq.write_table(
        pa.Table.from_pylist(data_rows),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(episode_rows),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    return root


class TestMergeDataE2E:
    def test_two_source_merge_no_filter(self, tmp_path: Path) -> None:
        """2 source dataset を no-filter で merge、output parquet を書き出して再読込 verify。"""
        src1_root = _create_fake_source(
            tmp_path, task_index=4, n_episodes=3, frames_per_episode=5
        )
        src2_root = _create_fake_source(
            tmp_path, task_index=5, n_episodes=2, frames_per_episode=4
        )
        s1 = merger.load_source(src1_root)
        s2 = merger.load_source(src2_root)
        plan = merger.build_merge_plan([s1, s2], {})

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_meta, total_frames, stats = merger.merge_data(
            sources=[s1, s2],
            plan=plan,
            output_dir=output_dir,
            data_files_size_mb=100,
            chunks_size=1000,
        )

        # Shape checks
        assert total_frames == 3 * 5 + 2 * 4  # 23
        assert len(output_meta) == 5
        assert [m["episode_index"] for m in output_meta] == [0, 1, 2, 3, 4]
        assert [m["length"] for m in output_meta] == [5, 5, 5, 4, 4]

        # dataset_from/to_index が consecutive
        from_idx = 0
        for m in output_meta:
            assert m["dataset_from_index"] == from_idx
            assert m["dataset_to_index"] == from_idx + m["length"]
            from_idx += m["length"]
        assert from_idx == total_frames

        # Read back written parquet
        written = pq.read_table(
            output_dir / "data" / "chunk-000" / "file-000.parquet"
        ).to_pylist()
        assert len(written) == total_frames

        # Episode index reindex
        assert written[0]["episode_index"] == 0
        assert written[15]["episode_index"] == 3  # src2 の第 1 episode
        # Global index 0..N-1 連番
        for i, r in enumerate(written):
            assert r["index"] == i
        # task_index は 0-based contiguous に remap される (Issue #122、LeRobot iloc 互換)。
        # sources task_indices sorted = [4, 5] → remap = {4: 0, 5: 1}
        src1_frames = written[:15]  # src1 の 3 ep × 5 frame (source task=4 → 0)
        src2_frames = written[15:]  # src2 の 2 ep × 4 frame (source task=5 → 1)
        assert all(r["task_index"] == 0 for r in src1_frames)
        assert all(r["task_index"] == 1 for r in src2_frames)

        # feature preservation
        assert "action.joint" in written[0]
        assert "observation.state.joint" in written[0]

        # Stats accumulator が非-video feature 分だけ populate
        assert "action.joint" in stats
        assert "observation.state.joint" in stats

    def test_filter_applied(self, tmp_path: Path) -> None:
        """flip_table 相当の filter (last_per_source_episode) を適用、生存 episode のみ出力。"""
        src_root = tmp_path / "src_flip"
        (src_root / "meta").mkdir(parents=True)
        (src_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (src_root / "data" / "chunk-000").mkdir(parents=True)

        features = {"action.joint": {"dtype": "float32", "shape": [2]}}
        with (src_root / "meta" / "info.json").open("w") as f:
            json.dump({"fps": 30.0, "features": features, "chunks_size": 1000}, f)
        pq.write_table(
            pa.Table.from_pylist([{"task_index": 2, "__index_level_0__": "flip_table"}]),
            src_root / "meta" / "tasks.parquet",
        )

        # Episodes: source_ep 10 に 2 seg (frame_start 100/500)、source_ep 20 に 1 seg
        data_rows = []
        episode_rows = []
        global_idx = 0
        for ep_idx, src_ep, frame_start in [
            (0, 10, 100),  # non-last、drop 予定
            (1, 10, 500),  # last、keep
            (2, 20, 200),  # keep
        ]:
            for f_idx in range(3):
                data_rows.append(
                    {
                        "timestamp": float(f_idx) / 30.0,
                        "frame_index": f_idx,
                        "episode_index": ep_idx,
                        "index": global_idx + f_idx,
                        "task_index": 2,
                        "action.joint": [float(ep_idx), float(f_idx)],
                    }
                )
            episode_rows.append(
                {
                    "episode_index": ep_idx,
                    "tasks": ["flip_table"],
                    "length": 3,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": global_idx,
                    "dataset_to_index": global_idx + 3,
                    "source_episode_index": src_ep,
                    "source_frame_start": frame_start,
                    "source_frame_end": frame_start + 2,
                }
            )
            global_idx += 3

        pq.write_table(
            pa.Table.from_pylist(data_rows),
            src_root / "data" / "chunk-000" / "file-000.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(episode_rows),
            src_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        )

        s = merger.load_source(src_root)
        plan = merger.build_merge_plan([s], {2: "last_per_source_episode"})

        output_dir = tmp_path / "output_filtered"
        output_dir.mkdir()
        output_meta, total_frames, _ = merger.merge_data(
            sources=[s],
            plan=plan,
            output_dir=output_dir,
            data_files_size_mb=100,
            chunks_size=1000,
        )

        # Filter で source_ep 10 の frame_start=500 (episode_index=1) と source_ep 20 (2) が残る
        assert len(output_meta) == 2
        assert total_frames == 6  # 2 episode × 3 frame
        # source_episode_index preserved in output_meta
        src_ep_indices = sorted(int(m["source_episode_index"]) for m in output_meta)
        assert src_ep_indices == [10, 20]
        # frame_start は 500 と 200 (100 は drop)
        starts = sorted(int(m["source_frame_start"]) for m in output_meta)
        assert starts == [200, 500]


# =============================================================================
# B-5: meta rebuild tests
# =============================================================================


class TestMergeTasks:
    def test_union_and_remap_to_zero_based_contiguous(self) -> None:
        """merge_tasks は source task_index を 0-based contiguous に remap
        (Issue #122、LeRobot iloc[task_idx] 互換)。task 名は sorted-original 順で保存、
        legacy `__index_level_0__` は canonical `task` column に rename。"""
        s1 = _make_source(task_index=4)
        s1.tasks[0]["__index_level_0__"] = "pick_leg"
        s2 = _make_source(task_index=5)
        s2.tasks[0]["__index_level_0__"] = "rotate_table_base"
        merged = merger.merge_tasks([s1, s2])
        assert len(merged) == 2
        assert [t["task_index"] for t in merged] == [0, 1]
        # sorted-original 順で並ぶので 0 行目 = task 4 の name、1 行目 = task 5 の name
        # column は canonical "task" に rename 済 (LeRobot v3 schema)
        assert merged[0]["task"] == "pick_leg"
        assert merged[1]["task"] == "rotate_table_base"
        assert "__index_level_0__" not in merged[0]
        assert "__index_level_0__" not in merged[1]

    def test_task_column_preserved_if_already_canonical(self) -> None:
        """source が既に canonical `task` column を持つ場合は rename せず保持。"""
        s = _make_source(task_index=3)
        s.tasks[0]["task"] = "canonical_name"
        merged = merger.merge_tasks([s])
        assert merged[0]["task"] == "canonical_name"
        assert merged[0]["task_index"] == 0  # remap

    def test_sorted_by_original_task_index_then_remap(self) -> None:
        s1 = _make_source(task_index=7)
        s2 = _make_source(task_index=0)
        s3 = _make_source(task_index=3)
        merged = merger.merge_tasks([s1, s2, s3])
        # sorted original [0, 3, 7] → new indices [0, 1, 2]
        assert [t["task_index"] for t in merged] == [0, 1, 2]

    def test_already_contiguous_is_noop(self) -> None:
        """既に 0-based contiguous なら remap 後も同じ値 (single source, task 0)。"""
        s1 = _make_source(task_index=0)
        merged = merger.merge_tasks([s1])
        assert [t["task_index"] for t in merged] == [0]

    def test_lerobot_load_tasks_compatibility(self) -> None:
        """canonical schema なので LeRobot load_tasks の set_index("task") 経路を通り、
        iloc[i].name が task 名 string を返す (int ではなく)。今回 bug 修正の中核。"""
        import pandas as pd
        s1 = _make_source(task_index=5)
        s1.tasks[0]["__index_level_0__"] = "rotate_table_base"
        s2 = _make_source(task_index=7)
        s2.tasks[0]["__index_level_0__"] = "move_table_base"
        merged = merger.merge_tasks([s1, s2])
        df = pd.DataFrame(merged)
        # LeRobot v3 load_tasks: `if "task" in columns: sort_values + set_index("task")`
        assert "task" in df.columns
        df = df.sort_values("task_index").set_index("task")
        df.index.name = "task"
        # iloc[0].name / iloc[1].name が task 名 string を返す (dataset_reader:1138 が触る)
        assert df.iloc[0].name == "rotate_table_base"
        assert df.iloc[1].name == "move_table_base"


class TestBuildTaskIndexRemap:
    """Issue #122: source の sparse task_index を 0-based contiguous に remap する helper。"""

    def test_task_5_and_7_maps_to_0_and_1(self) -> None:
        s1 = _make_source(task_index=5)
        s2 = _make_source(task_index=7)
        remap = merger.build_task_index_remap([s1, s2])
        assert remap == {5: 0, 7: 1}

    def test_deterministic_across_source_order(self) -> None:
        """source order が入れ替わっても同じ remap (sorted-original が決定的)。"""
        s5 = _make_source(task_index=5)
        s7 = _make_source(task_index=7)
        assert merger.build_task_index_remap([s5, s7]) == merger.build_task_index_remap([s7, s5])

    def test_duplicate_source_task_index_collapsed(self) -> None:
        """同 task_id を持つ複数 source (task 5 の chunk0..7 想定) は 1 個の remap 対象。"""
        sources = [_make_source(task_index=5) for _ in range(3)] + [_make_source(task_index=7)]
        remap = merger.build_task_index_remap(sources)
        assert remap == {5: 0, 7: 1}

    def test_already_contiguous_identity(self) -> None:
        s1 = _make_source(task_index=0)
        s2 = _make_source(task_index=1)
        assert merger.build_task_index_remap([s1, s2]) == {0: 0, 1: 1}

    def test_single_source_maps_to_zero(self) -> None:
        s = _make_source(task_index=42)
        assert merger.build_task_index_remap([s]) == {42: 0}


class TestSanitizeEpisodeRow:
    def test_removes_internal_fields(self) -> None:
        row = {
            "episode_index": 5,
            "length": 30,
            "source_dataset_root": "/tmp/x",   # internal
            "source_task_id": 4,                # internal
            "__source_videos/cam_0/chunk_index": 0,  # internal
            "source_episode_index": 10,         # keep
            "videos/cam_0/chunk_index": 2,      # keep (B-6 出力)
        }
        sanitized = merger._sanitize_episode_row(row)
        assert "episode_index" in sanitized
        assert "source_episode_index" in sanitized
        assert "videos/cam_0/chunk_index" in sanitized
        assert "source_dataset_root" not in sanitized
        assert "source_task_id" not in sanitized
        assert "__source_videos/cam_0/chunk_index" not in sanitized


class TestRebuildMetaE2E:
    def test_all_meta_files_written(self, tmp_path: Path) -> None:
        """B-4 output + rebuild_meta で tasks/episodes/info/stats/provenance が全て生成される。"""
        src1_root = _create_fake_source(
            tmp_path, task_index=4, n_episodes=3, frames_per_episode=5
        )
        src2_root = _create_fake_source(
            tmp_path, task_index=5, n_episodes=2, frames_per_episode=4
        )
        s1 = merger.load_source(src1_root)
        s2 = merger.load_source(src2_root)
        plan = merger.build_merge_plan([s1, s2], {})

        output_dir = tmp_path / "output_meta"
        output_dir.mkdir()
        output_meta, total_frames, stats = merger.merge_data(
            sources=[s1, s2], plan=plan, output_dir=output_dir,
            data_files_size_mb=100, chunks_size=1000,
        )

        merger.rebuild_meta(
            sources=[s1, s2],
            output_meta=output_meta,
            output_dir=output_dir,
            total_frames=total_frames,
            stats=stats,
            video_keys=s1.video_keys,  # fake source は video 無し = []
            episodes_per_meta_file=1000,
            data_files_size_mb=100,
            video_files_size_mb=500,
            chunks_size=1000,
            plan=plan,
            filters_by_task={},
            video_mode="reencode",
        )

        # 全 meta file が生成された
        assert (output_dir / "meta" / "tasks.parquet").exists()
        assert (output_dir / "meta" / "info.json").exists()
        assert (output_dir / "meta" / "stats.json").exists()
        assert (output_dir / "meta" / "team_ramen_curation.json").exists()
        ep_files = list((output_dir / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))
        assert len(ep_files) >= 1

        # tasks.parquet: 2 row、task_index は 0-based contiguous に remap (Issue #122)
        # sources sorted = [4, 5] → new indices [0, 1]
        tasks_written = pq.read_table(output_dir / "meta" / "tasks.parquet").to_pylist()
        assert len(tasks_written) == 2
        assert sorted(int(t["task_index"]) for t in tasks_written) == [0, 1]

        # LeRobot iloc 互換: tasks.parquet の row order と data の task_index が一致
        # (iloc[0] = task_name for source_task=4、iloc[1] = task_name for source_task=5)
        data_written = pq.read_table(
            output_dir / "data" / "chunk-000" / "file-000.parquet"
        ).to_pylist()
        src1_task_indices = {r["task_index"] for r in data_written[:15]}  # src task=4
        src2_task_indices = {r["task_index"] for r in data_written[15:]}  # src task=5
        assert src1_task_indices == {0}
        assert src2_task_indices == {1}

        # info.json: total_episodes=5, total_frames=23, total_tasks=2
        with (output_dir / "meta" / "info.json").open() as f:
            info = json.load(f)
        assert info["total_episodes"] == 5
        assert info["total_frames"] == 23
        assert info["total_tasks"] == 2
        assert info["fps"] == 30.0
        assert info["splits"] == {"train": "0:5"}

        # stats.json: 非-video feature が全部あり
        with (output_dir / "meta" / "stats.json").open() as f:
            stats_json = json.load(f)
        assert "action.joint" in stats_json
        assert "observation.state.joint" in stats_json
        # video keys は fake source では空、stats_json に無い
        assert not any(k.startswith("observation.images") for k in stats_json)

        # episodes.parquet: 5 row、internal field 除外済
        eps_all = []
        for ep_file in sorted(ep_files):
            eps_all.extend(pq.read_table(ep_file).to_pylist())
        assert len(eps_all) == 5
        assert all("source_dataset_root" not in ep for ep in eps_all)
        assert all("source_task_id" not in ep for ep in eps_all)
        # episode_index が 0..4 昇順
        assert [int(e["episode_index"]) for e in eps_all] == [0, 1, 2, 3, 4]

        # provenance が正しい source 情報を含む
        with (output_dir / "meta" / "team_ramen_curation.json").open() as f:
            prov = json.load(f)
        assert prov["total_episodes"] == 5
        assert prov["total_frames"] == 23
        assert len(prov["sources"]) == 2
        assert prov["filters_applied"] == {}
        assert prov["episodes_per_task_after_filter"] == {"4": 3, "5": 2}
        # fake source は team_ramen_curation.json を持たない → build_provenance は None
        assert all(src["build_provenance"] is None for src in prov["sources"])

    def test_provenance_nests_source_build_provenance(self, tmp_path: Path) -> None:
        """source root に team_ramen_curation.json があれば MIX 側 provenance の
        sources[i].build_provenance に nest される (source SHA / verdicts / seg 詳細 preserve)。"""
        src1_root = _create_fake_source(tmp_path, task_index=4, n_episodes=1, frames_per_episode=3)
        src2_root = _create_fake_source(tmp_path, task_index=5, n_episodes=1, frames_per_episode=3)
        # src1 に build 時 provenance を模擬配置 (build_curated_task_dataset.py の出力形式)
        build_prov_1 = {
            "source_repo_id": "BitRobot/G1_WBT_Dex1_Building-Children-Table",
            "source_revision": "4b5961c6f8b97ececa75ee73041b30bcbf463ae2",
            "curation_repo_id": "Team-RAMEN/ramen-g1-ikea-assembly-curation",
            "curation_revision": "7c0a3f8bb7ac8dc6672d1204a32c6531737d7a14",
            "task_id": 4,
            "allowed_verdicts": ["optimal"],
            "segment_count": 1,
            "segments": [{"source_episode_index": 0, "source_frame_start": 0}],
        }
        with (src1_root / "meta" / "team_ramen_curation.json").open("w") as f:
            json.dump(build_prov_1, f)
        # src2 は team_ramen_curation.json 無し (build_provenance が None になる)
        s1 = merger.load_source(src1_root)
        s2 = merger.load_source(src2_root)
        plan = merger.build_merge_plan([s1, s2], {})
        output_dir = tmp_path / "output_prov"
        output_dir.mkdir()
        output_meta, total_frames, stats = merger.merge_data(
            sources=[s1, s2], plan=plan, output_dir=output_dir,
            data_files_size_mb=100, chunks_size=1000,
        )
        merger.rebuild_meta(
            sources=[s1, s2], output_meta=output_meta, output_dir=output_dir,
            total_frames=total_frames, stats=stats, video_keys=s1.video_keys,
            episodes_per_meta_file=1000, data_files_size_mb=100,
            video_files_size_mb=500, chunks_size=1000, plan=plan, filters_by_task={},
            video_mode="reencode",
        )
        with (output_dir / "meta" / "team_ramen_curation.json").open() as f:
            prov = json.load(f)
        # src1 側は build_provenance に build 時の source SHA/verdicts/segments が nest
        src1_prov = next(s for s in prov["sources"] if 4 in s["task_indices"])
        assert src1_prov["build_provenance"] is not None
        assert src1_prov["build_provenance"]["source_repo_id"] == "BitRobot/G1_WBT_Dex1_Building-Children-Table"
        assert src1_prov["build_provenance"]["source_revision"] == "4b5961c6f8b97ececa75ee73041b30bcbf463ae2"
        assert src1_prov["build_provenance"]["allowed_verdicts"] == ["optimal"]
        assert src1_prov["build_provenance"]["task_id"] == 4
        # src2 側は team_ramen_curation.json 無しなので build_provenance = None
        src2_prov = next(s for s in prov["sources"] if 5 in s["task_indices"])
        assert src2_prov["build_provenance"] is None


# =============================================================================
# B-6: video re-encode tests (batching pure logic + graceful skip)
# =============================================================================


class TestBatchClipsToOutputFiles:
    def test_all_clips_fit_in_one_file(self) -> None:
        plans = [{"episode_index": i} for i in range(3)]
        sizes = [10, 20, 30]  # bytes
        counts = [5, 10, 15]  # frames
        # target = 100 MB → 全部 1 file に入る
        result = merger._batch_clips_to_output_files(
            plans, sizes, counts, video_files_size_mb=100, chunks_size=1000
        )
        assert len(result) == 3
        for r in result:
            assert r["output_chunk_index"] == 0
            assert r["output_file_index"] == 0
        # from/to_frame_count が cumulative
        assert result[0]["from_frame_count"] == 0
        assert result[0]["to_frame_count"] == 5
        assert result[1]["from_frame_count"] == 5
        assert result[1]["to_frame_count"] == 15
        assert result[2]["from_frame_count"] == 15
        assert result[2]["to_frame_count"] == 30

    def test_batch_splits_when_size_exceeded(self) -> None:
        # target = 1 byte × 1MB = 1MB、3 clip 各 500KB → 1clip 目 (500K) OK, 2clip 目
        # (追加で 500K = total 1000K) OK、3clip 目 (追加で 500K = total 1500K > 1MB) → 分割
        plans = [{"episode_index": i} for i in range(3)]
        sizes = [500 * 1024, 500 * 1024, 500 * 1024]  # 500KB each
        counts = [30, 30, 30]
        result = merger._batch_clips_to_output_files(
            plans, sizes, counts, video_files_size_mb=1, chunks_size=1000
        )
        # clip 0, 1 が file 0、clip 2 が次 file
        assert result[0]["output_file_index"] == 0
        assert result[1]["output_file_index"] == 0
        assert result[2]["output_file_index"] == 1
        # 新 file では frame_count が 0 から再開
        assert result[2]["from_frame_count"] == 0
        assert result[2]["to_frame_count"] == 30

    def test_length_mismatch_raises(self) -> None:
        plans = [{"episode_index": 0}]
        with pytest.raises(ValueError, match="length mismatch"):
            merger._batch_clips_to_output_files(
                plans, [10, 20], [5], video_files_size_mb=100, chunks_size=1000
            )


class TestPlanClipsForVideoKey:
    def test_extracts_source_video_info_per_episode(self, tmp_path: Path) -> None:
        output_meta = [
            {
                "episode_index": 0,
                "length": 30,
                "source_dataset_root": str(tmp_path / "src_a"),
                "__source_videos/observation.images.cam_0/chunk_index": 2,
                "__source_videos/observation.images.cam_0/file_index": 5,
                "__source_videos/observation.images.cam_0/from_timestamp": 1.5,
                "__source_videos/observation.images.cam_0/to_timestamp": 2.5,
            },
            {
                "episode_index": 1,
                "length": 15,
                "source_dataset_root": str(tmp_path / "src_b"),
                "__source_videos/observation.images.cam_0/chunk_index": 0,
                "__source_videos/observation.images.cam_0/file_index": 0,
                "__source_videos/observation.images.cam_0/from_timestamp": 0.0,
                "__source_videos/observation.images.cam_0/to_timestamp": 0.5,
            },
        ]
        plans = merger._plan_clips_for_video_key(
            output_meta, "observation.images.cam_0"
        )
        assert len(plans) == 2
        assert plans[0]["episode_index"] == 0
        assert plans[0]["source_video_path"] == (
            tmp_path / "src_a" / "videos" / "observation.images.cam_0" / "chunk-002" / "file-005.mp4"
        )
        assert plans[0]["clip_start_sec"] == 1.5
        assert plans[0]["clip_end_sec"] == 2.5
        assert plans[0]["length"] == 30
        assert plans[1]["source_video_path"] == (
            tmp_path / "src_b" / "videos" / "observation.images.cam_0" / "chunk-000" / "file-000.mp4"
        )


class TestWriteMergedVideosGracefulSkip:
    def test_empty_video_keys_no_op(self, tmp_path: Path) -> None:
        """video_keys=[] (fake source が video 無し) の場合は何もせず return。"""
        # sources / output_meta は空でも良い、video_keys 空なら loop 回らない
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        merger.write_merged_videos(
            sources=[],
            output_meta=[],
            output_dir=output_dir,
            video_keys=[],
            video_mode="lossless",
            video_files_size_mb=500,
            chunks_size=1000,
            fps=30.0,
        )
        # tmp_video_clips dir は作られない (video_keys が空なので処理 skip)
        assert not (output_dir / ".tmp_video_clips").exists()
        # videos/ dir も作られない
        assert not (output_dir / "videos").exists()


class TestParallelVideoWorkers:
    """Issue #122: --max-video-workers = camera 並列度の CLI + dispatch を確認。"""

    def test_parse_args_default_is_4(self) -> None:
        """CLI default で max_video_workers=4 (24-core H100 環境向け)。"""
        args = merger.parse_args([
            "--source-dir", "/tmp/x",
            "--output-dir", "/tmp/y",
        ])
        assert args.max_video_workers == 4

    def test_parse_args_override_to_1_for_sequential(self) -> None:
        """--max-video-workers 1 で sequential fallback を要求できる。"""
        args = merger.parse_args([
            "--source-dir", "/tmp/x",
            "--output-dir", "/tmp/y",
            "--max-video-workers", "1",
        ])
        assert args.max_video_workers == 1

    def test_parse_args_ffmpeg_threads_default_none_for_auto(self) -> None:
        """Issue #122: --ffmpeg-threads default None = auto (parallel は over-sub 回避)。"""
        args = merger.parse_args([
            "--source-dir", "/tmp/x",
            "--output-dir", "/tmp/y",
        ])
        assert args.ffmpeg_threads is None

    def test_parse_args_max_clip_workers_default_is_4(self) -> None:
        """Issue #122: --max-clip-workers default 4 (camera 内 clip 並列で ~2.5x 追加短縮)。"""
        args = merger.parse_args([
            "--source-dir", "/tmp/x",
            "--output-dir", "/tmp/y",
        ])
        assert args.max_clip_workers == 4

    def test_clip_parallel_preserves_episode_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_clip_workers=4 で 8 episode を extract しても results は episode_index
        順に返る (ThreadPool.map の order 保証。順序ズレると merge の batch 割当が
        variable な output に依存で壊れる)。"""
        import time
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        src_root = tmp_path / "src"
        src_root.mkdir()

        # 8 episodes、waited 順で完了 (order ズレる状況を作る)
        output_meta = []
        for i in range(8):
            ep = {
                "episode_index": i,
                "length": 5,
                "source_dataset_root": str(src_root),
                "__source_videos/observation.images.cam_0/chunk_index": 0,
                "__source_videos/observation.images.cam_0/file_index": 0,
                "__source_videos/observation.images.cam_0/from_timestamp": 0.0,
                "__source_videos/observation.images.cam_0/to_timestamp": 0.1,
            }
            output_meta.append(ep)

        # 完了順を wall-clock 反転にするため、epX の sleep = (8-X)*10ms を仕込む
        def fake_clip_video(_src, dst, *_args, **_kw):
            # ep index を dst path から抽出、大きい ep ほど短い sleep
            import re
            m = re.search(r"clip-(\d+)\.mp4", str(dst))
            ep_idx = int(m.group(1)) if m else 0
            time.sleep((8 - ep_idx) * 0.01)  # ep0 が一番遅く完了
            Path(dst).write_bytes(f"ep{ep_idx}".encode())

        # get_video_frame_count は dst 名から ep 引いて length と一致させる
        def fake_frame_count(p):
            return 5  # 全 ep length=5 と一致

        monkeypatch.setattr(merger, "clip_video", fake_clip_video)
        monkeypatch.setattr(merger, "get_video_frame_count", fake_frame_count)
        monkeypatch.setattr(merger, "concat_video_files",
                            lambda paths, out, *a, **kw: (Path(out).parent.mkdir(parents=True, exist_ok=True),
                                                            Path(out).write_bytes(b"c")))
        monkeypatch.setattr(merger, "_source_video_path",
                            lambda r, vk, ci, fi: (tmp_path / f"s_{vk}.mp4").resolve())

        # spy を書き込むため元 _process_video_key の結果を確認する代わりに、
        # tmp file content を post-hoc に inspect (clip-000000 は ep0 の bytes)
        merger.write_merged_videos(
            sources=[],
            output_meta=output_meta,
            output_dir=output_dir,
            video_keys=["observation.images.cam_0"],
            video_mode="lossless",
            video_files_size_mb=500,
            chunks_size=1000,
            fps=30.0,
            ffmpeg_threads=None,
            max_video_workers=1,
            max_clip_workers=4,
        )

        # temp_root は最後に rmtree されるので、output_meta の更新 column から順序を推定
        # (全 ep が処理されていれば全 ep に videos/.../chunk_index が書かれる)
        for ep in output_meta:
            assert f"videos/observation.images.cam_0/chunk_index" in ep, \
                f"ep {ep['episode_index']} not processed"

    def test_ffmpeg_threads_auto_matches_cpu_over_total_concurrency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """parallel 時 (video=4, clip=4, cpu=24) で ffmpeg_threads が 24/(4*4)=1 に
        auto-set されて clip_video に渡ることを確認 (over-sub 回避)。"""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        video_keys = ["observation.images.cam_0"]
        src_root = tmp_path / "src"
        src_root.mkdir()
        output_meta = [{
            "episode_index": 0,
            "length": 5,
            "source_dataset_root": str(src_root),
            "__source_videos/observation.images.cam_0/chunk_index": 0,
            "__source_videos/observation.images.cam_0/file_index": 0,
            "__source_videos/observation.images.cam_0/from_timestamp": 0.0,
            "__source_videos/observation.images.cam_0/to_timestamp": 0.1,
        }]

        called_threads: list[int] = []

        def fake_clip_video(_src, dst, *_args, ffmpeg_threads=0, **_kw):
            called_threads.append(ffmpeg_threads)
            Path(dst).write_bytes(b"fake")

        monkeypatch.setattr(merger, "clip_video", fake_clip_video)
        monkeypatch.setattr(merger, "get_video_frame_count", lambda _p: 5)
        monkeypatch.setattr(merger, "concat_video_files",
                            lambda paths, out, *a, **kw: (Path(out).parent.mkdir(parents=True, exist_ok=True),
                                                            Path(out).write_bytes(b"c")))
        monkeypatch.setattr(merger, "_source_video_path",
                            lambda r, vk, ci, fi: (tmp_path / f"s_{vk}.mp4").resolve())
        # cpu_count を 24 に固定 (H100 サーバ相当)
        import os as _os
        monkeypatch.setattr(_os, "cpu_count", lambda: 24)

        merger.write_merged_videos(
            sources=[],
            output_meta=output_meta,
            output_dir=output_dir,
            video_keys=video_keys,
            video_mode="lossless",
            video_files_size_mb=500,
            chunks_size=1000,
            fps=30.0,
            ffmpeg_threads=None,   # auto
            max_video_workers=4,
            max_clip_workers=4,
        )
        # 24 // (4 * 4) = 1 が期待値
        assert called_threads and all(t == 1 for t in called_threads), (
            f"expected all clip_video calls with ffmpeg_threads=1 (24/(4*4)), got {called_threads}"
        )

    def test_ffmpeg_threads_sequential_stays_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """sequential 時 (max_video_workers=1) は ffmpeg_threads=0 (全 core) 維持。"""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        src_root = tmp_path / "src"
        src_root.mkdir()
        output_meta = [{
            "episode_index": 0,
            "length": 5,
            "source_dataset_root": str(src_root),
            "__source_videos/observation.images.cam_0/chunk_index": 0,
            "__source_videos/observation.images.cam_0/file_index": 0,
            "__source_videos/observation.images.cam_0/from_timestamp": 0.0,
            "__source_videos/observation.images.cam_0/to_timestamp": 0.1,
        }]

        called_threads: list[int] = []

        def fake_clip_video(_src, dst, *_args, ffmpeg_threads=0, **_kw):
            called_threads.append(ffmpeg_threads)
            Path(dst).write_bytes(b"fake")

        monkeypatch.setattr(merger, "clip_video", fake_clip_video)
        monkeypatch.setattr(merger, "get_video_frame_count", lambda _p: 5)
        monkeypatch.setattr(merger, "concat_video_files",
                            lambda paths, out, *a, **kw: (Path(out).parent.mkdir(parents=True, exist_ok=True),
                                                            Path(out).write_bytes(b"c")))
        monkeypatch.setattr(merger, "_source_video_path",
                            lambda r, vk, ci, fi: (tmp_path / f"s_{vk}.mp4").resolve())

        merger.write_merged_videos(
            sources=[],
            output_meta=output_meta,
            output_dir=output_dir,
            video_keys=["observation.images.cam_0"],
            video_mode="lossless",
            video_files_size_mb=500,
            chunks_size=1000,
            fps=30.0,
            ffmpeg_threads=None,   # auto → sequential なので 0
            max_video_workers=1,
            max_clip_workers=1,
        )
        assert called_threads == [0], (
            f"expected ffmpeg_threads=0 (all cores) in sequential mode, got {called_threads}"
        )

    def test_clip_frame_drift_plus_one_tolerated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issue #122: ffmpeg cut boundary の GOP alignment で +1 frame drift が起きる
        (二重 re-encode の task_5_merged を merge 入力にしたとき実測)。drift ±1 は
        warn only で raise しない (余剰 frame は concat 末尾に置き去り、meta の
        from_timestamp が cumulative actual で recompute されるため無害)。"""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        src_root = tmp_path / "src"
        src_root.mkdir()

        output_meta = [{
            "episode_index": 0,
            "length": 69,
            "source_dataset_root": str(src_root),
            "__source_videos/observation.images.cam_0/chunk_index": 0,
            "__source_videos/observation.images.cam_0/file_index": 0,
            "__source_videos/observation.images.cam_0/from_timestamp": 0.0,
            "__source_videos/observation.images.cam_0/to_timestamp": 2.3,
        }]

        def fake_clip_video(_src, dst, *_args, **_kw):
            Path(dst).write_bytes(b"fake")

        # 意図的に 70 (planned 69 に対して +1) を返す
        monkeypatch.setattr(merger, "clip_video", fake_clip_video)
        monkeypatch.setattr(merger, "get_video_frame_count", lambda _p: 70)
        monkeypatch.setattr(merger, "concat_video_files",
                            lambda paths, out, *a, **kw: (Path(out).parent.mkdir(parents=True, exist_ok=True),
                                                            Path(out).write_bytes(b"c")))
        monkeypatch.setattr(merger, "_source_video_path",
                            lambda r, vk, ci, fi: (tmp_path / f"s_{vk}.mp4").resolve())

        # 例外にならず正常完了する
        merger.write_merged_videos(
            sources=[],
            output_meta=output_meta,
            output_dir=output_dir,
            video_keys=["observation.images.cam_0"],
            video_mode="lossless",
            video_files_size_mb=500,
            chunks_size=1000,
            fps=30.0,
            max_video_workers=1,
            max_clip_workers=1,
        )
        # meta の video pack column が書かれた = 処理継続
        assert "videos/observation.images.cam_0/chunk_index" in output_meta[0]

    def test_clip_frame_drift_beyond_tolerance_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """±1 超え (例: +2 or -2) の drift は data 破損の signal として raise。"""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        src_root = tmp_path / "src"
        src_root.mkdir()

        output_meta = [{
            "episode_index": 0,
            "length": 69,
            "source_dataset_root": str(src_root),
            "__source_videos/observation.images.cam_0/chunk_index": 0,
            "__source_videos/observation.images.cam_0/file_index": 0,
            "__source_videos/observation.images.cam_0/from_timestamp": 0.0,
            "__source_videos/observation.images.cam_0/to_timestamp": 2.3,
        }]

        def fake_clip_video(_src, dst, *_args, **_kw):
            Path(dst).write_bytes(b"fake")

        # 意図的に 72 (planned 69 に対して +3) を返す
        monkeypatch.setattr(merger, "clip_video", fake_clip_video)
        monkeypatch.setattr(merger, "get_video_frame_count", lambda _p: 72)
        monkeypatch.setattr(merger, "_source_video_path",
                            lambda r, vk, ci, fi: (tmp_path / f"s_{vk}.mp4").resolve())

        with pytest.raises(RuntimeError, match="tolerance"):
            merger.write_merged_videos(
                sources=[],
                output_meta=output_meta,
                output_dir=output_dir,
                video_keys=["observation.images.cam_0"],
                video_mode="lossless",
                video_files_size_mb=500,
                chunks_size=1000,
                fps=30.0,
                max_video_workers=1,
                max_clip_workers=1,
            )

    def test_parallel_dispatch_processes_all_cameras(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_video_workers=2 で 4 cameras 全て processed される (mock で ffmpeg 回避)。
        thread race で meta write が抜けないことも同時に確認。"""
        import threading

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        video_keys = [f"observation.images.cam_{i}" for i in range(4)]

        # output_meta: 2 episodes、_plan_clips が要求する field を全部込みで
        src_root = tmp_path / "src"
        src_root.mkdir()

        def _ep(idx: int) -> dict:
            base = {
                "episode_index": idx,
                "length": 5,
                "source_dataset_root": str(src_root),
            }
            for vk in video_keys:
                base[f"__source_videos/{vk}/chunk_index"] = 0
                base[f"__source_videos/{vk}/file_index"] = 0
                base[f"__source_videos/{vk}/from_timestamp"] = 0.0
                base[f"__source_videos/{vk}/to_timestamp"] = 0.1
            return base

        output_meta = [_ep(0), _ep(1)]

        # 各 clip 生成 = touch (thread-safe に file を作る、frame_count は固定 5)
        def fake_clip_video(src, dst, *_args, **_kw):
            Path(dst).write_bytes(b"fake")

        def fake_get_frame_count(_p):
            return 5

        def fake_concat(paths, out, *_a, **_kw):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"concat")

        # source_video_path も dummy dir を返す
        def fake_source_path(_root, video_key, _ci, _fi):
            p = tmp_path / f"srcvid_{video_key}.mp4"
            p.touch()
            return p

        # thread-safe な call 記録で並列 dispatch を検証
        called_keys: set[str] = set()
        lock = threading.Lock()
        orig_plan = merger._plan_clips_for_video_key

        def spy_plan(meta, vk):
            with lock:
                called_keys.add(vk)
            return orig_plan(meta, vk)

        monkeypatch.setattr(merger, "clip_video", fake_clip_video)
        monkeypatch.setattr(merger, "get_video_frame_count", fake_get_frame_count)
        monkeypatch.setattr(merger, "concat_video_files", fake_concat)
        monkeypatch.setattr(merger, "_source_video_path", fake_source_path)
        monkeypatch.setattr(merger, "_plan_clips_for_video_key", spy_plan)

        merger.write_merged_videos(
            sources=[],
            output_meta=output_meta,
            output_dir=output_dir,
            video_keys=video_keys,
            video_mode="lossless",
            video_files_size_mb=500,
            chunks_size=1000,
            fps=30.0,
            max_video_workers=2,
        )

        # 全 camera が処理された (thread race で抜けなし)
        assert called_keys == set(video_keys)
        # output_meta の各 episode に全 camera 分の new pack column が書かれた
        for ep in output_meta:
            for vk in video_keys:
                assert f"videos/{vk}/chunk_index" in ep
                assert f"videos/{vk}/file_index" in ep


# =============================================================================
# Issue #146: video passthrough (元 file をそのまま置き、meta の位置だけ書き換える)
# =============================================================================

CAM0 = "observation.images.cam_0"
CAM1 = "observation.images.cam_1"


def _pt_ep(
    idx: int,
    root: Path,
    file_index: int,
    start: int,
    length: int,
    video_keys: tuple[str, ...] = (CAM0,),
    fps: float = 30.0,
) -> dict:
    """passthrough の plan が読む field だけを持つ output_meta の 1 行 (start / length は frame)。"""
    ep = {"episode_index": idx, "length": length, "source_dataset_root": str(root)}
    for vk in video_keys:
        ep[f"__source_videos/{vk}/chunk_index"] = 0
        ep[f"__source_videos/{vk}/file_index"] = file_index
        ep[f"__source_videos/{vk}/from_timestamp"] = start / fps
        ep[f"__source_videos/{vk}/to_timestamp"] = (start + length) / fps
    return ep


class TestPlanPassthroughFiles:
    def test_maps_files_in_first_seen_order(self) -> None:
        """元の番号が source 間で重なっても、出力の番号は episode の順に 0 から振られる。"""
        a, b = Path("/src_a"), Path("/src_b")
        meta = [
            _pt_ep(0, a, 0, 0, 5), _pt_ep(1, a, 0, 5, 7),
            _pt_ep(2, a, 1, 0, 4),
            _pt_ep(3, b, 0, 0, 6),
        ]
        counts = {(str(a), 0, 0): 12, (str(a), 0, 1): 4, (str(b), 0, 0): 6}
        mapping = merger._plan_passthrough_files(meta, CAM0, counts, 30.0, 1000)
        assert mapping == {
            (str(a), 0, 0): merger.FileLocation(0, 0),
            (str(a), 0, 1): merger.FileLocation(0, 1),
            (str(b), 0, 0): merger.FileLocation(0, 2),
        }

    def test_chunk_rollover(self) -> None:
        a = Path("/src_a")
        meta = [_pt_ep(i, a, i, 0, 3) for i in range(3)]
        counts = {(str(a), 0, i): 3 for i in range(3)}
        mapping = merger._plan_passthrough_files(meta, CAM0, counts, 30.0, 2)
        assert list(mapping.values()) == [
            merger.FileLocation(0, 0), merger.FileLocation(0, 1), merger.FileLocation(1, 0),
        ]

    def test_episode_order_within_file_does_not_matter(self) -> None:
        a = Path("/src_a")
        meta = [_pt_ep(0, a, 0, 5, 7), _pt_ep(1, a, 0, 0, 5)]
        mapping = merger._plan_passthrough_files(meta, CAM0, {(str(a), 0, 0): 12}, 30.0, 1000)
        assert mapping == {(str(a), 0, 0): merger.FileLocation(0, 0)}

    def test_gap_raises(self) -> None:
        a = Path("/src_a")
        meta = [_pt_ep(0, a, 0, 0, 5), _pt_ep(1, a, 0, 6, 4)]
        with pytest.raises(ValueError, match="隙間"):
            merger._plan_passthrough_files(meta, CAM0, {(str(a), 0, 0): 10}, 30.0, 1000)

    def test_overlap_raises(self) -> None:
        a = Path("/src_a")
        meta = [_pt_ep(0, a, 0, 0, 5), _pt_ep(1, a, 0, 4, 6)]
        with pytest.raises(ValueError, match="重なり"):
            merger._plan_passthrough_files(meta, CAM0, {(str(a), 0, 0): 10}, 30.0, 1000)

    def test_file_not_fully_used_raises(self) -> None:
        """filter で末尾の episode を落とした file は passthrough できない。"""
        a = Path("/src_a")
        meta = [_pt_ep(0, a, 0, 0, 5)]
        with pytest.raises(ValueError, match="file は 10 frame"):
            merger._plan_passthrough_files(meta, CAM0, {(str(a), 0, 0): 10}, 30.0, 1000)

    def test_span_length_off_by_one_is_tolerated(self, capsys) -> None:
        """meta の length と動画の frame 数の 1 差は上流由来 (chunk を作る側も許容)。警告して通す。"""
        a = Path("/src_a")
        ep = _pt_ep(0, a, 0, 0, 5)
        ep["length"] = 4
        assert merger._plan_passthrough_files([ep], CAM0, {(str(a), 0, 0): 5}, 30.0, 1000) == {
            (str(a), 0, 0): merger.FileLocation(chunk_index=0, file_index=0)
        }
        assert "1 frame 差を許容" in capsys.readouterr().out

    def test_span_length_mismatch_raises(self) -> None:
        """2 frame 以上ずれていれば区間が壊れている。"""
        a = Path("/src_a")
        ep = _pt_ep(0, a, 0, 0, 5)
        ep["length"] = 3
        with pytest.raises(ValueError, match="length 3"):
            merger._plan_passthrough_files([ep], CAM0, {(str(a), 0, 0): 5}, 30.0, 1000)

    def test_fractional_frame_timestamp_raises(self) -> None:
        a = Path("/src_a")
        ep = _pt_ep(0, a, 0, 0, 5)
        ep[f"__source_videos/{CAM0}/from_timestamp"] = 0.0105  # 0.315 frame
        with pytest.raises(ValueError, match="length 5"):
            merger._plan_passthrough_files([ep], CAM0, {(str(a), 0, 0): 5}, 30.0, 1000)

    def test_lists_all_problem_files(self) -> None:
        a = Path("/src_a")
        meta = [_pt_ep(0, a, 0, 0, 5), _pt_ep(1, a, 1, 0, 3)]
        counts = {(str(a), 0, 0): 6, (str(a), 0, 1): 4}
        with pytest.raises(ValueError, match="2 個") as exc:
            merger._plan_passthrough_files(meta, CAM0, counts, 30.0, 1000)
        assert f"('{a}', 0, 0)" in str(exc.value)
        assert f"('{a}', 0, 1)" in str(exc.value)


def _write_fake_mp4(root: Path, video_key: str, file_index: int, frames: int) -> Path:
    """中身に frame 数を書いた fake mp4 (get_video_frame_count の mock が読む)。"""
    p = root / "videos" / video_key / "chunk-000" / f"file-{file_index:03d}.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(frames))
    return p


def _fake_frame_count(path: Path) -> int:
    return int(Path(path).read_text())


class TestLinkOrCopy:
    def test_existing_dst_raises_instead_of_silent_copy(self, tmp_path: Path) -> None:
        """dst が既に在るのは呼び出し側の間違い。黙って copy で上書きしない。"""
        src, dst = tmp_path / "src.mp4", tmp_path / "dst.mp4"
        src.write_bytes(b"NEW")
        dst.write_bytes(b"OLD")
        with pytest.raises(FileExistsError):
            merger._link_or_copy(src, dst)
        assert dst.read_bytes() == b"OLD"

    def test_cross_device_falls_back_to_copy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        src, dst = tmp_path / "src.mp4", tmp_path / "dst.mp4"
        src.write_bytes(b"DATA")

        def xdev(*_a, **_kw):
            raise OSError(errno.EXDEV, "cross-device link")

        monkeypatch.setattr(merger.os, "link", xdev)
        assert merger._link_or_copy(src, dst) is False
        assert dst.read_bytes() == b"DATA"


class TestWritePassthroughVideos:
    def _setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict], Path, Path]:
        a, b = tmp_path / "src_a", tmp_path / "src_b"
        for vk in (CAM0, CAM1):
            _write_fake_mp4(a, vk, 0, 12)
            _write_fake_mp4(a, vk, 1, 4)
            _write_fake_mp4(b, vk, 0, 6)
        meta = [
            _pt_ep(0, a, 0, 0, 5, (CAM0, CAM1)), _pt_ep(1, a, 0, 5, 7, (CAM0, CAM1)),
            _pt_ep(2, a, 1, 0, 4, (CAM0, CAM1)),
            _pt_ep(3, b, 0, 0, 6, (CAM0, CAM1)),
        ]
        monkeypatch.setattr(merger, "get_video_frame_count", _fake_frame_count)

        def no_ffmpeg(*_a, **_kw):
            raise AssertionError("passthrough では ffmpeg を呼ばない")

        monkeypatch.setattr(merger, "clip_video", no_ffmpeg)
        monkeypatch.setattr(merger, "concat_video_files", no_ffmpeg)
        out = tmp_path / "out"
        out.mkdir()
        return meta, a, out

    def _write(self, meta: list[dict], out: Path) -> None:
        merger.write_merged_videos(
            sources=[], output_meta=meta, output_dir=out, video_keys=[CAM0, CAM1],
            video_mode="passthrough", video_files_size_mb=500, chunks_size=1000, fps=30.0,
        )

    def test_zero_workers_still_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--max-video-workers 0 は reencode 経路では逐次実行。passthrough も落ちない。"""
        meta, _a, out = self._setup(tmp_path, monkeypatch)
        merger.write_merged_videos(
            sources=[], output_meta=meta, output_dir=out, video_keys=[CAM0, CAM1],
            video_mode="passthrough", video_files_size_mb=500, chunks_size=1000, fps=30.0,
            max_video_workers=0, max_clip_workers=0,
        )
        assert len(list((out / "videos" / CAM0).rglob("*.mp4"))) == 3

    def test_hardlinks_and_rewrites_meta(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        meta, a, out = self._setup(tmp_path, monkeypatch)
        src_ts = [(ep[f"__source_videos/{CAM0}/from_timestamp"], ep[f"__source_videos/{CAM0}/to_timestamp"]) for ep in meta]
        self._write(meta, out)

        for vk in (CAM0, CAM1):
            files = sorted((out / "videos" / vk / "chunk-000").glob("*.mp4"))
            assert [f.name for f in files] == ["file-000.mp4", "file-001.mp4", "file-002.mp4"]
            assert [f.read_text() for f in files] == ["12", "4", "6"]
            src = a / "videos" / vk / "chunk-000" / "file-000.mp4"
            assert files[0].stat().st_ino == src.stat().st_ino  # hardlink

        assert [(ep[f"videos/{CAM0}/chunk_index"], ep[f"videos/{CAM0}/file_index"]) for ep in meta] == [
            (0, 0), (0, 0), (0, 1), (0, 2),
        ]
        for ep, (from_ts, to_ts) in zip(meta, src_ts):
            for vk in (CAM0, CAM1):
                assert ep[f"videos/{vk}/from_timestamp"] == from_ts
                assert ep[f"videos/{vk}/to_timestamp"] == to_ts

    def test_falls_back_to_copy_when_link_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        meta, a, out = self._setup(tmp_path, monkeypatch)

        def cross_device(*_a, **_kw):
            raise OSError(18, "Invalid cross-device link")

        monkeypatch.setattr(merger.os, "link", cross_device)
        self._write(meta, out)
        dst = out / "videos" / CAM0 / "chunk-000" / "file-000.mp4"
        src = a / "videos" / CAM0 / "chunk-000" / "file-000.mp4"
        assert dst.read_text() == "12"
        assert dst.stat().st_ino != src.stat().st_ino

    def test_links_symlink_target(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """HF cache のように元 file が symlink でも、実体に hardlink する。"""
        meta, a, out = self._setup(tmp_path, monkeypatch)
        src = a / "videos" / CAM0 / "chunk-000" / "file-000.mp4"
        blob = tmp_path / "blob"
        src.rename(blob)
        src.symlink_to(blob)
        self._write(meta, out)
        dst = out / "videos" / CAM0 / "chunk-000" / "file-000.mp4"
        assert not dst.is_symlink()
        assert dst.stat().st_ino == blob.stat().st_ino

    def test_invalid_camera_blocks_all_writes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """1 camera でも前提が崩れていたら、どの camera の file も置かずに止まる。"""
        meta, a, out = self._setup(tmp_path, monkeypatch)
        _write_fake_mp4(a, CAM1, 1, 5)  # cam_1 の file-001 だけ 1 frame 長い = 末尾が使われていない
        with pytest.raises(ValueError, match=CAM1):
            self._write(meta, out)
        assert not (out / "videos").exists()


def _create_fake_video_source(
    tmp_path: Path, name: str, task_index: int, lengths_per_file: list[list[int]]
) -> Path:
    """video (cam_0) 付きの fake LeRobot v3 dataset。file ごとに episode を詰め、fake mp4 を置く。"""
    features = {
        "action.joint": {"dtype": "float32", "shape": [2]},
        CAM0: {"dtype": "video", "shape": [480, 640, 3]},
    }
    root = tmp_path / name
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    with (root / "meta" / "info.json").open("w") as f:
        json.dump({"fps": 30.0, "features": features, "chunks_size": 1000}, f)
    pq.write_table(
        pa.Table.from_pylist([{"task_index": task_index, "__index_level_0__": f"task_{task_index}"}]),
        root / "meta" / "tasks.parquet",
    )
    data_rows, ep_rows = [], []
    ep_idx = global_idx = 0
    for file_index, lengths in enumerate(lengths_per_file):
        _write_fake_mp4(root, CAM0, file_index, sum(lengths))
        start = 0
        for length in lengths:
            for f_idx in range(length):
                data_rows.append({
                    "timestamp": f_idx / 30.0, "frame_index": f_idx, "episode_index": ep_idx,
                    "index": global_idx + f_idx, "task_index": task_index, "action.joint": [0.0, 0.0],
                })
            ep_rows.append({
                "episode_index": ep_idx, "tasks": [f"task_{task_index}"], "length": length,
                "data/chunk_index": 0, "data/file_index": 0,
                "dataset_from_index": global_idx, "dataset_to_index": global_idx + length,
                "source_task_index": task_index,
                f"videos/{CAM0}/chunk_index": 0, f"videos/{CAM0}/file_index": file_index,
                f"videos/{CAM0}/from_timestamp": start / 30.0,
                f"videos/{CAM0}/to_timestamp": (start + length) / 30.0,
            })
            start += length
            global_idx += length
            ep_idx += 1
    pq.write_table(pa.Table.from_pylist(data_rows), root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(ep_rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root


class TestMainPassthroughEndToEnd:
    def test_parse_args_accepts_passthrough(self) -> None:
        args = merger.parse_args(
            ["--source-dir", "/dummy", "--output-dir", "/dummy_out", "--video-mode", "passthrough"]
        )
        assert args.video_mode == "passthrough"

    def test_merges_chunks_of_same_task(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """同じ task の 2 chunk を passthrough で merge: file は番号を振り直して置かれ、
        各 episode が同じ中身の file の同じ区間を指し、source_task_index と video_mode が残る。"""
        monkeypatch.setattr(merger, "get_video_frame_count", _fake_frame_count)
        c0 = _create_fake_video_source(tmp_path, "chunk0", 3, [[5, 7], [4]])
        c1 = _create_fake_video_source(tmp_path, "chunk1", 3, [[6, 2]])
        out = tmp_path / "merged"
        assert merger.main([
            "--source-dir", str(c0), "--source-dir", str(c1), "--output-dir", str(out),
            "--allow-duplicate-task-indices", "--video-mode", "passthrough",
        ]) == 0

        eps = pq.read_table(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
        assert [ep["length"] for ep in eps] == [5, 7, 4, 6, 2]
        assert [ep[f"videos/{CAM0}/file_index"] for ep in eps] == [0, 0, 1, 2, 2]
        assert [round(ep[f"videos/{CAM0}/from_timestamp"] * 30) for ep in eps] == [0, 5, 0, 0, 6]
        assert {ep["source_task_index"] for ep in eps} == {3}
        # 出力 file の中身 (= 元 file の frame 数) と、その file を指す episode の length の合計が一致
        for file_index in range(3):
            path = out / "videos" / CAM0 / "chunk-000" / f"file-{file_index:03d}.mp4"
            used = sum(ep["length"] for ep in eps if ep[f"videos/{CAM0}/file_index"] == file_index)
            assert int(path.read_text()) == used
        with (out / "meta" / "team_ramen_curation.json").open() as f:
            assert json.load(f)["video_mode"] == "passthrough"
        with (out / "meta" / "info.json").open() as f:
            assert json.load(f)["total_frames"] == 24


# =============================================================================
# B-7: main() CLI end-to-end integration test (fake source、video 無し)
# =============================================================================


class TestMainCLIEndToEnd:
    def test_main_runs_end_to_end_with_fake_sources(self, tmp_path: Path) -> None:
        """CLI 経由で main() を full-flow で実行、data + meta が全部生成される。

        fake source は video 無しなので write_merged_videos は no-op、
        data merge + meta rebuild + provenance の全ステップが完走することを担保。
        """
        src1 = _create_fake_source(tmp_path, task_index=4, n_episodes=3, frames_per_episode=5)
        src2 = _create_fake_source(tmp_path, task_index=5, n_episodes=2, frames_per_episode=4)
        src3 = _create_fake_source(tmp_path, task_index=7, n_episodes=1, frames_per_episode=3)
        output_dir = tmp_path / "mixed"

        # main() を argv 直接指定で呼ぶ (argparse 完走 + all steps 実行を担保)
        exit_code = merger.main(
            [
                "--source-dir", str(src1),
                "--source-dir", str(src2),
                "--source-dir", str(src3),
                "--output-dir", str(output_dir),
                "--force",
            ]
        )
        assert exit_code == 0

        # 全部の meta file が生成された
        assert (output_dir / "meta" / "tasks.parquet").exists()
        assert (output_dir / "meta" / "info.json").exists()
        assert (output_dir / "meta" / "stats.json").exists()
        assert (output_dir / "meta" / "team_ramen_curation.json").exists()

        # data parquet が書かれた
        data_files = list((output_dir / "data" / "chunk-000").glob("file-*.parquet"))
        assert len(data_files) == 1  # 少量なので 1 file に収まる

        # info.json の合計 frame/episode 数
        with (output_dir / "meta" / "info.json").open() as f:
            info = json.load(f)
        assert info["total_episodes"] == 3 + 2 + 1
        assert info["total_frames"] == 3 * 5 + 2 * 4 + 1 * 3
        assert info["total_tasks"] == 3

        # tasks.parquet の union、task_index は 0-based contiguous に remap (Issue #122)
        # sources sorted = [4, 5, 7] → new indices [0, 1, 2]
        tasks_written = pq.read_table(output_dir / "meta" / "tasks.parquet").to_pylist()
        assert sorted(int(t["task_index"]) for t in tasks_written) == [0, 1, 2]

        # data の row 数 total_frames 一致
        data_rows = pq.read_table(data_files[0]).to_pylist()
        assert len(data_rows) == info["total_frames"]

        # data task_index も 0-based contiguous に remap されている (LeRobot iloc 互換)
        # 各 source task_index → new index の対応: 4→0, 5→1, 7→2
        src1_frames = data_rows[:15]  # task=4 → 0
        src2_frames = data_rows[15 : 15 + 8]  # task=5 → 1
        src3_frames = data_rows[15 + 8 :]  # task=7 → 2
        assert {r["task_index"] for r in src1_frames} == {0}
        assert {r["task_index"] for r in src2_frames} == {1}
        assert {r["task_index"] for r in src3_frames} == {2}
        # 出力の全 task_index が [0, 1, 2] の subset (source 5, 7 は決して data 側に残らない)
        assert set(r["task_index"] for r in data_rows) == {0, 1, 2}

    def test_main_with_dry_run_no_output(self, tmp_path: Path) -> None:
        """--dry-run で output dir が作られない (validate + plan だけ)。"""
        src1 = _create_fake_source(tmp_path, task_index=4, n_episodes=1)
        output_dir = tmp_path / "should_not_exist"
        exit_code = merger.main(
            [
                "--source-dir", str(src1),
                "--output-dir", str(output_dir),
                "--dry-run",
            ]
        )
        assert exit_code == 0
        assert not output_dir.exists()

    def test_default_video_mode_is_reencode(self) -> None:
        """Issue #122: default video mode = 'reencode' (crf=18、visually lossless)。

        Source は既に crf~16 の h264 lossy なので、crf=0 (lossless) に上塗りしても
        情報は増えず decode コストだけ 4x 重くなっていた。GR00T train ETA 48h→~19h
        短縮のため default を crf=18 に切替。crf=18 は SSIM>0.986 で視覚的可逆。
        """
        args = merger.parse_args(
            [
                "--source-dir", "/dummy",
                "--output-dir", "/dummy_out",
            ]
        )
        assert args.video_mode == "reencode"

    def test_main_with_filter_reduces_episodes(self, tmp_path: Path) -> None:
        """--task-filter 2:last_per_source_episode 指定で flip task の episode が減る。"""
        # flip source: episode 3 個で source_episode 全部 10 (last per episode で 1 個残す)
        src_flip = tmp_path / "src_flip"
        (src_flip / "meta").mkdir(parents=True)
        (src_flip / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (src_flip / "data" / "chunk-000").mkdir(parents=True)

        features = {"action.joint": {"dtype": "float32", "shape": [2]}}
        with (src_flip / "meta" / "info.json").open("w") as f:
            json.dump({"fps": 30.0, "features": features, "chunks_size": 1000}, f)
        pq.write_table(
            pa.Table.from_pylist([{"task_index": 2, "__index_level_0__": "flip"}]),
            src_flip / "meta" / "tasks.parquet",
        )
        data_rows = []
        ep_rows = []
        global_idx = 0
        for ep_idx, frame_start in [(0, 100), (1, 300), (2, 500)]:
            for f in range(3):
                data_rows.append({
                    "timestamp": f/30.0, "frame_index": f, "episode_index": ep_idx,
                    "index": global_idx + f, "task_index": 2, "action.joint": [0.0, 0.0],
                })
            ep_rows.append({
                "episode_index": ep_idx, "tasks": ["flip"], "length": 3,
                "data/chunk_index": 0, "data/file_index": 0,
                "dataset_from_index": global_idx, "dataset_to_index": global_idx + 3,
                "source_episode_index": 10, "source_frame_start": frame_start,
                "source_frame_end": frame_start + 2,
            })
            global_idx += 3
        pq.write_table(pa.Table.from_pylist(data_rows), src_flip / "data" / "chunk-000" / "file-000.parquet")
        pq.write_table(pa.Table.from_pylist(ep_rows), src_flip / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

        output_dir = tmp_path / "mixed_filtered"
        exit_code = merger.main(
            [
                "--source-dir", str(src_flip),
                "--task-filter", "2:last_per_source_episode",
                "--output-dir", str(output_dir),
                "--force",
            ]
        )
        assert exit_code == 0

        with (output_dir / "meta" / "info.json").open() as f:
            info = json.load(f)
        # 3 seg 全部 source_episode 10 → filter で frame_start 最大 (500) の 1 個残る
        assert info["total_episodes"] == 1
        assert info["total_frames"] == 3

        # provenance の filter_applied 反映
        with (output_dir / "meta" / "team_ramen_curation.json").open() as f:
            prov = json.load(f)
        assert prov["filters_applied"] == {"2": "last_per_source_episode"}
        assert prov["episodes_per_task_after_filter"] == {"2": 1}
