"""Issue #122 recipe test: WarmupCosineScheduler, EMA, ImageAugPipeline, split_train_val_test。

- WarmupCosineScheduler: warmup 期の線形 up、cosine 期の end_lr 到達、境界一致
- EMA: shadow 更新 = decay 加重平均、apply/restore で weight swap 復元、frozen skip
- ImageAugPipeline: enabled=false は Normalize のみ、enabled=true で head 強 wrist 弱、
  prev+current に同 aug params (fork_rng 保証)
- split_train_val_test: 全 index 網羅、重複なし、seed 固定で再現、ratio 尊重
- RamenOriSplitView: is_train フラグが _transform_item に伝達される
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from model.ramen_ori.data_lerobot import (
    ImageAugPipeline,
    RamenOriLerobotDataset,
    RamenOriSplitView,
    split_episode_indices,
    split_train_val_test,
)
from model.ramen_ori.train import EMA, WarmupCosineScheduler, _resolve_autocast_dtype

# reuse FakeLeRobotBase from test_data_lerobot
from model.ramen_ori.tests.test_data_lerobot import FakeLeRobotBase


# ============================================================================
# WarmupCosineScheduler
# ============================================================================


def _dummy_optim(lr: float = 1e-4) -> torch.optim.Optimizer:
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([p], lr=lr)


class TestWarmupCosineScheduler:
    def test_warmup_starts_at_almost_zero(self):
        opt = _dummy_optim(lr=1e-4)
        sched = WarmupCosineScheduler(opt, base_lr=1e-4, warmup_steps=100, max_steps=1000, end_lr=1e-5)
        # step 0 => lr = base_lr / warmup = tiny
        lr0 = sched.get_lr(0)
        assert lr0 == pytest.approx(1e-4 / 100)

    def test_warmup_reaches_base_lr_at_boundary(self):
        opt = _dummy_optim(lr=1e-4)
        sched = WarmupCosineScheduler(opt, base_lr=1e-4, warmup_steps=100, max_steps=1000, end_lr=1e-5)
        # step = warmup - 1 => lr = base_lr * warmup / warmup = base_lr
        lr = sched.get_lr(99)
        assert lr == pytest.approx(1e-4)

    def test_cosine_starts_at_base_lr(self):
        opt = _dummy_optim(lr=1e-4)
        sched = WarmupCosineScheduler(opt, base_lr=1e-4, warmup_steps=100, max_steps=1000, end_lr=1e-5)
        # step = warmup => cosine t=0 => lr = base_lr
        lr = sched.get_lr(100)
        assert lr == pytest.approx(1e-4)

    def test_cosine_ends_at_end_lr(self):
        opt = _dummy_optim(lr=1e-4)
        sched = WarmupCosineScheduler(opt, base_lr=1e-4, warmup_steps=100, max_steps=1000, end_lr=1e-5)
        lr = sched.get_lr(999)
        # ほぼ end_lr、cosine 端の精度は 5% 以内 (最終 step の代入誤差)
        assert lr == pytest.approx(1e-5, rel=0.05)

    def test_step_advances_and_updates_optim_lr(self):
        opt = _dummy_optim(lr=1e-4)
        sched = WarmupCosineScheduler(opt, base_lr=1e-4, warmup_steps=10, max_steps=100, end_lr=1e-5)
        lr0 = sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(lr0)
        assert sched._step == 1
        lr1 = sched.step()
        assert lr1 > lr0  # warmup 期は上昇

    def test_invalid_config_raises(self):
        opt = _dummy_optim(lr=1e-4)
        with pytest.raises(ValueError, match="need max"):
            WarmupCosineScheduler(opt, base_lr=1e-4, warmup_steps=100, max_steps=100, end_lr=1e-5)


# ============================================================================
# EMA
# ============================================================================


class _TinyModel(torch.nn.Module):
    def __init__(self, freeze_first: bool = False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(4))
        self.b = torch.nn.Parameter(torch.ones(4) * 2.0)
        if freeze_first:
            self.a.requires_grad = False


class TestEMA:
    def test_shadow_only_covers_learnable(self):
        m = _TinyModel(freeze_first=True)
        ema = EMA(m, decay=0.9)
        assert "b" in ema.shadow
        assert "a" not in ema.shadow  # frozen skip

    def test_update_is_weighted_average(self):
        m = _TinyModel(freeze_first=False)
        ema = EMA(m, decay=0.9)
        # 初期 shadow = zeros / twos (deep copy)
        assert torch.allclose(ema.shadow["a"], torch.zeros(4))
        # a を 1 に変えて update: shadow_a = 0.9 * 0 + 0.1 * 1 = 0.1
        m.a.data.fill_(1.0)
        ema.update(m)
        assert torch.allclose(ema.shadow["a"], torch.full((4,), 0.1))

    def test_apply_and_restore_swaps_correctly(self):
        m = _TinyModel(freeze_first=False)
        ema = EMA(m, decay=0.9)
        # shadow を 100 に強制、apply で m.a が 100 になり、restore で 0 に戻る
        ema.shadow["a"].fill_(100.0)
        m.a.data.fill_(0.0)
        ema.apply_to(m)
        assert torch.allclose(m.a.data, torch.full((4,), 100.0))
        ema.restore(m)
        assert torch.allclose(m.a.data, torch.zeros(4))

    def test_context_manager(self):
        m = _TinyModel(freeze_first=False)
        ema = EMA(m, decay=0.9)
        ema.shadow["a"].fill_(50.0)
        m.a.data.fill_(1.0)
        with ema.applied(m):
            assert torch.allclose(m.a.data, torch.full((4,), 50.0))
        assert torch.allclose(m.a.data, torch.full((4,), 1.0))

    def test_double_apply_raises(self):
        m = _TinyModel()
        ema = EMA(m, decay=0.9)
        ema.apply_to(m)
        with pytest.raises(RuntimeError, match="twice"):
            ema.apply_to(m)

    def test_state_dict_round_trip(self):
        m = _TinyModel()
        ema = EMA(m, decay=0.9)
        ema.shadow["b"].fill_(7.0)
        state = ema.state_dict()
        # 別 EMA インスタンスへ load
        m2 = _TinyModel()
        ema2 = EMA(m2, decay=0.9)
        ema2.load_state_dict(state)
        assert torch.allclose(ema2.shadow["b"], torch.full((4,), 7.0))


# ============================================================================
# ImageAugPipeline
# ============================================================================


def _cfg_head_wrist(enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "wrist_key_substring": "wrist",
        "head": {
            "color_jitter": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.02},
            "gaussian_blur": {"kernel_size": 3, "sigma_min": 0.1, "sigma_max": 1.0, "p": 1.0},
        },
        "wrist": {
            "color_jitter": {"brightness": 0.05, "contrast": 0.05, "saturation": 0.05, "hue": 0.005},
            "gaussian_blur": {"kernel_size": 3, "sigma_min": 0.1, "sigma_max": 0.5, "p": 1.0},
        },
        "random_erasing": {"p": 0.0, "scale_min": 0.02, "scale_max": 0.15},  # p=0 で無効化 (test 決定性)
    }


class TestImageAugPipeline:
    """Issue #129 Phase G (2026-08-31): apply が (imgs, affine) tuple を返す形に refactor 済。
    affine は wrist / geometric disabled で identity (2, 3)。"""

    def test_disabled_only_normalizes(self):
        pipe = ImageAugPipeline(None)
        img_2 = torch.ones(2, 3, 8, 8) * 0.5
        out, affine = pipe.apply(img_2, "observation.images.cam_0", idx=0, is_train=True)
        assert out.shape == (2, 3, 8, 8)
        # 0.5 が Normalize (mean=0.485, std=0.229) 後は近い値
        expected_ch0 = (0.5 - 0.485) / 0.229
        assert out[0, 0].mean().item() == pytest.approx(expected_ch0, abs=1e-4)
        # aug disabled では affine は identity
        assert torch.allclose(affine, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_val_mode_skips_aug(self):
        """val mode は plain Normalize のみ、aug params 適用なし。"""
        from torchvision import transforms
        pipe = ImageAugPipeline(_cfg_head_wrist(enabled=True))
        img_2 = torch.rand(2, 3, 16, 16)
        val_out, val_affine = pipe.apply(img_2, "observation.images.cam_head", idx=1, is_train=False)
        # val 出力 == 生画像を素直に Normalize したもの
        normalize = transforms.Normalize(mean=pipe.mean, std=pipe.std)
        expected = torch.stack([normalize(img_2[0]), normalize(img_2[1])], dim=0)
        assert torch.allclose(val_out, expected)
        # val は affine identity
        assert torch.allclose(val_affine, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        # train 版は aug で変化、val とは異なる
        train_out, _ = pipe.apply(img_2, "observation.images.cam_head", idx=1, is_train=True)
        assert not torch.allclose(train_out, val_out)

    def test_head_and_wrist_are_dispatched(self):
        pipe = ImageAugPipeline(_cfg_head_wrist(enabled=True))
        assert pipe.is_wrist("observation.images.cam_wrist_left")
        assert not pipe.is_wrist("observation.images.cam_head")

    def test_prev_and_current_same_color_seed(self):
        """delta encoder 保護: prev + current が同じ ColorJitter params で aug されること。

        ColorJitter は同 seed で同 param → constant image に適用すれば prev == current。
        """
        pipe = ImageAugPipeline(_cfg_head_wrist(enabled=True))
        # 2 frame とも constant (同 value)、GaussianBlur は input が constant なら output も constant
        img_2 = torch.ones(2, 3, 16, 16) * 0.5
        out, _ = pipe.apply(img_2, "observation.images.cam_head", idx=42, is_train=True)
        # RandomErasing p=0 なので変化なし → prev/current の Aug 出力は完全一致すべき
        assert torch.allclose(out[0], out[1]), (
            f"prev/current aug differ (max abs delta = {(out[0]-out[1]).abs().max().item():.4f})"
        )

    def test_different_idx_produces_different_aug(self):
        """異なる sample には異なる aug seed → 異なる結果。"""
        pipe = ImageAugPipeline(_cfg_head_wrist(enabled=True))
        img_2 = torch.rand(2, 3, 16, 16)
        out_a, _ = pipe.apply(img_2, "observation.images.cam_head", idx=1, is_train=True)
        out_b, _ = pipe.apply(img_2, "observation.images.cam_head", idx=999, is_train=True)
        assert not torch.allclose(out_a, out_b)


# ============================================================================
# split_train_val_test
# ============================================================================


class TestSplitTrainValTest:
    def test_coverage_and_disjointness(self):
        ep_lengths = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int64)
        train, val, test = split_train_val_test(ep_lengths, val_ratio=0.2, test_ratio=0.2, seed=42)
        all_indices = set(train) | set(val) | set(test)
        assert all_indices == set(range(int(ep_lengths.sum())))
        # disjoint
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_seed_deterministic(self):
        ep_lengths = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        a = split_train_val_test(ep_lengths, seed=7)
        b = split_train_val_test(ep_lengths, seed=7)
        assert a == b
        # 別 seed で違う結果
        c = split_train_val_test(ep_lengths, seed=8)
        assert a != c

    def test_episode_boundaries_not_broken(self):
        """episode 3 の全 frame が同じ split (train/val/test 内でまとまる) こと。"""
        ep_lengths = np.array([5, 5, 5, 5, 5], dtype=np.int64)
        train, val, test = split_train_val_test(ep_lengths, val_ratio=0.2, test_ratio=0.2, seed=42)
        for split in (train, val, test):
            for i in range(0, 25, 5):
                ep_i_frames = set(range(i, i + 5))
                # 各 episode は 5 frame まとまり → split と交わるなら全 frame 内包
                intersect = ep_i_frames & set(split)
                assert len(intersect) == 0 or len(intersect) == 5

    def test_too_few_episodes_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            split_train_val_test(np.array([10, 10], dtype=np.int64))


class TestSplitEpisodeIndices:
    """Issue #122: episode-level split (GR00T apples-to-apples 用)。"""

    def test_returns_disjoint_and_covers_all(self):
        train, val, test = split_episode_indices(10, val_ratio=0.2, test_ratio=0.2, seed=42)
        assert set(train) | set(val) | set(test) == set(range(10))
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0

    def test_seed_deterministic(self):
        a = split_episode_indices(20, seed=7)
        b = split_episode_indices(20, seed=7)
        assert a == b

    def test_sorted_output(self):
        """Episode indices は昇順で返る (JSON schema と一致)。"""
        train, val, test = split_episode_indices(20, seed=99)
        assert train == sorted(train)
        assert val == sorted(val)
        assert test == sorted(test)

    def test_consistent_with_frame_level_split(self):
        """split_episode_indices と split_train_val_test が同じ episode 割当を使う (Issue #122 apples-to-apples 保証)。"""
        ep_lengths = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=np.int64)
        train_eps, val_eps, test_eps = split_episode_indices(
            len(ep_lengths), val_ratio=0.2, test_ratio=0.2, seed=42
        )
        train_frames, val_frames, test_frames = split_train_val_test(
            ep_lengths, val_ratio=0.2, test_ratio=0.2, seed=42
        )
        # Frame level split から episode assignments を逆算 (frame -> episode)
        cursor = 0
        frame_to_ep: list[int] = []
        for ep_i, length in enumerate(ep_lengths):
            frame_to_ep.extend([ep_i] * int(length))
            cursor += int(length)
        derived_train_eps = sorted(set(frame_to_ep[i] for i in train_frames))
        derived_val_eps = sorted(set(frame_to_ep[i] for i in val_frames))
        derived_test_eps = sorted(set(frame_to_ep[i] for i in test_frames))
        assert derived_train_eps == train_eps
        assert derived_val_eps == val_eps
        assert derived_test_eps == test_eps

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            split_episode_indices(2)


class TestSplitSourceGrouped:
    """Issue #122 fix: source_episode_name grouping による leakage-safe split。

    resolve_training_split.py が要求する source_episode_names を満たすため、
    task 5+7 combined (同一 raw 収録から複数 curated segment が抽出されるケース) で
    train/val 間の source recording leakage を排除する。
    """

    def test_no_source_leakage(self):
        """同一 source_episode_name は同一 split に集約される (leakage-safe の中核性質)。"""
        # 10 raw sources → 30 episodes (各 raw から 3 curated segment)
        source_names = [f"src{i:03d}" for i in range(10) for _ in range(3)]
        train, val, test = split_episode_indices(
            30, val_ratio=0.2, test_ratio=0.2, seed=42,
            source_episode_names=source_names,
        )
        train_sources = {source_names[i] for i in train}
        val_sources = {source_names[i] for i in val}
        test_sources = {source_names[i] for i in test}
        assert train_sources & val_sources == set()
        assert train_sources & test_sources == set()
        assert val_sources & test_sources == set()
        # 全 source が どこかの split には入る
        assert train_sources | val_sources | test_sources == set(source_names)

    def test_all_episodes_from_same_source_stay_together(self):
        """1 source の全 episode が同一 split (segments 分割禁止)。"""
        source_names = [f"src{i:03d}" for i in range(6) for _ in range(4)]  # 24 eps / 6 src
        train, val, test = split_episode_indices(
            24, val_ratio=0.2, test_ratio=0.2, seed=1,
            source_episode_names=source_names,
        )
        for split_indices in (train, val, test):
            per_source_counts: dict[str, int] = {}
            for idx in split_indices:
                per_source_counts[source_names[idx]] = per_source_counts.get(source_names[idx], 0) + 1
            for src, cnt in per_source_counts.items():
                assert cnt == 4, f"source {src} was split across boundaries: {cnt} episodes in this split"

    def test_seed_deterministic(self):
        source_names = [f"src{i:03d}" for i in range(20) for _ in range(2)]
        a = split_episode_indices(40, seed=7, source_episode_names=source_names)
        b = split_episode_indices(40, seed=7, source_episode_names=source_names)
        assert a == b
        c = split_episode_indices(40, seed=8, source_episode_names=source_names)
        assert a != c

    def test_sorted_output(self):
        source_names = [f"src{i:03d}" for i in range(10) for _ in range(2)]
        train, val, test = split_episode_indices(20, seed=99, source_episode_names=source_names)
        assert train == sorted(train)
        assert val == sorted(val)
        assert test == sorted(test)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            split_episode_indices(10, source_episode_names=["a", "b", "c"])

    def test_too_few_groups_raises(self):
        """distinct source_names が 3 未満なら error (train/val/test 3 way split 不可能)。"""
        # 6 eps だが source は 2 種類しか無い
        source_names = ["src0"] * 3 + ["src1"] * 3
        with pytest.raises(ValueError, match="at least 3"):
            split_episode_indices(6, source_episode_names=source_names)

    def test_empty_name_rejected(self):
        source_names = [f"src{i:03d}" for i in range(9)] + [""]
        with pytest.raises(ValueError, match="must be non-empty"):
            split_episode_indices(10, source_episode_names=source_names)

    def test_frame_level_split_matches_source_grouped(self):
        """split_train_val_test も source_episode_names を伝達すれば同じ episode 割当になる。"""
        source_names = [f"src{i:03d}" for i in range(8) for _ in range(2)]
        ep_lengths = np.array([5] * 16, dtype=np.int64)
        train_eps, val_eps, test_eps = split_episode_indices(
            16, val_ratio=0.25, test_ratio=0.25, seed=42,
            source_episode_names=source_names,
        )
        train_frames, val_frames, test_frames = split_train_val_test(
            ep_lengths, val_ratio=0.25, test_ratio=0.25, seed=42,
            source_episode_names=source_names,
        )
        # Frame → episode 逆算
        frame_to_ep = [ep_i for ep_i, length in enumerate(ep_lengths) for _ in range(int(length))]
        assert sorted({frame_to_ep[i] for i in train_frames}) == train_eps
        assert sorted({frame_to_ep[i] for i in val_frames}) == val_eps
        assert sorted({frame_to_ep[i] for i in test_frames}) == test_eps


# ============================================================================
# sample_episode_metadata + build_split_payload (round-trip)
# ============================================================================


class _FakeSub:
    """RamenOriLerobotDataset._sub_datasets の meta.episodes 部分だけ mock する fake sub。"""

    def __init__(self, rows: list[dict]) -> None:
        self.meta = type("M", (), {"episodes": rows})()


def _make_fake_dataset_with_source_indices(source_indices_per_sub: list[list[int]]):
    """`RamenOriLerobotDataset` を DI で作り、_sub_datasets を差し替える。

    Args:
        source_indices_per_sub: 各 sub_dataset に含まれる episode の source_episode_index list
    """
    # base_dataset は使わないので dummy (RamenOriLerobotDataset の __getitem__ を叩かない)
    fake_base = FakeLeRobotBase(num_samples=1, chunk_len=4, img_size=32)
    ds = RamenOriLerobotDataset(base_dataset=fake_base, chunk_len=4, img_size=32)
    # _sub_datasets を複数 sub の meta に差し替える
    ds._sub_datasets = [
        _FakeSub([
            {"length": 5, "source_task_index": 5, "source_episode_index": src_idx}
            for src_idx in src_list
        ])
        for src_list in source_indices_per_sub
    ]
    return ds


class TestSampleEpisodeMetadataWithSourceNames:
    def test_returns_source_episode_names(self):
        ds = _make_fake_dataset_with_source_indices([[10, 11, 12], [10, 20]])
        meta = ds.sample_episode_metadata()
        # 各 sub の各 row の source_episode_index を "src{:05d}" に normalize
        assert meta["source_episode_names"] == [
            "src00010", "src00011", "src00012", "src00010", "src00020",
        ]
        assert meta["num_episodes"] == 5

    def test_fallback_when_source_index_missing(self):
        # source_episode_index を除いた rows で fallback を verify
        fake_base = FakeLeRobotBase(num_samples=1, chunk_len=4, img_size=32)
        ds = RamenOriLerobotDataset(base_dataset=fake_base, chunk_len=4, img_size=32)
        ds._sub_datasets = [
            _FakeSub([{"length": 5, "source_task_index": 5}, {"length": 5, "source_task_index": 5}]),
            _FakeSub([{"length": 5, "source_task_index": 5}]),
        ]
        meta = ds.sample_episode_metadata()
        assert meta["source_episode_names"] == ["local_0_0", "local_0_1", "local_1_0"]


class TestLoadSplitFromJson:
    """Issue #122 refactor: `--split-json` option (cross-server で GR00T と strict identical split)。"""

    def test_valid_json_returns_split(self, tmp_path):
        from model.ramen_ori.data_lerobot import _load_split_from_json
        payload = {
            "schema_version": "team_ramen_grouped_episode_split_v1",
            "num_episodes": 10,
            "splits": {
                "train": {"episode_indices": [0, 1, 2, 3, 4, 5, 6]},
                "validation": {"episode_indices": [7, 8]},
                "test": {"episode_indices": [9]},
            },
        }
        p = tmp_path / "split.json"
        import json as _json
        p.write_text(_json.dumps(payload))
        train, val, test = _load_split_from_json(p, num_episodes=10)
        assert train == [0, 1, 2, 3, 4, 5, 6]
        assert val == [7, 8]
        assert test == [9]

    def test_missing_file_raises(self, tmp_path):
        from model.ramen_ori.data_lerobot import _load_split_from_json
        with pytest.raises(FileNotFoundError):
            _load_split_from_json(tmp_path / "missing.json", num_episodes=10)

    def test_wrong_schema_raises(self, tmp_path):
        from model.ramen_ori.data_lerobot import _load_split_from_json
        import json as _json
        p = tmp_path / "bad.json"
        p.write_text(_json.dumps({"schema_version": "other_v1", "splits": {}}))
        with pytest.raises(ValueError, match="unsupported split_json schema"):
            _load_split_from_json(p, num_episodes=10)

    def test_num_episodes_mismatch_raises(self, tmp_path):
        """HF revision 差検知: JSON の num_episodes と dataset 実数の mismatch → error。"""
        from model.ramen_ori.data_lerobot import _load_split_from_json
        import json as _json
        p = tmp_path / "split.json"
        payload = {
            "schema_version": "team_ramen_grouped_episode_split_v1",
            "num_episodes": 5,
            "splits": {
                "train": {"episode_indices": [0, 1, 2]},
                "validation": {"episode_indices": [3]},
                "test": {"episode_indices": [4]},
            },
        }
        p.write_text(_json.dumps(payload))
        with pytest.raises(RuntimeError, match="episode count.*mismatch|revision"):
            _load_split_from_json(p, num_episodes=10)  # JSON=5、dataset=10 mismatch

    def test_coverage_mismatch_raises(self, tmp_path):
        """全 episode を cover していない split JSON → error (data 欠損検知)。"""
        from model.ramen_ori.data_lerobot import _load_split_from_json
        import json as _json
        p = tmp_path / "gap.json"
        payload = {
            "schema_version": "team_ramen_grouped_episode_split_v1",
            "num_episodes": 5,
            "splits": {
                "train": {"episode_indices": [0, 1]},
                "validation": {"episode_indices": [2]},
                "test": {"episode_indices": [3]},  # 4 が抜けてる
            },
        }
        p.write_text(_json.dumps(payload))
        with pytest.raises(ValueError, match="coverage mismatch"):
            _load_split_from_json(p, num_episodes=5)


class TestExpandEpisodeSplitToFrames:
    def test_expands_correctly(self):
        from model.ramen_ori.data_lerobot import _expand_episode_split_to_frames
        # 4 episodes, length [3, 2, 3, 2] → global frames 0..9
        ep_lengths = np.array([3, 2, 3, 2], dtype=np.int64)
        train_idx, val_idx, test_idx = _expand_episode_split_to_frames(
            train_eps=[0, 2],  # frames 0-2 + 5-7
            val_eps=[1],       # frames 3-4
            test_eps=[3],      # frames 8-9
            episode_lengths=ep_lengths,
        )
        assert train_idx == [0, 1, 2, 5, 6, 7]
        assert val_idx == [3, 4]
        assert test_idx == [8, 9]


class TestBuildSplitPayload:
    def test_emits_source_episode_names_per_split(self):
        from model.ramen_ori.scripts.export_split_json import build_split_payload

        # 8 raw source × 2 curated segment = 16 episodes
        source_indices = list(range(8)) * 2
        ds = _make_fake_dataset_with_source_indices([source_indices])
        payload = build_split_payload(
            ds, val_ratio=0.25, test_ratio=0.25, seed=42,
            config_name="test_task5_7", repo_ids=["dummy/repo"],
        )
        assert payload["schema_version"] == "team_ramen_grouped_episode_split_v1"
        assert payload["group_key"] == "source_episode_name"
        assert payload["num_episodes"] == 16
        for name in ("train", "validation", "test"):
            entry = payload["splits"][name]
            assert isinstance(entry["source_episode_names"], list)
            assert all(isinstance(n, str) and n for n in entry["source_episode_names"])
            assert entry["source_episode_names"] == sorted(entry["source_episode_names"])
        # split 間で source_episode_names が disjoint (resolve の leakage check と同じ)
        train_names = set(payload["splits"]["train"]["source_episode_names"])
        val_names = set(payload["splits"]["validation"]["source_episode_names"])
        test_names = set(payload["splits"]["test"]["source_episode_names"])
        assert train_names & val_names == set()
        assert train_names & test_names == set()
        assert val_names & test_names == set()
        # sha256 は payload 全体を hash した後 field 追加 (resolve と同 計算)
        import hashlib as _h, json as _j
        unsigned = {k: v for k, v in payload.items() if k != "sha256"}
        expected = _h.sha256(_j.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert payload["sha256"] == expected

    def test_main_disables_auto_precompute_frame_cache(self, tmp_path, monkeypatch):
        """Issue #122 D-1: export_split_json は precompute (5-10 min) を trigger しない。

        split export は parquet 側 meta しか使わないため frame cache 不要。
        以前は env `FRAME_CACHE_PRECOMPUTE=false` を手動 export しないと precompute が
        走ってしまったが、export_split_json.main が constructor で
        `auto_precompute_frame_cache=False` を明示 override してユーザーが env を
        意識しなくて済むようにした。この事を verify する。
        """
        from model.ramen_ori.scripts import export_split_json as es

        # main() の hydra.compose を bypass、instantiate に渡る kwargs だけ捕捉
        captured: dict = {}

        class _StubDataset:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

        # RamenOriLerobotDataset instance check を通すため型偽装
        monkeypatch.setattr(es, "RamenOriLerobotDataset", _StubDataset)

        # hydra.utils.instantiate → 我々の Stub に kwargs 込みで転送
        def _fake_instantiate(cfg, **kwargs):
            return es.RamenOriLerobotDataset(**kwargs)

        monkeypatch.setattr(es.hydra.utils, "instantiate", _fake_instantiate)

        # hydra.initialize_config_dir + compose → fake DictConfig
        class _FakeCM:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        monkeypatch.setattr(es.hydra, "initialize_config_dir", lambda **kw: _FakeCM())
        fake_cfg = es.OmegaConf.create({
            "data": {"_target_": "unused"},
            "val": {"val_ratio": 0.1, "test_ratio": 0.1, "seed": 42},
        })
        monkeypatch.setattr(es.hydra, "compose", lambda config_name: fake_cfg)

        # build_split_payload は dataset の内部を触るので stub 化
        monkeypatch.setattr(
            es, "build_split_payload",
            lambda *a, **kw: {
                "num_episodes": 0,
                "provenance": {"num_source_recordings": 0},
                "splits": {
                    "train": {"episode_indices": []},
                    "validation": {"episode_indices": []},
                    "test": {"episode_indices": []},
                },
                "group_key": "source_episode_name",
                "sha256": "0" * 64,
            },
        )

        # CLI 引数を argv 経由で
        out_path = tmp_path / "split.json"
        monkeypatch.setattr(
            "sys.argv",
            ["export_split_json.py", "--config-name", "test", "--output", str(out_path)],
        )
        es.main()

        # 本命: auto_precompute_frame_cache=False が propagate されている
        assert captured.get("auto_precompute_frame_cache") is False, (
            f"expected auto_precompute_frame_cache=False, got kwargs: {captured}"
        )
        # augmentation_cfg=None も一緒に (memory 節約の意図保持)
        assert captured.get("augmentation_cfg") is None

    def test_payload_satisfies_resolve_training_split_schema(self):
        """resolve_training_split.resolve_split が 例外なく受け付ける payload である事を verify。"""
        from model.ramen_ori.scripts.export_split_json import build_split_payload
        from model.subtask_policy_training.scripts.resolve_training_split import resolve_split
        import tempfile
        from pathlib import Path

        source_indices = list(range(10)) * 2  # 20 eps / 10 src
        ds = _make_fake_dataset_with_source_indices([source_indices])
        payload = build_split_payload(
            ds, val_ratio=0.1, test_ratio=0.1, seed=42,
            config_name="test", repo_ids=["dummy/repo"],
        )
        # resolve は <dataset_root>/meta/team_ramen_episode_split.json を読む。tmpdir で再現。
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "meta").mkdir()
            (root / "meta" / "team_ramen_episode_split.json").write_text(
                __import__("json").dumps(payload)
            )
            # 例外なく resolve できれば schema 適合済
            out = resolve_split(root)
            assert out["TRAIN_EPISODE_COUNT"] > 0
            assert out["VALIDATION_EPISODE_COUNT"] > 0
            assert out["SPLIT_SHA256"] == payload["sha256"]


# ============================================================================
# RamenOriSplitView
# ============================================================================


class TestRamenOriSplitView:
    def _make_dataset(self):
        fake = FakeLeRobotBase(num_samples=8, chunk_len=4, img_size=32)
        return RamenOriLerobotDataset(
            base_dataset=fake, chunk_len=4, img_size=32, augmentation_cfg=_cfg_head_wrist(enabled=True)
        )

    def test_train_view_applies_aug(self):
        # Issue #129 Phase A: N_cams 4→3 (head_right 除外)
        ds = self._make_dataset()
        view = RamenOriSplitView(ds, indices=[0, 1, 2], is_train=True)
        item = view[0]
        assert item["images"].shape == (3, 3, 32, 32)
        # aug がかかっているので、val view と直接 image 値が違うはず (see next test)

    def test_val_view_skips_aug(self):
        ds = self._make_dataset()
        train_view = RamenOriSplitView(ds, indices=[0], is_train=True)
        val_view = RamenOriSplitView(ds, indices=[0], is_train=False)
        train_item = train_view[0]
        val_item = val_view[0]
        # 同じ global_idx=0、is_train フラグだけ違う → images は違う値
        assert not torch.allclose(train_item["images"], val_item["images"]), (
            "train/val で aug 有無が反映されていない"
        )

    def test_len_matches_indices(self):
        ds = self._make_dataset()
        view = RamenOriSplitView(ds, indices=[0, 3, 5], is_train=True)
        assert len(view) == 3


# ============================================================================
# _resolve_autocast_dtype
# ============================================================================


class TestResolveAutocastDtype:
    def test_none_returns_none(self):
        assert _resolve_autocast_dtype(None) is None

    @pytest.mark.parametrize("name,expected", [
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("float16", torch.float16),
        ("fp16", torch.float16),
    ])
    def test_known_names(self, name, expected):
        assert _resolve_autocast_dtype(name) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="not in"):
            _resolve_autocast_dtype("float64")
