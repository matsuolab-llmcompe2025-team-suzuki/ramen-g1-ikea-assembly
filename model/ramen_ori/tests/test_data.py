"""DummyRamenOriDataset の unit test (Issue #115、Phase 1 M9)。

- len / __getitem__ keys / per-sample shape
- 決定的 (同 idx で同 tensor)
- DataLoader integration: batch shape 正しい
- RamenOriPolicy との e2e integration: DataLoader → batch → compute_loss 完走
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from model.ramen_ori.data import DummyRamenOriDataset
from model.ramen_ori.model import RamenOriPolicy


class TestBasics:
    def test_len_matches_num_samples(self) -> None:
        ds = DummyRamenOriDataset(num_samples=42)
        assert len(ds) == 42

    def test_getitem_returns_all_expected_keys(self) -> None:
        ds = DummyRamenOriDataset()
        sample = ds[0]
        expected_keys = {
            "images",
            "images_prev",
            "cam_id",
            "obb_verts",
            "obb_conf",
            "obb_class_id",
            "obb_cam_id",
            "obb_valid_mask",
            "state",
            "skill_id",
            "action",
            "action_is_pad",
        }
        assert set(sample.keys()) == expected_keys

    def test_per_sample_shapes(self) -> None:
        ds = DummyRamenOriDataset(
            N_cams=4, top_K=4, state_dim=71, chunk_len=16, action_dim=19, img_size=224
        )
        s = ds[0]
        assert s["images"].shape == (4, 3, 224, 224)
        assert s["images_prev"].shape == (4, 3, 224, 224)
        assert s["cam_id"].shape == (4,)
        assert s["obb_verts"].shape == (4, 4, 8)
        assert s["obb_conf"].shape == (4, 4, 1)
        assert s["obb_class_id"].shape == (4, 4)
        assert s["obb_cam_id"].shape == (4, 4)
        assert s["obb_valid_mask"].shape == (4, 4)
        assert s["obb_valid_mask"].dtype == torch.bool
        assert s["state"].shape == (71,)
        assert s["skill_id"].shape == ()  # scalar
        assert s["action"].shape == (16, 19)


class TestDeterminism:
    def test_same_idx_gives_same_tensors(self) -> None:
        ds1 = DummyRamenOriDataset()
        ds2 = DummyRamenOriDataset()
        s1 = ds1[7]
        s2 = ds2[7]
        for k in s1:
            assert torch.equal(s1[k], s2[k]), f"key {k} not deterministic"

    def test_different_idx_gives_different_tensors(self) -> None:
        ds = DummyRamenOriDataset()
        s0 = ds[0]
        s1 = ds[1]
        # images 等の連続値 tensor は idx 違えば異なる、代表として images で verify
        assert not torch.equal(s0["images"], s1["images"])


class TestDataLoaderIntegration:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_batch_shapes(self, batch_size: int) -> None:
        ds = DummyRamenOriDataset(
            num_samples=16, N_cams=4, top_K=4, chunk_len=8, action_dim=4
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        batch = next(iter(loader))
        assert batch["images"].shape == (batch_size, 4, 3, 224, 224)
        assert batch["cam_id"].shape == (batch_size, 4)
        assert batch["obb_valid_mask"].shape == (batch_size, 4, 4)
        assert batch["state"].shape == (batch_size, 71)
        assert batch["skill_id"].shape == (batch_size,)
        assert batch["action"].shape == (batch_size, 8, 4)


class TestPolicyIntegration:
    def test_dataloader_batch_flows_through_compute_loss(self, stub_vit) -> None:
        """DummyRamenOriDataset → DataLoader → RamenOriPolicy.compute_loss 完走。"""
        ds = DummyRamenOriDataset(
            num_samples=4,
            N_cams=4,
            top_K=4,
            chunk_len=8,
            action_dim=4,
            state_dim=71,
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        # Phase 2 A-0: RamenOriPolicy は sub-module DI 型に変わったので test_model.py の
        # _make_small_policy と同じ minimal config を組む
        from model.ramen_ori.tests.test_model import _make_small_policy
        policy = _make_small_policy(stub_vit, chunk_len=8, action_dim=4)
        batch = next(iter(loader))
        loss = policy.compute_loss(batch)
        assert loss.dim() == 0
        loss.backward()
        # 少なくとも 1 個の learnable param に grad が付いてる
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in policy.parameters() if p.requires_grad
        )
