"""学習と推論の約束 (contract.py、Issue #141 束 2-14) の test。"""

from __future__ import annotations

import json

import pytest
import yaml
from omegaconf import OmegaConf

from model.ramen_ori.contract import CONTRACT_VERSION, REPO_ROOT, build_contract
from model.ramen_ori.data_lerobot import RamenOriLerobotDataset, RamenOriMultiDataset
from model.ramen_ori.state_derive import STATE71_GROUPS
from model.ramen_ori.tests.test_data_lerobot import FakeLeRobotBase

BAKE_CONFIG = "data/bitrobot_lerobot_subtask_datasets/configs/unified_frame_cache.yaml"
INFERENCE_POLICY_CONFIG = REPO_ROOT / "inference/desktop/lower_policy/configs/policy_config.yaml"


def _cfg(bake_config: str | None = BAKE_CONFIG, chunk_len: int = 32) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {"action_dim": 16, "chunk_len": chunk_len, "normalization": {"enabled": True}},
            "contract": {"bake_config": bake_config},
        }
    )


def _baked_multi(tmp_path, num_variants: int = 10) -> RamenOriMultiDataset:
    """rotate 12 frame + move 8 frame の baked_overlay (前 frame なし)。"""
    subs = [
        {
            "base_dataset": FakeLeRobotBase(num_samples=n, chunk_len=32, source_task_index=tid, n_image_frames=1),
            "frame_cache_root_baked": tmp_path,
        }
        for n, tid in ((12, 5), (8, 7))
    ]
    return RamenOriMultiDataset(
        sub_datasets=subs,
        chunk_len=32,
        load_prev_image=False,
        frame_cache_mode="baked_overlay",
        frame_cache_num_variants=num_variants,
    )


class TestBakedOverlayContract:
    def test_contents(self, tmp_path) -> None:
        contract = build_contract(_cfg(), _baked_multi(tmp_path))

        assert contract["version"] == CONTRACT_VERSION
        assert contract["skills"] == [
            {"id": 4, "name": "rotate_table_base", "frames": 12},
            {"id": 5, "name": "move_table_base", "frames": 8},
        ]
        assert contract["state"]["variant"] == "71d"
        assert contract["state"]["version"] == 2
        assert contract["state"]["groups"] == {k: [s.start, s.stop] for k, s in STATE71_GROUPS.items()}
        assert contract["action"]["space"] == "abs"
        assert contract["action"]["dim"] == 16
        assert contract["action"]["chunk_len"] == 32
        assert contract["normalization"] is True

        images = contract["images"]
        assert images["camera_keys"] == [
            "observation.images.cam_0",
            "observation.images.cam_2",
            "observation.images.cam_3",
        ]
        assert images["img_size"] == 224
        assert images["prev_frame"] is False
        assert images["source"] == "baked_overlay"
        overlay = images["overlay"]
        assert overlay["camera"] == "observation.images.cam_0"
        assert overlay["top_k"] == 32
        assert overlay["conf"] == pytest.approx(0.30)
        assert overlay["thickness"] == 2
        assert overlay["class_filter"] is None
        assert overlay["bake_config"] == BAKE_CONFIG

        commit = contract["provenance"]["git_commit"]
        assert commit is not None and len(commit) == 40
        assert isinstance(contract["provenance"]["git_dirty"], bool)

    def test_yolo_is_the_inference_yolo(self, tmp_path) -> None:
        """焼き込みの YOLO と推論が overlay に使う YOLO が同じ (推論は起動時にこの値と比べる)。

        repo の名前は違ってよい。焼き込みの設定は cache の置き場所の hash のために改名の前の名前のままで
        (Issue #190)、推論は commit の sha で比べる。ここも推論と同じ判定を使う (Issue #196)。
        """
        from inference.desktop.lower_policy.policies.ramen_ori import _same_yolo_weight

        contract = build_contract(_cfg(), _baked_multi(tmp_path))
        inference_yolo = yaml.safe_load(INFERENCE_POLICY_CONFIG.read_text())["yolo"]["ckpt_ref"]
        assert _same_yolo_weight(contract["images"]["overlay"]["yolo_ckpt"], inference_yolo)

    def test_plain_data_for_ckpt_and_wandb(self, tmp_path) -> None:
        """numpy の型などが混ざらない (yaml / json にそのまま出せる)。"""
        contract = build_contract(_cfg(), _baked_multi(tmp_path))
        assert yaml.safe_load(yaml.safe_dump(contract, allow_unicode=True)) == contract
        json.dumps(contract)

    def test_requires_bake_config(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="contract.bake_config"):
            build_contract(_cfg(bake_config=None), _baked_multi(tmp_path))

    def test_num_variants_must_match_bake(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="num_variants"):
            build_contract(_cfg(), _baked_multi(tmp_path, num_variants=5))


class TestOnlineContract:
    def test_no_overlay(self) -> None:
        ds = RamenOriLerobotDataset(base_dataset=FakeLeRobotBase(num_samples=10), chunk_len=16)
        contract = build_contract(_cfg(bake_config=None, chunk_len=16), ds)
        assert contract["images"]["source"] == "online"
        assert contract["images"]["prev_frame"] is True
        assert contract["images"]["overlay"] is None
        assert contract["skills"] == [{"id": 4, "name": "rotate_table_base", "frames": 10}]


class TestMemoryContract:
    def test_memory_block(self, tmp_path) -> None:
        from model.ramen_ori.memory_features import MEMORY_LAYOUT
        from model.ramen_ori.tests.test_data_memory import HASH, _dataset

        ds, _, _ = _dataset(tmp_path, [5, 4])
        memory = build_contract(_cfg(bake_config=None), ds)["memory"]
        assert memory["version"] == 1
        assert memory["layout"] == list(MEMORY_LAYOUT) and len(memory["layout"]) == 51
        assert memory["camera"] == "observation.images.cam_0"
        assert memory["yolo_ckpt"] == f"dummy@{HASH}"
        assert memory["conf_min"] == pytest.approx(0.30)
        assert memory["taus_s"] == [0.25, 0.5, 1.0] and memory["fps"] == 30.0
        assert memory["single_classes"] == {"workspace": 0, "table_top": 4, "hand_left": 6, "hand_right": 5}
        assert memory["multi_classes"] == {"leg": 1, "leg_tip": 2, "hole": 3}

    def test_no_memory_is_null(self, tmp_path) -> None:
        assert build_contract(_cfg(), _baked_multi(tmp_path))["memory"] is None


@pytest.mark.parametrize("source", ["dataset", "fk"])
def test_contract_records_the_ee_pose_source(source) -> None:
    """Issue #183: 手先の位置の入力の出どころを約束に書く (既定は記録値 = dataset)。"""
    multi = RamenOriMultiDataset(
        sub_datasets=[{"base_dataset": FakeLeRobotBase(num_samples=10, chunk_len=32, source_task_index=2)}],
        chunk_len=32,
        ee_state_source=source,
    )
    assert build_contract(_cfg(bake_config=None), multi)["state"]["ee_pose_source"] == source
