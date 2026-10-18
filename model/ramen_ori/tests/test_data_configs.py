"""configs/data の unit test (Issue #146)。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from model.ramen_ori.data_lerobot import _SUB_DATASET_KEYS

DATA = Path(__file__).resolve().parents[1] / "configs" / "data"
ALL6 = [
    "rotate_table_base", "move_table_base", "insert_table_leg",
    "pick_table_leg", "flip_table", "rotate_leg_to_tighten",
]


def _load(name: str) -> dict:
    return yaml.safe_load((DATA / name).read_text(encoding="utf-8"))


def test_unified_all6_is_beta_plus_three_skills() -> None:
    """6 skill の data は β と共通の設定が同じで、sub は β の 3 本 + 新しい 3 skill (置き場所の規則も同じ)。"""
    beta, all6 = _load("unified_beta.yaml"), _load("unified_all6.yaml")
    assert {k: v for k, v in all6.items() if k != "sub_datasets"} == {
        k: v for k, v in beta.items() if k != "sub_datasets"
    }
    subs = all6["sub_datasets"]
    assert subs[:3] == beta["sub_datasets"]
    assert [s["repo_id"] for s in subs] == [f"local/ramen_ori_{n}" for n in ALL6]
    for s, n in zip(subs, ALL6):
        assert set(s) <= _SUB_DATASET_KEYS
        root = f"/nvme/unified_cache/{n}"
        assert (s["merged_source_root"], s["frame_cache_root_baked"], s["obb_precomputed_root"]) == (
            f"{root}/src", f"{root}/v2/overlay", f"{root}/obb_yolo"
        )
        assert s["obb_precomputed_hash"] == beta["sub_datasets"][0]["obb_precomputed_hash"]


@pytest.mark.parametrize(
    "name, skill",
    [
        ("unified_flip.yaml", "flip_table"),
        # Issue #188: 会場の腕の追加学習の 1 skill の data
        ("unified_rotate.yaml", "rotate_table_base"),
        ("unified_pick.yaml", "pick_table_leg"),
        ("unified_insert.yaml", "insert_table_leg"),
    ],
)
def test_single_skill_data_is_the_all6_sub_only(name: str, skill: str) -> None:
    """1 skill だけの data は all6 と共通の設定が同じで、sub は all6 のその skill の 1 本だけ。"""
    all6, single = _load("unified_all6.yaml"), _load(name)
    assert {k: v for k, v in single.items() if k != "sub_datasets"} == {
        k: v for k, v in all6.items() if k != "sub_datasets"
    }
    assert single["sub_datasets"] == [all6["sub_datasets"][ALL6.index(skill)]]
