"""RAMEN-Ori の ckpt をネット無しで解決できること (Issue #159 B3d-03)。

会場の実行時はネットに繋がない (`HF_HUB_OFFLINE=1`)。GR00T と YOLO は commit を
固定した `snapshot_download` なので cache から読めるが、RAMEN-Ori だけは読み込みの
たびに `HfApi.list_repo_files` を無条件に呼んでいて、ネットが無いと落ちていた。
rotate_table_base の既定は RAMEN-Ori で、1 run で 4 回読み直す。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inference.desktop.lower_policy.policies import ramen_ori
from inference.desktop.lower_policy.policies.config_loader import load_policy_variant

_POLICY_CONFIG = "inference/desktop/lower_policy/configs/policy_config.yaml"
_REPO = "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base"
_REV = "aaa73be26e795376ca9e2fcf2528e0b51b362c67"


class _NoNetwork:
    """呼ばれたら落ちる HfApi (オフライン)。"""

    def __init__(self, *a, **k) -> None:
        pass

    def list_repo_files(self, *a, **k):
        raise AssertionError("list_repo_files must not be called offline")


@pytest.fixture
def offline_hub(monkeypatch, tmp_path):
    """cache に `ckpt_step_{50000,100000}.pt` だけがある状態の huggingface_hub。"""
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for step in (50000, 100000):
        (snapshot / f"ckpt_step_{step}.pt").write_bytes(b"x")
    calls: dict[str, list] = {"hf_hub_download": [], "snapshot_download": []}

    def _hf_hub_download(repo_id, filename, revision=None, **kwargs):
        calls["hf_hub_download"].append((repo_id, filename, revision))
        path = snapshot / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return str(path)

    def _snapshot_download(repo_id, revision=None, **kwargs):
        calls["snapshot_download"].append((repo_id, revision, kwargs))
        assert kwargs.get("local_files_only") is True
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "HfApi", _NoNetwork)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot_download)
    return calls


def test_a_pinned_file_is_read_without_the_hf_api(offline_hub):
    path = ramen_ori.resolve_hf_ckpt(_REPO, _REV, "ckpt_step_100000.pt")
    assert Path(path).name == "ckpt_step_100000.pt"
    assert offline_hub["hf_hub_download"] == [(_REPO, "ckpt_step_100000.pt", _REV)]


def test_an_unpinned_slot_picks_the_latest_cached_step_without_the_hf_api(offline_hub):
    path = ramen_ori.resolve_hf_ckpt(_REPO, _REV, None)
    assert Path(path).name == "ckpt_step_100000.pt"


def test_offline_without_the_file_says_what_to_prefetch(monkeypatch, tmp_path):
    import huggingface_hub

    empty = tmp_path / "empty"
    empty.mkdir()

    class _Offline:
        def __init__(self, *a, **k) -> None:
            pass

        def list_repo_files(self, *a, **k):
            raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "HfApi", _Offline)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, revision=None, **k: str(empty),
    )
    with pytest.raises(FileNotFoundError, match="prefetch|ckpt_filename"):
        ramen_ori.resolve_hf_ckpt(_REPO, _REV, None)


def test_the_slot_passes_its_pinned_file_name():
    entry = load_policy_variant(
        _POLICY_CONFIG, "rotate_table_base_ramen_ori_venue_all6_20k"
    )
    assert entry.policy_config.ckpt_filename == "ckpt_step_020000.pt"


def test_every_ramen_ori_slot_used_at_the_venue_pins_its_checkpoint():
    """既定と組み合わせ (variant_sets) で選べる RAMEN-Ori は、ファイル名を固定する。

    固定しないと、読み込みのたびに repo の一覧 (HF API) か cache の中身で ckpt が決まる。
    """
    with open(_POLICY_CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    used = set(config["default_variant_by_skill"].values())
    for variants in (config.get("variant_sets") or {}).values():
        used |= set(variants.values())
    ramen_ori_slots = [
        name
        for name in used
        if config["policies"][name].get("policy_type") == "ramen_ori"
    ]
    assert ramen_ori_slots
    for name in ramen_ori_slots:
        assert config["policies"][name].get("ckpt_filename"), name
