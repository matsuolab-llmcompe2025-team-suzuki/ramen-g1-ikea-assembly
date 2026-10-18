"""prepare_multi_repo_source.py の unit test (Issue #122、Data DL 高速化 flag)。

download_all の実 HF アクセスは mock で bypass、CLI default 値 + 並列/sequential
両 path で repo_ids 順序保持を確認。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_multi_repo_source.py"


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_multi_repo_source", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestParseArgs:
    def test_defaults_include_parallel_flags(self) -> None:
        """Issue #122: --max-repo-concurrency=4 と --max-workers=16 が default。"""
        prepare = load_prepare_module()
        argv = [
            "prepare_multi_repo_source.py",
            "--repo-ids",
            '["Team-RAMEN/foo_v1"]',
            "--output-dir",
            "/tmp/test_out",
        ]
        with patch.object(sys, "argv", argv):
            args = prepare.parse_args()
        assert args.max_repo_concurrency == 4
        assert args.max_workers == 16

    def test_parallel_flags_override(self) -> None:
        prepare = load_prepare_module()
        argv = [
            "prepare_multi_repo_source.py",
            "--repo-ids",
            '["Team-RAMEN/foo_v1"]',
            "--output-dir",
            "/tmp/test_out",
            "--max-repo-concurrency",
            "1",
            "--max-workers",
            "8",
        ]
        with patch.object(sys, "argv", argv):
            args = prepare.parse_args()
        assert args.max_repo_concurrency == 1
        assert args.max_workers == 8


class TestDownloadAll:
    """snapshot_download を mock して download_all の path 順序 + 並列 dispatch を確認。"""

    def test_sequential_preserves_order(self) -> None:
        """--max-repo-concurrency=1 の sequential path で repo_ids 順に返す。"""
        prepare = load_prepare_module()
        repos = ["Team-RAMEN/a_v1", "Team-RAMEN/b_v1", "Team-RAMEN/c_v1"]
        with patch.object(prepare, "snapshot_download", side_effect=lambda repo_id, **kw: f"/fake/{repo_id}"):
            paths = prepare.download_all(repos, revision=None, max_repo_concurrency=1, max_workers=8)
        assert [str(p) for p in paths] == [f"/fake/{r}" for r in repos]

    def test_parallel_preserves_order(self) -> None:
        """並列 (max_repo_concurrency=4) でも as_completed 経由で repo_ids 順に返す
        (merge 側の episode_index 再採番が順序依存のため必須)。"""
        prepare = load_prepare_module()
        repos = [f"Team-RAMEN/x{i}_v1" for i in range(6)]
        with patch.object(prepare, "snapshot_download", side_effect=lambda repo_id, **kw: f"/fake/{repo_id}"):
            paths = prepare.download_all(repos, revision=None, max_repo_concurrency=4, max_workers=16)
        assert [str(p) for p in paths] == [f"/fake/{r}" for r in repos]

    def test_parallel_passes_max_workers_to_snapshot(self) -> None:
        """--max-workers が snapshot_download に伝播することを確認。"""
        prepare = load_prepare_module()
        received_kwargs: list[dict] = []

        def _spy(repo_id, **kw):
            received_kwargs.append(kw)
            return f"/fake/{repo_id}"

        with patch.object(prepare, "snapshot_download", side_effect=_spy):
            prepare.download_all(["Team-RAMEN/a_v1"], revision=None, max_repo_concurrency=2, max_workers=32)
        assert received_kwargs[0]["max_workers"] == 32
