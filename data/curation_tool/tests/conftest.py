"""tests/conftest.py — curation_tool を sys.path に追加、tmp workspace fixture。"""

import sys
from pathlib import Path

import pytest

# root iros_2026_ramen/pytest.ini が pythonpath=. なので、curation_tool 側を
# top-level import (ui.labels_io, loader.curation_loader) 可能にするために追加。
_TOOL_ROOT = Path(__file__).resolve().parent.parent
if str(_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOL_ROOT))


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """テスト用の空 workspace。pytest 標準 tmp_path をそのまま返す。"""
    return tmp_path
