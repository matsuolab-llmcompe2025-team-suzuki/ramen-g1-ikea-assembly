"""運営の interface package の読み込み (Issue #190)。

公開用 repo には ``boundary/`` が無い。会場の経路を選んだときに、何を置けばよいかが分かる
error で止まること。
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from inference.desktop import organizer_boundary
from inference.desktop.organizer_boundary import (
    BOUNDARY_PACKAGE,
    MISSING_BOUNDARY_MESSAGE,
    import_boundary,
)


def _raise_missing(name: str):
    def _import(requested: str):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    return _import


def test_missing_package_says_where_to_get_it(monkeypatch):
    monkeypatch.setattr(
        organizer_boundary.importlib, "import_module", _raise_missing(BOUNDARY_PACKAGE)
    )
    with pytest.raises(ModuleNotFoundError) as info:
        import_boundary("states")
    assert str(info.value) == MISSING_BOUNDARY_MESSAGE
    assert info.value.name == BOUNDARY_PACKAGE
    assert "iacevaltest/iros_g1_orin_package" in str(info.value)
    assert "497f3ab" in str(info.value)


def test_a_missing_dependency_inside_the_package_is_not_hidden(monkeypatch):
    """boundary/ はあるが、その中の import (zmq など) が無いときは元の error のまま。"""
    monkeypatch.setattr(
        organizer_boundary.importlib, "import_module", _raise_missing("zmq")
    )
    with pytest.raises(ModuleNotFoundError) as info:
        import_boundary()
    assert info.value.name == "zmq"
    assert str(info.value) != MISSING_BOUNDARY_MESSAGE


def test_an_already_loaded_module_is_returned_as_is(monkeypatch):
    """会場の腕の模擬は boundary を偽物に差し替えてから送り口を作る。その偽物を返すこと。"""
    fake = types.ModuleType(BOUNDARY_PACKAGE)
    monkeypatch.setitem(sys.modules, BOUNDARY_PACKAGE, fake)
    assert import_boundary() is fake


def test_the_real_package_loads_when_present():
    pytest.importorskip(BOUNDARY_PACKAGE)
    assert import_boundary() is importlib.import_module(BOUNDARY_PACKAGE)
