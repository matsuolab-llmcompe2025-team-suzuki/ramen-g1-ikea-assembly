"""repo 全体の pytest の設定 (Issue #190)。

``@pytest.mark.requires_boundary`` の test は、運営の interface package
(``inference/desktop/boundary/``) が無いときに skip する。公開用 repo にはこの package を
入れない (license が無く、自前の環境では使わない) ので、そこでは会場の経路の test が skip になる。
この private repo には package があるので、全部走る。
"""

from pathlib import Path

import pytest

_BOUNDARY_INIT = Path(__file__).resolve().parent / "inference" / "desktop" / "boundary" / "__init__.py"


def _boundary_available() -> bool:
    # import では確かめない: data/・model/ の test だけを回すと inference.desktop が import できず、
    # find_spec が例外を投げて pytest 全体が止まる
    return _BOUNDARY_INIT.is_file()


def pytest_collection_modifyitems(config, items):
    if _boundary_available():
        return
    skip = pytest.mark.skip(
        reason="運営の interface package (inference/desktop/boundary/) が無い。会場の経路の test"
    )
    for item in items:
        if "requires_boundary" in item.keywords:
            item.add_marker(skip)
