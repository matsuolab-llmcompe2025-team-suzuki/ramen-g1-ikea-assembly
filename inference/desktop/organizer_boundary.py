"""運営の interface package (``inference/desktop/boundary/``) を読み込む (Issue #190)。

公開用 repo には入れない (license が無い)。要るのは会場の経路 (``--action-sink boundary``・
``--joint-source boundary``・会場の腕の学習) だけなので、無ければ置き場所を案内して止める。
"""

from __future__ import annotations

import importlib
from types import ModuleType

BOUNDARY_PACKAGE = "inference.desktop.boundary"

MISSING_BOUNDARY_MESSAGE = (
    "The organizer's interface package `inference/desktop/boundary/` is not included in this "
    "repository (it has no license). It is needed only for the competition-venue path "
    "(`--action-sink boundary`, `--joint-source boundary`, the venue-arm training), which this "
    "repository does not support. To use it anyway, copy `boundary/` from "
    "https://github.com/iacevaltest/iros_g1_orin_package at commit 497f3ab to "
    "`inference/desktop/boundary/`."
)


def import_boundary(submodule: str = "") -> ModuleType:
    """``inference.desktop.boundary`` (か、その下の ``submodule``) を import して返す。

    boundary/ そのものが無いときだけ、置き場所を案内する ModuleNotFoundError にする。
    boundary/ の中の別の import (zmq など) の失敗は、元の error をそのまま出す。
    すでに読み込まれている物 (会場の腕の模擬が差し替えた偽物も含む) はそのまま返す。
    """
    name = f"{BOUNDARY_PACKAGE}.{submodule}" if submodule else BOUNDARY_PACKAGE
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != BOUNDARY_PACKAGE:
            raise
        raise ModuleNotFoundError(
            MISSING_BOUNDARY_MESSAGE, name=BOUNDARY_PACKAGE
        ) from exc
