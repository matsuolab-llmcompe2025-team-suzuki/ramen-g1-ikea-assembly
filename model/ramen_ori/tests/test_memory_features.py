"""memory の入力 (memory_features.MemoryTracker、Issue #141 Phase 7) の test。"""

from __future__ import annotations

import numpy as np
import pytest

from model.ramen_ori.memory_features import (
    FPS,
    MEMORY_DIM,
    MEMORY_LAYOUT,
    MOTION_SLICE,
    TAUS_S,
    MemoryTracker,
)

IDX = {name: i for i, name in enumerate(MEMORY_LAYOUT)}
NO_DET = (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros((0, 4, 2)))


def _box(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    """中心・辺の長さ・最初の辺の角度から 4 頂点 (4, 2)。"""
    a = np.radians(angle_deg)
    u = np.array([np.cos(a), np.sin(a)]) * w
    v = np.array([-np.sin(a), np.cos(a)]) * h
    p0 = np.array([cx, cy]) - (u + v) / 2
    return np.stack([p0, p0 + u, p0 + u + v, p0 + v])


def _dets(*items: tuple[int, float, np.ndarray]):
    if not items:
        return NO_DET
    cls, conf, verts = zip(*items)
    return np.asarray(cls), np.asarray(conf), np.stack(verts)


def test_layout() -> None:
    assert MEMORY_DIM == 51 == len(set(MEMORY_LAYOUT))
    assert MEMORY_LAYOUT[:6] == ("workspace.dcx", "workspace.dcy", "workspace.dw", "workspace.dh", "workspace.dangle", "workspace.visible")
    assert all(n.startswith("motion.") for n in MEMORY_LAYOUT[MOTION_SLICE])
    assert MOTION_SLICE.stop - MOTION_SLICE.start == 18


class TestObservation:
    def test_first_seen_reference_and_hold(self) -> None:
        tr = MemoryTracker()
        m0 = tr.update(*NO_DET, None)                                   # 天板はまだ見えていない
        assert m0[IDX["table_top.visible"]] == 0 and m0[IDX["table_top.dcx"]] == 0
        m1 = tr.update(*_dets((4, 0.9, _box(0.5, 0.5, 0.2, 0.1, 10.0))), None)   # 最初に見えた frame = 基準
        assert m1[IDX["table_top.visible"]] == 1
        np.testing.assert_allclose(m1[[IDX[f"table_top.{f}"] for f in ("dcx", "dcy", "dw", "dh", "dangle")]], 0, atol=1e-6)
        m2 = tr.update(*_dets((4, 0.9, _box(0.55, 0.48, 0.22, 0.1, 25.0))), None)
        np.testing.assert_allclose(
            m2[[IDX[f"table_top.{f}"] for f in ("dcx", "dcy", "dw", "dh", "dangle")]],
            [0.05, -0.02, 0.02, 0.0, 15.0], atol=1e-5,
        )
        m3 = tr.update(*NO_DET, None)                                   # 見逃し → 前の差を保つ
        assert m3[IDX["table_top.visible"]] == 0
        np.testing.assert_array_equal(m3[IDX["table_top.dcx"]:IDX["table_top.dangle"] + 1],
                                      m2[IDX["table_top.dcx"]:IDX["table_top.dangle"] + 1])

    def test_low_confidence_is_ignored(self) -> None:
        tr = MemoryTracker()
        m = tr.update(*_dets((4, 0.29, _box(0.5, 0.5, 0.2, 0.1, 0.0))), None)
        assert m[IDX["table_top.visible"]] == 0

    def test_highest_confidence_box_is_used(self) -> None:
        tr = MemoryTracker()
        tr.update(*_dets((6, 0.9, _box(0.3, 0.3, 0.1, 0.1, 0.0))), None)
        m = tr.update(*_dets((6, 0.5, _box(0.9, 0.9, 0.1, 0.1, 0.0)), (6, 0.8, _box(0.35, 0.3, 0.1, 0.1, 0.0))), None)
        assert m[IDX["hand_left.dcx"]] == pytest.approx(0.05, abs=1e-6)

    def test_angle_difference_wraps_at_180(self) -> None:
        tr = MemoryTracker()
        tr.update(*_dets((4, 0.9, _box(0.5, 0.5, 0.2, 0.1, 179.0))), None)
        m = tr.update(*_dets((4, 0.9, _box(0.5, 0.5, 0.2, 0.1, 1.0))), None)
        assert m[IDX["table_top.dangle"]] == pytest.approx(2.0, abs=1e-4)

    def test_multi_class_count_and_mean_center(self) -> None:
        tr = MemoryTracker()
        m0 = tr.update(*_dets((1, 0.9, _box(0.2, 0.5, 0.05, 0.05, 0.0)), (1, 0.9, _box(0.4, 0.5, 0.05, 0.05, 0.0))), None)
        assert m0[IDX["leg.count"]] == 2 and m0[IDX["leg.dcx"]] == pytest.approx(0.0, abs=1e-6)
        m1 = tr.update(*_dets((1, 0.9, _box(0.5, 0.6, 0.05, 0.05, 0.0))), None)   # 平均の中心 (0.3, 0.5) → (0.5, 0.6)
        assert m1[IDX["leg.count"]] == 1
        assert m1[IDX["leg.dcx"]] == pytest.approx(0.2, abs=1e-6)
        assert m1[IDX["leg.dcy"]] == pytest.approx(0.1, abs=1e-6)
        m2 = tr.update(*NO_DET, None)
        assert m2[IDX["leg.count"]] == 0 and m2[IDX["leg.dcx"]] == pytest.approx(0.2, abs=1e-6)

    def test_reset_clears_references(self) -> None:
        tr = MemoryTracker()
        tr.update(*_dets((4, 0.9, _box(0.5, 0.5, 0.2, 0.1, 0.0))), None)
        tr.update(*_dets((4, 0.9, _box(0.6, 0.5, 0.2, 0.1, 0.0))), None)
        tr.reset()
        m = tr.update(*_dets((4, 0.9, _box(0.6, 0.5, 0.2, 0.1, 0.0))), None)
        assert m[IDX["table_top.dcx"]] == 0


class TestMotion:
    def test_velocity_is_zero_for_first_two_ticks_and_ema_follows_formula(self) -> None:
        rng = np.random.default_rng(0)
        hands = [None] + [rng.normal(size=6) * 0.1 for _ in range(6)]   # tick 0 はまだ指令が無い
        tr = MemoryTracker()
        out = [tr.update(*NO_DET, h) for h in hands]
        np.testing.assert_array_equal(out[0][MOTION_SLICE], 0)
        np.testing.assert_array_equal(out[1][MOTION_SLICE], 0)   # 手先は 1 点だけ → 速度 0

        alpha = np.exp(-(1.0 / FPS) / np.asarray(TAUS_S))[:, None]
        ema = np.zeros((len(TAUS_S), 6))
        for t in range(2, len(hands)):
            v = (hands[t] - hands[t - 1]) * FPS
            ema = alpha * ema + (1 - alpha) * v[None]
            np.testing.assert_allclose(out[t][MOTION_SLICE], ema.ravel(), rtol=1e-5, atol=1e-6)

    def test_layout_order_is_tau_then_side_then_axis(self) -> None:
        tr = MemoryTracker()
        tr.update(*NO_DET, np.zeros(6))
        m = tr.update(*NO_DET, np.array([0, 0, 0, 0.1, 0, 0]))   # 右手先の x だけ動く
        moving = [MEMORY_LAYOUT[MOTION_SLICE.start + i] for i in np.flatnonzero(m[MOTION_SLICE])]
        assert moving == ["motion.tau0.25.right.x", "motion.tau0.5.right.x", "motion.tau1.right.x"]
