"""PolicyDetectionFilter の unit test (Issue #141 束 1-9)。

主眼:
    - 遅れが無い (続いている物は今の frame の位置で返る、None を返さない)
    - 1 frame だけの枠を消す / 新しい物は persist_frames 遅れて出る
    - class ごとの上限 / continue_frames による見逃しの扱い / 過去 3 frame の中央値の遅れ
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.perception.policy_filter import (
    PolicyDetectionFilter,
    load_policy_filter_config,
)
from inference.desktop.perception.yolo_obb import OBBDetection


def _det(class_name: str, cx: float, cy: float, size: float = 0.1, confidence: float = 0.9) -> OBBDetection:
    """中心と大きさから axis-aligned の OBB (時計回り 4 頂点) を作る。"""
    h = size / 2
    verts = np.array(
        [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]], dtype=np.float32
    )
    return OBBDetection(class_id=0, class_name=class_name, confidence=confidence, verts=verts)


def _config(**overrides) -> dict:
    config = {
        "conf_min": 0.30,
        "max_count": {"table_top": 1, "hand_left": 1, "leg": 4},
        "over_max_continue_iou": 0.9,
        "over_cap": "keep_continuing",
        "continue_iou": 0.5,
        "continue_frames": 1,
        "persist_frames": 1,
        "smoothing": "none",
        "smoothing_iou_min": 0.5,
    }
    config.update(overrides)
    return config


def _names(dets: list[OBBDetection]) -> list[str]:
    return sorted(d.class_name for d in dets)


def _center_x(dets: list[OBBDetection], name: str) -> float:
    (d,) = [d for d in dets if d.class_name == name]
    return float(d.verts[:, 0].mean())


class TestPersistence:
    def test_new_object_appears_after_persist_frames(self):
        f = PolicyDetectionFilter(_config(persist_frames=1))
        assert f.push([_det("leg", 0.3, 0.3)]) == []           # 初出の frame は出さない
        assert _names(f.push([_det("leg", 0.3, 0.3)])) == ["leg"]

    def test_persist_frames_two_delays_two_frames(self):
        f = PolicyDetectionFilter(_config(persist_frames=2))
        assert f.push([_det("leg", 0.3, 0.3)]) == []
        assert f.push([_det("leg", 0.3, 0.3)]) == []
        assert _names(f.push([_det("leg", 0.3, 0.3)])) == ["leg"]

    def test_persist_frames_zero_keeps_immediately(self):
        f = PolicyDetectionFilter(_config(persist_frames=0))
        assert _names(f.push([_det("leg", 0.3, 0.3)])) == ["leg"]

    def test_single_frame_spike_is_dropped(self):
        f = PolicyDetectionFilter(_config())
        f.push([_det("table_top", 0.5, 0.5)])
        f.push([_det("table_top", 0.5, 0.5)])
        out = f.push([_det("table_top", 0.5, 0.5), _det("hand_left", 0.8, 0.2)])   # hand_left は 1 frame だけ
        assert _names(out) == ["table_top"]
        assert _names(f.push([_det("table_top", 0.5, 0.5)])) == ["table_top"]


class TestNoLag:
    def test_continuing_box_is_returned_at_current_position(self):
        f = PolicyDetectionFilter(_config())
        xs = [0.30, 0.32, 0.34, 0.36, 0.38]
        outs = [f.push([_det("hand_left", x, 0.5)]) for x in xs]
        assert outs[0] == []
        for x, out in zip(xs[1:], outs[1:]):
            assert _center_x(out, "hand_left") == pytest.approx(x, abs=1e-6)

    def test_never_returns_none(self):
        f = PolicyDetectionFilter(_config())
        assert f.push([]) == []


class TestConf:
    def test_below_conf_min_is_dropped_before_rules(self):
        f = PolicyDetectionFilter(_config(persist_frames=0))
        assert f.push([_det("leg", 0.3, 0.3, confidence=0.29)]) == []
        assert _names(f.push([_det("leg", 0.3, 0.3, confidence=0.30)])) == ["leg"]


class TestMaxCount:
    def test_over_cap_keeps_only_box_continuing_from_previous_frame(self):
        f = PolicyDetectionFilter(_config(persist_frames=0))
        f.push([_det("table_top", 0.5, 0.5)])
        out = f.push([_det("table_top", 0.5, 0.5), _det("table_top", 0.2, 0.2)])
        assert len(out) == 1 and _center_x(out, "table_top") == pytest.approx(0.5)

    def test_over_cap_without_continuing_box_drops_class(self):
        f = PolicyDetectionFilter(_config(persist_frames=0))
        f.push([_det("table_top", 0.5, 0.5)])
        out = f.push([_det("table_top", 0.8, 0.8), _det("table_top", 0.2, 0.2)])
        assert out == []


class TestOverCapFillByConf:
    def test_fills_up_to_cap_by_confidence_instead_of_dropping_class(self):
        """上限を超えても class は消えず、conf の高い順で上限まで残る。"""
        f = PolicyDetectionFilter(_config(over_cap="fill_by_conf", persist_frames=0))
        out = f.push([
            _det("leg", 0.1, 0.1, confidence=0.9),
            _det("leg", 0.3, 0.1, confidence=0.8),
            _det("leg", 0.5, 0.1, confidence=0.7),
            _det("leg", 0.7, 0.1, confidence=0.6),
            _det("leg", 0.9, 0.1, confidence=0.5),   # cap 4 を超える 5 個目 (conf 最小)
        ])
        assert len(out) == 4
        assert min(d.confidence for d in out) == pytest.approx(0.6)

    def test_continuing_box_is_kept_even_with_lower_confidence(self):
        """続いている枠は、conf が低くても先に残る。"""
        f = PolicyDetectionFilter(_config(over_cap="fill_by_conf", persist_frames=0))
        f.push([_det("table_top", 0.5, 0.5, confidence=0.4)])
        out = f.push([
            _det("table_top", 0.5, 0.5, confidence=0.4),   # 続いている (cap 1)
            _det("table_top", 0.1, 0.1, confidence=0.95),
        ])
        assert len(out) == 1 and _center_x(out, "table_top") == pytest.approx(0.5)

    def test_keep_continuing_mode_drops_class_without_continuing_box(self):
        """今の規則では、上限を超えて続いている枠が無いと class ごと消える。"""
        f = PolicyDetectionFilter(_config(over_cap="keep_continuing", persist_frames=0))
        assert f.push([_det("table_top", 0.2, 0.2), _det("table_top", 0.8, 0.8)]) == []

    def test_unknown_over_cap_mode_raises(self):
        with pytest.raises(ValueError):
            PolicyDetectionFilter(_config(over_cap="top_conf"))


class TestContinueFrames:
    def test_one_missed_frame_needs_persistence_again_with_continue_frames_one(self):
        f = PolicyDetectionFilter(_config(continue_frames=1))
        f.push([_det("leg", 0.3, 0.3)])
        f.push([_det("leg", 0.3, 0.3)])
        f.push([])                                              # 1 frame 見逃し
        assert f.push([_det("leg", 0.3, 0.3)]) == []           # 新しい物扱い、前の生にも無い
        assert _names(f.push([_det("leg", 0.3, 0.3)])) == ["leg"]

    def test_one_missed_frame_is_bridged_with_continue_frames_two(self):
        f = PolicyDetectionFilter(_config(continue_frames=2))
        f.push([_det("leg", 0.3, 0.3)])
        f.push([_det("leg", 0.3, 0.3)])
        f.push([])
        assert _names(f.push([_det("leg", 0.3, 0.3)])) == ["leg"]


class TestSmoothing:
    def test_median3_past_lags_moving_box_by_one_frame(self):
        f = PolicyDetectionFilter(_config(smoothing="median3_past"))
        xs = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35]
        outs = [f.push([_det("hand_left", x, 0.5)]) for x in xs]
        # 3 frame 残った後は、直前 2 frame と合わせた中央値 = 1 frame 前の位置
        for t in range(3, len(xs)):
            assert _center_x(outs[t], "hand_left") == pytest.approx(xs[t - 1], abs=1e-6)

    def test_median3_past_removes_single_frame_jitter_of_static_box(self):
        f = PolicyDetectionFilter(_config(smoothing="median3_past"))
        for x in (0.50, 0.50, 0.50):
            f.push([_det("table_top", x, 0.5)])
        out = f.push([_det("table_top", 0.52, 0.5)])           # 1 frame だけずれた
        assert _center_x(out, "table_top") == pytest.approx(0.50, abs=1e-6)

    def test_unknown_smoothing_mode_raises(self):
        with pytest.raises(ValueError):
            PolicyDetectionFilter(_config(smoothing="median3"))


class TestReset:
    def test_reset_forgets_history(self):
        f = PolicyDetectionFilter(_config())
        f.push([_det("leg", 0.3, 0.3)])
        f.reset()
        assert f.push([_det("leg", 0.3, 0.3)]) == []


class TestConfigFile:
    def test_policy_section_builds_filter(self):
        config = load_policy_filter_config()
        assert config["conf_min"] == pytest.approx(0.30)
        PolicyDetectionFilter(config)

    def test_decided_defaults(self):
        """2026-09-14 の決定 (計測の結果から): 上限だけを効かせ、1 frame の枠もならしも触らない。"""
        config = load_policy_filter_config()
        assert config["over_cap"] == "fill_by_conf"
        assert config["persist_frames"] == 0
        assert config["smoothing"] == "none"
        assert config["max_count"] == {
            "workspace": 1, "table_top": 1, "hand_right": 1, "hand_left": 1,
            "leg": 4, "leg_tip": 4, "hole": 4,
        }

    def test_planner_cleaner_uses_fill_by_conf(self):
        """Planner 用 cleaner も上限超えを conf の高い順で埋める (2026-09-14 決定)。"""
        from inference.desktop.perception.cleaner import load_cleanup_config

        assert load_cleanup_config()["over_max_mode"] == "fill_by_conf"
