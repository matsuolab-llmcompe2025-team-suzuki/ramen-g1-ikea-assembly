"""RotateProgressMonitor の unit test (Issue #137 Phase 2)。"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from inference.desktop.lower_policy.skills.rotate_progress import (
    RotateProgressMonitor,
    load_progress_monitor_for_skill,
    _obb_angle_deg,
    _wrap_deg,
)


class _Det:
    """OBBDetection の duck-typed stub。"""

    def __init__(self, angle_deg: float, confidence: float = 0.9,
                 class_name: str = "table_top") -> None:
        self.class_name = class_name
        self.confidence = confidence
        rad = np.radians(angle_deg)
        edge = np.array([np.cos(rad), np.sin(rad)]) * 0.2
        normal = np.array([-edge[1], edge[0]]) * 0.6
        p0 = np.array([0.5, 0.5])
        self.verts = np.stack([p0, p0 + edge, p0 + edge + normal, p0 + normal])


def _feed(monitor: RotateProgressMonitor, angles, start_ns: int = 0):
    step = int(1e9 / 30)
    out = None
    for i, a in enumerate(angles):
        dets = None if a is None else [_Det(a)]
        out = monitor.update(dets, start_ns + i * step)
    return out


def test_wrap_folds_180_degree_period() -> None:
    assert _wrap_deg(170.0) == pytest.approx(-10.0)
    assert _wrap_deg(-170.0) == pytest.approx(10.0)
    assert _wrap_deg(5.0) == pytest.approx(5.0)


def test_obb_angle_uses_first_edge() -> None:
    assert _obb_angle_deg(_Det(30.0).verts) == pytest.approx(30.0, abs=1e-6)
    # 180deg 周期なので 210deg は 30deg と同一視される
    assert _obb_angle_deg(_Det(210.0).verts) == pytest.approx(30.0, abs=1e-6)


def test_not_usable_until_median_window_is_filled() -> None:
    monitor = RotateProgressMonitor(median_window=9, sustain_frames=6)
    for i in range(8):
        result = monitor.update([_Det(70.0)], i * int(1e9 / 30))
        assert not result.usable
    result = monitor.update([_Det(70.0)], 8 * int(1e9 / 30))
    assert result.usable
    assert result.rotation_deg == pytest.approx(0.0, abs=1e-6)


def test_steady_rotation_is_detected_after_sustain() -> None:
    monitor = RotateProgressMonitor(success_deg=8.0, median_window=5, sustain_frames=3)
    # 参照確立 → 20deg まで回す → 閾値超えが続く
    result = _feed(monitor, [70.0] * 5 + [90.0] * 8)
    assert result.usable
    assert result.rotation_deg == pytest.approx(20.0, abs=0.5)
    assert result.sustained


def test_single_frame_flip_does_not_trigger_success() -> None:
    """1 フレームだけ 90deg 飛ぶ誤検出で成功判定しない (median + sustain)。"""
    monitor = RotateProgressMonitor(success_deg=8.0, median_window=9, sustain_frames=6)
    angles = [70.0] * 9 + [70.0, 70.0, 160.0, 70.0, 70.0, 70.0, 70.0, 70.0]
    result = _feed(monitor, angles)
    assert not result.sustained
    assert abs(result.rotation_deg) < 8.0


def test_brief_detection_gap_keeps_last_judgement() -> None:
    """単発の検出漏れでは判定を捨てない (手が天板にかぶる区間があるため)。"""
    monitor = RotateProgressMonitor(
        median_window=3, sustain_frames=1, success_deg=8.0, stale_timeout_s=1.0
    )
    step = int(1e9 / 30)
    _feed(monitor, [70.0] * 3 + [85.0] * 3)
    held = monitor.update(None, 6 * step)  # 1 frame 落ち = 0.033s
    assert held.usable
    assert held.sustained
    assert not held.detected
    assert held.rotation_deg == pytest.approx(15.0, abs=0.5)


def test_detection_gap_beyond_timeout_becomes_unusable() -> None:
    monitor = RotateProgressMonitor(
        median_window=3, sustain_frames=1, success_deg=8.0, stale_timeout_s=1.0
    )
    step = int(1e9 / 30)
    _feed(monitor, [70.0] * 3 + [85.0] * 3)
    stale = monitor.update(None, 6 * step + 2_000_000_000)  # 2s 欠測
    assert not stale.usable
    assert not stale.sustained
    assert stale.stale_s == pytest.approx(2.0, abs=0.1)


def test_never_detected_reports_infinite_stale() -> None:
    monitor = RotateProgressMonitor(median_window=3, sustain_frames=1)
    result = monitor.update(None, 0)
    assert not result.usable
    assert result.stale_s == float("inf")


def test_other_classes_and_low_confidence_are_ignored() -> None:
    monitor = RotateProgressMonitor(median_window=3, sustain_frames=1)
    step = int(1e9 / 30)
    result = monitor.update([_Det(70.0, class_name="leg")], 0)
    assert not result.detected
    result = monitor.update([_Det(70.0, confidence=0.05)], step)
    assert not result.detected


def test_highest_confidence_table_top_wins() -> None:
    monitor = RotateProgressMonitor(median_window=1, sustain_frames=1, success_deg=8.0)
    monitor.update([_Det(70.0, confidence=0.9), _Det(10.0, confidence=0.2)], 0)
    result = monitor.update(
        [_Det(85.0, confidence=0.9), _Det(10.0, confidence=0.2)], int(1e9 / 30)
    )
    assert result.rotation_deg == pytest.approx(15.0, abs=0.5)


def test_reset_reestablishes_reference() -> None:
    monitor = RotateProgressMonitor(median_window=3, sustain_frames=1, success_deg=8.0)
    result = _feed(monitor, [70.0] * 3 + [90.0] * 3)
    assert result.rotation_deg == pytest.approx(20.0, abs=0.5)
    monitor.reset()
    result = _feed(monitor, [90.0] * 3)
    assert result.rotation_deg == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("bad", [
    {"success_deg": 0.0}, {"success_deg": 90.0},
    {"median_window": 0}, {"sustain_frames": 0}, {"stale_timeout_s": 0.0},
])
def test_invalid_config_is_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        RotateProgressMonitor(**bad)


def test_vla_skill_emits_progress_metadata_without_changing_behaviour() -> None:
    """Phase 2: monitor は metadata を足すだけで arm target を変えない。"""
    import time

    from inference.desktop.lower_policy.skills.tests.test_vla_skill import _StubPolicy
    from inference.desktop.lower_policy.skills.vla_skill import MoveTableBaseVlaSkill
    from inference.desktop.lower_policy.actuators.hand import MockHandActuator
    from inference.desktop.lower_policy.actuators.waist import MockWaistActuator

    def build(monitor):
        return MoveTableBaseVlaSkill(
            policy=_StubPolicy(chunk_value=0.0),
            waist_actuator=MockWaistActuator(),
            hand_actuator=MockHandActuator(),
            progress_monitor=monitor,
        )

    obs = {
        "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_left_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_right_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "joint_state": None,
        "cleaned": [_Det(70.0)],
        "t": time.monotonic_ns(),
    }
    skill_without = build(None)
    skill_with = build(RotateProgressMonitor(median_window=1, sustain_frames=1))
    for skill in (skill_without, skill_with):
        skill.start({})
    target_without = skill_without.step(dict(obs))
    target_with = skill_with.step(dict(obs))

    assert np.allclose(target_without, target_with)
    assert "table_rotation_deg" not in skill_without.last_action.metadata
    meta = skill_with.last_action.metadata
    assert meta["table_rotation_deg"] == pytest.approx(0.0)
    assert meta["table_top_detected"] is True


def test_loader_returns_none_without_progress_block() -> None:
    cfg = {"skills": {"flip_table": {"dispatch_waist": False}}}
    assert load_progress_monitor_for_skill(cfg, "flip_table") is None
    assert load_progress_monitor_for_skill(cfg, "missing_skill") is None
    assert load_progress_monitor_for_skill({}, "flip_table") is None


def test_loader_builds_monitor_from_config() -> None:
    cfg = {"skills": {"rotate_table_base": {"progress": {
        "success_deg": 12.0, "median_window": 5, "sustain_frames": 2,
    }}}}
    monitor = load_progress_monitor_for_skill(cfg, "rotate_table_base")
    assert isinstance(monitor, RotateProgressMonitor)
    result = _feed(monitor, [70.0] * 5 + [85.0] * 4)
    assert result.sustained  # 15deg > 12deg 閾値


def test_loader_rejects_unknown_keys() -> None:
    cfg = {"skills": {"rotate_table_base": {"progress": {"succes_deg": 8.0}}}}
    with pytest.raises(ValueError, match="unknown keys"):
        load_progress_monitor_for_skill(cfg, "rotate_table_base")


def test_repo_skill_config_progress_block_is_loadable() -> None:
    """repo の skill_config.yaml がそのまま monitor を build できること。"""
    import yaml

    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "configs" / "skill_config.yaml"
    )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    monitor = load_progress_monitor_for_skill(cfg, "rotate_table_base")
    assert isinstance(monitor, RotateProgressMonitor)
