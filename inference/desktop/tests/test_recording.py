"""RunRecorder unit test。"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from inference.desktop.recording import (
    CAMERA_ROLES,
    RunRecorder,
    record_policy_tick,
    state_record,
)


class TestRunRecorder:
    def test_writes_all_streams(self, tmp_path):
        with RunRecorder(tmp_path) as rec:
            rec.write_metadata({"skill": "test", "variant": "test_variant"})
            rec.write_event({"kind": "start", "phase": "pre_motion"})
            rec.write_state({"joint_state": [0.0] * 29})
            rec.write_action({"chunk_0": [0.1] * 19})
            rec.write_action({"chunk_0": [0.2] * 19})
        assert (tmp_path / "metadata.json").exists()
        assert (tmp_path / "events.jsonl").exists()
        assert (tmp_path / "states.jsonl").exists()
        assert (tmp_path / "actions.jsonl").exists()
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["skill"] == "test"
        actions = [
            json.loads(line)
            for line in (tmp_path / "actions.jsonl").read_text().splitlines()
        ]
        assert len(actions) == 2
        assert actions[0]["chunk_0"] == [0.1] * 19

    def test_tick_count_increments(self, tmp_path):
        with RunRecorder(tmp_path) as rec:
            for i in range(5):
                rec.write_action({"chunk_0": [float(i)] * 19})
            assert rec.tick_count == 5

    def test_close_flushes_pending_writes(self, tmp_path):
        rec = RunRecorder(tmp_path)
        rec.write_event({"kind": "test"})
        rec.write_state({"kind": "test"})
        rec.close()
        # 明示 close 後、file 内容が読める
        assert "test" in (tmp_path / "events.jsonl").read_text()
        assert "test" in (tmp_path / "states.jsonl").read_text()

    def test_policy_interval_packages_four_unique_camera_streams(self, tmp_path):
        base_ns = time.monotonic_ns()
        with RunRecorder(tmp_path) as rec:
            rec.start_policy_capture(base_ns)
            for index in range(6):
                frames = {
                    role: np.full((48, 64, 3), index * 20, dtype=np.uint8)
                    for role in CAMERA_ROLES
                }
                observation = {
                    "t": base_ns + index * 33_333_333,
                    "_camera_frames": frames,
                    "_camera_generations": {
                        role: index for role in CAMERA_ROLES
                    },
                    "_camera_received_monotonic_ns": {
                        role: base_ns + index * 33_333_333
                        for role in CAMERA_ROLES
                    },
                }
                rec.record_camera_observation(observation)
                # A duplicate policy poll must not duplicate camera frames.
                rec.record_camera_observation(observation)
            assert rec.stop_policy_capture(base_ns + 6 * 33_333_333)

        report = json.loads(
            (tmp_path / "capture/capture_report.json").read_text()
        )
        manifest = json.loads(
            (tmp_path / "capture/video_manifest.json").read_text()
        )
        assert report["complete"] is True
        assert report["queue_drop_count"] == 0
        assert report["policy_interval_s"] == pytest.approx(0.2, rel=0.02)
        assert report["camera_frame_count"] == {role: 6 for role in CAMERA_ROLES}
        assert report["camera_written_frame_count"] == {
            role: 6 for role in CAMERA_ROLES
        }
        assert manifest["complete"] is True
        assert set(manifest["videos"]) == set(CAMERA_ROLES)
        for role in CAMERA_ROLES:
            assert report["camera_effective_hz"][role] >= 29.0
            assert report["camera_coverage_fraction"][role] >= 0.99
            assert report["camera_max_interframe_gap_s"][role] < 0.04
            video = tmp_path / manifest["videos"][role]["relative_path"]
            assert video.stat().st_size > 0

    def test_abnormal_short_capture_keeps_partial_diagnostics(self, tmp_path):
        with RunRecorder(tmp_path) as rec:
            rec.start_policy_capture(time.monotonic_ns())
            rec.record_camera_observation(
                {
                    "t": time.monotonic_ns(),
                    "_camera_frames": {
                        "head_left": np.zeros((16, 16, 3), dtype=np.uint8)
                    },
                    "_camera_generations": {"head_left": 1},
                    "_camera_received_monotonic_ns": {
                        "head_left": time.monotonic_ns()
                    },
                }
            )

        report = json.loads(
            (tmp_path / "capture/capture_report.json").read_text()
        )
        manifest = json.loads(
            (tmp_path / "capture/video_manifest.json").read_text()
        )
        assert report["recording_started"] is True
        assert report["complete"] is False
        assert report["camera_written_frame_count"]["head_left"] == 1
        assert manifest["complete"] is False
        assert manifest["videos"]["head_left"]["frame_count"] == 1
        assert (tmp_path / "capture/camera_frames.jsonl").stat().st_size > 0

    def test_camera_sampler_runs_independently_of_policy_loop(self, tmp_path):
        generation = 0

        def supply():
            nonlocal generation
            generation += 1
            now = time.monotonic_ns()
            return {
                "t": now,
                "_camera_frames": {
                    role: np.full((24, 32, 3), generation, dtype=np.uint8)
                    for role in CAMERA_ROLES
                },
                "_camera_generations": {
                    role: generation for role in CAMERA_ROLES
                },
                "_camera_received_monotonic_ns": {
                    role: now for role in CAMERA_ROLES
                },
            }

        with RunRecorder(tmp_path) as rec:
            rec.start_policy_capture(
                time.monotonic_ns(),
                observation_supplier=supply,
                camera_sample_hz=30.0,
            )
            # Simulate a slow policy forward that never calls the recorder.
            time.sleep(0.22)
            rec.stop_policy_capture(time.monotonic_ns())

        report = json.loads(
            (tmp_path / "capture/capture_report.json").read_text()
        )
        assert report["requested_camera_sample_hz"] == 30.0
        assert report["complete"] is True
        assert all(count >= 5 for count in report["camera_frame_count"].values())

    def test_truncated_role_fails_full_interval_validation(self, tmp_path):
        base_ns = time.monotonic_ns()
        interval_ns = 3_000_000_000
        with RunRecorder(tmp_path) as rec:
            rec.start_policy_capture(base_ns)
            for index in range(90):
                timestamp = base_ns + index * 33_333_333
                roles = list(CAMERA_ROLES)
                if index >= 8:
                    roles.remove("left_wrist")
                rec.record_camera_observation(
                    {
                        "t": timestamp,
                        "_camera_frames": {
                            role: np.zeros((8, 8, 3), dtype=np.uint8)
                            for role in roles
                        },
                        "_camera_generations": {role: index for role in roles},
                        "_camera_received_monotonic_ns": {
                            role: timestamp for role in roles
                        },
                    }
                )
            rec.stop_policy_capture(base_ns + interval_ns)

        report = json.loads(
            (tmp_path / "capture/capture_report.json").read_text()
        )
        manifest = json.loads(
            (tmp_path / "capture/video_manifest.json").read_text()
        )
        assert report["complete"] is False
        assert report["camera_effective_hz"]["left_wrist"] == pytest.approx(8 / 3)
        assert report["camera_coverage_fraction"]["left_wrist"] < 0.10
        assert report["camera_end_stale_s"]["left_wrist"] > 2.7
        assert any(
            "left_wrist" in error for error in report["validation_errors"]
        )
        assert manifest["complete"] is False


class _StubRecorder:
    def __init__(self) -> None:
        self.states: list[dict] = []
        self.actions: list[dict] = []
        self.events: list[dict] = []

    def write_state(self, record: dict) -> None:
        self.states.append(record)

    def write_action(self, record: dict) -> None:
        self.actions.append(record)

    def write_event(self, event: dict) -> None:
        self.events.append(event)


class _StubPolicyAction:
    def __init__(self) -> None:
        self.action_chunk = np.zeros((16, 19), dtype=np.float32)
        self.latency_ms = 12.5
        self.metadata = {"stub": True}


class _StubActiveSkill:
    def __init__(self, name: str, last_action=None) -> None:
        self.name = name
        self.last_action = last_action


def _obs_for_record() -> dict:
    class _JointState:
        position = np.zeros(29)
        velocity = np.zeros(29)
        effort = np.arange(29, dtype=float)

    class _HandState:
        position_rad = np.zeros(2)

    return {
        "t": 1234,
        "joint_state": _JointState(),
        "hand_state": _HandState(),
    }


class TestRecordPolicyTick:
    """tick ごとの記録 (Issue #141 D1 #4 / D6-3)。評価と本番の両方が使う。"""

    def test_state_and_action_are_written_as_a_pair(self):
        recorder = _StubRecorder()
        skill = _StubActiveSkill("rotate_table_base", _StubPolicyAction())
        record_policy_tick(
            recorder, _obs_for_record(), skill, learned_skills={"rotate_table_base"}
        )
        assert len(recorder.states) == 1 and len(recorder.actions) == 1
        assert recorder.states[0]["skill"] == "rotate_table_base"
        assert recorder.actions[0]["skill"] == "rotate_table_base"

    def test_a_tick_without_an_action_writes_neither(self):
        """model を読んだ直後の tick で行番号の対がずれていた既存のずれを無くす。"""
        recorder = _StubRecorder()
        skill = _StubActiveSkill("rotate_table_base", None)
        record_policy_tick(
            recorder, _obs_for_record(), skill, learned_skills={"rotate_table_base"}
        )
        assert recorder.states == [] and recorder.actions == []

    def test_head_procedure_ticks_are_not_recorded(self):
        recorder = _StubRecorder()
        skill = _StubActiveSkill("arm_pre_motion_for_rotate_table_base", _StubPolicyAction())
        record_policy_tick(
            recorder, _obs_for_record(), skill, learned_skills={"rotate_table_base"}
        )
        assert recorder.states == [] and recorder.actions == []

    def test_effort_probe_is_written_once(self):
        recorder = _StubRecorder()
        skill = _StubActiveSkill("rotate_table_base", _StubPolicyAction())
        probe: dict = {}
        for _ in range(3):
            record_policy_tick(
                recorder, _obs_for_record(), skill,
                learned_skills={"rotate_table_base"}, probe_state=probe,
            )
        probes = [e for e in recorder.events if e["kind"] == "joint_effort_probe"]
        assert len(probes) == 1
        assert probes[0]["nonzero_count"] == 28      # effort = arange(29) の非ゼロ


def test_state_record_carries_joint_effort() -> None:
    """Issue #137: tau_est (JointState.effort) を states.jsonl に落とす。"""

    class Joints:
        position = np.arange(29, dtype=np.float64)
        velocity = np.full(29, 0.5)
        effort = np.full(29, -1.25)

    class Hand:
        position_rad = np.asarray([0.9, 3.8])

    record = state_record({
        "t": 7,
        "joint_state": Joints(),
        "hand_state": Hand(),
        "_camera_received_monotonic_ns": {"head_left": 3},
    })

    assert record["joint_effort"] == [-1.25] * 29
    assert record["joint_velocity"] == [0.5] * 29
    assert record["hand_position_rad"] == [0.9, 3.8]
    assert record["camera_received_monotonic_ns"] == {"head_left": 3}


def test_state_record_keeps_missing_effort_as_null_not_zero() -> None:
    """effort 未 publish は 0 埋めせず null。「0 を返した」と区別するため。"""

    class JointsWithoutEffort:
        position = np.zeros(29)
        velocity = np.zeros(29)
        effort = None

    record = state_record({"t": 1, "joint_state": JointsWithoutEffort()})

    assert record["joint_effort"] is None
    assert record["joint_position"] == [0.0] * 29


def test_state_record_without_joint_state_is_all_null() -> None:
    record = state_record({"t": 2})

    assert record["joint_position"] is None
    assert record["joint_velocity"] is None
    assert record["joint_effort"] is None
    assert record["hand_position_rad"] is None
