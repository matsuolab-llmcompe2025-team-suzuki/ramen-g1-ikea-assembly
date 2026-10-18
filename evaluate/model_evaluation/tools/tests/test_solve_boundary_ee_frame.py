"""solve_boundary_ee_frame の unit test。

**本番の準備 20 分枠で初めて動かす script** なので、既知の transform を仕込んで
それが復元できることを事前に確かめておく。実機データは会場でしか取れない。
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from evaluate.model_evaluation.tools.solve_boundary_ee_frame import (
    main,
    orthonormalize,
    quat_wxyz_to_matrix,
    read_records,
    se3,
    solve_side,
)


def _rot_z(deg: float) -> np.ndarray:
    th = np.radians(deg)
    return np.array(
        [[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]]
    )


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    delta = a.T @ b
    return float(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1.0, 1.0))))


class TestQuatWxyzToMatrix:
    def test_identity(self):
        np.testing.assert_allclose(quat_wxyz_to_matrix([1, 0, 0, 0]), np.eye(3))

    def test_w_first_order_is_assumed(self):
        """boundary は w-first。(x,y,z,w) を渡しても unit なので検証は通ってしまう
        (README が警告している罠) ため、ここで順序を固定しておく。"""
        # z 軸 90deg = (w, x, y, z) = (cos45, 0, 0, sin45)
        s = np.sqrt(0.5)
        np.testing.assert_allclose(
            quat_wxyz_to_matrix([s, 0.0, 0.0, s]), _rot_z(90.0), atol=1e-12
        )

    def test_normalizes_a_non_unit_quaternion(self):
        np.testing.assert_allclose(quat_wxyz_to_matrix([2, 0, 0, 0]), np.eye(3))

    def test_zero_quaternion_is_rejected(self):
        with pytest.raises(ValueError, match="zero norm"):
            quat_wxyz_to_matrix([0, 0, 0, 0])


class TestOrthonormalize:
    def test_a_rotation_is_unchanged(self):
        rot = _rot_z(33.0)
        np.testing.assert_allclose(orthonormalize(rot), rot, atol=1e-12)

    def test_an_averaged_rotation_is_repaired(self):
        """回転行列の平均は回転群を外れる。SVD で戻せること。"""
        averaged = (_rot_z(10.0) + _rot_z(20.0)) / 2.0
        assert not np.isclose(np.linalg.det(averaged), 1.0)
        fixed = orthonormalize(averaged)
        np.testing.assert_allclose(fixed.T @ fixed, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(fixed), 1.0)


class TestSolveSide:
    def _make_pairs(self, x_true: np.ndarray, n: int = 120, noise_m: float = 0.0):
        rng = np.random.default_rng(0)
        commanded, achieved = [], []
        for _ in range(n):
            angles = np.radians(rng.uniform(-30, 30, 2))
            rot = _rot_z(np.degrees(angles[0]))
            achieved_pose = se3(rot, rng.uniform(-0.4, 0.4, 3))
            commanded_pose = x_true @ achieved_pose
            if noise_m:
                commanded_pose[:3, 3] += rng.normal(0.0, noise_m, 3)
            commanded.append(commanded_pose)
            achieved.append(achieved_pose)
        return commanded, achieved

    def test_recovers_a_known_transform(self):
        """X = P_cmd ∘ P_ach⁻¹ が解けること。これが script の存在理由。"""
        x_true = se3(_rot_z(8.0), np.array([0.05, -0.02, 0.18]))
        result = solve_side(*self._make_pairs(x_true))

        estimated = np.asarray(result["transform"])
        np.testing.assert_allclose(estimated[:3, 3], x_true[:3, 3], atol=1e-9)
        assert _angle_between(x_true[:3, :3], estimated[:3, :3]) < 1e-6

    def test_recovers_it_under_measurement_noise(self):
        """実機は 2mm order のばらつきが乗る。平均で吸収できること。"""
        x_true = se3(_rot_z(8.0), np.array([0.05, -0.02, 0.18]))
        result = solve_side(*self._make_pairs(x_true, noise_m=0.002))

        estimated = np.asarray(result["transform"])
        np.testing.assert_allclose(estimated[:3, 3], x_true[:3, 3], atol=5e-3)
        # ばらつきが出力に現れる = 区間選択の良し悪しを判断できる
        assert np.all(np.asarray(result["pos_std_m"]) > 1e-4)

    def test_identity_when_the_frames_already_agree(self):
        """frame が一致していれば単位行列が出る (= 直す必要なしと分かる)。"""
        result = solve_side(*self._make_pairs(np.eye(4)))
        np.testing.assert_allclose(
            np.asarray(result["transform"]), np.eye(4), atol=1e-9
        )
        assert result["pos_norm_m"] < 1e-9

    def test_rejects_an_empty_sample_set(self):
        with pytest.raises(ValueError, match="no samples"):
            solve_side([], [])


class TestMain:
    """CLI を通しで 1 回走らせる。

    この script は **本番の準備 20 分枠で初めて実行される**。module 関数の
    unit test だけでは `main()` の中でしか起きない失敗 (FK の組み立て方を
    間違える等) を会場まで持ち込んでしまうので、実 URDF を使って端から端まで
    通す test を必ず 1 本置いておく。
    """

    def _write_log(self, path, samples: int = 4) -> None:
        rng = np.random.default_rng(0)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "tick", "seq": 0}) + "\n")
            for i in range(samples):
                row = np.zeros(25)
                row[4:7] = (0.20, 0.15, 0.10)  # left ee pos
                row[7:11] = (1.0, 0.0, 0.0, 0.0)  # left quat (w-first)
                row[11:14] = (0.20, -0.15, 0.10)  # right ee pos
                row[14:18] = (1.0, 0.0, 0.0, 0.0)
                handle.write(
                    json.dumps(
                        {
                            "event": "boundary_taskspace",
                            "seq": i,
                            "taskspace_25": row.tolist(),
                            "measured_body_q29": rng.normal(0.0, 0.05, 29).tolist(),
                        }
                    )
                    + "\n"
                )

    def test_runs_end_to_end_against_the_repo_urdf(self, tmp_path, monkeypatch, capsys):
        log = tmp_path / "orch_run.jsonl"
        self._write_log(log)
        monkeypatch.setattr(
            sys,
            "argv",
            ["solve_boundary_ee_frame.py", "--log", str(log), "--skip-first", "0.0"],
        )

        main()

        out = capsys.readouterr().out
        assert "=== left ===" in out
        assert "=== right ===" in out
        assert "ee_frame_transform (4x4)" in out

    def test_exits_when_the_log_has_no_boundary_rows(self, tmp_path, monkeypatch):
        """`--action-sink boundary` を付け忘れたログを渡したときに分かること。"""
        log = tmp_path / "orch_run.jsonl"
        log.write_text(json.dumps({"event": "tick"}) + "\n", encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["solve_boundary_ee_frame.py", "--log", str(log)]
        )

        with pytest.raises(SystemExit, match="boundary_taskspace"):
            main()


class TestReadRecords:
    def test_picks_only_boundary_rows_and_tolerates_others(self, tmp_path):
        """orchestrator の他の行が同じ JSONL に混ざっていても読めること。"""
        log = tmp_path / "orch_run.jsonl"
        log.write_text(
            "\n".join(
                [
                    json.dumps({"event": "tick", "seq": 1}),
                    json.dumps({"event": "boundary_taskspace", "seq": 1}),
                    "",
                    "not json at all",
                    json.dumps({"event": "boundary_taskspace", "seq": 2}),
                ]
            ),
            encoding="utf-8",
        )
        got = list(read_records(log))
        assert [r["seq"] for r in got] == [1, 2]
