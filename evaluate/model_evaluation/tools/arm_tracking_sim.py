"""会場の log を、本物の送り方 (BoundaryActionSink + ArmTrackingAssist) に流して、送り方と
運営 adapter / WBC の起動の引数を比べる (Issue #188)。

模擬するもの:
- 私たちの sink: 本物のコード。ZMQ の送り口だけを記録係に差し替える。tick ごとに会場の指令
  (``actions.jsonl`` の ``chunk_0``) と模擬の実測を渡す。
- 運営 adapter (``--max-joint-vel`` / ``--chunk-hz`` / ``--max-waypoints``): chunk を受け取ると
  1 行目を**実測の腕**に合わせ直し、1 行 (1/chunk_hz) ごとに max_joint_vel/chunk_hz までしか
  進めない (``wbc_driver.py:_step_clamp``)。行は受け取った時刻から 1/chunk_hz おき。
- WBC の目標 (``--upper-body-joint-speed``): waypoint を時間で線形に補間し、その速さを上限で抑える。
- 腕: 目標へ一次遅れ (時定数 ``tau``。0.15 s で 09-29 の会場の遅れ pick 370〜390 ms・
  台を回す 260〜280 ms を再現した)。重力の垂れは入れない (比べ合いに使う)。
- 台 (``wall_joint``): その関節は会場の実測より押す向きへは進めない。押す力の目安 = WBC の
  目標が壁を押す向きに越えた量 (柔らかい PD では力 ∝ この量)。

- 先の予定 (束の ``preview_s``、Issue #188): ``compare`` が腕の指令を ``preview_s`` 先へずらして
  流し、遅れ・ずれは元の指令 (予定の今の値) と比べる。後で実際に実行された指令を予定とみなす
  ので**上限**の数字 (実際の予定は後から立て直されて変わる)。``simulate`` は渡された指令をそのまま送る。

限界: 会場の指令を流すだけ (policy の閉ループは再現しない)。台は動かない壁の近似。

    python evaluate/model_evaluation/tools/arm_tracking_sim.py \\
        ~/iros_venue_logs/0929/production_runs/2026-09-29T22-10-37 --skill rotate_table_base --wall-joint R.el
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from inference.desktop.lower_policy.actuators.arm_tracking_assist import (  # noqa: E402
    ArmTrackingAssist,
    ArmTrackingConfig,
)
from inference.desktop.lower_policy.scripts.tracking_summary import (  # noqa: E402
    ARM_NAMES,
    _read_jsonl,
    tracking_metrics,
)

SKILL_CONFIG = REPO / "inference/desktop/lower_policy/configs/skill_config.yaml"
DT = 0.005
_SEGMENT_GAP_S = 0.5


@dataclass(frozen=True)
class Case:
    """比べる 1 つの組み合わせ。``arm`` は束の名前か ArmTrackingConfig (None = 補いなし)。"""

    label: str
    arm: Union[None, str, ArmTrackingConfig] = "today"
    max_joint_vel: float = 1.0  # adapter --max-joint-vel
    chunk_hz: float = 20.0  # adapter --chunk-hz
    max_waypoints: int = 16  # adapter --max-waypoints
    chunk_rows: int = 16  # 私たちの --boundary-chunk-rows
    wbc_speed: Optional[float] = 3.0  # WBC --upper-body-joint-speed (None = 上限なし)
    extra: dict = field(default_factory=dict)


class _RecordingJointSink:
    """``inference.desktop.boundary.JointSink`` の代わり。送った chunk を時刻つきで残す。"""

    clock = None

    def __init__(self, port: int = 0, host: str = "") -> None:
        self.sent: list[tuple[float, np.ndarray]] = []

    def send_chunk(self, actions: np.ndarray, issued_at=None) -> None:
        self.sent.append(
            (float(type(self).clock()), np.asarray(actions, dtype=np.float64))
        )

    def close(self) -> None:
        pass


@dataclass
class _Clock:
    t: float = 0.0

    def __call__(self) -> float:
        return self.t


def _make_sink(assist: Optional[ArmTrackingAssist], clock: _Clock, chunk_rows: int):
    """本物の BoundaryActionSink を、送り口だけ差し替えて作る。"""
    fake = types.ModuleType("inference.desktop.boundary")
    fake.JointSink = _RecordingJointSink
    _RecordingJointSink.clock = clock
    saved = sys.modules.get("inference.desktop.boundary")
    sys.modules["inference.desktop.boundary"] = fake
    try:
        from inference.desktop.lower_policy.actuators.boundary_sink import (
            BoundaryActionSink,
        )

        return BoundaryActionSink(
            lane="joint",
            publish_period_s=0.1,
            chunk_rows=chunk_rows,
            hold_refresh_s=1.0,
            clock=clock,
            arm_tracking=assist,
        )
    finally:
        if saved is None:
            sys.modules.pop("inference.desktop.boundary", None)
        else:
            sys.modules["inference.desktop.boundary"] = saved


def load_segments(
    run: Path, skill: str
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """[(時刻 s, 19-D の指令 (n,19), 実測の腕 (n,14))]。途切れ (0.5 s 超) で分ける。"""
    actions = _read_jsonl(run / "actions.jsonl")
    states = {r.get("t"): r for r in _read_jsonl(run / "states.jsonl")}
    rows = []
    for a in actions:
        s = states.get(a.get("t"))
        if (
            a.get("skill") != skill
            or not a.get("chunk_0")
            or s is None
            or not s.get("joint_position")
        ):
            continue
        chunk = list(a["chunk_0"])
        # 先の予定を送った run は、送った腕ではなく予定の今の腕を指令とみなす (Issue #188)
        plan = (a.get("policy_metadata") or {}).get("plan_target_arms")
        if isinstance(plan, list) and len(plan) == 14 and len(chunk) >= 17:
            chunk[3:17] = plan
        rows.append((a["t"] / 1e9, chunk, s["joint_position"][15:29]))
    if not rows:
        raise ValueError(f"no {skill} rows in {run}")
    t = np.array([r[0] for r in rows])
    action = np.array([r[1] for r in rows], dtype=np.float64)
    measured = np.array([r[2] for r in rows], dtype=np.float64)
    cuts = np.concatenate(
        [[0], np.flatnonzero(np.diff(t) > _SEGMENT_GAP_S) + 1, [len(t)]]
    )
    return [
        (t[a:b] - t[a], action[a:b], measured[a:b])
        for a, b in zip(cuts[:-1], cuts[1:])
        if b - a >= 60
    ]


def _assist(case: Case, skill_config: dict) -> Optional[ArmTrackingAssist]:
    if case.arm is None:
        return None
    if isinstance(case.arm, ArmTrackingConfig):
        return ArmTrackingAssist(case.arm, name=case.label)
    config, name = ArmTrackingConfig.from_config(skill_config, case.arm)
    return ArmTrackingAssist(config, name=name)


def _preview_s(case: Case, skill_config: dict) -> float:
    """その組の束の先の予定 [s] (補いなしなら 0)。"""
    if case.arm is None:
        return 0.0
    if isinstance(case.arm, ArmTrackingConfig):
        return case.arm.preview_s
    return ArmTrackingConfig.from_config(skill_config, case.arm)[0].preview_s


def with_preview(t: np.ndarray, action19: np.ndarray, preview_s: float) -> np.ndarray:
    """腕だけ preview_s 先の指令を今送る形にする (最後は最後の値のまま)。上限の模擬。"""
    if preview_s <= 0.0:
        return action19
    sent = action19.copy()
    for j in range(3, 17):
        sent[:, j] = np.interp(t + preview_s, t, action19[:, j])
    return sent


def simulate(
    t: np.ndarray,
    action19: np.ndarray,
    measured_real: np.ndarray,
    assist: Optional[ArmTrackingAssist],
    *,
    tau: float,
    case: Case,
    latency_s: float = 0.02,
    wall_joint: Optional[int] = None,
) -> dict:
    clock = _Clock()
    sink = _make_sink(assist, clock, case.chunk_rows)
    recorder = sink._sink
    step = case.max_joint_vel / case.chunk_hz
    arm = measured_real[0].copy()
    goal = arm.copy()
    waypoints = times = start = start_t = None
    pending: list[tuple[float, np.ndarray]] = []
    seen = k = 0
    out = np.zeros((len(t), 14))
    push = []
    # 押す向き: 会場で「指令 − 実測」が平均で負なら、実測は指令より大きい側で止まっている
    sign = 0.0
    if wall_joint is not None:
        diff = action19[:, 3 + wall_joint] - measured_real[:, wall_joint]
        sign = -1.0 if np.mean(diff) < 0 else 1.0
    # 最後の tick まで回す (t[-1] / DT の切り捨てで最後の tick を落とさない)
    for i in range(int(np.ceil(t[-1] / DT)) + 2):
        now = i * DT
        while k < len(t) and t[k] <= now + 1e-9:
            clock.t = float(t[k])
            body = np.zeros(29)
            body[15:29] = arm
            sink.send_action(action19[k], body)
            out[k] = arm
            k += 1
        while seen < len(recorder.sent):
            sent_at, chunk = recorder.sent[seen]
            pending.append((sent_at + latency_s, chunk[: case.max_waypoints, 4:18]))
            seen += 1
        while pending and pending[0][0] <= now:
            _, rows = pending.pop(0)
            anchor = arm.copy()
            ws = []
            for row in rows:
                anchor = anchor + np.clip(row - anchor, -step, step)
                ws.append(anchor.copy())
            waypoints = np.array(ws)
            times = now + (np.arange(len(ws)) + 1) / case.chunk_hz
            start, start_t = goal.copy(), now
        if waypoints is not None:
            if now <= times[0]:
                target = start + (now - start_t) / (times[0] - start_t) * (
                    waypoints[0] - start
                )
            elif now >= times[-1]:
                target = waypoints[-1]
            else:
                j = np.searchsorted(times, now) - 1
                f = (now - times[j]) / (times[j + 1] - times[j])
                target = waypoints[j] + f * (waypoints[j + 1] - waypoints[j])
            if case.wbc_speed is None:
                goal = target
            else:
                limit = case.wbc_speed * DT
                goal = goal + np.clip(target - goal, -limit, limit)
        arm = arm + (goal - arm) / tau * DT
        if wall_joint is not None:
            wall = measured_real[min(max(k - 1, 0), len(t) - 1), wall_joint]
            # 押す向き (sign) の反対側が壁: 会場の実測より押す向きへは進めない
            if sign < 0 and arm[wall_joint] < wall:
                arm[wall_joint] = wall
            if sign > 0 and arm[wall_joint] > wall:
                arm[wall_joint] = wall
            push.append(max(sign * (goal[wall_joint] - wall), 0.0))
    sink.close()
    return {
        "measured": out,
        "published": len(recorder.sent),
        "push": np.asarray(push) if push else None,
    }


def compare(
    run: Path,
    skill: str,
    cases: list[Case],
    *,
    tau: float,
    wall_joint: Optional[str],
    skill_config: dict,
) -> list[dict]:
    segments = load_segments(run, skill)
    wall = ARM_NAMES.index(wall_joint) if wall_joint else None
    results = []
    for case in cases:
        sims, pushes, inst, published, seconds = [], [], [], 0, 0.0
        preview_s = _preview_s(case, skill_config)
        for t, action, measured in segments:
            r = simulate(
                t, with_preview(t, action, preview_s), measured, _assist(case, skill_config),
                tau=tau, case=case, wall_joint=wall,
            )  # fmt: skip
            m = r["measured"].copy()
            if wall is not None:
                m[:, wall] = action[:, 3 + wall]  # 遅れ・ずれは壁のない関節で見る
            sims.append((t, action[:, 3:17], m))
            inst.append(np.abs(action[:, 3:17] - m).mean())
            if r["push"] is not None:
                pushes.append(r["push"])
            published += r["published"]
            seconds += float(t[-1])
        metrics = tracking_metrics(sims)
        push = np.concatenate(pushes) if pushes else None
        results.append(
            {
                "case": case.label,
                "lag_ms": metrics["lag_ms"],
                # policy から見たずれ: その瞬間の |指令 − 実測| (遅れをそろえない)
                "instant_error_rad": float(np.mean(inst)),
                "aligned_error_rad": metrics["mean_abs_error_rad"],
                "push_mean_rad": None if push is None else float(push.mean()),
                "push_p90_rad": None
                if push is None
                else float(np.percentile(push, 90)),
                "publish_hz": published / seconds if seconds else None,
            }
        )
    return results


def default_cases(skill_config: dict) -> list[Case]:
    """09-29 の結論 (Issue #188) を確かめる組み合わせ。束と、会場で回せる adapter / WBC の引数。"""
    standard, _ = ArmTrackingConfig.from_config(skill_config, "standard")
    return [
        Case("today (09-29)", "today"),
        Case("standard (先の予定なし)", replace(standard, preview_s=0.0)),
        Case("standard", "standard"),
        Case("lead", "lead"),
        Case("standard + 上限 1.5", "standard", max_joint_vel=1.5),
        Case("standard + 上限 2.0", "standard", max_joint_vel=2.0),
        Case("standard + chunk-hz 50", "standard", chunk_hz=50.0),
        Case("standard + WBC 速さ上限なし", "standard", wbc_speed=None),
    ]  # fmt: skip


def format_results(title: str, results: list[dict]) -> str:
    lines = [title]
    for r in results:
        push = (
            f"  押す力 {r['push_mean_rad']:.3f} / 上位10% {r['push_p90_rad']:.3f}"
            if r["push_mean_rad"] is not None
            else ""
        )
        lines.append(
            f"  {r['case']:30s} 遅れ {r['lag_ms']:4.0f} ms  その瞬間のずれ {r['instant_error_rad']:.3f}"
            f"  (そろえた後 {r['aligned_error_rad']:.3f})  送る {r['publish_hz']:.1f}/s{push}"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="production_runs/<run>")
    parser.add_argument("--skill", required=True)
    parser.add_argument("--wall-joint", default=None, choices=ARM_NAMES)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(SKILL_CONFIG.read_text())
    results = compare(
        args.run, args.skill, default_cases(cfg),
        tau=args.tau, wall_joint=args.wall_joint, skill_config=cfg,
    )  # fmt: skip
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    else:
        title = f"== {args.run.name} {args.skill} (腕の反応 {args.tau:g}s、壁 {args.wall_joint or 'なし'})"
        print(format_results(title, results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
