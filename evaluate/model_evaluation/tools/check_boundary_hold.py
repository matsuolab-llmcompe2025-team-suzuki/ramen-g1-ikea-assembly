"""保持中に腕が指令から離れていかないかを、会場の boundary ログで確かめる (Issue #172)。

# なぜ要るか

運営 adapter は chunk の 1 行目を**実測の腕から ±max_joint_vel/chunk_hz** (既定
1.0/20 = 0.05 rad) に切り詰める (`wbc_driver.py:_step_clamp`)。運営の実測では、負荷の
かかった腕は指令から 0.05〜0.07 rad 遅れて止まる (PD、重力補償なし)。遅れがこの幅を
超えると、送るたびに指令が実測の側へ引き戻され、腕が下がり続ける (模擬: 遅れ 0.06 rad、
1 行 @30 Hz で 6 s に -0.74 rad)。

entrypoint は既定で「変わったときだけ・16 行」送るので保持中は下がらないはずだが、
遅れそのものの大きさと、動いている間の追従は実機でしか分からない。接続テストで
**腕を上げた姿勢の保持** (開始待ち・準備の保持など) を 10 s 以上含む run を 1 回回し、
その `--log` をこれに通す。

# 使い方

    python evaluate/model_evaluation/tools/check_boundary_hold.py --log <orch_log.jsonl>

`event=boundary_joint` (joint lane) の記録 (送った `(T,22)` の 1 行と、そのときの実測
`measured_body_q29`) から、腕の指令が変わらない区間 (`--min-hold-s` 以上) ごとに
「指令 − 実測」の始めと終わり、流れる速さを関節ごとに出す。

# 判定

- ``DRIFT``: 指令が同じなのに、どれかの関節で差が ``--drift-rad`` 以上広がり、速さが
  ``--drift-rate`` 以上。adapter の切り詰めで腕が引きずられている。送り方
  (`--boundary-publish-period-s` / `--boundary-chunk-rows`) を確認し、結果を運営に共有する。
- ``SAG``: 保持の終わりの差が切り詰め幅 (max_joint_vel/chunk_hz) を超える関節がある。
  保持はできても、動かす間は追従が遅れる。結果を運営に共有して判断を仰ぐ。

adapter の ``--max-joint-vel`` は自分たちで上げない。運営は WBC の安全監視 (6 rad/s) での
停止を受けて 4.0 → 2.0 → 1.0 と下げており (実速度が指令上限の 3.6〜3.9 倍になる原因が
未解明)、2.0 では -7.7 rad/s での停止例がある。
- ``OK``: どちらでもない。

重力の垂れ補正 (`--boundary-gravity-offset on`、既定) のときは、記録の `arms_intent`
(policy の目標) と実測の差で判定する (補正が合っていればほぼ 0)。あわせて保持中の
「送った腕 − 実測」(= 実際の垂れ) を補正量で割った比から、`--boundary-gravity-offset-scale`
の提案値を出す (1 前後なら YAML の kp が会場の WBC と合っている)。

pose lane (`boundary_taskspace`) の記録は腕の指令角を持たないので対象外。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

#: 腕 14 関節 ((T,22) の [4:18] と body_q[15:29] の並び、Unitree G1JointIndex 順)
ARM_JOINTS = (
    "L.shoulder_pitch", "L.shoulder_roll", "L.shoulder_yaw", "L.elbow",
    "L.wrist_roll", "L.wrist_pitch", "L.wrist_yaw",
    "R.shoulder_pitch", "R.shoulder_roll", "R.shoulder_yaw", "R.elbow",
    "R.wrist_roll", "R.wrist_pitch", "R.wrist_yaw",
)
JOINT_ROW_ARMS = slice(4, 18)
BODY_Q_ARMS = slice(15, 29)


@dataclass(frozen=True)
class Sample:
    t_s: float
    command: np.ndarray  # (14,) 狙う腕 (補正ありなら arms_intent、無ければ送った腕)
    measured: np.ndarray  # (14,)
    sent: np.ndarray | None = None  # (14,) 実際に送った腕 (補正ありのときだけ)
    offset_scale1: np.ndarray | None = None  # (14,) 倍率 1 に直した補正量


@dataclass(frozen=True)
class HoldResult:
    start_s: float
    duration_s: float
    samples: int
    worst_joint: str
    start_error_rad: float
    end_error_rad: float
    drift_rate_rad_s: float
    verdict: str
    #: 重力の垂れ補正の倍率の提案 (補正ありの記録で、補正量 0.01 rad 超の関節があるとき)
    suggested_offset_scale: float | None = None


def load_samples(lines: Iterable[str]) -> list[Sample]:
    """`boundary_joint` の記録を時刻順の Sample にする (壊れた行・他の event は飛ばす)。"""
    samples: list[Sample] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != "boundary_joint":
            continue
        row = record.get("joint_22")
        body = record.get("measured_body_q29")
        stamp = record.get("monotonic_ns")
        if row is None or body is None or stamp is None:
            continue
        row = np.asarray(row, dtype=np.float64)
        body = np.asarray(body, dtype=np.float64)
        if row.shape != (22,) or body.shape != (29,):
            continue
        if record.get("state_stale"):
            continue  # 古い実測は比べない
        sent = row[JOINT_ROW_ARMS].copy()
        intent = record.get("arms_intent")
        offset = record.get("gravity_offset_rad")
        scale = record.get("gravity_offset_scale")
        if intent is not None and offset is not None and scale:
            samples.append(Sample(
                float(stamp) * 1e-9,
                np.asarray(intent, dtype=np.float64),
                body[BODY_Q_ARMS].copy(),
                sent=sent,
                offset_scale1=np.asarray(offset, dtype=np.float64) / float(scale),
            ))
        else:
            samples.append(Sample(float(stamp) * 1e-9, sent, body[BODY_Q_ARMS].copy()))
    samples.sort(key=lambda s: s.t_s)
    return samples


def hold_segments(samples: list[Sample], *, min_hold_s: float) -> list[list[Sample]]:
    """腕の指令が変わらない連続区間 (min_hold_s 以上) を返す。"""
    segments: list[list[Sample]] = []
    current: list[Sample] = []
    for sample in samples:
        if current and not np.array_equal(sample.command, current[0].command):
            segments.append(current)
            current = []
        current.append(sample)
    if current:
        segments.append(current)
    return [
        seg for seg in segments
        if len(seg) >= 2 and seg[-1].t_s - seg[0].t_s >= min_hold_s
    ]


def assess_hold(
    segment: list[Sample], *, clamp_rad: float, drift_rad: float, drift_rate: float
) -> HoldResult:
    t = np.asarray([s.t_s for s in segment])
    error = np.asarray([s.command - s.measured for s in segment])  # (N, 14)
    start = np.abs(error[0])
    end = np.abs(error[-1])
    span = t - t[0]
    # 「指令から離れる向き」の速さ = |誤差| の時間変化の傾き
    slopes = (
        np.polyfit(span, np.abs(error), 1)[0]
        if span[-1] > 0 else np.zeros(14)
    )
    grew = end - start
    drifting = (grew >= drift_rad) & (slopes >= drift_rate)
    if drifting.any():
        worst = int(np.argmax(np.where(drifting, grew, -np.inf)))
        verdict = "DRIFT"
    elif (end > clamp_rad).any():
        worst = int(np.argmax(end))
        verdict = "SAG"
    else:
        worst = int(np.argmax(end))
        verdict = "OK"
    suggested = None
    last = segment[-1]
    if last.sent is not None and last.offset_scale1 is not None:
        # 保持の終わりの実際の垂れ (送った腕 − 実測) / 倍率 1 の補正量 = 会場の kp に合う倍率
        usable = np.abs(last.offset_scale1) > 0.01
        if usable.any():
            ratio = (last.sent - last.measured)[usable] / last.offset_scale1[usable]
            suggested = float(np.median(ratio))
    return HoldResult(
        suggested_offset_scale=suggested,
        start_s=float(t[0]),
        duration_s=float(span[-1]),
        samples=len(segment),
        worst_joint=ARM_JOINTS[worst],
        start_error_rad=float(start[worst]),
        end_error_rad=float(end[worst]),
        drift_rate_rad_s=float(slopes[worst]),
        verdict=verdict,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, required=True, help="entrypoint の --log (JSONL)")
    parser.add_argument("--min-hold-s", type=float, default=2.0)
    parser.add_argument(
        "--max-joint-vel", type=float, default=1.0,
        help="運営 adapter の --max-joint-vel (既定 1.0 rad/s)",
    )
    parser.add_argument("--chunk-hz", type=float, default=20.0, help="運営 adapter の --chunk-hz")
    parser.add_argument("--drift-rad", type=float, default=0.03)
    parser.add_argument("--drift-rate", type=float, default=0.01, help="rad/s")
    parser.add_argument("--json", action="store_true", help="結果を JSON で出す")
    args = parser.parse_args(argv)

    samples = load_samples(args.log.read_text(encoding="utf-8").splitlines())
    if not samples:
        print(f"no boundary_joint records with monotonic_ns in {args.log}")
        return 2
    clamp_rad = args.max_joint_vel / args.chunk_hz
    t0 = samples[0].t_s
    results = [
        assess_hold(seg, clamp_rad=clamp_rad, drift_rad=args.drift_rad,
                    drift_rate=args.drift_rate)
        for seg in hold_segments(samples, min_hold_s=args.min_hold_s)
    ]
    if args.json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        print(f"{len(samples)} samples, {len(results)} hold(s) >= {args.min_hold_s:g}s, "
              f"adapter clamp {clamp_rad:.3f} rad/row")
        for r in results:
            scale = (
                "" if r.suggested_offset_scale is None
                else f"  gravity offset scale ~{r.suggested_offset_scale:.2f}"
            )
            print(f"  t=+{r.start_s - t0:7.1f}s {r.duration_s:5.1f}s {r.verdict:<5} "
                  f"{r.worst_joint:<17} |target-meas| {r.start_error_rad:.3f} -> "
                  f"{r.end_error_rad:.3f} rad ({r.drift_rate_rad_s:+.3f} rad/s){scale}")
        scales = [r.suggested_offset_scale for r in results if r.suggested_offset_scale is not None]
        if scales:
            print(f"suggested --boundary-gravity-offset-scale {float(np.median(scales)):.2f} "
                  "(about 1 = the YAML kp matches the venue WBC; far from 1: check the "
                  "WBC's MOTOR_KP on PC2)")
        verdicts = {r.verdict for r in results}
        if "DRIFT" in verdicts:
            print("DRIFT: the arm moves away from a constant command. Check the publish "
                  "pacing flags and share this result with the organizer.")
        elif "SAG" in verdicts:
            print(f"SAG: steady error exceeds the {clamp_rad:.3f} rad adapter clamp; holds "
                  "are fine but motion will lag. Share this result with the organizer "
                  "(do not raise the adapter's --max-joint-vel yourselves).")
        elif not results:
            print("no hold long enough; record a run with a >= 10 s raised-arm hold")
        else:
            print("OK: every hold stays within the adapter clamp")
    return 1 if any(r.verdict == "DRIFT" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
