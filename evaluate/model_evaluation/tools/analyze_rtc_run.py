"""実機評価 run から RTC / async / TE の効き具合を数値化する (Issue #137)。

`docs/handoff/realmachine_rotate_table_base_eval.md` の実行層診断を
毎回同じ手順で出すための script。#137 の診断は ad-hoc に行われて手順が残らず、
同じ比較を再現できなかったので、ここに固定する。

# 何を出すか

1. **手先の 1.2-3.0Hz 帯振幅 [mm]** — #137 の主症状 (chunk 境界の不連続による
   手先振動) の直接指標。実測関節 (`state_joint_00-28`) を URDF FK に通した
   手先位置から算出する。
2. **指令速度** — arms 14D の per-tick |Δ| の median。teacher 比を見たい時は
   `--teacher-dq` に teacher の値を渡す (実測: rotate 0.011 / insert 0.00772)。
3. **hold 率** — 指令が前 tick と完全一致した tick の割合。#137 で
   「集計値が teacher 一致に見えても hold 混入で歪んでいた」事例があったので、
   指令速度は hold 込み / hold 除外の両方を出す。
4. **実行層の診断集計** — `chunk_source` / `blended_from_n_candidates` /
   `rtc_*` の分布。handoff §A-2 の表を手作業でなく埋めるためのもの。
5. **tick 周期の実測 [ms] と制御周波数 [Hz]**。

# 出せないもの

chunk 境界の position jump / 速度方向 cos (本家 deployment guide の残り 2 metric)
は **chunk 全体がログされていない**ため計算できない (`actions.jsonl` は dispatch
した 1 行だけを持つ)。ただしそれらは「手先が振動するか」の proxy であり、本 script
は実測関節から本体を直接測っているので、判定には支障しない。

# 帯振幅の定義

policy_time_s は僅かに不均一なので median 周期の等間隔格子へ線形補間してから
rFFT する。軸ごとに振幅スペクトル `A_k = 2|X_k| / N` [mm] を取り、1.2-3.0Hz の
帯で二乗和平方根を取って軸を合成する。**この定義は本 script 固有**なので、
比較は必ず同じ script の出力同士で行うこと。

特に memory `reference_groot_chunk_jitter_findings` に残っている値
(teacher 左 3.85 / 右 4.72 mm 等) は ad-hoc な別定義で算出されたもので、
**本 script の出力と直接比較してはいけない**。本 script は run 間の相対比較
(RTC on/off など、同一 model・同一 skill での比較) に使う。

指令 |dq| も同様で、teacher 値の定義 (平均 joint か最大 joint か) が揃って
いないと比が意味を持たない。本 script は arm 14 joint の**平均**を
`--teacher-dq` の比較対象に使う。

# 使い方

```bash
# HF_TOKEN / WANDB_API_KEY が要る (.env)
set -a && . .env && set +a

# 単一 run
pixi run -e runtime python -m evaluate.model_evaluation.tools.analyze_rtc_run \\
    --run 0xp667og

# 複数 run を比較 (Phase B の 4 run など)。--output で markdown 表を書き出す
pixi run -e runtime python -m evaluate.model_evaluation.tools.analyze_rtc_run \\
    --run <B-1> --run <B-2> --run <B-3> --run <B-4> \\
    --teacher-dq 0.011 --output outputs/rtc_phase_b.md
```
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


DEFAULT_ENTITY = "ken05-matuo-llm-88_llm_2025_suzuki"
DEFAULT_PROJECT = "iros-2026-ramen-real-policy-evaluation"

# #137 で実測した手先振動の帯域。chunk 入替周波数 (1.5-2.0Hz) を含む。
BAND_LOW_HZ = 1.2
BAND_HIGH_HZ = 3.0

# VlaSkill 19D contract の arms slice (waist 3 + arms 14 + hand 2)。
ARMS_SLICE = slice(3, 17)

# `policy_metadata` から拾う実行層の診断キー。
_RTC_SCALAR_KEYS = (
    "rtc_enabled",
    "rtc_prefix_rows",
    "rtc_frozen_steps",
    "rtc_overlap_steps",
    "rtc_async_execution",
    "rtc_error",
)


@dataclass
class RunSummary:
    """1 run 分の集計結果。"""

    name: str
    ticks: int
    tick_period_ms: float
    control_hz: float
    predict_latency_ms: dict[str, float]
    hand_band_mm: dict[str, float]
    hand_peak_hz: dict[str, float]
    command_dq: dict[str, float]
    hold_fraction: float
    chunk_source: dict[str, int]
    blended_candidates: dict[str, int]
    rtc: dict[str, object] = field(default_factory=dict)


# ---- 入出力 ---- #


def _load_run_files(run_id: str, entity: str, project: str, cache: Path) -> tuple[Path, Path]:
    """wandb から actions.jsonl と state_action_trace table を落とす。"""
    import wandb  # lazy: env-isolated (runtime env only)

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    target = cache / run_id
    target.mkdir(parents=True, exist_ok=True)

    actions = run.file("actions.jsonl").download(root=str(target), replace=True)
    traces = [f for f in run.files() if "state_action_trace" in f.name]
    if not traces:
        raise FileNotFoundError(f"run {run_id} has no state_action_trace table")
    trace = traces[0].download(root=str(target), replace=True)
    return Path(actions.name), Path(trace.name)


def _read_trace(path: Path) -> tuple[list[str], np.ndarray]:
    table = json.loads(path.read_text(encoding="utf-8"))
    columns = list(table["columns"])
    data = np.asarray(table["data"], dtype=np.float64)
    return columns, data


def _column_block(columns: list[str], data: np.ndarray, prefix: str) -> np.ndarray:
    """`prefix` で始まる列を番号順にまとめて (rows, n) で返す。"""
    idx = sorted(
        (i for i, c in enumerate(columns) if c.startswith(prefix)),
        key=lambda i: columns[i],
    )
    if not idx:
        raise KeyError(f"trace table has no column starting with {prefix!r}")
    return data[:, idx]


# ---- 指標 ---- #


def band_amplitude_mm(
    positions_m: np.ndarray, times_s: np.ndarray
) -> tuple[float, float]:
    """(rows, 3) 手先位置 → 帯域振幅 [mm] と帯内のピーク周波数 [Hz]。

    不均一な時刻を median 周期の等間隔格子へ線形補間してから rFFT する。
    """
    if positions_m.shape[0] < 8:
        return float("nan"), float("nan")
    dt = float(np.median(np.diff(times_s)))
    if not np.isfinite(dt) or dt <= 0.0:
        return float("nan"), float("nan")
    grid = np.arange(times_s[0], times_s[-1], dt)
    if grid.size < 8:
        return float("nan"), float("nan")

    freqs = np.fft.rfftfreq(grid.size, d=dt)
    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    if not band.any():
        return float("nan"), float("nan")

    per_axis_power = np.zeros(int(band.sum()))
    total = 0.0
    for axis in range(positions_m.shape[1]):
        resampled = np.interp(grid, times_s, positions_m[:, axis])
        spectrum = np.fft.rfft(resampled - resampled.mean())
        amplitude_mm = 2.0 * np.abs(spectrum) / grid.size * 1000.0
        selected = amplitude_mm[band]
        per_axis_power += selected**2
        total += float(np.sum(selected**2))
    peak_hz = float(freqs[band][int(np.argmax(per_axis_power))])
    return float(np.sqrt(total)), peak_hz


def _percentiles(values: list[float]) -> dict[str, float]:
    """percentile と **サンプル数** を返す。

    `n` を必ず持たせるのは、async 経路では `latency_ms` が非ゼロになるのが seed
    の同期推論 1 tick だけで、median / p90 / max が同一の 1 サンプルになるため
    (実測: RAMEN-Ori async 1/899、GR00T async 1/898、sync のみ 542/542)。
    3 つ並ぶと分布に見えるので、読み手が n を見ずに解釈できないようにする。
    """
    if not values:
        return {
            "median": float("nan"), "p90": float("nan"), "max": float("nan"), "n": 0.0
        }
    ordered = sorted(values)
    return {
        "median": float(statistics.median(ordered)),
        "p90": float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
        "max": float(ordered[-1]),
        "n": float(len(ordered)),
    }


def summarize(name: str, actions_path: Path, trace_path: Path) -> RunSummary:
    columns, data = _read_trace(trace_path)
    times = data[:, columns.index("policy_time_s")]
    latency = data[:, columns.index("latency_ms")]
    joints = _column_block(columns, data, "state_joint_")
    actions = _column_block(columns, data, "action_")

    dt_ms = np.diff(times) * 1000.0
    tick_period_ms = float(np.median(dt_ms)) if dt_ms.size else float("nan")

    # 実測関節 → URDF FK → 手先位置 (pelvis frame、m)
    from inference.desktop.perception.g1_urdf_fk import (  # lazy: URDF parse を持つ
        DEFAULT_URDF_PATH,
        G1WristFK,
    )

    fk = G1WristFK.from_urdf(DEFAULT_URDF_PATH)
    ee = np.stack([fk.compute_ee_state(row) for row in joints])  # (rows, 12)
    left_mm, left_hz = band_amplitude_mm(ee[:, 0:3], times)
    right_mm, right_hz = band_amplitude_mm(ee[:, 6:9], times)

    # 指令速度。hold (前 tick と完全一致) は #137 で集計を歪めた実績があるので分ける
    arms = actions[:, ARMS_SLICE]
    delta = np.abs(np.diff(arms, axis=0))
    step_mean = delta.mean(axis=1)   # arm 14 joint の平均
    step_max = delta.max(axis=1)     # 最も動いた joint
    holds = step_max == 0.0
    hold_fraction = float(holds.mean()) if step_max.size else float("nan")
    moving_mean = step_mean[~holds]

    metadata = [json.loads(line) for line in actions_path.read_text().splitlines() if line]
    sources = Counter(m["policy_metadata"].get("chunk_source") for m in metadata)
    blended = Counter(
        m["policy_metadata"].get("blended_from_n_candidates") for m in metadata
    )
    rtc: dict[str, object] = {}
    for key in _RTC_SCALAR_KEYS:
        seen = Counter(
            m["policy_metadata"][key] for m in metadata if key in m["policy_metadata"]
        )
        if seen:
            rtc[key] = dict(seen)

    return RunSummary(
        name=name,
        ticks=len(metadata),
        tick_period_ms=tick_period_ms,
        control_hz=1000.0 / tick_period_ms if tick_period_ms else float("nan"),
        predict_latency_ms=_percentiles([float(x) for x in latency if x > 0.5]),
        hand_band_mm={"left": left_mm, "right": right_mm},
        hand_peak_hz={"left": left_hz, "right": right_hz},
        command_dq={
            "mean_all": float(np.median(step_mean)) if step_mean.size else float("nan"),
            "mean_moving": (
                float(np.median(moving_mean)) if moving_mean.size else float("nan")
            ),
            "max_all": float(np.median(step_max)) if step_max.size else float("nan"),
        },
        hold_fraction=hold_fraction,
        chunk_source={str(k): v for k, v in sorted(sources.items(), key=lambda kv: str(kv[0]))},
        blended_candidates={str(k): v for k, v in sorted(blended.items(), key=lambda kv: str(kv[0]))},
        rtc=rtc,
    )


# ---- 出力 ---- #


def render(summaries: list[RunSummary], teacher_dq: float | None) -> str:
    lines: list[str] = []
    for s in summaries:
        lines.append(f"## {s.name}")
        lines.append("")
        lines.append(f"- ticks: {s.ticks}")
        lines.append(
            f"- tick 周期: {s.tick_period_ms:.1f} ms ({s.control_hz:.1f} Hz)"
        )
        lat = s.predict_latency_ms
        lines.append(
            f"- predict latency: median {lat['median']:.1f} / p90 {lat['p90']:.1f} "
            f"/ max {lat['max']:.1f} ms (n={int(lat['n'])})"
        )
        lines.append(
            f"- **手先 {BAND_LOW_HZ}-{BAND_HIGH_HZ}Hz 帯振幅**: "
            f"左 {s.hand_band_mm['left']:.2f} mm ({s.hand_peak_hz['left']:.2f} Hz) / "
            f"右 {s.hand_band_mm['right']:.2f} mm ({s.hand_peak_hz['right']:.2f} Hz)"
        )
        dq = s.command_dq
        ratio = ""
        if teacher_dq:
            ratio = (
                f" (teacher 比 {dq['mean_all'] / teacher_dq:.2f}x / "
                f"hold 除外 {dq['mean_moving'] / teacher_dq:.2f}x)"
            )
        lines.append(
            f"- 指令 per-tick |dq| (arm 14 joint 平均): 全体 {dq['mean_all']:.5f} / "
            f"hold 除外 {dq['mean_moving']:.5f} rad{ratio}"
        )
        lines.append(
            f"- 指令 per-tick |dq| (最大 joint): {dq['max_all']:.5f} rad"
        )
        lines.append(f"- hold 率: {s.hold_fraction:.1%}")
        lines.append(f"- chunk_source: {s.chunk_source}")
        lines.append(f"- blended_from_n_candidates: {s.blended_candidates}")
        lines.append(f"- rtc: {s.rtc if s.rtc else '(診断キー無し = 旧 build の run)'}")
        lines.append("")

    if len(summaries) > 1:
        lines.append("## 比較")
        lines.append("")
        lines.append("| run | Hz | 手先帯振幅 左/右 [mm] | hold | blend率 | RTC |")
        lines.append("|---|---:|---|---:|---:|---|")
        for s in summaries:
            total = sum(s.blended_candidates.values()) or 1
            blend_rate = (
                sum(v for k, v in s.blended_candidates.items() if k.isdigit() and int(k) >= 2)
                / total
            )
            enabled = s.rtc.get("rtc_enabled", {})
            rtc_state = "on" if isinstance(enabled, dict) and enabled.get(True) else "off"
            lines.append(
                f"| {s.name} | {s.control_hz:.1f} | "
                f"{s.hand_band_mm['left']:.2f} / {s.hand_band_mm['right']:.2f} | "
                f"{s.hold_fraction:.0%} | {blend_rate:.0%} | {rtc_state} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", default=[], help="wandb run id (複数可)")
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument(
        "--local-dir", type=Path, action="append", default=[],
        help="actions.jsonl と state_action_trace*.table.json を含む directory",
    )
    p.add_argument(
        "--cache", type=Path, default=Path("outputs/rtc_analysis_cache"),
        help="wandb からの download 先",
    )
    p.add_argument(
        "--teacher-dq", type=float, default=None,
        help="teacher の per-tick |dq| [rad] (rotate 0.011 / insert 0.00772)",
    )
    p.add_argument("--output", type=Path, default=None, help="markdown 書き出し先")
    args = p.parse_args(argv)

    if not args.run and not args.local_dir:
        p.error("--run か --local-dir のどちらかが要る")

    summaries: list[RunSummary] = []
    for run_id in args.run:
        actions, trace = _load_run_files(run_id, args.entity, args.project, args.cache)
        summaries.append(summarize(run_id, actions, trace))
    for directory in args.local_dir:
        actions = directory / "actions.jsonl"
        traces = sorted(directory.rglob("*state_action_trace*.table.json"))
        if not traces:
            raise FileNotFoundError(f"{directory} has no state_action_trace table")
        summaries.append(summarize(directory.name, actions, traces[0]))

    report = render(summaries, args.teacher_dq)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"[analyze_rtc_run] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
