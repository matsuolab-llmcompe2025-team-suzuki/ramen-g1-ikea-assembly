"""HARA curated per-skill dataset の length column から hard timeout の推奨値を出す。

各 skill の training episode 長 (frame 単位) を meta parquet から集めて、min /
p50 / p90 / p95 / max を frames と seconds 両方で出力する。単体 skill runner
に per-skill hard timeout を入れる際の base 値決めに使う。

# 想定 formula

    hard_timeout_s = ceil(p95_frames / fps × 1.5)

- p95: 訓練 episode の上位 5% は「難しめ demo」= 実 policy でも起きうる遅延目安
- × 1.5: policy は teleop より慎重で遅くなる傾向、初回 chunk warmup 5-10s も
  吸収したいが、arm pre-motion / hand init overhead は timeout の外側で数える
  想定 (script は policy 純粋実行時間だけ計算、orchestrator 側で加算する)

# 使い方

```bash
cd iros_2026_ramen/inference/desktop  # pyarrow + huggingface_hub が揃った env
set -a && . ../../.env && set +a  # HF_TOKEN 必要 (Team-RAMEN read 権限)
pixi run python -m evaluate.model_evaluation.tools.compute_skill_timeout
```

引数無し = 6 skill 全部集計、default 出力先へ書き込み:

- CSV: `evaluate/model_evaluation/tools/skill_timeout_stats.csv`
- Markdown: `evaluate/model_evaluation/tools/skill_timeout_report.md`

`--skill <name>` で単一 skill 集計、`--refresh` で cache 無視して再 DL。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq


# skill_config.yaml と 1:1 対応。skill ごとに 1 つの dataset (chunk は 2026-10-06 に消した、Issue #190)。
SKILL_REPO_MAP: dict[str, list[str]] = {
    "move_table_base": [
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-move-table-base",
    ],
    "rotate_table_base": [
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base",
    ],
    "pick_table_leg": [
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-pick-table-leg",
    ],
    "insert_table_leg": [
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-insert-table-leg",
    ],
    "rotate_leg_to_tighten": [
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-leg-to-tighten",
    ],
    "flip_table": [
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-flip-table",
    ],
}

P95_SAFETY_FACTOR = 1.5
DEFAULT_OUT_DIR = Path(__file__).parent


@dataclass(frozen=True)
class SkillStats:
    skill: str
    repo_count: int
    n_episodes: int
    fps: float
    length_min: int
    length_p50: int
    length_p90: int
    length_p95: int
    length_max: int

    @property
    def sec_min(self) -> float:
        return self.length_min / self.fps

    @property
    def sec_p50(self) -> float:
        return self.length_p50 / self.fps

    @property
    def sec_p90(self) -> float:
        return self.length_p90 / self.fps

    @property
    def sec_p95(self) -> float:
        return self.length_p95 / self.fps

    @property
    def sec_max(self) -> float:
        return self.length_max / self.fps

    @property
    def recommended_hard_timeout_s(self) -> int:
        return math.ceil(self.sec_p95 * P95_SAFETY_FACTOR)


def _download_meta(repo_id: str, refresh: bool) -> Path:
    """Snapshot download meta only. Return snapshot dir Path."""
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "allow_patterns": ["meta/**"],
    }
    if refresh:
        kwargs["force_download"] = True
    try:
        path = snapshot_download(**kwargs)
    except Exception as exc:  # 401 / 404 / network etc.
        raise RuntimeError(
            f"[dl] failed to fetch {repo_id!r} meta: {type(exc).__name__}: {exc}"
        ) from exc
    return Path(path)


def _read_fps(snapshot_root: Path) -> float:
    import json

    info_path = snapshot_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing meta/info.json in {snapshot_root}")
    info = json.loads(info_path.read_text())
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError(f"invalid fps={fps} in {info_path}")
    return fps


def _iter_episodes_parquets(snapshot_root: Path) -> Iterable[Path]:
    root = snapshot_root / "meta" / "episodes"
    if not root.exists():
        return
    for chunk_dir in sorted(root.glob("chunk-*")):
        for parquet in sorted(chunk_dir.glob("file-*.parquet")):
            yield parquet


def _collect_lengths(snapshot_root: Path) -> np.ndarray:
    lengths: list[int] = []
    # v3: meta/episodes/chunk-XXX/file-XXX.parquet
    v3_parquets = list(_iter_episodes_parquets(snapshot_root))
    if v3_parquets:
        for parquet in v3_parquets:
            table = pq.read_table(parquet, columns=["length"])
            col = table.column("length").to_numpy(zero_copy_only=False)
            lengths.extend(int(v) for v in col)
    else:
        # v2: meta/episodes.jsonl (1 line per episode with `length` field)
        import json

        jsonl = snapshot_root / "meta" / "episodes.jsonl"
        if not jsonl.exists():
            raise FileNotFoundError(
                f"neither meta/episodes/ (v3) nor meta/episodes.jsonl (v2) in {snapshot_root}"
            )
        with jsonl.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                lengths.append(int(json.loads(line)["length"]))
    if not lengths:
        raise RuntimeError(f"no episodes found under {snapshot_root}")
    return np.asarray(lengths, dtype=np.int64)


def compute_skill_stats(skill: str, repo_ids: list[str], refresh: bool) -> SkillStats:
    all_lengths: list[int] = []
    fps_values: list[float] = []
    for repo_id in repo_ids:
        print(f"  [{skill}] downloading meta: {repo_id}", flush=True)
        snapshot = _download_meta(repo_id, refresh=refresh)
        fps_values.append(_read_fps(snapshot))
        lengths = _collect_lengths(snapshot)
        all_lengths.extend(int(v) for v in lengths)

    fps_set = set(fps_values)
    if len(fps_set) > 1:
        print(
            f"  [{skill}] WARNING: fps mismatch across repos: {sorted(fps_set)}"
            " -- using first repo's fps",
            file=sys.stderr,
        )
    fps = fps_values[0]
    if fps != 30.0:
        print(f"  [{skill}] WARNING: unexpected fps={fps} (script assumes 30)", file=sys.stderr)

    arr = np.asarray(all_lengths, dtype=np.int64)
    return SkillStats(
        skill=skill,
        repo_count=len(repo_ids),
        n_episodes=int(arr.size),
        fps=fps,
        length_min=int(arr.min()),
        length_p50=int(np.percentile(arr, 50)),
        length_p90=int(np.percentile(arr, 90)),
        length_p95=int(np.percentile(arr, 95)),
        length_max=int(arr.max()),
    )


def emit_stdout(rows: list[SkillStats]) -> None:
    print()
    print(f"{'skill':<24} {'n_ep':>6} {'p50s':>7} {'p95s':>7} {'maxs':>7} {'rec_s':>7}")
    print("-" * 62)
    for r in rows:
        print(
            f"{r.skill:<24} {r.n_episodes:>6} "
            f"{r.sec_p50:>7.1f} {r.sec_p95:>7.1f} {r.sec_max:>7.1f} "
            f"{r.recommended_hard_timeout_s:>7d}"
        )


def emit_csv(rows: list[SkillStats], path: Path) -> None:
    lines = [
        "skill,repo_count,n_episodes,fps,"
        "length_min,length_p50,length_p90,length_p95,length_max,"
        "sec_min,sec_p50,sec_p90,sec_p95,sec_max,"
        "recommended_hard_timeout_s"
    ]
    for r in rows:
        lines.append(
            f"{r.skill},{r.repo_count},{r.n_episodes},{r.fps:g},"
            f"{r.length_min},{r.length_p50},{r.length_p90},{r.length_p95},{r.length_max},"
            f"{r.sec_min:.3f},{r.sec_p50:.3f},{r.sec_p90:.3f},{r.sec_p95:.3f},{r.sec_max:.3f},"
            f"{r.recommended_hard_timeout_s}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_markdown(rows: list[SkillStats], path: Path) -> None:
    header = (
        "# Per-skill hard timeout 推奨値 (HARA curated dataset length から算出)\n"
        "\n"
        "Auto-generated by `evaluate/model_evaluation/tools/compute_skill_timeout.py`。\n"
        "各 skill の HARA per-task datasets の `meta/episodes/**/*.parquet.length` を集計。\n"
        "\n"
        f"**推奨式**: `hard_timeout_s = ceil(p95_frames / fps × {P95_SAFETY_FACTOR})`\n"
        "\n"
        "**注意**:\n"
        "- policy 純粋実行時間 (chunk warmup 含む)。arm pre-motion / hand init overhead は\n"
        "  timeout の外側で加算する想定 (orchestrator 側)。\n"
        "- 訓練 episode は成功 demo のみ ⇒ p95 は最良ケース側に寄る bias あり。\n"
        "  実機 N=3 trial 後に slide 前提 (max × 1.3 で置き直す)。\n"
        "\n"
        "## 集計結果\n"
        "\n"
        "| skill | n_ep | fps | min | p50 | p90 | p95 | max | 推奨 hard timeout |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body: list[str] = []
    for r in rows:
        body.append(
            f"| `{r.skill}` | {r.n_episodes} | {r.fps:g} | "
            f"{r.sec_min:.1f}s | {r.sec_p50:.1f}s | {r.sec_p90:.1f}s | "
            f"{r.sec_p95:.1f}s | {r.sec_max:.1f}s | **{r.recommended_hard_timeout_s}s** |"
        )
    footer = (
        "\n\n## Frame 単位 (30fps 前提)\n"
        "\n"
        "| skill | n_ep | min | p50 | p90 | p95 | max |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in rows:
        footer += (
            f"| `{r.skill}` | {r.n_episodes} | "
            f"{r.length_min} | {r.length_p50} | {r.length_p90} | "
            f"{r.length_p95} | {r.length_max} |\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(body) + footer, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skill",
        default=None,
        choices=sorted(SKILL_REPO_MAP.keys()),
        help="単一 skill だけ集計 (default: 全 6 skill)",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="HF cache 無視して再 DL",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_OUT_DIR / "skill_timeout_stats.csv",
    )
    ap.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_OUT_DIR / "skill_timeout_report.md",
    )
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print(
            "[error] HF_TOKEN が未設定。Team-RAMEN read 権限のトークンを export してください:\n"
            "    set -a && . <repo>/.env && set +a",
            file=sys.stderr,
        )
        sys.exit(2)

    skills = [args.skill] if args.skill else list(SKILL_REPO_MAP.keys())
    rows: list[SkillStats] = []
    for skill in skills:
        repo_ids = SKILL_REPO_MAP[skill]
        print(f"[skill] {skill} ({len(repo_ids)} repo{'s' if len(repo_ids) > 1 else ''})", flush=True)
        stats = compute_skill_stats(skill, repo_ids, refresh=args.refresh)
        rows.append(stats)

    emit_stdout(rows)
    emit_csv(rows, args.csv)
    emit_markdown(rows, args.markdown)
    print()
    print(f"[out] CSV      : {args.csv}")
    print(f"[out] Markdown : {args.markdown}")


if __name__ == "__main__":
    main()
