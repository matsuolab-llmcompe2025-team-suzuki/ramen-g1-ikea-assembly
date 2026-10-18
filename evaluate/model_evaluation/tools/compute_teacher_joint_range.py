"""教師データで腕 14 関節が取った min / max を出す (Issue #141 束 1-4 / D5-1)。

推論は「教師の範囲を出た関節だけ、端から margin_rad 内側へ戻す」補正を入れる
(`inference/desktop/lower_policy/skills/teacher_range.py`)。その範囲をここで実測する。
基準は p1 / p99 ではなく min / max (09-09 の実測で、p1/p99 だと補正が常時かかるため)。

出力は `skill_config.yaml` の `skills.<name>.teacher_joint_range` にそのまま貼れる YAML。
統計は `compute_initial_pose_median.py` と同じ `action.robot_q_desired[22:36]` (arm 14D) から取る。

memory: parquet は batch 単位で読み、min / max だけを持ち回る (全 frame を materialize しない)。
実測の peak RSS を最後に出す。

# 使い方

```bash
cd iros_2026_ramen/model/ramen_ori   # pyarrow + huggingface_hub がある env
export PYTHONPATH=$(git rev-parse --show-toplevel)   # inference/ と evaluate/ を import する
# 1) 落とす前に data parquet の合計容量を見る
pixi run python -m evaluate.model_evaluation.tools.compute_teacher_joint_range \
    --dataset Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base \
    --revision 9fb876364c8bf436446a36487dad3353802ee412 \
    --skill-name rotate_table_base --show-size
# 2) 計算する (data parquet は HF の cache に入る)
pixi run python -m evaluate.model_evaluation.tools.compute_teacher_joint_range \
    --dataset Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base \
    --revision 9fb876364c8bf436446a36487dad3353802ee412 \
    --skill-name rotate_table_base \
    --output /tmp/rotate_table_base_teacher_range.yaml
```
"""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pyarrow.parquet as pq

from inference.desktop.lower_policy.pose_utils import JOINT_NAMES, NUM_ARM_JOINTS

ACTION_COLUMN = "action.robot_q_desired"
ACTION_DIM = 36  # G1 36D action layout: root7 + legs12 + waist3 + arms14
ARM_SLICE = slice(22, 36)


def _iter_data_parquets(root: Path) -> Iterator[Path]:
    """snapshot 直下の data/chunk-*/file-*.parquet を順に返す。"""
    data_root = root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"no data/ dir in snapshot: {data_root}")
    for chunk_dir in sorted(data_root.glob("chunk-*")):
        for parquet in sorted(chunk_dir.glob("file-*.parquet")):
            yield parquet


def _arm_matrix(column: object) -> np.ndarray:
    """list<double>[36] の Arrow 列を (n, 14) の arm 行列にする。

    ListArray / FixedSizeListArray のどちらでも、値を平らにして reshape する
    (batch ごとに呼ぶので、ここで確保されるのは batch 1 個分だけ)。
    """
    values = column.flatten()  # type: ignore[attr-defined]
    flat = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)
    if flat.size % ACTION_DIM != 0:
        raise ValueError(
            f"{ACTION_COLUMN} length {flat.size} is not a multiple of {ACTION_DIM}"
        )
    return flat.reshape(-1, ACTION_DIM)[:, ARM_SLICE]


def accumulate_min_max(
    parquets: Iterable[Path],
    *,
    task_index: Optional[int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """全 parquet を batch で舐めて、関節ごとの min / max と frame 数を返す。"""
    lo = np.full(NUM_ARM_JOINTS, np.inf)
    hi = np.full(NUM_ARM_JOINTS, -np.inf)
    total = 0
    columns = [ACTION_COLUMN] if task_index is None else [ACTION_COLUMN, "task_index"]
    for parquet in parquets:
        file_frames = 0
        for batch in pq.ParquetFile(parquet).iter_batches(
            batch_size=batch_size, columns=columns
        ):
            arm = _arm_matrix(batch.column(ACTION_COLUMN))
            if task_index is not None:
                mask = np.asarray(batch.column("task_index")) == task_index
                arm = arm[mask]
            if arm.size == 0:
                continue
            lo = np.minimum(lo, arm.min(axis=0))
            hi = np.maximum(hi, arm.max(axis=0))
            file_frames += arm.shape[0]
        total += file_frames
        print(f"[scan] {parquet.name}: {file_frames} frames", file=sys.stderr)
    if total == 0:
        raise RuntimeError(
            "no frames matched"
            + (f" task_index={task_index}" if task_index is not None else "")
        )
    return lo, hi, total


def emit_yaml(
    *,
    skill_name: str,
    dataset_repo_id: str,
    dataset_revision: str,
    task_index: Optional[int],
    frame_count: int,
    margin_rad: float,
    lo: np.ndarray,
    hi: np.ndarray,
) -> str:
    """skill_config.yaml の `skills.<name>` 配下に貼る YAML 片を組む。"""
    lines = [
        f"  # ---- paste into skills.{skill_name} ----",
        "    # 教師データで各腕関節が取った範囲 (Issue #141 束 1-4 / D5-1)。範囲を出た関節だけ",
        "    # 端から margin_rad 内側を目標にして戻す (戻す速さは MotionLimiter 任せ)。",
        "    teacher_joint_range:",
        f"      margin_rad: {margin_rad}",
        "      source:",
        f'        dataset_repo_id: "{dataset_repo_id}"',
        f'        dataset_revision: "{dataset_revision}"',
    ]
    if task_index is not None:
        lines.append(f"        task_index: {task_index}")
    lines += [
        f"        frame_count: {frame_count}",
        f'        statistic: "min_max_{ACTION_COLUMN.replace(".", "_")}_arms14"',
        "      arm_position_rad:",
    ]
    width = max(len(name) for name in JOINT_NAMES)
    for name, low, high in zip(JOINT_NAMES, lo, hi):
        lines.append(f"        {name + ':':<{width + 1}} [{low:+.10f}, {high:+.10f}]")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="HF dataset repo id")
    ap.add_argument("--revision", default=None, help="commit SHA (省略時は main の解決結果)")
    ap.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="この task_index の frame だけ使う (skill 専用 dataset なら省略)",
    )
    ap.add_argument("--skill-name", required=True, help="skill_config.yaml の key")
    ap.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="snapshot の置き場所 (省略時は HF の cache。--skip-download では必須)",
    )
    ap.add_argument("--output", type=Path, default=None, help="YAML 片の出力先")
    ap.add_argument("--margin-rad", type=float, default=0.02)
    ap.add_argument("--batch-size", type=int, default=8192, help="parquet の読み単位 (行)")
    ap.add_argument(
        "--show-size",
        action="store_true",
        help="data parquet の合計容量を出して終了する (download しない)",
    )
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    revision = args.revision
    if revision is None:
        revision = api.dataset_info(args.dataset).sha

    if args.show_size:
        files = api.list_repo_tree(
            args.dataset, repo_type="dataset", revision=revision,
            path_in_repo="data", recursive=True,
        )
        total = 0
        count = 0
        for entry in files:
            size = getattr(entry, "size", None)
            if size is None or not entry.path.endswith(".parquet"):
                continue
            total += size
            count += 1
        print(f"{args.dataset}@{revision[:7]}: data parquet {count} files, "
              f"{total / 1e6:.1f} MB")
        return

    if args.skip_download:
        if args.local_dir is None:
            ap.error("--skip-download requires --local-dir")
        snapshot_root = args.local_dir
    else:
        from huggingface_hub import snapshot_download

        print(f"[dl] fetching {args.dataset}@{revision[:7]} data parquets", file=sys.stderr)
        kwargs = {} if args.local_dir is None else {"local_dir": str(args.local_dir)}
        snapshot_root = Path(
            snapshot_download(
                args.dataset,
                repo_type="dataset",
                revision=revision,
                allow_patterns=["data/**/*.parquet"],
                **kwargs,
            )
        )

    lo, hi, frame_count = accumulate_min_max(
        _iter_data_parquets(snapshot_root),
        task_index=args.task_index,
        batch_size=args.batch_size,
    )
    fragment = emit_yaml(
        skill_name=args.skill_name,
        dataset_repo_id=args.dataset,
        dataset_revision=revision,
        task_index=args.task_index,
        frame_count=frame_count,
        margin_rad=args.margin_rad,
        lo=lo,
        hi=hi,
    )
    print(f"[range] frames={frame_count} width min={np.min(hi - lo):.3f} "
          f"max={np.max(hi - lo):.3f} rad", file=sys.stderr)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[mem] peak RSS {peak_mb:.0f} MB", file=sys.stderr)
    if args.output is not None:
        args.output.write_text(fragment, encoding="utf-8")
        print(f"[out] {args.output}", file=sys.stderr)
    print(fragment)


if __name__ == "__main__":
    main()
