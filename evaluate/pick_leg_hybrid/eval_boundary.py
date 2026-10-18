"""区間 1→2 の境界検出をデータセットでオフライン評価する (Issue #136、planning.md P-11)。

実機もロボットも要らない。`IROS2026_RAMEN_suzuki_pick_leg_1` のホールドアウト
episode を replay し、境界検出が**何フレーム遅れる / 早まる**かを測る。

# 正解ラベルはタダで手に入る

`hand_state` の遷移から自動生成できる:

    掴んだ = (hand_state − hand_cmd > follow_error_min)
             AND (hold_min ≤ hand_state ≤ hold_max)

右手が最初にこれを満たした frame が 1→2 の正解。実測では 2114 episode 中
2112 で検出できる (`derive_thresholds.py`)。

# 何を比べるか

`--mode` で切り替える:

    interlock  インターロックだけ (VLM を呼ばない)。**基準線**
    overlay    インターロック AND VLM、画像に OBB overlay を描いて渡す
    raw        インターロック AND VLM、overlay なしの生画像

`overlay` と `raw` の差が、D-7 の overlay が効いているかどうか。
repo の GR00T overlay は overlay 済み画像で fine-tune したモデルだが、
ここでは overlay を見たことのない Qwen3-VL にゼロショットで渡すため、
効くとは限らない。効かなければ overlay を捨てる判断材料になる。

# 実行できる環境

`interlock` モードは numpy + pyarrow だけで走る。
**`overlay` / `raw` は VLM endpoint が要る** (Thor 上の vLLM 等)。
映像 frame の decode には opencv が要るので runtime env で実行すること。

    # 基準線 (VLM 不要、どこでも走る)
    python evaluate/pick_leg_hybrid/eval_boundary.py --mode interlock --limit 200

    # VLM 込み (Thor など、endpoint がある環境)
    pixi run -e runtime python evaluate/pick_leg_hybrid/eval_boundary.py \
        --mode overlay --limit 200 --endpoint http://127.0.0.1:8000/v1/chat/completions
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.desktop.pick_leg_hybrid.boundary import (  # noqa: E402
    BoundaryConfig,
    GraspBoundaryDetector,
)
from inference.desktop.pick_leg_hybrid.interlock import (  # noqa: E402
    RIGHT,
    GraspInterlockConfig,
    is_grasping,
)
from inference.desktop.pick_leg_hybrid.vlm import VlmConfig  # noqa: E402

MODES = ("interlock", "overlay", "raw")


def ground_truth_frame(
    hand_state: np.ndarray, hand_cmd: np.ndarray, cfg: GraspInterlockConfig
) -> int:
    """右手が最初に「掴んだ」を満たす frame index。無ければ -1。"""
    grip = (
        ((hand_state[:, RIGHT] - hand_cmd[:, RIGHT]) > cfg.follow_error_min)
        & (hand_state[:, RIGHT] >= cfg.hold_min)
        & (hand_state[:, RIGHT] <= cfg.hold_max)
    )
    idx = np.flatnonzero(grip)
    return int(idx[0]) if idx.size else -1


def run_episode(
    hand_state: np.ndarray,
    hand_cmd: np.ndarray,
    detector: GraspBoundaryDetector,
    fps: float,
    images_for_frame=None,
) -> int:
    """1 episode を replay して、検出器が発火した frame を返す。無ければ -1。

    Args:
        hand_state / hand_cmd: (N, 2)。
        detector: 評価する検出器。
        fps: 時刻を作るための frame rate。
        images_for_frame: `f -> list[str]` の callable、または None
            (None なら画像を渡さない = interlock モード)。
    """
    detector.reset()
    for f in range(len(hand_state)):
        images = images_for_frame(f) if images_for_frame is not None else ()
        out = detector.update(
            t=f / fps,
            hand_state=hand_state[f],
            hand_cmd=hand_cmd[f],
            images_b64=images,
        )
        if out.fire:
            return f
    return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/IROS2026_RAMEN_suzuki_pick_leg_1")
    ap.add_argument("--mode", choices=MODES, default="interlock")
    ap.add_argument("--limit", type=int, default=200, help="評価する episode 数")
    ap.add_argument("--confirm-count", type=int, default=2)
    ap.add_argument("--min-interval-sec", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--endpoint", default=VlmConfig().endpoint)
    ap.add_argument("--model", default=VlmConfig().model)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    if args.mode != "interlock":
        print(
            f"[warn] mode={args.mode} は VLM endpoint ({args.endpoint}) と "
            "映像 frame の decode (opencv) が要ります。"
            "本 script の frame 供給は未実装です — todo.md「4. 未実装」参照。",
            file=sys.stderr,
        )
        return 2

    icfg = GraspInterlockConfig()
    bcfg = BoundaryConfig(
        confirm_count=args.confirm_count,
        min_interval_sec=args.min_interval_sec,
        require_interlock=True,
    )

    class _AlwaysYes:
        """interlock モード用。VLM を「常に成立」に固定して純粋な基準線を測る。"""

        def ask(self, question, images, *, current_phase_text=""):
            from inference.desktop.pick_leg_hybrid.vlm import VlmAnswer

            return VlmAnswer(value=1)

    files = sorted(glob.glob(f"{args.dataset}/data/chunk-*/file-*.parquet"))
    if not files:
        print(f"no parquet under {args.dataset}", file=sys.stderr)
        return 1

    cols = ["episode_index", "observation.state.hand_state", "action.hand_cmd"]
    deltas: list[int] = []
    misses = 0
    n = 0

    for path in files:
        if n >= args.limit:
            break
        t = pq.read_table(path, columns=cols)
        ep = np.asarray(t["episode_index"].to_pylist(), dtype=np.int64)
        hs = np.asarray(t["observation.state.hand_state"].to_pylist(), dtype=np.float64)
        hc = np.asarray(t["action.hand_cmd"].to_pylist(), dtype=np.float64)
        for e in np.unique(ep):
            if n >= args.limit:
                break
            m = ep == e
            HS, HC = hs[m], hc[m]
            truth = ground_truth_frame(HS, HC, icfg)
            if truth < 0:
                continue
            n += 1
            det = GraspBoundaryDetector(
                cfg=bcfg, interlock_cfg=icfg, vlm_client=_AlwaysYes()
            )
            fired = run_episode(HS, HC, det, args.fps)
            if fired < 0:
                misses += 1
            else:
                deltas.append(fired - truth)

    d = np.array(deltas, dtype=np.float64)
    print("=" * 74)
    print(f"1→2 境界検出   mode={args.mode}   episode={n}   "
          f"confirm_count={args.confirm_count}   min_interval={args.min_interval_sec}s")
    print("=" * 74)
    print(f"  検出できなかった episode: {misses} / {n}")
    if d.size:
        print(f"  正解からのずれ [frame] (正 = 遅れ、負 = 早まり):")
        print(f"    median {np.median(d):+.1f}   mean {d.mean():+.2f}   "
              f"IQR [{np.percentile(d,25):+.0f}, {np.percentile(d,75):+.0f}]")
        print(f"    早まった episode (< 0): {int((d < 0).sum())} / {d.size}")
        print(f"    遅れ [秒]: median {np.median(d)/args.fps:+.2f}")
        print()
        print("  ※ 早まり = 偽陽性で回復不能。遅れ = ただの遅延 (安全側)。")

    result = {
        "mode": args.mode, "n_episodes": n, "misses": misses,
        "confirm_count": args.confirm_count,
        "delta_median_frames": float(np.median(d)) if d.size else None,
        "early_count": int((d < 0).sum()) if d.size else 0,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\n  → {args.out} に保存")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
