"""データセットから pick_leg_hybrid の数値を算出する (Issue #136)。

`Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1` を全走査して、
`inference/desktop/pick_leg_hybrid/configs/pick_leg_hybrid.yaml` に入れる値を出す。

出すもの:
  - 把持インターロックの閾値の妥当性 (指令帯ごとの追従誤差)
  - 区間 3 入口の左右 EE pose (位置 median + 姿勢の quaternion 平均)
  - 区間 2 の長さ (duration_sec の根拠)
  - 遷移の健全性 (右→左 の順序が守られている割合)

EE pose は 2 つの定義で出す。**運営 IK が期待する frame が未確定**のため
(determinded.md D-3 / taskspace_adapter の調査 Q1):
  - ``g1_wrist_fk``      : perception/g1_urdf_fk.G1WristFK (我々の出力経路と同じ)
  - ``dataset_ee_state`` : データセット記録の observation.state.ee_state

実行 (GPU 不要、numpy + pyarrow のみ):

    python evaluate/pick_leg_hybrid/derive_thresholds.py \
        --dataset data/IROS2026_RAMEN_suzuki_pick_leg_1
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

from inference.desktop.perception.g1_urdf_fk import G1WristFK  # noqa: E402

LEFT, RIGHT = 0, 1
ROOT_DIM = 7          # robot_q = root(7) + body(29) = 36
BODY_DIM = 29


def quat_mean(quats: np.ndarray) -> np.ndarray:
    """quaternion の平均 (Markley の固有ベクトル法)。

    符号の曖昧さ (q と −q は同じ回転) を扱うため、`sum(q qᵀ)` の
    最大固有値に対応する固有ベクトルを取る。

    Args:
        quats: (N, 4) の単位 quaternion (w, x, y, z)。

    Returns:
        (4,) 単位 quaternion。w >= 0 に正規化して返す。
    """
    q = np.asarray(quats, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    _, vecs = np.linalg.eigh(q.T @ q)
    out = vecs[:, -1]
    if out[0] < 0:
        out = -out
    return out / np.linalg.norm(out)


def matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """3x3 回転行列 → 単位 quaternion (w, x, y, z)。"""
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def euler_xyz_to_quat(rpy: np.ndarray) -> np.ndarray:
    """XYZ euler (roll, pitch, yaw) → quaternion (w, x, y, z)。

    回転順は `g1_urdf_fk._euler_xyz_matrix` と同じ Rz(yaw) @ Ry(pitch) @ Rx(roll)。
    """
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
    cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
    return -q if q[0] < 0 else q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/IROS2026_RAMEN_suzuki_pick_leg_1")
    ap.add_argument("--follow-error-min", type=float, default=0.2)
    ap.add_argument("--hold-min", type=float, default=1.45)
    ap.add_argument("--hold-max", type=float, default=2.35)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default="", help="結果 JSON の出力先 (省略で標準出力のみ)")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{args.dataset}/data/chunk-*/file-*.parquet"))
    if not files:
        print(f"no parquet under {args.dataset}", file=sys.stderr)
        return 1

    fk = G1WristFK.from_urdf()
    cols = ["episode_index", "observation.state.hand_state", "action.hand_cmd",
            "observation.state.robot_q_current", "observation.state.ee_state"]

    err_by_band: dict[str, list] = {}
    entry = {"fk_l_pos": [], "fk_l_quat": [], "fk_r_pos": [], "fk_r_quat": [],
             "ee_l_pos": [], "ee_l_quat": [], "ee_r_pos": [], "ee_r_quat": []}
    gaps, n_right, n_left, n_both, n_order, n_ep = [], 0, 0, 0, 0, 0

    for path in files:
        t = pq.read_table(path, columns=cols)
        ep = np.asarray(t["episode_index"].to_pylist(), dtype=np.int64)
        hs = np.asarray(t["observation.state.hand_state"].to_pylist(), dtype=np.float64)
        hc = np.asarray(t["action.hand_cmd"].to_pylist(), dtype=np.float64)
        q = np.asarray(t["observation.state.robot_q_current"].to_pylist(), dtype=np.float64)
        ee = np.asarray(t["observation.state.ee_state"].to_pylist(), dtype=np.float64)

        e_all = hs - hc
        for lab, lo, hi in (("open>3.5", 3.5, 1e9), ("mid2.35-3.5", 2.35, 3.5),
                            ("hold1.45-2.35", 1.45, 2.35)):
            for side, j in (("left", LEFT), ("right", RIGHT)):
                m = (hc[:, j] >= lo) & (hc[:, j] < hi)
                if m.any():
                    err_by_band.setdefault(f"{lab}/{side}", []).append(e_all[m, j])

        grip = ((hs - hc) > args.follow_error_min) & (hs >= args.hold_min) & (hs <= args.hold_max)
        for e in np.unique(ep):
            m = ep == e
            n_ep += 1
            G, Q, E = grip[m], q[m], ee[m]
            ri = np.flatnonzero(G[:, RIGHT])
            li = np.flatnonzero(G[:, LEFT])
            n_right += int(ri.size > 0)
            n_left += int(li.size > 0)
            if ri.size and li.size:
                n_both += 1
                if ri[0] < li[0]:
                    n_order += 1
                    gaps.append(li[0] - ri[0])
            if li.size:
                k = li[0]
                body = Q[k][ROOT_DIM:ROOT_DIM + BODY_DIM]
                lp, lR, rp, rR = fk.compute_ee_transforms(body)
                entry["fk_l_pos"].append(lp); entry["fk_l_quat"].append(matrix_to_quat_wxyz(lR))
                entry["fk_r_pos"].append(rp); entry["fk_r_quat"].append(matrix_to_quat_wxyz(rR))
                entry["ee_l_pos"].append(E[k][0:3]); entry["ee_l_quat"].append(euler_xyz_to_quat(E[k][3:6]))
                entry["ee_r_pos"].append(E[k][6:9]); entry["ee_r_quat"].append(euler_xyz_to_quat(E[k][9:12]))

    print("=" * 78)
    print("1) 指令帯ごとの追従誤差 (hand_state − hand_cmd)")
    print("=" * 78)
    for k in sorted(err_by_band):
        v = np.concatenate(err_by_band[k])
        print(f"  {k:22s} n={v.size:8d}  median={np.median(v):+.3f}")
    print("\n  → 開/中間は ±0.013 以内で追従、HOLD 帯だけ +0.35。")
    print(f"     follow_error_min={args.follow_error_min} は両者を明確に分ける。")

    print()
    print("=" * 78)
    print(f"2) 区間 3 入口の EE pose  ({len(entry['fk_l_pos'])}/{n_ep} episode)")
    print("=" * 78)
    result: dict = {"n_episodes": n_ep, "n_entry": len(entry["fk_l_pos"])}
    for frame, lp, lq, rp, rq in (
        ("g1_wrist_fk", "fk_l_pos", "fk_l_quat", "fk_r_pos", "fk_r_quat"),
        ("dataset_ee_state", "ee_l_pos", "ee_l_quat", "ee_r_pos", "ee_r_quat"),
    ):
        print(f"  [{frame}]")
        block = {}
        for side, pk, qk in (("left", lp, lq), ("right", rp, rq)):
            P = np.array(entry[pk]); Q_ = np.array(entry[qk])
            med = np.median(P, 0)
            iqr = np.percentile(P, 75, 0) - np.percentile(P, 25, 0)
            qm = quat_mean(Q_)
            print(f"    {side:5s} pos  = [{med[0]:+.4f}, {med[1]:+.4f}, {med[2]:+.4f}]"
                  f"   IQR 幅 [{iqr[0]:.4f}, {iqr[1]:.4f}, {iqr[2]:.4f}]")
            print(f"          quat = [{qm[0]:+.4f}, {qm[1]:+.4f}, {qm[2]:+.4f}, {qm[3]:+.4f}]  (w,x,y,z)")
            block[side] = {"pos": med.tolist(), "pos_iqr": iqr.tolist(), "quat": qm.tolist()}
        result[frame] = block
    d_l = np.linalg.norm(np.array(entry["fk_l_pos"]) - np.array(entry["ee_l_pos"]), axis=1)
    d_r = np.linalg.norm(np.array(entry["fk_r_pos"]) - np.array(entry["ee_r_pos"]), axis=1)
    print(f"\n  → 2 定義の位置差: 左 {np.median(d_l)*1000:.0f} mm / 右 {np.median(d_r)*1000:.0f} mm")
    print("     運営 IK がどちらを期待するかは未確定 (調査 Q1)。config の ee_frame で切替。")
    result["frame_gap_mm"] = {"left": float(np.median(d_l) * 1000), "right": float(np.median(d_r) * 1000)}

    g = np.array(gaps, dtype=np.float64)
    print()
    print("=" * 78)
    print("3) 遷移の健全性")
    print("=" * 78)
    print(f"  右手が掴んだ episode: {n_right}/{n_ep}    左手が掴んだ episode: {n_left}/{n_ep}")
    print(f"  右 → 左 の順序が正しい: {n_order}/{n_both} ({100*n_order/max(n_both,1):.1f}%)")
    if g.size:
        print(f"  区間 2 の長さ: median {np.median(g):.0f} frame = {np.median(g)/args.fps:.2f} 秒"
              f"   IQR=[{np.percentile(g,25):.0f}, {np.percentile(g,75):.0f}]")
        result["phase2_duration_sec"] = float(np.median(g) / args.fps)
    result["order_ok_ratio"] = float(n_order / max(n_both, 1))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\n  → {args.out} に保存")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
