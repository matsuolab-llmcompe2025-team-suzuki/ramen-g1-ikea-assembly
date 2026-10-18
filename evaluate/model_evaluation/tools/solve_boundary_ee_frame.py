"""運営 IK が期待する EE frame を、実機のログから逆算する。

⚠️ **この問いは 2026-09-22 (Issue #154) に決着した。通常このスクリプトは不要。**

```
CONTRACT.md:60-64   "End-effector frame: left_wrist_yaw_link / right_wrist_yaw_link,
                     with **zero tool offset**"
ik.py WRIST_FRAMES              = {"left": "left_wrist_yaw_link", ...}
IKSettings.tool_offset_xyz      = (0,0,0)   "ZERO by default and deliberately so"
PinkArmIK.solve                 target は **pelvis 基準**として解釈される
```

実測でも裏を取った。pod の 3012 行を運営 model で FK し、記録済み `taskspace` と
突き合わせて **位置ずれ median 4.85e-07 m**、quaternion は **w-first 確定**
(`evaluate/model_evaluation/tools/measure_organizer_ik_accept.py`)。
我々の `compute_ee_transforms` は既にこの規約どおり。

以下の「どの一次資料にも書かれていない」は **2026-09-15 時点の記述で、
2026-09-21 配布の運営 package が明文化したことで覆った。** 経緯として残す。

本スクリプトの用途は、**将来 package が規約を変えた / 実機で系統偏りが再発した**
ときの逆算診断に限る。

# なぜ書かれたか (当時の状況)

`decoupled` lane は `(T,25)` で **EE 姿勢**を送り、運営側が IK を解く。その EE を
**どの原点基準で解釈するか** (pelvis / torso / root-link、wrist tool offset の有無) が
**どの一次資料にも書かれていなかった** — `boundary/` / submit README / INSTRUCTIONS /
WBC_RUNBOOK / ADMINISTRATIVE_MANIFEST / Onsite Details を全文検索して 0 件、
運営連絡 (Slack 全 72 件) にも無かった。2026-09-15 に質問済みで、当時は未回答。

frame がずれていると **エラーは出ず、腕が違う場所へ行く**。運営の 09-14 dry-run では
pick の IK 残差が許容の約 60 倍、しかも "steady bias, not scattered noise" だった
= 固定オフセットの signature。

# 考え方

我々は自分の root-link frame で `P_cmd` を送る。運営 IK がそれを別 frame `X` で
解釈して関節を解き、ロボットが動く。その関節を**自前 FK**に通すと `P_ach` が出る。

    X ∘ FK_ours(joints) = P_cmd   →   P_ach = FK_ours(joints) = X⁻¹ ∘ P_cmd
    ∴ X = P_cmd ∘ P_ach⁻¹

`taskspace_adapter._apply_frame` は送信前に姿勢へ transform を掛けるので、
求めた `X` をそのまま `ee_frame_transform` に渡せばよい。

# 使い方

`--action-sink boundary` で 1 回走らせると、`--log` の JSONL に
`event=boundary_taskspace` が 1 tick 1 行で残る (指令 `(T,25)` と実測関節角の両方)。
**静止保持している区間**を選んで:

    python evaluate/model_evaluation/tools/solve_boundary_ee_frame.py \\
        --log /out/orch_run.jsonl --skip-first 0.5

本番の準備 20 分枠で回せる想定。ただし **`~/wbc_adapter/ik.py` を読める場合は
そちらが確実** (実装そのもの)。本 script はそれが読めなかったときの保険。

→ **2026-09-21 配布の運営 package にその `ik.py` が同梱された**ので、この保険は
通常は使わない。冒頭の警告を参照。

# 限界

- 静止していない区間を混ぜると IK の追従遅れが混入する。`--skip-first` で立ち上がりを捨て、
  残差の散らばり (出力の `rot_std_deg` / `pos_std_m`) を見て判断すること
- 左右で別々に解く。**両者が大きく食い違うなら frame ではなく別の原因**を疑う
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np

# (T,25) の列レイアウト (boundary/actions.py と一致、並べ替え禁止)
LEFT_POS = slice(4, 7)
LEFT_QUAT = slice(7, 11)  # w-first
RIGHT_POS = slice(11, 14)
RIGHT_QUAT = slice(14, 18)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """単位 quaternion (w, x, y, z) → 3x3 回転行列。"""
    w, x, y, z = (float(v) for v in quat)
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    if n == 0.0:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def se3(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


def read_records(log_path: Path) -> Iterator[dict]:
    """`event=boundary_taskspace` の行だけを順に返す。"""
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # orchestrator の他の行が混ざっていてもよい
            if record.get("event") == "boundary_taskspace":
                yield record


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """平均した回転行列を SVD で直交化する (平均は回転群を外れるため)。"""
    u, _, vt = np.linalg.svd(matrix)
    rot = u @ vt
    if np.linalg.det(rot) < 0:  # 反射になったら最小特異値の軸を反転
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def solve_side(
    commanded: list[np.ndarray], achieved: list[np.ndarray]
) -> dict[str, object]:
    """片腕ぶんの `X = P_cmd ∘ P_ach⁻¹` を平均で求める。"""
    if not commanded:
        raise ValueError("no samples")
    transforms = [c @ np.linalg.inv(a) for c, a in zip(commanded, achieved)]
    positions = np.stack([t[:3, 3] for t in transforms])
    rotations = np.stack([t[:3, :3] for t in transforms])

    mean_rot = orthonormalize(rotations.mean(axis=0))
    mean_pos = positions.mean(axis=0)

    # 散らばり = 静止できていたかの目安。大きいなら区間選択が悪い。
    angles = []
    for rot in rotations:
        delta = mean_rot.T @ rot
        cos = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cos))))

    return {
        "samples": len(transforms),
        "transform": se3(mean_rot, mean_pos),
        "pos_m": mean_pos.tolist(),
        "pos_std_m": positions.std(axis=0).tolist(),
        "pos_norm_m": float(np.linalg.norm(mean_pos)),
        "rot_std_deg": float(np.std(angles)),
        "rot_max_dev_deg": float(np.max(angles)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="orch_run.jsonl")
    parser.add_argument(
        "--fk-urdf", type=Path, default=None, help="G1 URDF (未指定なら既定)"
    )
    parser.add_argument(
        "--skip-first",
        type=float,
        default=0.3,
        help="先頭の何割を捨てるか (立ち上がりの追従遅れを除く)。既定 0.3",
    )
    args = parser.parse_args()

    from inference.desktop.perception.g1_urdf_fk import G1WristFK

    # `G1WristFK.__init__` は parse 済みの chain を取る。URDF から組むのは
    # `from_urdf`。直接 __init__ を呼ぶと TypeError になる。
    fk = (
        G1WristFK.from_urdf(str(args.fk_urdf))
        if args.fk_urdf is not None
        else G1WristFK.from_urdf()
    )

    records = list(read_records(args.log))
    if not records:
        raise SystemExit(
            f"{args.log} に event=boundary_taskspace の行が無い "
            "(--action-sink boundary で走らせたログか確認すること)"
        )
    start = int(len(records) * args.skip_first)
    records = records[start:]
    print(f"[solve] {len(records)} samples (先頭 {start} 件を破棄)")

    sides: dict[str, tuple[list, list]] = {"left": ([], []), "right": ([], [])}
    for record in records:
        row = np.asarray(record["taskspace_25"], dtype=np.float64)
        body_q29 = np.asarray(record["measured_body_q29"], dtype=np.float64)
        left_pos, left_rot, right_pos, right_rot = fk.compute_ee_transforms(body_q29)

        sides["left"][0].append(se3(quat_wxyz_to_matrix(row[LEFT_QUAT]), row[LEFT_POS]))
        sides["left"][1].append(se3(np.asarray(left_rot), np.asarray(left_pos)))
        sides["right"][0].append(
            se3(quat_wxyz_to_matrix(row[RIGHT_QUAT]), row[RIGHT_POS])
        )
        sides["right"][1].append(se3(np.asarray(right_rot), np.asarray(right_pos)))

    for name, (commanded, achieved) in sides.items():
        result = solve_side(commanded, achieved)
        print(f"\n=== {name} ===")
        print(f"  samples      : {result['samples']}")
        print(f"  offset [m]   : {np.round(result['pos_m'], 4).tolist()}")
        print(f"  |offset| [m] : {result['pos_norm_m']:.4f}")
        print(f"  pos std [m]  : {np.round(result['pos_std_m'], 4).tolist()}")
        print(f"  rot std [deg]: {result['rot_std_deg']:.2f}")
        print(f"  rot max dev  : {result['rot_max_dev_deg']:.2f} deg")
        print("  ee_frame_transform (4x4):")
        for row_ in np.asarray(result["transform"]):
            print("    " + "  ".join(f"{v: .6f}" for v in row_))

    print(
        "\n判断の目安: pos std が cm order を超える / rot std が数 deg を超えるなら、"
        "\n静止していない区間が混ざっている。--skip-first を上げるか保持を長く取ること。"
        "\n左右の offset が大きく食い違うなら frame 以外の原因を疑う。"
    )


if __name__ == "__main__":
    main()
