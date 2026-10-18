"""運営の `wbc_adapter/ik.py` を、記録済みの `(T,25)` 行にそのまま当てて accept 率を測る。

運営コードは **import して呼ぶだけ**で、一切改変しない。

---------------------------------------------------------------------------
## ✅ 決着済み — 以下は再検証不要 (2026-09-22, Issue #154)
---------------------------------------------------------------------------

pod Phase B の 3012 行 (`outputs/venue_capture/phaseB_rows.jsonl`、**会場 bridge の
実画像 30 frame + 実 body_q + 実 gripper_q** から生成) に対して測定済み。

```
FK 照合          位置ずれ median 4.85e-07 m / max 9.65e-07 m
                 → 我々の URDF と運営の腕 chain は等価
                 → EE は pelvis 基準 / wrist_yaw_link 原点 / tool offset ゼロ

quaternion 順序  w-first 7.07e-07  vs  w-last 1.319   (6 桁差)
                 → **w-first で確定**
                 CONTRACT.md:55-58 が "the single most common silent failure"
                 と書いている項目。実データで否定済み

accept 率        3012/3012 = 100.0% (左右とも、4 skill すべて)
                 solve 0.9 ms/row
```

**この 3 つのために再実行する必要はない。** 結論は
当時の `docs/handoff/venue_runbook.md`（Git 履歴）に記録した。

---------------------------------------------------------------------------
## 🔴 まだ測れていない一点 — このスクリプトが残っている唯一の理由
---------------------------------------------------------------------------

上の 100% は **静止シーン**の行に対する数字である。ロボットは動いておらず、固定
観測に対する policy 出力を見ているだけなので、**把持や回転の最中に出る端値に到達
していない**。

運営自身が同じ落とし穴を記録している (`reference/wbc_adapter/README.md:176-186`):

```
mock 画像                 L=100%, R=67-100%
実カメラ + 実際に動くロボット   1-3%
"Treat this as 'the plumbing and IK work',
 not as evidence about the policy."
```

**accept されなかった waypoint は `wbc_driver` が直前の腕 target を保持する**
(`ik.py` の `UpperBodyIK.solve_row`、"Hold the last known-good target per arm")。1-3% は腕がほぼ凍結することを意味する。

→ **#152 の実機通し run が「動いているロボット」の行を出す。それをこのスクリプトに
通すのが、我々が本番の accept 率を知る唯一の手段。** 実機も Thor も会場機材も要らない。

そのとき見るべき運営側の制約 (`ik.py` の既定値、solver 側 override):

```
max_err 1e-3            1mm 超えたら reject → 保持
elbow  upper 1.4        URDF の 2.0944 ではない
wrist_roll ±0.9         URDF の ±1.9722 ではない
```

静止 run での実指令は elbow 最大 1.221 / wrist_roll 最大 0.336 で、どちらも 0% 超過。

---------------------------------------------------------------------------
## 実行環境
---------------------------------------------------------------------------

`pinocchio` / `pink` / `qpsolvers` / `quadprog` が要る。**root の pixi env には
入れないこと** (default env は numpy+pyyaml+pytest の最小構成、`runtime` は SDK 用)。
解析専用の venv を切る:

```bash
python3 -m venv /tmp/ikvenv
/tmp/ikvenv/bin/pip install pin==4.1.0 pin-pink==4.4.0 qpsolvers==4.13.0 quadprog==0.1.13
```

## 使い方

```bash
/tmp/ikvenv/bin/python evaluate/model_evaluation/tools/measure_organizer_ik_accept.py \
    --package ~/iros_g1_orin_package \
    --rows outputs/venue_capture/phaseB_rows.jsonl
```

`--rows` の JSONL は 1 行 1 tick で、最低限 `taskspace` (25) と `step19` (19) を持つこと
(`orchestrator_driver` / `entrypoint` の記録形式)。

## 限界

- seed (`body_q29`) は **直前 tick の指令**であって実測ではない。実機では追従誤差と
  運営側 clamp (`--max-joint-vel 1.0` → 1 waypoint あたり 0.05 rad) の分だけ遅れる。
  実測 `body_q` が記録に含まれるなら `--body-q-key` で指定したほうが正確
- 脚は 0 (立位) 固定。pelvis が root なので pelvis 基準の EE には影響しない
  (FK 照合がサブミクロンで一致していることが根拠)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# step19   = [waist(0:3), left_arm(3:10), right_arm(10:17), grip(17:19)]
# body_q29 = [legs(0:12), waist(12:15), left_arm(15:22), right_arm(22:29)]
_STEP19_WAIST = slice(0, 3)
_STEP19_LEFT = slice(3, 10)
_STEP19_RIGHT = slice(10, 17)


def load_organizer_ik(package: Path):
    """運営 package の `ik.py` をそのまま import する (改変しない)。"""
    path = package / "reference/wbc_adapter/ik.py"
    if not path.is_file():
        raise SystemExit(f"運営の ik.py が無い: {path}")
    spec = importlib.util.spec_from_file_location("organizer_ik", path)
    mod = importlib.util.module_from_spec(spec)
    # dataclass が cls.__module__ を sys.modules から引くので、exec 前に登録が要る
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def body_q29_from_step19(step19) -> np.ndarray:
    q = np.zeros(29, dtype=np.float64)
    q[12:15] = step19[_STEP19_WAIST]
    q[15:22] = step19[_STEP19_LEFT]
    q[22:29] = step19[_STEP19_RIGHT]
    return q


def fk_check(ik, solver, rows, n: int) -> None:
    """運営 model で step19 を FK し、記録済み taskspace の EE と突き合わせる。

    一致すれば URDF の等価性・EE 規約・quaternion 順序が同時に確定する。
    """
    import pinocchio as pin
    from scipy.spatial.transform import Rotation

    arm = solver.left  # PinkArmIK から model / q_index を借りる
    model, data = arm.model, arm.model.createData()
    fid = {s: model.getFrameId(ik.WRIST_FRAMES[s]) for s in ("left", "right")}
    pid = model.getFrameId(ik.PELVIS_FRAME)

    # ⚠️ 下の `zip` は長さ不一致を **黙って切り捨てる**。運営 package が
    # `_BODY_Q_NAMES` を改名・要素数変更していると、余った関節が neutral のまま
    # FK され、症状は「位置ずれが少し大きい」としか出ない = 原因に辿り着けない。
    # このツールは **#152 の実機行で別バージョンの package に対して再実行する
    # 前提**なので、ここで止める。
    if len(ik._BODY_Q_NAMES) != 29:
        raise SystemExit(
            f"運営 ik.py の _BODY_Q_NAMES が 29 要素でない "
            f"({len(ik._BODY_Q_NAMES)})。body_q29 の対応付けを見直すこと"
        )

    dp = {"left": [], "right": []}
    dq_w, dq_x = [], []
    for r in rows[:n]:
        q = pin.neutral(model)
        b = body_q29_from_step19(r["step19"])
        for name, i in zip(ik._BODY_Q_NAMES, range(29)):
            if name in arm.q_index:
                q[arm.q_index[name]] = b[i]
        pin.framesForwardKinematics(model, data, q)
        t_pelvis = data.oMf[pid]
        ts = np.asarray(r["taskspace"], dtype=np.float64)
        for side, pcol, qcol in (
            ("left", slice(4, 7), slice(7, 11)),
            ("right", slice(11, 14), slice(14, 18)),
        ):
            t = t_pelvis.inverse() * data.oMf[fid[side]]  # pelvis 基準へ
            dp[side].append(float(np.linalg.norm(t.translation - ts[pcol])))
            quat_xyzw = Rotation.from_matrix(t.rotation).as_quat()
            got = ts[qcol]
            w_first = np.roll(quat_xyzw, 1)
            # quaternion は q と -q が同じ姿勢なので両極性の近いほうを採る
            dq_w.append(
                min(np.linalg.norm(w_first - got), np.linalg.norm(w_first + got))
            )
            dq_x.append(
                min(np.linalg.norm(quat_xyzw - got), np.linalg.norm(quat_xyzw + got))
            )

    print(f"\n=== FK 照合 ({min(n, len(rows))} 行) ===")
    for s in ("left", "right"):
        print(
            f"  {s:<6} 位置ずれ  median {statistics.median(dp[s]):.3e} m  "
            f"max {max(dp[s]):.3e} m"
        )
    med_w, med_x = statistics.median(dq_w), statistics.median(dq_x)
    order = "w-first (wxyz)" if med_w < med_x else "w-last (xyzw)"
    print(f"  quaternion  w-first {med_w:.3e}  /  w-last {med_x:.3e}   -> {order}")


def rows_from_capture_evidence(log_jsonl: Path) -> list[dict]:
    """運営 `tools/capture_evidence.py` の `log.jsonl` を本 script の入力に直す。

    **これが本番の accept 率を測る現実的な手段。** あちらは read-only の SUB を
    3 本張るだけ (`:5555` / `:5557` / 我々が bind した `:5556`) で、
    「safe to run *alongside* a live session since it never publishes」と
    明記されている = **大会経路の走行中にそのまま回せる**。

    あちらの行:

        {"t", "type": "action", "issued_at", "shape", "row0", "rows"}   (T,25) 全行
        {"t", "type": "state",  "body_q", "gripper_q"}                  実測 29-D

    ⚠️ **seed は実測 `body_q` を使う。** 本 script の既定 (直前 tick の `step19`
    から再構成) は**指令**であって実測ではなく、追従誤差と運営側 clamp の分だけ
    ずれる。あちらは実測をそのまま載せているので、そちらが正確。
    呼び出し側は `--body-q-key body_q` を付けること。

    action 1 行につき chunk の全 row を展開する (adapter は row ごとに IK を解く)。
    大会経路は `ACTION_CHUNK=1` なので通常は 1 行 = 1 row。
    """
    out: list[dict] = []
    latest_body_q: list | None = None
    with log_jsonl.open() as fh:
        for raw in fh:  # 36 分の run は数万行になるので流し読みする
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("type")
            if kind == "state":
                bq = rec.get("body_q")
                if bq:  # bridge が旧版だと None が来る
                    latest_body_q = bq
                continue
            if kind != "action":
                continue
            if latest_body_q is None:
                continue  # 最初の state が来る前の action は seed が無い
            chunk = rec.get("rows") or ([rec["row0"]] if "row0" in rec else [])
            for row in chunk:
                out.append(
                    {
                        "taskspace": row,
                        "body_q": latest_body_q,
                        "t": rec.get("t"),
                        "issued_at": rec.get("issued_at"),
                    }
                )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--package", required=True, type=Path, help="運営 interface package の root"
    )
    ap.add_argument("--rows", required=True, type=Path, help="記録済み JSONL")
    ap.add_argument(
        "--urdf",
        type=Path,
        default=Path(
            "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf"
        ),
    )
    ap.add_argument(
        "--assets",
        type=Path,
        default=Path("inference/orin/ros2_ws/src"),
        help="package://g1_description/... の解決元",
    )
    ap.add_argument(
        "--body-q-key", default="", help="実測 body_q(29) を持つ key があれば指定"
    )
    ap.add_argument("--fk-rows", type=int, default=200, help="FK 照合に使う行数")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 行だけ (0=全部)")
    ap.add_argument("--out", type=Path, default=None, help="結果 JSON の出力先")
    args = ap.parse_args()

    ik = load_organizer_ik(args.package.expanduser())
    # 運営 capture_evidence の log.jsonl なら、その場で本 script の形に直す。
    # 判定は先頭 1 行だけ。本 script 本来の形式は `taskspace` / `step19` を持ち
    # `"type"` を持たないので衝突しない。
    with args.rows.open() as fh:
        first = fh.readline()
    if '"type"' in first and ('"action"' in first or '"state"' in first):
        print("[setup] 運営 capture_evidence の log.jsonl と判断して変換する")
        rows = rows_from_capture_evidence(args.rows)
        if not args.body_q_key:
            args.body_q_key = "body_q"  # 実測 seed を使う (再構成より正確)
            print("[setup] --body-q-key body_q を自動で有効化 (実測 seed)")
    else:
        with args.rows.open() as fh:
            rows = [json.loads(line) for line in fh]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"行が無い: {args.rows}")

    settings = ik.IKSettings()  # 運営の既定値そのまま。触らない
    solver = ik.UpperBodyIK(
        settings,
        include_waist=False,
        urdf=args.urdf,
        assets=args.assets,
        warm_start="current",
    )
    print(f"[setup] rows={len(rows)}  urdf={args.urdf.name}")
    print(
        f"[setup] max_err={settings.max_err}  max_iters={settings.max_iters}  "
        f"elbow_cap={settings.elbow_upper_limit_override}  "
        f"wrist_roll_cap={settings.wrist_roll_limit_override}"
    )

    fk_check(ik, solver, rows, args.fk_rows)

    per_skill: dict[str, list] = {}
    l_err, r_err, l_ok, r_ok = [], [], 0, 0
    t0 = time.monotonic()
    prev_step19 = rows[0]["step19"]

    for i, r in enumerate(rows):
        row = np.asarray(r["taskspace"], dtype=np.float64)
        if args.body_q_key and args.body_q_key in r:
            body_q = np.asarray(r[args.body_q_key], dtype=np.float64).reshape(-1)[:29]
        else:
            body_q = body_q29_from_step19(prev_step19)
        res = solver.solve_row(row, body_q)
        l_err.append(res.left_err)
        r_err.append(res.right_err)
        l_ok += int(res.left_ok)
        r_ok += int(res.right_ok)
        per_skill.setdefault(r.get("skill", "?"), []).append(
            (res.left_ok, res.right_ok)
        )
        prev_step19 = r["step19"]
        if (i + 1) % 500 == 0:
            print(
                f"  ... {i + 1}/{len(rows)}  L={100 * l_ok / (i + 1):.1f}%  "
                f"R={100 * r_ok / (i + 1):.1f}%",
                flush=True,
            )

    n = len(rows)
    dt = time.monotonic() - t0
    print(f"\n=== accept 率 (max_err={settings.max_err}) ===")
    print(f"  left  {l_ok}/{n} = {100 * l_ok / n:.1f}%")
    print(f"  right {r_ok}/{n} = {100 * r_ok / n:.1f}%")
    print(f"  solve {dt:.1f}s = {1000 * dt / n:.1f} ms/row")

    print(
        "\n=== 残差 (m) — converged_err == max_err なので 1e-3 を切った時点で打ち切る ==="
    )
    for name, errs in (("left", l_err), ("right", r_err)):
        e = sorted(errs)
        print(
            f"  {name:<6} median {statistics.median(e):.2e}  "
            f"p95 {e[int(0.95 * len(e))]:.2e}  max {e[-1]:.2e}"
        )

    print("\n=== skill 別 ===")
    for skill in sorted(per_skill):
        v = per_skill[skill]
        lo = sum(x[0] for x in v)
        ro = sum(x[1] for x in v)
        print(
            f"  {skill:<24} n={len(v):<5} L={100 * lo / len(v):5.1f}%  R={100 * ro / len(v):5.1f}%"
        )

    if l_ok < n or r_ok < n:
        print(
            "\n🔴 reject がある。運営 `wbc_driver` はその waypoint で "
            "**直前の腕 target を保持**する (ik.py UpperBodyIK.solve_row)。"
        )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "n": n,
                    "left_accept": l_ok / n,
                    "right_accept": r_ok / n,
                    "left_err_median": statistics.median(l_err),
                    "right_err_median": statistics.median(r_err),
                    "settings": {
                        "max_err": settings.max_err,
                        "elbow_cap": settings.elbow_upper_limit_override,
                        "wrist_roll_cap": settings.wrist_roll_limit_override,
                    },
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
