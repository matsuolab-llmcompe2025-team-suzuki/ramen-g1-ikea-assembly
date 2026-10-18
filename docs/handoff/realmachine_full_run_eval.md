# 自前実機での通し評価 handoff

**本番前に、自前の G1 で Stage 0〜5 を一度は通しておく。** 会場でしか確認できない
ものは別。ここは**自前経路
（`entrypoint.py` → SDK 直）だけ**を扱う。

> 🔴 **2026-09-22 以降、グリッパの速度が変わっている。**
> 自前機は制限なしで 5.43 unit/s だったが、学習データの実測上限 **4.20 unit/s**
> に合わせた（`Dex1Limits.max_speed`）。**掴む動作がゆっくりに見えるのが正常。**
> あわせて把持判定が slew 後の実 publish 値を見るようになった（目標値と比べると
> 閉じ始めの過渡で誤って HOLDING になっていた）。
>
> **これ以前の run と直接比較しないこと。** 条件が違う。掴めない / 遅くて後続が
> timeout する場合は `Dex1Limits(max_speed=None)` で戻せるが、学習分布からは外れる。

## 会場の経路は扱わない

運営の WBC を通す会場の経路（`--action-sink boundary`）はこの手順では扱わない。自前リグには運営の WBC が無いので、
その経路を足してもロボットは動かない。この repo の公開版は会場の経路をサポートしない。

---

## Stage の構成（`orchestrator.py:104` の `STAGE_SKILL_SEQUENCES`）

```
Stage 0   setup -> move_to_table -> post_walk_settle        歩行 (+2 点)
Stage 1   pick -> insert -> rotate_leg                      **卓を回さない**
Stage 2   rotate_table_base -> pick -> insert -> rotate_leg
Stage 3   同上
Stage 4   同上
Stage 5   flip_table                                        **別 policy 経路**
```

配点: Stage 0 **+2** / Stage 1 **5** / Stage 2〜4 各 **6** / Stage 5 **4**

⚠️ `--phase3-start-stage` の既定は `0`、`--phase3-end-stage` は `5`。
つまり **`--phase3-full` は Stage 0〜5 を全部通る。**

⚠️ `STAGE_HEAD_SKILL`（`orchestrator.py:135`）の分岐で、**通しと単体起動で挙動が
違う**。stage 0 は go-live 待ちだけが先頭で、setup で腕を `walk_lowered_pose` へ下ろして
から歩き、残りの頭の手順は歩いた後に入る（2026-09-24）。stage 1 は「stage 1 から起動
したときだけ」先頭に入る。だから単体起動も別に踏む必要がある。

---

## 1. 通し

```bash
set -a && . .env && set +a
pixi run -e runtime python -m inference.desktop.entrypoint \
    --interface eth0 --phase3-full --actuate --use-real-waist --use-real-hand \
    --setup-hold-sec 3 --gpu-models 2 \
    --log /tmp/run_$(date +%H%M).jsonl
```

🔴 **`--log` を付ける。** 残るのは skill 遷移と `n_cleaned`（YOLO の検出数）。
止まった run で「YOLO が 0 個」なのか「検出はあるが遷移条件が立たない」のかが
分かれる。**録画だけでは区別できない。**

⚠️ この log に `taskspace` の行は入らない（`(T,25)` を残すのは
`--action-sink boundary` のときだけ）。IK accept 率の測定はこの run ではできない。

variant は `policy_config.yaml` の `default_variant_by_skill` から自動で入る
（Python 側に ckpt 名を直書きしない）。起動ログの
`[init] policy variants: ... (config) / (cli)` で入った値を確認する。

pick は既定で、学習 policy（RAMEN-Ori の `pick_table_leg_ramen_ori_venue_all6_20k`）だけで pick 全体を動かす
（`policy_config.yaml` の `pick_leg_hybrid: false`、2026-09-30 から）。hybrid 経路（VLM → 学習 policy で掴む → IK/MP →
rule-based 持ち替え → insert 初期姿勢）にするなら `--pick-leg-hybrid`。hybrid の既定で掴む `groot_pick_legs_v1` は公開していない。

各 learned policy の終了後は、現在の腕・Dex1 targetを保持したまま、次policyの
dataset frame-zeroへ有限遷移する。順序は「衝突回避付き腕pre-motion → 次policyの
Dex1初期値 → 安定hold → 次policy開始」。腕は実測角、実測速度、最後にpublishされた
targetが連続して許容内に入るまで完了しない。未到達・timeout・遷移skill欠落時は
HOLDして停止し、次policyを開始しない。hybrid pick→insertだけはpick内部のrule-based
handoverが同じ到達確認を行い、insert初期姿勢を終端状態とする。

**見るもの**

| 項目 | 期待 |
|---|---|
| Stage 0 の歩行 | `move_to_table` は `vx 0.185 × 1.0s = 18.5cm` の決め打ち。届くか |
| Stage 1 で卓を回さない | 回転は 3 回（Stage 2,3,4 の頭）。4 回なら開始 skill がずれている |
| `unsafe_position_target_held` | **5 割を超えたら止める。** URDF 限界 −0.03 rad に触れると 19D target ごと破棄され、上半身が固まる |
| Stage 5 に入れるか | flip の policy（既定は RAMEN-Ori の `flip_table_ramen_ori_flip_sel_a_n15_20k`、2026-10-10 から。実機では未確認）。ここでしか通らない |

**共有**: `[init] recording -> <path>` の path と、`--log` の JSONL。

---

## 2. 各 stage を単体で（6 本）

Stageを単体指定した場合、Stage 1はpick、Stage 2〜4はrotate-table、Stage 5は
flip-tableの初期腕・Dex1姿勢へ到達した後、保持指令を継続したままEnterを待つ。
Enter後に初めて先頭policyを実行する。Stage内の後続サブタスク間にはこの確認を
挿入しない。Stage 0は歩行後にpick初期姿勢へ移動して終了するため確認を挿入しない。
したがってStage 1〜5の単体実行では、1回目のEnterがpre-motion開始、2回目のEnterが
policy開始である。待機中も腕・Dex1・boundary指令とカメラ監視は継続する。

```bash
for N in 0 1 2 3 4 5; do  # 1 本ずつ、手で
pixi run -e runtime python -m inference.desktop.entrypoint \
    --interface eth0 --stage $N \
    --actuate --use-real-waist --use-real-hand \
    --setup-hold-sec 3 --gpu-models 1
done
```

**なぜ通しと別に要るか**: 上の `STAGE_HEAD_SKILL` の分岐に加えて、
**単体起動は物理的な前提を運用側が作る**ことになる。Stage 3 を単体で叩くなら
「2 本挿さっていて卓が回っている」状態を人が作ってから始める。会場では運営が
次 stage の初期状態を作ってくれる（FAQ 明記）ので、そこと同じ形になる。

⚠️ **各 model の 1 本目は `--actuate` を外して 1 回通す。**

---

## 3. 途中から最後まで（2 本）

```bash
# 3 本目の脚から最後まで
pixi run -e runtime python -m inference.desktop.entrypoint \
    --interface eth0 --phase3-full --phase3-start-stage 3 \
    --actuate --use-real-waist --use-real-hand --setup-hold-sec 3 --gpu-models 2

# 1 本目から (stage 0 を飛ばす)
    ... --phase3-full --phase3-start-stage 1 ...
```

単体（2.）と違い、**stage をまたぐ引き継ぎ**が入る。腕の姿勢・グリッパの開度が
前の stage の終わりから次の stage の頭へ渡るところを見る。

会場での回復・再起動は、運営と物理状態を確認して判断する。
自前経路での stage 再開コマンドを、そのまま会場に持ち込まない。

---

## 4. 止めて戻す訓練（1 本）🔴

**これをぶっつけ本番にしない。** 既定は全 skill が `on_timeout: advance` で、
時間切れなら次の skill へ進む (2026-09-22 に復帰)。止まるのは安全判定が
失敗したときで、会場は学習データと違うシーンなので**止まること自体は普通**と
思っておく。この訓練では `on_timeout` を `stop` にしたコピーで、その状態を作る。

### 止まったとき / 終わったときに何が起きるか

**入力は求められない**（2026-09-22 に Enter 待ちを廃止）。正常終了・Ctrl+C・安全停止の
どれでも、解放の直前に同じ後始末が走る。

```
1. 腕は最後の姿勢を保持（いきなり脱力しない）
2. Dex1 を全開にする（脚を掴んだままにしない）
3. 起動時の退避を逆にたどって下ろす
   forward-outward → lateral-high → shoulder-back → 下がった姿勢
   (skills.walk_lowered_pose.pose_rad = 肩 roll ±0.2・肘 0.9、前腕が斜め下。
    全関節 0 は前腕が水平に前を向く姿勢で、下ろした姿勢ではない)
4. arm_sdk weight 1 → 0 の controlled release（2 秒）
5. 安全停止のときは SystemExit(2) で終了。自動復帰は無い
```

端末に出る順番:

```
[safety-stop] ...                          ← 安全停止のときだけ
[Phase 3 HOLD] ... E-stop ... の案内        ← 安全停止のときだけ
[return] opening Dex1 and lowering the arms
[pre-motion 1/4] return_forward_outward_clearance started
...
[return] arms are lowered; controlled release follows
```

⚠️ **危険な動きなら待たずに E-stop。** 戻す動きも腕が動くので、人が近くにいるときは
手を出す前に止めること。**Ctrl+C をもう一度押すと、戻すのをやめて即解放**になる。

記録（JSONL）には `return_started` / `return_complete` / `return_failed` /
`return_timeout` が残る。

### 訓練のやり方

`max_seconds_hard` を短くした skill で走らせれば必ず安全停止に入る。
**正本 (`skill_config.yaml`) は書き換えない。** コピーを作って `--skill-config` で差す。

```bash
CFG=inference/desktop/lower_policy/configs/skill_config.yaml
cp $CFG /tmp/skill_config_holddrill.yaml
python3 - <<'EOF'
import yaml, pathlib
p = pathlib.Path("/tmp/skill_config_holddrill.yaml")
d = yaml.safe_load(p.read_text())
# rotate_table_base を 5 秒で切る (本来 30 秒)。既定は advance = 次の skill へ進むので、
# 訓練のときだけ stop にして安全停止を起こす
d["skills"]["rotate_table_base"]["max_seconds_hard"] = 5.0
d["skills"]["rotate_table_base"]["on_timeout"] = "stop"
p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
print("  max_seconds_hard =", d["skills"]["rotate_table_base"]["max_seconds_hard"],
      "/ on_timeout =", d["skills"]["rotate_table_base"].get("on_timeout"))
EOF

# Stage 2 の頭 (rotate_table_base) で必ず止まる
pixi run -e runtime python -m inference.desktop.entrypoint \
    --interface eth0 --phase3-full \
    --phase3-start-stage 2 --phase3-end-stage 2 \
    --skill-config /tmp/skill_config_holddrill.yaml \
    --actuate --use-real-waist --use-real-hand --setup-hold-sec 3 --gpu-models 1
```

⚠️ 終わったら `/tmp/skill_config_holddrill.yaml` を消す。**本番でこれを差したまま
走らせない**（5 秒で全部止まる）。

**見るもの**

| 項目 | 期待 |
|---|---|
| 腕が落ちないか | 停止の瞬間は最後の target を保持。ガクッと落ちたら publisher が死んでいる |
| 手が開くか | Dex1 が全開になる（脚を持っていれば落ちる。下に物を置かない） |
| 戻る経路 | 頭上・側方を経由して下がった姿勢まで。天板や脚に当たらないか |
| controlled release | 戻り切ってから 2 秒かけて weight 1→0 |
| **解放後の腕** | weight 0 で本体側の制御に戻る。**そのとき腕が動くかどうかをここで確認する**（未確認） |
| 入り直せるか | 3. の `--phase3-start-stage N` で復帰できるか |
| Ctrl+C 2 回 | 戻す動きが止まって即解放になるか |

---

## 安全

- 各 model の 1 本目は `--actuate` を外して 1 回通す
- `unsafe_position_target_held` が 5 割を超えたら止める
- E-stop 担当を必ず配置する

## 記録

各 run の `[init] recording -> <path>` を残す。止めた run も残す（止まり方が資料になる）。

| # | 内容 | 状態 |
|---|---|---|
| 1 | 通し（Stage 0〜5） | ⬜ |
| 2 | 各 stage 単体 ×6 | ⬜ |
| 3 | 途中から最後まで ×2 | ⬜ |
| 4 | 止めて戻す訓練 | ⬜ |
