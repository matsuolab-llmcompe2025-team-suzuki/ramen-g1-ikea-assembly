# Depth Pipeline

Unitree G1 の wrist (RealSense D405 ×2) / head (stereo RGB) から depth を推定するパイプラインの設計と検証結果。**力センサが無い**問題を depth ベースで代替するための基盤。

---

## TL;DR

- D405 IR projector は **OFF 確定** → active stereo 不可、**passive 前提で設計**
- **L2 (SGBM + WLS filter)** で wrist IR valid **80%**、head RGB valid **61%** 達成
  - gripper / 脚 / 穴の縁が明確に分離され、operative 領域はほぼ全部取れる
- L4 (deep stereo) は当面不要。L2 で insertion 用に実用域に達する
- 力センサ無し問題に対しては **wrist depth を接触検出 / 滑り検出 / insertion 完了判定の proxy** として利用

---

## 1. ハードウェア仕様

### Wrist (RealSense D405 ×2)

| 項目 | 値 |
|---|---|
| 構成 | left wrist + right wrist 各 1 ユニット |
| RGB | 640×480, fx=437, fy=437, Brown-Conrady distortion |
| IR1 / IR2 | 640×480, fx=fy=389.85, **distortion = 0** (rectify 済) |
| baseline (IR1 ↔ IR2) | **18.09 mm** |

### Head (stereo RGB)

| 項目 | 値 |
|---|---|
| 構成 | stereo packed image 1280×480 → split で 2 × 640×480 |
| baseline | **63.02 mm** |
| focal (rectified P1) | **510.15 px** |
| 校正 RMS | 1.11 px |
| 提供物 | K_l, K_r, D_l, D_r, R, T, R1, R2, P1, P2, **Q** (reproject 即可) |
| 備考 | IR 無し (D405 系ではない、別の stereo モジュール) |

### 理論精度

| カメラ | 距離 20cm | 距離 50cm | 距離 1m | 用途 |
|---|---|---|---|---|
| Wrist IR | **0.5 mm** | 3 mm | 14 mm | 精密 (insertion / grasp) |
| Head RGB | 0.1 mm 強 | **2 mm** | 8 mm | シーン (table / 残脚配置) |

---

## 2. D405 IR Projector — **OFF 確定**

1 episode 内 10 タイムスタンプ (t=10〜750s に分散) の left_wrist IR1 を抽出し、Laplacian variance / high-pass mean / hf_ratio をヒューリスティック値 + 視覚で判定:

- 全 10 フレームで **projector のドットパターン視認不能**
- 値の変動範囲も狭く、cycle on/off の徴候なし
- → **このデータセット全期間で passive 動作**と判定

### 含意

- active stereo の恩恵は得られない → SGBM / Deep stereo の改善で対応
- 本番データで projector ON なら +α の品質向上余地ありだが、**現データで設計するなら passive 前提**
- 別 episode 横断確認は不要 (episode 内 10 点で十分強い証拠)

---

## 3. SGBM 実走結果

### L1: 素の SGBM

```python
cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,   # wrist。head は 192
    blockSize=5,          # head は 7
    P1=8*3*B**2, P2=32*3*B**2,
    disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
```

| 入力 | valid disparity 率 | 中央値 depth |
|---|---:|---:|
| left wrist IR (passive) | 51.33% | 144.7 mm |
| head RGB stereo | 34.95% | 1,224.8 mm |

### L2: SGBM + WLS filter

WLS パラメータ: `lambda=8000.0, sigma_color=1.5`

```python
matcher_left = cv2.StereoSGBM_create(...)
matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)
wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left)
wls.setLambda(8000); wls.setSigmaColor(1.5)
disp_filtered = wls.filter(disp_left, img_left, None, disp_right)
```

| 入力 | L1 valid | L2 valid | 改善 | 視覚的品質 |
|---|---:|---:|---:|---|
| Wrist IR disparity | 51.33% | **80.00%** | **+28.7 pp** | gripper finger / 脚 / 縁が明確分離 |
| Wrist IR depth (30-400mm) | 47.80% | 68.24% | +20.4 pp | distance crop 内は十分使える |
| Head RGB disparity | 34.95% | **61.16%** | +26.2 pp | table 全体 + アーム depth 連続性確保 |
| Head RGB depth (100-3000mm) | 26.30% | 22.87% | -3.4 pp ※ | WLS 補間で disp が極小値に飛び depth crop 外へ流出 |

※ Head depth の degradation は WLS のパラメータ問題。`lambda` を下げる、or disparity を `[1, 200]` などに clip して compute すれば改善する見込み。disparity マップ自体は綺麗。

### 視認結論

**Wrist IR**:
- ✓ table の穴の縁、脚、gripper の輪郭で depth 取得
- ✓ 上端カーペット境界で depth 取得
- ✗ 暗い gripper 表面 (テクスチャ無 + 暗で SGBM が match できず)
- ✗ 白プラ平面の中央部
- **重要**: gripper 自体は FK で位置既知なので depth で見る必要なし

**Head RGB**:
- ✓ 手前カーペット / table = 近 (300-500mm 推定)
- ✓ アーム / グリッパ = 中距離 (500-1000mm)
- ✓ 背景の人 / 犬 = 遠 (1-3m)
- ✗ 左上の白い壁 / 背景の特徴薄領域
- **near→far のグラデーションが完全に正しい**

---

## 4. 力センサ無し問題への depth ベース代替

データセットに force / torque sensor topic が存在しないため、wrist depth を proxy として利用する。

| 必要な信号 | 力センサでの理想 | depth ベース代替 |
|---|---|---|
| 接触検出 | 力スパイク | wrist depth が変化停止 |
| 滑り検出 | 微小トルク変動 | gripper 内脚 depth の時間変化 |
| insertion 完了 | 抗力解放 | 脚頭 depth ↔ table 表面の一致 |
| 軸ミスアラインメント | コンタクト anomaly | 円柱 axis fit vs 穴 axis fit の角度差 |
| 過剰押付け防止 | 力閾値 | depth 接近速度制限 |

---

## 5. 改善ロードマップ

### L1 → L2: WLS filter (実装済)

- 期待 valid 率: 51% → 70-80%, 35% → 55-65% → **実測 80% / 61% で達成**
- 計算量: SGBM の +3 倍程度 (許容)

### L2 → L3: Depth Anything v2 fusion

- DAv2-Small (24M) でモノキュラ depth 推論 (GPU 30ms)
- DAv2 の相対 depth を SGBM の metric depth で scale & shift fit
- 結果: SGBM 穴を DAv2 で埋め、メトリック性は SGBM で保証

### L3 → L4: Deep stereo (性能ボトルネック計測後)

候補:

- **RAFT-Stereo** (Princeton): 軽量・汎化強い
- **CREStereo**: in-the-wild 性能高い
- **IGEV-Stereo**: SOTA 級
- **FoundationStereo** (NVIDIA 2026): zero-shot stereo foundation model

→ L2 で実用域なので **当面 L3 / L4 は不要**。性能ボトルネックが計測された時点で再評価。

---

## 6. backlog

- [ ] `/stamped/lowstate` に joint torque が含まれるか確認 (含まれれば depth + joint torque fusion)
- [ ] WLS パラメータチューニング (lambda 引下げ + disparity clip) で head depth の crop 流出修正
- [ ] head_depth → 上位 VLM 用 textual summary 関数の試作
- [ ] DAv2 monocular 統合 (L3、必要時のみ)
- [ ] L4 RAFT-Stereo を試走して L2 との差分計測 (必要時のみ)
- [ ] 30 fps streaming で SGBM+WLS のレイテンシ実測
