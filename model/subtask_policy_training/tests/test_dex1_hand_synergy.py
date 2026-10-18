"""Dex1-1 ↔ official G1 7-joint hand synergy port の verify (Issue #128)。

Ported from issue-70-flip-table-data-augmentation branch (verbatim `.py` + JSON
asset)。ここでは port の一貫性 + Gr00tPolicy adapter 統合前提の round-trip
closure を確認する。実データ (HF ckpt の policy_postprocessor.json:raw_stats.
action.{left,right}_hand) が全 4 GR00T ckpt (hara / takada×2 / suzuki) で 7 dim
active であることを 2026-08-30 に verify 済 → 全 GR00T variant で synergy 逆射影が
必須、adapter 側は variant 分岐なし。
"""

from __future__ import annotations

import math

import pytest

from model.subtask_policy_training.gr00t.dex1_hand_synergy import (
    DEX1_CLOSED,
    DEX1_OPEN,
    HAND_DIM,
    dex1_closed_fraction,
    dex1_to_hand,
    hand_to_dex1,
    load_synergy_manifest,
    projection_residual,
    synergy_manifest_sha256,
)


# ---- Manifest asset の pin (SHA 固定で port の drift を検知) ---- #


class TestManifestAssetIntegrity:
    # Port した dex1_g1_synergy.json の SHA256 を固定。upstream issue-70 の
    # 同 asset と bit-identical であることを保証、drift すれば test 失敗。
    EXPECTED_SHA256 = (
        "16c5c500607441255d447d7c15e948b84fada96d41034f01fff2be1d951d465d"
    )

    def test_asset_sha256_pinned(self) -> None:
        assert synergy_manifest_sha256() == self.EXPECTED_SHA256, (
            "dex1_g1_synergy.json が改変されている。upstream issue-70 と bit-"
            "identical に保つ (synergy calibration は NVIDIA 公式 G1 dataset "
            "由来、勝手に変更しない)"
        )

    def test_manifest_schema_version(self) -> None:
        manifest = load_synergy_manifest()
        assert manifest["schema_version"] == "dex1_g1_synergy_v1"

    def test_manifest_dex1_open_closed_constants(self) -> None:
        # Module-level constants と manifest の値が一致
        manifest = load_synergy_manifest()
        assert manifest["dex1"]["open"] == DEX1_OPEN == 4.5
        assert manifest["dex1"]["closed"] == DEX1_CLOSED == 0.0

    def test_manifest_joint_order_matches_official_g1_hand(self) -> None:
        # 7 joint synergy の順序: index_0/1, middle_0/1, thumb_0/1/2
        # (NVIDIA GR00T-N1.7-3B の real_g1_relative_eef_relative_joints 準拠)
        manifest = load_synergy_manifest()
        assert manifest["joint_order"] == [
            "index_0", "index_1", "middle_0", "middle_1",
            "thumb_0", "thumb_1", "thumb_2",
        ]

    def test_manifest_hand_open_closed_vectors_are_7d(self) -> None:
        manifest = load_synergy_manifest()
        for side in ("left", "right"):
            for kind in ("state_open", "state_closed", "action_open", "action_closed"):
                assert len(manifest[side][kind]) == HAND_DIM, (
                    f"{side}.{kind} must have {HAND_DIM} dim"
                )


# ---- dex1_closed_fraction (Dex1 rad → [0,1] 閉度) ---- #


class TestDex1ClosedFraction:
    def test_open_maps_to_zero(self) -> None:
        assert dex1_closed_fraction(DEX1_OPEN) == 0.0

    def test_closed_maps_to_one(self) -> None:
        assert dex1_closed_fraction(DEX1_CLOSED) == 1.0

    def test_midpoint_maps_to_half(self) -> None:
        assert abs(dex1_closed_fraction((DEX1_OPEN + DEX1_CLOSED) / 2) - 0.5) < 1e-9

    def test_clamped_above_open(self) -> None:
        # Dex1 rad が open 越え (>4.5) でも [0,1] にクランプ
        assert dex1_closed_fraction(DEX1_OPEN + 1.0) == 0.0

    def test_clamped_below_closed(self) -> None:
        assert dex1_closed_fraction(DEX1_CLOSED - 1.0) == 1.0

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError):
            dex1_closed_fraction(float("nan"))
        with pytest.raises(ValueError):
            dex1_closed_fraction(float("inf"))


# ---- dex1_to_hand / hand_to_dex1 round-trip ---- #


class TestDex1ToHandRoundTrip:
    """1D Dex1 → 7D synergy → 1D Dex1 の closure。"""

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("kind", ["state", "action"])
    def test_open_round_trip(self, side, kind) -> None:
        seven = dex1_to_hand(DEX1_OPEN, side=side, kind=kind)
        assert len(seven) == HAND_DIM
        recovered = hand_to_dex1(seven, side=side, kind=kind)
        assert abs(recovered - DEX1_OPEN) < 1e-9

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("kind", ["state", "action"])
    def test_closed_round_trip(self, side, kind) -> None:
        seven = dex1_to_hand(DEX1_CLOSED, side=side, kind=kind)
        recovered = hand_to_dex1(seven, side=side, kind=kind)
        assert abs(recovered - DEX1_CLOSED) < 1e-9

    @pytest.mark.parametrize("dex1_value", [0.5, 1.5, 2.25, 3.0, 4.0])
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("kind", ["state", "action"])
    def test_intermediate_round_trip(self, dex1_value, side, kind) -> None:
        # synergy は linear interp なので中間値でも exact round-trip
        seven = dex1_to_hand(dex1_value, side=side, kind=kind)
        recovered = hand_to_dex1(seven, side=side, kind=kind)
        assert abs(recovered - dex1_value) < 1e-9, (
            f"{side}/{kind}: {dex1_value} → 7D → {recovered}"
        )


# ---- hand_to_dex1 projection の robustness (synergy 外部の 7D) ---- #


class TestHandToDex1Projection:
    """任意 7D vector を Dex1 [0, 4.5] に射影する least-squares の挙動。"""

    def test_projection_clamps_above_open(self) -> None:
        # Open pose に近い方向へ大きく振っても [0, DEX1_OPEN] にクランプ
        manifest = load_synergy_manifest()
        opened = manifest["left"]["action_open"]
        overshoot_open = [float(v) - 5.0 for v in opened]  # open 方向 exaggerate
        result = hand_to_dex1(overshoot_open, side="left", kind="action")
        assert 0.0 <= result <= DEX1_OPEN

    def test_projection_clamps_below_closed(self) -> None:
        # Closed 方向へ大きく振ってもクランプ
        manifest = load_synergy_manifest()
        closed = manifest["left"]["action_closed"]
        overshoot_closed = [float(v) + 5.0 for v in closed]
        result = hand_to_dex1(overshoot_closed, side="left", kind="action")
        assert 0.0 <= result <= DEX1_OPEN

    def test_projection_residual_zero_for_synergy_conformant(self) -> None:
        # dex1_to_hand → hand_to_dex1 → dex1_to_hand の再射影 residual が 0
        # (完全に synergy 上にある 7D は元 vector そのものに戻る)
        seven = dex1_to_hand(2.5, side="left", kind="action")
        residual = projection_residual(seven, side="left", kind="action")
        assert residual < 1e-9

    def test_projection_residual_positive_for_off_synergy(self) -> None:
        # synergy 直線から外れた 7D は non-zero residual
        off_synergy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        residual = projection_residual(off_synergy, side="left", kind="action")
        assert residual > 1e-3


# ---- Input validation ---- #


class TestInputValidation:
    def test_rejects_wrong_hand_dim(self) -> None:
        for bad_len in (0, 1, 2, 6, 8, 14):
            with pytest.raises(ValueError, match=str(HAND_DIM)):
                hand_to_dex1([0.0] * bad_len, side="left", kind="action")

    def test_rejects_non_finite_hand(self) -> None:
        bad = [0.0, 0.0, math.nan, 0.0, 0.0, 0.0, 0.0]
        with pytest.raises(ValueError):
            hand_to_dex1(bad, side="left", kind="action")
