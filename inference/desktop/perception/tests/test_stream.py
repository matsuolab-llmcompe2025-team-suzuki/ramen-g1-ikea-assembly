"""DetectionStream の unit test。

主眼:
    - warmup 期間の Optional[None] 挙動
    - flush の残り frame emit
    - batch API (`clean_all_frames`) との等価性 (中央 index 範囲で)
    - median enabled / disabled 両方の latency 差分
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from inference.desktop.perception.cleaner import clean_all_frames
from inference.desktop.perception.stream import DetectionStream
from inference.desktop.perception.yolo_obb import OBBDetection


def _det(
    class_name: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    class_id: int = 0,
    confidence: float = 0.9,
) -> OBBDetection:
    """axis-aligned bbox から OBB 4 頂点 (時計回り) を作って detection にする。"""
    verts = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
    )
    return OBBDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        verts=verts,
    )


def _base_config(median_enabled: bool) -> dict:
    """test 用 config (yaml 経由じゃなく直接 dict を渡す)。"""
    return {
        "max_count": {"leg": 4, "hand_left": 1, "hand_right": 1},
        "over_max_continue_iou": 0.3,
        "under_max_similar_iou": 0.3,
        "median_filter": {"enabled": median_enabled, "iou_match_min": 0.5},
    }


# =========================================================================
# median disabled (1 frame delay)
# =========================================================================
class TestMedianDisabled:
    def test_first_push_returns_none(self):
        stream = DetectionStream(_base_config(median_enabled=False))
        assert stream.push([_det("leg", 0.1, 0.1, 0.2, 0.2)]) is None

    def test_second_push_emits_first_cleaned(self):
        """1 frame delay: 2 回目の push で 1 frame 目の cleaned が emit される。"""
        stream = DetectionStream(_base_config(median_enabled=False))
        f0 = [_det("leg", 0.10, 0.10, 0.20, 0.20)]
        f1 = [_det("leg", 0.11, 0.11, 0.21, 0.21)]  # 1 frame 目に似た位置
        stream.push(f0)
        out = stream.push(f1)
        assert out is not None
        # f0 の leg は f1 に similar → continuing で保持される
        assert len(out) == 1
        assert out[0].class_name == "leg"

    def test_flush_emits_last_frame(self):
        """flush で最後の raw を next_raw=[] で clean、emit。"""
        stream = DetectionStream(_base_config(median_enabled=False))
        f0 = [_det("leg", 0.10, 0.10, 0.20, 0.20)]
        f1 = [_det("leg", 0.11, 0.11, 0.21, 0.21)]
        f2 = [_det("leg", 0.12, 0.12, 0.22, 0.22)]
        stream.push(f0)
        stream.push(f1)
        stream.push(f2)
        remaining = stream.flush()
        # flush で cleaned_2 emit
        assert len(remaining) == 1


# =========================================================================
# median enabled (2 frame delay)
# =========================================================================
class TestMedianEnabled:
    def test_first_three_pushes_return_none(self):
        """median enabled: warmup は Step 1 x1 + median x2 = 計 3 push 必要。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        f0 = [_det("leg", 0.10, 0.10, 0.20, 0.20)]
        f1 = [_det("leg", 0.11, 0.11, 0.21, 0.21)]
        f2 = [_det("leg", 0.12, 0.12, 0.22, 0.22)]
        assert stream.push(f0) is None
        assert stream.push(f1) is None
        assert stream.push(f2) is None

    def test_fourth_push_emits_first_median(self):
        """4 push 目で cleaned_1 の 3-tap median が emit される。

        flow:
            push(f0): Step 1 warmup (raw_curr=None) → None
            push(f1): Step 1 で cleaned_0、median_ring size 1 → None
            push(f2): Step 1 で cleaned_1、median_ring size 2 → None
            push(f3): Step 1 で cleaned_2、median_ring size 3 → emit median(c0,c1,c2)
        """
        stream = DetectionStream(_base_config(median_enabled=True))
        f0 = [_det("leg", 0.10, 0.10, 0.20, 0.20)]
        f1 = [_det("leg", 0.11, 0.11, 0.21, 0.21)]
        f2 = [_det("leg", 0.12, 0.12, 0.22, 0.22)]
        f3 = [_det("leg", 0.13, 0.13, 0.23, 0.23)]
        stream.push(f0)
        stream.push(f1)
        stream.push(f2)
        out = stream.push(f3)
        assert out is not None
        assert len(out) == 1


# =========================================================================
# batch API との等価性 (最重要 regression 防止)
# =========================================================================
def _run_streaming(
    all_dets: list[list[OBBDetection]], config: dict
) -> list[list[OBBDetection]]:
    """streaming で全 frame push + flush して emit を集める。"""
    stream = DetectionStream(config)
    emitted: list[list[OBBDetection]] = []
    for raw in all_dets:
        out = stream.push(raw)
        if out is not None:
            emitted.append(out)
    emitted.extend(stream.flush())
    return emitted


def _sample_ep(n: int) -> list[list[OBBDetection]]:
    """n frame の simple ep (leg が徐々に移動する) を生成。"""
    ep = []
    for t in range(n):
        offset = 0.001 * t
        ep.append(
            [
                _det("leg", 0.10 + offset, 0.10 + offset, 0.20 + offset, 0.20 + offset),
                _det("hand_left", 0.50, 0.50, 0.60, 0.60),
                _det("hand_right", 0.65, 0.55, 0.75, 0.65),
            ]
        )
    return ep


class TestBatchEquivalence:
    @pytest.mark.parametrize("median_enabled", [True, False])
    @pytest.mark.parametrize("n_frames", [5, 10, 20])
    def test_middle_frames_match_batch(
        self, median_enabled: bool, n_frames: int
    ):
        """streaming の emit と batch API の中央 index 範囲が verts で一致する。

        - median disabled 時: streaming は全 n frame emit、batch と 1:1 一致すべき
        - median enabled 時: streaming は index 1..n-2 emit、batch との対応 index で一致
        """
        config = _base_config(median_enabled=median_enabled)
        ep = _sample_ep(n_frames)

        batch_out = clean_all_frames(ep, config)
        streaming_out = _run_streaming(ep, config)

        if not median_enabled:
            # median disabled: streaming は n frame emit (batch と同数)
            assert len(streaming_out) == n_frames
            for t in range(n_frames):
                _assert_frame_equal(streaming_out[t], batch_out[t])
        else:
            # median enabled: streaming は index 1..n-2 emit (batch の n-2 個と対応)
            assert len(streaming_out) == n_frames - 2
            for i, t in enumerate(range(1, n_frames - 1)):
                _assert_frame_equal(streaming_out[i], batch_out[t])


def _assert_frame_equal(
    a: list[OBBDetection], b: list[OBBDetection]
) -> None:
    """2 つの frame の detection list が (class_name, verts) で等価か検証。"""
    assert len(a) == len(b), f"detection count mismatch: {len(a)} vs {len(b)}"
    # class_name でグループ化して比較 (順序不問)
    def _key(d: OBBDetection) -> tuple:
        return (d.class_name, tuple(d.verts.flatten().tolist()))

    a_sorted = sorted(a, key=_key)
    b_sorted = sorted(b, key=_key)
    for da, db in zip(a_sorted, b_sorted):
        assert da.class_name == db.class_name
        np.testing.assert_allclose(da.verts, db.verts, atol=1e-5)


# =========================================================================
# 端 case
# =========================================================================
class TestEdgeCases:
    def test_empty_raw_all_frames(self):
        """全 frame で detection 0 個 → streaming も空 emit。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        for _ in range(5):
            stream.push([])
        remaining = stream.flush()
        # median enabled で 5 frame → emit は index 1..3 (3 個) + flush で index 4 の中央 = 1 個
        # ただし全 detection 空なので中身も空
        # 具体的 count は buffer 動作次第、全 emit が [] であることのみ確認
        assert all(len(r) == 0 for r in remaining)

    def test_flush_with_no_pushes(self):
        """push 0 回で flush → 空 list。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        assert stream.flush() == []

    def test_single_push_then_flush(self):
        """1 push だけ + flush の極端 case。median enabled では端 frame drop で emit 0。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        stream.push([_det("leg", 0.10, 0.10, 0.20, 0.20)])
        remaining = stream.flush()
        # ring size < 3 なので flush の median emit は 0 個 (design: 端 frame drop)
        assert remaining == []


# =========================================================================
# snapshot_buffer (skill 遷移時 resume 用 dump)
# =========================================================================
class TestSnapshotBuffer:
    """DetectionStream.snapshot_buffer の contract。

    JSON serializable dict を返し、buffer が空でも None を返さない (form 保証)。
    """

    def test_snapshot_shape_before_any_push(self):
        stream = DetectionStream(_base_config(median_enabled=True))
        snap = stream.snapshot_buffer()

        assert snap["raw_curr"] is None
        assert snap["cleaned_prev"] is None
        assert snap["median_ring"] == []
        assert snap["median_enabled"] is True

    def test_snapshot_median_enabled_flag_reflects_config(self):
        stream_off = DetectionStream(_base_config(median_enabled=False))
        assert stream_off.snapshot_buffer()["median_enabled"] is False

    def test_snapshot_after_first_push_captures_raw_curr(self):
        """1 push 目: raw_curr にその raw が入り、cleaned_prev はまだ None。"""
        stream = DetectionStream(_base_config(median_enabled=False))
        stream.push([_det("leg", 0.1, 0.1, 0.2, 0.2)])

        snap = stream.snapshot_buffer()
        assert snap["raw_curr"] is not None
        assert len(snap["raw_curr"]) == 1
        assert snap["raw_curr"][0]["class_name"] == "leg"
        # verts は list[list[float]] に変換されている
        assert isinstance(snap["raw_curr"][0]["verts"], list)
        assert len(snap["raw_curr"][0]["verts"]) == 4  # 4 頂点
        assert len(snap["raw_curr"][0]["verts"][0]) == 2  # (x, y)
        assert snap["cleaned_prev"] is None

    def test_snapshot_after_steady_state_has_all_fields(self):
        """median enabled で 4 push (定常 emit 開始後): raw_curr / cleaned_prev / median_ring 全て埋まる。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        for i in range(4):
            stream.push([_det("leg", 0.1 + 0.01 * i, 0.1, 0.2, 0.2)])

        snap = stream.snapshot_buffer()
        assert snap["raw_curr"] is not None
        assert snap["cleaned_prev"] is not None
        assert len(snap["median_ring"]) == 3  # maxlen=3 の deque が満杯
        assert snap["median_enabled"] is True

    def test_snapshot_is_json_serializable(self):
        """snapshot dict は json.dumps でそのまま serialize できる (log emit 用)。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        for i in range(3):
            stream.push([_det("leg", 0.1 + 0.01 * i, 0.1, 0.2, 0.2)])

        snap = stream.snapshot_buffer()
        # 例外なく serialize できる
        s = json.dumps(snap)
        # round-trip で dict 復元可能
        recovered = json.loads(s)
        assert recovered["median_enabled"] is True
        assert recovered["raw_curr"][0]["class_name"] == "leg"


# =========================================================================
# restore_from_buffer (skill 遷移時 snapshot からの復元)
# =========================================================================
class TestRestoreFromBuffer:
    """DetectionStream.restore_from_buffer の contract。

    snapshot_buffer() の逆操作で buffer を prewarm し、次 push() から warmup を
    経由せず emit に入れることを保証する。
    """

    def test_roundtrip_median_disabled_preserves_state(self):
        """median disabled: raw_curr + cleaned_prev を round-trip 復元。"""
        stream = DetectionStream(_base_config(median_enabled=False))
        stream.push([_det("leg", 0.1, 0.1, 0.2, 0.2)])
        stream.push([_det("hand_left", 0.3, 0.3, 0.4, 0.4)])
        snap = stream.snapshot_buffer()

        restored = DetectionStream(_base_config(median_enabled=False))
        # 復元経由 → JSON round trip も含めて検証 (実運用 flow に近い)
        restored.restore_from_buffer(json.loads(json.dumps(snap)))

        assert restored._raw_curr is not None
        assert len(restored._raw_curr) == 1
        assert restored._raw_curr[0].class_name == "hand_left"
        assert restored._cleaned_prev is not None

    def test_roundtrip_median_enabled_preserves_ring(self):
        """median enabled: median_ring (最大 3 frame) も復元される。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        for i in range(4):
            stream.push([_det("leg", 0.1 + 0.01 * i, 0.1, 0.2, 0.2)])
        snap = stream.snapshot_buffer()

        restored = DetectionStream(_base_config(median_enabled=True))
        restored.restore_from_buffer(json.loads(json.dumps(snap)))

        assert len(restored._median_ring) == len(snap["median_ring"])
        assert restored._raw_curr is not None
        assert restored._cleaned_prev is not None

    def test_median_enabled_mismatch_raises(self):
        """snapshot 側と現 instance で median_enabled が異なると ValueError。"""
        stream_on = DetectionStream(_base_config(median_enabled=True))
        stream_on.push([_det("leg", 0.1, 0.1, 0.2, 0.2)])
        snap = stream_on.snapshot_buffer()

        stream_off = DetectionStream(_base_config(median_enabled=False))
        with pytest.raises(ValueError, match="median_enabled mismatch"):
            stream_off.restore_from_buffer(snap)

    def test_restore_preserves_verts_dtype(self):
        """復元後の verts は np.ndarray (float32) で shape (4, 2)。"""
        stream = DetectionStream(_base_config(median_enabled=False))
        stream.push([_det("leg", 0.1, 0.2, 0.3, 0.4)])
        snap = stream.snapshot_buffer()

        restored = DetectionStream(_base_config(median_enabled=False))
        restored.restore_from_buffer(json.loads(json.dumps(snap)))

        det = restored._raw_curr[0]
        assert isinstance(det.verts, np.ndarray)
        assert det.verts.dtype == np.float32
        assert det.verts.shape == (4, 2)

    def test_dict_to_obb_rejects_wrong_verts_shape(self):
        """malformed log / 手動編集 log の防護: verts shape != (4, 2) は ValueError。"""
        from inference.desktop.perception.stream import _dict_to_obb

        # shape (1, 2) — vertex 数が足りない
        bad = {
            "class_id": 0,
            "class_name": "leg",
            "confidence": 0.9,
            "verts": [[0.1, 0.2]],
        }
        with pytest.raises(ValueError, match="shape.*4, 2"):
            _dict_to_obb(bad)

    def test_dict_to_obb_accepts_correct_shape(self):
        from inference.desktop.perception.stream import _dict_to_obb

        good = {
            "class_id": 0,
            "class_name": "leg",
            "confidence": 0.9,
            "verts": [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]],
        }
        det = _dict_to_obb(good)
        assert det.verts.shape == (4, 2)
        assert det.verts.dtype == np.float32

    def test_push_after_restore_emits_immediately(self):
        """restore 後の 1 回目 push で cleaned が emit される (warmup skip)。"""
        # 1) 元 stream で warmup 通して定常 emit 開始
        stream = DetectionStream(_base_config(median_enabled=False))
        stream.push([_det("leg", 0.1, 0.1, 0.2, 0.2)])
        snap = stream.snapshot_buffer()

        # 2) 新 stream で restore、次 push は warmup ゼロで emit を返す
        restored = DetectionStream(_base_config(median_enabled=False))
        restored.restore_from_buffer(snap)
        out = restored.push([_det("leg", 0.15, 0.1, 0.25, 0.2)])
        # median disabled + raw_curr が既にある → 1 push で cleaned emit
        assert out is not None


# =========================================================================
# case A (max 超え) の扱い (Issue #141 束 1-9)
# =========================================================================
class TestOverMaxMode:
    def _dets(self):
        """cap 1 の hand_left が 2 個 (前 frame と重なるものは無い)。"""
        return [
            _det("hand_left", 0.1, 0.1, 0.2, 0.2, confidence=0.6),
            _det("hand_left", 0.7, 0.7, 0.8, 0.8, confidence=0.9),
        ]

    def test_keep_continuing_drops_class(self):
        from inference.desktop.perception.cleaner import clean_frame

        out = clean_frame(self._dets(), None, self._dets(), {"hand_left": 1}, 0.9, 0.5)
        assert out == []

    def test_fill_by_conf_keeps_highest_confidence(self):
        from inference.desktop.perception.cleaner import clean_frame

        out = clean_frame(
            self._dets(), None, self._dets(), {"hand_left": 1}, 0.9, 0.5, "fill_by_conf"
        )
        assert len(out) == 1 and out[0].confidence == pytest.approx(0.9)

    def test_fill_by_conf_prefers_continuing_box(self):
        from inference.desktop.perception.cleaner import clean_frame

        prev = [_det("hand_left", 0.1, 0.1, 0.2, 0.2, confidence=0.6)]
        out = clean_frame(
            self._dets(), prev, self._dets(), {"hand_left": 1}, 0.9, 0.5, "fill_by_conf"
        )
        assert len(out) == 1 and out[0].confidence == pytest.approx(0.6)

    def test_stream_passes_mode_from_config(self):
        config = _base_config(median_enabled=False)
        config["over_max_mode"] = "fill_by_conf"
        stream = DetectionStream(config)
        stream.push(self._dets())
        out = stream.push(self._dets())
        assert len(out) == 1 and out[0].confidence == pytest.approx(0.9)


# =========================================================================
# Issue #140: median 対応づけの centroid fallback (振動対策)
# =========================================================================
from inference.desktop.perception.cleaner import (  # noqa: E402
    _find_best_match,
    _median_single_frame,
    resolve_median_match_params,
)


def _shift(det: OBBDetection, dx: float, dy: float) -> OBBDetection:
    """det を (dx, dy) 平行移動した複製 (振動によるフレーム間移動の模擬)。"""
    return OBBDetection(
        class_id=det.class_id,
        class_name=det.class_name,
        confidence=det.confidence,
        verts=(det.verts + np.array([dx, dy], dtype=np.float32)),
    )


def test_find_best_match_iou_only_default_returns_none_on_large_shift():
    """既定 (max_centroid_dist=0): 幅40%ずれ=IoU<0.5 なら None (従来挙動、回帰保護)。"""
    target = _det("leg", 0.40, 0.40, 0.50, 0.50)  # width 0.1
    shifted = _shift(target, 0.04, 0.04)  # 40% of width → IoU ~0.22 < 0.5
    assert _find_best_match(target.verts, [shifted], iou_min=0.5) is None


def test_find_best_match_centroid_fallback_matches_under_shift():
    """max_centroid_dist>0: IoU 全滅でも重心が近ければ fallback 採用。"""
    target = _det("leg", 0.40, 0.40, 0.50, 0.50)
    shifted = _shift(target, 0.04, 0.04)  # centroid dist ~0.057
    got = _find_best_match(
        target.verts, [shifted], iou_min=0.5, max_centroid_dist=0.15
    )
    assert got is shifted
    # 距離上限を下回る設定なら不採用
    assert (
        _find_best_match(
            target.verts, [shifted], iou_min=0.5, max_centroid_dist=0.03
        )
        is None
    )


def test_find_best_match_accepts_exactly_the_ambiguity_ratio():
    """ちょうど ratio 倍のときは採用する (docstring の「倍以上近い」に合わせる)。"""
    target = _det("leg", 0.10, 0.10, 0.14, 0.14)
    near = _shift(target, 0.05, 0.0)      # 重心の距離 0.05
    far = _shift(target, 0.15, 0.0)       # 重心の距離 0.15 = 0.05 * 3.0
    match = _find_best_match(target.verts, [near, far], 0.5, 0.20, 3.0)
    assert match is near


def test_find_best_match_ambiguous_two_legs_rejected():
    """同 class 2 個が等距離 (どちらか曖昧) なら誤対応を避けて不採用。"""
    target = _det("leg", 0.40, 0.40, 0.50, 0.50)
    near_a = _shift(target, 0.04, 0.04)   # centroid dist ~0.057
    near_b = _shift(target, -0.04, -0.04)  # 同距離 (対称)
    got = _find_best_match(
        target.verts, [near_a, near_b], iou_min=0.5,
        max_centroid_dist=0.15, ambiguity_ratio=2.0,
    )
    assert got is None
    # 片方が十分遠ければ nearest を採用
    far_b = _shift(target, 0.30, 0.30)
    got2 = _find_best_match(
        target.verts, [near_a, far_b], iou_min=0.5,
        max_centroid_dist=0.15, ambiguity_ratio=2.0,
    )
    assert got2 is near_a


def test_median_single_frame_fallback_off_passthrough_vs_on_smooths():
    """fallback off=素通し / on=median 適用 を対比 (B3 の core)。"""
    curr = [_det("leg", 0.40, 0.40, 0.50, 0.50)]
    prev = [_shift(curr[0], 0.04, 0.04)]   # 40% shift → IoU<0.5
    nxt = [_shift(curr[0], 0.05, 0.05)]
    # off: 対応切れ → raw passthrough (verts 不変)
    off = _median_single_frame(prev, curr, nxt, iou_min=0.5)
    assert np.array_equal(off[0].verts, curr[0].verts)
    # on: 対応成立 → median 適用 (verts 変化)
    on = _median_single_frame(
        prev, curr, nxt, iou_min=0.5, max_centroid_dist=0.15
    )
    assert not np.array_equal(on[0].verts, curr[0].verts)


def test_resolve_median_match_params_env_override(monkeypatch):
    """env override が YAML より優先されること。"""
    mf = {"match_max_centroid_dist": 0.0, "match_ambiguity_ratio": 2.0}
    assert resolve_median_match_params(mf) == (0.0, 2.0)
    monkeypatch.setenv("RAMEN_MEDIAN_MATCH_MAX_CENTROID_DIST", "0.12")
    monkeypatch.setenv("RAMEN_MEDIAN_MATCH_AMBIGUITY_RATIO", "3.0")
    assert resolve_median_match_params(mf) == (0.12, 3.0)


def test_detection_stream_reads_env_toggle(monkeypatch):
    """DetectionStream が env の fallback 設定を拾うこと。"""
    monkeypatch.setenv("RAMEN_MEDIAN_MATCH_MAX_CENTROID_DIST", "0.12")
    stream = DetectionStream(_base_config(median_enabled=True))
    assert stream._match_max_dist == 0.12
