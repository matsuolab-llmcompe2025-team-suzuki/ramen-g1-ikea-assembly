"""FrameSource / LerobotFrameSource / Ros2FrameSource の unit test。

Ros2FrameSource の test は cyclonedds / SDK / cv2 install 無しで動くよう、__init__ を
bypass して _cb を直接叩く形。fake msg (data / header.stamp を持つ Namespace) を
渡す。cyclonedds が deserialize する CompressedImage_ instance と同じ shape
(data: bytes-like / header.stamp.sec / header.stamp.nanosec) を SimpleNamespace で
再現する。decode は cv2 の代わりに fake cv2 を self._cv2 seam に注入し、可逆 serialize
で round-trip を検証する (JPEG 圧縮そのものは cv2 の責務なので test 対象外)。
"""

from __future__ import annotations

import ast
import threading
import time
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.desktop.perception.frame_source import (
    FrameData,
    LerobotFrameSource,
    Ros2FrameSource,
    adapt_wrist_to_training_shape,
)


def _make_frame(t: int, h: int = 4, w: int = 4) -> FrameData:
    rgb = np.full((h, w, 3), t % 256, dtype=np.uint8)
    return FrameData(rgb=rgb, t=t)


class TestFrameData:
    def test_frame_data_immutable(self):
        """frozen dataclass: field 書き換えは FrozenInstanceError。"""
        frame = _make_frame(t=0)
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            frame.t = 1

    def test_frame_data_holds_bgr_shape(self):
        frame = _make_frame(t=5, h=8, w=12)
        assert frame.rgb.shape == (8, 12, 3)
        assert frame.t == 5


class TestLerobotFrameSource:
    def test_iterates_and_ends_with_none(self):
        """3 frame の iter → get() 3 回で順に返す、4 回目は None、以降永久 None。"""
        frames = [_make_frame(t=i) for i in range(3)]
        source = LerobotFrameSource(frames)

        got = [source.get() for _ in range(3)]
        assert all(g is not None for g in got)
        assert [g.t for g in got] == [0, 1, 2]

        # 4 回目以降は None (exhausted)
        for _ in range(5):
            assert source.get() is None

    def test_empty_iter_immediately_none(self):
        """空 iter → 最初の get() から None。"""
        source = LerobotFrameSource([])
        assert source.get() is None
        assert source.get() is None  # 繰り返しでも None

    def test_generator_iter_supported(self):
        """iterable は list じゃなく generator でも動く (LeRobot dataset は generator 想定)。"""

        def _gen():
            for i in range(2):
                yield _make_frame(t=i * 10)

        source = LerobotFrameSource(_gen())
        f0 = source.get()
        f1 = source.get()
        f2 = source.get()
        assert f0 is not None and f0.t == 0
        assert f1 is not None and f1.t == 10
        assert f2 is None


# =========================================================================
# Ros2FrameSource (rclpy install 不要、_cb を直接呼ぶ形で test)
# =========================================================================
class _FakeCv2:
    """cv2 の imencode / imdecode / IMREAD_COLOR を満たす最小 fake。

    default env (cv2 未 install) でも _cb の decode ロジックを test できるよう、
    Ros2FrameSource が __init__ で保持する `self._cv2` seam にこれを注入する。
    JPEG 圧縮は再現せず可逆 serialize (magic + h + w + raw bytes) で round-trip する
    — 圧縮そのものは cv2 の責務で test 対象外。magic 不一致は「壊れ画像」= None を返す
    (cv2.imdecode の None 返り相当)。
    """

    IMREAD_COLOR = 1
    _MAGIC = b"FAKEIMG0"

    def imencode(self, ext: str, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        buf = (
            self._MAGIC
            + int(h).to_bytes(4, "little")
            + int(w).to_bytes(4, "little")
            + rgb.astype(np.uint8).tobytes()
        )
        return True, np.frombuffer(buf, dtype=np.uint8)

    def imdecode(self, arr: np.ndarray, flag: int):
        buf = arr.tobytes()
        if not buf.startswith(self._MAGIC):
            return None
        body = buf[len(self._MAGIC) :]
        h = int.from_bytes(body[:4], "little")
        w = int.from_bytes(body[4:8], "little")
        return np.frombuffer(body[8:], dtype=np.uint8).reshape(h, w, 3).copy()

    def hconcat(self, images: list):
        """ZmqFrameSource が packed を組むのに使う。cv2.hconcat = 横方向の連結。"""
        return np.hstack(images)


_FAKE_CV2 = _FakeCv2()


def _make_ros2_source_no_init() -> Ros2FrameSource:
    """__init__ を bypass して internal state だけ初期化した instance を返す。

    cyclonedds / SDK / cv2 install 無しの main env で test するため、SDK
    ChannelFactory 経由の subscription 登録を skip し、decode は fake cv2 を
    self._cv2 に注入して _cb / get の behavior だけ検証する。
    """
    source = Ros2FrameSource.__new__(Ros2FrameSource)
    source._latest = None  # type: ignore[attr-defined]
    source._pending = None  # type: ignore[attr-defined]
    source._received_count = 0  # type: ignore[attr-defined]
    source._decoded_count = 0  # type: ignore[attr-defined]
    source._last_jpeg_size = 0  # type: ignore[attr-defined]
    source._lock = threading.Lock()  # type: ignore[attr-defined]
    source._stereo_view = "packed"  # type: ignore[attr-defined]
    source._closed = False  # type: ignore[attr-defined]
    source._channel = MagicMock()  # type: ignore[attr-defined]
    source._cv2 = _FAKE_CV2  # type: ignore[attr-defined]
    source._decode_event = threading.Event()  # type: ignore[attr-defined]
    source._decode_thread = None  # type: ignore[attr-defined]
    return source


def _fake_compressed_image_msg(
    rgb: np.ndarray, stamp_sec: int, stamp_nanosec: int
) -> object:
    """rgb を fake cv2 で encode して CompressedImage 互換の fake msg を作る。"""
    ok, encoded = _FAKE_CV2.imencode(".jpg", rgb)
    assert ok, "encode failed"
    stamp = types.SimpleNamespace(sec=stamp_sec, nanosec=stamp_nanosec)
    header = types.SimpleNamespace(stamp=stamp)
    return types.SimpleNamespace(data=bytes(encoded.tobytes()), header=header)


class TestRos2FrameSource:
    def test_invalid_stereo_view_rejected_before_subscription_setup(self):
        with pytest.raises(ValueError, match="stereo_view"):
            Ros2FrameSource("/camera", stereo_view="invalid")

    def test_get_returns_none_before_any_callback(self):
        source = _make_ros2_source_no_init()
        assert source.get() is None

    def test_close_is_idempotent_and_releases_reader(self):
        source = _make_ros2_source_no_init()
        channel = source._channel  # type: ignore[attr-defined]
        source._latest = _make_frame(t=1)  # type: ignore[attr-defined]

        source.close()
        source.close()

        channel.CloseReader.assert_called_once_with()
        assert source.get() is None

    def test_cb_updates_latest(self):
        """_cb を直接呼んで latest が更新される + get() で取れる。"""
        source = _make_ros2_source_no_init()
        rgb_in = np.full((8, 8, 3), 100, dtype=np.uint8)
        msg = _fake_compressed_image_msg(rgb_in, stamp_sec=2, stamp_nanosec=500_000_000)

        source._cb(msg)

        result = source.get()
        assert result is not None
        # t = 2 * 1e9 + 5e8 = 2_500_000_000
        assert result.t == 2_500_000_000
        assert result.received_monotonic_ns is not None
        assert result.received_monotonic_ns > 0
        assert source.diagnostics() == {
            "received_count": 1,
            "decoded_count": 1,
            "last_jpeg_size": len(msg.data),
        }
        # JPEG round-trip なので値は完全一致しないが shape は保持
        assert result.rgb.shape == (8, 8, 3)

    def test_cb_overwrites_with_latest(self):
        """後続 callback で latest が上書きされる (latest-only policy)。"""
        source = _make_ros2_source_no_init()
        rgb_a = np.full((4, 4, 3), 50, dtype=np.uint8)
        rgb_b = np.full((4, 4, 3), 200, dtype=np.uint8)

        source._cb(_fake_compressed_image_msg(rgb_a, 1, 0))
        source._cb(_fake_compressed_image_msg(rgb_b, 3, 0))

        result = source.get()
        assert result is not None
        assert result.t == 3_000_000_000  # 後の msg
        # 値の平均で B の方に近いことを確認 (JPEG 圧縮誤差込み)
        assert result.rgb.mean() > 100  # 200 に近い側

    @pytest.mark.parametrize(
        ("view", "expected_mean_check"),
        [
            ("left", lambda mean: mean < 100),
            ("right", lambda mean: mean > 100),
        ],
    )
    def test_cb_splits_packed_stereo(self, view, expected_mean_check):
        source = _make_ros2_source_no_init()
        source._stereo_view = view  # type: ignore[attr-defined]
        left = np.full((8, 8, 3), 20, dtype=np.uint8)
        right = np.full((8, 8, 3), 220, dtype=np.uint8)
        packed = np.concatenate([left, right], axis=1)

        source._cb(_fake_compressed_image_msg(packed, 1, 0))

        result = source.get()
        assert result is not None
        assert result.rgb.shape == (8, 8, 3)
        assert expected_mean_check(float(result.rgb.mean()))

    def test_cb_broken_jpeg_drops_frame_and_keeps_previous(self):
        """壊れ JPEG (imdecode → None) は drop、latest は前回維持。"""
        source = _make_ros2_source_no_init()
        # 1 回目: 正常 frame で latest 更新
        rgb_ok = np.full((4, 4, 3), 100, dtype=np.uint8)
        source._cb(_fake_compressed_image_msg(rgb_ok, 1, 0))
        prev = source.get()
        assert prev is not None

        # 2 回目: 壊れ JPEG (適当な bytes)、latest は前回維持
        stamp = types.SimpleNamespace(sec=2, nanosec=0)
        header = types.SimpleNamespace(stamp=stamp)
        broken_msg = types.SimpleNamespace(
            data=b"\x00\x01\x02not_a_jpeg", header=header
        )
        source._cb(broken_msg)  # 例外なく return

        after = source.get()
        assert after is not None
        assert after.t == prev.t  # 1 回目の t を維持

    def test_cb_broad_exception_does_not_propagate(self):
        """msg 構造 mismatch 等の例外を _cb 内で catch (rclpy executor thread 保護)。"""
        source = _make_ros2_source_no_init()
        # data 属性が無い msg → np.frombuffer の中で AttributeError が起きる
        bad_msg = types.SimpleNamespace()
        source._cb(bad_msg)  # 例外が伝播しないことを確認 (raise すれば test fail)
        # latest は None のまま (最初から)
        assert source.get() is None


class TestAdaptWristToTrainingShape:
    def test_640_passthrough(self):
        frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        result = adapt_wrist_to_training_shape(frame)
        assert result.shape == (480, 640, 3)
        assert np.array_equal(result, frame)

    def test_848_center_crops_to_640(self):
        frame = np.arange(480 * 848 * 3, dtype=np.uint8).reshape(480, 848, 3)
        result = adapt_wrist_to_training_shape(frame)
        assert result.shape == (480, 640, 3)
        # center 640 = columns 104..744
        assert np.array_equal(result, frame[:, 104:744, :])

    def test_848_result_is_contiguous(self):
        frame = np.random.randint(0, 255, size=(480, 848, 3), dtype=np.uint8)
        result = adapt_wrist_to_training_shape(frame)
        assert result.flags["C_CONTIGUOUS"]

    def test_unexpected_shape_raises(self):
        for bad_shape in [(240, 320, 3), (480, 1280, 3), (480, 640, 1), (480, 848, 4)]:
            frame = np.zeros(bad_shape, dtype=np.uint8)
            with pytest.raises(ValueError, match="wrist frame must be"):
                adapt_wrist_to_training_shape(frame)


# =========================================================================
# ZmqFrameSource (zmq / msgpack / cv2 install 不要、__init__ を bypass)
# =========================================================================
class _FakeMsgpack:
    """msgpack.unpackb の最小 fake。fixmap header 1 byte + repr で round-trip する。

    default env に msgpack が無くても `_unpack` の分岐を test できるようにする。
    header byte を持たせてあるのは、`_unpack` が topic prefix 付き配信に備えて
    行う marker 探索をそのまま通すため。
    """

    _HEADER = b"\x82"

    @classmethod
    def packb(cls, obj) -> bytes:
        return cls._HEADER + repr(obj).encode()

    @classmethod
    def unpackb(cls, buf: bytes, raw: bool = False):
        if not buf.startswith(cls._HEADER):
            raise ValueError("not a fake msgpack payload")
        return ast.literal_eval(buf[len(cls._HEADER) :].decode())


_FAKE_MSGPACK = _FakeMsgpack()


def _fake_jpeg(h: int, w: int, fill: int) -> bytes:
    """_FakeCv2 が decode できる「JPEG 相当」のバイト列。"""
    ok, arr = _FAKE_CV2.imencode(".jpg", np.full((h, w, 3), fill, dtype=np.uint8))
    assert ok
    return arr.tobytes()


def _make_zmq_source_no_init(stereo_view: str = "packed", image_key: str = "ego_view"):
    """__init__ を bypass し、socket / thread を持たない instance を返す。

    `Ros2FrameSource` 側と同じ方針で、zmq socket の生成と受信 thread の起動を
    skip して `_build_frame` / `_unpack` / `get` / `close` の behavior だけ見る。
    """
    from inference.desktop.perception.frame_source import ZmqFrameSource

    source = ZmqFrameSource.__new__(ZmqFrameSource)
    source._cv2 = _FAKE_CV2  # type: ignore[attr-defined]
    source._msgpack = _FAKE_MSGPACK  # type: ignore[attr-defined]
    source._zmq = types.SimpleNamespace(Again=RuntimeError)  # type: ignore[attr-defined]
    source._stereo_view = stereo_view  # type: ignore[attr-defined]
    source._image_key = image_key  # type: ignore[attr-defined]
    source._latest = None  # type: ignore[attr-defined]
    source._lock = threading.Lock()  # type: ignore[attr-defined]
    source._closed = False  # type: ignore[attr-defined]
    source._received_count = 0  # type: ignore[attr-defined]
    source._decoded_count = 0  # type: ignore[attr-defined]
    source._last_jpeg_size = 0  # type: ignore[attr-defined]
    source._layout = "(未受信)"  # type: ignore[attr-defined]
    source._warned_fallback = False  # type: ignore[attr-defined]
    source._key_payload = {}  # type: ignore[attr-defined]
    source._key_changed_monotonic_ns = {}  # type: ignore[attr-defined]
    source._sender_clock_samples = []  # type: ignore[attr-defined]
    source._sender_clock_offset = None  # type: ignore[attr-defined]
    source._sender_clock_logged = False  # type: ignore[attr-defined]
    source._thread = None  # type: ignore[attr-defined]
    source._sock = MagicMock()  # type: ignore[attr-defined]
    return source


class TestZmqSenderClockOffset:
    """運営 bridge (PC2) の時計との差の推定 (Issue #161、C2-01)。"""

    KEYS = ("ego_view_left", "ego_view_right")

    def _msg(self, sender_s: float) -> dict:
        return {"timestamps": {key: sender_s for key in self.KEYS}}

    def test_offset_is_the_sample_with_the_smallest_delivery_delay(self, capsys):
        source = _make_zmq_source_no_init()
        assert source.sender_clock_offset_s() is None
        # PC2 の時計が 5 s 進んでいる。届くまでの遅れ 30 ms と 5 ms の 2 sample。
        source._note_sender_clock(
            self._msg(1000.0 - 0.030 + 5.0), self.KEYS, 1000.0, 1_000_000_000
        )
        source._note_sender_clock(
            self._msg(1001.0 - 0.005 + 5.0), self.KEYS, 1001.0, 2_000_000_000
        )
        assert source.sender_clock_offset_s() == pytest.approx(4.995)
        err = capsys.readouterr().err
        assert err.count("sender clock offset") == 1

    def test_samples_older_than_the_window_are_forgotten(self):
        source = _make_zmq_source_no_init()
        source._note_sender_clock(self._msg(1010.0), self.KEYS, 1000.0, 0)
        # 窓 (5 s) より後の sample だけが残る = 時計が合わせ直されたら追従する。
        source._note_sender_clock(self._msg(1011.0), self.KEYS, 1010.0, 10_000_000_000)
        assert source.sender_clock_offset_s() == pytest.approx(1.0)

    def test_the_last_estimate_survives_a_camera_gap(self):
        """カメラが窓より長く途切れても、最後の推定を返す (Thor の時計に戻さない)。"""
        source = _make_zmq_source_no_init()
        source._note_sender_clock(self._msg(1003.0), self.KEYS, 1000.0, 0)
        # 次の frame は 20 s 後。途切れの間も推定は残る
        assert source.sender_clock_offset_s() == pytest.approx(3.0)
        source._sender_clock_samples.clear()  # 窓の外に出た状態
        assert source.sender_clock_offset_s() == pytest.approx(3.0)
        source._note_sender_clock(self._msg(1022.5), self.KEYS, 1020.0, 20_000_000_000)
        assert source.sender_clock_offset_s() == pytest.approx(2.5)

    def test_messages_without_the_used_timestamps_are_not_samples(self):
        source = _make_zmq_source_no_init()
        source._note_sender_clock({}, self.KEYS, 1000.0, 0)
        source._note_sender_clock(
            {"timestamps": {"left_wrist": 2000.0}}, self.KEYS, 1000.0, 1
        )
        assert source.sender_clock_offset_s() is None

    def test_nanosecond_timestamps_are_converted(self):
        source = _make_zmq_source_no_init()
        source._note_sender_clock(self._msg(1003.0 * 1e9), self.KEYS, 1000.0, 0)
        assert source.sender_clock_offset_s() == pytest.approx(3.0)


class TestZmqFrameSource:
    def test_invalid_stereo_view_rejected_before_any_import(self):
        """stereo_view の検証は zmq/cv2 の import より前 = 未 install 環境でも効く。"""
        from inference.desktop.perception.frame_source import ZmqFrameSource

        with pytest.raises(ValueError, match="stereo_view must be one of"):
            ZmqFrameSource("tcp://127.0.0.1:5555", stereo_view="bogus")

    def test_packed_hconcats_left_and_right(self):
        """会場 bridge の ego_view_left/right を左右連結して packed に揃える。"""
        source = _make_zmq_source_no_init(stereo_view="packed")
        images = {
            "ego_view": _fake_jpeg(4, 6, 1),
            "ego_view_left": _fake_jpeg(4, 6, 10),
            "ego_view_right": _fake_jpeg(4, 6, 20),
        }
        frame, layout, size = source._build_frame(images)

        assert frame.shape == (4, 12, 3)  # 幅が 2 倍 = hconcat
        assert np.all(frame[:, :6, :] == 10)  # 左が left
        assert np.all(frame[:, 6:, :] == 20)  # 右が right
        assert layout == "hconcat(ego_view_left, ego_view_right)"
        assert size == len(images["ego_view_left"]) + len(images["ego_view_right"])

    def test_single_uses_image_key_only(self):
        """手首カメラは単一キーをそのまま返す (連結しない)。"""
        source = _make_zmq_source_no_init(stereo_view="single", image_key="left_wrist")
        images = {
            "ego_view": _fake_jpeg(4, 6, 1),
            "left_wrist": _fake_jpeg(4, 6, 7),
        }
        frame, layout, _ = source._build_frame(images)

        assert frame.shape == (4, 6, 3)
        assert np.all(frame == 7)
        assert layout == "left_wrist"

    def test_single_missing_key_returns_none(self):
        source = _make_zmq_source_no_init(stereo_view="single", image_key="right_wrist")
        frame, layout, size = source._build_frame({"ego_view": _fake_jpeg(4, 6, 1)})
        assert frame is None and layout == "" and size == 0

    def test_packed_without_stereo_keys_duplicates_mono_and_warns_once(self, capsys):
        """mono 配信でも packed の 2W 契約を維持し、左右へ同じ像を置く。"""
        source = _make_zmq_source_no_init(stereo_view="packed")
        images = {"ego_view": _fake_jpeg(4, 6, 3)}

        frame, layout, _ = source._build_frame(images)
        assert frame.shape == (4, 12, 3)
        assert np.array_equal(frame[:, :6], frame[:, 6:])
        assert "単眼複製 fallback" in layout
        first = capsys.readouterr().err
        assert "ego_view_left/right" in first

        # 30Hz で出し続けないこと
        source._build_frame(images)
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize("stereo_view", ["packed", "left", "right"])
    def test_missing_ego_view_never_substitutes_another_camera(self, stereo_view):
        """ego_view が無い配信で手首カメラを head として使わない (frame を捨てる)。"""
        source = _make_zmq_source_no_init(stereo_view=stereo_view)
        images = {"left_wrist": _fake_jpeg(4, 6, 7), "right_wrist": _fake_jpeg(4, 6, 8)}

        frame, layout, size = source._build_frame(images)

        assert frame is None and layout == "" and size == 0
        assert source._used_image_keys(images) == ()

    def test_one_frozen_eye_keeps_the_pair_stale(self, monkeypatch):
        """Codex #11: 右眼だけ凍結なら、左が変わっても packed は古いまま。"""
        from inference.desktop.perception import frame_source as fs

        clock = iter([1_000, 2_000, 3_000])
        monkeypatch.setattr(fs.time, "monotonic_ns", lambda: next(clock))
        source = _make_zmq_source_no_init(stereo_view="packed")
        keys = ("ego_view_left", "ego_view_right")
        frozen = _fake_jpeg(4, 6, 20)
        assert source._content_changed_ns(
            {"ego_view_left": _fake_jpeg(4, 6, 1), "ego_view_right": frozen}, keys
        ) == 1_000
        assert source._content_changed_ns(
            {"ego_view_left": _fake_jpeg(4, 6, 2), "ego_view_right": frozen}, keys
        ) == 1_000
        assert source._content_changed_ns(
            {"ego_view_left": _fake_jpeg(4, 6, 3), "ego_view_right": _fake_jpeg(4, 6, 21)},
            keys,
        ) == 3_000

    def test_broken_jpeg_in_stereo_pair_drops_frame(self):
        """片眼が decode 不能なら frame を捨てる (半端な連結をしない)。"""
        source = _make_zmq_source_no_init(stereo_view="packed")
        images = {
            "ego_view_left": _fake_jpeg(4, 6, 10),
            "ego_view_right": b"not-an-image",
        }
        frame, layout, size = source._build_frame(images)
        assert frame is None and layout == "" and size == 0

    def test_unpack_plain_payload(self):
        source = _make_zmq_source_no_init()
        payload = _FAKE_MSGPACK.packb({"images": {"ego_view": b"x"}})
        assert source._unpack(payload) == {"images": {"ego_view": b"x"}}

    def test_unpack_tolerates_topic_prefix(self):
        """state 側 (:5557) のような prefix 付き配信でも読めること。"""
        source = _make_zmq_source_no_init()
        payload = b"topic" + _FAKE_MSGPACK.packb({"images": {}})
        assert source._unpack(payload) == {"images": {}}

    def test_unpack_returns_none_for_garbage(self):
        source = _make_zmq_source_no_init()
        assert source._unpack(b"\x00\x01\x02") is None

    @pytest.mark.parametrize(
        "timestamps, expected",
        [
            ({"ego_view": 1.5}, 1_500_000_000),  # 秒 → ns
            (
                {"ego_view": 1_700_000_000_000_000_000},
                1_700_000_000_000_000_000,
            ),  # ns はそのまま
        ],
    )
    def test_stamp_ns_normalizes_units(self, timestamps, expected):
        from inference.desktop.perception.frame_source import ZmqFrameSource

        assert ZmqFrameSource._stamp_ns(
            {"timestamps": timestamps}, ("ego_view",)
        ) == expected

    def test_stamp_ns_uses_only_selected_camera_and_older_stereo_eye(self):
        from inference.desktop.perception.frame_source import ZmqFrameSource

        msg = {
            "timestamps": {
                "unrelated": 99.0,
                "ego_view_left": 3.0,
                "ego_view_right": 2.0,
            }
        }
        assert ZmqFrameSource._stamp_ns(
            msg, ("ego_view_left", "ego_view_right")
        ) == 2_000_000_000

    def test_stamp_ns_does_not_borrow_unrelated_camera_timestamp(self):
        from inference.desktop.perception.frame_source import ZmqFrameSource

        before = time.time_ns()
        got = ZmqFrameSource._stamp_ns(
            {"timestamps": {"left_wrist": 1.0}}, ("right_wrist",)
        )
        assert before <= got <= time.time_ns()

    def test_stamp_ns_without_timestamps_uses_wall_clock(self):
        from inference.desktop.perception.frame_source import ZmqFrameSource

        before = time.time_ns()
        got = ZmqFrameSource._stamp_ns({})
        assert before <= got <= time.time_ns()

    def test_get_returns_none_before_any_frame(self):
        assert _make_zmq_source_no_init().get() is None

    def test_close_is_idempotent_and_closes_socket(self):
        source = _make_zmq_source_no_init()
        source.close()
        source.close()  # 2 回目は no-op
        assert source._sock.close.call_count == 1
        assert source.get() is None  # close 後は frame を返さない

    def test_get_returns_none_after_close_even_with_latest(self):
        """close 済みの source は、保持していた frame を返さない。"""
        source = _make_zmq_source_no_init()
        source._latest = _make_frame(t=1)
        assert source.get() is not None
        source.close()
        assert source.get() is None
