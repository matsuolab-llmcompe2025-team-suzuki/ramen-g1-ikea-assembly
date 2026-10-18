"""VLM クライアントと 1→2 境界検出の unit test (Issue #136)。

実 VLM は呼ばない (Thor 前提のため)。`_session` / `vlm_client` を差し替えて
戻り値を制御する。
"""

from __future__ import annotations

import pytest

from inference.desktop.pick_leg_hybrid.boundary import (
    BoundaryConfig,
    GraspBoundaryDetector,
)
from inference.desktop.pick_leg_hybrid.vlm import (
    DEFAULT_SYSTEM_PROMPT,
    VlmAnswer,
    VlmBoundaryClient,
    VlmConfig,
    build_messages,
    parse_answer,
)

OPEN_STATE, OPEN_CMD = [4.49, 4.49], [4.50, 4.50]
HOLD_STATE, HOLD_CMD = [2.19, 2.24], [1.84, 1.89]


class _FakeResponse:
    def __init__(self, content: str, status: int = 200) -> None:
        self._content = content
        self._status = status

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}

    @property
    def text(self) -> str:
        return self._content


class _FakeSession:
    """post 呼び出しを記録し、決められた応答を順に返す。"""

    def __init__(self, contents, status: int = 200, raises: bool = False) -> None:
        self._contents = list(contents)
        self._status = status
        self._raises = raises
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None):  # noqa: A002
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self._raises:
            raise ConnectionError("boom")
        content = self._contents.pop(0) if self._contents else self._contents_last
        self._contents_last = content
        return _FakeResponse(content, self._status)

    _contents_last = "{\"answer\": 0}"


class _StubVlm:
    """常に同じ答えを返す VLM。呼び出し回数を数える。"""

    def __init__(self, values) -> None:
        self._values = list(values)
        self.calls = 0

    def ask(self, question, images, *, current_phase_text=""):  # noqa: D102
        self.calls += 1
        v = self._values.pop(0) if self._values else self._last
        self._last = v
        return VlmAnswer(value=v)

    _last = 0


class TestParseAnswer:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ('{"answer": 1}', 1),
            ('{"answer": 0}', 0),
            ('{"answer": true}', 1),
            ('{"answer": "1"}', 1),
            ('  {"answer":1}  ', 1),
            ('はい、判定します。{"answer": 1} 以上です。', 1),
            ("```json\n{\"answer\": 0}\n```", 0),
            ("1", 1),
            ("0", 0),
        ],
    )
    def test_extracts_the_bit(self, text: str, expected: int) -> None:
        assert parse_answer(text) == expected

    @pytest.mark.parametrize(
        "text", ["", "yes", '{"answer": 2}', '{"answer": null}', "{broken", '{"a": 1}']
    )
    def test_returns_none_when_unparseable(self, text: str) -> None:
        """解釈できない出力は None。呼び出し側が「まだ」として扱う。"""
        assert parse_answer(text) is None


class TestBuildMessages:
    def test_images_are_explicitly_labeled_before_the_question(self) -> None:
        msgs = build_messages("Q?", ["aaa", "bbb"])
        content = msgs[1]["content"]
        assert [c["type"] for c in content] == [
            "text", "image_url", "text", "image_url", "text"
        ]
        assert content[0]["text"].startswith("参考画像 A")
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,aaa")
        assert content[2]["text"].startswith("参考画像 B")

    def test_last_non_reference_image_is_labeled_current(self) -> None:
        msgs = build_messages("Q?", ["ref-a", "ref-b", "old", "now"])
        content = msgs[1]["content"]
        assert content[-3]["text"] == "現在画像（判定対象）"
        assert content[-2]["image_url"]["url"].endswith("now")

    def test_system_prompt_biases_toward_not_yet(self) -> None:
        """偽陽性は回復不能なので「迷ったら 0」を明示する (D-5)。"""
        assert "0 を返して" in DEFAULT_SYSTEM_PROMPT
        assert "まだ起きていない" in DEFAULT_SYSTEM_PROMPT

    def test_current_phase_is_prepended_as_context(self) -> None:
        msgs = build_messages("Q?", [], current_phase_text="脚を掴む")
        assert "現在の区間: 脚を掴む" in msgs[1]["content"][-1]["text"]


class TestVlmBoundaryClient:
    def test_parses_a_successful_response(self) -> None:
        client = VlmBoundaryClient(VlmConfig(), _session=_FakeSession(['{"answer": 1}']))
        assert client.ask("Q?", ["img"]).is_yes is True

    def test_network_failure_becomes_none_not_an_exception(self) -> None:
        """VLM の障害で制御ループを止めない。None = 「まだ」扱い。"""
        client = VlmBoundaryClient(VlmConfig(), _session=_FakeSession([], raises=True))
        ans = client.ask("Q?", ["img"])
        assert ans.value is None
        assert ans.is_yes is False
        assert "ConnectionError" in ans.error

    def test_http_error_becomes_none(self) -> None:
        client = VlmBoundaryClient(
            VlmConfig(), _session=_FakeSession(['{"answer": 1}'], status=500)
        )
        answer = VlmBoundaryClient(
            VlmConfig(), _session=_FakeSession(['{"answer": 1}'], status=500)
        ).ask("Q?", [])
        assert answer.value is None
        assert "VLM HTTP request failed" in answer.error

    def test_sends_temperature_zero(self) -> None:
        session = _FakeSession(['{"answer": 0}'])
        VlmBoundaryClient(VlmConfig(), _session=session).ask("Q?", ["img"])
        assert session.calls[0]["json"]["temperature"] == 0.0

    def test_config_rejects_bad_values(self) -> None:
        with pytest.raises(ValueError, match="timeout_sec"):
            VlmConfig(timeout_sec=0.0)
        with pytest.raises(ValueError, match="jpeg_quality"):
            VlmConfig(jpeg_quality=0)


class TestGraspBoundaryDetector:
    def _detector(self, values, **kw):
        cfg = BoundaryConfig(**{"confirm_count": 2, "min_interval_sec": 1.0, **kw})
        return GraspBoundaryDetector(cfg=cfg, vlm_client=_StubVlm(values))

    def test_does_not_call_the_vlm_while_the_interlock_is_open(self) -> None:
        """明らかに掴んでいないときは VLM を呼ばない (Thor の資源節約)。"""
        det = self._detector([1, 1])
        out = det.update(t=0.0, hand_state=OPEN_STATE, hand_cmd=OPEN_CMD, images_b64=[])
        assert out.interlock is False
        assert out.vlm_called is False
        assert out.fire is False
        assert det.vlm.calls == 0

    def test_fires_only_after_two_consecutive_yes(self) -> None:
        det = self._detector([1, 1])
        first = det.update(t=0.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[])
        assert first.vlm_called is True and first.hits == 1 and first.fire is False
        second = det.update(t=1.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[])
        assert second.hits == 2 and second.fire is True

    def test_a_single_stray_yes_does_not_fire(self) -> None:
        """まぐれの 1 回で進まない。後戻りできないため。"""
        det = self._detector([1, 0, 1])
        assert det.update(t=0.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[]).hits == 1
        assert det.update(t=1.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[]).hits == 0
        assert det.update(t=2.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[]).fire is False

    def test_interlock_dropping_resets_the_count(self) -> None:
        det = self._detector([1, 1])
        det.update(t=0.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[])
        det.update(t=1.0, hand_state=OPEN_STATE, hand_cmd=OPEN_CMD, images_b64=[])
        assert det.hits == 0

    def test_confirm_count_one_fires_immediately(self) -> None:
        det = self._detector([1], confirm_count=1)
        assert det.update(t=0.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[]).fire is True

    def test_polling_is_rate_limited(self) -> None:
        """30Hz で回っても VLM は min_interval_sec ごとにしか呼ばない。"""
        det = self._detector([1, 1], confirm_count=1)
        det.update(t=0.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[])
        for i in range(1, 30):
            out = det.update(
                t=i / 30.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[]
            )
            assert out.vlm_called is False
        assert det.vlm.calls == 1

    def test_undecidable_answer_counts_as_not_yet(self) -> None:
        det = self._detector([None, None], confirm_count=1)
        out = det.update(t=0.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[])
        assert out.vlm_value is None
        assert out.fire is False

    def test_references_are_prepended_to_the_images(self) -> None:
        class _Capture(_StubVlm):
            def ask(self, question, images, *, current_phase_text=""):
                self.seen = list(images)
                return super().ask(question, images, current_phase_text=current_phase_text)

        vlm = _Capture([1])
        det = GraspBoundaryDetector(
            cfg=BoundaryConfig(confirm_count=1),
            vlm_client=vlm,
            reference_images_b64=["REF_A", "REF_B"],
        )
        det.update(t=0.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=["PAST", "NOW"])
        assert vlm.seen == ["REF_A", "REF_B", "PAST", "NOW"]

    def test_reset_clears_state(self) -> None:
        det = self._detector([1, 1])
        det.update(t=0.0, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[])
        det.reset()
        assert det.hits == 0
        out = det.update(t=0.5, hand_state=HOLD_STATE, hand_cmd=HOLD_CMD, images_b64=[])
        assert out.vlm_called is True   # 間引き履歴も消えている

    def test_config_rejects_bad_values(self) -> None:
        with pytest.raises(ValueError, match="confirm_count"):
            BoundaryConfig(confirm_count=0)
