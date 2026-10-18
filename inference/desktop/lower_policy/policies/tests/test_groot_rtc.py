"""GR00T RTC prefix builder の test (Issue #137 Phase B)。

要は **decode の逆になっているか**。実機で確かめようがない部分なので、
`GrootN17ActionDecodeStep` を実際に構築して往復させることで担保する。
重みは load せず decode step だけ組むので GPU 不要 (数十 ms で回る)。

stats は実 ckpt の両形式に合わせた合成値を使う (group 構成
9/9/7/7/7/7/3/1/3、rep フラグ、1-D / per-horizon 2-D stats)。往復の正しさに
実 stat 値は不要なため、ckpt 由来データを
repo に持ち込まずに済む。実 ckpt が HF キャッシュに在るときだけ走る構造 assert を
別途置き、合成 fixture が実物とずれたら気付けるようにしている。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot.policies.groot.processor_groot import (  # noqa: E402
    GrootN17ActionDecodeStep,
)
from lerobot.policies.groot.utils import relative_eef_to_absolute  # noqa: E402
from lerobot.processor import create_transition  # noqa: E402
from lerobot.processor import TransitionKey  # noqa: E402

from inference.desktop.lower_policy.policies.groot_rtc import (  # noqa: E402
    absolute_eef_to_relative,
    encode_absolute_chunk_to_model_space,
    normalize_min_max,
)


# 実 ckpt (Team-RAMEN rotate/insert 系) で実測した action group 構成。
# (key, dim, rep, type, format)
_GROUPS = [
    ("left_wrist_eef_9d", 9, "RELATIVE", "EEF", "XYZ_ROT6D"),
    ("right_wrist_eef_9d", 9, "RELATIVE", "EEF", "XYZ_ROT6D"),
    ("left_hand", 7, "ABSOLUTE", "NON_EEF", "DEFAULT"),
    ("right_hand", 7, "ABSOLUTE", "NON_EEF", "DEFAULT"),
    ("left_arm", 7, "RELATIVE", "NON_EEF", "DEFAULT"),
    ("right_arm", 7, "RELATIVE", "NON_EEF", "DEFAULT"),
    ("waist", 3, "ABSOLUTE", "NON_EEF", "DEFAULT"),
    ("base_height_command", 1, "ABSOLUTE", "NON_EEF", "DEFAULT"),
    ("navigate_command", 3, "ABSOLUTE", "NON_EEF", "DEFAULT"),
]
_ENV_ACTION_DIM = sum(dim for _, dim, *_ in _GROUPS)  # 53

# 単位回転の rot6d (回転行列の先頭 2 行)。reference pose に使う。
_IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class _FakePackStep:
    """`get_cached_raw_state()` だけを持つ decode step 用のスタブ。"""

    def __init__(self, raw_state: dict[str, np.ndarray]) -> None:
        self._raw_state = raw_state

    def get_cached_raw_state(self) -> dict[str, np.ndarray]:
        return self._raw_state


def _make_raw_state(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    for key, dim, _rep, action_type, _fmt in _GROUPS:
        if action_type == "EEF":
            # xyz は任意、rot6d は正規直交でないと reference として意味を持たない
            state[key] = np.array(
                [[*rng.normal(size=3), *_IDENTITY_ROT6D]], dtype=np.float32
            )
        else:
            state[key] = rng.normal(size=(1, dim)).astype(np.float32)
    return state


def _make_decode_step(*, use_percentiles: bool = True) -> GrootN17ActionDecodeStep:
    rng = np.random.default_rng(12345)
    action_stats: dict[str, dict] = {}
    relative_stats: dict[str, dict] = {}
    for key, dim, rep, _type, _fmt in _GROUPS:
        low = rng.uniform(-2.0, -0.5, size=dim)
        high = low + rng.uniform(0.5, 3.0, size=dim)
        action_stats[key] = {
            "min": low.tolist(),
            "max": high.tolist(),
            "q01": (low + 0.05).tolist(),
            "q99": (high - 0.05).tolist(),
        }
        if rep == "RELATIVE":
            rel_low = rng.uniform(-1.0, -0.2, size=dim)
            rel_high = rel_low + rng.uniform(0.4, 2.0, size=dim)
            if _type == "EEF":
                # rot6d 成分は回転行列の要素なので必ず [-1, 1] に収まる。実 ckpt の
                # relative_action stats は (既に正規直交な) 相対 pose から計算される
                # のでこの範囲を覆う。合成値でここを狭めると、decode の
                # Gram-Schmidt 正規直交化で範囲外に出た値が正規化時に飽和し、
                # 往復が壊れる (encoder ではなく fixture 側の非現実性)。
                rel_low[3:] = -1.05
                rel_high[3:] = 1.05
            relative_stats[key] = {
                "min": rel_low.tolist(),
                "max": rel_high.tolist(),
            }
    modality_config = {
        "action": {
            "modality_keys": [key for key, *_ in _GROUPS],
            "action_configs": [
                {"rep": rep, "type": action_type, "format": fmt, "state_key": key}
                for key, _dim, rep, action_type, fmt in _GROUPS
            ],
            "delta_indices": list(range(40)),
        }
    }
    return GrootN17ActionDecodeStep(
        env_action_dim=_ENV_ACTION_DIM,
        raw_stats={"action": action_stats, "relative_action": relative_stats},
        modality_config=modality_config,
        use_percentiles=use_percentiles,
        use_relative_action=True,
        pack_step=_FakePackStep(_make_raw_state()),
    )


def _decode(step: GrootN17ActionDecodeStep, normalized: np.ndarray) -> np.ndarray:
    """(rows, 53) 正規化値 → (rows, 53) 絶対値。"""
    tensor = torch.from_numpy(np.asarray(normalized, dtype=np.float32))[None]
    out = step(create_transition(action=tensor))[TransitionKey.ACTION]
    return out[0].detach().cpu().numpy()


class TestAbsoluteEefToRelative:
    def test_is_inverse_of_relative_eef_to_absolute(self) -> None:
        rng = np.random.default_rng(7)
        reference = np.array([[*rng.normal(size=3), *_IDENTITY_ROT6D]])
        # 相対側も正規直交な rot6d でないと、復元時の Gram-Schmidt 射影で
        # 元の値には戻らない (射影後の値には戻る)。
        relative = np.array(
            [[[*rng.normal(size=3), *_IDENTITY_ROT6D] for _ in range(4)]]
        )
        absolute = relative_eef_to_absolute(relative, reference)
        recovered = absolute_eef_to_relative(absolute, reference)
        np.testing.assert_allclose(recovered, relative, rtol=1e-5, atol=1e-5)

    def test_round_trip_with_rotated_reference(self) -> None:
        # reference が単位回転でない場合も逆変換が成立すること
        angle = 0.7
        rot = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
            ]
        )
        reference = np.array([[0.1, -0.2, 0.3, *rot.reshape(-1)]])
        relative = np.array([[[0.05, 0.0, -0.02, *_IDENTITY_ROT6D]]])
        absolute = relative_eef_to_absolute(relative, reference)
        recovered = absolute_eef_to_relative(absolute, reference)
        np.testing.assert_allclose(recovered, relative, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        ("action_shape", "ref_shape"),
        [((4, 9), (1, 9)), ((1, 4, 9), (9,)), ((1, 4, 6), (1, 9))],
    )
    def test_invalid_shapes_rejected(
        self, action_shape: tuple, ref_shape: tuple
    ) -> None:
        with pytest.raises(ValueError):
            absolute_eef_to_relative(
                np.zeros(action_shape), np.zeros(ref_shape)
            )


class TestNormalizeMinMax:
    def test_inverse_of_unnormalize(self) -> None:
        from lerobot.policies.groot.processor_groot import _unnormalize_min_max

        min_v = np.array([-1.0, 0.0, 2.0])
        max_v = np.array([1.0, 4.0, 3.0])
        normalized = np.array([[-1.0, 0.0, 0.5]])
        absolute = _unnormalize_min_max(normalized, min_v, max_v)
        np.testing.assert_allclose(
            normalize_min_max(absolute, min_v, max_v), normalized, atol=1e-6
        )

    def test_clips_out_of_range(self) -> None:
        min_v = np.array([0.0])
        max_v = np.array([1.0])
        assert normalize_min_max(np.array([[5.0]]), min_v, max_v)[0, 0] == 1.0
        assert normalize_min_max(np.array([[-5.0]]), min_v, max_v)[0, 0] == -1.0

    def test_degenerate_span_does_not_divide_by_zero(self) -> None:
        # min == max の定数次元。decode 側も常に min を返すので情報は元から無い。
        out = normalize_min_max(np.array([[3.0]]), np.array([2.0]), np.array([2.0]))
        assert np.isfinite(out).all()
        assert out[0, 0] == 0.0


class TestEncodeAbsoluteChunk:
    def test_decode_encode_decode_is_stable(self) -> None:
        """実運用と同じ向きの往復: 絶対 leftover → prefix → 絶対。

        我々が保持しているのは絶対値なので、それを encode した prefix が
        同じ絶対値に decode されることが要件。rot6d は decode 出力の時点で
        正規直交なので、この向きなら厳密に一致する。
        """
        step = _make_decode_step()
        rng = np.random.default_rng(3)
        normalized = rng.uniform(-0.9, 0.9, size=(8, _ENV_ACTION_DIM))
        absolute = _decode(step, normalized)

        prefix = encode_absolute_chunk_to_model_space(
            absolute,
            reference_raw_state=step.pack_step.get_cached_raw_state(),
            decode_step=step,
        )
        assert prefix.shape == (8, _ENV_ACTION_DIM)
        assert prefix.dtype == np.float32
        assert np.all(np.abs(prefix) <= 1.0 + 1e-6)

        np.testing.assert_allclose(_decode(step, prefix), absolute, rtol=1e-4, atol=1e-4)

    def test_non_eef_groups_round_trip_exactly(self) -> None:
        """EEF 以外は decode(y)→encode で y そのものに戻ること。

        EEF は rot6d の Gram-Schmidt 射影が入るため raw の y には戻らない
        (射影後の値には戻る = 上の test でカバー)。
        """
        step = _make_decode_step()
        rng = np.random.default_rng(11)
        normalized = rng.uniform(-0.9, 0.9, size=(5, _ENV_ACTION_DIM))
        absolute = _decode(step, normalized)
        recovered = encode_absolute_chunk_to_model_space(
            absolute,
            reference_raw_state=step.pack_step.get_cached_raw_state(),
            decode_step=step,
        )
        non_eef = slice(18, _ENV_ACTION_DIM)  # 先頭 2 group (9+9) が EEF
        np.testing.assert_allclose(
            recovered[:, non_eef], normalized[:, non_eef], rtol=1e-4, atol=1e-4
        )

    def test_reference_state_changes_the_prefix(self) -> None:
        """re-anchor が効いていること: 基準 state が変われば prefix も変わる。

        ここが no-op だと「絶対値をそのまま渡す」のと同じになり、RTC が
        意味を失う (本家が relative ckpt を弾いていた理由そのもの)。
        """
        step = _make_decode_step()
        rng = np.random.default_rng(5)
        absolute = _decode(step, rng.uniform(-0.5, 0.5, size=(4, _ENV_ACTION_DIM)))

        first = encode_absolute_chunk_to_model_space(
            absolute,
            reference_raw_state=step.pack_step.get_cached_raw_state(),
            decode_step=step,
        )
        second = encode_absolute_chunk_to_model_space(
            absolute,
            reference_raw_state=_make_raw_state(seed=99),
            decode_step=step,
        )
        # RELATIVE group (eef 0:18 と arm 32:46) は基準が変われば値も変わる
        assert not np.allclose(first[:, 0:18], second[:, 0:18], atol=1e-3)
        assert not np.allclose(first[:, 32:46], second[:, 32:46], atol=1e-3)
        # ABSOLUTE group (hand 18:32) は基準に依存しない
        np.testing.assert_allclose(first[:, 18:32], second[:, 18:32], atol=1e-6)

    def test_single_row_prefix(self) -> None:
        step = _make_decode_step()
        # 正規化 0 は rot6d が全ゼロ = 退化した回転になるので避ける
        absolute = _decode(step, np.full((1, _ENV_ACTION_DIM), 0.3))
        prefix = encode_absolute_chunk_to_model_space(
            absolute,
            reference_raw_state=step.pack_step.get_cached_raw_state(),
            decode_step=step,
        )
        assert prefix.shape == (1, _ENV_ACTION_DIM)

    def test_missing_reference_state_rejected(self) -> None:
        step = _make_decode_step()
        raw_state = step.pack_step.get_cached_raw_state()
        partial = {k: v for k, v in raw_state.items() if k != "left_arm"}
        absolute = _decode(step, np.full((2, _ENV_ACTION_DIM), 0.3))
        with pytest.raises(KeyError, match="left_arm"):
            encode_absolute_chunk_to_model_space(
                absolute, reference_raw_state=partial, decode_step=step
            )

    def test_action_decode_transform_rejected(self) -> None:
        step = _make_decode_step()
        step.action_decode_transform = "some_transform"
        with pytest.raises(ValueError, match="action_decode_transform"):
            encode_absolute_chunk_to_model_space(
                np.zeros((2, _ENV_ACTION_DIM)),
                reference_raw_state=step.pack_step.get_cached_raw_state(),
                decode_step=step,
            )

    def test_per_timestep_stats_round_trip_uses_new_prefix_rows(self) -> None:
        # RTC leftover is inserted at the start of the new chunk.  Match the
        # official decoder by applying per-horizon statistics from row zero.
        step = _make_decode_step()
        for modality in ("action", "relative_action"):
            for entry in step.raw_stats[modality].values():
                for name in tuple(entry):
                    base = np.asarray(entry[name], dtype=np.float32)
                    # Same span but a different center at every horizon row.
                    entry[name] = np.stack(
                        [base + row * 0.001 for row in range(40)]
                    ).tolist()
        normalized = np.random.default_rng(137).uniform(
            -0.7, 0.7, size=(8, _ENV_ACTION_DIM)
        ).astype(np.float32)
        absolute = _decode(step, normalized)
        prefix = encode_absolute_chunk_to_model_space(
            absolute,
            reference_raw_state=step.pack_step.get_cached_raw_state(),
            decode_step=step,
        )
        np.testing.assert_allclose(
            _decode(step, prefix), absolute, rtol=1e-4, atol=1e-4
        )

    def test_per_timestep_stats_shorter_than_prefix_rejected(self) -> None:
        step = _make_decode_step()
        entry = step.raw_stats["relative_action"]["left_arm"]
        entry["min"] = [entry["min"]]
        entry["max"] = [entry["max"]]
        absolute = np.ones((2, _ENV_ACTION_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="cannot cover prefix"):
            encode_absolute_chunk_to_model_space(
                absolute,
                reference_raw_state=step.pack_step.get_cached_raw_state(),
                decode_step=step,
            )

    def test_non_finite_input_rejected(self) -> None:
        step = _make_decode_step()
        absolute = _decode(step, np.full((2, _ENV_ACTION_DIM), 0.3))
        absolute[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            encode_absolute_chunk_to_model_space(
                absolute,
                reference_raw_state=step.pack_step.get_cached_raw_state(),
                decode_step=step,
            )

    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_degenerate_rotation_rejected(self) -> None:
        # 全ゼロ rot6d は Gram-Schmidt で NaN になる。prefix として model に
        # 渡すと chunk 全体が壊れるので、silent に通さないこと。
        step = _make_decode_step()
        with pytest.raises(ValueError, match="non-finite"):
            encode_absolute_chunk_to_model_space(
                np.zeros((2, _ENV_ACTION_DIM)),
                reference_raw_state=step.pack_step.get_cached_raw_state(),
                decode_step=step,
            )

    @pytest.mark.parametrize("bad", [np.zeros(53), np.zeros((0, 53)), np.zeros((2, 4))])
    def test_invalid_chunk_rejected(self, bad: np.ndarray) -> None:
        step = _make_decode_step()
        with pytest.raises(ValueError):
            encode_absolute_chunk_to_model_space(
                bad,
                reference_raw_state=step.pack_step.get_cached_raw_state(),
                decode_step=step,
            )


class TestRealCheckpointStructure:
    """実 ckpt が HF キャッシュに在るときだけ走る構造 assert。

    合成 fixture (`_GROUPS`) が実物とずれたら気付けるようにするためのもの。
    キャッシュが無い環境では skip する (network を張らない)。
    """

    def _find_postprocessor(self) -> Path | None:
        roots = [
            Path.home() / ".cache/huggingface/hub",
            Path("/tmp/claude-1000"),
        ]
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("policy_postprocessor.json"):
                if "Team-RAMEN" in str(path):
                    return path
        return None

    def test_group_layout_matches_fixture(self) -> None:
        import json

        path = self._find_postprocessor()
        if path is None:
            pytest.skip("no Team-RAMEN policy_postprocessor.json in local HF cache")
        data = json.loads(path.read_text(encoding="utf-8"))
        steps = [
            s
            for s in data.get("steps", [])
            if s.get("registry_name") == "groot_n1_7_action_decode_v1"
        ]
        assert len(steps) == 1, "expected exactly one N1.7 action decode step"
        cfg = steps[0]["config"]
        assert cfg["use_relative_action"] is True
        assert cfg["env_action_dim"] == _ENV_ACTION_DIM

        modality = cfg["modality_config"]["action"]
        assert modality["modality_keys"] == [key for key, *_ in _GROUPS]
        for (key, dim, rep, action_type, fmt), real in zip(
            _GROUPS, modality["action_configs"], strict=True
        ):
            assert real.get("rep") == rep, key
            assert real.get("type") == action_type, key
            assert real.get("format") == fmt, key
            entry = cfg["raw_stats"]["action"][key]
            assert np.asarray(entry["min"]).shape == (dim,), key


class TestRtcPrefixMixin:
    """`_prepare_n1_7_rtc_inputs` の override。

    上流は use_relative_actions=True で早期 return して RTC を無効化する。
    我々は prefix を model space に変換済みで渡すので、その判定を通さずに
    options を返す必要がある。
    """

    def _prep(self, *, ramp_rate=None, action_horizon=16, max_action_dim=32):
        from types import SimpleNamespace

        from inference.desktop.lower_policy.policies.groot_rtc import _RtcPrefixMixin

        class _Prep(_RtcPrefixMixin):
            def __init__(self) -> None:
                self._groot_model = SimpleNamespace(
                    config=SimpleNamespace(
                        action_horizon=action_horizon,
                        max_action_dim=max_action_dim,
                        rtc_ramp_rate=6.0,
                    )
                )
                self.config = SimpleNamespace(
                    chunk_size=action_horizon,
                    max_action_dim=max_action_dim,
                    rtc_ramp_rate=ramp_rate,
                    use_relative_actions=True,
                )

        return _Prep()

    def _inputs(self, batch: int = 1) -> dict:
        return {"state": torch.zeros(batch, 49)}

    def test_none_prefix_disables_rtc(self) -> None:
        prep = self._prep()
        inputs = self._inputs()
        out, options = prep._prepare_n1_7_rtc_inputs(
            inputs, inference_delay=3, prev_chunk_left_over=None
        )
        assert options is None
        assert out is inputs

    def test_relative_checkpoint_still_gets_options(self) -> None:
        # ここが上流との差分。use_relative_actions=True でも None を返さない。
        prep = self._prep()
        _out, options = prep._prepare_n1_7_rtc_inputs(
            self._inputs(),
            inference_delay=3,
            prev_chunk_left_over=torch.zeros(8, _ENV_ACTION_DIM),
        )
        assert options is not None
        assert options["rtc_overlap_steps"] == 8
        assert options["action_horizon"] == 8
        assert options["rtc_frozen_steps"] == 3

    def test_overlap_equals_supplied_rows(self) -> None:
        # 呼出側が overlap 分だけを渡す契約なので、行数がそのまま overlap になる
        prep = self._prep()
        for rows in (1, 4, 16):
            _out, options = prep._prepare_n1_7_rtc_inputs(
                self._inputs(),
                inference_delay=0,
                prev_chunk_left_over=torch.zeros(rows, _ENV_ACTION_DIM),
            )
            assert options["rtc_overlap_steps"] == rows

    def test_frozen_steps_clamped_to_rows(self) -> None:
        prep = self._prep()
        _out, options = prep._prepare_n1_7_rtc_inputs(
            self._inputs(),
            inference_delay=99,
            prev_chunk_left_over=torch.zeros(4, _ENV_ACTION_DIM),
        )
        assert options["rtc_frozen_steps"] == 4

    def test_action_dim_padded_to_max_action_dim(self) -> None:
        prep = self._prep(max_action_dim=64)
        out, _options = prep._prepare_n1_7_rtc_inputs(
            self._inputs(),
            inference_delay=1,
            prev_chunk_left_over=torch.ones(4, _ENV_ACTION_DIM),
        )
        assert out["action"].shape == (1, 4, 64)
        # 実データはそのまま、余りはゼロ埋め
        assert torch.all(out["action"][0, :, :_ENV_ACTION_DIM] == 1.0)
        assert torch.all(out["action"][0, :, _ENV_ACTION_DIM:] == 0.0)

    def test_rows_beyond_model_horizon_truncate_the_tail(self) -> None:
        # prefix は新 chunk の先頭に載るので、溢れたら末尾を捨てる
        prep = self._prep(action_horizon=8)
        prefix = torch.arange(16, dtype=torch.float32)[:, None].repeat(
            1, _ENV_ACTION_DIM
        )
        out, options = prep._prepare_n1_7_rtc_inputs(
            self._inputs(), inference_delay=0, prev_chunk_left_over=prefix
        )
        assert options["action_horizon"] == 8
        assert out["action"][0, 0, 0] == 0.0
        assert out["action"][0, -1, 0] == 7.0

    def test_config_ramp_rate_takes_precedence(self) -> None:
        prep = self._prep(ramp_rate=2.5)
        _out, options = prep._prepare_n1_7_rtc_inputs(
            self._inputs(),
            inference_delay=1,
            prev_chunk_left_over=torch.zeros(4, _ENV_ACTION_DIM),
        )
        assert options["rtc_ramp_rate"] == 2.5

    def test_checkpoint_ramp_rate_is_the_fallback(self) -> None:
        prep = self._prep(ramp_rate=None)
        _out, options = prep._prepare_n1_7_rtc_inputs(
            self._inputs(),
            inference_delay=1,
            prev_chunk_left_over=torch.zeros(4, _ENV_ACTION_DIM),
        )
        assert options["rtc_ramp_rate"] == 6.0

    def test_batch_broadcast(self) -> None:
        prep = self._prep()
        out, _options = prep._prepare_n1_7_rtc_inputs(
            self._inputs(batch=3),
            inference_delay=1,
            prev_chunk_left_over=torch.zeros(4, _ENV_ACTION_DIM),
        )
        assert out["action"].shape[0] == 3

    def test_non_tensor_prefix_rejected(self) -> None:
        prep = self._prep()
        with pytest.raises(TypeError, match="torch.Tensor"):
            prep._prepare_n1_7_rtc_inputs(
                self._inputs(),
                inference_delay=1,
                prev_chunk_left_over=np.zeros((4, _ENV_ACTION_DIM)),
            )

    def test_missing_state_rejected(self) -> None:
        prep = self._prep()
        with pytest.raises(ValueError, match="state"):
            prep._prepare_n1_7_rtc_inputs(
                {},
                inference_delay=1,
                prev_chunk_left_over=torch.zeros(4, _ENV_ACTION_DIM),
            )


class TestBuildRtcPrefixEndToEnd:
    """Gr00tPolicy._build_rtc_prefix を実 decode step 込みで通す。

    最重要は「prefix を decode すると保持していた絶対 leftover に戻る」こと。
    ここが崩れると model には別の軌道が拘束として渡ることになる。
    """

    def _policy(self, **rtc_kwargs):
        from types import SimpleNamespace

        from inference.desktop.lower_policy.policies.base import PolicyConfig
        from inference.desktop.lower_policy.policies.groot import (
            CAMERAS,
            GrootArtifactContract,
            Gr00tPolicy,
        )
        from inference.desktop.lower_policy.rtc import RtcConfig

        rtc_kwargs.setdefault("enabled", True)
        policy = Gr00tPolicy(
            PolicyConfig(
                mode="none",
                ckpt_ref="x",
                cams=CAMERAS,
                execution_steps=8,
                # 実 config の GR00T variant は全て async (replan_family 設定済み)
                replan_family="groot_relative_eef_v1",
                rtc=RtcConfig(**rtc_kwargs),
            )
        )
        policy._artifact_contract = GrootArtifactContract(
            policy_type="groot",
            chunk_size=16,
            n_action_steps=16,
            video_horizon=1,
            history_offset_ticks=1,
            expert_role=None,
        )
        step = _make_decode_step()
        policy._post_processor = SimpleNamespace(steps=[step])
        # tick 周期と latency を実測済みの状態にする (30Hz / 90ms → d=3)
        for _ in range(4):
            policy._rtc_tick_deltas.append(0.0333)
        policy._rtc_delay.add(0.090)
        return policy, step

    def test_prefix_decodes_back_to_the_stored_leftover(self) -> None:
        policy, step = self._policy()
        rng = np.random.default_rng(21)
        absolute = _decode(step, rng.uniform(-0.8, 0.8, size=(16, _ENV_ACTION_DIM)))
        policy._rtc_leftover.store(absolute, origin_step=0)

        kwargs, metadata = policy._build_rtc_prefix(8)
        assert set(kwargs) == {"inference_delay", "prev_chunk_left_over"}
        prefix = kwargs["prev_chunk_left_over"].numpy()
        # execution_steps=8 消費した残り 8 行が prefix になる
        assert prefix.shape == (8, _ENV_ACTION_DIM)
        assert metadata["rtc_prefix_rows"] == 8
        np.testing.assert_allclose(
            _decode(step, prefix), absolute[8:], rtol=1e-4, atol=1e-4
        )

    def test_frozen_steps_from_measured_latency(self) -> None:
        # 90ms / 33.3ms = 2.7 → 切り上げて 3
        policy, step = self._policy()
        absolute = _decode(step, np.full((16, _ENV_ACTION_DIM), 0.3))
        policy._rtc_leftover.store(absolute, origin_step=0)
        kwargs, metadata = policy._build_rtc_prefix(8)
        assert kwargs["inference_delay"] == 3
        assert metadata["rtc_frozen_steps"] == 3
        assert metadata["rtc_tick_period_ms"] == pytest.approx(33.3, abs=0.1)
        assert metadata["rtc_delay_max_ms"] == pytest.approx(90.0, abs=0.1)

    def test_fixed_frozen_steps_override_the_estimate(self) -> None:
        policy, step = self._policy(frozen_steps=1)
        absolute = _decode(step, np.full((16, _ENV_ACTION_DIM), 0.3))
        policy._rtc_leftover.store(absolute, origin_step=0)
        kwargs, _metadata = policy._build_rtc_prefix(8)
        assert kwargs["inference_delay"] == 1

    def test_overlap_clamped_by_available_rows(self) -> None:
        # overlap_steps=8 でも残りが 4 行しか無ければ 4 行だけ渡す
        policy, step = self._policy(overlap_steps=8)
        absolute = _decode(step, np.full((16, _ENV_ACTION_DIM), 0.3))
        policy._rtc_leftover.store(absolute, origin_step=0)
        kwargs, metadata = policy._build_rtc_prefix(12)
        assert kwargs["prev_chunk_left_over"].shape[0] == 4
        assert metadata["rtc_prefix_rows"] == 4

    def test_exhausted_leftover_yields_no_prefix(self) -> None:
        policy, step = self._policy()
        absolute = _decode(step, np.full((16, _ENV_ACTION_DIM), 0.3))
        policy._rtc_leftover.store(absolute, origin_step=0)
        kwargs, metadata = policy._build_rtc_prefix(16)
        assert kwargs == {}
        assert metadata["rtc_prefix_rows"] == 0


class TestRtcClassCompositionAgainstLerobot:
    """実 LeRobot クラスとの接合部の退行ガード。

    lerobot は `inference/desktop/pixi.toml` で ==0.6.1 に pin されているが、
    upgrade 時にここが真っ先に壊れる (private シンボル / メソッド名 / 署名)。
    実 ckpt を load せずに検証できる範囲を固定しておく。
    """

    def test_mixin_composes_and_overrides(self) -> None:
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        from inference.desktop.lower_policy.policies.groot_rtc import (
            _RtcPrefixMixin,
            make_rtc_policy_class,
        )

        cls = make_rtc_policy_class()
        assert issubclass(cls, GrootPolicy)
        # MRO 上 mixin が base より前 = override が効く
        assert cls._prepare_n1_7_rtc_inputs is _RtcPrefixMixin._prepare_n1_7_rtc_inputs

    def test_custom_base_is_preserved(self) -> None:
        # furniture_groot は独自の runtime policy class を使うので、そちらにも
        # 同じ override が載ること
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        from inference.desktop.lower_policy.policies.groot_rtc import (
            make_rtc_policy_class,
        )

        class _CustomBase(GrootPolicy):
            pass

        cls = make_rtc_policy_class(_CustomBase)
        assert issubclass(cls, _CustomBase)

    def test_predict_action_chunk_accepts_our_kwargs(self) -> None:
        import inspect

        from lerobot.policies.groot.modeling_groot import GrootPolicy

        inspect.signature(GrootPolicy.predict_action_chunk).bind(
            object(), object(), inference_delay=3, prev_chunk_left_over=None
        )

    def test_upstream_declares_rtc_support(self) -> None:
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        assert GrootPolicy.supports_rtc(object()) is True


class TestFrozenStepsWithoutAsyncPipeline:
    """同期実行 (replan_family=None) では frozen=0 が正しい。

    推論中ロボットは最後の指令を保持したままで、新 chunk から先行送信される
    action が 1 本も無い。latency から d を推定すると、前 chunk の再現に数 step
    を費やして応答性を捨てることになる。
    """

    def _policy(self, replan_family):
        from types import SimpleNamespace

        from inference.desktop.lower_policy.policies.base import PolicyConfig
        from inference.desktop.lower_policy.policies.groot import (
            CAMERAS,
            GrootArtifactContract,
            Gr00tPolicy,
        )
        from inference.desktop.lower_policy.rtc import RtcConfig

        policy = Gr00tPolicy(
            PolicyConfig(
                mode="none",
                ckpt_ref="x",
                cams=CAMERAS,
                execution_steps=8,
                replan_family=replan_family,
                rtc=RtcConfig(enabled=True),
            )
        )
        policy._artifact_contract = GrootArtifactContract(
            policy_type="groot",
            chunk_size=16,
            n_action_steps=16,
            video_horizon=1,
            history_offset_ticks=1,
            expert_role=None,
        )
        step = _make_decode_step()
        policy._post_processor = SimpleNamespace(steps=[step])
        for _ in range(4):
            policy._rtc_tick_deltas.append(0.0333)
        policy._rtc_delay.add(0.090)  # async なら d=3 になる latency
        absolute = _decode(step, np.full((16, _ENV_ACTION_DIM), 0.3))
        policy._rtc_leftover.store(absolute, origin_step=0)
        return policy

    def test_sync_execution_freezes_nothing(self) -> None:
        policy = self._policy(replan_family=None)
        kwargs, metadata = policy._build_rtc_prefix(8)
        assert kwargs["inference_delay"] == 0
        assert metadata["rtc_frozen_steps"] == 0
        assert metadata["rtc_async_execution"] is False
        # prefix 自体は渡る (soft ramp によるchunk 間連続性は同期でも効く)
        assert metadata["rtc_prefix_rows"] == 8

    def test_async_execution_uses_the_measured_delay(self) -> None:
        policy = self._policy(replan_family="groot_relative_eef_v1")
        kwargs, metadata = policy._build_rtc_prefix(8)
        assert kwargs["inference_delay"] == 3
        assert metadata["rtc_async_execution"] is True

    def test_sync_execution_does_not_need_a_tick_period(self) -> None:
        # frozen=0 が確定するので、tick 周期が未計測でも初回 replan から RTC が効く
        policy = self._policy(replan_family=None)
        policy._rtc_tick_deltas.clear()
        kwargs, metadata = policy._build_rtc_prefix(8)
        assert kwargs["inference_delay"] == 0
        assert metadata["rtc_prefix_rows"] == 8
