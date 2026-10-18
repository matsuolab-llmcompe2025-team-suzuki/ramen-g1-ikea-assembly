import json

import numpy as np
import pytest

from inference.desktop.lower_policy.scripts.smoke_policy_forward import (
    _load_captured_observation,
    _load_captured_state,
    _smoke_language,
)


def test_load_captured_state_accepts_finite_matching_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "mode": "ramen_ori_71d",
                "last_tick_state": [float(i) for i in range(71)],
            }
        ),
        encoding="utf-8",
    )

    state = _load_captured_state(path, 71)

    assert state.shape == (71,)
    assert state.dtype == np.float32
    assert state[-1] == 70.0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"mode": "unknown", "last_tick_state": [0.0] * 71}, "unsupported"),
        ({"mode": "ramen_ori_71d"}, "last_tick_state"),
        ({"mode": "ramen_ori_71d", "last_tick_state": [0.0] * 70}, "shape"),
        (
            {"mode": "ramen_ori_71d", "last_tick_state": [float("nan")] * 71},
            "NaN/Inf",
        ),
    ],
)
def test_load_captured_state_rejects_invalid_dump(tmp_path, payload, message):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_captured_state(path, 71)


def test_load_captured_observation_accepts_v2_and_native_d405(tmp_path):
    from inference.desktop.lower_policy.policies.base import CameraKey

    path = tmp_path / "observation.npz"
    arrays = {
        "state": np.zeros(71, dtype=np.float32),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema": "ramen_policy_observation_v2",
                    "actuator_constructed": False,
                    "camera_timestamp_source": "desktop_post_decode_monotonic",
                }
            )
        ),
    }
    cameras = (
        CameraKey.HEAD_LEFT,
        CameraKey.HEAD_RIGHT,
        CameraKey.WRIST_LEFT,
        CameraKey.WRIST_RIGHT,
    )
    for camera in cameras:
        shape = (480, 848, 3) if camera.is_wrist else (480, 640, 3)
        arrays[camera.value] = np.ones(shape, dtype=np.uint8)
        arrays[f"{camera.value}_prev"] = np.ones(shape, dtype=np.uint8)
    np.savez_compressed(path, **arrays)

    policy = type(
        "PolicyContract",
        (),
        {"STATE_DIM": 71, "CAMERAS": cameras},
    )()

    state, frames, previous, metadata = _load_captured_observation(path, policy)

    assert state.shape == (71,)
    # 848 native D405 は load 時に 640 (training aspect) に center crop される
    assert frames[CameraKey.WRIST_LEFT].shape == (480, 640, 3)
    assert previous[CameraKey.WRIST_LEFT].shape == (480, 640, 3)
    assert previous[CameraKey.HEAD_LEFT].shape == (480, 640, 3)
    assert metadata["schema"] == "ramen_policy_observation_v2"


class _PolicyWithPrompt:
    """groot_pick_legs のように既定 prompt を公開する policy。"""

    DEFAULT_LANGUAGE_PROMPT = "pick table leg"


class _PolicyWithoutPrompt:
    """prompt を持たない policy (53D の groot 系など)。"""


def test_smoke_language_uses_the_policy_default_prompt():
    """worker が manifest task と厳密一致を要求するので既定 prompt を使う。"""
    assert (
        _smoke_language(_PolicyWithPrompt(), "smoke dry-run") == "pick table leg"
    )


def test_smoke_language_falls_back_when_the_policy_has_no_prompt():
    assert _smoke_language(_PolicyWithoutPrompt(), "smoke dry-run") == "smoke dry-run"


def test_load_only_and_dry_run_agree_on_the_language():
    """load-only と dry-run で prompt がずれないこと。

    以前は 2 箇所に別々に書かれており、dry-run だけ任意文字列を送っていた。
    その結果、正常な pick policy が
    `ValueError: task must exactly match manifest task` で全 tick 落ち、
    「pick が壊れている」と読めてしまった (2026-09-20、実 image 上で確認)。
    """
    policy = _PolicyWithPrompt()

    assert _smoke_language(policy, "smoke test") == _smoke_language(
        policy, "smoke dry-run"
    )


def test_smoke_calls_build_state_on_the_instance_not_the_class():
    """`policy.__class__.build_state_from_raw(raw)` を使わないこと。

    GR00T 系は staticmethod なので class 経由でも通るが、RAMEN-Ori は
    instance method で raw が self に入り TypeError になる
    (実 image で `rotate_table_base_ramen_ori_141_c32` を回して確認、2026-09-21)。
    VlaSkill (vla_skill.py) は instance 経由なので、そちらに揃える。
    """
    import inspect

    from inference.desktop.lower_policy.scripts import smoke_policy_forward as mod

    # コメントに文字列として書いてあるので、コード行だけを見る。
    code = [
        line
        for line in inspect.getsource(mod).splitlines()
        if not line.lstrip().startswith("#")
    ]
    src = "\n".join(code)

    assert "__class__.build_state_from_raw" not in src
    assert "policy.build_state_from_raw(raw)" in src
