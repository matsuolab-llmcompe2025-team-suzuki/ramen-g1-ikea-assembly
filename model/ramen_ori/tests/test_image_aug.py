"""image_aug.ImageAugPipeline の unit test (Issue #139)。

# 対象
- wrist 判定: source 名 (cam_2 / cam_3) を `wrist_cam_keys` で wrist 扱いできる
- refactor 後の `apply` が refactor 前の実装 (下の `_legacy_apply`) とビット一致する
- offline bake の高速経路 (photometric 1 回 → token / overlay に分岐) が `apply` と一致する
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torchvision.transforms.v2 import functional as tvF

from model.ramen_ori.image_aug import (
    ImageAugPipeline,
    _identity_affine_matrix,
    _make_head_affine_matrix,
)

BASE_YAML = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
HEAD = "observation.images.cam_0"
LEFT_WRIST = "observation.images.cam_2"
RIGHT_WRIST = "observation.images.cam_3"


def _aug_cfg(**overrides) -> dict:
    """base.yaml の augmentation section を enabled=true + geometric on で返す。"""
    cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))["augmentation"]
    cfg = copy.deepcopy(cfg)
    cfg["enabled"] = True
    cfg["geometric"]["enabled"] = True
    cfg["wrist_cam_keys"] = [LEFT_WRIST, RIGHT_WRIST]
    cfg.update(overrides)
    return cfg


def _legacy_apply(pipe, img_2, cam_key, idx, overlay_callback=None, skip_geometric=False):
    """refactor 前の `ImageAugPipeline.apply` (is_train=True, skip_normalize=True) の写し。"""
    import zlib

    seed = zlib.crc32(f"{idx}:{cam_key}".encode()) & 0x7FFFFFFF
    is_wrist = pipe.is_wrist(cam_key)
    color_jitter = pipe.wrist_color_jitter if is_wrist else pipe.head_color_jitter
    gaussian_blur = pipe.wrist_gaussian_blur if is_wrist else pipe.head_gaussian_blur
    sharpness = pipe.wrist_sharpness if is_wrist else pipe.head_sharpness
    gaussian_noise = pipe.wrist_gaussian_noise if is_wrist else pipe.head_gaussian_noise
    jpeg = pipe.wrist_jpeg if is_wrist else pipe.head_jpeg
    use_geometric = pipe.geometric_enabled and (not is_wrist) and (not skip_geometric)
    if use_geometric:
        geom_rng = np.random.RandomState(seed)
        angle_deg = float(geom_rng.uniform(pipe.geometric_degrees[0], pipe.geometric_degrees[1]))
        tx_frac = float(geom_rng.uniform(-pipe.geometric_translate[0], pipe.geometric_translate[0]))
        ty_frac = float(geom_rng.uniform(-pipe.geometric_translate[1], pipe.geometric_translate[1]))
        affine_matrix = _make_head_affine_matrix(angle_deg, tx_frac, ty_frac)
    else:
        angle_deg = tx_frac = ty_frac = 0.0
        affine_matrix = _identity_affine_matrix()
    results = []
    for frame_i in range(2):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            img = color_jitter(img_2[frame_i])
            img = gaussian_blur(img)
            img = sharpness(img)
            img = gaussian_noise(img)
            img = jpeg(img)
            img = pipe.random_erasing(img)
            if overlay_callback is not None:
                img = overlay_callback(img, frame_i)
            if use_geometric:
                H, W = img.shape[-2:]
                img = tvF.affine(
                    img,
                    angle=angle_deg,
                    translate=[int(round(tx_frac * W)), int(round(ty_frac * H))],
                    scale=1.0,
                    shear=[0.0, 0.0],
                )
            results.append(img)
    return torch.stack(results, dim=0), affine_matrix


def _random_img_2(seed: int, h: int = 48, w: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    img = torch.rand((3, h, w), generator=g)
    return img.unsqueeze(0).repeat(2, 1, 1, 1)


def _box_overlay(img: torch.Tensor, _frame_i: int) -> torch.Tensor:
    """test 用 overlay: 左上に固定の赤枠を描く (決定的)。"""
    out = img.clone()
    out[0, 2:6, 2:20] = 1.0
    out[1:, 2:6, 2:20] = 0.0
    return out


# ---------------------------------------------------------------------------
# wrist 判定
# ---------------------------------------------------------------------------


def test_is_wrist_uses_explicit_cam_keys_for_source_names() -> None:
    pipe = ImageAugPipeline(_aug_cfg())
    assert pipe.is_wrist(LEFT_WRIST)
    assert pipe.is_wrist(RIGHT_WRIST)
    assert not pipe.is_wrist(HEAD)
    # 学習 view 名は従来どおり substring で判定
    assert pipe.is_wrist("observation.images.left_wrist")
    assert not pipe.is_wrist("observation.images.head_left")


def test_is_wrist_without_explicit_keys_keeps_substring_only() -> None:
    pipe = ImageAugPipeline(_aug_cfg(wrist_cam_keys=None))
    assert not pipe.is_wrist(LEFT_WRIST)
    assert pipe.is_wrist("observation.images.right_wrist")


def test_uses_geometric_only_for_head_when_enabled() -> None:
    pipe = ImageAugPipeline(_aug_cfg())
    assert pipe.uses_geometric(HEAD)
    assert not pipe.uses_geometric(LEFT_WRIST)
    cfg = _aug_cfg()
    cfg["geometric"]["enabled"] = False
    assert not ImageAugPipeline(cfg).uses_geometric(HEAD)


# ---------------------------------------------------------------------------
# refactor 前後の一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cam_key", [HEAD, LEFT_WRIST, RIGHT_WRIST])
@pytest.mark.parametrize("skip_geometric", [False, True])
@pytest.mark.parametrize("with_overlay", [False, True])
def test_apply_matches_legacy_implementation(cam_key, skip_geometric, with_overlay) -> None:
    pipe = ImageAugPipeline(_aug_cfg())
    cb = _box_overlay if with_overlay else None
    for i in range(12):
        img_2 = _random_img_2(i)
        idx = f"{cam_key}:chunk-000:file-000:{i:06d}:v{1 + i % 9}"
        got, got_affine = pipe.apply(
            img_2, cam_key, idx, True,
            overlay_callback=cb, skip_normalize=True, skip_geometric=skip_geometric,
        )
        want, want_affine = _legacy_apply(
            pipe, img_2, cam_key, idx, overlay_callback=cb, skip_geometric=skip_geometric
        )
        assert torch.equal(got, want)
        assert torch.equal(got_affine, want_affine)


def test_apply_leaves_global_rng_state_untouched() -> None:
    pipe = ImageAugPipeline(_aug_cfg())
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    pipe.apply(_random_img_2(0), HEAD, "x", True, skip_normalize=True)
    assert torch.equal(torch.random.get_rng_state(), before)


# ---------------------------------------------------------------------------
# offline bake の高速経路
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cam_key", [HEAD, LEFT_WRIST])
def test_fast_path_token_and_overlay_match_apply(cam_key) -> None:
    """photometric を 1 回だけ計算して分岐しても、apply を 2 回呼んだ結果と一致する。"""
    pipe = ImageAugPipeline(_aug_cfg())
    for i in range(12):
        img_2 = _random_img_2(100 + i)
        idx = f"{cam_key}:chunk-000:file-001:{i:06d}:v{1 + i % 9}"

        base = pipe.photometric(img_2[1], cam_key, idx)
        token_fast = base
        overlay_fast = pipe.apply_geometric(
            _box_overlay(base.clone(), 1), pipe.geometric_params(cam_key, idx)
        )

        token_ref = pipe.apply(
            img_2, cam_key, idx, True, skip_normalize=True, skip_geometric=True
        )[0][1]
        overlay_ref = pipe.apply(
            img_2, cam_key, idx, True, overlay_callback=_box_overlay, skip_normalize=True
        )[0][1]
        assert torch.equal(token_fast, token_ref)
        assert torch.equal(overlay_fast, overlay_ref)


def test_wrist_token_equals_overlay_without_overlay_callback() -> None:
    """wrist は overlay も geometric も無いので token と overlay が同一 (common 共有の前提)。"""
    pipe = ImageAugPipeline(_aug_cfg())
    img_2 = _random_img_2(7)
    idx = "wrist-share-check:v3"
    token = pipe.apply(img_2, LEFT_WRIST, idx, True, skip_normalize=True, skip_geometric=True)[0]
    overlay = pipe.apply(img_2, LEFT_WRIST, idx, True, skip_normalize=True)[0]
    assert torch.equal(token, overlay)


class TestSingleFrame:
    """Issue #141 RO-11: 今の frame だけ (T=1) でも同じ pipeline が通る。"""

    def test_single_frame_train_and_val(self) -> None:
        import torch

        from model.ramen_ori.image_aug import ImageAugPipeline

        pipe = ImageAugPipeline(_aug_cfg())
        img = torch.rand(1, 3, 32, 32)
        for cam in (HEAD, LEFT_WRIST):
            for is_train in (True, False):
                out, affine = pipe.apply(img, cam, idx=3, is_train=is_train)
                assert out.shape == (1, 3, 32, 32)
                assert affine.shape == (2, 3)

    def test_single_frame_equals_current_frame_of_pair(self) -> None:
        """1 枚で通した結果は、同じ idx で 2 枚 (前 + 今) を通したときの今の frame と同じ aug になる。"""
        import torch

        from model.ramen_ori.image_aug import ImageAugPipeline

        pipe = ImageAugPipeline(_aug_cfg())
        pair = _random_img_2(seed=5)
        torch.manual_seed(0)
        two, _ = pipe.apply(pair, HEAD, idx=7, is_train=True)
        torch.manual_seed(0)
        one, _ = pipe.apply(pair[1:], HEAD, idx=7, is_train=True)
        torch.testing.assert_close(one[0], two[1])
