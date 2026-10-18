"""Phase 4: preprocess_frame が training pipeline と 1 pixel 一致することを verify (Issue #125)。

# 目的

Inference-side `preprocess_frame` が training-side `data_lerobot.py` の val-mode
image pipeline と **element-wise diff < 1e-5** で一致することを保証する。
不一致は distribution shift となり、fine-tuned model 性能を silent に落とす。

# 検証対象の training pipeline (val mode = augmentation 無し)

1. LeRobot Dataset return: uint8 (H, W, 3) RGB → float32 (C, H, W) [0, 1]
2. `transforms.Compose([transforms.Resize((224, 224), antialias=True)])`
   (data_lerobot.py:make_image_transform)
3. `transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)`
   (data_lerobot.py:ImageAugPipeline.normalize、val は aug スキップして normalize のみ)

# サンプル画像

`research/data/subtask_frames/` から実 rotate_table_base / move_table_base
サンプルを使用。head は 480x1280 packed stereo で split 必要 (`split_head_stereo`
経由)、wrist は 480x640 mono で直接使える。

# Env

torch + torchvision 依存 = runtime env or model/ramen_ori env で走る前提。
default env では cv2/torch 無くて全 test skip。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None

SAMPLE_DIR = Path("/home/ubuntu/work_dir/iros/research/data/subtask_frames")

# 実サンプル (rotate_table_base 系 + move_table_base 系 + head/wrist 全 pattern)
SAMPLE_FRAMES = {
    "wrist_rotate": SAMPLE_DIR / "03_t0081.62s_at_rotate_table_base_lwrist.jpg",
    "wrist_move": SAMPLE_DIR / "00_t0004.25s_at_move_to_table_lwrist.jpg",
    "wrist_pick": SAMPLE_DIR / "01_t0011.43s_at_pick_table_leg_rwrist.jpg",
    "head_packed_rotate": SAMPLE_DIR / "03_t0081.62s_at_rotate_table_base_head.jpg",
}

# 一致 tolerance: torchvision と training は同じ path で走るので数値誤差のみ
TOLERANCE_MAX = 1e-5
TOLERANCE_MEAN = 1e-6


pytestmark = pytest.mark.skipif(
    not (HAS_TORCH and HAS_CV2),
    reason="torch + torchvision + cv2 required (runtime env or model/ramen_ori env)",
)


def _load_bgr(path: Path) -> np.ndarray:
    import cv2

    if not path.exists():
        pytest.skip(f"sample not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cv2.imread returned None for {path}")
    return img


def _training_pipeline(frame_bgr: np.ndarray) -> np.ndarray:
    """Training-side pipeline (data_lerobot.py val mode) を単独再現する reference。

    make_image_transform + ImageAugPipeline.normalize (val mode = aug 無し) を
    inline で復元する。model.ramen_ori.data_lerobot を直接 import すると重い
    (LeRobot fork の Dataset class collection が走る) ので、pipeline の骨格
    (Compose Resize antialias + Normalize) だけ再現。
    """
    import torch
    from torchvision import transforms

    from inference.desktop.lower_policy.policies.ramen_ori import (
        IMAGE_HW,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

    # cv2 は BGR、training LeRobot は RGB return なので変換
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    # LeRobot Dataset: uint8 HWC → float32 CHW [0,1] に変換される
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
    # make_image_transform: transforms.Compose([Resize((224, 224), antialias=True)])
    resize = transforms.Compose(
        [transforms.Resize((IMAGE_HW[0], IMAGE_HW[1]), antialias=True)]
    )
    t = resize(t)
    # ImageAugPipeline.normalize (val mode: aug スキップ、normalize のみ)
    normalize = transforms.Normalize(
        mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)
    )
    t = normalize(t)
    return t.numpy()


class TestPixelExactMatch:
    """training pipeline と inference の element-wise diff が tolerance 以下か検証。"""

    @pytest.mark.parametrize(
        "sample_name",
        ["wrist_rotate", "wrist_move", "wrist_pick"],
    )
    def test_wrist_frame_matches_training(self, sample_name: str) -> None:
        from inference.desktop.lower_policy.policies.ramen_ori import preprocess_frame

        bgr = _load_bgr(SAMPLE_FRAMES[sample_name])
        assert bgr.shape == (480, 640, 3), f"wrist expected 480x640, got {bgr.shape}"

        inference_out = preprocess_frame(bgr)
        training_out = _training_pipeline(bgr)

        assert inference_out.shape == (3, 224, 224)
        assert inference_out.dtype == np.float32
        assert inference_out.shape == training_out.shape

        diff = np.abs(inference_out - training_out)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        assert max_diff < TOLERANCE_MAX, (
            f"[{sample_name}] max pixel diff {max_diff:.2e} exceeds tolerance "
            f"{TOLERANCE_MAX:.2e} (training vs inference preprocess drift)"
        )
        assert mean_diff < TOLERANCE_MEAN, (
            f"[{sample_name}] mean pixel diff {mean_diff:.2e} exceeds tolerance "
            f"{TOLERANCE_MEAN:.2e}"
        )

    def test_head_stereo_split_then_match(self) -> None:
        """head 480x1280 packed → split 480x640 → preprocess → training match。"""
        from inference.desktop.lower_policy.policies.ramen_ori import (
            preprocess_frame,
            split_head_stereo,
        )

        packed = _load_bgr(SAMPLE_FRAMES["head_packed_rotate"])
        assert packed.shape == (480, 1280, 3), (
            f"head expected 480x1280, got {packed.shape}"
        )
        head_l, head_r = split_head_stereo(packed)
        assert head_l.shape == (480, 640, 3)
        assert head_r.shape == (480, 640, 3)

        # split 済 head_left で pixel-exact match verify
        inference_out = preprocess_frame(head_l)
        training_out = _training_pipeline(head_l)
        max_diff = float(np.abs(inference_out - training_out).max())
        assert max_diff < TOLERANCE_MAX, (
            f"head_left max pixel diff {max_diff:.2e} exceeds tolerance"
        )

    def test_synthetic_frame_deterministic(self) -> None:
        """合成 gradient frame でも一致 (edge / high-freq content の resize 一致)。"""
        from inference.desktop.lower_policy.policies.ramen_ori import preprocess_frame

        # 決定的 gradient (row * col + channel offset)
        H, W = 480, 640
        row = np.arange(H, dtype=np.uint8)[:, None]
        col = np.arange(W, dtype=np.uint8)[None, :]
        bgr = np.stack(
            [
                (row + col) % 256,
                (row * 2 + col) % 256,
                (row + col * 2) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        assert bgr.shape == (H, W, 3)

        inference_out = preprocess_frame(bgr)
        training_out = _training_pipeline(bgr)
        max_diff = float(np.abs(inference_out - training_out).max())
        assert max_diff < TOLERANCE_MAX, (
            f"synthetic gradient max pixel diff {max_diff:.2e} > {TOLERANCE_MAX:.2e}"
        )


class TestCv2ImplementationDivergence:
    """旧 cv2 実装との比較 (前 Phase 3 実装は cv2、差分を明示的に文書化)。

    cv2.INTER_LINEAR は torchvision antialias=True と一致しない。この test は
    cv2 実装を採用するとどれくらい drift するかを numerical に示し、torchvision
    切り替えの正当性を残す (regression 監視用ではなく documentation 意図)。
    """

    def _cv2_impl(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Phase 3 実装 (cv2.resize) の再現。documentation 用の reference。"""
        import cv2

        from inference.desktop.lower_policy.policies.ramen_ori import (
            IMAGE_HW,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

        rgb = frame_bgr[:, :, ::-1]
        resized = cv2.resize(
            rgb, (IMAGE_HW[1], IMAGE_HW[0]), interpolation=cv2.INTER_LINEAR
        )
        float_img = resized.astype(np.float32) / 255.0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        normalized = (float_img - mean) / std
        return np.transpose(normalized, (2, 0, 1)).copy()

    def test_cv2_diverges_from_training(self) -> None:
        """cv2 実装は training と非無視な diff を出す (torchvision 採用の正当化)。"""
        bgr = _load_bgr(SAMPLE_FRAMES["wrist_rotate"])
        cv2_out = self._cv2_impl(bgr)
        training_out = _training_pipeline(bgr)
        max_diff = float(np.abs(cv2_out - training_out).max())
        mean_diff = float(np.abs(cv2_out - training_out).mean())
        # cv2 vs training は 1e-2 order の diff が出る (documentation 意図で assert)
        # ここでは pytest.approx で "cv2 は training と drift する" を assert する
        # (数値は informational、tolerance を超えていることが正当化の証拠)
        assert max_diff > TOLERANCE_MAX, (
            f"unexpected: cv2 impl max_diff={max_diff:.2e} within torchvision "
            f"tolerance — torchvision と cv2 が偶然一致した可能性 (要確認)"
        )
        # 値を print (pytest -s で確認できる、CI 依存無し informational)
        print(
            f"\n[cv2 divergence] max={max_diff:.4e} mean={mean_diff:.4e} "
            f"(training uses torchvision, cv2 as baseline for comparison)"
        )
