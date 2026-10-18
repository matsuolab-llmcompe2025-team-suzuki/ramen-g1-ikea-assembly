"""Issue #129: scripts/rotate_specialist/patch_lerobot.py の test。

Phase 0 では実 patch 内容は未実装、skeleton 部分 (env 解析 + no-op 判定) のみ検証。
Phase 1-4 で patch content 追加時に本 test も拡張する。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def patch_mod(monkeypatch):
    """env をクリーンな状態で patch module を import (fresh state)。"""
    # ENV を default state に (以前 test の残留があれば掃除)
    for env in [
        "GROOT_LEFT_HAND_LOSS_WEIGHT",
        "GROOT_PER_CAM_SCALAR_ENABLE",
        "GROOT_BILATERAL_LOSS_WEIGHT",
        "GROOT_FK_ANCHOR_LOSS_WEIGHT",
    ]:
        monkeypatch.delenv(env, raising=False)
    # freshly import module (毎 test 独立)
    from model.subtask_policy_training.scripts.rotate_specialist import patch_lerobot as m
    importlib.reload(m)
    return m


def test_defaults_yield_no_feature_enabled(patch_mod) -> None:
    features = patch_mod.read_features_from_env()
    assert features[patch_mod.ENV_LEFT_HAND_LOSS_WEIGHT] == 1.0
    assert features[patch_mod.ENV_PER_CAM_SCALAR_ENABLE] is False
    assert features[patch_mod.ENV_BILATERAL_LOSS_WEIGHT] == 0.0
    assert features[patch_mod.ENV_FK_ANCHOR_LOSS_WEIGHT] == 0.0
    assert patch_mod.any_feature_enabled(features) is False


def test_left_hand_weight_enable_detected(patch_mod, monkeypatch) -> None:
    monkeypatch.setenv("GROOT_LEFT_HAND_LOSS_WEIGHT", "2.0")
    features = patch_mod.read_features_from_env()
    assert features[patch_mod.ENV_LEFT_HAND_LOSS_WEIGHT] == 2.0
    assert patch_mod.any_feature_enabled(features) is True


def test_per_cam_scalar_enable_detected(patch_mod, monkeypatch) -> None:
    monkeypatch.setenv("GROOT_PER_CAM_SCALAR_ENABLE", "true")
    features = patch_mod.read_features_from_env()
    assert features[patch_mod.ENV_PER_CAM_SCALAR_ENABLE] is True
    assert patch_mod.any_feature_enabled(features) is True


def test_bilateral_weight_enable_detected(patch_mod, monkeypatch) -> None:
    monkeypatch.setenv("GROOT_BILATERAL_LOSS_WEIGHT", "0.5")
    features = patch_mod.read_features_from_env()
    assert features[patch_mod.ENV_BILATERAL_LOSS_WEIGHT] == 0.5
    assert patch_mod.any_feature_enabled(features) is True


def test_fk_anchor_weight_enable_detected(patch_mod, monkeypatch) -> None:
    monkeypatch.setenv("GROOT_FK_ANCHOR_LOSS_WEIGHT", "0.3")
    features = patch_mod.read_features_from_env()
    assert features[patch_mod.ENV_FK_ANCHOR_LOSS_WEIGHT] == 0.3
    assert patch_mod.any_feature_enabled(features) is True


def test_env_bool_parsing_variants(patch_mod, monkeypatch) -> None:
    for truthy in ["true", "TRUE", "1", "yes", "on"]:
        monkeypatch.setenv("GROOT_PER_CAM_SCALAR_ENABLE", truthy)
        assert patch_mod.read_features_from_env()[patch_mod.ENV_PER_CAM_SCALAR_ENABLE] is True
    for falsy in ["false", "0", "no", "off", ""]:
        monkeypatch.setenv("GROOT_PER_CAM_SCALAR_ENABLE", falsy)
        assert patch_mod.read_features_from_env()[patch_mod.ENV_PER_CAM_SCALAR_ENABLE] is False


def test_env_float_invalid_raises(patch_mod, monkeypatch) -> None:
    monkeypatch.setenv("GROOT_LEFT_HAND_LOSS_WEIGHT", "not-a-float")
    with pytest.raises(ValueError, match="not a float"):
        patch_mod.read_features_from_env()


def _require_matching_lerobot(expected_version: str) -> None:
    """LeRobot が install されて version 一致してれば OK、そうでなければ skip。"""
    import importlib.metadata
    try:
        actual = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("lerobot not installed in this env")
    if actual != expected_version:
        pytest.skip(f"lerobot version mismatch (need {expected_version}, have {actual})")


def test_patch_processor_noop_when_all_defaults(patch_mod, monkeypatch) -> None:
    """全 feature default → LeRobot install + version 一致環境では False return。"""
    _require_matching_lerobot(patch_mod.LEROBOT_VERSION)
    changed = patch_mod.patch_processor(check_only=False)
    assert changed is False


def test_loss_side_patch_transform_on_copy(patch_mod, tmp_path) -> None:
    """Loss-side patch (T0-1 + L1) を lerobot source の copy に適用、transformation を verify。

    実 install file には触らない (test 分離)、tmp copy に対して patch 動作を確認。
    """
    import importlib.util
    try:
        spec = importlib.util.find_spec("lerobot.policies.groot.groot_n1_7")
    except ModuleNotFoundError:
        pytest.skip("lerobot not installed in this env")
    if spec is None or spec.origin is None:
        pytest.skip("lerobot not installed in this env")
    src_path = Path(spec.origin)
    tmp_target = tmp_path / "groot_n1_7.py"
    tmp_target.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Fresh, unpatched
    orig = tmp_target.read_text(encoding="utf-8")
    assert patch_mod.PATCH_MARKER not in orig
    assert "GROOT_LEFT_HAND_LOSS_WEIGHT" not in orig
    assert "GROOT_BILATERAL_LOSS_WEIGHT" not in orig

    # Apply
    applied = patch_mod._apply_loss_side_patch(target_path=tmp_target)
    assert applied is True

    # Verify transformation
    patched = tmp_target.read_text(encoding="utf-8")
    assert patch_mod.PATCH_MARKER in patched
    # T0-1 features
    assert "GROOT_LEFT_HAND_LOSS_WEIGHT" in patched
    assert "(3, 4, 5, 6, 7, 8, 9, 17)" in patched  # LEFT_HAND_INDICES
    # L1 features
    assert "GROOT_BILATERAL_LOSS_WEIGHT" in patched
    assert "pred_actions[..., 3:10]" in patched   # BILATERAL_LEFT_INDICES
    assert "pred_actions[..., 10:17]" in patched  # BILATERAL_RIGHT_INDICES
    assert "_tr_bi_loss" in patched
    # L4 features
    assert "GROOT_FK_ANCHOR_LOSS_WEIGHT" in patched
    assert "noisy_trajectory + (1.0 - t) * pred_actions" in patched  # 1-step Euler
    assert "G1WristFKTorch" in patched
    assert "_tr_fk_singleton" in patched
    assert "_tr_fk_loss" in patched
    # 単一 marker
    assert patched.count(patch_mod.PATCH_MARKER) == 1
    # loss 計算式が保存されてる (else 節に残る + normalization line 変わらず)
    assert 'F.mse_loss(pred_actions, velocity, reduction="none") * action_mask' in patched
    assert "loss = action_loss.sum() / (action_mask.sum() + 1e-6)" in patched


def test_loss_side_patch_idempotent(patch_mod, tmp_path) -> None:
    """Loss-side patch を 2 回適用しても 2 回目は no-op (marker 検出)。"""
    import importlib.util
    try:
        spec = importlib.util.find_spec("lerobot.policies.groot.groot_n1_7")
    except ModuleNotFoundError:
        pytest.skip("lerobot not installed in this env")
    if spec is None or spec.origin is None:
        pytest.skip("lerobot not installed in this env")
    tmp_target = tmp_path / "groot_n1_7.py"
    tmp_target.write_text(Path(spec.origin).read_text(encoding="utf-8"), encoding="utf-8")

    applied_1 = patch_mod._apply_loss_side_patch(target_path=tmp_target)
    text_after_1 = tmp_target.read_text(encoding="utf-8")
    applied_2 = patch_mod._apply_loss_side_patch(target_path=tmp_target)
    text_after_2 = tmp_target.read_text(encoding="utf-8")

    assert applied_1 is True
    assert applied_2 is False
    assert text_after_1 == text_after_2


def test_loss_side_patch_check_only_raises_on_unpatched(patch_mod, tmp_path) -> None:
    """check_only=True で未 patch file を渡すと RuntimeError。"""
    import importlib.util
    try:
        spec = importlib.util.find_spec("lerobot.policies.groot.groot_n1_7")
    except ModuleNotFoundError:
        pytest.skip("lerobot not installed in this env")
    if spec is None or spec.origin is None:
        pytest.skip("lerobot not installed in this env")
    tmp_target = tmp_path / "groot_n1_7.py"
    tmp_target.write_text(Path(spec.origin).read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not active"):
        patch_mod._apply_loss_side_patch(check_only=True, target_path=tmp_target)


def test_loss_side_patch_sha256_guard(patch_mod, tmp_path) -> None:
    """SHA256 が違う file (改変済 or 別 version) には patch 拒否。"""
    tmp_target = tmp_path / "fake_groot_n1_7.py"
    tmp_target.write_text("# fake content, wrong SHA\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to patch"):
        patch_mod._apply_loss_side_patch(target_path=tmp_target)


def test_patch_processor_t01_dispatch(patch_mod, monkeypatch) -> None:
    """patch_processor が T0-1 branch を dispatch する (real lerobot install 環境)。

    NOTE: 本 test は installed lerobot を実際に触るため、実 file を書き換える。
    tmp copy 経由の test_t01_patch_transform_on_copy と対で存在。
    """
    _require_matching_lerobot(patch_mod.LEROBOT_VERSION)
    pytest.skip("real install 書換は destructive、skip default (手動 verify 用に残す)")


def test_l1_bilateral_loss_math_reference() -> None:
    """L1 bilateral loss の数式が「左右 error の非独立性 penalize」の意図と一致することを
    独立実装で verify (patch 経由ではなく、numpy 参照実装で expected 値を計算)。

    Expected:
        L_bi = mean_over_masked( ((pred_L - pred_R) - (true_L - true_R))^2 )
             = mean_over_masked( ((pred_L - true_L) - (pred_R - true_R))^2 )
             = mean_over_masked( (L_err - R_err)^2 )
    """
    import numpy as np
    rng = np.random.default_rng(0)
    B, T = 2, 4
    # 19 dim action、L=3..10、R=10..17
    pred = rng.normal(size=(B, T, 19)).astype(np.float32)
    true = rng.normal(size=(B, T, 19)).astype(np.float32)
    mask = np.ones((B, T, 19), dtype=np.float32)

    L_pred, R_pred = pred[..., 3:10], pred[..., 10:17]
    L_true, R_true = true[..., 3:10], true[..., 10:17]
    L_mask, R_mask = mask[..., 3:10], mask[..., 10:17]
    bi_mask = L_mask * R_mask
    bi_diff = (L_pred - R_pred) - (L_true - R_true)
    bi_err = (bi_diff ** 2) * bi_mask
    expected = bi_err.sum() / (bi_mask.sum() + 1e-6)

    # 別 form: 左右 error の非独立性
    L_err = L_pred - L_true
    R_err = R_pred - R_true
    alt_expected = (((L_err - R_err) ** 2) * bi_mask).sum() / (bi_mask.sum() + 1e-6)

    np.testing.assert_allclose(expected, alt_expected, atol=1e-6)
    # 2 form が数式的に等価 = intent が「左右 error 非独立性 penalize」で正しい


def test_patch_processor_t02_skipped_for_groot(patch_mod, monkeypatch) -> None:
    """T0-2 (per-cam scalar) enable → GR00T arch mismatch で skip 判断 (NotImplementedError)。"""
    monkeypatch.setenv("GROOT_PER_CAM_SCALAR_ENABLE", "true")
    _require_matching_lerobot(patch_mod.LEROBOT_VERSION)
    with pytest.raises(NotImplementedError, match="T0-2 .* skipped"):
        patch_mod.patch_processor(check_only=False)


def test_l4_fk_reconstruction_math_reference() -> None:
    """L4 の velocity → actions 復元式が正しいことを numpy で verify。

    Flow matching の定義:
        velocity = actions - noise
        noisy_trajectory = (1 - t) * noise + t * actions
    復元式:
        actions_pred = noisy_trajectory + (1 - t) * velocity
                     = (1-t)*noise + t*actions + (1-t)*(actions - noise)
                     = t*actions + (1-t)*actions
                     = actions ✓
    """
    import numpy as np
    rng = np.random.default_rng(3)
    B, T, D = 2, 4, 19
    actions = rng.normal(size=(B, T, D)).astype(np.float64)
    noise = rng.normal(size=(B, T, D)).astype(np.float64)
    t = rng.uniform(0.1, 0.9, size=(B, 1, 1)).astype(np.float64)
    velocity = actions - noise
    noisy_trajectory = (1 - t) * noise + t * actions
    # 復元
    actions_reconstructed = noisy_trajectory + (1 - t) * velocity
    np.testing.assert_allclose(actions_reconstructed, actions, atol=1e-10)


def test_l4_fk_import_verify_when_pythonpath_ok(patch_mod, monkeypatch) -> None:
    """L4 enable + FK module import 可能 → patch install に進む (実 install はしない)。

    monkeypatch で _apply_loss_side_patch を捕捉、install 呼ばれるか確認。
    """
    monkeypatch.setenv("GROOT_FK_ANCHOR_LOSS_WEIGHT", "0.3")
    _require_matching_lerobot(patch_mod.LEROBOT_VERSION)
    # FK module import 可能か先に確認 (default env で fail する場合 skip)
    try:
        from inference.desktop.perception.g1_urdf_fk_torch import G1WristFKTorch  # noqa
    except ImportError:
        pytest.skip("g1_urdf_fk_torch not importable in this env (torch missing)")
    # 実 install は destructive なので mock で置換
    called = {"n": 0}
    def mock_apply(check_only=False):
        called["n"] += 1
        return True
    monkeypatch.setattr(patch_mod, "_apply_loss_side_patch", mock_apply)
    result = patch_mod.patch_processor(check_only=False)
    assert called["n"] == 1
    assert result is True


def test_patch_processor_raises_on_lerobot_version_mismatch(patch_mod, monkeypatch) -> None:
    """LeRobot version が違う env で feature enabled + patch_processor 呼ぶと version guard で fail。

    inference/desktop env (lerobot 0.6.1) から呼んだ場合の behaviour を明示化。
    実 Sakura training env (lerobot 0.6.0) では version 一致で patch 経路に入る。
    """
    import importlib.metadata
    try:
        actual = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("lerobot not installed in this env")
    if actual == patch_mod.LEROBOT_VERSION:
        pytest.skip("lerobot version matches, no mismatch to test here")
    # version 違いなら feature enabled でも RuntimeError (version guard)
    monkeypatch.setenv("GROOT_LEFT_HAND_LOSS_WEIGHT", "2.0")
    with pytest.raises(RuntimeError, match="expected lerobot=="):
        patch_mod.patch_processor(check_only=False)
