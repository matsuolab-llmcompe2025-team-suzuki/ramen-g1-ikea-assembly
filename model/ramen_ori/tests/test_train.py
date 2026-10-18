"""train.py の unit test (Issue #115、Phase 1 M10)。

- config loads and has expected structure (Hydra compose API 使用)
- CLI override が正しく反映される
- 実 train loop の smoke は M11 (RTX 5090 100 iter smoke) で verify、本 unit test では
  static config 部分だけ検証
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestConfigLoad:
    def test_base_config_loads_with_expected_structure(self) -> None:
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR.absolute()), version_base=None
        ):
            cfg = compose(config_name="base")
        assert cfg.device == "cuda"
        assert cfg.model.d_model == 512
        assert cfg.model.chunk_len == 16
        # Issue #129 Phase A (2026-08-31): action 19→16 (waist 除外)、N_cams 4→3 (head_right 除外)。
        assert cfg.model.action_dim == 16
        # Issue #129 Phase B (2026-08-31): L4 FK anchor (base default: disabled、weight=0.3)。
        assert cfg.model.l4_enabled is False
        # Issue #141 RO-16 / RO-12: FK の loss の重みは毎 step 自動 (BC の 0.2 を 3 項に配分、上限あり)
        assert cfg.model.fk_loss.ratio == pytest.approx(0.2)
        assert cfg.model.fk_loss.weight_max > 0
        # Issue #141 RO-4 / RO-6: 正規化は既定 off (推論が base.yaml から Phase K の構造で組み立てるため)
        assert cfg.model.normalization.enabled is False
        assert cfg.model.state.input_norm is True
        # Issue #141: QK-norm は retrain だけ (base は Phase K と同じ nn.MultiheadAttention のまま)
        assert "qk_norm" not in cfg.model.fusion and "qk_norm" not in cfg.model.action_expert
        assert cfg.val.ee_error_rows == 8
        assert cfg.model.fk is None
        # Issue #141: backbone は凍結 + eval (LingBot の loader が凍結して返し、train mode では RoPE が乱れる)
        assert cfg.model.vision.freeze is True
        # Issue #129 Phase C (2026-08-31): partial-train + LLRD (base default: disabled)。
        assert cfg.optim.partial_train.enabled is False
        assert cfg.optim.name == "adamw"  # Issue #141: muon は retrain/c32_muon.yaml だけ
        assert cfg.optim.partial_train.last_n_blocks == 4
        assert cfg.optim.partial_train.llrd_decay == pytest.approx(0.75)
        assert cfg.optim.partial_train.backbone_lr_ratio == pytest.approx(0.2)

        # Phase 2 A-0 で sub-module 化: num_skills は skill._target_ block の param に移動
        assert cfg.model.skill.num_skills == 6
        assert cfg.model.skill._target_ == "model.ramen_ori.skill.SkillEmbedding"
        assert cfg.model.fusion._target_ == "model.ramen_ori.fusion.FusionTransformer"
        assert cfg.model.action_expert._target_ == "model.ramen_ori.action_expert.FlowMatchingDiT"
        assert cfg.data._target_ == "model.ramen_ori.data.DummyRamenOriDataset"
        assert cfg.data.N_cams == 3
        assert cfg.optim.lr == 1.0e-4
        assert cfg.training.max_steps == 100
        assert cfg.wandb.enabled is False

    def test_real_task5_7_relative_action_defaults_disabled(self) -> None:
        """Issue #129 Phase F (2026-08-31): use_relative_action は default false。"""
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR.absolute()), version_base=None
        ):
            cfg = compose(config_name="real_task5_7")
        assert cfg.data.use_relative_action is False
        assert cfg.data.relative_stats_path is None

    def test_cli_override_works(self) -> None:
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR.absolute()), version_base=None
        ):
            cfg = compose(
                config_name="base",
                overrides=[
                    "training.max_steps=5",
                    "device=cpu",
                    "wandb.enabled=true",
                ],
            )
        assert cfg.training.max_steps == 5
        assert cfg.device == "cpu"
        assert cfg.wandb.enabled is True

    def test_train_module_importable(self) -> None:
        """train.py の import が壊れていない (module-level 副作用がないこと)。"""
        from model.ramen_ori import train

        assert callable(train.main)
        assert callable(train._build_model)

    def test_real_task5_7_config_defaults_obb_none(self) -> None:
        """D-2: real_task5_7 は default で obb_source=none (OBB path 完全 skip)、
        obb_precomputed_root は null (opt-in で precomputed_token に切替)。"""
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR.absolute()), version_base=None
        ):
            cfg = compose(config_name="real_task5_7")
        assert cfg.data.obb_source == "none"
        assert cfg.data.obb_precomputed_root is None
        assert cfg.data.obb_precomputed_hash is None

    def test_real_task5_7_config_defaults_overlay_params(self) -> None:
        """D-3: overlay 系 field は default で v1 決定値 (conf=0.30, thickness=2, class_filter=null)。"""
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR.absolute()), version_base=None
        ):
            cfg = compose(config_name="real_task5_7")
        assert cfg.data.overlay_conf_threshold == pytest.approx(0.30)
        assert cfg.data.overlay_line_thickness == 2
        assert cfg.data.overlay_class_filter is None

    def test_real_task5_7_config_overlay_override(self) -> None:
        """D-3: CLI override で overlay 有効化 + filter 指定可能。"""
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR.absolute()), version_base=None
        ):
            cfg = compose(
                config_name="real_task5_7",
                overrides=[
                    "data.obb_source=overlay",
                    "data.obb_precomputed_root=outputs/yolo_obb_cache",
                    "data.overlay_conf_threshold=0.40",
                    "data.overlay_line_thickness=3",
                    "data.overlay_class_filter=[3,4]",
                ],
            )
        assert cfg.data.obb_source == "overlay"
        assert cfg.data.overlay_conf_threshold == pytest.approx(0.40)
        assert cfg.data.overlay_line_thickness == 3
        assert list(cfg.data.overlay_class_filter) == [3, 4]

    def test_real_task5_7_config_precomputed_token_override(self) -> None:
        """D-2: CLI override で precomputed_token + root 指定可能。"""
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR.absolute()), version_base=None
        ):
            cfg = compose(
                config_name="real_task5_7",
                overrides=[
                    "data.obb_source=precomputed_token",
                    "data.obb_precomputed_root=outputs/yolo_obb_cache",
                    "data.obb_precomputed_hash=abc123def456",
                ],
            )
        assert cfg.data.obb_source == "precomputed_token"
        assert cfg.data.obb_precomputed_root == "outputs/yolo_obb_cache"
        assert cfg.data.obb_precomputed_hash == "abc123def456"

    def test_real_task5_7_hf_autopush_defaults_disabled(self) -> None:
        """Issue #122: hf_autopush は default 無効、Run 起動時に override で on 化する運用。"""
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR.absolute()), version_base=None
        ):
            cfg = compose(config_name="real_task5_7")
        assert cfg.training.hf_autopush.enabled is False
        assert cfg.training.hf_autopush.repo_id is None
        assert cfg.training.hf_autopush.every_n_ckpt == 2   # ckpt_every=5000 と積で 10k step 毎
        assert cfg.training.hf_autopush.private is True

    def test_real_task5_7_hf_autopush_override(self) -> None:
        """Issue #122: Run 起動時の override で enabled + repo_id + every_n_ckpt 差替可能。"""
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR.absolute()), version_base=None
        ):
            cfg = compose(
                config_name="real_task5_7",
                overrides=[
                    "training.hf_autopush.enabled=true",
                    "training.hf_autopush.repo_id=Team-RAMEN/dummy_test_repo",
                    "training.hf_autopush.every_n_ckpt=4",
                ],
            )
        assert cfg.training.hf_autopush.enabled is True
        assert cfg.training.hf_autopush.repo_id == "Team-RAMEN/dummy_test_repo"
        assert cfg.training.hf_autopush.every_n_ckpt == 4


class TestEmaCompilePrefix:
    """Issue #137: EMA が torch.compile 後の key 空間でも動くことを守る。

    Phase K の 6 run は EMA を compile 前に構築して compile 後の model で
    update() していたため、key が一致せず shadow が初期化のまま保存された。
    推論がそれを load して未学習重みで走った。
    """

    @staticmethod
    def _tiny_model():
        import torch

        m = torch.nn.Linear(3, 2)
        torch.nn.init.zeros_(m.weight)   # zero-init 層 (output_proj 相当) を模す
        torch.nn.init.ones_(m.bias)
        return m

    class _CompiledLike:
        """`named_parameters()` が `_orig_mod.` prefix を返す wrapper (compile 相当)。"""

        def __init__(self, inner):
            self._inner = inner

        def named_parameters(self):
            for name, p in self._inner.named_parameters():
                yield f"_orig_mod.{name}", p

    def test_update_matches_shadow_after_compile_wrapper(self) -> None:
        import torch

        from model.ramen_ori.train import EMA

        model = self._tiny_model()
        ema = EMA(model, decay=0.5)
        compiled = self._CompiledLike(model)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(compiled)          # prefix 付きでも一致すること
        assert float(ema.shadow["weight"].abs().max()) > 0.0

    def test_update_fails_fast_when_no_key_matches(self) -> None:
        import pytest as _pytest

        from model.ramen_ori.train import EMA

        model = self._tiny_model()
        ema = EMA(model, decay=0.5)
        ema.shadow = {"totally.different.key": ema.shadow["weight"]}
        with _pytest.raises(RuntimeError, match="0 of 1 shadow tensors updated"):
            ema.update(model)

    def test_apply_to_swaps_through_compile_wrapper(self) -> None:
        import torch

        from model.ramen_ori.train import EMA

        model = self._tiny_model()
        ema = EMA(model, decay=0.5)
        with torch.no_grad():
            ema.shadow["bias"].fill_(7.0)
            model.bias.fill_(1.0)
        compiled = self._CompiledLike(model)
        ema.apply_to(compiled)
        assert float(model.bias[0].detach()) == 7.0
        ema.restore(compiled)
        assert float(model.bias[0].detach()) == 1.0

    def test_apply_to_fails_fast_when_no_key_matches(self) -> None:
        import pytest as _pytest

        from model.ramen_ori.train import EMA

        model = self._tiny_model()
        ema = EMA(model, decay=0.5)
        ema.shadow = {"totally.different.key": ema.shadow["weight"]}
        with _pytest.raises(RuntimeError, match="matched 0 of"):
            ema.apply_to(model)


class TestEmaStrictKeys:
    """Issue #141 RO-9: 一部の key だけずれても止める (partial-train で学習する param の範囲が変わるため)。"""

    @staticmethod
    def _model():
        import torch

        return torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))

    def test_learnable_param_without_shadow_raises(self) -> None:
        import pytest as _pytest

        from model.ramen_ori.train import EMA

        model = self._model()
        ema = EMA(model, decay=0.5)
        del ema.shadow["1.bias"]
        with _pytest.raises(RuntimeError, match="learnable params without shadow"):
            ema.update(model)

    def test_shadow_not_updated_raises(self) -> None:
        """EMA を作った後に param を凍結すると、その shadow は更新されない → 止める。"""
        import pytest as _pytest

        from model.ramen_ori.train import EMA

        model = self._model()
        ema = EMA(model, decay=0.5)
        model[1].weight.requires_grad_(False)
        with _pytest.raises(RuntimeError, match="shadow not updated"):
            ema.update(model)

    def test_partial_apply_raises_before_swapping(self) -> None:
        """一部の key が model に無ければ、1 つも入れ替えずに止める (raw と EMA が混ざった model を残さない)。"""
        import pytest as _pytest
        import torch

        from model.ramen_ori.train import EMA

        model = self._model()
        ema = EMA(model, decay=0.5)
        with torch.no_grad():
            for v in ema.shadow.values():
                v.fill_(9.0)
        ema.shadow["extra.key"] = torch.zeros(1)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with _pytest.raises(RuntimeError, match="matched 4 of 5"):
            ema.apply_to(model)
        for k, v in model.state_dict().items():
            assert torch.equal(v, before[k])
        assert ema._backup is None

    def test_all_keys_ok(self) -> None:
        from model.ramen_ori.train import EMA

        model = self._model()
        ema = EMA(model, decay=0.5)
        ema.update(model)
        with ema.applied(model):
            pass


class TestSetTrainingStats:
    """Issue #141 RO-4 / RO-6 / RO-16: 起動時に全 frame から skill ごとに同じ重みの統計を計算して model に入れる。"""

    class _FakeDataset:
        def __init__(self) -> None:
            rng = np.random.default_rng(0)
            self.items = [
                {"state": rng.normal(sid, 1.0, 71).astype(np.float32),
                 "action": rng.normal(-sid, 1.0, (4, 16)).astype(np.float32),
                 "action_waist_teacher": rng.normal(0.0, 0.05, (4, 3)).astype(np.float32),
                 "skill_id": sid}
                for sid, n in ((4, 40), (5, 8)) for _ in range(n)
            ]

        def __len__(self) -> int:
            return len(self.items)

        def state_action_item(self, i: int) -> dict:
            return self.items[i]

    class _Recorder:
        """set_*_stats を記録する model の代わり (正規化 on、FK の loss は fk_loss で切り替え)。"""

        def __init__(self, fk_loss: bool = False) -> None:
            self.state_normalizer = object()
            self.fk_normalizer = object() if fk_loss else None
            self.fk_stats = None

        def set_normalization_stats(self, state_mean, state_std, action_mean, action_std) -> None:
            self.stats = (state_mean, state_std, action_mean, action_std)

        def set_fk_stats(self, fk_mean, fk_std) -> None:
            self.fk_stats = (fk_mean, fk_std)

    def test_sets_equal_weight_stats_and_logs(self, capsys) -> None:
        from model.ramen_ori.train import _set_training_stats

        ds = self._FakeDataset()
        rec = self._Recorder()
        _set_training_stats(rec, ds, std_min=1e-3, num_workers=0)
        state_mean, _, action_mean, _ = rec.stats
        s4 = np.stack([it["state"] for it in ds.items if it["skill_id"] == 4]).mean(axis=0)
        s5 = np.stack([it["state"] for it in ds.items if it["skill_id"] == 5]).mean(axis=0)
        np.testing.assert_allclose(state_mean, (s4 + s5) / 2, rtol=1e-5)   # frame 数ではなく skill で等分
        a4 = np.stack([it["action"][0] for it in ds.items if it["skill_id"] == 4]).mean(axis=0)
        a5 = np.stack([it["action"][0] for it in ds.items if it["skill_id"] == 5]).mean(axis=0)
        np.testing.assert_allclose(action_mean, (a4 + a5) / 2, rtol=1e-5)
        out = capsys.readouterr().out
        assert "[norm] stats from 48 frames" in out
        assert "skill rotate_table_base" in out and "skill move_table_base" in out
        # state は入力なので WARN を出さない (縮尺の違いは最初の層が吸収する)
        assert not any("WARN" in line for line in out.splitlines() if line.startswith("[norm] state"))

    def test_warns_only_on_action_dims_below_half(self, capsys) -> None:
        from model.ramen_ori.train import _set_training_stats

        # 2 skill とも同じ分布 (正規化後の std ≈ 1)。skill 5 の action 次元 15 と state 次元 40 だけほぼ動かない
        ds = self._FakeDataset()
        rng = np.random.default_rng(1)
        ds.items = []
        for sid in (4, 5):
            for _ in range(400):
                state = rng.normal(0.0, 1.0, 71).astype(np.float32)
                action = rng.normal(0.0, 1.0, (4, 16)).astype(np.float32)
                if sid == 5:
                    action[0, 15] *= 0.1
                    state[40] *= 0.1
                ds.items.append({"state": state, "action": action,
                                 "action_waist_teacher": np.zeros((4, 3), np.float32), "skill_id": sid})
        _set_training_stats(self._Recorder(), ds, std_min=1e-3, num_workers=0)
        lines = capsys.readouterr().out.splitlines()
        warn = [ln for ln in lines if "WARN" in ln]
        assert len(warn) == 1
        assert warn[0].startswith("[norm] action") and "move_table_base" in warn[0] and "15" in warn[0]

    def test_sets_fk_stats_when_fk_loss_is_on(self, capsys) -> None:
        from model.ramen_ori.fk import FK_FEATURE_DIM
        from model.ramen_ori.train import _set_training_stats

        rec = self._Recorder(fk_loss=True)
        _set_training_stats(rec, self._FakeDataset(), std_min=1e-3, num_workers=0)
        fk_mean, fk_std = rec.fk_stats
        assert fk_mean.shape == (FK_FEATURE_DIM,) and fk_std.shape == (FK_FEATURE_DIM,)
        assert np.all(fk_std > 0)
        assert "[norm] fk std" in capsys.readouterr().out


class TestVisionTrainability:
    """Issue #141: backbone の凍結と partial-train の組み合わせの確認。"""

    @staticmethod
    def _policy(stub_vit, freeze: bool, unfreeze_blocks: bool):
        from model.ramen_ori.tests.test_model import _StubViTWithBlocks, _make_small_policy

        backbone = _StubViTWithBlocks()
        for p in backbone.parameters():   # LingBot の loader と同じく凍結して渡す
            p.requires_grad_(False)
        policy = _make_small_policy(backbone, use_temporal=False, freeze_vision=freeze)
        if unfreeze_blocks:
            for p in policy.vision.backbone.blocks[-4:].parameters():
                p.requires_grad_(True)
        return policy

    def test_frozen_eval_ok(self, stub_vit, capsys) -> None:
        from model.ramen_ori.train import _check_vision_trainability

        _check_vision_trainability(self._policy(stub_vit, freeze=True, unfreeze_blocks=False))
        assert "vision backbone learnable 0.0M (freeze=True)" in capsys.readouterr().out

    def test_partial_train_with_freeze_false_ok(self, stub_vit) -> None:
        from model.ramen_ori.train import _check_vision_trainability

        _check_vision_trainability(self._policy(stub_vit, freeze=False, unfreeze_blocks=True))

    def test_freeze_true_with_partial_train_raises(self, stub_vit) -> None:
        from model.ramen_ori.train import _check_vision_trainability

        with pytest.raises(ValueError, match="勾配が流れない"):
            _check_vision_trainability(self._policy(stub_vit, freeze=True, unfreeze_blocks=True))

    def test_freeze_false_without_partial_train_raises(self, stub_vit) -> None:
        """Phase K の設定 (freeze=false でも loader が凍結して返すので学習されない) は止める。"""
        from model.ramen_ori.train import _check_vision_trainability

        with pytest.raises(ValueError, match="学習する param が無い"):
            _check_vision_trainability(self._policy(stub_vit, freeze=False, unfreeze_blocks=False))


def _compose(name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=name, overrides=overrides or [])


TOP_LEVEL_CONFIGS = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
RETRAIN_RUNS = [
    "c32", "c16", "c32_no_fk", "c32_partial_vision", "c32_state_dropout", "c32_memory", "c32_muon",
    "c32_muon_ns32", "c32_muon_ns32_nocompile",  # c32_muon が 2 回落ちた後の回し直し用 (直交化の計算の経路と compile だけ変える)
]


class TestUnknownConfigKeys:
    """Issue #141: base.yaml に無い key (読むコードが無く黙って無視される) を起動時に止める。"""

    @pytest.mark.parametrize("name", TOP_LEVEL_CONFIGS + [f"retrain/{r}" for r in RETRAIN_RUNS])
    def test_existing_configs_have_no_unknown_keys(self, name) -> None:
        from model.ramen_ori.train import _unknown_config_keys

        assert _unknown_config_keys(_compose(name)) == []

    def test_plus_key_is_detected(self) -> None:
        """Phase K の handoff の書き方 (`+model.l4_weight=15`) は今の key (model.fk_loss.*) に届かない。"""
        from model.ramen_ori.train import _unknown_config_keys

        cfg = _compose("base", ["+model.l4_weight=15", "+training.foo.bar=1"])
        assert _unknown_config_keys(cfg) == ["model.l4_weight", "training.foo"]

    def test_target_block_and_data_are_left_to_constructors(self) -> None:
        """`_target_` の block と data は constructor が知らない引数でエラーにするので見ない。"""
        from model.ramen_ori.train import _unknown_config_keys

        cfg = _compose("base", ["+model.fusion.foo=1", "+data.foo=1", "model.aux_head={_target_: x, a: 1}"])
        assert _unknown_config_keys(cfg) == []


class TestRetrainConfigs:
    """Issue #141 Phase 5: 再学習の yaml (configs/retrain/)。共通の設定と run ごとの違い。"""

    @pytest.mark.parametrize("run", RETRAIN_RUNS)
    def test_common_settings(self, run) -> None:
        cfg = _compose(f"retrain/{run}")
        assert cfg.data.chunk_len == cfg.model.chunk_len        # data は model.chunk_len を参照
        assert cfg.model.temporal is None and cfg.data.load_prev_image is False
        assert cfg.model.normalization.enabled is True and cfg.model.state.input_norm is False
        assert cfg.data.frame_cache_mode == "baked_overlay" and cfg.data.frame_cache_num_variants == 10
        assert cfg.data.auto_precompute_frame_cache is False
        assert cfg.contract.bake_config == "data/bitrobot_lerobot_subtask_datasets/configs/unified_frame_cache.yaml"
        assert cfg.speedup.torch_compile is (run != "c32_muon_ns32_nocompile") and cfg.speedup.torch_compile_mode == "default"
        assert cfg.precision.autocast_dtype == "bfloat16"   # Phase K と同じ (FK は fp32 で計算、fk.py)
        assert cfg.optim.grad_clip_norm == 1.0             # c32 の 1 回目の発散の対策 (base の既定は null)
        assert cfg.model.normalization.state_clip == 10.0
        # c32 の 1・2 回目の発散の対策 (attention の logit が増え続けた)
        assert cfg.model.fusion.qk_norm is True and cfg.model.action_expert.qk_norm is True
        assert cfg.training.batch_size == 32 and cfg.training.max_steps == 100000
        assert cfg.training.balancing.enabled is True
        assert cfg.lr_schedule.enabled and cfg.ema.enabled and cfg.val.enabled and cfg.wandb.enabled
        assert [s.merged_source_root for s in cfg.data.sub_datasets] == [
            f"/nvme/unified_cache/{skill}/src"
            for skill in ("rotate_table_base", "move_table_base", "insert_table_leg")
        ]
        assert cfg.training.hf_autopush.enabled is True
        assert cfg.training.hf_autopush.repo_id == f"Team-RAMEN/IROS2026_RAMEN_hara_ramen_ori_141_{run}"
        assert cfg.wandb.run_name == f"ramen_ori_141_{run}"
        assert cfg.training.ckpt_dir == f"/nvme/train_outputs/ramen_ori_141_{run}"

    @pytest.mark.parametrize(
        "run, chunk_len, l4_enabled, freeze, partial",
        [
            ("c32", 32, True, True, False),
            ("c16", 16, True, True, False),
            ("c32_no_fk", 32, False, True, False),
            ("c32_partial_vision", 32, True, False, True),
            ("c32_state_dropout", 32, True, True, False),
            ("c32_memory", 32, True, True, False),
            ("c32_muon", 32, True, True, False),
            ("c32_muon_ns32", 32, True, True, False),
            ("c32_muon_ns32_nocompile", 32, True, True, False),
        ],
    )
    def test_run_differences(self, run, chunk_len, l4_enabled, freeze, partial) -> None:
        cfg = _compose(f"retrain/{run}")
        assert cfg.model.chunk_len == chunk_len
        assert cfg.model.l4_enabled is l4_enabled
        assert cfg.model.vision.freeze is freeze
        assert cfg.optim.partial_train.enabled is partial
        assert cfg.optim.name == ("muon" if run.startswith("c32_muon") else "adamw")  # 6 本目だけ optimizer を変える
        # 直交化の計算の経路: c32_muon は torch.optim.Muon と同じ既定、回し直し用は fp32 + matmul
        ns = (cfg.optim.muon.ns_dtype, cfg.optim.muon.ns_impl)
        assert ns == (("float32", "matmul") if run.startswith("c32_muon_ns32") else ("bfloat16", "baddbmm"))
        if run == "c32_state_dropout":
            assert dict(cfg.model.state.dropout) == {
                "joint": 0.1, "hand": 0.1, "tracking_err": 0.3, "velocity": 0.3, "ee_pose": 0.3
            }
        else:
            assert cfg.model.state.dropout is None
        if run == "c32_memory":
            assert cfg.data.memory is True
            assert cfg.model.memory._target_ == "model.ramen_ori.memory.MemoryEncoder"
            assert cfg.model.memory.memory_dim == 51 and cfg.model.memory.motion_dropout == 0.3
        else:
            assert cfg.data.memory is False and cfg.model.memory is None
        assert [s.obb_precomputed_hash for s in cfg.data.sub_datasets] == ["ce877e8237be"] * 3


def test_flip_state_dropout_is_c32_state_dropout_on_flip_only() -> None:
    """flip だけの学習は c32_state_dropout と data の sub・保存先・名前だけが違う (学習手法は同じ)。"""
    from omegaconf import OmegaConf

    from model.ramen_ori.train import _unknown_config_keys

    flip = _compose("retrain/flip_state_dropout")
    assert _unknown_config_keys(flip) == []
    assert [s.merged_source_root for s in flip.data.sub_datasets] == ["/nvme/unified_cache/flip_table/src"]
    assert flip.training.max_steps == 100000
    assert flip.training.ckpt_dir == "/nvme/train_outputs/ramen_ori_flip_state_dropout"
    assert flip.training.hf_autopush.repo_id == "Team-RAMEN/IROS2026_RAMEN_hara_ramen_ori_flip_state_dropout"
    assert flip.wandb.run_name == "ramen_ori_flip_state_dropout"

    def _rest(cfg) -> dict:
        c = OmegaConf.to_container(cfg, resolve=True)
        del c["data"]["sub_datasets"], c["training"]["ckpt_dir"], c["training"]["hf_autopush"]["repo_id"]
        del c["wandb"]["run_name"]
        return c

    assert _rest(flip) == _rest(_compose("retrain/c32_state_dropout"))


def test_multi_data_yamls_write_only_locations_in_subs() -> None:
    """Issue #141 RO-10: data yaml の sub に共通の key (null を含む) を書かない。"""
    import yaml

    from model.ramen_ori.data_lerobot import _SUB_DATASET_KEYS

    for path in sorted((CONFIG_DIR / "data").glob("*.yaml")):
        for sub in yaml.safe_load(path.read_text()).get("sub_datasets", []):
            assert set(sub) <= _SUB_DATASET_KEYS, f"{path.name}: {sorted(set(sub) - _SUB_DATASET_KEYS)}"


class TestMemoryStats:
    """Issue #141 Phase 7: memory の切り詰めの範囲 (skill ごとに同じ数の frame の p1 / p99) と、切り詰めた後の統計。"""

    class _Dataset:
        def __init__(self, table, skill_ids) -> None:
            self._arrays = (table, skill_ids)

        def memory_arrays(self):
            return self._arrays

    class _Model:
        def set_memory_stats(self, low, high, mean, std) -> None:
            self.stats = (low, high, mean, std)

    def test_skill_balanced_clip_and_moments(self, capsys) -> None:
        from model.ramen_ori.memory_features import MEMORY_DIM
        from model.ramen_ori.train import _set_memory_stats

        rng = np.random.default_rng(0)
        # skill 0 は 9000 frame で値 0 付近、skill 4 は 1000 frame で値 10 付近。frame 数の比では skill 0 に偏る
        table = np.concatenate([rng.normal(0, 1, (9000, MEMORY_DIM)), rng.normal(10, 1, (1000, MEMORY_DIM))]).astype(np.float32)
        skill_ids = np.array([0] * 9000 + [4] * 1000)
        model = self._Model()
        _set_memory_stats(model, self._Dataset(table, skill_ids), std_min=1e-3)
        low, high, mean, std = model.stats
        assert np.all(low < 0) and np.all(high > 10)     # 2 つの skill を同じ数ずつ引いた範囲
        np.testing.assert_allclose(mean, 5.0, atol=0.2)   # 切り詰めた後、skill ごとに同じ重みの平均
        assert "見えている割合" in capsys.readouterr().out


class TestSkillBalancedValPositions:
    """Issue #155 §2: val が最初の sub しか読まない問題の修正。

    `RamenOriMultiSplitView` は sub を順に並べるので、`shuffle=False` の val loader を
    先頭 N batch だけ回すと最初の skill しか読まない。実測では 6 skill 統合 cache の
    val 先頭 640 sample が rotate_table_base 100% だった。
    """

    @staticmethod
    def _skills_in_sub_order(counts: dict[int, int]) -> np.ndarray:
        """sub が順に並んだ val view を模す (skill_id → 件数)。"""
        return np.concatenate([np.full(n, s, dtype=np.int64) for s, n in counts.items()])

    def test_every_skill_gets_the_same_number_of_batches(self) -> None:
        from model.ramen_ori.train import skill_balanced_val_positions

        # 実際の 6 skill の val 件数 (frames × 0.1)
        counts = {4: 14687, 5: 4654, 0: 48246, 3: 23256, 1: 21342, 2: 23263}
        val_skills = self._skills_in_sub_order(counts)

        pos = skill_balanced_val_positions(val_skills, batch_size=32, max_batches=42)

        picked = val_skills[pos]
        per_skill = {int(s): int((picked == s).sum()) for s in np.unique(picked)}
        assert set(per_skill) == set(counts), "6 skill 全部が入ること"
        # 42 // 6 = 7 batch × 32 = 224 sample ずつ
        assert set(per_skill.values()) == {224}
        assert len(pos) == 224 * 6

    def test_each_batch_is_a_single_skill(self) -> None:
        """per-skill の val を batch の skill_id だけで出せるよう、batch が skill で揃うこと。"""
        from model.ramen_ori.train import skill_balanced_val_positions

        counts = {4: 5000, 5: 5000, 0: 5000}
        val_skills = self._skills_in_sub_order(counts)
        batch_size = 32

        pos = skill_balanced_val_positions(val_skills, batch_size=batch_size, max_batches=9)

        picked = val_skills[pos]
        for b in range(len(pos) // batch_size):
            batch = picked[b * batch_size : (b + 1) * batch_size]
            assert len(np.unique(batch)) == 1, f"batch {b} に skill が混ざっている"

    def test_positions_are_stable_across_calls(self) -> None:
        """eval ごとに引く sample が変わると val 曲線がぶれ、ckpt を選べない。"""
        from model.ramen_ori.train import skill_balanced_val_positions

        val_skills = self._skills_in_sub_order({4: 3000, 0: 9000})
        a = skill_balanced_val_positions(val_skills, batch_size=32, max_batches=4)
        b = skill_balanced_val_positions(val_skills, batch_size=32, max_batches=4)
        np.testing.assert_array_equal(a, b)

    def test_positions_span_the_whole_split_not_just_the_head(self) -> None:
        """等間隔に抜くので、各 skill の val の端から端まで使うこと。"""
        from model.ramen_ori.train import skill_balanced_val_positions

        val_skills = self._skills_in_sub_order({4: 10000})
        pos = skill_balanced_val_positions(val_skills, batch_size=32, max_batches=2)
        assert pos.min() == 0
        assert pos.max() == 9999

    def test_a_skill_smaller_than_the_quota_is_not_repeated(self) -> None:
        """val が少ない skill は持っている分だけ取り、同じ frame を水増ししない。"""
        from model.ramen_ori.train import skill_balanced_val_positions

        val_skills = self._skills_in_sub_order({4: 10000, 5: 50})
        pos = skill_balanced_val_positions(val_skills, batch_size=32, max_batches=4)
        picked = val_skills[pos]
        assert int((picked == 5).sum()) == 50
        assert len(np.unique(pos)) == len(pos), "同じ位置を 2 回入れない"

    def test_single_skill_dataset_still_works(self) -> None:
        """β 以前の単一 skill の config でも壊れないこと。"""
        from model.ramen_ori.train import skill_balanced_val_positions

        val_skills = self._skills_in_sub_order({4: 5000})
        pos = skill_balanced_val_positions(val_skills, batch_size=32, max_batches=10)
        assert len(pos) == 320
        assert set(val_skills[pos].tolist()) == {4}

    def test_fewer_batches_than_skills_is_rejected(self) -> None:
        """1 skill 1 batch に足りないと、後ろの skill が val から黙って抜けるので止める。"""
        from model.ramen_ori.train import skill_balanced_val_positions

        val_skills = self._skills_in_sub_order({4: 1000, 5: 1000, 0: 1000})
        with pytest.raises(ValueError, match="smaller than the number of skills"):
            skill_balanced_val_positions(val_skills, batch_size=32, max_batches=2)


class TestFlipFineTuneConfigs:
    """Issue #183: all6 150k から flip へ追加学習する 2 本 (retrain/flip_ft_fk、flip_ft_fk_progress)。"""

    @staticmethod
    def _flat(cfg) -> dict:
        from omegaconf import OmegaConf

        def _walk(d, prefix=""):
            out = {}
            for k, v in d.items():
                key = f"{prefix}{k}"
                if isinstance(v, dict):
                    out.update(_walk(v, key + "."))
                else:
                    out[key] = v
            return out

        return _walk(OmegaConf.to_container(cfg, resolve=True))

    def test_no_unknown_keys(self) -> None:
        from model.ramen_ori.train import _unknown_config_keys

        for run in ("flip_ft_fk", "flip_ft_fk_progress"):
            assert _unknown_config_keys(_compose(f"retrain/{run}")) == []

    def test_ft_differs_from_c32_state_dropout_only_where_intended(self) -> None:
        """学習手法は c32_state_dropout のまま。違うのは data (flip・FK の手先)・初期値・LR・EMA・val・保存先だけ。"""
        ft, base = self._flat(_compose("retrain/flip_ft_fk")), self._flat(_compose("retrain/c32_state_dropout"))
        diff = {k for k in set(ft) | set(base) if ft.get(k) != base.get(k)}
        assert diff == {
            "data.sub_datasets",
            "data.ee_state_source",
            "training.init_from",
            "training.max_steps",
            "training.ckpt_every",
            "training.ckpt_dir",
            "training.hf_autopush.repo_id",
            "optim.lr",
            "lr_schedule.warmup_steps",
            "lr_schedule.cosine_end_lr",
            "ema.decay",
            "val.val_every",
            "val.at_start",
            "wandb.run_name",
        }
        assert ft["data.ee_state_source"] == "fk" and ft["training.init_from_weights"] == "ema"
        assert ft["training.init_from"].endswith("all6_state_dropout_17b5658/ckpt_step_150000.pt")
        assert (ft["training.max_steps"], ft["optim.lr"], ft["ema.decay"]) == (20000, 3.0e-5, 0.999)
        assert ft["training.hf_autopush.repo_id"] == "Team-RAMEN/IROS2026_RAMEN_hara_ramen_ori_flip_ft_fk"

    def test_progress_adds_only_the_aux_head(self) -> None:
        prog, ft = self._flat(_compose("retrain/flip_ft_fk_progress")), self._flat(_compose("retrain/flip_ft_fk"))
        diff = {k for k in set(prog) | set(ft) if prog.get(k) != ft.get(k)}
        assert diff == {
            "data.progress_target",
            "model.aux_head._target_",   # ft は null (flat では model.aux_head = None で、prog 側には無い key)
            "model.aux_head.d_model",
            "model.aux_head.hidden_dim",
            "model.aux_head.num_heads",
            "model.aux_weight",
            "training.ckpt_dir",
            "training.hf_autopush.repo_id",
            "wandb.run_name",
        }
        assert prog["model.aux_head._target_"] == "model.ramen_ori.aux_progress.ProgressHead"
        assert prog["model.aux_head.d_model"] == prog["model.d_model"]


class TestVenueArmConfigs:
    """Issue #188: 会場の腕に慣れさせる追加学習 5 本 (retrain/venue_*)。flip_ft_fk と同じ追加学習に、会場の腕の置き換えを足しただけ。"""

    RUNS = {
        # run: (skill, 初期値の ckpt の末尾)
        "venue_rotate_141": ("rotate_table_base", "141_c32_state_dropout_b19c777/ckpt_step_100000.pt"),
        "venue_rotate_all6": ("rotate_table_base", "all6_state_dropout_17b5658/ckpt_step_300000.pt"),
        "venue_pick_all6": ("pick_table_leg", "all6_state_dropout_17b5658/ckpt_step_300000.pt"),
        "venue_flip_all6": ("flip_table", "all6_state_dropout_17b5658/ckpt_step_150000.pt"),
        "venue_insert_all6": ("insert_table_leg", "all6_state_dropout_17b5658/ckpt_step_400000.pt"),
    }

    @pytest.mark.parametrize("run", sorted(RUNS))
    def test_no_unknown_keys(self, run) -> None:
        from model.ramen_ori.train import _unknown_config_keys

        assert _unknown_config_keys(_compose(f"retrain/{run}")) == []

    @pytest.mark.parametrize("run", sorted(RUNS))
    def test_differs_from_flip_ft_fk_only_where_intended(self, run) -> None:
        """学習手法・追加学習の設定は flip_ft_fk と同じ。違うのは会場の腕・data・初期値・保存先だけ。"""
        flat = TestFlipFineTuneConfigs._flat
        cfg, ft = flat(_compose(f"retrain/{run}")), flat(_compose("retrain/flip_ft_fk"))
        diff = {k for k in set(cfg) | set(ft) if cfg.get(k) != ft.get(k)}
        skill, init = self.RUNS[run]
        intended = {
            # eval_tau は null (flip_ft_fk には key が無く、.get が同じ None になる)
            "data.venue_arm.prob", "data.venue_arm.taus", "data.venue_arm.preset", "data.venue_arm.workers",
            "training.ckpt_dir", "training.hf_autopush.repo_id", "wandb.run_name",
        }
        if skill != "flip_table":
            intended.add("data.sub_datasets")
        if not init.endswith("ckpt_step_150000.pt"):
            intended.add("training.init_from")
        assert diff == intended
        assert cfg["data.ee_state_source"] == "fk" and cfg["training.init_from_weights"] == "ema"
        assert cfg["training.init_from"] == f"/nvme/init/{init}"
        assert cfg["data.sub_datasets"][0]["repo_id"] == f"local/ramen_ori_{skill}" and len(cfg["data.sub_datasets"]) == 1
        assert (cfg["data.venue_arm.prob"], cfg["data.venue_arm.taus"]) == (0.5, [0.05, 0.1, 0.15, 0.2, 0.3])
        assert cfg["data.venue_arm.preset"] == "standard" and cfg["data.venue_arm.eval_tau"] is None
        assert cfg["training.ckpt_dir"] == f"/nvme/train_outputs/ramen_ori_{run}"
        assert cfg["training.hf_autopush.repo_id"] == f"Team-RAMEN/IROS2026_RAMEN_hara_ramen_ori_{run}"

    def test_venue_arm_config_is_accepted(self) -> None:
        """yaml の venue_arm は dataset が受け付ける形 (VenueArmConfig)。"""
        from omegaconf import OmegaConf

        from model.ramen_ori.venue_arm import VenueArmConfig

        venue = OmegaConf.to_container(_compose("retrain/venue_rotate_141").data.venue_arm, resolve=True)
        assert VenueArmConfig.from_dict(venue).taus == (0.05, 0.1, 0.15, 0.2, 0.3)


class TestFlipSelectConfigs:
    """Issue #188: flip を手順の揃った区間 + 腕の状態のずらしで追加学習する 3 本。venue_flip_all6 との違いはその 2 つだけ。"""

    RUNS = {
        # run: (max_peak_gap_s, sigma_rad, max_rad)
        "flip_sel_a_n15": (None, 0.15, 0.4),
        "flip_sel_a_n30": (None, 0.3, 0.6),
        "flip_sel_c_n15": (2.0, 0.15, 0.4),
    }

    @pytest.mark.parametrize("run", sorted(RUNS))
    def test_differs_from_venue_flip_all6_only_in_selection_and_noise(self, run) -> None:
        flat = TestFlipFineTuneConfigs._flat
        cfg, base = flat(_compose(f"retrain/{run}")), flat(_compose("retrain/venue_flip_all6"))
        diff = {k for k in set(cfg) | set(base) if cfg.get(k) != base.get(k)}
        gap, sigma, mx = self.RUNS[run]
        intended = {
            "data.episode_select.left_peak_first", "data.episode_select.right_push_max_rad",
            "data.state_noise.prob", "data.state_noise.sigma_rad", "data.state_noise.max_rad",
            "training.ckpt_dir", "training.hf_autopush.repo_id", "wandb.run_name",
        }
        if gap is not None:
            intended.add("data.episode_select.max_peak_gap_s")
        assert diff == intended
        assert (cfg["data.episode_select.left_peak_first"], cfg["data.episode_select.right_push_max_rad"]) == (True, -0.8)
        assert cfg["data.episode_select.max_peak_gap_s"] == gap
        assert (cfg["data.state_noise.prob"], cfg["data.state_noise.sigma_rad"], cfg["data.state_noise.max_rad"]) == (0.5, sigma, mx)
        assert cfg["training.init_from"].endswith("all6_state_dropout_17b5658/ckpt_step_150000.pt")
        assert cfg["training.hf_autopush.repo_id"] == f"Team-RAMEN/IROS2026_RAMEN_hara_ramen_ori_{run}"

    @pytest.mark.parametrize("run", sorted(RUNS))
    def test_settings_are_accepted(self, run) -> None:
        from omegaconf import OmegaConf

        from model.ramen_ori.episode_select import EpisodeSelectConfig
        from model.ramen_ori.state_noise import StateNoiseConfig

        data = OmegaConf.to_container(_compose(f"retrain/{run}").data, resolve=True)
        EpisodeSelectConfig.from_dict(data["episode_select"])
        StateNoiseConfig.from_dict(data["state_noise"])


class TestLoadInitWeights:
    """Issue #183: training.init_from (学習済みの ckpt の重みだけを初期値にする)。"""

    @staticmethod
    def _policy(stub_vit, seed: int, aux: bool = False):
        import torch

        from model.ramen_ori.aux_progress import ProgressHead
        from model.ramen_ori.tests.test_model import _make_small_policy, _set_random_stats

        torch.manual_seed(seed)
        head = ProgressHead(d_model=64, hidden_dim=32, num_heads=4) if aux else None
        policy = _make_small_policy(stub_vit, normalization=True, aux_head=head)
        _set_random_stats(policy, action_dim=4, seed=seed)
        return policy

    @staticmethod
    def _save(policy, path: Path, ema_offset: float | None = 1.0) -> Path:
        import torch

        ckpt = {"step": 150000, "model_state_dict": policy.state_dict(), "contract": {"provenance": {"git_commit": "abc"}}}
        if ema_offset is not None:
            ckpt["ema_state_dict"] = {
                n: p.detach() + ema_offset for n, p in policy.named_parameters() if p.requires_grad
            }
        torch.save(ckpt, path)
        return path

    def test_ema_weights_for_params_and_ckpt_buffers(self, stub_vit, tmp_path) -> None:
        """学習する param は EMA の重み、正規化の統計 (buffer) は ckpt のもの。記録を返す。"""
        import torch

        from model.ramen_ori.train import _load_init_weights

        src = self._policy(stub_vit, seed=0)
        path = self._save(src, tmp_path / "ckpt.pt", ema_offset=1.0)
        dst = self._policy(stub_vit, seed=1)
        info = _load_init_weights(dst, str(path), "ema")
        src_p, dst_p = dict(src.named_parameters()), dict(dst.named_parameters())
        for name, p in src_p.items():
            expected = p + 1.0 if p.requires_grad else p
            torch.testing.assert_close(dst_p[name], expected)
        torch.testing.assert_close(dst.state_normalizer.mean, src.state_normalizer.mean)
        assert info == {"path": str(path), "step": 150000, "weights": "ema", "provenance": {"git_commit": "abc"}}

    def test_model_weights(self, stub_vit, tmp_path) -> None:
        import torch

        from model.ramen_ori.train import _load_init_weights

        src = self._policy(stub_vit, seed=0)
        dst = self._policy(stub_vit, seed=1)
        _load_init_weights(dst, str(self._save(src, tmp_path / "ckpt.pt")), "model")
        for name, p in src.named_parameters():
            torch.testing.assert_close(dict(dst.named_parameters())[name], p)

    def test_new_aux_head_is_allowed(self, stub_vit, tmp_path) -> None:
        """ckpt に無い補助 head は新しく作ったまま (all6 の ckpt から flip_ft_fk_progress を始める形)。"""
        import torch

        from model.ramen_ori.train import _load_init_weights

        src = self._policy(stub_vit, seed=0)
        dst = self._policy(stub_vit, seed=1, aux=True)
        head_before = {n: p.detach().clone() for n, p in dst.aux_head.named_parameters()}
        _load_init_weights(dst, str(self._save(src, tmp_path / "ckpt.pt")), "ema")
        for n, p in dst.aux_head.named_parameters():
            torch.testing.assert_close(p, head_before[n])

    def test_other_structure_differences_stop(self, stub_vit, tmp_path) -> None:
        """補助 head 以外の key が欠けている・余っている ckpt は止める (ckpt 側に head がある = 余っている)。"""
        from model.ramen_ori.train import _load_init_weights

        src = self._policy(stub_vit, seed=0, aux=True)
        dst = self._policy(stub_vit, seed=1)
        with pytest.raises(ValueError, match="構造が違う"):
            _load_init_weights(dst, str(self._save(src, tmp_path / "ckpt.pt")), "model")

    def test_ema_requested_but_missing_stops(self, stub_vit, tmp_path) -> None:
        from model.ramen_ori.train import _load_init_weights

        src = self._policy(stub_vit, seed=0)
        with pytest.raises(ValueError, match="ema_state_dict が無い"):
            _load_init_weights(self._policy(stub_vit, seed=1), str(self._save(src, tmp_path / "c.pt", None)), "ema")

    def test_bad_weights_name_and_missing_file(self, stub_vit, tmp_path) -> None:
        from model.ramen_ori.train import _load_init_weights

        dst = self._policy(stub_vit, seed=1)
        with pytest.raises(ValueError, match="'ema' か 'model'"):
            _load_init_weights(dst, str(tmp_path / "x.pt"), "raw")
        with pytest.raises(FileNotFoundError):
            _load_init_weights(dst, str(tmp_path / "missing.pt"), "ema")
