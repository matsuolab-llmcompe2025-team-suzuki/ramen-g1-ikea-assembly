"""Phase 7 smoke: policy_config.yaml loader + factory dispatch (Issue #125)。

- YAML file の schema (variant name / policy_type / mode / ckpt_ref / dtype)
- list_variants() で全 variant 列挙
- load_policy_variant() で PolicyConfig 組立て
- resolve_policy_class() で Policy class dispatch
- Null ckpt / 誤 policy_type 等の error 経路
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inference.desktop.lower_policy.policies.base import CameraKey, PolicyConfig
from inference.desktop.lower_policy.policies.config_loader import (
    PolicyAlternative,
    VariantEntry,
    list_variants,
    load_alternatives_by_skill,
    load_default_variant_by_skill,
    load_model_loading,
    load_policy_variant,
    load_yolo_settings,
    resolve_policy_class,
)
from inference.desktop.lower_policy.policies.groot import (
    Gr00tPolicy,
    CAMERAS as GROOT_CAMS,
)
from inference.desktop.lower_policy.policies.ramen_ori import (
    RamenOriPolicy,
    CAMERAS as RAMEN_CAMS,
)
from inference.desktop.lower_policy.policies.groot_pick_legs import (
    Gr00tPolicyPickLegs,
    CAMERAS as PICK_LEGS_CAMS,
)
from inference.desktop.lower_policy.rtc import RtcConfig


# `rtc:` key 自体を書かない場合を表す sentinel (None は「null 明示」と区別する)
_OMITTED = object()


REAL_CONFIG = Path(
    "inference/desktop/lower_policy/configs/policy_config.yaml"
)


class TestRealConfig:
    """実 policy_config.yaml に対する schema + content check。"""

    def test_config_file_exists(self) -> None:
        assert REAL_CONFIG.exists(), f"config file missing: {REAL_CONFIG}"

    def test_expected_variants_present(self) -> None:
        # GR00T pick_legs
        # (別 embodiment) + skill 別 GR00T + RAMEN-Ori (3 mode + 4 architecture
        # variant)。GR00T は precomputed_token 非対応。
        variants = list_variants(REAL_CONFIG)
        assert "groot_insert_leg_200k" in variants
        assert "groot_rotate_leg_200k" in variants
        assert "groot_rotate_leg_160k" in variants
        assert "groot_insert_and_tighten_160k" in variants
        assert "groot_insert_and_tighten_200k" in variants
        for step in (4, 8, 16):
            assert f"groot_insert_and_tighten_success_140k_steps{step}" in variants
            assert f"groot_insert_and_tighten_success_200k_steps{step}" in variants
        assert "groot_flip_table_n17_4" in variants
        assert "groot_flip_table_n17_2_baseline" in variants
        assert "groot_flip_table_insertion_candidate" in variants
        assert "groot_pick_legs_v1" in variants
        # RAMEN-Ori: Phase 1.11 用 default + Phase K R-6 bench 6 run 対応 3 slot
        assert "ramen_ori_default" in variants
        # Issue #190 (2026-10-08): model を消した slot は未登録であること
        # (GR00T overlay・Phase K bench の 9 slot・#139 の ACT / Diffusion・#141 の 7 本と同じ重みの insert 2 本・
        # 会場の腕の flip・flip の学び直し 2 本。#141 の state_dropout と会場の腕の flip は 2026-10-09)
        for removed in (
            "groot_overlay",
            "groot_overlay_cached",
            "rotate_table_base_ramen_ori_overlay_abs",
            "rotate_table_base_ramen_ori_overlay_rel",
            "rotate_table_base_ramen_ori_coord_abs",
            *(
                f"rotate_table_base_ramen_ori_{run_slot}"
                for run_slot in (
                    "run1_overlay_abs_alpha",
                    "run2_coord_abs_alpha",
                    "run3_overlay_rel_alpha",
                    "run4_overlay_abs_beta",
                    "run5_overlay_rel_beta",
                    "run6_coord_abs_beta",
                )
            ),
            "rotate_table_base_act",
            "rotate_table_base_diffusion",
            "rotate_table_base_ramen_ori_141_c32",
            "rotate_table_base_ramen_ori_141_c16",
            "rotate_table_base_ramen_ori_141_c32_memory",
            "rotate_table_base_ramen_ori_141_c32_muon",
            "rotate_table_base_ramen_ori_141_c32_no_fk",
            "rotate_table_base_ramen_ori_141_c32_partial_vision",
            "insert_table_leg_ramen_ori_141_c32",
            "rotate_table_base_ramen_ori_141_c32_state_dropout",
            "insert_table_leg_ramen_ori_141_c32_state_dropout",
            "flip_table_ramen_ori_venue_all6_20k",
            "flip_table_ramen_ori_flip_sel_a_n30_20k",
            "flip_table_ramen_ori_flip_sel_c_n15_20k",
        ):
            assert removed not in variants
        # Issue #132 Phase B: 削除済 slot は未登録であること
        assert "groot_baseline" not in variants
        assert "rotate_table_base_groot" not in variants
        # Issue #132 Phase D: 削除済 RAMEN-Ori architecture variants + null ckpt 予約
        assert "ramen_ori_g3" not in variants
        assert "ramen_ori_meanflow" not in variants
        assert "ramen_ori_awr" not in variants
        assert "ramen_ori_radio" not in variants
        assert "ramen_ori_token" not in variants
        assert "ramen_ori_overlay" not in variants
        # Phase C+ で 1 slot 版 (rotate_table_base_ramen_ori) は 3 slot に再編されたので absence
        assert "rotate_table_base_ramen_ori" not in variants
        # GR00T + precomputed_token combination は無い
        assert "groot_token" not in variants
        assert "groot_precomputed_token" not in variants

    def test_groot_insert_leg_200k_config(self) -> None:
        entry = load_policy_variant(REAL_CONFIG, "groot_insert_leg_200k")
        assert entry.policy_type == "groot"
        assert entry.policy_config.mode == "none"
        assert entry.policy_config.dtype == "bf16"
        assert len(entry.policy_config.cams) == 3  # LeRobot GR00T 3 cam layout
        assert entry.policy_config.ckpt_ref == (
            "Team-RAMEN/IROS2026_RAMEN_takada_insert_leg_optimal_gr00t_200k"
            "@0f0927bcc1aed6927fb26b82c017009961ee3d35"
        )

    def test_groot_rotate_leg_200k_config(self) -> None:
        entry = load_policy_variant(REAL_CONFIG, "groot_rotate_leg_200k")
        assert entry.policy_type == "groot"
        assert entry.policy_config.mode == "none"
        assert entry.policy_config.dtype == "bf16"
        assert len(entry.policy_config.cams) == 3
        assert entry.policy_config.ckpt_ref == (
            "Team-RAMEN/IROS2026_RAMEN_takada_rotate_leg_to_tighten_optimal_gr00t_200k"
            "@51e306fd19dfbcba81932c6b6f073615fdb54b26"
        )

    @pytest.mark.parametrize(
        ("name", "repo", "revision"),
        (
            (
                "groot_rotate_leg_160k",
                "Team-RAMEN/IROS2026_RAMEN_takada_rotate_leg_to_tighten_optimal_gr00t_160k",
                "4c6bc5c29e3b4f8dfaaefd6285d0fe004780dbda",
            ),
            (
                "groot_insert_and_tighten_160k",
                "Team-RAMEN/IROS2026_RAMEN_takada_insert_and_tighten_optimal_gr00t_160k",
                "23ea1d01677107511d6de2d28a0408e13aaf8fa9",
            ),
            (
                "groot_insert_and_tighten_200k",
                "Team-RAMEN/IROS2026_RAMEN_takada_insert_and_tighten_optimal_gr00t_200k",
                "3988504d73e6cca137c12c1c43b7354bed67128c",
            ),
        ),
    )
    def test_new_takada_groot_variants(
        self, name: str, repo: str, revision: str
    ) -> None:
        entry = load_policy_variant(REAL_CONFIG, name)
        assert entry.policy_type == "groot"
        assert entry.policy_config.mode == "none"
        assert entry.policy_config.dtype == "bf16"
        assert entry.policy_config.cams == GROOT_CAMS
        assert entry.policy_config.ckpt_ref == f"{repo}@{revision}"
        assert entry.policy_config.replan_family == "groot_relative_eef_v1"
        # Issue #137: chunk_len 16 に対し重なりを作るため 8 (旧 10)。
        assert entry.policy_config.execution_steps == 8

    @pytest.mark.parametrize(
        ("training_step", "revision"),
        (
            (140, "d7ba8012df30f0128e66b191092f049aed1cb8cf"),
            (200, "e8b349b1651876a14c554ce289b453f375f24df5"),
        ),
    )
    @pytest.mark.parametrize("execution_steps", (4, 8, 16))
    def test_success_optimal_insert_and_tighten_variants(
        self, training_step: int, revision: str, execution_steps: int
    ) -> None:
        name = (
            f"groot_insert_and_tighten_success_{training_step}k_"
            f"steps{execution_steps}"
        )
        entry = load_policy_variant(REAL_CONFIG, name)
        assert entry.policy_type == "groot"
        assert entry.policy_config.mode == "none"
        assert entry.policy_config.dtype == "bf16"
        assert entry.policy_config.cams == GROOT_CAMS
        assert entry.policy_config.ckpt_ref == (
            "Team-RAMEN/IROS2026_RAMEN_takada_insert_and_tighten_"
            f"optimal_success_gr00t_{training_step}k@{revision}"
        )
        assert entry.policy_config.replan_family == "groot_relative_eef_v1"
        assert entry.policy_config.execution_steps == execution_steps

    def test_groot_flip_table_n17_4_config(self) -> None:
        entry = load_policy_variant(REAL_CONFIG, "groot_flip_table_n17_4")
        assert entry.policy_type == "groot"
        assert entry.policy_config.mode == "none"
        assert entry.policy_config.dtype == "bf16"
        assert len(entry.policy_config.cams) == 3
        assert entry.policy_config.ckpt_ref == (
            "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_groot_n17_4"
            "@737ce9179ee2bba8594687a8603f9b8ab1b33281"
        )

    def test_groot_flip_table_n17_2_baseline_config(self) -> None:
        entry = load_policy_variant(
            REAL_CONFIG, "groot_flip_table_n17_2_baseline"
        )
        assert entry.policy_type == "groot"
        assert entry.policy_config.ckpt_ref == (
            "Team-RAMEN/"
            "IROS2026_RAMEN_suzuki_flip_table_groot_n17_2_baseline_checkpoints"
            "@1a408d87eda8d01f9b79113f1aed97a5d0811bff"
        )
        assert entry.policy_config.checkpoint_subdir == (
            "checkpoints/020000/pretrained_model"
        )
        assert entry.policy_config.temporal_lambda == -0.1
        # Issue #137: chunk_len 16 に対し重なりを作るため 8 (旧 10)。
        assert entry.policy_config.execution_steps == 8

    def test_groot_flip_table_kori_step100_config(self) -> None:
        entry = load_policy_variant(REAL_CONFIG, "groot_flip_table_kori_step100")
        assert entry.policy_type == "groot"
        assert entry.policy_config.ckpt_ref == (
            ".checkpoints/pana_kori_flip_table_step100_770fc5ed/pretrained_model"
        )
        assert entry.policy_config.dtype == "bf16"
        assert entry.policy_config.replan_family == "groot_relative_eef_v1"
        assert entry.policy_config.execution_steps == 8

    def test_groot_flip_table_insertion_candidate_config(self) -> None:
        entry = load_policy_variant(
            REAL_CONFIG, "groot_flip_table_insertion_candidate"
        )
        assert entry.policy_type == "groot"
        assert entry.policy_config.mode == "none"
        assert entry.policy_config.dtype == "bf16"
        assert entry.policy_config.cams == GROOT_CAMS
        assert entry.policy_config.ckpt_ref == (
            "Team-RAMEN/"
            "IROS2026_RAMEN_suzuki_flip_table_groot_n17_insertion_candidate"
            "@13b0295b1a4b618d504ed8245a00fa31e63f5ab5"
        )
        assert entry.policy_config.replan_family == "groot_relative_eef_v1"
        assert entry.policy_config.temporal_lambda is None
        assert entry.policy_config.execution_steps == 4

    def test_groot_pick_legs_v1_config(self) -> None:
        entry = load_policy_variant(REAL_CONFIG, "groot_pick_legs_v1")
        assert entry.policy_type == "groot_pick_legs"
        assert entry.policy_config.mode == "none"
        assert entry.policy_config.dtype == "bf16"
        # pick_legs は 4 cam layout (task_5+7 の 3cam と違う)
        assert len(entry.policy_config.cams) == 4
        assert entry.policy_config.ckpt_ref == (
            "Team-RAMEN/groot-n1.7-pick-legs-ver1"
            "@b63d9c49482cfc8998d1a3a0e3e0ac3ceb1d620a"
        )

    def test_ramen_ori_default_config(self) -> None:
        # Issue #132 Phase D: architecture variant (g3/meanflow/awr/radio) と
        # null ckpt 予約 slot (token/overlay) は削除。default のみ保持
        # (entrypoint.py:PHASE1_11_POLICY_VARIANT 依存 + smoke_policy_forward
        # の --variant 例で使用)。
        entry = load_policy_variant(REAL_CONFIG, "ramen_ori_default")
        assert entry.policy_type == "ramen_ori"
        assert entry.policy_config.mode == "none"
        assert entry.policy_config.dtype == "fp32"
        assert entry.policy_config.cams == RAMEN_CAMS  # 4 cam layout
        assert entry.policy_config.hydra_overrides == ()
        assert entry.policy_config.action_space == "abs"  # rel は Phase K slot 用
        assert "ramen_ori_default" in entry.policy_config.ckpt_ref

    def test_overlay_jpeg_subsampling_follows_bake_generation(self) -> None:
        """overlay slot の jpg 色差は、学習した焼き込み cache の保存形式と一致させる。

        #139 より前の焼き込み (GR00T #129 / RAMEN-Ori Phase K) は PIL の 4:2:0、
        #139 の統合 cache は 4:4:4 (既定)。新しい slot は既定のままでよい。
        """
        # #139 より前の焼き込みで学習した slot (GR00T overlay・Phase K bench) は、model を消したので
        # 2026-10-08 (Issue #190) から無い
        pre_139_bakes: set[str] = set()
        subsampling = {
            name: entry.policy_config.overlay_jpeg_subsampling
            for name in list_variants(REAL_CONFIG)
            if (entry := load_policy_variant(REAL_CONFIG, name)).policy_config.mode == "overlay"
        }
        assert {name for name, value in subsampling.items() if value == "4:2:0"} == pre_139_bakes
        assert {name for name, value in subsampling.items() if value == "4:4:4"} == {
            # Issue #155 の 6 skill 統合学習は統合 cache で焼いた overlay で学習した
            *TestAll6EvalSlots.NAMES,
            # Issue #188 の会場の腕の追加学習 (各 skill の統合 cache)
            *TestVenueArmSlots.SLOTS,
            # flip の学び直し (Issue #188) も同じ統合 cache
            *TestFlipSelSlots.SLOTS,
        }

    def test_yolo_ckpt_ref_pinned(self) -> None:
        # v11b SHA pin 済 (Issue #122 D-3 決定値)。slot 共通の YOLO の設定を ramen_ori_default で確認
        # (model は 2026-10-08 に消したが、設定は残している)。
        entry = load_policy_variant(REAL_CONFIG, "ramen_ori_default")
        assert entry.yolo_ckpt_ref is not None
        assert "ramen-g1-ikea-assembly-yolo-obb" in entry.yolo_ckpt_ref
        assert "8221d0aef45d63666537c1f030810fb9b56dd1ed" in entry.yolo_ckpt_ref


class TestDefaultVariantBySkill:
    """本番 run の既定 variant (Issue #148)。"""

    def test_real_config_defaults_are_registered_variants(self) -> None:
        defaults = load_default_variant_by_skill(REAL_CONFIG)
        variants = set(list_variants(REAL_CONFIG))
        assert set(defaults) == {
            "rotate_table_base",
            "pick_table_leg",
            "insert_table_leg",
            "rotate_leg_to_tighten",
            "flip_table",
        }
        assert set(defaults.values()) <= variants

    def test_real_config_defaults_can_be_loaded(self) -> None:
        # 既定に書いた名前は全部 PolicyConfig まで組み立てられる (ckpt_ref 付き)。
        for skill_name, variant in load_default_variant_by_skill(REAL_CONFIG).items():
            entry = load_policy_variant(REAL_CONFIG, variant)
            assert entry.policy_config.ckpt_ref, skill_name

    def test_missing_section(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump({"policies": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="default_variant_by_skill.*required"):
            load_default_variant_by_skill(path)

    def test_unregistered_variant_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "default_variant_by_skill": {"pick_table_leg": "typo_v9"},
                    "policies": {
                        "groot_pick_legs_v1": {
                            "policy_type": "groot_pick_legs",
                            "mode": "none",
                            "ckpt_ref": "x@1",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="not registered under policies"):
            load_default_variant_by_skill(path)


class TestResolvePolicyClass:
    def test_groot(self) -> None:
        assert resolve_policy_class("groot") is Gr00tPolicy

    def test_ramen_ori(self) -> None:
        assert resolve_policy_class("ramen_ori") is RamenOriPolicy

    def test_groot_pick_legs(self) -> None:
        assert resolve_policy_class("groot_pick_legs") is Gr00tPolicyPickLegs

    def test_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown policy_type"):
            resolve_policy_class("nvidia_pi0")


class TestSyntheticConfig:
    """小さい synthetic YAML で error 経路 test (real config を汚さない)。"""

    def _write_yaml(self, tmp_path: Path, data: dict) -> Path:
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    def test_unknown_variant(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            {"policies": {"foo": {"policy_type": "groot", "mode": "none", "ckpt_ref": "x"}}},
        )
        with pytest.raises(KeyError, match="not in policy config"):
            load_policy_variant(path, "bar")

    def test_invalid_policy_type(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            {"policies": {"foo": {"policy_type": "unknown_backbone", "mode": "none", "ckpt_ref": "x"}}},
        )
        with pytest.raises(ValueError, match="policy_type must be"):
            load_policy_variant(path, "foo")

    def test_null_ckpt_ref(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            {"policies": {"foo": {"policy_type": "groot", "mode": "none", "ckpt_ref": None}}},
        )
        with pytest.raises(ValueError, match="ckpt_ref is null"):
            load_policy_variant(path, "foo")

    def test_no_policies_section(self, tmp_path: Path) -> None:
        path = self._write_yaml(tmp_path, {"other": {}})
        # policies section 無し → empty variants
        assert list_variants(path) == []

    def test_default_device_cuda(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            {
                "policies": {
                    "foo": {
                        "policy_type": "ramen_ori",
                        "mode": "none",
                        "ckpt_ref": "x",
                    }
                }
            },
        )
        entry = load_policy_variant(path, "foo")
        assert entry.policy_config.device == "cuda"

    def test_default_dtype_fp32(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            {
                "policies": {
                    "foo": {
                        "policy_type": "ramen_ori",
                        "mode": "none",
                        "ckpt_ref": "x",
                    }
                }
            },
        )
        entry = load_policy_variant(path, "foo")
        assert entry.policy_config.dtype == "fp32"

    def test_action_space_rel_rejected_for_groot(self, tmp_path: Path) -> None:
        # Issue #132 Phase C: action_space=rel は RAMEN-Ori 専用、GR00T では reject。
        path = self._write_yaml(
            tmp_path,
            {
                "policies": {
                    "foo": {
                        "policy_type": "groot",
                        "mode": "none",
                        "ckpt_ref": "x",
                        "action_space": "rel",
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="policy_type='ramen_ori'"):
            load_policy_variant(path, "foo")

    def test_action_space_invalid_rejected(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            {
                "policies": {
                    "foo": {
                        "policy_type": "ramen_ori",
                        "mode": "none",
                        "ckpt_ref": "x",
                        "action_space": "delta_ee",
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="'abs' or 'rel'"):
            load_policy_variant(path, "foo")

    def test_action_space_default_abs(self, tmp_path: Path) -> None:
        # Issue #132 Phase C: field 未指定なら default "abs" (既存 slot backward compat)
        path = self._write_yaml(
            tmp_path,
            {
                "policies": {
                    "foo": {
                        "policy_type": "ramen_ori",
                        "mode": "none",
                        "ckpt_ref": "x",
                    }
                }
            },
        )
        entry = load_policy_variant(path, "foo")
        assert entry.policy_config.action_space == "abs"

    def test_cams_yaml_override(self, tmp_path: Path) -> None:
        # Issue #132 Phase C: slot YAML で cams: [head_left, wrist_left, wrist_right]
        # 指定時、default 4 cam を override して 3 cam layout になる。
        path = self._write_yaml(
            tmp_path,
            {
                "policies": {
                    "foo": {
                        "policy_type": "ramen_ori",
                        "mode": "none",
                        "ckpt_ref": "x",
                        "cams": ["head_left", "wrist_left", "wrist_right"],
                    }
                }
            },
        )
        entry = load_policy_variant(path, "foo")
        assert tuple(c.value for c in entry.policy_config.cams) == (
            "head_left",
            "wrist_left",
            "wrist_right",
        )

    def test_cams_unknown_camera_key_rejected(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            {
                "policies": {
                    "foo": {
                        "policy_type": "ramen_ori",
                        "mode": "none",
                        "ckpt_ref": "x",
                        "cams": ["head_middle"],
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="unknown CameraKey"):
            load_policy_variant(path, "foo")

    def _overlay_variant(self, **extra) -> dict:
        return {
            "policies": {
                "foo": {"policy_type": "groot", "mode": "overlay", "ckpt_ref": "x", **extra}
            }
        }

    def test_overlay_jpeg_subsampling_default_444(self, tmp_path: Path) -> None:
        path = self._write_yaml(tmp_path, self._overlay_variant())
        entry = load_policy_variant(path, "foo")
        assert entry.policy_config.overlay_jpeg_subsampling == "4:4:4"

    def test_overlay_jpeg_subsampling_explicit_420(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path, self._overlay_variant(overlay_jpeg_subsampling="4:2:0")
        )
        entry = load_policy_variant(path, "foo")
        assert entry.policy_config.overlay_jpeg_subsampling == "4:2:0"

    def test_overlay_jpeg_subsampling_unquoted_yaml_rejected(self, tmp_path: Path) -> None:
        # YAML 1.1 はクォート無しの 4:2:0 を 60 進数の int (14520) として読む
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "policies:\n"
            "  foo:\n"
            "    policy_type: groot\n"
            "    mode: overlay\n"
            "    ckpt_ref: x\n"
            "    overlay_jpeg_subsampling: 4:2:0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="quote it"):
            load_policy_variant(path, "foo")

    def test_overlay_jpeg_subsampling_unknown_value_rejected(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path, self._overlay_variant(overlay_jpeg_subsampling="4:2:2")
        )
        with pytest.raises(ValueError, match="overlay_jpeg_subsampling must be one of"):
            load_policy_variant(path, "foo")

    def test_overlay_jpeg_subsampling_rejected_outside_overlay_mode(
        self, tmp_path: Path
    ) -> None:
        data = self._overlay_variant(overlay_jpeg_subsampling="4:2:0")
        data["policies"]["foo"]["mode"] = "none"
        path = self._write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="only used in mode='overlay'"):
            load_policy_variant(path, "foo")


class TestRtcConfigParsing:
    """Issue #137 Phase A: `rtc:` ブロックの parse + 制約。"""

    def _write_yaml(self, tmp_path: Path, data: dict) -> Path:
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    def _variant(self, rtc: object, policy_type: str = "groot") -> dict:
        entry: dict = {
            "policy_type": policy_type,
            "mode": "none",
            "ckpt_ref": "x",
        }
        if rtc is not _OMITTED:
            entry["rtc"] = rtc
        return {"policies": {"foo": entry}}

    def test_default_is_disabled(self, tmp_path: Path) -> None:
        # rtc 未記載の既存 variant は従来経路のまま (回帰ゼロ)
        path = self._write_yaml(tmp_path, self._variant(_OMITTED))
        cfg = load_policy_variant(path, "foo").policy_config
        assert cfg.rtc == RtcConfig()
        assert cfg.rtc.enabled is False

    def test_real_config_has_rtc_disabled_after_phase_b(self) -> None:
        # B-3/B-4 rejected RTC for relative RAMEN-Ori, and B'-2 was worse than
        # B'-1 for GR00T.  Phase C therefore keeps RTC disabled everywhere.
        for name in list_variants(REAL_CONFIG):
            cfg = load_policy_variant(REAL_CONFIG, name).policy_config
            assert cfg.rtc.enabled is False, f"{name} unexpectedly enables rtc"
            assert cfg.rtc.allow_experimental_relative_action is False

    def test_rejects_rtc_for_relative_ramen_ori(self, tmp_path: Path) -> None:
        data = self._variant(
            {
                "enabled": True,
                "frozen_steps": "auto",
                "overlap_steps": 8,
                "ramp_rate": 6.0,
            },
            policy_type="ramen_ori",
        )
        data["policies"]["foo"]["action_space"] = "rel"
        path = self._write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="unsafe for RAMEN-Ori"):
            load_policy_variant(path, "foo")

    def test_explicit_experimental_relative_rtc_override(self, tmp_path: Path) -> None:
        data = self._variant(
            {
                "enabled": True,
                "frozen_steps": "auto",
                "overlap_steps": 8,
                "ramp_rate": 6.0,
                "allow_experimental_relative_action": True,
            },
            policy_type="ramen_ori",
        )
        data["policies"]["foo"]["action_space"] = "rel"
        path = self._write_yaml(tmp_path, data)
        cfg = load_policy_variant(path, "foo").policy_config
        assert cfg.rtc.enabled is True
        assert cfg.rtc.allow_experimental_relative_action is True

    def test_null_block_is_default(self, tmp_path: Path) -> None:
        path = self._write_yaml(tmp_path, self._variant(None))
        assert load_policy_variant(path, "foo").policy_config.rtc == RtcConfig()

    def test_full_block_parsed(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            self._variant(
                {
                    "enabled": True,
                    "frozen_steps": 3,
                    "overlap_steps": 8,
                    "ramp_rate": 4.5,
                }
            ),
        )
        rtc = load_policy_variant(path, "foo").policy_config.rtc
        assert rtc == RtcConfig(
            enabled=True, frozen_steps=3, overlap_steps=8, ramp_rate=4.5
        )

    def test_auto_frozen_steps(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path, self._variant({"enabled": True, "frozen_steps": "auto"})
        )
        rtc = load_policy_variant(path, "foo").policy_config.rtc
        assert rtc.frozen_steps == "auto"
        assert rtc.overlap_steps is None

    def test_ramen_ori_supported(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path, self._variant({"enabled": True}, policy_type="ramen_ori")
        )
        assert load_policy_variant(path, "foo").policy_config.rtc.enabled is True

    def test_pick_legs_rejected_when_enabled(self, tmp_path: Path) -> None:
        # 別 embodiment / 別 action schema なので silent no-op にせず落とす
        path = self._write_yaml(
            tmp_path, self._variant({"enabled": True}, policy_type="groot_pick_legs")
        )
        with pytest.raises(ValueError, match="rtc.enabled is only supported"):
            load_policy_variant(path, "foo")

    def test_pick_legs_allowed_when_disabled(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path, self._variant({"enabled": False}, policy_type="groot_pick_legs")
        )
        assert load_policy_variant(path, "foo").policy_config.rtc.enabled is False

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        # typo を silent 無視すると「設定したのに効かない」事故になる
        path = self._write_yaml(
            tmp_path, self._variant({"enabled": True, "frozen_step": 3})
        )
        with pytest.raises(ValueError, match="unknown keys"):
            load_policy_variant(path, "foo")

    def test_non_mapping_rejected(self, tmp_path: Path) -> None:
        path = self._write_yaml(tmp_path, self._variant(True))
        with pytest.raises(ValueError, match="rtc must be a mapping"):
            load_policy_variant(path, "foo")

    @pytest.mark.parametrize(
        ("rtc", "match"),
        [
            ({"frozen_steps": -1}, "frozen_steps"),
            ({"frozen_steps": "always"}, "frozen_steps"),
            ({"overlap_steps": 0}, "overlap_steps"),
            ({"frozen_steps": 9, "overlap_steps": 8}, "frozen_steps"),
            ({"ramp_rate": 0}, "ramp_rate"),
            ({"ramp_rate": "fast"}, "ramp_rate"),
            ({"enabled": "yes"}, "enabled"),
        ],
    )
    def test_invalid_values_rejected(
        self, tmp_path: Path, rtc: dict, match: str
    ) -> None:
        path = self._write_yaml(tmp_path, self._variant(rtc))
        with pytest.raises(ValueError, match=match):
            load_policy_variant(path, "foo")


class TestActDiffusionVariant:
    """Issue #139: ACT / Diffusion Policy slot の読込と、非対応設定の起動時 reject。"""

    def _write(self, tmp_path: Path, **extra) -> Path:
        path = tmp_path / "cfg.yaml"
        entry = {"policy_type": "act_diffusion", "mode": "overlay", "ckpt_ref": "x", **extra}
        path.write_text(yaml.safe_dump({"policies": {"foo": entry}}), encoding="utf-8")
        return path

    def test_default_cams_match_training(self, tmp_path: Path) -> None:
        entry = load_policy_variant(self._write(tmp_path), "foo")
        assert entry.policy_type == "act_diffusion"
        assert entry.policy_config.cams == (
            CameraKey.HEAD_LEFT,
            CameraKey.WRIST_LEFT,
            CameraKey.WRIST_RIGHT,
        )

    def test_rtc_rejected(self, tmp_path: Path) -> None:
        # ACT / DP は flow matching ではないので RTC の対象外
        path = self._write(tmp_path, rtc={"enabled": True})
        with pytest.raises(ValueError, match="rtc.enabled is only supported"):
            load_policy_variant(path, "foo")

    def test_relative_action_space_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, action_space="rel")
        with pytest.raises(ValueError, match="policy_type='ramen_ori'"):
            load_policy_variant(path, "foo")

    def test_hydra_overrides_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, hydra_overrides=["model.x=1"])
        with pytest.raises(ValueError, match="hydra_overrides are only supported"):
            load_policy_variant(path, "foo")


class TestYoloSettings:
    """`yolo` section (両経路が共有する検出の設定、Issue #141 INF-10)。"""

    REF = "owner/repo@" + "a" * 40
    FILE = "runs/x/weights/best.pt"

    def _write(self, tmp_path: Path, yolo: dict | None) -> Path:
        data: dict = {"policies": {}}
        if yolo is not None:
            data["yolo"] = yolo
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    def _yolo(self, **overrides) -> dict:
        yolo = {"ckpt_ref": self.REF, "ckpt_file": self.FILE, "conf": 0.25, "imgsz": 800}
        yolo.update(overrides)
        return yolo

    def test_reads_ckpt_ref_ckpt_file_conf_imgsz(self, tmp_path: Path) -> None:
        settings = load_yolo_settings(self._write(tmp_path, self._yolo()))
        assert settings.ckpt_ref == self.REF
        assert settings.ckpt_file == self.FILE
        assert settings.conf == pytest.approx(0.25)
        assert settings.imgsz == 800

    def test_shipped_config_pins_the_training_bake_imgsz(self) -> None:
        """学習 cache の焼き込み (unified_frame_cache.yaml: imgsz 800) と同じ値。"""
        settings = load_yolo_settings(REAL_CONFIG)
        assert settings.imgsz == 800
        assert settings.ckpt_ref.endswith("@8221d0aef45d63666537c1f030810fb9b56dd1ed")

    def test_shipped_config_pins_the_weight_path_inside_the_repo(self) -> None:
        """HF の実物 (@8221d0ae) の .pt は入れ子のこの 1 本 (B7-01、hf_fs で確認)。"""
        settings = load_yolo_settings(REAL_CONFIG)
        assert settings.ckpt_file == "runs/m_lowaug_v11b/weights/best_20260818.pt"

    def test_missing_section_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="'yolo' section is required"):
            load_yolo_settings(self._write(tmp_path, None))

    def test_ckpt_ref_without_revision_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, self._yolo(ckpt_ref="repo"))
        with pytest.raises(ValueError, match="repo@revision"):
            load_yolo_settings(path)

    def test_branch_revision_rejected(self, tmp_path: Path) -> None:
        """branch 名だと cache を使うにも問い合わせが要る。commit hash で固定する。"""
        path = self._write(tmp_path, self._yolo(ckpt_ref="owner/repo@main"))
        with pytest.raises(ValueError, match="commit hash"):
            load_yolo_settings(path)

    def test_missing_ckpt_file_rejected(self, tmp_path: Path) -> None:
        yolo = self._yolo()
        del yolo["ckpt_file"]
        with pytest.raises(ValueError, match="ckpt_file"):
            load_yolo_settings(self._write(tmp_path, yolo))

    def test_imgsz_not_multiple_of_32_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, self._yolo(imgsz=801))
        with pytest.raises(ValueError, match="multiple of 32"):
            load_yolo_settings(path)

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, self._yolo(iou=0.5))
        with pytest.raises(ValueError, match="unknown keys"):
            load_yolo_settings(path)


_ALL6_SKILL_IDS = {
    "insert_table_leg": 0, "flip_table": 1, "rotate_leg_to_tighten": 2,
    "pick_table_leg": 3, "rotate_table_base": 4, "move_table_base": 5,
}


class TestAll6EvalSlots:
    """Issue #155 の 6 skill 統合学習の実機評価 slot (3 step × 6 skill)。

    1 つの repo の途中の ckpt を `ckpt_filename` で選ぶ。取り違えると、別 step や
    別 skill の指令で 1 本を評価したことに気付かないまま消費する。
    """

    REPO = "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-all-skills"
    SKILL_IDS = _ALL6_SKILL_IDS
    NAMES = tuple(f"{skill}_ramen_ori_all6_{s}k" for s in (150, 300, 400) for skill in _ALL6_SKILL_IDS)

    def test_every_skill_and_step_has_a_slot(self) -> None:
        names = set(list_variants(REAL_CONFIG))
        assert set(self.NAMES) <= names
        assert len([n for n in names if "_ramen_ori_all6_" in n]) == 18

    @pytest.mark.parametrize("name", NAMES)
    def test_slot_picks_its_step_and_skill(self, name: str) -> None:
        from model.ramen_ori.skill_mapping import skill_id_name

        entry = load_policy_variant(REAL_CONFIG, name)
        cfg = entry.policy_config
        skill, step = name.split("_ramen_ori_all6_")
        assert entry.policy_type == "ramen_ori"
        assert cfg.skill_id == self.SKILL_IDS[skill]
        assert skill_id_name(cfg.skill_id) == skill       # 学習側の対応表とも一致
        assert cfg.ckpt_filename == f"ckpt_step_{int(step[:-1]) * 1000:06d}.pt"
        assert cfg.ckpt_ref.startswith(self.REPO + "@")
        assert len(cfg.ckpt_ref.split("@", 1)[1]) == 40   # commit に固定

    @pytest.mark.parametrize("name", NAMES)
    def test_slot_matches_the_training_recipe(self, name: str) -> None:
        cfg = load_policy_variant(REAL_CONFIG, name).policy_config
        assert cfg.mode == "overlay"
        assert cfg.overlay_jpeg_subsampling == "4:4:4"
        assert cfg.dtype == "fp32"
        assert cfg.action_space == "abs"
        assert cfg.hydra_overrides == ()
        assert [cam.value for cam in cfg.cams] == ["head_left", "wrist_left", "wrist_right"]
        assert cfg.replan_family == "ramen_ori_flow_v1"
        assert cfg.execution_steps == 8
        assert cfg.temporal_lambda == -0.1
        assert not cfg.rtc.enabled


class TestVenueArmSlots:
    """Issue #188: 会場の腕に慣れさせた追加学習 (20k) の slot。all6 の slot と同じ約束で、ckpt だけが違う。

    4 本のうち flip は 2026-10-09 (Issue #190) に model を消した。
    """

    # slot: (repo, commit)。commit は 20k を push した HF の commit (USB の V1〜V4 と同じ)
    SLOTS = {
        "rotate_table_base_ramen_ori_venue_all6_20k": (
            "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base", "aaa73be26e795376ca9e2fcf2528e0b51b362c67"),
        "pick_table_leg_ramen_ori_venue_all6_20k": (
            "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-pick-table-leg", "5384b7ed641af16edf10f6b1ecf67713146996af"),
        "insert_table_leg_ramen_ori_venue_all6_20k": (
            "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-insert-table-leg", "427d42dd29c0f8a8ff81a3c2319ab7710cdb1f8c"),
    }

    @staticmethod
    def _skill(name: str) -> str:
        return name.removesuffix("_ramen_ori_venue_all6_20k")

    @pytest.mark.parametrize("name", sorted(SLOTS))
    def test_slot_picks_the_20k_ckpt_of_its_commit(self, name: str) -> None:
        entry = load_policy_variant(REAL_CONFIG, name)
        cfg = entry.policy_config
        assert entry.policy_type == "ramen_ori"
        assert cfg.skill_id == _ALL6_SKILL_IDS[self._skill(name)]
        assert cfg.ckpt_filename == "ckpt_step_020000.pt"
        assert tuple(cfg.ckpt_ref.split("@", 1)) == self.SLOTS[name]

    @pytest.mark.parametrize("name", sorted(SLOTS))
    def test_slot_matches_the_all6_slot_except_the_ckpt(self, name: str) -> None:
        """学習の約束 (overlay・JPEG・カメラ・chunk の実行・skill) は all6 の同じ skill の slot と同じ。"""
        venue = load_policy_variant(REAL_CONFIG, name).policy_config
        all6 = load_policy_variant(REAL_CONFIG, f"{self._skill(name)}_ramen_ori_all6_150k").policy_config
        same = ("mode", "overlay_jpeg_subsampling", "dtype", "action_space", "hydra_overrides", "cams",
                "replan_family", "execution_steps", "temporal_lambda", "rtc", "skill_id", "device")
        assert {k: getattr(venue, k) for k in same} == {k: getattr(all6, k) for k in same}

    def test_venue_arm_set_replaces_rotate_insert_but_not_the_hybrid_pick(self) -> None:
        """pick の既定は hybrid (groot_pick_legs_v1 に固定) なので、set には入れない。"""
        defaults = load_default_variant_by_skill(REAL_CONFIG)
        loaded = load_default_variant_by_skill(REAL_CONFIG, variant_set="venue_arm")
        for skill in ("rotate_table_base", "insert_table_leg"):
            assert loaded[skill] == f"{skill}_ramen_ori_venue_all6_20k", skill
        for skill in ("pick_table_leg", "rotate_leg_to_tighten", "flip_table"):
            assert loaded[skill] == defaults[skill], skill

    def test_flip_ft_slots_are_gone(self) -> None:
        """Issue #183 の flip の追加学習 2 本は、学習の手先の FK の関節の位置がずれていたので選べない。"""
        variants = list_variants(REAL_CONFIG)
        assert "flip_table_ramen_ori_flip_ft_fk_20k" not in variants
        assert "flip_table_ramen_ori_flip_ft_fk_progress_20k" not in variants


class TestCkptFilename:
    """Issue #155: slot から repo 内の ckpt file を選ぶ `ckpt_filename`。"""

    def _write(self, tmp_path: Path, policy_type: str) -> Path:
        path = tmp_path / "policy_config.yaml"
        path.write_text(
            "policies:\n"
            "  v:\n"
            f"    policy_type: {policy_type}\n"
            "    mode: overlay\n"
            "    ckpt_ref: Team-RAMEN/x@0123456789abcdef0123456789abcdef01234567\n"
            "    ckpt_filename: ckpt_step_400000.pt\n"
        )
        return path

    def test_ramen_ori_reads_it(self, tmp_path: Path) -> None:
        cfg = load_policy_variant(self._write(tmp_path, "ramen_ori"), "v").policy_config
        assert cfg.ckpt_filename == "ckpt_step_400000.pt"

    def test_other_policies_reject_it(self, tmp_path: Path) -> None:
        """GR00T などは読まないので、書いたら黙って無視されずに止まること。"""
        with pytest.raises(ValueError, match="ckpt_filename is only read by"):
            load_policy_variant(self._write(tmp_path, "groot"), "v")

    @pytest.mark.parametrize("bad", ["", "/abs/ckpt_step_1.pt", "../x.pt"])
    def test_must_be_a_relative_path_without_traversal(self, bad: str) -> None:
        from inference.desktop.lower_policy.policies.base import PolicyConfig

        with pytest.raises(ValueError, match="ckpt_filename"):
            PolicyConfig(mode="overlay", ckpt_ref="x", ckpt_filename=bad)

    def test_default_is_none(self) -> None:
        from inference.desktop.lower_policy.policies.base import PolicyConfig

        assert PolicyConfig(mode="overlay", ckpt_ref="x").ckpt_filename is None


# ---------------------------------------------------------------- variant_sets (Issue #159)
#
# 会場で model の組み合わせを 1 つの名前で切り替える (RAMEN_VARIANT_SET /
# --policy-variant-set)。書いた skill だけ差し替え、未知の名前は起動時に落とす。


def test_every_variant_set_in_the_real_config_loads():
    data = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    defaults = load_default_variant_by_skill(REAL_CONFIG)
    for name in data.get("variant_sets", {}):
        loaded = load_default_variant_by_skill(REAL_CONFIG, variant_set=name)
        assert set(loaded) == set(defaults), name
        for skill, variant in loaded.items():
            assert variant in data["policies"], (name, skill, variant)


def test_the_ramen_ori_set_is_gone():
    """#141 の 2 slot だけの組み合わせ。model を消したので外した (Issue #190、2026-10-09)。"""
    data = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    assert "ramen_ori" not in data["variant_sets"]


def _config_with_sets(tmp_path: Path, sets) -> Path:
    data = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    data["variant_sets"] = sets
    path = tmp_path / "policy_config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_an_unknown_variant_set_is_rejected_with_the_available_names(tmp_path):
    path = _config_with_sets(tmp_path, {"ramen_ori": {"insert_table_leg": "groot_insert_leg_200k"}})
    with pytest.raises(ValueError, match=r"unknown variant_set 'ramen_orii'.*ramen_ori"):
        load_default_variant_by_skill(path, variant_set="ramen_orii")


def test_a_set_naming_an_unknown_skill_or_variant_is_rejected(tmp_path):
    path = _config_with_sets(tmp_path, {"typo_skill": {"insert_leg": "groot_insert_leg_200k"}})
    with pytest.raises(ValueError, match="is not a skill"):
        load_default_variant_by_skill(path, variant_set="typo_skill")
    path = _config_with_sets(tmp_path, {"typo_variant": {"insert_table_leg": "no_such_slot"}})
    with pytest.raises(ValueError, match="not registered"):
        load_default_variant_by_skill(path, variant_set="typo_variant")


# ---------------------------------------------------------------- alternatives_by_skill (Issue #188)
#
# R のやり直しで既定の代わりに選べる候補。候補の中身は後で選ぶので、本物の config は
# 空の list を並べておく。誤った書き方は起動時に落とす。


def _config_with_alternatives(tmp_path: Path, alternatives) -> Path:
    data = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    data["alternatives_by_skill"] = alternatives
    path = tmp_path / "policy_config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_real_config_lists_the_chosen_alternatives():
    """選んだ候補 (2026-09-30、Issue #188)。pick は hybrid の 2 通り (既定は hybrid なし)。締め付けは候補なし。

    台を回すのは前の既定 (#141) を候補にしていたが、2026-10-09 (Issue #190) に model を消したので候補なし。
    """
    venue_pick = "pick_table_leg_ramen_ori_venue_all6_20k"
    alternatives = load_alternatives_by_skill(REAL_CONFIG)
    assert set(alternatives) == set(load_default_variant_by_skill(REAL_CONFIG))
    assert alternatives == {
        "rotate_table_base": (),
        "pick_table_leg": (
            PolicyAlternative("groot_pick_legs_v1", True),
            PolicyAlternative(venue_pick, True),
        ),
        "insert_table_leg": (PolicyAlternative("insert_table_leg_diffusion_optimal_100k"),),
        "rotate_leg_to_tighten": (),
        "flip_table": (PolicyAlternative("groot_flip_table_n17_2_legacy_fk"),),  # 前の既定 (2026-10-10 に a_n15 を既定に)
    }


def test_alternatives_keep_their_order(tmp_path):
    path = _config_with_alternatives(
        tmp_path,
        {"rotate_table_base": ["rotate_table_base_ramen_ori_all6_400k", "rotate_table_base_ramen_ori_all6_300k"]},
    )
    assert load_alternatives_by_skill(path) == {
        "rotate_table_base": (
            PolicyAlternative("rotate_table_base_ramen_ori_all6_400k"),
            PolicyAlternative("rotate_table_base_ramen_ori_all6_300k"),
        ),
    }


def test_a_config_without_the_section_has_no_alternatives(tmp_path):
    data = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    del data["alternatives_by_skill"]
    path = tmp_path / "policy_config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_alternatives_by_skill(path) == {}


@pytest.mark.parametrize(
    ("alternatives", "message"),
    [
        ({"insert_leg": ["groot_insert_leg_200k"]}, "is not a skill"),
        ({"insert_table_leg": ["no_such_slot"]}, "not registered"),
        ({"insert_table_leg": "groot_insert_leg_200k"}, "must be a list"),
        ({"insert_table_leg": ["groot_insert_leg_200k", "groot_insert_leg_200k"]}, "twice"),
        (["insert_table_leg"], "must map skills"),
    ],
)
def test_a_wrong_alternatives_section_is_rejected(tmp_path, alternatives, message):
    path = _config_with_alternatives(tmp_path, alternatives)
    with pytest.raises(ValueError, match=message):
        load_alternatives_by_skill(path)


# ---------------------------------------------------------------- model_loading (Issue #188)
#
# `--gpu-models plan` の読み方。本物の config は pick まで先に読み、全部腕が止まっている
# 間だけ読む (ユーザー決定 2026-09-30)。誤った書き方は起動時に落とす。


def _config_with_model_loading(tmp_path: Path, section) -> Path:
    data = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    if section is None:
        del data["model_loading"]
    else:
        data["model_loading"] = section
    path = tmp_path / "policy_config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_real_config_preloads_through_pick_and_loads_only_while_the_arms_are_still():
    loading = load_model_loading(REAL_CONFIG)
    assert loading.preload_through_skill == "pick_table_leg"
    assert loading.load_while_policy_runs == ()


def test_listed_kinds_keep_their_order(tmp_path):
    path = _config_with_model_loading(
        tmp_path,
        {"preload_through_skill": "rotate_table_base", "load_while_policy_runs": ["vlm", "groot"]},
    )
    loading = load_model_loading(path)
    assert loading.preload_through_skill == "rotate_table_base"
    assert loading.load_while_policy_runs == ("vlm", "groot")


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (None, "section is required"),
        ({"preload_through_skill": "pick_table_leg"}, "must have exactly"),
        (
            {"preload_through_skill": "pick_table_leg", "load_while_policy_runs": [], "x": 1},
            "must have exactly",
        ),
        ({"preload_through_skill": "pick_leg", "load_while_policy_runs": []}, "is not a skill"),
        (
            {"preload_through_skill": "pick_table_leg", "load_while_policy_runs": "groot"},
            "must be a list",
        ),
        (
            {"preload_through_skill": "pick_table_leg", "load_while_policy_runs": ["gr00t"]},
            "unknown kinds",
        ),
        (
            {"preload_through_skill": "pick_table_leg", "load_while_policy_runs": ["vlm", "vlm"]},
            "twice",
        ),
    ],
)
def test_a_wrong_model_loading_section_is_rejected(tmp_path, section, message):
    path = _config_with_model_loading(tmp_path, section)
    with pytest.raises(ValueError, match=message):
        load_model_loading(path)


def test_a_pick_alternative_can_say_it_runs_without_the_hybrid(tmp_path):
    """pick だけ、同じ variant を hybrid なし (学習 policy に全部任せる) でも並べられる (Issue #188)。"""
    venue = "pick_table_leg_ramen_ori_venue_all6_20k"
    path = _config_with_alternatives(
        tmp_path,
        {"pick_table_leg": [venue, {"variant": venue, "pick_leg_hybrid": False}]},
    )
    assert load_alternatives_by_skill(path) == {
        "pick_table_leg": (PolicyAlternative(venue), PolicyAlternative(venue, False)),
    }


@pytest.mark.parametrize(
    ("alternatives", "message"),
    [
        (
            {"insert_table_leg": [{"variant": "groot_insert_leg_200k", "pick_leg_hybrid": False}]},
            "only for pick_table_leg",
        ),
        (
            {"pick_table_leg": [{"variant": "groot_pick_legs_v1", "pick_leg_hybrid": "yes"}]},
            "must be true or false",
        ),
        ({"pick_table_leg": [{"variant": "groot_pick_legs_v1"}]}, "only for pick_table_leg"),
        (
            {"pick_table_leg": [{"variant": "no_such_slot", "pick_leg_hybrid": False}]},
            "not registered",
        ),
        (
            {
                "pick_table_leg": [
                    {"variant": "groot_pick_legs_v1", "pick_leg_hybrid": False},
                    {"variant": "groot_pick_legs_v1", "pick_leg_hybrid": False},
                ]
            },
            "twice",
        ),
    ],
)
def test_a_wrong_pick_mapping_is_rejected(tmp_path, alternatives, message):
    path = _config_with_alternatives(tmp_path, alternatives)
    with pytest.raises(ValueError, match=message):
        load_alternatives_by_skill(path)


def test_a_pick_alternative_can_also_ask_for_the_hybrid(tmp_path):
    """既定が hybrid でない run で、hybrid の候補を並べる (Issue #188)。"""
    path = _config_with_alternatives(
        tmp_path,
        {"pick_table_leg": [{"variant": "groot_pick_legs_v1", "pick_leg_hybrid": True}]},
    )
    assert load_alternatives_by_skill(path) == {
        "pick_table_leg": (PolicyAlternative("groot_pick_legs_v1", True),),
    }


@pytest.mark.parametrize(("value", "expected"), [(None, True), (True, True), (False, False)])
def test_the_pick_mode_default_is_read_from_the_config(tmp_path, value, expected):
    from inference.desktop.lower_policy.policies.config_loader import (
        load_pick_leg_hybrid_default,
    )

    data = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    data.pop("pick_leg_hybrid", None)
    if value is not None:
        data["pick_leg_hybrid"] = value
    path = tmp_path / "policy_config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_pick_leg_hybrid_default(path) is expected  # 無ければ hybrid (Issue #148 からの既定)


def test_a_wrong_pick_mode_default_is_rejected(tmp_path):
    from inference.desktop.lower_policy.policies.config_loader import (
        load_pick_leg_hybrid_default,
    )

    data = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    data["pick_leg_hybrid"] = "no"
    path = tmp_path / "policy_config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="pick_leg_hybrid must be true or false"):
        load_pick_leg_hybrid_default(path)


def test_the_real_config_runs_the_learned_venue_pick_by_default():
    """pick の既定は hybrid なしの会場の腕の RAMEN-Ori (2026-09-30、ユーザー決定)。"""
    from inference.desktop.lower_policy.policies.config_loader import (
        load_pick_leg_hybrid_default,
    )

    assert load_pick_leg_hybrid_default(REAL_CONFIG) is False
    assert (
        load_default_variant_by_skill(REAL_CONFIG)["pick_table_leg"]
        == "pick_table_leg_ramen_ori_venue_all6_20k"
    )


class TestFlipSelSlots:
    """Issue #188: flip の学び直し (手順の揃った区間 + 腕の状態のずらし) の 20k の slot。all6 の slot と ckpt だけが違う。"""

    # slot: commit (20k を push した HF の commit)
    SLOTS = {
        "flip_table_ramen_ori_flip_sel_a_n15_20k": "072c3757b781cfd547eefaead8e2c55de705d25b",
    }

    @pytest.mark.parametrize("name", sorted(SLOTS))
    def test_slot_picks_the_20k_ckpt_of_its_commit(self, name: str) -> None:
        entry = load_policy_variant(REAL_CONFIG, name)
        cfg = entry.policy_config
        assert entry.policy_type == "ramen_ori"
        assert cfg.ckpt_filename == "ckpt_step_020000.pt"
        assert tuple(cfg.ckpt_ref.split("@", 1)) == ("Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-flip-table", self.SLOTS[name])

    @pytest.mark.parametrize("name", sorted(SLOTS))
    def test_slot_matches_the_all6_slot_except_the_ckpt(self, name: str) -> None:
        new = load_policy_variant(REAL_CONFIG, name).policy_config
        venue = load_policy_variant(REAL_CONFIG, "flip_table_ramen_ori_all6_150k").policy_config
        same = ("mode", "overlay_jpeg_subsampling", "dtype", "action_space", "hydra_overrides", "cams",
                "replan_family", "execution_steps", "temporal_lambda", "rtc", "skill_id", "device")
        assert {k: getattr(new, k) for k in same} == {k: getattr(venue, k) for k in same}
