"""Insert optimal DP is the production default and keeps named alternatives."""

from pathlib import Path

import yaml

from inference.desktop.lower_policy.policies.act_diffusion import (
    CAMERAS,
    ActDiffusionPolicy,
    validate_act_diffusion_config,
)
from inference.desktop.lower_policy.policies.config_loader import (
    list_variants,
    load_default_variant_by_skill,
    load_policy_variant,
    resolve_policy_class,
)
from evaluate.model_evaluation.common.skill_registry import SKILL_REGISTRY


CONFIG = Path(__file__).resolve().parents[2] / "configs/policy_config.yaml"
VARIANT = "insert_table_leg_diffusion_optimal_100k"
CHECKPOINT = (
    "Team-RAMEN/IROS2026_RAMEN_takada_insert_leg_optimal_diffusion"
    "@6c9911b8185ea185e6c929fbe9ede7d586c1bf09"
)


def test_insert_diffusion_checkpoint_is_pinned_and_uses_existing_adapter():
    assert VARIANT in list_variants(CONFIG)
    entry = load_policy_variant(CONFIG, VARIANT)
    assert entry.policy_type == "act_diffusion"
    assert resolve_policy_class(entry.policy_type) is ActDiffusionPolicy
    assert entry.policy_config.ckpt_ref == CHECKPOINT
    assert entry.policy_config.checkpoint_subdir is None


def test_insert_diffusion_matches_raw_image_and_absolute_action_contract():
    config = load_policy_variant(CONFIG, VARIANT).policy_config
    validate_act_diffusion_config(config)
    assert config.mode == "none"
    assert config.dtype == "fp32"
    assert config.device == "cuda"
    assert config.cams == CAMERAS
    assert config.action_space == "abs"
    assert config.hydra_overrides == ()
    assert config.language_prompt is None


def test_insert_diffusion_replans_without_blending_or_rtc():
    config = load_policy_variant(CONFIG, VARIANT).policy_config
    assert config.replan_family == "diffusion"
    assert config.execution_steps == 8
    assert config.temporal_lambda is None
    assert config.rtc.enabled is False


def test_insert_diffusion_is_the_r_time_candidate_and_the_single_skill_evaluation_default():
    """本番の insert の既定は会場の腕の RAMEN-Ori に替えた (Issue #188、2026-09-30)。Diffusion 100k は
    R で選べる候補に残し、自前の実機評価 (run_skill) の既定はそのまま。"""
    from inference.desktop.lower_policy.policies.config_loader import (
        PolicyAlternative,
        load_alternatives_by_skill,
    )

    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    defaults = load_default_variant_by_skill(CONFIG)
    assert defaults["insert_table_leg"] == "insert_table_leg_ramen_ori_venue_all6_20k"
    assert PolicyAlternative(VARIANT) in load_alternatives_by_skill(CONFIG)["insert_table_leg"]
    assert SKILL_REGISTRY["insert_table_leg"].default_variant == VARIANT
    assert set(data["variant_sets"]) >= {"all6_400k"}
    assert load_default_variant_by_skill(CONFIG, variant_set="all6_400k")[
        "insert_table_leg"
    ] == "insert_table_leg_ramen_ori_all6_400k"
    assert "groot_insert_leg_200k" in list_variants(CONFIG)


def test_new_variant_does_not_request_overlay_jpeg_or_other_policy_overrides():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    entry = data["policies"][VARIANT]
    assert "overlay_jpeg_subsampling" not in entry
    assert "skill_id" not in entry
    assert "hydra_overrides" not in entry
