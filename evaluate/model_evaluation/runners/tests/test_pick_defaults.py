"""CLI-only tests: no model load, sensor subscription, or robot commands."""

import sys

import pytest

from evaluate.model_evaluation.runners import run_skill


def _parse(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_skill", *args])
    return run_skill.parse_args()


def test_pick_defaults_to_full_hybrid_and_rule_based_handover(monkeypatch):
    args = _parse(monkeypatch, "--skill", "pick_table_leg")
    assert args.pick_leg_hybrid is True
    assert args.pick_leg_phase3_executor == "rule_based"
    assert args.actuate is False


@pytest.mark.parametrize("skill", ["rotate_table_base", "insert_table_leg", "flip_table"])
def test_other_skills_do_not_enable_pick_hybrid(monkeypatch, skill):
    args = _parse(monkeypatch, "--skill", skill)
    assert args.pick_leg_hybrid is False


def test_old_vla_evaluation_requires_explicit_opt_out(monkeypatch):
    args = _parse(monkeypatch, "--skill", "pick_table_leg", "--no-pick-leg-hybrid", "--variant", "comparison_model")
    assert args.pick_leg_hybrid is False
    assert args.pick_leg_phase3_executor == "vla"
    assert args.variant == "comparison_model"


def test_hybrid_accepts_the_venue_ramen_ori_for_phase_one(monkeypatch):
    """Issue #188: hybrid の区間 1 を会場の腕の RAMEN-Ori にできる (持ち替えまで hybrid のまま)。"""
    args = _parse(monkeypatch, "--skill", "pick_table_leg", "--variant", "pick_table_leg_ramen_ori_venue_all6_20k")
    assert args.pick_leg_hybrid is True
    assert args.pick_leg_phase3_executor == "rule_based"
    assert args.variant == "pick_table_leg_ramen_ori_venue_all6_20k"


def test_hybrid_phase_one_variants_agree_with_the_defaults():
    """一覧の先頭 = 評価の既定 = entrypoint の固定値。一覧は全部 policy_config にある。

    本番 (policy_config) の pick の既定は hybrid なしの RAMEN-Ori に替えた (Issue #188)。hybrid の
    既定 (一覧の先頭) は R で選べる候補に残っていること。
    """
    from evaluate.model_evaluation.common.skill_registry import SKILL_REGISTRY
    from inference.desktop.entrypoint import PRODUCTION_HYBRID_PICK_VARIANT
    from inference.desktop.lower_policy.policies.config_loader import (
        DEFAULT_CONFIG_PATH,
        PolicyAlternative,
        list_variants,
        load_alternatives_by_skill,
    )
    from inference.desktop.pick_leg_hybrid import HYBRID_PHASE1_VARIANTS

    default = HYBRID_PHASE1_VARIANTS[0]
    assert default == PRODUCTION_HYBRID_PICK_VARIANT == SKILL_REGISTRY["pick_table_leg"].default_variant
    assert PolicyAlternative(default, True) in load_alternatives_by_skill(DEFAULT_CONFIG_PATH)[
        "pick_table_leg"
    ]
    assert set(HYBRID_PHASE1_VARIANTS) <= set(list_variants(DEFAULT_CONFIG_PATH))


def test_explicit_vla_phase_three_remains_available_for_comparison(monkeypatch):
    args = _parse(monkeypatch, "--skill", "pick_table_leg", "--pick-leg-phase3-executor", "vla")
    assert args.pick_leg_hybrid is True
    assert args.pick_leg_phase3_executor == "vla"


@pytest.mark.parametrize("options", [
    ["--skill", "pick_table_leg", "--variant", "comparison_model"],
    ["--skill", "flip_table", "--pick-leg-hybrid"],
    ["--skill", "pick_table_leg", "--no-pick-leg-hybrid", "--pick-leg-phase3-executor", "rule_based"],
])
def test_incompatible_hybrid_arguments_fail_before_startup(monkeypatch, options):
    with pytest.raises(SystemExit) as exc:
        _parse(monkeypatch, *options)
    assert exc.value.code == 2


def test_missing_vlm_does_not_fall_back_or_load_a_policy(monkeypatch, tmp_path):
    from inference.desktop.pick_leg_hybrid import real_skill

    monkeypatch.setattr(sys, "argv", [
        "run_skill", "--skill", "pick_table_leg", "--actuate", "--use-real-hand",
        "--log-dir", str(tmp_path),
    ])
    monkeypatch.setattr(real_skill, "load_reference_images", lambda *a, **kw: (object(), []))

    def unavailable(*args):
        raise RuntimeError("test VLM unavailable")

    monkeypatch.setattr(real_skill, "probe_vlm_endpoint", unavailable)
    monkeypatch.setattr(run_skill.assembly, "build_vla_skill", lambda *a, **kw: pytest.fail("loaded policy after VLM failure"))
    with pytest.raises(RuntimeError, match="test VLM unavailable"):
        run_skill.main()
