"""PolicyChoice の unit test (Issue #188 ② 段 3)。

R の後の待ちで、既定と候補から skill の model を選ぶ。読み込み済みの物だけ選べ、
選んだら dispatcher の中身を差し替える (名前はそのまま)。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference.desktop.lower_policy.skills.policy_choice import (
    ChoiceOption,
    PolicyChoice,
)

SKILL = "rotate_table_base"


def _choice(states: dict[str, str] | None = None):
    skills = {
        name: SimpleNamespace(name=SKILL, variant=name) for name in ("main", "a", "b")
    }
    choice = PolicyChoice(
        SKILL, [ChoiceOption(name, skill) for name, skill in skills.items()]
    )
    replaced = []
    choice.replace_fn = replaced.append
    states = dict(states or {})
    choice.state_fn = lambda variant: states.get(variant, "loaded")
    return choice, skills, replaced, states


def test_keys_labels_and_status_list_the_default_first():
    choice, _, _, _ = _choice({"b": "loading"})
    assert choice.keys == ("0", "1", "2")
    assert choice.labels() == {
        "r": "R 初期姿勢へ（main のまま）",
        "0": "0 main（既定）",
        "1": "1 a",
        "2": "2 b",
    }
    assert choice.status() == (
        f"[model] {SKILL}: いま main ｜ 0 読み込み済み ｜ 1 読み込み済み ｜ 2 読み込み中"
    )


def test_a_loaded_candidate_replaces_the_skill_and_becomes_current():
    choice, skills, replaced, _ = _choice()
    accepted, message = choice.choose("1")
    assert accepted and replaced == [skills["a"]]
    assert choice.current_variant == "a"
    assert "main → a に切り替えた" in message
    assert choice.labels()["r"] == "R 初期姿勢へ（a のまま）"
    # 既定へ戻す
    assert choice.choose("0")[0] and replaced[-1] is skills["main"]
    assert choice.current_variant == "main"


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        ("loading", "まだ読み込み中"),
        ("failed", "読み込みに失敗"),
        ("released", "解放したので"),
    ],
)
def test_a_candidate_that_is_not_loaded_is_refused(state, reason):
    choice, _, replaced, _ = _choice({"b": state})
    accepted, message = choice.choose("2")
    assert not accepted and replaced == []
    assert choice.current_variant == "main"
    assert reason in message


def test_choosing_the_current_model_just_proceeds():
    choice, _, replaced, states = _choice()
    states["main"] = "loading"  # 今の model は読み込みの状態を見ない
    accepted, message = choice.choose("0")
    assert accepted and replaced == [] and "main のまま" in message


def test_a_choice_needs_candidates_with_the_same_skill_name_and_no_duplicates():
    main = SimpleNamespace(name=SKILL)
    with pytest.raises(ValueError, match="at least one candidate"):
        PolicyChoice(SKILL, [ChoiceOption("main", main)])
    with pytest.raises(ValueError, match="named"):
        PolicyChoice(
            SKILL,
            [ChoiceOption("main", main), ChoiceOption("a", SimpleNamespace(name="x"))],
        )
    with pytest.raises(ValueError, match="twice"):
        PolicyChoice(SKILL, [ChoiceOption("main", main), ChoiceOption("main", main)])


def test_the_same_variant_used_two_ways_is_told_apart_by_its_note():
    """pick の hybrid あり・なし (同じ RAMEN-Ori)。画面と切り替えの 1 行に添え書きが出る。"""
    skill = SimpleNamespace(name="pick_table_leg")
    choice = PolicyChoice(
        "pick_table_leg",
        [
            ChoiceOption("groot", skill, note="hybrid"),
            ChoiceOption("venue", skill, note="hybrid"),
            ChoiceOption("venue", skill, note="hybrid なし"),
        ],
    )
    choice.replace_fn = lambda _skill: None
    assert choice.labels() == {
        "r": "R 初期姿勢へ（groot（hybrid） のまま）",
        "0": "0 groot（既定・hybrid）",
        "1": "1 venue（hybrid）",
        "2": "2 venue（hybrid なし）",
    }
    accepted, message = choice.choose("2")
    assert accepted and "groot（hybrid） → venue（hybrid なし）" in message
    assert choice.current_variant == "venue" and choice.current_label == "venue（hybrid なし）"
    with pytest.raises(ValueError, match="twice"):
        PolicyChoice(
            "pick_table_leg",
            [ChoiceOption("groot", skill), ChoiceOption("venue", skill, note="x"), ChoiceOption("venue", skill, note="x")],
        )


@pytest.mark.parametrize(
    ("vlm", "expected"),
    [("loaded", "loaded"), ("loading", "loading"), ("failed", "failed"), ("released", "released")],
)
def test_a_hybrid_candidate_needs_the_vlm_too(vlm, expected):
    """既定が hybrid でない run の hybrid の候補は、VLM も読めて初めて選べる (Issue #188)。"""
    skill = SimpleNamespace(name="pick_table_leg")
    choice = PolicyChoice(
        "pick_table_leg",
        [
            ChoiceOption("venue", skill, note="hybrid なし"),
            ChoiceOption("groot", skill, note="hybrid", requires=("vlm",)),
        ],
    )
    replaced = []
    choice.replace_fn = replaced.append
    states = {"venue": "loaded", "groot": "loaded", "vlm": vlm}
    choice.state_fn = states.__getitem__
    accepted, _ = choice.choose("1")
    assert accepted is (expected == "loaded")
    assert choice.status().endswith(
        {"loaded": "読み込み済み", "loading": "読み込み中", "failed": "読み込み失敗", "released": "解放済み"}[expected]
    )
