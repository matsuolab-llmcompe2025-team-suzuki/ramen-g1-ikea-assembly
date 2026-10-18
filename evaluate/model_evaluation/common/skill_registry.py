"""Eval runner の対応 skill 一覧と VlaSkill class への mapping。

各 skill の evaluation を回すために、以下 3 種の identity を静的に持つ:

- **skill_name**: skill_config.yaml key、CLI `--skill` に渡す文字列
- **VlaSkill subclass**: `inference.desktop.lower_policy.skills.vla_skill` から
  import して instance を組む
- **default policy variant**: `policy_config.yaml` の該当 variant 名 (CLI で override 可)

`SKILL_REGISTRY` はこの 3 種をまとめた dataclass の read-only table。
Eval で扱う skill が増えたらここに 1 entry 追加すれば runner が対応する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EvalSkillEntry:
    skill_name: str            # skill_config.yaml key + CLI --skill 値
    vla_skill_cls_name: str    # inference.desktop.lower_policy.skills.vla_skill class 名
    default_variant: str       # policy_config.yaml:policies の default variant 名


SKILL_REGISTRY: dict[str, EvalSkillEntry] = {
    "move_table_base": EvalSkillEntry(
        skill_name="move_table_base",
        vla_skill_cls_name="MoveTableBaseVlaSkill",
        default_variant="ramen_ori_default",
    ),
    "rotate_table_base": EvalSkillEntry(
        skill_name="rotate_table_base",
        vla_skill_cls_name="RotateTableBaseVlaSkill",  # 実装 pending (新設 skill)
        default_variant="ramen_ori_default",  # task 5+7 combined 学習 Model
    ),
    "pick_table_leg": EvalSkillEntry(
        skill_name="pick_table_leg",
        vla_skill_cls_name="PickTableLegVlaSkill",
        default_variant="groot_pick_legs_v1",
    ),
    "insert_table_leg": EvalSkillEntry(
        skill_name="insert_table_leg",
        vla_skill_cls_name="InsertTableLegVlaSkill",
        default_variant="insert_table_leg_diffusion_optimal_100k",
    ),
    "rotate_leg_to_tighten": EvalSkillEntry(
        skill_name="rotate_leg_to_tighten",
        vla_skill_cls_name="RotateLegToTightenVlaSkill",
        default_variant="groot_rotate_leg_200k",
    ),
    "insert_and_tighten": EvalSkillEntry(
        skill_name="insert_and_tighten",
        vla_skill_cls_name="InsertAndTightenVlaSkill",
        default_variant="groot_insert_and_tighten_200k",
    ),
    "flip_table": EvalSkillEntry(
        skill_name="flip_table",
        vla_skill_cls_name="FlipTableVlaSkill",
        default_variant="groot_flip_table_n17_2_legacy_fk",
    ),
}


def resolve_vla_skill_class(entry: EvalSkillEntry) -> Callable:
    """Late-import VlaSkill subclass by name (runtime env 依存を lazy 化)。"""
    from inference.desktop.lower_policy.skills import vla_skill

    cls = getattr(vla_skill, entry.vla_skill_cls_name, None)
    if cls is None:
        raise LookupError(
            f"VlaSkill class {entry.vla_skill_cls_name!r} not found in "
            f"inference.desktop.lower_policy.skills.vla_skill (skill={entry.skill_name!r})"
        )
    return cls
