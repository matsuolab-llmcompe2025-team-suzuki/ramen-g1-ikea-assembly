"""Backward-compat re-export: SkillInitialPose / load_initial_pose を
inference.desktop.lower_policy.initial_pose に移動 (production 側で使うため)。
既存 eval code は本 module 経由の import で継続動作。
"""

from inference.desktop.lower_policy.initial_pose import (  # noqa: F401
    ARM_JOINT_ORDER,
    DEX1_MAX_RAD,
    SkillInitialPose,
    load_initial_pose,
)
