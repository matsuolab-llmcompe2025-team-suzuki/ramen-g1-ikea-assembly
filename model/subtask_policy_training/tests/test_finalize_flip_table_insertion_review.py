from __future__ import annotations

from model.subtask_policy_training.scripts.finalize_flip_table_insertion_review import (
    _transition_mask,
)


def test_transition_mask_dilates_dex1_changes() -> None:
    command = [[4.5, 4.5] for _ in range(30)]
    command[15][1] = 4.0
    mask = _transition_mask(command, radius=2)
    assert mask[12] is False
    assert all(mask[index] for index in range(13, 18))
    assert mask[18] is True


def test_transition_mask_ignores_small_noise() -> None:
    command = [[4.5, 4.5] for _ in range(10)]
    command[5][0] = 4.49
    assert not any(_transition_mask(command, radius=1))
