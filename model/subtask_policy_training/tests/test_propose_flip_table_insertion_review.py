from __future__ import annotations

import numpy as np

from model.subtask_policy_training.scripts.propose_flip_table_insertion_review import (
    propose_interval,
)


def test_proposal_requires_left_support_and_persistent_right_close() -> None:
    command = np.full((180, 2), 4.5)
    state = command.copy()
    command[40:, 0] = 2.0
    state[45:, 0] = 2.0
    command[110:, 1] = 2.5
    state[115:, 1] = 2.5

    proposal = propose_interval(state, command)
    assert proposal is not None
    m3, m4 = proposal
    assert m3 == m4 - 90
    assert 100 <= m4 <= 110


def test_proposal_rejects_right_close_without_left_support() -> None:
    command = np.full((180, 2), 4.5)
    state = command.copy()
    command[110:, 1] = 2.5
    state[115:, 1] = 2.5

    assert propose_interval(state, command) is None


def test_proposal_rejects_non_finite_signal() -> None:
    command = np.full((180, 2), 4.5)
    state = command.copy()
    state[0, 0] = np.nan
    assert propose_interval(state, command) is None


def test_proposal_selects_later_right_regrasp_not_initial_grasp() -> None:
    command = np.full((360, 2), 4.5)
    state = command.copy()
    command[40:, 0] = 2.0
    state[45:, 0] = 2.0
    command[100:160, 1] = 2.5
    state[105:165, 1] = 2.5
    command[260:, 1] = 2.5
    state[265:, 1] = 2.5

    proposal = propose_interval(state, command)
    assert proposal is not None
    _, m4 = proposal
    assert 250 <= m4 <= 260
