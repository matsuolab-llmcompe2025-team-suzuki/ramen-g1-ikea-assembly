"""skill_mapping の unit test (Issue #120 Phase A-3)。"""

from __future__ import annotations

import pytest

from model.ramen_ori.skill_mapping import (
    NUM_SKILLS,
    PHASE2_SKILL_IDS,
    SKILL_ID_TO_NAME,
    SKILL_INSERT_TABLE_LEG,
    SKILL_MOVE_TABLE_BASE,
    SKILL_ROTATE_TABLE_BASE,
    TASK_INDEX_TO_SKILL_ID,
    skill_id_name,
    task_index_to_skill_id,
)


class TestConstants:
    def test_num_skills_is_6(self) -> None:
        assert NUM_SKILLS == 6

    def test_skill_ids_are_0_through_5(self) -> None:
        assert set(TASK_INDEX_TO_SKILL_ID.values()) == set(range(NUM_SKILLS))

    def test_skill_id_to_name_covers_all_ids(self) -> None:
        assert set(SKILL_ID_TO_NAME.keys()) == set(range(NUM_SKILLS))

    def test_mapped_task_indices_are_0_2_3_4_5_7(self) -> None:
        assert set(TASK_INDEX_TO_SKILL_ID.keys()) == {0, 2, 3, 4, 5, 7}


class TestTaskIndexToSkillId:
    @pytest.mark.parametrize(
        "task_index,expected_skill_id,expected_name",
        [
            (0, 0, "insert_table_leg"),
            (2, 1, "flip_table"),
            (3, 2, "rotate_leg_to_tighten"),
            (4, 3, "pick_table_leg"),
            (5, 4, "rotate_table_base"),
            (7, 5, "move_table_base"),
        ],
    )
    def test_mapping_matches_design(
        self, task_index: int, expected_skill_id: int, expected_name: str
    ) -> None:
        skill_id = task_index_to_skill_id(task_index)
        assert skill_id == expected_skill_id
        assert skill_id_name(skill_id) == expected_name

    @pytest.mark.parametrize("bad_tid", [1, 6, -1, 8, 100])
    def test_unmapped_task_index_raises(self, bad_tid: int) -> None:
        with pytest.raises(KeyError, match=str(bad_tid)):
            task_index_to_skill_id(bad_tid)


class TestPhase2Scope:
    def test_phase2_covers_only_task5_and_task7(self) -> None:
        assert PHASE2_SKILL_IDS == frozenset(
            {SKILL_ROTATE_TABLE_BASE, SKILL_MOVE_TABLE_BASE}
        )

    def test_phase2_skill_ids_are_valid(self) -> None:
        assert all(0 <= sid < NUM_SKILLS for sid in PHASE2_SKILL_IDS)

    def test_task5_and_task7_map_into_phase2(self) -> None:
        assert task_index_to_skill_id(5) in PHASE2_SKILL_IDS
        assert task_index_to_skill_id(7) in PHASE2_SKILL_IDS


class TestSkillIdName:
    def test_known_id(self) -> None:
        assert skill_id_name(SKILL_INSERT_TABLE_LEG) == "insert_table_leg"

    def test_unknown_id_fallback(self) -> None:
        assert skill_id_name(999) == "unknown_skill_999"
