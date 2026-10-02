# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import pytest
import torch

from skill_memory.evaluation.independent_evaluator import (
    EvaluationMemory,
    select_evaluation_memory,
)


def _memory(class_id: int, experience_index: int, size: int = 2):
    return EvaluationMemory(
        inputs=torch.zeros(size, 1),
        targets=torch.full((size,), class_id, dtype=torch.long),
        class_id=class_id,
        experience_index=experience_index,
    )


def test_history_update_replays_all_retained_classes():
    memory = [_memory(2, 0), _memory(5, 1), _memory(7, 2)]

    selected = select_evaluation_memory(memory, update_mode="history")

    assert [item.class_id for item in selected] == [2, 5, 7]
    assert [item.size for item in selected] == [2, 2, 2]


def test_new_class_update_uses_only_latest_experience():
    memory = [_memory(2, 0), _memory(5, 1), _memory(7, 2), _memory(8, 2)]

    selected = select_evaluation_memory(memory, update_mode="new_class")

    assert [item.class_id for item in selected] == [7, 8]
    assert all(item.experience_index is None for item in selected)


def test_new_class_update_does_not_replay_older_examples():
    old = _memory(2, 0, size=20)
    new = _memory(5, 1, size=3)

    selected = select_evaluation_memory([old, new], update_mode="new_class")

    assert sum(item.size for item in selected) == 3


@pytest.mark.parametrize("mode", ["invalid", "replay", "current"])
def test_unknown_update_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="history.*new_class"):
        select_evaluation_memory([], update_mode=mode)
