# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import torch

from skill_memory.demos.candidate_skill_routing_patch import (
    _skill_verification_probabilities,
)


def test_singleton_skill_confidence_is_not_trivially_one() -> None:
    logits = torch.tensor([[3.0, 0.0, 0.0, 2.0, 0.0]])
    global_probabilities, owned_probabilities = _skill_verification_probabilities(
        logits, [3]
    )

    assert 0.0 < owned_probabilities.item() < 1.0
    assert torch.isclose(owned_probabilities, global_probabilities[:, 3]).all()


def test_multi_class_skill_uses_full_classifier_probability() -> None:
    logits = torch.tensor([[0.0, 0.0, 4.0, 3.0, 0.0]])
    global_probabilities, owned_probabilities = _skill_verification_probabilities(
        logits, [2, 3]
    )

    assert torch.isclose(owned_probabilities[0, 0], global_probabilities[0, 2])
    assert torch.isclose(owned_probabilities[0, 1], global_probabilities[0, 3])
    assert owned_probabilities.sum().item() < 1.0


def test_compact_singleton_skill_uses_nontrivial_applicability_score() -> None:
    logits = torch.tensor([[2.0]])
    _, owned_probabilities = _skill_verification_probabilities(logits, [3])

    assert 0.5 < owned_probabilities.item() < 1.0
