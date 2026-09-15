import torch

from behavior import (
    ClassBehaviorRecord,
    compare_fingerprint_evolution,
    fingerprint_similarity,
    pairwise_reference_similarity,
)


def _record(class_id, skill_id, values, inputs=None):
    return ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=0,
        reference_inputs=(
            torch.tensor([[float(class_id)]]) if inputs is None else inputs
        ),
        output_class_ids=tuple(sorted(values)),
        reference_output=torch.tensor([values[c] for c in sorted(values)]),
        reference_summary=torch.ones(4),
    )


def test_fingerprint_evolution_uses_fixed_reference_behavior():
    before = {
        0: _record(0, 0, {0: 0.9, 1: 0.1}),
        1: _record(1, 1, {0: 0.1, 1: 0.9}),
    }
    after = {
        0: _record(0, 0, {0: 0.9, 1: 0.1}, torch.tensor([[0.0]])),
        1: _record(1, 1, {0: 0.8, 1: 0.2}, torch.tensor([[1.0]])),
        2: _record(2, 2, {0: 0.0, 1: 1.0}),
    }

    result = compare_fingerprint_evolution(before, after)

    assert result[0]["status"] == "updated"
    assert result[0]["similarity"] > 0.999
    assert result[0]["drift"] < 0.001
    assert result[1]["status"] == "updated"
    assert result[1]["drift"] > 0.0
    assert result[2] == {"status": "new"}


def test_pairwise_similarity_uses_global_class_coordinates():
    left = _record(10, 0, {10: 1.0, 20: 0.0})
    right = _record(20, 1, {10: 0.0, 20: 1.0})
    same = _record(30, 2, {10: 1.0, 20: 0.0})

    assert fingerprint_similarity(left, same) > 0.999
    assert fingerprint_similarity(left, right) < 0.01

    pairwise = pairwise_reference_similarity([left, right, same])
    assert pairwise[(10, 20)] < 0.01
    assert pairwise[(10, 30)] > 0.999
    assert pairwise[(20, 30)] < 0.01
