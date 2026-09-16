import torch

from skill_memory.persistent_skill_memory_plugin import (
    PersistentFingerprintSkillMemoryPlugin,
)
from skill_memory.reverse_engineering import (
    CandidateParameters,
    NormalMLReverseEngineer,
)


def test_normal_ml_reverse_engineer_learns_candidate_identity():
    x0 = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    x1 = torch.tensor([[0.0, 1.0], [0.1, 0.9]])
    candidate0 = CandidateParameters(torch.tensor([1.0, 0.0]), 0.0)
    candidate1 = CandidateParameters(torch.tensor([0.0, 1.0]), 0.0)
    pairs = [
        (x0, candidate0, 1.0),
        (x0, candidate1, 0.0),
        (x1, candidate0, 0.0),
        (x1, candidate1, 1.0),
    ]

    reverse_engineer = NormalMLReverseEngineer(epochs=120, seed=7)
    reverse_engineer.fit(pairs)

    p00 = reverse_engineer.predict_proba(x0, candidate0.weight, candidate0.bias)
    p01 = reverse_engineer.predict_proba(x0, candidate1.weight, candidate1.bias)
    p10 = reverse_engineer.predict_proba(x1, candidate0.weight, candidate0.bias)
    p11 = reverse_engineer.predict_proba(x1, candidate1.weight, candidate1.bias)

    assert float(p00.mean()) > 0.8
    assert float(p11.mean()) > 0.8
    assert float(p01.mean()) < 0.2
    assert float(p10.mean()) < 0.2


def test_normal_ml_reverse_engineer_does_not_mutate_candidate_weights():
    weight = torch.tensor([1.0, 2.0])
    params = CandidateParameters(weight.clone(), 0.5)
    reverse_engineer = NormalMLReverseEngineer(epochs=5)
    reverse_engineer.fit([(torch.ones(2, 2), params, 1.0)])
    assert torch.equal(params.weight, weight)


def test_normal_ml_reverse_engineer_state_round_trip():
    x0 = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    x1 = torch.tensor([[0.0, 1.0], [0.1, 0.9]])
    candidate0 = CandidateParameters(torch.tensor([1.0, 0.0]), 0.0)
    candidate1 = CandidateParameters(torch.tensor([0.0, 1.0]), 0.0)
    pairs = [
        (x0, candidate0, 1.0),
        (x0, candidate1, 0.0),
        (x1, candidate0, 0.0),
        (x1, candidate1, 1.0),
    ]

    original = NormalMLReverseEngineer(epochs=40, seed=3)
    original.fit(pairs)
    state = original.state_dict()

    restored = NormalMLReverseEngineer()
    restored.load_state_dict(state)

    for x, candidate in ((x0, candidate0), (x1, candidate1)):
        expected = original.predict_proba(x, candidate.weight, candidate.bias)
        actual = restored.predict_proba(x, candidate.weight, candidate.bias)
        assert torch.allclose(expected, actual)


def test_normal_ml_reverse_engineer_standardizes_features():
    pairs = [
        (torch.tensor([[1.0, 10.0], [2.0, 20.0]]), 1.0),
        (torch.tensor([[3.0, 30.0], [4.0, 40.0]]), 0.0),
    ]
    reverse_engineer = NormalMLReverseEngineer(epochs=5)
    reverse_engineer.fit_feature_pairs(pairs)

    assert reverse_engineer.feature_mean is not None
    assert reverse_engineer.feature_std is not None
    assert torch.all(reverse_engineer.feature_std > 0)


def test_normal_ml_reverse_engineer_handles_many_negative_candidates():
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[0.0, 1.0]])
    pairs = [(positive, 1.0)] + [(negative, 0.0)] * 19

    reverse_engineer = NormalMLReverseEngineer(epochs=120, seed=11)
    reverse_engineer.fit_feature_pairs(pairs)

    positive_probability = reverse_engineer.predict_proba_features(positive)
    negative_probability = reverse_engineer.predict_proba_features(negative)
    assert float(positive_probability.mean()) > float(negative_probability.mean())


def test_reverse_router_features_include_candidate_class_parameters():
    x = torch.tensor([[0.25, 0.75]])
    logits = torch.tensor([[2.0, -1.0]])
    weight_a = torch.tensor([1.0, 0.0])
    weight_b = torch.tensor([0.0, 1.0])

    feature_a = PersistentFingerprintSkillMemoryPlugin._make_features(
        x, logits, weight_a, 0.0, output_dim=2
    )
    feature_b = PersistentFingerprintSkillMemoryPlugin._make_features(
        x, logits, weight_b, 0.0, output_dim=2
    )

    assert not torch.equal(feature_a, feature_b)
    assert feature_a.shape == feature_b.shape
