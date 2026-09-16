import torch

from skill_memory.reverse_engineering import CandidateParameters, NormalMLReverseEngineer


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
