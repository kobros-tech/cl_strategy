from __future__ import annotations

import torch
from torch import nn

from skill_memory.reverse_engineering import NormalMLReverseEngineer


class _ShapeRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_shape: tuple[int, ...] | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.seen_shape = tuple(features.shape)
        return features[..., :1]


def test_listwise_inference_preserves_candidate_sequence_dimension() -> None:
    reverse = NormalMLReverseEngineer()
    model = _ShapeRecorder()
    reverse.model = model
    reverse.feature_dim = 3
    reverse.feature_mean = torch.zeros(3)
    reverse.feature_std = torch.ones(3)

    features = torch.randn(4, 5, 3)
    scores = reverse.predict_scores_candidate_sets(features)

    assert model.seen_shape == (4, 5, 3)
    assert scores.shape == (4, 5)


def test_single_candidate_set_scoring_keeps_candidates_as_sequence() -> None:
    reverse = NormalMLReverseEngineer()
    model = _ShapeRecorder()
    reverse.model = model
    reverse.feature_dim = 2
    reverse.feature_mean = torch.zeros(2)
    reverse.feature_std = torch.ones(2)

    features = torch.randn(1, 7, 2)
    scores = reverse.predict_scores_candidate_sets(features)

    assert model.seen_shape == (1, 7, 2)
    assert scores.shape == (1, 7)
