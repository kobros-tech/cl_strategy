"""Tests for weight-based reconstruction and continuous behavior fingerprints."""

import torch
import torch.nn as nn
from avalanche.models.dynamic_modules import IncrementalClassifier

from skill_memory import (
    reverse_engineer_scores_from_weights,
    reverse_engineer_y_from_weights,
)
from skill_memory.behavior import build_weight_behavior_statistics


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 3, bias=False)
        self.classifier = IncrementalClassifier(3, initial_out_features=3)

    def forward(self, x):
        return self.classifier(self.features(x))


def test_weight_reconstruction_matches_classifier_output():
    torch.manual_seed(0)
    model = TinyClassifier()
    x = torch.randn(8, 4)

    with torch.no_grad():
        expected = model(x)
    reconstructed = reverse_engineer_scores_from_weights(model, x)

    torch.testing.assert_close(reconstructed, expected)


def test_weight_reverse_engineering_returns_classifier_argmax_behavior():
    torch.manual_seed(1)
    model = TinyClassifier()
    x = torch.randn(8, 4)

    expected = model(x).argmax(dim=1)
    for class_id in range(3):
        actual = reverse_engineer_y_from_weights(model, x, class_id)
        torch.testing.assert_close(actual, expected.eq(class_id))


def test_weight_behavior_statistics_capture_reference_distribution():
    torch.manual_seed(2)
    model = TinyClassifier()
    x = torch.randn(12, 4)

    stats = build_weight_behavior_statistics(model, x, class_id=1)

    assert stats["feature_mean"].shape == (3,)
    assert stats["feature_std"].shape == (3,)
    assert stats["feature_std"].min().item() >= 1e-6
    assert isinstance(stats["margin_mean"], float)
    assert stats["margin_std"] >= 1e-6
    assert stats["weight"].shape == (3,)
    assert isinstance(stats["bias"], float)
