"""Standalone ML reverse engineering of anonymous class identity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CandidateParameters:
    """Frozen classifier parameters used as candidate-side ML features."""

    weight: Tensor
    bias: float


class NormalMLReverseEngineer:
    """Learn ``P(y=1 | x, candidate_weights)`` as a normal ML problem.

    The reverse model is deliberately independent of the continual-learning
    feature extractor. It consumes the raw sample and a frozen candidate
    classifier weight/bias pair. Candidate parameters come from a frozen skill
    generation, while this model has its own parameters and optimizer.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        epochs: int = 80,
        learning_rate: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.model: nn.Module | None = None
        self.input_dim: int | None = None

    @staticmethod
    def _flatten_samples(x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError("reverse-engineering samples must have a batch dimension")
        return x.detach().float().cpu().reshape(x.shape[0], -1)

    @staticmethod
    def _candidate_vector(
        weight: Tensor,
        bias: float,
        count: int,
    ) -> Tensor:
        weight = weight.detach().float().cpu().reshape(1, -1).expand(count, -1)
        bias_column = torch.full((count, 1), float(bias), dtype=weight.dtype)
        return torch.cat((weight, bias_column), dim=1)

    def _features(
        self,
        x: Tensor,
        weight: Tensor,
        bias: float,
    ) -> Tensor:
        samples = self._flatten_samples(x)
        candidate = self._candidate_vector(weight, bias, samples.shape[0])
        return torch.cat((samples, candidate), dim=1)

    def fit(self, pairs: list[tuple[Tensor, CandidateParameters, float]]) -> None:
        """Fit from positive/negative reference pairs.

        ``pairs`` contains reference samples paired with candidate parameters
        and a binary target. No evaluation labels are needed by this method.
        """
        if not pairs:
            self.model = None
            self.input_dim = None
            return
        torch.manual_seed(self.seed)
        features = torch.cat(
            [self._features(x, params.weight, params.bias) for x, params, _ in pairs],
            dim=0,
        )
        targets = torch.tensor(
            [target for _, _, target in pairs], dtype=torch.float32
        ).view(-1, 1)
        self.input_dim = int(features.shape[1])
        model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        self.model = model.eval()

    def predict_proba(
        self,
        x: Tensor,
        weight: Tensor,
        bias: float,
    ) -> Tensor:
        """Return ``P(y=1)`` for each sample/candidate pair."""
        if self.model is None or self.input_dim is None:
            raise RuntimeError("reverse-engineering model has not been fitted")
        features = self._features(x, weight, bias)
        if features.shape[1] != self.input_dim:
            raise ValueError("sample/candidate feature shape changed after fitting")
        with torch.no_grad():
            return torch.sigmoid(self.model(features).squeeze(-1))
