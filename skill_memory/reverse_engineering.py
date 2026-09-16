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


class _ReverseModel(nn.Module):
    """Sample encoder plus candidate-conditioned binary classifier."""

    def __init__(self, sample_dim: int, weight_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sample_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, weight_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(weight_dim * 3 + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, samples: Tensor, candidate: Tensor) -> Tensor:
        embedding = self.encoder(samples)
        interaction = embedding * candidate[:, :-1]
        features = torch.cat((embedding, candidate[:, :-1], interaction, candidate[:, -1:]), dim=1)
        return self.head(features)


class NormalMLReverseEngineer:
    """Learn ``P(y=1 | x, candidate_weights)`` as a normal ML problem.

    The reverse model is deliberately independent of the continual-learning
    feature extractor. It learns its own sample representation and consumes a
    frozen candidate classifier weight/bias pair. Candidate parameters come
    from a frozen skill generation, while this model has its own parameters and
    optimizer.
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
        self.model: _ReverseModel | None = None
        self.input_dim: int | None = None
        self.weight_dim: int | None = None

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
    ) -> tuple[Tensor, Tensor]:
        samples = self._flatten_samples(x)
        candidate = self._candidate_vector(weight, bias, samples.shape[0])
        return samples, candidate

    def fit(self, pairs: list[tuple[Tensor, CandidateParameters, float]]) -> None:
        """Fit from positive/negative reference pairs.

        Each pair may contain a batch of reference samples. The scalar target
        is expanded to every sample in that batch. No evaluation labels are
        needed by this method.
        """
        if not pairs:
            self.model = None
            self.input_dim = None
            self.weight_dim = None
            return
        torch.manual_seed(self.seed)
        sample_batches = []
        candidate_batches = []
        target_batches = []
        for x, params, target in pairs:
            samples, candidate = self._features(x, params.weight, params.bias)
            sample_batches.append(samples)
            candidate_batches.append(candidate)
            target_batches.append(
                torch.full(
                    (samples.shape[0], 1),
                    float(target),
                    dtype=torch.float32,
                )
            )
        samples = torch.cat(sample_batches, dim=0)
        candidates = torch.cat(candidate_batches, dim=0)
        targets = torch.cat(target_batches, dim=0)
        self.input_dim = int(samples.shape[1])
        self.weight_dim = int(candidates.shape[1] - 1)
        model = _ReverseModel(self.input_dim, self.weight_dim, self.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = model(samples, candidates)
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
        if self.model is None or self.input_dim is None or self.weight_dim is None:
            raise RuntimeError("reverse-engineering model has not been fitted")
        samples, candidate = self._features(x, weight, bias)
        if samples.shape[1] != self.input_dim:
            raise ValueError("sample shape changed after fitting")
        if candidate.shape[1] != self.weight_dim + 1:
            raise ValueError("candidate weight shape changed after fitting")
        with torch.no_grad():
            return torch.sigmoid(self.model(samples, candidate).squeeze(-1))
