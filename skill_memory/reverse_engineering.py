"""Standalone ML reverse engineering from frozen model behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CandidateParameters:
    """Frozen classifier parameters retained for compatibility."""

    weight: Tensor
    bias: float


class _FeatureReverseModel(nn.Module):
    """Small supervised model over frozen-candidate behavior features."""

    def __init__(self, feature_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class NormalMLReverseEngineer:
    """Learn anonymous candidate/class compatibility as a normal ML problem.

    Training consumes only detached samples and responses from frozen skill
    snapshots. Evaluation is a pure prediction operation; the reverse model is
    never fitted from inside an evaluation forward pass.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        epochs: int = 120,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.model: _FeatureReverseModel | None = None
        self.feature_dim: int | None = None

    def fit_feature_pairs(
        self,
        pairs: list[tuple[Tensor, float]],
    ) -> None:
        """Fit from detached frozen-model features and binary targets."""
        if not pairs:
            self.model = None
            self.feature_dim = None
            return
        torch.manual_seed(self.seed)
        features = torch.cat(
            [feature.detach().float().cpu() for feature, _ in pairs],
            dim=0,
        )
        targets = torch.cat(
            [
                torch.full(
                    (feature.shape[0], 1),
                    float(target),
                    dtype=torch.float32,
                )
                for feature, target in pairs
            ],
            dim=0,
        )
        self.feature_dim = int(features.shape[1])
        model = _FeatureReverseModel(self.feature_dim, self.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        with torch.enable_grad():
            for _ in range(self.epochs):
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
        self.model = model.eval()

    def predict_proba_features(self, features: Tensor) -> Tensor:
        """Return candidate-match probabilities for behavior features."""
        if self.model is None or self.feature_dim is None:
            raise RuntimeError("reverse-engineering model has not been fitted")
        features = features.detach().float().cpu()
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("reverse-engineering feature shape changed")
        with torch.no_grad():
            return torch.sigmoid(self.model(features).squeeze(-1))

    def fit(self, pairs: list[tuple[Tensor, CandidateParameters, float]]) -> None:
        """Backward-compatible raw-pair API."""
        if not pairs:
            self.model = None
            self.feature_dim = None
            return
        feature_pairs = []
        for samples, params, target in pairs:
            samples = samples.detach().float().cpu().reshape(samples.shape[0], -1)
            weight = params.weight.detach().float().cpu().reshape(1, -1)
            weight = weight.expand(samples.shape[0], -1)
            bias = torch.full((samples.shape[0], 1), float(params.bias))
            feature_pairs.append(
                (torch.cat((samples, weight, bias), dim=1), target)
            )
        self.fit_feature_pairs(feature_pairs)

    def predict_proba(
        self,
        x: Tensor,
        weight: Tensor,
        bias: float,
    ) -> Tensor:
        """Backward-compatible prediction API."""
        samples = x.detach().float().cpu().reshape(x.shape[0], -1)
        weight = weight.detach().float().cpu().reshape(1, -1)
        weight = weight.expand(samples.shape[0], -1)
        bias_column = torch.full((samples.shape[0], 1), float(bias))
        return self.predict_proba_features(
            torch.cat((samples, weight, bias_column), dim=1)
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize the fitted reverse model without optimizer state."""
        return {
            "hidden_size": self.hidden_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "feature_dim": self.feature_dim,
            "model": None
            if self.model is None
            else {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a previously fitted reverse model."""
        self.hidden_size = int(state.get("hidden_size", self.hidden_size))
        self.epochs = int(state.get("epochs", self.epochs))
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.seed = int(state.get("seed", self.seed))
        feature_dim = state.get("feature_dim")
        model_state = state.get("model")
        if feature_dim is None or model_state is None:
            self.model = None
            self.feature_dim = None
            return
        self.feature_dim = int(feature_dim)
        model = _FeatureReverseModel(self.feature_dim, self.hidden_size)
        model.load_state_dict(model_state)
        self.model = model.eval()
