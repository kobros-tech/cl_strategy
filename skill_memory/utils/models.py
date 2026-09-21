"""A minimal, dataset-agnostic MLP used both as a Skill Memory training
model and as an independent ML/CL evaluator (see
`skill_memory.evaluation.ml_cl_evaluator`).
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """One hidden layer, ReLU, then a fixed-width linear classifier head.

    Input is flattened to `(batch, input_dim)` before the first layer, so
    this accepts image tensors (e.g. `(batch, 1, 28, 28)`) directly.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_size: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(x.size(0), -1)
        return self.classifier(self.features(x))
