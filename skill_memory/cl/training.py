"""Single-class training loop."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..utils.probing import class_subset


def train_on_class(
    strategy,
    experience,
    target_class: int,
    epochs: int,
    batch_size: int,
    on_step=None,
) -> None:
    """Train only on samples whose label equals ``target_class``.

    ``on_step`` is called immediately after each optimizer update and receives
    the model, exact input/target batch, and zero-based optimization-step index.
    """
    dataset = class_subset(experience, target_class)
    if len(dataset) == 0:
        raise RuntimeError(f"class {target_class} has no samples to train on")
    if epochs < 1:
        return

    device = next(strategy.model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
    )

    criterion = getattr(strategy, "_criterion", None)
    if criterion is None:
        criterion = torch.nn.functional.cross_entropy

    strategy.model.train()
    for _ in range(epochs):
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            strategy.optimizer.zero_grad()
            logits = strategy.model(x)
            loss = criterion(logits, y)
            loss.backward()
            strategy.optimizer.step()

            if on_step is not None:
                on_step(strategy.model, x, y, strategy.clock.train_iterations)
            strategy.clock.train_iterations += 1
