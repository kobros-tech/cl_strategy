"""Input-only evaluation and routing helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RoutingResult:
    """Input-only skill-routing result for one minibatch."""

    skill_indices: Tensor
    probabilities: Tensor
    best_probability: Tensor
    second_probability: Tensor
    confidence_gap: Tensor


def _routing_scores(
    raw_skill_logits: Sequence[Tensor],
    states: Sequence[Mapping[str, torch.Tensor]],
    skill_classes: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Score each skill by probability mass on its owned classes."""
    del states

    scores = []
    for logits, owned_classes in zip(raw_skill_logits, skill_classes, strict=False):
        if not owned_classes:
            scores.append(torch.zeros(logits.shape[0], device=logits.device))
            continue

        valid_classes = sorted(
            class_id for class_id in owned_classes if 0 <= class_id < logits.shape[1]
        )
        if not valid_classes:
            scores.append(torch.zeros(logits.shape[0], device=logits.device))
            continue

        if logits.shape[1] == 1:
            scores.append(torch.sigmoid(logits[:, 0]))
            continue

        probabilities = torch.softmax(logits, dim=1)
        scores.append(probabilities[:, valid_classes].sum(dim=1))

    if not scores:
        raise RuntimeError("No skills available for probe routing.")
    return torch.stack(scores, dim=0)


def _normalize_routing_scores(scores: Tensor, temperature: float) -> Tensor:
    """Normalize bounded routing scores across candidate skills."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    powered = scores.clamp_min(0).pow(1.0 / temperature)
    totals = powered.sum(dim=0, keepdim=True)
    eps = torch.finfo(powered.dtype).eps
    probabilities = powered / totals.clamp_min(eps)

    zero_total = totals.squeeze(0) <= 0
    if zero_total.any():
        probabilities = probabilities.clone()
        probabilities[:, zero_total] = 1.0 / probabilities.shape[0]
    return probabilities


def find_best_routing_skill(
    raw_skill_logits: list[Tensor],
    states: list[Mapping[str, torch.Tensor]],
    skill_classes: list[set[int]],
    temperature: float = 1.0,
) -> RoutingResult:
    """Select the best stored skill for every unlabeled probe sample."""
    scores = _routing_scores(raw_skill_logits, states, skill_classes)
    probabilities = _normalize_routing_scores(scores, temperature)
    skill_indices = probabilities.argmax(dim=0)

    if probabilities.shape[0] == 1:
        best_probability = probabilities[0]
        second_probability = torch.zeros_like(best_probability)
    else:
        top2 = torch.topk(probabilities, k=2, dim=0).values
        best_probability = top2[0]
        second_probability = top2[1]

    return RoutingResult(
        skill_indices=skill_indices,
        probabilities=probabilities,
        best_probability=best_probability,
        second_probability=second_probability,
        confidence_gap=best_probability - second_probability,
    )


def route_probe_logits(
    raw_skill_logits: list[Tensor],
    states: list[Mapping[str, torch.Tensor]],
    skill_classes: list[set[int]],
) -> Tensor:
    """Compatibility wrapper returning only selected skill indices."""
    return find_best_routing_skill(
        raw_skill_logits, states, skill_classes
    ).skill_indices
