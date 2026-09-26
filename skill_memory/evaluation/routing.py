"""Input-only evaluation and routing helpers."""

from __future__ import annotations

from collections.abc import Sequence
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


def _normalize_routing_scores(scores: Tensor, temperature: float) -> Tensor:
    """Normalize bounded evaluator routing scores across candidate skills."""
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


def score_skill_compatibility(
    evaluator_logits: Tensor,
    skill_classes: Sequence[Sequence[int]],
) -> Tensor:
    """Score each skill from one shared independent evaluator."""
    if evaluator_logits.ndim != 2:
        raise ValueError("evaluator_logits must have shape [batch, classes]")
    probabilities = torch.softmax(evaluator_logits, dim=1)
    scores = []
    for classes in skill_classes:
        owned = sorted(int(class_id) for class_id in classes)
        if not owned:
            scores.append(
                torch.zeros(probabilities.shape[0], device=probabilities.device)
            )
            continue
        if max(owned) >= probabilities.shape[1]:
            raise RuntimeError(
                f"skill owns class {max(owned)}, but evaluator has only "
                f"{probabilities.shape[1]} output columns"
            )
        scores.append(probabilities[:, owned].sum(dim=1))
    if not scores:
        raise RuntimeError("No skills available for probe routing.")
    return torch.stack(scores, dim=0)



@torch.no_grad()
def build_skill_behavior_prototypes(
    evaluator: torch.nn.Module,
    skill_inputs: Sequence[Tensor],
    *,
    device: torch.device,
) -> Tensor:
    """Build one evaluator-behavior prototype for each stored skill."""
    if not skill_inputs:
        raise ValueError("skill_inputs must contain at least one skill")

    evaluator.eval()
    prototypes = []
    for inputs in skill_inputs:
        if inputs.ndim < 2 or inputs.shape[0] == 0:
            raise ValueError("each skill must provide non-empty batched inputs")
        logits = evaluator(inputs.to(device))
        if logits.ndim != 2:
            raise RuntimeError(
                "The evaluator must return [batch, classes] logits."
            )
        prototypes.append(torch.softmax(logits, dim=1).mean(dim=0))
    return torch.stack(prototypes, dim=0)


def score_skill_behavior_similarity(
    evaluator_logits: Tensor,
    behavior_prototypes: Tensor,
) -> Tensor:
    """Score samples by cosine similarity to skill behavior prototypes."""
    if evaluator_logits.ndim != 2:
        raise ValueError("evaluator_logits must have shape [batch, classes]")
    if behavior_prototypes.ndim != 2:
        raise ValueError("behavior_prototypes must have shape [skills, classes]")
    if evaluator_logits.shape[1] != behavior_prototypes.shape[1]:
        raise ValueError("evaluator and prototype class dimensions must match")

    probabilities = torch.softmax(evaluator_logits, dim=1)
    sample = torch.nn.functional.normalize(probabilities, dim=1)
    prototypes = torch.nn.functional.normalize(behavior_prototypes, dim=1)
    return (sample @ prototypes.T).transpose(0, 1)


def combine_skill_scores(
    compatibility_scores: Tensor,
    behavior_scores: Tensor,
    *,
    behavior_weight: float = 0.5,
) -> Tensor:
    """Combine class compatibility with evaluator behavior similarity."""
    if not 0.0 <= behavior_weight <= 1.0:
        raise ValueError("behavior_weight must be between 0 and 1")
    if compatibility_scores.shape != behavior_scores.shape:
        raise ValueError("skill score tensors must have the same shape")

    compatibility = _normalize_routing_scores(
        compatibility_scores,
        temperature=1.0,
    )
    behavior = _normalize_routing_scores(
        behavior_scores.clamp_min(0.0),
        temperature=1.0,
    )
    return (
        (1.0 - behavior_weight) * compatibility
        + behavior_weight * behavior
    )


def select_skill_from_scores(scores: Tensor) -> RoutingResult:
    """Select one skill per sample from compatibility scores."""
    if scores.ndim != 2 or scores.shape[0] < 1:
        raise ValueError("scores must have shape [skills, batch]")
    probabilities = _normalize_routing_scores(scores, temperature=1.0)
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
