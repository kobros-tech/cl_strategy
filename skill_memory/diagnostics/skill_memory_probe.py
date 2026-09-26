"""Direct Skill Memory diagnostics.

These utilities inspect stored Skill Memory states directly. They are not part
of the normal anonymous ML evaluation methodology.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..cl.skill_memory_plugin import SkillMemoryPlugin
from ..evaluation.routing import RoutingResult, select_skill_from_scores
from ..utils.probing import (
    apply_skill_state_exact,
    expand_skill_logits,
)


def _routing_scores(
    raw_skill_logits: list[torch.Tensor],
    states,
    skill_classes,
) -> torch.Tensor:
    """Score each stored skill from its own raw logits."""
    del states

    scores = []
    for logits, owned_classes in zip(raw_skill_logits, skill_classes, strict=False):
        if not owned_classes:
            scores.append(torch.zeros(logits.shape[0], device=logits.device))
            continue

        if logits.shape[1] == 1:
            scores.append(torch.sigmoid(logits[:, 0]))
            continue

        out_of_range = sorted(
            class_id
            for class_id in owned_classes
            if not 0 <= class_id < logits.shape[1]
        )
        if out_of_range:
            raise RuntimeError(
                f"skill owns classes {out_of_range} but its raw output only "
                f"has {logits.shape[1]} columns; class bookkeeping and the "
                "model's own output space have drifted apart"
            )

        probabilities = torch.softmax(logits, dim=1)
        scores.append(probabilities[:, sorted(owned_classes)].sum(dim=1))

    if not scores:
        raise RuntimeError("No skills available for probe routing.")
    return torch.stack(scores, dim=0)


def find_best_routing_skill(
    raw_skill_logits: list[torch.Tensor],
    states,
    skill_classes,
    temperature: float = 1.0,
) -> RoutingResult:
    """Select the best stored Skill Memory skill without labels."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scores = _routing_scores(raw_skill_logits, states, skill_classes)
    probabilities = scores.clamp_min(0).pow(1.0 / temperature)
    totals = probabilities.sum(dim=0, keepdim=True)
    eps = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities / totals.clamp_min(eps)

    zero_total = totals.squeeze(0) <= 0
    if zero_total.any():
        probabilities = probabilities.clone()
        probabilities[:, zero_total] = 1.0 / probabilities.shape[0]

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
    raw_skill_logits: list[torch.Tensor],
    states,
    skill_classes,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return the selected stored-skill index for each unlabeled sample."""
    return find_best_routing_skill(
        raw_skill_logits,
        states,
        skill_classes,
        temperature=temperature,
    ).skill_indices


def evaluate_skill_memory(
    model: nn.Module,
    plugin: SkillMemoryPlugin,
    test_stream,
    up_to_index: int,
    *,
    num_classes: int,
    routing: str,
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Evaluate stored Skill Memory states with oracle or probe routing."""
    if not plugin.memory:
        raise RuntimeError("Cannot evaluate Skill Memory before any skill exists.")
    if routing not in ("oracle", "probe"):
        raise ValueError(f"Unknown Skill Memory routing mode: {routing}")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    slot_ids = sorted(plugin.memory.slots())
    skill_states = [plugin.memory.state(slot) for slot in slot_ids]
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}
    class_loss: dict[int, float] = {}
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    try:
        model.eval()
        for experience_index in range(up_to_index + 1):
            experience = test_stream[experience_index]
            loader = DataLoader(
                experience.dataset, batch_size=batch_size, shuffle=False
            )
            for batch in loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                if routing == "oracle":
                    chosen_logits = torch.empty(
                        inputs.shape[0], num_classes, device=device
                    )
                    rows_by_skill: dict[int, list[int]] = {}
                    for row, label in enumerate(labels.detach().cpu().tolist()):
                        skill = plugin.class_map.find_skill_for_class_anywhere(
                            int(label)
                        )
                        if skill is None:
                            raise RuntimeError(
                                f"No canonical skill recorded for class {label}."
                            )
                        rows_by_skill.setdefault(int(skill), []).append(row)
                    for skill, rows in rows_by_skill.items():
                        state = plugin.memory.state(skill)
                        row_tensor = torch.tensor(rows, device=device)
                        apply_skill_state_exact(model, state)
                        raw_logits = model(inputs[row_tensor])
                        chosen_logits[row_tensor] = expand_skill_logits(
                            raw_logits,
                            state,
                            plugin.class_map.classes_for_skill(skill),
                            num_classes,
                        )
                else:
                    raw_logits = []
                    for state in skill_states:
                        apply_skill_state_exact(model, state)
                        raw_logits.append(model(inputs))
                    routing_result = find_best_routing_skill(
                        raw_logits,
                        skill_states,
                        [plugin.class_map.classes_for_skill(slot) for slot in slot_ids],
                    )
                    chosen = routing_result.skill_indices
                    expanded_by_skill = [
                        expand_skill_logits(
                            logits,
                            state,
                            plugin.class_map.classes_for_skill(slot),
                            num_classes,
                        )
                        for slot, state, logits in zip(
                            slot_ids, skill_states, raw_logits, strict=False
                        )
                    ]
                    stacked = torch.stack(expanded_by_skill, dim=0)
                    rows = torch.arange(inputs.shape[0], device=device)
                    chosen_logits = stacked[chosen, rows]

                per_sample_loss = nn.functional.cross_entropy(
                    chosen_logits, labels, reduction="none"
                )
                predictions = chosen_logits.argmax(dim=1)
                for class_id in torch.unique(labels).tolist():
                    class_id = int(class_id)
                    mask = labels == class_id
                    class_loss[class_id] = class_loss.get(class_id, 0.0) + float(
                        per_sample_loss[mask].sum().item()
                    )
                    class_correct[class_id] = class_correct.get(class_id, 0) + int(
                        (predictions[mask] == labels[mask]).sum().item()
                    )
                    class_total[class_id] = class_total.get(class_id, 0) + int(
                        mask.sum().item()
                    )
    finally:
        apply_skill_state_exact(model, original_state)

    results: dict[int, dict[str, float]] = {}
    for class_id in sorted(class_total):
        total = class_total[class_id]
        if total == 0:
            raise RuntimeError(f"Class {class_id} has no test samples.")
        results[class_id] = {
            "loss": class_loss[class_id] / total,
            "accuracy": class_correct[class_id] / total,
        }
    return results


def evaluate_class_oracle(
    model: nn.Module,
    plugin: SkillMemoryPlugin,
    test_stream,
    up_to_index: int,
    *,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Evaluate stored skills using the true class-to-skill mapping."""
    return evaluate_skill_memory(
        model,
        plugin,
        test_stream,
        up_to_index,
        num_classes=num_classes,
        routing="oracle",
        batch_size=batch_size,
        device=device,
    )


@torch.no_grad()
def diagnose_evaluator_probe(
    strategy,
    test_stream,
    *,
    batch_size: int,
) -> dict[str, object]:
    """Diagnose evaluator-driven probe routing without changing evaluation."""
    evaluator = strategy.ml_evaluation_plugin.evaluator_model
    if evaluator is None:
        raise RuntimeError("No independent evaluator is available.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    memory_plugin = strategy.skill_memory_plugin
    slot_ids = sorted(memory_plugin.memory.slots())
    if not slot_ids:
        raise RuntimeError("No stored skills are available for diagnostics.")

    skill_classes = [
        memory_plugin.class_map.classes_for_skill(slot) for slot in slot_ids
    ]

    evaluator.eval()
    total = 0
    correct = 0
    confidences: list[float] = []
    margins: list[float] = []
    diagnostics: list[dict[str, object]] = []

    for experience_index, experience in enumerate(test_stream):
        loader = DataLoader(
            experience.dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        for batch in loader:
            inputs = batch[0].to(strategy.device)
            labels = batch[1].to(strategy.device)
            evaluator_logits = evaluator(inputs)
            routing = select_skill_from_scores(
                _evaluator_skill_scores(evaluator_logits, skill_classes)
            )
            candidate_scores = routing.probabilities.detach().cpu().tolist()
            confidence = routing.best_probability.detach().cpu().tolist()
            margin = routing.confidence_gap.detach().cpu().tolist()

            labels_cpu = labels.detach().cpu().tolist()
            chosen_cpu = routing.skill_indices.detach().cpu().tolist()
            for row, (label, chosen_idx) in enumerate(
                zip(labels_cpu, chosen_cpu, strict=False)
            ):
                canonical = memory_plugin.class_map.find_skill_for_class_anywhere(
                    int(label)
                )
                confidences.append(float(confidence[row]))
                margins.append(float(margin[row]))
                if canonical is not None:
                    total += 1
                    if slot_ids[chosen_idx] == canonical:
                        correct += 1
                diagnostics.append(
                    {
                        "true_class": int(label),
                        "canonical_skill": (
                            int(canonical) if canonical is not None else None
                        ),
                        "evaluation_experience": experience_index,
                        "selected_skill": int(slot_ids[chosen_idx]),
                        "candidate_skills": [int(slot) for slot in slot_ids],
                        "candidate_scores": [
                            float(candidate_scores[index][row])
                            for index in range(len(candidate_scores))
                        ],
                        "top_candidates": [
                            {
                                "skill": int(slot_ids[index]),
                                "score": float(candidate_scores[index][row]),
                            }
                            for index in sorted(
                                range(len(slot_ids)),
                                key=lambda index: candidate_scores[index][row],
                                reverse=True,
                            )[:2]
                        ],
                        "confidence": float(confidence[row]),
                        "margin": float(margin[row]),
                        "correct": (
                            canonical is not None and slot_ids[chosen_idx] == canonical
                        ),
                    }
                )

    return {
        "probe_routing_accuracy": correct / total if total else float("nan"),
        "probe_mean_confidence": (
            float(torch.tensor(confidences).mean().item())
            if confidences
            else float("nan")
        ),
        "probe_mean_margin": (
            float(torch.tensor(margins).mean().item()) if margins else float("nan")
        ),
        "probe_diagnostics": diagnostics,
    }


def _evaluator_skill_scores(
    evaluator_logits: torch.Tensor,
    skill_classes,
) -> torch.Tensor:
    """Group independent-evaluator probabilities by canonical skill."""
    probabilities = torch.softmax(evaluator_logits, dim=1)
    scores = []
    for owned_classes in skill_classes:
        if not owned_classes:
            scores.append(
                torch.zeros(
                    probabilities.shape[0],
                    device=probabilities.device,
                )
            )
            continue
        out_of_range = [
            class_id
            for class_id in owned_classes
            if not 0 <= class_id < probabilities.shape[1]
        ]
        if out_of_range:
            raise RuntimeError(
                f"skill owns classes {sorted(out_of_range)} but evaluator output "
                f"has {probabilities.shape[1]} classes"
            )
        scores.append(probabilities[:, sorted(owned_classes)].sum(dim=1))
    return torch.stack(scores, dim=0)
