# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Public diagnostics API: anonymous routing and direct Skill Memory evaluation.

Everything in this module is deliberately **not** part of the normal
``SkillMemoryStrategy.eval()`` lifecycle (see :mod:`skill_memory.strategy`):
production evaluation always goes through the independent ML evaluator
(:mod:`skill_memory.evaluation.independent_evaluator`), which is the package's one
methodology for measuring non-forgetting. The functions here answer a
different, diagnostic question -- *could Skill Memory's own stored skills,
routed anonymously or with an oracle label, reproduce that accuracy?* -- and
are therefore opt-in, explicitly called, and never mixed into
``strategy.eval()``'s results.

Two families of helpers live here:

* :func:`find_best_routing_skill` / :func:`route_probe_logits` -- route a
  batch to one skill per sample using only each skill's *own* raw output at
  its *own* owned class columns (no shared evaluator, no learned router).
* :func:`evaluate_skill_memory` / :func:`evaluate_class_oracle` /
  :func:`diagnose_evaluator_probe` -- apply that routing (or a true-label
  oracle, or the evaluator-based probe from
  :class:`~skill_memory.evaluation.independent_evaluator.MLEvaluationPlugin`) end
  to end and report per-class accuracy/loss or routing-quality metrics.
* :func:`timing_report` -- read back where wall-clock time actually went
  during training/evaluation (decision/probing, class training, or the
  independent evaluator and Avalanche's own eval loop), so a slow run gets
  measured rather than guessed at.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluation.routing import (
    RoutingResult,
    _normalize_routing_scores,
    score_skill_compatibility,
    select_skill_from_scores,
)
from .utils.probing import expand_skill_logits, predict_logits

__all__ = [
    "find_best_routing_skill",
    "route_probe_logits",
    "diagnose_evaluator_probe",
    "evaluate_class_oracle",
    "evaluate_skill_memory",
    "timing_report",
    "reset_timing",
]


# ---------------------------------------------------------------------------
# Input-only routing from each skill's own raw response
# ---------------------------------------------------------------------------


def _skill_own_score(logits: torch.Tensor, owned_classes) -> torch.Tensor:
    r"""Score one skill's own raw logits at its own owned class columns.

    Owning zero classes scores as all-zero (handled by
    ``_normalize_routing_scores``'s uniform-fallback branch). A skill whose
    raw output has only one column (a genuine single-class head) cannot use
    a softmax at all -- softmax over one column is identically :math:`1` for
    every sample and carries no information -- so that case uses the
    column's own sigmoid instead:

    .. math::

        \text{score} =
        \begin{cases}
            0 & \text{no owned classes} \\[4pt]
            \sigma(z_0) & \text{single-column head } (C = 1) \\[4pt]
            \sum_{c \in \text{owned}} \operatorname{softmax}(z)_c
                & \text{otherwise}
        \end{cases}
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    owned = sorted(int(c) for c in owned_classes)
    if not owned:
        return torch.zeros(logits.shape[0])

    width = logits.shape[1]
    out_of_range = [c for c in owned if not 0 <= c < width]
    if out_of_range:
        raise RuntimeError(
            f"skill owns classes {out_of_range} but its own raw output is "
            f"only {width}-wide; class bookkeeping and the model have "
            "drifted apart"
        )

    if width == 1:
        return torch.sigmoid(logits[:, 0])
    return torch.softmax(logits, dim=1)[:, owned].sum(dim=1)


def find_best_routing_skill(
    logits_by_skill,
    states,
    classes_by_skill,
    *,
    temperature: float = 1.0,
) -> RoutingResult:
    """Route each sample to one skill using only each skill's own response.

    `logits_by_skill[i]` is skill ``i``'s own raw forward-pass output for
    the batch (shape ``[batch, classes_i]``); `classes_by_skill[i]` is the
    set of global class ids skill ``i`` owns. No label, task id, or
    experience id is required or used -- see :func:`_skill_own_score` for
    the per-skill scoring rule and
    :func:`skill_memory.evaluation.routing.select_skill_from_scores` for how
    the resulting per-skill scores become a probability and a selection.
    `states` (each skill's stored ``state_dict``) is accepted for interface
    symmetry with the rest of this module's state-aware helpers; it is not
    required for this computation.
    """
    del states
    scores = torch.stack(
        [
            _skill_own_score(logits, owned)
            for logits, owned in zip(logits_by_skill, classes_by_skill, strict=True)
        ],
        dim=0,
    )
    probabilities = _normalize_routing_scores(scores, temperature)
    skill_indices = probabilities.argmax(dim=0)
    if probabilities.shape[0] == 1:
        best_probability = probabilities[0]
        second_probability = torch.zeros_like(best_probability)
    else:
        top2 = torch.topk(probabilities, k=2, dim=0).values
        best_probability, second_probability = top2[0], top2[1]
    return RoutingResult(
        skill_indices=skill_indices,
        probabilities=probabilities,
        best_probability=best_probability,
        second_probability=second_probability,
        confidence_gap=best_probability - second_probability,
    )


def route_probe_logits(
    logits_by_skill,
    states,
    classes_by_skill,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return just the chosen skill index per sample.

    See :func:`find_best_routing_skill`. Kept for callers that only need
    the routing decision, not the full
    :class:`~skill_memory.evaluation.routing.RoutingResult`.
    """
    return find_best_routing_skill(
        logits_by_skill, states, classes_by_skill, temperature=temperature
    ).skill_indices


# ---------------------------------------------------------------------------
# Direct Skill Memory evaluation (oracle- or probe-routed)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_skill_memory(
    model: nn.Module,
    skill_memory_plugin,
    test_stream,
    up_to_index: int,
    *,
    num_classes: int,
    routing: str = "oracle",
    batch_size: int,
    device,
) -> dict[int, dict[str, float]]:
    """Evaluate Skill Memory's own stored skills directly, per class.

    Unlike
    :func:`~skill_memory.evaluation.independent_evaluator.evaluate_model_by_class`
    (which evaluates one independent evaluator model), this applies each
    *routed* skill's own frozen weights before scoring, so it measures
    whether Skill Memory's stored skills -- not an auxiliary learner --
    would reproduce the reported accuracy on their own. This is
    intentionally expensive (one forward pass per sample, or per sample per
    candidate skill) and is never called from the normal
    ``strategy.eval()`` lifecycle.

    ``routing="oracle"`` looks up each sample's canonical skill directly
    from its true label via
    ``skill_memory_plugin.class_map.find_skill_for_class_anywhere`` -- an
    upper bound on accuracy that presupposes knowing the label.
    ``routing="probe"`` instead routes anonymously with
    :func:`find_best_routing_skill`, using only every stored skill's own
    raw response to the very same input.

    Returns ``{class_id: {"accuracy": ..., "loss": ...}}`` for every class
    with at least one scored sample.
    """
    if routing not in ("oracle", "probe"):
        raise ValueError("routing must be one of 'oracle' or 'probe'")
    if up_to_index < 0:
        return {}
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    memory = skill_memory_plugin.memory
    class_map = skill_memory_plugin.class_map
    slot_ids = sorted(memory.slots())
    if not slot_ids:
        return {}
    owned_by_slot = [sorted(class_map.classes_for_skill(slot)) for slot in slot_ids]

    class_loss: dict[int, float] = {}
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}

    for experience_index in range(up_to_index + 1):
        experience = test_stream[experience_index]
        loader = DataLoader(experience.dataset, batch_size=batch_size, shuffle=False)

        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            if routing == "oracle":
                chosen_skill_ids: list[int | None] = [
                    class_map.find_skill_for_class_anywhere(int(label))
                    for label in y.tolist()
                ]
            else:
                per_skill_raw = [
                    predict_logits(model, memory.state(slot), x) for slot in slot_ids
                ]
                per_skill_states = [memory.state(slot) for slot in slot_ids]
                result = find_best_routing_skill(
                    per_skill_raw, per_skill_states, owned_by_slot
                )
                chosen_skill_ids = [
                    slot_ids[index] for index in result.skill_indices.tolist()
                ]

            for sample_index, skill_id in enumerate(chosen_skill_ids):
                if skill_id is None:
                    continue
                slot_index = slot_ids.index(skill_id)
                owned = owned_by_slot[slot_index]
                sample_x = x[sample_index : sample_index + 1]
                label = y[sample_index : sample_index + 1]

                raw_logits = predict_logits(model, memory.state(skill_id), sample_x)
                expanded = expand_skill_logits(
                    raw_logits.to(device),
                    memory.state(skill_id),
                    owned,
                    num_classes,
                )
                loss = nn.functional.cross_entropy(expanded, label)
                correct = int(expanded.argmax(dim=1).item() == int(label.item()))

                class_id = int(label.item())
                class_loss[class_id] = class_loss.get(class_id, 0.0) + float(
                    loss.item()
                )
                class_correct[class_id] = class_correct.get(class_id, 0) + correct
                class_total[class_id] = class_total.get(class_id, 0) + 1

    return {
        class_id: {
            "loss": class_loss[class_id] / class_total[class_id],
            "accuracy": class_correct[class_id] / class_total[class_id],
        }
        for class_id in sorted(class_total)
    }


def evaluate_class_oracle(
    model: nn.Module,
    skill_memory_plugin,
    test_stream,
    up_to_index: int,
    *,
    num_classes: int,
    batch_size: int,
    device,
) -> dict[int, dict[str, float]]:
    """Convenience wrapper: :func:`evaluate_skill_memory` with oracle routing."""
    return evaluate_skill_memory(
        model,
        skill_memory_plugin,
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
) -> dict[str, Any]:
    """Diagnose the evaluator-based probe router used by ``eval_routing="probe"``.

    Measures whether :func:`~skill_memory.evaluation.routing.score_skill_compatibility`
    plus :func:`~skill_memory.evaluation.routing.select_skill_from_scores` --
    the routing
    :class:`~skill_memory.evaluation.independent_evaluator.MLEvaluationPlugin`
    applies at evaluation time -- actually selects the skill that owns each
    sample's *true* class, independently of whether the final class
    prediction was correct. Requires `strategy.eval(...)` to have already
    been called at least once (so `strategy.evaluator_model` exists) and at
    least one skill to be stored.

    Returns ``probe_routing_accuracy`` (fraction of samples routed to their
    true owning skill), ``probe_mean_confidence``
    (mean :math:`\\max` routing probability), ``probe_mean_margin`` (mean
    gap between the best and second-best routing probability), and
    ``probe_diagnostics`` (the raw per-sample records).
    """
    plugin = strategy.skill_memory_plugin
    evaluator = strategy.evaluator_model
    if evaluator is None:
        raise RuntimeError(
            "No evaluator model is available; call strategy.eval() first."
        )
    slot_ids = sorted(plugin.memory.slots())
    if not slot_ids:
        raise RuntimeError("No skills have been stored yet.")
    skill_classes = [plugin.class_map.classes_for_skill(slot) for slot in slot_ids]

    evaluator.eval()
    device = strategy.device
    correct = 0
    total = 0
    confidences: list[float] = []
    margins: list[float] = []
    records: list[dict[str, Any]] = []

    for experience in test_stream:
        loader = DataLoader(experience.dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            evaluator_logits = evaluator(x)
            scores = score_skill_compatibility(evaluator_logits, skill_classes)
            routing = select_skill_from_scores(scores)

            for sample_index, label in enumerate(y.tolist()):
                true_skill = plugin.class_map.find_skill_for_class_anywhere(int(label))
                chosen_skill = slot_ids[int(routing.skill_indices[sample_index].item())]
                is_correct = true_skill is not None and chosen_skill == true_skill

                confidence = float(routing.best_probability[sample_index].item())
                margin = float(routing.confidence_gap[sample_index].item())
                correct += int(is_correct)
                total += 1
                confidences.append(confidence)
                margins.append(margin)
                records.append(
                    {
                        "true_class": int(label),
                        "true_skill": true_skill,
                        "chosen_skill": chosen_skill,
                        "correct": is_correct,
                        "confidence": confidence,
                        "margin": margin,
                    }
                )

    if total == 0:
        raise RuntimeError("No evaluation samples were available to diagnose.")

    return {
        "probe_routing_accuracy": correct / total,
        "probe_mean_confidence": float(np.mean(confidences)),
        "probe_mean_margin": float(np.mean(margins)),
        "probe_diagnostics": records,
    }


# ---------------------------------------------------------------------------
# Timing: find the slow part instead of guessing
# ---------------------------------------------------------------------------


def timing_report(strategy) -> dict[str, dict[str, float]]:
    """Return cumulative wall-clock time for each stage of `strategy`'s lifecycle.

    Three buckets, each accumulated since `strategy` was created (or since
    the last :func:`reset_timing`):

    - ``"skill_memory_decision_probing"`` -- every call to `decide_class`
      (the REUSE-vs-SCRATCH probing loop in
      :mod:`skill_memory.cl.decision`), timed once per class.
    - ``"skill_memory_class_training"`` -- every call to `train_on_class`
      (:mod:`skill_memory.cl.training`), timed once per class, whether it
      trained a brand-new skill from scratch or updated a reused one.
    - ``"independent_evaluator_and_test_evaluation"`` -- every call to
      `strategy.eval(...)`, timed once per call: training the independent
      evaluator on retained memory (see
      :class:`~skill_memory.evaluation.independent_evaluator.MLEvaluationPlugin`)
      plus the real Avalanche evaluation loop over the test stream.

    Each bucket reports ``total_seconds``, ``calls``, and
    ``mean_seconds`` -- comparing `total_seconds` across the three
    tells you which stage of a slow run to optimize next, instead of
    guessing from the outside.
    """
    report = strategy.skill_memory_plugin.timing.report()
    report.update(strategy.timing.report())
    return report


def reset_timing(strategy) -> None:
    """Clear every bucket :func:`timing_report` would otherwise accumulate.

    Useful for timing one specific experience or `eval()` call in
    isolation, e.g. immediately before the training/eval step you want to
    measure.
    """
    strategy.skill_memory_plugin.timing.reset()
    strategy.timing.reset()
