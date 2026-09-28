# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Evaluator-candidate -> Skill Memory arbitration.

This is the production candidate-routing implementation used by the
independent evaluator. The ML evaluator proposes candidates; Skill Memory
verifies them and may rescue a lower-ranked candidate.

The normal independent evaluator remains the candidate generator:

    x -> evaluator -> top-k candidate classes

For every candidate class, its canonical class->skill mapping identifies the
stored Skill Memory snapshot that is allowed to verify it. The skill answers
"yes" when that candidate is its own top prediction among the skill's owned
classes and its full-classifier probability reaches the configured threshold.
A singleton skill is therefore not automatically given confidence 1.0.

ML top-1 is the default decision. Skill Memory acts as a verifier/filter:
if the ML top-1 is verified, it remains the decision. If the ML top-1 is not
verified or is uncertain, a lower-ranked ML candidate may rescue it only when
that candidate has strong, independent Skill Memory verification. Among
admissible alternatives, the evaluator's ranking is preserved; independent
Skill Memory probabilities are never compared to elect a class. Evaluator and
Skill Memory probabilities are not multiplied because they are not calibrated
onto a common scale.

"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from skill_memory.utils.probing import expand_skill_logits, predict_logits


@dataclass(frozen=True)
class CandidateSkillRoutingConfig:
    """Configuration for the experimental candidate arbitration."""

    candidate_k: int = 3
    skill_confidence_threshold: float = 0.5
    rescue_skill_confidence_threshold: float = 0.85
    rescue_skill_margin: float = 0.10
    ml_uncertainty_threshold: float = 0.55
    calibration_precision_target: float = 0.90
    calibration_min_samples: int = 20
    debug: bool = False
    debug_max_samples: int = 20

    def __post_init__(self) -> None:
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if not 0.0 <= self.skill_confidence_threshold <= 1.0:
            raise ValueError("skill_confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.rescue_skill_confidence_threshold <= 1.0:
            raise ValueError("rescue_skill_confidence_threshold must be in [0, 1]")
        if self.rescue_skill_margin < 0.0:
            raise ValueError("rescue_skill_margin must be non-negative")
        if not 0.0 <= self.ml_uncertainty_threshold <= 1.0:
            raise ValueError("ml_uncertainty_threshold must be in [0, 1]")
        if not 0.0 < self.calibration_precision_target <= 1.0:
            raise ValueError("calibration_precision_target must be in (0, 1]")
        if self.calibration_min_samples < 2:
            raise ValueError("calibration_min_samples must be at least 2")
        if self.debug_max_samples < 0:
            raise ValueError("debug_max_samples must be non-negative")


def _skill_verification_probabilities(
    skill_logits: torch.Tensor,
    owned_classes: list[int],
    *,
    binary_one_vs_rest: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-classifier probabilities and owned-class scores.

    When ``binary_one_vs_rest`` is enabled, each classifier column is an
    independent YES/NO verifier trained with target-class positives and
    other-class negatives, so sigmoid(logit) is the verification probability.

    Otherwise, when the stored skill has a global classifier, confidence is
    the candidate probability under that classifier. This is different from
    softmax over
    only the owned classes: the latter makes a singleton skill's confidence
    exactly 1.0 for every input.

    A compact classifier has no non-owned output against which to normalize.
    In that case sigmoid(raw logit) is used as a conservative one-vs-rest
    applicability score rather than inventing a singleton probability of 1.0.
    """
    if skill_logits.ndim != 2:
        raise ValueError("skill_logits must have shape [batch, classes]")
    if not owned_classes:
        raise ValueError("owned_classes must not be empty")
    if binary_one_vs_rest:
        if max(owned_classes) >= skill_logits.shape[1]:
            raise ValueError(
                "binary one-vs-rest skill does not contain all owned class logits"
            )
        global_probabilities = torch.sigmoid(skill_logits)
        return global_probabilities, global_probabilities[:, owned_classes]

    if skill_logits.shape[1] > len(owned_classes):
        global_probabilities = torch.softmax(skill_logits, dim=1)
        owned_probabilities = global_probabilities[:, owned_classes]
        return global_probabilities, owned_probabilities

    if skill_logits.shape[1] != len(owned_classes):
        raise ValueError("skill classifier width does not match its owned classes")
    owned_probabilities = torch.sigmoid(skill_logits)
    return owned_probabilities, owned_probabilities


def _calibrate_skill_thresholds(
    strategy,
    skill: int,
    state,
    owned_classes: list[int],
    *,
    binary_one_vs_rest: bool,
    minimum_threshold: float,
    precision_target: float,
    min_samples: int,
) -> dict[int, float | None]:
    """Calibrate verification thresholds once from held-out skill data.

    The threshold search is vectorized over sorted validation scores. This
    avoids the previous O(classes * samples * unique_scores) loop.
    """
    metadata = strategy.ml_evaluation_plugin.memory_plugin.memory.metadata(skill)
    examples_by_class = metadata.get("verification_examples_by_class")
    if examples_by_class is not None:
        examples = list(examples_by_class.values())
    else:
        examples = metadata.get("verification_examples")
        if examples is None:
            return {class_id: minimum_threshold for class_id in owned_classes}
    # Calibration is optional evidence, not a requirement for verification.
    # Empty or undersized validation data cannot establish a precision
    # threshold, so use the configured verification floor rather than turning
    # the candidate into an impossible threshold of 1.0.
    if not examples:
        return {class_id: minimum_threshold for class_id in owned_classes}

    device = next(strategy.model.parameters()).device
    inputs = torch.cat([item[0] for item in examples], dim=0).to(device)
    targets = torch.cat([item[1] for item in examples], dim=0).to(device)
    if len(targets) < min_samples:
        return {class_id: minimum_threshold for class_id in owned_classes}

    with torch.no_grad():
        validation_logits = predict_logits(strategy.model, state, inputs)
    global_probabilities, owned_probabilities = _skill_verification_probabilities(
        validation_logits,
        owned_classes,
        binary_one_vs_rest=binary_one_vs_rest,
    )

    thresholds: dict[int, float | None] = {}
    for position, class_id in enumerate(owned_classes):
        if binary_one_vs_rest or validation_logits.shape[1] <= len(owned_classes):
            scores = owned_probabilities[:, position]
        else:
            scores = global_probabilities[:, class_id]

        positive = targets == class_id
        positive_count = int(positive.sum().item())
        negative_count = int((~positive).sum().item())
        # A singleton skill has positive validation examples but no negatives
        # inside the skill. Precision calibration is therefore undefined for
        # that class. Fall back to the configured verification floor instead
        # of using 1.0, which would reject every singleton candidate.
        if positive_count == 0 or negative_count == 0:
            thresholds[class_id] = minimum_threshold
            continue

        order = torch.argsort(scores, descending=True)
        sorted_positive = positive[order].to(torch.int64)
        sorted_scores = scores[order]
        tp = torch.cumsum(sorted_positive, dim=0)
        fp = torch.cumsum(1 - sorted_positive, dim=0)
        precision = tp.float() / (tp + fp).clamp_min(1).float()
        valid = precision >= precision_target
        if not valid.any():
            thresholds[class_id] = None
            continue

        # Collapse equal score values: a threshold must classify all samples
        # sharing that score identically.
        keep = torch.ones_like(sorted_scores, dtype=torch.bool)
        if len(sorted_scores) > 1:
            keep[1:] = sorted_scores[1:] != sorted_scores[:-1]
        valid &= keep
        if not valid.any():
            thresholds[class_id] = None
            continue

        valid_indices = valid.nonzero(as_tuple=False).flatten()
        # The lowest valid score gives the highest recall while satisfying
        # the requested precision.
        index = valid_indices[-1]
        threshold = max(minimum_threshold, float(sorted_scores[index].item()))
        predicted = scores >= threshold
        tp_final = int((predicted & positive).sum().item())
        fp_final = int((predicted & ~positive).sum().item())
        precision_final = tp_final / max(1, tp_final + fp_final)
        if precision_final < precision_target:
            thresholds[class_id] = None
        else:
            thresholds[class_id] = threshold

    return thresholds


def _cl_topk_classes(
    inputs: torch.Tensor,
    strategy,
    *,
    num_classes: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CL top-k classes and their verification scores.

    Each stored skill is evaluated with its own frozen CL snapshot, using the
    exact same probability semantics as candidate verification. The diagnostic
    returns both classes and scores so tied zero/near-zero values are visible
    instead of looking like meaningful ranked candidates.
    """
    memory = strategy.ml_evaluation_plugin.memory_plugin.memory
    class_map = strategy.ml_evaluation_plugin.memory_plugin.class_map
    slot_ids = sorted(memory.slots())
    if not slot_ids:
        raise RuntimeError("No Skill Memory skills are available.")

    class_scores = torch.full(
        (inputs.shape[0], num_classes),
        float("-inf"),
        device=inputs.device,
        dtype=torch.float32,
    )
    for skill in slot_ids:
        owned_classes = sorted(class_map.classes_for_skill(skill))
        if not owned_classes:
            continue
        state = memory.state(skill)
        metadata = memory.metadata(skill)
        binary_one_vs_rest = metadata.get("class_train_mode") == "binary_one_vs_rest"
        logits = predict_logits(strategy.model, state, inputs)
        global_probabilities, owned_probabilities = _skill_verification_probabilities(
            logits,
            owned_classes,
            binary_one_vs_rest=binary_one_vs_rest,
        )
        for position, class_id in enumerate(owned_classes):
            # Global-width heads use raw class IDs as output columns, including
            # binary one-vs-rest heads. Only compact heads use owned-class
            # positions.
            if logits.shape[1] > len(owned_classes):
                score = global_probabilities[:, class_id]
            else:
                score = owned_probabilities[:, position]
            class_scores[:, class_id] = score.float()

    topk = min(k, num_classes)
    scores, classes = torch.topk(class_scores, k=topk, dim=1)
    return classes, scores


def route_candidate_classes(
    evaluator_logits: torch.Tensor,
    inputs: torch.Tensor,
    strategy,
    *,
    candidate_k: int,
    skill_confidence_threshold: float,
    rescue_skill_confidence_threshold: float,
    rescue_skill_margin: float,
    ml_uncertainty_threshold: float,
    calibration_precision_target: float = 0.90,
    calibration_min_samples: int = 20,
    debug: bool = False,
    debug_max_samples: int = 20,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Arbitrate evaluator candidates with their canonical Skill Memory skills.

    The evaluator proposes top-k classes. Each proposed class is verified only
    by its canonical skill. A candidate must be the skill's best owned class,
    but its verification confidence is the candidate probability under the
    skill model's full classifier, not a softmax restricted to owned classes.
    This makes the score meaningful for singleton skills as well as multi-class
    skills. ML top-1 remains the default. Skill Memory acts only as a veto/filter:
    a verified ML top-1 is kept; if the ML top-1 is unverified or uncertain, the
    first strongly verified alternative in ML rank order may rescue it. The
    independent CL scores are not ranked against one another.
    """
    if evaluator_logits.ndim != 2:
        raise RuntimeError("The evaluator must return [batch, classes] logits.")

    if candidate_k < 1:
        raise ValueError("candidate_k must be positive")
    if not 0.0 <= skill_confidence_threshold <= 1.0:
        raise ValueError("skill_confidence_threshold must be in [0, 1]")
    if not 0.0 <= rescue_skill_confidence_threshold <= 1.0:
        raise ValueError("rescue_skill_confidence_threshold must be in [0, 1]")
    if rescue_skill_margin < 0.0:
        raise ValueError("rescue_skill_margin must be non-negative")
    if not 0.0 <= ml_uncertainty_threshold <= 1.0:
        raise ValueError("ml_uncertainty_threshold must be in [0, 1]")
    if not 0.0 < calibration_precision_target <= 1.0:
        raise ValueError("calibration_precision_target must be in (0, 1]")
    if calibration_min_samples < 2:
        raise ValueError("calibration_min_samples must be at least 2")

    batch_size, num_classes = evaluator_logits.shape
    candidate_k = min(candidate_k, num_classes)
    evaluator_probabilities = torch.softmax(evaluator_logits, dim=1)
    candidate_probabilities, candidate_classes = torch.topk(
        evaluator_probabilities,
        k=candidate_k,
        dim=1,
    )

    class_map = strategy.ml_evaluation_plugin.memory_plugin.class_map
    memory = strategy.ml_evaluation_plugin.memory_plugin.memory

    candidate_skills: dict[int, int] = {}
    for class_id in candidate_classes.unique().tolist():
        class_id = int(class_id)
        skill = class_map.find_skill_for_class_anywhere(class_id)
        if skill is not None:
            candidate_skills[class_id] = skill

    routed_logits = evaluator_logits.clone()
    top1_classes = candidate_classes[:, 0]
    top1_evaluator_confidence = candidate_probabilities[:, 0]
    winner_class = top1_classes.clone()

    candidate_skill_confidences = torch.zeros(
        (batch_size, candidate_k),
        device=inputs.device,
        dtype=evaluator_logits.dtype,
    )
    candidate_accepted = torch.zeros(
        (batch_size, candidate_k),
        device=inputs.device,
        dtype=torch.bool,
    )
    # Compare candidates in calibrated evidence space rather than raw
    # verifier probabilities. A class-specific threshold defines the point
    # at which a verifier is considered reliable; evidence is the amount by
    # which the candidate clears that threshold.
    candidate_thresholds = torch.full(
        (batch_size, candidate_k),
        float(skill_confidence_threshold),
        device=inputs.device,
        dtype=evaluator_logits.dtype,
    )
    debug_records: list[dict[int, dict[str, object]]] = [{} for _ in range(batch_size)]

    for row in range(batch_size):
        for rank in range(candidate_k):
            class_id = int(candidate_classes[row, rank].item())
            debug_records[row][class_id] = {
                "class": class_id,
                "rank": rank + 1,
                "evaluator_confidence": float(
                    candidate_probabilities[row, rank].item()
                ),
                "skill": candidate_skills.get(class_id),
                "skill_confidence": None,
                "accepted": False,
            }

    rows_by_skill: dict[int, list[int]] = {}
    for row in range(batch_size):
        for candidate_class in candidate_classes[row].tolist():
            skill = candidate_skills.get(int(candidate_class))
            if skill is not None:
                rows_by_skill.setdefault(skill, []).append(row)

    for skill, row_list in rows_by_skill.items():
        rows = torch.tensor(
            sorted(set(row_list)),
            device=inputs.device,
            dtype=torch.long,
        )
        state = memory.state(skill)
        skill_metadata = memory.metadata(skill)
        binary_one_vs_rest = (
            skill_metadata.get("class_train_mode") == "binary_one_vs_rest"
        )
        skill_logits = predict_logits(strategy.model, state, inputs[rows])
        owned_classes = sorted(class_map.classes_for_skill(skill))
        eval_experience = int(
            getattr(getattr(strategy, "experience", None), "current_experience", -1)
        )
        cache = getattr(strategy, "_candidate_skill_calibration_cache", None)
        if cache is None:
            cache = {}
            strategy._candidate_skill_calibration_cache = cache
        cache_key = (eval_experience, skill)
        calibrated_thresholds = cache.get(cache_key)
        if calibrated_thresholds is None:
            calibrated_thresholds = _calibrate_skill_thresholds(
                strategy,
                skill,
                state,
                owned_classes,
                binary_one_vs_rest=binary_one_vs_rest,
                minimum_threshold=skill_confidence_threshold,
                precision_target=calibration_precision_target,
                min_samples=calibration_min_samples,
            )
            cache[cache_key] = calibrated_thresholds
        if not owned_classes:
            continue

        # Verification has two separate questions:
        # 1. Is this class the skill's best owned class?
        # 2. Does the skill model actually assign substantial probability to
        #    it when all classifier outputs are considered?
        #
        # Never softmax only over owned_classes for confidence. For a
        # singleton skill that would make confidence exactly 1.0 for every
        # input, turning "owns this class" into a false verification signal.
        global_probabilities, owned_probabilities = _skill_verification_probabilities(
            skill_logits,
            owned_classes,
            binary_one_vs_rest=binary_one_vs_rest,
        )
        skill_best_index = owned_probabilities.argmax(dim=1)
        skill_class_tensor = torch.tensor(
            owned_classes,
            device=inputs.device,
            dtype=torch.long,
        )
        skill_best_classes = skill_class_tensor[skill_best_index]

        for rank in range(candidate_k):
            candidate_class_tensor = candidate_classes[rows, rank]
            candidate_skill_confidence = torch.zeros_like(
                candidate_probabilities[rows, rank]
            )
            candidate_is_best = torch.zeros(
                rows.shape[0],
                device=inputs.device,
                dtype=torch.bool,
            )

            for position, class_id in enumerate(owned_classes):
                mask = candidate_class_tensor == class_id
                if mask.any():
                    if skill_logits.shape[1] > len(owned_classes):
                        candidate_skill_confidence[mask] = global_probabilities[
                            mask, class_id
                        ]
                    else:
                        candidate_skill_confidence[mask] = owned_probabilities[
                            mask, position
                        ]
                    candidate_is_best[mask] = (
                        torch.ones_like(skill_best_classes[mask], dtype=torch.bool)
                        if binary_one_vs_rest
                        else skill_best_classes[mask] == class_id
                    )

            thresholds_for_rows = torch.full_like(
                candidate_skill_confidence,
                float(skill_confidence_threshold),
            )
            threshold_available = torch.ones(
                rows.shape[0],
                device=inputs.device,
                dtype=torch.bool,
            )
            if (
                skill_metadata.get("verification_examples_by_class") is not None
                or skill_metadata.get("verification_examples") is not None
            ):
                threshold_values = [
                    calibrated_thresholds.get(int(class_value))
                    for class_value in candidate_class_tensor.tolist()
                ]
                threshold_available = torch.tensor(
                    [value is not None for value in threshold_values],
                    device=inputs.device,
                    dtype=torch.bool,
                )
                thresholds_for_rows = torch.tensor(
                    [
                        float(value) if value is not None else 1.0
                        for value in threshold_values
                    ],
                    device=inputs.device,
                    dtype=candidate_skill_confidence.dtype,
                )
            accepted = (
                candidate_is_best
                & threshold_available
                & (candidate_skill_confidence >= thresholds_for_rows)
            )
            candidate_skill_confidences[rows, rank] = candidate_skill_confidence
            candidate_thresholds[rows, rank] = thresholds_for_rows
            candidate_accepted[rows, rank] = accepted

            for position, row in enumerate(rows.tolist()):
                class_id = int(candidate_class_tensor[position].item())
                # A debug record is keyed by candidate class, but this skill
                # only owns owned_classes. A candidate from another skill
                # must not be indexed into this skill's compact/binary logits.
                if class_id not in owned_classes:
                    continue
                record = debug_records[row].get(class_id)
                if record is not None:
                    record["skill_confidence"] = float(
                        candidate_skill_confidence[position].item()
                    )
                    # Match the exact output-column semantics used above
                    # for the verification score: global-width heads use the
                    # global class ID; compact heads use owned-class position.
                    class_position = owned_classes.index(class_id)
                    if skill_logits.shape[1] > len(owned_classes):
                        record["skill_logit"] = float(
                            skill_logits[position, class_id].item()
                        )
                    else:
                        record["skill_logit"] = float(
                            skill_logits[position, class_position].item()
                        )
                    record["verification_threshold"] = float(
                        thresholds_for_rows[position].item()
                    )
                    record["accepted"] = bool(accepted[position].item())

    top1_verified = candidate_accepted[:, 0]

    # Skill Memory is a verifier/filter, not a second classifier whose
    # independent sigmoid scores can be ranked against one another. In
    # particular, binary one-vs-rest skills can legitimately give several
    # classes high scores for the same input. Therefore:
    #
    #   1. ML evaluator chooses the primary top-1 class.
    #   2. If ML top-1 is verified, keep it.
    #   3. Only if ML top-1 is not verified/too uncertain may CL admit an
    #      alternative from the evaluator's existing top-k list.
    #   4. Among admissible alternatives, preserve the evaluator's ranking;
    #      CL never elects the alternative by comparing independent verifier
    #      probabilities.
    candidate_evidence = candidate_skill_confidences - candidate_thresholds
    rescue_candidate = torch.full(
        (batch_size,),
        -1,
        device=inputs.device,
        dtype=torch.long,
    )
    rescue_confidence = torch.zeros(
        batch_size,
        device=inputs.device,
        dtype=evaluator_logits.dtype,
    )
    rescue_evidence = torch.full(
        (batch_size,),
        float("-inf"),
        device=inputs.device,
        dtype=evaluator_logits.dtype,
    )
    rescue_mask = torch.zeros(
        (batch_size,),
        device=inputs.device,
        dtype=torch.bool,
    )

    top1_needs_rescue = (top1_evaluator_confidence < ml_uncertainty_threshold) | (
        ~top1_verified
    )

    # Scan ML candidates from rank 2 upward. The first candidate that is
    # strongly verified by its canonical skill becomes the rescue candidate.
    # This preserves the ML evaluator's ordering and makes CL a veto/filter.
    for rank in range(1, candidate_k):
        candidate_confidence = candidate_skill_confidences[:, rank]
        strong_rescue = candidate_accepted[:, rank] & (
            candidate_confidence >= rescue_skill_confidence_threshold
        )
        eligible = top1_needs_rescue & strong_rescue
        choose = eligible & ~rescue_mask
        rescue_candidate[choose] = candidate_classes[choose, rank]
        rescue_confidence[choose] = candidate_confidence[choose]
        rescue_evidence[choose] = candidate_evidence[choose, rank]
        rescue_mask |= choose

    accepted_rows = rescue_mask.clone()
    if rescue_mask.any():
        for skill in sorted(set(candidate_skills.values())):
            owned_classes = sorted(class_map.classes_for_skill(skill))
            if not owned_classes:
                continue
            owned_tensor = torch.tensor(
                owned_classes,
                device=inputs.device,
                dtype=torch.long,
            )
            rows = (
                (rescue_mask & torch.isin(rescue_candidate, owned_tensor))
                .nonzero(as_tuple=False)
                .flatten()
            )
            if not len(rows):
                continue

            state = memory.state(skill)
            skill_logits = predict_logits(strategy.model, state, inputs[rows])
            expanded = expand_skill_logits(
                skill_logits,
                state,
                owned_classes,
                evaluator_logits.shape[1],
            )
            if memory.metadata(skill).get("class_train_mode") == ("binary_one_vs_rest"):
                # Binary skill outputs are independent YES/NO verifiers.
                # Once this candidate wins arbitration, suppress the other
                # owned-class logits so the elected candidate is also the
                # routed-logit argmax.
                selected = rescue_candidate[rows]
                for class_id in owned_classes:
                    expanded[:, class_id] = torch.where(
                        selected == class_id,
                        expanded[:, class_id],
                        expanded.new_full((expanded.shape[0],), -20.0),
                    )
            routed_logits[rows] = expanded
            winner_class[rows] = rescue_candidate[rows]

    final_classes = routed_logits.argmax(dim=1)
    inconsistent = accepted_rows & (final_classes != winner_class)
    if inconsistent.any():
        rows = inconsistent.nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "Candidate-skill routing elected class does not match routed "
            f"logit argmax for rows {rows}."
        )

    fallback_rows = ~accepted_rows

    if debug:
        debug_rows = min(batch_size, debug_max_samples)
        cl_topk_classes, cl_topk_scores = _cl_topk_classes(
            inputs[:debug_rows],
            strategy,
            num_classes=num_classes,
            k=candidate_k,
        )
        for row in range(debug_rows):
            print(f"[CANDIDATE ROUTING] sample={row}")
            ml_topk = candidate_classes[row].tolist()
            cl_topk = cl_topk_classes[row].tolist()
            cl_scores = cl_topk_scores[row].tolist()
            cl_topk_with_scores = [
                (int(class_id), float(score))
                for class_id, score in zip(cl_topk, cl_scores, strict=True)
            ]
            intersection = sorted(set(ml_topk).intersection(cl_topk))
            target = int(strategy.mbatch[1][row].item())
            print(
                "  TOP-K INTERSECTION: "
                f"ML top{candidate_k}={ml_topk} "
                f"CL top{candidate_k}={cl_topk_with_scores} "
                f"intersection={intersection} "
                f"contains_correct={target in intersection}"
            )
            print("  ML candidates:")
            for record in sorted(
                debug_records[row].values(),
                key=lambda item: item["rank"],
            ):
                skill = record["skill"]
                skill_text = "none" if skill is None else str(skill)
                print(
                    f"    #{record['rank']} class={record['class']} "
                    f"confidence={record['evaluator_confidence']:.4f} "
                    f"skill={skill_text}"
                )
            print("  CL / Skill Memory verification:")
            for class_id, record in sorted(
                debug_records[row].items(),
                key=lambda item: item[1]["rank"],
            ):
                skill = record["skill"]
                if skill is None:
                    print(
                        f"    class={class_id} skill=none REJECT (no canonical skill)"
                    )
                    continue
                status = "ACCEPT" if record["accepted"] else "REJECT"
                print(
                    f"    class={class_id} skill={skill} "
                    f"logit={record['skill_logit']:.4f} "
                    f"verification_score={record['skill_confidence']:.4f} "
                    f"threshold={record['verification_threshold']:.4f} {status}"
                )
            if rescue_mask[row]:
                print(
                    f"  DECISION: ML top1=class={int(top1_classes[row].item())} "
                    "-> rescue by strong Skill Memory candidate"
                )
                print(
                    f"  ELECTED: class={int(winner_class[row].item())} "
                    "source=Skill Memory reason=strong_verified_rescue "
                    f"skill_confidence={rescue_confidence[row].item():.4f} "
                    f"calibrated_evidence={rescue_evidence[row].item():.4f}"
                )
            else:
                reason = (
                    "top1_skill_verified"
                    if top1_verified[row]
                    else "no_strong_verified_rescue"
                )
                print(
                    f"  ELECTED: class={int(winner_class[row].item())} "
                    f"source=ML evaluator reason={reason}"
                )
            print(f"  GROUND TRUTH (metrics only): class={target}")

    targets = strategy.mbatch[1]
    ml_correct = top1_classes == targets
    final_correct = final_classes == targets
    cl_override = accepted_rows & (final_classes != top1_classes)
    cl_corrected_ml_error = cl_override & ~ml_correct & final_correct
    cl_introduced_error = cl_override & ml_correct & ~final_correct
    stats = {
        "ml_top1_accuracy": float(ml_correct.float().mean().item()),
        "final_batch_accuracy": float(final_correct.float().mean().item()),
        "cl_override_rate": float(cl_override.float().mean().item()),
        "cl_corrected_ml_error_rate": float(
            cl_corrected_ml_error.float().mean().item()
        ),
        "cl_introduced_error_rate": float(cl_introduced_error.float().mean().item()),
        "cl_net_accuracy_gain": float(
            (final_correct.float() - ml_correct.float()).mean().item()
        ),
        "cl_override_precision": float(
            cl_corrected_ml_error.float().sum().item()
            / max(float(cl_override.float().sum().item()), 1.0)
        ),
        "elected_class_matches_routed_argmax": float(
            (winner_class == final_classes).float().mean().item()
        ),
        "candidate_top1_accuracy": float(
            (top1_classes == strategy.mbatch[1]).float().mean().item()
        ),
        "candidate_topk_recall": float(
            (candidate_classes == strategy.mbatch[1].unsqueeze(1))
            .any(dim=1)
            .float()
            .mean()
            .item()
        ),
        "skill_acceptance_rate": float(
            candidate_accepted.any(dim=1).float().mean().item()
        ),
        "fallback_rate": float(fallback_rows.float().mean().item()),
        "skill_override_rate": float(
            (accepted_rows & (final_classes != top1_classes)).float().mean().item()
        ),
        "rescue_rate": float(rescue_mask.float().mean().item()),
        "top1_skill_verified_rate": float(top1_verified.float().mean().item()),
        "mean_rescue_skill_confidence": float(
            rescue_confidence[rescue_mask].mean().item() if rescue_mask.any() else 0.0
        ),
    }
    return routed_logits, stats

