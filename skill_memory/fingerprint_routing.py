"""Evidence-first persistent fingerprint routing.

The original fingerprint router used reconstructed argmax behavior as a hard
compatibility gate. That is not an independent fingerprint when each stored
skill is trained on its own class: a one-class skill can predict its mastered
class for unrelated inputs. This module keeps that binary result as a
diagnostic signal, but uses the persistent feature/margin evidence to make the
actual class decision.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor

from .behavior import (
    compare_binary_behavior,
    extract_features_from_weights,
    reverse_engineer_scores_from_weights,
    reverse_engineer_y,
)
from .persistent_skill_memory_plugin import PersistentFingerprintSkillMemoryPlugin as _BaseFingerprintPlugin
from .probing import apply_skill_state_exact, predict_logits


class PersistentFingerprintSkillMemoryPlugin(_BaseFingerprintPlugin):
    """Persistent fingerprint router with independent continuous evidence."""

    def _fingerprint_route(
        self,
        strategy,
        x: Tensor,
        slot_ids: list[int],
    ) -> tuple[Tensor, list[int]]:
        """Identify a class from all stored evidence, then resolve its skill.

        Binary reconstructed ``y`` is recorded for diagnosis but is not a hard
        gate. With the current learning semantics a SCRATCH skill is trained on
        one class only, so ``argmax == candidate`` can remain true for unrelated
        inputs. Treating that result as identity creates the exact class-1
        routing collapse seen in the SplitMNIST artifact.
        """
        probe_model = deepcopy(strategy.model)
        candidate_records = [
            record
            for slot in slot_ids
            for record in self.behavior.records_for_skill(slot)
        ]
        skill_scores: dict[int, Tensor] = {}
        skill_features: dict[int, Tensor] = {}

        for slot in slot_ids:
            records = self.behavior.records_for_skill(slot)
            if not records:
                continue
            version = records[0].version
            frozen_state = self.behavior.skill_state(slot, version)
            if frozen_state is None:
                frozen_state = self.memory.state(slot)
            if self._custom_reverse_engineer_y is None:
                apply_skill_state_exact(probe_model, frozen_state)
                skill_features[slot] = extract_features_from_weights(probe_model, x)
                skill_scores[slot] = reverse_engineer_scores_from_weights(
                    probe_model, x
                )
            else:
                skill_scores[slot] = predict_logits(probe_model, frozen_state, x)

        chosen_skills: list[int] = []
        chosen_classes: list[int] = []
        routes: list[dict] = []

        for sample_index in range(x.shape[0]):
            candidates: list[dict] = []
            for record in candidate_records:
                scores = skill_scores.get(record.skill_id)
                if scores is None or not 0 <= record.class_id < scores.shape[-1]:
                    continue

                predicted_class = int(scores[sample_index].argmax().item())
                predicted_y = predicted_class == record.class_id
                comparison = compare_binary_behavior(
                    torch.tensor([predicted_y]), record.expected_y
                )
                candidate = {
                    "class": record.class_id,
                    "skill": record.skill_id,
                    "predicted_y": predicted_y,
                    "binary_compatible": bool(comparison["all_correct"]),
                    "predicted_class": predicted_class,
                    "expected_y": record.expected_y,
                    "reference_y": record.reference_y.tolist(),
                    "reference_accuracy": record.reference_accuracy,
                    "correct": bool(comparison["all_correct"]),
                    "class_score": float(
                        scores[sample_index, record.class_id].item()
                    ),
                }

                features = skill_features.get(record.skill_id)
                if features is not None:
                    evidence = self._continuous_evidence(
                        record, features, scores, sample_index
                    )
                    candidate.update(
                        {
                            "feature_similarity": evidence["feature_similarity"],
                            "margin_similarity": evidence["margin_similarity"],
                            "continuous_evidence": evidence["evidence"],
                        }
                    )
                else:
                    candidate["continuous_evidence"] = 0.0
                candidates.append(candidate)

            ranked = sorted(
                candidates,
                key=lambda item: float(item["continuous_evidence"]),
                reverse=True,
            )
            selected = self._select_continuous_candidate(ranked)
            if not candidates:
                status = "FAILED"
            elif selected is None:
                status = "AMBIGUOUS"
            else:
                status = "IDENTIFIED"
                chosen_classes.append(int(selected["class"]))
                chosen_skills.append(int(selected["skill"]))

            if status != "IDENTIFIED":
                chosen_classes.append(-1)
                chosen_skills.append(-1)

            routes.append(
                {
                    "sample_index": sample_index,
                    "status": status,
                    "class": None if selected is None else selected["class"],
                    "skill": None if selected is None else selected["skill"],
                    "binary_compatible_candidates": sum(
                        item["binary_compatible"] for item in candidates
                    ),
                    "candidates": candidates,
                }
            )

        self.last_fingerprint_routes = routes
        chosen = torch.tensor(chosen_skills, dtype=torch.long, device=x.device)
        return chosen, chosen_classes
