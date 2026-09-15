"""Skill Memory extension using persistent class behavior fingerprints."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor

from .behavior import (
    BehaviorFingerprintCache,
    ClassBehaviorRecord,
    extract_reference_behavior,
    probe_behavior_fingerprint,
)
from .probing import predict_logits, probe_class
from .skill_memory_plugin import SkillMemoryPlugin


class PersistentFingerprintSkillMemoryPlugin(SkillMemoryPlugin):
    """Skill Memory with persistent anonymous class-behavior routing.

    Reference behavior is generated after training and reused during every
    evaluation pass. Mutable REUSE changes create a new skill generation and
    refresh every class mastered by that skill. SCRATCH creates references for
    the new skill. Unchanged skills keep their existing references.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior = BehaviorFingerprintCache()
        self._behavior_initialized = False
        self._pending_behavior_skills: set[int] = set()
        self.last_fingerprint_routes: list[dict] = []

    def _build_record(
        self,
        strategy,
        skill_id: int,
        class_id: int,
        x: Tensor,
        version: int,
    ) -> ClassBehaviorRecord:
        model = deepcopy(strategy.model)
        logits = predict_logits(model, self.memory.state(skill_id), x)
        # Keep the complete current global classifier coordinate system. The
        # head may grow later; probe_behavior_fingerprint aligns it by ID.
        output_class_ids = list(range(logits.shape[-1]))
        output, summary, output_class_ids = extract_reference_behavior(
            logits,
            output_class_ids,
            class_id,
        )
        return ClassBehaviorRecord(
            class_id=class_id,
            skill_id=skill_id,
            version=version,
            reference_inputs=x.detach().cpu().clone(),
            output_class_ids=output_class_ids,
            reference_output=output,
            reference_summary=summary,
        )

    def _refresh_skill(self, strategy, skill_id: int, experience) -> None:
        """Refresh all class references for one current skill generation."""
        version = self.behavior.skill_version(skill_id)
        classes = sorted(self.class_map.classes_for_skill(skill_id))
        if not classes:
            return

        existing = {
            record.class_id: record
            for record in self.behavior.all_records_for_skill(skill_id)
        }
        for class_id in classes:
            record = existing.get(class_id)
            if record is None:
                x, _ = probe_class(
                    experience,
                    class_id,
                    self.probe_batch_size,
                    self.probe_batches,
                    self.probe_seed,
                )
            else:
                x = record.reference_inputs

            self.behavior.put(
                self._build_record(strategy, skill_id, class_id, x, version)
            )

    def _collect_changed_skills(self, experience_index: int) -> set[int]:
        decisions = self.last_class_decisions.get(experience_index, {})
        changed: set[int] = set()
        for item in decisions.values():
            decision = item.get("decision")
            skill_id = item.get("skill")
            if skill_id is None:
                continue
            if decision == self.SCRATCH:
                changed.add(int(skill_id))
            elif decision == self.REUSE and self.reuse_is_mutable:
                changed.add(int(skill_id))
        return changed

    def before_training_exp(self, strategy, **kwargs) -> None:
        super().before_training_exp(strategy, **kwargs)
        if self._current_training_experience_index is None:
            return
        changed = self._collect_changed_skills(self._current_training_experience_index)
        # A logical Avalanche experience may contain several sub-experiences.
        # Accumulate changes until the final sub-experience has finished.
        self._pending_behavior_skills.update(changed)

    def after_training_exp(self, strategy, **kwargs) -> None:
        experience = strategy.experience
        experience_index = self._current_training_experience_index
        is_last = self._is_last_subexp(experience)

        super().after_training_exp(strategy, **kwargs)
        if experience_index is None or not is_last:
            return

        for skill_id in sorted(self._pending_behavior_skills):
            self.behavior.bump_skill(skill_id)
            self._refresh_skill(strategy, skill_id, experience)

        self._pending_behavior_skills.clear()
        self._behavior_initialized = bool(self.behavior.state_dict()["records"])

    def _fingerprint_route(
        self,
        strategy,
        x: Tensor,
        slot_ids: list[int],
    ) -> tuple[Tensor, Tensor, list[int]]:
        """Match each probe to a canonical class, then its skill."""
        probe_model = deepcopy(strategy.model)
        raw_logits = {
            slot: predict_logits(probe_model, self.memory.state(slot), x)
            for slot in slot_ids
        }

        # A class fingerprint is canonical: it belongs to the skill that
        # learned that class, but matching is performed across every class.
        # This allows a skill to master multiple classes and allows an
        # anonymous probe to identify the class before resolving its skill.
        records = [
            record
            for slot in slot_ids
            for record in self.behavior.records_for_skill(slot)
        ]
        if not records:
            self.last_fingerprint_routes = []
            chosen = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            probabilities = torch.full(
                (len(slot_ids), x.shape[0]),
                1.0 / max(len(slot_ids), 1),
                device=x.device,
            )
            return chosen, probabilities, [-1] * x.shape[0]

        class_scores: list[Tensor] = []
        class_skills: list[int] = []
        class_ids: list[int] = []
        for record in records:
            similarity_by_skill = []
            for slot in slot_ids:
                similarity, _ = probe_behavior_fingerprint(
                    raw_logits[slot],
                    record.output_class_ids,
                    record.reference_output,
                    record.reference_summary,
                )
                similarity_by_skill.append(similarity)

            # A class can be evaluated under every available skill snapshot;
            # retain the strongest behavior match for that canonical class.
            best_for_class = torch.stack(similarity_by_skill, dim=0).max(dim=0).values
            class_scores.append(best_for_class)
            class_skills.append(record.skill_id)
            class_ids.append(record.class_id)

        stacked = torch.stack(class_scores, dim=0)
        best_class_scores, best_class_indices = stacked.max(dim=0)
        chosen_classes = [
            class_ids[index] for index in best_class_indices.cpu().tolist()
        ]
        chosen_skills = [
            class_skills[index] for index in best_class_indices.cpu().tolist()
        ]

        skill_scores = torch.full(
            (len(slot_ids), x.shape[0]),
            -1.0,
            device=x.device,
        )
        slot_to_row = {slot: row for row, slot in enumerate(slot_ids)}
        for record_index, record in enumerate(records):
            row = slot_to_row.get(record.skill_id)
            if row is None:
                continue
            skill_scores[row] = torch.maximum(
                skill_scores[row], class_scores[record_index]
            )

        probabilities = torch.softmax(skill_scores, dim=0)
        chosen = torch.tensor(
            [slot_to_row[skill] for skill in chosen_skills],
            dtype=torch.long,
            device=x.device,
        )

        self.last_fingerprint_routes = []
        for sample_index, skill in enumerate(chosen_skills):
            sample_scores = skill_scores[:, sample_index]
            if sample_scores.numel() > 1:
                top_scores = torch.topk(sample_scores, k=2).values
                second_score = float(top_scores[1].item())
            else:
                second_score = float("-inf")
            row = slot_to_row[skill]
            best_score = float(best_class_scores[sample_index].item())
            self.last_fingerprint_routes.append(
                {
                    "sample_index": sample_index,
                    "skill": int(skill),
                    "class": int(chosen_classes[sample_index]),
                    "score": best_score,
                    "second_score": second_score,
                    "gap": best_score - second_score,
                    "probabilities": {
                        int(slot): float(probabilities[index, sample_index].item())
                        for index, slot in enumerate(slot_ids)
                    },
                }
            )

        return chosen, probabilities, chosen_classes

    def after_eval_forward(self, strategy, **kwargs) -> None:
        if not self._eval_active or self.eval_routing != "probe":
            return super().after_eval_forward(strategy, **kwargs)
        if len(self.memory) == 0 or not self._behavior_initialized:
            return super().after_eval_forward(strategy, **kwargs)

        x = strategy.mbatch[0]
        slot_ids = sorted(self.memory.slots())
        chosen, probabilities, class_matches = self._fingerprint_route(
            strategy,
            x,
            slot_ids,
        )

        output_model = deepcopy(strategy.model)
        per_skill_logits = [
            predict_logits(output_model, self.memory.state(slot), x)
            for slot in slot_ids
        ]
        output_dim = strategy.mb_output.shape[-1]
        padded = []
        for logits in per_skill_logits:
            result = logits.new_full((logits.shape[0], output_dim), -1e4)
            width = min(logits.shape[-1], output_dim)
            result[:, :width] = logits[:, :width]
            padded.append(result)

        strategy.mb_output = torch.stack(padded, dim=0)[
            chosen,
            torch.arange(x.shape[0], device=x.device),
        ]

        best = probabilities.max(dim=0).values
        self._log(
            "[FINGERPRINT routing] samples="
            f"{x.shape[0]} mean_similarity={best.mean().item():.4f} "
            f"matched_classes={class_matches[:5]}"
        )

    def state_dict(self) -> dict:
        """Serialize behavior state; missing state remains backward compatible."""
        return {"behavior": self.behavior.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore behavior state from a current or legacy checkpoint."""
        behavior_state = state.get("behavior", {})
        self.behavior.load_state_dict(behavior_state)
        self._behavior_initialized = bool(behavior_state.get("records"))
