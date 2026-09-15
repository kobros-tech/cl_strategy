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
        skill_classes = sorted(self.class_map.classes_for_skill(skill_id))
        output, summary, output_class_ids = extract_reference_behavior(
            logits,
            skill_classes,
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
            for record in self.behavior.records_for_skill(skill_id)
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
        """Match each probe sample to a class, then to its canonical skill."""
        probe_model = deepcopy(strategy.model)
        raw_logits = {
            slot: predict_logits(probe_model, self.memory.state(slot), x)
            for slot in slot_ids
        }

        per_skill_scores: list[Tensor] = []
        per_skill_classes: list[list[int]] = []
        for slot in slot_ids:
            records = self.behavior.records_for_skill(slot)
            if not records:
                per_skill_scores.append(
                    torch.full((x.shape[0],), -1.0, device=x.device)
                )
                per_skill_classes.append([-1] * x.shape[0])
                continue

            class_scores = []
            for record in records:
                similarity, _ = probe_behavior_fingerprint(
                    raw_logits[slot],
                    record.output_class_ids,
                    record.reference_output,
                    record.reference_summary,
                )
                class_scores.append(similarity)

            stacked = torch.stack(class_scores, dim=0)
            best_scores, best_indices = stacked.max(dim=0)
            classes = [records[i].class_id for i in best_indices.cpu().tolist()]
            per_skill_scores.append(best_scores)
            per_skill_classes.append(classes)

        scores = torch.stack(per_skill_scores, dim=0)
        probabilities = torch.softmax(scores, dim=0)
        chosen = probabilities.argmax(dim=0)
        chosen_classes = [
            per_skill_classes[int(skill_index)][sample_index]
            for sample_index, skill_index in enumerate(chosen.cpu().tolist())
        ]
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

        # Compute the selected skill's logits from an isolated model. This
        # prevents routing from leaving the strategy model on an arbitrary
        # skill snapshot between minibatches.
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
