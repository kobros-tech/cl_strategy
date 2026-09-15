"""Skill Memory plugin extension using persistent class behavior fingerprints."""

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
from .probing import (
    classes_in_experience,
    predict_logits,
    probe_class,
)
from .skill_memory_plugin import SkillMemoryPlugin


class PersistentFingerprintSkillMemoryPlugin(SkillMemoryPlugin):
    """Skill Memory with persistent class-level anonymous routing fingerprints.

    Reference fingerprints are generated after training and reused during
    evaluation.  A mutable REUSE update invalidates the entire affected skill
    and refreshes all of its mastered classes from their persistent reference
    samples.  The routing decision itself remains label-free.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior = BehaviorFingerprintCache()
        self._behavior_initialized = False

    @staticmethod
    def _known_classes(class_map) -> list[int]:
        result: set[int] = set()
        for skill in class_map._by_skill:
            result.update(class_map.classes_for_skill(skill))
        return sorted(result)

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
        output, summary, output_class_ids = extract_reference_behavior(
            logits,
            self._known_classes(self.class_map),
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
            if record is not None:
                x = record.reference_inputs
            else:
                x, _ = probe_class(
                    experience,
                    class_id,
                    self.probe_batch_size,
                    self.probe_batches,
                    self.probe_seed,
                )
            refreshed = self._build_record(
                strategy,
                skill_id,
                class_id,
                x,
                version,
            )
            self.behavior.put(refreshed)

    def before_training_exp(self, strategy, **kwargs) -> None:
        super().before_training_exp(strategy, **kwargs)
        if self._current_training_experience_index is None:
            return
        experience_index = self._current_training_experience_index
        decisions = self.last_class_decisions.get(experience_index, {})
        changed = {
            int(item["skill"])
            for item in decisions.values()
            if item.get("decision") in (self.REUSE, self.SCRATCH)
        }
        # Training has completed for the explicit class loop before the hook
        # returns, so defer versioning/refresh until after_training_exp.
        self._pending_behavior_skills = changed

    def after_training_exp(self, strategy, **kwargs) -> None:
        experience = strategy.experience
        experience_index = self._current_training_experience_index
        super().after_training_exp(strategy, **kwargs)
        if experience_index is None:
            return

        changed = getattr(self, "_pending_behavior_skills", set())
        for skill_id in sorted(changed):
            # Every changed skill gets a new generation.  This invalidates all
            # mastered classes before their updated references are regenerated.
            self.behavior.bump_skill(skill_id)
            self._refresh_skill(strategy, skill_id, experience)
        self._behavior_initialized = bool(self.behavior.state_dict()["records"])

    def before_eval(self, strategy, **kwargs) -> None:
        super().before_eval(strategy, **kwargs)

    def _fingerprint_route(
        self,
        strategy,
        x: Tensor,
        slot_ids: list[int],
    ) -> tuple[Tensor, Tensor]:
        probe_model = deepcopy(strategy.model)
        raw_logits = {
            slot: predict_logits(probe_model, self.memory.state(slot), x)
            for slot in slot_ids
        }
        scores = []
        class_matches = []
        for sample_index in range(x.shape[0]):
            skill_scores = []
            best_classes = []
            for slot in slot_ids:
                best_similarity = torch.tensor(-1.0, device=x.device)
                best_class = -1
                for record in self.behavior.records_for_skill(slot):
                    similarity, _ = probe_behavior_fingerprint(
                        raw_logits[slot][sample_index : sample_index + 1],
                        record.output_class_ids,
                        record.reference_output,
                        record.reference_summary,
                    )
                    value = similarity[0]
                    if value > best_similarity:
                        best_similarity = value
                        best_class = record.class_id
                skill_scores.append(best_similarity)
                best_classes.append(best_class)
            values = torch.stack(skill_scores)
            scores.append(values)
            class_matches.append(best_classes[int(values.argmax().item())])

        score_tensor = torch.stack(scores, dim=1)
        probabilities = torch.softmax(score_tensor, dim=0)
        chosen = probabilities.argmax(dim=0)
        return chosen, probabilities

    def after_eval_forward(self, strategy, **kwargs) -> None:
        if not self._eval_active or self.eval_routing != "probe":
            return super().after_eval_forward(strategy, **kwargs)
        if len(self.memory) == 0 or not self._behavior_initialized:
            return super().after_eval_forward(strategy, **kwargs)

        x = strategy.mbatch[0]
        slot_ids = sorted(self.memory.slots())
        chosen, _ = self._fingerprint_route(strategy, x, slot_ids)

        output_dim = strategy.mb_output.shape[-1]
        per_skill_logits = []
        for slot in slot_ids:
            logits = predict_logits(strategy.model, self.memory.state(slot), x)
            padded = logits.new_full((logits.shape[0], output_dim), -1e4)
            width = min(logits.shape[-1], output_dim)
            padded[:, :width] = logits[:, :width]
            per_skill_logits.append(padded)
        strategy.mb_output = torch.stack(per_skill_logits, dim=0)[
            chosen, torch.arange(x.shape[0], device=x.device)
        ]

    def state_dict(self) -> dict:
        """Serialize persistent behavior state for checkpoint integration."""
        return {"behavior": self.behavior.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore behavior state; missing state is valid for old checkpoints."""
        self.behavior.load_state_dict(state.get("behavior", {}))
        self._behavior_initialized = bool(state.get("behavior", {}).get("records"))
