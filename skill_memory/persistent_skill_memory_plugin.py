"""Skill Memory extension using binary reverse-engineered class behavior."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import Tensor

from .behavior import (
    BehaviorFingerprintCache,
    ClassBehaviorRecord,
    build_weight_behavior_statistics,
    compare_binary_behavior,
    extract_features_from_weights,
    reverse_engineer_scores_from_weights,
    reverse_engineer_y_from_weights,
)
from .probing import apply_skill_state_exact, predict_logits, probe_class
from .skill_memory_plugin import SkillMemoryPlugin


class PersistentFingerprintSkillMemoryPlugin(SkillMemoryPlugin):
    """Skill Memory with persistent binary class-behavior identification."""

    def __init__(
        self,
        *args,
        reverse_engineer_y_fn: Callable | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.behavior = BehaviorFingerprintCache()
        self._custom_reverse_engineer_y = reverse_engineer_y_fn
        self.reverse_engineer_y = reverse_engineer_y_from_weights
        self._behavior_initialized = False
        self._pending_reference_inputs: dict[int, Tensor] = {}
        self.last_fingerprint_routes: list[dict] = []
        self.fingerprint_route_history: list[dict] = []
        self._fingerprint_batch_index = 0
        self._evaluation_experience_index: int | None = None

    def _reverse_engineer_y(
        self,
        model,
        state_dict: dict,
        x: Tensor,
        class_id: int,
    ) -> Tensor:
        """Run the configured reverse-engineering method without label leakage."""
        if self._custom_reverse_engineer_y is not None:
            logits = predict_logits(model, state_dict, x)
            return self._custom_reverse_engineer_y(logits, class_id).detach().bool()

        apply_skill_state_exact(model, state_dict)
        return reverse_engineer_y_from_weights(model, x, class_id).detach().bool()

    def _build_record(
        self,
        strategy,
        skill_id: int,
        class_id: int,
        x: Tensor,
        version: int,
        state_dict: dict | None = None,
    ) -> ClassBehaviorRecord:
        """Create a class record from the frozen skill-generation state."""
        if state_dict is None:
            state_dict = self.memory.state(skill_id)
        model = deepcopy(strategy.model)
        reference_y = self._reverse_engineer_y(
            model,
            state_dict,
            x,
            class_id,
        ).cpu()
        apply_skill_state_exact(model, state_dict)
        statistics = build_weight_behavior_statistics(model, x, class_id)
        return ClassBehaviorRecord(
            class_id=class_id,
            skill_id=skill_id,
            version=version,
            reference_inputs=x.detach().cpu().clone(),
            reference_y=reference_y,
            reference_feature_mean=statistics["feature_mean"],
            reference_feature_std=statistics["feature_std"],
            reference_margin_mean=statistics["margin_mean"],
            reference_margin_std=statistics["margin_std"],
            reference_weight=statistics["weight"],
            reference_bias=statistics["bias"],
        )

    def _capture_new_class_inputs(self, experience, experience_index: int) -> None:
        """Keep deterministic probe inputs for newly introduced classes."""
        decisions = self.last_class_decisions.get(experience_index, {})
        for class_id, item in decisions.items():
            skill_id = item.get("skill")
            if skill_id is None:
                continue
            if self.behavior.get(class_id, int(skill_id)) is not None:
                continue
            if class_id in self._pending_reference_inputs:
                continue
            x, _ = probe_class(
                experience,
                class_id,
                self.probe_batch_size,
                self.probe_batches,
                self.probe_seed,
            )
            self._pending_reference_inputs[class_id] = x.detach().cpu().clone()

    def _refresh_skill(self, strategy, skill_id: int, experience) -> None:
        """Refresh every class fingerprint owned by the current skill generation."""
        version = self.behavior.skill_version(skill_id)
        state_dict = self.memory.state(skill_id)
        self.behavior.put_skill_state(skill_id, version, state_dict)
        classes = sorted(self.class_map.classes_for_skill(skill_id))
        existing = {
            record.class_id: record
            for record in self.behavior.all_records_for_skill(skill_id)
        }
        for class_id in classes:
            record = existing.get(class_id)
            x = (
                record.reference_inputs
                if record is not None
                else self._pending_reference_inputs.get(class_id)
            )
            if x is None:
                x, _ = probe_class(
                    experience,
                    class_id,
                    self.probe_batch_size,
                    self.probe_batches,
                    self.probe_seed,
                )
            self.behavior.put(
                self._build_record(
                    strategy,
                    skill_id,
                    class_id,
                    x,
                    version,
                    state_dict,
                )
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

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Refresh changed skills after the complete logical experience."""
        experience = strategy.experience
        experience_index = self._current_training_experience_index
        is_last = self._is_last_subexp(experience)

        if experience_index is not None:
            self._capture_new_class_inputs(experience, experience_index)

        changed = (
            self._collect_changed_skills(experience_index)
            if is_last and experience_index is not None
            else set()
        )
        pending = dict(self._pending_reference_inputs)
        super().after_training_exp(strategy, **kwargs)
        if experience_index is None or not is_last:
            return

        for skill_id in sorted(changed):
            self.behavior.bump_skill(skill_id)
            self._refresh_skill(strategy, skill_id, experience)

        for class_id, x in sorted(pending.items()):
            skill_id = self.class_map.find_skill_for_class_anywhere(class_id)
            if skill_id is None:
                continue
            if self.behavior.get(class_id, int(skill_id)) is not None:
                continue
            version = self.behavior.skill_version(int(skill_id))
            state_dict = self.behavior.skill_state(int(skill_id), version)
            if state_dict is None:
                state_dict = self.memory.state(int(skill_id))
                self.behavior.put_skill_state(
                    int(skill_id), version, state_dict
                )
            self.behavior.put(
                self._build_record(
                    strategy,
                    int(skill_id),
                    int(class_id),
                    x,
                    version,
                    state_dict,
                )
            )

        self._pending_reference_inputs.clear()
        self._behavior_initialized = bool(self.behavior.state_dict()["records"])

    @staticmethod
    def _continuous_evidence(
        record: ClassBehaviorRecord,
        features: Tensor,
        scores: Tensor,
        sample_index: int,
    ) -> dict[str, float]:
        """Score one binary-compatible candidate using persistent evidence."""
        feature = features[sample_index].detach().float().cpu()
        components: list[float] = []

        if record.reference_feature_mean is not None:
            reference = record.reference_feature_mean.float()
            cosine = F.cosine_similarity(
                feature.unsqueeze(0), reference.unsqueeze(0), dim=1
            ).item()
            components.append((float(cosine) + 1.0) / 2.0)

        class_id = record.class_id
        row = scores[sample_index]
        own = row[class_id]
        if row.numel() > 1:
            other = row.clone()
            other[class_id] = -torch.inf
            margin = own - other.max()
        else:
            margin = own
        margin_similarity = torch.exp(
            -torch.abs(
                margin.detach().cpu()
                - float(record.reference_margin_mean)
            )
            / max(float(record.reference_margin_std), 1e-6)
        ).item()
        components.append(float(margin_similarity))

        if record.reference_weight is not None:
            weight = row.new_tensor(record.reference_weight).float().cpu()
            current_weight = record.reference_weight.new_tensor(
                record.reference_weight
            ).float()
            # The current classifier weight is recovered from the score model
            # through the persistent skill state in the caller.  The reference
            # weight is therefore retained for checkpoint compatibility, while
            # feature/margin evidence remains sample-specific.
            del weight, current_weight

        return {
            "feature_similarity": components[0] if record.reference_feature_mean is not None else 0.0,
            "margin_similarity": float(margin_similarity),
            "evidence": float(sum(components) / len(components)),
        }

    @staticmethod
    def _select_continuous_candidate(
        compatible: list[dict],
    ) -> dict | None:
        """Select a candidate only when the evidence has a clear top cluster."""
        if len(compatible) <= 1:
            return compatible[0] if compatible else None

        ranked = sorted(
            compatible,
            key=lambda item: float(item["continuous_evidence"]),
            reverse=True,
        )
        gaps = [
            float(ranked[index]["continuous_evidence"])
            - float(ranked[index + 1]["continuous_evidence"])
            for index in range(len(ranked) - 1)
        ]
        largest_gap_index = max(range(len(gaps)), key=gaps.__getitem__)
        if largest_gap_index != 0:
            return None
        if len(gaps) == 1:
            return ranked[0]

        remaining_gaps = gaps[1:]
        baseline = sum(remaining_gaps) / len(remaining_gaps)
        if gaps[0] > baseline:
            return ranked[0]
        return None

    def _fingerprint_route(
        self,
        strategy,
        x: Tensor,
        slot_ids: list[int],
    ) -> tuple[Tensor, list[int]]:
        """Identify a class, then resolve its canonical skill mapping."""
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
            matches: list[dict] = []
            for record in candidate_records:
                scores = skill_scores.get(record.skill_id)
                if scores is None or not 0 <= record.class_id < scores.shape[-1]:
                    continue
                predicted_class = int(scores[sample_index].argmax().item())
                predicted_y = predicted_class == record.class_id
                comparison = compare_binary_behavior(
                    torch.tensor([predicted_y]), record.expected_y
                )
                match = {
                    "class": record.class_id,
                    "skill": record.skill_id,
                    "predicted_y": predicted_y,
                    "predicted_class": predicted_class,
                    "expected_y": record.expected_y,
                    "reference_y": record.reference_y.tolist(),
                    "reference_accuracy": record.reference_accuracy,
                    "correct": bool(comparison["all_correct"]),
                    "class_score": float(
                        scores[sample_index, record.class_id].item()
                    ),
                }
                if match["correct"]:
                    features = skill_features.get(record.skill_id)
                    if features is not None:
                        evidence = self._continuous_evidence(
                            record, features, scores, sample_index
                        )
                        match.update(
                            {
                                "feature_similarity": evidence["feature_similarity"],
                                "margin_similarity": evidence["margin_similarity"],
                                "continuous_evidence": evidence["evidence"],
                            }
                        )
                matches.append(match)

            compatible = [item for item in matches if item["correct"]]
            selected = self._select_continuous_candidate(compatible)
            if not compatible:
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
                    "candidates": matches,
                }
            )

        self.last_fingerprint_routes = routes
        chosen = torch.tensor(chosen_skills, dtype=torch.long, device=x.device)
        return chosen, chosen_classes

    def before_eval(self, strategy, **kwargs) -> None:
        """Start a fresh routing-analysis trace for each evaluation pass."""
        super().before_eval(strategy, **kwargs)
        self.fingerprint_route_history = []
        self._fingerprint_batch_index = 0
        self._evaluation_experience_index = None

    def before_eval_exp(self, strategy, **kwargs) -> None:
        """Record the evaluation experience without using it for routing."""
        super().before_eval_exp(strategy, **kwargs)
        experience = strategy.experience
        index = getattr(experience, "current_experience", None)
        if index is None:
            index = getattr(experience, "experience_id", None)
        self._evaluation_experience_index = None if index is None else int(index)

    def after_eval_forward(self, strategy, **kwargs) -> None:
        if not self._eval_active or self.eval_routing != "probe":
            return super().after_eval_forward(strategy, **kwargs)
        if len(self.memory) == 0 or not self._behavior_initialized:
            return super().after_eval_forward(strategy, **kwargs)

        x = strategy.mbatch[0]
        y = strategy.mbatch[1]
        slot_ids = sorted(self.memory.slots())
        chosen, class_matches = self._fingerprint_route(strategy, x, slot_ids)

        for route, label in zip(
            self.last_fingerprint_routes,
            y.detach().cpu().tolist(),
            strict=False,
        ):
            route["batch_index"] = self._fingerprint_batch_index
            route["evaluation_experience"] = self._evaluation_experience_index
            route["evaluation_y"] = int(label)
        self.fingerprint_route_history.extend(self.last_fingerprint_routes)
        self._fingerprint_batch_index += 1

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

        valid = chosen.ge(0)
        if valid.any():
            positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            skill_to_row = {skill_id: row for row, skill_id in enumerate(slot_ids)}
            rows = torch.tensor(
                [skill_to_row[int(skill)] for skill in chosen[valid].tolist()],
                device=chosen.device,
                dtype=torch.long,
            )
            stacked = torch.stack(padded, dim=0)
            strategy.mb_output[positions] = stacked[rows, positions]

        final_predictions = strategy.mb_output.detach().argmax(dim=-1).cpu().tolist()
        labels = y.detach().cpu().tolist()
        for route, prediction, label in zip(
            self.last_fingerprint_routes,
            final_predictions,
            labels,
            strict=False,
        ):
            route["model_predicted_class"] = int(prediction)
            route["model_correct"] = int(prediction) == int(label)

        identified = sum(
            item["status"] == "IDENTIFIED" for item in self.last_fingerprint_routes
        )
        ambiguous = sum(
            item["status"] == "AMBIGUOUS" for item in self.last_fingerprint_routes
        )
        failed = sum(
            item["status"] == "FAILED" for item in self.last_fingerprint_routes
        )
        self._log(
            "[WEIGHT fingerprint routing] "
            f"eval_exp={self._evaluation_experience_index} "
            f"samples={x.shape[0]} identified={identified} "
            f"ambiguous={ambiguous} failed={failed} "
            f"matched_classes={class_matches[:5]}"
        )

    def state_dict(self) -> dict:
        """Serialize only the persistent binary behavior state."""
        return {"behavior": self.behavior.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore binary behavior state; old checkpoints remain loadable."""
        behavior_state = state.get("behavior", {})
        self.behavior.load_state_dict(behavior_state)
        self._behavior_initialized = bool(behavior_state.get("records"))
