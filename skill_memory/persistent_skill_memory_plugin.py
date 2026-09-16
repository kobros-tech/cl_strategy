"""Skill Memory extension using binary and continuous weight behavior."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import torch
from torch import Tensor

from .behavior import (
    BehaviorFingerprintCache,
    ClassBehaviorRecord,
    build_weight_behavior_statistics,
    compare_binary_behavior,
    extract_features_from_weights,
    reverse_engineer_y_from_weights,
    reverse_engineer_scores_from_weights,
)
from .probing import apply_skill_state_exact, predict_logits, probe_class
from .skill_memory_plugin import SkillMemoryPlugin


class PersistentFingerprintSkillMemoryPlugin(SkillMemoryPlugin):
    """Skill Memory with persistent binary and continuous class fingerprints."""

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
    ) -> ClassBehaviorRecord:
        model = deepcopy(strategy.model)
        state = self.memory.state(skill_id)
        apply_skill_state_exact(model, state)
        reference_y = reverse_engineer_y_from_weights(
            model, x, class_id
        ).cpu()
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
        """Refresh every class fingerprint owned by the current skill."""
        version = self.behavior.skill_version(skill_id)
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
            self.behavior.put(
                self._build_record(
                    strategy,
                    int(skill_id),
                    int(class_id),
                    x,
                    version,
                )
            )

        self._pending_reference_inputs.clear()
        self._behavior_initialized = bool(self.behavior.state_dict()["records"])

    @staticmethod
    def _feature_similarity(features: Tensor, record: ClassBehaviorRecord) -> Tensor:
        """Measure how closely a sample matches a persistent feature fingerprint."""
        if record.reference_feature_mean is None:
            return torch.ones(features.shape[0], device=features.device)
        mean = record.reference_feature_mean.to(features.device)
        std = record.reference_feature_std.to(features.device).clamp_min(1e-6)
        z = (features - mean) / std
        distance = z.square().mean(dim=-1)
        return torch.exp(-0.5 * distance)

    @staticmethod
    def _margin_similarity(margin: Tensor, record: ClassBehaviorRecord) -> Tensor:
        """Measure similarity to the learned class-vs-rest margin distribution."""
        mean = torch.tensor(
            record.reference_margin_mean, device=margin.device, dtype=margin.dtype
        )
        std = torch.tensor(
            max(record.reference_margin_std, 1e-6),
            device=margin.device,
            dtype=margin.dtype,
        )
        z = (margin - mean) / std
        return torch.exp(-0.5 * z.square())

    def _candidate_evidence(
        self,
        scores: Tensor,
        features: Tensor,
        record: ClassBehaviorRecord,
        sample_index: int,
    ) -> dict:
        class_id = record.class_id
        predicted_class = int(scores[sample_index].argmax().item())
        predicted_y = predicted_class == class_id
        comparison = compare_binary_behavior(
            torch.tensor([predicted_y]), record.expected_y
        )
        own = scores[sample_index, class_id]
        if scores.shape[-1] > 1:
            others = scores[sample_index].clone()
            others[class_id] = -torch.inf
            margin = own - others.max()
        else:
            margin = own
        feature_score = self._feature_similarity(
            features[sample_index : sample_index + 1], record
        )[0]
        margin_score = self._margin_similarity(
            margin.reshape(1), record
        )[0]
        evidence = 0.65 * feature_score + 0.35 * margin_score
        return {
            "class": class_id,
            "skill": record.skill_id,
            "predicted_y": predicted_y,
            "predicted_class": predicted_class,
            "expected_y": record.expected_y,
            "reference_y": record.reference_y.tolist(),
            "reference_accuracy": record.reference_accuracy,
            "binary_compatible": bool(comparison["all_correct"]),
            "class_score": float(own.item()),
            "margin": float(margin.item()),
            "feature_similarity": float(feature_score.item()),
            "margin_similarity": float(margin_score.item()),
            "evidence": float(evidence.item()),
        }

    @staticmethod
    def _select_fingerprint_candidate(matches: list[dict]) -> tuple[str, dict | None]:
        """Select only among behaviorally compatible candidates.

        Binary behavior is the identity gate inherited from the original
        persistent fingerprint router. Continuous fingerprints are supporting
        evidence: they can distinguish multiple compatible candidates, but
        cannot promote an incompatible candidate into a route.
        """
        compatible = [item for item in matches if item["binary_compatible"]]
        if not compatible:
            return "FAILED", None
        if len(compatible) == 1:
            return "IDENTIFIED", compatible[0]

        compatible.sort(key=lambda item: item["evidence"], reverse=True)
        top = compatible[0]
        second = compatible[1]
        evidence_values = torch.tensor(
            [item["evidence"] for item in compatible]
        )
        dispersion = float(evidence_values.std(unbiased=False).item())
        gap = top["evidence"] - second["evidence"]
        if gap > dispersion:
            return "IDENTIFIED", top
        return "AMBIGUOUS", None

    def _fingerprint_route(
        self,
        strategy,
        x: Tensor,
        slot_ids: list[int],
    ) -> tuple[Tensor, list[int]]:
        """Infer class from persistent behavior, then resolve its skill."""
        probe_model = deepcopy(strategy.model)
        candidate_records = [
            record
            for slot in slot_ids
            for record in self.behavior.records_for_skill(slot)
        ]
        skill_data: dict[int, tuple[Tensor, Tensor]] = {}
        for slot in slot_ids:
            state = self.memory.state(slot)
            apply_skill_state_exact(probe_model, state)
            skill_data[slot] = (
                reverse_engineer_scores_from_weights(probe_model, x),
                extract_features_from_weights(probe_model, x),
            )

        chosen_skills: list[int] = []
        chosen_classes: list[int] = []
        routes: list[dict] = []

        for sample_index in range(x.shape[0]):
            matches: list[dict] = []
            for record in candidate_records:
                scores, features = skill_data[record.skill_id]
                if not 0 <= record.class_id < scores.shape[-1]:
                    continue
                matches.append(
                    self._candidate_evidence(
                        scores, features, record, sample_index
                    )
                )

            status, selected = self._select_fingerprint_candidate(matches)
            if selected is None:
                chosen_classes.append(-1)
                chosen_skills.append(-1)
            else:
                chosen_classes.append(int(selected["class"]))
                chosen_skills.append(int(selected["skill"]))

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
        """Start a fresh, complete routing-analysis trace for this eval pass."""
        super().before_eval(strategy, **kwargs)
        self.fingerprint_route_history = []
        self._fingerprint_batch_index = 0

    def after_eval_forward(self, strategy, **kwargs) -> None:
        if not self._eval_active or self.eval_routing != "probe":
            return super().after_eval_forward(strategy, **kwargs)
        if len(self.memory) == 0 or not self._behavior_initialized:
            return super().after_eval_forward(strategy, **kwargs)

        x = strategy.mbatch[0]
        y = strategy.mbatch[1]
        slot_ids = sorted(self.memory.slots())
        chosen, class_matches = self._fingerprint_route(strategy, x, slot_ids)

        # Labels are copied into the analysis trace only after routing has
        # finished. They are never consumed by the routing decision.
        for route, label in zip(
            self.last_fingerprint_routes,
            y.detach().cpu().tolist(),
            strict=False,
        ):
            route["batch_index"] = self._fingerprint_batch_index
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
            f"samples={x.shape[0]} identified={identified} "
            f"ambiguous={ambiguous} failed={failed} "
            f"matched_classes={class_matches[:5]}"
        )

    def state_dict(self) -> dict:
        """Serialize only the persistent behavior state."""
        return {"behavior": self.behavior.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore behavior state; old checkpoints remain loadable."""
        behavior_state = state.get("behavior", {})
        self.behavior.load_state_dict(behavior_state)
        self._behavior_initialized = bool(behavior_state.get("records"))
