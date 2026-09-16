"""Persistent anonymous routing with a standalone ML reverse model."""

from __future__ import annotations

from copy import deepcopy

import torch
from avalanche.models.dynamic_modules import IncrementalClassifier
from torch import Tensor

from .global_fingerprint_refresh import refresh_all_fingerprints
from .persistent_skill_memory_plugin import (
    PersistentFingerprintSkillMemoryPlugin as _BaseFingerprintPlugin,
)
from .probing import apply_skill_state_exact
from .reverse_engineering import CandidateParameters, NormalMLReverseEngineer


class PersistentFingerprintSkillMemoryPlugin(_BaseFingerprintPlugin):
    """Persistent router using a normal ML reverse-engineering model.

    The reverse model is independent of the CL feature extractor. It is trained
    only from deterministic reference samples and frozen classifier weights
    belonging to frozen skill generations. Anonymous evaluation then uses only
    the raw sample and those frozen candidate parameters; no evaluation label,
    experience ID, or live CL representation enters the reverse model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reverse_engineer_model = NormalMLReverseEngineer()

    @staticmethod
    def _classifier_parameters(
        model,
        state_dict,
        class_id: int,
    ) -> CandidateParameters:
        """Extract one classifier row from a frozen skill state."""
        probe_model = deepcopy(model)
        apply_skill_state_exact(probe_model, state_dict)
        classifier = None
        for module in probe_model.modules():
            if isinstance(module, IncrementalClassifier):
                classifier = module.classifier
                break
        if classifier is None:
            raise ValueError("reverse engineering requires an IncrementalClassifier")
        class_id = int(class_id)
        if not 0 <= class_id < classifier.weight.shape[0]:
            raise ValueError("candidate class is outside the classifier output")
        bias = (
            float(classifier.bias[class_id].item())
            if classifier.bias is not None
            else 0.0
        )
        return CandidateParameters(
            weight=classifier.weight[class_id].detach().cpu().clone(),
            bias=bias,
        )

    def _fit_normal_reverse_model(self, strategy) -> None:
        """Train the independent ML model from frozen reference pairs."""
        records = []
        for slot in sorted(self.memory.slots()):
            records.extend(self.behavior.records_for_skill(slot))
        if len(records) < 2:
            self.reverse_engineer_model.fit([])
            return

        parameters = {}
        for record in records:
            frozen_state = self.behavior.skill_state(record.skill_id, record.version)
            if frozen_state is None:
                frozen_state = self.memory.state(record.skill_id)
            parameters[record.class_id] = self._classifier_parameters(
                strategy.model,
                frozen_state,
                record.class_id,
            )

        pairs = []
        for source in records:
            for candidate in records:
                pairs.append(
                    (
                        source.reference_inputs,
                        parameters[candidate.class_id],
                        float(candidate.class_id == source.class_id),
                    )
                )
        self.reverse_engineer_model.fit(pairs)

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Refresh frozen references and retrain the independent ML model."""
        super().after_training_exp(strategy, **kwargs)
        experience = strategy.experience
        if not self._is_last_subexp(experience):
            return
        if self._current_training_experience_index is None:
            return
        self.last_fingerprint_refresh = refresh_all_fingerprints(self, strategy)
        self._fit_normal_reverse_model(strategy)

    def _fingerprint_route(
        self,
        strategy,
        x: Tensor,
        slot_ids: list[int],
    ) -> tuple[Tensor, list[int]]:
        """Infer anonymous class identity with the standalone ML model."""
        records = [
            record
            for slot in slot_ids
            for record in self.behavior.records_for_skill(slot)
        ]
        if self.reverse_engineer_model.model is None:
            self._fit_normal_reverse_model(strategy)
        if self.reverse_engineer_model.model is None:
            self.last_fingerprint_routes = [
                {"sample_index": i, "status": "FAILED", "candidates": []}
                for i in range(x.shape[0])
            ]
            return torch.full(
                (x.shape[0],), -1, dtype=torch.long, device=x.device
            ), [-1] * x.shape[0]

        parameters = {}
        for record in records:
            frozen_state = self.behavior.skill_state(record.skill_id, record.version)
            if frozen_state is None:
                frozen_state = self.memory.state(record.skill_id)
            parameters[record.class_id] = self._classifier_parameters(
                strategy.model,
                frozen_state,
                record.class_id,
            )

        probabilities = {
            class_id: self.reverse_engineer_model.predict_proba(
                x, params.weight, params.bias
            )
            for class_id, params in parameters.items()
        }

        chosen_skills: list[int] = []
        chosen_classes: list[int] = []
        routes: list[dict] = []
        for sample_index in range(x.shape[0]):
            candidates = []
            for record in records:
                probability = float(
                    probabilities[record.class_id][sample_index].item()
                )
                candidates.append(
                    {
                        "class": record.class_id,
                        "skill": record.skill_id,
                        "predicted_y": probability >= 0.5,
                        "binary_compatible": probability >= 0.5,
                        "reverse_engineering_probability": probability,
                        "expected_y": record.expected_y,
                        "reference_accuracy": record.reference_accuracy,
                        "correct": probability >= 0.5,
                        "class_score": None,
                        "continuous_evidence": probability,
                    }
                )

            ranked = sorted(
                candidates,
                key=lambda item: item["reverse_engineering_probability"],
                reverse=True,
            )
            compatible = [item for item in ranked if item["predicted_y"]]
            selected = None
            if len(compatible) == 1:
                selected = compatible[0]
                status = "IDENTIFIED"
            elif not compatible:
                status = "FAILED"
            else:
                top = ranked[0]
                second = ranked[1] if len(ranked) > 1 else None
                status = "AMBIGUOUS"
                if second is not None:
                    gap = (
                        top["reverse_engineering_probability"]
                        - second["reverse_engineering_probability"]
                    )
                    remaining = [
                        abs(
                            item["reverse_engineering_probability"]
                            - second["reverse_engineering_probability"]
                        )
                        for item in ranked[1:]
                    ]
                    baseline = sum(remaining) / max(len(remaining), 1)
                    if top["predicted_y"] and gap > baseline:
                        selected = top
                        status = "IDENTIFIED"

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
                    "binary_compatible_candidates": len(compatible),
                    "candidates": candidates,
                }
            )

        self.last_fingerprint_routes = routes
        return (
            torch.tensor(chosen_skills, dtype=torch.long, device=x.device),
            chosen_classes,
        )
