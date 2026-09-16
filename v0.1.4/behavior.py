"""Persistent binary class-behavior fingerprints for anonymous routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class ClassBehaviorRecord:
    """Persistent binary behavior for one canonical class."""

    class_id: int
    skill_id: int
    version: int
    reference_inputs: Tensor
    reference_y: Tensor
    expected_y: bool = True
    valid: bool = True

    @property
    def reference_accuracy(self) -> float:
        result = compare_binary_behavior(self.reference_y, self.expected_y)
        return float(result["accuracy"])

    def state_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "skill_id": self.skill_id,
            "version": self.version,
            "reference_inputs": self.reference_inputs.detach().cpu(),
            "reference_y": self.reference_y.detach().cpu().bool(),
            "expected_y": self.expected_y,
            "valid": self.valid,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ClassBehaviorRecord":
        return cls(
            class_id=int(state["class_id"]),
            skill_id=int(state["skill_id"]),
            version=int(state["version"]),
            reference_inputs=state["reference_inputs"].detach().cpu(),
            reference_y=state["reference_y"].detach().cpu().bool(),
            expected_y=bool(state.get("expected_y", True)),
            valid=bool(state.get("valid", True)),
        )


class BehaviorFingerprintCache:
    """Version-aware persistent cache of binary class behavior references."""

    def __init__(self) -> None:
        self._records: dict[int, ClassBehaviorRecord] = {}
        self._skill_versions: dict[int, int] = {}

    def skill_version(self, skill_id: int) -> int:
        return self._skill_versions.get(int(skill_id), 0)

    def bump_skill(self, skill_id: int) -> int:
        skill_id = int(skill_id)
        version = self.skill_version(skill_id) + 1
        self._skill_versions[skill_id] = version
        self.invalidate_skill(skill_id)
        return version

    def invalidate_skill(self, skill_id: int) -> None:
        skill_id = int(skill_id)
        for record in self._records.values():
            if record.skill_id == skill_id:
                record.valid = False

    def put(self, record: ClassBehaviorRecord) -> None:
        self._records[record.class_id] = record
        self._skill_versions[record.skill_id] = max(
            self.skill_version(record.skill_id), record.version
        )

    def get(self, class_id: int, skill_id: int) -> ClassBehaviorRecord | None:
        record = self._records.get(int(class_id))
        if record is None or not record.valid:
            return None
        if record.skill_id != int(skill_id):
            return None
        if record.version != self.skill_version(skill_id):
            return None
        return record

    def records_for_skill(self, skill_id: int) -> list[ClassBehaviorRecord]:
        return [
            record
            for record in self._records.values()
            if self.get(record.class_id, int(skill_id)) is not None
        ]

    def all_records_for_skill(self, skill_id: int) -> list[ClassBehaviorRecord]:
        skill_id = int(skill_id)
        return [
            record for record in self._records.values() if record.skill_id == skill_id
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "skill_versions": dict(self._skill_versions),
            "records": [record.state_dict() for record in self._records.values()],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._skill_versions = {
            int(key): int(value)
            for key, value in state.get("skill_versions", {}).items()
        }
        self._records = {}
        for record_state in state.get("records", []):
            record = ClassBehaviorRecord.from_state_dict(record_state)
            self._records[record.class_id] = record


def reverse_engineer_y(logits: Tensor, target_class: int) -> Tensor:
    """Produce binary ``y`` for a candidate class from classifier output."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if not 0 <= int(target_class) < logits.shape[-1]:
        raise ValueError("target_class is outside the classifier output")
    return logits.argmax(dim=-1).eq(int(target_class))


def compare_binary_behavior(
    predicted_y: Tensor,
    expected_y: bool,
) -> dict[str, Any]:
    """Compare predicted binary ``y`` values with an expected value."""
    predicted_y = predicted_y.detach().cpu().bool()
    expected = torch.full_like(predicted_y, expected_y, dtype=torch.bool)
    correct = predicted_y.eq(expected)
    return {
        "predicted_y": predicted_y,
        "expected_y": bool(expected_y),
        "correct": correct,
        "accuracy": float(correct.float().mean().item())
        if correct.numel()
        else 0.0,
        "all_correct": bool(correct.all().item()) if correct.numel() else False,
    }


def identify_binary_behavior(
    predicted_y: Tensor,
    expected_y: bool = True,
) -> bool:
    """Return whether every anonymous probe agrees with expected ``y``."""
    return bool(compare_binary_behavior(predicted_y, expected_y)["all_correct"])
