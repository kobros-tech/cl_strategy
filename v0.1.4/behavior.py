"""Persistent, class-level behavioral fingerprints for probe routing.

The cache stores reference behavior produced by a canonical skill.  Evaluation
only computes the small probe-side fingerprint and compares it with these
persistent references; reference behavior is not recomputed on every eval.

Fingerprints are represented by class-aligned logits rather than a fixed
output-column tensor.  This keeps them compatible with Avalanche's growing
global classifier head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class ClassBehaviorRecord:
    """Persistent reference behavior for one canonical class."""

    class_id: int
    skill_id: int
    version: int
    reference_inputs: Tensor
    output_class_ids: tuple[int, ...]
    reference_output: Tensor
    reference_summary: Tensor
    valid: bool = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "skill_id": self.skill_id,
            "version": self.version,
            "reference_inputs": self.reference_inputs.detach().cpu(),
            "output_class_ids": self.output_class_ids,
            "reference_output": self.reference_output.detach().cpu(),
            "reference_summary": self.reference_summary.detach().cpu(),
            "valid": self.valid,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ClassBehaviorRecord":
        return cls(
            class_id=int(state["class_id"]),
            skill_id=int(state["skill_id"]),
            version=int(state["version"]),
            reference_inputs=state["reference_inputs"].detach().cpu(),
            output_class_ids=tuple(int(x) for x in state["output_class_ids"]),
            reference_output=state["reference_output"].detach().cpu(),
            reference_summary=state["reference_summary"].detach().cpu(),
            valid=bool(state.get("valid", True)),
        )


class BehaviorFingerprintCache:
    """Version-aware persistent class behavior cache."""

    def __init__(self) -> None:
        self._records: dict[int, ClassBehaviorRecord] = {}
        self._skill_versions: dict[int, int] = {}

    def skill_version(self, skill_id: int) -> int:
        return self._skill_versions.get(int(skill_id), 0)

    def bump_skill(self, skill_id: int) -> int:
        skill_id = int(skill_id)
        version = self._skill_versions.get(skill_id, 0) + 1
        self._skill_versions[skill_id] = version
        for record in self._records.values():
            if record.skill_id == skill_id:
                record.valid = False
        return version

    def invalidate_skill(self, skill_id: int) -> None:
        for record in self._records.values():
            if record.skill_id == int(skill_id):
                record.valid = False

    def put(self, record: ClassBehaviorRecord) -> None:
        self._records[record.class_id] = record
        self._skill_versions[record.skill_id] = max(
            self._skill_versions.get(record.skill_id, 0), record.version
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
            if record.skill_id == int(skill_id) and self.get(record.class_id, skill_id)
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "skill_versions": dict(self._skill_versions),
            "records": [record.state_dict() for record in self._records.values()],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._skill_versions = {
            int(key): int(value) for key, value in state.get("skill_versions", {}).items()
        }
        self._records = {}
        for record_state in state.get("records", []):
            record = ClassBehaviorRecord.from_state_dict(record_state)
            self._records[record.class_id] = record


def _normalize(values: Tensor) -> Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def summarize_behavior(logits: Tensor, owned_classes: list[int]) -> Tensor:
    """Return compact scale-independent behavior statistics."""
    if not owned_classes:
        return torch.zeros(4, dtype=logits.dtype, device=logits.device)
    valid = [c for c in owned_classes if 0 <= c < logits.shape[-1]]
    if not valid:
        return torch.zeros(4, dtype=logits.dtype, device=logits.device)
    owned = logits[:, valid]
    top2 = torch.topk(owned, k=min(2, owned.shape[-1]), dim=-1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[-1] > 1 else 0.0)
    return torch.stack(
        [owned.mean(dim=-1), owned.std(dim=-1, unbiased=False), top2[:, 0], margin],
        dim=-1,
    ).mean(dim=0)


def extract_reference_behavior(
    logits: Tensor,
    class_ids: list[int],
    target_class: int,
) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    """Extract class-aligned output and summary fingerprint from references."""
    valid = tuple(sorted({c for c in class_ids if 0 <= c < logits.shape[-1]}))
    if not valid:
        raise ValueError("no valid classifier classes available for fingerprint")
    output = _normalize(logits[:, list(valid)]).mean(dim=0).detach().cpu()
    summary = summarize_behavior(logits, list(valid)).detach().cpu()
    del target_class
    return output, summary, valid


def probe_behavior_fingerprint(
    logits: Tensor,
    output_class_ids: tuple[int, ...],
    reference_output: Tensor,
    reference_summary: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compare one probe batch against one persistent class fingerprint."""
    valid = [c for c in output_class_ids if 0 <= c < logits.shape[-1]]
    if not valid:
        return torch.full((logits.shape[0],), -1.0, device=logits.device), torch.zeros(
            4, device=logits.device
        )
    current = _normalize(logits[:, valid])
    reference = reference_output.to(device=logits.device, dtype=logits.dtype)
    similarity = torch.nn.functional.cosine_similarity(
        current, reference.unsqueeze(0), dim=-1
    )
    summary = summarize_behavior(logits, valid)
    return similarity, summary
