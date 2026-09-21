"""Independent ML/CL evaluation of a trained Skill Memory model.

`SkillMemoryPlugin` is responsible only for training the main model: skill
allocation, REUSE/SCRATCH decisions, per-class training, and skill-state
storage. This module adds a *separate, auxiliary* way to measure what was
learned, decoupled from Skill Memory's own routing:

- `EvaluationMemoryPlugin` records a small frozen sample of each class's
  data as it's introduced, independent of experience boundaries.
- An independent evaluator model (any `nn.Module` with a fixed global
  output head) is trained on the accumulated per-class memory and
  evaluated class-by-class, with no experience-level class mask.

The evaluator's *lifetime* is the only degree of freedom this module
exposes: a fresh evaluator per experience ("ML") vs one evaluator retained
across experiences ("CL"). Both are always trained on the complete
accumulated class memory seen so far, not just the latest experience -
callers choose which by either calling `build_evaluator` once (CL) or
before every experience (ML); see `skill_memory/demos/demo_splitmnist_ml_er.py`
for a complete example of both.

`evaluate_skill_memory` is a third, independent measurement: Skill
Memory's own stored skills, evaluated directly (via the true class-to-skill
mapping, or via anonymous routing), with no auxiliary evaluator involved.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..cl.skill_memory_plugin import SkillMemoryPlugin
from ..utils.probing import apply_skill_state_exact
from .routing import find_best_routing_skill


@dataclass
class EvaluationMemory:
    """A frozen, bounded sample of one class's own data."""

    inputs: torch.Tensor
    targets: torch.Tensor
    class_id: int

    @property
    def size(self) -> int:
        """Number of retained samples."""
        return int(self.targets.numel())


class EvaluationMemoryPlugin(SkillMemoryPlugin):
    """Skill Memory training plus frozen class-level evaluation memory.

    This class does not implement evaluation itself - `SkillMemoryPlugin`
    remains responsible for skill allocation, REUSE/SCRATCH decisions,
    training, and skill-state storage. This subclass only additionally
    records a bounded, deterministic sample of every class's own data as
    each training experience introduces it, for later use by
    `train_evaluator`/`evaluate_model_by_class`.
    """

    def __init__(
        self,
        *args,
        eval_memory_per_class: int = 20,
        eval_memory_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if eval_memory_per_class <= 0:
            raise ValueError("eval_memory_per_class must be positive")
        self.eval_memory_per_class = eval_memory_per_class
        self.eval_memory_seed = eval_memory_seed
        # Entries are class-level memories. The original experience is not
        # part of the evaluation unit.
        self.eval_memory: list[EvaluationMemory] = []

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Run Skill Memory's own hook, then snapshot evaluation data."""
        super().after_training_exp(strategy, **kwargs)
        experience = strategy.experience
        experience_index = int(getattr(experience, "current_experience", 0))
        memories = self._build_evaluation_memory(experience)
        self.eval_memory.extend(memories)
        if self.verbose:
            print(
                f"Evaluation memory {experience_index}: "
                f"{sum(memory.size for memory in memories)} samples, "
                f"classes={[memory.class_id for memory in memories]}"
            )

    def _build_evaluation_memory(self, experience) -> list[EvaluationMemory]:
        """Build a deterministic, bounded sample independently for each class."""
        dataset = experience.dataset
        samples_by_class: dict[int, list[int]] = {}
        for index in range(len(dataset)):
            target = int(dataset[index][1])
            samples_by_class.setdefault(target, []).append(index)

        generator = torch.Generator()
        generator.manual_seed(
            self.eval_memory_seed + int(getattr(experience, "current_experience", 0))
        )

        memories: list[EvaluationMemory] = []
        for class_id in sorted(samples_by_class):
            indices = samples_by_class[class_id]
            if len(indices) > self.eval_memory_per_class:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [
                    indices[position]
                    for position in permutation[: self.eval_memory_per_class]
                ]

            inputs: list[torch.Tensor] = []
            targets: list[int] = []
            for index in indices:
                sample = dataset[index]
                input_tensor = sample[0]
                if not isinstance(input_tensor, torch.Tensor):
                    input_tensor = torch.as_tensor(input_tensor)
                inputs.append(input_tensor.detach().cpu())
                targets.append(int(sample[1]))

            if not inputs:
                raise RuntimeError(
                    f"Class {class_id} produced an empty evaluation memory."
                )
            memories.append(
                EvaluationMemory(
                    inputs=torch.stack(inputs),
                    targets=torch.tensor(targets, dtype=torch.long),
                    class_id=class_id,
                )
            )
        return memories


def make_loader(
    memory: list[EvaluationMemory],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    """Build a `DataLoader` exclusively from frozen class-level memory."""
    if not memory:
        raise RuntimeError("Cannot build an evaluation loader from empty memory.")
    inputs = torch.cat([item.inputs for item in memory], dim=0)
    targets = torch.cat([item.targets for item in memory], dim=0)
    dataset = TensorDataset(inputs, targets)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def consolidate_evaluation_memory(
    memory: list[EvaluationMemory],
) -> list[EvaluationMemory]:
    """Combine evaluation data into one canonical entry per class.

    Experience boundaries are intentionally discarded here. The resulting
    memory represents accumulated class knowledge only.
    """
    by_class: dict[int, list[EvaluationMemory]] = {}
    for item in memory:
        by_class.setdefault(int(item.class_id), []).append(item)

    consolidated: list[EvaluationMemory] = []
    for class_id in sorted(by_class):
        inputs = torch.cat([item.inputs for item in by_class[class_id]], dim=0)
        targets = torch.cat([item.targets for item in by_class[class_id]], dim=0)
        consolidated.append(
            EvaluationMemory(inputs=inputs, targets=targets, class_id=class_id)
        )
    return consolidated


def train_evaluator(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    memory: list[EvaluationMemory],
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> None:
    """Train an evaluator exclusively from accumulated class memory."""
    if not memory:
        raise RuntimeError("Cannot train evaluator with empty memory.")
    if epochs < 1:
        return
    loader = make_loader(memory, batch_size=batch_size, shuffle=True, seed=seed)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_model_by_class(
    model: nn.Module,
    test_stream,
    up_to_index: int,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Evaluate every class seen up to `up_to_index`, independently.

    The model always uses its complete, fixed output space - no
    experience-level prediction mask or task label is supplied.
    """
    model.eval()
    results: dict[int, dict[str, float]] = {}
    for experience_index in range(up_to_index + 1):
        experience = test_stream[experience_index]
        loader = DataLoader(experience.dataset, batch_size=batch_size, shuffle=False)

        class_loss: dict[int, float] = {}
        class_correct: dict[int, int] = {}
        class_total: dict[int, int] = {}
        for batch in loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            logits = model(inputs)
            per_sample_loss = nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            predictions = logits.argmax(dim=1)
            for class_id in torch.unique(targets).tolist():
                class_id = int(class_id)
                mask = targets == class_id
                class_loss[class_id] = class_loss.get(class_id, 0.0) + float(
                    per_sample_loss[mask].sum().item()
                )
                class_correct[class_id] = class_correct.get(class_id, 0) + int(
                    (predictions[mask] == targets[mask]).sum().item()
                )
                class_total[class_id] = class_total.get(class_id, 0) + int(
                    mask.sum().item()
                )

        for class_id in class_loss:
            total = class_total[class_id]
            if total == 0:
                raise RuntimeError(f"Class {class_id} has no test samples.")
            results[class_id] = {
                "loss": class_loss[class_id] / total,
                "accuracy": class_correct[class_id] / total,
                "experience": float(experience_index),
            }
    return results


def aggregate_experience_metrics(
    class_results: dict[int, dict[str, float]],
    test_stream,
    up_to_index: int,
) -> tuple[list[float], list[float]]:
    """Aggregate class metrics for reporting by introducing experience.

    The aggregation is a simple mean over classes, not over experiences,
    which keeps the metric class-balanced when experiences contain
    different numbers of classes.
    """
    losses: list[float] = []
    accuracies: list[float] = []
    for experience_index in range(up_to_index + 1):
        classes = sorted(
            int(class_id)
            for class_id in test_stream[experience_index].classes_in_this_experience
        )
        missing = [class_id for class_id in classes if class_id not in class_results]
        if missing:
            raise RuntimeError(
                f"Missing evaluation results for classes {missing} in "
                f"experience {experience_index}."
            )
        losses.append(
            float(np.mean([class_results[class_id]["loss"] for class_id in classes]))
        )
        accuracies.append(
            float(
                np.mean([class_results[class_id]["accuracy"] for class_id in classes])
            )
        )
    return losses, accuracies


def compute_class_forgetting(
    accuracy_history: list[dict[int, float]],
    class_to_experience: dict[int, int],
    num_experiences: int,
) -> np.ndarray:
    """Compute forgetting independently for every class.

    Each class's reference point is its own accuracy immediately after the
    experience that introduced it; the result is then averaged over
    classes grouped by their introducing experience.
    """
    forgetting_by_experience: list[list[float]] = [[] for _ in range(num_experiences)]
    all_classes = sorted(
        {class_id for history in accuracy_history for class_id in history}
    )
    for class_id in all_classes:
        introduction = class_to_experience[class_id]
        if introduction >= len(accuracy_history):
            continue
        acquisition_accuracy = accuracy_history[introduction][class_id]
        final_accuracy = accuracy_history[-1].get(class_id)
        if final_accuracy is None:
            continue
        forgetting = max(0.0, acquisition_accuracy - final_accuracy)
        forgetting_by_experience[introduction].append(forgetting)

    result = np.zeros(num_experiences, dtype=np.float64)
    for experience_index, values in enumerate(forgetting_by_experience):
        if values:
            result[experience_index] = float(np.mean(values))
    return result


def evaluate_skill_memory(
    model: nn.Module,
    plugin: SkillMemoryPlugin,
    test_stream,
    up_to_index: int,
    *,
    num_classes: int,
    routing: str,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    """Evaluate the actual stored Skill Memory states, per experience.

    Deliberately independent from any auxiliary ML/CL evaluator: this
    reuses `model`'s architecture only as scratch space to load each
    stored skill's own frozen weights into, restoring `model`'s original
    weights afterward. `routing="oracle"` uses the true class-to-skill
    mapping; `routing="probe"` uses anonymous routing
    (`find_best_routing_skill`).
    """
    if not plugin.memory:
        raise RuntimeError("Cannot evaluate Skill Memory before any skill exists.")
    slot_ids = sorted(plugin.memory.slots())
    skill_states = [plugin.memory.state(slot) for slot in slot_ids]
    accuracies: list[float] = []
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    try:
        model.eval()
        for experience_index in range(up_to_index + 1):
            experience = test_stream[experience_index]
            loader = DataLoader(
                experience.dataset, batch_size=batch_size, shuffle=False
            )
            correct = 0
            total = 0
            for batch in loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                if routing == "oracle":
                    chosen_logits = torch.empty(
                        inputs.shape[0], num_classes, device=device
                    )
                    rows_by_skill: dict[int, list[int]] = {}
                    for row, label in enumerate(labels.tolist()):
                        skill = plugin.class_map.find_skill_for_class_anywhere(
                            int(label)
                        )
                        if skill is None:
                            raise RuntimeError(
                                f"No canonical skill recorded for class {label}."
                            )
                        rows_by_skill.setdefault(int(skill), []).append(row)
                    for skill, rows in rows_by_skill.items():
                        apply_skill_state_exact(model, plugin.memory.state(skill))
                        row_tensor = torch.tensor(rows, device=device)
                        chosen_logits[row_tensor] = model(inputs[row_tensor])
                elif routing == "probe":
                    raw_logits = []
                    for state in skill_states:
                        apply_skill_state_exact(model, state)
                        raw_logits.append(model(inputs))
                    routing_result = find_best_routing_skill(
                        raw_logits,
                        skill_states,
                        [plugin.class_map.classes_for_skill(slot) for slot in slot_ids],
                    )
                    chosen = routing_result.skill_indices
                    stacked = torch.stack(raw_logits, dim=0)
                    rows = torch.arange(inputs.shape[0], device=device)
                    chosen_logits = stacked[chosen, rows]
                else:
                    raise ValueError(f"Unknown Skill Memory routing mode: {routing}")

                correct += int((chosen_logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())

            if total == 0:
                raise RuntimeError(f"Test experience {experience_index} is empty.")
            accuracies.append(correct / total)
    finally:
        apply_skill_state_exact(model, original_state)

    return accuracies


def build_evaluator(
    model_factory: Callable[[], nn.Module],
    *,
    device: torch.device,
    learning_rate: float,
) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """Create a fresh, completely independent evaluation learner.

    `model_factory` takes no arguments and returns a new, untrained model
    (e.g. `lambda: SimpleMLP(num_classes=10)`).
    """
    model = model_factory().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion
