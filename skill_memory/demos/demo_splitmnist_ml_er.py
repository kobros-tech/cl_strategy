"""SplitMNIST Skill Memory training with independent ML/CL evaluation.

Skill Memory is responsible only for training the main model.

After every Skill Memory training experience, this demo stores a small frozen
evaluation memory for each class introduced by that experience.

A completely separate evaluator is then used to measure evaluation-time
ML/CL behavior.

The evaluator is deliberately class-dependent rather than experience-
dependent:

    ML:
        a fresh evaluator is created after every experience, but it is trained
        on the complete accumulated class memory seen so far.

    CL:
        one evaluator is retained across experiences and is trained on the
        complete accumulated class memory seen so far.

The important distinction between ML and CL is therefore the lifetime of the
evaluator, not whether it receives only the latest experience.

Evaluation never receives an experience-level class oracle. The evaluator
always uses the fixed global 10-class output head, and each class is evaluated
independently.

Experience boundaries are used only for reporting which classes were
introduced at each step.

The demo also optionally reports direct Skill Memory evaluation:

    oracle:
        the true class -> skill mapping is supplied.

    probe:
        the anonymous routing mechanism must select the skill.

These direct Skill Memory measurements are kept separate from the auxiliary
ML/CL evaluator measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training import Naive

from skill_memory.cl import SkillMemoryPlugin
from skill_memory.cl.skill_registry import SkillMemory
from skill_memory.evaluation.routing import find_best_routing_skill
from skill_memory.utils.probing import apply_skill_state_exact


class SplitMNISTMLP(nn.Module):
    """MLP with a fixed global 10-class head for Skill Memory training."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(x.size(0), -1)
        return self.classifier(self.features(x))


class EvaluationMLP(nn.Module):
    """Independent evaluator with a fixed global 10-class head."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_size: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(x.size(0), -1)
        return self.classifier(self.features(x))


@dataclass
class EvaluationMemory:
    """Frozen examples belonging to one class."""

    inputs: torch.Tensor
    targets: torch.Tensor
    class_id: int

    @property
    def size(self) -> int:
        return int(self.targets.numel())


class EvaluationMemoryPlugin(SkillMemoryPlugin):
    """Skill Memory training plus frozen class-level evaluation memory.

    This class does not implement evaluation.

    SkillMemoryPlugin remains responsible for:
        - skill allocation
        - REUSE/SCRATCH decisions
        - skill training
        - skill-state storage

    This subclass only records a bounded copy of examples for every class
    appearing in each training experience.
    """

    def __init__(
        self,
        *args,
        eval_memory_per_class: int = 20,
        eval_memory_seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if eval_memory_per_class <= 0:
            raise ValueError("eval_memory_per_class must be positive")

        self.eval_memory_per_class = eval_memory_per_class
        self.eval_memory_seed = eval_memory_seed

        # Entries are class-level memories. The original experience is not
        # part of the evaluation unit.
        self.eval_memory: list[EvaluationMemory] = []

    def after_training_exp(self, strategy, **kwargs):
        """Run Skill Memory's hook, then snapshot evaluation data."""
        super().after_training_exp(strategy, **kwargs)

        experience = strategy.experience

        experience_index = int(
            getattr(
                experience,
                "current_experience",
                0,
            )
        )

        memories = self._build_evaluation_memory(experience)

        self.eval_memory.extend(memories)

        print(
            f"Evaluation memory {experience_index}: "
            f"{sum(memory.size for memory in memories)} samples, "
            f"classes="
            f"{[memory.class_id for memory in memories]}"
        )

    def _build_evaluation_memory(
        self,
        experience,
    ) -> list[EvaluationMemory]:
        """Build deterministic bounded memory independently for each class."""
        dataset = experience.dataset

        samples_by_class: dict[int, list[int]] = {}

        for index in range(len(dataset)):
            sample = dataset[index]
            target = int(sample[1])

            samples_by_class.setdefault(target, []).append(index)

        generator = torch.Generator()
        generator.manual_seed(
            self.eval_memory_seed + int(getattr(experience, "current_experience", 0))
        )

        memories: list[EvaluationMemory] = []

        for class_id in sorted(samples_by_class):
            indices = samples_by_class[class_id]

            if len(indices) > self.eval_memory_per_class:
                permutation = torch.randperm(
                    len(indices),
                    generator=generator,
                ).tolist()

                indices = [
                    indices[position]
                    for position in permutation[: self.eval_memory_per_class]
                ]

            inputs: list[torch.Tensor] = []
            targets: list[int] = []

            for index in indices:
                sample = dataset[index]

                input_tensor = sample[0]
                target = int(sample[1])

                if not isinstance(input_tensor, torch.Tensor):
                    input_tensor = torch.as_tensor(input_tensor)

                inputs.append(input_tensor.detach().cpu())
                targets.append(target)

            if not inputs:
                raise RuntimeError(
                    f"Class {class_id} produced an empty evaluation memory."
                )

            memories.append(
                EvaluationMemory(
                    inputs=torch.stack(inputs),
                    targets=torch.tensor(
                        targets,
                        dtype=torch.long,
                    ),
                    class_id=class_id,
                )
            )

        return memories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Skill Memory on SplitMNIST and evaluate its learned "
            "knowledge using independent ML/CL evaluation learners."
        )
    )

    parser.add_argument(
        "--dataset-root",
        default="data",
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
    )

    parser.add_argument(
        "--n-experiences",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--eval-method",
        choices=("ml", "cl"),
        default="cl",
        help=(
            "Evaluation learner. "
            "'ml' creates a fresh evaluator for every experience, "
            "but each evaluator is trained on all accumulated class "
            "memories. "
            "'cl' retains one evaluator across experiences and also "
            "trains on all accumulated class memories."
        ),
    )

    parser.add_argument(
        "--eval-memory-per-class",
        type=int,
        default=20,
        help="Frozen evaluation examples retained per class.",
    )

    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=1,
        help="Number of epochs used by the independent evaluator.",
    )

    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--eval-learning-rate",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-skills",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--skill-eval-routing",
        choices=("oracle", "probe", "both", "none"),
        default="both",
        help=(
            "Direct Skill Memory diagnostic. "
            "'oracle' uses the true class-to-skill mapping; "
            "'probe' uses anonymous routing; "
            "'both' runs both; "
            "'none' disables direct Skill Memory evaluation."
        ),
    )

    return parser.parse_args()


def make_loader(
    memory: list[EvaluationMemory],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """Build a loader exclusively from frozen class-level memory."""
    if not memory:
        raise RuntimeError("Cannot build an evaluation loader from empty memory.")

    inputs = torch.cat(
        [item.inputs for item in memory],
        dim=0,
    )

    targets = torch.cat(
        [item.targets for item in memory],
        dim=0,
    )

    dataset = torch.utils.data.TensorDataset(
        inputs,
        targets,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
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
        by_class.setdefault(
            int(item.class_id),
            [],
        ).append(item)

    consolidated: list[EvaluationMemory] = []

    for class_id in sorted(by_class):
        inputs = torch.cat(
            [item.inputs for item in by_class[class_id]],
            dim=0,
        )

        targets = torch.cat(
            [item.targets for item in by_class[class_id]],
            dim=0,
        )

        consolidated.append(
            EvaluationMemory(
                inputs=inputs,
                targets=targets,
                class_id=class_id,
            )
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

    loader = make_loader(
        memory,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    model.train()

    for _ in range(epochs):
        for batch in loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, targets)

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
    """Evaluate every seen class independently.

    The model always uses the complete global 10-class output space.

    No experience-level prediction mask or task label is supplied.
    """
    model.eval()

    # criterion = nn.CrossEntropyLoss(reduction="sum")

    results: dict[int, dict[str, float]] = {}

    for experience_index in range(up_to_index + 1):
        experience = test_stream[experience_index]

        loader = torch.utils.data.DataLoader(
            experience.dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        class_loss: dict[int, float] = {}
        class_correct: dict[int, int] = {}
        class_total: dict[int, int] = {}

        for batch in loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            logits = model(inputs)

            per_sample_loss = nn.functional.cross_entropy(
                logits,
                targets,
                reduction="none",
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
                "accuracy": (class_correct[class_id] / total),
                "experience": float(experience_index),
            }

    return results


def aggregate_experience_metrics(
    class_results: dict[int, dict[str, float]],
    test_stream,
    up_to_index: int,
) -> tuple[list[float], list[float]]:
    """Aggregate class metrics for reporting by introducing experience.

    The aggregation is a simple mean over classes, not over experiences.
    This keeps the metric class-balanced when experiences contain different
    numbers of classes.
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
                "Missing evaluation results for classes "
                f"{missing} in experience {experience_index}."
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

    For each class, the reference point is its accuracy immediately after
    the experience in which that class was introduced.

    The result is then averaged over classes introduced by each experience.
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

        forgetting = max(
            0.0,
            acquisition_accuracy - final_accuracy,
        )

        forgetting_by_experience[introduction].append(forgetting)

    result = np.zeros(
        num_experiences,
        dtype=np.float64,
    )

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
    routing: str,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    """Evaluate the actual stored Skill Memory states.

    This is deliberately independent from the auxiliary ML/CL evaluator.
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

            loader = torch.utils.data.DataLoader(
                experience.dataset,
                batch_size=batch_size,
                shuffle=False,
            )

            correct = 0
            total = 0

            for batch in loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                if routing == "oracle":
                    chosen_logits = torch.empty(
                        inputs.shape[0],
                        10,
                        device=device,
                    )

                    rows_by_skill: dict[
                        int,
                        list[int],
                    ] = {}

                    for row, label in enumerate(labels.tolist()):
                        skill = plugin.class_map.find_skill_for_class_anywhere(
                            int(label)
                        )

                        if skill is None:
                            raise RuntimeError(
                                f"No canonical skill recorded for class {label}."
                            )

                        rows_by_skill.setdefault(
                            int(skill),
                            [],
                        ).append(row)

                    for skill, rows in rows_by_skill.items():
                        apply_skill_state_exact(
                            model,
                            plugin.memory.state(skill),
                        )

                        row_tensor = torch.tensor(
                            rows,
                            device=device,
                        )

                        chosen_logits[row_tensor] = model(inputs[row_tensor])

                elif routing == "probe":
                    raw_logits = []

                    for state in skill_states:
                        apply_skill_state_exact(
                            model,
                            state,
                        )

                        raw_logits.append(model(inputs))

                    routing_result = find_best_routing_skill(
                        raw_logits,
                        skill_states,
                        [plugin.class_map.classes_for_skill(slot) for slot in slot_ids],
                    )

                    chosen = routing_result.skill_indices

                    stacked = torch.stack(
                        raw_logits,
                        dim=0,
                    )

                    rows = torch.arange(
                        inputs.shape[0],
                        device=device,
                    )

                    chosen_logits = stacked[
                        chosen,
                        rows,
                    ]

                else:
                    raise ValueError(f"Unknown Skill Memory routing mode: {routing}")

                correct += int((chosen_logits.argmax(dim=1) == labels).sum().item())

                total += int(labels.numel())

            if total == 0:
                raise RuntimeError(f"Test experience {experience_index} is empty.")

            accuracies.append(correct / total)

    finally:
        apply_skill_state_exact(
            model,
            original_state,
        )

    return accuracies


def build_evaluator(
    *,
    device: torch.device,
    learning_rate: float,
) -> tuple[
    EvaluationMLP,
    torch.optim.Optimizer,
    nn.Module,
]:
    """Create a completely independent evaluation learner."""
    model = EvaluationMLP().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmark = SplitMNIST(
        n_experiences=args.n_experiences,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )

    print("=== SplitMNIST Skill Memory experiment ===")
    print("Training method: Skill Memory")
    print(f"Evaluation method: {args.eval_method.upper()}")
    print(f"Device: {device}")
    print(f"Experiences: {len(benchmark.train_stream)}")
    print(f"Evaluation memory per class: {args.eval_memory_per_class}")

    for index, experience in enumerate(benchmark.train_stream):
        print(
            f"  Exp {index}: "
            f"classes="
            f"{sorted(experience.classes_in_this_experience)} "
            f"samples={len(experience.dataset)}"
        )

    if args.download_only:
        print(f"SplitMNIST dataset prepared at {args.dataset_root}")
        return

    # ================================================================
    # Class -> introducing experience mapping.
    #
    # This mapping is used ONLY for reporting. It is never used to
    # restrict predictions or choose evaluator training data.
    # ================================================================

    class_to_experience: dict[int, int] = {}

    for experience_index, experience in enumerate(benchmark.train_stream):
        for class_id in experience.classes_in_this_experience:
            class_to_experience[int(class_id)] = experience_index

    # ================================================================
    # Skill Memory training model
    # ================================================================

    model = SplitMNISTMLP().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    skill_memory_plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=args.max_skills),
        eval_routing="none",
        eval_memory_per_class=(args.eval_memory_per_class),
        eval_memory_seed=args.seed,
        verbose=True,
    )

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=args.batch_size,
        train_epochs=args.train_epochs,
        eval_mb_size=args.eval_batch_size,
        device=device,
        plugins=[skill_memory_plugin],
    )

    # ================================================================
    # Independent evaluation learner
    #
    # ML:
    #     replaced with a fresh evaluator every step.
    #
    # CL:
    #     same evaluator persists across steps.
    #
    # BOTH:
    #     receive the accumulated class memory.
    # ================================================================

    evaluator: EvaluationMLP | None = None
    evaluator_optimizer: torch.optim.Optimizer | None = None
    evaluator_criterion: nn.Module | None = None

    if args.eval_method == "cl":
        (
            evaluator,
            evaluator_optimizer,
            evaluator_criterion,
        ) = build_evaluator(
            device=device,
            learning_rate=args.eval_learning_rate,
        )

    accuracy_history: list[dict[int, float]] = []
    loss_history: list[dict[int, float]] = []

    diagonal_accuracy_history: list[float] = []
    diagonal_loss_history: list[float] = []

    for train_index, train_exp in enumerate(benchmark.train_stream):
        print()
        print(f"========== Training experience {train_index} ==========")

        # ------------------------------------------------------------
        # Actual Skill Memory training.
        # ------------------------------------------------------------

        strategy.train(train_exp)

        # ------------------------------------------------------------
        # Direct Skill Memory diagnostic.
        #
        # This is NOT the auxiliary ML/CL evaluator.
        # ------------------------------------------------------------

        if args.skill_eval_routing != "none":
            print()
            print("Direct Skill Memory evaluation (actual stored skills):")

            if args.skill_eval_routing in (
                "oracle",
                "both",
            ):
                oracle_accuracy = evaluate_skill_memory(
                    model,
                    skill_memory_plugin,
                    benchmark.test_stream,
                    train_index,
                    routing="oracle",
                    batch_size=(args.eval_batch_size),
                    device=device,
                )

                print(
                    "  class_oracle_accuracy="
                    + ", ".join(
                        f"Exp{i}={value:.4f}" for i, value in enumerate(oracle_accuracy)
                    )
                )

            if args.skill_eval_routing in (
                "probe",
                "both",
            ):
                probe_accuracy = evaluate_skill_memory(
                    model,
                    skill_memory_plugin,
                    benchmark.test_stream,
                    train_index,
                    routing="probe",
                    batch_size=(args.eval_batch_size),
                    device=device,
                )

                print(
                    "  probe_accuracy="
                    + ", ".join(
                        f"Exp{i}={value:.4f}" for i, value in enumerate(probe_accuracy)
                    )
                )

        # ------------------------------------------------------------
        # Evaluation memory must exist for every trained experience.
        # ------------------------------------------------------------

        if len(skill_memory_plugin.eval_memory) == 0:
            raise RuntimeError("Evaluation memory is empty.")

        # ------------------------------------------------------------
        # Consolidate all evaluation memories into one class-level
        # memory. Experience boundaries are intentionally removed from
        # the evaluator's training unit.
        # ------------------------------------------------------------

        accumulated_memory = consolidate_evaluation_memory(
            skill_memory_plugin.eval_memory
        )

        accumulated_classes = [memory.class_id for memory in accumulated_memory]

        print()
        print("Accumulated evaluation memory:")
        print(f"  classes={accumulated_classes}")
        print(f"  samples={sum(memory.size for memory in accumulated_memory)}")

        # ------------------------------------------------------------
        # Auxiliary evaluation.
        # ------------------------------------------------------------

        print()
        print(f"========== Evaluation after experience {train_index} ==========")

        if args.eval_method == "ml":
            (
                evaluator,
                evaluator_optimizer,
                evaluator_criterion,
            ) = build_evaluator(
                device=device,
                learning_rate=args.eval_learning_rate,
            )

            print("Auxiliary evaluator: ML")
            print("Evaluator training memory: all accumulated classes")

        else:
            print("Auxiliary evaluator: CL")
            print("Evaluator training memory: all accumulated classes")

        assert evaluator is not None
        assert evaluator_optimizer is not None
        assert evaluator_criterion is not None

        train_evaluator(
            evaluator,
            evaluator_optimizer,
            evaluator_criterion,
            accumulated_memory,
            batch_size=args.eval_batch_size,
            epochs=args.eval_epochs,
            device=device,
            seed=args.seed + train_index,
        )

        # ------------------------------------------------------------
        # Evaluate globally, class by class.
        #
        # There is deliberately NO experience-based class mask.
        # ------------------------------------------------------------

        class_results = evaluate_model_by_class(
            evaluator,
            benchmark.test_stream,
            train_index,
            batch_size=args.eval_batch_size,
            device=device,
        )

        losses, accuracies = aggregate_experience_metrics(
            class_results,
            benchmark.test_stream,
            train_index,
        )

        current_accuracy = {
            class_id: values["accuracy"] for class_id, values in class_results.items()
        }

        current_loss = {
            class_id: values["loss"] for class_id, values in class_results.items()
        }

        accuracy_history.append(current_accuracy)
        loss_history.append(current_loss)

        current_classes = sorted(
            int(class_id) for class_id in (train_exp.classes_in_this_experience)
        )

        diagonal_accuracy = float(
            np.mean([current_accuracy[class_id] for class_id in current_classes])
        )

        diagonal_loss = float(
            np.mean([current_loss[class_id] for class_id in current_classes])
        )

        diagonal_accuracy_history.append(diagonal_accuracy)
        diagonal_loss_history.append(diagonal_loss)

        print(f"Step {train_index}: classes={current_classes}")

        print(
            "  class_accuracy="
            + ", ".join(
                f"class{class_id}={current_accuracy[class_id]:.4f}"
                for class_id in sorted(current_accuracy)
            )
        )

        print(
            "  class_loss="
            + ", ".join(
                f"class{class_id}={current_loss[class_id]:.4f}"
                for class_id in sorted(current_loss)
            )
        )

        print(
            "  experience_accuracy="
            + ", ".join(f"Exp{i}={value:.4f}" for i, value in enumerate(accuracies))
        )

        print(
            "  experience_loss="
            + ", ".join(f"Exp{i}={value:.4f}" for i, value in enumerate(losses))
        )

        print(f"  diagonal_accuracy={diagonal_accuracy:.4f}")

    # ================================================================
    # Final class-level summary
    # ================================================================

    final_class_accuracy = accuracy_history[-1]

    final_class_loss = loss_history[-1]

    forgetting = compute_class_forgetting(
        accuracy_history,
        class_to_experience,
        len(benchmark.train_stream),
    )

    diagonal_accuracy = np.asarray(
        diagonal_accuracy_history,
        dtype=np.float64,
    )

    diagonal_loss = np.asarray(
        diagonal_loss_history,
        dtype=np.float64,
    )

    final_accuracy = np.asarray(
        [final_class_accuracy[class_id] for class_id in sorted(final_class_accuracy)],
        dtype=np.float64,
    )

    final_loss = np.asarray(
        [final_class_loss[class_id] for class_id in sorted(final_class_loss)],
        dtype=np.float64,
    )

    print()
    print("=== Summary ===")
    print("training_method=SkillMemory")
    print(f"auxiliary_eval_method={args.eval_method}")
    print(f"eval_memory_per_class={args.eval_memory_per_class}")

    print(
        "diagonal_loss:",
        np.round(diagonal_loss, 4),
    )

    print(
        "diagonal_accuracy:",
        np.round(diagonal_accuracy, 4),
    )

    print(
        "final_class_loss:",
        np.round(final_loss, 4),
    )

    print(
        "final_class_accuracy:",
        np.round(final_accuracy, 4),
    )

    print(
        "class_forgetting_by_introducing_experience:",
        np.round(forgetting, 4),
    )

    print(f"mean_diagonal_loss={diagonal_loss.mean():.4f}")

    print(f"mean_auxiliary_diagonal_accuracy={diagonal_accuracy.mean():.4f}")

    print(f"mean_auxiliary_final_accuracy={final_accuracy.mean():.4f}")

    print(f"mean_class_forgetting={forgetting.mean():.4f}")


if __name__ == "__main__":
    main()
