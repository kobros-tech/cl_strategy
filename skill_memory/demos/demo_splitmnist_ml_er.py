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

All of the reusable machinery this demo drives (the frozen per-class
evaluation memory, the independent evaluator's train/evaluate loop, and
direct Skill Memory oracle/probe evaluation) lives in
`skill_memory.evaluation.ml_cl_evaluator`; this script only wires it up
against SplitMNIST, parses CLI flags, and prints results.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training import Naive

from skill_memory.cl.skill_registry import SkillMemory
from skill_memory.evaluation.ml_cl_evaluator import (
    EvaluationMemoryPlugin,
    aggregate_experience_metrics,
    build_evaluator,
    compute_class_forgetting,
    compute_peak_class_forgetting,
    consolidate_evaluation_memory,
    evaluate_model_by_class,
    evaluate_skill_memory,
    train_evaluator,
)
from skill_memory.utils.models import SimpleMLP

NUM_CLASSES = 10  # SplitMNIST's fixed global class count.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Skill Memory on SplitMNIST and evaluate its learned "
            "knowledge using independent ML/CL evaluation learners."
        )
    )
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--n-experiences", type=int, default=5)
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
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--eval-learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-skills", type=int, default=20)
    parser.add_argument(
        "--skill-eval-routing",
        choices=("oracle", "probe", "both", "none"),
        default="both",
        help=(
            "Direct Skill Memory diagnostic. 'oracle' uses the true "
            "class-to-skill mapping; 'probe' uses anonymous routing; "
            "'both' runs both; 'none' disables direct Skill Memory "
            "evaluation."
        ),
    )
    return parser.parse_args()


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
    print(f"Device: {device}")
    print(f"Experiences: {len(benchmark.train_stream)}")
    print(f"Evaluation memory per class: {args.eval_memory_per_class}")

    for index, experience in enumerate(benchmark.train_stream):
        print(
            f"  Exp {index}: "
            f"classes={sorted(experience.classes_in_this_experience)} "
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

    model = SimpleMLP(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    skill_memory_plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=args.max_skills),
        eval_routing="none",
        eval_memory_per_class=args.eval_memory_per_class,
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
    # ML: a fresh evaluator is created every step.
    # CL:  same evaluator persists across steps.
    # Both receive the accumulated class memory.
    # ================================================================
    skill_accuracy_history: dict[str, list[dict[int, float]]] = {
        "oracle": [],
        "probe": [],
    }

    def new_evaluator():
        return build_evaluator(
            lambda: SimpleMLP(num_classes=NUM_CLASSES),
            device=device,
            learning_rate=args.eval_learning_rate,
        )

    evaluator = evaluator_optimizer = evaluator_criterion = None
    evaluator, evaluator_optimizer, evaluator_criterion = new_evaluator()

    accuracy_history: list[dict[int, float]] = []
    loss_history: list[dict[int, float]] = []
    diagonal_accuracy_history: list[float] = []
    diagonal_loss_history: list[float] = []

    for train_index, train_exp in enumerate(benchmark.train_stream):
        print()
        print(f"========== Training experience {train_index} ==========")

        # Actual Skill Memory training.
        strategy.train(train_exp)

        if args.skill_eval_routing != "none":
            routings = (
                ("oracle", "probe")
                if args.skill_eval_routing == "both"
                else (args.skill_eval_routing,)
            )

            for routing in routings:
                class_results = evaluate_skill_memory(
                    model,
                    skill_memory_plugin,
                    benchmark.test_stream,
                    train_index,
                    num_classes=NUM_CLASSES,
                    routing=routing,
                    batch_size=args.eval_batch_size,
                    device=device,
                )

                current_accuracy = {
                    class_id: values["accuracy"]
                    for class_id, values in class_results.items()
                }
                skill_accuracy_history[routing].append(current_accuracy)

        # Evaluation memory must exist for every trained experience.
        if len(skill_memory_plugin.eval_memory) == 0:
            raise RuntimeError("Evaluation memory is empty.")

        # Consolidate all evaluation memories into one class-level memory.
        # Experience boundaries are intentionally removed from the
        # evaluator's training unit.
        accumulated_memory = consolidate_evaluation_memory(
            skill_memory_plugin.eval_memory
        )
        accumulated_classes = [memory.class_id for memory in accumulated_memory]
        print()
        print("Accumulated evaluation memory:")
        print(f"  classes={accumulated_classes}")
        print(f"  samples={sum(memory.size for memory in accumulated_memory)}")

        # Auxiliary evaluation.
        print()
        print(f"========== Evaluation after experience {train_index} ==========")
        print(
            "  [auxiliary evaluator only - trained on raw retained pixels, "
            "NOT on Skill Memory's weights/features/predictions; not "
            "evidence about Skill Memory's own learned representation, "
            "see skill_memory.evaluation.ml_cl_evaluator's module docstring]"
        )
        print("Auxiliary evaluator: ML")
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

        # Evaluate globally, class by class - deliberately NO
        # experience-based class mask.
        class_results = evaluate_model_by_class(
            evaluator,
            benchmark.test_stream,
            train_index,
            batch_size=args.eval_batch_size,
            device=device,
        )
        losses, accuracies = aggregate_experience_metrics(
            class_results, benchmark.test_stream, train_index
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
            int(class_id) for class_id in train_exp.classes_in_this_experience
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
        accuracy_history, class_to_experience, len(benchmark.train_stream)
    )
    peak_forgetting = compute_peak_class_forgetting(
        accuracy_history, class_to_experience, len(benchmark.train_stream)
    )
    diagonal_accuracy = np.asarray(diagonal_accuracy_history, dtype=np.float64)
    diagonal_loss = np.asarray(diagonal_loss_history, dtype=np.float64)
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
    print(
        "NOTE: the auxiliary_* / diagonal_* / final_class_* / "
        "*_forgetting figures below all come from the AUXILIARY ML/CL "
        "evaluator (trained on raw retained pixels, not on Skill Memory's "
        "learned representation) - they are not evidence for or against "
        "Skill Memory itself. For that, use the "
        "'Direct Skill Memory evaluation (actual stored skills)' "
        "oracle/probe accuracies printed after each training experience "
        "above."
    )
    print(f"eval_memory_per_class={args.eval_memory_per_class}")
    print("diagonal_loss:", np.round(diagonal_loss, 4))
    print("diagonal_accuracy:", np.round(diagonal_accuracy, 4))
    print("final_class_loss:", np.round(final_loss, 4))
    print("final_class_accuracy:", np.round(final_accuracy, 4))
    print(
        "class_forgetting_by_introducing_experience (acquisition-relative):",
        np.round(forgetting, 4),
    )
    print(
        "class_forgetting_by_introducing_experience (peak-relative, standard CL "
        "definition):",
        np.round(peak_forgetting, 4),
    )
    print(f"mean_diagonal_loss={diagonal_loss.mean():.4f}")
    print(f"mean_auxiliary_diagonal_accuracy={diagonal_accuracy.mean():.4f}")
    print(f"mean_auxiliary_final_accuracy={final_accuracy.mean():.4f}")
    print(f"mean_class_forgetting_acquisition_relative={forgetting.mean():.4f}")
    print(f"mean_class_forgetting_peak_relative={peak_forgetting.mean():.4f}")

    skill_peak_forgetting = {
        routing: compute_peak_class_forgetting(
            history,
            class_to_experience,
            len(benchmark.train_stream),
        )
        for routing, history in skill_accuracy_history.items()
        if history
    }

    print("Primary Skill Memory metrics (actual stored skills):")
    for routing, history in skill_accuracy_history.items():
        if not history:
            continue

        final = history[-1]
        values = np.asarray(list(final.values()), dtype=np.float64)
        forgetting = skill_peak_forgetting[routing]

        print(f"skill_memory_{routing}_final_class_accuracy={values.mean():.4f}")
        print(
            f"skill_memory_{routing}_peak_forgetting_by_introducing_experience=",
            np.round(forgetting, 4),
        )
        print(f"mean_skill_memory_{routing}_peak_forgetting={forgetting.mean():.4f}")


if __name__ == "__main__":
    main()
