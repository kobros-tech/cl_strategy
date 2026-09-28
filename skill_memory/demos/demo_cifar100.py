# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Selected-experience CIFAR-100 Skill Memory experiment.

The demo keeps the normal continual-learning order while allowing a subset
of SplitCIFAR100 experiences to be selected from the command line.

Examples
--------
Run experiences 0, 1 and 2 sequentially::

    python -m skill_memory.demos.demo_cifar100 --experience-indices 0,1,2

Run only experience 0::

    python -m skill_memory.demos.demo_cifar100 --experience-indices 0

The selected experiences are trained cumulatively in the order supplied.
Evaluation after each selected experience covers only the selected
experiences trained so far, so skipped/future classes are never evaluated.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.models import SlimResNet18
from torch import nn

from skill_memory import SkillMemoryStrategy
from skill_memory.diagnostics import (
    diagnose_evaluator_probe,
    evaluate_class_oracle,
    evaluate_skill_memory,
    timing_report,
)


def _parse_experience_indices(value: str) -> list[int]:
    """Parse a comma-separated list of unique CIFAR-100 experience indices."""
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "experience indices must be comma-separated integers"
        ) from exc

    if not indices:
        raise argparse.ArgumentTypeError(
            "at least one experience index is required"
        )

    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError(
            "experience indices must be unique"
        )

    if any(index < 0 or index >= 20 for index in indices):
        raise argparse.ArgumentTypeError(
            "CIFAR-100 experience indices must be between 0 and 19"
        )

    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate Skill Memory on selected SplitCIFAR100 experiences."
        )
    )
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--experience-indices",
        type=_parse_experience_indices,
        default=[0, 1, 2],
        help="Comma-separated experience indices to train sequentially, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--eval-memory-per-class",
        type=int,
        default=20,
        help="Frozen evaluation examples retained per class.",
    )
    parser.add_argument(
        "--skill-train-samples-per-class",
        type=int,
        default=20,
        help="Training examples per class used by Skill Memory.",
    )
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument(
        "--class-train-mode",
        choices=("multiclass", "binary_one_vs_rest"),
        default="multiclass",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--eval-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--eval-learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-skills", type=int, default=100)
    parser.add_argument(
        "--eval-routing",
        choices=("none", "probe"),
        default="none",
        help=(
            "Evaluation routing: independent evaluator only or anonymous "
            "Skill Memory probe."
        ),
    )
    parser.add_argument(
        "--probe-behavior-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run optional Skill Memory oracle/probe diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmark = SplitCIFAR100(
        n_experiences=20,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )

    selected_indices = args.experience_indices

    print("=== CIFAR-100 Skill Memory experiment ===")
    print("Training method: Skill Memory")
    print("Evaluation method: independent anonymous ML evaluator")
    print(f"Evaluation routing: {args.eval_routing}")
    print(f"Selected experiences: {selected_indices}")
    print(f"Device: {device}")
    print(f"Experiences in benchmark: {len(benchmark.train_stream)}")
    print(f"Evaluation memory per class: {args.eval_memory_per_class}")
    print(f"Skill training samples per class: {args.skill_train_samples_per_class}")
    print(f"Skill training epochs: {args.train_epochs}")
    print(f"Evaluation epochs: {args.eval_epochs}")

    for index in selected_indices:
        experience = benchmark.train_stream[index]
        print(
            f"  Exp {index}: "
            f"classes={sorted(experience.classes_in_this_experience)} "
            f"samples={len(experience.dataset)}"
        )

    if args.download_only:
        print(f"CIFAR-100 dataset prepared at {args.dataset_root}")
        return

    model = SlimResNet18(
        nclasses=100,
        input_size=(3, 32, 32),
    ).to(device)

    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
        ),
        criterion=nn.CrossEntropyLoss(),
        max_skills=args.max_skills,
        train_mb_size=args.batch_size,
        train_epochs=args.train_epochs,
        class_train_mode=args.class_train_mode,
        skill_train_samples_per_class=args.skill_train_samples_per_class,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        eval_mb_size=args.eval_batch_size,
        evaluator_model_factory=lambda: SlimResNet18(
            nclasses=100,
        ),
        eval_memory_per_class=args.eval_memory_per_class,
        eval_epochs=args.eval_epochs,
        eval_learning_rate=args.eval_learning_rate,
        probe_seed=args.seed,
        device=device,
        diagnose=args.diagnose,
        verbose=True,
        eval_routing=args.eval_routing,
        probe_behavior_weight=args.probe_behavior_weight,
    )

    trained_indices: list[int] = []

    for experience_index in selected_indices:
        experience = benchmark.train_stream[experience_index]

        print()
        print(f"========== Training experience {experience_index} ==========")
        print(
            "Classes:",
            sorted(
                int(class_id)
                for class_id in experience.classes_in_this_experience
            ),
        )

        strategy.train(experience)
        trained_indices.append(experience_index)

        eval_stream = [
            benchmark.test_stream[index]
            for index in trained_indices
        ]

        print("Skill Memory groups after training:")
        for skill in sorted(strategy.skill_memory.slots()):
            classes = sorted(
                strategy.skill_memory_plugin.class_map.classes_for_skill(skill)
            )
            print(f"  skill {skill}: classes={classes}")

        print(
            f"========== Evaluation after experience {experience_index} =========="
        )
        strategy.eval(eval_stream)

        if args.diagnose:
            class_oracle = evaluate_class_oracle(
                strategy.model,
                strategy.skill_memory_plugin,
                benchmark.test_stream,
                experience_index,
                num_classes=100,
                batch_size=args.eval_batch_size,
                device=device,
                diagnose=True,
            )
            direct_probe = evaluate_skill_memory(
                strategy.model,
                strategy.skill_memory_plugin,
                benchmark.test_stream,
                experience_index,
                num_classes=100,
                routing="probe",
                batch_size=args.eval_batch_size,
                device=device,
                diagnose=True,
            )
            evaluator_probe = diagnose_evaluator_probe(
                strategy,
                eval_stream,
                batch_size=args.eval_batch_size,
                diagnose=True,
            )
            print(
                "class_oracle_mean_accuracy=",
                f"{np.mean([item['accuracy'] for item in class_oracle.values()]):.4f}",
            )
            print(
                "direct_probe_mean_accuracy=",
                f"{np.mean([item['accuracy'] for item in direct_probe.values()]):.4f}",
            )
            print(
                "evaluator_probe_routing_accuracy=",
                f"{evaluator_probe['probe_routing_accuracy']:.4f}",
            )
            print(
                "evaluator_probe_mean_confidence=",
                f"{evaluator_probe['probe_mean_confidence']:.4f}",
            )
            print(
                "evaluator_probe_mean_margin=",
                f"{evaluator_probe['probe_mean_margin']:.4f}",
            )
            for bucket, stats in timing_report(strategy).items():
                print(
                    f"timing[{bucket}]: total={stats['total_seconds']:.2f}s "
                    f"calls={stats['calls']} mean={stats['mean_seconds']:.3f}s"
                )

    results = strategy.results()

    print()
    print("=== Summary ===")
    print(f"mean_final_accuracy={results['mean_final_accuracy']:.4f}")
    print(f"mean_final_loss={results['mean_final_loss']:.4f}")

    print("final_class_accuracy:")
    for class_id, accuracy in results["final_class_accuracy"].items():
        print(f"  class {class_id}: {accuracy:.4f}")

    print("final_class_loss:")
    for class_id, loss in results["final_class_loss"].items():
        print(f"  class {class_id}: {loss:.4f}")


if __name__ == "__main__":
    main()
