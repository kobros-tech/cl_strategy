"""SplitMNIST Skill Memory experiment with anonymous ML evaluation.

The public SkillMemoryStrategy owns the complete experiment lifecycle:

* Skill Memory training
* frozen per-class evaluation memory
* independent ML evaluator training
* anonymous x -> y evaluation through Avalanche's normal eval() lifecycle
* accuracy/loss tracking
* forgetting metrics

The demo only configures SplitMNIST and the strategy, then reports results.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from torch import nn

from skill_memory import SkillMemoryStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Skill Memory on SplitMNIST."
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
        "--weight-state-ml1-epochs",
        type=int,
        default=10,
        help="Number of epochs used to train ML-1 (x -> omega).",
    )
    parser.add_argument(
        "--weight-state-ml1-batch-size",
        type=int,
        default=64,
        help="Training batch size for ML-1 (x -> omega).",
    )
    parser.add_argument(
        "--weight-state-ml1-learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for ML-1 (x -> omega).",
    )
    parser.add_argument(
        "--weight-state-eval-epochs",
        type=int,
        default=10,
        help="Number of epochs used to train ML-2 (omega -> y).",
    )
    parser.add_argument(
        "--weight-state-eval-batch-size",
        type=int,
        default=64,
        help="Training batch size for ML-2 (omega -> y).",
    )
    parser.add_argument(
        "--weight-state-eval-learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for ML-2 (omega -> y).",
    )
    parser.add_argument(
        "--weight-state-snapshots-per-class",
        type=int,
        default=10,
        help="Maximum retained post-step omega snapshots per class.",
    )
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-skills", type=int, default=20)
    parser.add_argument(
        "--skill-eval-routing",
        choices=("none",),
        default="none",
        help=(
            "Use the anonymous two-stage weight-state evaluator: "
            "x -> ML-1 -> omega -> ML-2 -> y."
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
    print("Evaluation method: independent anonymous ML evaluator")
    print(f"Device: {device}")
    print(f"Experiences: {len(benchmark.train_stream)}")
    print(
        "Evaluation memory per class:",
        args.eval_memory_per_class,
    )

    for index, experience in enumerate(benchmark.train_stream):
        print(
            f"  Exp {index}: "
            f"classes={sorted(experience.classes_in_this_experience)} "
            f"samples={len(experience.dataset)}"
        )

    if args.download_only:
        print(f"SplitMNIST dataset prepared at {args.dataset_root}")
        return

    # ------------------------------------------------------------------
    # Main Skill Memory model.
    # ------------------------------------------------------------------

    model = SimpleMLP(
        num_classes=10,
    ).to(device)

    # ------------------------------------------------------------------
    # SkillMemoryStrategy owns:
    #
    #   1. Skill Memory training
    #   2. evaluation-memory retention
    #   3. independent ML evaluator
    #   4. normal Avalanche evaluation lifecycle
    #
    # The evaluator receives only x at evaluation time and predicts y
    # through ML-1 (x -> omega) and ML-2 (omega -> y).
    # No experience ID, task label, class oracle, or skill ID is supplied.
    # ------------------------------------------------------------------

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
        eval_mb_size=args.eval_batch_size,
        evaluator_model_factory=lambda: SimpleMLP(
            num_classes=10,
            input_size=28 * 28,
            hidden_size=2048,
            hidden_layers=1,
            drop_rate=0.1,
        ),
        eval_memory_per_class=args.eval_memory_per_class,
        skill_eval_routing=args.skill_eval_routing,
        weight_state_ml1_epochs=args.weight_state_ml1_epochs,
        weight_state_ml1_batch_size=args.weight_state_ml1_batch_size,
        weight_state_ml1_learning_rate=args.weight_state_ml1_learning_rate,
        weight_state_snapshots_per_class=args.weight_state_snapshots_per_class,
        weight_state_eval_epochs=args.weight_state_eval_epochs,
        weight_state_eval_batch_size=args.weight_state_eval_batch_size,
        weight_state_eval_learning_rate=args.weight_state_eval_learning_rate,
        probe_seed=args.seed,
        device=device,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Complete Avalanche lifecycle.
    #
    # Training:
    #     strategy.train(experience)
    #
    # Evaluation:
    #     strategy.eval(benchmark.test_stream)
    #
    # The ML evaluator is invoked automatically by its Avalanche plugin
    # during strategy.eval().
    # ------------------------------------------------------------------

    for experience_index, experience in enumerate(benchmark.train_stream):
        print()
        print(f"========== Training experience {experience_index} ==========")
        print(
            "Classes:",
            sorted(int(class_id) for class_id in experience.classes_in_this_experience),
        )

        # Normal Avalanche training lifecycle.
        strategy.train(experience)

        print(f"========== Evaluation after experience {experience_index} ==========")

        # Normal Avalanche evaluation lifecycle.
        #
        # The weight-state evaluator is trained automatically on all
        # accumulated post-step (x, omega, y) trajectory records before
        # this evaluation.
        strategy.eval(benchmark.test_stream)

    # ------------------------------------------------------------------
    # Final results.
    # ------------------------------------------------------------------

    results = strategy.results()["weight_state_evaluation"]

    print()
    print("=== Summary ===")
    print(
        "true_omega_accuracy=",
        f"{results['true_omega_accuracy']:.4f}",
    )
    print(
        "end_to_end_accuracy=",
        f"{results['end_to_end_accuracy']:.4f}",
    )
    print(
        "omega_mse=",
        f"{results['omega_mse']:.4f}",
    )

    print(
        "mean_final_accuracy=",
        f"{results['mean_final_accuracy']:.4f}",
    )
    print(
        "mean_final_loss=",
        f"{results['mean_final_loss']:.4f}",
    )

    print("final_class_accuracy:")
    for class_id, accuracy in results["final_class_accuracy"].items():
        print(f"  class {class_id}: {accuracy:.4f}")

    print("final_class_loss:")
    for class_id, loss in results["final_class_loss"].items():
        print(f"  class {class_id}: {loss:.4f}")


if __name__ == "__main__":
    main()
