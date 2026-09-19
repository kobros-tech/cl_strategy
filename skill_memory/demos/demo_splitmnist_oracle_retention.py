"""Diagnostic SplitMNIST comparison for Skill Memory and standard baselines.

Modes:
- skill-memory-class-oracle: Skill Memory + true class-to-skill routing.
- skill-memory-cl-probe: Skill Memory + label-free CL probe routing.
- skill-memory-ml-probe: Skill Memory + label-free standalone ML router.
- ml: ordinary Avalanche Naive supervised learning.
- er: Avalanche Experience Replay.

The oracle mode is diagnostic only.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.templates import SupervisedTemplate

from skill_memory import PersistentFingerprintSkillMemoryPlugin, SkillMemory
from skill_memory.cl import SkillMemoryPlugin


class SkillMemoryMLP(nn.Module):
    """Small MLP with an Avalanche growing classifier head."""

    def __init__(self, input_dim: int, hidden_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.classifier = IncrementalClassifier(hidden_size, initial_out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(x.size(0), -1)
        return self.classifier(self.features(x))


def evaluate_seen(strategy, test_stream, up_to_index: int) -> list[float]:
    """Evaluate seen experiences with canonical class-to-skill routing."""
    results = strategy.eval([test_stream[i] for i in range(up_to_index + 1)])
    keys = sorted(key for key in results if key.startswith("Top1_Acc_Exp"))
    return [float(results[key]) for key in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Skill Memory routing with ordinary ML and ER."
    )
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--n-experiences", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=(
            "skill-memory-class-oracle",
            "skill-memory-cl-probe",
            "skill-memory-ml-probe",
            "ml",
            "er",
        ),
        default="skill-memory-class-oracle",
    )
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-memory-size", type=int, default=200)
    parser.add_argument("--reverse-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_strategy(args: argparse.Namespace, device: torch.device):
    """Build the requested training/evaluation configuration."""
    model = SkillMemoryMLP(input_dim=784).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    if args.mode == "ml":
        return model, Naive(
            model=model, optimizer=optimizer, criterion=criterion,
            train_mb_size=args.batch_size, train_epochs=args.train_epochs,
            eval_mb_size=args.batch_size, device=device,
        ), None

    if args.mode == "er":
        return model, Naive(
            model=model, optimizer=optimizer, criterion=criterion,
            train_mb_size=args.batch_size, train_epochs=args.train_epochs,
            eval_mb_size=args.batch_size, device=device,
            plugins=[ReplayPlugin(mem_size=args.replay_memory_size)],
        ), None

    if args.mode == "skill-memory-ml-probe":
        plugin = PersistentFingerprintSkillMemoryPlugin(
            memory=SkillMemory(max_skills=10),
            forgetting_margin=0.05,
            probe_batch_size=10,
            probe_batches=5,
            probe_seed=args.seed,
            class_train_epochs=1,
            class_train_batch_size=args.batch_size,
            reuse_is_mutable=False,
            eval_routing="ml_probe",
            reverse_epochs=args.reverse_epochs,
            reverse_batch_size=256,
            verbose=True,
        )
    else:
        routing = (
            "class_oracle"
            if args.mode == "skill-memory-class-oracle"
            else "cl_probe"
        )
        plugin = SkillMemoryPlugin(
            memory=SkillMemory(max_skills=10),
            forgetting_margin=0.05,
            probe_batch_size=10,
            probe_batches=5,
            probe_seed=args.seed,
            class_train_epochs=1,
            class_train_batch_size=args.batch_size,
            reuse_is_mutable=True,
            eval_routing=routing,
            verbose=True,
        )

    strategy = SupervisedTemplate(
        model=model, optimizer=optimizer, criterion=criterion,
        train_mb_size=args.batch_size, train_epochs=args.train_epochs,
        eval_mb_size=args.batch_size, device=device, plugins=[plugin],
    )
    return model, strategy, plugin


def evaluate_seen(
    strategy, test_stream, up_to_index: int
) -> tuple[list[float], list[float]]:
    """Evaluate all seen experiences using Avalanche's normal evaluator."""
    results = strategy.eval([test_stream[i] for i in range(up_to_index + 1)])
    loss_keys = sorted(key for key in results if key.startswith("Loss_Exp"))
    acc_keys = sorted(key for key in results if key.startswith("Top1_Acc_Exp"))
    expected = up_to_index + 1
    if len(loss_keys) != expected or len(acc_keys) != expected:
        raise RuntimeError(
            f"Expected {expected} loss/accuracy metrics; "
            f"found {len(loss_keys)}/{len(acc_keys)}."
        )
    return (
        [float(results[key]) for key in loss_keys],
        [float(results[key]) for key in acc_keys],
    )


def summarize(history: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(history)
    curve = np.array([history[i][i] for i in range(n)])
    forgetting = np.zeros(n)
    for exp_index in range(n):
        seen = [row[exp_index] for row in history[exp_index:]]
        if len(seen) > 1:
            forgetting[exp_index] = max(seen[:-1]) - seen[-1]
    return curve, forgetting


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

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Experiences: {len(benchmark.train_stream)}")
    for index, experience in enumerate(benchmark.train_stream):
        print(
            f"  Exp {index}: classes="
            f"{sorted(experience.classes_in_this_experience)} "
            f"samples={len(experience.dataset)}"
        )

    if args.download_only:
        print(f"SplitMNIST dataset prepared at {args.dataset_root}")
        return

    _, strategy, plugin = build_strategy(args, device)
    history: list[list[float]] = []

    for train_index, train_exp in enumerate(benchmark.train_stream):
        strategy.train(train_exp)
        losses, accuracies = evaluate_seen(
            strategy, benchmark.test_stream, train_index
        )
        history.append(accuracies)

        print(
            f"Step {train_index}: classes="
            f"{sorted(train_exp.classes_in_this_experience)}"
        )
        print(
            "  loss="
            + ", ".join(
                f"Exp{i}={value:.4f}" for i, value in enumerate(losses)
            )
        )
        print(
            "  accuracy="
            + ", ".join(
                f"Exp{i}={value:.4f}" for i, value in enumerate(accuracies)
            )
        )

        if plugin is not None:
            print(
                "  class->skill=",
                plugin.class_map.class_skill_for_experience(train_index),
            )

    curve, forgetting = summarize(history)
    print("
=== Summary ===")
    print(f"mode={args.mode}")
    print("accuracy_curve:", np.round(curve, 4))
    print("forgetting:", np.round(forgetting, 4))
    print(f"mean_final_accuracy={curve.mean():.4f}")

    if args.mode == "skill-memory-class-oracle":
        print("NOTE: class_oracle uses true labels only for skill selection.")
    elif args.mode == "skill-memory-ml-probe":
        print("NOTE: ml_probe is label-free and uses the standalone ML router.")
    elif args.mode == "skill-memory-cl-probe":
        print("NOTE: cl_probe is label-free and uses the Skill Memory router.")
    else:
        print("NOTE: this baseline does not use Skill Memory routing.")


if __name__ == "__main__":
    main()
