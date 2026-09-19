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
from skill_memory.cl.skill_registry import ClassRecord
from skill_memory.utils.probing import classes_in_experience


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
    print("\n=== Summary ===")
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
        print(
            "NOTE: ML/ER training is unchanged; Skill Memory evaluation "
            f"routing={args.eval_routing!r} runs over post-experience snapshots."
        )


if __name__ == "__main__":
    main()
