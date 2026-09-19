"""Simple SplitMNIST ML-vs-ER evaluation demo.

This demo is independent from SkillMemory routing.

The same SplitMNIST benchmark is trained with either ordinary supervised ML
or Avalanche Experience Replay. After each training experience, the model is
evaluated on all experiences seen so far.
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


class SplitMNISTMLP(nn.Module):
    """Small MLP with an Avalanche growing classifier head."""

    def __init__(self, input_dim: int = 784, hidden_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.classifier = IncrementalClassifier(
            hidden_size,
            initial_out_features=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(x.size(0), -1)
        return self.classifier(self.features(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ordinary ML and Avalanche Experience Replay "
        "on SplitMNIST."
    )
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--n-experiences", type=int, default=10)
    parser.add_argument(
        "--use-cl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Avalanche Experience Replay instead of ordinary ML.",
    )
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-memory-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_strategy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    use_cl: bool,
    train_epochs: int,
    batch_size: int,
    device: torch.device,
    replay_memory_size: int,
):
    """Build the requested Avalanche strategy."""

    plugins = []
    if use_cl:
        plugins.append(ReplayPlugin(mem_size=replay_memory_size))

    return Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=batch_size,
        train_epochs=train_epochs,
        eval_mb_size=batch_size,
        device=device,
        plugins=plugins,
    )


def evaluate_seen(
    strategy,
    test_stream,
    up_to_index: int,
) -> tuple[list[float], list[float]]:
    """Evaluate all experiences seen so far."""

    results = strategy.eval([test_stream[i] for i in range(up_to_index + 1)])

    loss_keys = sorted(key for key in results if key.startswith("Loss_Exp"))
    acc_keys = sorted(
        key for key in results if key.startswith("Top1_Acc_Exp")
    )

    expected = up_to_index + 1
    if len(loss_keys) != expected or len(acc_keys) != expected:
        raise RuntimeError(
            "Expected one loss and accuracy metric per seen experience, "
            f"found losses={len(loss_keys)}, accuracies={len(acc_keys)}, "
            f"expected={expected}."
        )

    return (
        [float(results[key]) for key in loss_keys],
        [float(results[key]) for key in acc_keys],
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "ER-CL" if args.use_cl else "ML"

    benchmark = SplitMNIST(
        n_experiences=args.n_experiences,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )

    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Experiences: {len(benchmark.train_stream)}")

    for index, experience in enumerate(benchmark.train_stream):
        print(
            f"  Exp {index}: "
            f"classes={sorted(experience.classes_in_this_experience)} "
            f"samples={len(experience.dataset)}"
        )

    if args.download_only:
        print(f"SplitMNIST dataset prepared at {args.dataset_root}")
        return

    model = SplitMNISTMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    strategy = build_strategy(
        model,
        optimizer,
        criterion,
        use_cl=args.use_cl,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        device=device,
        replay_memory_size=args.replay_memory_size,
    )

    loss_history: list[list[float]] = []
    accuracy_history: list[list[float]] = []

    for train_index, train_exp in enumerate(benchmark.train_stream):
        strategy.train(train_exp)

        losses, accuracies = evaluate_seen(
            strategy,
            benchmark.test_stream,
            train_index,
        )
        loss_history.append(losses)
        accuracy_history.append(accuracies)

        print(
            f"Step {train_index}: "
            f"classes={sorted(train_exp.classes_in_this_experience)}"
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

    diagonal_loss = np.array(
        [loss_history[i][i] for i in range(len(loss_history))]
    )
    diagonal_accuracy = np.array(
        [accuracy_history[i][i] for i in range(len(accuracy_history))]
    )

    forgetting = np.zeros(len(accuracy_history))
    for exp_index in range(len(accuracy_history)):
        seen = [row[exp_index] for row in accuracy_history[exp_index:]]
        if len(seen) > 1:
            forgetting[exp_index] = max(seen[:-1]) - seen[-1]

    print("\n=== Summary ===")
    print(f"mode={mode}")
    print("loss_curve:", np.round(diagonal_loss, 4))
    print("accuracy_curve:", np.round(diagonal_accuracy, 4))
    print("forgetting:", np.round(forgetting, 4))
    print(f"mean_final_loss={diagonal_loss.mean():.4f}")
    print(f"mean_final_accuracy={diagonal_accuracy.mean():.4f}")


if __name__ == "__main__":
    main()
