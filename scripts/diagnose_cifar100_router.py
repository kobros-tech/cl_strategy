"""Diagnostic Split-CIFAR100 run for anonymous Skill Memory routing.

This is intentionally a diagnostic, not a benchmark implementation. It records
routing correctness separately from final prediction correctness so a failure
can be localized to routing or to the selected frozen skill.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.training.supervised import Naive
from torch import nn
from torch.utils.data import Subset
from torchvision import datasets, models, transforms

from skill_memory import PersistentFingerprintSkillMemoryPlugin


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", default="cifar100_router_diagnostic.json")
    parser.add_argument("--n-experiences", type=int, default=4)
    parser.add_argument("--train-samples-per-class", type=int, default=100)
    parser.add_argument("--eval-samples-per-class", type=int, default=100)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument("--reverse-epochs", type=int, default=30)
    parser.add_argument("--reverse-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _limit_per_class(dataset, samples_per_class: int):
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    selected = []
    counts = [0] * 100
    for index in range(len(dataset)):
        class_id = int(dataset[index][1])
        if counts[class_id] >= samples_per_class:
            continue
        selected.append(index)
        counts[class_id] += 1
        if all(count >= samples_per_class for count in counts):
            break
    return Subset(dataset, selected)


def _make_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


def _run_eval_diagnostics(plugin, strategy, experience_count: int) -> dict:
    routes = list(plugin.fingerprint_route_history)
    total = len(routes)
    identified = sum(route["status"] == "IDENTIFIED" for route in routes)
    route_correct = sum(
        route["class"] == route["evaluation_y"] for route in routes
    )
    prediction_correct = sum(
        route.get("model_correct", False) for route in routes
    )

    by_experience = {}
    for experience_id in range(experience_count):
        rows = [
            route
            for route in routes
            if route.get("evaluation_experience") == experience_id
        ]
        by_experience[str(experience_id)] = {
            "samples": len(rows),
            "routing_accuracy": (
                sum(row["class"] == row["evaluation_y"] for row in rows)
                / len(rows)
                if rows
                else None
            ),
            "prediction_accuracy": (
                sum(row.get("model_correct", False) for row in rows) / len(rows)
                if rows
                else None
            ),
            "skill_counts": {
                str(skill): sum(row["skill"] == skill for row in rows)
                for skill in sorted({row["skill"] for row in rows})
            },
        }

    return {
        "samples": total,
        "identified": identified,
        "routing_accuracy": route_correct / total if total else None,
        "prediction_accuracy": prediction_correct / total if total else None,
        "by_experience": by_experience,
    }


def main() -> None:
    args = _parse_args()
    _seed_everything(args.seed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    train_set = datasets.CIFAR100(
        root=args.data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.CIFAR100(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform,
    )

    train_subset = _limit_per_class(train_set, args.train_samples_per_class)
    test_subset = _limit_per_class(test_set, args.eval_samples_per_class)

    class_order = list(range(100))
    scenario = nc_benchmark(
        train_dataset=train_subset,
        test_dataset=test_subset,
        n_experiences=args.n_experiences,
        task_labels=False,
        shuffle=False,
        seed=args.seed,
        fixed_class_order=class_order,
    )

    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    plugin = PersistentFingerprintSkillMemoryPlugin(
        eval_routing="probe",
        probe_batch_size=args.probe_batch_size,
        probe_batches=args.probe_batches,
        reverse_epochs=args.reverse_epochs,
        reverse_batch_size=args.reverse_batch_size,
        class_train_epochs=args.train_epochs,
        class_train_batch_size=32,
        verbose=True,
    )
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=32,
        train_epochs=1,
        eval_mb_size=64,
        device=args.device,
        plugins=[plugin],
    )

    for experience in scenario.train_stream:
        strategy.train(experience)

    strategy.eval(scenario.test_stream)
    diagnostics = _run_eval_diagnostics(plugin, strategy, args.n_experiences)

    result = {
        "seed": args.seed,
        "n_experiences": args.n_experiences,
        "class_order": class_order,
        "routing": diagnostics,
        "class_skill_assignments": {
            str(experience_id): {
                str(class_id): int(skill)
                for class_id, skill in plugin.class_map.class_skill_for_experience(
                    experience_id
                ).items()
            }
            for experience_id in range(args.n_experiences)
        },
        "training_decisions": {
            str(experience_id): plugin.last_class_decisions.get(experience_id, {})
            for experience_id in range(args.n_experiences)
        },
        "routes": plugin.fingerprint_route_history,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
