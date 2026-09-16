"""SplitMNIST demo for anonymous routing from learned classifier weights."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.templates import SupervisedTemplate

from skill_memory import PersistentFingerprintSkillMemoryPlugin, SkillMemory


class Tee:
    """Write output to several streams at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


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
    """Evaluate seen experiences; labels are used only for metrics."""
    results = strategy.eval([test_stream[i] for i in range(up_to_index + 1)])
    keys = sorted(key for key in results if key.startswith("Top1_Acc_Exp"))
    return [float(results[key]) for key in keys]


def flatten_route(route: dict, train_index: int) -> dict:
    """Convert one nested routing record into a compact analysis row."""
    candidates = route.get("candidates", [])
    top = candidates[0] if candidates else {}
    second = candidates[1] if len(candidates) > 1 else {}
    evaluation_y = route.get("evaluation_y")
    return {
        "training_step": train_index,
        "batch_index": route.get("batch_index", -1),
        "sample_index": route.get("sample_index", -1),
        "evaluation_y": evaluation_y,
        "status": route.get("status"),
        "inferred_class": route.get("class"),
        "selected_skill": route.get("skill"),
        "top_candidate_class": top.get("class"),
        "top_candidate_skill": top.get("skill"),
        "top_evidence": top.get("evidence"),
        "top_binary_compatible": top.get("binary_compatible"),
        "top_feature_similarity": top.get("feature_similarity"),
        "top_margin_similarity": top.get("margin_similarity"),
        "top_margin": top.get("margin"),
        "second_candidate_class": second.get("class"),
        "second_candidate_skill": second.get("skill"),
        "second_evidence": second.get("evidence"),
        "reference_accuracy": top.get("reference_accuracy"),
    }


def write_analysis_files(log_dir: Path, run_id: str, rows: list[dict]) -> None:
    """Write compact CSV and JSON routing-analysis artifacts."""
    csv_path = log_dir / f"weight_reverse_engineering_{run_id}.csv"
    json_path = log_dir / f"weight_reverse_engineering_{run_id}.json"

    fieldnames = [
        "training_step",
        "batch_index",
        "sample_index",
        "evaluation_y",
        "status",
        "inferred_class",
        "selected_skill",
        "top_candidate_class",
        "top_candidate_skill",
        "top_evidence",
        "top_binary_compatible",
        "top_feature_similarity",
        "top_margin_similarity",
        "top_margin",
        "second_candidate_class",
        "second_candidate_skill",
        "second_evidence",
        "reference_accuracy",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("IDENTIFIED", "AMBIGUOUS", "FAILED")
        },
        "identified_class_accuracy": (
            sum(
                row["status"] == "IDENTIFIED"
                and row["inferred_class"] == row["evaluation_y"]
                for row in rows
            )
            / max(sum(row["status"] == "IDENTIFIED" for row in rows), 1)
        ),
        "analysis_csv": csv_path.name,
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print("Analysis CSV saved to:", csv_path)
    print("Analysis JSON saved to:", json_path)


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"weight_reverse_engineering_{run_id}.txt"
    log_file = open(log_path, "w")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log_file)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        benchmark = SplitMNIST(n_experiences=10, seed=0)
        train_stream = benchmark.train_stream
        test_stream = benchmark.test_stream

        print("Device:", device)
        print("Routing: persistent class fingerprint -> learned weights -> class -> skill")
        print("No experience ID or target class is supplied to routing.")

        model = SkillMemoryMLP(input_dim=784).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        plugin = PersistentFingerprintSkillMemoryPlugin(
            memory=SkillMemory(max_skills=10),
            forgetting_margin=0.05,
            probe_batch_size=10,
            probe_batches=5,
            probe_seed=0,
            class_train_epochs=1,
            class_train_batch_size=64,
            reuse_is_mutable=True,
            eval_routing="probe",
            verbose=True,
        )
        strategy = SupervisedTemplate(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=64,
            train_epochs=1,
            eval_mb_size=64,
            device=device,
            plugins=[plugin],
        )

        accuracy_history: list[list[float]] = []
        analysis_rows: list[dict] = []
        for train_index, train_exp in enumerate(train_stream):
            strategy.train(train_exp)
            accuracies = evaluate_seen(strategy, test_stream, train_index)
            accuracy_history.append(accuracies)
            analysis_rows.extend(
                flatten_route(route, train_index)
                for route in plugin.fingerprint_route_history
            )
            routes = plugin.last_fingerprint_routes
            identified = sum(r["status"] == "IDENTIFIED" for r in routes)
            ambiguous = sum(r["status"] == "AMBIGUOUS" for r in routes)
            failed = sum(r["status"] == "FAILED" for r in routes)
            print(
                f"Step {train_index}: classes="
                f"{sorted(train_exp.classes_in_this_experience)} "
                f"mean_seen_accuracy={np.mean(accuracies):.3f} "
                f"last_batch_routes=(identified={identified}, "
                f"ambiguous={ambiguous}, failed={failed})"
            )

        n = len(accuracy_history)
        accuracy_curve = np.array([accuracy_history[i][i] for i in range(n)])
        forgetting = np.zeros(n)
        for class_index in range(n):
            seen = [row[class_index] for row in accuracy_history[class_index:]]
            if len(seen) > 1:
                forgetting[class_index] = max(seen[:-1]) - seen[-1]

        print("Accuracy:", np.round(accuracy_curve, 3))
        print("Forgetting:", np.round(forgetting, 3))
        print("Final fingerprint records:", len(plugin.behavior.state_dict()["records"]))
        write_analysis_files(log_dir, run_id, analysis_rows)
        print("Log saved to:", log_path)
    finally:
        sys.stdout = real_stdout
        log_file.close()


if __name__ == "__main__":
    main()
