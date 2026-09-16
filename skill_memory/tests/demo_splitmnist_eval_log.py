# # Skill-memory continual learning demo (v0.1.4 package) -- eval log edition
#
# Deliberately named `demo_*.py`, not `test_*.py`: this is a runnable demo
# script (real SplitMNIST training, network access to download MNIST, no
# `test_` functions), not a pytest unit test, so it must NOT match pytest's
# default `test_*.py` / `*_test.py` collection pattern. Two independent
# reasons:
#   1. pytest would try to import and execute it during `pytest -q`,
#      running a full real training loop instead of a fast unit test.
#   2. Several of the package's real unit tests (test_plugin_eval.py,
#      test_probing.py, test_persistent_fingerprints.py) register a stub
#      `avalanche` module into `sys.modules` for isolated testing. Once
#      pytest has imported one of those in the same process, a later
#      `from avalanche... import ...` in this file would resolve against
#      that stub instead of the real installed package and fail with
#      "'avalanche' is not a package". Running this file on its own
#      (`python demo_splitmnist_eval_log.py`) is unaffected either way.
#
# This plays the same role as `learn_CL_3.ipynb` / the "refined" prototype
# you pasted, but rewritten against the *actual* `skill_memory` v0.1.4
# package (a real Avalanche plugin, not the `skill_memory2` module the
# prototype imported from -- see the previous version of this file for that
# mapping).
#
# The difference from the previous version of this script: instead of only
# reporting aggregate per-experience accuracy, a `PredictionLogger` plugin
# captures the actual routed prediction (predicted y) against the ground
# truth label (real y) for every evaluated sample, so you can inspect
# individual right/wrong calls rather than just an experience-id-level
# accuracy number.
#
# It assumes `train_stream`, `test_stream`, and `device` already exist from
# your Avalanche benchmark setup earlier in the notebook -- paste those
# setup cells above this one unchanged, or just run this script standalone;
# it builds its own SplitMNIST stream below if you haven't.
#
# NOTE on import path: this file assumes the v0.1.4 package has been
# installed/placed importable as `skill_memory` (e.g. `pip install -e .`
# from inside the unzipped `v0.1.4/` directory after adding a `pyproject
# .toml`, or by simply renaming the folder). Adjust the import line below
# if your project exposes it under a different name.

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate

from skill_memory import SkillMemory, SkillMemoryPlugin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


########################################################
# Model
########################################################
# skill_memory's probing/decision code (probing.py) resizes and reloads
# `IncrementalClassifier` heads when it swaps skill state dicts in and out
# during REUSE/SCRATCH, and grows the head on new classes via Avalanche's
# `avalanche_model_adaptation`. The classifier head therefore MUST be an
# `IncrementalClassifier`, not a plain `nn.Linear`.
class SkillMemoryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.classifier = IncrementalClassifier(hidden_size, initial_out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(x.size(0), -1)
        x = self.features(x)
        return self.classifier(x)


########################################################
# Prediction logger -- captures predicted y vs. real y per sample
########################################################
class PredictionLogger(SupervisedPlugin):
    """Logs predicted vs. real labels for every evaluated sample.

    Must appear AFTER `SkillMemoryPlugin` in `plugins=[...]`. Avalanche
    calls each hook in plugin-list order, and `SkillMemoryPlugin`'s own
    `after_eval_forward` is what overwrites `strategy.mb_output` with the
    routed (probe-selected) logits -- this plugin's `after_eval_forward`
    needs to fire afterwards so it reads that routed prediction rather than
    a single skill's raw, unrouted logits.
    """

    def __init__(self):
        super().__init__()
        self.records: list[dict] = []
        self._train_step: int | None = None  # which training step this eval is under
        self._exp_id: int | None = None

    def set_train_step(self, step: int) -> None:
        """Call before each `strategy.eval(...)` so records know which
        training step (t) they were logged under -- the same experience
        gets re-evaluated after every later training step, so this is what
        distinguishes those repeated passes in the log."""
        self._train_step = step

    def before_eval_exp(self, strategy, **kwargs) -> None:
        experience = strategy.experience
        self._exp_id = getattr(experience, "current_experience", None)

    def after_eval_forward(self, strategy, **kwargs) -> None:
        y_true = strategy.mbatch[1].detach().cpu()
        logits = strategy.mb_output.detach().cpu()
        probs = torch.softmax(logits, dim=1)
        y_pred = logits.argmax(dim=1)
        confidence = probs.gather(1, y_pred.unsqueeze(1)).squeeze(1)

        for i in range(len(y_true)):
            self.records.append(
                {
                    "train_step": self._train_step,
                    "experience": self._exp_id,
                    "true_y": int(y_true[i]),
                    "pred_y": int(y_pred[i]),
                    "confidence": float(confidence[i]),
                    "correct": bool(y_true[i] == y_pred[i]),
                }
            )

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def print_log(
        self,
        train_step: int | None = None,
        only_errors: bool = False,
        max_rows: int | None = 40,
    ) -> None:
        """Print true-y vs. predicted-y rows, optionally filtered/truncated."""
        rows = self.records
        if train_step is not None:
            rows = [r for r in rows if r["train_step"] == train_step]
        if only_errors:
            rows = [r for r in rows if not r["correct"]]
        shown = rows[:max_rows] if max_rows else rows

        header = f"{'step':>4} {'exp':>4} {'true_y':>7} {'pred_y':>7} {'conf':>6}  ok"
        print(header)
        print("-" * len(header))
        for r in shown:
            mark = "OK" if r["correct"] else "X"
            print(
                f"{r['train_step']:>4} {r['experience']:>4} "
                f"{r['true_y']:>7} {r['pred_y']:>7} {r['confidence']:>6.3f}  {mark}"
            )
        if max_rows and len(rows) > max_rows:
            print(f"... ({len(rows) - max_rows} more rows not shown)")
        if not rows:
            print("(no rows match this filter)")

    def to_csv(self, path: str) -> None:
        """Dump the full per-sample log (every row, no truncation) to CSV."""
        fieldnames = [
            "train_step",
            "experience",
            "true_y",
            "pred_y",
            "confidence",
            "correct",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"Wrote {len(self.records)} eval rows to {path}")


########################################################
# Metrics helpers (per-experience accuracy; not shipped by the package)
########################################################
def evaluate_seen_experiences(
    strategy,
    test_stream,
    up_to_index: int,
    pred_logger: PredictionLogger,
    train_step: int,
) -> list[float]:
    """Per-experience test accuracy for experiences 0..up_to_index.

    Evaluates one experience at a time so each accuracy is unambiguously
    attributable to a single experience, and so `pred_logger` tags every
    logged sample with the right experience id.
    """
    accuracies = []
    pred_logger.set_train_step(train_step)
    for i in range(up_to_index + 1):
        results = strategy.eval([test_stream[i]])
        acc_keys = [k for k in results if k.startswith("Top1_Acc_Exp")]
        if not acc_keys:
            raise RuntimeError(
                "No 'Top1_Acc_Exp' metric found -- is the default "
                "EvaluationPlugin (accuracy_metrics(experience=True)) still "
                "attached to the strategy?"
            )
        accuracies.append(float(results[acc_keys[0]]))
    return accuracies


def compute_cl_metrics(accuracy_history: list[list[float]]):
    """Build accuracy/forgetting curves from a ragged per-step accuracy history."""
    n = len(accuracy_history)
    matrix = np.full((n, n), np.nan)
    for t, row in enumerate(accuracy_history):
        matrix[t, : len(row)] = row

    accuracy_curve = np.array([matrix[t, t] for t in range(n)])

    forgetting_curve = np.zeros(n)
    for i in range(n):
        seen = matrix[i:, i]
        seen = seen[~np.isnan(seen)]
        if len(seen) > 1:
            forgetting_curve[i] = np.max(seen[:-1]) - seen[-1]

    return accuracy_curve, forgetting_curve


########################################################
# Benchmark (skip this cell if train_stream/test_stream already exist)
########################################################
benchmark = SplitMNIST(n_experiences=10, seed=0)

train_stream = benchmark.train_stream
test_stream = benchmark.test_stream

print("Number of experiences:", len(train_stream))
for i, exp in enumerate(train_stream):
    print(i, sorted(exp.classes_in_this_experience), len(exp.dataset))


########################################################
# Strategy
########################################################
model = SkillMemoryMLP(input_dim=784).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

skill_plugin = SkillMemoryPlugin(
    memory=SkillMemory(max_skills=10),
    forgetting_margin=0.05,
    probe_batch_size=10,
    probe_batches=5,  # was implicitly 1 batch of 10; now 5 (~50 samples)
    probe_seed=0,  # reproducible probe sampling across runs
    class_train_epochs=1,
    class_train_batch_size=64,
    reuse_is_mutable=True,  # flip to False to A/B against a frozen-skill version
    eval_routing="probe",  # task-free evaluation; see package README
    verbose=True,
)
pred_logger = PredictionLogger()

strategy = SupervisedTemplate(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=64,
    train_epochs=1,
    eval_mb_size=64,
    device=device,
    # pred_logger MUST come after skill_plugin -- see PredictionLogger docstring.
    plugins=[skill_plugin, pred_logger],
)

accuracy_history: list[list[float]] = []

for t, train_exp in enumerate(train_stream):
    strategy.train(train_exp)

    current_accuracies = evaluate_seen_experiences(
        strategy, test_stream, t, pred_logger, train_step=t
    )
    accuracy_history.append(current_accuracies)

    print(
        f"Experience {t}: "
        f"trained on {sorted(train_exp.classes_in_this_experience)}, "
        f"mean seen accuracy = {np.mean(current_accuracies):.3f}"
    )

    # Per-sample eval log for this training step: predicted y vs. real y.
    print(f"\n--- eval log after training step {t} (predicted y vs. real y) ---")
    pred_logger.print_log(train_step=t, only_errors=False, max_rows=40)
    print()

accuracy_curve, forgetting_curve = compute_cl_metrics(accuracy_history)

print("\n")
print("Accuracy:", np.round(accuracy_curve, 3))
print("Forgetting:", np.round(forgetting_curve, 3))

# Full per-sample log (every evaluated sample across every training step),
# for offline inspection -- this is the complete predicted-vs-real record,
# unlike the truncated console printouts above.
log_dir = Path(__file__).resolve().parent / "logs"
log_dir.mkdir(exist_ok=True)
pred_logger.to_csv(str(log_dir / "skill_memory_eval_log.csv"))

# Quick look at only the mistakes from the very last (final) evaluation pass.
final_step = len(train_stream) - 1
print(f"\n--- misclassified samples at final training step ({final_step}) ---")
pred_logger.print_log(train_step=final_step, only_errors=True, max_rows=40)
