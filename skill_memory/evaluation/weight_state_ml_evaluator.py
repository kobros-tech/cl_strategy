"""Anonymous two-stage ML evaluation from learned Skill Memory states."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class WeightStateSnapshot:
    """One post-update omega paired with the batch that produced it."""

    inputs: Tensor
    state: dict[str, Tensor]
    targets: Tensor
    class_id: int
    skill_id: int
    step: int


class WeightEvaluationMemory:
    """Bounded trajectory memory retaining x <-> omega <-> y records."""

    def __init__(self, max_snapshots_per_class: int = 10) -> None:
        if max_snapshots_per_class < 1:
            raise ValueError("max_snapshots_per_class must be positive")
        self.max_snapshots_per_class = int(max_snapshots_per_class)
        self._by_class: dict[int, deque[WeightStateSnapshot]] = {}
        self._next_step = 0

    @staticmethod
    def _clone_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {key: value.detach().cpu().clone() for key, value in state.items()}

    @staticmethod
    def _clone_batch(value: Tensor) -> Tensor:
        return value.detach().cpu().clone()

    def add(
        self,
        state: Mapping[str, Tensor],
        *,
        inputs: Tensor,
        targets: Tensor,
        class_id: int,
        skill_id: int,
        step: int | None = None,
    ) -> None:
        """Store an immutable post-step trajectory record."""
        inputs = self._clone_batch(inputs)
        targets = self._clone_batch(targets).to(dtype=torch.long)
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("inputs and targets must have the same batch size")

        class_id = int(class_id)
        if step is None:
            step = self._next_step
        step = int(step)
        if step < 0:
            raise ValueError("step must be non-negative")
        snapshot = WeightStateSnapshot(
            inputs=inputs,
            state=self._clone_state(state),
            targets=targets,
            class_id=class_id,
            skill_id=int(skill_id),
            step=step,
        )
        self._next_step = max(self._next_step, step + 1)
        bucket = self._by_class.setdefault(
            class_id, deque(maxlen=self.max_snapshots_per_class)
        )
        bucket.append(snapshot)

    def snapshots(self) -> list[WeightStateSnapshot]:
        snapshots = [
            snapshot
            for class_snapshots in self._by_class.values()
            for snapshot in class_snapshots
        ]
        return sorted(snapshots, key=lambda item: item.step)

    def snapshots_for_class(self, class_id: int) -> list[WeightStateSnapshot]:
        return list(self._by_class.get(int(class_id), ()))

    def classes(self) -> set[int]:
        return set(self._by_class)

    def __len__(self) -> int:
        return sum(len(items) for items in self._by_class.values())


def flatten_weight_state(state: Mapping[str, Tensor]) -> Tensor:
    """Flatten a complete model state using deterministic key ordering."""
    pieces = [
        value.detach().cpu().reshape(-1).to(dtype=torch.float32)
        for _, value in sorted(state.items())
    ]
    if not pieces:
        raise ValueError("cannot flatten an empty model state")
    return torch.cat(pieces)


def _flatten_inputs(inputs: Tensor) -> Tensor:
    """Flatten each input sample into a float feature vector."""
    return inputs.reshape(inputs.shape[0], -1).to(dtype=torch.float32)


def sketch_weight_state(
    state: Mapping[str, Tensor],
    representation_size: int,
    *,
    seed: int = 0,
) -> Tensor:
    """Map a complete model state to a bounded deterministic state sketch."""
    if representation_size < 1:
        raise ValueError("representation_size must be positive")
    flat = flatten_weight_state(state)
    indices = torch.arange(flat.numel(), dtype=torch.long)
    buckets = (indices * 1_000_003 + int(seed)) % representation_size
    signs = torch.where(
        ((indices * 9_176 + int(seed)) % 2) == 0,
        torch.ones_like(flat),
        -torch.ones_like(flat),
    )
    sketch = torch.zeros(representation_size, dtype=torch.float32)
    sketch.scatter_add_(0, buckets, flat * signs)
    counts = torch.zeros(representation_size, dtype=torch.float32)
    counts.scatter_add_(0, buckets, torch.ones_like(flat))
    return sketch / counts.clamp_min(1.0).sqrt()


def consolidate_weight_state_memory(
    memory: WeightEvaluationMemory,
    *,
    representation_size: int = 128,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build ML-1 pairs and ML-2 pairs from retained trajectory records."""
    snapshots = memory.snapshots()
    if not snapshots:
        raise RuntimeError("cannot consolidate empty weight evaluation memory")

    ml1_inputs: list[Tensor] = []
    ml1_targets: list[Tensor] = []
    ml2_inputs: list[Tensor] = []
    ml2_targets: list[int] = []

    for snapshot in snapshots:
        omega = sketch_weight_state(
            snapshot.state,
            representation_size,
            seed=seed,
        )
        batch_x = _flatten_inputs(snapshot.inputs)
        ml1_inputs.append(batch_x)
        ml1_targets.append(omega.unsqueeze(0).expand(batch_x.shape[0], -1))
        ml2_inputs.append(omega)
        ml2_targets.append(snapshot.class_id)

    return (
        torch.cat(ml1_inputs, dim=0),
        torch.cat(ml1_targets, dim=0),
        torch.stack(ml2_inputs),
        torch.tensor(ml2_targets, dtype=torch.long),
    )


def consolidate_weight_evaluation_memory(
    memory: WeightEvaluationMemory,
    *,
    representation_size: int = 128,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Build ML-2 training pairs from true stored omega states."""
    _, _, inputs, targets = consolidate_weight_state_memory(
        memory,
        representation_size=representation_size,
        seed=seed,
    )
    return inputs, targets


def build_weight_state_regressor(
    input_size: int,
    output_size: int,
    *,
    hidden_size: int = 128,
) -> nn.Module:
    if min(input_size, output_size, hidden_size) < 1:
        raise ValueError("model dimensions must be positive")
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


def build_weight_state_evaluator(
    input_size: int,
    num_classes: int,
    *,
    hidden_size: int = 128,
) -> nn.Module:
    """Build the ML-2 omega -> class classifier."""
    if min(input_size, num_classes, hidden_size) < 1:
        raise ValueError("model dimensions must be positive")
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes),
    )


def train_weight_state_regressor(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    memory: WeightEvaluationMemory,
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int = 0,
    representation_size: int = 128,
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if epochs < 1:
        return
    inputs, targets, _, _ = consolidate_weight_state_memory(
        memory,
        representation_size=representation_size,
        seed=seed,
    )
    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=min(batch_size, len(targets)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model.to(device).train()
    with torch.enable_grad():
        for _ in range(epochs):
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_inputs.to(device))
                loss = nn.functional.mse_loss(prediction, batch_targets.to(device))
                loss.backward()
                optimizer.step()


def train_weight_state_evaluator(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    memory: WeightEvaluationMemory,
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int = 0,
    representation_size: int = 128,
) -> None:
    """Train the ML-2 omega -> class evaluator."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if epochs < 1:
        return
    inputs, targets = consolidate_weight_evaluation_memory(
        memory,
        representation_size=representation_size,
        seed=seed,
    )
    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=min(batch_size, len(targets)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model.to(device).train()
    with torch.enable_grad():
        for _ in range(epochs):
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_inputs.to(device))
                loss = criterion(logits, batch_targets.to(device))
                loss.backward()
                optimizer.step()


@torch.no_grad()
def evaluate_weight_state_memory(
    model: nn.Module,
    memory: WeightEvaluationMemory,
    *,
    device: torch.device,
    representation_size: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    inputs, targets = consolidate_weight_evaluation_memory(
        memory,
        representation_size=representation_size,
        seed=seed,
    )
    model.to(device).eval()
    logits = model(inputs.to(device))
    targets = targets.to(device)
    if logits.shape[1] <= int(targets.max().item()):
        raise RuntimeError("Evaluator output space does not contain all class IDs.")
    loss = nn.functional.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return {"loss": float(loss.item()), "accuracy": float(accuracy.item())}


class WeightStateMLEvaluation:
    """Independent ML-1 + ML-2 anonymous evaluator."""

    def __init__(
        self,
        *,
        memory: WeightEvaluationMemory,
        num_classes: int,
        ml2_model_factory: Callable[[int, int], nn.Module] | None = None,
        ml1_model_factory: Callable[[int, int], nn.Module] | None = None,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        ml1_learning_rate: float = 0.001,
        ml1_epochs: int | None = None,
        ml1_batch_size: int | None = None,
        seed: int = 0,
        hidden_size: int = 128,
        state_representation_size: int = 128,
        verbose: bool = True,
    ) -> None:
        if epochs < 1 or (ml1_epochs is not None and ml1_epochs < 1):
            raise ValueError("epochs must be at least 1")
        if batch_size < 1 or (ml1_batch_size is not None and ml1_batch_size < 1):
            raise ValueError("batch_size must be positive")
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.memory = memory
        self.num_classes = int(num_classes)
        self.ml2_model_factory = ml2_model_factory
        self.ml1_model_factory = ml1_model_factory
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.ml1_learning_rate = float(ml1_learning_rate)
        self.ml1_epochs = int(ml1_epochs or epochs)
        self.ml1_batch_size = int(ml1_batch_size or batch_size)
        self.seed = int(seed)
        self.hidden_size = int(hidden_size)
        if state_representation_size < 1:
            raise ValueError("state_representation_size must be positive")
        self.state_representation_size = int(state_representation_size)
        self.verbose = verbose
        self.ml1_model: nn.Module | None = None
        self.ml2_model: nn.Module | None = None
        self.ml1_optimizer: torch.optim.Optimizer | None = None
        self.ml2_optimizer: torch.optim.Optimizer | None = None
        self.criterion = nn.CrossEntropyLoss()
        self._last_result: dict[str, object] = {}

    def train(self, *, device: torch.device, num_classes: int) -> dict[str, object]:
        ml1_x, ml1_y, _, _ = consolidate_weight_state_memory(
            self.memory,
            representation_size=self.state_representation_size,
            seed=self.seed,
        )
        input_size = ml1_x.shape[1]
        omega_size = ml1_y.shape[1]
        if self.ml1_model_factory is None:
            self.ml1_model = build_weight_state_regressor(
                input_size, omega_size, hidden_size=self.hidden_size
            )
        else:
            self.ml1_model = self.ml1_model_factory(input_size, omega_size)
        self.ml1_optimizer = torch.optim.SGD(
            self.ml1_model.parameters(), lr=self.ml1_learning_rate
        )
        train_weight_state_regressor(
            self.ml1_model,
            self.ml1_optimizer,
            self.memory,
            batch_size=self.ml1_batch_size,
            epochs=self.ml1_epochs,
            device=device,
            seed=self.seed,
            representation_size=self.state_representation_size,
        )

        if self.ml2_model_factory is None:
            self.ml2_model = build_weight_state_evaluator(
                omega_size,
                num_classes,
                hidden_size=self.hidden_size,
            )
        else:
            self.ml2_model = self.ml2_model_factory(omega_size, num_classes)

        self.ml2_optimizer = torch.optim.SGD(
            self.ml2_model.parameters(),
            lr=self.learning_rate,
        )
        train_weight_state_evaluator(
            self.ml2_model,
            self.ml2_optimizer,
            self.criterion,
            self.memory,
            batch_size=self.batch_size,
            epochs=self.epochs,
            device=device,
            seed=self.seed,
            representation_size=self.state_representation_size,
        )
        self._last_result = self.evaluate(device=device)
        return dict(self._last_result)

    @torch.no_grad()
    def evaluate(self, *, device: torch.device) -> dict[str, object]:
        if self.ml1_model is None or self.ml2_model is None:
            return {}

        ml1_x, ml1_targets, omega, targets = consolidate_weight_state_memory(
            self.memory,
            representation_size=self.state_representation_size,
            seed=self.seed,
        )
        self.ml1_model.to(device).eval()
        self.ml2_model.to(device).eval()

        predicted_omega = self.ml1_model(ml1_x.to(device))

        true_logits = self.ml2_model(omega.to(device))
        true_targets = targets.to(device)
        true_accuracy = (true_logits.argmax(1) == true_targets).float().mean()

        sample_targets = torch.cat(
            [
                torch.full(
                    (snapshot.inputs.shape[0],),
                    snapshot.class_id,
                    dtype=torch.long,
                )
                for snapshot in self.memory.snapshots()
            ]
        ).to(device)

        predicted_logits = self.ml2_model(predicted_omega)
        end_accuracy = (predicted_logits.argmax(1) == sample_targets).float().mean()

        return {
            "true_omega_accuracy": float(true_accuracy.item()),
            "end_to_end_accuracy": float(end_accuracy.item()),
            "omega_mse": float(
                nn.functional.mse_loss(
                    predicted_omega,
                    ml1_targets.to(device),
                ).item()
            ),
        }

    @property
    def current_result(self) -> dict[str, object]:
        return dict(self._last_result)


class WeightStateMLEvaluationPlugin(WeightStateMLEvaluation, SupervisedPlugin):
    """Avalanche plugin implementing x -> ML-1 -> omega -> ML-2 -> y."""

    def __init__(self, **kwargs) -> None:
        WeightStateMLEvaluation.__init__(self, **kwargs)
        SupervisedPlugin.__init__(self)
        self._active = False
        self._class_loss: dict[int, float] = {}
        self._class_correct: dict[int, int] = {}
        self._class_total: dict[int, int] = {}

    def before_eval(self, strategy, **kwargs) -> None:
        if not self.memory:
            self._active = False
            return
        self.train(device=strategy.device, num_classes=self.num_classes)
        self._active = True
        self._class_loss = {}
        self._class_correct = {}
        self._class_total = {}

    @torch.no_grad()
    def after_eval_forward(self, strategy, **kwargs) -> None:
        if not self._active or self.ml1_model is None or self.ml2_model is None:
            return

        inputs = _flatten_inputs(strategy.mbatch[0]).to(strategy.device)

        self.ml1_model.eval()
        self.ml2_model.eval()

        predicted_omega = self.ml1_model(inputs)
        strategy.mb_output = self.ml2_model(predicted_omega)

    def after_eval_iteration(self, strategy, **kwargs) -> None:
        if not self._active:
            return
        outputs = strategy.mb_output
        targets = strategy.mbatch[1]
        predictions = outputs.argmax(dim=1)
        losses = nn.functional.cross_entropy(outputs, targets, reduction="none")
        for class_id in torch.unique(targets).tolist():
            class_id = int(class_id)
            mask = targets == class_id
            self._class_loss[class_id] = self._class_loss.get(class_id, 0.0) + float(
                losses[mask].sum().item()
            )
            self._class_correct[class_id] = self._class_correct.get(class_id, 0) + int(
                (predictions[mask] == targets[mask]).sum().item()
            )
            self._class_total[class_id] = self._class_total.get(class_id, 0) + int(
                mask.sum().item()
            )

    def after_eval(self, strategy, **kwargs) -> None:
        if not self._active:
            return
        accuracy = {
            class_id: self._class_correct[class_id] / self._class_total[class_id]
            for class_id in self._class_total
        }
        loss = {
            class_id: self._class_loss[class_id] / self._class_total[class_id]
            for class_id in self._class_total
        }
        self._last_result.update(
            {
                "final_class_accuracy": accuracy,
                "final_class_loss": loss,
                "mean_final_accuracy": float(np.mean(list(accuracy.values()))),
                "mean_final_loss": float(np.mean(list(loss.values()))),
            }
        )
        self._active = False
