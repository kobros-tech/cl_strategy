"""High-level Skill Memory strategy with integrated ML evaluation.

This module provides the public orchestration layer for Skill Memory.

The strategy manages two learners:

1. Skill Memory:
   - class-level REUSE/SCRATCH decisions
   - skill allocation and storage
   - class-to-skill bookkeeping
   - optional oracle/probe diagnostics

2. Auxiliary ML evaluator:
   - receives frozen raw examples retained by Skill Memory
   - trains on all accumulated class memory
   - evaluates every seen class
   - tracks accuracy, loss, and forgetting

The low-level Skill Memory and evaluation components remain modular. This
class simply provides one entry point for applications such as ocl_survey.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from avalanche.training import Naive

from .cl.skill_memory_plugin import SkillMemoryPlugin
from .cl.skill_registry import SkillMemory
from .evaluation.ml_cl_evaluator import (
    EvaluationMemoryPlugin,
    compute_class_forgetting,
    compute_peak_class_forgetting,
    consolidate_evaluation_memory,
    evaluate_model_by_class,
    evaluate_skill_memory,
    train_evaluator,
)


class SkillMemoryStrategy:
    """High-level Skill Memory training and evaluation strategy.

    The class owns the underlying Avalanche strategy, Skill Memory plugin,
    evaluation memory, and auxiliary ML evaluator.

    Typical usage::

        strategy = SkillMemoryStrategy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )

        for experience in benchmark.train_stream:
            strategy.train(experience)

        results = strategy.results()

    The caller does not need to construct `Naive`, `SkillMemoryPlugin`,
    `EvaluationMemoryPlugin`, or the evaluator separately.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_stream: Iterable,
        test_stream: Iterable,
        max_skills: int = 200,
        forgetting_margin: float = 0.05,
        score_floor: float | None = 0.9,
        probe_batch_size: int = 64,
        probe_batches: int = 5,
        probe_seed: int | None = None,
        max_safety_candidates: int = 5,
        class_train_epochs: int = 1,
        class_train_batch_size: int = 64,
        reuse_is_mutable: bool = True,
        force_decision: str | None = None,
        skill_eval_routing: str = "none",
        eval_frequency: str = "every_experience",
        skill_eval_batch_size: int = 64,
        eval_memory_per_class: int = 20,
        eval_memory_seed: int = 0,
        eval_epochs: int = 1,
        eval_batch_size: int = 64,
        eval_learning_rate: float = 0.01,
        evaluator_model_factory: Callable[[], nn.Module],
        train_mb_size: int = 64,
        train_epochs: int = 1,
        eval_mb_size: int = 64,
        device: torch.device | str | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the complete Skill Memory + ML evaluation pipeline.

        Parameters controlling Skill Memory are passed to
        :class:`SkillMemoryPlugin`.

        The auxiliary evaluator is intentionally independent from the main
        Skill Memory model. By default it uses a fresh `SimpleMLP`, but a
        caller can provide `evaluator_model_factory` for another architecture.
        """
        if skill_eval_routing not in {
            "none",
            "oracle",
            "probe",
            "both",
        }:
            raise ValueError(
                "skill_eval_routing must be one of {'none', 'oracle', 'probe', 'both'}"
            )

        if eval_memory_per_class <= 0:
            raise ValueError("eval_memory_per_class must be positive")

        if eval_epochs < 1:
            raise ValueError("eval_epochs must be at least 1")

        if eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        if eval_frequency not in {"every_experience", "final", "none"}:
            raise ValueError(
                "evaluation_frequency must be one of "
                "{'every_experience', 'final', 'none'}"
            )

        self.eval_frequency = eval_frequency

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

        self.eval_epochs = eval_epochs
        self.eval_batch_size = eval_batch_size
        self.eval_learning_rate = eval_learning_rate
        self.skill_eval_routing = skill_eval_routing
        self.skill_eval_batch_size = skill_eval_batch_size
        self.verbose = verbose

        # ------------------------------------------------------------------
        # Skill Memory
        # ------------------------------------------------------------------

        self.memory = SkillMemory(max_skills=max_skills)

        self.plugin = EvaluationMemoryPlugin(
            memory=self.memory,
            max_skills=max_skills,
            forgetting_margin=forgetting_margin,
            score_floor=score_floor,
            probe_batch_size=probe_batch_size,
            probe_batches=probe_batches,
            probe_seed=probe_seed,
            max_safety_candidates=max_safety_candidates,
            class_train_epochs=class_train_epochs,
            class_train_batch_size=class_train_batch_size,
            reuse_is_mutable=reuse_is_mutable,
            force_decision=force_decision,
            eval_routing="none",
            eval_memory_per_class=eval_memory_per_class,
            eval_memory_seed=eval_memory_seed,
            verbose=verbose,
        )

        self.avalanche_strategy = Naive(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=self.device,
            plugins=[self.plugin],
        )

        # ------------------------------------------------------------------
        # Auxiliary ML evaluator
        # ------------------------------------------------------------------

        self.train_stream = train_stream
        self._test_stream = test_stream
        self.evaluator_model_factory = evaluator_model_factory

        self.evaluator = evaluator_model_factory().to(self.device)
        self.evaluator_optimizer = torch.optim.SGD(
            self.evaluator.parameters(),
            lr=eval_learning_rate,
        )
        self.evaluator_criterion = nn.CrossEntropyLoss()

        # ------------------------------------------------------------------
        # Experiment bookkeeping
        # ------------------------------------------------------------------

        self._class_to_experience: dict[int, int] = {}

        self._accuracy_history: list[dict[int, float]] = []
        self._loss_history: list[dict[int, float]] = []

        self._diagonal_accuracy_history: list[float] = []
        self._diagonal_loss_history: list[float] = []

        self._skill_accuracy_history: dict[str, list[dict[int, float]]] = {
            "oracle": [],
            "probe": [],
        }

        self._experience_count = 0

        self._last_evaluation: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, experience) -> None:
        """Train Skill Memory on one Avalanche experience."""
        experience_index = self._experience_count

        classes = sorted(
            int(class_id) for class_id in experience.classes_in_this_experience
        )

        for class_id in classes:
            if class_id in self._class_to_experience:
                continue
            self._class_to_experience[class_id] = experience_index

        if self.verbose:
            print()
            print(f"========== Training experience {experience_index} ==========")
            print(f"Classes: {classes}")

        # --------------------------------------------------------------
        # 1. Actual Skill Memory training
        # --------------------------------------------------------------

        self.avalanche_strategy.train(experience)
        self._experience_count += 1

        if self.eval_frequency == "every_experience":
            self._evaluate_after_experience(experience_index)

    # ------------------------------------------------------------------
    # Evaluation stream
    # ------------------------------------------------------------------

    def set_test_stream(self, test_stream: Iterable) -> None:
        """Set the test stream used by automatic evaluation.

        This is separated from ``train`` because the Avalanche training
        experience itself does not contain the complete test stream.
        """
        self._test_stream = test_stream

    # ------------------------------------------------------------------
    # Skill Memory evaluation
    # ------------------------------------------------------------------

    def _evaluate_skill_memory(self, experience_index: int) -> None:
        """Evaluate the actual stored Skill Memory when requested."""
        if self.skill_eval_routing == "none":
            return

        if not hasattr(self, "_test_stream"):
            raise RuntimeError(
                "set_test_stream() must be called before training when "
                "skill evaluation is enabled."
            )

        if self.skill_eval_routing == "both":
            routings = ("oracle", "probe")
        else:
            routings = (self.skill_eval_routing,)

        for routing in routings:
            class_results = evaluate_skill_memory(
                self.model,
                self.plugin,
                self._test_stream,
                experience_index,
                num_classes=self._num_classes(),
                routing=routing,
                batch_size=self.skill_eval_batch_size,
                device=self.device,
            )

            accuracy = {
                class_id: values["accuracy"]
                for class_id, values in class_results.items()
            }

            self._skill_accuracy_history[routing].append(accuracy)

    def _num_classes(self) -> int:
        """Return the global evaluator/output class count."""
        if hasattr(self.evaluator, "classifier"):
            classifier = self.evaluator.classifier
            if hasattr(classifier, "out_features"):
                return int(classifier.out_features)

        # Fall back to the largest class observed so far.
        if self._class_to_experience:
            return max(self._class_to_experience) + 1

        raise RuntimeError("Cannot determine the number of classes.")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _evaluate_after_experience(self, experience_index: int) -> None:
        """Run auxiliary and optional Skill Memory evaluation."""
        self._evaluate_skill_memory(experience_index)

        accumulated_memory = consolidate_evaluation_memory(self.plugin.eval_memory)

        if not accumulated_memory:
            raise RuntimeError("Evaluation memory is empty.")

        train_evaluator(
            self.evaluator,
            self.evaluator_optimizer,
            self.evaluator_criterion,
            accumulated_memory,
            batch_size=self.eval_batch_size,
            epochs=self.eval_epochs,
            device=self.device,
            seed=experience_index,
        )

        class_results = evaluate_model_by_class(
            self.evaluator,
            self._test_stream,
            experience_index,
            batch_size=self.eval_batch_size,
            device=self.device,
        )

        current_accuracy = {
            class_id: values["accuracy"] for class_id, values in class_results.items()
        }
        current_loss = {
            class_id: values["loss"] for class_id, values in class_results.items()
        }

        self._accuracy_history.append(current_accuracy)
        self._loss_history.append(current_loss)

        classes = [
            class_id
            for class_id, first_experience in self._class_to_experience.items()
            if first_experience <= experience_index
        ]

        diagonal_classes = [
            class_id
            for class_id in classes
            if self._class_to_experience[class_id] == experience_index
        ]

        if not diagonal_classes:
            raise RuntimeError(f"No classes found for experience {experience_index}.")

        missing_classes = [
            class_id
            for class_id in diagonal_classes
            if class_id not in current_accuracy
        ]

        if missing_classes:
            raise RuntimeError(
                f"Evaluation is missing classes {missing_classes} "
                f"for experience {experience_index}."
            )

        diagonal_accuracy = float(
            np.mean([current_accuracy[class_id] for class_id in diagonal_classes])
        )
        diagonal_loss = float(
            np.mean([current_loss[class_id] for class_id in diagonal_classes])
        )

        self._diagonal_accuracy_history.append(diagonal_accuracy)
        self._diagonal_loss_history.append(diagonal_loss)

        self._last_evaluation = {
            "class_results": class_results,
            "diagonal_accuracy": diagonal_accuracy,
            "diagonal_loss": diagonal_loss,
        }

        if self.verbose:
            print(f"  diagonal_accuracy={diagonal_accuracy:.4f}")
            print(f"  diagonal_loss={diagonal_loss:.4f}")

    def evaluate(self) -> dict[str, Any]:
        """Run final evaluation once after training."""
        if self._experience_count == 0:
            raise RuntimeError("No trained experiences are available for evaluation.")

        if self.eval_frequency == "every_experience":
            if self._last_evaluation is None:
                raise RuntimeError("No evaluation results are available.")
            return dict(self._last_evaluation["class_results"])

        if self.eval_frequency == "none":
            raise RuntimeError("Evaluation is disabled because eval_frequency='none'.")

        # Final-only evaluation.
        self._evaluate_after_experience(self._experience_count - 1)

        return dict(self._last_evaluation["class_results"])

    def results(self) -> dict[str, Any]:
        """Return the complete experiment metrics."""
        if not self._accuracy_history:
            raise RuntimeError(
                "No results are available. Train at least one experience "
                "and call evaluate()."
            )

        final_accuracy = self._accuracy_history[-1]
        final_loss = self._loss_history[-1]

        result: dict[str, Any] = {
            "final_class_accuracy": dict(final_accuracy),
            "final_class_loss": dict(final_loss),
            "mean_final_accuracy": float(np.mean(list(final_accuracy.values()))),
            "mean_final_loss": float(np.mean(list(final_loss.values()))),
        }

        if self.eval_frequency == "every_experience":
            acquisition_forgetting = compute_class_forgetting(
                self._accuracy_history,
                self._class_to_experience,
                self._experience_count,
            )

            peak_forgetting = compute_peak_class_forgetting(
                self._accuracy_history,
                self._class_to_experience,
                self._experience_count,
            )

            result.update(
                {
                    "diagonal_accuracy": np.asarray(
                        self._diagonal_accuracy_history,
                        dtype=np.float64,
                    ),
                    "diagonal_loss": np.asarray(
                        self._diagonal_loss_history,
                        dtype=np.float64,
                    ),
                    "class_forgetting_acquisition_relative": (acquisition_forgetting),
                    "class_forgetting_peak_relative": peak_forgetting,
                    "mean_diagonal_accuracy": float(
                        np.mean(self._diagonal_accuracy_history)
                    ),
                    "mean_diagonal_loss": float(np.mean(self._diagonal_loss_history)),
                    "mean_class_forgetting_acquisition_relative": float(
                        acquisition_forgetting.mean()
                    ),
                    "mean_class_forgetting_peak_relative": float(
                        peak_forgetting.mean()
                    ),
                }
            )
        else:
            result.update(
                {
                    "diagonal_accuracy": np.asarray(
                        self._diagonal_accuracy_history,
                        dtype=np.float64,
                    ),
                    "diagonal_loss": np.asarray(
                        self._diagonal_loss_history,
                        dtype=np.float64,
                    ),
                }
            )

        skill_results = self._skill_results()

        if skill_results:
            result["skill_memory"] = skill_results

        return result

    def _skill_results(self) -> dict[str, Any]:
        """Build metrics for direct Skill Memory evaluation."""
        results: dict[str, Any] = {}

        for routing, history in self._skill_accuracy_history.items():
            if not history:
                continue

            final = history[-1]

            forgetting = compute_peak_class_forgetting(
                history,
                self._class_to_experience,
                self._experience_count,
            )

            results[routing] = {
                "final_class_accuracy": dict(final),
                "mean_final_accuracy": float(np.mean(list(final.values()))),
                "peak_forgetting": forgetting,
                "mean_peak_forgetting": float(forgetting.mean()),
            }

        return results

    # ------------------------------------------------------------------
    # Convenient properties
    # ------------------------------------------------------------------

    @property
    def skill_memory(self) -> SkillMemory:
        """Return the underlying Skill Memory storage."""
        return self.memory

    @property
    def skill_memory_plugin(self) -> SkillMemoryPlugin:
        """Return the underlying Skill Memory plugin."""
        return self.plugin

    @property
    def evaluator_model(self) -> nn.Module:
        """Return the auxiliary ML evaluator."""
        return self.evaluator

    @property
    def current_accuracy(self) -> dict[int, float]:
        """Return the most recent per-class accuracy."""
        if not self._accuracy_history:
            return {}
        return dict(self._accuracy_history[-1])

    @property
    def current_loss(self) -> dict[int, float]:
        """Return the most recent per-class loss."""
        if not self._loss_history:
            return {}
        return dict(self._loss_history[-1])
