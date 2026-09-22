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

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from avalanche.training.templates import SupervisedTemplate

from .cl.skill_memory_plugin import SkillMemoryPlugin
from .cl.skill_registry import SkillMemory
from .evaluation.ml_cl_evaluator import (
    EvaluationMemoryPlugin,
    compute_peak_class_forgetting,
    consolidate_evaluation_memory,
    evaluate_model_by_class,
    evaluate_skill_memory,
    train_evaluator,
)


class SkillMemoryStrategy(SupervisedTemplate):
    """Avalanche strategy with integrated Skill Memory and ML evaluation."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
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
            device = next(model.parameters()).device
        else:
            device = torch.device(device)

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

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=[self.plugin],
        )

        # ------------------------------------------------------------------
        # Auxiliary ML evaluator
        # ------------------------------------------------------------------

        self.evaluator_model_factory = evaluator_model_factory

        self.ml_evaluator = evaluator_model_factory().to(self.device)
        self.evaluator_optimizer = torch.optim.SGD(
            self.ml_evaluator.parameters(),
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

        self._last_evaluation: dict[str, Any] | None = None

    def record_experience(self, experience, experience_index: int) -> None:
        """Record the first experience in which each class was introduced."""
        for class_id in experience.classes_in_this_experience:
            class_id = int(class_id)
            self._class_to_experience.setdefault(class_id, experience_index)

    # ------------------------------------------------------------------
    # Skill Memory evaluation
    # ------------------------------------------------------------------

    def _evaluate_skill_memory(
        self,
        test_stream,
        experience_index: int,
    ) -> None:
        """Evaluate the actual stored Skill Memory when requested."""
        if self.skill_eval_routing == "none":
            return

        if self.skill_eval_routing == "both":
            routings = ("oracle", "probe")
        else:
            routings = (self.skill_eval_routing,)

        for routing in routings:
            class_results = evaluate_skill_memory(
                self.model,
                self.plugin,
                test_stream,
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
        """Return the number of globally known classes."""
        if not self._class_to_experience:
            raise RuntimeError("Cannot determine the number of classes.")

        return max(self._class_to_experience) + 1

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def evaluate_ml(
        self,
        test_stream,
        experience_index: int,
    ) -> None:
        """Run auxiliary and optional Skill Memory evaluation."""
        self._evaluate_skill_memory(
            test_stream,
            experience_index,
        )

        accumulated_memory = consolidate_evaluation_memory(self.plugin.eval_memory)

        if not accumulated_memory:
            raise RuntimeError("Evaluation memory is empty.")

        train_evaluator(
            self.ml_evaluator,
            self.evaluator_optimizer,
            self.evaluator_criterion,
            accumulated_memory,
            batch_size=self.eval_batch_size,
            epochs=self.eval_epochs,
            device=self.device,
            seed=experience_index,
        )

        class_results = evaluate_model_by_class(
            self.ml_evaluator,
            test_stream,
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

    def results(self) -> dict[str, Any]:
        """Return the complete experiment metrics."""
        if not self._accuracy_history:
            raise RuntimeError(
                "No results are available. Train at least one experience "
                "and call evaluate_ml()."
            )

        final_accuracy = self._accuracy_history[-1]
        final_loss = self._loss_history[-1]

        result: dict[str, Any] = {
            "final_class_accuracy": dict(final_accuracy),
            "final_class_loss": dict(final_loss),
            "mean_final_accuracy": float(np.mean(list(final_accuracy.values()))),
            "mean_final_loss": float(np.mean(list(final_loss.values()))),
        }

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
                len(history),
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
        return self.ml_evaluator

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
