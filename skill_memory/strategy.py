"""High-level Avalanche strategy with Skill Memory and ML evaluation.

The strategy integrates two distinct learning/evaluation processes:

1. Skill Memory
   - class-level REUSE/SCRATCH decisions
   - skill allocation and storage
   - class-to-skill bookkeeping

2. Anonymous ML evaluator
   - receives only x at prediction time
   - learns x -> y from frozen examples retained by Skill Memory
   - is trained on all accumulated evaluation memory
   - evaluates all classes seen so far
   - provides the methodology used to measure non-forgetting

The ML evaluator is intentionally independent from the Skill Memory model.
Its purpose is to measure whether an independently trained classifier can
recover the class identity of anonymous samples from the accumulated
retained data after continual training.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate

from .cl.skill_memory_plugin import SkillMemoryPlugin
from .cl.skill_registry import SkillMemory
from .evaluation.ml_cl_evaluator import (
    EvaluationMemoryPlugin,
    MLEvaluationPlugin,
)


class SkillMemoryStrategy(SupervisedTemplate):
    """Avalanche strategy integrating Skill Memory and anonymous ML evaluation.

    The Avalanche strategy is responsible for lifecycle integration and for
    exposing the complete experiment through one public object.

    train_epochs controls the number of epochs used for each explicit
    class-training pass; Avalanche's mixed-experience training loop is disabled
    by SkillMemoryPlugin after that custom training has been scheduled.

    Skill Memory training and evaluation-memory retention remain owned by
    ``EvaluationMemoryPlugin``, which extends ``SkillMemoryPlugin``. The
    independent ML evaluator is the sole evaluation methodology used by the
    normal ``strategy.eval()`` lifecycle. Direct Skill Memory diagnostics are
    intentionally separate from this strategy.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        evaluator: EvaluationPlugin | None = None,
        plugins: list[SupervisedPlugin] | None = None,
        eval_every: int = -1,
        peval_mode: str = "epoch",
        max_skills: int = 200,
        forgetting_margin: float = 0.05,
        score_floor: float | None = 0.9,
        probe_batch_size: int = 64,
        probe_batches: int = 5,
        probe_seed: int | None = None,
        max_safety_candidates: int = 5,
        class_train_batch_size: int = 64,
        reuse_is_mutable: bool = True,
        force_decision: str | None = None,
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
        eval_routing: str = "none",
        probe_behavior_weight: float = 0.5,
    ) -> None:
        if eval_memory_per_class <= 0:
            raise ValueError("eval_memory_per_class must be positive")

        if train_epochs < 1:
            raise ValueError("train_epochs must be at least 1")

        if eval_epochs < 1:
            raise ValueError("eval_epochs must be at least 1")

        if eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")

        if eval_routing not in ("none", "probe"):
            raise ValueError("eval_routing must be one of 'none' or 'probe'")

        if not 0.0 <= probe_behavior_weight <= 1.0:
            raise ValueError("probe_behavior_weight must be between 0 and 1")

        if device is None:
            device = next(model.parameters()).device
        else:
            device = torch.device(device)

        self.eval_epochs = eval_epochs
        self.eval_batch_size = eval_batch_size
        self.eval_learning_rate = eval_learning_rate
        self.verbose = verbose
        self.eval_routing = eval_routing
        self.train_epochs = train_epochs
        self.probe_behavior_weight = float(probe_behavior_weight)

        # ------------------------------------------------------------------
        # Skill Memory
        # ------------------------------------------------------------------

        self.memory = SkillMemory(max_skills=max_skills)

        # EvaluationMemoryPlugin extends SkillMemoryPlugin. Therefore there
        # is exactly one Skill Memory plugin in the Avalanche plugin list.
        self.plugin = EvaluationMemoryPlugin(
            memory=self.memory,
            max_skills=max_skills,
            forgetting_margin=forgetting_margin,
            score_floor=score_floor,
            probe_batch_size=probe_batch_size,
            probe_batches=probe_batches,
            probe_seed=probe_seed,
            max_safety_candidates=max_safety_candidates,
            class_train_epochs=train_epochs,
            class_train_batch_size=class_train_batch_size,
            reuse_is_mutable=reuse_is_mutable,
            force_decision=force_decision,
            eval_memory_per_class=eval_memory_per_class,
            eval_memory_seed=eval_memory_seed,
            verbose=verbose,
        )

        self.ml_evaluation_plugin = MLEvaluationPlugin(
            memory_plugin=self.plugin,
            model_factory=evaluator_model_factory,
            epochs=eval_epochs,
            batch_size=eval_batch_size,
            learning_rate=eval_learning_rate,
            seed=eval_memory_seed,
            verbose=verbose,
            eval_routing=eval_routing,
            probe_behavior_weight=probe_behavior_weight,
        )

        strategy_plugins: list[SupervisedPlugin] = [
            self.plugin,
            self.ml_evaluation_plugin,
        ]

        if plugins:
            strategy_plugins.extend(plugins)

        if evaluator is None:
            evaluator = EvaluationPlugin(
                accuracy_metrics(stream=True),
                loss_metrics(stream=True),
            )

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=evaluator,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            eval_every=eval_every,
            peval_mode=peval_mode,
            device=device,
            plugins=strategy_plugins,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, exp_list, **kwargs):
        """Run normal Avalanche evaluation and include ML evaluator results."""
        avalanche_results = super().eval(exp_list, **kwargs)
        avalanche_results.update(self.ml_evaluation_plugin.results())
        return avalanche_results

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def results(self) -> dict[str, Any]:
        """Return the independent ML evaluation results."""
        return self.ml_evaluation_plugin.results()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def skill_memory(self) -> SkillMemory:
        """Return the underlying Skill Memory."""
        return self.memory

    @property
    def skill_memory_plugin(self) -> SkillMemoryPlugin:
        """Return the underlying Skill Memory plugin."""
        return self.plugin

    @property
    def evaluator_model(self) -> nn.Module | None:
        return self.ml_evaluation_plugin.model
