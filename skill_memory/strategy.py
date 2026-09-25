"""High-level Avalanche strategy with Skill Memory and ML evaluation.

The strategy integrates Skill Memory training with independent evaluation plugins:

1. Skill Memory
   - class-level REUSE/SCRATCH decisions
   - skill allocation and storage
   - class-to-skill bookkeeping
   - optional direct Skill Memory diagnostics

2. Anonymous two-stage ML evaluator
   - for eval_routing="none", receives only x at prediction time
   - learns x -> omega and omega -> y using retained CL trajectory states
   - predicts x -> ML-1 -> omega -> ML-2 -> y during evaluation

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
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate

from .cl.skill_memory_plugin import SkillMemoryPlugin
from .cl.skill_registry import SkillMemory
from .evaluation.ml_cl_evaluator import (
    EvaluationMemoryPlugin,
    MLEvaluationPlugin,
)
from .evaluation.weight_state_ml_evaluator import (
    WeightEvaluationMemory,
    WeightStateMLEvaluationPlugin,
)


class SkillMemoryStrategy(SupervisedTemplate):
    """Avalanche strategy integrating Skill Memory and anonymous ML evaluation.

    The Avalanche strategy is responsible for lifecycle integration and for
    exposing the complete experiment through one public object.

    Skill Memory training itself remains in ``EvaluationMemoryPlugin`` /
    ``SkillMemoryPlugin``. The independent ML evaluator is implemented as an
    Avalanche plugin. The strategy constructs and registers that plugin as
    part of the experiment's evaluation methodology; it is not part of Skill
    Memory's internal training algorithm.
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
        class_train_epochs: int = 10,
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
        weight_state_num_classes: int | None = None,
        weight_state_ml2_model_factory: Callable[[int, int], nn.Module] | None = None,
        weight_state_ml1_model_factory: Callable[[int, int], nn.Module] | None = None,
        weight_state_ml1_learning_rate: float = 0.001,
        weight_state_ml1_epochs: int | None = None,
        weight_state_ml1_batch_size: int | None = None,
        weight_state_snapshots_per_class: int = 10,
        weight_state_eval_epochs: int = 10,
        weight_state_eval_batch_size: int = 64,
        weight_state_eval_learning_rate: float = 0.01,
        weight_state_eval_hidden_size: int = 128,
        weight_state_representation_size: int = 128,
        train_mb_size: int = 64,
        train_epochs: int = 1,
        eval_mb_size: int = 64,
        device: torch.device | str | None = None,
        verbose: bool = True,
    ) -> None:
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
        self.weight_state_eval_epochs = weight_state_eval_epochs
        self.weight_state_eval_batch_size = weight_state_eval_batch_size
        self.weight_state_eval_learning_rate = weight_state_eval_learning_rate
        self.weight_state_representation_size = weight_state_representation_size
        self.skill_eval_routing = skill_eval_routing
        self.verbose = verbose

        if weight_state_num_classes is None:
            linear_layers = [
                module for module in model.modules() if isinstance(module, nn.Linear)
            ]
            if not linear_layers:
                raise ValueError(
                    "weight_state_num_classes must be provided when the model "
                    "has no final Linear classification layer"
                )
            weight_state_num_classes = linear_layers[-1].out_features
        if weight_state_num_classes < 1:
            raise ValueError("weight_state_num_classes must be positive")
        self.weight_state_num_classes = int(weight_state_num_classes)

        self.weight_evaluation_memory = WeightEvaluationMemory(
            max_snapshots_per_class=weight_state_snapshots_per_class,
        )

        # ------------------------------------------------------------------
        # Skill Memory
        # ------------------------------------------------------------------

        self.memory = SkillMemory(max_skills=max_skills)

        # EvaluationMemoryPlugin extends SkillMemoryPlugin. Therefore there
        # is exactly one Skill Memory plugin in the Avalanche plugin list.
        def capture_weight_state(
            model: nn.Module,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            class_id: int,
            skill_id: int,
            step: int,
        ) -> None:
            self.weight_evaluation_memory.add(
                model.state_dict(),
                inputs=inputs,
                targets=targets,
                class_id=class_id,
                skill_id=skill_id,
                step=step,
            )

        self.plugin = EvaluationMemoryPlugin(
            memory=self.memory,
            weight_state_callback=capture_weight_state,
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

        self.ml_evaluation_plugin = MLEvaluationPlugin(
            memory_plugin=self.plugin,
            model_factory=evaluator_model_factory,
            epochs=eval_epochs,
            batch_size=eval_batch_size,
            learning_rate=eval_learning_rate,
            seed=eval_memory_seed,
            verbose=verbose,
        )

        self.weight_state_ml_evaluation_plugin = WeightStateMLEvaluationPlugin(
            memory=self.weight_evaluation_memory,
            num_classes=self.weight_state_num_classes,
            ml2_model_factory=weight_state_ml2_model_factory,
            ml1_model_factory=weight_state_ml1_model_factory,
            epochs=weight_state_eval_epochs,
            batch_size=weight_state_eval_batch_size,
            learning_rate=weight_state_eval_learning_rate,
            ml1_learning_rate=weight_state_ml1_learning_rate,
            ml1_epochs=weight_state_ml1_epochs,
            ml1_batch_size=weight_state_ml1_batch_size,
            seed=eval_memory_seed,
            hidden_size=weight_state_eval_hidden_size,
            state_representation_size=weight_state_representation_size,
            verbose=verbose,
        )

        strategy_plugins: list[SupervisedPlugin] = [self.plugin]
        if skill_eval_routing == "none":
            strategy_plugins.append(self.weight_state_ml_evaluation_plugin)
        else:
            strategy_plugins.append(self.ml_evaluation_plugin)

        if plugins:
            strategy_plugins.extend(plugins)

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
    # Results
    # ------------------------------------------------------------------

    def results(self) -> dict[str, Any]:
        """Return results for the active independent evaluation path."""
        if self.skill_eval_routing == "none":
            return {
                "weight_state_evaluation": (
                    self.weight_state_ml_evaluation_plugin.current_result
                )
            }

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

    @property
    def weight_state_evaluator_model(self) -> nn.Module | None:
        return self.weight_state_ml_evaluation_plugin.ml2_model

    @property
    def weight_state_memory(self) -> WeightEvaluationMemory:
        return self.weight_evaluation_memory
