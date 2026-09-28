# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Avalanche plugin for class-level, probe-based Skill Memory.

An Avalanche experience is only a container.  The strategy extracts the
classes actually present in that experience and handles them independently:

    experience -> class -> REUSE/SCRATCH -> train only that class

Bookkeeping is explicit:

    experience -> [(skill, {classes})]
    class      -> canonical skill
    skill      -> all classes it currently masters

A class that has already been mastered is never re-assigned by the generic
probe heuristic. It deterministically returns to its canonical skill, and
REUSE may update the same reserved skill slot when `reuse_is_mutable=True`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from ..diagnostics.timing import TimingAccumulator
from ..utils.probing import (
    apply_skill_state_exact,
    classes_in_experience,
    origin_experience,
    prepare_for_experience,
    restore_initial_state,
)
from .decision import decide_class
from .skill_registry import ClassRecord, ExperienceClassMap, SkillMemory
from .training import train_on_class

logger = logging.getLogger(__name__)


class SkillMemoryPlugin(SupervisedPlugin):
    """Class-level Skill Memory plugin with explicit class bookkeeping."""

    REUSE, SCRATCH = "reuse", "scratch"

    #: Bucket names used with `self.timing` (see
    #: `skill_memory.diagnostics.timing_report`).
    TIMING_DECISION = "skill_memory_decision_probing"
    TIMING_CLASS_TRAINING = "skill_memory_class_training"

    def __init__(
        self,
        memory: SkillMemory | None = None,
        *,
        max_skills: int = 200,
        forgetting_margin: float = 0.05,
        score_floor: float | None = 0.9,
        probe_batch_size: int = 64,
        probe_batches: int = 5,
        probe_seed: int | None = None,
        max_safety_candidates: int | None = None,
        class_train_epochs: int = 1,
        class_train_batch_size: int = 64,
        reuse_is_mutable: bool = True,
        skill_name: Callable | None = None,
        force_decision: str | None = None,
        verbose: bool = True,
        diagnose: bool = False,
    ):
        """Configure per-class REUSE/SCRATCH decisions.

        `memory` stores each skill's model-state snapshot; a fresh one is
        created if not given. With `reuse_is_mutable=False`, snapshots remain
        unchanged after REUSE; with `reuse_is_mutable=True`, the canonical
        skill snapshot is updated in place. `diagnose` controls only whether
        `self.timing` actually records anything
        (see `skill_memory.diagnostics.timing_report`)
        -- it defaults to `False` so a production run never pays even the
        cost of `time.perf_counter()` calls it will not read back.
        """
        super().__init__()
        if force_decision not in (None, self.REUSE, self.SCRATCH):
            raise ValueError("invalid force_decision")

        self.memory = memory if memory is not None else SkillMemory(max_skills)
        self.class_map = ExperienceClassMap()
        self.forgetting_margin = forgetting_margin
        self.score_floor = score_floor
        self.probe_batch_size = probe_batch_size
        self.probe_batches = probe_batches
        self.probe_seed = probe_seed
        if max_safety_candidates is not None and max_safety_candidates < 1:
            raise ValueError("max_safety_candidates must be positive or None")
        self.max_safety_candidates = (
            None if max_safety_candidates is None else int(max_safety_candidates)
        )
        self.class_train_epochs = class_train_epochs
        self.class_train_batch_size = class_train_batch_size
        self.reuse_is_mutable = reuse_is_mutable
        self.skill_name = skill_name
        self.force_decision = force_decision
        self.verbose = verbose
        self.diagnose = bool(diagnose)

        self.last_class_decisions: dict[int, dict[int, dict[str, Any]]] = {}
        self.timing = TimingAccumulator(enabled=self.diagnose)
        self._initial_state: dict | None = None
        self._task_active = False
        self._seen_experiences: list = []
        self._training_experience_count = 0
        self._current_training_experience_index: int | None = None
        self._original_train_epochs: int | None = None
        self._pre_eval_state: dict | None = None
        self._eval_active = False

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
        else:
            logger.info(message)

    @staticmethod
    def _snapshot(model) -> dict:
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    @staticmethod
    def _is_first_subexp(experience) -> bool:
        return getattr(experience, "is_first_subexp", True)

    @staticmethod
    def _is_last_subexp(experience) -> bool:
        return getattr(experience, "is_last_subexp", True)

    @staticmethod
    def _capture_optimizer_groups(strategy) -> dict[str, int]:
        """Map parameter names to their optimizer parameter-group indices."""
        optimizer = getattr(strategy, "optimizer", None)
        if optimizer is None:
            return {}

        group_by_id = {}
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter in group["params"]:
                group_by_id[id(parameter)] = group_index

        return {
            name: group_by_id[id(parameter)]
            for name, parameter in strategy.model.named_parameters()
            if id(parameter) in group_by_id
        }

    def _reset_optimizer(
        self,
        strategy,
        group_by_name: dict[str, int] | None = None,
    ) -> None:
        """Rebind optimizer parameters and reset optimizer history.

        Optimizer state is intentionally not part of a stored Skill Memory
        snapshot. Switching skills therefore starts with a clean optimizer
        state instead of leaking momentum or adaptive moments from another
        skill into the restored weights.
        """
        optimizer = getattr(strategy, "optimizer", None)
        if optimizer is None:
            return

        params_by_name = dict(strategy.model.named_parameters())
        if not optimizer.param_groups:
            optimizer.add_param_group({"params": list(params_by_name.values())})
            optimizer.state.clear()
            return

        group_by_name = group_by_name or {}
        grouped_params = [[] for _ in optimizer.param_groups]
        for name, parameter in params_by_name.items():
            group_index = group_by_name.get(name, 0)
            if not 0 <= group_index < len(grouped_params):
                group_index = 0
            grouped_params[group_index].append(parameter)

        for group, parameters in zip(
            optimizer.param_groups,
            grouped_params,
            strict=True,
        ):
            group["params"] = parameters

        optimizer.state.clear()

    def _scratch_reset(self, strategy, experience) -> None:
        if self._initial_state is None:
            raise RuntimeError("Initial model state has not been captured")
        group_by_name = self._capture_optimizer_groups(strategy)
        restore_initial_state(strategy.model, self._initial_state)
        # A fresh skill must have the head required by the current
        # Avalanche experience before its single-class training pass.
        prepare_for_experience(strategy.model, experience)
        self._reset_optimizer(strategy, group_by_name)

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def before_training_exp(self, strategy, **kwargs) -> None:
        """Decide REUSE/SCRATCH and train each class in this experience.

        Avalanche experiences may be split into sub-experiences; this hook
        tracks logical experience boundaries so every sub-experience's
        classes get processed, not just the first.
        """
        experience = strategy.experience
        first_subexp = self._is_first_subexp(experience)

        # A logical Avalanche experience can be split into sub-experiences.
        # The old implementation processed ONLY the first sub-experience,
        # which silently dropped classes that lived in later sub-experiences.
        # Keep one logical experience index, but process every sub-experience.
        if first_subexp:
            if self._task_active:
                raise RuntimeError(
                    "A new first sub-experience arrived while the previous "
                    "logical experience is still active"
                )

            self._task_active = True
            self._current_training_experience_index = self._training_experience_count
            self._training_experience_count += 1
            experience_index = self._current_training_experience_index

            if self._initial_state is None:
                self._initial_state = self._snapshot(strategy.model)

            self.last_class_decisions[experience_index] = {}

            # Never let Avalanche's normal mixed-experience loop retrain the
            # data after our explicit class-by-class loop.
            self._original_train_epochs = getattr(strategy, "train_epochs", None)
            if self._original_train_epochs is not None:
                strategy.train_epochs = 0
        else:
            if not self._task_active or self._current_training_experience_index is None:
                raise RuntimeError(
                    "Received a non-first sub-experience without an active "
                    "logical training experience"
                )
            experience_index = self._current_training_experience_index

        classes = classes_in_experience(experience)
        self._log(
            f"Experience {experience_index} "
            f"subexp(first={first_subexp}, last={self._is_last_subexp(experience)}): "
            f"classes={classes}"
        )

        if not classes:
            self._log(
                f"Experience {experience_index}: empty sub-experience; nothing to train"
            )
            return

        for target_class in classes:
            decision_start = time.perf_counter() if self.diagnose else None
            with self.timing.track(self.TIMING_DECISION):
                decision = decide_class(
                    strategy,
                    experience,
                    target_class,
                    self.memory,
                    self.class_map,
                    self.probe_batch_size,
                    self.probe_batches,
                    self.probe_seed,
                    self._seen_experiences,
                    self.forgetting_margin,
                    self.score_floor,
                    self.force_decision,
                    self._log,
                    max_safety_candidates=self.max_safety_candidates,
                )
            if decision_start is not None:
                self._log(
                    f"Class {target_class}: imagination+decision "
                    f"time={time.perf_counter() - decision_start:.2f}s"
                )

            if decision["decision"] == self.REUSE:
                skill = decision["skill"]
                self._log(
                    f"Class {target_class}: REUSE skill {skill} "
                    f"(mutable={self.reuse_is_mutable})"
                )
                # A canonical class is already known to belong to this skill.
                # Restore its snapshot exactly: re-adapting the model to the
                # current sub-experience can traverse incompatible FlatData
                # indices and is unnecessary for deterministic class reuse.
                group_by_name = self._capture_optimizer_groups(strategy)
                apply_skill_state_exact(
                    strategy.model,
                    self.memory.state(skill),
                )
                # A new class may reuse an existing skill. Its snapshot can
                # have a narrower IncrementalClassifier head, so grow the
                # live head before training the new target class. Known-class
                # reuse remains an exact snapshot restore.
                if not decision.get("known_class", False):
                    prepare_for_experience(strategy.model, experience)
                self._reset_optimizer(strategy, group_by_name)

                if self.reuse_is_mutable:
                    with self.timing.track(self.TIMING_CLASS_TRAINING):
                        train_on_class(
                            strategy,
                            experience,
                            target_class,
                            self.class_train_epochs,
                            self.class_train_batch_size,
                        )
                    self.memory.store(
                        skill,
                        strategy.model.state_dict(),
                        metadata={
                            **self.memory.metadata(skill),
                            "last_updated_class": target_class,
                            "last_updated_experience": experience_index,
                        },
                    )
                    self._log(f"Class {target_class}: skill {skill} updated in place")
                else:
                    self._log(f"Class {target_class}: skill {skill} left unchanged")
            else:
                skill = self.memory.allocate()
                self._log(f"Class {target_class}: SCRATCH -> new skill {skill}")
                self._scratch_reset(strategy, experience)
                with self.timing.track(self.TIMING_CLASS_TRAINING):
                    train_on_class(
                        strategy,
                        experience,
                        target_class,
                        self.class_train_epochs,
                        self.class_train_batch_size,
                    )
                self.memory.store(
                    skill,
                    strategy.model.state_dict(),
                    metadata={
                        "acquisition_decision": self.SCRATCH,
                        "experience_index": experience_index,
                        "last_updated_class": target_class,
                        "probe_batch_size": self.probe_batch_size,
                        "probe_batches": self.probe_batches,
                        "probe_seed": self.probe_seed,
                    },
                )
                decision["skill"] = skill

            self.last_class_decisions[experience_index][target_class] = decision
            self.class_map.record(
                ClassRecord(
                    experience_index=experience_index,
                    class_id=target_class,
                    decision=decision["decision"],
                    skill=decision["skill"],
                    new_score=decision.get("new_score", 0.0),
                    old_score=decision.get("old_score", 0.0),
                    old_accuracy=decision.get("old_accuracy", 0.0),
                    new_accuracy=decision.get("new_accuracy", 0.0),
                )
            )

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Log the class->skill assignments for this experience and close it out.

        No-op until the last sub-experience of a logical experience has
        finished (see `before_training_exp`).
        """
        experience = strategy.experience
        if not self._is_last_subexp(experience):
            return

        experience_index = self._current_training_experience_index
        if experience_index is None:
            raise RuntimeError("Missing current training experience index")

        grouped = self.class_map.skills_for_experience(experience_index)
        for skill, classes in grouped:
            self._log(
                f"Experience {experience_index}: skill {skill} covers "
                f"classes {sorted(classes)}"
            )

        # Explicit class -> skill view.  This is intentionally printed for
        # every class in the logical experience, including classes that were
        # attached to an already-existing skill.  It makes it impossible to
        # mistake a skill index for a class index.
        assignments = self.class_map.class_skill_for_experience(experience_index)
        self._log(
            f"Experience {experience_index}: class->skill "
            + ", ".join(
                f"{class_id}->{skill}"
                for class_id, skill in sorted(assignments.items())
            )
        )

        if self._original_train_epochs is not None:
            strategy.train_epochs = self._original_train_epochs
        self._original_train_epochs = None
        self._seen_experiences.append(origin_experience(experience))
        self._current_training_experience_index = None
        self._task_active = False

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    def before_eval(self, strategy, **kwargs) -> None:
        """Snapshot model state before the evaluation phase."""
        self._pre_eval_state = self._snapshot(strategy.model)
        self._eval_active = True

    def before_eval_exp(self, strategy, **kwargs) -> None:
        """Prepare an evaluation experience.

        Skill Memory does not route evaluation samples here. Anonymous
        evaluation routing, when enabled, is owned by the independent ML
        evaluator.
        """
        return

    def after_eval_forward(self, strategy, **kwargs) -> None:
        """Leave evaluation outputs untouched."""
        return

    def after_eval(self, strategy, **kwargs) -> None:
        """Restore the model state that existed before evaluation."""
        if not self._eval_active:
            return
        try:
            if self._pre_eval_state is not None:
                group_by_name = self._capture_optimizer_groups(strategy)
                restore_initial_state(strategy.model, self._pre_eval_state)
                self._reset_optimizer(strategy, group_by_name)
        finally:
            self._pre_eval_state = None
            self._eval_active = False
