# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Machine-learning evaluation and anonymous routing components."""

from .reverse_engineering import (
    CandidateParameters as CandidateParameters,
)
from .reverse_engineering import (
    NormalMLReverseEngineer as NormalMLReverseEngineer,
)
from .routing import RoutingResult as RoutingResult
from .routing import score_skill_compatibility as score_skill_compatibility
from .routing import select_skill_from_scores as select_skill_from_scores

# `independent_evaluator` is deliberately NOT re-exported here (only from the
# top-level `skill_memory` package, and always importable directly as
# `skill_memory.evaluation.independent_evaluator`). It subclasses
# `skill_memory.cl.skill_memory_plugin.SkillMemoryPlugin`, and `cl/`'s own
# modules trigger this package's `__init__` before `cl` itself has finished
# loading. Importing `independent_evaluator` here would import `cl` back before
# it's ready - a real circular import, not just a lint warning. See
# `skill_memory/__init__.py`, where `cl` is already fully loaded by the time
# `independent_evaluator` is imported.
