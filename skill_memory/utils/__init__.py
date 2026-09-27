# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Shared, package-independent helpers used by ``skill_memory.cl`` and
``skill_memory.evaluation``.

Nothing in this sub-package may import from ``skill_memory.cl`` or
``skill_memory.evaluation`` -- it sits below both of them in the
dependency graph so that either can import it without risking a
circular import.
"""

from __future__ import annotations
