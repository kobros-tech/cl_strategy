# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""The one gate every public function in `skill_memory.diagnostics` calls first.

Every function in this package is either expensive (it is not something
`strategy.eval()` should ever pay for automatically), or capable of using
information a production prediction must never see (a true label, via
oracle routing). `require_diagnose` is the single, shared enforcement
point for "you must say `diagnose=True` yourself" -- so auditing where
diagnostic-only computation or ground truth could possibly reach a
result is one grep for ``diagnose=True``, not a review of every module
that might have forgotten to check a flag.
"""

from __future__ import annotations


def require_diagnose(diagnose: bool, function_name: str) -> None:
    """Raise unless the caller explicitly passed `diagnose=True`.

    `diagnose` has no default in any of this package's public functions --
    Python itself refuses the call before this even runs if it's left out
    entirely. This additionally rejects an explicit but falsy value
    (``diagnose=False``), so the failure mode is always the same clear
    error rather than a silently-empty result.
    """
    if not diagnose:
        raise RuntimeError(
            f"skill_memory.diagnostics.{function_name} refuses to run "
            "unless called with diagnose=True. This is not a convenience "
            "default: it exists so that ground truth (e.g. oracle "
            "routing) or the extra cost of this diagnostic can never "
            "reach a production result by accident. Pass diagnose=True "
            "explicitly at this call site to confirm that's what you want."
        )
