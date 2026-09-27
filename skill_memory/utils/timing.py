# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""A tiny cumulative wall-clock timer, used only for opt-in diagnostics.

This intentionally does not touch anything performance-sensitive itself:
`TimingAccumulator.track` is a context manager that records
`time.perf_counter()` before and after the wrapped block, so it adds one
timestamp pair per call, not per batch or per skill. It exists so that
"which part of a training/eval cycle is slow" is answered by reading a
report instead of guessing (see `skill_memory.diagnostics.timing_report`).
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class TimingAccumulator:
    """Cumulative wall-clock time and call counts, grouped by bucket name.

    A bucket is any string a caller chooses (e.g. ``"decision"``); the same
    name can be tracked many times (once per class, once per experience,
    ...) and the accumulator sums the elapsed time and counts the calls.
    """

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}

    @contextmanager
    def track(self, bucket: str):
        """Time one block of code, adding it to `bucket`'s running total."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._seconds[bucket] = self._seconds.get(bucket, 0.0) + elapsed
            self._calls[bucket] = self._calls.get(bucket, 0) + 1

    def report(self) -> dict[str, dict[str, float]]:
        """Return ``{bucket: {total_seconds, calls, mean_seconds}}``, per bucket."""
        return {
            bucket: {
                "total_seconds": total,
                "calls": self._calls[bucket],
                "mean_seconds": total / self._calls[bucket],
            }
            for bucket, total in self._seconds.items()
        }

    def reset(self) -> None:
        """Clear every recorded bucket, e.g. between independent experiments."""
        self._seconds.clear()
        self._calls.clear()
