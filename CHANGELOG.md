# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2026

### Breaking

- **All diagnostic code now lives in one package,
  `skill_memory/diagnostics/`**, replacing the top-level
  `skill_memory/diagnostics.py`, `skill_memory/evaluation/diagnostics.py`
  and `skill_memory/utils/timing.py`.
- **Every diagnostic entry point now requires `diagnose=True`** as a
  keyword-only argument with no default (`find_best_routing_skill`,
  `route_probe_logits`, `evaluate_skill_memory`, `evaluate_class_oracle`,
  `diagnose_evaluator_probe`, `routing_rank_diagnostics`,
  `class_index_alignment_report`). Omitting it is a `TypeError`;
  `diagnose=False` is a `RuntimeError`. Migration: add `diagnose=True` at
  each call site.
- `class_index_alignment_report` is no longer exported from the top-level
  `skill_memory` namespace; import it from `skill_memory.diagnostics`.
- `timing_report` / `reset_timing` now require a strategy built with
  `SkillMemoryStrategy(..., diagnose=True)` (new argument, default
  `False`). Previously timing was always recorded.

### Changed

- With `diagnose=False`, every internal `self.timing.track(...)` is a true
  no-op (no `time.perf_counter()` call), so production runs pay nothing
  for instrumentation. `PersistentFingerprintSkillMemoryPlugin`'s existing
  `diagnose` flag now also controls this.

### Added

- `tests/test_diagnostics_gate.py`: enforces the required-keyword
  signatures, the refusals, that no diagnostic name leaks into the
  top-level package, and (by parsing imports) that production modules
  import from `skill_memory.diagnostics` only in three known, gated places.
- The demo's `--diagnose` flag now drives the strategy's `diagnose=` and
  prints the timing report.

### Fixed

- README no longer claims probing operates on a `deepcopy` (it has been
  functional since 1.1.0).

## [1.1.0] - 2026

### Changed

- **Performance:** the "for every new class, for every skill: load skill
  state, forward probe batch" loop
  (`decision.score_class_against_skills`), and the equivalent per-batch
  routing in `MLEvaluationPlugin.after_eval_forward`, no longer mutate a
  model to switch between skills. `evaluate_state` and `predict_logits`
  now apply each skill's frozen weights with `torch.func.functional_call`
  instead of `load_state_dict`, removing an O(skills) full-parameter-copy
  cost from both hot loops (and the `deepcopy`s that existed only to make
  that mutation safe to undo). The full test suite's wall-clock time fell
  from ~111s to ~17s as a direct result.

### Added

- `skill_memory.diagnostics.timing_report(strategy)` /
  `reset_timing(strategy)`: cumulative wall-clock time and call counts for
  skill-memory decision/probing, skill-memory class training, and the
  independent evaluator + test-evaluation loop, so a slow run can be
  measured instead of guessed at.

## [1.0.0] - 2026

First public/production release.

### Fixed

- Restored `skill_memory/utils/` (dataset probing, exact state
  application, and `IncrementalClassifier` introspection) and the
  top-level `skill_memory/diagnostics.py` (anonymous routing and direct
  Skill Memory evaluation), both of which were missing from the previous
  internal snapshot. Without them, `pip install -e .` failed outright and
  roughly a third of the test suite could not even be collected.

### Changed

- Renamed `skill_memory/evaluation/ml_cl_evaluator.py` to
  `skill_memory/evaluation/independent_evaluator.py` to match the name
  used throughout its own documentation and the rest of the codebase.

### Added

- Copyright/SPDX headers on every source file, inserted and kept in sync
  automatically by a `pre-commit` hook (`insert-license`) rather than by
  hand.
- Mathematical description of the reuse-vs-scratch decision policy and
  the anonymous routing rule in the [README](README.md).
- Packaging metadata for a public release: license/author/classifier
  fields in `pyproject.toml`, and `skill_memory.demos` as an installable
  package (previously only importable from an editable checkout).

### Verified

- Full test suite (96 tests) passes.
- The `IncrementalClassifier` growth/resize path — not exercised by any
  existing test, since all of them use a fixed-size classifier — was
  additionally smoke-tested end to end against real Avalanche machinery.
