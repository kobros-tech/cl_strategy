# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
