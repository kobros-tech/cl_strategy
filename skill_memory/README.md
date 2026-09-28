# `skill_memory` — implementation notes

This is a contributor-facing companion to the [root README](../README.md),
which covers the concepts, the math, and how to use the package. This
document covers the internal contracts each module relies on, for anyone
changing the code rather than just calling it.

## Module dependency order

Lower modules never import from higher ones — breaking this ordering
reintroduces a real circular import, not just a lint warning:

```
utils/probing.py            (no dependency on cl/ or evaluation/)
        |
        v
cl/skill_registry.py  -->  cl/decision.py  -->  cl/skill_memory_plugin.py
        |                                              |
        v                                              v
cl/training.py                          cl/persistent_skill_memory_plugin.py
        |
        v
evaluation/routing.py
        |
        v
evaluation/behavior.py --> evaluation/reverse_engineering.py
        |
        v
evaluation/fingerprint_routing.py, evaluation/global_fingerprint_refresh.py
        |
        v
evaluation/independent_evaluator.py   (subclasses cl.skill_memory_plugin.SkillMemoryPlugin)
        |
        v
strategy.py                             (top-level; imports everything above)

diagnostics/   (depends only on utils/ and evaluation/routing.py; imported by
                production code in exactly three places -- see below)
```

`evaluation/independent_evaluator.py` is deliberately **not** re-exported
from `evaluation/__init__.py` — only from the top-level `skill_memory`
package — for exactly this reason (see the comment at the top of
`evaluation/__init__.py`).

## Bookkeeping invariants (`cl/skill_registry.py`)

- `SkillMemory` stores `state_dict` snapshots by integer slot;
  `ExperienceClassMap` stores which slot owns which class. These are kept
  as two separate objects on purpose: a skill can master more than one
  class, and one experience can therefore be associated with more than one
  `(skill, classes)` group.
- Once `ExperienceClassMap` records a class → skill mapping, it is never
  overwritten. `find_skill_for_class_anywhere` is the one lookup every
  other module should use rather than re-deriving it.
- `SkillMemory.allocate()` reserves the *lowest free* slot and raises
  `RuntimeError` once `max_skills` is reached — callers (`decision.py`,
  `skill_memory_plugin.py`) are expected to handle that as "memory full,"
  not as a bug.

## State application (`utils/probing.py`)

Two distinct contracts live side by side here; picking the wrong one for
a new call site either corrupts the live model or silently reintroduces
the O(skills) cost this module exists to avoid.

**Mutating** (a real, persistent state change): `apply_skill_state_exact`
first calls `resize_incremental_classifiers_for_state` so
`nn.Module.load_state_dict` never fails on a shape mismatch between the
model's current `IncrementalClassifier` width and the snapshot's recorded
width, then loads it for real. Use this (via `restore_initial_state`,
its own name for the same operation used to undo scratch-training
adaptation) wherever a skill's weights need to actually become the live
model's weights going forward — `SkillMemoryPlugin`'s REUSE/SCRATCH
training paths, and its before/after-eval snapshot restore.

**Functional** (a disposable probe): `predict_logits` and
`evaluate_state` apply a stored snapshot with
`torch.func.functional_call` instead, so the model they're given is
*never mutated* — no resize, no restore, and (critically) no per-skill
`load_state_dict` copy of every parameter tensor. This is what makes
`score_class_against_skills`' "for every stored skill, forward a probe
batch" loop, and `MLEvaluationPlugin.after_eval_forward`'s per-batch
routing, cheap: trying skill `k+1` costs one more forward pass, not one
more full parameter copy. `evaluate_state`'s classifier-growth rule for a
genuinely new class (`_functional_growth_for_experience`) deliberately
duplicates `IncrementalClassifier.adaptation`'s math rather than calling
`prepare_for_experience` (the mutating version), for the same reason.
When adding a new read-only probe, prefer this contract; reach for the
mutating one only when the caller genuinely needs the model itself to
keep the new state afterwards.

`classes_in_experience`/`class_indices` cache each dataset's full label
list, keyed by the dataset *object* (a `weakref.WeakKeyDictionary`, not
`id()`), so `decision.py`'s per-`(skill, class)` probing doesn't rescan
the same dataset once per pair. See
[`tests/test_probing_cache.py`](tests/test_probing_cache.py) for the
exact scanning-cost guarantee this cache makes.

## Decision policy (`cl/decision.py`)

See the [root README's math section](../README.md#how-it-decides-reuse-vs-scratch)
for the formulas. Implementation notes that don't belong in that
higher-level explanation:

- `score_class_against_skills` is two-staged on purpose: stage 1
  (`evaluate_state` against the new class) runs for *every* stored skill,
  cheaply — one functional forward pass each, no model copies (see
  above); stage 2 checks old classes for every candidate by default.
  `max_safety_candidates` can be set to a finite value as an explicit
  performance approximation when the full safety check is too expensive.
- `_strongest_candidates` finds the largest gap in a sorted metric
  ranking rather than a fixed threshold, so the "how much better than the
  runner-up does a candidate need to be" question doesn't need its own
  magic number.
- `evaluate_state` never takes a gradient step — "imagination" means
  measuring a frozen skill's *existing* representation on a class it may
  never have trained on, not training it further.

## Timing instrumentation (`diagnostics/timing.py`)

`SkillMemoryPlugin.timing` and `SkillMemoryStrategy.timing` are each a
`TimingAccumulator`; `diagnostics.timing_report(strategy)` merges both
into the three buckets described in the
[root README](../README.md#performance-functional-probing-and-where-the-time-goes).
If you add a new expensive stage to the lifecycle, wrap it with
`self.timing.track("some_bucket_name")` on whichever plugin/strategy owns
it, rather than adding another ad hoc `time.perf_counter()` call — the
existing three buckets are read together specifically so the report
stays comparable across runs.

## Anonymous routing (`evaluation/routing.py`, `diagnostics/routing.py`)

`evaluation/routing.py` holds the shared primitives
(`score_skill_compatibility`, `select_skill_from_scores`,
`_normalize_routing_scores`) used by both the evaluator-based probe router
(`evaluation/independent_evaluator.py`) and the evaluator-free anonymous
router (`find_best_routing_skill` in `diagnostics/routing.py`). If
you change the temperature/normalization rule in one, check whether the
other's tests
([`tests/test_routing.py`](tests/test_routing.py),
[`tests/test_continuous_fingerprint_routing.py`](tests/test_continuous_fingerprint_routing.py))
still hold — they intentionally share the same math.

## Running the tests

```bash
pytest skill_memory/tests -q
```

96 tests, no network access and no GPU required; the slowest ones
(`test_strategy.py`) build tiny synthetic Avalanche benchmarks rather than
downloading a real dataset, so the whole suite runs in well under two
minutes on CPU.

## Adding a new diagnostic

1. Put it in `skill_memory/diagnostics/`, never next to production code.
2. If it can read a true label or costs real forward passes, make
   `diagnose` a **required keyword-only argument with no default** and call
   `require_diagnose(diagnose, "your_function_name")` first (see
   `diagnostics/_gate.py`), then add it to
   `GATED_FUNCTIONS` in `tests/test_diagnostics_gate.py` — that test
   asserts the no-default rule mechanically.
3. Do **not** import it from production modules. If production code truly
   must (as `fingerprint_routing.py` does for its own already-flag-gated
   reports), add the import to the `allowed` set in
   `test_production_modules_do_not_import_diagnostic_functions` and say why
   in the docstring — the point of that test is that this list stays tiny
   and deliberate.
