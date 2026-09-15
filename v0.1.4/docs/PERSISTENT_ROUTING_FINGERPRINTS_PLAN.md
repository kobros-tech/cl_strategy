# Persistent Routing Fingerprints

## Status

Design/implementation plan for v0.1.4 best-skill probe routing.

This document records what is being changed now and what the next implementation step must do. It deliberately does **not** redefine the routing score or the training-time Skill Memory decision logic.

## Why this is needed

Probe routing can repeatedly derive information from stored skills during evaluation. The package already caches expensive dataset class-index lookups, but that cache is about locating samples; it is not a cache of model-derived routing fingerprints.

A persistent fingerprint cache can avoid recomputing unchanged skill/class fingerprints on every evaluation pass.

## Current branch observations

- `probing.py` already has `RoutingResult` and input-only `find_best_routing_skill()`.
- The current routing implementation is still being aligned with the intended routing-score design (owned-class margin normalized by classifier weight norm).
- `test_probing_cache.py` currently protects the dataset class-index cache. That cache must remain separate from model fingerprint caching.
- Training-time REUSE updates an existing mutable skill slot in place. Therefore any model-derived fingerprint for that skill becomes stale after the update.
- Class-to-canonical-skill ownership must remain stable. Fingerprints are routing metadata and must never remap or discard learned class ownership.

## Target design

### 1. Fingerprints are persistent routing metadata

Store fingerprints associated with the skill/class they describe rather than recomputing them from every evaluation minibatch.

A cache entry must identify at least:

- `skill_id`
- `class_id`
- the skill/model version from which the fingerprint was produced
- the fingerprint value itself

The cache must be owned by the Skill Memory lifecycle (or a dedicated cache object owned by it), not by an evaluation minibatch.

### 2. Version mutable skills

Each stored skill gets a monotonically increasing version/generation.

- Creating a skill starts its version at an initial value.
- REUSE training changes the skill weights and increments that skill's version.
- A cached fingerprint is valid only when its stored version equals the current skill version.

This gives a cheap stale-entry check and prevents old fingerprints from silently surviving a REUSE update.

### 3. REUSE invalidation/update boundary

The update sequence must be:

1. Load the canonical mutable skill.
2. Train/update that skill exactly as the existing REUSE path does.
3. Store the updated state dict.
4. Increment the skill version.
5. Invalidate or regenerate fingerprints belonging to that skill.

The fingerprint update must happen **after** the new weights are stored. It must never fingerprint the pre-update state and then mark that fingerprint as current.

### 4. Invalidate the whole changed skill by default

If fingerprints are derived from shared skill weights, changing one class can alter the representation/logits of other classes mastered by the same skill.

Therefore the safe default is to invalidate/recompute fingerprints for **all classes owned by the updated skill**, not only the class that triggered REUSE.

Only narrow this to the changed class if the fingerprint implementation can prove that its mathematical dependency is class-local.

### 5. Prefer eager refresh after training when practical

The preferred lifecycle is:

`REUSE update -> save new state -> increment version -> refresh that skill's fingerprints`

This keeps evaluation cheap and deterministic: evaluation should normally consume already-valid fingerprints rather than performing a full recomputation for a changed skill.

If eager refresh is too expensive for a particular integration, the version check permits a lazy refresh without correctness risk. In that case stale entries are recomputed once and replaced with entries carrying the new version.

### 6. Evaluation must not change semantics

Fingerprints are a performance/cache layer. They must not introduce:

- labels (`y`) into normal probe routing
- task IDs or experience IDs as an oracle
- class-oracle routing
- fixed correctness thresholds
- changes to canonical class ownership
- changes to REUSE/SCRATCH semantics

The existing input-only routing API remains responsible for selecting a skill per sample.

## Planned API shape

The exact fingerprint representation must follow the existing fingerprint implementation used by the project. Before coding the cache, inspect that implementation and preserve its value/shape semantics.

The cache should expose operations equivalent to:

- get fingerprint for `(skill_id, class_id)` if its skill version is current
- set/update a fingerprint with the current skill version
- invalidate all fingerprints for a skill
- refresh all fingerprints for a skill after a REUSE update

Avoid exposing mutable cache internals to the plugin.

## Required tests

Add focused tests for:

1. A fingerprint is reused when the skill version has not changed.
2. A REUSE update increments the skill version.
3. Fingerprints from the previous version are not returned as current.
4. Updating skill 5 invalidates/refreshes all classes mastered by skill 5.
5. Fingerprints for unchanged skills remain reusable.
6. A newly created skill gets a valid initial fingerprint/version state.
7. Repeated evaluation does not recompute fingerprints for unchanged skills.
8. A changed skill is recomputed exactly once when using lazy refresh.
9. Class-to-canonical-skill mappings are unchanged by fingerprint operations.
10. Existing routing-result tests continue to verify per-sample, label-free routing.

## Validation

Run the package test suite, the probing/routing tests, and the project's pre-commit/style checks before considering the implementation complete.

Because the GitHub connector cannot execute the repository's local test environment here, CI and the user's local test run are the source of truth for execution validation.

## Explicit non-goals

This work does **not**:

- change `find_best_skill()` training-time decision logic;
- add CLONE to the current version;
- change REUSE into immutable reuse;
- remap global classifier columns into local skill columns;
- use labels during normal evaluation routing;
- replace the routing score with an unrelated fingerprint similarity metric.

## Next implementation step

Inspect the project's existing fingerprint calculation and integrate the persistent versioned cache around that exact calculation. Then connect the cache update to the existing REUSE post-training state-save boundary and add the tests above.
