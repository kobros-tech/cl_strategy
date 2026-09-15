# Persistent class behavioral fingerprints

This branch implements persistent class-behavior references on top of
`v0.1.5-best-skill-routing`.

## Lifecycle

- Each canonical class gets a `ClassBehaviorRecord` containing reference
  inputs, global class IDs, a persistent probability-distribution fingerprint,
  and a skill version.
- References are computed after training and reused across evaluation passes.
- A mutable `REUSE` update creates a new generation for that skill and refreshes
  all classes mastered by that skill, because one shared state change can alter
  every class it owns.
- `SCRATCH` creates a new skill and its initial class references.
- Unchanged skills keep their cached references.
- A logical experience with multiple Avalanche sub-experiences accumulates all
  changed skills before refreshing them.

## Routing

`PersistentFingerprintSkillMemoryPlugin` performs task-free evaluation routing.
For each unlabeled sample it:

1. runs the sample through each stored skill;
2. compares the current behavior with the persistent fingerprints for the
   classes owned by that skill;
3. finds the best matching class within each skill;
4. compares the winning class match across skills;
5. uses the winning canonical skill to produce the final prediction.

The router never needs a task ID, experience ID, or target label. Labels are
only suitable for separate diagnostic evaluation such as routing accuracy.

The primary fingerprint signal is cosine similarity between class-aligned
mean softmax probability distributions. This preserves global classifier
columns and is invariant to arbitrary logit scale. A normalized four-value
behavior summary is an additional signal.

The tests explicitly verify that fingerprints generated from distinct class
reference logits rank their source class above an unrelated class and that
clear matches produce non-uniform skill-routing evidence.

## Fingerprint validation diagnostics

The implementation also measures whether the fingerprint itself is a stable
and discriminative representation of learned behavior before changing its
mathematical formula.

At the beginning of each logical training experience, existing persistent
fingerprints are snapshotted. After changed skills are refreshed, the plugin
compares the same reference inputs before and after training and logs:

- per-class fingerprint cosine similarity;
- per-class fingerprint drift (`1 - similarity`);
- the calculated fingerprint of each newly introduced class using its top
  global classifier coordinates;
- pairwise reference-fingerprint separation across all current classes.

This separates several possible failure modes: unstable fingerprints,
cross-class interference, poor reference separation, and a routing problem
where stored references are distinct but anonymous samples still produce
similar behavior across competing skills.

Pairwise reference similarity is intentionally diagnostic only. It compares
stored fingerprints with each other and does not establish anonymous routing,
because it does not run a sample through competing skill snapshots.

## Growing heads

Fingerprints store `output_class_ids` explicitly instead of assuming that a
fixed tensor column always represents a particular class. This keeps the
reference representation tied to global class IDs while Avalanche grows the
classifier head.

## Checkpoints

`BehaviorFingerprintCache.state_dict()` stores skill generations and all
persistent records. `load_state_dict()` accepts missing behavior state, so a
checkpoint produced before fingerprints were introduced can still be loaded.

## Usage

The extension is exported as `PersistentFingerprintSkillMemoryPlugin` so the
existing `SkillMemoryPlugin` remains available while this routing path is
validated against the current benchmark and integration tests.
