# Persistent binary class-behavior identification

This branch extends `v0.1.5-best-skill-routing` with one focused goal:

> Identify an anonymous class by testing whether the learned model produces the
> expected binary `y` behavior for that candidate class.

The implementation deliberately does **not** use class-weight probability
similarity as the identification criterion.

## Binary reverse-engineering model

For a candidate class `c`, the reverse-engineering primitive returns:

```text
y_hat = reverse_engineer_y(sample, c)
```

where `y_hat` is strictly `True` or `False`.

The current low-level adapter uses the classifier prediction as a baseline:
`True` means the predicted global class is `c`. The plugin accepts an injected
`reverse_engineer_y_fn`, so the actual research reverse-engineering algorithm
can replace this adapter without changing persistence, routing, or tests.

## Known-class validation

Reference samples for a known class have a known expected value:

```text
expected_y = True
```

The plugin stores the binary predictions produced by the reverse-engineering
primitive for those samples. `reference_accuracy` records how often the
primitive reproduces the known expected value.

This separates two questions:

1. Can the reverse-engineering method reproduce the expected `y` for known
   class samples?
2. Does an anonymous sample produce the same expected `y` for a candidate
   class?

## Anonymous identification

For every anonymous sample, each persistent class fingerprint is tested.
The routing result is intentionally discrete:

- `IDENTIFIED`: exactly one candidate class is compatible.
- `AMBIGUOUS`: more than one candidate class is compatible.
- `FAILED`: no candidate class is compatible.

The router identifies the **class first** and only then resolves the existing
canonical `class -> skill` mapping. A class is never remapped, and one skill
can own multiple classes.

No target label, task ID, or experience ID is consumed by anonymous routing.

## Persistent references

Each `ClassBehaviorRecord` stores:

- global `class_id`;
- canonical `skill_id`;
- skill generation/version;
- deterministic reference inputs;
- binary `reference_y` values;
- expected `y`.

If mutable `REUSE` changes a skill, every class mastered by that skill gets a
new generation. The original reference inputs are retained and reused when
refreshing the fingerprints. Unchanged skills are not refreshed. `SCRATCH`
creates the initial fingerprint for each newly mastered class.

## Diagnostics

Every anonymous routing record contains enough information to reconstruct the
decision:

- sample index;
- final status;
- selected class and skill, when identified;
- every candidate class and skill;
- candidate predicted `y`;
- candidate expected `y`;
- candidate reference accuracy;
- candidate correctness.

This makes `IDENTIFIED`, `AMBIGUOUS`, and `FAILED` decisions directly
inspectable instead of reducing the result to a skill index.

## Tests

The focused test suite covers:

- binary `y` generation and global class IDs;
- known-class expected-vs-predicted comparison;
- exact identification and failed identification;
- ambiguous identification;
- per-sample routing;
- multi-class skills with distinct class identities;
- mutable versus immutable `REUSE` refresh behavior;
- invalidation of all classes belonging to a changed skill;
- checkpoint round-trip and legacy checkpoint loading;
- reconstructable routing diagnostics.

The existing `SkillMemoryPlugin` remains unchanged. This extension is kept
opt-in until the binary reverse-engineering primitive is validated against the
benchmark.
