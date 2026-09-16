# Persistent binary class-behavior identification

This branch extends `v0.1.5-best-skill-routing` with one focused goal:

> Identify an anonymous class by reverse-engineering the learned classifier
> weights and testing the resulting binary `y` behavior for that candidate class.

The implementation does **not** use the benchmark experience ID or target label
for anonymous routing.

## Weight-based reverse engineering

For the default research path, a candidate class is evaluated from the stored
skill's learned parameters rather than by treating `model(x)` logits as the
reverse-engineering algorithm.

For an Avalanche `IncrementalClassifier`, the plugin captures the learned
feature representation `h` immediately before the classifier and explicitly
reconstructs:

```text
scores = h @ W.T + b
```

where `W` and `b` are the persisted classifier weights and bias. The candidate
binary behavior is then the candidate's own one-vs-rest score:

```text
y_hat(c) = scores[:, c] > 0
```

This is deliberately **not** `argmax(scores) == c`. Argmax is a multiclass
decision and guarantees that one candidate is compatible for every sample,
even when that candidate is simply the class produced by a misclassification.
The one-vs-rest rule allows zero, one, or multiple candidates to be compatible;
continuous fingerprint evidence is used only to resolve multiple compatible
candidates.

The package still accepts `reverse_engineer_y_fn` for experiments that need a
different research procedure. The injected function receives precomputed
scores/logits for backward compatibility; the default path is weight-based.

## Known-class validation

Reference samples for a known class have a known expected value:

```text
expected_y = True
```

The plugin persists the binary reference behavior and exposes
`reference_accuracy`. This validates whether the reverse-engineering procedure
can reproduce the expected class behavior before it is used for anonymous
routing.

## Anonymous identification

For every anonymous sample, the router evaluates every persistent class
fingerprint using its canonical skill's learned weights. The route is then:

1. reverse-engineer binary `y` independently for each candidate class;
2. keep candidates whose binary behavior matches the expected fingerprint;
3. if several candidates remain, use persistent feature/margin evidence to
   select a clear top candidate or return `AMBIGUOUS`;
4. resolve the selected class through the persistent canonical `class -> skill`
   map;
5. load the selected skill for the final prediction.

The routing result is intentionally discrete:

- `IDENTIFIED`: exactly one candidate remains after binary behavior and any
  continuous disambiguation.
- `AMBIGUOUS`: multiple binary-compatible candidates cannot be separated by
  the available continuous evidence.
- `FAILED`: no candidate matches the expected binary behavior.

There is no uniform-probability fallback to skill 0. A failed identification
remains failed instead of silently routing to the first slot.

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
- predicted multiclass class and binary `y`;
- candidate class score from the learned weights;
- expected `y`;
- reference accuracy;
- candidate correctness;
- continuous feature/margin evidence when available.

This makes `IDENTIFIED`, `AMBIGUOUS`, and `FAILED` decisions directly
inspectable instead of reducing the result to a skill index.

## Tests and benchmark

The focused tests include an exact reconstruction check: the weight-based
reverse-engineering scores must match the classifier's own affine output, and
the resulting binary `y` must be the candidate score threshold rather than the
classifier argmax.

`skill_memory/tests/demo_splitmnist_weight_reverse_engineering.py` runs the
SplitMNIST benchmark through the persistent fingerprint plugin. The CI demo
uses this plugin directly; it does not pass experience IDs or labels to the
router.

The original `SkillMemoryPlugin` remains available separately for the older
probe-routing experiments.
