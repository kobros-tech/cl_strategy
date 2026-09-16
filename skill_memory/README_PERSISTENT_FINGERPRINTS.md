# Persistent class identification with normal ML reverse engineering

This branch extends `v0.1.5-best-skill-routing` with one focused goal:

> Identify an anonymous class with a standalone ML model that learns to predict
> binary `y` from the raw sample and a frozen candidate classifier weight.

The reverse-engineering model is intentionally separated from continual
learning. It does not train on, update, or depend on the live CL feature
representation.

## Normal ML reverse engineering

For each mastered class, the plugin keeps deterministic reference samples and
the classifier weight/bias from the corresponding **frozen skill generation**.
Those are used to build ordinary supervised ML pairs:

```text
input  = [flattened sample x, candidate weight W_c, candidate bias b_c]
target = 1 if candidate class is the reference sample's class else 0
```

A small PyTorch MLP learns:

```text
P(y=1 | x, W_c, b_c)
```

This is a genuine learned reverse-engineering model rather than a handcrafted
`argmax`, cosine, or score-threshold fingerprint.

The reverse model is protected from CL drift in two ways:

1. candidate weights are copied from a frozen skill-generation snapshot;
2. the reverse model consumes the raw input rather than the mutable CL feature
   extractor.

Therefore later `REUSE`, `CLONE`, or `SCRATCH` training cannot silently change
the reverse model's input representation. When a mutable skill changes, a new
skill generation is created, its frozen candidate weights are refreshed, and the
normal ML reverse model is retrained from the new reference set.

The older `reverse_engineer_y_fn` injection remains available for experiments,
but the benchmark router uses the standalone normal-ML path.

## Known-class calibration

Reference samples provide the only labels used to train the reverse model.
For every reference sample, its canonical class is known while the model is
being calibrated. Negative candidate pairs are generated from the other known
class weights.

Those labels are never supplied during anonymous evaluation.

## Anonymous identification

For an anonymous sample, every persistent class candidate is evaluated by the
same trained reverse model:

```text
sample x + frozen candidate W_c,b_c
                |
                v
       normal ML reverse model
                |
                v
       P(y=1 | x, candidate)
```

Candidates are converted to binary-compatible candidates using the learned
probability, not the classifier's multiclass argmax. A single compatible
candidate is identified directly. Multiple compatible candidates are resolved
from the learned probability ordering; if the learned evidence does not form a
clear separation, routing remains `AMBIGUOUS`. If no candidate is compatible,
the result is `FAILED`.

No evaluation target label, task ID, or experience ID is consumed by the
reverse model or routing decision.

## Persistent references

Each `ClassBehaviorRecord` stores:

- global `class_id`;
- canonical `skill_id`;
- skill generation/version;
- deterministic reference inputs;
- binary reference diagnostics;
- expected `y`.

If mutable `REUSE` changes a skill, every class mastered by that skill receives
a new generation. The original reference inputs are retained. The frozen
classifier state for that generation is used to rebuild the normal ML training
pairs. `SCRATCH` creates the initial records for newly mastered classes.

## Diagnostics

Each anonymous routing record contains:

- sample index;
- final routing status;
- selected class and skill, when identified;
- every candidate class and skill;
- reverse-engineering probability;
- learned binary compatibility;
- reference accuracy;
- final classifier prediction/correctness when the evaluation plugin adds it.

This separates the reverse-engineering decision from the final classifier
prediction, so a bad route can be diagnosed without treating classifier
forgetting as evidence that the reverse model itself failed.

## Tests and benchmark

The focused reverse-engineering tests verify that the standalone model can
learn candidate identity from sample/weight pairs and does not mutate frozen
candidate weights.

`skill_memory/tests/demo_splitmnist_weight_reverse_engineering.py` runs the
SplitMNIST benchmark through the persistent fingerprint plugin. The benchmark
router does not pass experience IDs or evaluation labels to the reverse model.

The original `SkillMemoryPlugin` remains available separately for the older
probe-routing experiments.
