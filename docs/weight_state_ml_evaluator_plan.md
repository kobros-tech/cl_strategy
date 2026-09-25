# Anonymous Two-Stage ML Evaluation Plan

## 1. Objective

Implement an independent anonymous evaluation pipeline for Skill Memory.

When:

    eval_routing="none"

evaluation must use:

    x -> ML-1 -> predicted omega -> ML-2 -> predicted y

The pipeline is auxiliary evaluation only. It must not replace or modify Skill Memory training.

The CL model produces the learned model states omega. ML-1 learns to predict omega from x. ML-2 learns to predict y from omega.

The test label is never provided to either ML model during inference. It is used only afterward to calculate metrics.

---

## 2. Exact meaning of eval_routing="none"

For this experiment:

    eval_routing="none"

means:

1. Do not select a Skill Memory skill for the evaluation sample.
2. Do not use the evaluation label for routing.
3. Run the anonymous two-stage ML evaluator.
4. Give only x_test to ML-1.
5. Pass ML-1's predicted omega directly to ML-2.
6. Use the predicted y for evaluation.
7. Compare predicted y with the true test label only after prediction.

The runtime path is exactly:

    x_test
       |
       v
      ML-1
       |
       v
 predicted omega
       |
       v
      ML-2
       |
       v
 predicted y
       |
       v
 compare with y_test

Probe routing is a separate evaluation mode and must not be changed by this work.

The existing raw-input evaluator remains available for the other evaluation modes. It is not the evaluation path for `eval_routing="none"` after this change.
### Current implementation boundary

The current implementation is intentionally a trajectory-memory diagnostic:

- CL training captures post-step x, omega, y, skill_id, step records.
- WeightStateMLEvaluationPlugin trains fresh ML-1 and ML-2 models from the retained records in before_eval.
- During the anonymous evaluation pass, the plugin receives only x and performs x -> ML-1 -> predicted omega -> ML-2 -> predicted y.
- The current diagnostic trains and evaluates the evaluator from the same retained trajectory memory. This establishes the pipeline semantics, but it is **not** a held-out generalization measurement.
- Held-out x -> omega and omega -> y evaluation is a follow-up experiment and must not be confused with the current same-memory diagnostic.

The evaluator is implemented as an Avalanche plugin. It is not a second continual-learning strategy and it does not participate in CL optimization.


---

## 3. What is learned by each component

There are three distinct learning processes.

### 3.1 Stage 0: Skill Memory / CL produces omega

Existing CL training receives:

    x + y

and performs the normal optimization:

    model
      |
      v
    loss
      |
      v
    optimizer.step()
      |
      v
    learned omega_t

Immediately after each optimizer step, capture the resulting model state.

Do not introduce another learner into this training process.

A captured omega is currently the complete `model.state_dict()`. It is not a class-specific subset of parameters.

### 3.2 Stage 1: ML-1 learns x -> omega

ML-1 is an independent regression model.

Training:

    x -> ML-1 -> omega

Inputs:

- x from the CL training batch.

Targets:

- the complete learned omega produced immediately after the optimizer step for that batch.

The class label is not an ML-1 input.

At anonymous test time:

    x_test -> ML-1 -> predicted omega

ML-1 must therefore predict the same omega representation used by ML-2.

### 3.3 Stage 2: ML-2 learns omega -> y

ML-2 is an independent classification model.

Training:

    omega -> ML-2 -> y

Inputs:

- captured learned omega.

Targets:

- the associated class y.

At anonymous test time:

    predicted omega -> ML-2 -> predicted y

ML-2 must use the benchmark's global class output space.

---

## 4. Exact trajectory data that must be retained

Each retained optimization-step record must contain:

    {
        x: input batch associated with the optimizer step,
        omega: complete post-step model state,
        y: class target for the batch,
        class_id: class identifier,
        skill_id: selected Skill Memory skill identifier,
        step: optimization-step index
    }

The important invariant is:

    x_t <-> omega_t <-> y_t

The omega must be the model state **after** the optimizer update caused by the associated x batch.

The `step` field is the zero-based optimizer-step index from the class-training loop; it is not merely a memory insertion order.

The captured state must be:

- detached;
- cloned;
- moved to CPU;
- immutable with respect to later model updates.

The captured x must also be detached/cloned/moved to CPU so later training cannot mutate the stored input.

Because `train_on_class()` trains one class at a time, the batch has a single class target. Preserve the batch itself rather than inventing a different class assignment.

---

## 5. State retention

Retain only the most recent configurable number of omega records per class.

Strategy parameter:

    weight_state_snapshots_per_class=10

For example, with a limit of 3:

    class 3:
        record_18
        record_19
        record_20

    class 7:
        record_38
        record_39
        record_40

When a new record for a class is added, the oldest record for that class is discarded.

The x, omega, y, skill_id, and step belonging to a retained record must remain paired.

The retention limit applies to optimization-step/state records, not independently to x and omega.

Both SCRATCH and mutable REUSE training paths must capture records.

If `reuse_is_mutable=False` and no optimizer step occurs for a REUSE decision, no new post-step omega record is created for that decision.

---

## 6. Building ML-1 training pairs

A single optimizer step produces one omega for an entire training batch.

Therefore the implementation must preserve the batch-to-state relationship without pretending that each sample produced a different omega.

For ML-1 training, each sample in a retained batch may use the same post-step omega target:

    x_t[0] -> omega_t
    x_t[1] -> omega_t
    ...
    x_t[n] -> omega_t

This converts the retained batch/state record into sample-level `x -> omega` training pairs while preserving the fact that one optimizer step produced the shared omega.

Do not capture a separate omega for every sample unless the training loop actually performs a separate optimizer step for that sample.

This distinction is important for a correct implementation.

---

## 7. Dedicated evaluation memory

Create a dedicated weight-state trajectory memory. It must support both mappings:

    x -> omega

and:

    omega -> y

It must not change the existing raw `EvaluationMemory` semantics.

The trajectory memory should expose enough information to:

1. recover retained x samples and their paired omega targets for ML-1;
2. recover retained omega states and their class targets for ML-2;
3. preserve class, skill, and step metadata for diagnostics.

The raw-input evaluator remains a separate mechanism and must not be silently reused as the implementation of the weight-state pipeline.

---

## 8. Omega representation

The current Skill Memory representation is a complete `model.state_dict()`.

For ML evaluation, convert the captured state deterministically into a flat numeric vector.

Requirements:

- deterministic ordering of state keys;
- consistent representation for every snapshot;
- same representation dimension for ML-1 targets and ML-2 inputs;
- floating-point tensors represented consistently;
- no dependence on dictionary insertion order.

The implementation must not silently switch from complete-state omega to a class-specific parameter subset.

If a future experiment introduces another omega representation, that should be an explicit configuration/design change rather than an implicit optimization.

---

## 9. ML-1 model

ML-1 is a regression model:

    input:  x
    target: omega_vector

The output dimension is exactly the flattened omega dimension.

The initial implementation should use a lightweight one-hidden-layer MLP where practical.

The training objective must be a regression loss suitable for continuous omega targets, such as MSE.

ML-1 must be trained independently from the CL model.

ML-1 must not receive:

- class labels;
- skill IDs;
- test labels;
- Skill Memory routing decisions.

---

## 10. ML-2 model

ML-2 is a classification model:

    input:  omega_vector
    target: y

The output dimension is the benchmark's global number of classes. The strategy accepts `weight_state_num_classes` explicitly; when omitted, it infers the global class count from the main model's final `nn.Linear` classification layer.

The initial implementation should use a lightweight one-hidden-layer MLP where practical.

ML-2 must be trained independently from the CL model.

ML-2 must not receive:

- class labels as input features;
- skill IDs;
- test labels;
- Skill Memory routing decisions.

The class is used only as the supervised training target and later as the held-out evaluation target.

---

## 11. Evaluation lifecycle

The anonymous evaluator must be trained before anonymous test evaluation.

The intended lifecycle is:

### During CL training

    training batch x, y
          |
          v
      CL training
          |
          v
    optimizer.step()
          |
          v
    capture x, omega, y, skill_id, step

### Before anonymous evaluation

1. Read the retained trajectory memory.
2. Build ML-1 training pairs:
   `x -> omega`.
3. Build ML-2 training pairs:
   `omega -> y`.
4. Train fresh independent ML-1.
5. Train fresh independent ML-2.

The evaluator must not continue training during the anonymous test pass.

### During anonymous evaluation

For each test batch:

    x_test
       |
       v
      ML-1
       |
       v
 predicted omega
       |
       v
      ML-2
       |
       v
 predicted y

Only after prediction is produced:

    predicted y <-> true y_test

for metrics.

No test label may influence ML-1, ML-2, omega prediction, routing, or model selection during that pass.

---

## 12. Required diagnostics

The implementation should expose at least these distinguishable results:

### Stage 1 diagnostic

    x -> predicted omega

Measure a suitable omega reconstruction/regression metric.

This tells us whether ML-1 can recover the learned state from anonymous input.

### Stage 2 diagnostic using true omega

    true omega -> ML-2 -> y

This measures the class information contained in the learned states without ML-1 error.

### End-to-end anonymous result

    x -> ML-1 -> predicted omega -> ML-2 -> predicted y

This is the primary result for `eval_routing="none"`.

The distinction between true-omega and predicted-omega evaluation is important because it separates ML-1 reconstruction error from ML-2 classification capability.

---

## 13. Skill Memory behavior that must remain unchanged

Do not change:

- REUSE/SCRATCH decisions;
- canonical class-to-skill mapping;
- stored Skill Memory semantics;
- main CL optimization;
- optimizer behavior;
- probe routing;
- non-`none` evaluation routing modes.

The new evaluator is auxiliary evaluation only.

It must not become a second continual learner.

It must not replace Skill Memory training.

It must not feed predicted omega back into Skill Memory.

---

## 14. Existing raw evaluator

The existing raw-input evaluator is not the implementation of the new anonymous pipeline.

It may remain in the codebase for existing diagnostics or separate experiments, but:

    eval_routing="none"

must select the new:

    x -> ML-1 -> omega -> ML-2 -> y

path.

Existing raw evaluator tests should remain valid unless their API is explicitly changed for another reason.

---

## 15. Required implementation tests

Tests must verify behavior, not merely object construction.

### Capture

1. A state is captured immediately after `optimizer.step()` and the test proves it matches the post-step model state.
2. The captured omega reflects the post-step model state.
3. The associated x batch is retained.
4. x and omega are immutable after later model updates.
5. y, class_id, skill_id, and step remain attached.
6. SCRATCH captures the expected records.
7. Mutable REUSE captures the expected records.
8. The configured per-class retention limit is enforced.

### ML-1

9. ML-1 input shape matches the dataset input shape.
10. ML-1 output shape matches the flattened omega dimension.
11. ML-1 training uses x as input and omega as target.
12. ML-1 does not require class labels.

### ML-2

13. ML-2 input shape matches the flattened omega dimension.
14. ML-2 output shape matches the global class count.
15. ML-2 training uses omega as input and y as target.
16. ML-2 does not require class labels as input features.

### End-to-end

17. `eval_routing="none"` selects the anonymous two-stage evaluator.
18. Anonymous evaluation executes exactly:

        x -> ML-1 -> predicted omega -> ML-2 -> predicted y

19. Test labels are used only for metrics after prediction.
20. The true-omega -> ML-2 diagnostic and predicted-omega end-to-end result are distinguishable.
21. Existing Skill Memory training invariants remain unchanged.
22. Existing probe and other routing modes remain unchanged.
23. Existing raw evaluator tests remain valid.

---

## 16. Non-goals

This implementation does not:

- redesign Skill Memory;
- change REUSE/SCRATCH behavior;
- change probe routing;
- introduce class-aware routing into `none`;
- train another continual learner;
- use test labels to infer omega;
- replace the CL model with ML-1 or ML-2;
- claim that omega is a physically isolated class-specific parameter subset;
- silently change the omega representation.

---

## 17. Scientific question

The experiment asks:

> Can an anonymous input be mapped to the learned Skill Memory state, and can that learned state then be used by an independent classifier to recover the associated class?

The end-to-end answer is measured by:

    x -> ML-1 -> omega -> ML-2 -> y

The two diagnostic stages explain where performance is gained or lost:

    x -> ML-1 -> omega

and:

    omega -> ML-2 -> y

---

## 18. Follow-up experiments

Only after the basic pipeline is correct, investigate:

- different numbers of retained states per class;
- different omega representations or selected layers;
- final-state-only versus trajectory states;
- alternative ML-1 architectures;
- alternative ML-2 architectures;
- per-skill versus per-class analysis;
- held-out x/omega pairs for ML-1;
- held-out omega/y pairs for ML-2;
- propagation of ML-1 reconstruction error into final anonymous accuracy;
- memory and compute cost of complete-state omega targets.

These are follow-up experiments, not prerequisites for implementing the core pipeline.
