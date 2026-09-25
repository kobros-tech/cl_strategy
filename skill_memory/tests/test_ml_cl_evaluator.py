"""Tests for `skill_memory.evaluation.ml_cl_evaluator`.

Uses a small synthetic (non-MNIST, no download needed) benchmark built
with Avalanche's own `nc_benchmark` generator, exercising the same code
path as `skill_memory/demos/demo_splitmnist_ml_er.py` end to end: Skill
Memory training + evaluation-memory capture, the independent evaluator's
train/evaluate loop, and ML evaluation through the Avalanche plugin.
"""

import pytest
import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from torch.utils.data import TensorDataset

from skill_memory import SkillMemory
from skill_memory.evaluation.ml_cl_evaluator import (
    EvaluationMemory,
    EvaluationMemoryPlugin,
    MLEvaluationPlugin,
    aggregate_experience_metrics,
    build_evaluator,
    compute_class_forgetting,
    compute_peak_class_forgetting,
    consolidate_evaluation_memory,
    evaluate_model_by_class,
    make_loader,
    train_evaluator,
)
from skill_memory.evaluation.weight_state_ml_evaluator import (
    WeightEvaluationMemory,
    WeightStateMLEvaluationPlugin,
    build_weight_state_evaluator,
    sketch_weight_state,
    build_weight_state_regressor,
    consolidate_weight_evaluation_memory,
    consolidate_weight_state_memory,
    evaluate_weight_state_memory,
    flatten_weight_state,
    train_weight_state_evaluator,
    train_weight_state_regressor,
)


def _synthetic_benchmark(
    n_classes: int,
    n_experiences: int,
    n_per_class: int = 40,
):
    """A synthetic, well-separated benchmark with no ordering degeneracy.

    Each class perturbs its own feature dimension (`class_id % n_features`),
    not a shared scalar offset across every dimension - the latter makes
    every class linearly ordered along one axis, so a binary "is this class
    0" classifier accidentally also fires for classes far along that same
    axis, triggering REUSE decisions that have nothing to do with what
    these tests are actually checking.
    """
    torch.manual_seed(0)
    n_features = 6
    xs, ys = [], []
    for class_id in range(n_classes):
        offsets = torch.zeros(n_features)
        offsets[class_id % n_features] = 6.0 * (1 + class_id // n_features)
        xs.append(torch.randn(n_per_class, n_features) * 0.5 + offsets)
        ys.append(
            torch.full(
                (n_per_class,),
                class_id,
                dtype=torch.long,
            )
        )
    x = torch.cat(xs)
    y = torch.cat(ys)
    train_ds = TensorDataset(x, y)
    test_ds = TensorDataset(x, y)
    return nc_benchmark(
        train_ds,
        test_ds,
        n_experiences=n_experiences,
        task_labels=False,
        seed=0,
        shuffle=False,
    )


def test_consolidate_evaluation_memory_merges_by_class_across_experiences():
    memory = [
        EvaluationMemory(
            torch.zeros(2, 3),
            torch.tensor([1, 1]),
            class_id=1,
        ),
        EvaluationMemory(
            torch.ones(3, 3),
            torch.tensor([2, 2, 2]),
            class_id=2,
        ),
        EvaluationMemory(
            torch.full((1, 3), 5.0),
            torch.tensor([1]),
            class_id=1,
        ),
    ]
    consolidated = consolidate_evaluation_memory(memory)
    by_class = {item.class_id: item for item in consolidated}
    assert set(by_class) == {1, 2}
    assert by_class[1].size == 3
    assert by_class[2].size == 3


def test_make_loader_rejects_empty_memory():
    try:
        make_loader(
            [],
            batch_size=4,
            shuffle=False,
        )
    except RuntimeError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for empty memory")


def test_forgetting_definitions_diverge_on_a_dip_then_recovery_then_drop():
    """Worked example: a class that goes 60% -> 90% -> 70%.

    Acquisition-relative forgetting (`compute_class_forgetting`) compares
    only against the acquisition-time accuracy (60%), so a final accuracy
    of 70% (>= 60%) scores zero forgetting even though the class fell 20
    points from its peak of 90%. Peak-relative forgetting
    (`compute_peak_class_forgetting`, the standard continual-learning
    definition) catches exactly that drop. The two metrics must therefore
    give different answers on this example - if they ever agree here,
    one of them has been implemented wrong.
    """
    accuracy_history = [
        {0: 0.60},  # experience 0: class 0 introduced, accuracy on introduction = 60%
        {0: 0.90},  # experience 1: class 0's accuracy rises to 90%
        {0: 0.70},  # experience 2: class 0's accuracy falls to 70%
    ]
    class_to_experience = {0: 0}

    acquisition_relative = compute_class_forgetting(
        accuracy_history,
        class_to_experience,
        num_experiences=3,
    )
    peak_relative = compute_peak_class_forgetting(
        accuracy_history,
        class_to_experience,
        num_experiences=3,
    )

    assert acquisition_relative[0] == 0.0
    assert peak_relative[0] == pytest.approx(0.20)
    assert acquisition_relative[0] != peak_relative[0]


def test_train_evaluator_reduces_loss_on_a_separable_synthetic_problem():
    torch.manual_seed(0)
    memory = [
        EvaluationMemory(
            torch.randn(20, 6) + class_id * 4.0,
            torch.full(
                (20,),
                class_id,
                dtype=torch.long,
            ),
            class_id=class_id,
        )
        for class_id in range(3)
    ]
    model, optimizer, criterion = build_evaluator(
        lambda: SimpleMLP(
            input_size=6,
            hidden_size=16,
            num_classes=3,
        ),
        device=torch.device("cpu"),
        learning_rate=0.1,
    )

    loader = make_loader(
        memory,
        batch_size=8,
        shuffle=False,
    )

    model.eval()
    with torch.no_grad():
        initial_loss = sum(float(criterion(model(x), y)) for x, y in loader) / len(
            loader
        )

    train_evaluator(
        model,
        optimizer,
        criterion,
        memory,
        batch_size=8,
        epochs=20,
        device=torch.device("cpu"),
        seed=0,
    )

    model.eval()
    loader = make_loader(
        memory,
        batch_size=8,
        shuffle=False,
    )
    with torch.no_grad():
        final_loss = sum(float(criterion(model(x), y)) for x, y in loader) / len(loader)

    assert final_loss < initial_loss


def test_evaluation_memory_plugin_captures_bounded_per_class_samples():
    benchmark = _synthetic_benchmark(
        n_classes=4,
        n_experiences=2,
        n_per_class=40,
    )
    model = SimpleMLP(
        input_size=6,
        hidden_size=8,
        num_classes=4,
    )
    plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        eval_routing="none",
        eval_memory_per_class=5,
        eval_memory_seed=0,
        verbose=False,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=0.05,
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        plugins=[plugin],
    )
    for exp in benchmark.train_stream:
        strategy.train(exp)

    assert {m.class_id for m in plugin.eval_memory} == {
        0,
        1,
        2,
        3,
    }
    # eval_memory_per_class=5 must bound each class's retained sample count.
    assert all(m.size == 5 for m in plugin.eval_memory)


def test_evaluate_model_by_class_and_aggregate_and_forgetting_end_to_end():
    benchmark = _synthetic_benchmark(
        n_classes=4,
        n_experiences=2,
        n_per_class=40,
    )
    model = SimpleMLP(
        input_size=6,
        hidden_size=8,
        num_classes=4,
    )
    plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        eval_routing="none",
        eval_memory_per_class=10,
        eval_memory_seed=0,
        verbose=False,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=0.05,
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        plugins=[plugin],
    )
    class_to_experience: dict[int, int] = {}
    accuracy_history = []
    for train_index, exp in enumerate(benchmark.train_stream):
        strategy.train(exp)
        for class_id in exp.classes_in_this_experience:
            class_to_experience[int(class_id)] = train_index

        memory = consolidate_evaluation_memory(plugin.eval_memory)
        evaluator, optimizer, criterion = build_evaluator(
            lambda: SimpleMLP(
                input_size=6,
                hidden_size=8,
                num_classes=4,
            ),
            device=torch.device("cpu"),
            learning_rate=0.1,
        )
        train_evaluator(
            evaluator,
            optimizer,
            criterion,
            memory,
            batch_size=8,
            epochs=10,
            device=torch.device("cpu"),
            seed=0,
        )
        class_results = evaluate_model_by_class(
            evaluator,
            benchmark.test_stream,
            train_index,
            batch_size=16,
            device=torch.device("cpu"),
        )
        # Every class seen so far must have a result, and results are
        # class-level (not experience-level) as advertised.
        assert set(class_results) == set(class_to_experience)

        losses, accuracies = aggregate_experience_metrics(
            class_results,
            benchmark.test_stream,
            train_index,
        )
        assert len(losses) == len(accuracies) == train_index + 1

        accuracy_history.append(
            {class_id: values["accuracy"] for class_id, values in class_results.items()}
        )

    forgetting = compute_class_forgetting(
        accuracy_history,
        class_to_experience,
        len(benchmark.train_stream),
    )
    assert forgetting.shape == (len(benchmark.train_stream),)
    assert (forgetting >= 0).all()


def test_ml_evaluation_plugin_trains_and_reports_class_metrics():
    benchmark = _synthetic_benchmark(
        n_classes=4,
        n_experiences=2,
        n_per_class=40,
    )
    model = SimpleMLP(
        input_size=6,
        hidden_size=8,
        num_classes=4,
    )
    memory_plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        eval_routing="none",
        eval_memory_per_class=10,
        eval_memory_seed=0,
        verbose=False,
    )
    ml_plugin = MLEvaluationPlugin(
        memory_plugin=memory_plugin,
        model_factory=lambda: SimpleMLP(
            input_size=6,
            hidden_size=8,
            num_classes=4,
        ),
        epochs=10,
        batch_size=8,
        learning_rate=0.1,
        seed=0,
        verbose=False,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=0.05,
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        plugins=[
            memory_plugin,
            ml_plugin,
        ],
    )
    for experience in benchmark.train_stream:
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)
    results = ml_plugin.results()
    assert set(results["final_class_accuracy"]) == {
        0,
        1,
        2,
        3,
    }
    assert set(results["final_class_loss"]) == {
        0,
        1,
        2,
        3,
    }
    assert results["mean_final_accuracy"] > 0.95
    assert results["mean_final_loss"] >= 0.0
    assert ml_plugin.current_accuracy == results["final_class_accuracy"]
    assert ml_plugin.current_loss == results["final_class_loss"]
    assert len(results["diagonal_accuracy"]) == 2
    assert len(results["diagonal_loss"]) == 2
    assert results["peak_forgetting"].shape == (2,)


def test_ml_evaluator_uses_global_class_output_space():
    benchmark = _synthetic_benchmark(
        n_classes=4,
        n_experiences=1,
        n_per_class=40,
    )
    model = SimpleMLP(
        input_size=6,
        hidden_size=8,
        num_classes=4,
    )
    memory_plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        eval_routing="none",
        eval_memory_per_class=10,
        eval_memory_seed=0,
        verbose=False,
    )
    ml_plugin = MLEvaluationPlugin(
        memory_plugin=memory_plugin,
        model_factory=lambda: SimpleMLP(
            input_size=6,
            hidden_size=8,
            num_classes=4,
        ),
        epochs=10,
        batch_size=8,
        learning_rate=0.1,
        seed=0,
        verbose=False,
    )

    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=0.05,
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        plugins=[
            memory_plugin,
            ml_plugin,
        ],
    )
    for experience in benchmark.train_stream:
        strategy.train(experience)

    strategy.eval(benchmark.test_stream)
    evaluator = ml_plugin.evaluator_model
    assert evaluator is not None
    x = benchmark.test_stream[0].dataset[0][0]
    x = x.unsqueeze(0)
    with torch.no_grad():
        logits = evaluator(x)

    assert logits.shape == (1, 4)


def test_ml_evaluation_does_not_modify_main_model():
    benchmark = _synthetic_benchmark(
        n_classes=2,
        n_experiences=1,
        n_per_class=20,
    )
    model = SimpleMLP(
        input_size=6,
        hidden_size=8,
        num_classes=2,
    )
    memory_plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=5),
        eval_routing="none",
        eval_memory_per_class=5,
        eval_memory_seed=0,
        verbose=False,
    )
    ml_plugin = MLEvaluationPlugin(
        memory_plugin=memory_plugin,
        model_factory=lambda: SimpleMLP(
            input_size=6,
            hidden_size=8,
            num_classes=2,
        ),
        epochs=10,
        batch_size=8,
        learning_rate=0.1,
        seed=0,
        verbose=False,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=0.05,
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        plugins=[
            memory_plugin,
            ml_plugin,
        ],
    )
    for experience in benchmark.train_stream:
        strategy.train(experience)

    before = {key: value.clone() for key, value in model.state_dict().items()}
    strategy.eval(benchmark.test_stream)
    after = model.state_dict()
    for key in before:
        assert torch.equal(
            before[key],
            after[key],
        )

    assert ml_plugin.evaluator_model is not None
    assert ml_plugin.evaluator_model is not model


def _tiny_state(value: float) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([[value, value + 1.0]]),
        "bias": torch.tensor([value]),
    }


def _tiny_batch(
    value: float,
    class_id: int,
    n: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.full((n, 2), value)
    targets = torch.full((n,), class_id, dtype=torch.long)
    return inputs, targets


def test_weight_evaluation_memory_keeps_last_ten_snapshots_per_class():
    memory = WeightEvaluationMemory(max_snapshots_per_class=10)

    for step in range(12):
        inputs, targets = _tiny_batch(float(step), class_id=7)
        memory.add(
            _tiny_state(float(step)),
            inputs=inputs,
            targets=targets,
            class_id=7,
            skill_id=3,
        )

    snapshots = memory.snapshots_for_class(7)

    assert len(snapshots) == 10
    assert [snapshot.step for snapshot in snapshots] == list(range(2, 12))
    assert all(snapshot.class_id == 7 for snapshot in snapshots)
    assert all(snapshot.skill_id == 3 for snapshot in snapshots)


def test_weight_evaluation_memory_snapshots_are_immutable():
    memory = WeightEvaluationMemory(max_snapshots_per_class=10)
    state = _tiny_state(1.0)
    inputs, targets = _tiny_batch(1.0, class_id=2)

    memory.add(
        state,
        inputs=inputs,
        targets=targets,
        class_id=2,
        skill_id=4,
    )
    state["weight"].fill_(99.0)
    inputs.fill_(99.0)
    targets.fill_(9)

    stored = memory.snapshots_for_class(2)[0]
    assert torch.equal(stored.state["weight"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(stored.inputs, torch.ones(2, 2))
    assert torch.equal(stored.targets, torch.tensor([2, 2]))


def test_weight_state_memory_preserves_x_omega_y_pairing():
    memory = WeightEvaluationMemory(max_snapshots_per_class=10)

    first_x, first_y = _tiny_batch(1.0, class_id=0)
    second_x, second_y = _tiny_batch(2.0, class_id=1)

    memory.add(
        _tiny_state(10.0),
        inputs=first_x,
        targets=first_y,
        class_id=0,
        skill_id=4,
    )
    memory.add(
        _tiny_state(20.0),
        inputs=second_x,
        targets=second_y,
        class_id=1,
        skill_id=7,
    )

    snapshots = memory.snapshots()
    assert torch.equal(snapshots[0].inputs, first_x)
    assert torch.equal(snapshots[0].targets, first_y)
    assert torch.equal(
        flatten_weight_state(snapshots[0].state),
        torch.tensor([10.0, 10.0, 11.0]),
    )
    assert snapshots[0].skill_id == 4

    assert torch.equal(snapshots[1].inputs, second_x)
    assert torch.equal(snapshots[1].targets, second_y)
    assert snapshots[1].skill_id == 7


def test_consolidate_weight_state_memory_builds_both_training_stages():
    memory = WeightEvaluationMemory(max_snapshots_per_class=10)
    inputs, targets = _tiny_batch(3.0, class_id=1)

    memory.add(
        _tiny_state(5.0),
        inputs=inputs,
        targets=targets,
        class_id=1,
        skill_id=2,
    )

    ml1_x, ml1_targets, ml2_x, ml2_y = consolidate_weight_state_memory(
        memory,
        representation_size=3,
    )

    assert ml1_x.shape == (2, 2)
    assert ml1_targets.shape == (2, 3)
    assert ml2_x.shape == (1, 3)
    assert ml2_y.tolist() == [1]
    assert torch.equal(ml1_targets[0], ml1_targets[1])


def test_ml1_regressor_learns_x_to_omega_on_synthetic_trajectory():
    memory = WeightEvaluationMemory(max_snapshots_per_class=10)

    for class_id in range(2):
        for step in range(5):
            value = float(class_id * 5 + step)
            inputs, targets = _tiny_batch(value, class_id=class_id)
            memory.add(
                _tiny_state(value),
                inputs=inputs,
                targets=targets,
                class_id=class_id,
                skill_id=class_id,
            )

    model = build_weight_state_regressor(2, 3, hidden_size=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_weight_state_regressor(
        model,
        optimizer,
        memory,
        batch_size=4,
        epochs=100,
        device=torch.device("cpu"),
        seed=0,
        representation_size=3,
    )

    ml1_x, ml1_targets, _, _ = consolidate_weight_state_memory(
        memory,
        representation_size=3,
    )
    with torch.no_grad():
        prediction = model(ml1_x)

    assert prediction.shape == ml1_targets.shape
    assert torch.mean((prediction - ml1_targets) ** 2).item() < 1.0


def test_weight_state_sketch_has_bounded_representation():
    state = {
        "large": torch.arange(10_000, dtype=torch.float32),
    }

    sketch = sketch_weight_state(state, representation_size=32, seed=0)

    assert sketch.shape == (32,)
    assert sketch.dtype == torch.float32
    assert torch.isfinite(sketch).all()


def test_weight_state_evaluator_learns_class_from_anonymous_weight_states():
    memory = WeightEvaluationMemory(max_snapshots_per_class=10)

    for class_id in range(2):
        for step in range(10):
            value = float(class_id * 5 + step * 0.01)
            inputs, targets = _tiny_batch(value, class_id=class_id)
            memory.add(
                _tiny_state(value),
                inputs=inputs,
                targets=targets,
                class_id=class_id,
                skill_id=class_id,
            )

    inputs, targets = consolidate_weight_evaluation_memory(
        memory,
        representation_size=3,
    )

    assert inputs.shape[0] == 20
    assert inputs.shape[1] == 3
    assert set(targets.tolist()) == {0, 1}

    model = build_weight_state_evaluator(
        input_size=3,
        num_classes=2,
        hidden_size=8,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    train_weight_state_evaluator(
        model,
        optimizer,
        criterion,
        memory,
        batch_size=8,
        epochs=50,
        device=torch.device("cpu"),
        seed=0,
        representation_size=3,
    )

    result = evaluate_weight_state_memory(
        model,
        memory,
        device=torch.device("cpu"),
        representation_size=3,
    )

    assert result["accuracy"] >= 0.9


def test_skill_memory_plugin_captures_post_step_states_with_inputs():
    benchmark = _synthetic_benchmark(
        n_classes=2,
        n_experiences=1,
        n_per_class=12,
    )
    memory = WeightEvaluationMemory(max_snapshots_per_class=10)
    captured: list[tuple[int, int, int]] = []

    def callback(model, inputs, targets, class_id, skill_id, step):
        memory.add(
            model.state_dict(),
            inputs=inputs,
            targets=targets,
            class_id=class_id,
            skill_id=skill_id,
            step=step,
        )
        captured.append((class_id, skill_id, step))

    plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        eval_routing="none",
        class_train_epochs=2,
        class_train_batch_size=3,
        weight_state_callback=callback,
        verbose=False,
    )
    model = SimpleMLP(
        input_size=6,
        hidden_size=8,
        num_classes=2,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=0.05,
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,
        plugins=[plugin],
    )

    post_step_states = []
    original_step = strategy.optimizer.step

    def wrapped_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        post_step_states.append(
            {
                key: value.detach().cpu().clone()
                for key, value in strategy.model.state_dict().items()
            }
        )
        return result

    strategy.optimizer.step = wrapped_step

    for experience in benchmark.train_stream:
        strategy.train(experience)

    assert len(captured) > 0
    assert len(post_step_states) == len(memory.snapshots())
    for snapshot, post_step_state in zip(
        memory.snapshots(), post_step_states, strict=True
    ):
        assert all(
            torch.equal(snapshot.state[key], value)
            for key, value in post_step_state.items()
        )
    assert [step for _, _, step in captured] == [
        snapshot.step for snapshot in memory.snapshots()
    ]
    assert set(memory.classes()) == {0, 1}
    assert all(
        len(memory.snapshots_for_class(class_id)) <= 10 for class_id in memory.classes()
    )
    assert all(
        snapshot.inputs.shape[0] == snapshot.targets.shape[0]
        for snapshot in memory.snapshots()
    )


def test_mutable_reuse_captures_weight_states():
    benchmark = _synthetic_benchmark(n_classes=1, n_experiences=1, n_per_class=12)
    memory = WeightEvaluationMemory(max_snapshots_per_class=10)

    def callback(model, inputs, targets, class_id, skill_id, step):
        memory.add(
            model.state_dict(),
            inputs=inputs,
            targets=targets,
            class_id=class_id,
            skill_id=skill_id,
            step=step,
        )

    plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        eval_routing="none",
        class_train_epochs=1,
        class_train_batch_size=4,
        reuse_is_mutable=True,
        force_decision="scratch",
        weight_state_callback=callback,
        verbose=False,
    )
    model = SimpleMLP(input_size=6, hidden_size=8, num_classes=1)
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=8,
        train_epochs=1,
        plugins=[plugin],
    )

    experience = benchmark.train_stream[0]
    strategy.train(experience)
    first_count = len(memory.snapshots_for_class(0))
    plugin.force_decision = plugin.REUSE
    strategy.train(experience)

    snapshots = memory.snapshots_for_class(0)
    assert first_count > 0
    assert len(snapshots) > first_count
    assert all(snapshot.skill_id == snapshots[0].skill_id for snapshot in snapshots)


def test_anonymous_two_stage_plugin_replaces_evaluation_output():
    benchmark = _synthetic_benchmark(
        n_classes=2,
        n_experiences=1,
        n_per_class=12,
    )
    memory = WeightEvaluationMemory(max_snapshots_per_class=10)

    def callback(model, inputs, targets, class_id, skill_id, step):
        memory.add(
            model.state_dict(),
            inputs=inputs,
            targets=targets,
            class_id=class_id,
            skill_id=skill_id,
            step=step,
        )

    plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        eval_routing="none",
        class_train_epochs=1,
        class_train_batch_size=4,
        weight_state_callback=callback,
        verbose=False,
    )
    anonymous = WeightStateMLEvaluationPlugin(
        memory=memory,
        num_classes=3,
        epochs=2,
        batch_size=4,
        ml1_epochs=2,
        ml1_batch_size=4,
        verbose=False,
    )
    model = SimpleMLP(input_size=6, hidden_size=8, num_classes=2)
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=8,
        train_epochs=1,
        eval_mb_size=8,
        plugins=[plugin, anonymous],
    )

    for experience in benchmark.train_stream:
        strategy.train(experience)

    strategy.eval(benchmark.test_stream)
    result = anonymous.current_result

    assert anonymous.ml1_model is not None
    assert anonymous.model is not None
    assert anonymous.model[-1].out_features == 3
    assert "true_omega_accuracy" in result
    assert "end_to_end_accuracy" in result
    assert "final_class_accuracy" in result
    assert set(result["final_class_accuracy"]) == {0, 1}
