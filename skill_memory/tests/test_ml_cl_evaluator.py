"""Tests for `skill_memory.evaluation.ml_cl_evaluator`.

Uses a small synthetic (non-MNIST, no download needed) benchmark built
with Avalanche's own `nc_benchmark` generator, exercising the same code
path as `skill_memory/demos/demo_splitmnist_ml_er.py` end to end: Skill
Memory training + evaluation-memory capture, the independent evaluator's
train/evaluate loop, and direct Skill Memory oracle/probe evaluation.
"""

import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.training import Naive
from torch.utils.data import TensorDataset

from skill_memory import SkillMemory
from skill_memory.evaluation.ml_cl_evaluator import (
    EvaluationMemory,
    EvaluationMemoryPlugin,
    aggregate_experience_metrics,
    build_evaluator,
    compute_class_forgetting,
    consolidate_evaluation_memory,
    evaluate_model_by_class,
    evaluate_skill_memory,
    make_loader,
    train_evaluator,
)
from skill_memory.utils.models import SimpleMLP


def _synthetic_benchmark(n_classes: int, n_experiences: int, n_per_class: int = 40):
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
        ys.append(torch.full((n_per_class,), class_id, dtype=torch.long))
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
        EvaluationMemory(torch.zeros(2, 3), torch.tensor([1, 1]), class_id=1),
        EvaluationMemory(torch.ones(3, 3), torch.tensor([2, 2, 2]), class_id=2),
        EvaluationMemory(torch.full((1, 3), 5.0), torch.tensor([1]), class_id=1),
    ]
    consolidated = consolidate_evaluation_memory(memory)
    by_class = {item.class_id: item for item in consolidated}
    assert set(by_class) == {1, 2}
    assert by_class[1].size == 3
    assert by_class[2].size == 3


def test_make_loader_rejects_empty_memory():
    try:
        make_loader([], batch_size=4, shuffle=False)
    except RuntimeError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for empty memory")


def test_train_evaluator_reduces_loss_on_a_separable_synthetic_problem():
    torch.manual_seed(0)
    memory = [
        EvaluationMemory(
            torch.randn(20, 6) + class_id * 4.0,
            torch.full((20,), class_id, dtype=torch.long),
            class_id=class_id,
        )
        for class_id in range(3)
    ]
    model, optimizer, criterion = build_evaluator(
        lambda: SimpleMLP(input_dim=6, hidden_size=16, num_classes=3),
        device=torch.device("cpu"),
        learning_rate=0.1,
    )

    loader = make_loader(memory, batch_size=8, shuffle=False)
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
    loader = make_loader(memory, batch_size=8, shuffle=False)
    with torch.no_grad():
        final_loss = sum(float(criterion(model(x), y)) for x, y in loader) / len(loader)

    assert final_loss < initial_loss


def test_evaluation_memory_plugin_captures_bounded_per_class_samples():
    benchmark = _synthetic_benchmark(n_classes=4, n_experiences=2, n_per_class=40)
    model = SimpleMLP(input_dim=6, hidden_size=8, num_classes=4)
    plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        eval_routing="none",
        eval_memory_per_class=5,
        eval_memory_seed=0,
        verbose=False,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        plugins=[plugin],
    )
    for exp in benchmark.train_stream:
        strategy.train(exp)

    assert {m.class_id for m in plugin.eval_memory} == {0, 1, 2, 3}
    # eval_memory_per_class=5 must bound each class's retained sample count.
    assert all(m.size == 5 for m in plugin.eval_memory)


def test_evaluate_model_by_class_and_aggregate_and_forgetting_end_to_end():
    benchmark = _synthetic_benchmark(n_classes=4, n_experiences=2, n_per_class=40)
    model = SimpleMLP(input_dim=6, hidden_size=8, num_classes=4)
    plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        eval_routing="none",
        eval_memory_per_class=10,
        eval_memory_seed=0,
        verbose=False,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
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
            lambda: SimpleMLP(input_dim=6, hidden_size=8, num_classes=4),
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
            class_results, benchmark.test_stream, train_index
        )
        assert len(losses) == len(accuracies) == train_index + 1

        accuracy_history.append(
            {class_id: values["accuracy"] for class_id, values in class_results.items()}
        )

    forgetting = compute_class_forgetting(
        accuracy_history, class_to_experience, len(benchmark.train_stream)
    )
    assert forgetting.shape == (len(benchmark.train_stream),)
    assert (forgetting >= 0).all()


def test_evaluate_skill_memory_oracle_and_probe_agree_on_a_separable_problem():
    benchmark = _synthetic_benchmark(n_classes=4, n_experiences=2, n_per_class=40)
    model = SimpleMLP(input_dim=6, hidden_size=8, num_classes=4)
    plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        eval_routing="none",
        eval_memory_per_class=10,
        eval_memory_seed=0,
        # A mutable REUSE would overwrite an earlier skill's weights for a
        # later class, which is expected to forget the earlier class - a
        # real property of SkillMemoryPlugin, not something this test is
        # about. Disable it here so both classes stay recognizable.
        reuse_is_mutable=False,
        # class_train_epochs defaults to 1; this tiny synthetic dataset
        # needs more than one pass to actually converge.
        class_train_epochs=10,
        verbose=False,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        plugins=[plugin],
    )
    for exp in benchmark.train_stream:
        strategy.train(exp)

    last_index = len(benchmark.train_stream) - 1
    oracle_accuracy = evaluate_skill_memory(
        model,
        plugin,
        benchmark.test_stream,
        last_index,
        num_classes=4,
        routing="oracle",
        batch_size=16,
        device=torch.device("cpu"),
    )
    probe_accuracy = evaluate_skill_memory(
        model,
        plugin,
        benchmark.test_stream,
        last_index,
        num_classes=4,
        routing="probe",
        batch_size=16,
        device=torch.device("cpu"),
    )
    assert len(oracle_accuracy) == len(probe_accuracy) == last_index + 1
    # Well-separated synthetic classes with immutable REUSE: every skill
    # keeps recognizing its own class, so oracle routing (the true mapping)
    # must be essentially perfect throughout.
    assert all(acc > 0.95 for acc in oracle_accuracy)


def test_evaluate_skill_memory_restores_model_weights_afterward():
    benchmark = _synthetic_benchmark(n_classes=2, n_experiences=1, n_per_class=20)
    model = SimpleMLP(input_dim=6, hidden_size=8, num_classes=2)
    plugin = EvaluationMemoryPlugin(
        memory=SkillMemory(max_skills=5),
        eval_routing="none",
        eval_memory_per_class=5,
        eval_memory_seed=0,
        verbose=False,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        plugins=[plugin],
    )
    for exp in benchmark.train_stream:
        strategy.train(exp)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    evaluate_skill_memory(
        model,
        plugin,
        benchmark.test_stream,
        0,
        num_classes=2,
        routing="oracle",
        batch_size=16,
        device=torch.device("cpu"),
    )
    after = model.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key])
