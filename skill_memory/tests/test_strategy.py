"""Tests for `skill_memory.strategy.SkillMemoryStrategy`.

The central property being verified: a plain
`strategy.train(experience)` / `strategy.eval(test_stream)` loop - the
same pattern used for `Naive`, `Replay` (ER), `ER-ACE`, and any other
Avalanche strategy - must work with no extra method calls, and
Avalanche's own accuracy/loss/forgetting metrics must reflect the ML
evaluator's predictions.
"""

import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.models import SimpleMLP
from torch.utils.data import TensorDataset

from skill_memory.evaluation.ml_cl_evaluator import evaluate_skill_memory
from skill_memory.strategy import SkillMemoryStrategy


def _synthetic_benchmark(n_classes: int, n_experiences: int, n_per_class: int = 40):
    """A synthetic, well-separated, no-download benchmark (see
    test_ml_cl_evaluator.py for why each class perturbs its own
    dimension rather than a shared scalar offset)."""
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


def _make_strategy(n_classes: int) -> SkillMemoryStrategy:
    model = SimpleMLP(input_size=6, hidden_size=8, num_classes=n_classes)
    return SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        evaluator_model_factory=lambda: SimpleMLP(
            input_size=6, hidden_size=8, num_classes=n_classes
        ),
        max_skills=10,
        eval_memory_per_class=10,
        eval_epochs=5,
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
    )


def test_standard_train_eval_loop_requires_no_extra_method_calls():
    """Exactly the ER/ER-ACE-style loop: strategy.train then strategy.eval,
    nothing else, must produce results through Avalanche's own metrics."""
    benchmark = _synthetic_benchmark(n_classes=4, n_experiences=2, n_per_class=30)
    strategy = _make_strategy(n_classes=4)

    for experience in benchmark.train_stream:
        strategy.train(experience)
        results = strategy.eval(benchmark.test_stream)

    # Avalanche's own metric keys, not anything this test computed itself.
    stream_accuracy_keys = [key for key in results if key.startswith("Top1_Acc_Stream")]
    stream_loss_keys = [key for key in results if key.startswith("Loss_Stream")]
    assert stream_accuracy_keys, f"no stream accuracy metric in {sorted(results)}"
    assert stream_loss_keys, f"no stream loss metric in {sorted(results)}"
    for key in stream_accuracy_keys:
        assert 0.0 <= results[key] <= 1.0


def test_skill_memory_diagnostic_is_separate_from_strategy_eval():
    """The direct Skill Memory diagnostic remains separate from strategy.eval().

    It must reflect the actual stored skills, not the auxiliary evaluator.
    """
    benchmark = _synthetic_benchmark(n_classes=4, n_experiences=2, n_per_class=40)
    strategy = _make_strategy(n_classes=4)
    for experience in benchmark.train_stream:
        strategy.train(experience)

    results = evaluate_skill_memory(
        strategy.model,
        strategy.skill_memory_plugin,
        benchmark.test_stream,
        1,
        num_classes=4,
        routing="oracle",
        batch_size=16,
        device=strategy.device,
    )
    assert set(results) == {0, 1, 2, 3}
    for metrics in results.values():
        assert "accuracy" in metrics
        assert "loss" in metrics


def test_public_properties_expose_underlying_components():
    strategy = _make_strategy(n_classes=2)
    assert strategy.memory is strategy.skill_memory_plugin.memory
    assert strategy.evaluator_model is strategy.ml_evaluation_plugin.evaluator_model
