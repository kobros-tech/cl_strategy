# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Focused tests for the experimental candidate-skill routing patch."""

from types import SimpleNamespace

import torch

import skill_memory.demos.candidate_skill_routing_patch as routing


class FakeClassMap:
    def __init__(self, mapping, owned):
        self.mapping = mapping
        self.owned = owned

    def find_skill_for_class_anywhere(self, class_id):
        return self.mapping.get(int(class_id))

    def classes_for_skill(self, skill):
        return self.owned[int(skill)]


class FakeMemory:
    def __init__(self, states, metadata=None):
        self.states = states
        self.metadata_values = metadata or {}
        self.calls = []

    def state(self, skill):
        self.calls.append(int(skill))
        return self.states[int(skill)]

    def metadata(self, skill):
        return dict(self.metadata_values.get(int(skill), {}))


def _strategy(mapping, owned, states, targets=None, metadata=None):
    memory = FakeMemory(states, metadata=metadata)
    class_map = FakeClassMap(mapping, owned)
    plugin = SimpleNamespace(
        memory_plugin=SimpleNamespace(
            class_map=class_map,
            memory=memory,
        )
    )
    if targets is None:
        targets = torch.zeros(1, dtype=torch.long)
    return (
        SimpleNamespace(
            model=SimpleNamespace(),
            ml_evaluation_plugin=plugin,
            mbatch=(torch.zeros(len(targets), 1), targets),
        ),
        memory,
    )


def _install_fake_skill_predictor(monkeypatch, logits_by_skill):
    calls = []

    def predict(_model, state, inputs):
        skill = int(state)
        calls.append(skill)
        return logits_by_skill[skill][inputs.shape[0]].clone()

    def expand(skill_logits, _state, owned_classes, num_classes):
        output = torch.full(
            (skill_logits.shape[0], num_classes),
            -100.0,
            dtype=skill_logits.dtype,
        )
        output[:, owned_classes] = skill_logits
        return output

    monkeypatch.setattr(routing, "predict_logits", predict)
    monkeypatch.setattr(routing, "expand_skill_logits", expand)
    return calls


def test_accepted_candidate_replaces_evaluator_prediction(monkeypatch):
    strategy, _ = _strategy(
        mapping={0: 0, 1: 0},
        owned={0: [0, 1]},
        states={0: 0},
    )
    _install_fake_skill_predictor(
        monkeypatch,
        {0: {1: torch.tensor([[0.0, 5.0]])}},
    )

    evaluator = torch.tensor([[2.0, 1.9]])
    routed, stats = routing.route_candidate_classes(
        evaluator,
        torch.zeros(1, 1),
        strategy,
        candidate_k=2,
        skill_confidence_threshold=0.9,
    )

    assert int(evaluator.argmax(1).item()) == 0
    assert int(routed.argmax(1).item()) == 1
    assert stats["skill_acceptance_rate"] == 1.0
    assert stats["skill_override_rate"] == 1.0


def test_no_candidate_accepted_falls_back_to_ml_top1(monkeypatch):
    strategy, _ = _strategy(
        mapping={0: 0, 1: 0},
        owned={0: [0, 1]},
        states={0: 0},
    )
    _install_fake_skill_predictor(
        monkeypatch,
        {0: {1: torch.tensor([[0.0, 0.0]])}},
    )

    evaluator = torch.tensor([[2.0, 1.0]])
    routed, stats = routing.route_candidate_classes(
        evaluator,
        torch.zeros(1, 1),
        strategy,
        candidate_k=2,
        skill_confidence_threshold=0.9,
    )

    assert torch.equal(routed, evaluator)
    assert stats["fallback_rate"] == 1.0


def test_multiple_accepted_candidates_use_joint_confidence(monkeypatch):
    strategy, _ = _strategy(
        mapping={0: 0, 2: 1},
        owned={0: [0, 1], 1: [2, 3]},
        states={0: 0, 1: 1},
    )
    _install_fake_skill_predictor(
        monkeypatch,
        {
            0: {1: torch.tensor([[2.0, 0.0]])},
            1: {1: torch.tensor([[3.0, 0.0]])},
        },
    )

    evaluator = torch.tensor([[2.0, 1.9, 1.8, -5.0]])
    routed, stats = routing.route_candidate_classes(
        evaluator,
        torch.zeros(1, 1),
        strategy,
        candidate_k=3,
        skill_confidence_threshold=0.8,
    )

    # Class 0 has the larger evaluator*skill joint confidence than class 2.
    assert int(routed.argmax(1).item()) == 0
    assert stats["skill_acceptance_rate"] == 1.0


def test_candidates_sharing_one_skill_evaluate_that_skill_once(monkeypatch):
    strategy, memory = _strategy(
        mapping={0: 0, 1: 0, 2: 1},
        owned={0: [0, 1], 1: [2]},
        states={0: 0, 1: 1},
        targets=torch.tensor([0, 1]),
    )
    calls = _install_fake_skill_predictor(
        monkeypatch,
        {
            0: {2: torch.tensor([[3.0, 0.0], [3.0, 0.0]])},
            1: {2: torch.tensor([[3.0], [3.0]])},
        },
    )

    evaluator = torch.tensor([[2.0, 1.8, 0.1], [2.0, 1.7, 0.1]])
    routing.route_candidate_classes(
        evaluator,
        torch.zeros(2, 1),
        strategy,
        candidate_k=2,
        skill_confidence_threshold=0.8,
    )

    assert calls.count(0) == 1
    assert memory.calls.count(0) == 1


def test_skill_memory_cannot_select_class_outside_evaluator_top_k(monkeypatch):
    strategy, _ = _strategy(
        mapping={0: 0, 1: 0, 2: 1},
        owned={0: [0, 1], 1: [2]},
        states={0: 0, 1: 1},
    )
    _install_fake_skill_predictor(
        monkeypatch,
        {
            0: {1: torch.tensor([[0.0, 0.0]])},
            1: {1: torch.tensor([[10.0]])},
        },
    )

    evaluator = torch.tensor([[3.0, 2.0, 1.0]])
    routed, _ = routing.route_candidate_classes(
        evaluator,
        torch.zeros(1, 1),
        strategy,
        candidate_k=2,
        skill_confidence_threshold=0.5,
    )

    assert int(routed.argmax(1).item()) in {0, 1}


def test_routing_decision_is_independent_of_labels(monkeypatch):
    base_strategy, _ = _strategy(
        mapping={0: 0, 1: 0},
        owned={0: [0, 1]},
        states={0: 0},
        targets=torch.tensor([0]),
    )
    _install_fake_skill_predictor(
        monkeypatch,
        {0: {1: torch.tensor([[3.0, 0.0]])}},
    )

    evaluator = torch.tensor([[2.0, 1.0]])
    inputs = torch.zeros(1, 1)
    routed_a, _ = routing.route_candidate_classes(
        evaluator,
        inputs,
        base_strategy,
        candidate_k=2,
        skill_confidence_threshold=0.8,
    )

    other_strategy = base_strategy
    other_strategy.mbatch = (inputs, torch.tensor([1]))
    routed_b, _ = routing.route_candidate_classes(
        evaluator,
        inputs,
        other_strategy,
        candidate_k=2,
        skill_confidence_threshold=0.8,
    )

    assert torch.equal(routed_a, routed_b)


def test_debug_log_shows_candidates_verification_and_elected_value(monkeypatch, capsys):
    strategy, _ = _strategy(
        mapping={0: 0, 1: 0},
        owned={0: [0, 1]},
        states={0: 0},
    )
    _install_fake_skill_predictor(
        monkeypatch,
        {0: {1: torch.tensor([[4.0, 0.0]])}},
    )

    routing.route_candidate_classes(
        torch.tensor([[2.0, 1.0]]),
        torch.zeros(1, 1),
        strategy,
        candidate_k=2,
        skill_confidence_threshold=0.8,
        debug=True,
        debug_max_samples=1,
    )

    output = capsys.readouterr().out
    assert "[CANDIDATE ROUTING] sample=0" in output
    assert "ML candidates:" in output
    assert "CL / Skill Memory verification:" in output
    assert "ELECTED: class=1 source=Skill Memory" in output


def test_cl_override_reports_net_correction(monkeypatch):
    strategy, _ = _strategy(
        mapping={0: 0, 1: 0},
        owned={0: [0, 1]},
        states={0: 0},
        targets=torch.tensor([1]),
    )
    _install_fake_skill_predictor(
        monkeypatch,
        {0: {1: torch.tensor([[0.0, 5.0]])}},
    )

    routed, stats = routing.route_candidate_classes(
        torch.tensor([[2.0, 1.9]]),
        torch.zeros(1, 1),
        strategy,
        candidate_k=2,
        skill_confidence_threshold=0.8,
        rescue_skill_confidence_threshold=0.9,
        rescue_skill_margin=0.1,
        ml_uncertainty_threshold=0.55,
    )

    assert int(routed.argmax(1).item()) == 1
    assert stats["ml_top1_accuracy"] == 0.0
    assert stats["final_batch_accuracy"] == 1.0
    assert stats["cl_corrected_ml_error_rate"] == 1.0
    assert stats["cl_introduced_error_rate"] == 0.0
    assert stats["cl_net_accuracy_gain"] == 1.0
    assert stats["cl_override_precision"] == 1.0


def test_binary_skill_uses_independent_yes_no_scores(monkeypatch):
    strategy, _ = _strategy(
        mapping={0: 0, 1: 0},
        owned={0: [0, 1]},
        states={0: 0},
    )
    strategy.ml_evaluation_plugin.memory_plugin.memory.metadata = lambda _skill: {
        "class_train_mode": "binary_one_vs_rest"
    }
    _install_fake_skill_predictor(
        monkeypatch,
        {0: {1: torch.tensor([[4.0, -1.0]])}},
    )

    evaluator = torch.tensor([[2.0, 1.9]])
    routed, stats = routing.route_candidate_classes(
        evaluator,
        torch.zeros(1, 1),
        strategy,
        candidate_k=2,
        skill_confidence_threshold=0.9,
        rescue_skill_confidence_threshold=0.9,
        rescue_skill_margin=0.1,
        ml_uncertainty_threshold=0.55,
    )

    # Class 1 has a strong independent YES score even though it is not the
    # ML top-1. Binary verification must not softmax the two YES/NO heads.
    assert int(routed.argmax(1).item()) == 1
    assert stats["skill_override_rate"] == 1.0


def test_validation_gate_blocks_unvalidated_cl_override(monkeypatch):
    validation_inputs = torch.zeros(20, 1)
    validation_targets = torch.tensor([0] * 10 + [1] * 10)
    strategy, _ = _strategy(
        mapping={0: 0, 1: 0},
        owned={0: [0, 1]},
        states={0: 0},
        targets=torch.tensor([0]),
        metadata={
            0: {
                "class_train_mode": "binary_one_vs_rest",
                "verification_examples": [(validation_inputs, validation_targets)],
            }
        },
    )
    strategy.model = torch.nn.Linear(1, 2)
    _install_fake_skill_predictor(
        monkeypatch,
        {
            0: {
                20: torch.tensor([[4.0, -4.0]] * 20),
                1: torch.tensor([[0.0, 5.0]]),
            }
        },
    )

    routed, stats = routing.route_candidate_classes(
        torch.tensor([[2.0, 1.9]]),
        torch.zeros(1, 1),
        strategy,
        candidate_k=2,
        skill_confidence_threshold=0.5,
        rescue_skill_confidence_threshold=0.5,
        rescue_skill_margin=0.0,
        ml_uncertainty_threshold=0.55,
        calibration_precision_target=0.9,
        calibration_min_samples=20,
    )

    assert int(routed.argmax(1).item()) == 0
    assert stats["skill_override_rate"] == 0.0
