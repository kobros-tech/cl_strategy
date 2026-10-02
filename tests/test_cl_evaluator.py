# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import torch

import skill_memory.evaluation.cl_evaluator as cl_evaluator
from skill_memory.evaluation.cl_evaluator import CLEvaluationPlugin


class _Memory:
    def slots(self):
        return [0]

    def state(self, skill):
        assert skill == 0
        return {}

    def metadata(self, skill):
        return {}


class _ClassMap:
    def classes_for_skill(self, skill):
        assert skill == 0
        return [0, 1]


def test_cl_evaluator_reports_raw_accuracy_separately(monkeypatch):
    plugin = CLEvaluationPlugin(
        memory_plugin=SimpleNamespace(
            memory=_Memory(),
            class_map=_ClassMap(),
        ),
        verbose=False,
        strict_protocol=False,
    )
    plugin._active = True
    plugin._num_classes = 2
    plugin._calibrators = {
        0: (0, 1.0, -4.0),
        1: (0, 1.0, 0.0),
    }

    def fake_predict_logits(model, state, inputs):
        return torch.tensor([[2.0, 1.0]], device=inputs.device)

    monkeypatch.setattr(cl_evaluator, "predict_logits", fake_predict_logits)

    strategy = SimpleNamespace(
        model=SimpleNamespace(),
        mbatch=(torch.zeros(1, 1), torch.tensor([0])),
    )

    plugin.after_eval_forward(strategy)
    plugin.after_eval_iteration(strategy)
    plugin.after_eval(strategy)

    results = plugin.results()

    assert results["mean_final_accuracy"] == 0.0
    assert results["raw_mean_final_accuracy"] == 1.0
