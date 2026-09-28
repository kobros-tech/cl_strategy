# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import torch
from avalanche.models.dynamic_modules import (
    IncrementalClassifier,
    avalanche_model_adaptation,
)
from torch.utils.data import Dataset

from skill_memory.utils import probing as mod


class TinyDataset(Dataset):
    def __init__(self):
        self.samples = [(torch.tensor([i]), i % 3) for i in range(9)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class Experience:
    def __init__(self):
        self.dataset = TinyDataset()


def test_classes_and_probe_are_based_on_dataset_content():
    exp = Experience()

    assert mod.classes_in_experience(exp) == [0, 1, 2]

    _x, y = mod.probe_class(
        exp,
        1,
        batch_size=10,
        n_batches=1,
        seed=1,
    )

    assert set(y.tolist()) == {1}


class ClassExperience:
    classes_in_this_experience = [7]


def test_functional_growth_matches_incremental_classifier_adaptation():
    model = IncrementalClassifier(2, initial_out_features=1)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    actual = IncrementalClassifier(2, initial_out_features=1)
    actual.load_state_dict(state)
    actual.train()
    torch.manual_seed(123)
    avalanche_model_adaptation(actual, ClassExperience())

    torch.manual_seed(999)
    params = mod._functional_growth_for_experience(
        model,
        state,
        ClassExperience(),
        seed=123,
    )

    assert params["classifier.weight"].shape == actual.classifier.weight.shape
    assert torch.equal(params["classifier.weight"], actual.classifier.weight)
    assert torch.equal(params["classifier.bias"], actual.classifier.bias)
    assert torch.equal(params["active_units"], actual.active_units)


def test_functional_growth_activates_only_experience_classes():
    model = IncrementalClassifier(2, initial_out_features=2)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    params = mod._functional_growth_for_experience(
        model,
        state,
        ClassExperience(),
        seed=123,
    )

    assert params["active_units"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]


def test_functional_growth_with_seed_preserves_caller_rng_state():
    model = IncrementalClassifier(2, initial_out_features=1)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    torch.manual_seed(999)
    expected = torch.rand(4)

    torch.manual_seed(999)
    mod._functional_growth_for_experience(
        model,
        state,
        ClassExperience(),
        seed=123,
    )
    actual = torch.rand(4)

    assert torch.equal(actual, expected)


def test_predict_logits_restores_model_training_mode():
    model = IncrementalClassifier(2, initial_out_features=1)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.train()

    mod.predict_logits(model, state, torch.ones(2, 2))

    assert model.training


def test_evaluate_state_restores_model_training_mode():
    model = IncrementalClassifier(2, initial_out_features=1)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.train()

    mod.evaluate_state(
        model,
        state,
        torch.ones(2, 2),
        torch.tensor([7, 7]),
        torch.nn.CrossEntropyLoss(),
        ClassExperience(),
        seed=123,
    )

    assert model.training
