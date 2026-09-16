import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("binary_behavior", ROOT / "behavior.py")
behavior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = behavior
spec.loader.exec_module(behavior)


def _record(class_id=42, skill_id=3, version=0):
    return behavior.ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=version,
        reference_inputs=torch.ones(3, 2),
        reference_y=torch.tensor([True, True, True]),
    )


def test_reverse_engineer_y_is_binary_and_uses_global_class_id():
    logits = torch.zeros(3, 100)
    logits[:, 42] = 10.0
    logits[1, 87] = 20.0

    predicted = behavior.reverse_engineer_y(logits, 42)

    assert predicted.dtype == torch.bool
    assert predicted.tolist() == [True, False, True]


def test_known_class_prediction_is_compared_with_true_expected_y():
    predicted = torch.tensor([True, True, False, True])

    result = behavior.compare_binary_behavior(predicted, expected_y=True)

    assert result["expected_y"] is True
    assert result["correct"].tolist() == [True, True, False, True]
    assert result["accuracy"] == 0.75
    assert result["all_correct"] is False


def test_exact_anonymous_behavior_is_identified():
    predicted = torch.tensor([True, True, True])

    assert behavior.identify_binary_behavior(predicted, expected_y=True)


def test_nonmatching_anonymous_behavior_fails_identification():
    predicted = torch.tensor([True, False, True])

    assert not behavior.identify_binary_behavior(predicted, expected_y=True)


def test_cache_invalidates_every_class_of_changed_skill():
    cache = behavior.BehaviorFingerprintCache()
    cache.put(_record(class_id=42, skill_id=3))
    cache.put(_record(class_id=87, skill_id=3))
    cache.put(_record(class_id=12, skill_id=4))

    version = cache.bump_skill(3)

    assert version == 1
    assert cache.get(42, 3) is None
    assert cache.get(87, 3) is None
    assert cache.get(12, 4) is not None
    assert torch.equal(
        cache.all_records_for_skill(3)[0].reference_inputs,
        torch.ones(3, 2),
    )


def test_checkpoint_roundtrip_preserves_binary_fingerprint():
    cache = behavior.BehaviorFingerprintCache()
    record = _record(version=2)
    cache.put(record)

    restored = behavior.BehaviorFingerprintCache()
    restored.load_state_dict(cache.state_dict())
    result = restored.get(42, 3)

    assert result is not None
    assert result.version == 2
    assert result.expected_y is True
    assert torch.equal(result.reference_y, record.reference_y)
    assert torch.equal(result.reference_inputs, record.reference_inputs)


def test_legacy_checkpoint_without_behavior_state_loads():
    cache = behavior.BehaviorFingerprintCache()

    cache.load_state_dict({})

    assert cache.state_dict() == {"skill_versions": {}, "records": []}
