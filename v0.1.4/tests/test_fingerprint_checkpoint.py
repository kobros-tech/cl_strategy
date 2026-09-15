import torch

from v0_1_4_test_helpers import load_persistent_plugin


def test_behavior_checkpoint_roundtrip_preserves_fingerprints():
    mod = load_persistent_plugin()
    first = mod.PersistentFingerprintSkillMemoryPlugin(verbose=False)

    first.behavior.put(
        mod.ClassBehaviorRecord(
            class_id=42,
            skill_id=3,
            version=2,
            reference_inputs=torch.tensor([[1.0, 2.0]]),
            output_class_ids=(42, 87),
            reference_output=torch.tensor([0.8, 0.2]),
            reference_summary=torch.tensor([0.5, 0.5, 0.5, 0.5]),
        )
    )
    first._behavior_initialized = True

    checkpoint = first.state_dict()

    second = mod.PersistentFingerprintSkillMemoryPlugin(verbose=False)
    second.load_state_dict(checkpoint)

    restored = second.behavior.get(42, 3)
    assert restored is not None
    assert restored.version == 2
    assert restored.output_class_ids == (42, 87)
    assert torch.equal(restored.reference_inputs, first.behavior.get(42, 3).reference_inputs)
    assert torch.equal(restored.reference_output, first.behavior.get(42, 3).reference_output)
    assert torch.equal(restored.reference_summary, first.behavior.get(42, 3).reference_summary)
    assert second._behavior_initialized
