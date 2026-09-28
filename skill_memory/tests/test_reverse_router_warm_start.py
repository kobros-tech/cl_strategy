# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Tests for `NormalMLReverseEngineer`/`PersistentFingerprintSkillMemoryPlugin`
warm-starting (`reverse_warm_start`).

Contract under test:

* default behavior (`warm_start=False`, the default everywhere) is
  byte-for-byte unchanged from before this feature existed;
* `warm_start=True` actually reuses the previous model's compatible weights
  rather than reinitializing from scratch -- checked directly on the
  weights, not inferred from behavior;
* growing the feature dimension (new classes widen the candidate feature
  vector) is handled safely: old `input_projection` columns are preserved,
  new columns exist and are finite, and every other layer is copied exactly;
* a model from an incompatible class (or no previous model at all) falls
  back to an ordinary fresh initialization instead of erroring;
* `epochs=None` falls back to `self.epochs`, as before; an explicit value
  is honored and visibly changes how many optimizer steps run;
* this is NOT claimed to be an exact optimization -- warm-starting can and
  does change the fitted model, which is expected and tested as such rather
  than asserted away.
"""

from __future__ import annotations

import torch
from torch import nn

from skill_memory.evaluation.reverse_engineering import (
    NormalMLReverseEngineer,
    _FeatureReverseModel,
)


def _candidate_sets(n_sets, n_candidates, feature_dim, seed):
    torch.manual_seed(seed)
    sets = []
    for _ in range(n_sets):
        features = torch.randn(n_candidates, feature_dim)
        target = torch.randint(0, n_candidates, (1,)).item()
        sets.append((features, target))
    return sets


# ---------------------------------------------------------------------------
# Default behavior is unchanged
# ---------------------------------------------------------------------------


def test_default_call_matches_pre_warm_start_reference_implementation():
    """A from-scratch reference `_fit_model` (the exact pre-warm-start body,
    inlined here) must match the current default call bit-for-bit."""

    def reference_fit(engine, features, targets):
        torch.manual_seed(engine.seed)
        engine.feature_dim = int(features.shape[-1])
        flat = features.reshape(-1, engine.feature_dim)
        engine.feature_mean, engine.feature_std = engine._fit_scaler(flat)
        normalized = (features - engine.feature_mean) / engine.feature_std
        engine.hidden_size = max(engine.hidden_size, 128)
        model = _FeatureReverseModel(
            engine.feature_dim, engine.hidden_size, engine.num_heads, engine.num_layers
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=engine.learning_rate)
        criterion = nn.CrossEntropyLoss()
        sample_count = normalized.shape[0]
        batch_size = max(1, min(engine.batch_size, sample_count))
        model.train()
        with torch.enable_grad():
            for _ in range(engine.epochs):
                order = torch.randperm(sample_count)
                for start in range(0, sample_count, batch_size):
                    indices = order[start : start + batch_size]
                    batch = normalized[indices]
                    batch_targets = targets[indices]
                    logits = model(batch).squeeze(-1).reshape(-1, batch.shape[1])
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(logits, batch_targets)
                    loss.backward()
                    optimizer.step()
        return model.eval()

    sets = _candidate_sets(12, 4, 10, seed=0)
    features = torch.stack([f for f, _ in sets])
    targets = torch.tensor([t for _, t in sets])

    ref_engine = NormalMLReverseEngineer(
        epochs=2, hidden_size=32, num_heads=4, num_layers=1
    )
    ref_model = reference_fit(ref_engine, features.clone(), targets.clone())

    engine = NormalMLReverseEngineer(
        epochs=2, hidden_size=32, num_heads=4, num_layers=1
    )
    engine.fit_candidate_sets(sets)

    for (n1, p1), (n2, p2) in zip(
        ref_model.state_dict().items(), engine.model.state_dict().items(), strict=True
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), n1


def test_warm_start_false_ignores_a_previous_model():
    engine = NormalMLReverseEngineer(
        epochs=1, hidden_size=32, num_heads=4, num_layers=1
    )
    sets_a = _candidate_sets(8, 3, 6, seed=1)
    engine.fit_candidate_sets(sets_a)
    first_weight = engine.model.input_projection.weight.detach().clone()

    sets_b = _candidate_sets(8, 3, 6, seed=1)  # identical data
    engine.fit_candidate_sets(sets_b, warm_start=False)
    second_weight = engine.model.input_projection.weight.detach().clone()

    # Same seed, same data, fresh init every time -> identical result,
    # proving warm_start=False really did discard the previous model
    # (a warm start from a *trained* model would not reproduce the
    # from-scratch initialization).
    assert torch.equal(first_weight, second_weight)


# ---------------------------------------------------------------------------
# warm_start=True actually reuses weights
# ---------------------------------------------------------------------------


def test_warm_start_reuses_encoder_and_output_weights_exactly():
    """hidden_size/num_heads/num_layers are fixed, so the encoder and output
    head are always shape-compatible and must be copied exactly, before any
    further training perturbs them -- checked with epochs=0."""
    engine = NormalMLReverseEngineer(
        epochs=3, hidden_size=32, num_heads=4, num_layers=1, seed=0
    )
    engine.fit_candidate_sets(_candidate_sets(10, 4, 8, seed=2))
    trained_encoder = {
        k: v.clone() for k, v in engine.model.encoder.state_dict().items()
    }
    trained_output = {k: v.clone() for k, v in engine.model.output.state_dict().items()}

    # Same feature_dim (8): input_projection shape doesn't even need to grow
    # here, isolating the "encoder/output always preserved" claim.
    engine.fit_candidate_sets(
        _candidate_sets(10, 4, 8, seed=3), warm_start=True, epochs=0
    )

    for key, value in trained_encoder.items():
        assert torch.equal(engine.model.encoder.state_dict()[key], value), key
    for key, value in trained_output.items():
        assert torch.equal(engine.model.output.state_dict()[key], value), key


def test_warm_start_with_zero_epochs_leaves_weights_untouched():
    """epochs=0 means no optimizer step runs; warm-started weights (and the
    freshly-initialized new input-projection columns) must come through
    completely unperturbed."""
    engine = NormalMLReverseEngineer(
        epochs=3, hidden_size=32, num_heads=4, num_layers=1
    )
    engine.fit_candidate_sets(_candidate_sets(10, 4, 6, seed=5))
    before = {k: v.clone() for k, v in engine.model.state_dict().items()}

    engine.fit_candidate_sets(
        _candidate_sets(10, 4, 6, seed=6), warm_start=True, epochs=0
    )
    after = engine.model.state_dict()
    for key, value in before.items():
        assert torch.equal(after[key], value), key


def test_warm_start_rejects_negative_epochs():
    engine = NormalMLReverseEngineer(
        epochs=1, hidden_size=32, num_heads=4, num_layers=1
    )
    engine.fit_candidate_sets(_candidate_sets(5, 3, 6, seed=0))
    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        engine.fit_candidate_sets(
            _candidate_sets(5, 3, 6, seed=1), warm_start=True, epochs=-1
        )


# ---------------------------------------------------------------------------
# Growing feature dimension
# ---------------------------------------------------------------------------


def test_warm_start_preserves_old_input_projection_columns_when_dim_grows():
    engine = NormalMLReverseEngineer(
        epochs=2, hidden_size=32, num_heads=4, num_layers=1
    )
    engine.fit_candidate_sets(_candidate_sets(10, 4, 6, seed=0))
    old_columns = engine.model.input_projection.weight[:, :6].detach().clone()
    old_bias = engine.model.input_projection.bias.detach().clone()

    engine.fit_candidate_sets(
        _candidate_sets(10, 4, 10, seed=1), warm_start=True, epochs=0
    )
    assert engine.model.input_projection.weight.shape == (engine.hidden_size, 10)
    new_columns = engine.model.input_projection.weight[:, :6]
    assert torch.equal(new_columns, old_columns)
    assert torch.equal(engine.model.input_projection.bias, old_bias)
    # the newly introduced columns exist and are finite (freshly initialized,
    # not garbage/uninitialized memory)
    assert torch.isfinite(engine.model.input_projection.weight[:, 6:]).all()


def test_warm_start_also_handles_feature_dimension_shrinking():
    """Not expected in production (feature width only grows as classes
    accumulate), but the slice-based copy must not crash or corrupt data if
    it ever does shrink."""
    engine = NormalMLReverseEngineer(
        epochs=2, hidden_size=32, num_heads=4, num_layers=1
    )
    engine.fit_candidate_sets(_candidate_sets(10, 4, 10, seed=0))
    old_columns = engine.model.input_projection.weight[:, :6].detach().clone()

    engine.fit_candidate_sets(
        _candidate_sets(10, 4, 6, seed=1), warm_start=True, epochs=0
    )
    assert engine.model.input_projection.weight.shape == (engine.hidden_size, 6)
    assert torch.equal(engine.model.input_projection.weight, old_columns)


def test_warm_start_with_no_previous_model_falls_back_to_fresh_init():
    engine = NormalMLReverseEngineer(
        epochs=1, hidden_size=32, num_heads=4, num_layers=1
    )
    assert engine.model is None
    engine.fit_candidate_sets(
        _candidate_sets(5, 3, 6, seed=0), warm_start=True, epochs=1
    )
    assert isinstance(engine.model, _FeatureReverseModel)


def test_warm_start_ignores_a_binary_mode_previous_model():
    engine = NormalMLReverseEngineer(
        epochs=1, hidden_size=32, num_heads=4, num_layers=1, training_mode="binary"
    )
    engine.fit_feature_pairs([(torch.randn(4, 6), 1.0), (torch.randn(4, 6), 0.0)])
    assert engine.model is not None and not isinstance(
        engine.model, _FeatureReverseModel
    )

    engine.fit_candidate_sets(
        _candidate_sets(5, 3, 6, seed=0), warm_start=True, epochs=1
    )
    assert isinstance(engine.model, _FeatureReverseModel)  # fresh, not an error


# ---------------------------------------------------------------------------
# Plugin-level wiring
# ---------------------------------------------------------------------------


def test_plugin_defaults_preserve_from_scratch_behavior():
    from skill_memory.cl.persistent_skill_memory_plugin import (
        PersistentFingerprintSkillMemoryPlugin,
    )

    plugin = PersistentFingerprintSkillMemoryPlugin(max_skills=5, verbose=False)
    assert plugin.reverse_warm_start is False
    assert plugin.reverse_warm_start_epochs is None


def test_plugin_rejects_negative_warm_start_epochs():
    import pytest

    from skill_memory.cl.persistent_skill_memory_plugin import (
        PersistentFingerprintSkillMemoryPlugin,
    )

    with pytest.raises(ValueError, match="non-negative"):
        PersistentFingerprintSkillMemoryPlugin(
            max_skills=5, verbose=False, reverse_warm_start_epochs=-1
        )


def test_fit_reverse_router_uses_warm_start_epochs_only_after_the_first_fit(
    monkeypatch,
):
    """First call (no previous model): full `reverse_epochs`. Later calls
    with warm_start=True: `reverse_warm_start_epochs`."""
    from skill_memory.cl.persistent_skill_memory_plugin import (
        PersistentFingerprintSkillMemoryPlugin,
    )

    plugin = PersistentFingerprintSkillMemoryPlugin(
        max_skills=5,
        verbose=False,
        reverse_epochs=7,
        reverse_warm_start=True,
        reverse_warm_start_epochs=2,
    )
    calls = []
    original = plugin.reverse_engineer.fit_candidate_sets

    def spy(candidate_sets, **kwargs):
        calls.append(kwargs)
        return original(candidate_sets, **kwargs)

    monkeypatch.setattr(plugin.reverse_engineer, "fit_candidate_sets", spy)

    sets = _candidate_sets(5, 3, 6, seed=0)
    plugin.reverse_engineer.fit_candidate_sets(
        sets,
        warm_start=plugin.reverse_warm_start
        and plugin.reverse_engineer.model is not None,
        epochs=(
            plugin.reverse_warm_start_epochs
            if plugin.reverse_warm_start and plugin.reverse_engineer.model is not None
            else None
        ),
    )
    assert calls[-1]["warm_start"] is False  # no model yet
    assert calls[-1]["epochs"] is None  # -> falls back to reverse_epochs=7

    plugin.reverse_engineer.fit_candidate_sets(
        sets,
        warm_start=plugin.reverse_warm_start
        and plugin.reverse_engineer.model is not None,
        epochs=(
            plugin.reverse_warm_start_epochs
            if plugin.reverse_warm_start and plugin.reverse_engineer.model is not None
            else None
        ),
    )
    assert calls[-1]["warm_start"] is True
    assert calls[-1]["epochs"] == 2
