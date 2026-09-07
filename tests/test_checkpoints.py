"""Small integration checks using the real released weights.

Missing checkpoints skip this module explicitly. These tests are not visual
quality or real-physics benchmarks. Training-quality evaluation is separate.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest

from worldline.state import SCHEMA, WorldStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained_models():
    directory = Path(os.environ.get("WORLDLINE_TEST_CHECKPOINTS", ROOT / "checkpoints"))
    missing = [name for name in ("spatial-flow.pt", "ecology.pt") if not (directory / name).is_file()]
    if missing:
        pytest.skip("Real-checkpoint checks NOT RUN: missing " + ", ".join(missing))
    import torch
    from worldline.models import get_models
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return get_models(directory, device=os.environ.get("WORLDLINE_TEST_DEVICE", "cpu"))


def test_real_generator_repeats_seed_and_responds_to_seed_and_biome(trained_models):
    generator, _ = trained_models
    original = generator.sample(80001, 0, 8)
    repeated = generator.sample(80001, 0, 8)
    changed_seed = generator.sample(80002, 0, 8)
    changed_biome = generator.sample(80001, 1, 8)
    assert original.shape == (2, 64, 64)
    assert np.isfinite(original).all()
    assert original.min() >= -1 and original.max() <= 1
    assert np.array_equal(original, repeated), "Fixed seed changed within one device/process"
    assert np.mean(np.abs(original - changed_seed)) > 1e-4
    assert np.mean(np.abs(original - changed_biome)) > 1e-4


def test_real_rain_and_heat_branches_diverge_without_changing_source(tmp_path, trained_models):
    generator, dynamics = trained_models
    fields = generator.sample(80003, 0, 8)
    state = {"schema": SCHEMA, "prompt": "Checkpoint integration test", "seed": 80003, "biome": 0,
             "height": fields[0].tolist(), "ecology": np.full((3, 64, 64), .4, dtype=np.float32).tolist(),
             "tick": 0, "weather": {"rain": 0, "heat": 0}}
    store = WorldStore(tmp_path / "worlds.sqlite")
    try:
        source = store.create(state)
        rain = store.branch(source["id"], "Rain")
        heat = store.branch(source["id"], "Heat")
        values = []
        for child, controls in [(rain, (1., 0.)), (heat, (0., 1.))]:
            edited = copy.deepcopy(child["state"])
            ecology = np.asarray(edited["ecology"], dtype=np.float32)
            for _ in range(10):
                ecology = dynamics.step(ecology, *controls)
            assert np.isfinite(ecology).all()
            assert ecology.min() >= 0 and ecology.max() <= 1
            edited["ecology"] = ecology.tolist()
            edited["tick"] = 10
            edited["weather"] = {"rain": controls[0], "heat": controls[1]}
            store.save(child["id"], edited, {"type": "checkpoint-test"}, 0)
            values.append(ecology)
        assert np.mean(values[0][0]) > np.mean(values[1][0]), "Rain did not increase water relative to heat"
        assert np.mean(values[1][2]) > np.mean(values[0][2]), "Heat did not increase temperature relative to rain"
        assert store.get(source["id"]) == source
        rain_hash = store.get(rain["id"])["hash"]
        restored = store.restore(rain["id"], 0, 1)
        assert restored["hash"] == source["hash"]
        assert store.get(rain["id"], 1)["hash"] == rain_hash
        replayed = np.asarray(restored["state"]["ecology"], dtype=np.float32)
        for _ in range(10):
            replayed = dynamics.step(replayed, 1., 0.)
        assert np.array_equal(replayed, values[0]), "Restored deterministic transition path changed"
    finally:
        store.close()


def test_real_dynamics_zero_time_is_identity(trained_models):
    _, dynamics = trained_models
    state = np.full((3, 8, 8), .345, dtype=np.float32)
    actual = dynamics.step(state, .3, .7, dt=0)
    assert np.array_equal(actual, state)
