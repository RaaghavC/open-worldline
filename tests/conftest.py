"""API fixtures isolate routing/storage from trained-model quality.

The deterministic stand-ins in this file are never used for model-quality claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worldline.server import create_app
from worldline.state import SCHEMA


def fixture_state():
    return {
        "schema": SCHEMA,
        "seed": 17,
        "biome": 0,
        "prompt": "Fixture world used for software checks only",
        "height": np.full((64, 64), 0.123456789123).tolist(),
        "ecology": np.full((3, 64, 64), 0.234567891234).tolist(),
        "tick": 0,
        "weather": {"rain": 0, "heat": 0},
    }


class FixtureGenerator:
    def __init__(self):
        self.calls = []

    def sample(self, seed, biome, steps):
        self.calls.append((seed, biome, steps))
        rng = np.random.default_rng(seed)
        return rng.uniform(-0.25, 0.25, (2, 64, 64)).astype(np.float32)


class FixtureDynamics:
    def __init__(self):
        self.calls = []

    def step(self, ecology, rain, heat):
        self.calls.append((rain, heat))
        out = ecology.copy()
        out[0] = np.clip(out[0] + rain * 0.01, 0, 1)
        out[2] = np.clip(out[2] + heat * 0.01, 0, 1)
        return out


@pytest.fixture
def state():
    return fixture_state()


@pytest.fixture
def fixture_models():
    return FixtureGenerator(), FixtureDynamics()


@pytest.fixture
def app(tmp_path, fixture_models):
    application = create_app(tmp_path / "worlds.sqlite", tmp_path / "no-checkpoints", fixture_models)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def world(client, state):
    response = client.post("/api/import", json={"name": "Fixture", "state": state})
    assert response.status_code == 200, response.text
    return response.json()
