"""HTTP behavior with explicit stand-in models, not learned-model evaluation."""
from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient

from worldline.server import create_app


def test_generate_routes_keywords_seed_and_steps_to_generator(client, fixture_models):
    response = client.post("/api/worlds", json={"prompt": "An alien crystal garden", "seed": 987, "steps": 8})
    assert response.status_code == 200
    world = response.json()
    assert world["state"]["biome"] == 2
    assert world["state"]["seed"] == 987
    assert fixture_models[0].calls == [(987, 2, 8)]
    assert client.get(f'/api/worlds/{world["id"]}').json()["hash"] == world["hash"]


def test_api_branch_isolation_action_routing_and_restore(client, world, fixture_models):
    wid = world["id"]
    branch = client.post(f"/api/worlds/{wid}/branch", json={"name": "Rain branch", "revision": 0}).json()
    child_id = branch["id"]
    response = client.post(f"/api/worlds/{child_id}/step", json={"rain": .75, "heat": .1, "steps": 3, "revision": 0})
    assert response.status_code == 200
    assert response.json()["state"]["tick"] == 3
    assert fixture_models[1].calls == [(.75, .1)] * 3
    assert client.get(f"/api/worlds/{wid}").json() == world
    history_before = client.get(f"/api/worlds/{child_id}/history").json()
    restored = client.post(f"/api/worlds/{child_id}/restore", json={"target": 0, "revision": 1})
    assert restored.status_code == 200
    assert restored.json()["hash"] == world["hash"]
    assert restored.json()["revision"] == 2
    assert client.get(f"/api/worlds/{child_id}/history").json()[1:] == history_before


@pytest.mark.parametrize("operation,payload", [
    ("edit", {"kind": "raise", "x": .5, "z": .5}),
    ("step", {"rain": .5, "steps": 1}),
    ("restore", {"target": 0}),
])
def test_stale_revision_is_409_without_change(client, world, operation, payload):
    wid = world["id"]
    first = client.post(f"/api/worlds/{wid}/edit", json={"kind": "raise", "x": .5, "z": .5, "revision": 0})
    assert first.status_code == 200
    current = client.get(f"/api/worlds/{wid}").json()
    history = client.get(f"/api/worlds/{wid}/history").json()
    stale = client.post(f"/api/worlds/{wid}/{operation}", json={**payload, "revision": 0})
    assert stale.status_code == 409
    assert client.get(f"/api/worlds/{wid}").json() == current
    assert client.get(f"/api/worlds/{wid}/history").json() == history


def test_import_export_roundtrip_has_same_content_hash(client, world):
    exported = client.get(f'/api/worlds/{world["id"]}/export')
    assert exported.status_code == 200
    assert exported.headers["content-disposition"].endswith('.json"')
    imported = client.post("/api/import", json=exported.json())
    assert imported.status_code == 200
    assert imported.json()["id"] != world["id"]
    assert imported.json()["hash"] == world["hash"]
    assert imported.json()["state"] == world["state"]


@pytest.mark.parametrize("origin", ["https://example.com", "null", "http://testserver.evil.example", "http://localhost:9999"])
def test_cross_origin_mutation_rejected(client, world, origin):
    before = client.get("/api/worlds").json()
    response = client.post(f'/api/worlds/{world["id"]}/branch', json={}, headers={"Origin": origin})
    assert response.status_code == 403
    assert client.get("/api/worlds").json() == before


def test_same_origin_and_local_host_allowed(client):
    assert client.get("/api/worlds", headers={"Origin": "http://testserver"}).status_code == 200
    assert client.get("/api/worlds", headers={"Host": "evil.example"}).status_code == 400


@pytest.mark.parametrize("route,body", [
    ("/api/worlds", {"prompt": "x" * 601}),
    ("/api/worlds", {"seed": -1}),
    ("/api/worlds", {"seed": 2147483648}),
    ("/api/worlds", {"steps": 1000000}),
    ("/api/worlds", {"prompt": "test", "path": "/tmp/output"}),
    ("/api/import", {"name": "x" * 101, "state": {}}),
])
def test_oversized_or_unknown_parameters_rejected(client, route, body):
    assert client.post(route, json=body).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("height", {}), ("height", None), ("ecology", "bad field"),
    ("height", [[0] * 65] * 64), ("weather", {"rain": "NaN"}),
])
def test_malformed_fields_return_422_not_500(client, state, field, value):
    state[field] = value
    response = client.post("/api/import", json={"state": state})
    assert response.status_code == 422, response.text
    assert client.get("/api/worlds").json() == []


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_numbers_return_422(client, world, token):
    body = '{"rain":' + token + ',"steps":1,"revision":0}'
    response = client.post(f'/api/worlds/{world["id"]}/step', content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert client.get(f'/api/worlds/{world["id"]}').json() == world


@pytest.mark.parametrize("field", ["height", "ecology"])
def test_imported_nonfinite_numbers_rejected(client, state, field):
    if field == "height":
        state[field][0][0] = float("nan")
    else:
        state[field][0][0][0] = float("inf")
    response = client.post("/api/import", content=json.dumps({"state": state}), headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text


def test_declared_body_size_is_limited(client):
    assert client.post("/api/import", content=b"{}", headers={"Content-Length": "2000001"}).status_code == 413


def test_streamed_body_size_is_limited_without_content_length(client, state):
    state["ignored_extra"] = "x" * 2_100_000
    payload = json.dumps({"state": state}).encode()
    chunks = (payload[offset:offset + 16384] for offset in range(0, len(payload), 16384))
    response = client.post("/api/import", content=chunks, headers={"Content-Type": "application/json"})
    assert response.status_code == 413, response.text[:200]
    assert client.get("/api/worlds").json() == []


@pytest.mark.parametrize("bad_id", ["..%2F..%2Fetc%2Fpasswd", "%2E%2E%2FNOTICE", "x%27%20OR%201%3D1--", "not-a-world"])
def test_world_identifiers_cannot_read_files_or_inject_queries(client, bad_id):
    response = client.get(f"/api/worlds/{bad_id}/export")
    assert response.status_code == 404
    assert "root:" not in response.text
    assert "Apache License" not in response.text


def test_invalid_restore_target_does_not_append_history(client, world):
    wid = world["id"]
    response = client.post(f"/api/worlds/{wid}/restore", json={"target": 9999, "revision": 0})
    assert response.status_code == 404
    assert len(client.get(f"/api/worlds/{wid}/history").json()) == 1


def test_partial_checkpoints_do_not_report_ready_or_synthesize_fallback(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "unrelated.pt").write_bytes(b"not model weights")
    application = create_app(tmp_path / "worlds.sqlite", directory)
    try:
        with TestClient(application, raise_server_exceptions=False) as client:
            status = client.get("/api/status")
            assert status.status_code == 200
            assert status.json()["trained"] is False
            response = client.post("/api/worlds", json={"prompt": "alpine", "seed": 123})
            assert response.status_code == 503
            assert client.get("/api/worlds").json() == []
    finally:
        application.state.store.close()
