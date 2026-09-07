"""Browser session integrity, independent of trained model quality."""
import pytest
import torch
from fastapi.testclient import TestClient

from experiments.room_world import play
from experiments.room_world.model import RGBModel


class Probe(torch.nn.Module):
    kind = "predictor"

    def __init__(self):
        super().__init__()
        self.histories = []

    def forward(self, history, action):
        self.histories.append(history.clone())
        return torch.full_like(history[:, -1], .25 * int(action[0]) - .5)


def start(client, kind="predictor", seed=123):
    result = client.post("/api/rooms", json={"seed": seed, "kind": kind})
    assert result.status_code == 200, result.text
    return result.json()


def action(client, room, value):
    result = client.post(f"/api/rooms/{room['id']}/step", json={"revision": room["revision"], "action": value})
    assert result.status_code == 200, result.text
    return result.json()


def test_prediction_receives_its_own_pixels_and_never_calls_renderer(monkeypatch):
    probe = Probe()
    with TestClient(play.create_app(model_loader=lambda *a, **k: (probe, {}))) as client:
        room = start(client)
        assert room["frame_source"] == "initial renderer"

        def prohibited(*args, **kwargs):
            raise AssertionError("Future teacher state must not be used")

        monkeypatch.setattr(play.Room, "render", prohibited)
        monkeypatch.setattr(play.Room, "step", prohibited)
        room = action(client, room, 1)
        assert room["frame_source"] == "neural prediction"
        room = action(client, room, 2)
        assert room["revision"] == 2
        assert torch.equal(probe.histories[1][:, :-1], probe.histories[0][:, 1:])
        assert torch.all(probe.histories[1][:, -1] == -.25)


def test_branch_copies_flow_noise_state_and_remains_independent():
    torch.manual_seed(34)
    model = RGBModel("flow", width=8).eval()
    with TestClient(play.create_app(model_loader=lambda *a, **k: (model, {}))) as client:
        parent = action(client, start(client, kind="flow"), 1)
        child = client.post(f"/api/rooms/{parent['id']}/branch", json={"revision": 1}).json()
        assert child["image"] == parent["image"] and child["id"] != parent["id"]
        parent = action(client, parent, 3)
        child = action(client, child, 3)
        assert parent["image"] == child["image"]
        action(client, child, 4)
        untouched = client.get(f"/api/rooms/{parent['id']}").json()
        assert untouched["revision"] == 2 and untouched["image"] == parent["image"]


def test_stale_updates_bad_actions_and_cross_origin_requests_are_rejected():
    with TestClient(play.create_app(model_loader=lambda *a, **k: (Probe(), {}))) as client:
        room = start(client)
        action(client, room, 0)
        path = f"/api/rooms/{room['id']}/step"
        assert client.post(path, json={"revision": 0, "action": 1}).status_code == 409
        for invalid in (-1, 6, True, "1"):
            assert client.post(path, json={"revision": 1, "action": invalid}).status_code == 422
        assert client.post("/api/rooms", json={}, headers={"Origin": "https://example.com"}).status_code == 403
        oversized = client.post("/api/rooms", content=b" " * 4097)
        assert oversized.status_code == 413 and "4 KB" in oversized.json()["detail"]
        for invalid_seed in (True, "1", 1.5):
            assert client.post("/api/rooms", json={"seed": invalid_seed}).status_code == 422


def test_session_storage_is_bounded():
    with TestClient(play.create_app(model_loader=lambda *a, **k: (Probe(), {}), max_sessions=2)) as client:
        first = start(client)
        second = start(client, seed=456)
        third = start(client, seed=789)
        assert client.get(f"/api/rooms/{first['id']}").status_code == 404
        assert client.get(f"/api/rooms/{second['id']}").status_code == 200
        assert client.get(f"/api/rooms/{third['id']}").status_code == 200


def test_missing_checkpoint_explains_recovery(tmp_path):
    with TestClient(play.create_app(artifact_root=tmp_path)) as client:
        result = client.post("/api/rooms", json={})
        assert result.status_code == 503
        assert "checkpoint is missing" in result.json()["detail"]


@pytest.mark.parametrize("failure", ["nonfinite", "runtime"])
def test_failed_generation_preserves_history_revision_and_rng(monkeypatch, failure):
    torch.manual_seed(34)
    model = RGBModel("flow", width=8).eval()
    with TestClient(play.create_app(model_loader=lambda *a, **k: (model, {}))) as client:
        room = start(client, kind="flow")
        branch = client.post(f"/api/rooms/{room['id']}/branch", json={"revision": 0}).json()
        original = play.predict_tensor

        def fail_after_sampling(model, history, action, sample_steps, rng):
            torch.randn(97, generator=rng)
            if failure == "runtime":
                raise RuntimeError("injected failure after sampling")
            return torch.full_like(history[:, -1], float("nan"))

        monkeypatch.setattr(play, "predict_tensor", fail_after_sampling)
        failed = client.post(f"/api/rooms/{room['id']}/step", json={"revision": 0, "action": 3})
        assert failed.status_code == (503 if failure == "runtime" else 500)
        assert "not advanced" in failed.json()["detail"]
        unchanged = client.get(f"/api/rooms/{room['id']}").json()
        assert unchanged["image"] == room["image"] and unchanged["revision"] == 0
        monkeypatch.setattr(play, "predict_tensor", original)
        retried = action(client, unchanged, 3)
        reference = action(client, branch, 3)
        assert retried["revision"] == reference["revision"] == 1
        assert retried["image"] == reference["image"]


def test_corrupt_or_wrong_model_checkpoint_reports_recovery(tmp_path):
    checkpoint_dir = tmp_path / "predictor"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_bytes(b"invalid checkpoint")
    with TestClient(play.create_app(artifact_root=tmp_path)) as client:
        result = client.post("/api/rooms", json={})
        assert result.status_code == 503
        assert "checkpoint could not be loaded" in result.json()["detail"]
    with TestClient(play.create_app(model_loader=lambda *a, **k: (RGBModel("flow", 8), {}))) as client:
        result = client.post("/api/rooms", json={"kind": "predictor"})
        assert result.status_code == 503


def test_invalid_session_bound_is_rejected():
    for value in (0, -1, True):
        with pytest.raises(ValueError, match="max_sessions"):
            play.create_app(max_sessions=value)
