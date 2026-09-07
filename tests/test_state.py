"""Storage invariants, actual compact brush boundaries and independent branches."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from worldline.state import ConflictError, WorldStore, digest, edit_field, validate_state


def test_close_reopen_preserves_every_snapshot_and_history(tmp_path, state):
    path = tmp_path / "nested" / "worlds.sqlite"
    store = WorldStore(path)
    first = store.create(state, "Saved world")
    edited = edit_field(first["state"], "raise", 0.35, 0.45, 0.1, 0.2)
    latest = store.save(first["id"], edited, {"type": "test-edit"}, 0)
    history = store.history(first["id"])
    store.close()

    reopened = WorldStore(path)
    try:
        assert reopened.get(first["id"]) == latest
        assert reopened.get(first["id"], 0) == first
        assert reopened.history(first["id"]) == history
        assert digest(reopened.get(first["id"])["state"]) == latest["hash"]
    finally:
        reopened.close()


@pytest.mark.parametrize("kind", ["raise", "lower", "plant", "water", "heat"])
@pytest.mark.parametrize("center", [(0.0, 0.0), (0.53, 0.48), (1.0, 1.0)])
def test_brush_changes_no_values_outside_support(state, kind, center):
    original = copy.deepcopy(state)
    x, z = center
    radius = 0.09
    changed = edit_field(state, kind, x, z, radius, 0.2)
    yy, xx = np.indices((64, 64), dtype=np.float64)
    distance = np.hypot(xx / 63 - x, yy / 63 - z)
    outside = distance > radius + 1e-6
    inside = distance < radius - 1e-6
    if kind in ("raise", "lower"):
        before = np.asarray(original["height"])
        after = np.asarray(changed["height"])
        assert np.array_equal(before[outside], after[outside]), "Unselected terrain changed"
        assert np.any(before[inside] != after[inside])
        assert changed["ecology"] == original["ecology"]
    else:
        channel = {"water": 0, "plant": 1, "heat": 2}[kind]
        before = np.asarray(original["ecology"])
        after = np.asarray(changed["ecology"])
        assert np.array_equal(before[channel][outside], after[channel][outside])
        assert np.any(before[channel][inside] != after[channel][inside])
        for other in set(range(3)) - {channel}:
            assert np.array_equal(before[other], after[other]), "Unselected channel changed"
        assert changed["height"] == original["height"]
    assert state == original, "Editing mutated its caller's state"


def test_branch_mutation_and_restore_do_not_rewrite_parent(tmp_path, state):
    store = WorldStore(tmp_path / "worlds.sqlite")
    try:
        parent = store.create(state)
        child = store.branch(parent["id"], "Different future", 0)
        assert child["id"] != parent["id"]
        assert child["parent"] == parent["id"]
        assert child["hash"] == parent["hash"]
        altered = edit_field(child["state"], "water", .5, .5, .2, .5)
        child_next = store.save(child["id"], altered, {"type": "edit"}, 0)
        assert child_next["hash"] != parent["hash"]
        assert store.get(parent["id"]) == parent
        assert store.get(child["id"], 0)["state"] == parent["state"]
        before_restore = store.history(child["id"])
        restored = store.restore(child["id"], 0, 1)
        assert restored["hash"] == parent["hash"]
        assert restored["revision"] == 2
        history = store.history(child["id"])
        assert history[1:] == before_restore, "Restore rewrote existing history"
        assert history[0]["event"] == {"type": "restore", "revision": 0}
        assert store.get(child["id"], 1) == child_next
    finally:
        store.close()


def test_mutating_a_read_result_cannot_change_stored_content(tmp_path, state):
    store = WorldStore(tmp_path / "worlds.sqlite")
    try:
        created = store.create(state)
        read = store.get(created["id"])
        read["state"]["height"][0][0] = .9
        assert store.get(created["id"]) == created
    finally:
        store.close()


def test_only_one_writer_with_matching_revision_can_commit(tmp_path, state):
    store = WorldStore(tmp_path / "worlds.sqlite")
    try:
        world = store.create(state)

        def write(index):
            try:
                candidate = copy.deepcopy(state)
                candidate["tick"] = index + 1
                store.save(world["id"], candidate, {"writer": index}, 0)
                return "saved"
            except ConflictError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(write, range(8)))
        assert outcomes.count("saved") == 1
        assert outcomes.count("conflict") == 7
        assert len(store.history(world["id"])) == 2
    finally:
        store.close()


@pytest.mark.parametrize("key,value", [
    ("height", {}), ("height", None), ("ecology", "not a numeric field"),
    ("height", [[0] * 63] * 64), ("ecology", [[[0] * 64] * 64] * 4),
    ("tick", -1), ("seed", True), ("biome", 3),
    ("weather", {"rain": float("nan"), "heat": 0}),
])
def test_invalid_imports_raise_value_error(state, key, value):
    state[key] = value
    with pytest.raises(ValueError):
        validate_state(state)


def test_numeric_limits_checked_before_lossy_precision_conversion(state):
    state["height"][0][0] = 1 + 1e-8
    with pytest.raises(ValueError):
        validate_state(state)


def test_unknown_state_metadata_is_removed(state):
    state["file_path"] = "/private/example"
    validated = validate_state(state)
    assert "file_path" not in validated
