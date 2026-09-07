"""Browser JSON numeric spelling and non-destructive legacy-store migration."""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from worldline.state import WorldStore, canonical, digest


def browser_json_roundtrip(value):
    """Simulate JS number serialization without requiring Node in backend CI.

    JSON.parse represents these finite numbers as doubles; JSON.stringify
    emits integral doubles and negative zero without a fractional suffix.
    Python float parsing preserves the same fractional double values.
    """
    def parse_number(token):
        value = float(token)
        return int(value) if value.is_integer() else value
    return json.loads(json.dumps(value, allow_nan=False), parse_float=parse_number)


def legacy_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def legacy_digest(value):
    return hashlib.sha256(legacy_json(value).encode()).hexdigest()


def state_with_numeric_variants(state):
    state["height"][0][:3] = [0.0, -0.0, 1.0]
    state["ecology"][0][0][:3] = [0.0, 1.0, -0.0]
    state["weather"] = {"rain": 0.0, "heat": 1.0}
    return state


def test_equal_numeric_spellings_have_equal_hashes_without_merging_booleans():
    assert canonical([0, 1, -1]) == canonical([0.0, 1.0, -1.0])
    assert canonical([0, -0.0]) == "[0,0]"
    assert canonical({"v": False}) != canonical({"v": 0})
    assert canonical({"v": True}) != canonical({"v": 1})
    original = .12345678912345678
    adjacent = float(np.nextafter(original, 1.))
    assert canonical([original]) != canonical([adjacent])
    assert json.loads(canonical([original]))[0] == original


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numbers_still_cannot_be_hashed(nonfinite):
    with pytest.raises(ValueError):
        canonical({"value": nonfinite})


def test_browser_export_import_has_identical_hash_and_remains_editable(client, state):
    state = state_with_numeric_variants(state)
    created = client.post("/api/import", json={"name": "Original", "state": state}).json()
    exported = client.get(f'/api/worlds/{created["id"]}/export').json()
    # Exercise exactly the 0.0/1.0/-0.0 spelling change, even if the server now
    # emits integer spelling itself.
    exported["state"]["height"][0][:3] = [0.0, -0.0, 1.0]
    browser_export = browser_json_roundtrip(exported)
    assert type(browser_export["state"]["height"][0][0]) is int
    imported = client.post("/api/import", json=browser_export)
    assert imported.status_code == 200
    assert imported.json()["hash"] == created["hash"]
    assert imported.json()["state"] == created["state"]
    changed = client.post(f'/api/worlds/{imported.json()["id"]}/edit', json={"kind": "raise", "x": 0, "z": 0, "revision": 0})
    assert changed.status_code == 200
    assert changed.json()["hash"] != created["hash"]
    assert changed.json()["state"]["height"][63][63] == state["height"][63][63]


@pytest.mark.parametrize("cell", ["0.25", True, False])
@pytest.mark.parametrize("field", ["height", "ecology"])
def test_import_rejects_string_and_boolean_field_cells(client, state, field, cell):
    if field == "height":
        state[field][0][0] = cell
    else:
        state[field][0][0][0] = cell
    response = client.post("/api/import", json={"state": state})
    assert response.status_code == 422
    assert client.get("/api/worlds").json() == []


def create_legacy_store(path, state):
    original = state_with_numeric_variants(copy.deepcopy(state))
    edited = copy.deepcopy(original)
    edited["height"][20][30] = .8765432198765432
    browser_copy = browser_json_roundtrip(original)
    original_hash, edited_hash, browser_hash = map(legacy_digest, (original, edited, browser_copy))
    assert original_hash != browser_hash
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE blobs(hash TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE worlds(id TEXT PRIMARY KEY, name TEXT NOT NULL,
            parent TEXT, created REAL NOT NULL, revision INTEGER NOT NULL);
        CREATE TABLE history(world TEXT REFERENCES worlds(id), revision INTEGER,
            hash TEXT REFERENCES blobs(hash), event TEXT NOT NULL, created REAL NOT NULL,
            PRIMARY KEY(world,revision));
    """)
    for value in (original, edited, browser_copy):
        db.execute("INSERT INTO blobs VALUES (?,?)", (legacy_digest(value), legacy_json(value)))
    db.executemany("INSERT INTO worlds VALUES (?,?,?,?,?)", [
        ("source", "Source", None, 1000., 1),
        ("child", "Child", "source", 1001., 0),
        ("browser", "Browser import", None, 1002., 0),
    ])
    histories = [
        ("source", 0, original_hash, legacy_json({"type": "generate"}), 1000.),
        ("source", 1, edited_hash, legacy_json({"type": "edit", "strength": .25}), 1003.),
        ("child", 0, original_hash, legacy_json({"type": "branch", "source": "source", "revision": 0, "hash": original_hash}), 1001.),
        ("browser", 0, browser_hash, legacy_json({"type": "import"}), 1002.),
    ]
    db.executemany("INSERT INTO history VALUES (?,?,?,?,?)", histories)
    db.commit()
    blobs = db.execute("SELECT hash,payload FROM blobs ORDER BY hash").fetchall()
    worlds = db.execute("SELECT * FROM worlds ORDER BY id").fetchall()
    db.close()
    return original, edited, blobs, worlds, sorted(histories)


def test_legacy_migration_preserves_worlds_history_and_branch_references(tmp_path, state):
    path = tmp_path / "legacy.sqlite"
    original, edited, old_blobs, old_worlds, old_history = create_legacy_store(path, state)
    store = WorldStore(path)
    try:
        assert store.get("source", 0)["state"] == original
        assert store.get("source")["state"] == edited
        assert store.get("source")["state"]["height"][20][30] == .8765432198765432
        assert store.get("child")["hash"] == digest(original)
        assert store.get("browser")["hash"] == digest(original)
        assert store.history("child")[0]["event"]["hash"] == digest(original)
        assert [tuple(row) for row in store.db.execute("SELECT * FROM worlds ORDER BY id")] == old_worlds
        assert [tuple(row) for row in store.db.execute("SELECT world,revision,hash,event,created FROM canonical_number_history_backup ORDER BY world,revision")] == old_history
        archive = {row["old_hash"]: row["payload"] for row in store.db.execute("SELECT old_hash,payload FROM canonical_number_blob_backup")}
        current = {row["hash"]: row["payload"] for row in store.db.execute("SELECT hash,payload FROM blobs")}
        for old_hash, old_payload in old_blobs:
            assert archive.get(old_hash, current.get(old_hash)) == old_payload, "Original bytes were discarded"
        assert len(current) == 2, "Equivalent browser/numeric variants did not deduplicate"
        for new_hash, payload in current.items():
            assert hashlib.sha256(payload.encode()).hexdigest() == new_hash
            assert digest(json.loads(payload)) == new_hash
        current_history = {name: store.history(name) for name in ("source", "child", "browser")}
        backups = [tuple(row) for row in store.db.execute("SELECT * FROM canonical_number_blob_backup ORDER BY old_hash")]
    finally:
        store.close()
    reopened = WorldStore(path)
    try:
        assert {name: reopened.history(name) for name in current_history} == current_history
        assert [tuple(row) for row in reopened.db.execute("SELECT * FROM canonical_number_blob_backup ORDER BY old_hash")] == backups
        restored = reopened.restore("source", 0, 1)
        assert restored["hash"] == digest(original)
        assert restored["revision"] == 2
        assert reopened.history("source")[1:] == current_history["source"]
    finally:
        reopened.close()


def test_invalid_legacy_checksum_rolls_back_all_migration_changes(tmp_path, state):
    path = tmp_path / "damaged.sqlite"
    create_legacy_store(path, state)
    db = sqlite3.connect(path)
    old_hash = db.execute("SELECT hash FROM blobs ORDER BY hash LIMIT 1").fetchone()[0]
    db.execute("UPDATE blobs SET payload=? WHERE hash=?", ('{"changed":true}', old_hash))
    db.commit()
    before = list(db.iterdump())
    db.close()
    with pytest.raises(ValueError, match="checksum"):
        WorldStore(path)
    db = sqlite3.connect(path)
    try:
        assert list(db.iterdump()) == before
    finally:
        db.close()
