"""Immutable content-addressed world snapshots and branch histories.

Persistence here is explicit software state, not an emergent neural memory claim.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np

SCHEMA = "worldline.world.v1"
BIOMES = ("alpine", "desert", "alien")


def canonical(value):
    """Canonical JSON with the same spelling for equivalent finite numbers.

    Browsers write 0.0, -0.0 and 1.0 as 0, 0 and 1. Integral floats therefore
    use integer spelling here too. Fractional values retain their full Python
    float precision; booleans remain JSON booleans, not numbers.
    """
    def normalize(item):
        if isinstance(item, float) and item.is_integer():
            return int(item)
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        return item

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(state):
    return hashlib.sha256(canonical(state).encode()).hexdigest()


def parse_prompt(prompt):
    """Small explicit keyword vocabulary; this is not a language model."""
    lower = prompt.lower()
    desert = ("desert", "dune", "sand", "canyon", "arid")
    alien = ("alien", "crystal", "bioluminescent", "cosmic", "extraterrestrial")
    biome = 2 if any(x in lower for x in alien) else 1 if any(x in lower for x in desert) else 0
    return biome


def validate_state(state):
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ValueError("Unsupported world format")
    if type(state.get("seed")) is not int or not 0 <= state["seed"] <= 2**31 - 1:
        raise ValueError("Seed must be an integer from 0 to 2147483647")
    if type(state.get("biome")) is not int or state["biome"] not in (0, 1, 2):
        raise ValueError("Unknown biome")
    if not isinstance(state.get("prompt"), str) or len(state["prompt"]) > 600:
        raise ValueError("Prompt must be at most 600 characters")
    for key, shape, lo, hi in (("height", (64, 64), -1, 1), ("ecology", (3, 64, 64), 0, 1)):
        try:
            cells = np.asarray(state.get(key), dtype=object)
            if cells.shape != shape or any(type(cell) not in (int, float) for cell in cells.flat):
                raise ValueError(f"Invalid {key} field: cells must be JSON numbers")
            arr = np.asarray(state.get(key), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid {key} field") from exc
        if arr.shape != shape or not np.isfinite(arr).all() or arr.min() < lo or arr.max() > hi:
            raise ValueError(f"Invalid {key} field")
    tick = state.get("tick", 0)
    if type(tick) is not int or not 0 <= tick <= 10**9:
        raise ValueError("Invalid simulation time")
    weather = state.get("weather", {})
    if not isinstance(weather, dict):
        raise ValueError("Invalid weather")
    for key in ("rain", "heat"):
        value = weather.get(key, 0)
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Weather must be between zero and one")
    # Drop unknown fields instead of saving arbitrary imported content.
    return {"schema": SCHEMA, "seed": state["seed"], "biome": state["biome"],
            "prompt": state["prompt"], "height": state["height"], "ecology": state["ecology"],
            "tick": tick, "weather": {"rain": weather.get("rain", 0), "heat": weather.get("heat", 0)}}


def edit_field(state, kind, x, z, radius, strength):
    """Apply a compact-support edit. Every cell outside the brush stays identical."""
    out = copy.deepcopy(state)
    grid = np.linspace(0, 1, 64, dtype=np.float32)
    xx, zz = np.meshgrid(grid, grid)
    distance = np.sqrt((xx - x)**2 + (zz - z)**2)
    brush = np.maximum(0, 1 - distance / radius)**2 * strength
    target = out["height"] if kind in ("raise", "lower") else out["ecology"][{"water": 0, "plant": 1, "heat": 2}[kind]]
    lower = -1 if kind in ("raise", "lower") else 0
    sign = -1 if kind == "lower" else 1
    # Assign only affected cells, preserving even imported double precision values
    # and number representations outside the edit region.
    for row, col in zip(*np.nonzero(brush)):
        target[row][col] = float(np.clip(target[row][col] + sign * float(brush[row, col]), lower, 1))
    return out


class ConflictError(Exception):
    pass


class WorldStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS blobs(hash TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS worlds(id TEXT PRIMARY KEY, name TEXT NOT NULL,
              parent TEXT, created REAL NOT NULL, revision INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS history(world TEXT REFERENCES worlds(id), revision INTEGER,
              hash TEXT REFERENCES blobs(hash), event TEXT NOT NULL, created REAL NOT NULL,
              PRIMARY KEY(world,revision));
        """)
        try:
            self._migrate_numeric_hashes()
        except Exception:
            self.db.close()
            raise

    def _migrate_numeric_hashes(self):
        """Atomically upgrade legacy hashes without discarding saved state.

        Original payloads and history rows are retained in backup tables. IDs,
        revision numbers and timestamps are unchanged. Only numeric spelling
        and the content hashes that refer to it are updated.
        """
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("CREATE TABLE IF NOT EXISTS store_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            version = self.db.execute("SELECT value FROM store_metadata WHERE key='canonical_numbers'").fetchone()
            if version is not None and version["value"] == "1":
                return
            self.db.execute("""CREATE TABLE IF NOT EXISTS canonical_number_blob_backup(
                old_hash TEXT PRIMARY KEY, new_hash TEXT NOT NULL, payload TEXT NOT NULL)""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS canonical_number_history_backup(
                world TEXT, revision INTEGER, hash TEXT, event TEXT, created REAL,
                PRIMARY KEY(world,revision))""")
            # Freeze the IDs only, processing one payload at a time. New blobs
            # inserted during migration must not expand the scan.
            self.db.execute("CREATE TEMP TABLE numeric_hash_source AS SELECT hash FROM blobs")
            changed = False
            for row in self.db.execute("SELECT b.hash,b.payload FROM numeric_hash_source s JOIN blobs b ON b.hash=s.hash"):
                original_hash = hashlib.sha256(row["payload"].encode()).hexdigest()
                if original_hash != row["hash"]:
                    raise ValueError("Saved snapshot checksum is invalid; numeric migration was rolled back")
                payload = canonical(json.loads(row["payload"]))
                new_hash = hashlib.sha256(payload.encode()).hexdigest()
                if new_hash == row["hash"]:
                    continue
                changed = True
                self.db.execute("INSERT INTO canonical_number_blob_backup VALUES (?,?,?)",
                                (row["hash"], new_hash, row["payload"]))
                self.db.execute("INSERT OR IGNORE INTO blobs VALUES (?,?)", (new_hash, payload))
            if changed:
                self.db.execute("INSERT INTO canonical_number_history_backup SELECT world,revision,hash,event,created FROM history")
                self.db.execute("""UPDATE history SET hash=(
                    SELECT new_hash FROM canonical_number_blob_backup WHERE old_hash=history.hash)
                    WHERE hash IN (SELECT old_hash FROM canonical_number_blob_backup)""")
                for row in self.db.execute("SELECT world,revision,event FROM history"):
                    event = json.loads(row["event"])
                    if not isinstance(event, dict) or event.get("type") != "branch" or not isinstance(event.get("hash"), str):
                        continue
                    replacement = self.db.execute("SELECT new_hash FROM canonical_number_blob_backup WHERE old_hash=?",
                                                  (event.get("hash"),)).fetchone()
                    if replacement is not None:
                        event["hash"] = replacement["new_hash"]
                        self.db.execute("UPDATE history SET event=? WHERE world=? AND revision=?",
                                        (canonical(event), row["world"], row["revision"]))
                self.db.execute("DELETE FROM blobs WHERE hash IN (SELECT old_hash FROM canonical_number_blob_backup)")
            self.db.execute("DROP TABLE numeric_hash_source")
            self.db.execute("INSERT OR REPLACE INTO store_metadata VALUES ('canonical_numbers','1')")

    def _blob(self, state):
        h = digest(state)
        self.db.execute("INSERT OR IGNORE INTO blobs VALUES (?,?)", (h, canonical(state)))
        return h

    def create(self, state, name="Untitled world", parent=None, event=None):
        state = validate_state(state)
        with self.lock, self.db:
            wid = uuid.uuid4().hex[:16]
            now = time.time()
            self.db.execute("INSERT INTO worlds VALUES (?,?,?,?,0)", (wid, name[:100], parent, now))
            h = self._blob(state)
            self.db.execute("INSERT INTO history VALUES (?,0,?,?,?)", (wid, h, canonical(event or {"type": "generate"}), now))
            return self.get(wid)

    def get(self, wid, revision=None):
        with self.lock:
            meta = self.db.execute("SELECT * FROM worlds WHERE id=?", (wid,)).fetchone()
            if meta is None:
                raise KeyError(wid)
            rev = meta["revision"] if revision is None else revision
            row = self.db.execute("SELECT h.hash,b.payload FROM history h JOIN blobs b ON h.hash=b.hash WHERE h.world=? AND h.revision=?", (wid, rev)).fetchone()
            if row is None:
                raise KeyError(f"Revision {rev}")
            return {**dict(meta), "revision": rev, "hash": row["hash"], "state": json.loads(row["payload"])}

    def save(self, wid, state, event, expected_revision):
        state = validate_state(state)
        with self.lock, self.db:
            current = self.get(wid)
            if current["revision"] != expected_revision:
                raise ConflictError("World changed in another window. Reload it before editing.")
            rev = current["revision"] + 1
            h = self._blob(state)
            self.db.execute("INSERT INTO history VALUES (?,?,?,?,?)", (wid, rev, h, canonical(event), time.time()))
            self.db.execute("UPDATE worlds SET revision=? WHERE id=?", (rev, wid))
            return self.get(wid)

    def branch(self, wid, name, revision=None):
        source = self.get(wid, revision)
        return self.create(source["state"], name, wid, {"type": "branch", "source": wid, "revision": source["revision"], "hash": source["hash"]})

    def restore(self, wid, revision, expected_revision):
        old = self.get(wid, revision)
        return self.save(wid, old["state"], {"type": "restore", "revision": revision}, expected_revision)

    def history(self, wid):
        self.get(wid)
        with self.lock:
            return [{**dict(row), "event": json.loads(row["event"])} for row in self.db.execute("SELECT revision,hash,event,created FROM history WHERE world=? ORDER BY revision DESC", (wid,))]

    def list(self):
        with self.lock:
            return [dict(row) for row in self.db.execute("SELECT * FROM worlds ORDER BY created DESC LIMIT 100")]

    def close(self):
        self.db.close()
