# SPDX-License-Identifier: Apache-2.0
"""Validate the eight already captured development failed-interaction controls.

No capture file is modified. The black/white door renders are teacher-only
visibility probes. Neither their pixels nor their derived metadata are model
inputs, targets for training, or results from the neural predictor.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
from experiments.room_world.simulator import Room

SCENES = tuple(range(300000, 300008))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def audit(captures):
    captures = Path(captures).resolve()
    manifest_path = captures / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "worldline-room-memory-controls-v1"
            or manifest.get("status") != "complete" or manifest.get("split") != "validation"
            or manifest.get("scene_seeds") != list(SCENES)
            or manifest.get("reserved_test_generated") is not False):
        raise ValueError("Only the completed eight-scene development control capture is accepted")
    records = [record for record in manifest["records"] if record["kind"] == "out_of_reach_interaction"]
    if [record["scene_seed"] for record in records] != list(SCENES):
        raise ValueError("Missing, duplicated or unplanned development control scenes")
    source_files = {
        "audit_visibility.py": Path(__file__),
        "simulator.py": PROJECT / "experiments/room_world/simulator.py",
        "memory_controls.py": PROJECT / "experiments/room_world/memory_controls.py",
        "memory_data.py": PROJECT / "experiments/room_world/memory_data.py",
    }
    source_hashes = {name: file_hash(path) for name, path in source_files.items()}
    results = []
    for record in records:
        relative = Path(record["file"])
        path = (captures / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(captures):
            raise ValueError("Capture path leaves the declared directory")
        before_file_hash = file_hash(path)
        if before_file_hash != record["file_sha256"]:
            raise ValueError("Captured file hash differs")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"observations", "actions"}:
                raise ValueError("Unexpected capture arrays")
            observations, actions = archive["observations"], archive["actions"]
        if observations.shape != (1, 17, 3, 64, 64) or observations.dtype != np.uint8:
            raise ValueError("Unexpected observed RGB shape/dtype")
        if actions.shape != (1, 16) or actions.dtype != np.int64:
            raise ValueError("Unexpected action shape/dtype")
        if actions[0].tolist() != [2] * 6 + [5] + [0] * 3 + [1] * 6:
            raise ValueError("The fixed failed-interaction command sequence differs")
        if array_hash(observations) != record["observations_sha256"] or array_hash(actions) != record["actions_sha256"]:
            raise ValueError("Captured raw-array hash differs")
        scene, episode = record["scene_seed"], record["episode_seed"]
        if scene not in SCENES or episode != scene * 1009 + 37:
            raise ValueError("Unexpected development scene or episode seed")
        room = Room(scene, episode, 64)
        for action in actions[0, :6]:
            room.step(int(action))
        reconstructed = room.render()
        if not np.array_equal(reconstructed, observations[0, 6]):
            raise RuntimeError("Reconstructed teacher frame does not equal captured observation6")
        state = room.state()
        distance = math.hypot(room.x, room.z)
        facing = (-room.x * math.sin(room.yaw) - room.z * math.cos(room.yaw)) / max(distance, 0.001)
        if distance < 2.3 or facing <= 0.35 or room.door_open:
            raise RuntimeError("The camera is not an out-of-range, door-facing closed-door case")
        black, white = room.clone(), room.clone()
        black.door_color = np.zeros(3, dtype=np.float32)
        white.door_color = np.ones(3, dtype=np.float32)
        black_rgb, white_rgb = black.render(), white.render()
        difference = np.abs(white_rgb.astype(np.int16) - black_rgb.astype(np.int16))
        visible = (difference != 0).any(axis=0)
        if not visible.any():
            raise RuntimeError("Changing only door color produced no visible pixels")
        y, x = np.nonzero(visible)
        room.step(int(actions[0, 6]))
        if room.state() != state or not np.array_equal(room.render(), observations[0, 7]):
            raise RuntimeError("The failed interaction changed teacher state or differs from capture")
        if file_hash(path) != before_file_hash:
            raise RuntimeError("Capture changed during read-only validation")
        results.append({
            "scene_seed": scene, "episode_seed": episode, "file": relative.as_posix(),
            "capture_file_sha256": before_file_hash,
            "captured_observation6_sha256": array_hash(observations[0, 6]),
            "reconstructed_observation6_sha256": array_hash(reconstructed),
            "captured_image_exactly_reproduced": True, "action_index": 6, "action_id": 5,
            "door_distance": distance, "minimum_success_distance_excluded": 2.3, "facing_cosine": facing,
            "door_open_before": False, "door_open_after": room.door_open, "teacher_state_unchanged": True,
            "visible_door_pixel_count": int(visible.sum()), "image_pixel_count": int(visible.size),
            "visibility_bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
            "black_door_rgb_sha256": array_hash(black_rgb), "white_door_rgb_sha256": array_hash(white_rgb),
            "visibility_mask_sha256": array_hash(visible), "maximum_channel_difference_uint8": int(difference.max()),
        })
    if {name: file_hash(path) for name, path in source_files.items()} != source_hashes:
        raise RuntimeError("Source changed during validation")
    return {
        "schema": "worldline-room-control-visibility-audit-v1", "status": "passed",
        "device": "cpu", "numpy": np.__version__, "python": platform.python_version(),
        "scene_seeds": list(SCENES), "reserved_test_opened": False,
        "capture_manifest_sha256": file_hash(manifest_path), "source_sha256": source_hashes,
        "method": "Reconstruct captured state before action6; compare stored RGB exactly; independently render door black and white with all other scene properties fixed.",
        "visibility_scope": "Direct-light ray-box renderer has no indirect-light propagation; changed pixels identify direct door-color visibility.",
        "training_inputs_changed": False, "capture_files_changed": False,
        "validation_only": True, "neural_model_executed": False, "cases": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=PROJECT / "experiments/room_world/memory_artifacts/data/validation-controls")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Audit output must be new")
    report = audit(args.captures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                      "visible_pixels": [case["visible_door_pixel_count"] for case in report["cases"]]}))
