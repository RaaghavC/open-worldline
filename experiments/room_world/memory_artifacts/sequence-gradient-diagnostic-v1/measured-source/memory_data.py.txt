# SPDX-License-Identifier: Apache-2.0
"""Original paired RGB trajectories for the prospective recurrent-memory study.

Only this data/validation module imports the teacher. Models consume normalized
RGB and integer actions. The reserved test split is deliberately not exposed by
the development-data command while the training/evaluation protocol is unfinished.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .simulator import Room

DEVELOPMENT_SCENES = {"train": tuple(range(5000, 5032)),
                      "validation": tuple(range(300000, 300008))}
RESERVED_TEST_SCENES = tuple(range(400000, 400032))


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def make_pair(scene_seed, episode_seed, *, wait_steps=16, successful=True, size=64):
    """Observe the first interaction, then turn away, wait and return.

    observations[:,t+1] is the result of actions[:,t]. Both branches begin with
    exactly the same closed-door image. They differ only in the initial command.
    In the unsuccessful control, both cameras initially face away from the door.
    No state variable is returned as a model input.
    """
    if type(wait_steps) is not int or not 4 <= wait_steps <= 64:
        raise ValueError("wait_steps must be an integer between 4 and 64")
    if type(successful) is not bool:
        raise ValueError("successful must be boolean")
    initial = Room(scene_seed, episode_seed, size)
    if not successful:
        initial.yaw += np.pi
    shared = [3]*24 + [0]*wait_steps + [4]*24
    actions = np.asarray([[0, *shared], [5, *shared]], dtype=np.int64)
    observations, doors = [], []
    for commands in actions:
        room = initial.clone()
        frames, states = [room.render()], [room.door_open]
        for action in commands:
            room.step(int(action))
            frames.append(room.render())
            states.append(room.door_open)
        observations.append(np.stack(frames))
        doors.append(states)
    rgb = np.stack(observations)
    door_states = np.asarray(doors, dtype=bool)
    if not np.array_equal(rgb[0, 0], rgb[1, 0]):
        raise RuntimeError("Branches do not start from the same observed image")
    if bool(door_states[1, 1]) != successful or door_states[0].any():
        raise RuntimeError("Teacher intervention did not match the declared control")
    first_return_action = 1 + 24 + wait_steps
    prefix = rgb[:, first_return_action-3:first_return_action+1]
    if successful:
        if not np.array_equal(prefix[0], prefix[1]):
            raise RuntimeError("Door state is visible in the four-frame return input")
        if np.array_equal(rgb[0, -1], rgb[1, -1]):
            raise RuntimeError("Return does not reveal the intervened door")
    elif not np.array_equal(rgb[0], rgb[1]):
        raise RuntimeError("Unsuccessful interaction altered the observed trajectory")
    record = {
        "scene_seed": int(scene_seed), "episode_seed": int(episode_seed),
        "successful_initial_interaction": successful, "wait_steps": wait_steps,
        "steps": len(shared)+1, "size": size, "branch_order": ["wait", "interact"],
        "first_return_action_index": first_return_action,
        "observations_sha256": array_hash(rgb), "actions_sha256": array_hash(actions),
        "identical_initial_observation": True,
        "identical_last_four_observed_frames_before_return": bool(np.array_equal(prefix[0], prefix[1])),
        "final_rgb_differs": not np.array_equal(rgb[0, -1], rgb[1, -1]),
        "teacher_door_states": door_states.tolist(),
        "scope": "Original teacher data and validation metadata; model inputs are RGB and action ids only.",
    }
    return {"observations": rgb, "actions": actions, "record": record}


def observed_history(observations, transition):
    """Four RGB observations available before one transition; repeat start if needed."""
    if not isinstance(observations, np.ndarray) or observations.dtype != np.uint8:
        raise ValueError("observations must be uint8 RGB")
    if observations.ndim != 4 or observations.shape[1] != 3:
        raise ValueError("Expected [T,3,H,W] observations")
    if type(transition) is not int or not 0 <= transition < len(observations)-1:
        raise ValueError("transition outside observed trajectory")
    indices = np.maximum(np.arange(transition-3, transition+1), 0)
    return observations[indices].astype(np.float32)/127.5-1


def write_development(output, split, *, size=64):
    """Materialize standard 65-step training or validation pairs only."""
    if split not in DEVELOPMENT_SCENES:
        raise ValueError("Only train and validation development splits can be generated")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema": "worldline-room-memory-development-v1", "status": "running",
        "split": split, "scene_seeds": list(DEVELOPMENT_SCENES[split]),
        "reserved_test_generated": False, "size": size, "steps": 65,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("simulator.py")]},
        "records": [],
        "model_input_fields": ["observations", "actions"],
        "validation_only_fields": ["record", "scene_seed", "episode_seed", "teacher_door_states"],
        "limits": "Single renderer family, new scene seeds; no learned memory result or frontier graphics claim.",
    }
    def save_manifest():
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    save_manifest()
    try:
        for scene in DEVELOPMENT_SCENES[split]:
            case = make_pair(scene, scene*1009+37, size=size)
            path = output/f"scene-{scene}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, observations=case["observations"], actions=case["actions"])
            manifest["records"].append({**case["record"], "file": path.name,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "file_bytes": path.stat().st_size})
            save_manifest()
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic()-started
        save_manifest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=list(DEVELOPMENT_SCENES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    result = write_development(args.output, args.split, size=args.size)
    print(json.dumps({k:result[k] for k in ["status", "split", "elapsed_seconds", "reserved_test_generated"]}))


if __name__ == "__main__":
    main()
