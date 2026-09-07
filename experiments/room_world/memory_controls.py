# SPDX-License-Identifier: Apache-2.0
"""Fixed development controls; original RGB teacher data, never model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .memory_data import DEVELOPMENT_SCENES, array_hash, make_pair
from .simulator import Room

CONTROL_ACTIONS = {
    "translation_cycle": (1,)*4 + (2,)*8 + (1,)*4,
    "out_of_reach_interaction": (2,)*6 + (5,) + (0,)*3 + (1,)*6,
    "turn_open_close": (3,)*4 + (4,)*4 + (5,) + (0,)*3 + (5,) + (0,)*3,
}
PAIRED_WAITS = (8, 32)


def make_control(scene_seed, episode_seed, kind, *, size=64):
    if kind not in CONTROL_ACTIONS:
        raise ValueError("Unknown fixed control type")
    room = Room(scene_seed, episode_seed, size)
    rgb, states = [room.render()], [room.state()]
    actions = np.asarray(CONTROL_ACTIONS[kind], dtype=np.int64)
    for action in actions:
        room.step(int(action))
        rgb.append(room.render())
        states.append(room.state())
    moved = []
    toggles = []
    for step, action in enumerate(actions):
        before, after = states[step], states[step+1]
        if action in (1, 2):
            moved.append(bool(np.hypot(after["x"]-before["x"], after["z"]-before["z"]) > .17))
        if action == 5:
            toggles.append(bool(before["door_open"] != after["door_open"]))
    if moved and not all(moved):
        raise RuntimeError("A declared translation did not move the camera")
    if kind == "out_of_reach_interaction":
        if toggles != [False] or any(state["door_open"] for state in states):
            raise RuntimeError("The out-of-reach interaction unexpectedly changed the door")
    if kind == "turn_open_close":
        if toggles != [True, True] or states[-1]["door_open"]:
            raise RuntimeError("The opening/closing control did not toggle twice")
    observations = np.stack(rgb)[None]
    actions = actions[None]
    return {"observations": observations, "actions": actions, "record": {
        "scene_seed": int(scene_seed), "episode_seed": int(episode_seed), "kind": kind,
        "steps": 16, "size": size, "translations_verified": len(moved),
        "interaction_changed_door": toggles, "teacher_door_states": [s["door_open"] for s in states],
        "observations_sha256": array_hash(observations), "actions_sha256": array_hash(actions),
        "scope": "Teacher validation metadata is excluded from model inputs.",
    }}


def write_validation_controls(output, *, size=64):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema": "worldline-room-memory-controls-v1", "status": "running",
        "split": "validation", "scene_seeds": list(DEVELOPMENT_SCENES["validation"]),
        "size": size, "reserved_test_generated": False,
        "control_actions": {key:list(value) for key,value in CONTROL_ACTIONS.items()},
        "paired_waits": list(PAIRED_WAITS), "records": [],
        "source_sha256": {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ["memory_controls.py", "memory_data.py", "simulator.py"]},
        "model_input_fields": ["observations", "actions"],
        "limits": "Fixed development controls from one programmed room family. No learned-control result.",
    }
    def save():
        (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    save()
    try:
        for scene in DEVELOPMENT_SCENES["validation"]:
            episode = scene*1009+37
            cases = [(kind,make_control(scene,episode,kind,size=size)) for kind in CONTROL_ACTIONS]
            cases += [(f"paired_wait_{waiting}",make_pair(scene,episode,wait_steps=waiting,size=size)) for waiting in PAIRED_WAITS]
            for kind,case in cases:
                path=output/f"scene-{scene}-{kind}.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(handle,observations=case["observations"],actions=case["actions"])
                manifest["records"].append({**case["record"],"kind":kind,"file":path.name,
                    "file_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"file_bytes":path.stat().st_size})
                save()
        manifest["status"]="complete"
    except BaseException as error:
        manifest.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        manifest["elapsed_seconds"]=time.monotonic()-started
        save()
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--size",type=int,default=64)
    args=parser.parse_args()
    manifest=write_validation_controls(args.output,size=args.size)
    print(json.dumps({key:manifest[key] for key in ["status","elapsed_seconds","reserved_test_generated"]}))


if __name__=="__main__":
    main()
