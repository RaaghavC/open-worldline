"""Disjoint scene/trajectory splits and original action-conditioned RGB data."""
import hashlib

import numpy as np

from .simulator import Room


SPLIT_OFFSETS = {"train": 1_000, "validation": 100_000, "test": 200_000}


def trajectory_actions(seed, count, episode_index=0):
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < count:
        action = int(rng.choice(6, p=[.12, .26, .13, .18, .18, .13]))
        hold = 1 if action == 5 else int(rng.integers(1, 5))
        result.extend([action] * hold)
    if episode_index % 2 == 0:
        # A reproducible successful interaction followed by camera movement.
        prefix = [1, 1, 1, 5, 0, 0, 1, 1, 3, 3, 3, 4, 4, 4, 2, 2, 5]
        result[:min(count, len(prefix))] = prefix[:count]
    else:
        # An early unsuccessful interaction while facing away.
        prefix = [3]*12 + [5] + [4]*12 + [5]
        result[:min(count, len(prefix))] = prefix[:count]
    return np.asarray(result[:count], dtype=np.int64)


def make_dataset(split, scenes=16, episodes_per_scene=2, steps=32, size=64, log=None):
    if split not in SPLIT_OFFSETS or not 0 < scenes < 10_000:
        raise ValueError("invalid split or scene count")
    if episodes_per_scene < 1 or steps < 20:
        raise ValueError("at least one episode and twenty transitions are required")
    frames, actions, records = [], [], []
    for scene_index in range(scenes):
        scene_seed = SPLIT_OFFSETS[split] + scene_index
        for episode in range(episodes_per_scene):
            trajectory_seed = scene_seed * 1009 + episode * 9176 + 37
            room = Room(scene_seed, trajectory_seed, size)
            control = trajectory_actions(trajectory_seed, steps, episode)
            observation = room.render()
            sequence = [observation.copy() for _ in range(4)]
            states = [room.state()]
            for action in control:
                room.step(int(action))
                sequence.append(room.render())
                states.append(room.state())
            array = np.stack(sequence)
            frames.append(array)
            actions.append(control)
            records.append({"scene_seed": scene_seed, "trajectory_seed": trajectory_seed,
                            "episode_index": episode, "initial_state": states[0],
                            "door_states": [s["door_open"] for s in states],
                            "sha256": hashlib.sha256(array.tobytes()+control.tobytes()).hexdigest()})
        if log is not None:
            log({"event": "data", "split": split, "completed_scenes": scene_index+1, "scenes": scenes})
    return {"frames": np.stack(frames), "actions": np.stack(actions),
            "records": records, "split": split, "size": size, "steps": steps}


def sample_batch(dataset, count, rng):
    episode = rng.integers(0, len(dataset["frames"]), count)
    offset = rng.integers(0, dataset["steps"], count)
    history = np.stack([dataset["frames"][e, k:k+4] for e, k in zip(episode, offset)])
    target = dataset["frames"][episode, offset+4]
    action = dataset["actions"][episode, offset]
    return history.astype(np.float32)/127.5-1, action, target.astype(np.float32)/127.5-1
