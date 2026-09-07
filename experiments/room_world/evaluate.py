"""Held-out visual prediction measurements and fixed rollout contact sheet."""
import struct
import time
import zlib

import numpy as np
import torch

from .model import predict_tensor, rollout


def _mae(prediction, truth):
    return float(np.abs(prediction-truth).mean()/2)


def _psnr(prediction, truth):
    mse = float(np.square((prediction-truth)/2).mean())
    return float(-10*np.log10(max(mse, 1e-12)))


def write_png(path, rgb):
    """Write an RGB uint8 scientific contact sheet without an image dependency."""
    rgb = np.asarray(rgb, np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("RGB required")
    def chunk(kind, value):
        return struct.pack(">I", len(value))+kind+value+struct.pack(">I", zlib.crc32(kind+value)&0xffffffff)
    scanlines = b"".join(b"\x00"+row.tobytes() for row in rgb)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(scanlines, 6))+chunk(b"IEND", b"")
    path.write_bytes(payload)


def contact_sheet(path, sequences, horizons):
    """Each episode has truth, prediction, absolute-error rows; columns are horizons."""
    rows = []
    for truth, prediction in sequences:
        for value in (truth, prediction, np.abs(truth-prediction)-1):
            cells = [np.moveaxis(np.clip((value[h-1]+1)*127.5, 0, 255).astype(np.uint8), 0, -1)
                     for h in horizons]
            rows.append(np.concatenate(cells, 1))
    write_png(path, np.concatenate(rows, 0))


@torch.inference_mode()
def evaluate(model, dataset, device="cpu", sample_steps=8, output=None):
    model.eval()
    # Every transition is scored. Batch boundaries and random seeds are fixed.
    histories, targets, controls = [], [], []
    for episode in range(len(dataset["frames"])):
        for step in range(dataset["steps"]):
            histories.append(dataset["frames"][episode, step:step+4])
            targets.append(dataset["frames"][episode, step+4])
            controls.append(dataset["actions"][episode, step])
    histories = np.asarray(histories, np.float32)/127.5-1
    targets = np.asarray(targets, np.float32)/127.5-1
    controls = np.asarray(controls, np.int64)
    predictions, zero_predictions = [], []
    for offset in range(0, len(histories), 16):
        history = torch.from_numpy(histories[offset:offset+16]).to(device)
        action = torch.from_numpy(controls[offset:offset+16]).to(device)
        # Common noise isolates the effect of replacing the action with wait.
        rng = torch.Generator(device="cpu").manual_seed(400_001+offset)
        zero_rng = torch.Generator(device="cpu").manual_seed(400_001+offset)
        predictions.append(predict_tensor(model, history, action, sample_steps, rng).cpu().numpy())
        zero_predictions.append(predict_tensor(model, history, torch.zeros_like(action), sample_steps, zero_rng).cpu().numpy())
    predictions, zero_predictions = np.concatenate(predictions), np.concatenate(zero_predictions)
    previous = histories[:, -1]
    moving = (np.abs(targets-previous).max(1, keepdims=True) > .04)
    moving = np.broadcast_to(moving, targets.shape)
    def moving_mae(value):
        return float(np.abs(value-targets)[moving].mean()/2) if moving.any() else None
    one_step = {"count": len(targets), "model_mae": _mae(predictions, targets),
                "model_psnr": _psnr(predictions, targets),
                "repeat_last_mae": _mae(previous, targets), "zero_action_mae": _mae(zero_predictions, targets),
                "model_moving_mae": moving_mae(predictions), "repeat_last_moving_mae": moving_mae(previous),
                "zero_action_moving_mae": moving_mae(zero_predictions), "by_action": {}}
    for action in range(6):
        mask = controls == action
        if mask.any():
            one_step["by_action"][str(action)] = {"count": int(mask.sum()),
                "model_mae": _mae(predictions[mask], targets[mask]),
                "repeat_last_mae": _mae(previous[mask], targets[mask]),
                "zero_action_mae": _mae(zero_predictions[mask], targets[mask])}
    horizons = sorted(set(h for h in [1, 4, 16, 20, dataset["steps"]] if h <= dataset["steps"]))
    per_episode, sequences, latencies = [], [], []
    for episode in range(len(dataset["frames"])):
        initial = dataset["frames"][episode, :4].astype(np.float32)/127.5-1
        actions = dataset["actions"][episode]
        truth = dataset["frames"][episode, 4:].astype(np.float32)/127.5-1
        start = time.perf_counter()
        predicted = rollout(model, initial, actions, device, sample_steps, seed=500_001+episode)
        elapsed = time.perf_counter()-start
        latencies.append(elapsed/len(actions))
        zero = rollout(model, initial, np.zeros_like(actions), device, sample_steps, seed=500_001+episode)
        persistence = np.repeat(initial[-1:], len(actions), 0)
        scores = {str(h): {"model_mae": _mae(predicted[h-1], truth[h-1]),
                          "model_psnr": _psnr(predicted[h-1], truth[h-1]),
                          "repeat_last_mae": _mae(persistence[h-1], truth[h-1]),
                          "zero_action_mae": _mae(zero[h-1], truth[h-1])} for h in horizons}
        per_episode.append({"scene_seed": dataset["records"][episode]["scene_seed"],
                            "trajectory_seed": dataset["records"][episode]["trajectory_seed"],
                            "horizons": scores})
        if len(sequences) < 4:
            sequences.append((truth, predicted))
    aggregate = {str(h): {key: float(np.mean([record["horizons"][str(h)][key] for record in per_episode]))
                         for key in per_episode[0]["horizons"][str(h)]} for h in horizons}
    if output is not None:
        contact_sheet(output / "heldout-rollouts.png", sequences, horizons)
        np.savez_compressed(output / "heldout-first-rollout.npz", truth=sequences[0][0],
                            predicted=sequences[0][1], actions=dataset["actions"][0])
    return {"split": dataset["split"], "sample_steps": sample_steps, "one_step": one_step,
            "closed_loop": {"initial_true_frames": 4, "teacher_frames_after_start": 0,
                            "episodes": len(per_episode), "horizons": aggregate, "per_episode": per_episode},
            "timing": {"mean_episode_seconds_per_neural_frame": float(np.mean(latencies)),
                       "note": "Includes CPU transfer, first-call overhead and autoregressive history update; not renderer FPS."},
            "limits": ["One fixed noise seed per episode; this pilot is not a distribution-quality benchmark.",
                       "Pixel metrics can favor blur. Inspect the fixed contact sheet and saved rollout.",
                       "Both data and truth are original synthetic rooms; no broad world-model parity is tested."]}
