"""Measure a four-frame model's inability to remember an invisible door state.

The teacher creates and scores the two cases. Model inference receives only four
RGB frames, the common return actions, and identical sampled noise. It receives
no scene seed, door label, camera, geometry, or teacher callback.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .simulator import Room


def make_aliasing_case(scene_seed=200_000, episode_seed=20_260_907, size=64):
    """Return reachable open/closed cases after a 64-action shared excursion."""
    closed = Room(scene_seed, episode_seed, size)
    opened = closed.clone()
    opened.step(5)
    if closed.door_open or not opened.door_open:
        raise RuntimeError("starting pose did not permit the opening intervention")
    # Turn 180 degrees, wait with the door behind the camera, then turn back.
    # The 40-step prefix removes the door from the four-frame inference window.
    away_actions = np.asarray([3] * 24 + [0] * 16, dtype=np.int64)
    return_actions = np.asarray([4] * 24, dtype=np.int64)
    initial = np.stack([closed.render(), opened.render()])
    away, truth = [], []
    for room in (closed, opened):
        sequence = []
        for action in away_actions:
            room.step(int(action))
            sequence.append(room.render())
        away.append(np.stack(sequence))
        sequence = []
        for action in return_actions:
            room.step(int(action))
            sequence.append(room.render())
        truth.append(np.stack(sequence))
    away, truth = np.stack(away), np.stack(truth)
    histories = away[:, -4:].copy()
    if not np.array_equal(histories[0], histories[1]):
        raise RuntimeError("door remained visible in the four-frame input window")
    if np.array_equal(truth[0, -1], truth[1, -1]):
        raise RuntimeError("return trajectory did not reveal distinct door states")
    return {"scene_seed": int(scene_seed), "episode_seed": int(episode_seed),
            "initial": initial, "histories": histories, "away_frames": away,
            "truth": truth, "away_actions": away_actions, "actions": return_actions}


def _hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def evaluate_aliasing(model=None, device="cpu", sample_steps=8, output=None,
                      scene_seed=200_000, episode_seed=20_260_907, size=64):
    """Write a teacher-only diagnostic or evaluate a supplied RGB model on it.

    Scores use RGB in [0,1]. A shared prediction has average squared error at
    least one quarter of the squared difference between the two true futures.
    This lower bound is exact and requires no trained model or empirical guess.
    """
    case = make_aliasing_case(scene_seed, episode_seed, size)
    truth = case["truth"].astype(np.float64) / 255
    delta = truth[0] - truth[1]
    mse = np.square(delta).mean((1, 2, 3))
    changed_pixels = (case["truth"][0] != case["truth"][1]).any(1).mean((1, 2))
    differing_steps = np.flatnonzero(changed_pixels > 0)
    report = {
        "experiment": "four-frame invisible-door ambiguity",
        "scene_seed": case["scene_seed"], "episode_seed": case["episode_seed"],
        "size": size, "initial_intervention": "interact opens one reachable door",
        "away_actions": case["away_actions"].tolist(),
        "return_actions": case["actions"].tolist(),
        "shared_excursion_steps": len(case["away_actions"]) + len(case["actions"]),
        "observed_history_frames": 4, "identical_input_histories": True,
        "input_history_sha256": [_hash(x) for x in case["histories"]],
        "return_actions_sha256": _hash(case["actions"]),
        "first_different_return_step": int(differing_steps[0]) + 1,
        "ground_truth_mae_0_to_1_per_step": np.abs(delta).mean((1, 2, 3)).tolist(),
        "ground_truth_changed_pixel_fraction_per_step": changed_pixels.tolist(),
        "shared_prediction_mse_lower_bound_0_to_1_per_step": (mse / 4).tolist(),
        "mean_shared_prediction_mse_lower_bound_0_to_1": float(mse.mean() / 4),
        "final_shared_prediction_mse_lower_bound_0_to_1": float(mse[-1] / 4),
        "interpretation": (
            "These inputs omit the earlier opening action. Both futures are reachable, "
            "but four identical RGB frames and identical future actions cannot identify "
            "which hidden door state is present. A stochastic model may represent both "
            "possibilities; it cannot recover this episode's state from these inputs alone."
        ),
        "limits": [
            "One synthetic scene with two door states, not a population estimate.",
            "The 64-step teacher excursion contains 40 away steps and 24 return steps; "
            "model inference begins from the final four away frames and predicts 24 return frames.",
            "This tests information missing from a finite RGB window. It does not show "
            "that a model supplied with earlier history or learned persistent memory must fail.",
        ],
    }
    predictions, full_predictions = None, None
    if model is not None:
        from .model import rollout
        # No teacher data beyond these normalized RGB histories enters rollout.
        predictions = np.stack([
            rollout(model, history.astype(np.float32) / 127.5 - 1,
                    case["actions"], device, sample_steps, seed=20_260_907)
            for history in case["histories"]
        ])
        if not np.isfinite(predictions).all():
            raise RuntimeError("model produced non-finite RGB")
        predictions01 = (predictions.astype(np.float64) + 1) / 2
        pair_error = np.square(predictions01 - truth).mean((0, 2, 3, 4))
        identical = bool(np.array_equal(predictions[0], predictions[1]))
        report["model"] = {
            "input_protocol": "identical-history structural diagnostic",
            "kind": getattr(model, "kind", type(model).__name__),
            "device": str(device), "sample_steps": sample_steps, "seed": 20_260_907,
            "identical_paired_predictions": identical,
            "max_paired_prediction_difference_normalized_rgb": float(
                np.abs(predictions[0] - predictions[1]).max()),
            "paired_average_mse_0_to_1_per_step": pair_error.tolist(),
            "mean_paired_average_mse_0_to_1": float(pair_error.mean()),
            "lower_bound_applicable_to_this_same_noise_pair": identical,
        }
        # A distinct empirical experiment starts while the door is visible and
        # lets the network feed back its own predictions for the whole excursion.
        # Those generated histories need not become identical while looking away.
        full_actions = np.concatenate([case["away_actions"], case["actions"]])
        full_truth = np.concatenate([case["away_frames"], case["truth"]], axis=1)
        full_predictions = np.stack([
            rollout(model, np.repeat(initial[None], 4, axis=0).astype(np.float32) / 127.5 - 1,
                    full_actions, device, sample_steps, seed=20_260_907)
            for initial in case["initial"]
        ])
        if not np.isfinite(full_predictions).all():
            raise RuntimeError("full model rollout produced non-finite RGB")
        full01 = (full_predictions.astype(np.float64) + 1) / 2
        full_truth01 = full_truth.astype(np.float64) / 255
        full_mae = np.abs(full01 - full_truth01).mean((2, 3, 4))
        report["uninterrupted_model_rollout"] = {
            "input_protocol": "uninterrupted generated-history rollout",
            "initial_true_frames": 4, "initial_true_frames_are_repeated_start_image": True,
            "teacher_frames_after_start": 0, "generated_frames_per_case": len(full_actions),
            "case_order": ["closed", "open"],
            "mae_0_to_1_per_case_per_step": full_mae.tolist(),
            "mean_mae_0_to_1_per_case": full_mae.mean(1).tolist(),
            "final_mae_to_own_truth_0_to_1": full_mae[:, -1].tolist(),
            "final_mae_to_other_case_truth_0_to_1": np.abs(
                full01[:, -1] - full_truth01[::-1, -1]).mean((1, 2, 3)).tolist(),
            "paired_prediction_mae_0_to_1_per_step": np.abs(
                full01[0] - full01[1]).mean((1, 2, 3)).tolist(),
            "paired_ground_truth_mae_0_to_1_per_step": np.abs(
                full_truth01[0] - full_truth01[1]).mean((1, 2, 3)).tolist(),
            "interpretation": (
                "This empirical rollout begins with visibly different door states. "
                "Generated histories may retain differences while the camera looks away. "
                "Its errors measure this model's actual 64-step behavior, separately "
                "from the identical-input structural limit. Pixel distance to the "
                "correct or alternate future is not a semantic door-state classifier."
            ),
        }
    if output is not None:
        from .evaluate import contact_sheet, write_png
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "aliasing-case.npz", **{
            key: value for key, value in case.items() if isinstance(value, np.ndarray)
        }, **({"predictions": predictions, "full_predictions": full_predictions}
              if predictions is not None else {}))
        for name, value in (("initial-closed", case["initial"][0]),
                            ("initial-open", case["initial"][1]),
                            ("identical-away-input", case["histories"][0, -1]),
                            ("return-closed-truth", case["truth"][0, -1]),
                            ("return-open-truth", case["truth"][1, -1])):
            write_png(output / f"aliasing-{name}.png", np.moveaxis(value, 0, -1))
        if predictions is not None:
            for index, name in enumerate(("closed", "open")):
                image = np.clip(np.round((predictions[index, -1] + 1) * 127.5), 0, 255).astype(np.uint8)
                write_png(output / f"aliasing-return-{name}-prediction.png", np.moveaxis(image, 0, -1))
                full_image = np.clip(np.round((full_predictions[index, -1] + 1) * 127.5), 0, 255).astype(np.uint8)
                write_png(output / f"uninterrupted-return-{name}-prediction.png", np.moveaxis(full_image, 0, -1))
            contact_sheet(output / "uninterrupted-turn-away-return.png",
                          [(full_truth[index].astype(np.float32) / 127.5 - 1, full_predictions[index])
                           for index in range(2)], [1, 24, 40, 52, 64])
        (output / "aliasing-metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    model = None
    if args.checkpoint:
        from .model import load_model
        model, _ = load_model(args.checkpoint, args.device)
    report = evaluate_aliasing(model, args.device, args.sample_steps, args.output, size=args.size)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
