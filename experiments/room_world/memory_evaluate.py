# SPDX-License-Identifier: Apache-2.0
"""Separate observed-prefix and uninterrupted RGB memory evaluation.

Generators accept only available RGB, actions and explicit recurrent state.
Paired future truth enters scoring after both generated sequences are complete.
No renderer or simulator state is imported or supplied to a model step.
"""
import torch
from torch import nn


def history_at(observations, transition):
    """Available four-frame history at t, repeating observation0 if t<3."""
    if observations.ndim != 5 or observations.shape[2] != 3:
        raise ValueError("observations must be [B,T,3,H,W]")
    if type(transition) is not int or not 0 <= transition < observations.shape[1]:
        raise ValueError("transition is outside observed prefix")
    indices = torch.arange(transition - 3, transition + 1, device=observations.device).clamp_min(0)
    return observations.index_select(1, indices)


def _check_inputs(observations, actions):
    if observations.ndim != 5 or observations.shape[2] != 3 or not observations.is_floating_point():
        raise ValueError("RGB must be floating [B,T,3,H,W]")
    if not torch.isfinite(observations).all() or observations.abs().max() > 1.00001:
        raise ValueError("RGB must be finite and normalized to [-1,1]")
    if actions.ndim != 2 or actions.shape[0] != observations.shape[0] or actions.dtype != torch.int64:
        raise ValueError("actions must be int64 [B,T]")
    if actions.device != observations.device or (actions < 0).any() or (actions > 5).any():
        raise ValueError("actions must share the RGB device and use ids0..5")


@torch.inference_mode()
def generated_rollout(model, initial_history, actions, state=None, *, mode="carry"):
    """Every frame after the supplied history is recursively model-generated."""
    _check_inputs(initial_history, actions)
    if initial_history.shape[1] != 4 or actions.shape[1] < 1:
        raise ValueError("A rollout needs four initial RGB frames and at least one action")
    model.eval()
    state = model.initial_state(initial_history.shape[0]) if state is None else state.clone()
    history, frames = initial_history.clone(), []
    for step in range(actions.shape[1]):
        prediction, state = model(history, actions[:, step], state, mode=mode)
        if prediction.shape != history[:, -1].shape or not torch.isfinite(prediction).all():
            raise FloatingPointError("Invalid generated RGB frame")
        frames.append(prediction)
        history = torch.cat((history[:, 1:], prediction.unsqueeze(1)), dim=1)
    return torch.stack(frames, dim=1)


@torch.inference_mode()
def observed_prefix_return(model, observed_prefix, prefix_actions, return_actions, *, mode="carry"):
    """Accumulate memory from observed causes, then generate the return suffix.

    For the standard study P=41: warm actions0..40, then use the observed
    frames38..41 and actions41..64 to generate observations42..65. No true
    observation after41 is present in this function's inputs.
    """
    _check_inputs(observed_prefix, prefix_actions)
    if observed_prefix.shape[1] != prefix_actions.shape[1] + 1:
        raise ValueError("Observed prefix needs one more frame than executed prefix actions")
    model.eval()
    state = model.initial_state(observed_prefix.shape[0])
    for step in range(prefix_actions.shape[1]):
        _, state = model(history_at(observed_prefix, step), prefix_actions[:, step], state, mode=mode)
    return generated_rollout(model, history_at(observed_prefix, prefix_actions.shape[1]),
                             return_actions, state, mode=mode)


def score_return(prediction, truth):
    """Score all return frames only at pixels differing between paired truth.

    Primary pair correctness requires BOTH branch predictions to be strictly
    closer to their own truth than the opposite truth on this pooled region.
    Ties fail. No differing pixels makes a pair ineligible for this memory
    score, as with unsuccessful interactions; score those controls separately.
    """
    prediction, truth = prediction.detach().cpu().double(), truth.detach().cpu().double()
    if prediction.shape != truth.shape or truth.ndim != 5 or truth.shape[0] != 2 or truth.shape[2] != 3:
        raise ValueError("Paired predictions/truth must be equal [2,T,3,H,W] tensors")
    if not torch.isfinite(prediction).all() or not torch.isfinite(truth).all():
        raise ValueError("Scoring requires finite RGB")
    mask = (truth[0] != truth[1]).any(dim=1)
    expanded = mask.unsqueeze(1).expand_as(truth[0])
    result = {"return_frames": truth.shape[1], "different_pixel_count": int(mask.sum()),
              "return_full_frame_mae": float((prediction - truth).abs().mean() / 2),
              "eligible_pair": bool(mask.any()), "pair_correct": False,
              "return_region_mae": None, "branches": [],
              "mask_rule": "Any true RGB channel differs; same paired mask for both branches; pooled over all return frames",
              "tie_rule": "Both own errors must be strictly lower than opposite errors; ties fail"}
    if mask.any():
        for branch in range(2):
            own = float((prediction[branch] - truth[branch]).abs()[expanded].mean() / 2)
            opposite = float((prediction[branch] - truth[1 - branch]).abs()[expanded].mean() / 2)
            result["branches"].append({"branch": branch, "own_truth_mae": own,
                                       "opposite_truth_mae": opposite, "own_strictly_closer": own < opposite})
        result["return_region_mae"] = sum(row["own_truth_mae"] for row in result["branches"]) / 2
        result["pair_correct"] = all(row["own_strictly_closer"] for row in result["branches"])
    return result


@torch.inference_mode()
def observed_one_step_mae(model, observations, actions, *, mode="carry"):
    """Separate control-set metric using observed histories at every step."""
    report, _ = observed_one_step_measurements(model, observations, actions, mode=mode)
    return report["mae"]


def summarize_step_errors(errors, actions):
    """Retain each transition; action frequencies never change a scene's weight."""
    errors, actions = errors.detach().cpu().double(), actions.detach().cpu()
    if errors.ndim != 2 or errors.shape != actions.shape or not errors.numel() or actions.dtype != torch.int64:
        raise ValueError("Expected matching nonempty [B,T] errors and int64 actions")
    if not torch.isfinite(errors).all() or (errors < 0).any() or (actions < 0).any() or (actions > 5).any():
        raise ValueError("Invalid step errors or action IDs")
    per_step = [{"branch": branch, "transition": step, "action": int(actions[branch, step]),
                 "mae": float(errors[branch, step])}
                for branch in range(errors.shape[0]) for step in range(errors.shape[1])]
    return {"mae": float(errors.mean()), "branches": errors.shape[0], "steps": errors.shape[1],
            "per_step": per_step, "per_action": {str(action): {
                "count": int((actions == action).sum()),
                "mae": float(errors[actions == action].mean()) if (actions == action).any() else None}
                for action in range(6)}}


@torch.inference_mode()
def observed_one_step_measurements(model, observations, actions, *, mode="carry"):
    """Real histories and current actions only; next RGB is used after prediction."""
    _check_inputs(observations, actions)
    if observations.shape[1] != actions.shape[1] + 1 or actions.shape[1] < 1:
        raise ValueError("One observation is required before each action and after the final action")
    state = model.initial_state(observations.shape[0])
    model.eval()
    errors, frames = [], []
    for step in range(actions.shape[1]):
        predicted, state = model(history_at(observations, step), actions[:, step], state, mode=mode)
        if predicted.shape != observations[:, step + 1].shape or not torch.isfinite(predicted).all():
            raise FloatingPointError("Invalid observed-history prediction")
        errors.append((predicted - observations[:, step + 1]).abs().mean((1, 2, 3)) / 2)
        frames.append(predicted.cpu())
    return summarize_step_errors(torch.stack(errors, dim=1), actions), torch.stack(frames, dim=1)


@torch.inference_mode()
def evaluate_pair(model, observations, actions, *, mode="carry", first_return=41):
    """Return measurements and both complete prediction tensors for retention."""
    _check_inputs(observations, actions)
    if observations.shape[0] != 2 or observations.shape[1] != actions.shape[1] + 1:
        raise ValueError("Expected two branches with T+1 observations and T actions")
    if type(first_return) is not int or not 0 < first_return < actions.shape[1]:
        raise ValueError("first_return must identify an action inside the trajectory")
    # The generators receive only the explicitly sliced available prefix.
    full = generated_rollout(model, history_at(observations[:, :1], 0), actions, mode=mode)
    suffix = observed_prefix_return(model, observations[:, :first_return + 1],
                                    actions[:, :first_return], actions[:, first_return:], mode=mode)
    # Future truth is used only after generation has finished.
    truth_return = observations[:, first_return + 1:]
    report = {
        "mode": mode, "first_return_action_index": first_return,
        "uninterrupted_generated_history": {
            "generated_steps": actions.shape[1], "true_frames_after_start": 0,
            "full_trajectory_mae": float((full - observations[:, 1:]).abs().mean() / 2),
            **score_return(full[:, first_return:], truth_return)},
        "observed_prefix_then_generated_return": {
            "observed_prefix_actions": first_return, "observed_prefix_frames": first_return + 1,
            "generated_steps": actions.shape[1] - first_return, "true_frames_after_prefix": 0,
            **score_return(suffix, truth_return)},
    }
    return report, {"uninterrupted": full.cpu(), "observed_prefix_return": suffix.cpu()}


class FrozenPredictorReference(nn.Module):
    """The existing stateless predictor, exposed through the explicit-state API."""
    def __init__(self, base):
        super().__init__()
        self.base = base.eval().requires_grad_(False)

    def initial_state(self, batch_size):
        return next(self.base.parameters()).new_zeros((batch_size, 32))

    def forward(self, history, action, state, *, mode="carry"):
        return self.base(history, action), state


def aggregate_pairs(records):
    """Equal weight per whole scene, preserving separate evaluation protocols."""
    result = {}
    for protocol in ("uninterrupted_generated_history", "observed_prefix_then_generated_return"):
        eligible = [row[protocol] for row in records if row[protocol]["eligible_pair"]]
        result[protocol] = {"scenes": len(records), "eligible_scenes": len(eligible),
            "mean_scene_return_region_mae": sum(row["return_region_mae"] for row in eligible) / len(eligible) if eligible else None,
            "correct_pairs": sum(row["pair_correct"] for row in eligible),
            "correct_pair_fraction": sum(row["pair_correct"] for row in eligible) / len(eligible) if eligible else None}
    return result


def aggregate_control_cases(records, expected_scene_ids):
    from .memory_train import CONTROL_ACTIONS
    result = {"available": True, "types": {}, "overall_equal_type_mae": None,
              "weighting": "Mean of scene means within type, then mean of the three type means"}
    for kind in CONTROL_ACTIONS:
        rows = [row for row in records if row["kind"] == kind]
        scenes = [row["scene_seed"] for row in rows]
        complete = len(scenes) == len(expected_scene_ids) and set(scenes) == set(expected_scene_ids)
        result["available"] &= complete
        actions = {}
        for action in range(6):
            values = [row["per_action"][str(action)] for row in rows if row["per_action"][str(action)]["count"]]
            count = sum(value["count"] for value in values)
            actions[str(action)] = {"count": count, "scenes_with_action": len(values),
                "equal_scene_mae": sum(value["mae"] for value in values) / len(values) if values else None,
                "occurrence_weighted_mae": sum(value["mae"] * value["count"] for value in values) / count if count else None}
        result["types"][kind] = {"available": complete, "scenes": len(rows),
            "equal_scene_mae": sum(row["mae"] for row in rows) / len(rows) if complete else None,
            "per_action": actions}
    if result["available"]:
        result["overall_equal_type_mae"] = sum(row["equal_scene_mae"] for row in result["types"].values()) / len(CONTROL_ACTIONS)
    return result


def compare_evaluations(reports, *, matched_budgets=False, visibility_audit=None):
    """Fixed three-seed comparisons and shared whole-scene bootstrap intervals.

    Reports are measured evaluator outputs, never renderer metadata. Missing
    cases/references produce unavailable gates, never a passing default.
    """
    import hashlib
    import numpy as np
    from .memory_train import SEEDS, SCENES, CONTROL_ACTIONS
    scenes, seeds = SCENES["validation"], SEEDS
    protocols = ("uninterrupted_generated_history", "observed_prefix_then_generated_return")
    result = {"status": "unavailable", "numerical_results_available": False, "gates_available": False,
              "control_retention_passed": None, "control_retention": {}, "paired_memory": {},
              "bootstrap": {"status": "unavailable"}, "reasons": [],
              "limit": "Provisional development point-estimate gates; no protected-test or frontier-quality claim"}
    if not matched_budgets:
        result["reasons"].append("Complete matched training budgets are unavailable")
        return result
    keys = [f"{seed}-{mode}" for seed in seeds for mode in ("carry", "reset")] + ["frozen"]
    if any(key not in reports or reports[key].get("status") != "complete"
           or reports[key].get("controls_evaluated") is not True
           or reports[key].get("control_results", {}).get("status") != "complete" for key in keys):
        result["reasons"].append("All three completed seed pairs, frozen reference and control results are required")
        return result
    control_arrays, pair_arrays = {}, {}
    try:
        data_hash = reports["frozen"]["data"]["manifest_sha256"]
        controls_hash = reports["frozen"]["control_results"]["provenance"]["manifest_sha256"]
        if not isinstance(data_hash, str) or len(data_hash) != 64 or not isinstance(controls_hash, str) or len(controls_hash) != 64:
            raise ValueError("Measured data provenance is unavailable")
        for key in keys:
            measured = reports[key]
            if (measured["data"]["manifest_sha256"] != data_hash
                    or measured["control_results"]["provenance"]["manifest_sha256"] != controls_hash):
                raise ValueError("Measured data or control captures differ between models")
            identity = measured.get("provenance", {})
            if key == "frozen":
                if identity.get("mode") != "frozen":
                    raise ValueError("Frozen-reference identity differs")
            else:
                seed, mode = key.split("-")
                if identity.get("seed") != int(seed) or identity.get("mode") != mode:
                    raise ValueError("Measured model identity differs")
            controls = measured["control_results"]
            rows = controls["single_cases"]
            if len(rows) != len(scenes) * len(CONTROL_ACTIONS):
                raise ValueError("Missing or duplicate single control cases")
            values = []
            for kind in CONTROL_ACTIONS:
                selected = [row for row in rows if row["kind"] == kind]
                if len(selected) != len(scenes) or {row["scene_seed"] for row in selected} != set(scenes):
                    raise ValueError("Missing or duplicate scene/type scores")
                lookup = {row["scene_seed"]: row for row in selected}
                for row in selected:
                    steps = row["per_step"]
                    if (row["branches"] != 1 or row["steps"] != 16 or len(steps) != 16
                            or [(step["branch"], step["transition"], step["action"]) for step in steps]
                            != [(0, step, action) for step, action in enumerate(CONTROL_ACTIONS[kind])]
                            or not np.isclose(row["mae"], np.mean([step["mae"] for step in steps]), rtol=1e-7, atol=1e-12)):
                        raise ValueError("Control step count/action alignment or mean differs")
                values.append([lookup[scene]["mae"] for scene in scenes])
            control_arrays[key] = np.asarray(values, dtype=np.float64)
            for waiting in (8, 16, 32):
                rows = measured["scenes"] if waiting == 16 else controls["paired_waits"][str(waiting)]["cases"]
                if len(rows) != len(scenes) or {row["scene_seed"] for row in rows} != set(scenes):
                    raise ValueError("Missing or duplicate paired wait scenes")
                lookup = {row["scene_seed"]: row for row in rows}
                for protocol in protocols:
                    records = [lookup[scene][protocol] for scene in scenes]
                    if any(not row["eligible_pair"] for row in records):
                        raise ValueError("A paired memory case has no eligible return pixels")
                    pair_arrays[(key, waiting, protocol)] = (
                        np.asarray([row["return_region_mae"] for row in records], dtype=np.float64),
                        np.asarray([row["pair_correct"] for row in records], dtype=np.float64))
        if any(not np.isfinite(value).all() or (value < 0).any() for value in control_arrays.values()):
            raise ValueError("Control means are nonfinite or negative")
        if any(not np.isfinite(error).all() or (error < 0).any() for error, _ in pair_arrays.values()):
            raise ValueError("Paired errors are nonfinite or negative")
    except (KeyError, TypeError, ValueError) as error:
        result["reasons"].append(str(error))
        return result
    result["numerical_results_available"] = True
    capture_verified = (visibility_audit is not None and visibility_audit.get("status") == "verified"
                        and visibility_audit.get("capture_manifest_sha256") == controls_hash)
    result["gates_available"] = capture_verified
    result["status"] = "complete" if capture_verified else "unavailable"
    if not capture_verified:
        result["reasons"].append("The hash-bound visibility audit is unavailable")
    indices = np.random.default_rng(20260907).integers(0, len(scenes), size=(10000, len(scenes)), dtype=np.int64)

    def ci(values):
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values)
        return {"percentile_95": np.percentile(values, [2.5, 97.5]).tolist() if valid.all() else None,
                "defined_resamples": int(valid.sum()), "undefined_resamples": int((~valid).sum())}

    def ratio(numerator, denominator, gain=False):
        numerator, denominator = np.asarray(numerator), np.asarray(denominator)
        out = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=out, where=denominator != 0)
        return 1 - out if gain else out - 1

    def number(value):
        return float(value) if np.isfinite(value) else None

    result["bootstrap"] = {"status": "complete", "resamples": 10000, "seed": 20260907,
        "scene_ids": list(scenes), "shared_scene_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "method": "The same eight-scene resample is applied to every arm, seed, protocol, wait duration and control type",
        "interval_scope": "Scene variation conditional on three fixed trained seeds; undefined zero-denominator replicates make that interval unavailable"}
    frozen = control_arrays["frozen"]
    names = [*CONTROL_ACTIONS, "overall_equal_type"]
    controls_bootstrap = {name: {"carry": [], "reset": [], "increase_vs_reset": [], "increase_vs_frozen": []} for name in names}
    control_passes = []
    for seed in seeds:
        carry, reset = control_arrays[f"{seed}-carry"], control_arrays[f"{seed}-reset"]
        per_type = {}
        for index, name in enumerate(names):
            c, r, f = (array[index] if index < len(CONTROL_ACTIONS) else array.mean(0) for array in (carry, reset, frozen))
            cm, rm, fm = float(c.mean()), float(r.mean()), float(f.mean())
            cb, rb, fb = (value[indices].mean(1) for value in (c, r, f))
            inc_reset, inc_frozen = ratio(cb, rb), ratio(cb, fb)
            point_passed = cm <= 1.05 * rm and cm <= 1.05 * fm
            per_type[name] = {"carry_mae": cm, "reset_mae": rm, "frozen_mae": fm,
                "reset_limit_1_05": 1.05 * rm, "frozen_limit_1_05": 1.05 * fm,
                "exceeds_reset_by_more_than_5_percent": cm > 1.05 * rm,
                "exceeds_frozen_by_more_than_5_percent": cm > 1.05 * fm,
                "relative_increase_vs_reset": number(ratio(cm, rm)), "relative_increase_vs_frozen": number(ratio(cm, fm)),
                "point_comparison_passed": point_passed, "gate_passed": point_passed if capture_verified else None,
                "intervals": {"carry_mae": ci(cb), "reset_mae": ci(rb), "frozen_mae": ci(fb),
                              "relative_increase_vs_reset": ci(inc_reset), "relative_increase_vs_frozen": ci(inc_frozen)}}
            for key, value in (("carry", cb), ("reset", rb), ("increase_vs_reset", inc_reset), ("increase_vs_frozen", inc_frozen)):
                controls_bootstrap[name][key].append(value)
            control_passes.append(point_passed)
        result["control_retention"][str(seed)] = per_type
    result["control_retention_passed"] = all(control_passes) if capture_verified else None
    result["control_retention_across_seeds"] = {
        name: {"mean_seed_carry_mae": float(np.mean([result["control_retention"][str(seed)][name]["carry_mae"] for seed in seeds])),
               "mean_seed_reset_mae": float(np.mean([result["control_retention"][str(seed)][name]["reset_mae"] for seed in seeds])),
               "intervals": {key: ci(np.mean(value, axis=0)) for key, value in arrays.items()}}
        for name, arrays in controls_bootstrap.items()}
    for waiting in (8, 16, 32):
        result["paired_memory"][str(waiting)] = {}
        for protocol in protocols:
            seed_rows, gains_boot, correct_boot = {}, [], []
            for seed in seeds:
                c, correct = pair_arrays[(f"{seed}-carry", waiting, protocol)]
                r, reset_correct = pair_arrays[(f"{seed}-reset", waiting, protocol)]
                gain = ratio(c.mean(), r.mean(), gain=True)
                gb = ratio(c[indices].mean(1), r[indices].mean(1), gain=True)
                pb = correct[indices].mean(1)
                seed_rows[str(seed)] = {"carry_return_region_mae": float(c.mean()), "reset_return_region_mae": float(r.mean()),
                    "relative_gain": number(gain), "carry_pair_correct_fraction": float(correct.mean()),
                    "reset_pair_correct_fraction": float(reset_correct.mean()),
                    "intervals": {"relative_gain": ci(gb), "carry_pair_correct_fraction": ci(pb)}}
                gains_boot.append(gb)
                correct_boot.append(pb)
            gains = [row["relative_gain"] for row in seed_rows.values()]
            mean_gain = sum(gains) / len(seeds) if all(value is not None for value in gains) else None
            mean_correct = sum(row["carry_pair_correct_fraction"] for row in seed_rows.values()) / len(seeds)
            point_passed = mean_gain is not None and mean_gain >= .30 and mean_correct >= .80 and all(value > 0 for value in gains)
            result["paired_memory"][str(waiting)][protocol] = {
                "seeds": seed_rows, "mean_seed_relative_gain": mean_gain, "mean_seed_carry_pair_correct_fraction": mean_correct,
                "provisional_point_comparison_passed": point_passed,
                "provisional_gate_passed": point_passed if capture_verified else None,
                "thresholds": {"mean_relative_gain": .30, "mean_pair_correct": .80, "positive_gain_each_seed": True},
                "intervals": {"mean_seed_relative_gain": ci(np.mean(gains_boot, axis=0)),
                              "mean_seed_carry_pair_correct_fraction": ci(np.mean(correct_boot, axis=0))}}
    return result


def evaluate_model(model, dataset, output, *, mode="carry", provenance=None, controls=None):
    """Retain every generated tensor and per-scene score, including failures."""
    import time
    from .memory_train import new_directory, atomic_write, sha256, tensor_hashes, _sync, CONTROL_ACTIONS, PAIRED_WAIT_KINDS
    out = new_directory(output)
    device = next(model.parameters()).device
    report = {"schema": "worldline-room-memory-evaluation-v1", "status": "running",
              "mode": mode, "provenance": provenance or {}, "data": dataset.provenance(),
              "scenes": [], "scene_weighting": "Equal weight per scene; protocols reported separately",
              "controls_evaluated": False, "reserved_test_opened": False,
              "control_results": {"status": "pending" if controls is not None else "unavailable",
                  "provenance": controls.provenance() if controls is not None else None, "single_cases": [],
                  "paired_waits": {str(wait): {"cases": []} for wait in PAIRED_WAIT_KINDS.values()}},
              "limit": "Development validation only; model-quality comparisons require complete matched references and capture audit"}
    started = time.monotonic()
    try:
        atomic_write(out / "evaluation.json", report)
        for scene in dataset.scene_ids:
            observations, actions = dataset.load_pair(scene)
            observations, actions = observations.to(device), actions.to(device)
            _sync(device)
            tick = time.monotonic()
            scores, predictions = evaluate_pair(model, observations, actions, mode=mode)
            _sync(device)
            seconds = time.monotonic() - tick
            path = out / f"scene-{scene}.pt"
            atomic_write(path, predictions, tensor=True)
            report["scenes"].append(dict(scores, scene_seed=scene, inference_seconds=seconds,
                prediction_file=path.name, prediction_file_sha256=sha256(path), predictions=tensor_hashes(predictions)))
            atomic_write(out / "evaluation.json", report)
        report["aggregate"] = aggregate_pairs(report["scenes"])
        if controls is not None:
            for scene in controls.scene_ids:
                for kind in (*CONTROL_ACTIONS, *PAIRED_WAIT_KINDS):
                    observations, actions = controls.load_case(scene, kind)
                    observations, actions = observations.to(device), actions.to(device)
                    _sync(device)
                    tick = time.monotonic()
                    if kind in CONTROL_ACTIONS:
                        scores, frames = observed_one_step_measurements(model, observations, actions, mode=mode)
                        predictions = {"observed_next_rgb": frames}
                        target = report["control_results"]["single_cases"]
                    else:
                        scores, predictions = evaluate_pair(model, observations, actions, mode=mode,
                            first_return=controls.return_indices[(scene, kind)])
                        target = report["control_results"]["paired_waits"][str(PAIRED_WAIT_KINDS[kind])]["cases"]
                    _sync(device)
                    seconds = time.monotonic() - tick
                    path = out / f"control-{scene}-{kind}.pt"
                    atomic_write(path, predictions, tensor=True)
                    target.append(dict(scores, scene_seed=scene, kind=kind, inference_seconds=seconds,
                        prediction_file=path.name, prediction_file_sha256=sha256(path), predictions=tensor_hashes(predictions)))
                    atomic_write(out / "evaluation.json", report)
            control_result = report["control_results"]
            control_result["single_summary"] = aggregate_control_cases(control_result["single_cases"], controls.scene_ids)
            for value in control_result["paired_waits"].values():
                value["aggregate"] = aggregate_pairs(value["cases"])
            control_result["status"] = "complete"
            report["controls_evaluated"] = True
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_write(out / "evaluation.json", report)
    return report


def _evaluation_worker(config):
    from pathlib import Path
    from .memory_train import (DevelopmentDataset, ValidationControls, make_model, restore_memory, sha256, BASE_SHA256)
    torch.set_num_threads(2)
    for name, expected in config["source_sha256"].items():
        if Path(name).name != name or sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError("Evaluation source changed after its snapshot")
    if config["device"] == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        torch.mps.set_per_process_memory_fraction(min(1., 18 * 2**30 / torch.mps.recommended_max_memory()))
    model = make_model(config["base"], config["seed"], config["device"])
    if config["mode"] == "frozen":
        model = FrozenPredictorReference(model.base)
    else:
        if sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
            raise ValueError("Final checkpoint hash differs")
        payload = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
        if (payload["mode"] != config["mode"] or payload["seed"] != config["seed"]
                or payload["base_checkpoint_sha256"] != BASE_SHA256
                or payload["completed_updates"] != config["updates"]):
            raise ValueError("Checkpoint identity or completed budget differs")
        restore_memory(model, payload["memory_state_dict"])
    dataset = DevelopmentDataset(config["data"], "validation")
    if dataset.manifest_sha256 != config["data_sha256"]:
        raise ValueError("Validation manifest changed")
    controls = ValidationControls(config["controls"]) if config.get("controls") else None
    if controls is not None and controls.manifest_sha256 != config["controls_sha256"]:
        raise ValueError("Control manifest changed")
    evaluate_model(model, dataset, config["output"], mode="carry" if config["mode"] == "frozen" else config["mode"],
                   provenance={key: config[key] for key in ("mode", "seed", "updates", "checkpoint_sha256")}, controls=controls)


def evaluate_study(study, data, output, *, base=None, device="cpu", max_seconds=600, controls=None, control_audit=None):
    """Evaluate completed fixed training budgets; profile runs are ineligible."""
    import json
    import math
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    from .memory_train import (DevelopmentDataset, ValidationControls, verify_visibility_audit, BASE, BASE_SHA256,
                               new_directory, atomic_write, sha256, handoff_and_guard, matched_arms)
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 600 or device not in ("cpu", "mps"):
        raise ValueError("Use CPU/MPS and a positive time cap at most600 seconds")
    root, base = Path(study).resolve(), Path(base or BASE).resolve()
    source_report = json.loads((root / "study.json").read_text())
    if (source_report["status"] != "complete" or source_report["phase"] != "train"
            or source_report["matched_budgets"] is not True or sha256(base) != BASE_SHA256):
        raise ValueError("Only completed matched training arms with the pinned base are eligible")
    for name in ("memory_model.py", "model.py"):
        if sha256(Path(__file__).with_name(name)) != source_report.get("source_sha256", {}).get(name):
            raise ValueError("Model forward source differs from the completed training study")
    dataset = DevelopmentDataset(data, "validation")
    if dataset.manifest_sha256 != source_report["validation"]["manifest_sha256"]:
        raise ValueError("Validation capture differs from the declared study")
    control_data, control_issue, visibility = None, "No control capture was supplied", None
    if controls is not None:
        try:
            control_data = ValidationControls(controls)
            control_issue = None
        except (OSError, ValueError, KeyError, TypeError) as error:
            control_issue = f"Control capture unavailable: {type(error).__name__}: {error}"
    if control_data is not None and control_audit is not None:
        try:
            visibility = verify_visibility_audit(control_audit, control_data)
        except (OSError, ValueError, KeyError, TypeError) as error:
            control_issue = f"Visibility audit unavailable: {type(error).__name__}: {error}"
    jobs = []
    for seed in source_report["seeds"]:
        pair = []
        for mode in ("carry", "reset"):
            entries = [r for r in source_report["runs"] if r["seed"] == seed and r["mode"] == mode]
            if len(entries) != 1:
                raise ValueError("Missing or duplicate completed arm")
            row = entries[0]
            path = (root / row["metrics"]).resolve()
            if not path.is_relative_to(root) or sha256(path) != row["metrics_sha256"]:
                raise ValueError("Training metrics path or hash differs")
            arm = json.loads(path.read_text())
            pair.append(arm)
            checkpoint = path.parent / "memory-final.pt"
            if sha256(checkpoint) != arm["memory_final_sha256"]:
                raise ValueError("Final memory checkpoint changed")
            jobs.append({"seed": seed, "mode": mode, "updates": arm["completed_updates"],
                         "checkpoint": str(checkpoint), "checkpoint_sha256": arm["memory_final_sha256"]})
        if not matched_arms(*pair):
            raise ValueError("Realized arm budgets differ")
    jobs.append({"seed": source_report["seeds"][0], "mode": "frozen", "updates": 0,
                 "checkpoint": None, "checkpoint_sha256": BASE_SHA256})
    out = new_directory(output)
    snapshot = out / "measured-source"
    snapshot.mkdir()
    sources = {}
    for name in ("memory_evaluate.py", "memory_train.py", "memory_model.py", "model.py"):
        path = Path(__file__).with_name(name)
        shutil.copyfile(path, snapshot / (name + ".txt"))
        sources[name] = sha256(path)
    report = {"schema": "worldline-room-memory-validation-study-v1", "status": "running",
              "training_study_sha256": sha256(root / "study.json"), "source_sha256": sources,
              "base_checkpoint_sha256": BASE_SHA256, "validation": dataset.provenance(), "runs": [],
              "reserved_test_opened": False, "controls_evaluated": False,
              "controls": control_data.provenance() if control_data is not None else None,
              "control_input_issue": control_issue, "visibility_audit": visibility,
              "comparison": {"status": "unavailable", "reason": "Evaluation has not completed"},
              "checkpoint_selection": "Fixed completed training budget only; no validation selection",
              "limit": "Development evidence only; gates require complete matched three-seed results, all controls and the hash-bound visibility audit"}
    atomic_write(out / "validation.json", report)
    model_reports = {}
    try:
        for job in jobs:
            folder = out / ("frozen" if job["mode"] == "frozen" else f"{job['seed']}-{job['mode']}")
            folder.mkdir()
            config = dict(job, base=str(base), data=str(Path(data).resolve()), data_sha256=dataset.manifest_sha256,
                          output=str((folder / "predictions").resolve()), device=device, source_sha256=sources,
                          controls=str(control_data.root) if control_data is not None else None,
                          controls_sha256=control_data.manifest_sha256 if control_data is not None else None)
            with (folder / "worker.log").open("x") as log:
                process = subprocess.Popen([sys.executable, "-m", "experiments.room_world.memory_evaluate", "--worker"],
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0"))
                terminal = handoff_and_guard(process, config, folder, max_seconds=max_seconds)
            path = folder / "predictions/evaluation.json"
            measured = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
            report["runs"].append({"mode": job["mode"], "seed": job["seed"], "terminal_status": terminal["status"],
                "status": measured["status"], "evaluation": str(path.relative_to(out)),
                "evaluation_sha256": sha256(path) if path.exists() else None})
            atomic_write(out / "validation.json", report)
            if terminal["status"] != "complete" or measured["status"] != "complete":
                raise RuntimeError("Evaluation stopped or failed; partial outputs retained")
            model_reports[folder.name] = measured
        report["controls_evaluated"] = all(value.get("controls_evaluated") is True for value in model_reports.values())
        report["comparison"] = compare_evaluations(model_reports, matched_budgets=True, visibility_audit=visibility)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_write(out / "validation.json", report)
    return report


def main():
    import argparse
    import json
    from pathlib import Path
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--controls", type=Path, help="Separate captured validation controls; absent/incomplete controls leave gates unavailable")
    parser.add_argument("--control-audit", type=Path, help="Independent visibility audit bound to the control capture")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.worker:
        _evaluation_worker(json.loads(sys.stdin.read(65536)))
        return
    if any(value is None for value in (args.study, args.data, args.output)):
        parser.error("--study, --data and --output are required")
    result = evaluate_study(args.study, args.data, args.output, base=args.base, device=args.device, max_seconds=args.max_seconds,
                            controls=args.controls, control_audit=args.control_audit)
    print(json.dumps({key: result[key] for key in ("status", "reserved_test_opened", "controls_evaluated")}))


if __name__ == "__main__":
    main()
