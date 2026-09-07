"""Reproducible software measurements plus an actual-model control ablation.

No model is substituted when trained weights are unavailable. No comparison
against Genie 3, renderer graphics quality or real-world physics is performed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worldline.state import SCHEMA, WorldStore, digest, edit_field


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def elapsed_ms(operation):
    started = time.perf_counter()
    result = operation()
    return result, (time.perf_counter() - started) * 1000


def fixed_state():
    rng = np.random.default_rng(913700)
    return {"schema": SCHEMA, "seed": 913700, "biome": 0, "prompt": "Storage benchmark fixture",
            "height": rng.uniform(-.4, .4, (64, 64)).tolist(),
            "ecology": rng.uniform(.2, .6, (3, 64, 64)).tolist(),
            "tick": 0, "weather": {"rain": 0, "heat": 0}}


def benchmark_snapshots(directory, revisions=24):
    state = fixed_state()
    path = Path(directory) / "snapshots.sqlite"
    store = WorldStore(path)
    creation, creation_ms = elapsed_ms(lambda: store.create(state, "Measured fixture"))
    wid = creation["id"]
    timings = []
    hashes = [creation["hash"]]
    locality_violations = 0
    original_untouched = True
    for index in range(revisions):
        previous = store.get(wid)
        before = previous["state"]
        baseline = copy.deepcopy(before)
        x, z, radius = .2 + (index % 4) * .16, .3 + (index % 3) * .15, .08
        changed = edit_field(before, "raise", x, z, radius, .025)
        yy, xx = np.indices((64, 64), dtype=np.float64)
        outside = np.hypot(xx / 63 - x, yy / 63 - z) > radius + 1e-6
        locality_violations += int(np.count_nonzero(np.asarray(before["height"])[outside] != np.asarray(changed["height"])[outside]))
        original_untouched &= before == baseline
        saved, ms = elapsed_ms(lambda: store.save(wid, changed, {"type": "edit", "benchmark_index": index}, previous["revision"]))
        timings.append(ms)
        hashes.append(saved["hash"])
    prior_history = store.history(wid)
    restored, restore_ms = elapsed_ms(lambda: store.restore(wid, 0, revisions))
    restored_exact = restored["hash"] == creation["hash"] and restored["state"] == creation["state"]
    append_only = store.history(wid)[1:] == prior_history
    branch, branch_ms = elapsed_ms(lambda: store.branch(wid, "Independent branch"))
    child_state = edit_field(branch["state"], "plant", .5, .5, .1, .2)
    store.save(branch["id"], child_state, {"type": "edit"}, 0)
    parent_unchanged = store.get(wid) == restored
    expected_latest = store.get(wid)
    store.close()
    start = time.perf_counter()
    reopened = WorldStore(path)
    reload_exact = reopened.get(wid) == expected_latest
    all_revisions_exact = all(reopened.get(wid, index)["hash"] == value for index, value in enumerate(hashes))
    reload_ms = (time.perf_counter() - start) * 1000
    exported = json.loads(json.dumps({"name": expected_latest["name"], "state": expected_latest["state"]}))
    imported = reopened.create(exported["state"], exported["name"], event={"type": "import"})
    roundtrip_exact = imported["hash"] == expected_latest["hash"]
    reopened.close()
    return {
        "scope": "Explicit storage and compact-edit software, using a deterministic numeric fixture, no neural model",
        "revisions": revisions,
        "create_ms": creation_ms,
        "save_ms_median": statistics.median(timings),
        "save_ms_p95": float(np.percentile(timings, 95)),
        "restore_ms": restore_ms,
        "branch_ms": branch_ms,
        "close_reopen_and_all_history_read_ms": reload_ms,
        "database_bytes_after_close": path.stat().st_size,
        "outside_brush_changed_values": locality_violations,
        "caller_input_unchanged": bool(original_untouched),
        "restored_state_and_hash_exact": restored_exact,
        "history_append_only_after_restore": append_only,
        "branch_parent_unchanged": parent_unchanged,
        "reload_exact": reload_exact,
        "all_historical_hashes_retained": all_revisions_exact,
        "json_import_export_hash_exact": roundtrip_exact,
        "source_state_sha256": digest(state),
    }


def benchmark_models(checkpoint_dir, device=None):
    paths = [Path(checkpoint_dir) / name for name in ("spatial-flow.pt", "ecology.pt")]
    if not all(path.is_file() for path in paths):
        return {"status": "not_run", "reason": "Missing real checkpoints", "missing": [path.name for path in paths if not path.is_file()]}
    import torch
    from worldline.models import get_models
    from worldline.data import ecology_states, ecology_teacher

    torch.set_num_threads(min(4, torch.get_num_threads()))
    (generator, dynamics), load_ms = elapsed_ms(lambda: get_models(checkpoint_dir, device=device))
    samples, generation_ms = [], []
    for biome in range(3):
        fields, elapsed = elapsed_ms(lambda: generator.sample(913710, biome, 24))
        samples.append(fields)
        generation_ms.append(elapsed)
    repeated = generator.sample(913710, 0, 24)
    other_seed = generator.sample(913711, 0, 24)
    seed_mae = float(np.mean(np.abs(samples[0] - other_seed)))
    biome_mae = [float(np.mean(np.abs(samples[0] - samples[index]))) for index in (1, 2)]

    # Paired input ablation: same learned network, initial state and teacher
    # trajectory. Only rain/heat inputs are set to zero in the ablated arm.
    # This is not an alternative trained baseline and not a physical causality proof.
    cases = []
    total_transition_ms = []
    for seed in (913720, 913721, 913722):
        initial = ecology_states(1, size=64, seed=seed)[0].numpy()
        for rain, heat in ((1., 0.), (0., 1.), (.8, .7), (.25, .4)):
            predicted = initial.copy()
            ablated = initial.copy()
            teacher = torch.from_numpy(initial.copy()).unsqueeze(0)
            errors = {}
            for step in range(1, 61):
                predicted, elapsed = elapsed_ms(lambda: dynamics.step(predicted, rain, heat))
                total_transition_ms.append(elapsed)
                ablated = dynamics.step(ablated, 0., 0.)
                with torch.inference_mode():
                    teacher = ecology_teacher(teacher, torch.tensor([rain]), torch.tensor([heat]), torch.tensor([1.]))
                if step in (1, 10, 60):
                    truth = teacher[0].numpy()
                    errors[str(step)] = {
                        "conditioned_mse": float(np.mean((predicted - truth)**2)),
                        "controls_zeroed_mse": float(np.mean((ablated - truth)**2)),
                        "conditioned_vs_zeroed_output_mae": float(np.mean(np.abs(predicted - ablated))),
                    }
            cases.append({"seed": seed, "rain": rain, "heat": heat, "horizons": errors})
    aggregate = {}
    for horizon in ("1", "10", "60"):
        mse = np.asarray([row["horizons"][horizon]["conditioned_mse"] for row in cases])
        zero_mse = np.asarray([row["horizons"][horizon]["controls_zeroed_mse"] for row in cases])
        aggregate[horizon] = {
            "conditioned_mean_mse": float(mse.mean()),
            "controls_zeroed_mean_mse": float(zero_mse.mean()),
            "paired_cases_conditioned_better": int(np.sum(mse < zero_mse)),
            "case_count": len(cases),
        }
    return {
        "status": "measured",
        "scope": "Actual trained 64x64 field models; explicit synthetic ecology teacher; no rendered-image assessment",
        "device": str(generator.device), "torch_version": torch.__version__,
        "checkpoint_sha256": {path.name: file_hash(path) for path in paths},
        "model_load_ms": load_ms,
        "generation_seed": 913710, "flow_steps": 24,
        "generation_ms_by_biome": dict(zip(("alpine", "desert", "alien"), generation_ms)),
        "same_seed_exact_on_this_device": bool(np.array_equal(samples[0], repeated)),
        "different_seed_mean_absolute_difference": seed_mae,
        "same_seed_other_biomes_mean_absolute_difference": biome_mae,
        "finite_and_bounded_fields": bool(all(np.isfinite(x).all() and x.min() >= -1 and x.max() <= 1 for x in samples)),
        "learned_transition_ms_median": statistics.median(total_transition_ms),
        "learned_transition_ms_p95": float(np.percentile(total_transition_ms, 95)),
        "control_ablation_definition": "Same checkpoint and initial state, actual control in full arm, both controls zeroed every step in ablated arm; both compared against actual-control synthetic teacher",
        "control_ablation_aggregate": aggregate,
        "control_ablation_cases": cases,
        "limitations": [
            "Tests action-input use within one synthetic transition law, not real-world physics or unseen dynamics laws",
            "Cases use new deterministic seeds, with the same synthetic state family as training",
            "Generation timing includes inference and synchronization through NumPy output but excludes rendering and HTTP",
            "Single-process local measurements are not server concurrency or browser frame-rate measurements",
        ],
    }


def run(checkpoint_dir, device=None, revisions=24):
    with tempfile.TemporaryDirectory(prefix="worldline-eval-") as directory:
        software = benchmark_snapshots(directory, revisions)
    return {
        "schema": "worldline.evaluation.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "scope": "Software properties and limited synthetic-model ablation",
        "snapshot_metrics": software,
        "model_metrics": benchmark_models(checkpoint_dir, device),
        "not_tested": ["Genie 3 parity", "photorealism", "graphics preference", "scientific novelty", "general physical reasoning", "open-domain language understanding", "minute-long autoregressive video", "browser FPS", "multi-user concurrency performance"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "benchmark-metrics.json")
    parser.add_argument("--revisions", type=int, default=24)
    parser.add_argument("--require-models", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.revisions <= 1000:
        parser.error("--revisions must be between 1 and 1000")
    report = run(args.checkpoints, args.device, args.revisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output), "models": report["model_metrics"]["status"], "outside_brush_changed_values": report["snapshot_metrics"]["outside_brush_changed_values"]}))
    if args.require_models and report["model_metrics"]["status"] != "measured":
        return 2
    invariants = ("caller_input_unchanged", "restored_state_and_hash_exact", "history_append_only_after_restore", "branch_parent_unchanged", "reload_exact", "all_historical_hashes_retained", "json_import_export_hash_exact")
    return 0 if all(report["snapshot_metrics"][name] for name in invariants) and report["snapshot_metrics"]["outside_brush_changed_values"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
