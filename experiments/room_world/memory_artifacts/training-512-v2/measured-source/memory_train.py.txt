# SPDX-License-Identifier: Apache-2.0
"""Fixed-budget paired Room memory training; CLI defaults to plan-only.

One update is both branches of one scene and all65 teacher-observed transitions.
No validation checkpoint selection, truncated BPTT or door-weighted loss is used.
Each real CLI arm runs in its own supervised process. This module never renders
data and accepts only the declared training/validation development captures.
"""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil
import torch

from .memory_evaluate import history_at
from .memory_model import RoomMemoryModel
from .model import load_model

SEEDS = (20260907, 20260908, 20260909)
SCENES = {"train": tuple(range(5000, 5032)), "validation": tuple(range(300000, 300008))}
BASE = Path(__file__).parent / "artifacts/predictor/model.pt"
BASE_SHA256 = "9376e684e3abd103313d13d04bd4a1afc1b202fccb7e7b6fe133e7232a486d69"
CONTROL_ACTIONS = {
    "translation_cycle": (1,)*4 + (2,)*8 + (1,)*4,
    "out_of_reach_interaction": (2,)*6 + (5,) + (0,)*3 + (1,)*6,
    "turn_open_close": (3,)*4 + (4,)*4 + (5,) + (0,)*3 + (5,) + (0,)*3,
}
PAIRED_WAIT_KINDS = {"paired_wait_8": 8, "paired_wait_32": 32}
SEQUENCE_PATHS = ("stepwise", "batched")


def training_source_names(sequence_path="stepwise"):
    if sequence_path not in SEQUENCE_PATHS:
        raise ValueError("Sequence path must be stepwise or batched")
    sources = ("memory_train.py", "memory_evaluate.py", "memory_model.py", "memory_data.py", "model.py")
    return sources + (("memory_sequence.py",) if sequence_path == "batched" else ())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hashes(values):
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for key, value in sorted(values.items())}


def atomic_write(path, value, *, tensor=False):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".memory-", dir=path.parent)
    os.close(fd)
    try:
        if tensor:
            with open(temporary, "wb") as handle:
                torch.save(value, handle)
        else:
            Path(temporary).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def new_directory(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Output directory must be new")
    path.mkdir(parents=True)
    return path


class DevelopmentDataset:
    """Strict reader that cannot open the reserved test split or scene IDs."""
    def __init__(self, directory, split):
        if split not in SCENES:
            raise ValueError("Only train and validation development data are allowed")
        self.root = Path(directory).resolve()
        path = self.root / "manifest.json"
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Manifest must stay inside the capture directory")
        self.manifest = json.loads(path.read_text())
        self.manifest_sha256 = sha256(path)
        m = self.manifest
        if (m.get("schema") != "worldline-room-memory-development-v1" or m.get("status") != "complete"
                or m.get("split") != split or m.get("reserved_test_generated") is not False
                or m.get("scene_seeds") != list(SCENES[split]) or m.get("steps") != 65 or m.get("size") != 64):
            raise ValueError("Expected the completed declared64px/65-step development split")
        records = m.get("records", [])
        if len(records) != len(SCENES[split]) or [r.get("scene_seed") for r in records] != list(SCENES[split]):
            raise ValueError("Missing, duplicate or undeclared development scenes")
        self.scene_ids = SCENES[split]
        self.records = {r["scene_seed"]: r for r in records}
        self.split = split
        for record in records:
            if (record.get("steps") != 65 or record.get("wait_steps") != 16
                    or record.get("successful_initial_interaction") is not True
                    or record.get("first_return_action_index") != 41
                    or record.get("branch_order") != ["wait", "interact"]):
                raise ValueError("Standard paired training/validation protocol differs")
            self._path(record)

    def _path(self, record):
        relative = Path(record["file"])
        path = (self.root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Pair file must remain inside the capture")
        if path.stat().st_size != record["file_bytes"] or not 0 < path.stat().st_size < 64 * 2**20:
            raise ValueError("Pair file size differs or exceeds the bounded reader")
        return path

    def load_pair(self, scene):
        if scene not in self.records:
            raise ValueError("Scene is outside this declared development split")
        record = self.records[scene]
        path = self._path(record)
        if sha256(path) != record["file_sha256"]:
            raise ValueError("Paired data file hash differs")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"observations", "actions"}:
                raise ValueError("Only RGB observations and action arrays are allowed")
            rgb, actions = archive["observations"], archive["actions"]
        if rgb.dtype != np.uint8 or rgb.shape != (2, 66, 3, 64, 64) or actions.dtype != np.int64 or actions.shape != (2, 65):
            raise ValueError("Pair array shape/dtype differs from the fixed study")
        for key, array in (("observations", rgb), ("actions", actions)):
            if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != record[key + "_sha256"]:
                raise ValueError("Paired array hash differs")
        expected = np.asarray([[0] + [3]*24 + [0]*16 + [4]*24, [5] + [3]*24 + [0]*16 + [4]*24])
        if not np.array_equal(actions, expected) or not np.array_equal(rgb[0, 0], rgb[1, 0]) or not np.array_equal(rgb[0, 38:42], rgb[1, 38:42]):
            raise ValueError("Paired action order, common initial RGB or return alias differs")
        # Metadata, including teacher_door_states, never leaves this reader.
        return torch.from_numpy(rgb.copy()).float() / 127.5 - 1, torch.from_numpy(actions.copy())

    def provenance(self):
        return {"split": self.split, "manifest_sha256": self.manifest_sha256,
                "scene_ids": list(self.scene_ids), "steps": 65, "size": 64,
                "files": [{key: row[key] for key in ("scene_seed", "file", "file_sha256", "observations_sha256", "actions_sha256")}
                          for row in self.records.values()],
                "controls_present": "Successful interactions with fixed16-step wait only; other controls are separate"}


class ValidationControls:
    """Hash-checked development controls; only RGB/actions leave load_case.

    This reader does not import the capture module, simulator or teacher state.
    Exact fixed scripts are duplicated here so a changed manifest cannot silently
    redefine the predeclared controls. Return indices come from validated records.
    """
    _path = DevelopmentDataset._path

    def __init__(self, directory):
        self.root = Path(directory).resolve()
        path = self.root / "manifest.json"
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Control manifest must stay inside its capture")
        self.manifest = json.loads(path.read_text())
        self.manifest_sha256 = sha256(path)
        m = self.manifest
        if (m.get("schema") != "worldline-room-memory-controls-v1" or m.get("status") != "complete"
                or m.get("split") != "validation" or m.get("reserved_test_generated") is not False
                or m.get("scene_seeds") != list(SCENES["validation"]) or m.get("size") != 64
                or m.get("paired_waits") != [8, 32] or m.get("model_input_fields") != ["observations", "actions"]
                or m.get("control_actions") != {key: list(value) for key, value in CONTROL_ACTIONS.items()}):
            raise ValueError("Expected the complete declared validation control capture")
        self.scene_ids = SCENES["validation"]
        expected = {(scene, kind) for scene in self.scene_ids for kind in (*CONTROL_ACTIONS, *PAIRED_WAIT_KINDS)}
        rows = m.get("records", [])
        keys = [(row.get("scene_seed"), row.get("kind")) for row in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Control cases are missing, duplicated or outside the development scenes")
        self.records = dict(zip(keys, rows))
        self.return_indices = {}
        for (scene, kind), row in self.records.items():
            if row.get("size") != 64 or row.get("episode_seed") != scene * 1009 + 37:
                raise ValueError("Control size or episode differs from the fixed capture")
            if kind in CONTROL_ACTIONS:
                expected_toggles = {"translation_cycle": [], "out_of_reach_interaction": [False], "turn_open_close": [True, True]}[kind]
                expected_moves = sum(a in (1, 2) for a in CONTROL_ACTIONS[kind])
                if (row.get("steps") != 16 or row.get("interaction_changed_door") != expected_toggles
                        or row.get("translations_verified") != expected_moves):
                    raise ValueError("Declared single-control capture assertions differ")
            else:
                wait = PAIRED_WAIT_KINDS[kind]
                first = row.get("first_return_action_index")
                if (row.get("wait_steps") != wait or row.get("steps") != 49 + wait or first != 25 + wait
                        or row.get("steps") - first != 24 or row.get("successful_initial_interaction") is not True
                        or row.get("branch_order") != ["wait", "interact"]):
                    raise ValueError("Paired wait length or recorded return action index differs")
                self.return_indices[(scene, kind)] = first
            self._path(row)

    def load_case(self, scene, kind):
        if (scene, kind) not in self.records:
            raise ValueError("Control case is outside the declared validation capture")
        row = self.records[(scene, kind)]
        path = self._path(row)
        if sha256(path) != row["file_sha256"]:
            raise ValueError("Control file hash differs")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"observations", "actions"}:
                raise ValueError("Control arrays must contain RGB and actions only")
            rgb, actions = archive["observations"], archive["actions"]
        batch, steps = (1, 16) if kind in CONTROL_ACTIONS else (2, row["steps"])
        if rgb.shape != (batch, steps + 1, 3, 64, 64) or rgb.dtype != np.uint8 or actions.shape != (batch, steps) or actions.dtype != np.int64:
            raise ValueError("Control array shape or dtype differs")
        for key, value in (("observations", rgb), ("actions", actions)):
            if hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != row[key + "_sha256"]:
                raise ValueError("Control raw-array hash differs")
        if kind in CONTROL_ACTIONS:
            expected = np.asarray([CONTROL_ACTIONS[kind]], dtype=np.int64)
        else:
            wait, first = PAIRED_WAIT_KINDS[kind], self.return_indices[(scene, kind)]
            expected = np.asarray([[0] + [3]*24 + [0]*wait + [4]*24, [5] + [3]*24 + [0]*wait + [4]*24])
            if (not np.array_equal(rgb[0, 0], rgb[1, 0])
                    or not np.array_equal(rgb[0, first-3:first+1], rgb[1, first-3:first+1])
                    or np.array_equal(rgb[0, -1], rgb[1, -1])):
                raise ValueError("Paired control initial RGB, return alias or final difference fails")
        if not np.array_equal(actions, expected):
            raise ValueError("Control action sequence differs from the declared case")
        return torch.from_numpy(rgb.copy()).float() / 127.5 - 1, torch.from_numpy(actions.copy())

    def provenance(self):
        return {"manifest_sha256": self.manifest_sha256, "split": "validation", "scene_ids": list(self.scene_ids),
                "files": [{key: row[key] for key in ("scene_seed", "kind", "file", "file_sha256", "observations_sha256", "actions_sha256")}
                          for row in self.records.values()], "source_sha256": self.manifest.get("source_sha256", {}),
                "input_fields": ["observations", "actions"],
                "capture_limit": "Manifest records movement and toggle assertions; it does not record an independent visible-door or distance assertion"}


def verify_visibility_audit(path, controls):
    """Bind the independently recorded visibility evidence to these exact files."""
    path = Path(path)
    audit = json.loads(path.read_text())
    if (audit.get("schema") != "worldline-room-control-visibility-audit-v1" or audit.get("status") != "passed"
            or audit.get("scene_seeds") != list(controls.scene_ids) or audit.get("reserved_test_opened") is not False
            or audit.get("capture_manifest_sha256") != controls.manifest_sha256):
        raise ValueError("Visibility audit identity, completeness or capture hash differs")
    for name in ("simulator.py", "memory_controls.py", "memory_data.py"):
        if audit.get("source_sha256", {}).get(name) != controls.manifest["source_sha256"][name]:
            raise ValueError("Visibility audit capture-source hash differs")
    script = Path(__file__).parent / "memory_artifacts/control-validation/audit_visibility.py"
    if sha256(script) != audit["source_sha256"].get("audit_visibility.py"):
        raise ValueError("Visibility audit script differs from the retained source")
    cases = audit.get("cases", [])
    if len(cases) != len(controls.scene_ids) or {row.get("scene_seed") for row in cases} != set(controls.scene_ids):
        raise ValueError("Visibility audit scenes are missing or duplicated")
    for case in cases:
        scene = case["scene_seed"]
        row = controls.records[(scene, "out_of_reach_interaction")]
        controls.load_case(scene, "out_of_reach_interaction")
        with np.load(controls._path(row), allow_pickle=False) as archive:
            observed_hash = hashlib.sha256(archive["observations"][0, 6].tobytes()).hexdigest()
        distance, facing, visible = case.get("door_distance"), case.get("facing_cosine"), case.get("visible_door_pixel_count")
        if (case.get("file") != row["file"] or case.get("capture_file_sha256") != row["file_sha256"]
                or case.get("episode_seed") != row["episode_seed"] or case.get("action_index") != 6 or case.get("action_id") != 5
                or case.get("captured_observation6_sha256") != observed_hash or case.get("reconstructed_observation6_sha256") != observed_hash
                or case.get("captured_image_exactly_reproduced") is not True or case.get("teacher_state_unchanged") is not True
                or case.get("door_open_before") is not False or case.get("door_open_after") is not False
                or not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance <= 2.3
                or not isinstance(facing, (int, float)) or not math.isfinite(facing) or not .35 < facing <= 1
                or type(visible) is not int or not 0 < visible <= 4096 or case.get("image_pixel_count") != 4096):
            raise ValueError("Visibility audit case fails its hash, distance, visibility or failed-toggle assertions")
    return {"status": "verified", "file_sha256": sha256(path), "capture_manifest_sha256": controls.manifest_sha256,
            "source_sha256": audit["source_sha256"], "scene_seeds": list(controls.scene_ids),
            "visible_door_pixels": {str(row["scene_seed"]): row["visible_door_pixel_count"] for row in cases},
            "method": audit["method"]}


def paired_scene_schedule(scene_ids, seed, updates):
    if type(updates) is not int or updates < 1 or len(set(scene_ids)) != len(scene_ids) or not scene_ids:
        raise ValueError("Need distinct scenes and a positive integer update budget")
    rng = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < updates:
        schedule.extend(int(scene) for scene in rng.permutation(scene_ids))
    return schedule[:updates]


def sequence_loss(model, observations, actions, mode="carry", *, check=None):
    """Plain mean RGB[0,1] MAE, all65 observed-history steps, full BPTT."""
    if observations.ndim != 5 or observations.shape[1:3] != (66, 3) or actions.shape != (observations.shape[0], 65):
        raise ValueError("Training requires all65 transitions and66 RGB observations")
    state, errors = model.initial_state(observations.shape[0]), []
    for step in range(65):
        if check is not None:
            check()
        prediction, state = model(history_at(observations, step), actions[:, step], state, mode=mode)
        # t+1 is first used here, after the t-conditioned prediction.
        errors.append((prediction - observations[:, step + 1]).abs().mean() / 2)
    return torch.stack(errors).mean()


def memory_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_memory(model, values):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(values):
        raise ValueError("Memory checkpoint keys differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = values[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise ValueError("Memory checkpoint shape/dtype/finite validation failed")
            parameter.copy_(value.to(parameter.device))


def rng_state(device):
    state = np.random.get_state()
    result = {"torch_cpu": torch.get_rng_state(), "python": random.getstate(),
              "numpy": {"name": state[0], "keys": state[1].tolist(), "position": state[2],
                        "has_gauss": state[3], "cached_gaussian": state[4]}}
    if torch.device(device).type == "mps":
        result["torch_mps"] = torch.mps.get_rng_state().cpu()
    return result


def _sync(device):
    if torch.device(device).type == "mps":
        torch.mps.synchronize()


def run_arm(model, dataset, schedule, mode, output, *, seed, phase, learning_rate=3e-4,
            weight_decay=1e-4, check=None, base_checkpoint_sha256=BASE_SHA256, sequence_path="stepwise"):
    """Execute one already initialized arm; used by the supervised CLI worker.

    A failure preserves an explicit status and the current recoverable state.
    Checkpoints after each completed optimizer update also survive forced stops.
    CPU tests inject small synthetic models/data; the real CLI always constructs
    RoomMemoryModel with the pinned published predictor.
    """
    if mode not in ("carry", "reset") or phase not in ("profile", "train") or not schedule:
        raise ValueError("Expected an explicit carry/reset arm and nonempty budget")
    training_source_names(sequence_path)
    if sequence_path == "batched":
        from .memory_sequence import sequence_loss_batched
        loss_function = sequence_loss_batched
    else:
        loss_function = sequence_loss
    out = new_directory(output)
    device = next(model.parameters()).device
    initial = memory_state(model)
    base_before = tensor_hashes(model.base.state_dict())
    optimizer = torch.optim.AdamW(model.memory_parameters(), lr=learning_rate, weight_decay=weight_decay)
    report = {"schema": "worldline-room-memory-arm-v1", "status": "running", "phase": phase,
        "mode": mode, "seed": seed, "sequence_path": sequence_path,
        "loss_implementation": loss_function.__module__ + "." + loss_function.__name__,
        "requested_updates": len(schedule), "completed_updates": 0,
        "scene_schedule": list(schedule), "completed_scene_schedule": [], "initial_memory": tensor_hashes(initial),
        "base_checkpoint_sha256": base_checkpoint_sha256, "base_before": base_before,
        "model_class": type(model).__name__, "device": str(device), "torch": torch.__version__,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "psutil": psutil.__version__,
                    "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
                    "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset")},
        "parameter_counts": model.parameter_counts(), "data": dataset.provenance(),
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate, "weight_decay": weight_decay,
                      "betas": [0.9, 0.999], "eps": 1e-8, "gradient_clip_norm": 1.0},
        "loss": "Mean RGB[0,1] absolute error across both branches and all65 teacher-observed steps; no door weighting",
        "temporal_graph": "Full65 steps; no state detach; new zero state per paired episode",
        "checkpoint_selection": "Final fixed budget only; no validation selection", "updates": []}
    started = time.monotonic()

    def save_recovery():
        payload = {"schema": "worldline-room-memory-checkpoint-v1", "mode": mode, "seed": seed,
                   "sequence_path": sequence_path,
                   "completed_updates": report["completed_updates"], "base_checkpoint_sha256": base_checkpoint_sha256,
                   "memory_state_dict": memory_state(model)}
        # This one atomic bundle is authoritative if a stop occurs between files.
        atomic_write(out / "recovery-last.pt", dict(payload, optimizer_state_dict=optimizer.state_dict(),
                     rng=rng_state(device), completed_scene_schedule=list(report["completed_scene_schedule"])), tensor=True)
        atomic_write(out / "memory-last.pt", payload, tensor=True)

    try:
        atomic_write(out / "metrics.json", report)
        save_recovery()
        model.train()
        for scene in schedule:
            if check is not None:
                check()
            _sync(device)
            tick = time.monotonic()
            rgb, actions = dataset.load_pair(scene)
            rgb, actions = rgb.to(device), actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model, rgb, actions, mode, check=check)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite sequence loss")
            loss.backward()
            parameters = list(model.memory_parameters())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("Missing or nonfinite memory gradients")
            if any(p.grad is not None for p in model.base.parameters()):
                raise RuntimeError("Frozen base received a gradient")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError("Optimizer produced nonfinite memory parameters")
            _sync(device)
            report["completed_updates"] += 1
            report["completed_scene_schedule"].append(int(scene))
            save_recovery()
            row = {"update": report["completed_updates"], "scene": int(scene), "loss": float(loss.detach()),
                   "gradient_norm_before_clip": float(norm), "seconds_with_recovery_write": time.monotonic() - tick}
            if device.type == "mps":
                row.update(mps_active_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
            report["updates"].append(row)
            atomic_write(out / "metrics.json", report)
        report["base_after"] = tensor_hashes(model.base.state_dict())
        if report["base_after"] != base_before:
            raise RuntimeError("Frozen predictor tensors changed")
        report["final_memory"] = tensor_hashes(memory_state(model))
        shutil.copyfile(out / "memory-last.pt", out / "memory-final.pt")
        report["memory_final_sha256"] = sha256(out / "memory-final.pt")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            # Do not replace the last validated bundle with a partial/failed update.
            report["memory_last_sha256"] = sha256(out / "memory-last.pt")
            report["recovery_last_sha256"] = sha256(out / "recovery-last.pt")
        except BaseException as error:
            report["recovery_save_error"] = str(error)
            if report["status"] == "complete":
                report["status"] = "failed"
        report["elapsed_seconds"] = time.monotonic() - started
        if phase == "profile" and report["completed_updates"]:
            report["estimated_1200_update_seconds"] = report["elapsed_seconds"] * 1200 / report["completed_updates"]
            report["estimate_limit"] = "Linear estimate from this arm; excludes worker startup, model loading and separate evaluation; not a completed1200-update run"
        atomic_write(out / "metrics.json", report)
    return report


def guard_process(process, output, *, max_seconds=600, max_rss_gib=18, minimum_available_gib=2, interval=.5):
    """Supervise only the launched child, preserving a separate terminal record."""
    output = Path(output)
    started, reason, rows, error = time.monotonic(), None, [], None
    try:
        while process.poll() is None:
            try:
                rss = psutil.Process(process.pid).memory_info().rss
            except psutil.NoSuchProcess:
                break
            available = psutil.virtual_memory().available
            row = {"seconds": time.monotonic() - started, "rss_bytes": rss, "available_bytes": available}
            rows.append(row)
            if row["seconds"] > max_seconds:
                reason = "elapsed_time_limit"
            elif rss > max_rss_gib * 2**30:
                reason = "process_rss_limit"
            elif available < minimum_available_gib * 2**30:
                reason = "available_memory_limit"
            if reason:
                break
            time.sleep(interval)
    except BaseException as caught:
        error = caught
        reason = "parent_interrupted" if isinstance(caught, KeyboardInterrupt) else "parent_failure"
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except BaseException as cleanup_error:
                process.kill()
                if error is None and not isinstance(cleanup_error, subprocess.TimeoutExpired):
                    error = cleanup_error
                    reason = "parent_interrupted" if isinstance(error, KeyboardInterrupt) else "parent_failure"
        process.wait()
    terminal = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed" if error else "stopped" if reason else "complete" if process.returncode == 0 else "failed",
                "reason": reason, "exit_code": process.returncode, "elapsed_seconds": time.monotonic() - started,
                "max_seconds": max_seconds, "max_rss_gib": max_rss_gib, "minimum_available_gib": minimum_available_gib,
                "sample_interval_seconds": interval, "samples": rows,
                "limit": "RSS/available RAM sampled in parent; Metal allocator cap is set in worker; stopped runs retain last completed recovery files"}
    atomic_write(output / "terminal.json", terminal)
    if error is not None:
        raise error
    return terminal


def handoff_and_guard(process, config, output, **limits):
    """Cover the stdin handoff as well as the later supervised execution."""
    try:
        process.stdin.write(json.dumps(config).encode())
        process.stdin.close()
        return guard_process(process, output, **limits)
    except BaseException as error:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except BaseException:
                process.kill()
                process.wait()
        if not (Path(output) / "terminal.json").exists():
            atomic_write(Path(output) / "terminal.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "reason": "worker_configuration_handoff_failed", "error_type": type(error).__name__,
                "error": str(error), "exit_code": process.returncode})
        raise


def make_model(base, seed, device="cpu"):
    if sha256(base) != BASE_SHA256:
        raise ValueError("The published predictor checkpoint hash differs")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    predictor, _ = load_model(base, device="cpu")
    return RoomMemoryModel(predictor).to(device)


def matched_arms(carry, reset):
    return (carry["status"] == reset["status"] == "complete"
            and carry["mode"] == "carry" and reset["mode"] == "reset" and carry["seed"] == reset["seed"]
            and carry.get("sequence_path", "stepwise") in SEQUENCE_PATHS
            and carry.get("sequence_path", "stepwise") == reset.get("sequence_path", "stepwise")
            and carry["completed_updates"] == reset["completed_updates"] == carry["requested_updates"] == reset["requested_updates"]
            and carry["completed_scene_schedule"] == reset["completed_scene_schedule"] == carry["scene_schedule"] == reset["scene_schedule"]
            and carry["initial_memory"] == reset["initial_memory"] and carry["base_before"] == reset["base_before"]
            and carry["base_after"] == reset["base_after"] == carry["base_before"]
            and carry["optimizer"] == reset["optimizer"] and carry["data"] == reset["data"])


def _worker(config):
    torch.set_num_threads(2)
    sequence_path = config.get("sequence_path", "stepwise")
    if set(config["source_sha256"]) != set(training_source_names(sequence_path)):
        raise ValueError("Source snapshot does not cover the selected sequence path")
    for name, expected in config["source_sha256"].items():
        if Path(name).name != name or sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError("Training source changed after the study snapshot")
    if config["device"] == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        torch.mps.set_per_process_memory_fraction(min(1., 18 * 2**30 / torch.mps.recommended_max_memory()))
    model = make_model(config["base"], config["seed"], config["device"])
    checkpoint = Path(config["initial"])
    if sha256(checkpoint) != config["initial_sha256"]:
        raise ValueError("Initial memory file changed")
    values = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restore_memory(model, values)
    dataset = DevelopmentDataset(config["train"], "train")
    if dataset.manifest_sha256 != config["train_manifest_sha256"]:
        raise ValueError("Training manifest changed")
    result = run_arm(model, dataset, config["schedule"], config["mode"], config["output"],
            seed=config["seed"], phase=config["phase"], learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"], sequence_path=sequence_path)
    if result["status"] != "complete":
        raise RuntimeError("Training did not preserve a complete recovery record")


def run_study(train, validation, output, *, phase="plan", updates=50, seeds=SEEDS, device="cpu", base=BASE,
              learning_rate=3e-4, weight_decay=1e-4, max_seconds=600, sequence_path="stepwise"):
    source_names = training_source_names(sequence_path)
    if phase not in ("plan", "profile", "train") or type(updates) is not int or not 1 <= updates <= 1200:
        raise ValueError("Explicit phase and update budget1..1200 are required")
    if phase == "profile" and updates != 50:
        raise ValueError("The planned profile is exactly50 full-sequence updates per arm")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in SEEDS for seed in seeds):
        raise ValueError("Use unique planned initialization seeds20260907/08/09")
    if any(not math.isfinite(v) or v <= 0 for v in (learning_rate, max_seconds)) or not math.isfinite(weight_decay) or weight_decay < 0 or max_seconds > 600:
        raise ValueError("Invalid optimizer values or time cap; maximum600 seconds per arm")
    training, held = DevelopmentDataset(train, "train"), DevelopmentDataset(validation, "validation")
    if sha256(base) != BASE_SHA256:
        raise ValueError("Published predictor hash differs")
    out = new_directory(output)
    snapshot = out / "measured-source"
    snapshot.mkdir()
    sources = {}
    for name in source_names:
        path = Path(__file__).with_name(name)
        copied = snapshot / (name + ".txt")
        shutil.copyfile(path, copied)
        sources[name] = sha256(copied)
        if sources[name] != sha256(path):
            raise ValueError("Training source changed while copying its snapshot")
    report = {"schema": "worldline-room-memory-study-v1", "status": "planned" if phase == "plan" else "running",
        "phase": phase, "sequence_path": sequence_path, "requested_updates_per_arm": updates, "seeds": list(seeds), "arms": ["carry", "reset"],
        "max_seconds_per_arm": max_seconds, "base_checkpoint_sha256": BASE_SHA256, "source_sha256": sources,
        "training": training.provenance(), "validation": held.provenance(), "reserved_test_opened": False,
        "checkpoint_selection": "No selection; compare only complete equal-budget final checkpoints",
        "evaluation_executed": False, "controls_evaluated": False, "runs": [], "matched_budgets": False,
        "schedules": {str(seed): paired_scene_schedule(training.scene_ids, seed, updates) for seed in seeds}}
    atomic_write(out / "study.json", report)
    if phase == "plan":
        return report
    try:
        for seed in seeds:
            seed_dir = out / str(seed)
            seed_dir.mkdir()
            model = make_model(base, seed)
            initial_path = seed_dir / "initial-memory.pt"
            atomic_write(initial_path, memory_state(model), tensor=True)
            del model
            pair = []
            for mode in ("carry", "reset"):
                wrapper_dir = seed_dir / mode
                wrapper_dir.mkdir()
                config = {"train": str(Path(train).resolve()), "train_manifest_sha256": training.manifest_sha256,
                    "base": str(Path(base).resolve()), "initial": str(initial_path.resolve()), "initial_sha256": sha256(initial_path),
                    "output": str((wrapper_dir / "training").resolve()), "mode": mode, "seed": seed, "phase": phase,
                    "device": device, "sequence_path": sequence_path, "source_sha256": sources,
                    "schedule": report["schedules"][str(seed)], "learning_rate": learning_rate, "weight_decay": weight_decay}
                with (wrapper_dir / "worker.log").open("x") as log:
                    process = subprocess.Popen([sys.executable, "-m", "experiments.room_world.memory_train", "--worker"],
                        stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0"))
                    terminal = handoff_and_guard(process, config, wrapper_dir, max_seconds=max_seconds)
                path = wrapper_dir / "training/metrics.json"
                arm = json.loads(path.read_text()) if path.exists() else {"status": "failed", "completed_updates": 0}
                report["runs"].append({"seed": seed, "mode": mode, "terminal_status": terminal["status"],
                    "training_status": arm["status"], "sequence_path": arm.get("sequence_path", "stepwise"),
                    "completed_updates": arm["completed_updates"],
                    "metrics": str(path.relative_to(out)), "metrics_sha256": sha256(path) if path.exists() else None})
                atomic_write(out / "study.json", report)
                if terminal["status"] != "complete" or arm["status"] != "complete":
                    raise RuntimeError("Arm did not complete its fixed budget; no carry/reset comparison is valid")
                if arm.get("sequence_path", "stepwise") != sequence_path:
                    raise RuntimeError("Worker sequence path differs from the declared study")
                pair.append(arm)
            if not matched_arms(*pair):
                raise RuntimeError("Carry/reset realized budgets, initial parameters or schedules differ")
        report["matched_budgets"] = True
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_write(out / "study.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("plan", "profile", "train"), default="plan")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--sequence-path", choices=SEQUENCE_PATHS, default="stepwise")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.worker:
        _worker(json.loads(sys.stdin.read(65536)))
        return
    if any(value is None for value in (args.train, args.validation, args.output)):
        parser.error("--train, --validation and --output are required")
    if args.phase == "train" and args.updates is None:
        parser.error("Training requires an explicit equal update budget chosen after profiling")
    result = run_study(args.train, args.validation, args.output, phase=args.phase, updates=args.updates if args.updates is not None else 50,
                       seeds=args.seeds, device=args.device, base=args.base, max_seconds=args.max_seconds, sequence_path=args.sequence_path)
    print(json.dumps({key: result[key] for key in ("status", "phase", "matched_budgets", "evaluation_executed")}))


if __name__ == "__main__":
    main()
