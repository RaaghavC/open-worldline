# SPDX-License-Identifier: Apache-2.0
"""Numerical checks and two-branch accumulation, with no loader or launcher."""
import math
import time

import torch

from ..action_training.objective import (
    SEED, OPTIMIZER, fresh_adapter, make_draws, future_flow_mse,
    grad_norm, require_finite_tree,
)
from ..official_cpu.streaming import tensor_sha


PARITY = {"max_absolute": 1e-6, "relative_l2": 1e-6}
PREFIX = {"max_absolute": 1e-5, "relative_l2": 1e-5}


def cross_length_prefix(target_prefix, observation):
    """Separate native codec bound; never changes either saved latent tensor."""
    for value in (target_prefix, observation):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != "cpu"
                or value.ndim != 5 or value.shape[0] != 1 or value.shape[2] != 1 or not torch.isfinite(value).all()):
            raise ValueError("Finite CPU FP32 single-latent prefix tensors required")
    if target_prefix.shape != observation.shape:
        raise ValueError("Prefix dimensions differ")
    difference = target_prefix.double() - observation.double()
    norm, delta = float(observation.double().norm()), float(difference.norm())
    relative = delta/norm if norm else 0. if delta == 0. else None
    maximum = float(difference.abs().max())
    result = {"max_absolute": maximum, "relative_l2": relative,
              "exact_equal": torch.equal(target_prefix, observation), "tolerances": dict(PREFIX),
              "passed": maximum <= PREFIX["max_absolute"] and relative is not None and relative <= PREFIX["relative_l2"]}
    if not result["passed"]:
        raise ValueError("Cross-length native codec prefix check failed")
    return result


def flow_inputs(target, observation, noise, k):
    """Native CUDA FM input; raw target stays intact, observed input is exact.

    Unlike the frozen MPS helper, the new native cross-length codec gate has
    its separately declared 1e-5 bound. Only the newly constructed noisy input
    is clamped. The loss never scores the initial target/velocity latent.
    """
    for value in (target, observation, noise):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.device.type != "cpu" or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError("Training inputs must be finite non-gradient CPU FP32 tensors")
    if (type(k) is not int or not 50 <= k <= 950 or target.ndim != 5 or target.shape[0] != 1
            or target.shape[2] != 5 or target.shape != noise.shape
            or observation.shape != target[:, :, :1].shape or min(target.shape[-2:]) < 2
            or any(size % 2 for size in target.shape[-2:])):
        raise ValueError("Aligned B=1 five-latent video, one observation and integer k in [50,950] required")
    cross_length_prefix(target[:, :, :1], observation)
    sigma = k / 1000.
    noisy = (1-sigma)*target + sigma*noise
    noisy[:, :, :1] = observation
    velocity = noise-target
    prefix = (target.shape[-2]//2)*(target.shape[-1]//2)
    times = torch.full((1, 5*prefix), k, dtype=torch.int64, device="cpu")
    times[:, :prefix] = 0
    return noisy, times, velocity


def compare_velocity(native, bridged):
    """Both fixed bounds must pass. A zero reference has a defined exact case."""
    if (not isinstance(native, torch.Tensor) or not isinstance(bridged, torch.Tensor)
            or native.shape != bridged.shape or native.ndim != 5 or native.shape[0] != 1
            or native.shape[2] != 5 or min(native.shape) < 1
            or native.dtype != torch.float32 or bridged.dtype != torch.float32
            or native.device.type != "cpu" or bridged.device.type != "cpu"
            or not torch.isfinite(native).all() or not torch.isfinite(bridged).all()):
        raise ValueError("Matched finite CPU FP32 velocity tensors are required")
    # CPU float64 scoring avoids FP32 square/sum overflow or rounding the norm.
    reference = native.double()
    difference = bridged.double() - reference
    norm = float(reference.norm().item())
    difference_norm = float(difference.norm().item())
    relative = difference_norm / norm if norm else 0. if difference_norm == 0. else None
    maximum = float(difference.abs().max().item())
    return {
        "passed": maximum <= PARITY["max_absolute"] and relative is not None and relative <= PARITY["relative_l2"],
        "max_absolute": maximum, "relative_l2": relative,
        "reference_l2": norm, "difference_l2": difference_norm,
        "rms_difference": float(difference.square().mean().sqrt().item()),
        "exact_equal": torch.equal(native, bridged),
        "observed_velocity_exact_equal": torch.equal(native[:, :, :1], bridged[:, :, :1]),
        "native_sha256": tensor_sha(native), "bridged_sha256": tensor_sha(bridged),
        "tolerances": dict(PARITY),
        "relative_definition": "L2(bridged-native)/L2(native); 0 when both norms are zero, null and fail for nonzero difference with zero reference",
        "scoring_dtype": "float64 CPU", "automatic_tolerance_relaxation": False,
    }


def condition_identity(noisy, times, velocity, window):
    return {key: tensor_sha(value) for key, value in {
        "noisy": noisy, "token_times": times, "flow_target": velocity,
        "observation": window["observation"], "commands": window["commands"],
    }.items()}


def paired_update(bridge, windows, noise, k, context, optimizer, *, check=None):
    """Closed then open, half future loss each, exactly one optimizer step.

    Future target latents are used to form the usual noisy training input and
    FM loss target. The bridge only receives noisy latents, token times, text,
    requested commands and the independent initial observation.
    """
    if not isinstance(windows, (tuple, list)) or len(windows) != 2:
        raise ValueError("Exactly two ordered closed/open windows are required")
    if any(p.requires_grad or p.grad is not None for p in bridge.core.parameters()):
        raise ValueError("Core parameters must remain frozen and gradient-free")
    parameters = list(bridge.adapter.parameters())
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in parameters}:
        raise ValueError("Optimizer must own exactly the adapter parameters")
    device = parameters[0].device
    optimizer.zero_grad(set_to_none=True)
    branches = []
    for label, window in zip(("closed", "open"), windows):
        if check:
            check()
        if not isinstance(window, dict) or set(window) != {"target", "observation", "commands"}:
            raise ValueError("A window must contain only target, observation and commands")
        noisy, times, velocity = flow_inputs(window["target"], window["observation"], noise, k)
        identity = condition_identity(noisy, times, velocity, window)
        if not torch.equal(noisy[:, :, :1], window["observation"]):
            raise RuntimeError("Noisy training input lost its exact observed prefix")
        cuda_noisy = noisy.to(device)
        prediction = bridge(cuda_noisy, times.to(device), [context.to(device)],
                            commands=window["commands"].to(device), observation=window["observation"].to(device))
        loss = future_flow_mse(prediction, velocity.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite branch loss")
        (loss * .5).backward()
        prefix_exact = torch.equal(cuda_noisy[:, :, :1], window["observation"].to(device))
        if not prefix_exact:
            raise RuntimeError("Bridge mutated the clean training prefix")
        branches.append({"branch": label, "future_flow_mse": float(loss.detach().cpu()),
                         "input_sha256": identity, "observed_input_prefix_exact": prefix_exact})
        del prediction, loss, noisy, cuda_noisy, times, velocity
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("Every adapter gradient must be present and finite")
    before = grad_norm(parameters)
    recurrent = grad_norm(bridge.adapter.command_gru.parameters())
    output = grad_norm(bridge.adapter.output.parameters())
    if not math.isfinite(before) or before <= 0:
        raise FloatingPointError("A finite positive aggregate adapter gradient is required")
    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    after = grad_norm(parameters)
    optimizer.step()
    require_finite_tree(optimizer.state_dict())
    if any(not torch.isfinite(p).all() for p in parameters):
        raise FloatingPointError("Nonfinite adapter parameters after update")
    if any(p.requires_grad or p.grad is not None for p in bridge.core.parameters()):
        raise RuntimeError("Core gradient ownership changed")
    return {"branches": branches, "paired_mean_future_flow_mse": sum(row["future_flow_mse"] for row in branches) / 2,
            "gradient_l2_before_clip": before, "gradient_l2_after_clip": after,
            "command_gru_gradient_l2": recurrent, "output_gradient_l2": output,
            "live_sequential_forwards": 2, "optimizer_updates": 1,
            "loss": "Mean future latent velocity MSE; half each branch; initial latent excluded"}


def execute_steps(bridge, windows, schedule, draws, context, optimizer, *, native_predict,
                  retain, checkpoint, progress, check=lambda: None, synchronize=lambda: None):
    """Bounded numerical protocol; callbacks retain evidence and enforce guards.

    The production worker supplies the literal native prediction and CUDA
    synchronizer. CPU fixtures supply explicit stand-ins. Both zero-adapter
    comparisons finish before any optimizer update. No sampling or decoding.
    """
    if (len(schedule) != 2 or [row.get("start") for row in schedule] != [0, 8]
            or [row.get("branches") for row in schedule] != [["closed-0000", "open-0000"], ["closed-0008", "open-0008"]]):
        raise ValueError("Only the prescribed two-update start0/start8 schedule is accepted")
    if torch.count_nonzero(bridge.adapter.output.weight) or torch.count_nonzero(bridge.adapter.output.bias):
        raise ValueError("The numerical probe must start with the exact zero output projection")
    if optimizer.state:
        raise ValueError("The numerical probe must start with a fresh optimizer")
    report = {"completed_updates": 0, "native_comparisons": [], "updates": [],
              "native_reference_predictions": 0, "bridge_predictions": 0,
              "zero_adapter_gate_passed": False, "image_generation": False, "quality_assessed": False}
    checkpoint(0, report)
    progress(report)
    device = next(bridge.adapter.parameters()).device
    row = schedule[0]
    for identity in row["branches"]:
        check()
        window = windows[identity]
        noisy, times, velocity = flow_inputs(window["target"], window["observation"], draws[row["noise_key"]], row["k"])
        hashes = condition_identity(noisy, times, velocity, window)
        synchronize(); begin = time.monotonic()
        reference = native_predict(noisy, times, context)
        synchronize(); native_seconds = time.monotonic() - begin
        report["native_reference_predictions"] += 1
        retain("parity-" + identity + "-native", {"native_velocity": reference})
        report["stage"] = "zero-adapter/" + identity + "/bridge"
        progress(report)
        check()
        synchronize(); begin = time.monotonic()
        bridge_input = noisy.to(device)
        actual = bridge(bridge_input, times.to(device), [context.to(device)],
                        commands=window["commands"].to(device), observation=window["observation"].to(device), track_grad=False)
        synchronize(); bridge_seconds = time.monotonic() - begin
        actual = actual.detach().cpu()
        report["bridge_predictions"] += 1
        retain("parity-" + identity + "-bridge", {"bridged_velocity": actual})
        report["stage"] = "zero-adapter/" + identity + "/compare"
        progress(report)
        comparison = compare_velocity(reference, actual)
        prefix_exact = torch.equal(bridge_input[:, :, :1], window["observation"].to(device))
        unchanged = hashes == condition_identity(noisy, times, velocity, window)
        comparison.update(window_id=identity, input_sha256=hashes, native_seconds=native_seconds,
                          bridge_seconds=bridge_seconds, input_tensors_unchanged=unchanged,
                          observed_input_prefix_exact=prefix_exact)
        report["native_comparisons"].append(comparison)
        progress(report)
        if not prefix_exact or not unchanged:
            raise RuntimeError("Parity call mutated an input or its clean prefix")
        del noisy, times, velocity, reference, actual, bridge_input
    if not all(row["passed"] for row in report["native_comparisons"]):
        raise RuntimeError("Zero-adapter native parity gate failed; no optimizer updates are permitted")
    report["zero_adapter_gate_passed"] = True
    progress(report)
    for index, row in enumerate(schedule, 1):
        check()
        synchronize(); begin = time.monotonic()
        result = paired_update(bridge, [windows[identity] for identity in row["branches"]],
                               draws[row["noise_key"]], row["k"], context, optimizer, check=check)
        synchronize(); result["seconds"] = time.monotonic() - begin
        result["update"] = index
        result["start"] = row["start"]
        report["bridge_predictions"] += result["live_sequential_forwards"]
        report["updates"].append(result)
        progress(report)
        recurrent = result["command_gru_gradient_l2"]
        if (index == 1 and recurrent != 0.) or (index == 2 and (not math.isfinite(recurrent) or recurrent <= 0.)):
            raise RuntimeError("Expected first-zero then second-positive recurrent gradient did not occur")
        # Only fully validated updates may advance the immutable recovery.
        report["completed_updates"] = index
        checkpoint(index, report)
        progress(report)
    if report["native_reference_predictions"] != 2 or report["bridge_predictions"] != 6:
        raise RuntimeError("The prescribed native/bridge prediction counts differ")
    report["second_update_gru_gradient_nonzero"] = True
    progress(report)
    return report
