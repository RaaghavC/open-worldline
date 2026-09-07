# SPDX-License-Identifier: Apache-2.0
"""Cache verified original RGB/action windows with the external official VAE.

Run as a module from the repository root. This creates development inputs,
not a trained model or an independent train/test split.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time

import psutil
from safetensors.torch import save_file
import torch

from .capture_data import ACTION_CHANNELS, ACTION_SCHEMA, load_window, sha256
from .codec.helper import OfficialWanCodec, VAE_SHA256


GiB = 1024 ** 3
SELECTION = [(arm, start) for start in (0, 8, 32, 49) for arm in ("closed", "open")]


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 900:
        parser.error("Time limit must be positive and at most 900 seconds")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output directory must be new; earlier evidence is never replaced")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; choose CPU explicitly")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(min(1., 18 * GiB / torch.mps.recommended_max_memory()))
    report = {
        "schema": "worldline-wan-atrium-cache-v1", "status": "running",
        "protocol": "worldline-wan-atrium-17-v1", "split": "development", "independent_layouts": 1,
        "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md")),
        "reader_sha256": sha256(Path(__file__).with_name("capture_data.py")),
        "builder_sha256": sha256(Path(__file__)),
        "codec_helper_sha256": sha256(Path(__file__).parent / "codec/helper.py"),
        "vae_weights_sha256": VAE_SHA256, "device": args.device, "compute_dtype": "float16",
        "stored_dtype": "float32", "torch": torch.__version__, "platform": platform.platform(),
        "automatic_mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "max_seconds": args.max_seconds, "max_memory_gib": 18, "minimum_available_gib": 2,
        "action_schema": ACTION_SCHEMA, "action_channels": list(ACTION_CHANNELS),
        "selection": [{"arm": arm, "start": start, "frames": 17} for arm, start in SELECTION],
        "input_rgb": "Original native AgX PNG uint8, converted to float32 RGB/127.5-1; no resize/crop",
        "vae_normalization": "Pinned official Wan mean/std in codec/helper.py",
        "observed_input": "Separately encoded first RGB frame with fresh VAE temporal cache",
        "targets": "Independently encoded 17 RGB frames per window; never clean future conditioning",
        "causal_tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "cache_policy": "Official VAE clears temporal state per encode; MPS unused allocator cache cleared after each stage",
        "windows": [], "causal_checks": [], "timings": [],
    }
    started = time.perf_counter()
    stage = ["validate_inputs"]
    stopped = threading.Event()
    process = psutil.Process()
    peaks = {"rss_bytes": 0, "mps_active_bytes": 0, "mps_driver_bytes": 0}
    write_json(args.output / "manifest.json", report)

    def monitor():
        with (args.output / "memory.jsonl").open("x") as log:
            while not stopped.is_set():
                row = {"seconds": time.perf_counter() - started, "stage": stage[0],
                       "rss_bytes": process.memory_info().rss,
                       "available_system_bytes": psutil.virtual_memory().available}
                if args.device == "mps":
                    row.update(mps_active_bytes=torch.mps.current_allocated_memory(),
                               mps_driver_bytes=torch.mps.driver_allocated_memory())
                for key in peaks:
                    peaks[key] = max(peaks[key], row.get(key, 0))
                log.write(json.dumps(row) + "\n")
                log.flush()
                reason = None
                if row["seconds"] > args.max_seconds:
                    reason = "time limit"
                elif max(row["rss_bytes"], row.get("mps_driver_bytes", 0)) > 18 * GiB:
                    reason = "memory limit"
                elif row["available_system_bytes"] < 2 * GiB:
                    reason = "system available memory below 2 GiB"
                if reason:
                    write_json(args.output / "watchdog-stop.json", {"status": "stopped", "reason": reason, "last_sample": row})
                    os._exit(124)
                stopped.wait(.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        windows = {(arm, start): load_window(args.capture, arm, start) for arm, start in SELECTION}
        stage[0] = "load_vae"
        codec = OfficialWanCodec(args.weights, args.device, torch.float16)
        if args.device == "mps":
            torch.mps.empty_cache()

        def encode(label, video):
            stage[0] = label
            tick = time.perf_counter()
            with torch.inference_mode():
                result = codec.encode(video).float().cpu().contiguous()
            if not torch.isfinite(result).all():
                raise ValueError(f"Non-finite VAE output: {label}")
            gc.collect()
            if args.device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            report["timings"].append({"stage": label, "seconds": time.perf_counter() - tick})
            print(json.dumps(report["timings"][-1]), flush=True)
            return result

        def equal(label, a, b):
            if a.shape != b.shape or not torch.allclose(a, b, atol=1e-5, rtol=1e-5):
                raise ValueError(f"Causal/cache consistency failure: {label}")
            report["causal_checks"].append({"check": label, "passed": True,
                                            "max_absolute_difference": (a - b).abs().max().item()})

        # Actual complete A -> B -> A sequence checks state isolation, followed
        # by future perturbations for two different original initial images.
        a_video = torch.from_numpy(windows[("closed", 0)].video_array())
        b_video = torch.from_numpy(windows[("open", 49)].video_array())
        a_latent = encode("causal_A", a_video)
        b_latent = encode("causal_B", b_video)
        a_repeat = encode("causal_A_after_B", a_video)
        equal("complete A after B equals original A", a_latent, a_repeat)
        for name, video, latent in (("A", a_video, a_latent), ("B", b_video, b_latent)):
            changed = video.clone()
            changed[:, :, 1:] = -changed[:, :, 1:]
            if (changed[:, :, 1:] - video[:, :, 1:]).abs().mean() < .1:
                raise ValueError("Future perturbation must materially change the future images")
            altered = encode(f"{name}_changed_future", changed)
            equal(f"{name} first latent invariant to changed future", latent[:, :, :1], altered[:, :, :1])
        known = {("closed", 0): a_latent, ("open", 49): b_latent}
        for arm, start in SELECTION:
            sample = windows[(arm, start)]
            video = torch.from_numpy(sample.video_array())
            name = f"{arm}-{start:04d}"
            first = encode(name + "_first_only", video[:, :, :1])
            full = known[(arm, start)] if (arm, start) in known else encode(name + "_full", video)
            if full.shape != (1, 16, 5, 36, 64) or first.shape != (1, 16, 1, 36, 64):
                raise ValueError("Latent shapes disagree with native Wan encoding")
            equal(name + " first-only equals full-clip prefix", first, full[:, :, :1])
            filename = name + ".safetensors"
            tensors = {"target": full, "observation": first,
                       "actions": torch.from_numpy(sample.actions[None]).contiguous()}
            save_file(tensors, str(args.output / filename))
            report["windows"].append({"id": name, "file": filename,
                                       "sha256": sha256(args.output / filename),
                                       "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                           "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}
                                           for key, value in tensors.items()},
                                       "source": sample.provenance})
            write_json(args.output / "manifest.json", report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stopped.set()
        monitor_thread.join(timeout=2)
        report["peaks_sampled"] = peaks
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "manifest.json", report)
    print(json.dumps({"status": report["status"], "windows": len(report["windows"]),
                      "causal_checks": len(report["causal_checks"]), "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
