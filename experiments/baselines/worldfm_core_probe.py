"""Isolated, attributed WorldFM inference probe. This is not the Worldline model.

Without --guide, the reference image is also the guide. The published example
uses one original renderer image and tests execution, not novel-view accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import threading
import time

# This experiment never requests an inference service or missing model download.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from PIL import Image, ImageOps
import psutil
import torch

from modules.worldfm_infer import WorldFMInprocessConfig, WorldFMTriConditionInprocess

SOURCE_COMMIT = "d51ada211079d185285076b56fb803069a571838"
HF_REVISION = "48d206b813a6ff3ddc1977b6cce8b6769c3bbad6"
EXPECTED_HASHES = {
    "worldfm_2-step.pth": "7cf8462ca3e9acaa0c7cf6f5b99917f2dd0b9d03dee9837237c28bf5700c7b17",
    "vae/diffusion_pytorch_model.safetensors": "1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c",
    "vae/config.json": "f18b16fa4381c90ab44a3d7abd2e90afd05a2c31b8ca69998bc93c6453aeb7b7",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent.parent / "worldfm-weights")
    p.add_argument("--reference", type=Path, help="Required unless --verify-only is used.")
    p.add_argument("--guide", type=Path, help="Target-view RGB guide. Defaults to the same image as reference.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("mps", "cpu"), default="mps")
    p.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--verify-only", action="store_true", help="Verify hashes and parameter metadata without GPU inference.")
    args = p.parse_args()
    if not args.verify_only and args.reference is None:
        p.error("--reference is required for inference")
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True).strip()
    if actual_commit != SOURCE_COMMIT:
        raise RuntimeError("Expected the pinned upstream WorldFM source commit")
    args.output.mkdir(parents=True, exist_ok=True)
    weight_info = {}
    for rel, expected in EXPECTED_HASHES.items():
        path = args.weights / rel
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(f"Checkpoint hash mismatch: {rel}: {digest}")
        weight_info[rel] = {"sha256": digest, "bytes": path.stat().st_size}
    report = {
        "baseline": "InSpatio WorldFM official two-step release",
        "source_url": "https://github.com/inspatio/worldfm",
        "source_commit": SOURCE_COMMIT,
        "weights_url": "https://huggingface.co/inspatio/worldfm",
        "hf_revision": HF_REVISION,
        "weights": weight_info,
        "license_evidence": "Apache-2.0 repository license and HF card metadata; VAE upstream provenance not documented.",
        "input_claim": "Reference and guide are caller-supplied images. This probe bypasses the panorama pipeline and measures execution; it does not establish image ownership, photorealism, or novel-view accuracy.",
        "device": args.device,
        "dtype": args.dtype,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "diffusers", "timm", "transformers", "huggingface-hub", "numpy", "Pillow", "safetensors", "accelerate", "einops", "mmcv", "mmengine")},
        "seed": args.seed,
        "sampling_steps": 2,
        "resolution": [512, 512],
    }
    (args.output / "source.patch").write_text(subprocess.check_output(["git", "diff", "--", "modules", "worldfm"], cwd=Path(__file__).resolve().parent, text=True))
    if args.verify_only:
        checkpoint = torch.load(args.weights / "worldfm_2-step.pth", map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint.get("state_dict", checkpoint)
        report["checkpoint_top_level_keys"] = list(checkpoint.keys())[:20]
        report["checkpoint_tensor_count"] = len(state)
        report["checkpoint_tensor_elements"] = sum(t.numel() for t in state.values() if torch.is_tensor(t))
        report["checkpoint_tensor_dtypes"] = sorted({str(t.dtype) for t in state.values() if torch.is_tensor(t)})
        report["inference_run"] = False
        (args.output / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(4)
    reference = ImageOps.fit(Image.open(args.reference).convert("RGB"), (512, 512), method=Image.Resampling.LANCZOS)
    guide_path = args.guide or args.reference
    guide = ImageOps.fit(Image.open(guide_path).convert("RGB"), (512, 512), method=Image.Resampling.LANCZOS)
    report["same_reference_and_guide"] = bool(np.array_equal(np.asarray(reference), np.asarray(guide)))
    reference.save(args.output / "reference.png")
    guide.save(args.output / "guide.png")
    report["reference"] = {"path": str(args.reference.resolve()), "sha256": sha256(args.reference),
                           "model_input_rgb_sha256": hashlib.sha256(np.asarray(reference).tobytes()).hexdigest()}
    report["guide"] = {"path": str(guide_path.resolve()), "sha256": sha256(guide_path),
                       "model_input_rgb_sha256": hashlib.sha256(np.asarray(guide).tobytes()).hexdigest()}

    peak = {"rss_bytes": 0, "mps_tensor_bytes": 0, "mps_driver_bytes": 0}
    stop = threading.Event()
    process = psutil.Process()
    def sample_memory():
        while not stop.is_set():
            peak["rss_bytes"] = max(peak["rss_bytes"], process.memory_info().rss)
            if args.device == "mps":
                peak["mps_tensor_bytes"] = max(peak["mps_tensor_bytes"], torch.mps.current_allocated_memory())
                peak["mps_driver_bytes"] = max(peak["mps_driver_bytes"], torch.mps.driver_allocated_memory())
            stop.wait(.05)
    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    try:
        t0 = time.perf_counter()
        svc = WorldFMTriConditionInprocess(WorldFMInprocessConfig(
            model_path=str((args.weights / "worldfm_2-step.pth").resolve()),
            vae_path=str((args.weights / "vae").resolve()),
            image_size=512, version="sigma", disable_cross_attn=True, step=2,
            device=args.device, weight_dtype=dtype, compile_model=False,
            compile_vae=False, disable_vae_slicing=False, disable_vae_tiling=False,
        ))
        svc.vae.enable_slicing()
        svc.vae.enable_tiling()
        sync(args.device)
        report["load_seconds"] = time.perf_counter() - t0
        report["model_parameters"] = sum(p.numel() for p in svc.model.parameters())
        report["vae_parameters"] = sum(p.numel() for p in svc.vae.parameters())
        report["missing_keys"] = svc.checkpoint_missing_keys
        report["unexpected_keys"] = svc.checkpoint_unexpected_keys
        unexpected_missing = set(svc.checkpoint_missing_keys) - {"pos_embed"}
        if unexpected_missing or svc.checkpoint_unexpected_keys:
            raise RuntimeError(f"Incomplete checkpoint load: missing={sorted(unexpected_missing)}, unexpected={svc.checkpoint_unexpected_keys}")
        svc.set_cond2_from_array(np.asarray(reference, dtype=np.uint8))
        guide_tensor = torch.from_numpy(np.asarray(guide, dtype=np.uint8).copy()).to(args.device)
        report["frames"] = []
        for i in range(args.runs):
            seed = args.seed + i
            torch.manual_seed(seed)
            if args.device == "mps":
                torch.mps.manual_seed(seed)
            sync(args.device)
            t0 = time.perf_counter()
            result = svc.infer_from_render_u8(guide_tensor, profile=True)
            sync(args.device)
            seconds = time.perf_counter() - t0
            finite = bool(torch.isfinite(result).all().item())
            if not finite:
                raise RuntimeError("Model returned NaN or infinite pixels")
            pixels = ((result[0].detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
            outpath = args.output / f"generated-{i:02d}.png"
            Image.fromarray(pixels).save(outpath)
            frame = {"index": i, "seed": seed, "seconds": seconds, "file": outpath.name, "sha256": sha256(outpath), "finite": finite, "pixel_std": float(pixels.std()), "profile_ms": svc._last_profile}
            report["frames"].append(frame)
            print(json.dumps(frame), flush=True)
        steady = [f["seconds"] for f in report["frames"][1:]]
        report["steady_seconds_mean"] = float(np.mean(steady)) if steady else None
        report["steady_frames_per_second"] = 1 / float(np.mean(steady)) if steady else None
        report["inference_run"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        stop.set();monitor.join()
        report["sampled_peak_memory"] = peak
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_max_rss_bytes"] = max_rss if platform.system() == "Darwin" else max_rss * 1024
        (args.output / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
