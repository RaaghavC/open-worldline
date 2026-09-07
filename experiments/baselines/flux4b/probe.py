#!/usr/bin/env python3
"""Bounded, local MFLUX appearance baseline. Default preparation uses no GPU.

This invokes released third-party converted FLUX.2 klein 4B weights. It is not
Worldline training or a world-model benchmark. See the adjacent provenance note.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback

BASE = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE / "images/renderer-original.png"
ORIGINAL_RENDERER_SHA256 = "734f35de9d917a4d3d98873c29f08c1dee2e16c8c8329e341cd1309a064984ec"
DEFAULT_PROMPT = (
    "Convert this landscape rendering into a detailed, photorealistic photograph of the same place. "
    "Keep the camera position, mountain silhouettes, lake and river boundaries, and locations "
    "and sizes of the turquoise crystals unchanged. Give the purple-gray mountains natural "
    "fractured rock layers, mineral grain, small weathered stones and crevices. Make the "
    "turquoise crystals translucent quartz with realistic facets and varied light reflection. "
    "Make the water transparent blue-green with subtle ripples and reflections. Use soft warm "
    "sunrise light and light atmospheric haze. Keep every visible object and its position. "
    "No buildings, people, animals, added trees, text or labels."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bounds(width, height, steps, seed):
    if width <= 0 or height <= 0 or width % 16 or height % 16 or width * height > 768 * 432:
        raise ValueError("Dimensions must be positive multiples of 16, with at most 331776 total pixels.")
    if not 1 <= steps <= 64:
        raise ValueError("Steps must be between 1 and 64 for this bounded probe.")
    if not 1 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be a positive 32-bit unsigned integer.")


def source_provenance(digest):
    return ("Bundled original programmed renderer image, verified by SHA-256"
            if digest == ORIGINAL_RENDERER_SHA256 else
            "Caller-supplied image; this probe does not establish its origin or ownership")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Load the model and use the GPU.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True, help="A separate local results directory.")
    parser.add_argument("--weights", type=Path, required=True, help="Explicitly downloaded external weights directory.")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()
    try:
        validate_bounds(args.width, args.height, args.steps, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    from PIL import Image

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "result.json"
    if args.run and output.exists():
        parser.error("Use a fresh output directory to retain existing inference evidence.")
    manifest_path = BASE / "weights-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        path = args.weights / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Weight integrity failure: {entry['path']}")
    source_digest = sha256(args.source)
    source = Image.open(args.source).convert("RGB")
    if abs(source.width / source.height - args.width / args.height) > .001:
        parser.error("The bounded probe preserves the full source image aspect ratio.")
    reference_path = args.output_dir / "reference.png"
    source.resize((args.width, args.height), Image.Resampling.LANCZOS).save(reference_path)
    prompt = args.prompt_file.read_text().strip() if args.prompt_file else DEFAULT_PROMPT
    (args.output_dir / "prompt.txt").write_text(prompt + "\n")
    report = {
        "status": "prepared",
        "purpose": "External released image-editing baseline on a supplied image",
        "not_evidence_for": ["our own trained model", "world-model quality", "geometric consistency", "Genie 3 parity"],
        "source": {"path": args.source.name, "sha256": source_digest, "size": list(source.size),
                   "provenance": source_provenance(source_digest)},
        "reference": {"path": reference_path.name, "sha256": sha256(reference_path), "size": [args.width, args.height], "preprocessing": "RGB conversion and full-frame LANCZOS resize only"},
        "model": manifest,
        "prompt": prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance": 1.0,
        "scheduler": "flow_match_euler_discrete",
        "environment": {"python": sys.version, "platform": platform.platform(), "versions": {p: importlib.metadata.version(p) for p in ("mflux", "mlx", "mlx-metal", "numpy", "Pillow", "psutil", "transformers", "huggingface-hub")}},
        "script_sha256": sha256(__file__),
    }
    (args.output_dir / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.run:
        print(json.dumps({"status": "prepared", "weights_verified": len(manifest["files"]), "download_bytes": manifest["bytes"], "reference": str(reference_path), "gpu_used": False}))
        return

    # Require local files: missing dependencies or assets fail instead of fetching.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import mlx.core as mx
    import psutil
    from mflux.callbacks.instances.memory_saver import MemorySaver
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2KleinEdit

    # MLX's memory limit is an allocator guideline, not an operating-system cap.
    mx.set_memory_limit(16 * 1024**3)
    mx.set_cache_limit(1000**3)
    mx.reset_peak_memory()
    process = psutil.Process()
    samples = []
    stop = threading.Event()
    started = time.perf_counter()
    phase = "load"

    def sample():
        while not stop.is_set():
            samples.append({"seconds": time.perf_counter() - started, "phase": phase, "rss_bytes": process.memory_info().rss,
                            "mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(), "mlx_peak_bytes": mx.get_peak_memory()})
            stop.wait(.05)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        begin = time.perf_counter()
        model = Flux2KleinEdit(model_config=ModelConfig.flux2_klein_4b(), model_path=str(args.weights.resolve()))
        mx.eval(model.parameters())
        mx.synchronize()
        report["load_seconds"] = time.perf_counter() - begin
        report["load_mlx_peak_bytes"] = mx.get_peak_memory()
        report["loaded_quantization_bits"] = model.bits
        print(json.dumps({"loaded_seconds": report["load_seconds"], "quantization_bits": model.bits}), flush=True)
        saver = MemorySaver(model, keep_transformer=False, cache_limit_bytes=1000**3, num_seeds=1)
        model.callbacks.register(saver)
        phase = "generation"
        begin = time.perf_counter()
        generated = model.generate_image(seed=args.seed, prompt=prompt, width=args.width, height=args.height,
                                         image_paths=[reference_path], guidance=1.0,
                                         num_inference_steps=args.steps, scheduler="flow_match_euler_discrete")
        mx.synchronize()
        report["generation_seconds_full"] = time.perf_counter() - begin
        report["generation_seconds_reported_loop"] = generated.generation_time
        report["generation_mlx_peak_bytes"] = mx.get_peak_memory()
        phase = "save"
        image_path = args.output_dir / "generated.png"
        generated.image.save(image_path)  # Keep local file paths out of PNG metadata.
        report["generated"] = {"path": image_path.name, "sha256": sha256(image_path), "size": list(generated.image.size)}
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["exception"] = str(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        stop.set()
        monitor.join()
        report["max_rss_bytes_getrusage_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["max_sampled_rss_bytes"] = max((s["rss_bytes"] for s in samples), default=0)
        report["max_sampled_mlx_active_bytes"] = max((s["mlx_active_bytes"] for s in samples), default=0)
        report["memory_note"] = "MLX and RSS overlap. Do not add them. MLX peaks count array allocations; sampled RSS can miss brief peaks. Low-RAM mode releases text encoder and transformer after use."
        report["measurement"] = {"sample_interval_seconds": .05, "cache_limit_bytes": 1000**3, "mlx_memory_guideline_bytes": 16 * 1024**3, "cold_model_load": True, "single_generation": True, "warm_latency_measured": False}
        (args.output_dir / "memory-samples.json").write_text(json.dumps(samples, indent=2) + "\n")
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k in ("status", "load_seconds", "generation_seconds_full", "generation_seconds_reported_loop", "generation_mlx_peak_bytes", "max_rss_bytes_getrusage_macos", "exception")}), flush=True)


if __name__ == "__main__":
    main()
