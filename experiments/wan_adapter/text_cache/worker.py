# SPDX-License-Identifier: Apache-2.0
"""Internal worker for run_cache.py; evaluates real prompts in its own process."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

import psutil
import torch
from safetensors.torch import save_file

from .cache import tensor_sha256
from .encoder import OfficialUMT5Encoder, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("prompts.json"))
    parser.add_argument("--mode", choices=["cpu-bf16", "stream-cpu-fp32", "stream-mps-fp32"], required=True)
    args = parser.parse_args()
    if any((args.output / name).exists() for name in ["manifest.json", "embeddings.safetensors"]):
        parser.error("Cache output files must be new")
    prompt_manifest = json.loads(args.prompts.read_text())
    prompts = prompt_manifest["prompts"]
    if not 1 <= len(prompts) <= 8 or len({p["id"] for p in prompts}) != len(prompts):
        parser.error("Expected one through eight unique prompt ids")
    if any(not isinstance(p["id"], str) or not p["id"].isascii() or not p["id"].replace("_", "").isalnum() for p in prompts):
        parser.error("Prompt ids must use ASCII letters, digits or underscores")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    encoder = OfficialUMT5Encoder(args.weights, mode=args.mode)
    records, tensors = [], {}
    for prompt in prompts:
        def progress(event):
            event.update({"prompt_id": prompt["id"], "elapsed_seconds": time.monotonic() - started,
                          "rss_bytes": psutil.Process().memory_info().rss})
            if args.mode == "stream-mps-fp32":
                event.update({"mps_allocated_bytes": torch.mps.current_allocated_memory(),
                              "mps_driver_bytes": torch.mps.driver_allocated_memory()})
            print(json.dumps(event, allow_nan=False), flush=True)
        tick = time.monotonic()
        embedding, token_ids, details = encoder.encode([prompt["text"]], progress=progress)
        tensors[prompt["id"]] = embedding
        records.append({"id": prompt["id"], "text": prompt["text"],
            "text_sha256": hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest(),
            "token_length": len(token_ids), "active_token_ids": token_ids.tolist(),
            "tensor_sha256": tensor_sha256(embedding), "encoding_seconds": time.monotonic() - tick, **details})
    descriptor, temporary = tempfile.mkstemp(prefix=".embeddings-", suffix=".safetensors", dir=args.output)
    os.close(descriptor)
    try:
        save_file(tensors, temporary, metadata={"encoder": "official Wan UMT5", "original_worldline_model": "false",
                  "revision": encoder.provenance["revision"]})
        os.link(temporary, args.output / "embeddings.safetensors")
    finally:
        Path(temporary).unlink(missing_ok=True)
    result = {"schema": "worldline-real-umt5-cache-v1", "status": "complete",
        "external_model": "Wan UMT5-XXL encoder", "original_worldline_model": False,
        "weights_and_tokenizer": encoder.provenance,
        "source_manifest": json.loads(Path(__file__).with_name("source-manifest.json").read_text()),
        "loader_sha256": sha256(Path(__file__).with_name("encoder.py")), "worker_sha256": sha256(__file__),
        "prompt_manifest_sha256": sha256(args.prompts),
        "embeddings_file_sha256": sha256(args.output / "embeddings.safetensors"),
        "compute_mode": args.mode, "source_dtype": "bfloat16", "input_padding_length": 512,
        "prompt_scope": prompt_manifest.get("scope"), "prompts": records,
        "runtime_versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "tokenizers", "ftfy", "sentencepiece", "safetensors", "psutil"]},
        "elapsed_seconds": time.monotonic() - started,
        "limits": "Real text embeddings only; no new text model, image/video quality, action or memory claim."}
    with (args.output / "manifest.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "complete", "prompt_count": len(records), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
