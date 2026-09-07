# SPDX-License-Identifier: Apache-2.0
"""Load small, verified real text contexts without importing the text encoder."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def tensor_sha256(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_context(directory, prompt_id, *, expected_text=None, device="cpu", dtype=None):
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "worldline-real-umt5-cache-v1" or manifest.get("status") != "complete":
        raise ValueError("Missing completed real UMT5 cache")
    records = [entry for entry in manifest["prompts"] if entry["id"] == prompt_id]
    if len(records) != 1:
        raise ValueError("Prompt id missing or duplicated")
    record = records[0]
    if expected_text is not None and record["text"] != expected_text:
        raise ValueError("Cache text differs from requested prompt")
    if hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != record["text_sha256"]:
        raise ValueError("Prompt text hash mismatch")
    path = root / "embeddings.safetensors"
    if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["embeddings_file_sha256"]:
        raise ValueError("Cache file integrity mismatch")
    values = load_file(str(path), device="cpu")
    value = values[prompt_id]
    if type(record["token_length"]) is not int or value.shape != (record["token_length"], 4096) or not 1 <= value.shape[0] <= 512:
        raise ValueError("Invalid cached context shape")
    if not value.is_floating_point():
        raise ValueError("Context must contain floating-point hidden states")
    if not torch.isfinite(value).all().item() or not value.abs().max().item() > 0:
        raise ValueError("Nonfinite or all-zero text context is not accepted")
    if tensor_sha256(value) != record["tensor_sha256"]:
        raise ValueError("Context tensor hash mismatch")
    value = value.to(device=device, dtype=dtype or value.dtype)
    if not torch.isfinite(value).all().item():
        raise FloatingPointError("Context conversion overflowed the requested dtype")
    return value
