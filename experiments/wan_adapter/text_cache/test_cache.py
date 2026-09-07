# SPDX-License-Identifier: Apache-2.0
"""Synthetic file fixtures exercise integrity checks, not encoder quality."""
import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from .cache import load_context, tensor_sha256


def write_fixture(path, tensor=None):
    path.mkdir()
    tensor = torch.ones(1, 4096) if tensor is None else tensor
    save_file({"unconditional": tensor}, str(path / "embeddings.safetensors"))
    manifest = {"schema": "worldline-real-umt5-cache-v1", "status": "complete",
        "embeddings_file_sha256": hashlib.sha256((path / "embeddings.safetensors").read_bytes()).hexdigest(),
        "prompts": [{"id": "unconditional", "text": "", "text_sha256": hashlib.sha256(b"").hexdigest(),
                     "token_length": 1, "tensor_sha256": tensor_sha256(tensor)}]}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_cache_roundtrip_and_expected_text(tmp_path):
    root = tmp_path / "cache"
    write_fixture(root)
    assert load_context(root, "unconditional", expected_text="").shape == (1, 4096)
    with pytest.raises(ValueError, match="differs"):
        load_context(root, "unconditional", expected_text="other text")


@pytest.mark.parametrize("bad", [0., float("nan"), float("inf")])
def test_nonfinite_and_zero_files_are_not_success(tmp_path, bad):
    root = tmp_path / "cache"
    write_fixture(root, torch.full((1, 4096), bad))
    with pytest.raises(ValueError, match="Nonfinite or all-zero"):
        load_context(root, "unconditional")


def test_integrity_and_dtype_conversion_overflow(tmp_path):
    root = tmp_path / "cache"
    write_fixture(root, torch.full((1, 4096), 100000.))
    with pytest.raises(FloatingPointError):
        load_context(root, "unconditional", dtype=torch.float16)
    with (root / "embeddings.safetensors").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="integrity"):
        load_context(root, "unconditional")


def test_tensor_hash_is_checked_separately(tmp_path):
    root = tmp_path / "cache"
    manifest = write_fixture(root)
    manifest["prompts"][0]["tensor_sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tensor hash"):
        load_context(root, "unconditional")
