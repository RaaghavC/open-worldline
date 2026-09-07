# SPDX-License-Identifier: Apache-2.0
"""Meta construction and bounded one-tensor-at-a-time official weight loading."""
import gc
import hashlib
import json
from pathlib import Path
import struct

import torch
from torch import nn
from safetensors import safe_open

from .portable import create_model, declared_dtypes, PARAMETER_COUNTS, storage_report


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def tensor_sha256(value):
    if value.device.type != "cpu":
        raise ValueError("Tensor hashing accepts an explicit CPU copy only")
    array = value.detach().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def confined(root, name):
    relative = Path(name)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Weight file must stay inside the declared directory")
    return path


def read_header(path):
    """Read only the safetensors JSON header, never parameter storage."""
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if not 2 <= size <= 16 * 2**20:
            raise ValueError("Unexpected safetensors header size")
        content = stream.read(size)
        if len(content) != size:
            raise ValueError("Truncated safetensors header")
    value = json.loads(content)
    return {key: item for key, item in value.items() if key != "__metadata__"}


def inspect_index(model, directory, index):
    root = Path(directory).resolve()
    mapping = index["weight_map"]
    expected = dict(model.named_parameters())
    if set(mapping) != set(expected):
        raise ValueError("Pinned index keys differ from the exact model parameters")
    headers = {}
    for name in sorted(set(mapping.values())):
        headers[name] = read_header(confined(root, name))
    seen = set()
    for shard, header in headers.items():
        for name, value in header.items():
            if name in seen or mapping.get(name) != shard:
                raise ValueError("Unexpected, duplicate or wrongly indexed tensor")
            seen.add(name)
            if value["shape"] != list(expected[name].shape) or value["dtype"] != "F32":
                raise ValueError("Original tensor shape or FP32 dtype differs from official model")
    if seen != set(expected):
        raise ValueError("Shard headers omit indexed tensors")
    return {"parameter_count": sum(value.numel() for value in expected.values()),
            "tensor_count": len(expected), "shards": sorted(headers), "original_dtype": "F32"}


def stream_parameters(model, directory, index, artifacts, *, device="cpu", check=None):
    """Generic strict stream used by tiny CPU fixtures and the pinned core loader.

    One source tensor, a CPU conversion reference, the destination parameter and
    (on MPS) a temporary CPU verification copy coexist. CPU FP32 values are also
    copied, so final parameters cannot retain their source mmap. No full state
    dict exists; each source mapping is released before the next tensor.
    """
    root, device = Path(directory).resolve(), torch.device(device)
    if device.type not in ("cpu", "mps") or any(p.device.type != "meta" for p in model.parameters()):
        raise ValueError("A wholly meta model and explicit CPU/MPS target are required")
    info = inspect_index(model, root, index)
    if set(artifacts) != set(info["shards"]):
        raise ValueError("Artifact verification does not cover every indexed shard")
    for name, artifact in artifacts.items():
        if check:
            check()
        path = confined(root, name)
        if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise ValueError("Source shard differs from its pinned size/hash")
    dtypes = declared_dtypes(model, model._storage_policy)
    records = {}
    for name, old in list(model.named_parameters()):
        if check:
            check()
        shard = index["weight_map"][name]
        with safe_open(str(confined(root, shard)), framework="pt", device="cpu") as handle:
            source = handle.get_tensor(name)
            if source.dtype != torch.float32 or source.shape != old.shape or not torch.isfinite(source).all():
                raise ValueError("Source tensor differs or contains nonfinite values")
            original_hash = tensor_sha256(source)
            reference = source.to(dtype=dtypes[name], copy=True)
            if not torch.isfinite(reference).all():
                raise FloatingPointError("Declared precision conversion produced nonfinite weights")
            loaded = reference.to(device=device)
            if device.type == "mps":
                torch.mps.synchronize()
            verified = loaded.detach().cpu()
            if not torch.equal(verified, reference):
                raise RuntimeError("Loaded tensor differs from the declared source conversion")
            parent, field = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), field, nn.Parameter(loaded, requires_grad=False))
            records[name] = {"shape": list(source.shape), "original_dtype": "float32", "original_sha256": original_hash,
                "loaded_dtype": str(dtypes[name]).split(".")[-1], "loaded_sha256": tensor_sha256(verified),
                "converted_values_exact": True, "shard": shard}
            del source, reference, loaded, verified
        del handle
        if check:
            check()
    gc.collect()
    if any(p.device.type != device.type or (device.index is not None and p.device.index != device.index)
           or p.dtype != dtypes[name] for name, p in model.named_parameters()):
        raise RuntimeError("Final parameter device/dtype differs")
    if not torch.count_nonzero(model.head.head.weight).item():
        raise ValueError("Loaded output head is zero; this cannot establish a pretrained forward")
    return dict(info, storage=storage_report(model), tensors=records, nonzero_output_head=True,
                conversion="Each originalFP32 tensor converted directly to its declared dtype, then transferred; verified exactly",
                full_state_dict_materialized=False)


def load_core(directory, *, device="cpu", check=None):
    """Load only the pinned native5B transformer; never fetch or open VAE/text weights."""
    package = Path(__file__).parent
    provenance = json.loads((package / "provenance.json").read_text())
    for relative, expected in provenance["official_sources"].items():
        if sha256(package / relative) != expected["sha256"]:
            raise ValueError("Retained official source changed")
    root = Path(directory).resolve()
    index_path = confined(root, "diffusion_pytorch_model.safetensors.index.json")
    config_path = confined(root, "config.json")
    if sha256(index_path) != provenance["index_sha256"] or sha256(config_path) != provenance["config_sha256"]:
        raise ValueError("Official index/config hash differs")
    if sha256(package / "config.json") != provenance["config_sha256"]:
        raise ValueError("Portable constructor configuration changed")
    model = create_model(device="meta")
    if storage_report(model)["parameters"] != PARAMETER_COUNTS:
        raise ValueError("Native5B selective precision parameter count differs")
    artifacts = {row["file"]: row for row in provenance["weights"]}
    report = stream_parameters(model, root, json.loads(index_path.read_text()), artifacts, device=device, check=check)
    report.update(repository=provenance["repository"], revision=provenance["weight_revision"],
                  code_commit=provenance["code_commit"], input_files=artifacts,
                  index_sha256=provenance["index_sha256"], config_sha256=provenance["config_sha256"])
    return model, report
