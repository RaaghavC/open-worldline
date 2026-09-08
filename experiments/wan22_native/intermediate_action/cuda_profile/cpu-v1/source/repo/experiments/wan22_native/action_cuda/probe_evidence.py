# SPDX-License-Identifier: Apache-2.0
"""Read-only input/source/admission gates for the separate CUDA numerical probe."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open

from ..cuda_reference.guards import LIMITS
from ..official_cpu.streaming import sha, tensor_sha
from .bridge import PROFILES


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCOPE = "two-update-numerical-feasibility-only"
BRIDGE_REPORTS = {
    "cpu-results/v1/report.json": "7a8fe9c9335f1db6ee1aea5c4ecb9b32c56491575aa0639886e41dd06aa82ee9",
    "cpu-results/independent-v1/report.json": "4b9e72c1d5b1df59c959aee6558988a0790c503943ebaa4dcf6b78f972cdc4fd",
}


def relative_file(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("A bounded relative file path is required")
    path = root
    for part in Path(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink artifacts are not accepted")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Expected regular artifact is missing")
    return path


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read_json(path, *, maximum_bytes=16*2**20):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum_bytes:
        raise ValueError("Expected a bounded regular JSON artifact")
    value = json.loads(path.read_text(), object_pairs_hook=_pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _module_file(name):
    candidate = REPO.joinpath(*name.split("."))
    for file in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if file.is_file() and file.resolve().is_relative_to(REPO):
            return file
    return None


def source_paths(*, independent=False):
    """Include transitive project imports without importing or executing them.

    External package versions are captured separately. Literal native resource
    files are explicit because AST imports cannot find a JSON/license read.
    """
    entries = [HERE / name for name in ("__init__.py", "bridge.py", "probe_math.py", "probe_evidence.py",
                                        "probe.py", "data.py", "test_probe.py", "test_independent.py")]
    if independent:
        entries.append(HERE / "test_probe_independent.py")
    pending = list(entries)
    found = {}
    while pending:
        path = pending.pop()
        name = str(path.relative_to(REPO))
        if name in found:
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError("Required source is missing: " + name)
        found[name] = path
        parts = path.relative_to(REPO).with_suffix("").parts
        package = parts[:-1]
        for size in range(1, len(package)+1):
            init = REPO.joinpath(*package[:size], "__init__.py")
            if init.is_file():
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = package[:len(package)-node.level+1]
                    base = ".".join((*prefix, *((node.module or "").split(".") if node.module else ())))
                else:
                    base = node.module or ""
                names = [base] + [base + "." + alias.name for alias in node.names]
            for module in names:
                file = _module_file(module)
                if file is not None:
                    pending.append(file)
    native = HERE.parent / "cuda_reference"
    for name in ("config.json", "expected-weights.json", "upstream-provenance.json", "codec-source.json",
                 "requirements.txt", "LICENSE-APACHE-2.0.txt", "vendor/vae2_2.py"):
        path = native / name
        found[str(path.relative_to(REPO))] = path
    text_reuse = HERE.parent / "text-reuse.json"
    found[str(text_reuse.relative_to(REPO))] = text_reuse
    for name in BRIDGE_REPORTS:
        path = HERE/name
        found[str(path.relative_to(REPO))] = path
    return dict(sorted(found.items()))


def source_hashes(*, independent=False):
    return {name: sha(path) for name, path in source_paths(independent=independent).items()}


def snapshot(out):
    mapping = source_hashes()
    for name, path in source_paths().items():
        target = Path(out) / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        if sha(target) != mapping[name]:
            raise RuntimeError("Source changed during snapshot")
    return mapping


def validate_sources(out, mapping):
    if mapping != source_hashes():
        raise ValueError("Prepared source graph differs from the current probe")
    for name, digest in mapping.items():
        if sha(relative_file(Path(out)/"source", name)) != digest:
            raise ValueError("Retained source snapshot differs: " + name)


def bridge_reviews():
    results = {}
    for name, digest in BRIDGE_REPORTS.items():
        path = relative_file(HERE, name)
        report = read_json(path)
        if sha(path) != digest or report.get("status") != "passed" or report.get("sources_unchanged") is not True:
            raise ValueError("Pinned completed bridge CPU review is required")
        for source, expected in report["source_sha256"].items():
            if sha(relative_file(REPO, source)) != expected:
                raise ValueError("Bridge source changed since CPU review")
        results[name] = digest
    return results


def review(path, *, independent=False):
    value = read_json(path)
    if (value.get("status") != "passed" or type(value.get("tests")) is not int or value["tests"] < 4
            or value.get("sources_unchanged") is not True
            or any(type(value.get(key)) is not int or value[key] != 0
                   for key in ("pytest_exit_code", "failures", "errors", "skipped"))
            or value.get("source_sha256") != source_hashes(independent=independent)):
        raise ValueError("Completed current source-bound numerical-probe review required")
    if independent and value.get("independent_review") is not True:
        raise ValueError("Separate independent numerical-probe review required")
    return sha(path)


def tensors(path, expected, expected_hash):
    """Validate all names/dtypes/shapes before materializing any CPU tensor."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64*2**20 or sha(path) != expected_hash:
        raise ValueError("Bounded saved tensor artifact differs")
    result = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("Unexpected saved tensor keys")
        for name, (shape, dtype) in expected.items():
            view = handle.get_slice(name)
            if view.get_shape() != list(shape) or view.get_dtype() != dtype:
                raise ValueError("Saved tensor header differs: " + name)
        for name in expected:
            value = handle.get_tensor(name)
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("Nonfinite saved tensor")
            result[name] = value
    return result


def load_positive(directory):
    """Use only the exact retained genuine Atrium embedding, without negatives."""
    from ..official_cpu.inputs import TEXT_PINS
    root = Path(directory)
    for name, digest in TEXT_PINS.items():
        if sha(relative_file(root, name)) != digest:
            raise ValueError("Pinned genuine text cache changed")
    reuse = read_json(HERE.parent / "text-reuse.json")
    if (reuse.get("status") != "verified" or reuse.get("text_cache_manifest_sha256") != TEXT_PINS["manifest.json"]
            or reuse.get("text_embeddings_sha256") != TEXT_PINS["embeddings.safetensors"]):
        raise ValueError("Native text-reuse evidence differs")
    with safe_open(root/"embeddings.safetensors", framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"atrium", "native_negative"}:
            raise ValueError("Pinned text artifact keys differ")
        view = handle.get_slice("atrium")
        if view.get_shape() != [25, 4096] or view.get_dtype() != "F32":
            raise ValueError("Genuine Atrium context shape or dtype differs")
        positive = handle.get_tensor("atrium")
    if not torch.isfinite(positive).all():
        raise ValueError("Genuine positive context must be finite")
    return positive, {"text_files": dict(TEXT_PINS), "reuse_sha256": sha(HERE.parent/"text-reuse.json"),
                      "text_id": "atrium", "positive_text": reuse["positive_text"],
                      "positive_text_sha256": reuse["positive_sha256"],
                      "positive_tensor_sha256": tensor_sha(positive),
                      "negative_tensor_materialized": False, "negative_context_used": False}


def load_cache(directory, profile):
    """Require the full current guarded native CUDA cache, never an MPS cache."""
    from . import cache_run, data
    root = Path(directory)
    parent = read_json(relative_file(root, "metrics.json"))
    current = cache_run.sources()
    if (parent.get("schema") != cache_run.SCHEMA or parent.get("mode") != "cache"
            or parent.get("status") != "passed" or parent.get("model_execution") is not True
            or parent.get("profile") != profile or parent.get("source_sha256") != current
            or parent.get("inputs_unchanged") is not True or parent.get("sources_unchanged") is not True):
        raise ValueError("A completed current-source native CUDA cache parent is required")
    worker = cache_run.validate_completed(root, parent)
    if (sha(relative_file(root, "worker/metrics.json")) != parent.get("worker_metrics_sha256")
            or sha(relative_file(root, "result/completion.json")) != parent.get("result_completion_sha256")
            or worker.get("hardware") != parent.get("hardware")):
        raise ValueError("Completed cache worker/completion/hardware binding differs")
    for name, digest in current.items():
        if sha(relative_file(root/"source", name)) != digest:
            raise ValueError("Cache source snapshot differs")
    windows, records = {}, {}
    for arm, start in data.SELECTION:
        identity = f"{arm}-{start:04d}"
        values, provenance = data.read_window(root/"result", identity)
        if set(values) != {"target", "observation", "commands"}:
            raise ValueError("Only the three declared cached tensors are accepted")
        wanted = (1, *PROFILES[profile]["shape"])
        shapes = {"target": wanted, "observation": (1, 48, 1, *wanted[-2:]), "commands": (1, 16, 6)}
        for name, value in values.items():
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32
                    or tuple(value.shape) != shapes[name] or not torch.isfinite(value).all()):
                raise ValueError("Native cached tensor differs: " + name)
        from .probe_math import cross_length_prefix
        prefix_check = cross_length_prefix(values["target"][:, :, :1], values["observation"])
        windows[identity] = values
        records[identity] = {"provenance": provenance,
                             "cross_length_prefix": {key: prefix_check[key] for key in ("passed", "exact_equal", "tolerances")},
                             "tensor_sha256": {name: tensor_sha(value) for name, value in values.items()}}
    expected = {f"{arm}-{start:04d}" for start in (0, 8, 32, 49) for arm in ("closed", "open")}
    if set(windows) != expected:
        raise ValueError("Exactly the eight declared windows are required")
    if not torch.equal(windows["closed-0000"]["observation"], windows["open-0000"]["observation"]):
        raise ValueError("Start0 interventions must share one exact independent observation")
    return windows, {"profile": profile, "cache_parent_sha256": sha(root/"metrics.json"),
                     "cache_worker_sha256": sha(root/"worker/metrics.json"),
                     "cache_terminal_sha256": sha(root/"terminal.json"),
                     "cache_completion_sha256": sha(root/"result/completion.json"),
                     "cache_manifest_sha256": sha(root/"result/manifest.json"),
                     "cache_source_sha256": current, "hardware": parent["hardware"], "windows": records}


def validate_admission(path, prepared_directory, plan, expected_gpu):
    path = Path(path)
    value = read_json(path)
    if (value.get("schema") != "worldline-wan22-action-cuda-probe-admission-v1"
            or value.get("decision") != "admit" or value.get("issued_by") != "parent-agent"
            or value.get("scope") != SCOPE or value.get("profile") != plan["profile"]
            or value.get("plan_sha256") != sha(Path(prepared_directory)/"plan.json")
            or value.get("source_sha256") != plan["source_sha256"]
            or value.get("input_identity") != plan["input_identity"]
            or value.get("expected_gpu") != expected_gpu or value.get("limits") != LIMITS
            or value.get("image_generation_admitted") is not False
            or value.get("fixed16_admitted") is not False
            or not isinstance(value.get("visual_review"), str) or not value["visual_review"].strip()):
        raise ValueError("Exact source/input/profile/plan-bound numerical-only parent admission required")
    if value.get("foundation_diagnostic_visual_status") not in ("passed", "failed"):
        raise ValueError("Record the foundation diagnostic visual status explicitly")
    if value.get("mps_visual_status") != "failed":
        raise ValueError("Preserve the historical failed MPS visual result")
    audited = value.get("completed_audits")
    evidence = value.get("visual_evidence")
    if not isinstance(audited, list) or not audited or not isinstance(evidence, list) or not evidence:
        raise ValueError("Completed audit records and separate visual evidence are required")
    for rows, audit in ((audited, True), (evidence, False)):
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"file", "sha256"}:
                raise ValueError("Each admission evidence item must bind one file and SHA256")
            file = Path(row["file"])
            if not file.is_absolute():
                file = path.parent/file
            if file.is_symlink() or not file.is_file() or sha(file) != row["sha256"]:
                raise ValueError("Admission evidence differs")
            if audit and read_json(file).get("status") != "passed":
                raise ValueError("Admission audit must actually report passed")
    return {"admission_sha256": sha(path), "scope": SCOPE,
            "foundation_diagnostic_visual_status": value["foundation_diagnostic_visual_status"],
            "visual_review": value["visual_review"], "quality_admitted": False,
            "completed_audits": audited, "visual_evidence": evidence}
