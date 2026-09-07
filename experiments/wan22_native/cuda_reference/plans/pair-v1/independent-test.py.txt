# SPDX-License-Identifier: Apache-2.0
"""CPU-only protocol checks. No CUDA calls, cloud access or real model loading."""
import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from . import evidence, guards, native, sampling
from ..official_cpu import inputs as retained_inputs
from ..official_cpu.streaming import tensor_sha


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


@pytest.fixture(autouse=True)
def bounded_cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent tests must not access CUDA")
    for name in ("is_available", "device_count", "get_device_properties", "mem_get_info",
                 "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    yield
    torch.set_num_threads(previous)


def real_retained_inputs():
    return retained_inputs.load_inputs(
        REPO / "experiments/wan22_native/core-results/cpu-pair-v1",
        REPO / "experiments/wan_adapter/text_cache/native-results")


def synthetic_inputs():
    generator = torch.Generator().manual_seed(76541)
    noise = torch.randn(sampling.SHAPE, generator=generator)
    observation = torch.randn(1, 48, 1, 18, 32, generator=generator)
    initial = noise.clone()
    initial[:, :1] = observation[0]
    times = torch.full((1, 720), 999, dtype=torch.int64)
    times[:, :144] = 0
    return dict(initial_noise=noise, observation=observation, initial_latent=initial,
                token_times=times), {"atrium": torch.tensor([[0.75]]),
                                   "native_negative": torch.tensor([[-0.5]])}


def test_exact_upstream_sources_and_retained_conditioning_contract():
    native.verify_sources()
    expected = {
        "vendor/model.py": "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432",
        "vendor/attention.py": "23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc",
        "vendor/fm_solvers_unipc.py": "0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
    tree = ast.parse((HERE / "native.py").read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "vendor.model" in imports and "vendor" in imports
    assert not any("portable" in name or "reference" in name for name in imports if name)
    values, contexts, identity = real_retained_inputs()
    assert set(values) == {"initial_noise", "initial_latent", "observation", "token_times"}
    assert set(contexts) == {"atrium", "native_negative"}
    assert contexts["atrium"].shape == (25, 4096)
    assert contexts["native_negative"].shape == (126, 4096)
    assert all(value.dtype == torch.float32 for value in contexts.values())
    assert torch.equal(values["initial_latent"][:, :1], values["observation"][0])
    assert torch.equal(values["initial_latent"][:, 1:], values["initial_noise"][:, 1:])
    assert torch.equal(values["token_times"], sampling.times_at(torch.tensor(999)))
    assert not any(identity[key] for key in ("noise_regenerated", "actions_read", "future_target_read"))


def test_pair_clones_history_and_times_preserves_context_order_and_guidance():
    values, contexts = synthetic_inputs()
    original = {name: value.clone() for name, value in values.items()}
    calls, partial = [], []
    def prediction(x, times, context):
        calls.append((x.clone(), times.clone(), context))
        assert torch.equal(x[:, :1], values["observation"][0])
        # An adverse prediction callback must not corrupt the next CFG input.
        x.fill_(123)
        times.fill_(4)
        return torch.full(sampling.SHAPE, float(context.item()), dtype=torch.float32)
    output = sampling.pair(prediction, values["initial_latent"], values["token_times"],
                           contexts["atrium"], contexts["native_negative"], values["observation"],
                           lambda label, outputs: partial.append((label, set(outputs))))
    assert len(calls) == 2
    assert torch.equal(calls[0][0], calls[1][0]) and torch.equal(calls[0][1], calls[1][1])
    assert calls[0][2] is contexts["atrium"] and calls[1][2] is contexts["native_negative"]
    assert partial == [("positive", {"positive_velocity"}),
                       ("negative", {"positive_velocity", "negative_velocity"})]
    assert torch.equal(output["guided_velocity"], torch.full(sampling.SHAPE, 5.75))
    assert all(torch.equal(values[name], value) for name, value in original.items())


def test_fifty_step_loop_matches_separate_literal_solver_and_all_prefix_times():
    values, contexts = synthetic_inputs()
    original = {name: value.clone() for name, value in values.items()}
    spec = importlib.util.spec_from_file_location(
        "independent_literal_cuda_solver", HERE / "vendor/fm_solvers_unipc.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    oracle = module.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    oracle.set_timesteps(50, device="cpu", shift=5.0)
    product = sampling.scheduler()
    assert torch.equal(product.timesteps, oracle.timesteps)
    assert torch.equal(product.sigmas, oracle.sigmas)
    assert len(product.timesteps) == 50 and len(product.sigmas) == 51
    assert int(product.timesteps[0]) == 999 and float(product.sigmas[-1]) == 0
    calls, events = [], []
    def nonlinear(x, times, context):
        return (x.tanh() * 0.13 + float(context.item()) * 0.02 + float(times[0, -1]) * 0.00003).float()
    def observed(x, times, context):
        step = len(calls) // 2
        assert torch.equal(x[:, :1], values["observation"][0])
        assert times.shape == (1, 720) and times.dtype == torch.int64
        assert torch.count_nonzero(times[:, :144]) == 0
        assert torch.equal(times[:, 144:], torch.full((1, 576), int(oracle.timesteps[step]), dtype=torch.int64))
        calls.append((tensor_sha(x), tensor_sha(times), float(context.item())))
        return nonlinear(x, times, context)
    actual = sampling.sample(observed, values, contexts,
                             event=lambda i, t, x, outputs: events.append((i, int(t), x.clone())))
    expected = values["initial_latent"].clone()
    for i, timestep in enumerate(oracle.timesteps):
        times = torch.full((1, 720), int(timestep), dtype=torch.int64)
        times[:, :144] = 0
        expected[:, :1] = values["observation"][0]
        positive = nonlinear(expected.clone(), times.clone(), contexts["atrium"])
        negative = nonlinear(expected.clone(), times.clone(), contexts["native_negative"])
        guided = negative + 5.0 * (positive - negative)
        expected = oracle.step(guided.unsqueeze(0), timestep, expected.unsqueeze(0), return_dict=False)[0][0]
        expected[:, :1] = values["observation"][0]
        assert events[i][:2] == (i, int(timestep))
        assert torch.equal(events[i][2], expected)
    assert len(calls) == 100 and len(events) == 50
    for i in range(0, 100, 2):
        assert calls[i][:2] == calls[i + 1][:2]
        assert calls[i][2] == 0.75 and calls[i + 1][2] == -0.5
    assert torch.equal(actual, expected)
    assert all(torch.equal(values[name], value) for name, value in original.items())


@pytest.mark.parametrize("field", ["initial_noise", "token_times"])
def test_wrong_initial_evidence_rejected_before_prediction(field):
    values, contexts = synthetic_inputs()
    if field == "initial_noise": values[field][0, 1, 0, 0] += 1
    else: values[field][0, 144] -= 1
    def forbidden(*args): pytest.fail("Prediction must not start on mismatched evidence")
    with pytest.raises(ValueError, match="Saved initial"):
        sampling.sample(forbidden, values, contexts)


def test_failed_negative_prediction_retains_only_completed_positive():
    values, contexts = synthetic_inputs()
    partial = []
    def prediction(x, t, context):
        if context is contexts["native_negative"]: raise RuntimeError("independent negative failure")
        return torch.ones(sampling.SHAPE)
    with pytest.raises(RuntimeError, match="independent negative failure"):
        sampling.pair(prediction, values["initial_latent"], values["token_times"],
                      contexts["atrium"], contexts["native_negative"], values["observation"],
                      event=lambda label, outputs: partial.append((label, set(outputs))))
    assert partial == [("positive", {"positive_velocity"})]


def hardware_record():
    return dict(name="NVIDIA A100 80GB PCIe", total_memory_bytes=80 * 2**30,
                capability=[8, 0], bf16_supported=True, torch="2.5.1+cu124", cuda="12.4",
                flash_attn="2.7.4.post1", flash_attention_2_available=True,
                flash_attention_3_available=False)


def test_fixed_hardware_and_memory_caps_reject_weaker_or_unreviewed_modes():
    valid = hardware_record()
    assert guards.validate_hardware(valid, valid["name"]) == valid
    for key, wrong in [("total_memory_bytes", 69 * 2**30), ("bf16_supported", False),
                       ("capability", [7, 5]), ("flash_attention_2_available", False),
                       ("flash_attention_3_available", True), ("torch", "2.6.0+cu124"),
                       ("cuda", "12.1"), ("flash_attn", "2.7.3")]:
        row = copy.deepcopy(valid); row[key] = wrong
        with pytest.raises(ValueError): guards.validate_hardware(row, valid["name"])
    with pytest.raises(ValueError): guards.validate_hardware(valid, "NVIDIA H100 80GB HBM3")
    valid_memory = dict(host_rss_bytes=2**30, cuda_reserved_bytes=2**30,
                        host_available_bytes=16 * 2**30, cuda_available_bytes=16 * 2**30)
    guards.check_sample(valid_memory, 900.0, now=1.0)
    for key, value in [("host_rss_bytes", 48 * 2**30 + 1), ("cuda_reserved_bytes", 60 * 2**30 + 1),
                       ("host_available_bytes", 8 * 2**30 - 1), ("cuda_available_bytes", 8 * 2**30 - 1)]:
        row = dict(valid_memory); row[key] = value
        with pytest.raises(RuntimeError): guards.check_sample(row, 900.0, now=1.0)
    for deadline in (float("nan"), float("inf"), 1.0):
        with pytest.raises(RuntimeError): guards.check_sample(valid_memory, deadline, now=1.0)


def test_cuda_verification_copy_uses_cpu_hashing_without_cuda_execution(monkeypatch):
    """825 scalar stand-ins test ownership plumbing, not native tensor values."""
    copies, refs = [], []
    real = torch.tensor([0.125], dtype=torch.float32)
    digest = tensor_sha(real)
    class CudaStandIn:
        dtype = torch.float32
        device = types.SimpleNamespace(type="cuda")
        def detach(self): return self
        def cpu(self):
            value = real.clone(); copies.append(True); refs.append(weakref.ref(value)); return value
        def numel(self): return 1
        def element_size(self): return 4
    class OriginalStandIn:
        def to(self, *, device, dtype):
            assert device == "cuda:0" and dtype == torch.float32
            return CudaStandIn()
    class Model:
        def __init__(self):
            self.group = types.SimpleNamespace(**{f"p{i}": torch.empty(1, device="meta") for i in range(825)})
        def named_parameters(self): return [(f"group.p{i}", getattr(self.group, f"p{i}")) for i in range(825)]
        def parameters(self): return [p for _, p in self.named_parameters()]
        def get_submodule(self, name): assert name == "group"; return self.group
    class Source:
        def __init__(self, *args, **kwargs): self.records = {}
        def load(self, name, shape):
            assert tuple(shape) == (1,)
            self.records[name] = dict(source_sha256=digest)
            return OriginalStandIn()
    monkeypatch.setattr(native, "meta_model", Model)
    monkeypatch.setattr(native, "ShardSource", Source)
    monkeypatch.setattr(native.torch.nn, "Parameter", lambda value, requires_grad: value)
    model, record = native.load_model("not-a-real-weight-directory", lambda: None)
    assert record["tensor_count"] == 825 and record["parameter_bytes"] == 3300
    assert len(copies) == 825 and all(ref() is None for ref in refs)
    assert all(p.device.type == "cuda" for p in model.parameters())


class StubbornChild:
    def __init__(self): self.returncode = None; self.terminated = 0; self.killed = 0
    def poll(self): return self.returncode
    def terminate(self): self.terminated += 1
    def kill(self): self.killed += 1
    def wait(self, timeout):
        if not self.killed: raise subprocess.TimeoutExpired("CPU process stand-in", timeout)
        self.returncode = -9; return -9


def test_parent_deadline_kills_stubborn_worker_and_retains_failure(tmp_path, monkeypatch):
    process = types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=2**20), children=lambda recursive: [])
    monkeypatch.setattr(guards.psutil, "Process", lambda: process)
    monkeypatch.setattr(guards.psutil, "virtual_memory", lambda: types.SimpleNamespace(available=16 * 2**30))
    child = StubbornChild()
    with pytest.raises(RuntimeError, match="Parent resource guard"):
        guards.supervise(child, tmp_path, deadline=0.0)
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["status"] == "failed" and terminal["exit_code"] == -9
    assert child.terminated == child.killed == 1
    assert (tmp_path / "watchdog-stop.json").is_file()


def synthetic_completed_pair(root):
    """Synthetic admission metadata and CPU arrays, explicitly not a GPU result."""
    _, _, identity = real_retained_inputs()
    mapping = evidence.sources()
    destination = root / "core/result"
    destination.mkdir(parents=True)
    for name, digest in mapping.items():
        path = root / "measured-source" / (name + ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(evidence.source_path(name), path)
        assert evidence.sha(path) == digest
    pairs = REPO / "experiments/wan22_native/core-results/cpu-pair-v1"
    text = REPO / "experiments/wan_adapter/text_cache/native-results"
    for source, name in [(pairs / "inputs.safetensors", "inputs.safetensors"),
                         (text / "embeddings.safetensors", "contexts.safetensors"),
                         (text / "manifest.json", "text-manifest.json")]:
        shutil.copyfile(source, root / name)
    # Only prior JSON identities are used, never the actual parameter values.
    retained = json.loads((REPO / "experiments/wan22_native/official_cpu/results/pair-v1/result/weights-used.json").read_text())
    rows = copy.deepcopy(retained["original_fp32_tensors"])
    for row in rows.values(): row["cuda_copy_exact"] = True
    weights = dict(convert_model_dtype=False, all_shards_verified=True, cuda_copy_exact=True,
                   tensor_count=825, parameter_count=4999787712, parameter_bytes=19999150848,
                   tensors=rows, synthetic_no_cuda=True)
    guards.atomic(destination / "weight-load.json", weights)
    positive = torch.full(sampling.SHAPE, 0.125)
    negative = torch.full(sampling.SHAPE, 0.25)
    save_file({"positive_velocity": positive}, str(destination / "completed-positive.safetensors"))
    save_file({"positive_velocity": positive, "negative_velocity": negative},
              str(destination / "completed-negative.safetensors"))
    save_file({"positive_velocity": positive, "negative_velocity": negative,
               "guided_velocity": negative + 5 * (positive - negative)}, str(destination / "outputs.safetensors"))
    terminal = {"status": "complete", "exit_code": 0}
    guards.atomic(root / "core/terminal.json", terminal)
    result = dict(status="passed", mode="pair", stage="core", source_sha256=mapping,
                  input_identity=identity, settings=copy.deepcopy(sampling.SETTINGS),
                  predictions=2, solver_updates=0, finite_outputs=True,
                  limits=copy.deepcopy(guards.LIMITS), precision=copy.deepcopy(evidence.PRECISION),
                  hardware=hardware_record(), input_tensors_unchanged=True,
                  load_seconds=10.0, pair_seconds=2.0, synthetic_no_cuda=True,
                  output_sha256={name: evidence.sha(destination / name) for name in (
                      "weight-load.json", "outputs.safetensors", "completed-positive.safetensors", "completed-negative.safetensors")})
    guards.atomic(destination / "metrics.json", result)
    parent = dict(status="passed", mode="pair", source_sha256=mapping, input_identity=identity,
                  limits=copy.deepcopy(guards.LIMITS), elapsed_seconds=15.0, synthetic_no_cuda=True,
                  core_report_sha256=evidence.sha(destination / "metrics.json"),
                  core_terminal_sha256=evidence.sha(root / "core/terminal.json"),
                  copied_input_sha256={name: evidence.sha(root / name) for name in (
                      "inputs.safetensors", "contexts.safetensors", "text-manifest.json")})
    guards.atomic(root / "metrics.json", parent)
    return mapping, identity, parent, result, weights


def test_pair_to_clip_gate_rejects_incomplete_precision_timing_and_hash_evidence(tmp_path):
    mapping, identity, parent, result, weights = synthetic_completed_pair(tmp_path)
    gpu = hardware_record()["name"]
    admitted = evidence.validate_pair(tmp_path, mapping, identity, gpu)
    assert admitted["estimated_seconds"] == 282.0
    assert admitted["decoder_cost_measured"] is False
    assert admitted["estimate_is_not_a_completion_guarantee"] is True
    assert admitted["unmeasured_decode_load_and_decode_allowance_seconds"] == 120
    for kind in ("no_outputs", "no_input_files", "wrong_precision", "missing_hashes",
                 "wrong_weight_count", "incomplete", "wrong_mode", "slow_pair", "nonfinite_time"):
        p, r, w = copy.deepcopy(parent), copy.deepcopy(result), copy.deepcopy(weights)
        if kind == "no_outputs": r["output_sha256"] = {}
        elif kind == "no_input_files": p["copied_input_sha256"] = {}
        elif kind == "wrong_precision": r["precision"]["convert_model_dtype"] = True
        elif kind == "missing_hashes":
            row = next(iter(w["tensors"].values()))
            del row["source_sha256"], row["loaded_sha256"]
        elif kind == "wrong_weight_count": w["tensor_count"] = 824
        elif kind == "incomplete": r["predictions"] = 1
        elif kind == "wrong_mode": p["mode"] = "clip"
        elif kind == "slow_pair": r["pair_seconds"] = 20.0; p["elapsed_seconds"] = 35.0
        elif kind == "nonfinite_time": r["pair_seconds"] = float("nan")
        guards.atomic(tmp_path / "core/result/weight-load.json", w)
        if "weight-load.json" in r["output_sha256"]:
            r["output_sha256"]["weight-load.json"] = evidence.sha(tmp_path / "core/result/weight-load.json")
        guards.atomic(tmp_path / "core/result/metrics.json", r)
        p["core_report_sha256"] = evidence.sha(tmp_path / "core/result/metrics.json")
        guards.atomic(tmp_path / "metrics.json", p)
        with pytest.raises((ValueError, KeyError), match="."):
            evidence.validate_pair(tmp_path, mapping, identity, gpu)


def test_pair_to_clip_gate_rejects_artifact_source_terminal_or_watchdog_changes(tmp_path):
    mapping, identity, _, _, _ = synthetic_completed_pair(tmp_path)
    gpu = hardware_record()["name"]
    for name in ("inputs.safetensors", "contexts.safetensors", "core/result/outputs.safetensors",
                 "measured-source/native.py.txt", "core/terminal.json"):
        path = tmp_path / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError): evidence.validate_pair(tmp_path, mapping, identity, gpu)
        path.write_bytes(original)
    (tmp_path / "core/result/watchdog-stop.json").write_text("{}")
    with pytest.raises(ValueError, match="stopped"):
        evidence.validate_pair(tmp_path, mapping, identity, gpu)


def test_source_review_gate_rejects_stale_sampling_hash(tmp_path):
    mapping = evidence.sources()
    mapping["test_independent.py"] = evidence.sha(HERE / "test_independent.py")
    path = tmp_path / "review.json"
    guards.atomic(path, dict(status="passed", tests=12, source_sha256=mapping))
    evidence.preflight(path, independent=True)
    mapping["sampling.py"] = "0" * 64
    guards.atomic(path, dict(status="passed", tests=12, source_sha256=mapping))
    with pytest.raises(ValueError, match="source-matching"):
        evidence.preflight(path, independent=True)
