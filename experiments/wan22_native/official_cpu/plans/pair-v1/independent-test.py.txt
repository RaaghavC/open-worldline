# SPDX-License-Identifier: Apache-2.0
"""Independent tiny CPU checks; never open the official 5B weight files.

The oracle executes the pinned original model, retaining its forward equations.
Only its seven precision contexts and imported attention are replaced here.
No portable model/forward implementation is imported.
"""
import ast
import hashlib
import json
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from . import reference, streaming
from .vendor import attention


HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432"
CONFIG = dict(model_type="ti2v", patch_size=(1, 2, 2), text_len=8,
              in_dim=48, out_dim=48, dim=128, ffn_dim=192, freq_dim=16,
              text_dim=16, num_heads=1, num_layers=2, qk_norm=True,
              cross_attn_norm=True, eps=1e-6)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def oracle_attention(q, k, v, k_lens=None, **kwargs):
    """Native varlen semantics by independent per-item key truncation."""
    original = q.dtype
    cast = lambda x: x if x.dtype in (torch.float16, torch.bfloat16) else x.bfloat16()
    q, k, v = cast(q), cast(k), cast(v)
    q, k = q.to(v.dtype), k.to(v.dtype)
    rows = []
    with torch.autocast("cpu", enabled=False):
        for b in range(q.shape[0]):
            length = k.shape[1] if k_lens is None else int(k_lens[b])
            rows.append(torch.nn.functional.scaled_dot_product_attention(
                q[b].transpose(0, 1).unsqueeze(0),
                k[b, :length].transpose(0, 1).unsqueeze(0),
                v[b, :length].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0, is_causal=False)[0].transpose(0, 1))
    return torch.stack(rows).to(original)


def literal_oracle():
    """Independent context interpreter, not the product AST transformer."""
    raw = (HERE / "vendor/model.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA
    tree = ast.parse(raw)
    replacements = []

    class PrecisionCalls(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node.func) == "torch.amp.autocast":
                replacements.append(node.lineno)
                node.func = ast.copy_location(ast.Name("_native_context", ast.Load()), node.func)
            return node

        def visit_ImportFrom(self, node):
            if node.level == 1 and node.module == "attention":
                return ast.copy_location(ast.Assign(
                    [ast.Name("flash_attention", ast.Store())],
                    ast.Name("_oracle_attention", ast.Load())), node)
            return node

    def native_context(device, **kwargs):
        assert device == "cuda"
        assert kwargs in ({"enabled": False}, {"dtype": torch.float32})
        return torch.autocast("cpu", enabled=False)

    tree = ast.fix_missing_locations(PrecisionCalls().visit(tree))
    assert len(replacements) == 7
    name = __package__ + "._independent_literal"
    module = types.ModuleType(name)
    module.__dict__.update(_native_context=native_context, _oracle_attention=oracle_attention)
    sys.modules[name] = module
    exec(compile(tree, str(HERE / "vendor/model.py") + " [independent contexts]", "exec"), module.__dict__)
    return module


def fixture_models():
    torch.manual_seed(20260926)
    resident = literal_oracle().WanModel(**CONFIG).eval().requires_grad_(False)
    with torch.no_grad():
        resident.head.head.weight.normal_(0, 0.025)
        resident.head.head.bias.normal_(0, 0.01)
    state = {n: p.detach().clone() for n, p in resident.named_parameters()}
    meta, _ = reference.create_meta(CONFIG)
    return resident, meta, state


class MemorySource:
    def __init__(self, state):
        self.state = state
        self.calls = []
        self.return_refs = []

    def load(self, name, shape):
        assert tuple(shape) == tuple(self.state[name].shape)
        value = self.state[name].clone()
        self.calls.append(name)
        self.return_refs.append(weakref.ref(value))
        return value


def inputs():
    generator = torch.Generator().manual_seed(7231)
    latent = torch.randn(48, 3, 4, 6, generator=generator)
    times = torch.full((1, 20), 999, dtype=torch.int64)
    times[:, :6] = 0
    context = torch.randn(3, 16, generator=generator)
    return latent, times, context, 20


def run_resident(model, args, cache=False):
    latent, times, context, seq_len = args
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=cache):
        return model([latent], times, [context], seq_len)[0]


def assert_clean(model):
    assert all(p.device.type == "meta" and p.dtype == torch.float32 for p in model.parameters())
    for module in model.modules():
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


def test_complete_ast_diff_is_only_seven_native_precision_calls():
    original = ast.parse((HERE / "vendor/model.py").read_text())
    translated, report = reference.translated_source()
    changed = ast.parse(translated)

    def normalized(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.amp.autocast":
                node.args = [ast.Constant("reviewed-device-context")]
                node.keywords = []
        return ast.dump(tree, include_attributes=False)

    assert normalized(original) == normalized(changed)
    assert len(report["changed_calls"]) == 7
    assert report["original_source_sha256"] == ORIGINAL_SHA
    assert "portable" not in translated


def test_literal_complex_rope_axes_padding_and_dtype():
    module, _ = reference.native_module()
    oracle = literal_oracle()
    generator = torch.Generator().manual_seed(128)
    x = torch.randn(2, 22, 2, 128, generator=generator)
    grids = torch.tensor([[2, 3, 3], [3, 2, 3]])
    freqs = torch.cat([oracle.rope_params(1024, 44), oracle.rope_params(1024, 42),
                       oracle.rope_params(1024, 42)], dim=1)
    assert freqs.dtype == torch.complex128
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = module.rope_apply(x, grids, freqs)
        expected = oracle.rope_apply(x, grids, freqs)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.equal(actual[:, 18:], x[:, 18:])
    assert not torch.equal(actual[0, :18], actual[1, :18])
    torch.testing.assert_close(module.sinusoidal_embedding_1d(16, torch.tensor([0, 999])),
                               oracle.sinusoidal_embedding_1d(16, torch.tensor([0, 999])), rtol=0, atol=0)


def test_attention_matches_native_key_truncation_and_preserves_query_dtype():
    generator = torch.Generator().manual_seed(47)
    q = torch.randn(2, 7, 2, 16, generator=generator)
    k = torch.randn(2, 9, 2, 16, generator=generator)
    v = torch.randn(2, 9, 2, 16, generator=generator).bfloat16()
    lengths = torch.tensor([4, 9])
    actual = attention.flash_attention(q, k, v, k_lens=lengths)
    expected = oracle_attention(q, k, v, k_lens=lengths)
    assert actual.dtype == q.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    k[0, 4:] = 10000
    v[0, 4:] = -10000
    assert torch.equal(actual, attention.flash_attention(q, k, v, k_lens=lengths))
    assert not torch.equal(actual, attention.flash_attention(q, k, v))


def test_streamed_original_storage_matches_literal_resident_and_operation_dtypes():
    resident, meta, state = fixture_models()
    source = MemorySource(state)
    seen = {}
    observers = []
    wanted = {"patch_embedding", "time_embedding.0", "time_projection.1", "text_embedding.0",
              "blocks.0.self_attn.q", "blocks.0.self_attn.norm_q", "blocks.0.ffn.0", "head.head"}
    for name, module in meta.named_modules():
        if name in wanted:
            def observe(module, args, output, name=name):
                seen[name] = dict(dtype=output.dtype, cpu_autocast=torch.is_autocast_enabled("cpu"),
                                  cache=torch.is_autocast_cache_enabled(),
                                  weights=[p.dtype for p in module.parameters()])
            observers.append(module.register_forward_hook(observe))
    try:
        resident_seen = {}
        resident_observers = []
        for name, module in resident.named_modules():
            if name in wanted:
                def observe_resident(module, args, output, name=name):
                    resident_seen[name] = dict(dtype=output.dtype, cpu_autocast=torch.is_autocast_enabled("cpu"),
                                               cache=torch.is_autocast_cache_enabled(),
                                               weights=[p.dtype for p in module.parameters()])
                resident_observers.append(module.register_forward_hook(observe_resident))
        try:
            expected = run_resident(resident, inputs(), cache=False)
        finally:
            for handle in resident_observers:
                handle.remove()
        assert torch.equal(expected, run_resident(resident, inputs(), cache=True))
        with streaming.StreamedModules(meta, source) as stream:
            actual = reference.forward(meta, *inputs())
            complete = stream.completed()
            assert meta.freqs.device.type == "cpu" and meta.freqs.dtype == torch.complex128
            assert complete["all_parameters_meta_after_pass"]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert actual.abs().max() > 0.1
        assert set(source.calls) == set(state) and len(source.calls) == len(state)
        assert all(ref() is None for ref in source.return_refs)
        assert seen["patch_embedding"]["dtype"] == torch.bfloat16
        for name in ("text_embedding.0", "blocks.0.self_attn.q", "blocks.0.ffn.0"):
            assert seen[name]["dtype"] == torch.bfloat16 and seen[name]["cpu_autocast"]
        for name in ("time_embedding.0", "time_projection.1", "head.head"):
            assert seen[name]["dtype"] == torch.float32 and not seen[name]["cpu_autocast"]
        assert seen["blocks.0.self_attn.norm_q"]["dtype"] == torch.float32
        assert seen == resident_seen
        assert all(not row["cache"] for row in seen.values())
        assert all(all(dtype == torch.float32 for dtype in row["weights"]) for row in seen.values())
        for group in streaming.groups(meta):
            expected_bytes = sum(value.numel() * 4 for name, value in state.items() if name.startswith(group + "."))
            load = [row for row in stream.rows if row["event"] == "loaded" and row["group"] == group]
            assert len(load) == 1 and load[0]["live_parameter_bytes"] == expected_bytes
        # A fresh context must be able to execute another pass after complete eviction.
        with streaming.StreamedModules(meta, source) as second:
            different = list(inputs())
            different[2] = different[2].neg() + 0.5
            changed = reference.forward(meta, *different)
            second.completed()
        assert not torch.equal(actual, changed)
        torch.testing.assert_close(changed, run_resident(resident, different), atol=0, rtol=0)
        with streaming.StreamedModules(meta, source) as third:
            again = reference.forward(meta, *inputs())
            third.completed()
        assert torch.equal(actual, again)
    finally:
        for handle in observers:
            handle.remove()
    assert_clean(meta)


@pytest.mark.parametrize("failure", ["load", "native", "event"])
def test_partial_load_native_and_observer_failure_release_parameters(failure):
    _, meta, state = fixture_models()
    source = MemorySource(state)
    if failure == "load":
        original = source.load
        def failing_load(name, shape):
            if name.endswith("patch_embedding.bias"):
                raise RuntimeError("independent load sentinel")
            return original(name, shape)
        source.load = failing_load
    if failure == "native":
        def failed_native(*args, **kwargs):
            raise RuntimeError("independent native sentinel")
        meta.blocks[0].self_attn.forward = failed_native
    def event(row):
        if failure == "event" and row["event"] == "loaded" and row["group"] == "blocks.0":
            raise RuntimeError("independent event sentinel")
    with pytest.raises(RuntimeError, match="independent .* sentinel"):
        with streaming.StreamedModules(meta, source, event=event):
            reference.forward(meta, *inputs())
    assert_clean(meta)
    assert all(ref() is None for ref in source.return_refs)


def test_partial_hook_registration_failure_cleans_previously_registered_hooks(monkeypatch):
    _, meta, state = fixture_models()
    def fail(*args, **kwargs):
        raise RuntimeError("independent registration sentinel")
    monkeypatch.setattr(meta.time_embedding, "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="independent registration sentinel"):
        with streaming.StreamedModules(meta, MemorySource(state)):
            pytest.fail("Unreachable after registration error")
    assert_clean(meta)


def test_retained_parameter_owner_is_detected_not_counted_as_released():
    _, meta, state = fixture_models()
    retained = []
    def event(row):
        if row["event"] == "loaded" and row["group"] == "blocks.0":
            retained.append(next(meta.blocks[0].parameters()))
    try:
        with pytest.raises(RuntimeError, match="Evicted parameter owner remains alive"):
            with streaming.StreamedModules(meta, MemorySource(state), event=event):
                reference.forward(meta, *inputs())
        assert len(retained) == 1 and retained[0].device.type == "cpu"
        assert_clean(meta)
    finally:
        retained.clear()


def test_shard_copy_is_owned_and_plan_only_cannot_read_values(tmp_path):
    model, _ = reference.create_meta(CONFIG)
    values = {name: torch.full(parameter.shape, (i + 1) / 128, dtype=torch.float32)
              for i, (name, parameter) in enumerate(model.named_parameters())}
    shard = tmp_path / "tiny.safetensors"
    save_file(values, str(shard))
    index = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {name: shard.name for name in values}}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(CONFIG))
    provenance = dict(index_sha256=streaming.sha(index), config_sha256=streaming.sha(config),
                      weights=[dict(file=shard.name, bytes=shard.stat().st_size, sha256=streaming.sha(shard))])
    planned = streaming.ShardSource(tmp_path, model, provenance, verify_bytes=False)
    name = "head.head.weight"
    with pytest.raises(RuntimeError, match="Plan-only"):
        planned.load(name, values[name].shape)
    source = streaming.ShardSource(tmp_path, model, provenance)
    value = source.load(name, values[name].shape)
    before = value.clone()
    # Mutating this synthetic shard must not mutate the retained returned tensor.
    spec = streaming.header(shard)[name]
    import struct
    with shard.open("r+b") as handle:
        header_bytes = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_bytes + spec["data_offsets"][0])
        handle.write(struct.pack("<f", 19.25))
    assert torch.equal(value, before)
    assert source.records[name]["source_owner_released"]
    with pytest.raises(RuntimeError, match="changed between passes"):
        source.load(name, values[name].shape)


class StubbornChild:
    """A process stand-in that requires kill after refusing terminate."""
    def __init__(self):
        self.returncode = None
        self.terminated = 0
        self.killed = 0
        self.stdin = self

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1

    def kill(self):
        self.killed += 1

    def wait(self, timeout):
        import subprocess
        if not self.killed:
            raise subprocess.TimeoutExpired("independent process stand-in", timeout)
        self.returncode = -9
        return self.returncode

    def write(self, value):
        raise BrokenPipeError("independent handoff sentinel")


def test_parent_watchdog_retains_failure_and_kills_stubborn_child(tmp_path):
    from . import run
    child = StubbornChild()
    with pytest.raises(RuntimeError, match="Parent resource guard"):
        run.supervise(child, tmp_path, deadline=0.0, interval=0)
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["status"] == "failed" and terminal["exit_code"] == -9
    assert child.terminated == child.killed == 1
    assert (tmp_path / "watchdog-stop.json").is_file()
    assert len((tmp_path / "memory.jsonl").read_text().splitlines()) == 1


def test_pre_supervisor_handoff_failure_also_kills_stubborn_child(tmp_path, monkeypatch):
    from . import run
    child = StubbornChild()
    output = tmp_path / "run"
    args = ["independent-test", "--execute"]
    for name in ("weights", "pair-directory", "text-directory", "cpu-report", "independent-report"):
        args.extend(["--" + name, str(tmp_path / name)])
    args.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(run, "preflight", lambda *args, **kwargs: "0" * 64)
    monkeypatch.setattr(run, "load_inputs", lambda *args: ({}, {}, {"synthetic": True}))
    monkeypatch.setattr(run, "snapshot", lambda *args: {})
    monkeypatch.setattr(run, "create_meta", lambda: (torch.nn.Linear(1, 1), {}))
    monkeypatch.setattr(run, "ShardSource", lambda *args, **kwargs: None)
    monkeypatch.setattr(run.shutil, "copyfile", lambda source, destination: Path(destination).write_bytes(b"fixture"))
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(BrokenPipeError, match="independent handoff sentinel"):
        run.main()
    assert child.terminated == child.killed == 1
    assert child.returncode == -9
    report = json.loads((output / "metrics.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "BrokenPipeError"
    assert not report["model_execution"]
