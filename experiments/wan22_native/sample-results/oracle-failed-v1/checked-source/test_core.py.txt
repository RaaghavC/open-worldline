# SPDX-License-Identifier: Apache-2.0
"""CPU/meta checks for the attributed native5B port; no real-weight inference."""
import importlib.util
import json
from pathlib import Path
import sys
import struct

import pytest
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from . import portable as port
from .load_weights import inspect_index, read_header, sha256, stream_parameters

SMALL = dict(model_type="ti2v", in_dim=4, out_dim=4, dim=32, ffn_dim=64, freq_dim=8,
             text_dim=16, text_len=8, num_heads=4, num_layers=2)


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def reference_attention(q, k, v, k_lens=None, window_size=(-1, -1), **kwargs):
    # Independent explicit CPU oracle: match projection storage, cast Q/K to V,
    # mask keys, restore the query's original dtype. No CUDA call or autocast.
    dtype = q.dtype
    mask = None
    if k_lens is not None:
        mask = torch.arange(k.shape[1])[None, None, None, :] < k_lens[:, None, None, None]
    result = F.scaled_dot_product_attention(q.to(v.dtype).transpose(1, 2), k.to(v.dtype).transpose(1, 2),
        v.transpose(1, 2), attn_mask=mask, dropout_p=0., is_causal=False)
    return result.transpose(1, 2).contiguous().to(dtype)


def literal_reference(policy="fp32"):
    name = __package__ + ".vendor._cpu_test_reference"
    spec = importlib.util.spec_from_file_location(name, Path(port.__file__).parent / "vendor/model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.flash_attention = reference_attention
    torch.manual_seed(31)
    model = module.WanModel(**SMALL).eval().requires_grad_(False)
    torch.nn.init.normal_(model.head.head.weight, std=.1)
    if policy == "selective_bf16":
        # This oracle derives selected FP32 names from the literal class itself,
        # without importing the port's dtype selection or forward equations.
        fp32 = set()
        for name, child in model.named_modules():
            if isinstance(child, (module.WanRMSNorm, module.WanLayerNorm)):
                fp32.update(f"{name}.{key}" for key, _ in child.named_parameters(recurse=False))
        for name, parameter in list(model.named_parameters()):
            keep = name in fp32 or name.startswith(("head.", "time_embedding.", "time_projection.")) or name.endswith(".modulation")
            parent, key = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), key, torch.nn.Parameter(parameter.to(torch.float32 if keep else torch.bfloat16), requires_grad=False))
        for child in model.modules():
            if isinstance(child, (torch.nn.Linear, torch.nn.Conv3d)):
                child.register_forward_pre_hook(lambda layer, args: (args[0].to(layer.weight.dtype),))
    return model


@pytest.mark.parametrize("policy", ["fp32", "selective_bf16"])
def test_literal_official_equations_with_nonzero_head_and_tokenwise_times(policy):
    reference = literal_reference(policy)
    model = port.create_model(device="cpu", policy=policy, configuration=SMALL)
    model.load_state_dict(reference.state_dict(), strict=True)
    torch.manual_seed(71)
    inputs = [torch.randn(4, 3, 4, 4), torch.randn(4, 2, 4, 4)]
    text = [torch.randn(3, 16), torch.randn(7, 16)]
    time = port.token_times(torch.tensor([[3, 2, 2], [2, 2, 2]]), torch.tensor([999, 500]), 12)
    actual_types = {}
    hooks = []
    for name in ("patch_embedding", "text_embedding", "time_embedding", "time_projection", "blocks.0", "head"):
        hooks.append(model.get_submodule(name).register_forward_hook(lambda module, args, output, key=name: actual_types.update({key: output.dtype})))
    with torch.no_grad():
        expected = reference(inputs, time, text, 12)
        actual = model(inputs, time, text, 12)
    for hook in hooks:
        hook.remove()
    assert torch.count_nonzero(model.head.head.weight) > 0
    assert all(torch.count_nonzero(value) > 0 for value in actual)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6 if policy == "fp32" else 2e-4,
                                   rtol=2e-5 if policy == "fp32" else 2e-3)
        assert a.dtype == torch.float32 and torch.isfinite(a).all()
    low = torch.float32 if policy == "fp32" else torch.bfloat16
    assert actual_types == {"patch_embedding": low, "text_embedding": low, "time_embedding": torch.float32,
                           "time_projection": torch.float32, "blocks.0": torch.float32, "head": torch.float32}
    assert model.state_dict().keys() == reference.state_dict().keys()


def test_exact_native_meta_parameter_counts_and_storage_policy():
    model = port.create_model()
    report = port.storage_report(model)
    assert all(value.device.type == "meta" for value in model.parameters())
    assert report["parameters"] == {"float32": 68_573_376, "bfloat16": 4_931_214_336}
    assert report["bytes"] == 10_136_722_176 and report["total_parameters"] == 4_999_787_712
    assert all(value.dtype == port.declared_dtypes(model)[name] for name, value in model.named_parameters())
    with pytest.raises(ValueError, match="Full-size model"):
        port.create_model(device="cpu")


def test_native_observed_prefix_time_and_padded_sequence_are_exact():
    grid = torch.tensor([[5, 9, 16], [3, 9, 16]])
    t = port.token_times(grid, torch.tensor([999, 500]), 728)
    assert t.shape == (2, 728) and t.dtype == torch.int64
    assert torch.equal(t[:, :144], torch.zeros(2, 144, dtype=torch.int64))
    assert (t[0, 144:] == 999).all() and (t[1, 144:] == 500).all()
    assert tuple(grid[0].tolist()) == port.NATIVE_GRID and grid[0].prod() == port.NATIVE_TOKENS
    with pytest.raises(ValueError):
        port.token_times(grid, torch.tensor([999, 500]), 719)


def test_self_padding_is_masked_and_text_padding_is_projected_without_mask():
    torch.manual_seed(40)
    q = torch.randn(2, 6, 2, 8)
    k, v = torch.randn(2, 6, 2, 8), torch.randn(2, 6, 2, 8).bfloat16()
    lengths = torch.tensor([6, 3])
    expected = port.attention(q, k, v, k_lens=lengths)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[1, 3:] = 1000
    changed_v[1, 3:] = -1000
    assert torch.equal(expected, port.attention(q, changed_k, changed_v, k_lens=lengths))
    assert expected.dtype == q.dtype
    model = port.create_model(device="cpu", configuration=SMALL)
    captured = []
    model.blocks[0].cross_attn.register_forward_pre_hook(lambda module, args: captured.append(args))
    model([torch.randn(4, 2, 4, 4)], torch.full((1, 8), 500, dtype=torch.int64), [torch.randn(2, 16)], 8)
    assert captured[0][1].shape == (1, 8, 32) and captured[0][2] is None


def test_port_disables_external_cpu_autocast_for_fp32_time_and_head():
    model = port.create_model(device="cpu", configuration=SMALL)
    torch.nn.init.normal_(model.head.head.weight, std=.1)
    inputs = ([torch.randn(4, 2, 4, 4)], torch.full((1, 8), 500, dtype=torch.int64), [torch.randn(2, 16)], 8)
    plain = model(*inputs)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = model(*inputs)[0]
    assert plain.dtype == cast.dtype == torch.float32 and torch.equal(plain, cast)


def tiny_shards(tmp_path):
    reference = literal_reference()
    state = {key: value.clone().contiguous() for key, value in reference.state_dict().items()}
    keys = list(state)
    shards = {"part-1.safetensors": {key: state[key] for key in keys[::2]},
              "part-2.safetensors": {key: state[key] for key in keys[1::2]}}
    mapping, artifacts = {}, {}
    for name, values in shards.items():
        path = tmp_path / name
        save_file(values, path)
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        mapping.update({key: name for key in values})
    return state, {"weight_map": mapping}, artifacts


def test_stream_loader_uses_meta_and_exact_per_tensor_conversion(tmp_path):
    state, index, artifacts = tiny_shards(tmp_path)
    model = port.create_model(configuration=SMALL)
    calls = []
    result = stream_parameters(model, tmp_path, index, artifacts, check=lambda: calls.append(True))
    assert result["tensor_count"] == len(state) and result["nonzero_output_head"] and not result["full_state_dict_materialized"]
    assert len(calls) >= 2 * len(state)
    for name, value in model.named_parameters():
        assert value.device.type == "cpu" and not value.requires_grad
        assert torch.equal(value, state[name].to(port.declared_dtypes(model)[name]))
        assert result["tensors"][name]["converted_values_exact"]
    # The final CPU FP32 parameters must own storage rather than retaining a
    # safetensors mmap. Change only a synthetic fixture after verified loading.
    key = "head.head.weight"
    path = tmp_path / index["weight_map"][key]
    header = read_header(path)
    with path.open("r+b") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        file.seek(8 + header_size + header[key]["data_offsets"][0])
        file.write(struct.pack("<f", 999.))
    assert torch.equal(model.head.head.weight, state[key])


@pytest.mark.parametrize("change", ["missing_key", "wrong_hash", "outside", "wrong_shape"])
def test_stream_loader_rejects_wrong_index_hash_paths_and_shapes(tmp_path, change):
    state, index, artifacts = tiny_shards(tmp_path)
    model = port.create_model(configuration=SMALL)
    key = next(iter(index["weight_map"]))
    if change == "missing_key":
        del index["weight_map"][key]
    elif change == "wrong_hash":
        artifacts["part-1.safetensors"]["sha256"] = "0" * 64
    elif change == "outside":
        index["weight_map"][key] = "../outside.safetensors"
    else:
        model.patch_embedding.weight = torch.nn.Parameter(torch.empty(1, device="meta"), requires_grad=False)
    with pytest.raises(ValueError):
        stream_parameters(model, tmp_path, index, artifacts)
    assert all(value.device.type == "meta" for value in model.parameters())


def test_official_source_files_and_index_are_byte_exact():
    root = Path(port.__file__).parent
    data = json.loads((root / "provenance.json").read_text())
    for path, entry in data["official_sources"].items():
        assert sha256(root / path) == entry["sha256"]
    assert sha256(root / "weights.index.json") == data["index_sha256"]


def test_native_profile_prepares_only_observation_and_reproducible_future_noise():
    from .profile_core import make_scheduler, prepare_input
    scheduler = make_scheduler()
    assert scheduler.timesteps.device.type == "cpu" and scheduler.timesteps.dtype == torch.int64
    assert len(scheduler.timesteps) == 50
    observation = torch.linspace(-1, 1, 48 * 18 * 32).reshape(1, 48, 1, 18, 32)
    noise, latent, times = prepare_input(observation, scheduler.timesteps[0])
    assert torch.equal(latent[:, :1], observation[0]) and torch.equal(latent[:, 1:], noise[:, 1:])
    assert (times[:, :144] == 0).all() and (times[:, 144:] == scheduler.timesteps[0]).all()
    other_noise, other_latent, other_times = prepare_input(observation + .2, scheduler.timesteps[0])
    assert torch.equal(noise, other_noise) and torch.equal(latent[:, 1:], other_latent[:, 1:])
    assert torch.equal(times, other_times)
    with pytest.raises(ValueError, match="native48-channel"):
        prepare_input(torch.zeros(1, 16, 1, 36, 64), scheduler.timesteps[0])


def test_native_profile_reuses_exact_verified_real_text_without_encoder():
    from .profile_core import load_text
    root = Path(port.__file__).parent.parent / "wan_adapter/text_cache/native-results"
    positive, negative, evidence = load_text(root)
    assert positive.dtype == negative.dtype == torch.float32
    assert positive.shape[1] == negative.shape[1] == 4096
    assert torch.isfinite(positive).all() and torch.isfinite(negative).all()
    assert torch.count_nonzero(positive) > 0 and torch.count_nonzero(negative) > 0
    assert len(evidence["manifest_sha256"]) == 64
