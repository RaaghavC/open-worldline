# SPDX-License-Identifier: Apache-2.0
"""Independent bounded CPU checks; no official weights or GPU access."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

from . import load_weights as loader
from . import portable as port


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def literal_module():
    name = __package__ + '.vendor._independent_reference'
    spec = importlib.util.spec_from_file_location(name, Path(port.__file__).parent / 'vendor/model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cpu_attention(q, k, v, k_lens=None, **kwargs):
    original = q.dtype
    mask = None
    if k_lens is not None:
        mask = torch.arange(k.shape[1])[None, None, None, :] < k_lens[:, None, None, None]
    value = F.scaled_dot_product_attention(q.to(v.dtype).transpose(1, 2),
        k.to(v.dtype).transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
        is_causal=False, dropout_p=0.)
    return value.transpose(1, 2).contiguous().to(original)


def test_production_width_rotary_axes_and_padding_match_official_complex_math():
    official = literal_module()
    generator = torch.Generator().manual_seed(133)
    value = torch.randn(2, 34, 1, 128, generator=generator)
    grid = torch.tensor([[2, 3, 5], [3, 2, 4]])
    dimensions = [44, 42, 42]
    complex_table = torch.cat([official.rope_params(1024, d) for d in dimensions], dim=1)
    real_table = torch.cat([port.real_rope_params(1024, d) for d in dimensions], dim=1)
    expected = official.rope_apply(value, grid, complex_table)
    actual = port.real_rope_apply(value, grid, real_table)
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=2e-6)
    for index, length in enumerate([30, 24]):
        assert torch.equal(actual[index, length:], value[index, length:])


@pytest.mark.parametrize('policy', ['fp32', 'selective_bf16'])
def test_production_head_width_literal_forward_and_batch_isolation(policy):
    official = literal_module()
    official.flash_attention = cpu_attention
    config = dict(model_type='ti2v', in_dim=4, out_dim=4, dim=128, ffn_dim=192,
                  freq_dim=16, text_dim=16, text_len=8, num_heads=1, num_layers=1)
    torch.manual_seed(621)
    reference = official.WanModel(**config).eval().requires_grad_(False)
    nn.init.normal_(reference.head.head.weight, std=.05)
    if policy == 'selective_bf16':
        norm_names = {name + '.' + key for name, child in reference.named_modules()
            if isinstance(child, (official.WanRMSNorm, official.WanLayerNorm))
            for key, _ in child.named_parameters(recurse=False)}
        for name, value in list(reference.named_parameters()):
            fp32 = name in norm_names or name.startswith(('time_embedding.', 'time_projection.', 'head.')) or name.endswith('.modulation')
            parent, field = name.rsplit('.', 1)
            setattr(reference.get_submodule(parent), field, nn.Parameter(value.to(torch.float32 if fp32 else torch.bfloat16), requires_grad=False))
        for child in reference.modules():
            if isinstance(child, (nn.Linear, nn.Conv3d)):
                child.register_forward_pre_hook(lambda layer, args: (args[0].to(layer.weight.dtype),))
    model = port.create_model(device='cpu', policy=policy, configuration=config)
    model.load_state_dict(reference.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(710)
    inputs = [torch.randn(4, f, 4, 6, generator=generator) for f in (3, 2)]
    text = [torch.randn(n, 16, generator=generator) for n in (2, 7)]
    # Nonuniform future times independently exercise both block and head token axes.
    times = torch.randint(1, 1000, (2, 20), generator=generator)
    times[:, :6] = 0
    with torch.no_grad():
        expected = reference(inputs, times, text, 20)
        actual = model(inputs, times, text, 20)
        changed = model([inputs[0], inputs[1] * 9], times, [text[0], text[1] * -4], 20)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6 if policy == 'fp32' else 2e-4,
                                   rtol=2e-5 if policy == 'fp32' else 2e-3)
        assert torch.isfinite(a).all() and a.dtype == torch.float32 and torch.count_nonzero(a)
    assert torch.equal(actual[0], changed[0])
    assert not torch.equal(actual[1], changed[1])


def test_native_time_mask_matches_independent_spatial_patch_enumeration():
    mask = torch.ones(5, 18, 32, dtype=torch.int64)
    mask[0] = 0
    expected = (mask[:, ::2, ::2] * 987).flatten()
    actual = port.token_times(torch.tensor([[5, 9, 16]]), torch.tensor([987]), 720)[0]
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual == 0) == 144
    assert torch.count_nonzero(actual == 987) == 576
    with pytest.raises(ValueError):
        port.token_times(torch.tensor([[5, 9, 16]]), torch.tensor([987.]), 720)


def test_instance_extensions_leave_official_class_methods_unchanged():
    classes = [port.official.WanModel, port.official.WanAttentionBlock,
               port.official.WanSelfAttention, port.official.WanCrossAttention, port.official.Head]
    methods = [cls.forward for cls in classes]
    config = dict(model_type='ti2v', in_dim=4, out_dim=4, dim=32, ffn_dim=64,
                  freq_dim=8, text_dim=16, text_len=8, num_heads=4, num_layers=1)
    a = port.create_model(device='cpu', configuration=config)
    b = port.create_model(device='cpu', policy='fp32', configuration=config)
    assert all(cls.forward is original for cls, original in zip(classes, methods))
    assert a.forward.__self__ is a and b.forward.__self__ is b
    assert a.patch_embedding.weight.dtype == torch.bfloat16
    assert b.patch_embedding.weight.dtype == torch.float32


def tiny_stream(tmp_path, *, nonfinite=False):
    model = nn.Module()
    with torch.device('meta'):
        model.body = nn.Linear(2, 2)
        model.head = nn.Module()
        model.head.head = nn.Linear(2, 2)
    model._storage_policy = 'selective_bf16'
    values = {name: torch.full(tuple(value.shape), 1.2345) for name, value in model.named_parameters()}
    if nonfinite:
        values['body.weight'][0, 0] = float('nan')
    file = tmp_path / 'tiny.safetensors'
    save_file(values, file)
    index = {'weight_map': {name: file.name for name in values}}
    artifacts = {file.name: {'bytes': file.stat().st_size, 'sha256': loader.sha256(file)}}
    return model, values, index, artifacts


def test_loader_opens_one_handle_per_tensor_and_closes_on_guard_stop(tmp_path, monkeypatch):
    model, values, index, artifacts = tiny_stream(tmp_path)
    real = loader.safe_open
    state = {'active': 0, 'maximum': 0, 'reads': [], 'exits': 0}

    class Spy:
        def __init__(self, *args, **kwargs):
            self.inner = real(*args, **kwargs)
        def __enter__(self):
            state['active'] += 1
            state['maximum'] = max(state['maximum'], state['active'])
            self.handle = self.inner.__enter__()
            return self
        def get_tensor(self, name):
            state['reads'].append(name)
            return self.handle.get_tensor(name)
        def __exit__(self, *args):
            state['active'] -= 1
            state['exits'] += 1
            return self.inner.__exit__(*args)

    monkeypatch.setattr(loader, 'safe_open', Spy)
    report = loader.stream_parameters(model, tmp_path, index, artifacts)
    assert state == {'active': 0, 'maximum': 1, 'reads': list(values), 'exits': len(values)}
    assert all(torch.equal(value, values[name].to(value.dtype)) for name, value in model.named_parameters())
    assert not report['full_state_dict_materialized']
    stopped, _, _, _ = tiny_stream(tmp_path)

    def stop_after_first_assignment():
        if any(parameter.device.type != 'meta' for parameter in stopped.parameters()):
            raise RuntimeError('independent guard stop')

    with pytest.raises(RuntimeError, match='independent guard stop'):
        loader.stream_parameters(stopped, tmp_path, index, artifacts, check=stop_after_first_assignment)
    assert state['active'] == 0
    assert sum(parameter.device.type != 'meta' for parameter in stopped.parameters()) == 1


def test_hash_valid_nonfinite_source_fails_before_parameter_assignment(tmp_path):
    model, _, index, artifacts = tiny_stream(tmp_path, nonfinite=True)
    with pytest.raises(ValueError, match='nonfinite'):
        loader.stream_parameters(model, tmp_path, index, artifacts)
    assert all(value.device.type == 'meta' for value in model.parameters())
