# SPDX-License-Identifier: Apache-2.0
"""Tiny literal native blocks with scoped FP32 CPU attention, never CUDA/FA2."""
import copy
from unittest import mock

import pytest
import torch

from experiments.wan22_native.action_adapter.model import PostBlockActionAdapter
from experiments.wan22_native.action_cuda.bridge import NativeCUDAActionBridge
from experiments.wan22_native.cuda_reference.vendor import model as vendor
from experiments.wan22_native.intermediate_action.bridge import IntermediateActionBridge


def cpu_attention(q, k, v, *, k_lens=None, window_size=(-1, -1)):
    """Small-test mathematical attention; deliberately not a CUDA substitute."""
    assert q.device.type == k.device.type == v.device.type == 'cpu'
    assert q.dtype == k.dtype == v.dtype == torch.float32
    assert window_size == (-1, -1)
    result = []
    for index in range(q.shape[0]):
        length = k.shape[1] if k_lens is None else int(k_lens[index])
        query = q[index].transpose(0, 1)
        key = k[index, :length].transpose(0, 1)
        value = v[index, :length].transpose(0, 1)
        weights = torch.softmax((query @ key.transpose(-1, -2)) / q.shape[-1]**.5, dim=-1)
        result.append((weights @ value).transpose(0, 1))
    return torch.stack(result)


@pytest.fixture(autouse=True)
def bounded_cpu(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    # Only this test fixture changes the imported attention symbol. The
    # implementation never patches it and production still requires FA2.
    monkeypatch.setattr(vendor, 'flash_attention', cpu_attention)
    yield
    torch.set_num_threads(previous)
    assert not torch.cuda.is_initialized()


def fixture(index=0, activated=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2031209)
        core = vendor.WanModel(model_type='ti2v', in_dim=2, out_dim=2, dim=8,
            ffn_dim=16, freq_dim=8, text_dim=4, text_len=8, num_heads=1, num_layers=2)
        with torch.no_grad():
            core.head.head.weight.normal_(0, .12)
            core.head.head.bias.normal_(0, .03)
        core.requires_grad_(False).eval()
        adapter = PostBlockActionAdapter(8, 2, 4)
        if activated:
            with torch.no_grad():
                adapter.output.weight.normal_(0, .05)
                adapter.output.bias.normal_(0, .02)
        bridge = IntermediateActionBridge(core, adapter, block_index=index, test_only=True)
        noisy = torch.randn(1, *bridge.shape)
        times = torch.full((1, bridge.tokens), 506, dtype=torch.int64)
        times[:, :144] = 0
        contexts = [torch.randn(3, 4)]
        commands = torch.zeros(1, 16, 6)
        commands[:, 1:, 3] = .1
        commands[:, 0, 5] = 1
        observation = noisy[:, :, :1].clone()
    return bridge, noisy, times, contexts, commands, observation


def call(values, **kwargs):
    bridge, noisy, times, contexts, commands, observation = values
    return bridge(noisy, times, contexts, commands=commands, observation=observation, **kwargs)


def no_hooks(core):
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in core.modules())


def direct_suffix_features(values):
    """Independent reference: capture constants, then explicitly run the suffix."""
    bridge, noisy, times, contexts, _, _ = values
    saved = {}
    def block_done(module, args, kwargs, output):
        saved['hidden'] = output.detach().clone()
        saved['kwargs'] = kwargs
    def head_start(module, args):
        saved['time'] = args[1].detach().clone()
    a = bridge.core.blocks[bridge.block_index].register_forward_hook(block_done, with_kwargs=True)
    b = bridge.core.head.register_forward_pre_hook(head_start)
    try:
        with torch.no_grad():
            bridge.core(list(noisy.unbind(0)), times, contexts, bridge.tokens)
    finally:
        a.remove(); b.remove()
    return saved


def direct_suffix(bridge, adapter, saved, commands, observation):
    value = adapter(saved['hidden'], commands, observation, bridge.grid)
    for block in bridge.core.blocks[bridge.block_index+1:]:
        value = block(value, **saved['kwargs'])
    patches = bridge.core.head(value, saved['time'])
    return torch.stack(bridge.core.unpatchify(patches, saved['kwargs']['grid_sizes']))


@pytest.mark.parametrize('index', [0, 1])
def test_zero_adapter_is_exact_literal_native_forward(index):
    values = fixture(index)
    bridge, noisy, times, contexts, _, _ = values
    with torch.no_grad():
        expected = torch.stack(bridge.core(list(noisy.unbind(0)), times, contexts, bridge.tokens))
    actual = call(values)
    assert torch.equal(actual, expected)
    assert actual.requires_grad and actual.dtype == torch.float32
    assert torch.equal(call(values, track_grad=False), expected)
    no_hooks(bridge.core)


@pytest.mark.parametrize('index', [0, 1])
def test_activated_adapter_all_gradients_match_explicit_native_suffix(index):
    values = fixture(index, activated=True)
    bridge, noisy, times, contexts, commands, observation = values
    saved = direct_suffix_features(values)
    reference = copy.deepcopy(bridge.adapter)
    weights = {name: p.detach().clone() for name, p in bridge.core.named_parameters()}
    alternate = commands.clone(); alternate[:, 0, 5] = 0
    for action in (commands, alternate):
        actual = bridge(noisy, times, contexts, commands=action, observation=observation)
        expected = direct_suffix(bridge, reference, saved, action, observation)
        assert torch.equal(actual, expected)
        (.5 * actual[:, :, 1:].square().mean()).backward()
        (.5 * expected[:, :, 1:].square().mean()).backward()
    for (name, actual), (other, expected) in zip(bridge.adapter.named_parameters(), reference.named_parameters()):
        assert name == other and actual.grad is not None
        assert torch.isfinite(actual.grad).all()
        assert torch.equal(actual.grad, expected.grad), name
    assert sum(p.grad.abs().sum() for p in bridge.adapter.command_gru.parameters()) > 0
    assert bridge.adapter.query.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None and torch.equal(p, weights[name])
               for name, p in bridge.core.named_parameters())
    no_hooks(bridge.core)


def test_graph_starts_at_insertion_and_second_update_reaches_gru():
    values = fixture(0)
    bridge = values[0]
    modes = []
    original = vendor.WanAttentionBlock.forward
    def inspect(module, x, **kwargs):
        output = original(module, x, **kwargs)
        modes.append((list(bridge.core.blocks).index(module), x.requires_grad,
                      output.requires_grad, output.grad_fn is not None))
        return output
    optimizer = torch.optim.AdamW(bridge.adapter.parameters(), lr=.01)
    norms = []
    with mock.patch.object(vendor.WanAttentionBlock, 'forward', inspect):
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = call(values)
            prediction[:, :, 1:].square().mean().backward()
            norms.append(sum(p.grad.square().sum() for p in bridge.adapter.command_gru.parameters()).sqrt().item())
            optimizer.step()
    assert modes == [(0, False, False, False), (1, True, True, True)] * 2
    assert norms[0] == 0 and norms[1] > 0
    assert all(p.grad is None for p in bridge.core.parameters())
    assert all(value.grad is None for value in (values[1], values[4], values[5], *values[3]))
    no_hooks(bridge.core)


def test_last_block_matches_existing_final_head_bridge_with_nonzero_adapter():
    values = fixture(1, activated=True)
    bridge, noisy, times, contexts, commands, observation = values
    control = NativeCUDAActionBridge(bridge.core, copy.deepcopy(bridge.adapter), test_only=True)
    actual = call(values)
    expected = control(noisy, times, contexts, commands=commands, observation=observation)
    assert torch.equal(actual, expected)
    actual[:, :, 1:].square().mean().backward()
    expected[:, :, 1:].square().mean().backward()
    for p, q in zip(bridge.adapter.parameters(), control.adapter.parameters()):
        assert torch.equal(p.grad, q.grad)
    no_hooks(bridge.core)


def test_normal_constants_and_context_restoration_inside_inference_mode():
    values = fixture(0, activated=True)
    with torch.inference_mode():
        actual = call(values)
        assert torch.is_inference_mode_enabled()
    assert actual.requires_grad and not actual.is_inference()
    actual[:, :, 1:].square().mean().backward()
    with torch.no_grad():
        actual = call(values)
        assert not torch.is_grad_enabled()
    assert actual.requires_grad
    inference = call(values, track_grad=False)
    assert not inference.requires_grad and inference.grad_fn is None


def test_clean_input_and_direct_adapter_mask_preserved_not_output_clamped():
    values = fixture(0, activated=True)
    bridge, noisy, times, contexts, commands, observation = values
    before = [x.clone() for x in (noisy, times, commands, observation, *contexts)]
    original = PostBlockActionAdapter.forward
    captured = []
    def inspect(module, hidden, actions, observed, grid):
        output = original(module, hidden, actions, observed, grid)
        captured.append((hidden.detach().clone(), output.detach().clone()))
        return output
    with mock.patch.object(PostBlockActionAdapter, 'forward', inspect):
        first = call(values)
        alternate = commands.clone(); alternate[:, 0, 5] = 0
        second = bridge(noisy, times, contexts, commands=alternate, observation=observation)
    for hidden, adapted in captured:
        assert torch.equal(hidden[:, :144], adapted[:, :144])
        assert not torch.equal(hidden[:, 144:], adapted[:, 144:])
    assert all(torch.equal(x, y) for x, y in zip((noisy, times, commands, observation, *contexts), before))
    # Later self-attention can mix the direct future-only residual back into
    # observed output tokens. This must not be mislabeled as input mutation.
    assert not torch.equal(first[:, :, :1], second[:, :, :1])
    padded = torch.randn(1, bridge.tokens+3, bridge.core.dim)
    adapted = bridge.adapter(padded, commands, observation, bridge.grid)
    assert torch.equal(adapted[:, :144], padded[:, :144])
    assert torch.equal(adapted[:, bridge.tokens:], padded[:, bridge.tokens:])


@pytest.mark.parametrize('where', ['preprocess', 'selected', 'adapter', 'suffix', 'head', 'unpatchify'])
def test_error_identity_hook_and_lock_cleanup(where):
    values = fixture(0)
    bridge = values[0]
    failure = KeyboardInterrupt('bounded native failure fixture')
    def fail(*args, **kwargs): raise failure
    if where in ('selected', 'suffix'):
        original = vendor.WanAttentionBlock.forward
        selected = bridge.core.blocks[0 if where == 'selected' else 1]
        def conditional(module, *args, **kwargs):
            if module is selected: raise failure
            return original(module, *args, **kwargs)
        patcher = mock.patch.object(vendor.WanAttentionBlock, 'forward', conditional)
    else:
        owner, attr = {'preprocess':(vendor.WanModel,'forward'),
                       'adapter':(PostBlockActionAdapter,'forward'),
                       'head':(vendor.Head,'forward'),
                       'unpatchify':(vendor.WanModel,'unpatchify')}[where]
        patcher = mock.patch.object(owner, attr, fail)
    with patcher:
        with pytest.raises(KeyboardInterrupt) as caught: call(values)
    assert caught.value is failure
    no_hooks(bridge.core)
    assert not bridge._contract._call_lock.locked()
    assert call(values).shape == values[1].shape


@pytest.mark.parametrize('index', [-1, 2, True, 0.5, '0'])
def test_invalid_index_rejected(index):
    values = fixture()
    with pytest.raises(ValueError, match='block_index'):
        IntermediateActionBridge(values[0].core, values[0].adapter, block_index=index, test_only=True)


@pytest.mark.parametrize('which', ['noisy', 'commands', 'observation', 'context'])
def test_gradient_inputs_rejected_before_native_execution(which):
    values = list(fixture())
    target = {'noisy':values[1], 'commands':values[4], 'observation':values[5], 'context':values[3][0]}[which]
    target.requires_grad_()
    with mock.patch.object(vendor.WanModel, 'forward', side_effect=AssertionError('must not execute')):
        with pytest.raises(ValueError, match='input tensors'): call(values)
    no_hooks(values[0].core)


def test_changed_prefix_frozen_state_and_foreign_hook_are_rejected():
    values = fixture()
    bridge = values[0]
    values[5].flatten()[0] += .1
    with pytest.raises(ValueError, match='exact clean initial prefix'): call(values)
    values[5].copy_(values[1][:, :, :1])
    handle = bridge.core.blocks[0].register_forward_hook(lambda module, args, output: None)
    try:
        with pytest.raises(ValueError, match='no hooks'): call(values)
        assert handle.id in bridge.core.blocks[0]._forward_hooks
    finally: handle.remove()
    p = next(bridge.core.parameters());p.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen FP32'): call(values)
    p.requires_grad_(False);p.grad=torch.zeros_like(p)
    with pytest.raises(ValueError, match='frozen FP32'): call(values)
    p.grad=None
    assert call(values).shape == values[1].shape


def test_production_does_not_accept_cpu_core_or_change_precision():
    values = fixture()
    bridge = values[0]
    with pytest.raises(ValueError, match='native CUDA core'):
        IntermediateActionBridge(bridge.core, bridge.adapter, block_index=0)
    assert all(p.dtype == torch.float32 for p in bridge.core.parameters())
    assert all(p.dtype == torch.float32 for p in bridge.adapter.parameters())
    assert not torch.cuda.is_initialized()
