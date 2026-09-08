# SPDX-License-Identifier: Apache-2.0
"""Small CPU stand-ins only. No vendor attention, real weights or CUDA calls."""
from dataclasses import replace
import gc
import weakref

import pytest
import torch
from torch import nn

from ..action_adapter.model import PostBlockActionAdapter
from .bridge import NativeCUDAActionBridge, PROFILES, _HeadBoundary


class FakeBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.grad_modes = []

    def forward(self, x, e):
        self.grad_modes.append(torch.is_grad_enabled())
        return x + .2 * torch.tanh(self.linear(x) + e)


class FakeHead(nn.Module):
    def __init__(self, dim, channels):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, 4 * channels)
        self.calls = 0
        self.fail = False
        self.inputs = []

    def forward(self, x, e):
        self.calls += 1
        self.inputs.append((x.detach().clone(), e.detach().clone(), x.requires_grad, torch.is_grad_enabled()))
        if self.fail:
            raise RuntimeError("head failure")
        return self.linear(self.norm(x) * (1 + .1 * e) + e)


class FakeCore(nn.Module):
    """Native-shaped interface with a deliberately nonzero, differentiable head."""
    def __init__(self):
        super().__init__()
        self.in_dim = self.out_dim = 2
        self.dim, self.text_dim, self.text_len = 8, 6, 8
        self.patch_size, self.model_type = (1, 2, 2), "ti2v"
        self.patch_embedding = nn.Conv3d(2, 8, (1, 2, 2), stride=(1, 2, 2))
        self.time_embedding = nn.Linear(1, 8)
        self.text_embedding = nn.Linear(6, 8)
        self.blocks = nn.ModuleList([FakeBlock(8), FakeBlock(8)])
        self.head = FakeHead(8, 2)
        self.forward_calls, self.unpatchify_calls = 0, 0
        self.fail = None
        self.last_sequence_length = None

    def forward(self, latents, times, contexts, seq_len):
        self.forward_calls += 1
        self.last_sequence_length = seq_len
        temporary = torch.empty(16)
        self.last_temporary = weakref.ref(temporary)
        if self.fail == "preprocess":
            raise RuntimeError("preprocess failure")
        if self.fail == "foreign-sentinel":
            raise _HeadBoundary()
        tokens = [self.patch_embedding(value[None]) for value in latents]
        grids = torch.tensor([list(value.shape[2:]) for value in tokens])
        x = torch.cat([value.flatten(2).transpose(1, 2) for value in tokens])
        assert x.shape[1] == seq_len
        e = self.time_embedding(times.float()[..., None] / 1000)
        x = x + self.text_embedding(torch.stack([value.mean(0) for value in contexts]))[:, None]
        for block in self.blocks:
            if self.fail == "block":
                raise RuntimeError("block failure")
            x = block(x, e)
        if self.fail == "missing-boundary":
            return []
        if self.fail == "head-signature":
            return self.head(x, e, None)
        if self.fail == "head-dtype":
            return self.head(x.double(), e)
        x = self.head(x, e)
        return [value.float() for value in self.unpatchify(x, grids)]

    def unpatchify(self, patches, grids):
        self.unpatchify_calls += 1
        if self.fail == "unpatchify":
            raise RuntimeError("unpatchify failure")
        result = []
        for value, (frames, height, width) in zip(patches, grids.tolist()):
            value = value.view(frames, height, width, 1, 2, 2, self.out_dim)
            value = torch.einsum("fhwpqrc->cfphqwr", value)
            result.append(value.reshape(self.out_dim, frames, height*2, width*2))
        return result


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(profile="baseline"):
    torch.manual_seed(20260907)
    core = FakeCore().requires_grad_(False).eval()
    adapter = PostBlockActionAdapter(8, 2, 8)
    bridge = NativeCUDAActionBridge(core, adapter, profile=profile, test_only=True)
    x = torch.randn(1, *bridge.shape)
    observation = x[:, :, :1].clone()
    commands = torch.randn(1, 16, 6) * .05
    commands[..., 5] = 0
    commands[:, 0, 5] = 1
    times = torch.full((1, bridge.tokens), 506, dtype=torch.int64)
    times[:, :PROFILES[profile]["prefix"]] = 0
    contexts = [torch.randn(3, core.text_dim)]
    return bridge, x, times, contexts, commands, observation


def call(values, **kwargs):
    bridge, x, times, contexts, commands, observation = values
    return bridge(x, times, contexts, commands=commands, observation=observation, **kwargs)


def assert_no_hooks(core):
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in core.modules())


@pytest.mark.parametrize("profile", ["baseline", "spatial"])
def test_zero_adapter_identity_and_original_head_capture(profile):
    values = fixture(profile)
    bridge, x, times, contexts, commands, observation = values
    methods = [(module.forward, type(module)) for module in bridge.core.modules()]
    with torch.no_grad():
        expected = torch.stack(bridge.core(list(x.unbind(0)), times, contexts, bridge.tokens))
    reference_hidden, reference_time, _, _ = bridge.core.head.inputs[-1]
    features = bridge.extract_features(x, times, contexts)
    assert bridge.core.head.calls == 1 and bridge.core.unpatchify_calls == 1
    assert torch.equal(features.hidden, reference_hidden)
    assert torch.equal(features.time_embedding, reference_time)
    assert features.grid_sizes.tolist() == [list(bridge.grid)]
    assert bridge.core.last_sequence_length == PROFILES[profile]["tokens"]
    actual = bridge.predict_from_features(features, commands, observation)
    assert torch.equal(actual, expected) and actual.dtype == torch.float32
    assert actual.requires_grad
    assert bridge.core.head.calls == 2 and bridge.core.unpatchify_calls == 2
    assert methods == [(module.forward, type(module)) for module in bridge.core.modules()]
    assert_no_hooks(bridge.core)


def test_capture_constants_are_normal_inside_ambient_inference_mode():
    bridge, x, times, contexts, commands, observation = fixture()
    with torch.inference_mode():
        features = bridge.extract_features(x, times, contexts)
        output = bridge.predict_from_features(features, commands, observation)
    for tensor in (features.hidden, features.time_embedding, features.observed_prefix):
        assert not tensor.requires_grad and tensor.grad_fn is None and not tensor.is_inference()
    assert output.requires_grad and not output.is_inference()
    output.square().mean().backward()
    assert bridge.adapter.output.weight.grad.abs().sum() > 0
    assert all(not flag for block in bridge.core.blocks for flag in block.grad_modes)


def test_second_backward_reaches_gru_and_inputs_while_native_core_is_frozen():
    values = list(fixture())
    bridge, x, times, contexts, commands, observation = values
    saved = {name: p.detach().clone() for name, p in bridge.core.named_parameters()}
    values[1] = x.requires_grad_()
    values[4] = commands.requires_grad_()
    values[5] = observation.requires_grad_()
    optimizer = torch.optim.SGD(bridge.adapter.parameters(), lr=.1)
    gru_norms = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        values[4].grad, values[5].grad = None, None
        prediction = call(values)
        prediction[:, :, 1:].square().mean().backward()
        norms = [p.grad.square().sum() for p in bridge.adapter.command_gru.parameters()]
        gru_norms.append(torch.stack(norms).sum().sqrt().item())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bridge.adapter.parameters())
        optimizer.step()
    assert gru_norms[0] == 0 and gru_norms[1] > 0
    assert values[4].grad.abs().sum() > 0 and values[5].grad.abs().sum() > 0
    assert values[1].grad is None
    assert all(p.grad is None and torch.equal(p, saved[name]) for name, p in bridge.core.named_parameters())
    assert all(not flag for block in bridge.core.blocks for flag in block.grad_modes)
    assert all(requires and enabled for _, _, requires, enabled in bridge.core.head.inputs)
    assert_no_hooks(bridge.core)


def test_nonzero_adapter_preserves_observed_prefix_and_changes_future_only():
    bridge, x, times, contexts, commands, observation = fixture()
    with torch.no_grad():
        bridge.adapter.output.weight.normal_(std=.025)
        bridge.adapter.output.bias.normal_(std=.025)
    features = bridge.extract_features(x, times, contexts)
    first = bridge.predict_from_features(features, commands, observation)
    alternate = commands.clone()
    alternate[..., 3] *= -1
    alternate[:, 0, 5] = 0
    second = bridge.predict_from_features(features, alternate, observation)
    assert torch.equal(first[:, :, :1], second[:, :, :1])
    assert not torch.equal(first[:, :, 1:], second[:, :, 1:])
    for hidden, _, _, _ in bridge.core.head.inputs:
        prefix = PROFILES[bridge.profile]["prefix"]
        assert torch.equal(hidden[:, :prefix], features.hidden[:, :prefix])


@pytest.mark.parametrize("failure", ["preprocess", "block", "missing-boundary", "head-signature", "head-dtype"])
def test_capture_hook_removed_on_all_native_or_boundary_failures(failure):
    bridge, x, times, contexts, _, _ = fixture()
    bridge.core.fail = failure
    with pytest.raises((RuntimeError, ValueError)):
        bridge.extract_features(x, times, contexts)
    assert bridge.core.head.calls == bridge.core.unpatchify_calls == 0
    assert_no_hooks(bridge.core)
    bridge.core.fail = None
    assert bridge.extract_features(x, times, contexts).hidden.shape == (1, bridge.tokens, bridge.core.dim)


def test_foreign_private_sentinel_is_not_swallowed():
    bridge, x, times, contexts, _, _ = fixture()
    bridge.core.fail = "foreign-sentinel"
    with pytest.raises(_HeadBoundary):
        bridge.extract_features(x, times, contexts)
    assert_no_hooks(bridge.core)


def test_caught_sentinel_releases_native_frame_without_cyclic_collection():
    bridge, x, times, contexts, _, _ = fixture()
    enabled = gc.isenabled()
    gc.disable()
    try:
        features = bridge.extract_features(x, times, contexts)
        assert bridge.core.last_temporary() is None
        assert features.hidden.shape == (1, bridge.tokens, bridge.core.dim)
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("failure", ["head", "unpatchify"])
def test_post_boundary_failure_leaves_no_hook_or_lock(failure):
    values = fixture()
    bridge, x, times, contexts, commands, observation = values
    features = bridge.extract_features(x, times, contexts)
    if failure == "head":
        bridge.core.head.fail = True
    else:
        bridge.core.fail = "unpatchify"
    with pytest.raises(RuntimeError):
        bridge.predict_from_features(features, commands, observation)
    assert_no_hooks(bridge.core)
    bridge.core.head.fail, bridge.core.fail = False, None
    assert bridge.predict_from_features(features, commands, observation).shape == x.shape


def test_existing_unrelated_hook_is_rejected_and_preserved():
    bridge, x, times, contexts, _, _ = fixture()
    handle = bridge.core.head.register_forward_pre_hook(lambda module, args: None)
    keys = tuple(bridge.core.head._forward_pre_hooks)
    try:
        with pytest.raises(ValueError, match="hooks"):
            bridge.extract_features(x, times, contexts)
        assert tuple(bridge.core.head._forward_pre_hooks) == keys
        assert bridge.core.forward_calls == 0
    finally:
        handle.remove()


@pytest.mark.parametrize("bad", ["shape", "time-prefix", "time-range", "time-nonuniform", "text-nan", "text-empty",
                                  "commands-shape", "commands-nan", "pulse", "observation", "track-grad"])
def test_invalid_inputs_fail_before_frozen_forward(bad):
    values = list(fixture())
    kwargs = {}
    if bad == "shape":
        values[1] = values[1][:, :, :, :-1]
    elif bad == "time-prefix":
        values[2][0, 0] = 1
    elif bad == "time-range":
        values[2][:, PROFILES["baseline"]["prefix"]:] = 1000
    elif bad == "time-nonuniform":
        values[2][0, -1] += 1
    elif bad == "text-nan":
        values[3][0][0, 0] = float("nan")
    elif bad == "text-empty":
        values[3] = [values[3][0][:0]]
    elif bad == "commands-shape":
        values[4] = values[4][:, :-1]
    elif bad == "commands-nan":
        values[4][0, 0, 0] = float("nan")
    elif bad == "pulse":
        values[4][0, 0, 5] = .5
    elif bad == "observation":
        values[5][0, 0, 0, 0, 0] += 1
    else:
        kwargs["track_grad"] = 1
    with pytest.raises(ValueError):
        call(values, **kwargs)
    assert values[0].core.forward_calls == 0
    assert_no_hooks(values[0].core)


def test_owner_grid_and_nonconstant_features_are_rejected():
    bridge, x, times, contexts, commands, observation = fixture()
    features = bridge.extract_features(x, times, contexts)
    bad = [replace(features, owner=object()), replace(features, profile="spatial"),
           replace(features, grid_sizes=torch.tensor([[5, 16, 9]])),
           replace(features, hidden=features.hidden.clone().requires_grad_())]
    for value in bad:
        with pytest.raises(ValueError):
            bridge.predict_from_features(value, commands, observation)
    assert bridge.core.head.calls == 0


def test_explicit_gradient_policy_and_training_mode():
    values = fixture()
    bridge = values[0]
    bridge.train()
    assert bridge.adapter.training and not bridge.core.training
    with torch.no_grad():
        tracked = call(values)
    assert tracked.requires_grad
    untracked = call(values, track_grad=False)
    assert not untracked.requires_grad and torch.equal(tracked, untracked)


def test_frozen_parameter_and_concurrent_capture_checks():
    bridge, x, times, contexts, _, _ = fixture()
    parameter = next(bridge.core.parameters())
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        bridge.extract_features(x, times, contexts)
    parameter.requires_grad_(False)
    assert bridge._call_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="Concurrent"):
            bridge.extract_features(x, times, contexts)
    finally:
        bridge._call_lock.release()
    assert bridge.core.forward_calls == 0
    assert_no_hooks(bridge.core)


def test_production_rejects_cpu_fixture_without_touching_cuda(monkeypatch):
    bridge, *_ = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU tests must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with pytest.raises(ValueError, match="CUDA"):
        NativeCUDAActionBridge(bridge.core, bridge.adapter)
    original = PostBlockActionAdapter()
    assert sum(p.numel() for p in original.parameters()) == 947_712
    assert all(p.dtype == torch.float32 for p in original.parameters())
