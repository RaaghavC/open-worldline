# SPDX-License-Identifier: Apache-2.0
"""Independent CPU bridge checks using literal upstream head/unpatchify.

The tiny vendor model has zero attention blocks. These tests therefore check
native boundary and gradient mechanics, not CUDA/FA2 or full-model parity.
"""
import copy
from unittest import mock

import pytest
import torch

from ..action_adapter.model import PostBlockActionAdapter
from ..cuda_reference.vendor.model import WanModel
from . import bridge as implementation
from .bridge import NativeCUDAActionBridge, PROFILES


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(profile='baseline'):
    torch.manual_seed(93731)
    core = WanModel(model_type='ti2v', in_dim=2, out_dim=2, dim=8,
                    ffn_dim=16, freq_dim=8, text_dim=4, text_len=8,
                    num_heads=1, num_layers=0)
    with torch.no_grad():
        core.head.head.weight.normal_(0, .12)
        core.head.head.bias.normal_(0, .03)
    core.requires_grad_(False).eval()
    adapter = PostBlockActionAdapter(8, 2, 4)
    wrapped = NativeCUDAActionBridge(core, adapter, profile=profile, test_only=True)
    settings = PROFILES[profile]
    noisy = torch.randn(1, 2, *settings['shape'][1:])
    times = torch.full((1, settings['tokens']), 617, dtype=torch.int64)
    times[:, :settings['prefix']] = 0
    contexts = [torch.randn(3, 4)]
    commands = torch.zeros(1, 16, 6)
    commands[:, 2:, 3] = .08
    observation = noisy[:, :, :1].clone()
    return wrapped, noisy, times, contexts, commands, observation


@pytest.mark.parametrize('profile', ['baseline', 'spatial'])
def test_literal_head_boundary_time_grid_and_zero_adapter_identity(profile):
    wrapped, noisy, times, contexts, commands, observation = fixture(profile)
    captured = {}
    def inspect_head(module, args):
        captured['hidden'], captured['time'] = [value.detach().clone() for value in args]
    handle = wrapped.core.head.register_forward_pre_hook(inspect_head)
    try:
        with torch.no_grad():
            expected = torch.stack(wrapped.core(list(noisy.unbind(0)), times, contexts, wrapped.tokens))
    finally:
        handle.remove()
    # The caller's inference context must not create constants that autograd
    # cannot save later, and it must remain active after extraction returns.
    with torch.inference_mode():
        features = wrapped.extract_features(noisy, times, contexts)
        assert torch.is_inference_mode_enabled()
    assert torch.equal(features.hidden, captured['hidden'])
    assert torch.equal(features.time_embedding, captured['time'])
    assert features.time_embedding.dtype == torch.float32
    assert features.grid_sizes.tolist() == [list(wrapped.grid)]
    assert not features.hidden.is_inference()
    assert not features.time_embedding.is_inference()
    assert not features.hidden.requires_grad and features.hidden.grad_fn is None
    actual = wrapped.predict_from_features(features, commands, observation)
    assert torch.equal(actual, expected)
    assert actual.requires_grad
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in wrapped.core.modules())
    bad = times.clone(); bad[0, PROFILES[profile]['prefix'] - 1] = 1
    with pytest.raises(ValueError, match='Observed times'):
        wrapped.extract_features(noisy, bad, contexts)
    bad = times.clone(); bad[0, -1] += 1
    with pytest.raises(ValueError, match='future times'):
        wrapped.extract_features(noisy, bad, contexts)


def test_activated_adapter_paired_gradients_match_direct_literal_head():
    wrapped, noisy, times, contexts, commands, observation = fixture()
    with torch.no_grad():
        wrapped.adapter.output.weight.normal_(0, .07)
        wrapped.adapter.output.bias.normal_(0, .02)
    comparison = copy.deepcopy(wrapped.adapter)
    base_before = {name: parameter.detach().clone() for name, parameter in wrapped.core.named_parameters()}
    features = wrapped.extract_features(noisy, times, contexts)
    alternate = commands.clone(); alternate[0, 0, 5] = 1
    upstream = [torch.randn(1, 2, 4, 18, 32) / 100 for _ in range(2)]
    for action, derivative in zip((commands, alternate), upstream):
        with torch.no_grad():
            prediction = wrapped.predict_from_features(features, action, observation, track_grad=True)
            assert not torch.is_grad_enabled()
        (.5 * (prediction[:, :, 1:] * derivative).sum()).backward()
        direct = comparison(features.hidden, action, observation, wrapped.grid)
        direct = torch.stack(wrapped.core.unpatchify(wrapped.core.head(direct, features.time_embedding), features.grid_sizes))
        (.5 * (direct[:, :, 1:] * derivative).sum()).backward()
        assert torch.equal(prediction, direct)
    for (name, parameter), (other_name, reference) in zip(wrapped.adapter.named_parameters(), comparison.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.equal(parameter.grad, reference.grad), name
    assert sum(parameter.grad.abs().sum() for parameter in wrapped.adapter.command_gru.parameters()) > 0
    assert wrapped.adapter.query.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None and torch.equal(parameter, base_before[name])
               for name, parameter in wrapped.core.named_parameters())
    inference = wrapped.predict_from_features(features, commands, observation, track_grad=False)
    assert not inference.requires_grad and inference.grad_fn is None


def test_sentinel_traceback_release_foreign_exception_and_failure_cleanup():
    wrapped, noisy, times, contexts, commands, observation = fixture()
    recorded = []
    original_boundary = implementation._HeadBoundary
    class RecordedBoundary(original_boundary):
        def __init__(self):
            recorded.append(self)
    with mock.patch.object(implementation, '_HeadBoundary', RecordedBoundary):
        wrapped.extract_features(noisy, times, contexts)
    assert len(recorded) == 1
    assert recorded[0].__traceback__ is None
    # An upstream error must propagate unchanged while our hook and lock clear.
    failure = KeyboardInterrupt('independent native-forward fixture')
    with mock.patch.object(WanModel, 'forward', side_effect=failure):
        with pytest.raises(KeyboardInterrupt) as caught:
            wrapped.extract_features(noisy, times, contexts)
    assert caught.value is failure
    assert not wrapped.core.head._forward_pre_hooks
    assert not wrapped._call_lock.locked()
    foreign = original_boundary()
    with mock.patch.object(WanModel, 'forward', side_effect=foreign):
        with pytest.raises(original_boundary) as caught:
            wrapped.extract_features(noisy, times, contexts)
    assert caught.value is foreign
    assert not wrapped.core.head._forward_pre_hooks
    assert not wrapped._call_lock.locked()
    # A later clean call proves failure did not leave the interception armed.
    assert wrapped.extract_features(noisy, times, contexts).hidden.shape[1] == 720


def test_external_hook_and_changed_observation_are_rejected_without_deletion():
    wrapped, noisy, times, contexts, commands, observation = fixture()
    handle = wrapped.core.head.register_forward_pre_hook(lambda _m, _a: None)
    try:
        with pytest.raises(ValueError, match='no hooks'):
            wrapped.extract_features(noisy, times, contexts)
        assert handle.id in wrapped.core.head._forward_pre_hooks
    finally:
        handle.remove()
    features = wrapped.extract_features(noisy, times, contexts)
    changed = observation.clone(); changed.flatten()[0] += .1
    with pytest.raises(ValueError, match='exact clean initial prefix'):
        wrapped.predict_from_features(features, commands, changed)
