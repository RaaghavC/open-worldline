# SPDX-License-Identifier: Apache-2.0
"""CPU equation checks for the isolated MPS observation-pooling implementation."""
import copy
from unittest import mock

import pytest
import torch
from torch.nn import functional as F

from . import model, pooling


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape,output_size", [
    ((1, 48, 18, 32), (4, 8)),
    ((2, 3, 18, 31), (4, 8)),
    ((2, 2, 13, 19), (4, 8)),
    ((1, 2, 2, 3), (4, 8)),
    ((2, 4, 16, 24), (4, 8)),
])
def test_values_and_input_gradients_match_cpu_adaptive_oracle(shape, output_size):
    generator = torch.Generator().manual_seed(20260907)
    source = torch.randn(shape, generator=generator)
    explicit = source.clone().requires_grad_()
    oracle = source.clone().requires_grad_()
    upstream = torch.randn((*shape[:2], *output_size), generator=generator) * .25
    actual = pooling.adaptive_bin_avg_pool2d(explicit, output_size)
    expected = F.adaptive_avg_pool2d(oracle, output_size)
    actual_gradient, = torch.autograd.grad(actual, explicit, upstream)
    expected_gradient, = torch.autograd.grad(expected, oracle, upstream)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=3e-6)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-7, rtol=3e-6)
    assert actual.dtype == source.dtype and actual.device == source.device
    assert torch.equal(explicit.detach(), source) and torch.equal(oracle.detach(), source)
    assert torch.count_nonzero(actual_gradient) > 0


def test_overlapping_height_bins_reuse_rows_four_and_thirteen():
    source = torch.zeros(1, 1, 18, 32, requires_grad=True)
    with torch.no_grad():
        source[0, 0, 4, 0] = 1
        source[0, 0, 13, 0] = 2
    output = pooling.adaptive_bin_avg_pool2d(source, (4, 8))
    expected = torch.zeros(1, 1, 4, 8)
    expected[0, 0, :, 0] = torch.tensor([1/20, 1/20, 2/20, 2/20])
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    gradient, = torch.autograd.grad(output.sum(), source)
    expected_gradient = torch.full_like(source, 1/20)
    expected_gradient[:, :, (4, 13), :] = 2/20
    torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)


def test_noncontiguous_inputs_batch_isolation_and_no_adaptive_kernel_call():
    source = torch.randn(2, 3, 31, 18).transpose(-2, -1).requires_grad_()
    assert not source.is_contiguous()
    expected = F.adaptive_avg_pool2d(source, (4, 8))
    with mock.patch.object(F, 'adaptive_avg_pool2d', side_effect=AssertionError('Unsupported kernel must not be called')):
        actual = pooling.adaptive_bin_avg_pool2d(source, (4, 8))
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=3e-6)
    gradient, = torch.autograd.grad(actual[0].square().sum(), source)
    assert torch.count_nonzero(gradient[0]) > 0 and torch.count_nonzero(gradient[1]) == 0
    changed = source.detach().clone()
    changed[1] += 7
    assert torch.equal(pooling.adaptive_bin_avg_pool2d(changed, (4, 8))[0], actual[0])


def test_cpu_dispatch_keeps_original_kernel_and_exact_values():
    source = torch.randn(1, 48, 18, 32, requires_grad=True)
    original = F.adaptive_avg_pool2d
    with mock.patch.object(pooling, 'adaptive_bin_avg_pool2d', side_effect=AssertionError('CPU path changed')):
        with mock.patch.object(F, 'adaptive_avg_pool2d', wraps=original) as wrapped:
            actual = pooling.observation_pool2d(source, (4, 8))
    wrapped.assert_called_once_with(source, (4, 8))
    assert torch.equal(actual, original(source, (4, 8)))


def test_full_adapter_cpu_equations_and_gradients_with_explicit_pool():
    torch.manual_seed(315)
    reference = model.PostBlockActionAdapter(hidden_dim=12, observation_channels=48, width=8)
    with torch.no_grad():
        reference.output.weight.normal_(std=.03)
        reference.output.bias.normal_(std=.01)
    portable = copy.deepcopy(reference)
    features = torch.randn(1, 720, 12)
    commands = torch.zeros(1, 16, 6)
    commands[0, 0, 5] = 1
    observation = torch.randn(1, 48, 1, 18, 32)
    reference_observation = observation.clone().requires_grad_()
    portable_observation = observation.clone().requires_grad_()
    upstream = torch.randn_like(features) / 128
    expected = reference(features, commands, reference_observation, (5, 9, 16))
    with mock.patch.object(model, 'observation_pool2d', pooling.adaptive_bin_avg_pool2d):
        actual = portable(features, commands, portable_observation, (5, 9, 16))
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=3e-6)
    torch.testing.assert_close(portable_observation.grad, reference_observation.grad, atol=2e-7, rtol=3e-6)
    for (name, a), (other_name, b) in zip(reference.named_parameters(), portable.named_parameters()):
        assert name == other_name and a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=3e-6)
    assert reference.observation.weight.grad.abs().sum() > 0
    assert torch.equal(actual[:, :144], features[:, :144])


@pytest.mark.parametrize("value,size", [
    (torch.zeros(1, 2, 18, 32, dtype=torch.float64), (4, 8)),
    (torch.zeros(2, 18, 32), (4, 8)),
    (torch.zeros(1, 2, 0, 32), (4, 8)),
    (torch.zeros(1, 2, 18, 32), (True, 8)),
    (torch.zeros(1, 2, 18, 32), (4, 0)),
    (torch.zeros(1, 2, 18, 32), None),
])
def test_invalid_explicit_pool_contract(value, size):
    with pytest.raises(ValueError):
        pooling.adaptive_bin_avg_pool2d(value, size)
