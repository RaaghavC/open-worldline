# SPDX-License-Identifier: Apache-2.0
"""Bounded random-weight CPU checks. No real checkpoint or accelerator access."""
import copy
from dataclasses import replace

import pytest
import torch

from ..portable import create_model, token_times
from .model import NATIVE_PARAMETER_COUNT, PostBlockActionAdapter
from .wrapper import NativeActionWrapper

SMALL = dict(model_type="ti2v", in_dim=4, out_dim=4, dim=32, ffn_dim=64,
             freq_dim=8, text_dim=16, text_len=8, num_heads=4, num_layers=2)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(policy="fp32", batch=2):
    torch.manual_seed(271)
    core = create_model(device="cpu", policy=policy, configuration=SMALL)
    torch.nn.init.normal_(core.head.head.weight, std=.12)
    adapter = PostBlockActionAdapter(32, 4, 8)
    wrapper = NativeActionWrapper(core, adapter, test_only=True)
    noisy = torch.randn(batch, 4, 3, 4, 4)
    observation = noisy[:, :, :1].clone()
    commands = torch.randn(batch, 8, 6) * .1
    commands[:, :, 5] = torch.randint(0, 2, (batch, 8)).float()
    contexts = [torch.randn(3+i, 16) for i in range(batch)]
    times = token_times(torch.tensor([[3, 2, 2]]*batch), torch.arange(batch)*200+500, 12)
    return wrapper, noisy, times, contexts, commands, observation


def predict(case):
    w, x, t, context, commands, obs = case
    return w(x, t, context, commands=commands, observation=obs, seq_len=12)


def activate(adapter):
    with torch.no_grad():
        adapter.output.weight.normal_(0, .025)
        adapter.output.bias.normal_(0, .025)


def test_native_parameter_count_and_all_fp32():
    adapter = PostBlockActionAdapter()
    assert sum(p.numel() for p in adapter.parameters()) == NATIVE_PARAMETER_COUNT == 947_712
    assert all(p.dtype == torch.float32 and p.requires_grad for p in adapter.parameters())
    assert torch.count_nonzero(adapter.output.weight) == torch.count_nonzero(adapter.output.bias) == 0


@pytest.mark.parametrize("policy", ["fp32", "selective_bf16"])
def test_exact_zero_identity_to_untouched_native_core(policy):
    case = example(policy)
    w, x, t, contexts, commands, obs = case
    original_methods = [m.forward for m in w.core.modules()]
    expected = torch.stack(w.core(list(x.unbind(0)), t, contexts, 12))
    actual = predict(case)
    assert torch.count_nonzero(w.core.head.head.weight) > 0
    assert torch.equal(expected, actual) and actual.dtype == torch.float32
    altered = commands.clone()
    altered[:, :, 3] *= -1
    altered[:, :, 5] = 1-altered[:, :, 5]
    assert torch.equal(actual, w(x, t, contexts, commands=altered, observation=obs, seq_len=12))
    assert [m.forward for m in w.core.modules()] == original_methods
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in w.core.modules())


def test_encoder_retains_order_and_is_causal_with_fresh_per_call_state():
    torch.manual_seed(729)
    a = PostBlockActionAdapter(32, 4, 8)
    commands = torch.zeros(2, 16, 6)
    commands[:, 0, 3], commands[:, 1, 5] = .1, 1
    original = a.command_prefix(commands)
    assert torch.equal(original[:, 0], torch.zeros(2, 8))
    changed = commands.clone()
    changed[:, -4:, 3] = .3
    later = a.command_prefix(changed)
    assert torch.equal(later[:, :4], original[:, :4]) and not torch.equal(later[:, 4], original[:, 4])
    reversed_order = commands.clone()
    reversed_order[:, :2] = commands[:, [1, 0]]
    assert torch.equal(commands.sum(1), reversed_order.sum(1))
    assert not torch.equal(a.command_prefix(reversed_order)[:, 1], original[:, 1])
    removed_toggle = commands.clone()
    removed_toggle[:, 1, 5] = 0
    assert not torch.equal(a.command_prefix(removed_toggle)[:, -1], original[:, -1])
    assert torch.equal(a.command_prefix(commands), original)
    # Fixed-batch session isolation is exact. Changing GEMM batch shape may
    # change FP32 rounding, so the separate B=1 comparison has a numeric bound.
    another_session = commands.clone()
    another_session[1, :, 3] += .4
    assert torch.equal(a.command_prefix(another_session)[:1], original[:1])
    torch.testing.assert_close(a.command_prefix(commands[:1]), original[:1], atol=1e-7, rtol=1e-6)
    differentiable = commands.clone().requires_grad_(True)
    a.command_prefix(differentiable)[:, 2].sum().backward()
    assert torch.count_nonzero(differentiable.grad[:, :8]) > 0
    assert torch.count_nonzero(differentiable.grad[:, 8:]) == 0


def test_residual_masks_observed_tokens_and_padding_without_mutation():
    torch.manual_seed(92)
    a = PostBlockActionAdapter(32, 4, 8)
    activate(a)
    features = torch.randn(2, 15, 32)
    commands = torch.zeros(2, 8, 6)
    obs = torch.randn(2, 4, 1, 4, 4)
    before = [v.clone() for v in (features, commands, obs)]
    output = a(features, commands, obs, (3, 2, 2))
    assert torch.equal(output[:, :4], features[:, :4])
    assert torch.equal(output[:, 12:], features[:, 12:])
    assert not torch.equal(output[:, 4:12], features[:, 4:12])
    assert all(torch.equal(x, y) for x, y in zip(before, (features, commands, obs)))


@pytest.mark.parametrize("policy", ["fp32", "selective_bf16"])
def test_nonzero_head_keeps_adapter_gradient_path_and_frozen_prefix(policy):
    case = example(policy)
    w, x, t, contexts, commands, obs = case
    base_before = {name: p.detach().clone() for name, p in w.core.named_parameters()}
    x.requires_grad_(True)
    contexts[0].requires_grad_(True)
    w.train()
    assert not w.core.training and w.adapter.training
    optimizer = torch.optim.SGD(w.adapter.parameters(), lr=.2)
    target = torch.randn(2, 4, 2, 4, 4)
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        features = w.extract_features(x, t, contexts, 12)
        assert all(not value.requires_grad and value.grad_fn is None and not value.is_inference()
                   for value in (features.hidden, features.time_embedding, features.observed_prefix))
        output = w.predict_from_features(features, commands, obs)
        assert output.requires_grad
        loss = (output[:, :, 1:]-target).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in w.adapter.parameters())
        assert w.adapter.output.weight.grad.abs().sum() > 0
        gru_norm = sum(p.grad.square().sum() for p in w.adapter.command_gru.parameters())
        assert (gru_norm == 0) if update == 0 else (gru_norm > 0)
        if update == 1:
            for module in (w.adapter.action, w.adapter.observation, w.adapter.query, w.adapter.key, w.adapter.value):
                assert sum(p.grad.square().sum() for p in module.parameters()) > 0
        optimizer.step()
    assert x.grad is None and contexts[0].grad is None
    assert all(p.grad is None and torch.equal(p, base_before[name]) for name, p in w.core.named_parameters())


@pytest.mark.parametrize("policy", ["fp32", "selective_bf16"])
def test_sequential_half_losses_equal_batch_pair_gradients(policy):
    case = example(policy)
    w, x, t, contexts, commands, obs = case
    activate(w.adapter)
    single = example(policy)[0]
    single.load_state_dict(copy.deepcopy(w.state_dict()))
    truth = torch.randn(2, 4, 2, 4, 4)
    batch_loss = (predict(case)[:, :, 1:]-truth).square().mean()
    batch_loss.backward()
    branch_losses = []
    for index in range(2):
        result = single(x[index:index+1], t[index:index+1], [contexts[index]],
                        commands=commands[index:index+1], observation=obs[index:index+1], seq_len=12)
        loss = (result[:, :, 1:]-truth[index:index+1]).square().mean()/2
        branch_losses.append(loss.detach())
        loss.backward()
    torch.testing.assert_close(sum(branch_losses), batch_loss.detach(), atol=2e-6, rtol=2e-6)
    for (_, a), (_, b) in zip(w.adapter.named_parameters(), single.adapter.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)


def test_disabled_external_autocast_and_normal_features_from_inference_caller():
    case = example("selective_bf16")
    w, x, t, contexts, commands, obs = case
    activate(w.adapter)
    expected = predict(case)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = predict(case)
    assert torch.equal(expected, actual) and actual.dtype == torch.float32
    with torch.inference_mode():
        features = w.extract_features(x, t, contexts, 12)
    assert not features.hidden.is_inference()
    w.predict_from_features(features, commands, obs).square().mean().backward()
    assert w.adapter.output.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("field", ["noisy_shape", "noisy_nan", "command_shape", "command_nan", "pulse",
    "observation_mismatch", "time_dtype", "time_prefix", "time_future", "context_nan", "context_shape", "adapter_dtype"])
def test_invalid_inputs_are_rejected_before_frozen_extraction(field):
    case = list(example())
    w, x, t, contexts, commands, obs = case
    if field == "noisy_shape": case[1] = x[:, :3]
    elif field == "noisy_nan": x[0, 0, 1, 0, 0] = float("nan")
    elif field == "command_shape": case[4] = commands[:, :-1]
    elif field == "command_nan": commands[0, 0, 0] = float("nan")
    elif field == "pulse": commands[0, 0, 5] = .5
    elif field == "observation_mismatch": obs[0, 0, 0, 0, 0] += 1
    elif field == "time_dtype": case[2] = t.float()
    elif field == "time_prefix": t[0, 0] = 500
    elif field == "time_future": t[0, -1] = 499
    elif field == "context_nan": contexts[0][0, 0] = float("nan")
    elif field == "context_shape": contexts[0] = contexts[0][:, :15]
    elif field == "adapter_dtype": w.adapter.bfloat16()
    calls = []
    original = w.extract_features
    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    w.extract_features = spy
    with pytest.raises(ValueError): predict(case)
    assert not calls


def test_feature_ownership_and_metadata_rejection():
    w, x, t, contexts, commands, obs = example()
    features = w.extract_features(x, t, contexts, 12)
    other = example()[0]
    with pytest.raises(ValueError, match="another wrapper"):
        other.predict_from_features(features, commands, obs)
    with pytest.raises(ValueError, match="sequence lengths"):
        w.predict_from_features(replace(features, sequence_lengths=torch.tensor([11, 12])), commands, obs)
    with pytest.raises(ValueError, match="non-gradient"):
        w.predict_from_features(replace(features, hidden=features.hidden.clone().requires_grad_(True)), commands, obs)


def test_native_default_rejects_tiny_core_and_unfrozen_head():
    w, *_ = example()
    with pytest.raises(ValueError, match="exact native5B"):
        NativeActionWrapper(w.core, w.adapter)
    w.core.head.head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        NativeActionWrapper(w.core, w.adapter, test_only=True)
