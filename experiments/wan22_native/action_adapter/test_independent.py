# SPDX-License-Identifier: Apache-2.0
"""Independent tiny CPU equations and live-gradient checks. No real weights."""
import copy
import math

import pytest
import torch
from torch.nn import functional as F

from ..portable import create_model
from .model import PostBlockActionAdapter
from .wrapper import NativeActionWrapper


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def explicit_prefix(adapter, commands):
    """Expand GRU gates directly, without using GRUCell or command_prefix."""
    states = [torch.zeros(commands.shape[0], adapter.width)]
    for start in range(0, commands.shape[1], 4):
        # Explicitly enumerate time then channel, making the 24-value order clear.
        group = torch.stack([commands[:, t, c] for t in range(start, start+4)
                             for c in range(6)], dim=1)
        embedded = F.linear(F.silu(F.linear(group, adapter.action[0].weight,
                                            adapter.action[0].bias)),
                            adapter.action[2].weight, adapter.action[2].bias)
        gru = adapter.command_gru
        ir, iz, inn = F.linear(embedded, gru.weight_ih, gru.bias_ih).chunk(3, -1)
        hr, hz, hn = F.linear(states[-1], gru.weight_hh, gru.bias_hh).chunk(3, -1)
        reset, update = torch.sigmoid(ir+hr), torch.sigmoid(iz+hz)
        candidate = torch.tanh(inn+reset*hn)
        states.append((1-update)*candidate+update*states[-1])
    return torch.stack(states, 1)


def activated_adapter():
    torch.manual_seed(92013)
    adapter = PostBlockActionAdapter(12, 4, 8)
    with torch.no_grad():
        adapter.output.weight.normal_(0, .2)
        adapter.output.bias.normal_(0, .1)
    return adapter


def tiny_wrapper(policy):
    torch.manual_seed(92014)
    configuration = dict(model_type="ti2v", in_dim=4, out_dim=4, dim=32,
                         ffn_dim=64, freq_dim=8, text_dim=16, text_len=8,
                         num_heads=4, num_layers=2)
    core = create_model(device="cpu", policy=policy, configuration=configuration)
    with torch.no_grad():
        core.head.head.weight.normal_(0, .1)
        core.head.head.bias.normal_(0, .01)
    wrapper = NativeActionWrapper(core, PostBlockActionAdapter(32, 4, 8), test_only=True)
    noisy = torch.randn(1, 4, 3, 4, 4)
    times = torch.tensor([[0]*4+[731]*8], dtype=torch.int64)
    context = [torch.randn(5, 16)]
    command = torch.zeros(1, 8, 6)
    command[:, 1:, 3] = .1
    return wrapper, noisy, times, context, command, noisy[:, :, :1].clone()


def test_manual_gru_attention_and_fhw_mask_equations():
    adapter = activated_adapter()
    commands = torch.randn(2, 8, 6)*.1
    commands[..., 5] = 0
    commands[1, 0, 5] = 1
    features = torch.randn(2, 99, 12)  # 96 valid tokens, then three padding tokens.
    observation = torch.randn(2, 4, 1, 8, 16)
    prefix = explicit_prefix(adapter, commands)
    torch.testing.assert_close(adapter.command_prefix(commands), prefix, atol=4e-8, rtol=1e-6)

    # Every 4x8 observed token is the mean of its original 2x2 spatial cell.
    pooled = observation[:, :, 0].reshape(2, 4, 4, 2, 8, 2).mean((3, 5))
    observed = F.linear(pooled.flatten(2).transpose(1, 2),
                        adapter.observation.weight, adapter.observation.bias)
    q = F.linear(features, adapter.query.weight, adapter.query.bias)
    k = F.linear(observed, adapter.key.weight, adapter.key.bias)
    v = F.linear(observed, adapter.value.weight, adapter.value.bias)
    attended = torch.softmax(q@k.transpose(-2, -1)/math.sqrt(8), -1)@v
    expected = features.clone()
    for frame in range(1, 3):
        for row in range(4):
            for column in range(8):
                index = frame*32+row*8+column
                residual = F.linear(F.silu(prefix[:, frame]+attended[:, index]),
                                    adapter.output.weight, adapter.output.bias)
                expected[:, index] += residual
    actual = adapter(features, commands, observation, (3, 4, 8))
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-6)
    assert torch.equal(actual[:, :32], features[:, :32])
    assert torch.equal(actual[:, 96:], features[:, 96:])


def test_later_commands_cannot_enter_earlier_direct_residual_or_its_gradient():
    adapter = activated_adapter()
    features = torch.randn(2, 36, 12)
    observation = torch.randn(2, 4, 1, 4, 6)
    commands = torch.zeros(2, 20, 6)
    commands[:, 0, 5] = 1
    commands.requires_grad_()
    baseline = adapter(features, commands, observation, (6, 2, 3))
    altered = commands.detach().clone()
    altered[:, 8:, 0] = 7
    altered[:, 8:, 5] = 1
    changed = adapter(features, altered, observation, (6, 2, 3))
    assert torch.equal(baseline[:, :18], changed[:, :18])
    assert not torch.equal(baseline[:, 18:], changed[:, 18:])
    gradient, = torch.autograd.grad(baseline[:, 12:18].square().sum(), commands)
    assert gradient[:, :8].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, 8:]) == 0
    # Changing another batch member cannot change the first member's residual.
    changed_session = commands.detach().clone()
    changed_session[1, :, 4] = 2
    assert torch.equal(adapter(features, changed_session, observation, (6, 2, 3))[0], baseline[0])


@pytest.mark.parametrize("policy", ["fp32", "selective_bf16"])
def test_live_two_branch_updates_preserve_base_and_reach_command_gru(policy):
    wrapper, noisy, times, contexts, closed, observation = tiny_wrapper(policy)
    opened = closed.clone()
    opened[:, 0, 5] = 1
    base_before = {name: value.detach().clone() for name, value in wrapper.core.named_parameters()}
    native = torch.stack(wrapper.core(list(noisy.unbind(0)), times, contexts, 12))
    zero = wrapper(noisy, times, contexts, commands=closed, observation=observation, seq_len=12)
    assert torch.equal(zero, native)
    assert torch.equal(zero, wrapper(noisy, times, contexts, commands=opened,
                                     observation=observation, seq_len=12))
    events = []
    handles = [block.register_forward_hook(lambda _m, _a, result:
                events.append(("block", torch.is_grad_enabled(), result.requires_grad)))
               for block in wrapper.core.blocks]
    handles.append(wrapper.core.head.register_forward_hook(lambda _m, _a, result:
                   events.append(("head", torch.is_grad_enabled(), result.requires_grad))))
    optimizer = torch.optim.AdamW(wrapper.adapter.parameters(), lr=.01)
    targets = [torch.randn(1, 4, 2, 4, 4), torch.randn(1, 4, 2, 4, 4)]
    try:
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            for commands, truth in zip((closed, opened), targets):
                # Each branch gets a fresh live graph and exactly half the pair loss.
                prediction = wrapper(noisy, times, contexts, commands=commands,
                                     observation=observation, seq_len=12)
                loss = .5*(prediction[:, :, 1:]-truth).square().mean()
                loss.backward()
                del prediction, loss
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in wrapper.adapter.parameters())
            gru_gradient = sum(p.grad.square().sum() for p in wrapper.adapter.command_gru.parameters())
            assert (gru_gradient == 0) if update == 0 else (gru_gradient > 0)
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in wrapper.adapter.parameters())
        assert sum(int(state["step"].item()) == 2 for state in optimizer.state.values()) == len(list(wrapper.adapter.parameters()))
    finally:
        for handle in handles:
            handle.remove()
    assert len(events) == 12
    assert all(enabled == requires == (kind == "head") for kind, enabled, requires in events)
    assert all(value.grad is None and torch.equal(value, base_before[name])
               for name, value in wrapper.core.named_parameters())


@pytest.mark.parametrize("policy", ["fp32", "selective_bf16"])
def test_live_sequential_pair_common_upstream_matches_batched_gradient(policy):
    wrapper, noisy, times, contexts, commands, observation = tiny_wrapper(policy)
    with torch.no_grad():
        wrapper.adapter.output.weight.normal_(0, .025)
    batched = copy.deepcopy(wrapper)
    paired_commands = commands.repeat(2, 1, 1)
    paired_commands[1, 0, 5] = 1
    upstream = torch.randn(2, 4, 2, 4, 4)/128
    for branch in range(2):
        prediction = wrapper(noisy, times, contexts, commands=paired_commands[branch:branch+1],
                             observation=observation, seq_len=12)
        (.5*(prediction[:, :, 1:]*upstream[branch:branch+1]).sum()).backward()
    prediction = batched(noisy.repeat(2, 1, 1, 1, 1), times.repeat(2, 1), contexts*2,
                         commands=paired_commands, observation=observation.repeat(2, 1, 1, 1, 1), seq_len=12)
    (.5*(prediction[:, :, 1:]*upstream).sum()).backward()
    for (name, value), (other_name, other) in zip(wrapper.adapter.named_parameters(), batched.adapter.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(value.grad, other.grad, atol=2e-7, rtol=2e-5)
