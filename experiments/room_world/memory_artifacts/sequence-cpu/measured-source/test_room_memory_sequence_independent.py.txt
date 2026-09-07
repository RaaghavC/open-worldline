# SPDX-License-Identifier: Apache-2.0
"""Independent CPU checks for the separate observed-history batching path."""
from pathlib import Path

import pytest
import torch

from experiments.room_world.memory_evaluate import history_at
from experiments.room_world.memory_model import RoomMemoryModel
from experiments.room_world.memory_sequence import observed_sequence, sequence_loss_batched
from experiments.room_world.memory_train import sequence_loss
from experiments.room_world.model import load_model


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def active_model():
    base, _ = load_model(Path(__file__).resolve().parents[1] / 'experiments/room_world/artifacts/predictor/model.pt', device='cpu')
    torch.manual_seed(708)
    model = RoomMemoryModel(base)
    with torch.no_grad():
        model.condition.weight.normal_(0, .015)
        model.condition.bias.normal_(0, .005)
    return model


@pytest.mark.parametrize('mode', ['carry', 'reset'])
def test_full_65_step_loss_and_all_memory_rgb_gradients_match_stepwise_path(mode):
    stepwise, batched = active_model(), active_model()
    torch.manual_seed(84)
    rgb = torch.rand(2, 66, 3, 8, 8) * .4 - .2
    first, second = rgb.clone().requires_grad_(), rgb.clone().requires_grad_()
    actions = torch.stack([torch.arange(65) % 6, (torch.arange(65) + 3) % 6]).long()
    before = {name: value.clone() for name, value in batched.state_dict().items()}
    ordinary = sequence_loss(stepwise, first, actions, mode)
    batch_loss = sequence_loss_batched(batched, second, actions, mode)
    torch.testing.assert_close(ordinary, batch_loss, atol=1e-7, rtol=1e-6)
    ordinary.backward()
    batch_loss.backward()
    for (name, p), (other_name, q) in zip(stepwise.named_parameters(), batched.named_parameters()):
        assert name == other_name
        if name.startswith('base.'):
            assert p.grad is None and q.grad is None and not p.requires_grad and not q.requires_grad
        else:
            assert p.grad is not None and q.grad is not None
            assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
            torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=3e-4)
            if mode == 'reset' and name == 'gru.weight_hh':
                assert p.grad.count_nonzero() == q.grad.count_nonzero() == 0
            else:
                assert p.grad.abs().sum() > 0 and q.grad.abs().sum() > 0
    torch.testing.assert_close(first.grad, second.grad, atol=2e-8, rtol=3e-4)
    assert first.grad[:, :65].abs().sum() > 0
    assert all(torch.equal(value, before[name]) for name, value in batched.state_dict().items())
    assert not batched.base.training


@pytest.mark.parametrize('mode', ['carry', 'reset'])
def test_no_future_rgb_or_other_batch_session_enters_earlier_predictions(mode):
    model = active_model().eval()
    torch.manual_seed(219)
    observations = (torch.rand(2, 65, 3, 8, 8) * .4 - .2).requires_grad_()
    actions = (torch.arange(130).reshape(2, 65) % 6).long()
    initial = torch.rand(2, 32).requires_grad_()
    saved_rgb, saved_state, saved_actions = observations.detach().clone(), initial.detach().clone(), actions.clone()
    predictions, final_state = observed_sequence(model, observations, actions, initial, mode=mode)
    changed_rgb, changed_actions, changed_state = saved_rgb.clone(), saved_actions.clone(), saved_state.clone()
    changed_rgb[0, 29:] = -.85
    changed_actions[0, 29:] = (changed_actions[0, 29:] + 1) % 6
    changed_rgb[1] = .9
    changed_actions[1] = (changed_actions[1] + 2) % 6
    changed_state[1] += 5
    with torch.no_grad():
        other, _ = observed_sequence(model, changed_rgb, changed_actions, changed_state, mode=mode)
    torch.testing.assert_close(predictions[0, :29], other[0, :29], atol=0, rtol=0)
    gradient = torch.autograd.grad(predictions[0, 28].square().sum(), observations, retain_graph=True)[0]
    assert gradient[0, :29].abs().sum() > 0
    assert gradient[0, 29:].count_nonzero() == gradient[1].count_nonzero() == 0
    start_gradient = torch.autograd.grad(final_state[0].sum(), initial, allow_unused=True, retain_graph=True)[0]
    if mode == 'carry':
        assert start_gradient is not None and start_gradient[0].abs().sum() > 0
        assert start_gradient[1].count_nonzero() == 0
    else:
        assert start_gradient is None
    assert torch.equal(observations.detach(), saved_rgb) and torch.equal(initial.detach(), saved_state)
    assert torch.equal(actions, saved_actions)


def test_encoder_time_batch_order_and_final_state_match_explicit_calls():
    model = active_model().eval()
    observations = torch.arange(130).reshape(2, 65, 1, 1, 1).expand(-1, -1, 3, 8, 8).float() / 200 - .3
    actions = (torch.arange(130).reshape(2, 65) % 6).long()
    captured = {}

    def capture(_, args):
        captured['histories'] = args[0].detach().clone()
        captured['conditions'] = args[1].detach().clone()

    hook = model.base.a.register_forward_pre_hook(capture)
    with torch.no_grad():
        together, state = observed_sequence(model, observations, actions)
    hook.remove()
    for step in range(65):
        for batch in range(2):
            index = step * 2 + batch
            expected = observations[batch, torch.arange(step - 3, step + 1).clamp_min(0)]
            assert torch.equal(captured['histories'][index], expected.reshape(12, 8, 8))
            assert torch.equal(captured['conditions'][index], model.base.action(actions[batch, step]))
    with torch.no_grad():
        old, frames = model.initial_state(2), []
        for step in range(65):
            predicted, old = model(history_at(observations, step), actions[:, step], old)
            frames.append(predicted)
    torch.testing.assert_close(together, torch.stack(frames, 1), atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(state, old, atol=2e-6, rtol=2e-5)
