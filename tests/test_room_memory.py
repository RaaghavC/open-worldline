# SPDX-License-Identifier: Apache-2.0
"""CPU software checks with synthetic RGB; no new scenes or quality scores."""
import hashlib
from pathlib import Path

import pytest
import torch

from experiments.room_world.memory_model import RoomMemoryModel
from experiments.room_world.model import RGBModel, load_model


CHECKPOINT = Path(__file__).resolve().parents[1] / "experiments/room_world/artifacts/predictor/model.pt"
CHECKPOINT_SHA256 = "9376e684e3abd103313d13d04bd4a1afc1b202fccb7e7b6fe133e7232a486d69"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    base, _ = load_model(CHECKPOINT, device="cpu")
    torch.manual_seed(8201)
    return RoomMemoryModel(base)


def random_history(batch=1, size=12, seed=44):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(batch, 4, 3, size, size, generator=generator) * 0.5 - 0.25


def update_condition_once(model):
    """One synthetic loss update opens the zero-initialized conditioning path."""
    model.train()
    optimizer = torch.optim.SGD(model.memory_parameters(), lr=0.1)
    prediction, _ = model(random_history(batch=2), torch.tensor([1, 5]), model.initial_state(2))
    prediction.square().mean().backward()
    assert model.condition.weight.grad.abs().sum() > 0
    # The zero output map intentionally blocks GRU gradients on this first loss.
    assert all(parameter.grad is not None and not parameter.grad.count_nonzero()
               for parameter in model.gru.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert model.condition.weight.count_nonzero() > 0
    return optimizer


def test_actual_dimensions_counts_and_new_session_state(model):
    assert model.parameter_counts() == {
        "base": 498651, "memory": 24960, "trainable": 24960,
        "state_elements_per_session": 32, "state_bytes_per_session": 128,
    }
    assert model.gru.input_size == 160 and model.gru.hidden_size == 32
    assert model.condition.in_features == 32 and model.condition.out_features == 192
    assert sum(p.numel() for p in model.parameters()) == 523611
    first, second = model.initial_state(2), model.initial_state(2)
    assert first.shape == (2, 32) and first.dtype == torch.float32
    assert not first.count_nonzero() and not second.count_nonzero()
    assert first.data_ptr() != second.data_ptr()
    model.train()
    assert not model.base.training and model.gru.training
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())


@pytest.mark.parametrize("mode", ["carry", "reset"])
def test_zero_conditioning_exactly_matches_published_predictor(model, mode):
    history = random_history(batch=2, size=20)
    actions = torch.tensor([0, 5])
    state = torch.randn(2, 32)
    before = state.clone()
    with torch.no_grad():
        reference = model.base(history, actions)
        prediction, after = model(history, actions, state, mode=mode)
    assert torch.equal(reference, prediction)
    assert torch.equal(state, before)
    assert after.shape == state.shape and torch.isfinite(after).all()
    assert after.data_ptr() != state.data_ptr()


def test_condition_update_opens_gru_gradients_and_keeps_base_frozen(model):
    base_before = {name: tensor.clone() for name, tensor in model.base.state_dict().items()}
    optimizer = update_condition_once(model)
    state = model.initial_state(2)
    _, state = model(random_history(batch=2), torch.tensor([5, 0]), state)
    prediction, _ = model(random_history(batch=2, seed=45), torch.tensor([3, 4]), state)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.gru.parameters())
    assert all(p.grad.abs().sum() > 0 for p in model.gru.parameters())
    optimizer.step()
    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
    assert all(torch.equal(tensor, base_before[name]) for name, tensor in model.base.state_dict().items())
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == CHECKPOINT_SHA256


def test_final_frame_gradient_reaches_first_rgb_and_state_across_65_steps(model):
    update_condition_once(model)
    first_rgb = random_history(size=8).requires_grad_()
    first_state = model.initial_state(1).requires_grad_()
    state = first_state
    later_rgb = random_history(size=8, seed=95)
    # Later RGB is independent of the first RGB. Only recurrence connects them.
    for step in range(65):
        prediction, state = model(first_rgb if step == 0 else later_rgb,
                                  torch.tensor([5 if step == 0 else 0]), state)
    prediction.square().mean().backward()
    for value in (first_rgb, first_state):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())


def test_reset_arm_removes_temporal_state_path_with_identical_parameters(model):
    update_condition_once(model)
    parameters_before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    first_rgb = random_history(size=8).requires_grad_()
    first_state = model.initial_state(1).requires_grad_()
    state = first_state
    for step in range(5):
        prediction, state = model(first_rgb if step == 0 else random_history(size=8, seed=95),
                                  torch.tensor([5 if step == 0 else 0]), state, mode="reset")
    prediction.square().mean().backward()
    assert first_rgb.grad is None and first_state.grad is None
    assert all(torch.equal(tensor, parameters_before[name]) for name, tensor in model.state_dict().items())
    history, action = random_history(size=8), torch.tensor([1])
    with torch.no_grad():
        first, first_h = model(history, action, model.initial_state(1), mode="reset")
        other, other_h = model(history, action, torch.randn(1, 32), mode="reset")
    assert torch.equal(first, other) and torch.equal(first_h, other_h)
    assert model.parameter_counts()["trainable"] == 24960


def test_cloned_state_branches_are_independent_and_preserve_gradients(model):
    update_condition_once(model)
    initial = model.initial_state(1).requires_grad_()
    _, parent = model(random_history(), torch.tensor([5]), initial)
    saved = parent.detach().clone()
    left, right = model.clone_state(parent), model.clone_state(parent)
    assert len({parent.data_ptr(), left.data_ptr(), right.data_ptr()}) == 3
    assert left.grad_fn is not None and right.grad_fn is not None
    detached = model.clone_state(parent, detach=True)
    assert detached.grad_fn is None and not detached.requires_grad
    history = random_history(seed=46)
    left_rgb, left_after = model(history, torch.tensor([1]), left)
    right_rgb, right_after = model(history, torch.tensor([4]), right)
    assert not torch.equal(left_after, right_after)
    assert torch.equal(parent.detach(), saved) and torch.equal(right.detach(), saved)
    with torch.no_grad():
        left.add_(1)
        repeat_rgb, repeat_after = model(history, torch.tensor([4]), right)
    assert torch.equal(right.detach(), saved) and torch.equal(parent.detach(), saved)
    assert torch.equal(right_rgb.detach(), repeat_rgb) and torch.equal(right_after.detach(), repeat_after)
    # Forking the state itself must not detach the earlier temporal graph.
    right_rgb.square().mean().backward()
    assert initial.grad is not None and initial.grad.abs().sum() > 0


def test_forward_requires_explicit_rgb_action_state_only(model):
    history, action, state = random_history(), torch.tensor([0]), model.initial_state(1)
    with pytest.raises(TypeError):
        model(history, action, state, door_open=True)
    with pytest.raises(TypeError):
        model(history, action, state, camera_pose=torch.eye(4))
    with pytest.raises(TypeError):
        model(history, action)
    for bad_action in (torch.tensor([1.0]), torch.tensor([6]), torch.tensor([-1])):
        with pytest.raises(ValueError):
            model(history, bad_action, state)
    for bad_history in (history[:, :3], history.double(), history * float("nan"), history + 2):
        with pytest.raises(ValueError):
            model(bad_history, action, state)
    for bad_state in (None, torch.zeros(1, 31), state.double(), state + float("inf")):
        with pytest.raises(ValueError):
            model(history, action, bad_state)
    with pytest.raises(ValueError):
        model(history, action, state, mode="other")
    for batch_size in (0, -1, True, 1.0):
        with pytest.raises(ValueError):
            model.initial_state(batch_size)


def test_only_the_declared_existing_predictor_architecture_is_accepted():
    for base in (RGBModel("flow", width=24), RGBModel("predictor", width=8), object()):
        with pytest.raises(ValueError, match="width24"):
            RoomMemoryModel(base)
