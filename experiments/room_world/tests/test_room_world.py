import numpy as np
import pytest
import torch

from experiments.room_world.data import make_dataset, sample_batch
from experiments.room_world.model import RGBModel, load_model, rollout
from experiments.room_world.simulator import Room
from experiments.room_world.train import training_loss


def test_seeded_room_images_and_actions():
    first, second = Room(19, 27, 16), Room(19, 27, 16)
    before = first.render()
    assert before.shape == (3, 16, 16) and before.dtype == np.uint8
    assert np.array_equal(before, second.render())
    assert before.std() > 10
    first.step(5)
    assert first.door_open
    opened = first.render()
    assert not np.array_equal(before, opened)
    for _ in range(8):
        first.step(0)
    assert first.door_open and np.array_equal(opened, first.render())
    first.step(5)
    assert not first.door_open and np.array_equal(before, first.render())


def test_motion_changes_pixels_and_collision_blocks_closed_door():
    room = Room(19, 27, 16)
    initial_z = room.z
    before = room.render()
    room.step(1)
    assert room.z > initial_z
    assert not np.array_equal(before, room.render())
    for _ in range(30):
        room.step(1)
    assert room.z < 0
    room.step(5)
    for _ in range(15):
        room.step(1)
    assert room.z > 0


def test_data_splits_and_correct_image_action_alignment():
    train = make_dataset("train", scenes=1, episodes_per_scene=1, steps=20, size=16)
    test = make_dataset("test", scenes=1, episodes_per_scene=1, steps=20, size=16)
    assert train["records"][0]["scene_seed"] != test["records"][0]["scene_seed"]
    assert train["records"][0]["sha256"] != test["records"][0]["sha256"]
    record = train["records"][0]
    room = Room(record["scene_seed"], record["trajectory_seed"], 16)
    for index, action in enumerate(train["actions"][0]):
        room.step(int(action))
        assert np.array_equal(room.render(), train["frames"][0, index+4])
    history, action, target = sample_batch(train, 3, np.random.default_rng(8))
    assert history.shape == (3, 4, 3, 16, 16)
    assert target.shape == (3, 3, 16, 16)
    assert action.shape == (3,)


@pytest.mark.parametrize("kind", ["predictor", "flow"])
def test_model_training_loss_and_safe_checkpoint(kind, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(27)
    model = RGBModel(kind, width=8)
    history = torch.rand(2, 4, 3, 16, 16)*2-1
    target = torch.rand(2, 3, 16, 16)*2-1
    loss = training_loss(model, history, torch.tensor([1, 5]), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    path = tmp_path / "model.pt"
    torch.save({"config": {"kind": kind, "width": 8}, "state_dict": model.state_dict()}, path)
    loaded, _ = load_model(path)
    observation = history[0].numpy()
    controls = np.asarray([1, 0, 5]*7, dtype=np.int64)
    first = rollout(loaded, observation, controls, sample_steps=2, seed=37)
    second = rollout(loaded, observation, controls, sample_steps=2, seed=37)
    assert first.shape == (21, 3, 16, 16)
    assert np.isfinite(first).all() and np.abs(first).max() <= 1
    assert np.array_equal(first, second)


def test_untrained_predictor_is_exact_persistence_and_input_validation():
    model = RGBModel("predictor", width=8)
    initial = np.random.default_rng(9).uniform(-1, 1, (4, 3, 16, 16)).astype(np.float32)
    values = rollout(model, initial, np.asarray([1, 5, 3], dtype=np.int64))
    assert np.array_equal(values, np.repeat(initial[-1:], 3, 0))
    with pytest.raises(ValueError):
        rollout(model, initial, np.asarray([6], dtype=np.int64))
    with pytest.raises(ValueError):
        rollout(model, initial*5, np.asarray([1], dtype=np.int64))

