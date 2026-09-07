"""Independent checks of image-only inference and its finite-memory limit."""
import numpy as np
import pytest
import torch

from experiments.room_world.aliasing import evaluate_aliasing, make_aliasing_case
from experiments.room_world.data import make_dataset, sample_batch
from experiments.room_world.evaluate import evaluate
from experiments.room_world.model import RGBModel, rollout
from experiments.room_world.simulator import Room


def test_reachable_door_states_alias_after_sixty_four_step_excursion():
    case = make_aliasing_case(size=16)
    assert len(case["away_actions"]) + len(case["actions"]) == 64
    assert not np.array_equal(case["initial"][0], case["initial"][1])
    assert np.array_equal(case["histories"][0], case["histories"][1])
    assert not np.array_equal(case["truth"][0, -1], case["truth"][1, -1])
    # Both cases return to the same camera pose they had before turning away.
    assert np.array_equal(case["initial"], case["truth"][:, -1])


def test_shared_prediction_lower_bound_is_exact():
    case = make_aliasing_case(size=16)
    a, b = case["truth"].astype(np.float64) / 255
    midpoint = (a + b) / 2
    minimum = (np.square(midpoint - a).mean() + np.square(midpoint - b).mean()) / 2
    report = evaluate_aliasing(size=16)
    assert minimum > 0
    assert report["mean_shared_prediction_mse_lower_bound_0_to_1"] == pytest.approx(minimum)
    assert report["identical_input_histories"]


class FeedbackProbe(torch.nn.Module):
    """Routing probe, not a trained quality baseline."""
    kind = "predictor"

    def __init__(self):
        super().__init__()
        self.histories = []
        self.controls = []

    def forward(self, history, action):
        self.histories.append(history.clone())
        self.controls.append(action.clone())
        return (history[:, -1] + .125).clamp(-1, 1)


def test_rollout_uses_generated_frames_without_teacher_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("teacher must not be called inside model rollout")
    monkeypatch.setattr(Room, "render", forbidden)
    monkeypatch.setattr(Room, "step", forbidden)
    initial = np.zeros((4, 3, 16, 16), np.float32)
    model = FeedbackProbe()
    predicted = rollout(model, initial, np.asarray([1, 5, 3]), "cpu")
    assert np.all(initial == 0)
    np.testing.assert_array_equal(predicted[:, 0, 0, 0], [.125, .25, .375])
    assert [int(x.item()) for x in model.controls] == [1, 5, 3]
    assert torch.equal(model.histories[1][0, -1], torch.from_numpy(predicted[0]))
    assert torch.equal(model.histories[2][0, -2:], torch.from_numpy(predicted[:2]))


def test_actual_flow_network_uses_repeatable_noise_without_renderer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("renderer access during inference")
    monkeypatch.setattr(Room, "render", forbidden)
    model = RGBModel("flow", width=8).eval()
    initial = np.zeros((4, 3, 16, 16), np.float32)
    actions = np.asarray([1, 3])
    first = rollout(model, initial, actions, "cpu", sample_steps=2, seed=17)
    second = rollout(model, initial, actions, "cpu", sample_steps=2, seed=17)
    other_noise = rollout(model, initial, actions, "cpu", sample_steps=2, seed=18)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other_noise)
    # Untrained weights validate the inference/RNG contract, not prediction quality.


def test_aliasing_report_matches_model_pair_and_ground_truth_bound(tmp_path):
    model = RGBModel("predictor", width=8).eval()
    report = evaluate_aliasing(model, size=16, output=tmp_path)
    assert report["model"]["identical_paired_predictions"]
    assert report["model"]["mean_paired_average_mse_0_to_1"] >= (
        report["mean_shared_prediction_mse_lower_bound_0_to_1"] - 1e-12)
    saved = np.load(tmp_path / "aliasing-case.npz", allow_pickle=False)
    assert saved["predictions"].shape == (2, 24, 3, 16, 16)
    assert saved["full_predictions"].shape == (2, 64, 3, 16, 16)
    assert report["uninterrupted_model_rollout"]["teacher_frames_after_start"] == 0
    assert report["uninterrupted_model_rollout"]["generated_frames_per_case"] == 64
    assert (tmp_path / "aliasing-metrics.json").is_file()


def test_sample_batch_actions_target_the_immediately_following_frame():
    data = make_dataset("test", scenes=1, episodes_per_scene=1, steps=20, size=16)
    class FixedOffsets:
        def __init__(self):
            self.calls = 0
        def integers(self, low, high, size):
            self.calls += 1
            return np.zeros(size, np.int64) if self.calls == 1 else np.asarray([0, 3, 16, 19])
    histories, actions, targets = sample_batch(data, 4, FixedOffsets())
    record = data["records"][0]
    room = Room(record["scene_seed"], record["trajectory_seed"], size=16)
    selected = {0: 0, 3: 1, 16: 2, 19: 3}
    for step, action in enumerate(data["actions"][0]):
        if step in selected:
            index = selected[step]
            np.testing.assert_allclose(histories[index, -1], room.render().astype(np.float32) / 127.5 - 1)
            assert actions[index] == action
        room.step(int(action))
        if step in selected:
            np.testing.assert_allclose(targets[selected[step]], room.render().astype(np.float32) / 127.5 - 1)


def test_evaluation_closed_loop_and_repeat_last_baseline_use_only_initial_truth(tmp_path, monkeypatch):
    data = make_dataset("test", scenes=1, episodes_per_scene=1, steps=20, size=16)
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must not query teacher during prediction")
    monkeypatch.setattr(Room, "render", forbidden)
    monkeypatch.setattr(Room, "step", forbidden)
    report = evaluate(FeedbackProbe(), data, device="cpu", output=tmp_path)
    saved = np.load(tmp_path / "heldout-first-rollout.npz", allow_pickle=False)
    initial = data["frames"][0, 3].astype(np.float32) / 127.5 - 1
    truth = data["frames"][0, 4:].astype(np.float32) / 127.5 - 1
    current = initial.copy()
    for index, generated in enumerate(saved["predicted"]):
        current = np.clip(current + .125, -1, 1)
        np.testing.assert_array_equal(generated, current)
    for horizon, scores in report["closed_loop"]["horizons"].items():
        expected = float(np.abs(initial - truth[int(horizon) - 1]).mean() / 2)
        assert scores["repeat_last_mae"] == pytest.approx(expected)
    assert report["closed_loop"]["teacher_frames_after_start"] == 0


class NoiseProbe(torch.nn.Module):
    """Keep sampler noise unchanged to audit common-noise pairing only."""
    kind = "flow"

    def forward(self, history, action, noisy, time):
        return torch.zeros_like(noisy)


def test_evaluation_reuses_identical_noise_for_actual_and_zero_action_arms():
    data = make_dataset("test", scenes=1, episodes_per_scene=1, steps=20, size=16)
    report = evaluate(NoiseProbe(), data, device="cpu", sample_steps=2)
    assert report["one_step"]["model_mae"] == report["one_step"]["zero_action_mae"]
    for scores in report["closed_loop"]["horizons"].values():
        assert scores["model_mae"] == scores["zero_action_mae"]
