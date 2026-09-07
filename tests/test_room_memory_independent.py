"""Independent CPU protocol checks. No reserved scenes or GPU operations."""
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.room_world.memory_evaluate import (
    evaluate_pair, generated_rollout, observed_prefix_return, score_return,
)
from experiments.room_world.memory_model import RoomMemoryModel
from experiments.room_world.model import load_model


@pytest.fixture(autouse=True)
def small_cpu_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class RecordingModel(nn.Module):
    """Deterministic recurrence whose complete visible input can be inspected."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def initial_state(self, batch_size):
        return torch.zeros(batch_size, 32)

    def forward(self, history, action, state, *, mode="carry"):
        self.calls.append((history.clone(), action.clone(), state.clone()))
        prior = state if mode == "carry" else torch.zeros_like(state)
        after = prior + action[:, None] * 0.01 + history[:, -1].mean((1, 2, 3))[:, None] * 0.001
        prediction = (history[:, -1] + after[:, :1, None, None] * 0.01 + 0.002).clamp(-1, 1)
        return prediction, after


def marked_observations(batch=2, steps=65):
    return torch.arange(steps + 1).float()[None, :, None, None, None].expand(batch, -1, 3, 4, 4).clone() / 100


def test_observed_prefix_warms_actions_0_to_40_then_generates_only_suffix():
    observations = marked_observations()
    actions = torch.zeros(2, 65, dtype=torch.int64)
    actions[:, :41] = torch.arange(41) % 6
    actions[:, 41:] = 4
    model = RecordingModel()
    result = observed_prefix_return(model, observations[:, :42], actions[:, :41], actions[:, 41:])
    assert result.shape == (2, 24, 3, 4, 4)
    assert len(model.calls) == 65
    for step in range(41):
        indices = torch.arange(step - 3, step + 1).clamp_min(0)
        assert torch.equal(model.calls[step][0], observations[:, indices])
        assert torch.equal(model.calls[step][1], actions[:, step])
    assert torch.equal(model.calls[41][0], observations[:, 38:42])
    assert torch.equal(model.calls[41][1], actions[:, 41])
    # After the initialized return frame, generated RGB is the only new history.
    for step in range(42, 65):
        assert torch.equal(model.calls[step][0][:, -1], result[:, step - 42])
    assert model.calls[41][2].abs().sum() > 0


def test_future_truth_changes_scores_without_changing_either_generation_protocol():
    observations = marked_observations()
    actions = torch.zeros(2, 65, dtype=torch.int64)
    actions[1, 0] = 5
    first_report, first = evaluate_pair(RecordingModel(), observations, actions)
    modified = observations.clone()
    modified[0, 42:] = -0.5
    modified[1, 42:] = 0.9
    second_report, second = evaluate_pair(RecordingModel(), modified, actions)
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert not first_report["observed_prefix_then_generated_return"]["eligible_pair"]
    assert second_report["observed_prefix_then_generated_return"]["eligible_pair"]


def test_generated_rollout_clones_caller_state_and_history_and_uses_all_commands():
    model = RecordingModel()
    history = torch.full((2, 4, 3, 4, 4), 0.2)
    state = torch.full((2, 32), 0.3)
    initial_history, initial_state = history.clone(), state.clone()
    actions = torch.tensor([[5, 3, 0, 4], [0, 3, 0, 4]])
    result = generated_rollout(model, history, actions, state)
    assert torch.equal(history, initial_history) and torch.equal(state, initial_state)
    assert len(model.calls) == 4
    assert torch.equal(model.calls[0][0], history)
    assert torch.equal(model.calls[0][2], state)
    for index in range(4):
        assert torch.equal(model.calls[index][1], actions[:, index])
        if index:
            assert torch.equal(model.calls[index][0][:, -1], result[:, index - 1])
    # A second independent invocation starts from exactly the caller's state.
    assert torch.equal(result, generated_rollout(RecordingModel(), history, actions, state))


def test_pair_score_rejects_ties_and_requires_both_branches_on_same_true_region():
    truth = torch.zeros(2, 3, 3, 4, 4)
    truth[0, 2, :, 1, 1], truth[1, 2, :, 1, 1] = -1, 1
    exact = score_return(truth, truth)
    assert exact["different_pixel_count"] == 1 and exact["pair_correct"]
    assert exact["return_region_mae"] == 0
    midpoint = torch.zeros_like(truth)
    assert not score_return(midpoint, truth)["pair_correct"]
    one_correct = truth.clone()
    one_correct[1] = truth[0]
    assert not score_return(one_correct, truth)["pair_correct"]
    irrelevant_error = truth.clone()
    irrelevant_error[:, :2] = 0.8
    scored = score_return(irrelevant_error, truth)
    assert scored["pair_correct"] and scored["return_region_mae"] == 0
    assert scored["return_full_frame_mae"] > 0


def test_actual_memory_module_keeps_batched_sessions_independent():
    checkpoint = Path(__file__).resolve().parents[1] / "experiments/room_world/artifacts/predictor/model.pt"
    base, _ = load_model(checkpoint, device="cpu")
    torch.manual_seed(371)
    model = RoomMemoryModel(base).eval()
    # Exercise state conditioning rather than only the zero-initialized path.
    with torch.no_grad():
        model.condition.weight.normal_(0, 0.05)
    history = torch.rand(2, 4, 3, 8, 8) * 0.4 - 0.2
    state = torch.randn(2, 32) * 0.1
    commands = torch.tensor([5, 4])
    with torch.no_grad():
        together_rgb, together_state = model(history, commands, state)
        for index in range(2):
            rgb, after = model(history[index:index + 1], commands[index:index + 1], state[index:index + 1])
            torch.testing.assert_close(rgb, together_rgb[index:index + 1], atol=0.000001, rtol=0.00001)
            torch.testing.assert_close(after, together_state[index:index + 1], atol=0.000001, rtol=0.00001)
        mutated_history, mutated_state = history.clone(), state.clone()
        mutated_history[1] = -0.7
        mutated_state[1] = 20
        changed_rgb, changed_state = model(mutated_history, commands, mutated_state)
        torch.testing.assert_close(changed_rgb[0], together_rgb[0], atol=0.000001, rtol=0.00001)
        torch.testing.assert_close(changed_state[0], together_state[0], atol=0.000001, rtol=0.00001)


def test_training_function_aligns_all_65_observed_histories_actions_and_next_targets():
    from experiments.room_world.memory_train import sequence_loss

    class EchoCurrent(RecordingModel):
        def forward(self, history, action, state, *, mode="carry"):
            self.calls.append((history.clone(), action.clone(), state.clone()))
            return history[:, -1], state + 1

    observations = marked_observations()
    actions = (torch.arange(65) % 6)[None].repeat(2, 1)
    model = EchoCurrent()
    loss = sequence_loss(model, observations, actions)
    torch.testing.assert_close(loss, torch.tensor(0.005), atol=0.0000001, rtol=0)
    assert len(model.calls) == 65
    for step, (history, action, state) in enumerate(model.calls):
        indices = torch.arange(step - 3, step + 1).clamp_min(0)
        assert torch.equal(history, observations[:, indices])
        assert torch.equal(action, actions[:, step])
        assert torch.equal(state, torch.full_like(state, step))


@pytest.mark.parametrize("mode", ["carry", "reset"])
def test_trainer_final_only_loss_preserves_full_temporal_path_without_detach(mode):
    from experiments.room_world.memory_train import sequence_loss

    class FinalOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.retention = nn.Parameter(torch.tensor([0.97]))
            self.index = 0

        def initial_state(self, batch_size):
            self.start = torch.zeros(batch_size, 32, requires_grad=True)
            return self.start

        def forward(self, history, action, state, *, mode="carry"):
            previous = state if mode == "carry" else torch.zeros_like(state)
            after = previous * self.retention + history[:, -1].mean((1, 2, 3))[:, None] * 0.01
            prediction = torch.zeros_like(history[:, -1])
            if self.index == 64:
                prediction = prediction + after[:, :1, None, None]
            self.index += 1
            return prediction, after

    observations = torch.zeros(2, 66, 3, 4, 4)
    observations[:, 0] = 0.1
    observations.requires_grad_()
    actions = torch.zeros(2, 65, dtype=torch.int64)
    model = FinalOnly()
    loss = sequence_loss(model, observations, actions, mode)
    loss.backward()
    assert model.index == 65
    assert torch.isfinite(observations.grad).all()
    if mode == "carry":
        assert observations.grad[:, 0].abs().sum() > 0
        assert model.start.grad is not None and model.start.grad.abs().sum() > 0
    else:
        assert not observations.grad[:, 0].count_nonzero()
        assert model.start.grad is None


def test_all_actual_development_controls_preserve_hashes_commands_and_return_alias():
    """Read existing development files only; never call the teacher renderer."""
    import hashlib
    import json
    import numpy as np
    from experiments.room_world.memory_train import ValidationControls, verify_visibility_audit

    directory = Path(__file__).resolve().parents[1] / "experiments/room_world/memory_artifacts"
    capture = directory / "data/validation-controls"
    manifest = json.loads((capture / "manifest.json").read_text())
    reader = ValidationControls(capture)
    singles = {
        "translation_cycle": [1] * 4 + [2] * 8 + [1] * 4,
        "out_of_reach_interaction": [2] * 6 + [5] + [0] * 3 + [1] * 6,
        "turn_open_close": [3] * 4 + [4] * 4 + [5] + [0] * 3 + [5] + [0] * 3,
    }
    seen, counts, transitions = set(), np.zeros(6, dtype=np.int64), [0, 0]
    for row in manifest["records"]:
        key = row["scene_seed"], row["kind"]
        assert key not in seen and 300000 <= key[0] <= 300007
        seen.add(key)
        path = capture / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["file_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == {"observations", "actions"}
            rgb, actions = archive["observations"], archive["actions"]
        assert rgb.dtype == np.uint8 and actions.dtype == np.int64
        for name, value in (("observations", rgb), ("actions", actions)):
            assert hashlib.sha256(value.tobytes()).hexdigest() == row[name + "_sha256"]
        if row["kind"] in singles:
            assert rgb.shape == (1, 17, 3, 64, 64)
            assert actions.tolist() == [singles[row["kind"]]]
            counts += np.bincount(actions.ravel(), minlength=6)
            transitions[0] += actions.size
        else:
            wait = {"paired_wait_8": 8, "paired_wait_32": 32}[row["kind"]]
            first_return = {8: 33, 32: 57}[wait]
            assert actions.tolist() == [[0] + [3] * 24 + [0] * wait + [4] * 24,
                                       [5] + [3] * 24 + [0] * wait + [4] * 24]
            assert rgb.shape == (2, 50 + wait, 3, 64, 64)
            assert np.array_equal(rgb[0, 0], rgb[1, 0])
            assert np.array_equal(rgb[0, first_return - 3:first_return + 1],
                                  rgb[1, first_return - 3:first_return + 1])
            assert not np.array_equal(rgb[0, -1], rgb[1, -1])
            assert reader.return_indices[key] == first_return
            transitions[1] += actions.size
        normalized, loaded_actions = reader.load_case(*key)
        torch.testing.assert_close(normalized, torch.from_numpy(rgb.copy()).float() / 127.5 - 1, rtol=0, atol=0)
        assert torch.equal(loaded_actions, torch.from_numpy(actions.copy()))
    assert len(seen) == 40 and transitions == [384, 2208]
    assert counts.tolist() == [72, 112, 112, 32, 32, 24]
    verified = verify_visibility_audit(directory / "control-validation/visibility-report.json", reader)
    assert verified["status"] == "verified"
    assert list(verified["visible_door_pixels"].values()) == [965, 781, 962, 883, 912, 881, 885, 819]


def measured_score_fixture():
    """Unequal scene/seed errors distinguish ratios of means from pooled ratios."""
    import numpy as np
    from experiments.room_world.memory_train import SEEDS, SCENES, CONTROL_ACTIONS

    protocols = ("uninterrupted_generated_history", "observed_prefix_then_generated_return")
    reports, oracle = {}, {}
    for seed_index, seed in enumerate(SEEDS):
        for mode in ("carry", "reset"):
            key = f"{seed}-{mode}"
            report = {"status": "complete", "controls_evaluated": True,
                      "data": {"manifest_sha256": "1" * 64},
                      "provenance": {"seed": seed, "mode": mode}, "scenes": [],
                      "control_results": {"status": "complete", "single_cases": [],
                                          "provenance": {"manifest_sha256": "2" * 64},
                                          "paired_waits": {"8": {"cases": []}, "32": {"cases": []}}}}
            for scene_index, scene in enumerate(SCENES["validation"]):
                for kind_index, (kind, commands) in enumerate(CONTROL_ACTIONS.items()):
                    error = (1 + scene_index + kind_index) * .01
                    report["control_results"]["single_cases"].append({
                        "scene_seed": scene, "kind": kind, "mae": error, "branches": 1, "steps": 16,
                        "per_step": [{"branch": 0, "transition": t, "action": action, "mae": error}
                                     for t, action in enumerate(commands)]})
                for wait_index, waiting in enumerate((8, 16, 32)):
                    row = {"scene_seed": scene}
                    for protocol_index, protocol in enumerate(protocols):
                        reset = (seed_index + 1) * (.04 + .01 * scene_index) + .005 * wait_index
                        carry = reset * (.4 + .08 * seed_index + .01 * scene_index + .02 * protocol_index)
                        correct = (scene_index + seed_index + protocol_index + wait_index) % 8 != 0
                        row[protocol] = {"eligible_pair": True, "return_region_mae": carry if mode == "carry" else reset,
                                         "pair_correct": bool(correct) if mode == "carry" else False}
                        oracle.setdefault((waiting, protocol), {})[(seed_index, scene_index)] = (carry, reset, correct)
                    destination = report["scenes"] if waiting == 16 else report["control_results"]["paired_waits"][str(waiting)]["cases"]
                    destination.append(row)
            reports[key] = report
    import copy
    reports["frozen"] = copy.deepcopy(reports[f"{SEEDS[0]}-reset"])
    reports["frozen"]["provenance"] = {"mode": "frozen"}
    return reports, {key: np.asarray([[rows[(seed, scene)] for scene in range(8)] for seed in range(3)], dtype=np.float64)
                     for key, rows in oracle.items()}


def test_shared_scene_bootstrap_matches_independent_mean_of_seed_ratio_oracle():
    import hashlib
    import numpy as np
    from experiments.room_world.memory_evaluate import compare_evaluations

    reports, oracle = measured_score_fixture()
    result = compare_evaluations(reports, matched_budgets=True,
                                 visibility_audit={"status": "verified", "capture_manifest_sha256": "2" * 64})
    assert result["status"] == "complete" and result["control_retention_passed"]
    indices = np.random.default_rng(20260907).integers(0, 8, size=(10000, 8), dtype=np.int64)
    assert result["bootstrap"]["shared_scene_index_sha256"] == hashlib.sha256(indices.tobytes()).hexdigest()
    for (waiting, protocol), values in oracle.items():
        row = result["paired_memory"][str(waiting)][protocol]
        seed_gains = 1 - values[:, :, 0].mean(1) / values[:, :, 1].mean(1)
        expected = float(seed_gains.mean())
        pooled = float(1 - values[:, :, 0].mean() / values[:, :, 1].mean())
        assert abs(expected - pooled) > .01
        assert row["mean_seed_relative_gain"] == pytest.approx(expected)
        assert row["mean_seed_carry_pair_correct_fraction"] == pytest.approx(values[:, :, 2].mean())
        # Whole scenes are resampled together, preserving all three seed pairs.
        sampled = values[:, indices, :].mean(axis=2)
        gains = (1 - sampled[:, :, 0] / sampled[:, :, 1]).mean(axis=0)
        correctness = sampled[:, :, 2].mean(axis=0)
        np.testing.assert_allclose(row["intervals"]["mean_seed_relative_gain"]["percentile_95"],
                                   np.percentile(gains, [2.5, 97.5]), rtol=0, atol=1e-14)
        np.testing.assert_allclose(row["intervals"]["mean_seed_carry_pair_correct_fraction"]["percentile_95"],
                                   np.percentile(correctness, [2.5, 97.5]), rtol=0, atol=1e-14)


def test_one_seed_one_control_type_failure_is_not_hidden_by_averaging():
    from experiments.room_world.memory_evaluate import compare_evaluations
    from experiments.room_world.memory_train import SEEDS

    reports, _ = measured_score_fixture()
    key = f"{SEEDS[1]}-carry"
    for row in reports[key]["control_results"]["single_cases"]:
        if row["kind"] == "out_of_reach_interaction":
            row["mae"] *= 1.051
            for step in row["per_step"]:
                step["mae"] *= 1.051
    scored = compare_evaluations(reports, matched_budgets=True,
                                 visibility_audit={"status": "verified", "capture_manifest_sha256": "2" * 64})
    seed = scored["control_retention"][str(SEEDS[1])]
    assert not scored["control_retention_passed"]
    assert not seed["out_of_reach_interaction"]["gate_passed"]
    assert seed["overall_equal_type"]["gate_passed"]
    assert scored["control_retention"][str(SEEDS[0])]["out_of_reach_interaction"]["gate_passed"]


def test_control_retention_requires_frozen_reference_even_when_trained_reset_is_worse():
    from experiments.room_world.memory_evaluate import compare_evaluations
    from experiments.room_world.memory_train import SEEDS

    reports, _ = measured_score_fixture()
    for key, report in reports.items():
        factor = 1.06 if key.endswith("-carry") else 2 if key.endswith("-reset") else 1
        for row in report["control_results"]["single_cases"]:
            row["mae"] *= factor
            for step in row["per_step"]:
                step["mae"] *= factor
    scored = compare_evaluations(reports, matched_budgets=True,
                                 visibility_audit={"status": "verified", "capture_manifest_sha256": "2" * 64})
    assert not scored["control_retention_passed"]
    for seed in SEEDS:
        for value in scored["control_retention"][str(seed)].values():
            assert value["exceeds_frozen_by_more_than_5_percent"]
            assert not value["exceeds_reset_by_more_than_5_percent"]
            assert not value["gate_passed"]


def test_zero_reference_remains_undefined_for_ratios_and_exact_for_control_limit():
    from experiments.room_world.memory_evaluate import compare_evaluations
    from experiments.room_world.memory_train import SEEDS

    reports, _ = measured_score_fixture()
    for key, report in reports.items():
        if key == "frozen":
            for row in report["control_results"]["single_cases"]:
                row["mae"] = 0.
                for step in row["per_step"]:
                    step["mae"] = 0.
        if key.endswith("-reset"):
            for row in report["scenes"]:
                for protocol in ("uninterrupted_generated_history", "observed_prefix_then_generated_return"):
                    row[protocol]["return_region_mae"] = 0.
    scored = compare_evaluations(reports, matched_budgets=True,
                                 visibility_audit={"status": "verified", "capture_manifest_sha256": "2" * 64})
    assert not scored["control_retention_passed"]
    control = scored["control_retention"][str(SEEDS[0])]["translation_cycle"]
    assert control["frozen_limit_1_05"] == 0
    assert control["relative_increase_vs_frozen"] is None
    assert control["intervals"]["relative_increase_vs_frozen"]["undefined_resamples"] == 10000
    paired = scored["paired_memory"]["16"]["uninterrupted_generated_history"]
    assert paired["mean_seed_relative_gain"] is None
    assert not paired["provisional_gate_passed"]
    assert paired["intervals"]["mean_seed_relative_gain"]["percentile_95"] is None
