# SPDX-License-Identifier: Apache-2.0
"""CPU software checks with tiny synthetic tensors; no Room model training."""
import copy
import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from experiments.room_world import memory_train as mt
from experiments.room_world.memory_evaluate import (evaluate_model, evaluate_pair, evaluate_study,
    observed_one_step_measurements, summarize_step_errors, aggregate_control_cases, compare_evaluations)


class TinyMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1, 1).requires_grad_(False)
        self.weight = nn.Parameter(torch.tensor([.05, .01]))

    def initial_state(self, batch_size):
        return self.weight.new_zeros((batch_size, 1))

    def memory_parameters(self):
        return [self.weight]

    def parameter_counts(self):
        return {"memory_trainable": 2, "fixture_only": True}

    def forward(self, history, action, state, *, mode="carry"):
        old = state if mode == "carry" else torch.zeros_like(state)
        state = .8 * old + self.weight[0] * history[:, -1].mean((1, 2, 3))[:, None]
        state = state + self.weight[1] * (action[:, None] + 1) * .01
        return (history[:, -1] + .05 * state.tanh()[:, :, None, None]).clamp(-1, 1), state


class TinyData:
    scene_ids = (5000, 5001)

    def load_pair(self, scene):
        assert scene in self.scene_ids
        rgb = torch.linspace(-.5, .5, 66)[None, :, None, None, None].expand(2, 66, 3, 4, 4).clone()
        actions = torch.zeros(2, 65, dtype=torch.int64)
        actions[1, 0] = 5
        return rgb, actions

    def provenance(self):
        return {"fixture_only": True, "scene_ids": list(self.scene_ids)}


def test_full_sequence_loss_is_plain_mean_and_temporally_connected():
    model = TinyMemory()
    rgb, actions = TinyData().load_pair(5000)
    rgb.requires_grad_()
    state = model.initial_state(2)
    values = []
    for step in range(65):
        predicted, state = model(mt.history_at(rgb, step), actions[:, step], state, mode="carry")
        values.append((predicted - rgb[:, step + 1]).abs().mean() / 2)
    result = mt.sequence_loss(model, rgb, actions)
    assert torch.equal(result, torch.stack(values).mean())
    last_prediction, _ = model(mt.history_at(rgb, 64), actions[:, 64], state, mode="carry")
    grad = torch.autograd.grad(last_prediction.sum(), rgb)[0]
    assert grad[:, 0].abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())


def test_exact_initialization_schedule_and_atomic_recovery(tmp_path):
    torch.manual_seed(9)
    initial = TinyMemory()
    schedule = mt.paired_scene_schedule(TinyData.scene_ids, mt.SEEDS[0], 2)
    assert schedule == mt.paired_scene_schedule(TinyData.scene_ids, mt.SEEDS[0], 2)
    reports = []
    for mode in ("carry", "reset"):
        model = copy.deepcopy(initial)
        output = tmp_path / mode
        reports.append(mt.run_arm(model, TinyData(), schedule, mode, output, seed=mt.SEEDS[0], phase="train"))
        recovery = torch.load(output / "recovery-last.pt", weights_only=True)
        final = torch.load(output / "memory-final.pt", weights_only=True)
        assert recovery["completed_updates"] == final["completed_updates"] == 2
        assert recovery["completed_scene_schedule"] == schedule
        assert recovery["optimizer_state_dict"]["state"]
        assert set(recovery["rng"]) == {"torch_cpu", "python", "numpy"}
        assert torch.equal(recovery["memory_state_dict"]["weight"], model.weight)
        assert torch.equal(final["memory_state_dict"]["weight"], model.weight)
        assert set(final["memory_state_dict"]) == {"weight"}
    assert mt.matched_arms(*reports)
    for field, value in (("seed", 1), ("mode", "carry"), ("completed_updates", 1), ("completed_scene_schedule", [5001])):
        changed = copy.deepcopy(reports[1])
        changed[field] = value
        assert not mt.matched_arms(reports[0], changed)
    with pytest.raises(FileExistsError):
        mt.run_arm(TinyMemory(), TinyData(), schedule, "carry", tmp_path / "carry", seed=1, phase="train")


@pytest.mark.parametrize("error_class,status", [(KeyboardInterrupt, "interrupted"), (RuntimeError, "failed")])
def test_partial_failure_preserves_last_completed_bundle(tmp_path, monkeypatch, error_class, status):
    model = TinyMemory()
    original = torch.optim.AdamW.step
    calls = 0

    def bad_second_update(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            with torch.no_grad():
                model.weight.fill_(float("nan"))
            raise error_class("Deliberate partial-update fixture")
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", bad_second_update)
    out = tmp_path / "failed"
    with pytest.raises(error_class):
        mt.run_arm(model, TinyData(), [5000, 5001], "carry", out, seed=1, phase="train")
    report = json.loads((out / "metrics.json").read_text())
    recovery = torch.load(out / "recovery-last.pt", weights_only=True)
    assert report["status"] == status
    assert report["completed_updates"] == recovery["completed_updates"] == 1
    assert recovery["completed_scene_schedule"] == [5000]
    assert torch.isfinite(recovery["memory_state_dict"]["weight"]).all()
    assert report["recovery_last_sha256"] == mt.sha256(out / "recovery-last.pt")
    assert not (out / "memory-final.pt").exists()


def test_guard_time_limit_reaps_own_child(tmp_path):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    terminal = mt.guard_process(process, tmp_path, max_seconds=.04, minimum_available_gib=0, interval=.01)
    assert terminal["status"] == "stopped"
    assert terminal["reason"] == "elapsed_time_limit"
    assert process.poll() is not None
    assert json.loads((tmp_path / "terminal.json").read_text())["exit_code"] != 0


def test_guard_parent_interrupt_reaps_own_child(tmp_path, monkeypatch):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    def interrupted(_):
        raise KeyboardInterrupt("Synthetic parent stop")

    monkeypatch.setattr(mt.time, "sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        mt.guard_process(process, tmp_path, minimum_available_gib=0)
    assert process.poll() is not None
    assert json.loads((tmp_path / "terminal.json").read_text())["status"] == "interrupted"


def test_future_truth_isolation_and_prediction_retention(tmp_path):
    model, data = TinyMemory(), TinyData()
    observations, actions = data.load_pair(5000)
    _, predictions = evaluate_pair(model, observations, actions)
    changed = observations.clone()
    changed[:, 42:] *= -.75
    _, other = evaluate_pair(model, changed, actions)
    assert all(torch.equal(predictions[key], other[key]) for key in predictions)
    report = evaluate_model(model, data, tmp_path / "eval", provenance={"fixture_only": True})
    assert report["status"] == "complete" and not report["controls_evaluated"]
    for row in report["scenes"]:
        path = tmp_path / "eval" / row["prediction_file"]
        assert mt.sha256(path) == row["prediction_file_sha256"]
        assert mt.tensor_hashes(torch.load(path, weights_only=True)) == row["predictions"]


def test_evaluation_failure_retains_status(tmp_path):
    class FailingData(TinyData):
        def load_pair(self, scene):
            if scene == 5001:
                raise RuntimeError("Synthetic second-scene failure")
            return super().load_pair(scene)

    out = tmp_path / "eval"
    with pytest.raises(RuntimeError):
        evaluate_model(TinyMemory(), FailingData(), out)
    report = json.loads((out / "evaluation.json").read_text())
    assert report["status"] == "failed" and len(report["scenes"]) == 1
    assert (out / "scene-5000.pt").exists()


def test_reserved_split_rejected_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="Only train and validation"):
        mt.DevelopmentDataset(tmp_path / "never-open", "test")


def test_profile_checkpoint_cannot_be_used_for_quality_evaluation(tmp_path):
    study = tmp_path / "profile"
    study.mkdir()
    (study / "study.json").write_text(json.dumps({"status": "complete", "phase": "profile", "matched_budgets": True}))
    with pytest.raises(ValueError, match="Only completed matched training arms"):
        evaluate_study(study, tmp_path / "unread-data", tmp_path / "uncreated-output")
    assert not (tmp_path / "uncreated-output").exists()


def test_profile_and_training_cli_budget_rules(monkeypatch, tmp_path):
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return {"status": "fixture", "phase": kwargs["phase"], "matched_budgets": False, "evaluation_executed": False}

    monkeypatch.setattr(mt, "run_study", fake_run)
    monkeypatch.setattr(sys, "argv", ["memory_train", "--train", str(tmp_path), "--validation", str(tmp_path),
                        "--output", str(tmp_path), "--phase", "train", "--updates=50"])
    mt.main()
    assert seen["updates"] == 50
    monkeypatch.setattr(sys, "argv", sys.argv[:-1])
    with pytest.raises(SystemExit):
        mt.main()


@pytest.mark.parametrize("steps,batch", [(1, 1), (7, 1), (16, 1), (57, 2), (81, 2)])
def test_variable_length_observed_controls(steps, batch):
    model = TinyMemory()
    rgb = torch.linspace(-.4, .4, steps + 1)[None, :, None, None, None].expand(batch, steps + 1, 3, 2, 2).clone()
    actions = torch.arange(steps).remainder(6).expand(batch, -1).clone()
    scores, frames = observed_one_step_measurements(model, rgb, actions)
    assert frames.shape == (batch, steps, 3, 2, 2)
    expected = (frames - rgb[:, 1:]).abs().mean((2, 3, 4)) / 2
    assert scores["mae"] == pytest.approx(float(expected.double().mean()))
    assert len(scores["per_step"]) == steps * batch
    assert sum(value["count"] for value in scores["per_action"].values()) == steps * batch
    for step in scores["per_step"]:
        assert step["action"] == int(actions[step["branch"], step["transition"]])
        assert step["mae"] == float(expected[step["branch"], step["transition"]])


def test_equal_scene_type_weights_with_unequal_action_counts():
    rows = []
    for kind in mt.CONTROL_ACTIONS:
        for scene, errors, actions in ((1, [.1, .3], [0, 1]), (2, [.6, .6, .6, .6], [0, 0, 0, 1])):
            row = summarize_step_errors(torch.tensor([errors]), torch.tensor([actions]))
            rows.append(dict(row, scene_seed=scene, kind=kind))
    result = aggregate_control_cases(rows, (1, 2))
    assert result["available"]
    assert result["overall_equal_type_mae"] == pytest.approx(.4)
    for value in result["types"].values():
        assert value["equal_scene_mae"] == pytest.approx(.4)
        action = value["per_action"]["0"]
        assert action["count"] == 4
        assert action["equal_scene_mae"] == pytest.approx(.35)
        assert action["occurrence_weighted_mae"] == pytest.approx(.475)
    incomplete = aggregate_control_cases(rows[:-1], (1, 2))
    assert not incomplete["available"] and incomplete["overall_equal_type_mae"] is None


def test_missing_control_results_and_unmatched_budgets_leave_gates_unavailable():
    result = compare_evaluations({}, matched_budgets=True, visibility_audit={"status": "verified"})
    assert result["status"] == "unavailable" and result["control_retention_passed"] is None
    assert not result["gates_available"] and result["bootstrap"]["status"] == "unavailable"
    assert "three completed seed pairs" in result["reasons"][0]
    result = compare_evaluations({}, matched_budgets=False)
    assert "budgets" in result["reasons"][0]


def test_missing_or_incomplete_control_capture_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        mt.ValidationControls(tmp_path)
    manifest = {"schema": "worldline-room-memory-controls-v1", "status": "complete", "split": "validation",
        "scene_seeds": list(mt.SCENES["validation"]), "size": 64, "reserved_test_generated": False,
        "paired_waits": [8, 32], "model_input_fields": ["observations", "actions"],
        "control_actions": {key: list(value) for key, value in mt.CONTROL_ACTIONS.items()}, "records": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing, duplicated"):
        mt.ValidationControls(tmp_path)


@pytest.mark.parametrize("error_class,status", [(KeyboardInterrupt, "interrupted"), (BrokenPipeError, "failed")])
def test_configuration_handoff_failure_reaps_child(tmp_path, error_class, status):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.PIPE)
    actual_stdin = process.stdin

    class FailingPipe:
        def write(self, _):
            raise error_class("Synthetic configuration handoff failure")

    process.stdin = FailingPipe()
    try:
        with pytest.raises(error_class):
            mt.handoff_and_guard(process, {"fixture": True}, tmp_path)
        assert process.poll() is not None
        terminal = json.loads((tmp_path / "terminal.json").read_text())
        assert terminal["status"] == status
        assert terminal["reason"] == "worker_configuration_handoff_failed"
    finally:
        actual_stdin.close()


def test_variable_wait_control_predictions_are_retained(tmp_path):
    class TinyControls:
        scene_ids = TinyData.scene_ids
        return_indices = {(scene, kind): 25 + wait for scene in TinyData.scene_ids for kind, wait in mt.PAIRED_WAIT_KINDS.items()}

        def provenance(self):
            return {"fixture_only": True, "input_fields": ["observations", "actions"]}

        def load_case(self, scene, kind):
            if kind in mt.CONTROL_ACTIONS:
                actions = torch.tensor([mt.CONTROL_ACTIONS[kind]])
            else:
                wait = mt.PAIRED_WAIT_KINDS[kind]
                actions = torch.tensor([[0] + [3]*24 + [0]*wait + [4]*24, [5] + [3]*24 + [0]*wait + [4]*24])
            rgb = torch.zeros(actions.shape[0], actions.shape[1] + 1, 3, 2, 2)
            if actions.shape[0] == 2:
                rgb[1, -1] = .5
            return rgb, actions

    out = tmp_path / "with-controls"
    measured = evaluate_model(TinyMemory(), TinyData(), out, controls=TinyControls())
    assert measured["controls_evaluated"] and measured["control_results"]["status"] == "complete"
    assert len(measured["control_results"]["single_cases"]) == 6
    for wait, first, steps in ((8, 33, 57), (32, 57, 81)):
        for row in measured["control_results"]["paired_waits"][str(wait)]["cases"]:
            assert row["first_return_action_index"] == first
            saved = torch.load(out / row["prediction_file"], weights_only=True)
            assert saved["uninterrupted"].shape[1] == steps
            assert saved["observed_prefix_return"].shape[1] == 24
