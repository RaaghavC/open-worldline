# SPDX-License-Identifier: Apache-2.0
"""CPU software checks with tiny synthetic tensors; no Room model training."""
import copy
import json
from pathlib import Path
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


@pytest.mark.parametrize("sequence_path", ["stepwise", "batched"])
def test_sequence_path_plan_snapshots_the_actual_implementation(tmp_path, monkeypatch, sequence_path):
    monkeypatch.setattr(mt, "DevelopmentDataset", lambda *_: TinyData())
    out = tmp_path / sequence_path
    result = mt.run_study(tmp_path, tmp_path, out, phase="plan", seeds=(mt.SEEDS[0],), sequence_path=sequence_path)
    assert result["sequence_path"] == sequence_path
    assert set(result["source_sha256"]) == set(mt.training_source_names(sequence_path))
    assert ("memory_sequence.py" in result["source_sha256"]) == (sequence_path == "batched")
    for name, expected in result["source_sha256"].items():
        assert mt.sha256(out / "measured-source" / (name + ".txt")) == expected
        assert mt.sha256(Path(mt.__file__).with_name(name)) == expected
    assert result["schedules"][str(mt.SEEDS[0])] == mt.paired_scene_schedule(TinyData.scene_ids, mt.SEEDS[0], 50)


def test_batched_dispatch_and_checkpoint_path_are_explicit(tmp_path, monkeypatch):
    from experiments.room_world import memory_sequence
    calls = []

    def synthetic_batched_fixture(model, rgb, actions, mode, *, check=None):
        # Isolate dispatch/provenance. Actual batched numerical checks are independent.
        calls.append(mode)
        return mt.sequence_loss(model, rgb, actions, mode, check=check)

    monkeypatch.setattr(memory_sequence, "sequence_loss_batched", synthetic_batched_fixture)
    out = tmp_path / "batched"
    result = mt.run_arm(TinyMemory(), TinyData(), [5000], "carry", out, seed=mt.SEEDS[0], phase="train", sequence_path="batched")
    assert calls == ["carry"]
    assert result["sequence_path"] == "batched"
    assert result["loss_implementation"].endswith("synthetic_batched_fixture")
    for name in ("memory-last.pt", "memory-final.pt", "recovery-last.pt"):
        assert torch.load(out / name, weights_only=True)["sequence_path"] == "batched"


@pytest.mark.parametrize("sequence_path", ["stepwise", "batched"])
def test_worker_forwards_path_and_requires_its_source(tmp_path, monkeypatch, sequence_path):
    from pathlib import Path
    class Dataset(TinyData):
        manifest_sha256 = "fixture"

    monkeypatch.setattr(mt, "make_model", lambda *_: TinyMemory())
    monkeypatch.setattr(mt, "DevelopmentDataset", lambda *_: Dataset())
    called = []

    def synthetic_worker_fixture(*args, **kwargs):
        called.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(mt, "run_arm", synthetic_worker_fixture)
    initial = tmp_path / "initial.pt"
    mt.atomic_write(initial, mt.memory_state(TinyMemory()), tensor=True)
    config = {"device": "cpu", "base": str(mt.BASE), "seed": mt.SEEDS[0], "initial": str(initial),
        "initial_sha256": mt.sha256(initial), "train": str(tmp_path), "train_manifest_sha256": "fixture",
        "schedule": [5000], "mode": "carry", "output": str(tmp_path / "unused"), "phase": "profile",
        "learning_rate": 3e-4, "weight_decay": 1e-4,
        "source_sha256": {name: mt.sha256(Path(mt.__file__).with_name(name)) for name in mt.training_source_names(sequence_path)}}
    if sequence_path == "batched":
        config["sequence_path"] = "batched"
    # Missing path is the legacy/default stepwise case.
    mt._worker(config)
    assert called[0]["sequence_path"] == sequence_path
    if sequence_path == "batched":
        del config["source_sha256"]["memory_sequence.py"]
        with pytest.raises(ValueError, match="does not cover"):
            mt._worker(config)
        assert len(called) == 1


def test_legacy_stepwise_reports_match_but_mixed_paths_do_not():
    common = {"status": "complete", "seed": mt.SEEDS[0], "completed_updates": 1, "requested_updates": 1,
        "completed_scene_schedule": [5000], "scene_schedule": [5000], "initial_memory": {"same": 1},
        "base_before": {"same": 2}, "base_after": {"same": 2}, "optimizer": {"same": 3}, "data": {"same": 4}}
    carry, reset = dict(common, mode="carry"), dict(common, mode="reset")
    assert mt.matched_arms(carry, reset)
    assert mt.matched_arms(dict(carry, sequence_path="stepwise"), reset)
    assert not mt.matched_arms(dict(carry, sequence_path="batched"), reset)
    assert mt.matched_arms(dict(carry, sequence_path="batched"), dict(reset, sequence_path="batched"))
    assert not mt.matched_arms(dict(carry, sequence_path="unknown"), dict(reset, sequence_path="unknown"))


def test_invalid_sequence_path_is_rejected_before_output_creation(tmp_path):
    out = tmp_path / "untouched"
    with pytest.raises(ValueError, match="Sequence path"):
        mt.run_arm(TinyMemory(), TinyData(), [5000], "carry", out, seed=1, phase="profile", sequence_path="automatic")
    assert not out.exists()


def test_sequence_path_cli_defaults_to_stepwise_and_accepts_batched(tmp_path, monkeypatch):
    seen = []

    def plan_fixture(*args, **kwargs):
        seen.append(kwargs["sequence_path"])
        return {"status": "planned", "phase": "plan", "matched_budgets": False, "evaluation_executed": False}

    monkeypatch.setattr(mt, "run_study", plan_fixture)
    common = ["memory_train", "--train", str(tmp_path), "--validation", str(tmp_path), "--output", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", common)
    mt.main()
    monkeypatch.setattr(sys, "argv", common + ["--sequence-path", "batched"])
    mt.main()
    assert seen == ["stepwise", "batched"]


def _evaluation_study_fixture(tmp_path, monkeypatch, sequence_path="batched"):
    """Synthetic completed metadata only; no optimizer, real training or scoring."""
    class Dataset(TinyData):
        manifest_sha256 = "f" * 64

    monkeypatch.setattr(mt, "DevelopmentDataset", lambda *_: Dataset())
    root = tmp_path / "study"
    root.mkdir()
    study = {"status": "complete", "phase": "train", "matched_budgets": True,
        "requested_updates_per_arm": 1, "seeds": [mt.SEEDS[0]],
        "validation": {"manifest_sha256": Dataset.manifest_sha256}, "runs": [],
        "source_sha256": {name: mt.sha256(Path(mt.__file__).with_name(name))
                           for name in mt.training_source_names(sequence_path or "stepwise")}}
    if sequence_path is not None:
        study["sequence_path"] = sequence_path
    snapshot = root / "measured-source"
    snapshot.mkdir()
    for name in study["source_sha256"]:
        (snapshot / (name + ".txt")).write_bytes(Path(mt.__file__).with_name(name).read_bytes())
    initial = mt.memory_state(TinyMemory())
    common = {"status": "complete", "seed": mt.SEEDS[0], "completed_updates": 1, "requested_updates": 1,
        "completed_scene_schedule": [5000], "scene_schedule": [5000], "initial_memory": {"fixture": True},
        "base_before": {"fixture": True}, "base_after": {"fixture": True}, "optimizer": {}, "data": {}}
    for mode in ("carry", "reset"):
        folder = root / mode
        folder.mkdir()
        payload = {"mode": mode, "seed": mt.SEEDS[0], "completed_updates": 1,
                   "base_checkpoint_sha256": mt.BASE_SHA256, "memory_state_dict": initial}
        arm, row = dict(common, mode=mode), {"seed": mt.SEEDS[0], "mode": mode}
        if sequence_path is not None:
            payload["sequence_path"] = arm["sequence_path"] = row["sequence_path"] = sequence_path
        torch.save(payload, folder / "memory-final.pt")
        arm["memory_final_sha256"] = mt.sha256(folder / "memory-final.pt")
        mt.atomic_write(folder / "metrics.json", arm)
        row.update(metrics=f"{mode}/metrics.json", metrics_sha256=mt.sha256(folder / "metrics.json"))
        study["runs"].append(row)
    mt.atomic_write(root / "study.json", study)
    return root, study


@pytest.mark.parametrize("location", ["row", "arm", "checkpoint", "missing_source", "unknown_study"])
def test_evaluation_rejects_inconsistent_sequence_path_before_launch(tmp_path, monkeypatch, location):
    root, study = _evaluation_study_fixture(tmp_path, monkeypatch)
    row = study["runs"][0]
    metrics_path = root / row["metrics"]
    arm = json.loads(metrics_path.read_text())
    if location == "row":
        row["sequence_path"] = "stepwise"
    elif location == "arm":
        arm["sequence_path"] = "stepwise"
    elif location == "checkpoint":
        path = metrics_path.parent / "memory-final.pt"
        payload = torch.load(path, weights_only=True)
        payload["sequence_path"] = "stepwise"
        torch.save(payload, path)
        arm["memory_final_sha256"] = mt.sha256(path)
    elif location == "missing_source":
        del study["source_sha256"]["memory_sequence.py"]
    else:
        study["sequence_path"] = "unknown"
    mt.atomic_write(metrics_path, arm)
    row["metrics_sha256"] = mt.sha256(metrics_path)
    mt.atomic_write(root / "study.json", study)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Worker must not start"))
    output = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="sequence path"):
        evaluate_study(root, tmp_path, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["missing", "modified"])
def test_evaluation_rejects_changed_retained_training_source(tmp_path, monkeypatch, change):
    root, _ = _evaluation_study_fixture(tmp_path, monkeypatch)
    path = root / "measured-source/memory_sequence.py.txt"
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"\n# changed after training\n")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Worker must not start"))
    with pytest.raises(ValueError, match="Retained training source snapshot"):
        evaluate_study(root, tmp_path, tmp_path / "uncreated")
    assert not (tmp_path / "uncreated").exists()


@pytest.mark.parametrize("sequence_path", [None, "stepwise", "batched"])
def test_evaluation_records_path_and_training_provenance(tmp_path, monkeypatch, sequence_path):
    from experiments.room_world import memory_evaluate as me
    root, study = _evaluation_study_fixture(tmp_path, monkeypatch, sequence_path)
    configs = []

    def synthetic_handoff(process, config, folder, **kwargs):
        configs.append(config)
        out = Path(config["output"])
        out.mkdir()
        identity = {key: config[key] for key in ("mode", "seed", "updates", "checkpoint_sha256",
                    "sequence_path", "training_sequence_path", "training_source_sha256", "training_study_sha256")}
        mt.atomic_write(out / "evaluation.json", {"status": "complete", "provenance": identity})
        return {"status": "complete"}

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: object())
    monkeypatch.setattr(mt, "handoff_and_guard", synthetic_handoff)
    monkeypatch.setattr(me, "compare_evaluations", lambda *a, **k: {"status": "unavailable", "fixture_only": True})
    report = me.evaluate_study(root, tmp_path, tmp_path / "evaluation")
    expected = sequence_path or "stepwise"
    assert report["sequence_path"] == expected
    assert report["training_source_sha256"] == study["source_sha256"]
    assert [c["sequence_path"] for c in configs] == [expected, expected, None]
    assert all(c["training_sequence_path"] == expected for c in configs)
    assert all(c["training_source_sha256"] == study["source_sha256"] for c in configs)
    assert all(c["training_study_sha256"] == mt.sha256(root / "study.json") for c in configs)
    assert [r["sequence_path"] for r in report["runs"]] == [expected, expected, None]


def test_evaluation_worker_rechecks_checkpoint_path_and_records_real_config(tmp_path, monkeypatch):
    from experiments.room_world import memory_evaluate as me
    root, study = _evaluation_study_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(mt, "make_model", lambda *a: TinyMemory())
    seen = []
    monkeypatch.setattr(me, "evaluate_model", lambda *a, **k: seen.append(k))
    path = root / "carry/memory-final.pt"
    config = {"mode": "carry", "seed": mt.SEEDS[0], "updates": 1, "device": "cpu", "base": str(mt.BASE),
        "checkpoint": str(path), "checkpoint_sha256": mt.sha256(path), "sequence_path": "batched",
        "training_sequence_path": "batched", "training_source_sha256": study["source_sha256"],
        "training_study_sha256": mt.sha256(root / "study.json"), "data": str(tmp_path),
        "data_sha256": "f" * 64, "output": str(tmp_path / "unused"),
        "source_sha256": {name: mt.sha256(Path(me.__file__).with_name(name))
                           for name in ("memory_evaluate.py", "memory_train.py", "memory_model.py", "model.py")}}
    me._evaluation_worker(config)
    assert seen[0]["provenance"]["sequence_path"] == "batched"
    assert seen[0]["provenance"]["training_source_sha256"] == study["source_sha256"]
    payload = torch.load(path, weights_only=True)
    payload["sequence_path"] = "stepwise"
    torch.save(payload, path)
    config["checkpoint_sha256"] = mt.sha256(path)
    with pytest.raises(ValueError, match="sequence path"):
        me._evaluation_worker(config)
    assert len(seen) == 1
    config["training_sequence_path"] = "stepwise"
    with pytest.raises(ValueError, match="sequence paths"):
        me._evaluation_worker(config)
