# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU checks for probe orchestration and evidence, never real CUDA."""
import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from . import probe, probe_evidence as evidence, probe_math as numerical
from .test_cpu import fixture


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")
    return path


def test_original_fixed_hyperparameters_and_separate_prefix_bound():
    record = probe.protocol("spatial")
    assert record["paired_updates"] == 2 and record["starts"] == [0, 8]
    assert record["shape"] == [1, 48, 5, 44, 78]
    assert record["adapter_parameters"] == 947712
    assert record["native_parity"] == {"max_absolute": 1e-6, "relative_l2": 1e-6}
    assert record["cross_length_codec_prefix"] == {"max_absolute": 1e-5, "relative_l2": 1e-5}
    assert record["limits"]["seconds"] == 900 and record["limits"]["cuda_reserved_bytes"] == 60*2**30
    assert record["optimizer"] == {"lr": 1e-4, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01}
    assert not record["fixed16_admitted"] and not record["image_generation"]


def test_cross_length_tolerance_never_rewrites_saved_target():
    target = torch.ones(1, 2, 5, 4, 4)
    observation = torch.ones(1, 2, 1, 4, 4)
    target[:, :, :1] += 2e-6
    before = target.clone()
    noise = torch.randn_like(target)
    noisy, times, velocity = numerical.flow_inputs(target, observation, noise, 631)
    assert torch.equal(target, before) and torch.equal(noisy[:, :, :1], observation)
    assert torch.equal(velocity, noise-target)
    assert torch.equal(noisy[:, :, 1:], (1-.631)*target[:, :, 1:]+.631*noise[:, :, 1:])
    assert torch.equal(times[:, :4], torch.zeros(1, 4, dtype=torch.int64))
    assert (times[:, 4:] == 631).all()
    target[:, :, :1] += .001
    with pytest.raises(ValueError, match="prefix"):
        numerical.flow_inputs(target, observation, noise, 631)


def test_future_loss_ignores_the_entire_initial_velocity():
    prediction = torch.randn(1, 2, 5, 4, 4, requires_grad=True)
    velocity = torch.randn_like(prediction)
    loss = numerical.future_flow_mse(prediction, velocity)
    changed = velocity.clone(); changed[:, :, :1] += 1000
    assert torch.equal(loss, numerical.future_flow_mse(prediction, changed))
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, :, :1]) == 0
    assert prediction.grad[:, :, 1:].abs().sum() > 0


def math_case():
    bridge, x, times, contexts, commands, observation = fixture()
    schedule, draws = numerical.make_draws("probe", shape=tuple(x.shape))
    windows = {}
    for start in (0, 8):
        for arm in ("closed", "open"):
            target = x.clone()
            if arm == "open":
                target[:, :, 1:] *= .7
            command = commands.clone(); command[:, 0, 5] = int(arm == "open")
            windows[f"{arm}-{start:04d}"] = {"target": target, "observation": observation.clone(), "commands": command}
    optimizer = torch.optim.AdamW(bridge.adapter.parameters(), **numerical.OPTIMIZER)
    return bridge, schedule, draws, windows, contexts[0], optimizer


def test_two_update_protocol_checkpoints_order_counts_and_saved_comparisons():
    bridge, schedule, draws, windows, context, optimizer = math_case()
    retained, checkpoints, progress = {}, [], []
    def native(x, t, c):
        with torch.no_grad():
            return torch.stack(bridge.core(list(x.unbind(0)), t, [c], t.shape[1]))
    result = numerical.execute_steps(bridge, windows, schedule, draws, context, optimizer,
        native_predict=native, retain=lambda name, values: retained.update({name: values}),
        checkpoint=lambda step, report: checkpoints.append((step, len(optimizer.state))),
        progress=lambda value: progress.append(copy.deepcopy(value)))
    assert [step for step, _ in checkpoints] == [0, 1, 2]
    assert checkpoints[0][1] == 0 and all(count == 18 for _, count in checkpoints[1:])
    assert result["completed_updates"] == 2 and result["native_reference_predictions"] == 2 and result["bridge_predictions"] == 6
    assert len(retained) == 4 and all(row["exact_equal"] for row in result["native_comparisons"])
    assert result["updates"][0]["command_gru_gradient_l2"] == 0
    assert result["updates"][1]["command_gru_gradient_l2"] > 0
    assert all(item["completed_updates"] == 0 for item in progress if not item["zero_adapter_gate_passed"])
    assert all(parameter.grad is None for parameter in bridge.core.parameters())


def test_nonfinite_reference_is_retained_and_prevents_optimizer_step():
    bridge, schedule, draws, windows, context, optimizer = math_case()
    retained, steps = {}, []
    before = {name: value.clone() for name, value in bridge.adapter.state_dict().items()}
    def bad(x, t, c):
        value = torch.zeros_like(x); value.flatten()[0] = float("nan"); return value
    with pytest.raises(ValueError, match="finite"):
        numerical.execute_steps(bridge, windows, schedule, draws, context, optimizer,
            native_predict=bad, retain=lambda name, values: retained.update({name: values}),
            checkpoint=lambda step, report: steps.append(step), progress=lambda value: None)
    assert any(torch.isnan(value).any() for values in retained.values() for value in values.values())
    assert steps == [0] and not optimizer.state
    assert all(torch.equal(before[name], value) for name, value in bridge.adapter.state_dict().items())


def test_weight_catalog_requires_all_825_original_fp32_identities():
    expected = evidence.read_json(evidence.HERE.parent/"cuda_reference/expected-weights.json")["tensors"]
    rows = {name: {"source_sha256": row["original_sha256"], "loaded_sha256": row["original_sha256"],
                  "original_dtype": "float32", "loaded_dtype": "float32", "shape": row["shape"], "shard": row["shard"],
                  "cuda_copy_exact": True, "source_owner_released": True} for name, row in expected.items()}
    record = {"tensor_count": 825, "parameter_count": 4999787712, "parameter_bytes": 19999150848,
              "convert_model_dtype": False, "all_shards_verified": True, "cuda_copy_exact": True, "tensors": rows}
    assert len(probe._original_weights(record)) == 825
    one = next(iter(rows)); rows[one]["loaded_dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="identity"):
        probe._original_weights(record)
    rows.pop(one)
    with pytest.raises(ValueError):
        probe._original_weights(record)


def admission_fixture(tmp_path):
    root = tmp_path/"prepared"; root.mkdir()
    plan = {"profile": "spatial", "source_sha256": {"bridge": "a"*64}, "input_identity": {"cache": "b"*64}}
    put(root/"plan.json", plan)
    audit = put(tmp_path/"audit.json", {"status": "passed"})
    visual = tmp_path/"visual.txt"; visual.write_text("Recognizable near-static room. No action-quality result.")
    record = {"schema": "worldline-wan22-action-cuda-probe-admission-v1", "decision": "admit", "issued_by": "parent-agent",
              "scope": evidence.SCOPE, "profile": "spatial", "plan_sha256": evidence.sha(root/"plan.json"),
              "source_sha256": plan["source_sha256"], "input_identity": plan["input_identity"],
              "expected_gpu": "NVIDIA A100-SXM4-80GB", "limits": probe.LIMITS,
              "image_generation_admitted": False, "fixed16_admitted": False,
              "visual_review": visual.read_text(), "foundation_diagnostic_visual_status": "passed", "mps_visual_status": "failed",
              "completed_audits": [{"file": str(audit), "sha256": evidence.sha(audit)}],
              "visual_evidence": [{"file": str(visual), "sha256": evidence.sha(visual)}]}
    path = put(tmp_path/"admission.json", record)
    return root, plan, path, record


@pytest.mark.parametrize("field", ["scope", "plan_sha256", "profile", "fixed16_admitted", "input_identity"])
def test_admission_is_exact_numerical_only(tmp_path, field):
    root, plan, path, record = admission_fixture(tmp_path)
    accepted = evidence.validate_admission(path, root, plan, "NVIDIA A100-SXM4-80GB")
    assert accepted["quality_admitted"] is False
    record[field] = True if field == "fixed16_admitted" else "changed"
    put(path, record)
    with pytest.raises(ValueError):
        evidence.validate_admission(path, root, plan, "NVIDIA A100-SXM4-80GB")


def test_failed_or_changed_audit_cannot_admit(tmp_path):
    root, plan, path, record = admission_fixture(tmp_path)
    audit = Path(record["completed_audits"][0]["file"])
    put(audit, {"status": "failed"})
    with pytest.raises(ValueError, match="evidence"):
        evidence.validate_admission(path, root, plan, "NVIDIA A100-SXM4-80GB")
    record["completed_audits"][0]["sha256"] = evidence.sha(audit)
    put(path, record)
    with pytest.raises(ValueError, match="passed"):
        evidence.validate_admission(path, root, plan, "NVIDIA A100-SXM4-80GB")


def test_tensor_headers_and_symlinks_are_rejected_before_loading(tmp_path):
    path = tmp_path/"value.safetensors"
    save_file({"x": torch.ones(2, 3)}, str(path))
    assert evidence.tensors(path, {"x": ((2, 3), "F32")}, evidence.sha(path))["x"].shape == (2, 3)
    with pytest.raises(ValueError, match="header"):
        evidence.tensors(path, {"x": ((3, 2), "F32")}, evidence.sha(path))
    with pytest.raises(ValueError, match="keys"):
        evidence.tensors(path, {"other": ((2, 3), "F32")}, evidence.sha(path))
    link = tmp_path/"link"; link.symlink_to(path)
    with pytest.raises(ValueError):
        evidence.tensors(link, {"x": ((2, 3), "F32")}, evidence.sha(path))
    with pytest.raises(ValueError):
        evidence.relative_file(tmp_path, "../elsewhere")


def test_review_rejects_failed_skipped_or_contradictory_exit(tmp_path, monkeypatch):
    mapping = {"test": "a"*64}
    monkeypatch.setattr(evidence, "source_hashes", lambda **kwargs: mapping)
    report = {"status": "passed", "tests": 5, "source_sha256": mapping, "sources_unchanged": True,
              "pytest_exit_code": 0, "failures": 0, "errors": 0, "skipped": 0, "independent_review": True}
    path = put(tmp_path/"review.json", report)
    assert evidence.review(path, independent=True) == evidence.sha(path)
    for field in ("pytest_exit_code", "failures", "errors", "skipped"):
        changed = {**report, field: 1}; put(path, changed)
        with pytest.raises(ValueError):
            evidence.review(path, independent=True)


def test_prepared_bytes_are_consumed_without_seed_regeneration(tmp_path, monkeypatch):
    source = evidence.HERE/"bridge.py"
    mapping = {str(source.relative_to(evidence.REPO)): evidence.sha(source)}
    monkeypatch.setattr(evidence, "source_paths", lambda **kwargs: {name: source for name in mapping})
    monkeypatch.setattr(evidence, "source_hashes", lambda **kwargs: mapping)
    monkeypatch.setattr(evidence, "bridge_reviews", lambda: {})
    monkeypatch.setattr(evidence, "review", lambda path, **kwargs: evidence.sha(path))
    cache_identity = {"canonical_manifest": "b"*64}
    monkeypatch.setattr(evidence, "load_cache", lambda *args: ({}, cache_identity))
    positive = torch.ones(25, 4096)
    monkeypatch.setattr(evidence, "load_positive", lambda *args: (positive, {"positive_tensor_sha256": evidence.tensor_sha(positive)}))
    cpu = put(tmp_path/"cpu.json", {"fixture": True})
    independent = put(tmp_path/"independent.json", {"fixture": True})
    out = tmp_path/"prepared"
    plan = probe.prepare(cache_run=tmp_path/"cache", text_directory=tmp_path/"text", profile="baseline",
                         cpu_report=cpu, independent_report=independent, output=out)
    def forbidden(*args, **kwargs):
        raise AssertionError("Execution must consume saved bytes, not regenerate")
    monkeypatch.setattr(numerical, "fresh_adapter", forbidden)
    monkeypatch.setattr(numerical, "make_draws", forbidden)
    actual, _, initial, draws, text, rng = probe.read_prepared(out, tmp_path/"cache")
    assert actual == plan and torch.equal(text, positive)
    assert torch.count_nonzero(initial["output.weight"]) == 0
    assert draws["noise_0000"].shape == (1, 48, 5, 18, 32) and rng.dtype == torch.uint8
    path = out/"draws.safetensors"
    with path.open("r+b") as stream:
        stream.seek(-1, 2); byte = stream.read(1); stream.seek(-1, 2); stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="differs"):
        probe.read_prepared(out, tmp_path/"cache")


@pytest.mark.parametrize("partial", [None, True, False])
def test_supervisor_failure_retains_unknown_or_measured_execution(tmp_path, monkeypatch, partial):
    root, plan, admission, _ = admission_fixture(tmp_path)
    plan.update(protocol=probe.protocol("spatial"))
    monkeypatch.setattr(probe, "read_prepared", lambda *args: (plan,))
    monkeypatch.setattr(evidence, "validate_admission", lambda *args: {"admission_sha256": "a"*64})
    class Process:
        returncode = None
    process = Process()
    captured = {}
    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        if partial is not None:
            put(root/"result/metrics.json", {"status": "failed", "model_execution": partial})
        return process
    def stop(value):
        value.returncode = -15
    def fail(*args):
        raise KeyboardInterrupt("fixture supervisor interruption")
    monkeypatch.setattr(probe.subprocess, "Popen", popen)
    monkeypatch.setattr(probe, "stop_child", stop)
    monkeypatch.setattr(probe, "supervise", fail)
    with pytest.raises(KeyboardInterrupt):
        probe.execute(prepared_directory=root, cache_run=tmp_path/"cache", weights=tmp_path/"weights",
                      expected_gpu="NVIDIA A100-SXM4-80GB", admission=admission)
    report = evidence.read_json(root/"metrics.json")
    assert report["model_execution"] is partial and report["status"] == "interrupted"
    assert captured["stdin"] is probe.subprocess.DEVNULL
    assert "--worker-config" in captured["command"]
    assert evidence.read_json(root/"terminal.json")["exit_code"] == -15


def test_expired_deadline_rejected_before_child_creation(tmp_path, monkeypatch):
    root, plan, admission, _ = admission_fixture(tmp_path)
    monkeypatch.setattr(probe, "read_prepared", lambda *args: (plan,))
    monkeypatch.setattr(evidence, "validate_admission", lambda *args: {})
    ticks = iter([0., 901., 902.])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    def forbidden(*args, **kwargs):
        raise AssertionError("Expired deadline must never launch")
    monkeypatch.setattr(probe.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="deadline"):
        probe.execute(prepared_directory=root, cache_run=tmp_path/"cache", weights=tmp_path/"weights",
                      expected_gpu="NVIDIA A100-SXM4-80GB", admission=admission)
    assert evidence.read_json(root/"metrics.json")["model_execution"] is False


def test_cleanup_error_does_not_replace_original_interruption(tmp_path, monkeypatch):
    root, plan, admission, _ = admission_fixture(tmp_path)
    monkeypatch.setattr(probe, "read_prepared", lambda *args: (plan,))
    monkeypatch.setattr(evidence, "validate_admission", lambda *args: {})
    class Process:
        returncode = None
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *args, **kwargs: Process())
    def interruption(*args):
        raise KeyboardInterrupt("original interruption")
    def cleanup(*args):
        raise RuntimeError("cleanup failed")
    monkeypatch.setattr(probe, "supervise", interruption)
    monkeypatch.setattr(probe, "stop_child", cleanup)
    with pytest.raises(KeyboardInterrupt, match="original"):
        probe.execute(prepared_directory=root, cache_run=tmp_path/"cache", weights=tmp_path/"weights",
                      expected_gpu="NVIDIA A100-SXM4-80GB", admission=admission)
    terminal = evidence.read_json(root/"terminal.json")
    assert terminal["cleanup_error"] == "cleanup failed"
    assert terminal["error"] == "original interruption"
