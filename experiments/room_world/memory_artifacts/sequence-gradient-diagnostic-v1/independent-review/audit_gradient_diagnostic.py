"""CPU-only independent audit of retained MPS tensors; performs no model forward."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = RUN = FAILED = PROFILE = TRAIN = REPO = OUTPUT = TENSORS = None


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument('--diagnostic', type=Path)
    parser.add_argument('--tensors', type=Path, required=True,
                        help='Exact raw .pt file reconstructed from diagnostic-tensors.pt.gz')
    parser.add_argument('--failed-probe', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--train', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New JSON output file')
    args = parser.parse_args()
    global RUN, FAILED, PROFILE, TRAIN, REPO, OUTPUT, TENSORS
    REPO = args.repo.resolve()
    artifacts = REPO / 'experiments/room_world/memory_artifacts'
    RUN = (args.diagnostic or artifacts / 'sequence-gradient-diagnostic-v1').resolve()
    FAILED = (args.failed_probe or artifacts / 'sequence-mps-failure-v1').resolve()
    PROFILE = (args.profile or artifacts / 'profile-v1').resolve()
    TRAIN = (args.train or artifacts / 'data/train').resolve()
    OUTPUT, TENSORS = args.output.resolve(), args.tensors.resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-20), (a, b)


def difference(candidate, reference):
    candidate, reference = candidate.double(), reference.double()
    delta, norm = candidate - reference, reference.norm().item()
    return dict(max_abs_difference=delta.abs().max().item(),
                l2_difference=delta.norm().item(),
                relative_l2_difference=delta.norm().item() / norm if norm else None,
                reference_l2_norm=norm, candidate_l2_norm=candidate.norm().item(),
                finite=bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all()))


def verify_difference(candidate, reference, reported):
    actual = difference(candidate, reference)
    for key, value in actual.items():
        close(value, reported[key])
    return actual


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(2)
    r, launch, terminal = (read(RUN / name) for name in ("diagnostic.json", "launch.json", "terminal.json"))
    failed, study, manifest = read(FAILED / "probe.json"), read(PROFILE / "study.json"), read(TRAIN / "manifest.json")
    assert r["status"] == "complete" and r["optimizer_updates"] == 0 and r["device"] == "mps"
    assert r["mode"] == "carry" and r["seed"] == 20260907 and r["scene"] == 5023
    assert failed["status"] == "failed" and r["original_l1_result"] == "failed and retained unchanged"
    assert sha(FAILED / "probe.json") == r["failed_probe_sha256"] == launch["failed_probe_sha256"]
    assert sha(PROFILE / "study.json") == r["profile_study_sha256"]
    assert sha(TRAIN / "manifest.json") == r["train_manifest_sha256"] == study["training"]["manifest_sha256"]
    assert r["thresholds"] == failed["thresholds"] == dict(loss_abs=2e-6,
        parameter_gradient_abs_max=1e-6, parameter_gradient_relative_l2=1e-3,
        rgb_gradient_abs_max=1e-7, rgb_gradient_relative_l2=1e-3)
    assert r["source_sha256"] == launch["source_sha256"] == failed["source_sha256"]
    source_checks = {}
    for name, expected in r["source_sha256"].items():
        assert sha(RUN / "measured-source" / (name + ".txt")) == expected
        current = REPO / "experiments/room_world" / name
        source_checks[name] = {"measured_sha256": expected, "snapshot_verified": True,
                              "current_sha256": sha(current)}
    assert sha(RUN / "measured-source/diagnose-room-memory-gradients.py.txt") == r["script_sha256"] == launch["script_sha256"]
    assert sha(RUN / "measured-source/run-room-memory-sequence-mps.py.txt") == launch["original_probe_script_sha256"] == failed["script_sha256"]
    old_file = FAILED / failed["modes"]["carry"][0]["gradient_file"]
    assert sha(old_file) == r["previous_gradients_sha256"] == failed["modes"]["carry"][0]["gradient_file_sha256"]
    old = torch.load(old_file, weights_only=True, map_location="cpu")
    checkpoint = PROFILE / "20260907/carry/training/memory-final.pt"
    assert sha(checkpoint) == r["checkpoint_sha256"]
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert payload["completed_updates"] == 50 and payload["mode"] == "carry" and payload["seed"] == r["seed"]
    assert sha(REPO / "experiments/room_world/artifacts/predictor/model.pt") == r["base_checkpoint_sha256"] == payload["base_checkpoint_sha256"]
    assert study["schedules"][str(r["seed"])][0] == r["scene"]
    record = next(row for row in manifest["records"] if row["scene_seed"] == r["scene"])
    capture = TRAIN / record["file"]
    assert sha(capture) == record["file_sha256"]
    assert manifest["reserved_test_generated"] is False and manifest["scene_seeds"] == list(range(5000, 5032))
    with np.load(capture, allow_pickle=False) as archive:
        assert set(archive.files) == {"observations", "actions"}
        rgb, actions = archive["observations"], archive["actions"]
    for name, value in (("observations", rgb), ("actions", actions)):
        assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == record[name + "_sha256"]
    assert list(rgb.shape) == r["shape"] and list(actions.shape) == r["action_shape"]
    assert np.array_equal(actions, [[0] + [3]*24 + [0]*16 + [4]*24, [5] + [3]*24 + [0]*16 + [4]*24])
    artifact = TENSORS
    assert sha(artifact) == r["artifact_sha256"]
    tensors = torch.load(artifact, weights_only=True, map_location="cpu")
    assert set(tensors) == set(r["tensor_hashes"]) and len(tensors) == 39
    for name, value in tensors.items():
        assert value.device.type == "cpu" and torch.isfinite(value).all()
        assert {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": raw(value)} == r["tensor_hashes"][name]
    truth = torch.from_numpy(rgb.copy()).float() / 127.5 - 1
    assert torch.equal(tensors["targets"], truth[:, 1:])
    mismatch = []
    for prefix in ("reference", "batched"):
        residual = tensors[prefix + "_prediction"] - tensors["targets"]
        assert torch.equal(residual, tensors[prefix + "_residual"])
        signs = residual.sign()
        assert torch.equal(signs.to(torch.int8), tensors[prefix + "_residual_sign"])
        assert torch.equal(signs, tensors[prefix + "_native_upstream"].sign())
        unit = torch.tensor(1 / (2 * residual.numel()), dtype=torch.float32)
        assert torch.equal(signs * unit, tensors[prefix + "_native_upstream"])
    assert torch.equal(tensors["common_upstream"], tensors["reference_native_upstream"])
    sr, br = tensors["reference_residual"], tensors["batched_residual"]
    coords = (sr.sign() != br.sign()).nonzero().tolist()
    assert coords == [[0, 14, 0, 9, 18], [0, 17, 0, 13, 24], [0, 37, 0, 17, 61]]
    assert not ((sr.sign() * br.sign()) < 0).any()
    for coord, example in zip(coords, r["residual_signs"]["examples"]):
        idx = tuple(coord)
        assert coord == example["coordinate"] and (sr[idx] == 0 or br[idx] == 0)
        assert max(abs(sr[idx]), abs(br[idx])) == 2**-24
        for key in ("reference_prediction", "batched_prediction", "reference_residual", "batched_residual"):
            assert tensors[key][idx].item() == example[key]
        mismatch.append(example)
    verify_difference(tensors["batched_prediction"], tensors["reference_prediction"], r["prediction_comparison"])
    scalar_delta = abs(r["paths"]["stepwise"]["ordinary_l1"] - r["paths"]["batched"]["ordinary_l1"])
    assert scalar_delta == 0 == r["ordinary_l1_loss_abs_difference"]
    # CPU reductions can differ from MPS. Reconstructing in double is a cross-check, not a claim of bit equality.
    loss_reconstruction = {p: float(tensors[p + "_residual"].double().abs().mean() / 2) for p in ("reference", "batched")}
    assert all(abs(v - r["paths"]["stepwise"]["ordinary_l1"]) < 2e-9 for v in loss_reconstruction.values())
    names = list(r["ordinary_l1_gradient_comparison"])
    checked, decomposition = {}, {}
    for name in names:
        sn, sc, bn, bc = (tensors[f"{path}.{kind}.{name}"] for path, kind in
                          (("stepwise", "native"), ("stepwise", "common"), ("batched", "native"), ("batched", "common")))
        native = verify_difference(bn, sn, r["ordinary_l1_gradient_comparison"][name])
        common = verify_difference(bc, sc, r["common_upstream_gradient_comparison"][name])
        for path, value in (("stepwise", sn), ("batched", bn)):
            verify_difference(value, old[path][name], r["paths"][path]["native_vs_original_failed_probe"][name])
        sn, sc, bn, bc = [v.double() for v in (sn, sc, bn, bc)]
        terms = dict(native_delta=bn-sn, common_backward_delta=bc-sc,
                     candidate_native_minus_common=bn-bc, reference_common_minus_native=sc-sn)
        terms["identity_residual"] = terms["native_delta"] - sum(terms[k] for k in
            ("common_backward_delta", "candidate_native_minus_common", "reference_common_minus_native"))
        assert terms["identity_residual"].count_nonzero() == 0
        for key, value in terms.items():
            verify_difference(value, torch.zeros_like(value), r["gradient_decomposition"][name][key])
        threshold = "rgb_gradient" if name == "rgb_input" else "parameter_gradient"
        passed = lambda v: v["finite"] and v["max_abs_difference"] <= r["thresholds"][threshold + "_abs_max"] and (
            v["relative_l2_difference"] <= r["thresholds"][threshold + "_relative_l2"] if v["relative_l2_difference"] is not None else v["l2_difference"] == 0)
        checked[name] = {"native": native, "native_pass": passed(native), "common": common, "common_pass": passed(common)}
        decomposition[name] = {key: difference(value, torch.zeros_like(value)) for key, value in terms.items()}
    assert not all(v["native_pass"] for v in checked.values()) and all(v["common_pass"] for v in checked.values())
    assert [k for k, v in checked.items() if not v["native_pass"]] == ["rgb_input"]
    direct = torch.zeros_like(tensors["stepwise.native.rgb_input"])
    direct[:, 1:] = -(tensors["batched_native_upstream"] - tensors["reference_native_upstream"])
    assert torch.equal(direct, tensors["direct_target_gradient_change"])
    assert torch.count_nonzero(direct) == 3
    verify_difference(tensors["batched.native.rgb_input"] - tensors["stepwise.native.rgb_input"], direct,
                      r["direct_target_term"]["comparison_to_native_rgb_gradient_change"])
    assert terminal["status"] == "complete" and terminal["exit_code"] == 0 and terminal["elapsed_seconds"] <= 180
    assert max(s["rss_bytes"] for s in terminal["samples"]) <= 18 * 2**30
    assert min(s["available_bytes"] for s in terminal["samples"]) >= 2 * 2**30
    report = dict(schema="worldline-room-memory-gradient-independent-audit-v1", status="passed",
        audit_device="cpu", model_forwards=0, optimizer_updates=0, reserved_test_opened=False,
        source_sha256=sha(__file__), diagnostic_sha256=sha(RUN / "diagnostic.json"),
        diagnostic_tensors_sha256=sha(artifact), failed_probe_sha256=sha(FAILED / "probe.json"),
        original_l1_probe_status="failed, unchanged", ordinary_l1_rgb_gradient_gate="failed, unchanged",
        measured_source_checks=source_checks, verified_tensor_count=len(tensors),
        verified_target="Exact normalized next observations from hash-verified training scene 5023",
        common_upstream_equals_reference_autograd=True, residual_sign_mismatch_count=3,
        opposite_nonzero_sign_count=0, residual_examples=mismatch,
        scalar_mps_loss_difference=scalar_delta, cpu_double_loss_crosscheck=loss_reconstruction,
        gradient_checks=checked, decomposition=decomposition,
        finding="On this one carry-state pair, three zero-to-one-ULP residual changes alter the ordinary L1 subgradient. With the identical retained upstream, both backward paths meet the unchanged limits. The direct target derivative is only part of the change; prediction derivatives propagate the changed upstream to earlier RGB.",
        readiness="Ready for a separate, explicitly batched 50-update-per-arm timing and integrity profile with fresh matched initialization and a shared schedule. The previous failed probe is not reclassified. This audit does not establish identical optimization trajectories, reset-mode MPS equivalence, generalization, or memory quality.")
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "verified_tensor_count", "residual_sign_mismatch_count", "original_l1_probe_status", "readiness")}, indent=2))


if __name__ == "__main__":
    arguments()
    main()
