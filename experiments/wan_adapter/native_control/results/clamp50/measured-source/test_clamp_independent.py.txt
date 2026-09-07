"""Independent CPU checks for the single-factor observed-image clamp."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open as actual_safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from native_control import clamp_cache, clamp_loop, sampling
from native_control.clamp_run import validate_control

torch.set_num_threads(2)


def fixtures():
    noise = sampling.initial_noise()
    observation = torch.randn(16, 1, 36, 64, generator=torch.Generator().manual_seed(591))
    return noise, observation, torch.full((1, 4096), 2.0), torch.full((3, 4096), 3.0)


def test_published_cache_reader_materializes_only_the_independent_observation():
    root = Path(__file__).parent.parent / "data_cache"
    requested = []

    class Spy:
        def __init__(self, path, **kwargs):
            self.handle = actual_safe_open(path, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def get_tensor(self, key):
            requested.append(key)
            assert key == "observation", "Actions or target values reached the clamp reader"
            return self.handle.get_tensor(key)

    with patch.object(clamp_cache, "safe_open", Spy):
        observation, provenance = clamp_cache.read_observation(root)
    assert requested == ["observation"]
    assert observation.shape == (16, 1, 36, 64) and observation.dtype == torch.float32
    assert torch.isfinite(observation).all()
    assert provenance["materialized_tensor_keys"] == ["observation"]
    assert provenance["target_materialized"] is False and provenance["actions_materialized"] is False
    assert provenance["causal_cache_checks_passed"] == 11
    assert provenance["encoding_compute_dtype"] == "float16"


def test_clamp_replaces_only_first_frame_without_mutating_input_storage():
    noise, observation, _, _ = fixtures()
    saved_noise, saved_observation = noise.clone(), observation.clone()
    result = clamp_loop.clamp_first(noise, observation)
    assert torch.equal(result[:, :1], observation)
    assert torch.equal(result[:, 1:], noise[:, 1:])
    result.add_(1)
    assert torch.equal(noise, saved_noise) and torch.equal(observation, saved_observation)


def test_all_100_cfg_inputs_and_all_50_solver_outputs_have_the_exact_prefix():
    noise, observation, negative, positive = fixtures()
    saved_noise, saved_observation = noise.clone(), observation.clone()
    calls, records = [], []

    def model(values, timestep, context, seq_len):
        assert len(values) == len(context) == 1 and seq_len == 2880
        assert torch.equal(values[0][:, :1], observation)
        assert context[0] is (negative if len(calls) % 2 == 0 else positive)
        if len(calls) % 2:
            assert torch.equal(calls[-1][1], values[0])
        calls.append((timestep.item(), values[0].clone()))
        return [torch.full_like(values[0], float(context[0][0, 0]))]

    def callback(index, timestep, latent, timing):
        assert torch.equal(latent[:, :1], observation)
        assert timing["clamped_prefix_max_abs_difference"] == 0
        records.append((index, timestep.item()))

    actual = clamp_loop.integrate_clamped(model, noise, observation, negative, positive,
                                          device="cpu", callback=callback)
    scheduler = sampling.make_scheduler()
    assert len(calls) == 100 and len(records) == 50
    assert [row[0] for row in calls[::2]] == scheduler.timesteps.tolist()
    expected_suffix = noise[:, 1:] - scheduler.sigmas[0] * 8.0
    torch.testing.assert_close(actual[:, 1:], expected_suffix, atol=0.00003, rtol=0.00001)
    assert torch.equal(actual[:, :1], observation)
    assert torch.equal(noise, saved_noise) and torch.equal(observation, saved_observation)


def test_changing_velocity_matches_independent_projected_unipc_reference():
    noise, observation, negative, positive = fixtures()

    def velocity(value, timestep, context):
        return value * 0.02 + float(context[0, 0]) * 0.001 + timestep.item() * 0.000001

    def model(values, timestep, contexts, seq_len):
        return [velocity(values[0], timestep, contexts[0])]

    actual = clamp_loop.integrate_clamped(model, noise, observation, negative, positive, device="cpu")
    expected = noise.clone()
    expected[:, :1] = observation
    scheduler = sampling.make_scheduler()
    for timestep in scheduler.timesteps:
        expected[:, :1] = observation
        negative_value = velocity(expected, timestep, negative)
        positive_value = velocity(expected, timestep, positive)
        guided = negative_value + 6.0 * (positive_value - negative_value)
        expected = scheduler.step(guided.unsqueeze(0), timestep, expected.unsqueeze(0), return_dict=False)[0][0]
        expected[:, :1] = observation
    assert torch.equal(actual, expected)


def test_control_gate_rejects_different_settings_and_changed_successful_source():
    here = Path(__file__).parent
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        text_cache = root / "text"
        text_cache.mkdir()
        (text_cache / "manifest.json").write_text("{}")
        profile = {"base": {"test": "same"},
                   "text": {"manifest_sha256": sha(text_cache / "manifest.json")},
                   "scheduler": {"test": "same"}}
        source_names = ("portable.py", "sampling.py", "loop.py", "run_clip.py", "profile_pair.py",
                        "evidence.py", "vendor/model.py", "vendor/fm_solvers_unipc.py",
                        "../codec/helper.py", "../codec/decode_policy.py", "../codec/vendor/wan_vae.py")
        control = {"status": "passed", "steps": list(range(50)), "observed_frames": 0,
                   "generated_frames": 17, "seed": sampling.SEED, "shift": sampling.SHIFT,
                   "guidance": sampling.GUIDANCE, "declared_solver_steps": 50, "declared_model_calls": 100,
                   **profile, "source_sha256": {name: sha(here / name) for name in source_names}}
        path = root / "metrics.json"

        def validate():
            path.write_text(json.dumps(control))
            return validate_control(root, profile, text_cache)

        assert validate()["status"] == "passed"
        control["guidance"] = 5
        with unittest.TestCase().assertRaisesRegex(ValueError, "settings differ"):
            validate()
        control["guidance"] = sampling.GUIDANCE
        control["source_sha256"]["loop.py"] = "0" * 64
        with unittest.TestCase().assertRaisesRegex(ValueError, "Successful source changed"):
            validate()


def load_tests(loader, standard_tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in sorted(globals().items())
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Test evidence file must be new")
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(load_tests(None, None, None))
    if args.output is not None:
        names = ("test_clamp_independent.py", "clamp_cache.py", "clamp_loop.py", "clamp_run.py",
                 "sampling.py", "loop.py", "run_clip.py", "portable.py", "evidence.py")
        report = {"status": "passed" if result.wasSuccessful() else "failed", "device": "cpu",
                  "tests_run": result.testsRun, "elapsed_seconds": time.monotonic() - started,
                  "errors": [(str(test), error) for test, error in result.errors],
                  "failures": [(str(test), error) for test, error in result.failures],
                  "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                    for name in names},
                  "limits": "Equation and access checks only; no GPU, foundation load or quality result."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
