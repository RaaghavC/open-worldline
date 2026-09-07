# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU boundary tests: no foundation forward, weights download or GPU."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenwise_time import pilot_common as pc
from native_control.sampling import make_scheduler


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rng_draws_match_independent_order_and_ignore_target_values(self):
        shape = (1, 16, 5, 4, 6)
        windows = {name: {"target": torch.full(shape, float(i))} for i, name in enumerate(pc.WINDOWS)}
        global_before = torch.random.get_rng_state().clone()
        inputs, rows = pc.prepare_inputs(windows)
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))
        for kind, order, seed in (("train", pc.WINDOWS * 2, 20260913), ("diagnostic", pc.WINDOWS, 20260914)):
            rng = torch.Generator(device="cpu").manual_seed(seed)
            for index, window in enumerate(order):
                prefix = f"{kind}.{index:02d}"
                expected_k = torch.randint(50, 951, (1,), generator=rng, dtype=torch.int64)
                expected_noise = torch.randn(shape, generator=rng, dtype=torch.float32)
                self.assertTrue(torch.equal(inputs[prefix + ".k"], expected_k))
                self.assertTrue(torch.equal(inputs[prefix + ".noise"], expected_noise))
        changed = {name: {"target": value["target"] + 123} for name, value in windows.items()}
        again, again_rows = pc.prepare_inputs(changed)
        self.assertEqual(rows, again_rows)
        self.assertTrue(all(torch.equal(value, again[key]) for key, value in inputs.items()))

    def test_example_uses_independent_observation_and_exact_future_targets(self):
        target = torch.arange(1 * 16 * 5 * 4 * 6, dtype=torch.float32).reshape(1, 16, 5, 4, 6) / 200
        noise = torch.flip(target, [2])
        observation = torch.full((1, 16, 1, 4, 6), -.75)
        actions = torch.arange(96, dtype=torch.float32).reshape(1, 16, 6)
        window = dict(target=target, observation=observation, actions=actions)
        inputs = {"x.noise": noise, "x.k": torch.tensor([731], dtype=torch.int64)}
        noisy, velocity, times, observed, commands = pc.example(window, inputs, "x")
        self.assertTrue(torch.equal(noisy[:, :, :1], observation))
        sigma = torch.tensor(731, dtype=torch.float32) / 1000
        self.assertTrue(torch.equal(noisy[:, :, 1:], (1-sigma)*target[:, :, 1:] + sigma*noise[:, :, 1:]))
        self.assertTrue(torch.equal(velocity[:, :, 1:], noise[:, :, 1:]-target[:, :, 1:]))
        self.assertEqual(int(velocity[:, :, :1].count_nonzero()), 0)
        self.assertEqual(times.tolist(), [[0]*6 + [731]*24])
        altered = dict(window, target=target.clone())
        altered["target"][:, :, :1] = 10000
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(pc.example(altered, inputs, "x"), (noisy, velocity, times, observed, commands))))

    def test_atomic_bundle_rejects_nonfinite_optimizer_and_owns_tensor_copies(self):
        adapter = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
        adapter(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as temp:
            digest = pc.save_recovery(temp, adapter, optimizer, 1)
            path = Path(temp) / "recovery-last.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["completed_updates"], 1)
            self.assertEqual(saved["schedule_completed"], list(pc.SCHEDULE[:1]))
            self.assertTrue(all(value.device.type == "cpu" for value in saved["adapter"].values()))
            with torch.no_grad():
                adapter.weight.add_(3)
            optimizer.state[adapter.weight]["exp_avg"].fill_(float("nan"))
            with self.assertRaises(ValueError):
                pc.save_recovery(temp, adapter, optimizer, 2)
            self.assertEqual(pc.sha(path), digest)
            retained = torch.load(path, map_location="cpu", weights_only=True)
            self.assertTrue(torch.equal(retained["adapter"]["weight"], saved["adapter"]["weight"]))

    def test_every_native_time_reaches_both_cfg_calls_with_prefix_zero(self):
        observed = torch.full((16, 1, 36, 64), -.125)
        latent = torch.cat((observed, torch.full((16, 4, 36, 64), .25)), dim=1)
        saved = latent.clone()
        actions = torch.arange(96, dtype=torch.float32).reshape(1, 16, 6)
        adapter = type("Adapter", (), {"_hooks": []})()
        wrapped = pc.TokenwiseConditionedModel(None, adapter, observed, actions)
        seen = []
        def predict(core, attached, value, times, image, commands, contexts):
            seen.append((times.clone(), contexts[0].item()))
            self.assertTrue(torch.equal(value[0], latent))
            self.assertTrue(torch.equal(image[0], observed))
            self.assertTrue(torch.equal(commands, actions))
            return torch.zeros_like(value)
        schedule = make_scheduler("cpu")
        with mock.patch.object(pc, "predict", side_effect=predict):
            for time_value in schedule.timesteps:
                for context in (torch.tensor([[-2.]]), torch.tensor([[3.]])):
                    wrapped([latent], time_value.reshape(1), [context], 2880)
        self.assertEqual(wrapped.calls, 100)
        self.assertEqual(wrapped.prefix_differences, [0.]*100)
        for index, time_value in enumerate(schedule.timesteps):
            for times, context in seen[2*index:2*index+2]:
                self.assertEqual(times.dtype, torch.int64)
                self.assertTrue((times[:, :576] == 0).all())
                self.assertTrue((times[:, 576:] == time_value).all())
            self.assertEqual([item[1] for item in seen[2*index:2*index+2]], [-2., 3.])
        self.assertTrue(torch.equal(latent, saved))

    def test_sampling_reader_never_materializes_any_future_tensor(self):
        accessed = []
        class Reader:
            def __init__(self, *args, **kwargs): self.inner = safe_open(*args, **kwargs)
            def __enter__(self): self.value = self.inner.__enter__(); return self
            def __exit__(self, *args): return self.inner.__exit__(*args)
            def get_tensor(self, key):
                accessed.append(key)
                if key not in ("observation", "actions"):
                    raise AssertionError("Future data is not an inference input")
                return self.value.get_tensor(key)
        with mock.patch("native_control.clamp_cache.safe_open", Reader), mock.patch.object(pc, "safe_open", Reader):
            observation, actions, metadata = pc.read_conditions(pc.PARENT / "data_cache")
        self.assertEqual(accessed, ["observation", "actions"])
        self.assertEqual(tuple(observation.shape), (16, 1, 36, 64))
        self.assertEqual(tuple(actions.shape), (1, 16, 6))
        self.assertFalse(metadata["target_materialized"])

    def test_incomplete_or_nonfinite_training_evidence_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            names = ("initial-adapter.safetensors", "final-adapter.safetensors", "fixed-inputs.safetensors")
            for name in names:
                save_file({"synthetic_fixture": torch.ones(1)}, str(directory/name))
            diagnostic = [{"window": window, **{key: "a"*64 for key in
                ("noisy_latent_sha256", "target_velocity_sha256", "token_times_sha256")}} for window in pc.WINDOWS]
            report = {"status": "passed", "requested_updates": 16, "completed_updates": 16,
                "schedule": list(pc.SCHEDULE), "adapter_seed": pc.SEED, "base_parameters_unchanged": True,
                "base_tensor_sha256_before": "b"*64, "base_tensor_sha256_after": "b"*64,
                "actual_zero_init_equal": True, "hooks_removed": True,
                "updates": [{"update": i+1, "window": window, "gradient_tensors": 42,
                    "residual_gradient_norms": [.1, .2, .3], "gradient_norm_before_clip": .5,
                    "future_flow_mse": .1} for i, window in enumerate(pc.SCHEDULE)],
                "diagnostics": {"before": diagnostic, "after": copy.deepcopy(diagnostic)},
                "source_sha256": {name: pc.sha(pc.PARENT/name) for name in pc.SHARED_NAMES +
                    ["tokenwise_time/pilot_common.py", "tokenwise_time/train_pilot.py"]},
                "output_sha256": {name: pc.sha(directory/name) for name in names}}
            file = directory / "metrics.json"
            file.write_text(json.dumps(report))
            pc.validate_training_run(directory)
            bad_cases = []
            partial = copy.deepcopy(report); partial["completed_updates"] = 15; bad_cases.append(partial)
            missing = copy.deepcopy(report); missing["updates"].pop(); bad_cases.append(missing)
            changed = copy.deepcopy(report); changed["base_tensor_sha256_after"] = "c"*64; bad_cases.append(changed)
            for field, value in (("future_flow_mse", float("nan")), ("future_flow_mse", -1.),
                                 ("gradient_norm_before_clip", float("inf")), ("gradient_norm_before_clip", 0.)):
                bad = copy.deepcopy(report); bad["updates"][7][field] = value; bad_cases.append(bad)
            for bad in bad_cases:
                file.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): pc.validate_training_run(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Output must be new")
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'.*")
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    names = pc.PILOT_NAMES + pc.SHARED_NAMES + ["tokenwise_time/test_pilot_independent.py"]
    report = {"status": "passed" if result.wasSuccessful() else "failed", "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "seconds": time.perf_counter()-started,
        "device": "cpu", "foundation_forward_calls": 0, "external_weights_loaded": False,
        "torch": str(torch.__version__), "source_sha256": {name: pc.sha(pc.PARENT/name) for name in names}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == "__main__":
    main()
