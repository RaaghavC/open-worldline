"""Independent CPU protocol tests; no pretrained model, renderer or GPU.

Run with the isolated Wan environment: python test_training_protocol.py.
Synthetic fixtures test access boundaries and equations, not visual quality.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open as actual_safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import training_common as common


class TrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(3207)

    def test_flow_equation_and_fixed_observation(self):
        clean = torch.randn(2, 16, 5, 3, 4)
        noise = torch.randn_like(clean)
        observation = torch.randn(2, 16, 1, 3, 4)
        for sigma in (0., .2, .5, 1.):
            mixed, velocity = common.flow_training_pair(clean, noise, observation, sigma)
            self.assertTrue(torch.equal(mixed[:, :, :1], observation))
            self.assertTrue(torch.allclose(mixed[:, :, 1:],
                                          ((1 - sigma) * clean + sigma * noise)[:, :, 1:]))
            self.assertTrue(torch.equal(velocity, noise - clean))

    def test_loss_excludes_observed_latent_even_if_its_prediction_is_invalid(self):
        target = torch.randn(1, 16, 5, 3, 4)
        predicted = target.clone()
        predicted[:, :, 0] = float("nan")
        self.assertEqual(common.future_mse(predicted, target).item(), 0.)
        predicted[:, :, 1:] += 2
        self.assertAlmostEqual(common.future_mse(predicted, target).item(), 4.)

    def test_shift_five_schedule_has_exact_endpoints_and_descends(self):
        schedule = common.shifted_schedule(20, 5.)
        self.assertEqual(len(schedule), 21)
        self.assertEqual(schedule[0], 1.)
        self.assertEqual(schedule[-1], 0.)
        self.assertAlmostEqual(schedule[10], 5 / 6)
        self.assertTrue(all(a > b for a, b in zip(schedule, schedule[1:])))
        for steps, shift in ((0, 5.), (20, 0.), (20, float("nan")), (20, float("inf"))):
            with self.assertRaises(ValueError):
                common.shifted_schedule(steps, shift)

    def test_constant_velocity_oracle_returns_clean_future_with_observation_clamped(self):
        clean = torch.randn(1, 16, 5, 3, 4)
        noise = torch.randn_like(clean)
        observation = torch.randn(1, 16, 1, 3, 4)
        current = torch.cat((observation, noise[:, :, 1:]), dim=2)
        schedule = common.shifted_schedule(20, 5.)
        for sigma, following in zip(schedule, schedule[1:]):
            current = common.euler_update(current, noise - clean, sigma, following, observation)
            self.assertTrue(torch.equal(current[:, :, :1], observation))
        self.assertTrue(torch.allclose(current[:, :, 1:], clean[:, :, 1:], atol=2e-6, rtol=1e-6))

    def test_inference_cache_never_materializes_target_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = {
                "observation": torch.randn(1, 16, 1, 36, 64),
                "actions": torch.randn(1, 16, 6),
                # A poison target makes accidental validation/loading visible.
                "target": torch.full((1, 16, 5, 36, 64), float("nan")),
            }
            path = root / "window.safetensors"
            save_file(values, str(path))
            entry = {
                "file": path.name,
                "sha256": common.sha(path),
                "tensors": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": common.tensor_sha(value)}
                    for key, value in values.items()
                },
            }
            manifest = {
                "schema": "worldline-wan-atrium-cache-v1", "status": "passed",
                "protocol": common.PROTOCOL,
                "protocol_sha256": common.sha(Path(common.__file__).with_name("PROTOCOL.md")),
                "vae_weights_sha256": common.FILES["Wan2.1_VAE.pth"][1],
                "causal_checks": [{"passed": True} for _ in range(11)],
                "windows": [{"id": window_id, **entry} for window_id in common.ORDER],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            reads = []

            class ReadBoundary:
                def __init__(self, *args, **kwargs):
                    self.real = actual_safe_open(*args, **kwargs)

                def __enter__(self):
                    self.real.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.real.__exit__(*args)

                def keys(self):
                    return self.real.keys()

                def get_tensor(self, key):
                    if key == "target":
                        raise AssertionError("Inference attempted to load clean future targets")
                    reads.append(key)
                    return self.real.get_tensor(key)

            with patch.object(common, "safe_open", ReadBoundary):
                result = common.CaptureCache(root).read(common.ORDER[0], include_target=False)
            self.assertEqual(reads, ["observation", "actions"])
            self.assertEqual(set(result), {"observation", "actions"})
            with self.assertRaisesRegex(ValueError, "Invalid target"):
                common.CaptureCache(root).read(common.ORDER[0], include_target=True)

    def test_predict_removes_only_observation_time_dimension_and_detaches_hooks(self):
        latent = torch.randn(2, 16, 5, 36, 64)
        observation = torch.randn(2, 16, 1, 36, 64)
        commands = torch.randn(2, 16, 6)
        context = [torch.randn(3, 4096), torch.randn(3, 4096)]

        class Adapter:
            def __init__(self):
                self.active = False

            def attach(self, core, actions, first, grid):
                self.active = True
                self.first = first
                self.actions = actions
                self.grid = grid

            def detach(self):
                self.active = False

        adapter = Adapter()

        def core(frames, time, text, count):
            self.assertTrue(adapter.active)
            self.assertEqual(tuple(adapter.first.shape), (2, 16, 36, 64))
            self.assertTrue(torch.equal(adapter.first, observation[:, :, 0]))
            self.assertIs(adapter.actions, commands)
            self.assertEqual(adapter.grid, (5, 18, 32))
            self.assertTrue(torch.equal(time, torch.full((2,), 500.)))
            self.assertIs(text, context)
            self.assertEqual(count, 2880)
            return torch.stack(frames)

        output = common.predict(core, adapter, latent, observation, commands, context, 500.)
        self.assertTrue(torch.equal(output, latent))
        self.assertFalse(adapter.active)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            common.predict(lambda *args: (_ for _ in ()).throw(RuntimeError("deliberate")),
                           adapter, latent, observation, commands, context, 500.)
        self.assertFalse(adapter.active)

    def test_sampler_cfg_uses_real_context_inputs_and_identical_other_conditions(self):
        import sample_clip
        observed = torch.randn(1, 16, 1, 3, 4)
        commands = torch.randn(1, 16, 6)
        noise = torch.ones(1, 16, 5, 3, 4)
        positive = torch.full((25, 4096), 3.)
        unconditional = torch.full((1, 4096), -2.)
        times = []

        def predict(core, adapter, latent, observation, actions, contexts, timestep):
            self.assertIs(contexts[0], unconditional)
            self.assertIs(contexts[1], positive)
            self.assertEqual(latent.dtype, torch.float16)
            self.assertTrue(torch.equal(latent[0], latent[1]))
            self.assertTrue(torch.equal(observation[0], observation[1]))
            self.assertTrue(torch.equal(actions[0], actions[1]))
            self.assertTrue(torch.equal(latent[:1, :, :1], observed.half()))
            times.append(timestep)
            return torch.cat((torch.full_like(latent[:1], 2.), torch.full_like(latent[:1], 5.)))

        with patch.object(sample_clip, "predict", predict):
            result = sample_clip.sample_latents(None, None, observed, commands, positive,
                                                unconditional, noise, steps=20, shift=5., cfg=5.)
        self.assertTrue(torch.equal(result[:, :, :1], observed))
        # CFG velocity = 2 + 5*(5-2) =17; integrating sigma1→0 gives1-17=-16.
        self.assertTrue(torch.allclose(result[:, :, 1:], torch.full_like(result[:, :, 1:], -16.), atol=3e-6))
        self.assertEqual(times, [1000 * s for s in common.shifted_schedule(20, 5.)[:-1]])
        self.assertTrue(torch.equal(noise, torch.ones_like(noise)))

    def test_sampler_reuses_supplied_noise_independent_of_global_rng(self):
        import sample_clip
        observed = torch.randn(1, 16, 1, 3, 4)
        commands = torch.randn(1, 16, 6)
        noise = torch.randn(1, 16, 5, 3, 4)
        original_noise = noise.clone()
        inputs = (None, None, observed, commands, torch.ones(2, 4096), torch.ones(1, 4096), noise)
        with patch.object(sample_clip, "predict", lambda core, adapter, latent, *args: latent * .125):
            first = sample_clip.sample_latents(*inputs, steps=4)
            torch.manual_seed(9)
            torch.randn(10000)
            second = sample_clip.sample_latents(*inputs, steps=4)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(noise, original_noise))

    def test_interrupted_trainer_does_not_report_passed(self):
        import train_clip
        self._check_interruption_report(train_clip, sampling=False)

    def test_interrupted_sampler_does_not_report_passed_and_requests_no_target(self):
        import sample_clip
        self._check_interruption_report(sample_clip, sampling=True)

    def _check_interruption_report(self, module, *, sampling):
        reads = []

        class Cache:
            manifest_sha256 = "c" * 64

            def __init__(self, directory):
                pass

            def read(self, key, *, include_target):
                reads.append(include_target)
                return {}

        context_info = {"identity": "synthetic-test-fixture-only"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "new-evidence"
            args = [module.__file__, "--weights", str(root / "unused-weights"),
                    "--capture-cache", str(root / "unused-capture"),
                    "--text-cache", str(root / "unused-text"),
                    "--output", str(output), "--device", "cpu", "--max-seconds", "10"]
            if sampling:
                training = root / "training"
                training.mkdir()
                checkpoint = training / "trained-adapter.safetensors"
                checkpoint.write_bytes(b"not loaded by this interruption test")
                (training / "metrics.json").write_text(json.dumps({
                    "status": "passed", "capture_cache_manifest_sha256": Cache.manifest_sha256,
                    "text": context_info, "adapter_checkpoint_sha256": common.sha(checkpoint),
                    "updates": [{"step": 0}],
                }))
                args.extend(("--training-run", str(training)))
            with patch.object(sys, "argv", args), patch.object(module, "CaptureCache", Cache), \
                    patch.object(module, "load_texts", return_value=(torch.ones(25, 4096),
                                                                      torch.ones(1, 4096), context_info)), \
                    patch.object(module, "load_core", side_effect=KeyboardInterrupt("test interruption")):
                with self.assertRaises(KeyboardInterrupt):
                    module.main()
            report = json.loads((output / "metrics.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error_type"], "KeyboardInterrupt")
            self.assertTrue(reads)
            self.assertTrue(all(flag is (not sampling) for flag in reads))
if __name__ == "__main__":
    unittest.main(verbosity=2)
