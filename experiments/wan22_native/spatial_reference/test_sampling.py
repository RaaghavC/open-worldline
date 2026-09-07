# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU sampling fixtures. No weights, CUDA, cloud or model calls."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import unittest
from unittest import mock

import torch

from ..cuda_reference import sampling as frozen
from ..cuda_reference.vendor.fm_solvers_unipc import FlowUniPCMultistepScheduler
from . import sampling


def fixture(shape):
    channels, frames, height, width = shape
    count = channels * frames * height * width
    noise = torch.linspace(-0.3, 0.5, count).reshape(shape)
    observation = torch.arange(channels * height * width, dtype=torch.float32)
    observation = (observation.remainder(97) / 97).reshape(1, channels, 1, height, width)
    initial = noise.clone()
    initial[:, :1] = observation[0]
    values = {"initial_noise": noise, "initial_latent": initial,
              "observation": observation, "token_times": sampling.times_at(999, shape)}
    contexts = {"atrium": torch.full((2, 3), 0.03),
                "native_negative": torch.full((7, 3), -0.02)}
    return values, contexts


def velocity(x, times, context):
    return x.square() * 0.001 + x * 0.02 + context[0, 0] + times[0, -1].float() * 0.00001


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_two_explicit_shapes_and_non_square_token_flattening(self):
        for shape, geometry in zip(sampling.ALLOWED_SHAPES, ((720, 144), (4290, 858))):
            self.assertEqual(sampling.validate_shape(shape), shape)
            self.assertEqual(sampling.token_geometry(shape), geometry)
            _, frames, height, width = shape
            times = sampling.times_at(621, shape).reshape(frames, height // 2, width // 2)
            self.assertEqual(tuple(times.shape), (5, height // 2, width // 2))
            self.assertTrue((times[0] == 0).all())
            self.assertTrue((times[1:] == 621).all())
            self.assertEqual(int(sampling.times_at(621, shape)[0, geometry[1] - 1]), 0)
            self.assertEqual(int(sampling.times_at(621, shape)[0, geometry[1]]), 621)
        for bad in ((48, 31, 18, 32), (48, 5, 32, 18), (48, 5, 44, 80),
                    (48, 5, 44.0, 78), (48, True, 18, 32), "48,5,18,32"):
            with self.subTest(shape=bad), self.assertRaises(ValueError):
                sampling.validate_shape(bad)

    def test_exact_frozen_schedule_and_integer_times(self):
        actual = sampling.scheduler()
        expected = frozen.scheduler()
        self.assertTrue(torch.equal(actual.timesteps, expected.timesteps))
        self.assertTrue(torch.equal(actual.sigmas, expected.sigmas))
        self.assertEqual(len(actual.timesteps), 50)
        self.assertEqual(int(actual.timesteps[0]), 999)
        self.assertEqual(float(actual.sigmas[-1]), 0.0)
        for t in actual.timesteps:
            self.assertTrue(torch.equal(sampling.times_at(t, sampling.ALLOWED_SHAPES[0]), frozen.times_at(t)))
        for bad in (999.0, True, -1, 1000, torch.tensor([1, 2]), torch.tensor(999, dtype=torch.int32)):
            with self.subTest(time=bad), self.assertRaises(ValueError):
                sampling.times_at(bad, sampling.ALLOWED_SHAPES[1])

    def test_both_shapes_have_100_conditioned_calls_and_50_recursive_updates(self):
        for shape in sampling.ALLOWED_SHAPES:
            with self.subTest(shape=shape):
                values, contexts = fixture(shape)
                before = {key: value.clone() for key, value in values.items()}
                context_before = {key: value.clone() for key, value in contexts.items()}
                rng_before = torch.random.get_rng_state().clone()
                calls = []
                previous = {"latent": values["initial_latent"].clone(), "positive": None}
                _, prefix = sampling.token_geometry(shape)
                steps = []

                def predict(x, times, context):
                    index = len(calls)
                    label = "atrium" if index % 2 == 0 else "native_negative"
                    self.assertTrue(torch.equal(context, contexts[label]))
                    self.assertTrue(torch.equal(x, previous["latent"]))
                    self.assertTrue(torch.equal(x[:, :1], values["observation"][0]))
                    self.assertTrue((times[:, :prefix] == 0).all())
                    self.assertTrue((times[:, prefix:] == times[0, prefix]).all())
                    if index % 2 == 0:
                        previous["positive"] = (x.clone(), times.clone())
                    else:
                        self.assertTrue(torch.equal(x, previous["positive"][0]))
                        self.assertTrue(torch.equal(times, previous["positive"][1]))
                    calls.append((label, int(times[0, -1])))
                    return velocity(x, times, context)

                def event(i, t, x, result):
                    self.assertEqual(i, len(steps))
                    self.assertTrue(torch.equal(x[:, :1], values["observation"][0]))
                    self.assertTrue(torch.equal(result["guided_velocity"], result["negative_velocity"]
                                                + 5.0 * (result["positive_velocity"] - result["negative_velocity"])))
                    steps.append(int(t))
                    previous["latent"] = x.clone()

                actual = sampling.sample(predict, values, contexts, latent_shape=shape, event=event)
                self.assertEqual(len(calls), 100)
                self.assertEqual(len(steps), 50)
                self.assertEqual(steps, sampling.scheduler().timesteps.tolist())
                for key, value in before.items():
                    self.assertTrue(torch.equal(value, values[key]))
                for key, value in context_before.items():
                    self.assertTrue(torch.equal(value, contexts[key]))
                self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

                # Independent loop using the unchanged vendor solver and explicit
                # nonlinear branch equations. No generalized sampler helper here.
                solver = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
                solver.set_timesteps(50, device="cpu", shift=5.0)
                expected = values["initial_latent"].clone()
                for t in solver.timesteps:
                    positive = expected.square() * 0.001 + expected * 0.02 + contexts["atrium"][0, 0] + t.float() * 0.00001
                    negative = expected.square() * 0.001 + expected * 0.02 + contexts["native_negative"][0, 0] + t.float() * 0.00001
                    guided = negative + 5.0 * (positive - negative)
                    expected = solver.step(guided.unsqueeze(0), t, expected.unsqueeze(0), return_dict=False)[0][0]
                    expected[:, :1] = values["observation"][0]
                self.assertTrue(torch.equal(actual, expected))
                self.assertFalse(torch.equal(actual[:, 1:], values["initial_latent"][:, 1:]))

    def test_baseline_matches_frozen_sampler_and_event_cannot_mutate_solver(self):
        shape = sampling.ALLOWED_SHAPES[0]
        values, contexts = fixture(shape)
        expected = frozen.sample(velocity, values, contexts)

        def mutate_event(i, t, x, outputs):
            x.fill_(123)
            t.fill_(0)
            for value in outputs.values():
                value.fill_(456)

        actual = sampling.sample(velocity, values, contexts, latent_shape=shape, event=mutate_event)
        self.assertTrue(torch.equal(actual, expected))

    def test_pair_copies_predictor_inputs_and_partial_outputs(self):
        shape = sampling.ALLOWED_SHAPES[1]
        values, contexts = fixture(shape)
        snapshots = {key: value.clone() for key, value in {**values, **contexts}.items()}
        seen = []

        def predict(x, times, context):
            label = "atrium" if not seen else "native_negative"
            self.assertTrue(torch.equal(x, values["initial_latent"]))
            self.assertTrue(torch.equal(times, values["token_times"]))
            self.assertTrue(torch.equal(context, contexts[label]))
            result = torch.full_like(x, float(context[0, 0]))
            x.zero_(); times.zero_(); context.zero_()
            seen.append(label)
            return result

        partials = []
        def event(label, outputs):
            partials.append((label, tuple(outputs)))
            for value in outputs.values():
                value.zero_()

        result = sampling.pair(predict, values["initial_latent"], values["token_times"],
                               contexts["atrium"], contexts["native_negative"], values["observation"],
                               latent_shape=shape, event=event)
        self.assertEqual(seen, ["atrium", "native_negative"])
        self.assertEqual(partials[0], ("positive", ("positive_velocity",)))
        expected = torch.full(shape, -0.02) + 5.0 * (torch.full(shape, 0.03) - torch.full(shape, -0.02))
        self.assertTrue(torch.equal(result["guided_velocity"], expected))
        for key, value in {**values, **contexts}.items():
            self.assertTrue(torch.equal(value, snapshots[key]))

    def test_malformed_saved_inputs_rejected_before_predicting(self):
        shape = sampling.ALLOWED_SHAPES[0]
        values, contexts = fixture(shape)
        bad_inputs = []
        bad_inputs.append(dict(values, target=torch.zeros(1)))
        bad_inputs.append({k: v for k, v in values.items() if k != "initial_noise"})
        bad_inputs.append(dict(values, initial_noise=values["initial_noise"].double()))
        bad_inputs.append(dict(values, observation=values["observation"][0]))
        changed = values["initial_latent"].clone(); changed[0, 1, 0, 0] += 1
        bad_inputs.append(dict(values, initial_latent=changed))
        bad_inputs.append(dict(values, token_times=sampling.times_at(500, shape)))
        for bad in bad_inputs:
            predict = mock.Mock(side_effect=AssertionError("Prediction before valid inputs"))
            with self.assertRaises(ValueError):
                sampling.sample(predict, bad, contexts, latent_shape=shape)
            predict.assert_not_called()
        for bad in (dict(contexts, future_target=torch.zeros(1)),
                    dict(contexts, atrium=torch.zeros(0, 3)),
                    dict(contexts, native_negative=torch.zeros(2, 4))):
            predict = mock.Mock()
            with self.assertRaises(ValueError):
                sampling.sample(predict, values, bad, latent_shape=shape)
            predict.assert_not_called()

    def test_invalid_token_masks_and_embedding_types_rejected(self):
        shape = sampling.ALLOWED_SHAPES[1]
        values, contexts = fixture(shape)
        bad_times = [values["token_times"].int(), values["token_times"][:, :-1]]
        for index in (0, 859):
            changed = values["token_times"].clone(); changed[0, index] = 12
            bad_times.append(changed)
        for bad in bad_times:
            predict = mock.Mock()
            with self.assertRaises(ValueError):
                sampling.pair(predict, values["initial_latent"], bad, contexts["atrium"],
                              contexts["native_negative"], values["observation"], latent_shape=shape)
            predict.assert_not_called()
        with self.assertRaises(ValueError):
            sampling.pair(mock.Mock(), values["initial_latent"], values["token_times"],
                          contexts["atrium"].half(), contexts["native_negative"], values["observation"], latent_shape=shape)

    def test_nonfinite_saved_values_and_contexts_rejected(self):
        shape = sampling.ALLOWED_SHAPES[0]
        values, contexts = fixture(shape)
        for key in ("initial_noise", "initial_latent", "observation"):
            changed = values[key].clone(); changed.flatten()[0] = float("nan")
            predict = mock.Mock()
            with self.assertRaises(FloatingPointError):
                sampling.sample(predict, dict(values, **{key: changed}), contexts, latent_shape=shape)
            predict.assert_not_called()
        changed = contexts["native_negative"].clone(); changed[0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            sampling.sample(mock.Mock(), values, dict(contexts, native_negative=changed), latent_shape=shape)

    def test_bad_predictions_and_guidance_overflow_rejected(self):
        shape = sampling.ALLOWED_SHAPES[0]
        values, contexts = fixture(shape)
        for result in (torch.zeros(shape, dtype=torch.float64), torch.zeros(1),
                       torch.empty(shape, device="meta"), torch.full(shape, float("nan"))):
            with self.assertRaises((ValueError, FloatingPointError)):
                sampling.pair(lambda *args: result, values["initial_latent"], values["token_times"],
                              contexts["atrium"], contexts["native_negative"], values["observation"], latent_shape=shape)

        def overflow(x, times, context):
            return torch.full_like(x, 2e38 if context[0, 0] > 0 else -2e38)

        with self.assertRaises(FloatingPointError):
            sampling.pair(overflow, values["initial_latent"], values["token_times"], contexts["atrium"],
                          contexts["native_negative"], values["observation"], latent_shape=shape)

    def test_partial_positive_survives_negative_interrupt(self):
        shape = sampling.ALLOWED_SHAPES[0]
        values, contexts = fixture(shape)
        kept = []

        def predict(x, times, context):
            if context[0, 0] < 0:
                raise KeyboardInterrupt("Injected second-branch interruption")
            return torch.ones_like(x)

        with self.assertRaises(KeyboardInterrupt):
            sampling.pair(predict, values["initial_latent"], values["token_times"], contexts["atrium"],
                          contexts["native_negative"], values["observation"], latent_shape=shape,
                          event=lambda label, outputs: kept.append((label, outputs)))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][0], "positive")
        self.assertTrue((kept[0][1]["positive_velocity"] == 1).all())

    def test_nonfinite_solver_state_rejected_before_event(self):
        shape = sampling.ALLOWED_SHAPES[0]
        values, contexts = fixture(shape)
        solver = sampling.scheduler()
        event = mock.Mock()
        with mock.patch.object(solver, "step", return_value=(torch.full((1, *shape), float("inf")),)), \
                mock.patch.object(sampling, "scheduler", return_value=solver):
            with self.assertRaises(FloatingPointError):
                sampling.sample(velocity, values, contexts, latent_shape=shape, event=event)
        event.assert_not_called()

    def test_cpu_autocast_is_disabled_for_pair_math(self):
        shape = sampling.ALLOWED_SHAPES[0]
        values, contexts = fixture(shape)

        def predict(x, times, context):
            scalar = context[:1] @ torch.ones(3, 1)
            self.assertEqual(scalar.dtype, torch.float32)
            return x * 0.01 + scalar[0, 0]

        expected = sampling.pair(predict, values["initial_latent"], values["token_times"], contexts["atrium"],
                                 contexts["native_negative"], values["observation"], latent_shape=shape)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = sampling.pair(predict, values["initial_latent"], values["token_times"], contexts["atrium"],
                                   contexts["native_negative"], values["observation"], latent_shape=shape)
        self.assertTrue(torch.equal(actual["guided_velocity"], expected["guided_velocity"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError("A fresh output directory is required")
    files = {"sampling.py": Path(sampling.__file__), "test_sampling.py": Path(__file__),
             "frozen_sampling.py": Path(frozen.__file__),
             "vendor_solver.py": Path(frozen.__file__).parent / "vendor/fm_solvers_unipc.py"}
    before = {name: sha(path) for name, path in files.items()}
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    after = {name: sha(path) for name, path in files.items()}
    report = {"status": "passed" if result.wasSuccessful() and before == after else "failed",
              "tests": result.testsRun, "elapsed_seconds": time.monotonic() - started,
              "source_sha256": after, "sources_unchanged": before == after,
              "scope": "CPU injected-predictor protocol checks only",
              "model_calls": 0, "weight_values_read": False, "CUDA_used": False, "cloud_calls": 0}
    if args.output:
        args.output.mkdir(parents=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        checked = args.output / "checked-source"
        checked.mkdir()
        for name, path in files.items():
            shutil.copyfile(path, checked / name)
    print(json.dumps(report))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
