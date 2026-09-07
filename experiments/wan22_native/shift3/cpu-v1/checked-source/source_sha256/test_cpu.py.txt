# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU-only shift3 scheduler, source isolation and lifecycle tests."""
import argparse
import ast
import inspect
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file
from . import sample as s


def inputs():
    noise = torch.randn(s.SHAPE, generator=torch.Generator().manual_seed(49))
    observation = torch.full((1, 48, 1, 18, 32), .25)
    latent = noise.clone(); latent[:, :1] = observation[0]
    times = s.token_times(torch.tensor([s.NATIVE_GRID]), s.native.make_scheduler().timesteps[:1], 720)
    return {'initial_noise': noise, 'initial_latent': latent, 'observation': observation,
            'token_times': times, 'positive': torch.ones(25, 4096), 'negative': torch.zeros(126, 4096)}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous)

    def test_isolated_loops_are_literal_native_equations_and_globals_unchanged(self):
        for name in ('integrate', 'solver_benchmark'):
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(s, name)))),
                             ast.dump(ast.parse(inspect.getsource(getattr(s.native, name)))))
        native_factory = s.native.make_scheduler
        old = native_factory(); new = s.make_scheduler()
        self.assertIs(s.native.make_scheduler, native_factory)
        self.assertEqual(s.native.SETTINGS, {'steps': 50, 'shift': 5., 'guidance': 5.})
        self.assertEqual(s.SETTINGS, {'steps': 50, 'shift': 3., 'guidance': 5.})
        self.assertEqual((len(old.timesteps), len(new.timesteps)), (50, 50))
        self.assertEqual((int(old.timesteps[0]), int(new.timesteps[0])), (999, 999))
        self.assertFalse(torch.equal(old.timesteps[1:], new.timesteps[1:]))
        self.assertNotEqual(float(old.sigmas[0]), float(new.sigmas[0]))
        self.assertEqual(float(new.sigmas[-1]), 0.)

    def test_exact_retained_initial_inputs_and_changed_time_rejection(self):
        actual = s.native.HERE/'sample-results/clip50-v1/inputs.safetensors'
        values = s.native.read_tensors(actual, s.native.RUN_KEYS)
        hashes = {name: s.tensor_sha256(value) for name, value in values.items()}
        with mock.patch.object(torch, 'randn', side_effect=AssertionError('No RNG regeneration')):
            record = s.first_forward_identity(values)
        self.assertTrue(record['passed']); self.assertEqual(record['first_model_time_both'], 999)
        self.assertFalse(record['initial_noise_rescaled'])
        self.assertEqual(hashes, {name: s.tensor_sha256(value) for name, value in values.items()})
        values['token_times'] = values['token_times'].clone(); values['token_times'][0, -1] = 998
        with self.assertRaises(ValueError): s.first_forward_identity(values)

    def test_all50_solver_steps_match_independent_native_shift3_oracle(self):
        values = inputs(); noise = values['initial_noise']; obs = values['observation']
        before = noise.clone(); calls = []
        oracle = s.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        oracle.set_timesteps(50, device='cpu', shift=3.)
        def model(latents, times, contexts, seq_len):
            i = len(calls)//2; positive = len(calls)%2 == 0
            self.assertEqual(seq_len, 720)
            self.assertTrue(torch.equal(latents[0][:, :1], obs[0]))
            self.assertTrue((times[:, :144] == 0).all())
            self.assertTrue((times[:, 144:] == oracle.timesteps[i]).all())
            self.assertEqual(float(contexts[0][0, 0]), float(positive))
            calls.append(int(times[0, -1]))
            return [latents[0]*.05+int(times[0, -1])*.00001+float(positive)*.01]
        actual, rows = s.integrate(model, noise, obs, values['positive'], values['negative'])
        expected = noise.clone()
        for timestep in oracle.timesteps:
            expected[:, :1] = obs[0]
            pos = expected*.05+int(timestep)*.00001+.01
            neg = expected*.05+int(timestep)*.00001
            expected = oracle.step((neg+5*(pos-neg))[None], timestep, expected[None], return_dict=False)[0][0]
            expected[:, :1] = obs[0]
        self.assertTrue(torch.equal(actual, expected)); self.assertTrue(torch.equal(noise, before))
        self.assertEqual(len(calls), 100); self.assertEqual(len(rows), 50)
        self.assertEqual([row['model_timestep'] for row in rows], oracle.timesteps.tolist())
        self.assertTrue(all(row['post_step_prefix_exact'] and row['cfg_calls_prefix_exact'] == [True, True] for row in rows))

    def test_first_positive_and_negative_forward_arguments_match_shift5(self):
        values = inputs(); retained = []
        for integrate in (s.native.integrate, s.integrate):
            calls = []
            def model(latents, times, contexts, seq_len):
                if len(calls) < 2:
                    retained.append((latents[0].clone(), times.clone(), contexts[0].clone(), seq_len))
                calls.append(True)
                return [latents[0]*.01]
            integrate(model, values['initial_noise'], values['observation'], values['positive'], values['negative'])
        for old, new in zip(retained[:2], retained[2:]):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(old[:3], new[:3])))
            self.assertEqual(old[3], new[3])

    def test_solver_benchmark_is50_cpu_steps_and_preserves_input(self):
        values = inputs(); before = values['initial_noise'].clone()
        result = s.solver_benchmark(values['initial_noise'], values['observation'])
        self.assertEqual((result['status'], result['updates'], result['device']), ('passed', 50, 'cpu'))
        self.assertFalse(result['model_executed']); self.assertGreater(result['seconds'], 0.)
        self.assertTrue(torch.equal(before, values['initial_noise']))

    def test_nonfinite_velocity_or_bad_source_stops_before_completed_result(self):
        values = inputs()
        with self.assertRaises(FloatingPointError):
            s.integrate(lambda x, *a: [torch.full_like(x[0], float('nan'))], values['initial_noise'],
                        values['observation'], values['positive'], values['negative'])
        with tempfile.TemporaryDirectory() as tmp:
            report = {'status':'passed','tests':99,'source_sha256':{},'native_source_sha256':{},'reused_source_sha256':{}}
            path = Path(tmp)/'tests.json';path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):s.validate_cpu(path)

    def test_decode_dispatch_calls_unchanged_native_worker(self):
        config = {'ablation_settings': s.SETTINGS, 'ablation_source_sha256': {'fixture': 'source'}, 'stage': 'decode',
            'source_sha256': {name:s.native.sha256(s.native.HERE/name) for name in s.native.SOURCES},
            'reused_source_sha256': {name:s.native.sha256(s.native.REPO/name) for name in s.native.REUSED_NAMES}}
        with mock.patch.object(s, 'source_hashes', return_value={'fixture':'source'}), mock.patch.object(s.native, 'child') as original:
            s.child(config)
        original.assert_called_once_with(config)
        self.assertIs(s.CombinedGuard.__init__, s.native.CombinedGuard.__init__)
        self.assertIs(s.CombinedGuard.terminate, s.native.CombinedGuard.terminate)
        self.assertIs(s.CombinedGuard.close, s.native.CombinedGuard.close)

    def test_early_handoff_error_terminates_child_and_preserves_terminal(self):
        class Process:
            returncode = None
            def poll(self):return self.returncode
            def communicate(self, data):raise BrokenPipeError('injected early handoff')
            def terminate(self):self.returncode = -15
            def wait(self, timeout=None):return self.returncode
        process = Process(); guard = object.__new__(s.CombinedGuard)
        guard.deadline = time.monotonic()+900;guard.process = None
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp)/'core';config = {'runtime_environment': {'environment': {}}}
            with mock.patch.object(s.subprocess, 'Popen', return_value=process) as spawn:
                with self.assertRaises(BrokenPipeError):guard.launch(config, wrapper)
            self.assertEqual(process.returncode, -15)
            self.assertIn('experiments.wan22_native.shift3.sample', spawn.call_args.args[0])
            terminal = json.loads((wrapper/'terminal.json').read_text())
            self.assertEqual((terminal['status'], terminal['error_type']), ('failed', 'BrokenPipeError'))


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--output', type=Path, required=True);args=parser.parse_args()
    if args.output.exists():parser.error('Fresh CPU evidence required')
    before = s.source_hashes(); native_before = {name:s.native.sha256(s.native.HERE/name) for name in s.native.SOURCES}
    reused_before = {name:s.native.sha256(s.native.REPO/name) for name in s.native.REUSED_NAMES}
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    assert before==s.source_hashes()
    assert native_before=={name:s.native.sha256(s.native.HERE/name) for name in s.native.SOURCES}
    assert reused_before=={name:s.native.sha256(s.native.REPO/name) for name in s.native.REUSED_NAMES}
    report={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,
        'errors':len(result.errors),'failures':len(result.failures),'seconds':time.monotonic()-started,
        'source_sha256':before,'native_source_sha256':native_before,'reused_source_sha256':reused_before,
        'source_unchanged_after_tests':True,'device':'cpu','real_model_constructed':False,'official_weight_values_loaded':False,
        'gpu_used':False,'scope':'Exact copied-loop equations, independent50-step synthetic solver, retained actual initial tensors, process failures. No model quality or runtime inference.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
