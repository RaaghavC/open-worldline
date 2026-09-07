# SPDX-License-Identifier: Apache-2.0
"""Independent CPU sampler math and failed-handoff lifecycle checks."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import torch

from . import sample_clip as sample
from experiments.wan_adapter.native_control.vendor.fm_solvers_unipc import FlowUniPCMultistepScheduler


def tensors():
    generator = torch.Generator().manual_seed(20260916)
    return torch.randn(sample.SHAPE, generator=generator), torch.linspace(-.5,.5,48*18*32).reshape(1,48,1,18,32)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all50_native_steps_match_independent_timed_velocity_and_feedback(self):
        noise, observation = tensors()
        before_noise, before_observation = noise.clone(), observation.clone()
        schedule = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        schedule.set_timesteps(50, device='cpu', shift=5.)
        expected = noise.clone()
        references, before_step = [], []
        for timestep in schedule.timesteps:
            expected[:, :1] = observation[0]
            before_step.append(expected.clone())
            shared = expected*.03 + float(timestep)*1e-6
            positive, negative = shared+.04, shared-.02
            guided = negative+5*(positive-negative)
            expected = schedule.step(guided[None], timestep, expected[None], return_dict=False)[0][0]
            expected[:, :1] = observation[0]
            references.append(expected.clone())
        calls = []
        def denoiser(latents, times, contexts, seq_len):
            index, arm = divmod(len(calls),2)
            self.assertEqual(seq_len,720)
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(times.dtype,torch.int64)
            self.assertTrue(torch.equal(times[:,:144],torch.zeros(1,144,dtype=torch.int64)))
            self.assertTrue((times[:,144:]==schedule.timesteps[index]).all())
            self.assertTrue(torch.equal(latents[0],before_step[index]))
            self.assertEqual(float(contexts[0][0,0]),2. if arm==0 else -1.)
            calls.append((index,arm))
            return [latents[0]*.03+float(schedule.timesteps[index])*1e-6+float(contexts[0][0,0])*.02]
        seen = []
        def after(row, value):
            index = len(seen)
            self.assertTrue(torch.equal(value,references[index]))
            self.assertEqual(row['step'],index+1)
            self.assertTrue(row['post_step_prefix_exact'])
            seen.append(row)
        actual, rows = sample.integrate(denoiser,noise,observation,torch.tensor([[2.]]),torch.tensor([[-1.]]),on_step=after)
        self.assertEqual(len(calls),100)
        self.assertEqual(len(rows),len(seen))
        self.assertEqual(len(rows),50)
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(noise,before_noise))
        self.assertTrue(torch.equal(observation,before_observation))

    def test_failed_stdin_or_interrupt_terminates_child_and_retains_terminal(self):
        for failure in (BrokenPipeError('stdin failure'),KeyboardInterrupt('parent interruption')):
            class Process:
                returncode = None
                pid = 7654321
                terminated = False
                def poll(self): return self.returncode
                def communicate(self,*args): raise failure
                def terminate(self): self.terminated=True; self.returncode=-15
                def wait(self,**kwargs): return self.returncode
            child = Process()
            guard = sample.CombinedGuard.__new__(sample.CombinedGuard)
            guard.process = None
            guard.deadline = time.monotonic()+100
            config = {'runtime_environment':{'environment':{name:None for name in sample.RUNTIME_VARIABLES}}}
            with tempfile.TemporaryDirectory() as directory:
                wrapper = Path(directory)/'core'
                with mock.patch.object(sample.subprocess,'Popen',return_value=child):
                    with self.assertRaises(type(failure)):
                        guard.launch(config,wrapper)
                terminal = json.loads((wrapper/'terminal.json').read_text())
                self.assertTrue(child.terminated)
                self.assertEqual(terminal['status'],'failed')
                self.assertEqual(terminal['exit_code'],-15)
                self.assertEqual(terminal['error_type'],type(failure).__name__)
                self.assertFalse((wrapper/'result').exists())

    def test_expired_deadline_stops_before_process_or_output_stage(self):
        guard = sample.CombinedGuard.__new__(sample.CombinedGuard)
        guard.deadline = time.monotonic()-1
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory)/'decode'
            with mock.patch.object(sample.subprocess,'Popen',side_effect=AssertionError('No launch after deadline')):
                with self.assertRaisesRegex(RuntimeError,'deadline'):
                    guard.launch({},wrapper)
            self.assertFalse(wrapper.exists())

    def test_full_budget_accounts_for_loading_both_runtime_stages_and_preflight(self):
        pair = {'elapsed_seconds':11.,'pair_seconds':4.,'timings':[
            {'stage':'verify and stream official transformer','seconds':6.},
            {'stage':'positive native forward','seconds':2.},
            {'stage':'negative native forward','seconds':2.}]}
        decode = {'elapsed_seconds':30.,'timings':[{'stage':'decode','seconds':29.}]}
        gate = sample.timing_gate(pair,decode,.25,2.)
        self.assertEqual(gate['total_estimate_seconds'],6+50*4+50*1+.25+30+45+2)
        pair['elapsed_seconds']=100.
        with self.assertRaisesRegex(ValueError,'exceeds'):
            sample.timing_gate(pair,decode,.25,2.)

    def test_current_implementation_report_and_unexpected_input_keys(self):
        self.assertEqual(len(sample.validate_cpu(sample.HERE/'sample-results/cpu-v1/tests.json')),64)
        reads = []
        class Mixed:
            def __init__(self,*args,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def keys(self): return sample.RUN_KEYS|{'actions','target'}
            def get_tensor(self,name): reads.append(name); raise AssertionError('Rejected inputs must not materialize')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mixed';path.write_bytes(b'header')
            with mock.patch.object(sample,'safe_open',Mixed):
                with self.assertRaises(ValueError):
                    sample.read_tensors(path,sample.RUN_KEYS)
        self.assertEqual(reads,[])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():parser.error('New evidence file required')
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report = {'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,
        'errors':len(result.errors),'failures':len(result.failures),'seconds':time.perf_counter()-started,
        'source_sha256':{name:sample.sha256(sample.HERE/name) for name in sample.SOURCES},
        'reused_source_sha256':{name:sample.sha256(sample.REPO/name) for name in sample.REUSED_NAMES},
        'independent_test_sha256':sample.sha256(__file__),'device':'cpu','actual_gpu_calls':False,
        'actual_pretrained_inference':False,'official_weight_values_loaded':False,
        'scope':'Full native CPU solver with synthetic time/state/context-dependent velocity and mocked failed child handoff; source/evidence and exact input boundaries. No generated-quality or runtime guarantee.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__ == '__main__':main()
