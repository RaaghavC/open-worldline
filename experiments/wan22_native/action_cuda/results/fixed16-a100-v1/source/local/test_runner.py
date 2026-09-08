"""Small preparation and mocked process tests. No actual cache, weights or GPU."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

import runner as r
import test_prototype as tiny_fixture


class RunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix='fixed16-runner-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def fixture(self, stack):
        case = tiny_fixture.Fixed16Tests().inputs()
        bridge, windows, schedule, draws, _, _, initial, *_ = case
        context = torch.randn(25, 4096)
        cache_id, text_id, probe_id = {'fixture': 'cache'}, {'fixture': 'genuine-text'}, {'fixture': 'complete-probe'}
        prior = {'schedule': schedule[:2], 'input_identity': {'cache': cache_id, 'text': text_id,
                 'prepared_artifacts': {'initial-adapter.safetensors': {'tensors': {
                     name: {'sha256': r.prototype.tensor_sha(value)} for name, value in initial.items()}}}}}
        shape = tuple(windows['closed-0000']['target'].shape)
        stack.enter_context(patch.dict(r.PROFILES, {'baseline': {'shape': shape[1:]}}))
        stack.enter_context(patch.object(r, 'verify_probe', return_value=(prior, probe_id)))
        stack.enter_context(patch.object(r.evidence, 'load_cache', return_value=(windows, cache_id)))
        stack.enter_context(patch.object(r.evidence, 'load_positive', return_value=(context, text_id)))
        stack.enter_context(patch.object(r.prototype, 'fresh_inputs', return_value=(bridge.adapter, initial, schedule, draws)))
        stack.enter_context(patch.object(r.probe, '_initial_specs', return_value={k: (tuple(v.shape), 'F32') for k, v in initial.items()}))
        stack.enter_context(patch.object(r, 'sources', return_value={'local': {'runner.py': r.sha(r.HERE/'runner.py')}}))
        args = dict(completed_probe=self.root/'prior', cache_run=self.root/'cache', text_directory=self.root/'text')
        return args, initial, draws, prior

    def test_fresh_preparation_bytes_identity_and_mutation_rejection(self):
        with ExitStack() as stack:
            args, initial, draws, _ = self.fixture(stack)
            out = self.root/'prepared'
            with patch.object(r, 'load_model', side_effect=AssertionError('No foundation in prepare')):
                plan = r.prepare(**args, profile='baseline', output=out)
                reread = r.read_prepared(out, **args)
            self.assertFalse(plan['warm_start'])
            self.assertEqual(len(plan['schedule']), 16)
            self.assertTrue(all(torch.equal(v, reread[2][k]) for k, v in initial.items()))
            self.assertTrue(all(torch.equal(v, reread[3][k]) for k, v in draws.items()))
            self.assertTrue(torch.equal(torch.get_rng_state(), reread[5]))
            self.assertEqual(set(plan['input_identity']['artifacts']), {
                'initial-adapter.safetensors', 'fresh-draws.safetensors', 'positive.safetensors', 'initial-cpu-rng.safetensors'})
            decision = {'schema': 'worldline-action-cuda-fixed16-admission-v1', 'decision': 'admit',
                        'issued_by': 'parent-agent', 'scope': r.SCOPE, 'plan_sha256': r.sha(out/'plan.json'),
                        'source_sha256': plan['source_sha256'], 'input_identity': plan['input_identity'],
                        'profile': plan['profile'], 'limits': r.LIMITS, 'expected_gpu': plan['expected_gpu'],
                        'image_generation_admitted': False, 'resume_admitted': False, 'reason': 'CPU fixture only'}
            path = self.root/'decision.json'; r.atomic(path, decision)
            self.assertEqual(r.admission(path, out, plan)['sha256'], r.sha(path))
            decision['plan_sha256'] = '0'*64; r.atomic(path, decision)
            with self.assertRaisesRegex(ValueError, 'Exact-plan'): r.admission(path, out, plan)
            with (out/'fresh-draws.safetensors').open('ab') as handle: handle.write(b'changed')
            with self.assertRaises(ValueError): r.read_prepared(out, **args)

    def test_prepare_rejects_prior_initialization_mismatch(self):
        with ExitStack() as stack:
            args, _, _, prior = self.fixture(stack)
            next(iter(prior['input_identity']['prepared_artifacts']['initial-adapter.safetensors']['tensors'].values()))['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'Fresh initialized tensor bytes'):
                r.prepare(**args, profile='baseline', output=self.root/'rejected')
            self.assertEqual(json.loads((self.root/'rejected/metrics.json').read_text())['status'], 'failed')
            self.assertFalse((self.root/'rejected/initial-adapter.safetensors').exists())

    def test_parent_interrupt_stops_child_retains_failure_and_refuses_reuse(self):
        out = self.root/'prepared'; out.mkdir(); r.atomic(out/'plan.json', {'fixture': True})
        decision = self.root/'decision.json'; r.atomic(decision, {'fixture': True})
        args = dict(prepared=out, completed_probe=self.root/'prior', cache_run=self.root/'cache',
                    text_directory=self.root/'text', weights=self.root/'absent-weights', decision=decision)
        proc = Mock(); proc.returncode = -15
        plan = {'source_sha256': {'fixture': True}}
        def interrupted(child, root, deadline, mode):
            self.assertIs(child, proc); self.assertEqual(mode, 'pair')
            self.assertEqual(deadline, 1000.)
            (root/'worker').mkdir(); (root/'worker/metrics.json').write_text('{partial')
            raise KeyboardInterrupt('fixture interrupt after handoff')
        with patch.object(r, 'read_prepared', return_value=(plan,)), \
             patch.object(r, 'admission', return_value={'sha256': r.sha(decision), 'scope': r.SCOPE}), \
             patch.object(r.time, 'monotonic', return_value=100.), \
             patch.object(r.subprocess, 'Popen', return_value=proc) as popen, \
             patch.object(r, 'supervise', side_effect=interrupted), patch.object(r, 'stop_child') as stop:
            with self.assertRaises(KeyboardInterrupt): r.execute(**args)
            stop.assert_called_once_with(proc)
            with self.assertRaisesRegex(ValueError, 'single-use'): r.execute(**args)
            popen.assert_called_once()
        report = json.loads((out/'metrics.json').read_text())
        self.assertEqual(report['status'], 'failed'); self.assertIsNone(report['model_execution'])
        self.assertIn('partial_metrics_read_error', report)
        self.assertEqual(json.loads((out/'terminal.json').read_text())['exit_code'], -15)
        self.assertTrue((out/'execution-attempt.json').is_file())

    def test_failed_probe_prevents_launch_and_marks_attempt(self):
        out = self.root/'prepared'; out.mkdir()
        args = dict(prepared=out, completed_probe=self.root/'failed-probe', cache_run=self.root/'cache',
                    text_directory=self.root/'text', weights=self.root/'absent-weights', decision=self.root/'decision')
        with patch.object(r, 'read_prepared', side_effect=ValueError('actual probe failed')), \
             patch.object(r.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'actual probe failed'): r.execute(**args)
            popen.assert_not_called()
        report = json.loads((out/'metrics.json').read_text())
        self.assertEqual(report['status'], 'failed'); self.assertIs(report['model_execution'], False)
        self.assertTrue((out/'execution-attempt.json').is_file())

    def test_parameter_hash_uses_explicit_detached_cpu_copy(self):
        copied = torch.tensor([.3, .7], requires_grad=False)
        value = Mock(); detached = Mock(); value.detach.return_value = detached
        detached.to.return_value = copied
        self.assertEqual(r.prototype.initialization_hash(value), r.prototype.tensor_sha(copied))
        value.detach.assert_called_once_with()
        detached.to.assert_called_once_with(device='cpu', copy=True)


if __name__ == '__main__':
    unittest.main()
