# SPDX-License-Identifier: Apache-2.0
"""Independent CPU checks of the one-update mechanics and evidence gate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from tokenwise_time.portable import create_model
from tokenwise_time.training import (ActionObservationAdapter, training_pair, future_loss,
    conditioning, checkpoint_blocks, optimizer_update)
from tokenwise_time.training_data import read_training_window, tensor_sha
from tokenwise_time.profile_update import validate_preflight

HERE = Path(__file__).parent
PARENT = HERE.parent
CONFIG = dict(model_type='t2v', dim=256, ffn_dim=384, freq_dim=64, num_heads=2,
              num_layers=4, text_dim=32, text_len=8, in_dim=16, out_dim=16)


def fixture():
    torch.manual_seed(752)
    core = create_model(**CONFIG).eval().requires_grad_(False)
    adapter = ActionObservationAdapter(256, block_indices=(0, 1, 2), width=16)
    with torch.no_grad():
        core.head.head.weight.normal_(0, .02)
        for residual in adapter.residuals:
            residual.output.weight.normal_(0, .01)
    observation = torch.randn(1, 16, 1, 4, 6)
    noisy, velocity, times = training_pair(torch.randn(1, 16, 3, 4, 6),
        torch.randn(1, 16, 3, 4, 6), observation, torch.tensor([613]))
    actions, contexts = torch.arange(48).reshape(1, 8, 6).float() / 48, [torch.randn(3, 32)]
    return core, adapter, noisy, velocity, times, observation, actions, contexts


class IndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_noise_and_time_are_matched_per_example_and_observed_loss_is_excluded(self):
        target = torch.zeros(2, 16, 3, 4, 6)
        target[0], target[1] = 2, -3
        noise, observation = torch.ones_like(target) * 4, torch.ones(2, 16, 1, 4, 6) * 9
        times = torch.tensor([50, 950])
        x, v, t = training_pair(target, noise, observation, times)
        self.assertTrue(torch.equal(x[:, :, :1], observation))
        torch.testing.assert_close(x[0, :, 1:], torch.full_like(x[0, :, 1:], 2.1), atol=2e-7, rtol=0)
        torch.testing.assert_close(x[1, :, 1:], torch.full_like(x[1, :, 1:], 3.65), atol=3e-7, rtol=0)
        self.assertEqual(t[0].tolist(), [0] * 6 + [50] * 12)
        self.assertEqual(t[1].tolist(), [0] * 6 + [950] * 12)
        altered = target.clone()
        altered[:, :, :1] = -10000
        second = training_pair(altered, noise, observation, times)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip((x, v, t), second)))
        prediction = (v + 1).requires_grad_()
        loss = future_loss(prediction, v)
        loss.backward()
        self.assertEqual(loss.item(), 1)
        self.assertEqual(prediction.grad[:, :, :1].count_nonzero().item(), 0)
        self.assertTrue((prediction.grad[:, :, 1:] > 0).all())

    def test_checkpoint_replay_excludes_adapter_hooks_and_preserves_early_gradients(self):
        core, adapter, x, v, t, obs, actions, contexts = fixture()
        before = {name: p.clone() for name, p in core.state_dict().items()}
        handles, hook_counts = [], [0, 0, 0]
        for index, residual in enumerate(adapter.residuals):
            def count(_, args, output, index=index):
                hook_counts[index] += 1
            handles.append(residual.register_forward_hook(count))
        with conditioning(adapter, core, actions, obs, (3, 2, 3)), checkpoint_blocks(core) as calls:
            prediction = torch.stack(core(list(x.unbind(0)), t, contexts, 18))
            future_loss(prediction, v).backward()
        for handle in handles:
            handle.remove()
        self.assertEqual(hook_counts, [1, 1, 1])
        self.assertGreater(calls[-1], 1)
        self.assertFalse(adapter._hooks)
        gradients = {name: p.grad.clone() for name, p in adapter.named_parameters()}
        self.assertGreater(gradients['residuals.0.query.weight'].abs().sum().item(), 0)
        adapter.zero_grad(set_to_none=True)
        with conditioning(adapter, core, actions, obs, (3, 2, 3)):
            prediction = torch.stack(core(list(x.unbind(0)), t, contexts, 18))
            future_loss(prediction, v).backward()
        for name, p in adapter.named_parameters():
            torch.testing.assert_close(p.grad, gradients[name], atol=2e-7, rtol=2e-5, msg=name)
        self.assertTrue(all(torch.equal(p, before[name]) for name, p in core.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in core.parameters()))

    def test_disconnected_adapter_tensor_stops_before_optimizer_update(self):
        core, adapter, x, v, t, obs, actions, contexts = fixture()
        before = {name: p.clone() for name, p in adapter.state_dict().items()}
        # Detach one actual projection, while the other residuals still receive gradients.
        handle = adapter.residuals[0].query.register_forward_hook(lambda _, args, output: output.detach())
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.001)
        try:
            with self.assertRaises(RuntimeError):
                optimizer_update(core, adapter, optimizer, x, v, t, obs, actions, contexts)
        finally:
            handle.remove()
        self.assertFalse(adapter._hooks)
        self.assertTrue(all(torch.equal(p, before[name]) for name, p in adapter.state_dict().items()))
        self.assertFalse(optimizer.state)

    def test_actual_cache_observation_actions_and_target_are_separate_hashed_tensors(self):
        values, provenance = read_training_window(PARENT / 'data_cache', 'open-0000')
        self.assertEqual(set(values), {'target', 'observation', 'actions'})
        self.assertEqual(values['actions'].shape, (1, 16, 6))
        self.assertEqual(provenance['causal_checks_passed'], 11)
        for key, value in values.items():
            self.assertEqual(tensor_sha(value), provenance['tensors'][key]['sha256'])
        saved_observation = values['observation'].clone()
        saved_actions = values['actions'].clone()
        values['target'].fill_(37)
        self.assertTrue(torch.equal(values['observation'], saved_observation))
        self.assertTrue(torch.equal(values['actions'], saved_actions))
        with self.assertRaises(ValueError):
            read_training_window(PARENT / 'data_cache', 'unknown-window')

    def test_preflight_rejects_stale_sources_and_incomplete_reports(self):
        groups = [
            ['tokenwise_time/portable.py', 'tokenwise_time/test_parity.py', 'native_control/portable.py', 'native_control/vendor/model.py'],
            ['tokenwise_time/training.py', 'tokenwise_time/test_training.py', 'tokenwise_time/portable.py', 'adapter.py', 'native_control/portable.py', 'native_control/vendor/model.py'],
            ['tokenwise_time/training.py', 'tokenwise_time/training_data.py', 'tokenwise_time/profile_update.py',
             'tokenwise_time/test_training_independent.py', 'tokenwise_time/portable.py', 'adapter.py'],
        ]
        with tempfile.TemporaryDirectory() as directory:
            files, payloads = [], []
            for index, names in enumerate(groups):
                payload = {'status': 'passed', 'tests': 6, 'rows': [{}] * 56,
                           'source_sha256': {name: hashlib.sha256((PARENT / name).read_bytes()).hexdigest() for name in names}}
                path = Path(directory) / f'{index}.json'
                path.write_text(json.dumps(payload))
                files.append(path)
                payloads.append(payload)
            self.assertEqual(len(validate_preflight(*files)), 3)
            for index, field in ((0, 'rows'), (1, 'tests'), (2, 'source_sha256')):
                bad = copy.deepcopy(payloads[index])
                bad[field] = [] if field == 'rows' else 0 if field == 'tests' else {}
                files[index].write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    validate_preflight(*files)
                files[index].write_text(json.dumps(payloads[index]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Evidence output must be new')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        warnings.filterwarnings('ignore', message="User provided device_type of 'cuda'.*")
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTests))
    names = ['tokenwise_time/' + name for name in ('test_training_independent.py', 'training.py', 'training_data.py', 'profile_update.py', 'portable.py')]
    names += ['adapter.py', 'native_control/portable.py', 'native_control/vendor/model.py']
    report = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'seconds': time.perf_counter() - start,
              'device': 'cpu', 'torch': torch.__version__, 'pretrained_core_loaded': False,
              'source_sha256': {name: hashlib.sha256((PARENT / name).read_bytes()).hexdigest() for name in names},
              'scope': 'Independent training mechanics and actual development-cache integrity; no GPU, quality training, sampler or generalization result.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
