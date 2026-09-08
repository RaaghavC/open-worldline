"""Small random CPU bridge only. No checkpoints, weights or models from a run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import prototype as p
from experiments.wan22_native.action_cuda.test_cpu import fixture


class Fixed16Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix='action-fixed16-fixture-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def inputs(self):
        bridge, target, _, contexts, _, observation = fixture()
        adapter, initial, schedule, draws = p.fresh_inputs(tuple(target.shape), test_configuration={
            'hidden_dim': 8, 'observation_channels': 2, 'width': 8})
        bridge.adapter = adapter
        windows = {}
        for start in (0, 8, 32, 49):
            for arm in ('closed', 'open'):
                value = target.clone(); value[:, :, 1:] *= (1 + start/100) * (.7 if arm == 'open' else 1.)
                windows[f'{arm}-{start:04d}'] = {'target': value, 'observation': observation.clone(),
                                               'commands': p.expected_commands(arm, start)}
        optimizer = torch.optim.AdamW(adapter.parameters(), **p.original.OPTIMIZER)
        before = p.original.parameter_records(bridge.core, expected_count=len(list(bridge.core.parameters())))
        def native(x, times, context):
            with torch.no_grad():
                return torch.stack(bridge.core(list(x.unbind(0)), times, [context], times.shape[1]))
        return bridge, windows, schedule, draws, contexts[0], optimizer, initial, before, native

    def run_case(self, case, output):
        bridge, windows, schedule, draws, context, optimizer, initial, before, native = case
        return p.run_sequence(bridge, windows, schedule, draws, context, optimizer, output,
                              expected_initial=initial, expected_core=before, native_predict=native,
                              identity={'fixture': True, 'profile': 'baseline', 'fresh': True})

    def test_all16_actual_paired_updates_and_immutable_checkpoint_order(self):
        case = self.inputs(); bridge, windows, schedule, draws, _, optimizer, initial, _, _ = case
        output = self.root / 'full'; checkpoint_order = []
        original_save = p.original.save_checkpoint
        def saved(*args, **kwargs):
            record = original_save(*args, **kwargs)
            completed = args[3]; checkpoint_order.append(completed)
            pointer = json.loads((output / 'last-valid.json').read_text())
            self.assertEqual(pointer['completed_updates'], completed)
            return record
        with patch.object(p.original, 'save_checkpoint', side_effect=saved):
            report = self.run_case(case, output)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(checkpoint_order, list(range(17)))
        self.assertEqual([row['start'] for row in report['updates']], [0, 8, 32, 49] * 4)
        self.assertEqual(report['training_forwards'], 32)
        self.assertEqual(report['bridge_predictions'], 34)
        self.assertEqual(report['native_reference_predictions'], 2)
        self.assertTrue(report['base_unchanged'])
        self.assertEqual(report['updates'][0]['command_gru_gradient_l2'], 0.)
        self.assertGreater(report['updates'][1]['command_gru_gradient_l2'], 0.)
        self.assertTrue(all(p.grad is None for p in bridge.core.parameters()))
        self.assertEqual(len(list(output.glob('prediction-*.safetensors'))), 32)
        self.assertEqual(len(list(output.glob('gradients-after-clip-*.safetensors'))), 16)
        for index, row in enumerate(report['updates']):
            for branch, label in zip(row['branches'], ('closed', 'open')):
                key = f'{label}-{schedule[index]["start"]:04d}'
                self.assertEqual(branch['input_sha256']['commands'], p.tensor_sha(windows[key]['commands']))
                self.assertEqual(branch['branch'], label)
            recovery = torch.load(output / f'checkpoint-{index+1:04d}' / 'optimizer-and-rng.pt', weights_only=True)
            self.assertEqual(recovery['completed_updates'], index + 1)
            self.assertTrue(all(float(value['step']) == index + 1 for value in recovery['optimizer']['state'].values()))
            self.assertTrue(torch.equal(recovery['draw_rng_state'], draws[f'rng_after_{index:04d}']))
        saved = torch.load(output / 'checkpoint-0000/optimizer-and-rng.pt', weights_only=True)
        self.assertFalse(saved['optimizer']['state'])
        self.assertTrue(all(float(value['step']) == 16 for value in optimizer.state.values()))
        self.assertTrue(any(not torch.equal(initial[name], value) for name, value in bridge.adapter.state_dict().items()))

    def test_later_optimizer_failure_does_not_advance_last_valid_checkpoint(self):
        case = self.inputs(); optimizer = case[5]; step = optimizer.step; count = 0
        def fail_fourth(*args, **kwargs):
            nonlocal count
            count += 1
            result = step(*args, **kwargs)
            if count == 4:
                next(iter(optimizer.state.values()))['exp_avg'].fill_(float('nan'))
            return result
        output = self.root / 'failed'
        with patch.object(optimizer, 'step', side_effect=fail_fourth):
            with self.assertRaises(FloatingPointError):
                self.run_case(case, output)
        pointer = json.loads((output / 'last-valid.json').read_text())
        self.assertEqual(pointer['completed_updates'], 3)
        self.assertFalse((output / 'checkpoint-0004').exists())
        recovery = torch.load(output / 'checkpoint-0003/optimizer-and-rng.pt', weights_only=True)
        p.original.require_finite_tree(recovery)
        self.assertEqual(json.loads((output / 'metrics.json').read_text())['status'], 'failed')

    def test_fresh_draw_prefix_and_original_command_alignment(self):
        case = self.inputs(); _, windows, rows, draws, *_ = case
        first, two = p.original.make_draws('probe', shape=tuple(windows['closed-0000']['target'].shape))
        self.assertEqual(rows[:2], first)
        self.assertTrue(all(torch.equal(draws[name], value) for name, value in two.items()))
        self.assertEqual(rows[2]['branches'], ['closed-0032', 'open-0032'])
        self.assertEqual(torch.count_nonzero(windows['open-0000']['commands'][:, :, 5]), 1)
        self.assertEqual(torch.count_nonzero(windows['open-0032']['commands'][:, :9]), 0)
        changed = copy.deepcopy(windows); changed['open-0000']['commands'][:, 0, 5] = 0
        with self.assertRaisesRegex(ValueError, 'command'):
            p.validate_inputs(changed, rows, draws, tuple(windows['closed-0000']['target'].shape))
        swapped = copy.deepcopy(rows); swapped[2], swapped[3] = swapped[3], swapped[2]
        with self.assertRaisesRegex(ValueError, 'chronological'):
            p.validate_inputs(windows, swapped, draws, tuple(windows['closed-0000']['target'].shape))

    def test_resumed_state_or_changed_initialization_is_rejected(self):
        case = self.inputs(); adapter = case[0].adapter
        with torch.no_grad(): next(adapter.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, 'initialization'):
            self.run_case(case, self.root / 'not-fresh')
        case = self.inputs(); case[5].state[next(case[0].adapter.parameters())] = {'step': torch.tensor(2.)}
        with self.assertRaisesRegex(ValueError, 'resumed'):
            self.run_case(case, self.root / 'resumed')


if __name__ == '__main__':
    unittest.main()
