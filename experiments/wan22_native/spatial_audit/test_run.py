# SPDX-License-Identifier: Apache-2.0
"""CPU checks for prior-report arithmetic, result bounds and failure retention."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.codec = self.root / 'codec'; self.pair = self.root / 'pair'; self.clip = self.root / 'clip'

    def prior(self, pair_seconds=10.25):
        write(self.codec / 'metrics.json', {'synthetic': 'codec'})
        codec = {'hardware': {'fixture': True}, 'load_seconds': 37.1,
                 'profiles': {'spatial': {'decode_seconds': 8.5}}}
        write(self.codec / 'codec/result/metrics.json', codec)
        write(self.pair / 'metrics.json', {'codec_result_report_sha256': run.sha(self.codec / 'metrics.json')})
        core = {'hardware': codec['hardware'], 'load_seconds': 230.7, 'pair_seconds': pair_seconds}
        write(self.pair / 'core/result/metrics.json', core)
        write(self.clip / 'core/result/metrics.json', {'hardware': codec['hardware']})
        write(self.clip / 'decode/result/metrics.json', {'hardware': codec['hardware']})
        values = {'core_load_seconds': 230.7, 'pair_seconds': pair_seconds, 'codec_load_seconds': 37.1, 'decode_seconds': 8.5}
        estimate = 230.7 + 50 * pair_seconds * 2.0 + 1.2 * (37.1 + 8.5) + 30
        clip = {'pair_result_report_sha256': run.sha(self.pair / 'metrics.json'),
                'codec_result_report_sha256': run.sha(self.codec / 'metrics.json'),
                'admission': {'estimated_seconds': estimate, 'limit_seconds': 1800, 'profile': 'spatial',
                              'measured_seconds': values, 'factors': {'pairs': 50, 'pair_safety': 2.0,
                                                                   'codec_safety': 1.2, 'artifact_seconds': 30}}}
        write(self.clip / 'metrics.json', clip)
        return clip

    def test_prior_binding_and_arithmetic(self):
        clip = self.prior()
        result = run.check_prior(self.clip, self.pair, self.codec, 'spatial')
        self.assertEqual(result['estimated_seconds_recomputed'], clip['admission']['estimated_seconds'])
        self.assertTrue(result['does_not_verify_provider_deadline_or_billing'])
        clip['admission']['estimated_seconds'] += .0001
        write(self.clip / 'metrics.json', clip)
        with self.assertRaises(ValueError):
            run.check_prior(self.clip, self.pair, self.codec, 'spatial')

    def test_wrong_prior_report_and_cross_hardware_rejected(self):
        self.prior(); write(self.pair / 'metrics.json', {'changed': True})
        with self.assertRaises(ValueError):
            run.check_prior(self.clip, self.pair, self.codec, 'spatial')
        self.prior()
        core = run.obj(self.pair / 'core/result/metrics.json'); core['hardware'] = {'different': True}
        write(self.pair / 'core/result/metrics.json', core)
        with self.assertRaises(ValueError):
            run.check_prior(self.clip, self.pair, self.codec, 'spatial')

    def test_over_limit_and_nonpositive_times_rejected(self):
        for duration in (20, 0, -1, True):
            self.prior(duration)
            with self.assertRaises(ValueError):
                run.check_prior(self.clip, self.pair, self.codec, 'spatial')

    def test_inventory_size_symlink_and_exact_bytes(self):
        root = self.root / 'artifacts'; root.mkdir(); path = root / 'one'; path.write_bytes(b'abc')
        self.assertEqual(run.inventory(root), {'one': {'bytes': 3, 'sha256': run.sha(path)}})
        link = root / 'link'; link.symlink_to(path)
        with self.assertRaises(ValueError):
            run.inventory(root)
        link.unlink()
        with patch.object(run, 'MAX_RUN_BYTES', 2), self.assertRaises(ValueError):
            run.inventory(root)
        with path.open('wb') as stream:
            stream.truncate(100_000_001)
        with self.assertRaises(ValueError):
            run.inventory(root)

    def test_empty_input_failure_retains_failed_report(self):
        for name in ('run', 'codec', 'packet'):
            (self.root / name).mkdir(exist_ok=True)
        output = self.root / 'audit'
        argv = ['--mode', 'pair', '--profile', 'baseline', '--input-manifest-sha256', '0' * 64,
                '--expected-gpu', 'NVIDIA A100-SXM4-80GB', '--run-root', str(self.root / 'run'),
                '--codec-root', str(self.root / 'codec'), '--packet-root', str(self.root / 'packet'),
                '--output', str(output)]
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            run.main(argv)
        report = run.obj(output / 'report.json')
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['new_model_execution'])
        self.assertFalse(report['provider_access'])
        self.assertEqual(list((self.root / 'run').iterdir()), [])
        with self.assertRaisesRegex(ValueError, 'Fresh audit'):
            run.main(argv)

    def test_full_input_inventory_is_bound_even_on_later_failure(self):
        for name in ('run', 'codec', 'packet'):
            folder = self.root / name; folder.mkdir(exist_ok=True)
            (folder / 'payload').write_bytes(name.encode())
        output = self.root / 'audit'
        argv = ['--mode', 'pair', '--profile', 'baseline', '--input-manifest-sha256', '0' * 64,
                '--expected-gpu', 'NVIDIA A100-SXM4-80GB', '--run-root', str(self.root / 'run'),
                '--codec-root', str(self.root / 'codec'), '--packet-root', str(self.root / 'packet'),
                '--output', str(output)]
        with patch.object(run, 'validate_input_packet', side_effect=ValueError('Injected packet rejection')):
            with self.assertRaisesRegex(ValueError, 'Injected packet rejection'):
                run.main(argv)
        report = run.obj(output / 'report.json')
        inventory = run.obj(output / 'input-inventory.json')
        self.assertEqual(report['input_inventory_sha256'], run.sha(output / 'input-inventory.json'))
        for name in ('run', 'codec', 'packet'):
            self.assertEqual(inventory[name], run.inventory(self.root / name))
        self.assertEqual(report['status'], 'failed')

    def test_output_inside_measured_root_rejected_before_creation(self):
        root = self.root / 'measured'; root.mkdir(); out = root / 'audit'
        argv = ['--mode', 'pair', '--profile', 'baseline', '--input-manifest-sha256', '0' * 64,
                '--expected-gpu', 'NVIDIA A100-SXM4-80GB', '--run-root', str(root),
                '--codec-root', str(root), '--packet-root', str(root), '--output', str(out)]
        with self.assertRaises(ValueError):
            run.main(argv)
        self.assertFalse(out.exists())

    def test_output_inside_sibling_source_or_symlink_parent_rejected(self):
        root = self.root / 'measured'; root.mkdir()
        argv = ['--mode', 'pair', '--profile', 'baseline', '--input-manifest-sha256', '0' * 64,
                '--expected-gpu', 'NVIDIA A100-SXM4-80GB', '--run-root', str(root),
                '--codec-root', str(root), '--packet-root', str(root)]
        sibling = run.HERE.parent / 'spatial_reference' / 'forbidden-audit-test-output'
        self.assertFalse(sibling.exists())
        with self.assertRaises(ValueError):
            run.main(argv + ['--output', str(sibling)])
        self.assertFalse(sibling.exists())
        link = self.root / 'link'; link.symlink_to(root, target_is_directory=True)
        with self.assertRaises(ValueError):
            run.main(argv + ['--output', str(link / 'new')])

    def test_json_duplicates_nonfinite_and_size_rejected(self):
        path = self.root / 'report.json'
        for text in ('{"x":1,"x":2}', '{"x":Infinity}', '[]', ' ' * (4 * 2**20 + 1)):
            path.write_text(text)
            with self.assertRaises(ValueError):
                run.obj(path)


if __name__ == '__main__':
    unittest.main()
