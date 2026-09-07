# SPDX-License-Identifier: Apache-2.0
"""CPU integration checks and a source-bound report; no CUDA model execution."""
import argparse
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock
import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
from ..official_cpu.streaming import sha, tensor_sha
from .config import SPECS
from .inputs import prepare
from .output import save_rgb, read_tensors, first_frame, difference
from . import run
from .evidence import sources

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name).resolve()
        cls.packet = cls.folder / 'packet'
        prepare(REPO / 'experiments/wan22_native/action_data/source-images/open-0000.png',
                REPO / 'experiments/wan22_native/core-results/cpu-pair-v1',
                REPO / 'experiments/wan_adapter/text_cache/native-results', cls.packet)
        cls.packet_sha = sha(cls.packet / 'manifest.json')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_lossless_rgb_frame_split_and_pixel_rounding(self):
        selected = SPECS['baseline']
        video = torch.linspace(-1, 1, 1 * 3 * 17 * 288 * 512).reshape(1, 3, 17, 288, 512)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = save_rgb(video, root / 'rgb', selected, purpose='generated_clip')
            self.assertEqual(record['shape'], list(video.shape))
            self.assertEqual(len(record['frames']), 17)
            for item in record['frames']:
                target = root / 'rgb' / item['file']
                restored = load_file(target)['rgb']
                self.assertTrue(torch.equal(restored, video[:, :, item['index']:item['index'] + 1]))
                self.assertEqual(sha(target), item['sha256'])
                self.assertEqual(tensor_sha(restored), item['tensor_sha256'])
                self.assertLess(item['bytes'], 100_000_000)
            first_frame(video, root / 'first.png')
            with Image.open(root / 'first.png') as image:
                oracle = np.rint((video[0, :, 0].permute(1, 2, 0).numpy() + 1) * np.float32(127.5)).astype(np.uint8)
                self.assertTrue(np.array_equal(np.asarray(image), oracle))
            with self.assertRaises(FileExistsError):
                save_rgb(video, root / 'rgb', selected, purpose='generated_clip')

    def test_rgb_rejects_wrong_shape_range_and_purpose(self):
        selected = SPECS['baseline']
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'bad'
            for value in (torch.zeros(1, 3, 1, 288, 512), torch.full((1, 3, 17, 288, 512), 2.0)):
                with self.assertRaises(ValueError):
                    save_rgb(value, out, selected, purpose='generated_clip')
            with self.assertRaises(ValueError):
                save_rgb(torch.zeros(1, 3, 17, 288, 512), out, selected, purpose='unspecified')
            self.assertFalse(out.exists())

    def test_bounded_tensor_headers_and_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'tensor.safetensors'
            save_file({'latent': torch.arange(6).float().reshape(2, 3)}, target)
            self.assertTrue(torch.equal(read_tensors(target, {'latent': (2, 3)})['latent'], torch.arange(6).float().reshape(2, 3)))
            for shapes in ({'other': (2, 3)}, {'latent': (3, 2)}):
                with self.assertRaises(ValueError):
                    read_tensors(target, shapes)
            with self.assertRaises(ValueError):
                read_tensors(target, {'latent': (2, 3)}, maximum=10)
            link = Path(folder) / 'link'; link.symlink_to(target)
            with self.assertRaises(ValueError):
                read_tensors(link, {'latent': (2, 3)})
            save_file({'latent': torch.full((2, 3), float('nan'))}, target)
            with self.assertRaises(ValueError):
                read_tensors(target, {'latent': (2, 3)})

    def test_difference_declares_reference_normalizer(self):
        result = difference(torch.tensor([3., 5.]), torch.tensor([1., 1.]))
        self.assertEqual(result['reference_rms'], 1)
        self.assertAlmostEqual(result['rmse'], 10**.5)
        self.assertEqual(difference(torch.zeros(2), torch.zeros(2))['relative_rmse'], None)
        with self.assertRaises(ValueError):
            difference(torch.ones(2), torch.ones(3))

    def test_codec_plan_performs_no_hardware_or_model_access(self):
        # This test isolates the CLI plan. The actual source-bound CPU report is
        # produced after all test modules finish, not fabricated by this mock.
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder).resolve() / 'plan'
            review = Path(folder) / 'review.json'; review.write_text('{}')
            argv = ['--mode', 'codec', '--profile', 'both', '--expected-gpu', 'NVIDIA A100-SXM4-80GB',
                    '--weights', str(Path(folder) / 'nonexistent-weights'), '--inputs', str(self.packet),
                    '--input-manifest-sha256', self.packet_sha, '--cpu-report', str(review), '--output', str(out)]
            with (patch.object(run, 'preflight', return_value=sha(review)),
                  patch.object(run, 'hardware') as hardware, patch.object(run, 'launch') as launch):
                result = run.main(argv)
            self.assertEqual(result['status'], 'planned')
            self.assertFalse(result['model_execution'])
            hardware.assert_not_called(); launch.assert_not_called()
            self.assertEqual(sha(out / 'inputs/manifest.json'), self.packet_sha)
            self.assertEqual(result['limits']['seconds'], 600)

    def test_worker_source_change_fails_before_hardware(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'result'
            config = {'output': str(out), 'mode': 'codec', 'profile': 'both', 'stage': 'codec',
                      'source_sha256': {'wrong': 'wrong'}, 'input_manifest_sha256': self.packet_sha}
            with patch.object(run, 'hardware') as hardware:
                with self.assertRaisesRegex(ValueError, 'Source changed'):
                    run.worker(config)
            hardware.assert_not_called()
            report = json.loads((out / 'metrics.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['predictions'], 0)

    def test_monitor_exit_failure_cannot_report_passed(self):
        from .inputs import load_prepared
        tensors, _, _ = load_prepared(self.packet)
        observations = {name: torch.zeros(selected.observation_shape) for name, selected in SPECS.items()}
        class FailingMonitor:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def check(self): pass
            def __exit__(self, *args): raise RuntimeError('Injected monitor exit failure')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); observation = root / 'observations.safetensors'
            save_file(observations, observation)
            config = {'output': str(root / 'result'), 'mode': 'pair', 'profile': 'baseline', 'stage': 'core',
                      'source_sha256': sources(), 'input_manifest_sha256': self.packet_sha,
                      'expected_gpu': 'injected', 'deadline': time.monotonic() + 900,
                      'inputs': str(self.packet), 'weights': str(root / 'unused'),
                      'observation_file': str(observation), 'observation_sha256': sha(observation),
                      'observation_tensor_sha256': tensor_sha(observations['baseline'])}
            with (patch.object(run, 'hardware', return_value={}), patch.object(run, 'Monitor', FailingMonitor),
                  patch(__package__ + '.native.load_model', return_value=(object(), [])),
                  patch(__package__ + '.native.predict', side_effect=lambda model, x, t, context: torch.zeros_like(x)),
                  patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'empty_cache')):
                with self.assertRaisesRegex(RuntimeError, 'Injected monitor exit failure'):
                    run.worker(config)
            report = json.loads((root / 'result/metrics.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['predictions'], 2)
            self.assertTrue((root / 'result/outputs.safetensors').exists())

    def test_handoff_retains_terminal_when_cleanup_also_fails(self):
        child = Mock(); child.returncode = None
        child.stdin.write.side_effect = BrokenPipeError('Injected handoff failure')
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'child'
            with (patch.object(run.subprocess, 'Popen', return_value=child),
                  patch.object(run, 'stop_child', side_effect=RuntimeError('Injected cleanup failure'))):
                with self.assertRaisesRegex(BrokenPipeError, 'Injected handoff failure'):
                    run.launch({'mode': 'codec', 'deadline': time.monotonic() + 600}, out)
            terminal = json.loads((out / 'terminal.json').read_text())
            self.assertEqual(terminal['status'], 'failed')
            self.assertEqual(terminal['cleanup_error'], 'Injected cleanup failure')
            self.assertEqual(terminal['error'], 'Injected handoff failure')

    def test_preflight_rejects_stale_test_or_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'review.json'
            report = {'schema': 'wan22-spatial-cpu-review-v1', 'status': 'passed', 'source_sha256': sources(),
                      'test_sha256': {p.name: sha(p) for p in sorted(HERE.glob('test_*.py'))},
                      'failures': 0, 'errors': 0, 'skipped': 0, 'tests_run': 100}
            path.write_text(json.dumps(report))
            self.assertEqual(run.preflight(path), sha(path))
            report['test_sha256']['test_protocol.py'] = '0' * 64
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                run.preflight(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh CPU report required')
    before = sources(); tests = {p.name: sha(p) for p in sorted(HERE.glob('test_*.py'))}
    suite = unittest.defaultTestLoader.loadTestsFromNames([__package__ + '.' + p.stem for p in sorted(HERE.glob('test_*.py'))])
    started = time.monotonic(); stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    unchanged = before == sources() and tests == {p.name: sha(p) for p in sorted(HERE.glob('test_*.py'))}
    passed = result.wasSuccessful() and not result.skipped and unchanged
    report = {'schema': 'wan22-spatial-cpu-review-v1', 'status': 'passed' if passed else 'failed',
              'source_sha256': before, 'test_sha256': tests, 'sources_unchanged': unchanged,
              'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'elapsed_seconds': time.monotonic() - started,
              'python': sys.version, 'dependencies': {k: importlib.metadata.version(k)
                  for k in ('torch', 'numpy', 'diffusers', 'safetensors', 'psutil', 'Pillow')},
              'model_execution': False, 'cuda_initialized': torch.cuda.is_initialized(), 'log': stream.getvalue()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'tests_run', 'failures', 'errors', 'skipped', 'elapsed_seconds')}))
    if not passed:
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == '__main__':
    main()
