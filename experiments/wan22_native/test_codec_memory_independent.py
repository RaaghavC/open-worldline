# SPDX-License-Identifier: Apache-2.0
"""Independent CPU checks of allocator-only hooks and event failure cleanup."""
import argparse
import json
from pathlib import Path
import time
import unittest
from unittest import mock

import torch

from .codec import HERE, Wan22Codec, cleanup_temporal_chunks, make_model, sha
from .codec_memory import cleanup_causal_convolutions
from .vendor.vae2_2 import CausalConv3d

MEASUREMENTS = {}


def fixture():
    torch.manual_seed(20260915)
    return Wan22Codec.from_model(make_model(small=True, device='cpu')), torch.randn(1, 48, 5, 2, 2)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_full17_and_preserved_existing_hooks_and_output_objects(self):
        codec, latent = fixture()
        reference = codec.decode(latent)
        before = latent.clone()
        native, output_objects, owned = [], [], []
        for name, module in codec.model.named_modules():
            if isinstance(module, CausalConv3d):
                def observe(module, args, output, name=name):
                    native.append(name)
                    output_objects.append(output)
                owned.append((module, module.register_forward_hook(observe)))
        cleanups = []
        try:
            with cleanup_temporal_chunks(codec, lambda: None), cleanup_causal_convolutions(codec, lambda: cleanups.append(True)) as report:
                actual = codec.decode(latent)
            self.assertTrue(all(handle.id in module._forward_hooks for module, handle in owned))
            self.assertEqual([row['layer'] for row in report['events']], native)
            self.assertEqual(len(cleanups), len(native))
            self.assertEqual(report['registered_layers'], len(owned))
            self.assertTrue(report['hooks_removed'])
            self.assertTrue(all(torch.isfinite(value).all() for value in output_objects))
            self.assertTrue(torch.equal(actual, reference))
            self.assertTrue(torch.equal(latent, before))
            self.assertEqual(actual.shape, (1, 3, 17, 32, 32))
            self.assertTrue(codec.cache_is_clear())
            MEASUREMENTS.update(full17_max_absolute_difference=0., registered_layers=len(owned), native_calls=len(native))
        finally:
            for _, handle in owned:
                handle.remove()

    def test_event_sink_failure_removes_only_owned_hooks_and_clears_cache(self):
        codec, latent = fixture()
        seen = []
        retained = codec.model.conv2.register_forward_hook(lambda *args: seen.append(True))
        def broken_event(row):
            raise OSError('event storage unavailable')
        try:
            with self.assertRaisesRegex(OSError, 'event storage'):
                with cleanup_causal_convolutions(codec, lambda: None, broken_event) as report:
                    codec.decode(latent)
            self.assertEqual(report['cleanup_calls'], 1)
            self.assertEqual(report['completed_cleanups'], 1)
            self.assertTrue(report['hooks_removed'])
            self.assertTrue(codec.cache_is_clear())
            self.assertEqual(set(codec.model.conv2._forward_hooks), {retained.id})
            self.assertEqual(len(seen), 1)
        finally:
            retained.remove()
        self.assertTrue(torch.isfinite(codec.decode(latent)).all())

    def test_failed_synchronization_never_calls_empty_cache_and_removes_hooks(self):
        codec, latent = fixture()
        codec.device = torch.device('mps')
        with mock.patch.object(torch.mps, 'synchronize', side_effect=RuntimeError('sync failed')), mock.patch.object(torch.mps, 'empty_cache') as empty:
            with self.assertRaisesRegex(RuntimeError, 'sync failed'):
                with cleanup_causal_convolutions(codec) as report:
                    codec.model.conv2(latent)
        empty.assert_not_called()
        self.assertEqual(report['events'][0]['status'], 'failed')
        self.assertEqual(report['completed_cleanups'], 0)
        self.assertTrue(report['hooks_removed'])
        self.assertTrue(all(not module._forward_hooks for module in codec.model.modules()))
        self.assertTrue(codec.cache_is_clear())

    def test_source_bound_gates_and_unchanged_codec_math(self):
        from . import codec_decode_profile as profile
        self.assertEqual(sha(HERE/'codec.py'), '16f4a1df1219e91edcfebb56a9fee209dd67c7b9ac43b24519ea778f3ed26e30')
        self.assertEqual(sha(HERE/'codec_profile.py'), '1ea6268a7a74b35f06864ff525a6bb967318e8eae9ef6bfc5098a2ab835ea24a')
        self.assertEqual(len(profile.validate_cpu(HERE/'codec-results/cpu-v3/tests.json')), 64)
        self.assertEqual(len(profile.validate_decode_cpu(HERE/'codec-results/decode-cpu-v4/tests.json')), 64)
        self.assertEqual(len(profile.validate_memory_cpu(HERE/'codec-results/memory-cpu-v1/tests.json')), 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Evidence file must be new')
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    names = ('codec.py', 'codec_profile.py', 'codec_memory.py', 'codec_decode_profile.py',
             'test_codec_memory.py', 'test_codec_memory_independent.py', 'test_codec_decode.py', 'vendor/vae2_2.py')
    report = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
        'errors': len(result.errors), 'failures': len(result.failures), 'seconds': time.perf_counter()-started,
        'source_sha256': {name: sha(HERE/name) for name in names}, 'measurements': MEASUREMENTS,
        'device': 'cpu', 'actual_gpu_calls': False, 'external_weight_values_loaded': False,
        'scope': 'Tiny native full17 equations, hook ownership and event/synchronization failure paths. No measured memory saving or full-weight decode.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
