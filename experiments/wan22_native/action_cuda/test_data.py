# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for original RGB, native encode boundaries and cached evidence."""
import contextlib
import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch

from . import data


def fixture_windows():
    images = {}
    for arm in ['closed', 'open']:
        path = data.original.HERE / 'source-images' / (arm + '-0000.png')
        with Image.open(path) as image:
            images[arm] = np.array(image.convert('RGB'), dtype=np.uint8)
    result = {}
    for arm, start in data.SELECTION:
        timeline = [None, 'interact' if arm == 'open' else 'wait'] + ['left'] * 24 + ['wait'] * 16 + ['right'] * 24
        actions = np.stack([data.original.original.command_vector(x) for x in timeline[start + 1:start + 17]])
        pixels = np.stack([images[arm].copy() for _ in range(17)])
        raw = data.original.original.RGBActionWindow(pixels, actions, {
            'arm': arm, 'start': start, 'capture_manifest_sha256': data.original.MANIFEST_SHA256,
            'sources': [{'frame': start + i, 'file': arm + '.png',
                         'sha256': data.original.array_sha(images[arm])} for i in range(17)]})
        result[(arm, start)] = data.original.derive_window(raw, images['open'])
    return result


def fake_provenance():
    # Value-free synthetic loader record. No original parameter values are used.
    rows = {f'fixture.{i}': {'shape': [1], 'sha256': '0' * 64, 'cuda_copy_exact': True} for i in range(195)}
    rows['fixture.last'] = {'shape': [704688668 - 195], 'sha256': '1' * 64, 'cuda_copy_exact': True}
    return {'weight_sha256': data.VAE_SHA256, 'compute_dtype': 'float32', 'parameters': 704688668,
            'tensors': rows, 'config': data.VAE_CONFIG.copy()}


def causal_encoder(video):
    indices = [0] if video.shape[2] == 1 else [0, 4, 8, 12, 16]
    return video[:, :1, indices, ::16, ::16].expand(-1, 48, -1, -1, -1).contiguous().clone()


def rebind(directory, manifest):
    data._json(directory / 'manifest.json', manifest)
    completion = json.loads((directory / 'completion.json').read_text())
    completion['manifest_sha256'] = data.sha(directory / 'manifest.json')
    completion['output_sha256'] = {str(p.relative_to(directory)): data.sha(p) for p in directory.rglob('*')
                                    if p.is_file() and p.name != 'completion.json'}
    data._json(directory / 'completion.json', completion)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix='native-action-data-fixture-')
        cls.root = Path(cls.temporary.name)
        cls.windows = fixture_windows()
        cls.calls = []

        def encoder(video):
            cls.calls.append(video.shape[2])
            return causal_encoder(video)

        with mock.patch.object(data.original, 'load_selection', return_value=cls.windows):
            cls.completion = data.encode_cache('fixture-not-an-original-capture', 'baseline', cls.root / 'cache',
                                               encoder, fake_provenance())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def local_copy(self, name):
        destination = self.root / name
        shutil.copytree(self.root / 'cache', destination)
        return destination

    def test_original_18_changes_precede_exact_spatial_preprocessing(self):
        window = self.windows[('closed', 0)]
        self.assertEqual(window.provenance['changed_channel_values'], 18)
        self.assertTrue(np.array_equal(window.rgb[0], self.windows[('open', 0)].rgb[0]))
        for profile, size in [('baseline', (512, 288)), ('spatial', (1248, 704))]:
            with self.subTest(profile=profile):
                pixels, transform = data.processed_rgb(window, profile)
                expected = Image.fromarray(window.rgb[0])
                if profile == 'spatial':
                    expected = expected.resize((1252, 704), Image.Resampling.LANCZOS).crop((2, 0, 1250, 704))
                self.assertEqual(expected.size, size)
                self.assertEqual(pixels[0].tobytes(), expected.tobytes())
                if profile == 'spatial':
                    self.assertEqual(transform, {'resize': [1252, 704], 'crop': [2, 0, 1250, 704], 'resize_applied': True})
                tensor = data.video_tensor(pixels[:1])
                expected_tensor = torch.from_numpy(pixels[:1].copy()).permute(3, 0, 1, 2).float().div(255).sub(.5).div(.5)[None]
                self.assertTrue(torch.equal(tensor, expected_tensor))

    def test_independent_destination_command_oracle(self):
        for arm, start in data.SELECTION:
            expected = np.zeros((16, 6), np.float32)
            if start == 0:
                expected[1:, 3] = np.float32(math.pi / 24)
                expected[0, 5] = int(arm == 'open')
            elif start == 8:
                expected[:, 3] = np.float32(math.pi / 24)
            elif start == 32:
                expected[9:, 3] = np.float32(-math.pi / 24)
            else:
                expected[:, 3] = np.float32(-math.pi / 24)
            self.assertTrue(np.array_equal(self.windows[(arm, start)].commands, expected), (arm, start))
        self.assertTrue(np.array_equal(self.windows[('closed', 8)].commands, self.windows[('open', 8)].commands))

    def test_complete_counts_shared_observation_and_unmodified_targets(self):
        self.assertEqual(self.calls.count(1), 8)
        self.assertEqual(self.calls.count(17), 10)
        self.assertEqual(len(self.calls), 18)
        self.assertEqual(self.completion['unique_observations'], 7)
        values, record = data.read_window(self.root / 'cache', 'closed-0000')
        other, _ = data.read_window(self.root / 'cache', 'open-0000')
        self.assertTrue(torch.equal(values['observation'], other['observation']))
        self.assertEqual(record['materialized_tensor_keys'], ['target', 'observation', 'commands'])
        pixels, _ = data.processed_rgb(self.windows[('closed', 0)], 'baseline')
        self.assertTrue(torch.equal(values['target'], causal_encoder(data.video_tensor(pixels))))
        self.assertEqual(self.completion['encoder_calls_completed'], {'one_frame': 8, 'seventeen_frames': 10, 'total': 18})

    def test_condition_reader_never_materializes_full_target(self):
        real = data.safe_open; names = []

        class Spy:
            def __init__(self, *args, **kwargs):
                self.handle = real(*args, **kwargs)
            def __enter__(self):
                self.handle.__enter__(); return self
            def __exit__(self, *args):
                return self.handle.__exit__(*args)
            def keys(self):
                return self.handle.keys()
            def get_slice(self, name):
                return self.handle.get_slice(name)
            def get_tensor(self, name):
                if name == 'target':
                    raise AssertionError('A conditioning read accessed the full future target')
                names.append(name); return self.handle.get_tensor(name)

        with mock.patch.object(data, 'safe_open', Spy):
            values, record = data.read_window(self.root / 'cache', 'open-0032', conditioning_only=True)
        self.assertEqual(set(values), {'observation', 'commands'})
        self.assertEqual(names[-2:], ['observation', 'commands'])
        self.assertEqual(record['materialized_tensor_keys'], ['observation', 'commands'])

    def test_cross_length_tolerance_is_distinct_from_bit_exact_causality(self):
        reference = torch.ones(1, 48, 1, 2, 3)
        candidate = reference + 1e-6
        self.assertTrue(data.prefix_check(candidate, reference, 'x', 'cross-length')['passed'])
        row = data.prefix_check(candidate, reference, 'x', 'same-length', require_bit_exact=True)
        self.assertFalse(row['passed']); self.assertFalse(row['bit_exact_equal'])
        self.assertFalse(data.prefix_check(reference + 2e-5, reference, 'x', 'cross-length')['passed'])
        self.assertFalse(data.prefix_check(reference * 1e-8, torch.zeros_like(reference), 'x', 'cross-length')['passed'])
        self.assertTrue(data.prefix_check(torch.zeros_like(reference), torch.zeros_like(reference), 'x', 'cross-length')['passed'])
        before = candidate.clone()
        data.prefix_check(candidate, reference, 'x', 'cross-length')
        self.assertTrue(torch.equal(candidate, before))

    def test_fabricated_exact_flag_is_recomputed_from_saved_prefix_tensors(self):
        directory = self.local_copy('changed-prefix')
        manifest = json.loads((directory / 'manifest.json').read_text())
        row = next(c for c in manifest['causal_checks'] if c['requires_bit_exact'])
        path = directory / row['evidence']['file']; values = load_file(str(path))
        values['candidate'].flatten()[0] += 1e-6
        save_file(values, str(path)); row['evidence']['sha256'] = data.sha(path)
        row['evidence']['bytes'] = path.stat().st_size
        row['evidence']['tensors']['candidate']['sha256'] = data.tensor_sha(values['candidate'])
        rebind(directory, manifest)
        with self.assertRaisesRegex(ValueError, 'Recomputed causal-prefix'):
            data.read_window(directory, 'open-0000')

    def test_diagnostic_norm_roundoff_does_not_change_model_gates(self):
        for case, factor, accepted in [('last-bits', 1 + 5e-13, True),
                                       ('changed-norm', 1 + 5e-10, False)]:
            with self.subTest(case=case):
                directory = self.local_copy(case)
                manifest = json.loads((directory / 'manifest.json').read_text())
                manifest['causal_checks'][0]['relative_l2_reference_norm'] *= factor
                rebind(directory, manifest)
                if accepted:
                    values, _ = data.read_window(directory, 'closed-0000')
                    self.assertEqual(set(values), {'target', 'observation', 'commands'})
                else:
                    with self.assertRaisesRegex(ValueError, 'Recomputed causal-prefix'):
                        data.read_window(directory, 'closed-0000')
        reference = torch.ones(1, 48, 1, 2, 3)
        measured = data.prefix_check(reference + 1e-6, reference, 'x', 'cross-length')
        saved = copy.deepcopy(measured); saved['relative_l2'] *= 1 + 5e-13
        self.assertTrue(data._matches_prefix_row(saved, measured))
        changed = copy.deepcopy(measured); changed['max_abs'] += 1e-15
        self.assertFalse(data._matches_prefix_row(changed, measured))
        failed = data.prefix_check(reference + 2e-5, reference, 'x', 'cross-length')
        self.assertFalse(data._matches_prefix_row(failed, failed))
        nonexact = data.prefix_check(reference + 1e-6, reference, 'x', 'same-length', require_bit_exact=True)
        self.assertFalse(data._matches_prefix_row(nonexact, nonexact))

    def test_file_paths_reject_traversal_before_reading_outside_cache(self):
        directory = (self.root / 'cache').resolve()
        with mock.patch.object(data, 'sha', side_effect=AssertionError('Invalid path was read')):
            for name in ['../outside', '/outside', 'a/../outside', 'a//b', 'a\\b']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    data._checked_file(directory, {'file': name, 'sha256': '0' * 64})

    def test_missing_hash_inventory_and_source_or_command_tampering_rejected(self):
        for case in ['missing-output', 'changed-source', 'changed-command-source']:
            with self.subTest(case=case):
                directory = self.local_copy(case)
                manifest = json.loads((directory / 'manifest.json').read_text())
                if case == 'missing-output':
                    completion = json.loads((directory / 'completion.json').read_text())
                    completion['output_sha256'] = {}; data._json(directory / 'completion.json', completion)
                elif case == 'changed-source':
                    manifest['windows'][0]['source']['processed_first_rgb_sha256'] = '0' * 64
                    rebind(directory, manifest)
                else:
                    manifest['windows'][0]['tensors']['commands']['sha256'] = '0' * 64
                    rebind(directory, manifest)
                with self.assertRaises(ValueError):
                    data.read_window(directory, 'closed-0000')

    def test_encoder_error_retains_failed_completion_without_admitting_partial_cache(self):
        calls = []
        def failing(video):
            calls.append(video.shape[2])
            if len(calls) == 2:
                raise RuntimeError('Synthetic full-video failure')
            return causal_encoder(video)
        directory = self.root / 'interrupted'
        with mock.patch.object(data.original, 'load_selection', return_value=self.windows):
            with self.assertRaisesRegex(RuntimeError, 'Synthetic'):
                data.encode_cache('fixture', 'baseline', directory, failing, fake_provenance())
        completion = json.loads((directory / 'completion.json').read_text())
        self.assertEqual(completion['status'], 'failed')
        self.assertEqual(completion['encoder_call_attempts']['total'], 2)
        self.assertEqual(completion['encoder_calls_completed']['total'], 1)
        with self.assertRaises(ValueError):
            data.read_window(directory, 'closed-0000')

    def test_native_cuda_boundary_disabled_autocast_and_final_cache_clear(self):
        class Codec:
            def __init__(self):
                self.clears = 0; self.seen = []
            def clear_cache(self):
                self.clears += 1
            def encode(self, video, scale):
                self.seen.append(video.shape)
                self.dtype = video.dtype
                return causal_encoder(video)
        codec = Codec(); scope = []
        def autocast(device, *, enabled):
            scope.append((device, enabled)); return contextlib.nullcontext()
        with mock.patch.object(data, '_to_cuda', side_effect=lambda value: value), \
             mock.patch.object(data, 'verify_sources'), mock.patch.object(torch.cuda, 'synchronize'), \
             mock.patch.object(torch, 'autocast', side_effect=autocast):
            for profile in ['baseline', 'spatial']:
                item = data.spec(profile)
                for frames in [1, 17]:
                    video = torch.zeros(1, 3, frames, item.height, item.width)
                    result = data.native_encode(codec, object(), video, profile)
                    self.assertEqual(result.shape, (1, 48, 1 if frames == 1 else 5, item.height // 16, item.width // 16))
            with mock.patch.object(codec, 'encode', side_effect=RuntimeError('fixture')):
                with self.assertRaises(RuntimeError):
                    data.native_encode(codec, object(), torch.zeros(1, 3, 1, 288, 512), 'baseline')
        self.assertEqual(codec.clears, 10)
        self.assertEqual(codec.dtype, torch.float32)
        self.assertEqual(scope, [('cuda', False)] * 5)

    def test_plan_is_deterministic_cpu_only_and_output_is_fresh(self):
        with mock.patch.object(data.original, 'load_selection', return_value=self.windows), \
             mock.patch.object(data, 'native_encode', side_effect=AssertionError('Model call')):
            first = data.plan('fixture', 'baseline'); second = data.plan('fixture', 'baseline')
            self.assertEqual(data.canonical_sha(first), data.canonical_sha(second))
            self.assertFalse(first['model_execution'])
            self.assertEqual([x['id'] for x in first['selection']], [f'{arm}-{start:04d}' for arm, start in data.SELECTION])
            output = self.root / 'plan'
            data.write_plan('fixture', 'baseline', output)
            with self.assertRaises(ValueError):
                data.write_plan('fixture', 'baseline', output)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
