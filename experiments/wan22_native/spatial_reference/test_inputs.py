# SPDX-License-Identifier: Apache-2.0
"""Independent CPU checks of the original-image and prepared-input contract.

These checks use the public retained image, pair and text artifacts. They never
load a model or initialize CUDA. Saved spatial noise is authoritative; no test
requires seed regeneration to produce equal values on another architecture.
"""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from . import config, inputs


HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent
EXPERIMENTS = NATIVE.parent
SOURCE = NATIVE / 'codec-results/first-image-v2/original.png'
PAIR = NATIVE / 'core-results/cpu-pair-v1'
TEXT = EXPERIMENTS / 'wan_adapter/text_cache/native-results'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def independent_rgb(pixels):
    # Preserve the native sequence of FP32 arithmetic, rather than the
    # algebraically similar expression pixels / 127.5 - 1.
    array = pixels.astype(np.float32)
    array = array / np.float32(255)
    array = array - np.float32(.5)
    array = array / np.float32(.5)
    return np.ascontiguousarray(array.transpose(2, 0, 1))[None, :, None]


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.packet = cls.root / 'prepared'
        cls.raw_hashes = {str(path): digest(path) for path in
                          [SOURCE, *(PAIR / name for name in inputs.PAIR_PINS),
                           *(TEXT / name for name in inputs.TEXT_PINS)]}
        with ExitStack() as stack:
            for name in ('init', 'synchronize', 'set_device'):
                stack.enter_context(patch.object(torch.cuda, name,
                    side_effect=AssertionError('CUDA is outside this CPU preparation')))
            draw = stack.enter_context(patch.object(inputs.torch, 'randn', wraps=torch.randn))
            cls.manifest = inputs.prepare(SOURCE, PAIR, TEXT, cls.packet)
        if draw.call_count != 1:
            raise AssertionError('Only the new spatial noise may be drawn')
        if draw.call_args.args != (config.SPECS['spatial'].latent_shape,):
            raise AssertionError('Unexpected newly drawn noise shape')
        if draw.call_args.kwargs['device'] != 'cpu' or draw.call_args.kwargs['dtype'] != torch.float32:
            raise AssertionError('Spatial noise must be prepared on CPU in FP32')
        cls.values, cls.contexts, _ = inputs.load_prepared(cls.packet)

    @classmethod
    def tearDownClass(cls):
        for name, expected in cls.raw_hashes.items():
            if digest(name) != expected:
                raise AssertionError('A retained original artifact changed')
        cls.temporary.cleanup()

    def packet_copy(self, name):
        destination = self.root / name
        shutil.copytree(self.packet, destination)
        self.addCleanup(shutil.rmtree, destination)
        return destination

    def update_manifest(self, root, transform):
        path = root / 'manifest.json'
        value = json.loads(path.read_text())
        transform(value)
        path.write_text(json.dumps(value, indent=2) + '\n')

    def rebind_file(self, root, name):
        self.update_manifest(root, lambda value: value['files'].__setitem__(name,
            {'bytes': (root / name).stat().st_size, 'sha256': digest(root / name)}))

    def rebind_tensors(self, root, values):
        save_file({key: value.contiguous() for key, value in values.items()}, str(root / 'prepared.safetensors'))
        self.rebind_file(root, 'prepared.safetensors')
        self.update_manifest(root, lambda record: record.__setitem__('tensor_sha256',
            {key: tensor_digest(value) for key, value in values.items()}))

    def test_exact_source_opaque_alpha_and_baseline_rgb_bytes(self):
        self.assertEqual(SOURCE.stat().st_size, 184595)
        self.assertEqual(digest(SOURCE), '7bdfa121eb2917b837af3ee1faae9e697751cdd3cdee0b21422c1b1f2a53e780')
        with Image.open(SOURCE) as source:
            self.assertEqual(source.mode, 'RGBA')
            self.assertEqual(source.size, (512, 288))
            rgba = np.asarray(source)
        np.testing.assert_array_equal(rgba[:, :, 3], np.full((288, 512), 255, np.uint8))
        with Image.open(self.packet / 'baseline.png') as baseline:
            self.assertEqual(baseline.mode, 'RGB')
            np.testing.assert_array_equal(np.asarray(baseline), rgba[:, :, :3])
        np.testing.assert_array_equal(self.values['baseline_rgb'].numpy(), independent_rgb(rgba[:, :, :3]))
        self.assertEqual(self.manifest['preprocessing']['baseline'],
                         {'resize': [512, 288], 'crop': [0, 0, 512, 288], 'resize_applied': False})

    def test_official_spatial_size_crop_and_fp32_pixels(self):
        self.assertEqual(inputs.best_output_size(512, 288, 32, 32, 1280 * 704), (1248, 704))
        with Image.open(SOURCE) as original:
            source = Image.fromarray(np.asarray(original)[:, :, :3].copy())
        # Independently fixed native result: scale 704/288, rounded width1252,
        # then the centered 1248-pixel crop removes two pixels on each side.
        expected = source.resize((1252, 704), Image.Resampling.LANCZOS).crop((2, 0, 1250, 704))
        with Image.open(self.packet / 'spatial.png') as retained:
            self.assertEqual(retained.size, (1248, 704))
            self.assertEqual(retained.tobytes(), expected.tobytes())
        np.testing.assert_array_equal(self.values['spatial_rgb'].numpy(), independent_rgb(np.asarray(expected)))
        self.assertEqual(self.manifest['preprocessing']['spatial'],
                         {'resize': [1252, 704], 'crop': [2, 0, 1250, 704], 'resize_applied': True})

    def test_specs_and_actual_saved_reference_mapping(self):
        baseline, spatial = config.spec('baseline'), config.spec('spatial')
        self.assertEqual((baseline.latent_shape, baseline.observation_shape, baseline.tokens, baseline.prefix_tokens),
                         ((48, 5, 18, 32), (1, 48, 1, 18, 32), 720, 144))
        self.assertEqual((spatial.latent_shape, spatial.observation_shape, spatial.tokens, spatial.prefix_tokens),
                         ((48, 5, 44, 78), (1, 48, 1, 44, 78), 4290, 858))
        self.assertEqual(config.SETTINGS, {'steps': 50, 'shift': 5., 'guidance': 5.})
        self.assertEqual(config.SEED, 20260908)
        for name in ('other', None, 17):
            with self.assertRaises(ValueError):
                config.spec(name)
        original = load_file(str(PAIR / 'inputs.safetensors'), device='cpu')
        self.assertTrue(torch.equal(self.values['baseline_noise'], original['initial_noise']))
        self.assertTrue(torch.equal(self.values['reference_observation'], original['observation']))
        for name, expected in inputs.PAIR_PINS.items():
            self.assertEqual(digest(self.packet / ('reference-' + name)), expected)
        for source, retained in [('embeddings.safetensors', 'contexts.safetensors'),
                                 ('manifest.json', 'text-manifest.json')]:
            self.assertEqual(digest(self.packet / retained), inputs.TEXT_PINS[source])
        self.assertEqual(set(self.contexts), {'atrium', 'native_negative'})
        self.assertEqual(tuple(self.contexts['atrium'].shape), (25, 4096))
        self.assertEqual(tuple(self.contexts['native_negative'].shape), (126, 4096))
        for value in [*self.values.values(), *self.contexts.values()]:
            self.assertEqual(value.device.type, 'cpu')
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).all())
        for name in ('baseline_noise_regenerated', 'observation_encoder_executed', 'actions_read', 'future_targets_read'):
            self.assertIs(self.manifest[name], False)

    def test_bad_source_hash_existing_output_and_nonopaque_alpha(self):
        with self.assertRaises(ValueError):
            inputs.prepare(SOURCE, PAIR, TEXT, self.packet)
        changed = self.root / 'changed-source.png'
        source_bytes = bytearray(SOURCE.read_bytes())
        source_bytes[-1] ^= 1
        changed.write_bytes(source_bytes)
        with self.assertRaisesRegex(ValueError, 'Pinned original'):
            inputs.prepare(changed, PAIR, TEXT, self.root / 'bad-hash-output')
        self.assertFalse((self.root / 'bad-hash-output').exists())
        with Image.open(SOURCE) as source:
            pixels = np.array(source, copy=True)
        for alpha in (0, 254):
            pixels[0, 0, 3] = alpha
            image = self.root / f'alpha-{alpha}.png'
            Image.fromarray(pixels).save(image)
            # Synthetic file pins only in this fixture reach the explicit
            # opacity guard. Production source constants remain unchanged.
            with patch.object(inputs, 'SOURCE_SHA', digest(image)), patch.object(inputs, 'SOURCE_BYTES', image.stat().st_size):
                with self.assertRaisesRegex(ValueError, 'nonopaque'):
                    inputs.prepare(image, PAIR, TEXT, self.root / f'alpha-output-{alpha}')

    def test_corrupt_manifest_schema_profiles_keys_and_seed(self):
        mutations = [lambda value: value.__setitem__('schema', 'unknown'),
                     lambda value: value.__setitem__('status', 'passed'),
                     lambda value: value['profiles']['spatial'].__setitem__('tokens', 720),
                     lambda value: value['files'].pop('contexts.safetensors'),
                     lambda value: value.__setitem__('spatial_noise_seed', 42),
                     lambda value: value.__setitem__('spatial_noise_device', 'cuda')]
        for index, mutation in enumerate(mutations):
            root = self.packet_copy(f'bad-schema-{index}')
            self.update_manifest(root, mutation)
            with self.assertRaises(ValueError):
                inputs.load_prepared(root)

    def test_corrupt_bytes_and_rebound_pinned_file_rejected(self):
        for index, name in enumerate(('original.png', 'reference-inputs.safetensors', 'reference-metrics.json',
                                       'reference-terminal.json', 'contexts.safetensors', 'text-manifest.json')):
            root = self.packet_copy(f'bad-pin-{index}')
            file = root / name
            content = bytearray(file.read_bytes())
            content[-1] ^= 1
            file.write_bytes(content)
            with self.assertRaisesRegex(ValueError, 'Prepared file changed'):
                inputs.load_prepared(root)
            self.rebind_file(root, name)
            with self.assertRaisesRegex(ValueError, 'Pinned reference or source'):
                inputs.load_prepared(root)

    def test_wrong_tensor_keys_or_shape_rejected_before_materialization(self):
        for label, mutation in [('extra', lambda values: values.__setitem__('target', torch.zeros(1))),
                                 ('shape', lambda values: values.__setitem__('spatial_noise', torch.zeros(1)))]:
            root = self.packet_copy('bad-header-' + label)
            values = dict(self.values)
            mutation(values)
            self.rebind_tensors(root, values)
            real_open = safe_open
            calls = []
            class Spy:
                def __init__(self, *args, **kwargs):
                    self.handle = real_open(*args, **kwargs)
                def __enter__(self):
                    self.handle.__enter__()
                    return self
                def __exit__(self, *args):
                    return self.handle.__exit__(*args)
                def keys(self):
                    return self.handle.keys()
                def get_slice(self, key):
                    return self.handle.get_slice(key)
                def get_tensor(self, key):
                    calls.append(key)
                    raise AssertionError('Rejected header must not materialize tensors')
            with patch.object(inputs, 'safe_open', Spy), self.assertRaises(ValueError):
                inputs.load_prepared(root)
            self.assertEqual(calls, [])

    def test_wrong_dtype_nonfinite_and_oversize_prepared_values(self):
        for label, changed in [('dtype', self.values['spatial_noise'].to(torch.float16)),
                               ('nan', torch.full_like(self.values['spatial_noise'], float('nan')))]:
            root = self.packet_copy('bad-values-' + label)
            values = dict(self.values)
            values['spatial_noise'] = changed
            self.rebind_tensors(root, values)
            with self.assertRaisesRegex(ValueError, 'Finite CPU FP32'):
                inputs.load_prepared(root)
        oversized = self.root / 'oversized.safetensors'
        with oversized.open('wb') as handle:
            handle.truncate(inputs.MAX_INPUT_BYTES + 1)
        with self.assertRaisesRegex(ValueError, 'size bound'):
            inputs._read(oversized, {})

    def test_rebound_baseline_noise_observation_and_pixels_are_rejected(self):
        for name in ('baseline_noise', 'reference_observation', 'baseline_rgb', 'spatial_rgb'):
            root = self.packet_copy('changed-' + name)
            values = dict(self.values)
            values[name] = values[name].clone()
            values[name].reshape(-1)[0] += .125
            self.rebind_tensors(root, values)
            with self.assertRaises(ValueError):
                inputs.load_prepared(root)
        root = self.packet_copy('changed-png')
        with Image.open(root / 'spatial.png') as opened:
            pixels = np.array(opened, copy=True)
        pixels[0, 0, 0] ^= 1
        Image.fromarray(pixels).save(root / 'spatial.png')
        self.rebind_file(root, 'spatial.png')
        with self.assertRaisesRegex(ValueError, 'preprocessing pixels'):
            inputs.load_prepared(root)

    def test_spatial_saved_noise_is_authoritative_and_manifest_hash_is_external(self):
        root = self.packet_copy('new-spatial-noise')
        original_manifest_sha = digest(root / 'manifest.json')
        values = dict(self.values)
        values['spatial_noise'] = values['spatial_noise'].clone()
        values['spatial_noise'].reshape(-1)[0] += .125
        self.rebind_tensors(root, values)
        # A coherently rebound packet is valid to the low-level reader. The
        # caller must pin its manifest SHA to select one exact saved draw.
        with patch.object(inputs.torch, 'randn', side_effect=AssertionError('No noise regeneration on loading')):
            restored, _, _ = inputs.load_prepared(root)
        self.assertTrue(torch.equal(restored['spatial_noise'], values['spatial_noise']))
        self.assertNotEqual(digest(root / 'manifest.json'), original_manifest_sha)
        self.assertFalse(torch.equal(restored['spatial_noise'], self.values['spatial_noise']))


if __name__ == '__main__':
    unittest.main()
