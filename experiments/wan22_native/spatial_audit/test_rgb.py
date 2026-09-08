# SPDX-License-Identifier: Apache-2.0
"""Temporary synthetic full-size files only; no Torch, models or CUDA."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from . import rgb


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<f4').tobytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def write_tensor(path, values):
    """Independent minimal F32 fixture writer, not the audited reader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {}; offset = 0
    for name, value in values.items():
        count = value.size * 4
        header[name] = {'dtype': 'F32', 'shape': list(value.shape), 'data_offsets': [offset, offset + count]}
        offset += count
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with path.open('wb') as stream:
        stream.write(struct.pack('<Q', len(raw))); stream.write(raw)
        for value in values.values():
            stream.write(np.ascontiguousarray(value, dtype='<f4').tobytes())


def pixels(value):
    return np.rint((value[0, :, 0].transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)


def seal(root, stage):
    result = root / stage / 'result'
    path = result / 'metrics.json'
    report = json.loads(path.read_text())
    report['output_sha256'] = {p.relative_to(result).as_posix(): sha(p)
                              for p in result.rglob('*') if p.is_file() and p != path}
    write_json(path, report)
    parent = json.loads((root / 'metrics.json').read_text())
    parent['child_reports'] = {stage + '/result/metrics.json': sha(path)}
    write_json(root / 'metrics.json', parent)


def base_reports(root, mode, profile, packet_sha=None):
    stage = 'codec' if mode == 'codec' else 'decode'
    common = {'status': 'passed', 'mode': mode, 'profile': profile,
              'input_manifest_sha256': packet_sha}
    parent = {**common, 'model_execution': True}
    report = {**common, 'stage': stage, 'finite_outputs': True, 'decoder_cache_clear': True}
    write_json(root / 'metrics.json', parent)
    write_json(root / stage / 'result/metrics.json', report)
    return report


def raw_frames(result, folder, height, width, purpose, make_frame, include_pngs=False):
    records = []; sequence = []
    (result / folder).mkdir(parents=True)
    if include_pngs:
        (result / 'frames').mkdir()
    for number in range(17):
        value = make_frame(number)
        name = f'{number:04d}.safetensors'
        path = result / folder / name
        write_tensor(path, {'rgb': value})
        records.append({'index': number, 'file': name, 'bytes': path.stat().st_size,
                        'sha256': sha(path), 'tensor_sha256': tensor_sha(value)})
        if include_pngs:
            frame = Image.fromarray(pixels(value))
            frame.save(result / 'frames' / f'{number:04d}.png')
            sequence.append(frame)
    write_json(result / folder / 'index.json', {'schema': 'wan22-rgb-frames-v1', 'purpose': purpose,
        'shape': [1, 3, 17, height, width], 'dtype': 'float32', 'range': [-1, 1], 'frames': records})
    return sequence


class RGBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.packet = cls.root / 'packet'; cls.packet.mkdir()
        values = {'reference_observation': np.full((1, 48, 1, 18, 32), .25, np.float32)}
        for name, (height, width) in rgb.SIZES.items():
            data = np.empty((height, width, 3), np.uint8)
            data[:, :, 0] = np.arange(width, dtype=np.uint16)[None, :] % 256
            data[:, :, 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
            data[:, :, 2] = 128
            Image.fromarray(data).save(cls.packet / (name + '.png'))
            mapped = ((data.astype(np.float32) / np.float32(255)) - np.float32(.5)) / np.float32(.5)
            values[name + '_rgb'] = np.ascontiguousarray(mapped.transpose(2, 0, 1))[None, :, None]
            values[name + '_noise'] = np.zeros((48, 5, height // 16, width // 16), np.float32)
        write_tensor(cls.packet / 'prepared.safetensors', values)
        manifest = {'schema': 'wan22-two-size-inputs-v1', 'status': 'prepared',
                    'tensor_sha256': {k: tensor_sha(v) for k, v in values.items()},
                    'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in cls.packet.iterdir()}}
        write_json(cls.packet / 'manifest.json', manifest)
        cls.codec = cls.root / 'codec-run'
        report = base_reports(cls.codec, 'codec', 'both', sha(cls.packet / 'manifest.json'))
        result = cls.codec / 'codec/result'; report['profiles'] = {}; observations = {}
        for name, (height, width) in rgb.SIZES.items():
            observed = np.full((1, 48, 1, height // 16, width // 16), .255 if name == 'baseline' else .125, np.float32)
            observations[name] = observed
            write_tensor(result / name / 'observation.safetensors', {'observation': observed})
            write_tensor(result / name / 'timing-proxy.safetensors', {'latent': np.repeat(observed[0], 5, axis=1)})
            reconstruction = np.clip(values[name + '_rgb'] + np.float32(.005), -1, 1)
            raw_frames(result, name + '/proxy-rgb', height, width, 'repeated_observation_codec_timing_proxy',
                       lambda number, data=reconstruction: data)
            Image.fromarray(pixels(reconstruction)).save(result / name / 'reconstructed-initial.png')
            delta = (reconstruction.astype(np.float64) - values[name + '_rgb'].astype(np.float64)) / 2
            mse = float(np.mean(delta * delta))
            row = {'observation_tensor_sha256': tensor_sha(observed),
                   'timing_proxy': 'Five repeated observation latents; no denoiser or future target',
                   'raw_rgb_index': name + '/proxy-rgb/index.json',
                   'initial_frame_mse_0_1': mse, 'initial_frame_psnr_db': -10 * math.log10(mse),
                   'exact_initial_rgb_tensor': False}
            if name == 'baseline':
                delta = observed.astype(np.float64) - values['reference_observation'].astype(np.float64)
                rmse = float(np.sqrt(np.mean(delta * delta)))
                row['retained_mps_observation_difference'] = {'equal': False, 'rmse': rmse,
                    'max_absolute_difference': float(np.max(np.abs(delta))), 'reference_rms': .25,
                    'relative_rmse': rmse / .25, 'normalizer': 'RMS of retained reference values; no equivalence threshold'}
            report['profiles'][name] = row
        write_tensor(result / 'observations.safetensors', observations)
        write_json(result / 'metrics.json', report); seal(cls.codec, 'codec')
        del values, observations
        cls.clips = {}
        for name, (height, width) in rgb.SIZES.items():
            root = cls.root / (name + '-clip'); cls.clips[name] = root
            report = base_reports(root, 'clip', name)
            result = root / 'decode/result'
            # Distinct baseline frames and identical spatial frames exercise
            # both a 17-frame GIF and legal identical-frame timing merging.
            def frame(number):
                return np.full((1, 3, 1, height, width), -1 + number / 8 if name == 'baseline' else .125, np.float32)
            sequence = raw_frames(result, 'rgb', height, width, 'generated_clip', frame, True)
            contact = Image.new('RGB', (2 * width, 5 * (height + 28)), (245, 242, 234))
            draw = ImageDraw.Draw(contact)
            for slot, number in enumerate(rgb.CONTACT):
                x, y = slot % 2 * width, slot // 2 * (height + 28)
                draw.text((x + 8, y + 6), ('Conditioned reconstruction' if number == 0 else 'Generated future') + f' {number}', fill=(20, 20, 20))
                contact.paste(sequence[number], (x, y + 28))
            contact.save(result / 'comparison.png'); contact.close()
            sequence[0].save(result / 'preview.gif', save_all=True, append_images=sequence[1:], duration=120, loop=0, optimize=False)
            with Image.open(result / 'preview.gif') as preview:
                durations = []
                for number in range(preview.n_frames):
                    preview.seek(number); durations.append(preview.info['duration'])
            report['images'] = {'height': height, 'width': width, 'frames': 17,
                'conditioned_initial_frames': 1, 'new_future_frames': 16,
                'contact_frame_indices': list(rgb.CONTACT), 'contact_images_resized': False,
                'preview_requested_frame_duration_ms': 120, 'preview_requested_playback_fps': 1000 / 120,
                'preview_encoded_frames': len(durations), 'preview_encoded_frame_durations_ms': durations,
                'preview_encoded_duration_ms': sum(durations)}
            write_json(result / 'metrics.json', report); seal(root, 'decode')
            for image in sequence:
                image.close()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @contextmanager
    def changed(self, root, stage, *relative):
        result = root / stage / 'result'
        paths = [root / 'metrics.json', result / 'metrics.json', *(result / name for name in relative)]
        originals = {path: path.read_bytes() for path in paths}
        try:
            yield result
        finally:
            for path, content in originals.items():
                path.write_bytes(content)

    def edit_report(self, root, stage, change):
        path = root / stage / 'result/metrics.json'
        report = json.loads(path.read_text()); change(report); write_json(path, report); seal(root, stage)

    def test_codec_both_full_sizes_and_independent_scores(self):
        result = rgb.audit_codec(self.codec, self.packet)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['generated_frames'], 0)
        self.assertEqual(set(result['profiles']), {'baseline', 'spatial'})
        for profile in result['profiles'].values():
            self.assertEqual(profile['raw_rgb']['frames'], 17)
            self.assertTrue(profile['proxy_exactly_repeats_observation'])
            self.assertTrue(math.isfinite(profile['initial_frame_psnr_db']))

    def test_clip_both_sizes_pixels_and_merged_gif_timing(self):
        for name, root in self.clips.items():
            result = rgb.audit_decoded_clip(root, name)
            self.assertEqual(result['png_frames_exact'], 17)
            self.assertEqual(result['contact_frame_regions_exact'], 10)
            self.assertEqual(result['gif_duration_ms'], 2040)
            self.assertFalse(result['gif_color_equality_checked'])
            self.assertEqual(result['gif_encoded_frames'], 17 if name == 'baseline' else 1)

    def test_changed_png_and_contact_pixels_even_after_file_rebinding(self):
        root = self.clips['baseline']
        for target, coordinate in [('frames/0001.png', (0, 0)), ('comparison.png', (0, 28))]:
            with self.changed(root, 'decode', target) as result:
                path = result / target
                with Image.open(path) as opened:
                    data = np.array(opened, copy=True)
                x, y = coordinate; data[y, x, 0] ^= 1
                Image.fromarray(data).save(path); seal(root, 'decode')
                with self.assertRaisesRegex(ValueError, 'pixels differ'):
                    rgb.audit_decoded_clip(root, 'baseline')

    def test_hash_index_name_shape_and_tensor_hash_corruption(self):
        root = self.clips['baseline']
        path = 'rgb/0000.safetensors'
        with self.changed(root, 'decode', path) as result:
            content = bytearray((result / path).read_bytes()); content[-1] ^= 1
            (result / path).write_bytes(content)
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                rgb.audit_decoded_clip(root, 'baseline')
        changes = [lambda value: value['frames'][0].update(file='../outside'),
                   lambda value: value['frames'][0].update(index=1),
                   lambda value: value.update(shape=[1, 3, 16, 288, 512]),
                   lambda value: value['frames'][0].update(tensor_sha256='0' * 64)]
        for change in changes:
            with self.changed(root, 'decode', 'rgb/index.json') as result:
                index = json.loads((result / 'rgb/index.json').read_text()); change(index)
                write_json(result / 'rgb/index.json', index); seal(root, 'decode')
                with self.assertRaises(ValueError):
                    rgb.audit_decoded_clip(root, 'baseline')

    def test_rebound_nonfinite_range_and_dtype_tensors_rejected(self):
        root = self.clips['baseline']
        for bad in ('range', 'nan', 'dtype'):
            with self.changed(root, 'decode', 'rgb/0000.safetensors', 'rgb/index.json') as result:
                path = result / 'rgb/0000.safetensors'
                value = np.zeros((1, 3, 1, 288, 512), np.float32)
                value.reshape(-1)[0] = 2 if bad == 'range' else float('nan') if bad == 'nan' else 0
                write_tensor(path, {'rgb': value})
                if bad == 'dtype':
                    content = path.read_bytes().replace(b'"F32"', b'"I32"', 1); path.write_bytes(content)
                index = json.loads((result / 'rgb/index.json').read_text())
                index['frames'][0].update(bytes=path.stat().st_size, sha256=sha(path), tensor_sha256=tensor_sha(value))
                write_json(result / 'rgb/index.json', index); seal(root, 'decode')
                with self.assertRaises(ValueError):
                    rgb.audit_decoded_clip(root, 'baseline')

    def test_codec_psnr_mse_reference_denominator_and_exact_flag_corruption(self):
        changes = [lambda row: row.update(initial_frame_psnr_db=10),
                   lambda row: row.update(initial_frame_mse_0_1=0),
                   lambda row: row.update(exact_initial_rgb_tensor=True),
                   lambda row: row['retained_mps_observation_difference'].update(relative_rmse=1),
                   lambda row: row['retained_mps_observation_difference'].update(reference_rms=1)]
        for change in changes:
            with self.changed(self.codec, 'codec'):
                self.edit_report(self.codec, 'codec', lambda value: change(value['profiles']['baseline']))
                with self.assertRaises(ValueError):
                    rgb.audit_codec(self.codec, self.packet)

    def test_codec_proxy_and_initial_png_corruption(self):
        target = 'baseline/timing-proxy.safetensors'
        with self.changed(self.codec, 'codec', target) as result:
            value = np.full((48, 5, 18, 32), .255, np.float32); value[0, 4, 0, 0] += .1
            write_tensor(result / target, {'latent': value}); seal(self.codec, 'codec')
            with self.assertRaisesRegex(ValueError, 'five repeated'):
                rgb.audit_codec(self.codec, self.packet)
        target = 'baseline/reconstructed-initial.png'
        with self.changed(self.codec, 'codec', target) as result:
            with Image.open(result / target) as opened:
                data = np.array(opened, copy=True)
            data[0, 0, 0] ^= 1; Image.fromarray(data).save(result / target); seal(self.codec, 'codec')
            with self.assertRaisesRegex(ValueError, 'PNG pixels'):
                rgb.audit_codec(self.codec, self.packet)

    def test_contact_label_and_gif_timing_corruption(self):
        root = self.clips['baseline']
        with self.changed(root, 'decode', 'comparison.png') as result:
            with Image.open(result / 'comparison.png') as opened:
                image = opened.copy()
            ImageDraw.Draw(image).rectangle((0, 0, 511, 27), fill=(245, 242, 234))
            image.save(result / 'comparison.png'); image.close(); seal(root, 'decode')
            with self.assertRaisesRegex(ValueError, 'label ink'):
                rgb.audit_decoded_clip(root, 'baseline')
        with self.changed(root, 'decode'):
            self.edit_report(root, 'decode', lambda value: value['images'].update(preview_encoded_duration_ms=2125))
            with self.assertRaisesRegex(ValueError, 'GIF timing'):
                rgb.audit_decoded_clip(root, 'baseline')
        with self.changed(root, 'decode', 'preview.gif') as result:
            with Image.open(result / 'preview.gif') as opened:
                frames = []
                for index in range(opened.n_frames):
                    opened.seek(index); frames.append(opened.copy())
            frames[0].save(result / 'preview.gif', save_all=True, append_images=frames[1:], duration=130, loop=0)
            for frame in frames: frame.close()
            seal(root, 'decode')
            with self.assertRaisesRegex(ValueError, '120 ms'):
                rgb.audit_decoded_clip(root, 'baseline')

    def test_tensor_bounds_and_header_overlap_before_payload_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'small.safetensors'
            write_tensor(path, {'x': np.zeros((2,), np.float32), 'y': np.ones((2,), np.float32)})
            content = path.read_bytes(); length = struct.unpack('<Q', content[:8])[0]
            header = json.loads(content[8:8 + length]); header['y']['data_offsets'] = [0, 8]
            encoded = json.dumps(header).encode(); encoded += b' ' * (-len(encoded) % 8)
            path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + content[8 + length:])
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                rgb._TensorFile(path, {'x': (2,), 'y': (2,)})
            huge = root / 'huge.safetensors'
            with huge.open('wb') as stream: stream.truncate(rgb.FRAME_LIMIT + 1)
            with self.assertRaisesRegex(ValueError, 'Bounded'):
                rgb._TensorFile(huge, {'x': (2,)})


if __name__ == '__main__':
    unittest.main()
