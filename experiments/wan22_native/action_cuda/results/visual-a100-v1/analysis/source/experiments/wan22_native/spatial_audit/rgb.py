# SPDX-License-Identifier: Apache-2.0
"""Independent, frame-at-a-time NumPy/Pillow audit of retained spatial RGB.

No Torch, VAE, denoiser or production output helper is imported. Binary and
pixel checks establish artifact identity, not visual quality or model accuracy.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct

import numpy as np
from PIL import Image


SIZES = {'baseline': (288, 512), 'spatial': (704, 1248)}
CONTACT = (0, 1, 2, 4, 6, 8, 10, 12, 14, 16)
FRAME_LIMIT = 16 * 2**20
JSON_LIMIT = 4 * 2**20
IMAGE_LIMIT = 64 * 2**20
PREPARED_SHAPES = {'reference_observation': (1, 48, 1, 18, 32)}
for _name, (_height, _width) in SIZES.items():
    PREPARED_SHAPES[_name + '_rgb'] = (1, 3, 1, _height, _width)
    PREPARED_SHAPES[_name + '_noise'] = (48, 5, _height // 16, _width // 16)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(char in '0123456789abcdef' for char in value), 'Invalid SHA256 value')
    return value


def _file(root, relative, maximum=None):
    _require(isinstance(relative, str) and relative and '\\' not in relative
             and not PurePosixPath(relative).is_absolute()
             and str(PurePosixPath(relative)) == relative
             and all(part not in ('', '.', '..') for part in relative.split('/')),
             'Canonical relative artifact path required')
    root = Path(root)
    _require(root.is_dir() and not root.is_symlink(), 'Regular artifact directory required')
    path = root
    for part in relative.split('/'):
        path = path / part
        _require(not path.is_symlink(), 'Symlink artifacts are not accepted')
    _require(path.is_file(), 'Missing artifact: ' + relative)
    if maximum is not None:
        _require(path.stat().st_size <= maximum, 'Artifact exceeds byte bound: ' + relative)
    return path


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def number(value):
        result = float(value)
        _require(math.isfinite(result), 'Nonfinite JSON number')
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=invalid)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid JSON evidence') from error


def _json(path):
    _require(path.stat().st_size <= JSON_LIMIT, 'JSON exceeds bound')
    result = _parse(path.read_bytes())
    _require(isinstance(result, dict), 'JSON object required')
    return result


class _TensorFile:
    """Validate a bounded FP32 safetensors header, then read one named payload."""
    def __init__(self, path, shapes, maximum=FRAME_LIMIT):
        self.path = Path(path)
        size = self.path.stat().st_size
        _require(not self.path.is_symlink() and 10 <= size <= maximum, 'Bounded regular tensor file required')
        with self.path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            _require(2 <= length <= 65536 and 8 + length <= size, 'Bounded safetensors header required')
            header = _parse(stream.read(length))
        _require(isinstance(header, dict), 'Tensor header object required')
        metadata = header.pop('__metadata__', {})
        _require(isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str)
                  for k, v in metadata.items()), 'Invalid tensor metadata')
        _require(set(header) == set(shapes), 'Exact tensor key set required')
        self.offset = 8 + length
        self.header = header
        intervals = []
        for name, shape in shapes.items():
            row = header[name]
            _require(isinstance(row, dict) and row.get('dtype') == 'F32'
                     and row.get('shape') == list(shape), 'Exact FP32 tensor shape required: ' + name)
            _require(all(type(item) is int and item > 0 for item in row['shape']), 'Positive tensor dimensions required')
            offsets = row.get('data_offsets')
            _require(isinstance(offsets, list) and len(offsets) == 2
                     and all(type(item) is int for item in offsets), 'Integer tensor offsets required')
            start, end = offsets
            _require(0 <= start < end <= size - self.offset
                     and end - start == math.prod(shape) * 4, 'Tensor byte range differs from shape')
            intervals.append((start, end))
        end = 0
        for start, stop in sorted(intervals):
            _require(start == end, 'Overlapping, missing or unordered tensor storage')
            end = stop
        _require(end == size - self.offset, 'Trailing tensor payload bytes')

    def read(self, name):
        row = self.header[name]
        start, end = row['data_offsets']
        with self.path.open('rb') as stream:
            stream.seek(self.offset + start)
            payload = stream.read(end - start)
        _require(len(payload) == end - start, 'Truncated tensor payload')
        value = np.frombuffer(payload, dtype='<f4').reshape(row['shape'])
        _require(np.isfinite(value).all(), 'Nonfinite FP32 tensor values')
        return value, hashlib.sha256(payload).hexdigest()


def _artifact(root, relative, report, maximum=None):
    path = _file(root, relative, maximum)
    outputs = report.get('output_sha256')
    _require(isinstance(outputs, dict) and relative in outputs, 'Worker did not bind artifact: ' + relative)
    _require(_sha(path) == _digest(outputs[relative]), 'Worker artifact hash differs: ' + relative)
    return path


def _pixels(value):
    return np.rint((value[0, :, 0].transpose(1, 2, 0) + np.float32(1))
                   * np.float32(127.5)).clip(0, 255).astype(np.uint8)


def _png(path, height, width):
    with Image.open(path) as opened:
        _require(opened.format == 'PNG' and opened.mode == 'RGB'
                 and opened.size == (width, height), 'Exact RGB PNG dimensions required')
        return np.array(opened, copy=True)


def _close(actual, expected, label):
    if expected is None:
        _require(actual is None, label + ' must be null')
        return
    _require(type(actual) in (int, float), 'Finite numeric ' + label + ' required')
    try:
        actual = float(actual)
    except OverflowError as error:
        raise ValueError('Unbounded ' + label) from error
    _require(math.isfinite(actual), 'Nonfinite ' + label)
    # Only FP64 reduction-order tolerance, not a quality or model-equivalence
    # threshold. Exact zero is checked separately rather than absorbed here.
    _require((actual == 0) == (expected == 0)
             and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-14),
             'Recomputed ' + label + ' differs')


def _worker(root, stage, mode, profile):
    root = Path(root)
    parent = _json(_file(root, 'metrics.json'))
    result = root / stage / 'result'
    metrics_path = _file(result, 'metrics.json')
    report = _json(metrics_path)
    for record in (parent, report):
        _require(record.get('status') == 'passed' and record.get('mode') == mode
                 and record.get('profile') == profile, 'Completed matching run required')
    _require(parent.get('model_execution') is True and report.get('stage') == stage
             and report.get('finite_outputs') is True and report.get('decoder_cache_clear') is True,
             'Completed finite codec output and cleared caches required')
    _require(parent.get('child_reports', {}).get(stage + '/result/metrics.json') == _sha(metrics_path),
             'Parent child-report hash differs')
    return root, result, parent, report


def _frames(result, relative, profile, purpose, report, inspect):
    height, width = SIZES[profile]
    index_path = _artifact(result, relative + '/index.json', report, JSON_LIMIT)
    index = _json(index_path)
    _require(index.get('schema') == 'wan22-rgb-frames-v1' and index.get('purpose') == purpose
             and index.get('shape') == [1, 3, 17, height, width]
             and index.get('dtype') == 'float32' and index.get('range') == [-1, 1],
             'Exact raw RGB index contract required')
    rows = index.get('frames')
    _require(isinstance(rows, list) and len(rows) == 17, 'Exactly 17 raw RGB frames required')
    expected_files = {'index.json'} | {f'{i:04d}.safetensors' for i in range(17)}
    _require({path.name for path in (result / relative).iterdir()} == expected_files,
             'Exact raw RGB shard file set required')
    records = []
    for number, row in enumerate(rows):
        name = f'{number:04d}.safetensors'
        _require(isinstance(row, dict) and type(row.get('index')) is int
                 and row['index'] == number and row.get('file') == name, 'Raw RGB order or file name differs')
        path = _artifact(result, relative + '/' + name, report, FRAME_LIMIT)
        file_sha = _sha(path)
        _require(type(row.get('bytes')) is int and row['bytes'] == path.stat().st_size
                 and row.get('sha256') == file_sha, 'Raw RGB byte count or file hash differs')
        value, tensor_sha = _TensorFile(path, {'rgb': (1, 3, 1, height, width)}).read('rgb')
        _require(row.get('tensor_sha256') == tensor_sha, 'Raw RGB tensor hash differs')
        _require(value.min() >= -1 and value.max() <= 1, 'Raw RGB must already be clamped [-1,1]')
        inspect(number, value)
        records.append({'index': number, 'file_sha256': file_sha, 'tensor_sha256': tensor_sha,
                        'pixel_sha256': hashlib.sha256(_pixels(value).tobytes()).hexdigest()})
        del value
    return {'index_sha256': _sha(index_path), 'frames': 17, 'frame_records': records,
            'retention': 'One raw FP32 frame payload read at a time; no 17-frame RGB stack'}


def audit_codec(run_root, packet_root):
    """Recompute both observation/proxy identities and first-image codec scores."""
    _, result, parent, report = _worker(run_root, 'codec', 'codec', 'both')
    packet = Path(packet_root)
    manifest_path = _file(packet, 'manifest.json', JSON_LIMIT)
    manifest = _json(manifest_path)
    _require(manifest.get('schema') == 'wan22-two-size-inputs-v1' and manifest.get('status') == 'prepared'
             and parent.get('input_manifest_sha256') == _sha(manifest_path)
             and report.get('input_manifest_sha256') == _sha(manifest_path), 'Prepared packet identity differs')
    def prepared_file(name):
        path = _file(packet, name, 32 * 2**20)
        row = manifest.get('files', {}).get(name)
        _require(isinstance(row, dict) and type(row.get('bytes')) is int
                 and row['bytes'] == path.stat().st_size and row.get('sha256') == _sha(path),
                 'Prepared file identity differs: ' + name)
        return path
    prepared = _TensorFile(prepared_file('prepared.safetensors'), PREPARED_SHAPES, 32 * 2**20)
    shapes = {name: (1, 48, 1, h // 16, w // 16) for name, (h, w) in SIZES.items()}
    combined = _TensorFile(_artifact(result, 'observations.safetensors', report, FRAME_LIMIT), shapes)
    _require(isinstance(report.get('profiles'), dict) and set(report['profiles']) == set(SIZES), 'Both codec profiles required')
    findings = {}
    for profile, (height, width) in SIZES.items():
        row = report['profiles'][profile]
        _require(isinstance(row, dict) and row.get('raw_rgb_index') == profile + '/proxy-rgb/index.json'
                 and row.get('timing_proxy') == 'Five repeated observation latents; no denoiser or future target',
                 'Explicit repeated-observation timing proxy required')
        observed, observation_sha = combined.read(profile)
        separate, separate_sha = _TensorFile(_artifact(result, profile + '/observation.safetensors', report),
                                            {'observation': shapes[profile]}).read('observation')
        _require(observation_sha == separate_sha == row.get('observation_tensor_sha256')
                 and np.array_equal(observed, separate), 'Saved observation copies differ')
        proxy, proxy_sha = _TensorFile(_artifact(result, profile + '/timing-proxy.safetensors', report),
             {'latent': (48, 5, height // 16, width // 16)}).read('latent')
        _require(all(np.array_equal(proxy[:, i:i + 1], observed[0]) for i in range(5)),
                 'Timing proxy is not exactly five repeated observation latents')
        expected, rgb_sha = prepared.read(profile + '_rgb')
        _require(manifest.get('tensor_sha256', {}).get(profile + '_rgb') == rgb_sha, 'Prepared RGB tensor identity differs')
        pixels = _png(prepared_file(profile + '.png'), height, width)
        normalized = (pixels.astype(np.float32) / np.float32(255) - np.float32(.5)) / np.float32(.5)
        _require(np.array_equal(expected, normalized.transpose(2, 0, 1)[None, :, None]), 'Prepared PNG and FP32 RGB differ')
        metrics = {}
        def inspect(number, value):
            if number != 0:
                return
            png = _artifact(result, profile + '/reconstructed-initial.png', report, IMAGE_LIMIT)
            _require(np.array_equal(_png(png, height, width), _pixels(value)), 'Initial reconstruction PNG pixels differ')
            delta = (value.astype(np.float64) - expected.astype(np.float64)) / 2
            mse = float(np.mean(np.square(delta), dtype=np.float64))
            psnr = -10 * math.log10(mse) if mse > 0 else None
            _close(row.get('initial_frame_mse_0_1'), mse, 'initial frame MSE')
            _close(row.get('initial_frame_psnr_db'), psnr, 'initial frame PSNR')
            _require(row.get('exact_initial_rgb_tensor') is (mse == 0), 'Exact initial RGB flag differs')
            metrics.update(initial_frame_mse_0_1=mse, initial_frame_psnr_db=psnr,
                           exact_initial_rgb_tensor=mse == 0, initial_png_pixels_exact=True)
        raw = _frames(result, profile + '/proxy-rgb', profile,
                      'repeated_observation_codec_timing_proxy', report, inspect)
        if profile == 'baseline':
            reference, reference_sha = prepared.read('reference_observation')
            _require(manifest.get('tensor_sha256', {}).get('reference_observation') == reference_sha,
                     'Prepared reference observation identity differs')
            delta = observed.astype(np.float64) - reference.astype(np.float64)
            rmse = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))
            denominator = float(np.sqrt(np.mean(np.square(reference.astype(np.float64)), dtype=np.float64)))
            difference = {'rmse': rmse, 'max_absolute_difference': float(np.max(np.abs(delta))),
                          'reference_rms': denominator, 'relative_rmse': rmse / denominator if denominator else None}
            claimed = row.get('retained_mps_observation_difference')
            _require(isinstance(claimed, dict)
                     and claimed.get('equal') is bool(np.array_equal(observed, reference))
                     and claimed.get('normalizer') == 'RMS of retained reference values; no equivalence threshold',
                     'Observation comparison contract differs')
            for key, number in difference.items():
                _close(claimed.get(key), number, 'observation ' + key)
            metrics['retained_mps_observation_difference'] = {**difference, 'equal': bool(np.array_equal(observed, reference))}
        findings[profile] = {'observation_sha256': observation_sha, 'timing_proxy_sha256': proxy_sha,
                             'proxy_exactly_repeats_observation': True, 'raw_rgb': raw, **metrics}
        del observed, separate, proxy, expected, pixels, normalized
    return {'status': 'passed', 'audit': 'codec', 'profiles': findings,
            'input_manifest_sha256': _sha(manifest_path), 'generated_frames': 0,
            'meaning': 'Timing-proxy identity and retained first-frame codec measurements; no future-image fidelity or visual-quality claim',
            'numeric_comparison': 'FP64 recomputation, relative tolerance 1e-10 / absolute 1e-14 for reduction order; exact zeros remain exact'}


def audit_decoded_clip(run_root, profile):
    """Verify all raw frames, exact PNG/contact pixels and GIF timing metadata."""
    _require(isinstance(profile, str) and profile in SIZES, 'Explicit baseline or spatial profile required')
    _, result, _, report = _worker(run_root, 'decode', 'clip', profile)
    height, width = SIZES[profile]
    images = report.get('images')
    _require(isinstance(images, dict) and images.get('height') == height and images.get('width') == width
             and images.get('frames') == 17 and images.get('conditioned_initial_frames') == 1
             and images.get('new_future_frames') == 16 and images.get('contact_frame_indices') == list(CONTACT)
             and images.get('contact_images_resized') is False, 'Declared clip presentation differs')
    _require({p.name for p in (result / 'frames').iterdir()} == {f'{i:04d}.png' for i in range(17)},
             'All 17 authoritative PNG frames required')
    contact_path = _artifact(result, 'comparison.png', report, IMAGE_LIMIT)
    with Image.open(contact_path) as opened:
        _require(opened.format == 'PNG' and opened.mode == 'RGB'
                 and opened.size == (2 * width, 5 * (height + 28)), 'Full-size contact-sheet layout differs')
        contact = opened.copy()
    def inspect(number, value):
        expected = _pixels(value)
        path = _artifact(result, f'frames/{number:04d}.png', report, IMAGE_LIMIT)
        _require(np.array_equal(_png(path, height, width), expected), 'Rounded PNG pixels differ')
        if number in CONTACT:
            slot = CONTACT.index(number)
            x, y = slot % 2 * width, slot // 2 * (height + 28)
            crop = np.asarray(contact.crop((x, y + 28, x + width, y + 28 + height)))
            _require(np.array_equal(crop, expected), 'Unresized contact-sheet frame pixels differ')
            label = np.asarray(contact.crop((x + 8, y + 6, x + width, y + 28)))
            # Antialiasing may contain no exact (20,20,20) pixel. This checks
            # dark header ink only; it is not a font/glyph byte comparison.
            _require(np.any(np.max(label, axis=-1) < 128), 'Expected dark label ink is missing from contact header')
    try:
        raw = _frames(result, 'rgb', profile, 'generated_clip', report, inspect)
    finally:
        contact.close()
    preview_path = _artifact(result, 'preview.gif', report, IMAGE_LIMIT)
    with Image.open(preview_path) as preview:
        _require(preview.format == 'GIF' and preview.size == (width, height)
                 and 1 <= preview.n_frames <= 17 and preview.info.get('loop') == 0, 'GIF preview layout differs')
        durations = []
        for frame in range(preview.n_frames):
            preview.seek(frame)
            value = preview.info.get('duration')
            _require(type(value) is int and value > 0 and value % 120 == 0, 'GIF must retain 120 ms timing units')
            durations.append(value)
    _require(sum(durations) == 17 * 120
             and images.get('preview_requested_frame_duration_ms') == 120
             and images.get('preview_encoded_frames') == len(durations)
             and images.get('preview_encoded_frame_durations_ms') == durations
             and images.get('preview_encoded_duration_ms') == sum(durations), 'GIF timing report differs from encoded metadata')
    _close(images.get('preview_requested_playback_fps'), 1000 / 120, 'requested preview FPS')
    return {'status': 'passed', 'audit': 'decoded_clip', 'profile': profile, 'raw_rgb': raw,
            'png_frames_exact': 17, 'contact_frame_regions_exact': len(CONTACT),
            'contact_label_check': 'Dark ink in each declared header band; glyph spelling and font pixels are not compared',
            'conditioned_initial_frames': 1, 'new_future_frames': 16,
            'gif_encoded_frames': len(durations), 'gif_encoded_frame_durations_ms': durations,
            'gif_duration_ms': sum(durations), 'requested_preview_playback_fps': 1000 / 120,
            'gif_color_equality_checked': False,
            'meaning': 'Retained raw/pixel identity and playback timing; no visual-quality or generation-throughput claim'}
