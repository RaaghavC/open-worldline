# SPDX-License-Identifier: Apache-2.0
"""Pinned source image, official resizing and immutable CPU-prepared inputs."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from ..official_cpu.inputs import load_inputs as load_reference, PAIR_PINS, TEXT_PINS
from ..official_cpu.streaming import bounded_file, sha, tensor_sha
from .config import SPECS, SEED, spec

SOURCE_SHA = '7bdfa121eb2917b837af3ee1faae9e697751cdd3cdee0b21422c1b1f2a53e780'
SOURCE_BYTES = 184595
MAX_INPUT_BYTES = 32 * 2**20
PREPARED_NAMES = ('original.png', 'baseline.png', 'spatial.png', 'prepared.safetensors',
                  'reference-inputs.safetensors', 'reference-metrics.json',
                  'reference-terminal.json', 'contexts.safetensors', 'text-manifest.json')


# Verbatim scalar helper from Wan2.2 commit 42bf4cfaa384bc21833865abc2f9e6c0e67233dc,
# wan/utils/utils.py:202. Upstream Apache-2.0 license remains in cuda_reference/.
def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio)**0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2,
                                                 ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2


def rgb_tensor(image):
    """Match TF.to_tensor(img).sub_(0.5).div_(0.5) for an RGB byte image."""
    if image.mode != 'RGB':
        raise ValueError('RGB image required')
    pixels = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float().div_(255).sub_(0.5).div_(0.5)[None, :, None]


def preprocess(image, profile):
    selected = spec(profile)
    if image.mode != 'RGB' or image.size != (512, 288):
        raise ValueError('The pinned 512 by 288 RGB source is required')
    if profile == 'baseline':
        result = image.copy()
        record = {'resize': [512, 288], 'crop': [0, 0, 512, 288], 'resize_applied': False}
    else:
        ow, oh = best_output_size(512, 288, 32, 32, 1280 * 704)
        if (ow, oh) != (selected.width, selected.height):
            raise RuntimeError('Official size calculation differs from the declared spatial test')
        scale = max(ow / image.width, oh / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        left, top = (resized.width - ow) // 2, (resized.height - oh) // 2
        result = resized.crop((left, top, left + ow, top + oh))
        record = {'resize': list(resized.size), 'crop': [left, top, left + ow, top + oh], 'resize_applied': True}
    return result, rgb_tensor(result), record


def _shapes():
    result = {'reference_observation': SPECS['baseline'].observation_shape}
    for name, value in SPECS.items():
        result[name + '_rgb'] = (1, 3, 1, value.height, value.width)
        result[name + '_noise'] = value.latent_shape
    return result


def _read(path, shapes, max_bytes=MAX_INPUT_BYTES):
    if Path(path).stat().st_size > max_bytes:
        raise ValueError('Prepared tensor file exceeds its declared size bound')
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(shapes):
            raise ValueError('Exact prepared tensor keys required')
        for name, shape in shapes.items():
            if tuple(handle.get_slice(name).get_shape()) != tuple(shape):
                raise ValueError('Prepared tensor shape differs: ' + name)
        values = {name: handle.get_tensor(name) for name in shapes}
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in values.values()):
        raise ValueError('Finite CPU FP32 prepared values required')
    return values


def prepare(source, reference_directory, text_directory, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Fresh preparation directory required')
    if source.stat().st_size != SOURCE_BYTES or sha(source) != SOURCE_SHA:
        raise ValueError('Pinned original source PNG required')
    reference, contexts, identity = load_reference(reference_directory, text_directory)
    with Image.open(source) as opened:
        if opened.mode == 'RGBA' and opened.getchannel('A').getextrema() != (255, 255):
            raise ValueError('Unexpected nonopaque alpha in the source image')
        image = opened.convert('RGB')
    if image.size != (512, 288):
        raise ValueError('Pinned source dimensions differ')
    output.mkdir(parents=True)
    shutil.copyfile(source, output / 'original.png')
    tensors = {'reference_observation': reference['observation'].clone()}
    preprocessing = {}
    for name in SPECS:
        processed, values, preprocessing[name] = preprocess(image, name)
        processed.save(output / (name + '.png'))
        tensors[name + '_rgb'] = values
    tensors['baseline_noise'] = reference['initial_noise'].clone()
    tensors['spatial_noise'] = torch.randn(SPECS['spatial'].latent_shape,
        generator=torch.Generator(device='cpu').manual_seed(SEED), dtype=torch.float32, device='cpu')
    save_file(tensors, str(output / 'prepared.safetensors'))
    for original, renamed in [('inputs.safetensors', 'reference-inputs.safetensors'),
                              ('metrics.json', 'reference-metrics.json'), ('terminal.json', 'reference-terminal.json')]:
        shutil.copyfile(Path(reference_directory) / original, output / renamed)
    shutil.copyfile(Path(text_directory) / 'embeddings.safetensors', output / 'contexts.safetensors')
    shutil.copyfile(Path(text_directory) / 'manifest.json', output / 'text-manifest.json')
    record = {'schema': 'wan22-two-size-inputs-v1', 'status': 'prepared',
              'profiles': {name: item.record() for name, item in SPECS.items()},
              'source_image_sha256': SOURCE_SHA, 'preprocessing': preprocessing,
              'reference_identity': identity, 'spatial_noise_seed': SEED,
              'spatial_noise_device': 'cpu', 'equal_seeds_mean_equal_cross_shape_noise': False,
              'baseline_noise_regenerated': False, 'observation_encoder_executed': False,
              'actions_read': False, 'future_targets_read': False,
              'torch': torch.__version__, 'tensor_sha256': {k: tensor_sha(v) for k, v in tensors.items()},
              'files': {name: {'sha256': sha(output / name), 'bytes': (output / name).stat().st_size}
                        for name in PREPARED_NAMES}}
    (output / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def load_prepared(directory):
    root = Path(directory)
    manifest = json.loads(bounded_file(root, 'manifest.json').read_text())
    if (manifest.get('schema') != 'wan22-two-size-inputs-v1' or manifest.get('status') != 'prepared'
            or manifest.get('profiles') != {k: v.record() for k, v in SPECS.items()}
            or manifest.get('source_image_sha256') != SOURCE_SHA
            or set(manifest.get('files', {})) != set(PREPARED_NAMES)):
        raise ValueError('Exact two-size preparation contract required')
    for name, record in manifest['files'].items():
        path = bounded_file(root, name)
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Prepared file changed: ' + name)
    pins = {'original.png': SOURCE_SHA, 'contexts.safetensors': TEXT_PINS['embeddings.safetensors'],
            'text-manifest.json': TEXT_PINS['manifest.json'],
            **{'reference-' + name: digest for name, digest in PAIR_PINS.items()}}
    for name, digest in pins.items():
        if sha(bounded_file(root, name)) != digest:
            raise ValueError('Pinned reference or source file differs: ' + name)
    tensors = _read(root / 'prepared.safetensors', _shapes())
    # Original mixed int64/FP32 file is validated by its already pinned bytes.
    from ..official_cpu.inputs import read_exact, SHAPES
    reference = read_exact(root / 'reference-inputs.safetensors', SHAPES)
    if not torch.equal(reference['initial_noise'], tensors['baseline_noise']) or not torch.equal(reference['observation'], tensors['reference_observation']):
        raise ValueError('Baseline noise or cached observation differs from retained reference')
    if manifest.get('tensor_sha256') != {k: tensor_sha(v) for k, v in tensors.items()}:
        raise ValueError('Prepared tensor identity differs')
    with Image.open(root / 'original.png') as opened:
        original = opened.convert('RGB')
    for name in SPECS:
        expected_image, expected_rgb, record = preprocess(original, name)
        with Image.open(root / (name + '.png')) as retained:
            if retained.mode != 'RGB' or retained.size != expected_image.size or retained.tobytes() != expected_image.tobytes():
                raise ValueError('Saved preprocessing pixels differ')
        if record != manifest['preprocessing'][name] or not torch.equal(expected_rgb, tensors[name + '_rgb']):
            raise ValueError('Saved image tensor differs from source preprocessing')
    if manifest.get('spatial_noise_seed') != SEED or manifest.get('spatial_noise_device') != 'cpu':
        raise ValueError('Declared noise preparation differs')
    contexts = _read(root / 'contexts.safetensors', {'atrium': (25, 4096), 'native_negative': (126, 4096)})
    return tensors, contexts, manifest
