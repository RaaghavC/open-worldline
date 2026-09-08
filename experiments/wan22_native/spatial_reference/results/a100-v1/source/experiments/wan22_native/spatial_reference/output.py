# SPDX-License-Identifier: Apache-2.0
"""Bounded tensor reads and lossless frame-sized RGB retention."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from ..official_cpu.streaming import sha, tensor_sha


def read_tensors(path, shapes, maximum=16 * 2**20):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('A bounded regular tensor file is required')
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(shapes):
            raise ValueError('Exact tensor keys required')
        for name, shape in shapes.items():
            if tuple(handle.get_slice(name).get_shape()) != tuple(shape):
                raise ValueError('Tensor header shape differs: ' + name)
        values = {name: handle.get_tensor(name) for name in shapes}
    if any(x.dtype != torch.float32 or not torch.isfinite(x).all() for x in values.values()):
        raise ValueError('Finite CPU FP32 tensors required')
    return values


def save_rgb(video, out, selected, *, purpose):
    """Retain every FP32 sample without producing a >100 MB Git file."""
    if (video.device.type != 'cpu' or video.dtype != torch.float32 or video.requires_grad
            or tuple(video.shape) != (1, 3, 17, selected.height, selected.width)
            or not torch.isfinite(video).all() or video.min() < -1 or video.max() > 1):
        raise ValueError('Detached finite clamped CPU FP32 video of declared size required')
    if purpose not in ('generated_clip', 'repeated_observation_codec_timing_proxy'):
        raise ValueError('Explicit retained RGB purpose required')
    out = Path(out); out.mkdir()
    records = []
    for index in range(17):
        frame = video[:, :, index:index + 1].contiguous()
        target = out / f'{index:04d}.safetensors'
        save_file({'rgb': frame}, str(target))
        records.append({'index': index, 'file': target.name, 'bytes': target.stat().st_size,
                        'sha256': sha(target), 'tensor_sha256': tensor_sha(frame)})
    record = {'schema': 'wan22-rgb-frames-v1', 'purpose': purpose,
              'shape': list(video.shape), 'dtype': 'float32', 'range': [-1, 1], 'frames': records}
    (out / 'index.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def first_frame(video, path):
    raw = video[0, :, 0].permute(1, 2, 0).numpy()
    pixels = np.rint((raw + 1) * np.float32(127.5)).clip(0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def difference(actual, reference):
    if actual.shape != reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError('Finite identically shaped difference inputs required')
    delta = actual.double() - reference.double()
    rmse = delta.square().mean().sqrt().item()
    denominator = reference.double().square().mean().sqrt().item()
    return {'equal': torch.equal(actual, reference), 'rmse': rmse,
            'max_absolute_difference': delta.abs().max().item(), 'reference_rms': denominator,
            'relative_rmse': rmse / denominator if denominator else None,
            'normalizer': 'RMS of retained reference values; no equivalence threshold'}
