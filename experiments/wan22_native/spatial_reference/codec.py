# SPDX-License-Identifier: Apache-2.0
"""Native CUDA FP32 VAE helpers for the two declared spatial conditions.

The model and normalization come from the unchanged, pinned CUDA loader.
No image resizing, cropping, new weights, or VAE equation changes occur here.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..cuda_reference.decode import load_codec
from ..cuda_reference.native import verify_sources


SIZES = frozenset({(288, 512), (704, 1248)})
PREVIEW_DURATION_MS = 120
CONTACT_FRAMES = (0, 1, 2, 4, 6, 8, 10, 12, 14, 16)


def _size(height, width):
    if type(height) is not int or type(width) is not int or (height, width) not in SIZES:
        raise ValueError('Only height,width = (288,512) or (704,1248) are supported')
    return height // 16, width // 16


def _tensor(value, shape, label):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f'{label} must have shape {shape}')
    if value.dtype != torch.float32 or not torch.isfinite(value).all().item():
        raise ValueError(f'{label} must contain finite FP32 values')


def _to_cuda(value):
    return value.to(device='cuda:0')


def encode(model, scale, image, height, width):
    """Encode one independent RGB image, returning a CPU FP32 native latent.

    ``image`` is [1,3,1,H,W] in [-1,1]. It is transferred to CUDA without a
    dtype change before the literal native encode call. No other RGB is read.
    """
    latent_h, latent_w = _size(height, width)
    _tensor(image, (1, 3, 1, height, width), 'Independent image')
    if image.min().item() < -1 or image.max().item() > 1:
        raise ValueError('Independent image must use the [-1,1] RGB range')
    verify_sources()
    try:
        model.clear_cache()
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            latent = model.encode(_to_cuda(image), scale)
        torch.cuda.synchronize()
        _tensor(latent, (1, 48, 1, latent_h, latent_w), 'Encoded observation')
        return latent.detach().cpu()
    finally:
        model.clear_cache()


def decode(model, scale, latent, height, width):
    """Decode five native latents into 17 CPU FP32 RGB frames, clamped [-1,1]."""
    latent_h, latent_w = _size(height, width)
    _tensor(latent, (48, 5, latent_h, latent_w), 'Saved native latent')
    verify_sources()
    try:
        model.clear_cache()
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            video = model.decode(_to_cuda(latent.unsqueeze(0)), scale)
        torch.cuda.synchronize()
        _tensor(video, (1, 3, 17, height, width), 'Decoded clip')
        return video.detach().clamp(-1, 1).cpu()
    finally:
        model.clear_cache()


def images(video, out, height, width):
    """Save full-size PNGs/contact crops and a separately labeled GIF preview.

    PNGs and each contact-sheet image retain every rounded RGB pixel. GIF
    palette quantization is for preview only; its declared frame delay is
    exactly 120 ms, a representable GIF delay, independent of generation time.
    """
    _size(height, width)
    _tensor(video, (1, 3, 17, height, width), 'Retained decoded RGB')
    if video.device.type != 'cpu' or video.requires_grad:
        raise ValueError('Image export requires detached CPU RGB')
    if video.min().item() < -1 or video.max().item() > 1:
        raise ValueError('Image export requires the retained clamped [-1,1] RGB')
    out = Path(out)
    frames = out / 'frames'
    destinations = (frames, out / 'preview.gif', out / 'comparison.png')
    if any(path.exists() for path in destinations):
        raise FileExistsError('Image artifacts already exist; use a fresh output')
    frames.mkdir(parents=True)
    sequence = []
    for index in range(17):
        raw = video[0, :, index].permute(1, 2, 0).numpy()
        pixels = np.rint((raw + 1) * np.float32(127.5)).clip(0, 255).astype(np.uint8)
        frame = Image.fromarray(pixels)
        frame.save(frames / f'{index:04d}.png')
        sequence.append(frame)
    sequence[0].save(out / 'preview.gif', save_all=True, append_images=sequence[1:],
                     duration=PREVIEW_DURATION_MS, loop=0, optimize=False)
    canvas = Image.new('RGB', (2 * width, 5 * (height + 28)), (245, 242, 234))
    draw = ImageDraw.Draw(canvas)
    for slot, index in enumerate(CONTACT_FRAMES):
        x = slot % 2 * width
        y = slot // 2 * (height + 28)
        label = 'Conditioned reconstruction' if index == 0 else 'Generated future'
        draw.text((x + 8, y + 6), f'{label} {index}', fill=(20, 20, 20))
        canvas.paste(sequence[index], (x, y + 28))
    canvas.save(out / 'comparison.png')
    with Image.open(out / 'preview.gif') as preview:
        durations = []
        for index in range(preview.n_frames):
            preview.seek(index)
            durations.append(preview.info.get('duration', 0))
    return {
        'pixel_rule': 'rint((clamped FP32 RGB + 1) * 127.5), uint8',
        'height': height, 'width': width, 'frames': 17,
        'conditioned_initial_frames': 1, 'new_future_frames': 16,
        'contact_frame_indices': list(CONTACT_FRAMES), 'contact_images_resized': False,
        'preview_requested_frame_duration_ms': PREVIEW_DURATION_MS,
        'preview_requested_playback_fps': 1000 / PREVIEW_DURATION_MS,
        'preview_encoded_frames': len(durations),
        'preview_encoded_frame_durations_ms': durations,
        'preview_encoded_duration_ms': sum(durations),
        'preview_note': 'GIF palette quantization may merge identical frames; PNGs retain all 17 frames. Playback rate is not generation speed.',
        'quality_assessment': 'Not assessed automatically; all frames retained for review',
    }
