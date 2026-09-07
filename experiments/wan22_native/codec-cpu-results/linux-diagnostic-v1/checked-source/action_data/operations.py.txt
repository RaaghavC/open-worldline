# SPDX-License-Identifier: Apache-2.0
"""Codec operations with explicit initial-image and reconstruction boundaries."""
import hashlib
import math

import numpy as np
import torch


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def exact_prefix(target, observation, *, name):
    if target.ndim != 5 or observation.shape != target[:, :, :1].shape:
        raise ValueError('Observation must have exactly one matching latent frame')
    if not torch.isfinite(target).all() or not torch.isfinite(observation).all():
        raise ValueError('Nonfinite codec values')
    difference = (target[:, :, :1].cpu() - observation.cpu()).abs()
    row = {'name': name, 'passed': bool(torch.equal(target[:, :, :1].cpu(), observation.cpu())),
           'max_abs': float(difference.max()), 'target_prefix_sha256': tensor_sha(target[:, :, :1]),
           'observation_sha256': tensor_sha(observation), 'target_prefix_replaced': False}
    return row


def observation_only(codec, video):
    if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] < 1:
        raise ValueError('RGB video required')
    # Independent storage and a one-frame encode prevent any future pixel access.
    return codec.encode(video[:, :, :1].clone())


def reconstruct_target(codec, target):
    """Decoder accepts only the encoded target latent, without RGB or commands."""
    return codec.decode(target)


def score_reconstruction(truth, reconstruction):
    if truth.ndim != 4 or truth.shape[-1] != 3 or truth.dtype != np.uint8:
        raise ValueError('Native uint8 [frames,height,width,3] truth required')
    expected = (1, 3, truth.shape[0], truth.shape[1], truth.shape[2])
    if tuple(reconstruction.shape) != expected or reconstruction.dtype != torch.float32:
        raise ValueError('Matching FP32 codec reconstruction required')
    if not torch.isfinite(reconstruction).all():
        raise ValueError('Nonfinite reconstructed pixels')
    predicted = ((reconstruction.detach().cpu()[0].permute(1, 2, 3, 0).numpy()+1)/2).clip(0, 1)
    reference = truth.astype(np.float32)/255.
    difference = predicted-reference
    def metrics(delta):
        mse = float(np.mean(delta.astype(np.float64)**2))
        return {'mse_rgb_0_1': mse, 'mae_rgb_0_1': float(np.mean(np.abs(delta.astype(np.float64)))),
                'psnr_db': None if mse == 0 else -10*math.log10(mse), 'exact_reconstruction': mse == 0}
    return {'scope': 'All frames are codec reconstructions of supplied RGB, not generated futures',
            'all_frames': metrics(difference), 'first_frame': metrics(difference[:1]),
            'later_frames': metrics(difference[1:]) if len(truth)>1 else None,
            'per_frame': [{'frame': i, **metrics(difference[i])} for i in range(len(truth))]}, np.rint(predicted*255).astype(np.uint8)
