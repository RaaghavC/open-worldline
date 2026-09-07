# SPDX-License-Identifier: Apache-2.0
"""Declared native-style T2V solver settings; no action or image condition."""
import torch
from .vendor.fm_solvers_unipc import FlowUniPCMultistepScheduler

STEPS = 50
SHIFT = 8.
GUIDANCE = 6.
SEED = 20260908
LATENT_SHAPE = (16, 5, 36, 64)


def make_scheduler(device='cpu'):
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(STEPS, device=device, shift=SHIFT)
    return scheduler


def initial_noise(seed=SEED, device='cpu'):
    """Noise every frame, including frame zero; no capture/cache input exists."""
    return torch.randn(LATENT_SHAPE, generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32).to(device)


@torch.inference_mode()
def negative_positive_pair(model, latent, timestep, negative, positive):
    if latent.shape != LATENT_SHAPE or latent.dtype != torch.float32:
        raise ValueError('Expected full FP32 native-control latent shape')
    t = torch.as_tensor(timestep, dtype=torch.int64, device=latent.device).reshape(1)
    negative_velocity = model([latent], t, [negative], 2880)[0]
    positive_velocity = model([latent], t, [positive], 2880)[0]
    if not torch.isfinite(negative_velocity).all() or not torch.isfinite(positive_velocity).all():
        raise RuntimeError('Non-finite native-control velocity')
    guided = negative_velocity+GUIDANCE*(positive_velocity-negative_velocity)
    if not torch.isfinite(guided).all():
        raise RuntimeError('Non-finite guided native-control velocity')
    return guided


@torch.inference_mode()
def sample(model, latent, negative, positive, callback=None):
    scheduler = make_scheduler(latent.device)
    for index, timestep in enumerate(scheduler.timesteps):
        velocity = negative_positive_pair(model, latent, timestep, negative, positive)
        latent = scheduler.step(velocity.unsqueeze(0), timestep, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)
        if latent.dtype != torch.float32 or not torch.isfinite(latent).all():
            raise RuntimeError('Non-finite or non-FP32 solver state')
        if callback:
            callback(index, timestep, latent)
    return latent
