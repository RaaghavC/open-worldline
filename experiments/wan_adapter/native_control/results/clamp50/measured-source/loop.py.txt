# SPDX-License-Identifier: Apache-2.0
"""Unchanged CPU UniPC state with explicit transfers to the denoiser device."""
import time
import torch
from .sampling import make_scheduler, negative_positive_pair, LATENT_SHAPE


def synchronize(device):
    if torch.device(device).type == 'mps':
        torch.mps.synchronize()


@torch.inference_mode()
def integrate(model, initial_cpu, negative_cpu, positive_cpu, *, device='mps', callback=None):
    if initial_cpu.device.type != 'cpu' or initial_cpu.dtype != torch.float32 or initial_cpu.shape != LATENT_SHAPE:
        raise ValueError('Expected CPU FP32 full initial noise')
    scheduler = make_scheduler('cpu')
    latent = initial_cpu.clone()
    negative, positive = negative_cpu.to(device), positive_cpu.to(device)
    for index, timestep in enumerate(scheduler.timesteps):
        synchronize(device)
        started = time.perf_counter()
        model_input = latent.to(device)
        synchronize(device)
        input_seconds = time.perf_counter()-started
        started = time.perf_counter()
        velocity = negative_positive_pair(model, model_input, timestep, negative, positive)
        synchronize(device)
        forward_seconds = time.perf_counter()-started
        started = time.perf_counter()
        velocity_cpu = velocity.to('cpu')
        synchronize(device)
        output_seconds = time.perf_counter()-started
        started = time.perf_counter()
        latent = scheduler.step(velocity_cpu.unsqueeze(0), timestep, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)
        solver_seconds = time.perf_counter()-started
        if latent.device.type != 'cpu' or latent.dtype != torch.float32 or not torch.isfinite(latent).all():
            raise RuntimeError('Invalid CPU FP32 UniPC state')
        if callback:
            callback(index, timestep, latent, {'input_transfer_seconds': input_seconds,
                     'negative_positive_seconds': forward_seconds, 'output_transfer_seconds': output_seconds,
                     'cpu_solver_seconds': solver_seconds})
    return latent
