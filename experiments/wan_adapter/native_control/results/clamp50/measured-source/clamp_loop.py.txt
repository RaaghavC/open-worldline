# SPDX-License-Identifier: Apache-2.0
"""Single treatment: observed first latent before each model call and update."""
import time
import torch
from .sampling import make_scheduler, negative_positive_pair, LATENT_SHAPE
from .loop import synchronize


def clamp_first(latent, observation):
    if latent.shape != LATENT_SHAPE or observation.shape != (16,1,36,64):
        raise ValueError('Expected complete video latent and one observed latent frame')
    if latent.dtype != torch.float32 or observation.dtype != torch.float32 or latent.device != observation.device:
        raise ValueError('Clamp inputs must share device and float32 dtype')
    return torch.cat((observation,latent[:,1:]),dim=1)


class ClampedModel:
    def __init__(self, model, observation):
        self.model = model
        self.observation = observation
        self.calls = 0

    def __call__(self, values, timestep, context, seq_len):
        # Preserve original guidance arithmetic and call order. This boundary
        # applies the only treatment independently to both real model calls.
        inputs = [clamp_first(value,self.observation) for value in values]
        self.calls += 1
        return self.model(inputs,timestep,context,seq_len)


@torch.inference_mode()
def integrate_clamped(model, initial_cpu, observation_cpu, negative_cpu, positive_cpu, *, device='mps', callback=None):
    if initial_cpu.device.type != 'cpu' or observation_cpu.device.type != 'cpu':
        raise ValueError('Initial noise and observed frame must be on CPU')
    if not torch.isfinite(observation_cpu).all():
        raise ValueError('Observation contains non-finite values')
    scheduler = make_scheduler('cpu')
    latent = clamp_first(initial_cpu,observation_cpu)
    negative,positive = negative_cpu.to(device),positive_cpu.to(device)
    wrapped = ClampedModel(model,observation_cpu.to(device))
    for index,timestep in enumerate(scheduler.timesteps):
        synchronize(device); started=time.perf_counter()
        model_input=latent.to(device); synchronize(device)
        input_seconds=time.perf_counter()-started
        started=time.perf_counter()
        velocity=negative_positive_pair(wrapped,model_input,timestep,negative,positive)
        synchronize(device); forward_seconds=time.perf_counter()-started
        started=time.perf_counter(); velocity_cpu=velocity.to('cpu'); synchronize(device)
        output_seconds=time.perf_counter()-started
        started=time.perf_counter()
        updated=scheduler.step(velocity_cpu.unsqueeze(0),timestep,latent.unsqueeze(0),return_dict=False)[0].squeeze(0)
        latent=clamp_first(updated,observation_cpu)
        solver_seconds=time.perf_counter()-started
        if latent.dtype != torch.float32 or not torch.isfinite(latent).all():
            raise RuntimeError('Invalid clamped CPU UniPC state')
        if callback:
            callback(index,timestep,latent,{'input_transfer_seconds':input_seconds,
                'negative_positive_seconds':forward_seconds,'output_transfer_seconds':output_seconds,
                'cpu_solver_seconds':solver_seconds,'clamped_prefix_max_abs_difference':float((latent[:,:1]-observation_cpu).abs().max())})
    if wrapped.calls != 100:
        raise RuntimeError('Expected exactly100 original-model calls')
    return latent
