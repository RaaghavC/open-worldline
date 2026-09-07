# SPDX-License-Identifier: Apache-2.0
"""Fixed paired flow-matching objective and small adapter-only recovery files."""
import math
from pathlib import Path
import os

import torch
from safetensors.torch import save_file

from ..action_adapter.model import PostBlockActionAdapter
from ..load_weights import tensor_sha256, sha256
from ..portable import token_times
from experiments.room_world.memory_train import atomic_write

SEED = 20260907
STARTS = (0, 8, 32, 49)
SHAPE = (1, 48, 5, 18, 32)
OPTIMIZER = dict(lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)


def update_count(mode):
    if mode not in ('probe', 'fixed16'):
        raise ValueError('Only the two-update probe and fresh fixed16 run are specified')
    return 2 if mode == 'probe' else 16


def fresh_adapter(*, test_configuration=None):
    """Initialization is independent of load-time/global RNG and previous probes."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        return PostBlockActionAdapter(**(test_configuration or {}))


def make_draws(mode, *, shape=SHAPE):
    """Private CPU RNG: draw inclusive integer k, then full FP32 noise, per pair."""
    if len(shape) != 5 or shape[0] != 1 or min(shape) < 1 or shape[2] < 2:
        raise ValueError('Positive B=1 video shape with a future frame required')
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    rows, tensors = [], {}
    for index in range(update_count(mode)):
        k = int(torch.randint(50, 951, (1,), generator=generator).item())
        noise = torch.randn(shape, generator=generator, dtype=torch.float32, device='cpu')
        key = f'noise_{index:04d}'
        tensors[key] = noise
        tensors[f'rng_after_{index:04d}'] = generator.get_state().clone()
        start = STARTS[index % 4]
        rows.append(dict(update=index+1, start=start, branches=[f'closed-{start:04d}', f'open-{start:04d}'],
                         k=k, sigma=k/1000., noise_key=key, noise_sha256=tensor_sha256(noise),
                         rng_after_sha256=tensor_sha256(tensors[f'rng_after_{index:04d}'])))
    return rows, tensors


def checked_float(value, label):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or not torch.isfinite(value).all().item()):
        raise ValueError(label+' must be finite FP32')


def flow_inputs(target, observation, noise, k):
    """Future truth enters standard noisy latents and the loss target only."""
    for name, value in (('target', target), ('observation', observation), ('noise', noise)):
        checked_float(value, name)
        if value.device.type != 'cpu' or value.requires_grad:
            raise ValueError('Dataset and saved draw inputs must be non-gradient CPU tensors')
    if type(k) is not int or not 50 <= k <= 950:
        raise ValueError('Uniform integer k must lie in [50,950]')
    if (target.ndim != 5 or target.shape[0] != 1 or target.shape[2] < 2
            or target.shape != noise.shape or observation.shape != target[:, :, :1].shape
            or min(target.shape[-2:]) < 2 or any(d % 2 for d in target.shape[-2:])):
        raise ValueError('Require aligned B=1 latents, one observed frame and spatial patches')
    if not torch.equal(target[:, :, :1], observation):
        raise ValueError('Full target prefix differs from the independent observation')
    sigma = k/1000.
    noisy = (1-sigma)*target + sigma*noise
    noisy[:, :, :1] = observation
    velocity = noise-target
    grid = torch.tensor([[target.shape[2], target.shape[3]//2, target.shape[4]//2]], dtype=torch.int64)
    count = math.prod(grid[0].tolist())
    times = token_times(grid, k, count)
    return noisy, times, velocity


def future_flow_mse(prediction, velocity):
    checked_float(prediction, 'prediction'); checked_float(velocity, 'flow target')
    if prediction.shape != velocity.shape or prediction.ndim != 5 or prediction.shape[2] < 2:
        raise ValueError('Aligned predictions and future flow targets required')
    if prediction.device != velocity.device:
        raise ValueError('Prediction and loss target devices differ')
    return (prediction[:, :, 1:]-velocity[:, :, 1:]).square().mean()


def parameter_records(model, *, check=None, expected_count=825, expected=None):
    """Hash one current parameter at a time. No whole-core clone or state dict."""
    parameters = list(model.named_parameters())
    if len(parameters) != expected_count or expected is not None and set(expected) != {n for n,_ in parameters}:
        raise ValueError('Frozen parameter coverage differs')
    result = {}
    for name, parameter in parameters:
        if check: check()
        if parameter.requires_grad or parameter.grad is not None:
            raise RuntimeError('A core parameter became trainable or acquired a gradient: '+name)
        value = parameter.detach().cpu()
        if not torch.isfinite(value).all().item():
            raise FloatingPointError('Nonfinite frozen core value: '+name)
        row = dict(shape=list(value.shape), dtype=str(value.dtype).removeprefix('torch.'), sha256=tensor_sha256(value))
        del value
        if expected is not None and row != expected[name]:
            raise RuntimeError('Frozen core value changed: '+name)
        result[name] = row
    return result


def loaded_records(report):
    return {name:dict(shape=row['shape'], dtype=row['loaded_dtype'], sha256=row['loaded_sha256'])
            for name,row in report['tensors'].items()}


def grad_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt().item()) if values else 0.


def paired_update(wrapper, windows, noise, k, context, optimizer, *, check=None):
    """Closed then open, two live B=1 forwards and half-loss backpropagations."""
    if len(windows) != 2:
        raise ValueError('Exactly closed/open windows are required')
    device = next(wrapper.adapter.parameters()).device
    if any(p.requires_grad or p.grad is not None for p in wrapper.core.parameters()):
        raise ValueError('Core must be entirely frozen and gradient-free')
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in wrapper.adapter.parameters()}:
        raise ValueError('Optimizer must own exactly the adapter parameters')
    optimizer.zero_grad(set_to_none=True)
    branches = []
    for index, window in enumerate(windows):
        if check: check()
        if set(window) != {'target','observation','commands'}:
            raise ValueError('Only the three declared training window tensors are accepted')
        checked_float(window['commands'], 'commands')
        if window['commands'].shape != (1, 4*(noise.shape[2]-1), 6):
            raise ValueError('Commands must cover the exact RGB transitions')
        noisy, times, velocity = flow_inputs(window['target'], window['observation'], noise, k)
        identity = {name:tensor_sha256(value) for name,value in [('noisy',noisy),('token_times',times),
            ('observation',window['observation']),('commands',window['commands']),('flow_target',velocity)]}
        prediction = wrapper(noisy.to(device), times.to(device), [context.to(device)],
            commands=window['commands'].to(device), observation=window['observation'].to(device), seq_len=times.shape[1])
        loss = future_flow_mse(prediction, velocity.to(device))
        if not torch.isfinite(loss).item(): raise FloatingPointError('Nonfinite paired loss')
        (loss*.5).backward()
        branches.append(dict(branch=('closed','open')[index], future_flow_mse=float(loss.detach().cpu()), input_sha256=identity))
        del prediction, loss, noisy, times, velocity
    parameters = list(wrapper.adapter.parameters())
    if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in parameters):
        raise FloatingPointError('Every adapter gradient must be present and finite')
    before = grad_norm(parameters)
    gru = grad_norm(wrapper.adapter.command_gru.parameters())
    output = grad_norm(wrapper.adapter.output.parameters())
    if not math.isfinite(before) or before <= 0:
        raise FloatingPointError('Nonzero finite adapter gradient required')
    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    after = grad_norm(parameters)
    optimizer.step()
    require_finite_tree(optimizer.state_dict())
    if any(not torch.isfinite(p).all().item() for p in parameters):
        raise FloatingPointError('Nonfinite adapter after optimizer update')
    if any(p.requires_grad or p.grad is not None for p in wrapper.core.parameters()):
        raise RuntimeError('Frozen core changed gradient ownership')
    return dict(branches=branches, paired_mean_future_flow_mse=sum(x['future_flow_mse'] for x in branches)/2,
        gradient_l2_before_clip=before, gradient_l2_after_clip=after, command_gru_gradient_l2=gru,
        output_gradient_l2=output, optimizer_updates=1, live_sequential_forwards=2,
        loss='Mean future latent flow MSE; each branch contributes one half; initial latent excluded')


def require_finite_tree(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():raise FloatingPointError('Nonfinite optimizer/recovery tensor')
    elif isinstance(value, float):
        if not math.isfinite(value):raise FloatingPointError('Nonfinite optimizer/recovery scalar')
    elif isinstance(value, dict):
        for v in value.values():require_finite_tree(v)
    elif isinstance(value, (list,tuple)):
        for v in value:require_finite_tree(v)


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def save_checkpoint(output, adapter, optimizer, completed, *, identity, draw_rng_state):
    """Publish an immutable validated bundle, then advance a last-valid pointer."""
    output = Path(output)
    final = output/f'checkpoint-{completed:04d}'
    temporary = output/f'.checkpoint-{completed:04d}.partial'
    if final.exists() or final.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise ValueError('Refuse overwriting any checkpoint or partial bundle')
    if {id(p) for g in optimizer.param_groups for p in g['params']} != {id(p) for p in adapter.parameters()}:
        raise ValueError('Recovery optimizer must contain only adapter parameters')
    require_finite_tree(optimizer.state_dict())
    temporary.mkdir()
    values = {name:p.detach().cpu().contiguous().clone() for name,p in adapter.named_parameters()}
    if any(not torch.isfinite(v).all().item() for v in values.values()):
        raise FloatingPointError('Invalid checkpoint parameters')
    save_file(values, str(temporary/'adapter.safetensors'))
    recovery = dict(optimizer=cpu_tree(optimizer.state_dict()), torch_cpu_rng=torch.get_rng_state().clone(),
                    draw_rng_state=draw_rng_state.detach().cpu().clone(), completed_updates=completed,
                    identity=identity, schema='worldline-wan22-action-recovery-v1')
    torch.save(recovery, temporary/'optimizer-and-rng.pt')
    record = dict(schema='worldline-wan22-action-checkpoint-v1', completed_updates=completed, identity=identity,
        tensors={k:dict(shape=list(v.shape),dtype='float32',sha256=tensor_sha256(v)) for k,v in values.items()},
        files={name:sha256(temporary/name) for name in ('adapter.safetensors','optimizer-and-rng.pt')},
        external_core_weights_included=False, resume_supported=False)
    atomic_write(temporary/'manifest.json',record)
    os.rename(temporary,final)
    atomic_write(output/'last-valid.json', dict(directory=final.name, manifest_sha256=sha256(final/'manifest.json'), completed_updates=completed))
    return dict(directory=final.name, manifest_sha256=sha256(final/'manifest.json'), **record)
