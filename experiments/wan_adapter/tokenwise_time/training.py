# SPDX-License-Identifier: Apache-2.0
"""CPU-testable mechanics for a fresh original adapter and tokenwise Wan times.

This pilot uses 100% image-conditioned examples, with no uniformly noised T2V
mixture. The observed latent is independently encoded and clean. Only future
positions contribute to the flow loss. No checkpoint or dataset is loaded here.
"""
from contextlib import contextmanager
import math
import torch
from torch.utils.checkpoint import checkpoint
from adapter import ActionObservationAdapter
from tokenwise_time.portable import token_times


def training_pair(target, noise, observation, k):
    """Noising/loss target plus integer times; never derive observation from target."""
    if target.ndim != 5 or target.shape[1] != 16 or target.shape[2] < 2:
        raise ValueError('target must be [B,16,F,H,W] with future frames')
    if target.shape != noise.shape or observation.shape != (*target.shape[:2],1,*target.shape[3:]):
        raise ValueError('Noise/independent observation shapes do not match target')
    if any(v.dtype != torch.float32 or v.device != target.device for v in (target,noise,observation)):
        raise ValueError('Latent tensors must be float32 on the same device')
    if any(not torch.isfinite(v).all().item() for v in (target,noise,observation)):
        raise ValueError('All latent tensors must be finite')
    if k.dtype != torch.int64 or k.shape != (target.shape[0],) or k.device != target.device:
        raise ValueError('k must be int64 [B] on the latent device')
    if (k < 50).any() or (k > 950).any():
        raise ValueError('The declared integer time distribution is 50..950 inclusive')
    b,_,frames,height,width=target.shape
    if height%2 or width%2:
        raise ValueError('Spatial latent dimensions must divide into 2x2 patches')
    sigma=k.float().reshape(b,1,1,1,1)/1000
    future=(1-sigma)*target[:,:,1:]+sigma*noise[:,:,1:]
    noisy=torch.cat((observation,future),dim=2)
    velocity=torch.cat((torch.zeros_like(observation),noise[:,:,1:]-target[:,:,1:]),dim=2)
    grid=(frames,height//2,width//2)
    times=token_times(torch.tensor([grid]*b,dtype=torch.int64),k,math.prod(grid))
    return noisy,velocity,times


def future_loss(prediction, velocity):
    if prediction.shape != velocity.shape or prediction.ndim != 5 or prediction.shape[2] < 2:
        raise ValueError('Predicted velocity must match the full target shape')
    return torch.nn.functional.mse_loss(prediction[:,:,1:].float(),velocity[:,:,1:].float())


def sample_training_inputs(target, generator):
    """CPU RNG supplies exact integer k and FP32 noise; caller moves to its device."""
    if generator.device.type != 'cpu':
        raise ValueError('Use an explicitly seeded CPU generator')
    k=torch.randint(50,951,(target.shape[0],),generator=generator,dtype=torch.int64)
    noise=torch.randn(target.shape,generator=generator,dtype=torch.float32)
    return k.to(target.device),noise.to(target.device)


@contextmanager
def conditioning(adapter,core,actions,observation,grid):
    """Retain all hooks through backward and remove even partially attached hooks."""
    try:
        adapter.attach(core,actions,observation[:,:,0],grid)
        yield
    finally:
        adapter.detach()


@contextmanager
def checkpoint_blocks(core):
    """Checkpoint each frozen block forward, with adapter hooks outside replay.

    Only instance forward methods are wrapped. nn.Module forward hooks execute
    outside the checkpoint function, so the adapter is called once per original
    forward and its output remains in autograd. Earlier adapter output gradients
    pass through later block recomputation. No reentrant checkpoint is used.
    """
    if any(p.requires_grad for p in core.parameters()):
        raise ValueError('The whole pretrained core must be frozen')
    old=[block.forward for block in core.blocks]
    calls=[0]*len(old)
    def wrapper(original,index):
        def call(*args,**kwargs):
            calls[index]+=1
            return original(*args,**kwargs)
        def run(*args,**kwargs):
            return checkpoint(call,*args,use_reentrant=False,**kwargs)
        return run
    try:
        for index,block in enumerate(core.blocks):block.forward=wrapper(old[index],index)
        yield calls
    finally:
        for block,original in zip(core.blocks,old):block.forward=original


def predict(core,adapter,noisy,times,observation,actions,contexts):
    grid=(noisy.shape[2],noisy.shape[3]//2,noisy.shape[4]//2)
    with conditioning(adapter,core,actions,observation,grid):
        return torch.stack(core(list(noisy.unbind(0)),times,contexts,math.prod(grid)))


def optimizer_update(core,adapter,optimizer,noisy,velocity,times,observation,actions,contexts):
    """Exactly one adapter update; no CFG, extra examples, or image decoding."""
    if any(p.requires_grad for p in core.parameters()):raise ValueError('Core is not frozen')
    if any(p.dtype != torch.float32 for p in adapter.parameters()):raise ValueError('Adapter must be FP32')
    grid=(noisy.shape[2],noisy.shape[3]//2,noisy.shape[4]//2)
    optimizer.zero_grad(set_to_none=True)
    with conditioning(adapter,core,actions,observation,grid),checkpoint_blocks(core) as replay_calls:
        prediction=torch.stack(core(list(noisy.unbind(0)),times,contexts,math.prod(grid)))
        if not torch.isfinite(prediction).all().item():raise RuntimeError('Non-finite predicted velocity')
        loss=future_loss(prediction,velocity)
        if not torch.isfinite(loss).item():raise RuntimeError('Non-finite future-only loss')
        loss.backward()
        parameters=list(adapter.parameters())
        if not parameters or any(p.grad is None for p in parameters):
            raise RuntimeError('Every adapter parameter must have a gradient tensor')
        gradients=[p.grad for p in parameters]
        if not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError('Non-finite adapter gradients')
        per_residual=[]
        for residual in adapter.residuals:
            squares=sum(float(p.grad.float().square().sum()) for p in residual.parameters() if p.grad is not None)
            residual_norm=math.sqrt(squares)
            if not math.isfinite(residual_norm) or residual_norm<=0:
                raise RuntimeError('Each adapter residual must receive finite positive aggregate gradient norm')
            per_residual.append(residual_norm)
        norm=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.))
        if not math.isfinite(norm) or norm <= 0:raise RuntimeError('Gradient norm must be finite and positive')
        if any(p.grad is not None for p in core.parameters()):raise RuntimeError('Frozen core accumulated gradients')
        optimizer.step()
        if not all(torch.isfinite(p).all().item() for p in adapter.parameters()):
            raise RuntimeError('Non-finite updated adapter parameters')
        result={'future_flow_mse':float(loss),'gradient_norm_before_clip':norm,
                'residual_gradient_norms':per_residual,'gradient_tensors':len(gradients),
                'block_forward_and_recompute_calls':list(replay_calls)}
    return result
