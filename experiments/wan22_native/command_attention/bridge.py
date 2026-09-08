# SPDX-License-Identifier: Apache-2.0
"""Temporary native projection hooks and an optional frozen native-prefix cache.

This file loads no model. The caller owns verified foundation loading, external
value hashing, deadlines and resource monitoring. No method is replaced.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping
import math
import threading
import torch
from torch import nn

from controller import CommandAttentionController, DEFAULT_BLOCKS, DEFAULT_PARAMETER_COUNT, PROJECTIONS, constant
from experiments.wan22_native.cuda_reference.vendor import model as native_model

PROFILES = {'baseline': (48, 5, 18, 32), 'spatial': (48, 5, 44, 78)}
BLOCK_KEYS = frozenset(('e', 'seq_lens', 'grid_sizes', 'freqs', 'context', 'context_lens'))


class _Boundary(BaseException):
    pass


@dataclass(frozen=True)
class FrozenCommandFeatures:
    hidden: torch.Tensor
    time_embedding: torch.Tensor
    block_kwargs: Mapping
    observed_prefix: torch.Tensor
    owner: object = field(repr=False)
    core_stamp: tuple = field(repr=False)
    tensor_stamp: tuple = field(repr=False)


def _cache_stamp(hidden, time, kwargs, observation):
    values = [('hidden', hidden), ('time', time), ('observation', observation)]
    values += [(k, kwargs[k]) for k in sorted(BLOCK_KEYS) if kwargs[k] is not None]
    return tuple((name, id(v), v._version) for name, v in values)


class NativeCommandAttentionBridge(nn.Module):
    """B=1, 17 frames, six conditioned native blocks in production.

    Exclusive core ownership is required. The bridge rejects preexisting hooks,
    overrides and recursive calls. Cache version checks catch ordinary in-place
    mutations, not writes through .data or external storage. External original
    foundation hashes remain required before/after a measured CUDA run.
    """
    def __init__(self, core, controller, *, profile='spatial', test_only=False,
                 test_shape=(2, 5, 4, 6), padding_tokens=0):
        super().__init__()
        if (profile not in PROFILES or type(test_only) is not bool
                or type(controller) is not CommandAttentionController
                or type(padding_tokens) is not int or not 0 <= padding_tokens <= 128):
            raise ValueError('Invalid controller, profile, test mode or token padding')
        self.core, self.controller = core, controller
        self.profile, self.test_only = profile, test_only
        self.shape = tuple(test_shape) if test_only else PROFILES[profile]
        if (len(self.shape) != 4 or self.shape[1] != 5
                or any(type(v) is not int or v < 1 for v in self.shape)
                or any(v % 2 for v in self.shape[-2:])):
            raise ValueError('Require five latent groups and even spatial dimensions')
        self.grid = (5, self.shape[2]//2, self.shape[3]//2)
        self.valid_tokens = math.prod(self.grid)
        self.tokens = self.valid_tokens + padding_tokens
        self.prefix = self.grid[1]*self.grid[2]
        self._lock, self._owner = threading.Lock(), object()
        self._foundation_stamp = None
        self._check()
        self._foundation_stamp = self._core_stamp()[:-1]
        self.core.eval()

    def train(self, mode=True):
        super().train(mode); self.core.eval(); return self

    def _autocast(self):
        return nullcontext() if self.test_only else torch.autocast('cuda', dtype=torch.bfloat16)

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent or recursive bridge calls are unsupported')
        try:
            yield
        finally:
            self._lock.release()

    def _core_stamp(self):
        values = list(self.core.named_parameters()) + list(self.core.named_buffers()) + [('freqs', self.core.freqs)]
        return tuple((name, id(v), v._version) for name, v in values)

    def _check(self):
        core = self.core
        if (type(core) is not native_model.WanModel or type(core.head) is not native_model.Head
                or any(type(b) is not native_model.WanAttentionBlock for b in core.blocks)
                or len({id(b) for b in core.blocks}) != len(core.blocks)
                or any(type(b.self_attn) is not native_model.WanSelfAttention for b in core.blocks)):
            raise ValueError('Require literal native Wan model, blocks and head')
        if (tuple(core.patch_size) != (1,2,2) or core.model_type != 'ti2v'
                or core.in_dim != core.out_dim or core.in_dim != self.shape[0]
                or core.dim != self.controller.hidden_dim
                or self.controller.blocks != tuple(range(self.controller.blocks[0], len(core.blocks)))):
            raise ValueError('Require a contiguous conditioned native suffix and matching TI2V dimensions')
        for module in core.modules():
            if module._forward_hooks or module._forward_pre_hooks or 'forward' in module.__dict__:
                raise ValueError('Native modules must not have hooks or instance forward overrides')
        if 'unpatchify' in core.__dict__:
            raise ValueError('Native unpatchify must not be overridden')
        device = core.patch_embedding.weight.device
        for value in core.parameters():
            if (value.device != device or value.dtype != torch.float32 or value.requires_grad
                    or value.grad is not None or value.is_inference()):
                raise ValueError('Every original foundation parameter must be normal frozen FP32 with no gradient')
        for value in list(core.buffers()) + [core.freqs]:
            if value.requires_grad or value.grad_fn is not None or value.is_inference():
                raise ValueError('Native buffers and RoPE must be normal constants')
        if self.controller.parameter_device() != device:
            raise ValueError('Controller and core devices must agree')
        if self._foundation_stamp is not None and self._core_stamp()[:-1] != self._foundation_stamp:
            raise ValueError('Original foundation parameters/buffers changed since bridge construction')
        if self.test_only:
            if (device.type != 'cpu' or core.dim > 32 or core.in_dim > 4 or len(core.blocks) > 4
                    or self.valid_tokens > 128 or self.controller.rank > 8 or self.controller.command_width > 16):
                raise ValueError('test_only permits small literal CPU fixtures only')
        else:
            if device.type != 'cuda':
                raise ValueError('Production requires CUDA; no CPU model fallback')
            from experiments.wan22_native.cuda_reference import native
            from experiments.wan22_native.cuda_reference.vendor import attention
            native.verify_sources()
            if not attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE:
                raise ValueError('Production requires the pinned FlashAttention2 implementation')
            if (len(tuple(core.parameters())) != 825 or core.dim != 3072 or core.in_dim != 48
                    or len(core.blocks) != 30 or core.text_dim != 4096 or core.text_len != 512
                    or self.controller.blocks != DEFAULT_BLOCKS or self.controller.rank != 32
                    or self.controller.command_width != 128
                    or sum(p.numel() for p in self.controller.parameters()) != DEFAULT_PARAMETER_COUNT):
                raise ValueError('Require original 5B core and declared six-block rank32 controller')
        return device

    def _inputs(self, noisy, times, contexts):
        device = self._check()
        constant(noisy, 'noisy', (1,*self.shape), torch.float32, device)
        constant(times, 'times', (1,self.tokens), torch.int64, device)
        t = int(times[0,self.prefix])
        if (not 0 <= t <= 999 or not bool((times[:,:self.prefix] == 0).all())
                or not bool((times[:,self.prefix:self.valid_tokens] == t).all())
                or not bool((times[:,self.valid_tokens:] == 0).all())):
            raise ValueError('Times must be zero on observation/padding and one shared future integer in [0,999]')
        if (not isinstance(contexts, list) or len(contexts) != 1 or not isinstance(contexts[0], torch.Tensor)
                or contexts[0].ndim != 2 or not 1 <= contexts[0].shape[0] <= self.core.text_len):
            raise ValueError('Require one bounded native text context')
        constant(contexts[0], 'context', (contexts[0].shape[0],self.core.text_dim), torch.float32, device)

    def _conditions(self, commands, observation, expected):
        device = self.core.patch_embedding.weight.device
        constant(commands, 'commands', (1,16,6), torch.float32, device)
        if not bool(((commands[...,5] == 0) | (commands[...,5] == 1)).all()):
            raise ValueError('Interaction channel must contain binary transition pulses')
        constant(observation, 'observation', (1,self.shape[0],1,*self.shape[-2:]), torch.float32, device)
        if not torch.equal(observation, expected):
            raise ValueError('Observation must equal the exact clean input prefix')

    @contextmanager
    def _projection_hooks(self, gates):
        handles, counts = [], {}
        try:
            for site, (block, projection) in enumerate((b,p) for b in self.controller.blocks for p in PROJECTIONS):
                module = getattr(self.core.blocks[block].self_attn, projection)
                if type(module) is not nn.Linear:
                    raise ValueError('Native projection must remain the original Linear module')
                key = (block,projection); counts[key] = 0
                def inject(module, args, output, *, key=key, site=site):
                    counts[key] += 1
                    if (counts[key] != 1 or len(args) != 1 or not isinstance(output,torch.Tensor)
                            or tuple(output.shape) != (1,self.tokens,self.core.dim)
                            or output.dtype not in (torch.float32,torch.bfloat16)):
                        raise RuntimeError('Native projection call signature/count changed')
                    delta = self.controller.residual(*key, args[0], gates[site])
                    # Preserve each literal projection output dtype. Q/K norms,
                    # RoPE and native attention follow this hook unchanged.
                    return output + delta.to(output.dtype)
                handles.append(module.register_forward_hook(inject))
            yield
            if any(count != 1 for count in counts.values()):
                raise RuntimeError('Every selected Q/K/V/O projection must execute exactly once')
        finally:
            for handle in reversed(handles):
                handle.remove()

    def _output(self, output):
        if (tuple(output.shape) != (1,*self.shape) or output.dtype != torch.float32
                or not bool(torch.isfinite(output).all())):
            raise FloatingPointError('Require finite FP32 native latent velocity')
        self._check()
        return output

    def forward(self, noisy, times, contexts, *, commands, observation, track_grad=True):
        if type(track_grad) is not bool:
            raise ValueError('track_grad must be boolean')
        with self._exclusive():
            self._inputs(noisy,times,contexts); self._conditions(commands,observation,noisy[:,:,:1])
            with torch.inference_mode(False), torch.set_grad_enabled(track_grad), self._autocast():
                gates = self.controller.gates(commands,self.grid,self.tokens)
                with self._projection_hooks(gates):
                    output = torch.stack(self.core(list(noisy.unbind(0)),times,contexts,self.tokens))
            return self._output(output)

    def extract_features(self, noisy, times, contexts):
        """Original native preprocessing + blocks0..23, stopped before block24."""
        with self._exclusive():
            self._inputs(noisy,times,contexts)
            captured, handles, boundary = {}, [], _Boundary()
            selected = self.core.blocks[self.controller.blocks[0]]
            def time_hook(module,args,output):
                if 'time' in captured or len(args) != 1:
                    raise RuntimeError('Native time embedding call changed')
                captured['time'] = output
            def boundary_hook(module,args,kwargs):
                if len(args) != 1 or set(kwargs) != BLOCK_KEYS or 'hidden' in captured or 'time' not in captured:
                    raise RuntimeError('Native prefix call boundary changed')
                captured['hidden'], captured['kwargs'] = args[0], MappingProxyType(dict(kwargs))
                raise boundary
            try:
                handles.append(self.core.time_embedding.register_forward_hook(time_hook))
                handles.append(selected.register_forward_pre_hook(boundary_hook,with_kwargs=True))
                with torch.inference_mode(False), torch.no_grad(), self._autocast():
                    self.core(list(noisy.unbind(0)),times,contexts,self.tokens)
            except _Boundary as error:
                if error is not boundary:
                    raise
                error.__traceback__ = None
            else:
                raise RuntimeError('Native prefix did not stop at its declared boundary')
            finally:
                for handle in reversed(handles):handle.remove()
            if set(captured) != {'hidden','time','kwargs'}:
                raise RuntimeError('Incomplete native prefix capture')
            with torch.inference_mode(False), torch.no_grad():
                observed = noisy[:,:,:1].clone()
                result = FrozenCommandFeatures(captured['hidden'],captured['time'],captured['kwargs'],observed,
                    self._owner,self._core_stamp(),_cache_stamp(captured['hidden'],captured['time'],captured['kwargs'],observed))
            self._features(result)
            return result

    def _features(self, value):
        device = self._check()
        if (type(value) is not FrozenCommandFeatures or value.owner is not self._owner
                or value.core_stamp != self._core_stamp()):
            raise ValueError('Cache owner or frozen foundation identity changed')
        k = value.block_kwargs
        if not isinstance(k,Mapping) or set(k) != BLOCK_KEYS or k['context_lens'] is not None:
            raise ValueError('Require literal native block keyword arguments')
        if value.tensor_stamp != _cache_stamp(value.hidden,value.time_embedding,k,value.observed_prefix):
            raise ValueError('Cached tensors were replaced or modified')
        constant(value.hidden,'cached hidden',(1,self.tokens,self.core.dim),torch.float32,device,finite=False)
        constant(value.time_embedding,'cached time',(1,self.tokens,self.core.dim),torch.float32,device,finite=False)
        constant(k['e'],'cached projected time',(1,self.tokens,6,self.core.dim),torch.float32,device,finite=False)
        constant(k['context'],'projected text',(1,self.core.text_len,self.core.dim),
                 torch.float32 if self.test_only else torch.bfloat16,device,finite=False)
        constant(k['grid_sizes'],'grid',(1,3),torch.int64,torch.device('cpu'))
        constant(k['seq_lens'],'sequence lengths',(1,),torch.int64,torch.device('cpu'))
        if (k['grid_sizes'].tolist() != [list(self.grid)] or k['seq_lens'].tolist() != [self.valid_tokens]
                or k['freqs'] is not self.core.freqs):
            raise ValueError('Native grid, unpadded sequence length or rotary constant changed')
        constant(value.observed_prefix,'cached observation',(1,self.shape[0],1,*self.shape[-2:]),torch.float32,device)

    def predict_from_features(self, features, commands, observation, *, track_grad=True):
        if type(track_grad) is not bool:
            raise ValueError('track_grad must be boolean')
        with self._exclusive():
            self._features(features); self._conditions(commands,observation,features.observed_prefix)
            with torch.inference_mode(False), torch.set_grad_enabled(track_grad), self._autocast():
                gates = self.controller.gates(commands,self.grid,self.tokens)
                with self._projection_hooks(gates):
                    hidden = features.hidden
                    for block in self.core.blocks[self.controller.blocks[0]:]:
                        hidden = block(hidden,**features.block_kwargs)
                    patches = self.core.head(hidden,features.time_embedding)
                    output = torch.stack([x.float() for x in self.core.unpatchify(patches,features.block_kwargs['grid_sizes'])])
            self._features(features)
            return self._output(output)
