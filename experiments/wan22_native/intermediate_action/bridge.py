# SPDX-License-Identifier: Apache-2.0
"""Experimental single-site adapter insertion in the literal native Wan loop.

No model methods, weights, loader, sampler or published bridge are changed.
This module performs no weight loading and admits no actual CUDA experiment.
"""
import torch
from torch import nn

from experiments.wan22_native.action_cuda.bridge import NativeCUDAActionBridge


class IntermediateActionBridge(nn.Module):
    """Insert the unchanged adapter after one zero-based native block index.

    The existing final-head bridge supplies its unchanged input, precision and
    model checks. It is never used to execute or capture features here. There
    is one temporary output hook and one original native forward per call.

    With all native parameters and inputs frozen, ordinary autograd creates
    no graph before the adapter. Its output starts the graph needed through
    subsequent frozen blocks and the original head/unpatchify. Those later
    operations still retain backward activations; frozen weights do not make
    that activation memory disappear.

    Exclusive core ownership is required. This is neither a cache nor a
    replacement for a measured CUDA parity and resource admission.
    """

    def __init__(self, core, adapter, *, block_index=28, profile="baseline", test_only=False):
        super().__init__()
        self._contract = NativeCUDAActionBridge(core, adapter, profile=profile, test_only=test_only)
        self.block_index = block_index
        self._check_site()

    @property
    def core(self):
        return self._contract.core

    @property
    def adapter(self):
        return self._contract.adapter

    @property
    def shape(self):
        return self._contract.shape

    @property
    def tokens(self):
        return self._contract.tokens

    @property
    def grid(self):
        return self._contract.grid

    def train(self, mode=True):
        super().train(mode)
        self.core.eval()
        return self

    def _check_site(self):
        from experiments.wan22_native.cuda_reference.vendor.model import WanModel, WanAttentionBlock, Head
        if (type(self.core) is not WanModel or type(self.core.head) is not Head
                or any(type(block) is not WanAttentionBlock for block in self.core.blocks)):
            raise ValueError("Require the literal native Wan model, blocks and head")
        if (type(self.block_index) is not int
                or not 0 <= self.block_index < len(self.core.blocks)):
            raise ValueError("block_index must be a zero-based integer naming an existing native block")
        if len({id(block) for block in self.core.blocks}) != len(self.core.blocks):
            raise ValueError("Each native block must be a distinct module")
        constants = list(self.core.buffers()) + [self.core.freqs]
        if any(value.requires_grad or value.grad_fn is not None or value.is_inference()
               for value in constants):
            raise ValueError("Core buffers and rotary frequencies must be normal non-gradient constants")

    def forward(self, noisy, times, contexts, *, commands, observation, track_grad=True):
        """Return FP32 B=1 velocity without mutating or clamping input latents.

        Inputs must be normal non-gradient tensors. With track_grad=True this
        explicitly builds the adapter/suffix graph even under outer no_grad or
        inference_mode contexts; the caller's context is restored afterward.
        track_grad=False retains no graph. The adapter masks its own residual;
        later self-attention may change the observed-token output velocity.
        """
        if type(track_grad) is not bool:
            raise ValueError("track_grad must be an explicit boolean")
        with self._contract._exclusive():
            self._contract._inputs(noisy, times, contexts)
            self._contract._conditions(commands, observation, noisy[:, :, :1])
            self._check_site()
            if any(value.requires_grad or value.grad_fn is not None
                   for value in (noisy, times, commands, observation, *contexts)):
                raise ValueError("All input tensors must have requires_grad=False and no prior graph")
            selected = self.core.blocks[self.block_index]
            calls = 0

            def inject(module, args, kwargs, hidden):
                nonlocal calls
                if module is not selected or calls:
                    raise RuntimeError("Selected native block must execute exactly once")
                calls += 1
                if (len(args) != 1 or set(kwargs) != {
                        "e", "seq_lens", "grid_sizes", "freqs", "context", "context_lens"}):
                    raise RuntimeError("Literal native block call signature changed")
                if (not isinstance(hidden, torch.Tensor) or hidden.dtype != torch.float32
                        or tuple(hidden.shape) != (1, self.tokens, self.core.dim)
                        or hidden.requires_grad or hidden.grad_fn is not None or hidden.is_inference()):
                    raise RuntimeError("Selected block must provide normal frozen FP32 hidden tokens before insertion")
                if kwargs["grid_sizes"].tolist() != [list(self.grid)] or kwargs["seq_lens"].tolist() != [self.tokens]:
                    raise RuntimeError("Native grid and sequence length differ from the declared profile")
                # No detach or changed dtype is needed: hidden is already a
                # constant because every upstream input and parameter is frozen.
                # The existing adapter disables autocast for its FP32 equations.
                return self.adapter(hidden, commands, observation, self.grid)

            handle = selected.register_forward_hook(inject, with_kwargs=True)
            try:
                with torch.inference_mode(False), torch.set_grad_enabled(track_grad), self._contract._autocast():
                    values = self.core(list(noisy.unbind(0)), times, contexts, self.tokens)
                    if not isinstance(values, list) or len(values) != 1:
                        raise RuntimeError("Native forward must return exactly one latent velocity")
                    output = torch.stack(values)
            finally:
                # Also handles errors before the selected block, in the
                # adapter, following blocks, head or unpatchify. No sentinel
                # exception or captured native frame survives this call.
                handle.remove()
            if calls != 1:
                raise RuntimeError("Native forward did not execute the selected block exactly once")
            if (output.shape != (1, *self.shape) or output.dtype != torch.float32
                    or not bool(torch.isfinite(output).all())):
                raise FloatingPointError("Require a finite FP32 native velocity at the declared shape")
            self._contract._check_core()
            return output
