# SPDX-License-Identifier: Apache-2.0
"""Capture the literal native forward before its head, then train the adapter.

No native method or equation is replaced. A temporary local prehook raises a
private sentinel before the head runs; finally removes it on every exit path.
This module neither loads weights nor implements a trainer or sampler.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import threading

import torch
from torch import nn

from ..action_adapter.model import NATIVE_PARAMETER_COUNT, PostBlockActionAdapter


PROFILES = {
    "baseline": {"shape": (48, 5, 18, 32), "tokens": 720, "prefix": 144},
    "spatial": {"shape": (48, 5, 44, 78), "tokens": 4290, "prefix": 858},
}


class _HeadBoundary(BaseException):
    """Caught by identity only, within the single extraction that created it."""


@dataclass(frozen=True)
class FrozenNativeFeatures:
    hidden: torch.Tensor
    time_embedding: torch.Tensor
    grid_sizes: torch.Tensor
    observed_prefix: torch.Tensor
    profile: str
    owner: object = field(repr=False, compare=False)


def _float_tensor(value, name, shape, device):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.dtype != torch.float32 or value.device != device
            or tuple(value.shape) != tuple(shape) or value.is_inference()):
        raise ValueError(f"{name} must be a normal dense FP32 tensor of shape {tuple(shape)} on {device}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


class NativeCUDAActionBridge(nn.Module):
    """B=1 original adapter bridge at the two declared 17-frame resolutions.

    Production requires the pinned literal WanModel, original FP32 CUDA
    parameters, FlashAttention 2 only, and a preverified external weight load.
    test_only permits small CPU stand-ins at the same spatial grids. It is not
    a CPU fallback or evidence of real CUDA numerical parity.

    The caller must give this bridge exclusive use of the core while calling
    it; other threads, wrappers, hooks and compiled wrappers are unsupported.
    Frozen feature bundles are ephemeral and belong to this bridge instance.
    """

    def __init__(self, core, adapter, *, profile="baseline", test_only=False):
        super().__init__()
        if type(profile) is not str or profile not in PROFILES or type(test_only) is not bool:
            raise ValueError("Require profile baseline or spatial and an explicit boolean test_only")
        if type(adapter) is not PostBlockActionAdapter:
            raise ValueError("Use the existing unchanged PostBlockActionAdapter")
        if (not isinstance(core, nn.Module) or not hasattr(core, "patch_embedding")
                or not hasattr(core, "head") or not callable(getattr(core, "unpatchify", None))):
            raise ValueError("Require a populated native core with its original head and unpatchify")
        self.core, self.adapter = core, adapter
        self.profile, self.test_only = profile, test_only
        self._owner = object()
        self._call_lock = threading.Lock()
        self._check_core()
        self.core.eval()

    @property
    def shape(self):
        return (self.core.in_dim, *PROFILES[self.profile]["shape"][1:])

    @property
    def tokens(self):
        return PROFILES[self.profile]["tokens"]

    @property
    def grid(self):
        _, frames, height, width = self.shape
        return (frames, height // 2, width // 2)

    def train(self, mode=True):
        super().train(mode)
        self.core.eval()
        return self

    @contextmanager
    def _exclusive(self):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent or recursive calls to this bridge are unsupported")
        try:
            yield
        finally:
            self._call_lock.release()

    def _autocast(self):
        # CPU fixtures never request CUDA initialization or imitate FA2.
        return nullcontext() if self.test_only else torch.autocast("cuda", dtype=torch.bfloat16)

    def _check_core(self):
        parameters = tuple(self.core.parameters())
        if not parameters:
            raise ValueError("Core parameters must be populated")
        device = self.core.patch_embedding.weight.device
        if any(p.device != device or p.dtype != torch.float32 or p.requires_grad
               or p.grad is not None or p.is_inference() for p in parameters):
            raise ValueError("All core parameters must be normal frozen FP32 tensors with no gradients on one device")
        if tuple(self.core.patch_size) != (1, 2, 2) or self.core.model_type != "ti2v":
            raise ValueError("Require the native TI2V spatial patch architecture")
        if self.core.in_dim != self.core.out_dim:
            raise ValueError("Native input/output latent channels must match")
        for module in self.core.modules():
            if module._forward_pre_hooks or module._forward_hooks or "forward" in module.__dict__:
                raise ValueError("The native core must have no hooks or instance forward overrides")
        if "unpatchify" in self.core.__dict__:
            raise ValueError("Native unpatchify must not be replaced")
        if self.adapter.hidden_dim != self.core.dim or self.adapter.observation_channels != self.core.in_dim:
            raise ValueError("Adapter dimensions do not match the native core")
        if self.adapter._parameter_device() != device or any(not p.requires_grad for p in self.adapter.parameters()):
            raise ValueError("The existing FP32 adapter must be trainable on the core device")
        if self.test_only:
            if (device.type != "cpu" or self.core.dim > 32 or self.core.in_dim > 4
                    or len(self.core.blocks) > 2 or self.adapter.width > 16):
                raise ValueError("test_only accepts bounded small CPU fixtures only")
        else:
            # Reject a CPU/meta core before consulting any CUDA runtime state.
            if device.type != "cuda":
                raise ValueError("Production requires the native CUDA core; there is no CPU fallback")
            from ..cuda_reference import native
            native.verify_sources()
            from ..cuda_reference.vendor import attention
            from ..cuda_reference.vendor.model import WanModel, Head, WanAttentionBlock
            if (type(self.core) is not WanModel or type(self.core.head) is not Head
                    or any(type(block) is not WanAttentionBlock for block in self.core.blocks)):
                raise ValueError("Production requires the literal pinned native model, blocks and head")
            if (not attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE):
                raise RuntimeError("Production requires upstream FlashAttention 2 only")
            if (len(parameters) != 825 or self.core.in_dim != 48 or self.core.dim != 3072
                    or len(self.core.blocks) != 30 or self.core.text_dim != 4096 or self.core.text_len != 512
                    or self.adapter.width != 128
                    or sum(p.numel() for p in self.adapter.parameters()) != NATIVE_PARAMETER_COUNT):
                raise ValueError("Production requires exact native 5B and 947712-parameter adapter dimensions")
        return device

    def _inputs(self, noisy, times, contexts):
        device = self._check_core()
        _float_tensor(noisy, "noisy", (1, *self.shape), device)
        if (not isinstance(times, torch.Tensor) or times.layout != torch.strided
                or times.dtype != torch.int64 or times.device != device
                or times.shape != (1, self.tokens) or times.is_inference()):
            raise ValueError("times must be normal int64[1,tokens] on the core device")
        prefix = PROFILES[self.profile]["prefix"]
        future_time = int(times[0, prefix])
        if (not 0 <= future_time <= 999 or not bool((times[:, :prefix] == 0).all())
                or not bool((times[:, prefix:] == future_time).all())):
            raise ValueError("Observed times must be zero; future times must share one integer in [0,999]")
        if not isinstance(contexts, list) or len(contexts) != 1:
            raise ValueError("contexts must contain exactly one genuine text tensor")
        context = contexts[0]
        if (not isinstance(context, torch.Tensor) or context.ndim != 2
                or not 1 <= context.shape[0] <= self.core.text_len):
            raise ValueError("Text length must lie within the native limits")
        _float_tensor(context, "context", (context.shape[0], self.core.text_dim), device)
        return device

    def _conditions(self, commands, observation, prefix):
        device = self.core.patch_embedding.weight.device
        _float_tensor(commands, "commands", (1, 16, 6), device)
        if not bool(((commands[..., 5] == 0) | (commands[..., 5] == 1)).all()):
            raise ValueError("Interaction commands must be zero or one-transition pulses")
        _float_tensor(observation, "observation", (1, self.shape[0], 1, *self.shape[-2:]), device)
        if not torch.equal(observation, prefix):
            raise ValueError("Observation must equal the exact clean initial prefix")

    def extract_features(self, noisy, times, contexts):
        """Run the original core to its final head under no_grad; no target input."""
        with self._exclusive():
            self._inputs(noisy, times, contexts)
            captured = {}
            boundary = _HeadBoundary()

            def capture(module, args, kwargs):
                if module is not self.core.head or len(args) != 2 or kwargs or captured:
                    raise RuntimeError("The literal native head boundary changed")
                captured["hidden"], captured["time"] = args
                raise boundary

            handle = self.core.head.register_forward_pre_hook(capture, with_kwargs=True, prepend=True)
            try:
                with torch.inference_mode(False), torch.no_grad(), self._autocast():
                    self.core(list(noisy.unbind(0)), times, contexts, self.tokens)
            except _HeadBoundary as error:
                if error is not boundary:
                    raise
                # The traceback otherwise retains the entire native forward
                # frame (and a cycle through this local sentinel) until cyclic
                # GC. Keep only the two intentional captured tensor values.
                error.__traceback__ = None
            else:
                raise RuntimeError("Native forward did not reach the declared head boundary")
            finally:
                handle.remove()
            if set(captured) != {"hidden", "time"}:
                raise RuntimeError("The native head did not supply both tensors")
            # Fixed (1,2,2) patches and an unpadded input uniquely determine the
            # CPU grid metadata used by the unchanged native unpatchify method.
            with torch.inference_mode(False), torch.no_grad():
                result = FrozenNativeFeatures(
                    captured["hidden"].detach(), captured["time"].detach(),
                    torch.tensor([self.grid], dtype=torch.int64, device="cpu"),
                    noisy[:, :, :1].detach().clone(), self.profile, self._owner,
                )
            self._features(result)
            return result

    def _features(self, value):
        device = self._check_core()
        if (not isinstance(value, FrozenNativeFeatures) or value.owner is not self._owner
                or value.profile != self.profile):
            raise ValueError("Feature bundle belongs to another bridge or profile")
        for name, tensor in (("hidden", value.hidden), ("time_embedding", value.time_embedding)):
            _float_tensor(tensor, name, (1, self.tokens, self.core.dim), device)
            if tensor.requires_grad or tensor.grad_fn is not None:
                raise ValueError("Frozen features must be non-gradient constants")
        grid = value.grid_sizes
        if (not isinstance(grid, torch.Tensor) or grid.dtype != torch.int64 or grid.device.type != "cpu"
                or grid.shape != (1, 3) or grid.tolist() != [list(self.grid)]):
            raise ValueError("Feature grid must exactly match the declared native profile")
        _float_tensor(value.observed_prefix, "observed_prefix", (1, self.shape[0], 1, *self.shape[-2:]), device)
        if value.observed_prefix.requires_grad or value.observed_prefix.grad_fn is not None:
            raise ValueError("Captured prefix must be a non-gradient constant")

    def predict_from_features(self, features, commands, observation, *, track_grad=True):
        """Apply the existing FP32 adapter, original head and native unpatchify.

        track_grad=True explicitly enables the adapter/head graph, including
        inside an outer no_grad context. False is an explicit inference choice.
        Returned predictions stay on the core device; nothing is detached or
        copied to CPU when tracking gradients.
        """
        if type(track_grad) is not bool:
            raise ValueError("track_grad must be an explicit boolean")
        with self._exclusive():
            self._features(features)
            self._conditions(commands, observation, features.observed_prefix)
            with torch.inference_mode(False), torch.set_grad_enabled(track_grad), self._autocast():
                # Existing adapter internally disables autocast for its FP32
                # arithmetic. Original Head keeps its own native FP32 context.
                adapted = self.adapter(features.hidden, commands, observation, self.grid)
                patches = self.core.head(adapted, features.time_embedding)
                output = torch.stack([value.float() for value in self.core.unpatchify(patches, features.grid_sizes)])
            if output.shape != (1, *self.shape) or output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
                raise FloatingPointError("Native head must return finite FP32 velocity at the declared shape")
            return output

    def forward(self, noisy, times, contexts, *, commands, observation, track_grad=True):
        # Reject malformed action/observation inputs before running 30 blocks.
        self._inputs(noisy, times, contexts)
        self._conditions(commands, observation, noisy[:, :, :1])
        if type(track_grad) is not bool:
            raise ValueError("track_grad must be an explicit boolean")
        features = self.extract_features(noisy, times, contexts)
        return self.predict_from_features(features, commands, observation, track_grad=track_grad)
