# SPDX-License-Identifier: Apache-2.0
"""Reusable frozen prefix for the separately tested intermediate action bridge.

Native preprocessing and the selected prefix execute once. The cache contains
no commands or targets; each prediction executes a fresh adapter/native suffix
graph. This file does not load weights, change native equations or admit CUDA.
"""
from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping

import torch

from experiments.wan22_native.intermediate_action.bridge import IntermediateActionBridge


BLOCK_KEYS = frozenset({"e", "seq_lens", "grid_sizes", "freqs", "context", "context_lens"})


class _PrefixBoundary(BaseException):
    """Private per-extraction sentinel, caught by object identity only."""


@dataclass(frozen=True)
class FrozenIntermediateFeatures:
    hidden: torch.Tensor
    time_embedding: torch.Tensor
    block_kwargs: Mapping
    observed_prefix: torch.Tensor
    profile: str
    block_index: int
    owner: object = field(repr=False, compare=False)
    core_stamp: tuple = field(repr=False, compare=False)
    tensor_stamp: tuple = field(repr=False, compare=False)

    @property
    def grid_sizes(self):
        return self.block_kwargs["grid_sizes"]


def _constant(value, name, shape, dtype, device):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or tuple(value.shape) != tuple(shape) or value.dtype != dtype or value.device != device
            or value.requires_grad or value.grad_fn is not None or value.is_inference()):
        raise ValueError(f"{name} must be a normal frozen {dtype} tensor of shape {tuple(shape)} on {device}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _tensor_stamp(hidden, time_embedding, block_kwargs, observed_prefix):
    values = [("hidden", hidden), ("time_embedding", time_embedding), ("observed_prefix", observed_prefix)]
    values += [(name, block_kwargs[name]) for name in sorted(BLOCK_KEYS) if block_kwargs[name] is not None]
    return tuple((name, id(value), value._version) for name, value in values)


class CachedIntermediateActionBridge(IntermediateActionBridge):
    """Keep inherited full forward and add the prior auxiliary's cache API.

    A feature bundle belongs to one live wrapper, profile, block index and
    frozen core parameter/buffer version. Adapter optimization is allowed
    between uses. Bundles are ephemeral, read-only constants, not serialized
    datasets. External value-hash verification of foundation weights remains
    the caller's responsibility; version checks are not replacement hashes.
    """

    def __init__(self, core, adapter, *, block_index=28, profile="baseline", test_only=False):
        super().__init__(core, adapter, block_index=block_index, profile=profile, test_only=test_only)
        self._cache_owner = object()

    def _core_stamp(self):
        values = list(self.core.named_parameters()) + list(self.core.named_buffers()) + [("rotary_frequencies", self.core.freqs)]
        return tuple((name, id(value), value._version) for name, value in values)

    def _features(self, features):
        device = self._contract._check_core()
        self._check_site()
        if (type(features) is not FrozenIntermediateFeatures or features.owner is not self._cache_owner
                or features.profile != self._contract.profile or type(features.block_index) is not int
                or features.block_index != self.block_index):
            raise ValueError("Features must belong to this owner, profile and block index")
        if features.core_stamp != self._core_stamp():
            raise ValueError("Frozen core parameters or buffers changed after extraction")
        kwargs = features.block_kwargs
        if not isinstance(kwargs, Mapping) or set(kwargs) != BLOCK_KEYS or kwargs["context_lens"] is not None:
            raise ValueError("Features require the exact native block keyword arguments")
        _constant(features.hidden, "hidden", (1, self.tokens, self.core.dim), torch.float32, device)
        _constant(features.time_embedding, "raw time embedding", (1, self.tokens, self.core.dim), torch.float32, device)
        _constant(kwargs["e"], "projected block time embedding", (1, self.tokens, 6, self.core.dim), torch.float32, device)
        context_dtype = torch.float32 if self._contract.test_only else torch.bfloat16
        _constant(kwargs["context"], "native projected text", (1, self.core.text_len, self.core.dim), context_dtype, device)
        _constant(kwargs["grid_sizes"], "grid_sizes", (1, 3), torch.int64, torch.device("cpu"))
        _constant(kwargs["seq_lens"], "seq_lens", (1,), torch.int64, torch.device("cpu"))
        _constant(kwargs["freqs"], "native rotary frequencies", tuple(self.core.freqs.shape), self.core.freqs.dtype, device)
        if kwargs["grid_sizes"].tolist() != [list(self.grid)] or kwargs["seq_lens"].tolist() != [self.tokens]:
            raise ValueError("Cached grid/sequence length must match the declared native profile")
        if kwargs["freqs"] is not self.core.freqs:
            raise ValueError("Cached rotary frequencies must be the original live core constant")
        _constant(features.observed_prefix, "observed prefix", (1, self.shape[0], 1, *self.shape[-2:]), torch.float32, device)
        if features.tensor_stamp != _tensor_stamp(features.hidden, features.time_embedding, kwargs, features.observed_prefix):
            raise ValueError("Cached constants were replaced or mutated after extraction")

    def extract_features(self, noisy, times, contexts):
        """Capture the literal native prefix once, without any adapter or target."""
        with self._contract._exclusive():
            self._contract._inputs(noisy, times, contexts)
            self._check_site()
            if any(value.requires_grad or value.grad_fn is not None for value in (noisy, times, *contexts)):
                raise ValueError("Prefix inputs must be non-gradient constants")
            stamp = self._core_stamp()
            captured = {}
            boundary = _PrefixBoundary()
            selected = self.core.blocks[self.block_index]

            def capture_time(module, args, output):
                if module is not self.core.time_embedding or len(args) != 1 or "time" in captured:
                    raise RuntimeError("Native raw time embedding must execute exactly once")
                captured["time"] = output

            def capture_block(module, args, kwargs, output):
                if (module is not selected or len(args) != 1 or set(kwargs) != BLOCK_KEYS
                        or "hidden" in captured or "time" not in captured):
                    raise RuntimeError("Native prefix boundary or execution order changed")
                captured["hidden"] = output
                captured["kwargs"] = MappingProxyType(dict(kwargs))
                raise boundary

            handles = []
            try:
                handles.append(self.core.time_embedding.register_forward_hook(capture_time))
                handles.append(selected.register_forward_hook(capture_block, with_kwargs=True))
                with torch.inference_mode(False), torch.no_grad(), self._contract._autocast():
                    self.core(list(noisy.unbind(0)), times, contexts, self.tokens)
            except _PrefixBoundary as error:
                if error is not boundary:
                    raise
                # Do not retain the original forward's frame and intermediates
                # through the sentinel traceback cycle after prefix extraction.
                error.__traceback__ = None
            else:
                raise RuntimeError("Native forward did not stop at the selected prefix boundary")
            finally:
                for handle in reversed(handles):
                    handle.remove()
            if set(captured) != {"hidden", "time", "kwargs"}:
                raise RuntimeError("Incomplete native prefix capture")
            # Literal Wan forward lazily moves its unregistered rotary tensor
            # to the model device on the first call. Bind that resulting live
            # tensor, while rejecting changes to parameters/registered buffers.
            after = self._core_stamp()
            if after[:-1] != stamp[:-1]:
                raise ValueError("Frozen core parameters or buffers changed during extraction")
            with torch.inference_mode(False), torch.no_grad():
                prefix = noisy[:, :, :1].clone()
                features = FrozenIntermediateFeatures(
                    captured["hidden"], captured["time"], captured["kwargs"], prefix,
                    self._contract.profile, self.block_index, self._cache_owner, after,
                    _tensor_stamp(captured["hidden"], captured["time"], captured["kwargs"], prefix),
                )
            self._features(features)
            return features

    def predict_from_features(self, features, commands, observation, *, track_grad=True):
        """Build a fresh adapter/suffix graph from one unchanged frozen cache."""
        if type(track_grad) is not bool:
            raise ValueError("track_grad must be an explicit boolean")
        with self._contract._exclusive():
            self._features(features)
            self._contract._conditions(commands, observation, features.observed_prefix)
            if any(value.requires_grad or value.grad_fn is not None for value in (commands, observation)):
                raise ValueError("Commands and observation must be non-gradient constants")
            with torch.inference_mode(False), torch.set_grad_enabled(track_grad), self._contract._autocast():
                hidden = self.adapter(features.hidden, commands, observation, self.grid)
                for block in self.core.blocks[self.block_index+1:]:
                    hidden = block(hidden, **features.block_kwargs)
                patches = self.core.head(hidden, features.time_embedding)
                values = self.core.unpatchify(patches, features.grid_sizes)
                output = torch.stack([value.float() for value in values])
            if (output.shape != (1, *self.shape) or output.dtype != torch.float32
                    or not bool(torch.isfinite(output).all())):
                raise FloatingPointError("Require a finite FP32 native suffix output at the declared shape")
            self._features(features)
            return output
