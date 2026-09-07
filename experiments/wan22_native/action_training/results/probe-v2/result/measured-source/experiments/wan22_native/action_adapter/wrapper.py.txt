# SPDX-License-Identifier: Apache-2.0
"""Explicit post-block29 adapter boundary. No hooks or core method replacement.

The frozen prefix repeats the measured portable forward equations up to the
head, using the original core modules and helpers. Wan's external source and
weights retain their own Apache attribution in the parent package.
"""
from dataclasses import dataclass, field
import math

import torch
from torch import nn

from ..portable import cpu_sinusoidal_embedding
from .model import NATIVE_PARAMETER_COUNT, PostBlockActionAdapter, checked_float


@dataclass(frozen=True)
class FrozenFeatures:
    hidden: torch.Tensor
    time_embedding: torch.Tensor
    grid_sizes: torch.Tensor
    sequence_lengths: torch.Tensor
    observed_prefix: torch.Tensor
    owner: object = field(repr=False, compare=False)


class NativeActionWrapper(nn.Module):
    """Extract frozen features, then differentiate adapter plus frozen head.

    Native calls accept FP32[B,48,5,18,32] and exactly 720 tokens. test_only=True
    allows tiny CPU native-architecture fixtures, not a production shape mode.
    The feature bundle is ephemeral and belongs to this wrapper instance; no
    feature serialization, cache trainer, multi-site adapter or optimizer exists.
    """
    def __init__(self, core, adapter=None, *, test_only=False):
        super().__init__()
        if type(test_only) is not bool:
            raise ValueError("test_only must be a boolean")
        if (any(p.requires_grad for p in core.parameters())
                or any(p.is_inference() for p in core.parameters())):
            raise ValueError("Core parameters must be frozen normal tensors, not inference tensors")
        if tuple(core.patch_size) != (1, 2, 2) or core.model_type != "ti2v":
            raise ValueError("Require the native TI2V patch architecture")
        if not test_only and (core.in_dim != 48 or core.out_dim != 48 or core.dim != 3072
                              or len(core.blocks) != 30 or core.text_dim != 4096 or core.text_len != 512):
            raise ValueError("Production wrapper requires exact native5B dimensions")
        if test_only and (core.patch_embedding.weight.device.type != "cpu" or core.dim > 128 or len(core.blocks) > 4):
            raise ValueError("test_only permits bounded CPU fixtures only")
        if core.patch_embedding.weight.device.type == "meta":
            raise ValueError("Core must be populated before wrapping")
        self.core = core.eval()
        self.adapter = adapter if adapter is not None else PostBlockActionAdapter()
        if self.adapter.hidden_dim != core.dim or self.adapter.observation_channels != core.in_dim:
            raise ValueError("Adapter and native core dimensions differ")
        if any(not p.requires_grad for p in self.adapter.parameters()):
            raise ValueError("Every adapter parameter must be trainable")
        if not test_only and (self.adapter.width != 128 or sum(p.numel() for p in self.adapter.parameters()) != NATIVE_PARAMETER_COUNT):
            raise ValueError("Production adapter must have exactly 947712 parameters at width128")
        self.test_only = test_only
        self._feature_owner = object()

    def train(self, mode=True):
        super().train(mode)
        self.core.eval()
        return self

    def _validate_inputs(self, noisy, times, contexts, seq_len):
        device = self.core.patch_embedding.weight.device
        if any(p.requires_grad for p in self.core.parameters()):
            raise ValueError("Core must remain frozen")
        checked_float(noisy, name="noisy latents", device=device)
        if noisy.ndim != 5 or noisy.shape[0] < 1 or noisy.shape[1] != self.core.in_dim:
            raise ValueError("noisy latents must be [B,C,F,H,W]")
        if not self.test_only and tuple(noisy.shape[1:]) != (48, 5, 18, 32):
            raise ValueError("Production latents must have native48-channel 17-frame shape")
        if noisy.shape[2] < 2 or min(noisy.shape[3:]) < 2 or any(v % 2 for v in noisy.shape[3:]):
            raise ValueError("Require future latent frames and exact spatial patches")
        grid = (noisy.shape[2], noisy.shape[3]//2, noisy.shape[4]//2)
        if type(seq_len) is not int or seq_len != math.prod(grid):
            raise ValueError("Wrapper accepts exactly the unpadded token count")
        if not isinstance(times, torch.Tensor) or times.dtype != torch.int64 or times.device != device or times.shape != (len(noisy), seq_len):
            raise ValueError("times must be int64[B,L] on the native device")
        first = grid[1]*grid[2]
        if (times < 0).any() or (times > 1000).any() or (times[:, :first] != 0).any():
            raise ValueError("Observed token times must be zero; all times must lie in [0,1000]")
        if not torch.equal(times[:, first:], times[:, first:first+1].expand(-1, seq_len-first)):
            raise ValueError("All future token times must share the branch's solver time")
        if not isinstance(contexts, list) or len(contexts) != len(noisy):
            raise ValueError("contexts must be a matching list of genuine text tensors")
        for text in contexts:
            checked_float(text, name="text context", device=device)
            if text.ndim != 2 or text.shape[1] != self.core.text_dim or not 1 <= text.shape[0] <= self.core.text_len:
                raise ValueError("Text shape exceeds native limits")
        return device, grid

    def extract_features(self, noisy, times, contexts, seq_len=720):
        """No actions or teacher targets; no gradient graph through any block."""
        device, grid = self._validate_inputs(noisy, times, contexts, seq_len)
        # Explicitly disable inference mode so these constants remain usable by
        # trainable Linear backward, even if an inference caller requests them.
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            core = self.core
            if core.freqs.device != device:
                core.freqs = core.freqs.to(device)
            patches = [core.patch_embedding(x.unsqueeze(0)) for x in noisy.unbind(0)]
            grids = torch.tensor([value.shape[2:] for value in patches], dtype=torch.int64, device="cpu")
            tokens = [value.flatten(2).transpose(1, 2) for value in patches]
            lengths = torch.tensor([value.shape[1] for value in tokens], dtype=torch.int64, device="cpu")
            if grids.max().item() > core.freqs.shape[0]:
                raise ValueError("Grid exceeds rotary table length")
            tokens = torch.cat(tokens)
            embedding = cpu_sinusoidal_embedding(core.freq_dim, times.flatten()).unflatten(0, (len(noisy), seq_len))
            e = core.time_embedding(embedding.float())
            e0 = core.time_projection(e).unflatten(2, (6, core.dim))
            text = core.text_embedding(torch.stack([
                torch.cat([value, value.new_zeros(core.text_len-value.shape[0], core.text_dim)]) for value in contexts]))
            for block in core.blocks:
                tokens = block(tokens, e=e0, seq_lens=lengths, grid_sizes=grids, freqs=core.freqs,
                               context=text, context_lens=None)
            result = FrozenFeatures(tokens, e, grids, lengths, noisy[:, :, :1].detach().clone(), self._feature_owner)
        for value in (result.hidden, result.time_embedding, result.observed_prefix):
            if value.requires_grad or value.grad_fn is not None or value.is_inference() or not torch.isfinite(value).all().item():
                raise RuntimeError("Frozen extraction must produce finite normal constants without a graph")
        return result

    def predict_from_features(self, features, commands, observation):
        """Autograd remains enabled here unless the caller explicitly disables it."""
        if not isinstance(features, FrozenFeatures) or features.owner is not self._feature_owner:
            raise ValueError("Feature bundle belongs to another wrapper")
        device = self.core.patch_embedding.weight.device
        if any(p.requires_grad for p in self.core.parameters()):
            raise ValueError("Core must remain frozen")
        hidden, e = features.hidden, features.time_embedding
        checked_float(hidden, name="frozen hidden", device=device)
        checked_float(e, name="frozen time", shape=hidden.shape, device=device)
        if any(v.requires_grad or v.grad_fn is not None or v.is_inference() for v in (hidden, e)):
            raise ValueError("Extracted features must remain normal non-gradient constants")
        if features.grid_sizes.dtype != torch.int64 or features.grid_sizes.shape != (len(hidden), 3):
            raise ValueError("Invalid feature grid metadata")
        if not torch.equal(features.grid_sizes, features.grid_sizes[:1].expand_as(features.grid_sizes)):
            raise ValueError("Batch grids differ")
        grid = tuple(features.grid_sizes[0].tolist())
        if hidden.shape != (len(features.grid_sizes), math.prod(grid), self.core.dim):
            raise ValueError("Invalid feature token dimensions")
        if (features.sequence_lengths.shape != (len(hidden),) or features.sequence_lengths.dtype != torch.int64
                or not (features.sequence_lengths == math.prod(grid)).all().item()):
            raise ValueError("Invalid feature sequence lengths")
        checked_float(observation, name="observation", shape=features.observed_prefix.shape, device=device)
        if not torch.equal(observation, features.observed_prefix):
            raise ValueError("Observation differs from the actual clean extraction prefix")
        with torch.autocast(device_type=device.type, enabled=False):
            adapted = self.adapter(hidden, commands, observation, grid)
            prediction = torch.stack(self.core.unpatchify(self.core.head(adapted, e), features.grid_sizes))
        if prediction.dtype != torch.float32 or not torch.isfinite(prediction).all().item():
            raise FloatingPointError("Predicted velocity must be finite FP32")
        return prediction

    def forward(self, noisy, times, contexts, *, commands, observation, seq_len=720):
        # Reject invalid conditions before the expensive frozen prefix runs.
        device, grid = self._validate_inputs(noisy, times, contexts, seq_len)
        if self.adapter._parameter_device() != device:
            raise ValueError("Adapter and native core must share one device")
        checked_float(commands, name="commands", shape=(len(noisy), 4*(grid[0]-1), 6), device=device)
        if not ((commands[..., 5] == 0) | (commands[..., 5] == 1)).all().item():
            raise ValueError("Interaction commands must be zero or one-transition pulses")
        checked_float(observation, name="observation", shape=noisy[:, :, :1].shape, device=device)
        if not torch.equal(observation, noisy[:, :, :1]):
            raise ValueError("Observation differs from the actual clean extraction prefix")
        features = self.extract_features(noisy, times, contexts, seq_len)
        return self.predict_from_features(features, commands, observation)
