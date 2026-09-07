# SPDX-License-Identifier: Apache-2.0
"""Explicit selective-BF16 execution of the pinned official Wan2.2 architecture.

The original parameter names and shapes are unchanged. The source copy remains
byte-exact; reviewed methods bind to each constructed instance only. No MPS
autocast, adapter, actions, hidden scene state or old16-channel codec is used.
"""
import json
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from experiments.wan_adapter.native_control.portable import (
    cpu_sinusoidal_embedding, real_rope_apply, real_rope_params)
from .vendor import model as official

POLICIES = ("selective_bf16", "fp32")
PARAMETER_COUNTS = {"float32": 68_573_376, "bfloat16": 4_931_214_336}
NATIVE_LATENT_SHAPE = (48, 5, 18, 32)
NATIVE_GRID = (5, 9, 16)
NATIVE_TOKENS = 720
NATIVE_OBSERVED_TOKENS = 144


def native_config():
    return {key: value for key, value in json.loads(Path(__file__).with_name("config.json").read_text()).items()
            if not key.startswith("_")}


def fp32_parameter_names(model):
    names = set()
    for name, child in model.named_modules():
        if isinstance(child, (official.WanRMSNorm, official.WanLayerNorm)):
            names.update(name + "." + key for key, _ in child.named_parameters(recurse=False))
    for name, _ in model.named_parameters():
        if name.startswith(("time_embedding.", "time_projection.", "head.")) or name.endswith(".modulation"):
            names.add(name)
    return names


def declared_dtypes(model, policy="selective_bf16"):
    if policy not in POLICIES:
        raise ValueError("Precision policy must be selective_bf16 or fp32")
    fp32 = fp32_parameter_names(model)
    return {name: torch.float32 if policy == "fp32" or name in fp32 else torch.bfloat16
            for name, _ in model.named_parameters()}


def storage_report(model):
    counts = {"float32": 0, "bfloat16": 0}
    for parameter in model.parameters():
        if parameter.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("Unexpected parameter dtype")
        counts[str(parameter.dtype).split(".")[-1]] += parameter.numel()
    return {"parameters": counts, "bytes": counts["float32"] * 4 + counts["bfloat16"] * 2,
            "total_parameters": sum(counts.values()), "policy": model._storage_policy}


def token_times(grid_sizes, timestep, seq_len, *, observed_latent_frames=1):
    """Native clean first-frame times in patch F,H,W order; padding stays future."""
    if grid_sizes.dtype != torch.int64 or grid_sizes.ndim != 2 or grid_sizes.shape[1] != 3:
        raise ValueError("grid_sizes must be int64[B,3]")
    if timestep.dtype != torch.int64 or timestep.shape != (len(grid_sizes),):
        raise ValueError("timestep must be int64[B]")
    if type(seq_len) is not int or seq_len < 1 or type(observed_latent_frames) is not int or observed_latent_frames < 0:
        raise ValueError("Invalid sequence length or observed-frame count")
    if (timestep < 0).any() or (timestep > 1000).any():
        raise ValueError("Times must lie in[0,1000]")
    values = []
    for index, (frames, height, width) in enumerate(grid_sizes.tolist()):
        if min(frames, height, width) <= 0 or frames * height * width > seq_len or observed_latent_frames >= frames:
            raise ValueError("Grid requires a future frame and enough padded tokens")
        row = timestep[index].expand(seq_len).clone()
        row[:observed_latent_frames * height * width] = 0
        values.append(row)
    return torch.stack(values)


def attention(q, k, v, *, k_lens=None, q_lens=None, window_size=(-1, -1), policy="selective_bf16"):
    """Global noncausal SDPA with official Q/K-to-V casts and query output dtype."""
    if tuple(window_size) != (-1, -1) or policy not in POLICIES:
        raise ValueError("Only declared global SDPA policies are supported")
    original_dtype = q.dtype
    dtype = torch.bfloat16 if policy == "selective_bf16" else torch.float32
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    mask = None
    if k_lens is not None:
        if k_lens.shape != (k.shape[0],) or (k_lens < 1).any() or (k_lens > k.shape[1]).any():
            raise ValueError("Invalid key sequence lengths")
        mask = torch.arange(k.shape[1], device=k.device)[None, None, None, :] < k_lens.to(k.device)[:, None, None, None]
    result = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, dropout_p=0., is_causal=False).transpose(1, 2).contiguous().to(original_dtype)
    if q_lens is not None:
        valid = torch.arange(q.shape[1], device=q.device)[None, :] < q_lens.to(q.device)[:, None]
        result = result * valid[:, :, None, None]
    return result


def _linear(self, value):
    return F.linear(value.to(self.weight.dtype), self.weight, self.bias)


def _patch(self, value):
    return self._conv_forward(value.to(self.weight.dtype), self.weight, self.bias)


def _self_attention(self, x, seq_lens, grid_sizes, freqs):
    batch, length = x.shape[:2]
    shape = (batch, length, self.num_heads, self.head_dim)
    q, k, v = self.norm_q(self.q(x)).view(shape), self.norm_k(self.k(x)).view(shape), self.v(x).view(shape)
    out = attention(real_rope_apply(q, grid_sizes, freqs), real_rope_apply(k, grid_sizes, freqs), v,
                    k_lens=seq_lens, window_size=self.window_size, policy=self._storage_policy)
    return self.o(out.flatten(2))


def _cross_attention(self, x, context, context_lens):
    shape = (x.shape[0], -1, self.num_heads, self.head_dim)
    q, k, v = self.norm_q(self.q(x)).view(shape), self.norm_k(self.k(context)).view(shape), self.v(context).view(shape)
    return self.o(attention(q, k, v, k_lens=context_lens, policy=self._storage_policy).flatten(2))


def _block(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
    if e.dtype != torch.float32:
        raise TypeError("Time modulation must remain FP32")
    parts = (self.modulation.unsqueeze(0) + e).unbind(dim=2)
    y = self.self_attn(self.norm1(x).float() * (1 + parts[1]) + parts[0], seq_lens, grid_sizes, freqs)
    x = x.float() + y.float() * parts[2]
    x = x + self.cross_attn(self.norm3(x), context, context_lens).float()
    y = self.ffn(self.norm2(x).float() * (1 + parts[4]) + parts[3])
    return x + y.float() * parts[5]


def _head(self, x, e):
    if e.dtype != torch.float32:
        raise TypeError("Head time embedding must remain FP32")
    shift, scale = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).unbind(dim=2)
    return self.head(self.norm(x.float()).float() * (1 + scale) + shift).float()


def _forward(self, x, t, context, seq_len):
    device = self.patch_embedding.weight.device
    if device.type == "meta":
        raise ValueError("Meta model must be populated before inference")
    if not isinstance(x, list) or not x or not isinstance(context, list) or len(x) != len(context):
        raise ValueError("Latents and real text contexts must be matching nonempty lists")
    if type(seq_len) is not int or seq_len < 1:
        raise ValueError("seq_len must be positive")
    if t.dtype != torch.int64 or t.shape != (len(x), seq_len) or t.device != device or (t < 0).any() or (t > 1000).any():
        raise ValueError("Tokenwise times must be int64[B,L] on the model device within[0,1000]")
    for latent, text in zip(x, context):
        if (latent.ndim != 4 or latent.shape[0] != self.in_dim or latent.dtype != torch.float32 or latent.device != device
                or any(n % p for n, p in zip(latent.shape[1:], self.patch_size))):
            raise ValueError("Each latent must be FP32[C,F,H,W], divisible into exact patches")
        if (text.ndim != 2 or text.shape[1] != self.text_dim or not 0 < text.shape[0] <= self.text_len
                or text.dtype != torch.float32 or text.device != device):
            raise ValueError("Each real text context must be FP32[L,text_dim] within the native limit")
        if not torch.isfinite(latent).all() or not torch.isfinite(text).all():
            raise ValueError("Latent and text inputs must be finite")
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    patches = [self.patch_embedding(latent.unsqueeze(0)) for latent in x]
    grid = torch.tensor([value.shape[2:] for value in patches], dtype=torch.int64, device="cpu")
    tokens = [value.flatten(2).transpose(1, 2) for value in patches]
    lengths = torch.tensor([value.shape[1] for value in tokens], dtype=torch.int64, device="cpu")
    if lengths.max().item() > seq_len or grid.max().item() > self.freqs.shape[0]:
        raise ValueError("Insufficient token padding or rotary table length")
    tokens = torch.cat([torch.cat([value, value.new_zeros(1, seq_len - value.shape[1], value.shape[2])], dim=1) for value in tokens])
    embeddings = cpu_sinusoidal_embedding(self.freq_dim, t.flatten()).unflatten(0, (len(x), seq_len))
    e = self.time_embedding(embeddings.float())
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))
    # Native text padding occurs before projection; context_lens stays None.
    text = self.text_embedding(torch.stack([torch.cat([value, value.new_zeros(self.text_len - value.shape[0], self.text_dim)]) for value in context]))
    for block in self.blocks:
        tokens = block(tokens, e=e0, seq_lens=lengths, grid_sizes=grid, freqs=self.freqs, context=text, context_lens=None)
    return [value.float() for value in self.unpatchify(self.head(tokens, e), grid)]


def model_forward(self, x, t, context, seq_len):
    # All casts are explicit; callers cannot accidentally autocast the FP32 path.
    if self.patch_embedding.weight.device.type == "meta":
        raise ValueError("Meta model must be populated before inference")
    with torch.autocast(device_type=self.patch_embedding.weight.device.type, enabled=False):
        return _forward(self, x, t, context, seq_len)


def create_model(*, device="meta", policy="selective_bf16", configuration=None):
    configuration = dict(native_config() if configuration is None else configuration)
    if configuration.get("model_type") != "ti2v" or tuple(configuration.get("patch_size", (1, 2, 2))) != (1, 2, 2):
        raise ValueError("This port requires native TI2V with patch(1,2,2)")
    if str(device) != "meta" and (configuration.get("dim", 3072) > 128 or configuration.get("num_layers", 30) > 4):
        raise ValueError("Full-size model must be built on meta and streamed, never allocated as a full FP32 copy")
    with torch.device(device):
        model = official.WanModel(**configuration)
    dtypes = declared_dtypes(model, policy)
    for name, parameter in list(model.named_parameters()):
        parent, field = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), field, nn.Parameter(parameter.to(dtypes[name]), requires_grad=False))
    model._storage_policy = policy
    head_dim = model.dim // model.num_heads
    model.freqs = torch.cat([real_rope_params(1024, head_dim - 4 * (head_dim // 6)),
                            real_rope_params(1024, 2 * (head_dim // 6)),
                            real_rope_params(1024, 2 * (head_dim // 6))], dim=1)
    for child in model.modules():
        if isinstance(child, nn.Linear):
            child.forward = MethodType(_linear, child)
    model.patch_embedding.forward = MethodType(_patch, model.patch_embedding)
    for block in model.blocks:
        block.forward = MethodType(_block, block)
        block.self_attn._storage_policy = block.cross_attn._storage_policy = policy
        block.self_attn.forward = MethodType(_self_attention, block.self_attn)
        block.cross_attn.forward = MethodType(_cross_attention, block.cross_attn)
    model.head.forward = MethodType(_head, model.head)
    model.forward = MethodType(model_forward, model)
    return model.eval().requires_grad_(False)
