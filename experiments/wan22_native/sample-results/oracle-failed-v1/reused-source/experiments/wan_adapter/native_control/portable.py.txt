# SPDX-License-Identifier: Apache-2.0
"""Load literal official Wan layers with isolated CPU/MPS compatibility hooks.

Full float32 parameters and activations are the first diagnostic precision.
The reference and portable model modules have separate globals. Their network
source is the same unmodified, pinned official file; only attention/RoPE and a
small CPU-double timestep calculation are substituted for the portable path.
"""
import importlib.util
from pathlib import Path
import sys
import torch
import torch.nn.functional as F


def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
              softmax_scale=None, q_scale=None, causal=False,
              window_size=(-1, -1), deterministic=False, dtype=None, version=None):
    """FP32 global SDPA reference; no FlashAttention BF16 rounding is claimed."""
    if causal or tuple(window_size) != (-1, -1):
        raise NotImplementedError('Only bidirectional global attention is validated')
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise TypeError('The native-style FP32 control requires float32 Q/K/V')
    mask = None
    if k_lens is not None:
        mask = torch.arange(k.shape[1], device=k.device)[None, None, None, :] < k_lens.to(k.device)[:, None, None, None]
    query = q.transpose(1, 2)
    if q_scale is not None:
        query = query*q_scale
    output = F.scaled_dot_product_attention(query, k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, dropout_p=dropout_p, is_causal=False, scale=softmax_scale).transpose(1, 2).contiguous()
    if q_lens is not None:
        valid = torch.arange(q.shape[1], device=q.device)[None, :] < q_lens.to(q.device)[:, None]
        output = output*valid[:, :, None, None]
    return output


def real_rope_params(max_seq_len, dim, theta=10000):
    if dim % 2:
        raise ValueError('RoPE dimensions must be even')
    positions = torch.arange(max_seq_len, device='cpu', dtype=torch.float64)
    rates = 1. / torch.pow(theta, torch.arange(0, dim, 2, device='cpu', dtype=torch.float64)/dim)
    angle = torch.outer(positions, rates)
    return torch.stack((angle.cos(), angle.sin()), dim=-1).float()


def real_rope_apply(x, grid_sizes, freqs):
    heads, pairs = x.size(2), x.size(3)//2
    tables = freqs.split([pairs-2*(pairs//3), pairs//3, pairs//3], dim=1)
    result = []
    for index, (frames, height, width) in enumerate(grid_sizes.tolist()):
        length = frames*height*width
        rotation = torch.cat([
            tables[0][:frames].view(frames, 1, 1, -1, 2).expand(frames, height, width, -1, 2),
            tables[1][:height].view(1, height, 1, -1, 2).expand(frames, height, width, -1, 2),
            tables[2][:width].view(1, 1, width, -1, 2).expand(frames, height, width, -1, 2),
        ], dim=-2).reshape(length, 1, pairs, 2)
        real, imag = x[index, :length].float().reshape(length, heads, pairs, 2).unbind(-1)
        cos, sin = rotation.unbind(-1)
        rotated = torch.stack((real*cos-imag*sin, real*sin+imag*cos), dim=-1).flatten(2)
        result.append(torch.cat((rotated, x[index, length:].float()), dim=0))
    return torch.stack(result).float()


def cpu_sinusoidal_embedding(dim, position):
    if dim % 2:
        raise ValueError('Timestep embedding dimensions must be even')
    half = dim//2
    # Separate operations prevent PyTorch from attempting the double cast on MPS
    # before completing the device transfer.
    positions = position.to(device='cpu').to(dtype=torch.float64)
    sinusoid = torch.outer(positions, torch.pow(10000, -torch.arange(half).to(positions).div(half)))
    return torch.cat((torch.cos(sinusoid), torch.sin(sinusoid)), dim=1).float().to(position.device)


def load_model_module(*, portable):
    """Separate module globals prevent portable substitutions altering reference."""
    name = 'native_control.vendor.'+('_portable_model' if portable else '_reference_model')
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent/'vendor/model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.flash_attention = attention
    if portable:
        module.rope_params = real_rope_params
        module.rope_apply = real_rope_apply
        module.sinusoidal_embedding_1d = cpu_sinusoidal_embedding
    return module


def create_model(*, portable=True, **configuration):
    return load_model_module(portable=portable).WanModel(**configuration)
