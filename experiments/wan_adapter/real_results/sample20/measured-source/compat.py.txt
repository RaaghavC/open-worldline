"""Original CPU/MPS compatibility substitutions for the attributed Wan core.

Only the bidirectional, non-windowed SDPA path is supported. Positional math is
checked against upstream complex128 CPU RoPE in test_cpu.py.
"""
import torch
import torch.nn.functional as F
import torch.utils.checkpoint


def real_rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    positions = torch.arange(max_seq_len, device='cpu', dtype=torch.float64)
    frequencies = 1.0 / torch.pow(theta, torch.arange(0, dim, 2, device='cpu', dtype=torch.float64) / dim)
    phases = torch.outer(positions, frequencies)
    return torch.stack((phases.cos(), phases.sin()), dim=-1).float()


def real_rope_apply(x, grid_sizes, freqs):
    pairs = x.shape[-1] // 2
    tables = freqs.split([pairs - 2 * (pairs // 3), pairs // 3, pairs // 3], dim=1)
    outputs = []
    for i, (frames, height, width) in enumerate(grid_sizes.tolist()):
        length = frames * height * width
        rotation = torch.cat([
            tables[0][:frames].view(frames, 1, 1, -1, 2).expand(frames, height, width, -1, 2),
            tables[1][:height].view(1, height, 1, -1, 2).expand(frames, height, width, -1, 2),
            tables[2][:width].view(1, 1, width, -1, 2).expand(frames, height, width, -1, 2),
        ], dim=-2).reshape(length, 1, pairs, 2)
        value = x[i, :length].float().reshape(length, x.shape[2], pairs, 2)
        real, imag = value.unbind(-1)
        cos, sin = rotation.unbind(-1)
        rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).flatten(2)
        outputs.append(torch.cat((rotated.to(x.dtype), x[i, length:]), dim=0))
    return torch.stack(outputs)


def cpu_timestep_embedding(dim, position):
    # Tiny CPU operation preserves upstream fp64 computation without MPS doubles.
    from vendor.common.embedding import sinusoidal_embedding
    result = sinusoidal_embedding(position.to('cpu'), dim, compute_dtype=torch.float64, pad_odd=False)
    return result.float().to(position.device)


def portable_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                       softmax_scale=None, q_scale=None, causal=False,
                       window_size=(-1, -1), deterministic=False, dtype=None,
                       fa_version=None):
    if causal or tuple(window_size) != (-1, -1):
        raise NotImplementedError('This probe implements bidirectional global attention only')
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError('Query, key and value must share a dtype')
    mask = None
    if k_lens is not None and bool((k_lens != k.shape[1]).any()):
        lens = k_lens.to(q.device)
        mask = torch.arange(k.shape[1], device=q.device)[None, None, None, :] < lens[:, None, None, None]
    qq = q.transpose(1, 2)
    if q_scale is not None:
        qq = qq * q_scale
    out = F.scaled_dot_product_attention(qq, k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, dropout_p=dropout_p, is_causal=False, scale=softmax_scale)
    out = out.transpose(1, 2).contiguous()
    if q_lens is not None and bool((q_lens != q.shape[1]).any()):
        valid = torch.arange(q.shape[1], device=q.device)[None, :] < q_lens.to(q.device)[:, None]
        out = out * valid[:, :, None, None]
    return out


def install():
    from vendor.wan21 import model, attention
    from vendor.wan21.layers import rope
    model.rope_params = real_rope_params
    model.sinusoidal_embedding_1d = cpu_timestep_embedding
    rope.rope_apply = real_rope_apply
    attention.attention = portable_attention
