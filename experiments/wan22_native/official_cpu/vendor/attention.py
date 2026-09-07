# SPDX-License-Identifier: Apache-2.0
"""CPU SDPA replacement for CUDA FlashAttention, independently specified here."""
import torch
from torch.nn import functional as F


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                    softmax_scale=None, q_scale=None, causal=False,
                    window_size=(-1,-1), deterministic=False,
                    dtype=torch.bfloat16, version=None):
    if any(x.device.type != 'cpu' or x.ndim != 4 for x in (q,k,v)):
        raise ValueError('CPU [B,L,heads,channels] attention only')
    if dtype != torch.bfloat16 or causal or tuple(window_size) != (-1,-1) or dropout_p != 0.:
        raise ValueError('Only native noncausal BF16 zero-dropout attention')
    if q_lens is not None or q_scale is not None or softmax_scale is not None:
        raise ValueError('Unused attention options must stay absent')
    original = q.dtype
    # Official half(): retain existing half inputs; otherwise choose BF16.
    half = lambda x: x if x.dtype in (torch.float16,torch.bfloat16) else x.to(dtype)
    q,k,v = half(q),half(k),half(v)
    q,k = q.to(v.dtype),k.to(v.dtype)
    if v.dtype != torch.bfloat16:
        raise ValueError('This reference requires BF16 attention computation')
    mask = None
    if k_lens is not None:
        if k_lens.shape != (k.shape[0],) or (k_lens < 1).any() or (k_lens > k.shape[1]).any():
            raise ValueError('Invalid key lengths')
        mask = torch.arange(k.shape[1])[None,None,None,:] < k_lens[:,None,None,None]
    # Explicit inputs make this substitution independent of outer autocast policy.
    with torch.autocast('cpu',enabled=False):
        result = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
            attn_mask=mask,dropout_p=0.,is_causal=False)
    return result.transpose(1,2).contiguous().to(original)
