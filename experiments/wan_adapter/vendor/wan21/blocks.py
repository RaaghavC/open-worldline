# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan transformer blocks (bidirectional and causal)."""

import math

import torch
import torch.nn as nn

from vendor.single_process import sp_all_to_all_4D
from vendor.single_process import get_parallel_state
from vendor.wan21.layers.norm import WanLayerNorm

from .attention import WAN_CROSSATTENTION_CLASSES, WanSelfAttention

__all__ = ["WanAttentionBlock"]


class WanAttentionBlock(nn.Module):
    """Full DiT transformer block: self-attn + cross-attn + FFN + AdaLN modulation.

    Args:
        cross_attn_type (str): key into ``WAN_CROSSATTENTION_CLASSES``.
        dim (int): model hidden dimension.
        ffn_dim (int): FFN intermediate dimension.
        num_heads (int): number of attention heads.
        window_size (tuple): local attention window for self-attention.
        qk_norm (bool): QK normalisation.
        cross_attn_norm (bool): extra LayerNorm before cross-attention.
        eps (float): epsilon for norms.
        use_prope (bool): enable PRoPE camera conditioning on the self-attention.
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        use_prope: bool = False,
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, use_prope)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, viewmats=None, Ks=None
    ):
        """
        Args:
            x (Tensor): token features, shape ``[B, L, C]``.
            e (Tensor): AdaLN conditioning, shape ``[B, 6, C]``.
            seq_lens (Tensor): valid sequence lengths, shape ``[B]``.
            grid_sizes (Tensor): ``(F, H, W)`` grids, shape ``[B, 3]``.
            freqs (Tensor): RoPE frequencies.
            context (Tensor): text/image context, shape ``[B, T, C]``.
            context_lens (Tensor, optional): valid context lengths.
            viewmats (Tensor, optional): camera extrinsics for PRoPE.
            Ks (Tensor, optional): camera intrinsics for PRoPE.

        Returns:
            Tensor: output tokens, shape ``[B, L, C]``.
        """
        e = (self.modulation + e).chunk(6, dim=1)
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs, viewmats=viewmats, Ks=Ks
        )
        x = x + y * e[2]
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        x = x + self.ffn(self.norm2(x) * (1 + e[4]) + e[3]) * e[5]
        return x

