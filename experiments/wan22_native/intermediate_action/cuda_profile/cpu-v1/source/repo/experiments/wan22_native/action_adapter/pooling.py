# SPDX-License-Identifier: Apache-2.0
"""Adaptive average pooling using explicit bins for the MPS observation path."""
import torch
from torch.nn import functional as F


def adaptive_bin_avg_pool2d(value, output_size):
    """Average the same floor/ceil bins as PyTorch, including their overlaps.

    This FP32 NCHW implementation keeps slices, reductions and gradients on the
    input device. It does not interpolate pixels or copy work to the CPU.
    """
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or value.ndim != 4 or any(size < 1 for size in value.shape)):
        raise ValueError("A nonempty FP32 NCHW tensor is required")
    if (not isinstance(output_size, (tuple, list)) or len(output_size) != 2
            or any(type(size) is not int or size < 1 for size in output_size)):
        raise ValueError("Output size must contain two positive integers")
    height, width = value.shape[-2:]
    out_height, out_width = output_size
    rows = []
    for row in range(out_height):
        top = row * height // out_height
        bottom = ((row + 1) * height + out_height - 1) // out_height
        columns = []
        for column in range(out_width):
            left = column * width // out_width
            right = ((column + 1) * width + out_width - 1) // out_width
            columns.append(value[..., top:bottom, left:right].mean(dim=(-2, -1)))
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=-2)


def observation_pool2d(value, output_size=(4, 8)):
    """Keep existing CPU behavior; use device-local explicit bins only on MPS."""
    if value.device.type == "mps":
        return adaptive_bin_avg_pool2d(value, output_size)
    return F.adaptive_avg_pool2d(value, output_size)
