# SPDX-License-Identifier: Apache-2.0
"""Command-gated low-rank self-attention updates; no foundation model loading."""
import math
import torch
from torch import nn
from torch.nn import functional as F

PROJECTIONS = ('q', 'k', 'v', 'o')
DEFAULT_BLOCKS = tuple(range(24, 30))
DEFAULT_PARAMETER_COUNT = 4_936_448


def constant(value, name, shape, dtype, device, *, finite=True):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or tuple(value.shape) != tuple(shape) or value.dtype != dtype or value.device != device
            or value.requires_grad or value.grad_fn is not None or value.is_inference()):
        raise ValueError(f'{name} must be a normal constant {dtype} tensor {tuple(shape)} on {device}')
    if finite and not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' must be finite')


class LowRankProjection(nn.Module):
    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.a = nn.Linear(hidden_dim, rank, bias=False)
        self.b = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.b.weight)

    def forward(self, value, gate):
        return self.b(self.a(value) * gate)


class CommandAttentionController(nn.Module):
    """One command encoder and distinct Q/K/V/O factors at each declared block.

    Four destination-aligned transitions form one future latent group. The GRU
    resets on every call. Gates are 1+tanh(linear(state)); B starts at zero.
    There is no LoRA alpha/rank multiplier and no persistent recurrent memory.
    """
    def __init__(self, hidden_dim=3072, rank=32, command_width=128, blocks=DEFAULT_BLOCKS):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (hidden_dim, rank, command_width)):
            raise ValueError('Dimensions must be positive integers')
        if (not isinstance(blocks, tuple) or not blocks or any(type(v) is not int or v < 0 for v in blocks)
                or tuple(sorted(set(blocks))) != blocks):
            raise ValueError('blocks must be a nonempty increasing tuple of distinct indices')
        self.hidden_dim, self.rank, self.command_width, self.blocks = hidden_dim, rank, command_width, blocks
        self.action = nn.Sequential(nn.Linear(24, command_width), nn.SiLU(), nn.Linear(command_width, command_width))
        self.command_gru = nn.GRUCell(command_width, command_width)
        self.gate = nn.Linear(command_width, len(blocks) * 4 * rank)
        self.projections = nn.ModuleDict({f'{b}_{p}': LowRankProjection(hidden_dim, rank) for b in blocks for p in PROJECTIONS})
        self.float()

    def parameter_device(self):
        values = tuple(self.parameters()); device = values[0].device
        if any(p.dtype != torch.float32 or p.device != device or not p.requires_grad or p.is_inference() for p in values):
            raise ValueError('Controller parameters must be trainable normal FP32 tensors on one device')
        return device

    def command_states(self, commands):
        device = self.parameter_device()
        constant(commands, 'commands', (1, 16, 6), torch.float32, device)
        if not bool(((commands[..., 5] == 0) | (commands[..., 5] == 1)).all()):
            raise ValueError('Interaction channel must contain binary transition pulses')
        with torch.autocast(device.type, enabled=False):
            embedded = self.action(commands.reshape(1, 4, 24))
            state = embedded.new_zeros(1, self.command_width)
            states = [state]
            for value in embedded.unbind(1):
                state = self.command_gru(value, state)
                states.append(state)
            return torch.stack(states, 1)

    def gates(self, commands, grid, sequence_length):
        if (not isinstance(grid, tuple) or len(grid) != 3 or grid[0] != 5
                or any(type(v) is not int or v < 1 for v in grid)
                or type(sequence_length) is not int or sequence_length < math.prod(grid)):
            raise ValueError('Require a five-group native grid and sufficient sequence length')
        states = self.command_states(commands)
        with torch.autocast(states.device.type, enabled=False):
            # [site, B, F, rank] then F,H,W token order, exactly as native patchify.
            values = (1 + torch.tanh(self.gate(states))).reshape(1, 5, len(self.blocks)*4, self.rank)
            values = values.permute(2, 0, 1, 3).repeat_interleave(grid[1]*grid[2], dim=2)
            values = F.pad(values, (0, 0, 0, sequence_length-math.prod(grid)))
            positions = torch.arange(sequence_length, device=states.device)
            mask = ((positions >= grid[1]*grid[2]) & (positions < math.prod(grid)))[None, None, :, None]
            return values * mask

    def residual(self, block, projection, value, gate):
        device = self.parameter_device()
        if (block not in self.blocks or projection not in PROJECTIONS or value.device != device
                or value.dtype not in (torch.float32, torch.bfloat16) or value.ndim != 3
                or value.shape[0] != 1 or value.shape[-1] != self.hidden_dim
                or tuple(gate.shape) != (1, value.shape[1], self.rank) or gate.dtype != torch.float32
                or gate.device != device or value.is_inference()):
            raise ValueError('Projection/gate does not match the declared controller')
        with torch.autocast(device.type, enabled=False):
            return self.projections[f'{block}_{projection}'](value.float(), gate)
