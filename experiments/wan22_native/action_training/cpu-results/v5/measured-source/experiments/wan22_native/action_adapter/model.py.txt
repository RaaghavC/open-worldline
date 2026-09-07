# SPDX-License-Identifier: Apache-2.0
"""Original post-final-block conditioning; not a persistent visual-memory model."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .pooling import observation_pool2d

NATIVE_PARAMETER_COUNT = 947_712


def checked_float(value, *, name, shape=None, device=None):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError(name + " must be an FP32 tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(name + " has an unexpected shape")
    if device is not None and value.device != device:
        raise ValueError(name + " must be on the declared device")
    if not torch.isfinite(value).all().item():
        raise ValueError(name + " must be finite")


class PostBlockActionAdapter(nn.Module):
    """Requested-command history plus first-observation attention, all FP32.

    Default dimensions have exactly 947,712 trainable parameters. Explicit
    smaller dimensions exist only for bounded CPU fixtures. Every call resets
    the command GRU; no hidden state is stored between sessions or clips.
    """
    def __init__(self, hidden_dim=3072, observation_channels=48, width=128):
        super().__init__()
        for value in (hidden_dim, observation_channels, width):
            if type(value) is not int or value < 1:
                raise ValueError("Adapter dimensions must be positive integers")
        self.hidden_dim, self.observation_channels, self.width = hidden_dim, observation_channels, width
        self.action = nn.Sequential(nn.Linear(24, width), nn.SiLU(), nn.Linear(width, width))
        self.command_gru = nn.GRUCell(width, width)
        self.observation = nn.Linear(observation_channels, width)
        self.query = nn.Linear(hidden_dim, width)
        self.key, self.value = nn.Linear(width, width), nn.Linear(width, width)
        self.output = nn.Linear(width, hidden_dim)
        self.float()
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _parameter_device(self):
        values = tuple(self.parameters())
        device = values[0].device
        if any(p.dtype != torch.float32 or p.device != device for p in values):
            raise ValueError("All adapter parameters must remain FP32 on one device")
        return device

    def command_prefix(self, commands):
        """Return [B,1+N/4,width]; position zero is reserved, never a GRU step."""
        device = self._parameter_device()
        checked_float(commands, name="commands", device=device)
        if commands.ndim != 3 or commands.shape[0] < 1 or commands.shape[1] < 4 or commands.shape[1] % 4 or commands.shape[2] != 6:
            raise ValueError("commands must be [B,4*K,6] for positive B,K")
        if not ((commands[..., 5] == 0) | (commands[..., 5] == 1)).all().item():
            raise ValueError("Interaction commands must be zero or one-transition pulses")
        with torch.autocast(device_type=device.type, enabled=False):
            groups = commands.reshape(commands.shape[0], commands.shape[1] // 4, 24)
            embedded = self.action(groups)
            state = embedded.new_zeros(commands.shape[0], self.width)
            states = [state]
            for group in embedded.unbind(1):
                state = self.command_gru(group, state)
                states.append(state)
            return torch.stack(states, dim=1)

    def forward(self, features, commands, observation, grid):
        """Mask the initial latent group and optional padding, in F,H,W order."""
        device = self._parameter_device()
        if (not isinstance(grid, (tuple, list)) or len(grid) != 3
                or any(type(v) is not int or v < 1 for v in grid) or grid[0] < 2):
            raise ValueError("grid must contain positive F,H,W with future frames")
        frames, height, width = grid
        checked_float(features, name="features", device=device)
        if (features.ndim != 3 or features.shape[0] < 1 or features.shape[1] < math.prod(grid)
                or features.shape[2] != self.hidden_dim):
            raise ValueError("features must be [B,L,hidden_dim] with sufficient token length")
        checked_float(commands, name="commands", shape=(features.shape[0], 4*(frames-1), 6), device=device)
        checked_float(observation, name="observation",
                      shape=(features.shape[0], self.observation_channels, 1, height*2, width*2), device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            command = self.command_prefix(commands)
            command = command[:, :, None].expand(-1, -1, height*width, -1).flatten(1, 2)
            command = F.pad(command, (0, 0, 0, features.shape[1]-command.shape[1]))
            observed = observation_pool2d(observation[:, :, 0], (4, 8)).flatten(2).transpose(1, 2)
            observed = self.observation(observed)
            attended = F.scaled_dot_product_attention(
                self.query(features).unsqueeze(1), self.key(observed).unsqueeze(1),
                self.value(observed).unsqueeze(1), dropout_p=0., is_causal=False).squeeze(1)
            delta = self.output(F.silu(command + attended))
            positions = torch.arange(features.shape[1], device=device)
            mask = ((positions >= height*width) & (positions < math.prod(grid)))[None, :, None]
            return features + delta * mask
