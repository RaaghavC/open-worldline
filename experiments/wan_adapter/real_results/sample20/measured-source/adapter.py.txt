"""Original action and observed-prefix conditioning for a frozen Wan clip model.

This module is newly implemented for Worldline. It is not a novelty claim.
The observation input contains only the first observed latent frame. It does
not implement persistent world memory or receive clean future video frames.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ActionObservationResidual(nn.Module):
    def __init__(self, hidden_dim, action_dim=6, width=128, observation_channels=16):
        super().__init__()
        self.action_dim = action_dim
        self.action = nn.Sequential(nn.Linear(4 * action_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.observation = nn.Linear(observation_channels, width)
        self.query = nn.Linear(hidden_dim, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, hidden_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, actions, first_observed_latent, grid):
        frames, height, width = grid
        if actions.shape != (x.shape[0], (frames - 1) * 4, self.action_dim):
            raise ValueError('Actions must retain four ordered frame intervals per later latent frame')
        if (first_observed_latent.ndim != 4 or first_observed_latent.shape[1] != 16
                or first_observed_latent.shape[0] != x.shape[0]):
            raise ValueError('Observation must be [B,16,H,W] from the first observed frame only')
        # Concatenation preserves action order inside each four-frame interval.
        groups = actions.float().reshape(x.shape[0], frames - 1, 4 * self.action_dim)
        initial = groups.new_zeros(x.shape[0], 1, 4 * self.action_dim)
        action = self.action(torch.cat((initial, groups), dim=1))
        action = action[:, :, None, :].expand(-1, -1, height * width, -1).flatten(1, 2)
        observed = F.adaptive_avg_pool2d(first_observed_latent.float(), (4, 8)).flatten(2).transpose(1, 2)
        observed = self.observation(observed)
        query = self.query(x.float()).unsqueeze(1)
        attended = F.scaled_dot_product_attention(query, self.key(observed).unsqueeze(1), self.value(observed).unsqueeze(1)).squeeze(1)
        return x + self.output(F.silu(action + attended)).to(x.dtype)


class ActionObservationAdapter(nn.Module):
    def __init__(self, hidden_dim, block_indices=(9, 19, 29), action_dim=6, width=128):
        super().__init__()
        self.block_indices = tuple(block_indices)
        self.residuals = nn.ModuleList([ActionObservationResidual(hidden_dim, action_dim, width) for _ in block_indices])
        self._hooks = []

    def attach(self, core, actions, first_observed_latent, grid):
        if self._hooks:
            raise RuntimeError('Detach previous conditioning hooks before attaching new inputs')
        for block_index, residual in zip(self.block_indices, self.residuals):
            def hook(module, args, output, residual=residual):
                return residual(output, actions, first_observed_latent, grid)
            self._hooks.append(core.blocks[block_index].register_forward_hook(hook))

    def detach(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
