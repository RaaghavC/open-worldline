# SPDX-License-Identifier: Apache-2.0
"""A small recurrent addition to the frozen original RGB room predictor.

The caller owns one [batch, 32] state tensor per session and must create a new
zero state at episode boundaries. A step receives four RGB observations, the
command for the next transition, and the state accumulated before that command.
It returns a predicted next RGB frame and the state after that command. No state
is retained on the module, and no renderer, pose or teacher labels are accepted.

This implements an untrained memory experiment. It does not establish learned
memory, better rollouts or a new scientific method.
"""
from itertools import chain

import torch
from torch import nn
from torch.nn import functional as F

from .model import RGBModel


class RoomMemoryModel(nn.Module):
    """GRU32 and channel scale/shift around the width24 direct predictor.

    Construction takes ownership of the supplied predictor's frozen/eval state.
    It does not change its parameter values. Use separate wrappers or a single
    shared wrapper with separate state tensors for separate world sessions.
    ``carry`` and ``reset`` use exactly the same parameters; ``reset`` supplies
    zeros to the GRU every step, while returning its computed state for the same
    interface. Training code decides when to detach explicitly. This module
    never detaches a carried state or an RGB-history gradient.
    """

    state_size = 32
    bottleneck_channels = 96
    action_channels = 64

    def __init__(self, predictor: RGBModel):
        super().__init__()
        if not isinstance(predictor, RGBModel) or predictor.kind != "predictor" or predictor.width != 24:
            raise ValueError("Room memory requires the width24 direct RGB predictor")
        self.base = predictor.eval().requires_grad_(False)
        for parameter in self.base.parameters():
            parameter.grad = None
        reference = self.base.out.weight
        self.gru = nn.GRUCell(self.bottleneck_channels + self.action_channels, self.state_size,
                             device=reference.device, dtype=reference.dtype)
        self.condition = nn.Linear(self.state_size, 2 * self.bottleneck_channels,
                                   device=reference.device, dtype=reference.dtype)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def memory_parameters(self):
        """Only these parameters belong in the memory experiment optimizer."""
        return chain(self.gru.parameters(), self.condition.parameters())

    def parameter_counts(self):
        return {
            "base": sum(parameter.numel() for parameter in self.base.parameters()),
            "memory": sum(parameter.numel() for parameter in self.memory_parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "state_elements_per_session": self.state_size,
            "state_bytes_per_session": self.state_size * self.gru.weight_ih.element_size(),
        }

    def initial_state(self, batch_size: int):
        """New zero state on the model's device/dtype, without shared storage."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return self.gru.weight_ih.new_zeros((batch_size, self.state_size))

    @staticmethod
    def clone_state(state: torch.Tensor, *, detach: bool = False):
        """Fork state storage, retaining temporal gradients unless requested."""
        if not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[1] != 32:
            raise ValueError("State must be a [B,32] tensor")
        if type(detach) is not bool:
            raise ValueError("detach must be a boolean")
        return state.detach().clone() if detach else state.clone()

    def _validate(self, history, action, state, mode):
        if mode not in ("carry", "reset"):
            raise ValueError("mode must be carry or reset")
        if not isinstance(history, torch.Tensor) or history.ndim != 5 or history.shape[1:3] != (4, 3):
            raise ValueError("history must be [B,4,3,H,W]")
        if history.shape[0] < 1 or min(history.shape[-2:]) < 4:
            raise ValueError("history requires a nonempty batch and spatial dimensions of at least four")
        if not isinstance(action, torch.Tensor) or action.shape != (history.shape[0],) or action.dtype != torch.int64:
            raise ValueError("action must be int64 [B] command ids")
        if not isinstance(state, torch.Tensor) or state.shape != (history.shape[0], self.state_size):
            raise ValueError("state must be [B,32]; initialize each new episode explicitly")
        reference = self.gru.weight_ih
        if history.device != reference.device or state.device != reference.device or action.device != reference.device:
            raise ValueError("history, action and state must be on the model device")
        if history.dtype != reference.dtype or state.dtype != reference.dtype:
            raise ValueError("RGB history and state must use the model floating-point dtype")
        if not torch.isfinite(history).all() or not torch.isfinite(state).all():
            raise ValueError("RGB history and state must be finite")
        if history.abs().max() > 1.00001 or (action < 0).any() or (action > 5).any():
            raise ValueError("RGB must be normalized to [-1,1] and command ids must be in [0,5]")

    def forward(self, history, action, state, *, mode="carry"):
        """Advance one world step; both outputs retain their autograd graph."""
        self._validate(history, action, state, mode)
        batch, _, _, height, width = history.shape
        condition = self.base.action(action)
        a = self.base.a(history.reshape(batch, 12, height, width), condition)
        b = self.base.b(F.avg_pool2d(a, 2), condition)
        c = self.base.c(F.avg_pool2d(b, 2), condition)
        c = self.base.middle(c, condition)
        recurrent_input = torch.cat((c.mean(dim=(-2, -1)), condition), dim=1)
        previous = state if mode == "carry" else torch.zeros_like(state)
        next_state = self.gru(recurrent_input, previous)
        scale, shift = self.condition(next_state).chunk(2, dim=1)
        c = c * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        b = self.base.upb(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b), dim=1), condition)
        a = self.base.upa(torch.cat((F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False), a), dim=1), condition)
        prediction = (history[:, -1] + self.base.out(a)).clamp(-1, 1)
        return prediction, next_state
