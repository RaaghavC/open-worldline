# SPDX-License-Identifier: Apache-2.0
"""Separate batched teacher-observed path; not wired into the measured trainer.

Frozen per-image features and decoding run across time in one batch. The GRU
still advances sequentially with a complete temporal gradient graph. This path
is for observed-history training, never for generated-history rollout: a future
generated frame cannot be prepared before its preceding model prediction.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .memory_evaluate import history_at
from .memory_model import RoomMemoryModel
from .model import RGBModel, Block


def observed_sequence(model, observations, actions, state=None, *, mode="carry", check=None):
    """Predict from T observed frames/actions; future loss targets are not inputs.

    observations: [B,T,3,H,W], containing the current observation for each step.
    actions: int64 [B,T]. Returned predictions are [B,T,3,H,W]. Each predicted
    step depends only on observations through that step and earlier state.
    The caller owns the optional initial state; the function does not mutate it.
    """
    if type(model) is not RoomMemoryModel:
        raise TypeError("This separate path supports the exact current RoomMemoryModel")
    if observations.ndim != 5 or observations.shape[2] != 3 or not 1 <= observations.shape[1] <= 65:
        raise ValueError("Observed RGB must be [B,T,3,H,W] with 1 through 65 steps")
    batch, steps, _, height, width = observations.shape
    if actions.shape != (batch, steps) or actions.dtype != torch.int64:
        raise ValueError("Actions must be int64 [B,T]")
    if not torch.isfinite(observations).all() or observations.abs().max() > 1.00001:
        raise ValueError("Observed RGB must be finite in [-1,1]")
    if (actions < 0).any() or (actions > 5).any():
        raise ValueError("Action ids must be 0 through 5")
    state = model.initial_state(batch) if state is None else state
    model._validate(history_at(observations, 0), actions[:, 0], state, mode)
    if model.base.training or any(parameter.requires_grad for parameter in model.base.parameters()):
        raise ValueError("The predictor must remain frozen and in evaluation mode")
    allowed = {RGBModel, Block, nn.Conv2d, nn.GroupNorm, nn.Linear, nn.Embedding, nn.SiLU, nn.Identity, nn.Sequential}
    if any(type(module) not in allowed for module in model.base.modules()):
        raise ValueError("The predictor contains a layer whose temporal batching is not verified")
    if check is not None:
        check()
    # Time-major layout makes the sample order explicit for the recurrent loop.
    histories = torch.stack([history_at(observations, step) for step in range(steps)], dim=0)
    histories = histories.reshape(steps * batch, 4, 3, height, width)
    condition = model.base.action(actions.transpose(0, 1).reshape(-1))
    a = model.base.a(histories.reshape(steps * batch, 12, height, width), condition)
    b = model.base.b(F.avg_pool2d(a, 2), condition)
    c = model.base.c(F.avg_pool2d(b, 2), condition)
    c = model.base.middle(c, condition)
    recurrent = torch.cat((c.mean(dim=(-2, -1)), condition), dim=1).reshape(steps, batch, -1)
    states = []
    for step in range(steps):
        if check is not None:
            check()
        previous = state if mode == "carry" else torch.zeros_like(state)
        state = model.gru(recurrent[step], previous)
        states.append(state)
    # No detach/no_grad: gradients can reach earlier state and observed RGB.
    all_states = torch.stack(states, dim=0).reshape(steps * batch, model.state_size)
    scale, shift = model.condition(all_states).chunk(2, dim=1)
    c = c * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
    b = model.base.upb(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b), dim=1), condition)
    a = model.base.upa(torch.cat((F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False), a), dim=1), condition)
    predictions = (histories[:, -1] + model.base.out(a)).clamp(-1, 1)
    if check is not None:
        check()
    return predictions.reshape(steps, batch, 3, height, width).transpose(0, 1), state


def sequence_loss_batched(model, observations, actions, mode="carry", *, check=None):
    """Same all-65-step RGB[0,1] MAE as the measured stepwise sequence_loss."""
    if observations.ndim != 5 or observations.shape[1:3] != (66, 3) or actions.shape != (observations.shape[0], 65):
        raise ValueError("Training requires all 65 transitions and 66 RGB observations")
    predictions, _ = observed_sequence(model, observations[:, :-1], actions, mode=mode, check=check)
    # Observation t+1 enters only this loss, after predictions are computed.
    return ((predictions - observations[:, 1:]).abs().mean(dim=(0, 2, 3, 4)) / 2).mean()
