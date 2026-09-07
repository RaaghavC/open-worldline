"""Learned RGB transitions. This module does not import or call a renderer."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, condition=64):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(4, cout)
        self.affine = nn.Linear(condition, 2*cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, condition):
        scale, shift = self.affine(condition).chunk(2, 1)
        h = self.norm(self.conv1(x))*(1+scale[:, :, None, None])+shift[:, :, None, None]
        return F.silu(self.conv2(F.silu(h))+self.skip(x))


class RGBModel(nn.Module):
    """Original compact action-conditioned U-Net; predictor or rectified flow."""
    def __init__(self, kind="predictor", width=24):
        super().__init__()
        if kind not in ("predictor", "flow") or width < 8 or width % 4:
            raise ValueError("invalid architecture")
        self.kind, self.width = kind, width
        self.action = nn.Embedding(6, 64)
        self.time = nn.Sequential(nn.Linear(9, 64), nn.SiLU(), nn.Linear(64, 64))
        self.register_buffer("frequencies", torch.tensor([1., 2., 4., 8.])*torch.pi)
        self.a = Block(15 if kind == "flow" else 12, width)
        self.b = Block(width, width*2)
        self.c = Block(width*2, width*4)
        self.middle = Block(width*4, width*4)
        self.upb = Block(width*6, width*2)
        self.upa = Block(width*3, width)
        self.out = nn.Conv2d(width, 3, 3, padding=1)
        if kind == "predictor":
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, history, action, noisy=None, time=None):
        batch, count, channels, height, width = history.shape
        if count != 4 or channels != 3:
            raise ValueError("history must be [B,4,3,H,W]")
        cond = self.action(action)
        x = history.reshape(batch, 12, height, width)
        if self.kind == "flow":
            if noisy is None or time is None:
                raise ValueError("flow requires noisy target and time")
            time = time.reshape(-1, 1)
            angle = time*self.frequencies[None]
            cond = cond+self.time(torch.cat([time, angle.sin(), angle.cos()], 1))
            x = torch.cat([x, noisy], 1)
        a = self.a(x, cond)
        b = self.b(F.avg_pool2d(a, 2), cond)
        c = self.c(F.avg_pool2d(b, 2), cond)
        c = self.middle(c, cond)
        b = self.upb(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1), cond)
        a = self.upa(torch.cat([F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1), cond)
        output = self.out(a)
        if self.kind == "predictor":
            return (history[:, -1]+output).clamp(-1, 1)
        return output


@torch.inference_mode()
def predict_tensor(model, history, action, sample_steps=8, rng=None):
    if model.kind == "predictor":
        return model(history, action)
    if sample_steps < 1:
        raise ValueError("sample_steps must be positive")
    shape = (history.shape[0], 3, *history.shape[-2:])
    x = torch.randn(shape, generator=rng, device="cpu").to(history.device)
    for i in range(sample_steps):
        time = torch.full((history.shape[0],), i/sample_steps, device=history.device)
        x = x+model(history, action, x, time)/sample_steps
    return x.clamp(-1, 1)


@torch.inference_mode()
def rollout(model, initial_history, actions, device="cpu", sample_steps=8, seed=0):
    """Only initial RGB, controls and model weights are used for all next frames."""
    initial_history = np.asarray(initial_history, dtype=np.float32)
    if initial_history.ndim != 4 or initial_history.shape[:2] != (4, 3):
        raise ValueError("initial_history must have shape [4,3,H,W]")
    if not np.isfinite(initial_history).all() or np.abs(initial_history).max() > 1.00001:
        raise ValueError("initial_history must contain finite normalized RGB in [-1,1]")
    controls = np.asarray(actions)
    if controls.ndim != 1 or controls.dtype.kind not in "iu" or ((controls < 0) | (controls > 5)).any():
        raise ValueError("actions must be a vector of integer action ids in [0,5]")
    model.eval()
    history = torch.from_numpy(initial_history).unsqueeze(0).to(device)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    frames = []
    for action in controls:
        value = predict_tensor(model, history, torch.tensor([int(action)], device=device), sample_steps, rng)
        frames.append(value[0].cpu().numpy())
        history = torch.cat([history[:, 1:], value.unsqueeze(1)], 1)
    return np.stack(frames) if frames else np.empty((0, 3, *initial_history.shape[-2:]), np.float32)


def load_model(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = RGBModel(config["kind"], config["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint
