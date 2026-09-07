"""Original Worldline spatial flow generator and learned ecosystem dynamics.

No pretrained models or external model services are used. The terrain model
learns synthetic fields; the ecosystem model learns a synthetic transition law.
Rendering, object geometry, and camera movement live outside these networks.
"""
from pathlib import Path
import threading

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SIZE = 64
DEFAULT_CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


def choose_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ConditionedBlock(nn.Module):
    def __init__(self, channels_in, channels_out, condition_width=64):
        super().__init__()
        self.first = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.norm = nn.GroupNorm(4, channels_out)
        self.condition = nn.Linear(condition_width, channels_out * 2)
        self.second = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.skip = nn.Conv2d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()

    def forward(self, x, condition):
        scale, shift = self.condition(condition).chunk(2, dim=1)
        y = self.norm(self.first(x))
        y = y * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        y = self.second(F.silu(y))
        return F.silu(y + self.skip(x))


class SpatialFlowNet(nn.Module):
    """Time/biome-conditioned U-Net, predicting rectified-flow velocity."""
    def __init__(self, width=16):
        super().__init__()
        self.biome = nn.Embedding(3, 64)
        self.time = nn.Sequential(nn.Linear(9, 64), nn.SiLU(), nn.Linear(64, 64))
        self.register_buffer("frequencies", torch.tensor([1., 2., 4., 8.]) * torch.pi)
        self.down0 = ConditionedBlock(2, width)
        self.down1 = ConditionedBlock(width, width * 2)
        self.down2 = ConditionedBlock(width * 2, width * 4)
        self.middle = ConditionedBlock(width * 4, width * 4)
        self.up1 = ConditionedBlock(width * 6, width * 2)
        self.up0 = ConditionedBlock(width * 3, width)
        self.output = nn.Conv2d(width, 2, 3, padding=1)

    def forward(self, x, time, biome):
        time = time.reshape(-1, 1)
        angles = time * self.frequencies[None]
        condition = self.time(torch.cat([time, angles.sin(), angles.cos()], 1)) + self.biome(biome)
        a = self.down0(x, condition)
        b = self.down1(F.avg_pool2d(a, 2), condition)
        c = self.down2(F.avg_pool2d(b, 2), condition)
        c = self.middle(c, condition)
        b = self.up1(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1), condition)
        a = self.up0(torch.cat([F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1), condition)
        return self.output(a)


class EcologyNet(nn.Module):
    """Learned field transition, channels water/vegetation/temperature in [0,1]."""
    def __init__(self, width=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(6, width, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(width, 3, 1),
        )

    def forward(self, state, rain, heat, dt):
        batch, _, h, w = state.shape
        controls = torch.stack([rain, heat, dt], dim=1).reshape(batch, 3, 1, 1).expand(-1, -1, h, w)
        derivative = self.layers(torch.cat([state, controls], dim=1)) * 0.1
        return (state + derivative * dt[:, None, None, None]).clamp(0, 1)


class Generator:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = choose_device(device)
        self.model = SpatialFlowNet().to(self.device).eval()
        self._lock = threading.RLock()
        checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINTS / "spatial-flow.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.metadata = checkpoint.get("metadata", {})

    @torch.inference_mode()
    def sample(self, seed: int, biome: int, steps: int = 24):
        """Return float32 [2,64,64] with height/density in [-1,1]."""
        if biome not in (0, 1, 2):
            raise ValueError("biome must be 0 (alpine), 1 (desert), or 2 (alien)")
        if not 4 <= int(steps) <= 100:
            raise ValueError("steps must be between 4 and 100")
        rng = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
        with self._lock:
            x = torch.randn(1, 2, SIZE, SIZE, generator=rng).to(self.device)
            label = torch.tensor([biome], device=self.device, dtype=torch.long)
            step_size = 1.0 / int(steps)
            for i in range(int(steps)):
                t = torch.tensor([i * step_size], device=self.device)
                velocity = self.model(x, t, label)
                # Heun integration reduces field artifacts at a modest step budget.
                proposal = x + velocity * step_size
                next_t = torch.tensor([(i + 1) * step_size], device=self.device)
                next_velocity = self.model(proposal, next_t, label)
                x = x + (velocity + next_velocity) * (step_size * .5)
            return x[0].clamp(-1, 1).cpu().numpy().astype(np.float32)


class Dynamics:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = choose_device(device)
        self.model = EcologyNet().to(self.device).eval()
        self._lock = threading.RLock()
        checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINTS / "ecology.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.metadata = checkpoint.get("metadata", {})

    @torch.inference_mode()
    def step(self, state, rain: float, heat: float, dt: float = 1.0):
        """Predict next fields [3,H,W]; rain/heat in [0,1], dt in [0,2]."""
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 3 or state.shape[0] != 3 or min(state.shape[1:]) < 4:
            raise ValueError("state must have shape [3,H,W] with H,W >= 4")
        if not np.isfinite(state).all() or not np.isfinite([rain, heat, dt]).all():
            raise ValueError("state and controls must be finite")
        if not (0 <= rain <= 1 and 0 <= heat <= 1 and 0 <= dt <= 2):
            raise ValueError("rain and heat must be in [0,1], dt in [0,2]")
        with self._lock:
            value = torch.from_numpy(np.clip(state, 0, 1)).unsqueeze(0).to(self.device)
            controls = [torch.tensor([float(c)], device=self.device) for c in (rain, heat, dt)]
            return self.model(value, *controls)[0].cpu().numpy().astype(np.float32)


def get_models(checkpoint_dir=None, device=None):
    directory = Path(checkpoint_dir or DEFAULT_CHECKPOINTS)
    return Generator(directory / "spatial-flow.pt", device), Dynamics(directory / "ecology.pt", device)
