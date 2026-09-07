"""Original, seed-reproducible synthetic data for Worldline.

This is the explicit training distribution. It is not a real-world corpus and
it is never called by the trained models during generation or simulation.
"""
import numpy as np
import torch
from torch.nn import functional as F


def terrain_batch(count=32, size=64, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    result, labels = [], []
    for index in range(count):
        biome = int(rng.integers(0, 3))
        angle = rng.uniform(0, np.pi * 2)
        u = x * np.cos(angle) + y * np.sin(angle)
        v = -x * np.sin(angle) + y * np.cos(angle)
        phase = rng.uniform(-np.pi, np.pi)
        field = np.zeros_like(x)
        for frequency, amplitude in [(1.0, .46), (2.0, .23), (4.0, .11), (8.0, .045)]:
            a = rng.uniform(0, 2 * np.pi)
            b = rng.uniform(0, 2 * np.pi)
            field += amplitude * np.sin(u * frequency * np.pi + a) * np.cos(v * frequency * np.pi + b)
        for _ in range(4):
            cx, cy = rng.uniform(-1.2, 1.2, 2)
            width = rng.uniform(.10, .65)
            field += rng.uniform(-.22, .35) * np.exp(-((x-cx)**2 + (y-cy)**2) / width**2)
        if biome == 0:
            ridge = .50 * (1 - np.abs(np.sin(3.2*u + .4*np.sin(4*v) + phase)))
            height = np.tanh((field + ridge - .20) * 1.65)
            vegetation = np.clip(.85 - .75 * np.maximum(height, 0) + .15 * np.sin(4*v + phase), 0, 1)
        elif biome == 1:
            dunes = np.sin(7 * u + .8 * np.sin(2.5 * v) + phase)
            height = np.tanh(.68 * field + .34 * dunes - .12)
            vegetation = np.clip(.07 + .14 * (1-height) * (1 + np.sin(3*v + phase)) / 2, 0, .35)
        else:
            rings = np.sin(9 * np.sqrt((u + .35*np.sin(phase))**2 + (v+.25*np.cos(phase))**2) + phase)
            height = np.tanh(field * 1.4 + .30 * rings + .10)
            vegetation = np.clip(.47 + .27*np.sin(height * 7 + phase) + .16 * np.cos(5*v), 0, 1)
        result.append(np.stack([height, vegetation * 2 - 1]).astype(np.float32))
        labels.append(biome)
    return torch.from_numpy(np.stack(result)), torch.tensor(labels, dtype=torch.long)


def ecology_states(count=32, size=32, seed=0):
    generator = torch.Generator().manual_seed(int(seed))
    coarse = torch.rand(count, 3, 6, 6, generator=generator)
    smooth = F.interpolate(coarse, size=(size, size), mode="bilinear", align_corners=False)
    offsets = torch.rand(count, 3, 1, 1, generator=generator) * .7
    state = (smooth * .5 + offsets - .10).clamp(0, 1)
    state += (torch.rand(count, 3, size, size, generator=generator) - .5) * .06
    return state.clamp(0, 1)


def ecology_teacher(state, rain, heat, dt):
    """Synthetic reaction/diffusion teacher used only for training and eval.

    Controls and state have no SI-unit interpretation. Conditions stay within
    the small bounded ecosystem specified here.
    """
    water, vegetation, temperature = state.split(1, dim=1)
    neighbourhood = F.avg_pool2d(F.pad(state, (1,1,1,1), mode="replicate"), 3, stride=1)
    spread = neighbourhood - state
    rain, heat, dt = (v.reshape(-1, 1, 1, 1) for v in (rain, heat, dt))
    change_w = .11 * spread[:, :1] + .040 * rain * (1-water) - .026 * temperature * water - .008 * vegetation * water
    change_v = .014 * spread[:, 1:2] + .027 * water * vegetation * (1-vegetation) - .014 * temperature.square() * vegetation - .012 * (1-water).square() * vegetation
    change_t = .080 * spread[:, 2:3] + .065 * (heat-temperature) - .022 * rain * temperature
    return (state + dt * torch.cat([change_w, change_v, change_t], dim=1)).clamp(0, 1)
