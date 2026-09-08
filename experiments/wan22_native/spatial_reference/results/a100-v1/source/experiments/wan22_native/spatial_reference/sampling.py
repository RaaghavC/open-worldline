# SPDX-License-Identifier: Apache-2.0
"""CPU UniPC for exactly two 17-frame spatial-reference shapes.

The injected predictor receives CPU FP32 ``x[C,F,H,W]``, CPU int64
``times[1,L]`` and CPU FP32 ``context[N,D]``. It must return a finite CPU
FP32 velocity with the same shape as x. It owns any denoiser device transfer.
No model, observation encoder, random draw or target reader is included here.
"""

from collections.abc import Mapping

import torch

from ..cuda_reference.vendor.fm_solvers_unipc import FlowUniPCMultistepScheduler


ALLOWED_SHAPES = ((48, 5, 18, 32), (48, 5, 44, 78))
SETTINGS = {"steps": 50, "shift": 5.0, "guidance": 5.0}
VALUE_KEYS = frozenset(("initial_noise", "initial_latent", "observation", "token_times"))
CONTEXT_KEYS = frozenset(("atrium", "native_negative"))


def validate_shape(latent_shape):
    """Return the declared C,F,H,W tuple; reject every other shape."""
    if not isinstance(latent_shape, (tuple, list)) or any(type(x) is not int for x in latent_shape):
        raise ValueError("An explicit integer latent-shape tuple is required")
    shape = tuple(latent_shape)
    if shape not in ALLOWED_SHAPES:
        raise ValueError("Only the baseline and spatial 17-frame shapes are supported")
    return shape


def token_geometry(latent_shape):
    """Return (total tokens, observed tokens) in native F,H/2,W/2 order."""
    _, frames, height, width = validate_shape(latent_shape)
    prefix = (height // 2) * (width // 2)
    return frames * prefix, prefix


def scheduler():
    """Construct the unchanged upstream CPU UniPC50 / shift-5 schedule."""
    solver = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    solver.set_timesteps(50, device="cpu", shift=5.0)
    return solver


def times_at(t, latent_shape):
    """Set the first latent frame's patch times to zero; preserve future t."""
    tokens, prefix = token_geometry(latent_shape)
    if isinstance(t, torch.Tensor) and t.device.type != "cpu":
        raise ValueError("Scheduler time must already be on CPU")
    value = torch.as_tensor(t, device="cpu")
    if value.numel() != 1 or value.dtype != torch.int64 or not 0 <= int(value) <= 999:
        raise ValueError("One exact integer scheduler time in [0,999] is required")
    result = value.reshape(1, 1).expand(1, tokens).clone()
    result[:, :prefix] = 0
    return result


def _float_tensor(value, shape, name):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != torch.float32 or tuple(value.shape) != tuple(shape)):
        raise ValueError(f"{name} must be CPU FP32 with shape {tuple(shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains nonfinite values")


def _context(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or min(value.shape) <= 0:
        raise ValueError(f"{name} must be a nonempty sequence of text embeddings")
    _float_tensor(value, value.shape, name)


def _times(value, shape):
    tokens, prefix = token_geometry(shape)
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != torch.int64 or tuple(value.shape) != (1, tokens)):
        raise ValueError("Token times must have the declared CPU int64 sequence shape")
    expected = times_at(value[0, prefix], shape)
    if not torch.equal(value, expected):
        raise ValueError("Observed times must be zero and future times one exact scheduler value")


def _copies(values):
    return {key: value.clone() for key, value in values.items()}


@torch.no_grad()
def pair(predict, x, times, positive, negative, observation, *, latent_shape, event=None):
    """Positive then negative prediction with the same restored image prefix.

    Inputs are copied before each prediction. ``event(label, partial_outputs)``
    receives detached copies after each completed branch, so a recorder cannot
    change the guided velocity. Exceptions propagate to the caller's guard.
    """
    shape = validate_shape(latent_shape)
    _float_tensor(x, shape, "latent")
    _float_tensor(observation, (1, shape[0], 1, shape[2], shape[3]), "observation")
    _times(times, shape)
    _context(positive, "positive context")
    _context(negative, "negative context")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("Positive and negative embedding widths must match")
    outputs = {}
    with torch.autocast("cpu", enabled=False):
        for label, context in (("positive", positive), ("negative", negative)):
            projected = x.clone()
            projected[:, :1] = observation[0]
            prediction = predict(projected, times.clone(), context.clone())
            _float_tensor(prediction, shape, f"{label} velocity")
            outputs[label + "_velocity"] = prediction.clone()
            if event is not None:
                event(label, _copies(outputs))
        guided = outputs["negative_velocity"] + 5.0 * (
            outputs["positive_velocity"] - outputs["negative_velocity"]
        )
        _float_tensor(guided, shape, "guided velocity")
        outputs["guided_velocity"] = guided
    return outputs


@torch.no_grad()
def sample(predict, values, contexts, *, latent_shape, event=None):
    """Return a generated CPU FP32 latent after exactly 50 solver updates.

    ``values`` contains initial_noise, initial_latent, observation and token_times.
    ``contexts`` contains atrium and native_negative. Future targets, actions and
    extra input fields are rejected. No noise is regenerated. The optional
    ``event(step_index, scheduler_time, latent, velocities)`` runs after each
    completed update and receives copies. It uses zero-based step indices.
    """
    shape = validate_shape(latent_shape)
    if not isinstance(values, Mapping) or set(values) != VALUE_KEYS:
        raise ValueError("Exactly the four saved sampling inputs are required")
    if not isinstance(contexts, Mapping) or set(contexts) != CONTEXT_KEYS:
        raise ValueError("Exactly the positive and native-negative text contexts are required")
    for key in ("initial_noise", "initial_latent"):
        _float_tensor(values[key], shape, key)
    observation = values["observation"]
    _float_tensor(observation, (1, shape[0], 1, shape[2], shape[3]), "observation")
    _times(values["token_times"], shape)
    for key in CONTEXT_KEYS:
        _context(contexts[key], key)

    with torch.autocast("cpu", enabled=False):
        solver = scheduler()
        x = values["initial_noise"].clone()
        x[:, :1] = observation[0]
        if (not torch.equal(x, values["initial_latent"])
                or not torch.equal(times_at(solver.timesteps[0], shape), values["token_times"])):
            raise ValueError("Saved initial state or time differs from the declared native schedule")
        for step, t in enumerate(solver.timesteps):
            x[:, :1] = observation[0]
            outputs = pair(
                predict, x, times_at(t, shape), contexts["atrium"],
                contexts["native_negative"], observation, latent_shape=shape
            )
            x = solver.step(
                outputs["guided_velocity"].unsqueeze(0), t, x.unsqueeze(0),
                return_dict=False
            )[0][0]
            x[:, :1] = observation[0]
            _float_tensor(x, shape, "solver state")
            if event is not None:
                event(step, t.clone(), x.clone(), _copies(outputs))
    return x
