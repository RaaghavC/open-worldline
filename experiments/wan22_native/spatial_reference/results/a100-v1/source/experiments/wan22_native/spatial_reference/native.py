# SPDX-License-Identifier: Apache-2.0
"""Two declared spatial shapes through the unchanged original CUDA model.

The loader and meta constructor are the frozen CUDA reference functions, not
new implementations. This wrapper changes only sequence-length dispatch and
adds bounded CPU input checks. It creates no second model or weight copy.
"""
import torch

from ..cuda_reference import native as _frozen


meta_model = _frozen.meta_model
load_model = _frozen.load_model

_SHAPES = {
    (48, 5, 18, 32): (720, 144),
    (48, 5, 44, 78): (4290, 858),
}


def _cpu_tensor(value, dtype, name):
    if (not isinstance(value, torch.Tensor)
            or value.device.type != 'cpu'
            or value.layout != torch.strided
            or value.dtype != dtype):
        raise ValueError(f'{name} must be a dense CPU {dtype} tensor')


def _validate_inputs(latent, times, context):
    _cpu_tensor(latent, torch.float32, 'latent')
    shape = tuple(latent.shape)
    if shape not in _SHAPES:
        raise ValueError('Only the declared 17-frame 512x288 and 1248x704 latent shapes are accepted')
    tokens, prefix = _SHAPES[shape]
    _cpu_tensor(times, torch.int64, 'times')
    if tuple(times.shape) != (1, tokens):
        raise ValueError('Token times must match the declared spatial shape')
    if not bool((times[:, :prefix] == 0).all()):
        raise ValueError('Every observed-prefix token must have time zero')
    time = int(times[0, prefix])
    if not 0 <= time <= 999 or not bool((times[:, prefix:] == time).all()):
        raise ValueError('Future token times must be one integer in [0, 999]')
    _cpu_tensor(context, torch.float32, 'context')
    if context.ndim != 2 or context.shape[1] != 4096 or not 1 <= context.shape[0] <= 512:
        raise ValueError('Context must have shape [C, 4096] with 1 <= C <= 512')
    if not bool(torch.isfinite(latent).all()) or not bool(torch.isfinite(context).all()):
        raise ValueError('Latent and context must contain only finite values')
    return tokens


def predict(model, latent, times, context):
    """Return a finite FP32 CPU prediction at one of the two allowed shapes.

    Both text branches use this same function. It neither edits the caller's
    observed prefix nor invents conditioning; the sampler supplies and checks
    the clean prefix independently. There is no CPU attention fallback.
    """
    tokens = _validate_inputs(latent, times, context)
    _frozen.verify_sources()
    from ..cuda_reference.vendor import attention
    if not attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError('The spatial reference requires upstream FlashAttention 2 only')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        values = model(
            [latent.to('cuda:0')], times.to('cuda:0'),
            [context.to('cuda:0')], tokens,
        )
    torch.cuda.synchronize()
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        raise RuntimeError('The native model must return exactly one prediction')
    output = values[0]
    if (not isinstance(output, torch.Tensor)
            or output.layout != torch.strided
            or not output.is_floating_point()
            or tuple(output.shape) != tuple(latent.shape)):
        raise RuntimeError('The native prediction has an unexpected shape or type')
    output = output.float().cpu()
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError('The native prediction is not finite FP32')
    return output
