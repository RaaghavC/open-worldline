# SPDX-License-Identifier: Apache-2.0
"""Work-only action sampling kernel. No launcher, admission, decoder or loader for Wan.

The future parent runner owns real-probe admission, source/input provenance,
original foundation verification, memory/time guards and image decoding.
"""
import hashlib
import json
from pathlib import Path
import struct

import torch
from safetensors.torch import load as load_safetensors_bytes

from experiments.wan22_native.action_adapter.model import (
    NATIVE_PARAMETER_COUNT, PostBlockActionAdapter,
)
from experiments.wan22_native.action_cuda.bridge import NativeCUDAActionBridge, PROFILES
from experiments.wan22_native.spatial_reference import sampling as native_sampling


MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate safetensors header key")
        result[key] = value
    return result


def load_adapter_checkpoint(path, expected_sha256):
    """Load only the complete 947,712-parameter FP32 adapter, on CPU.

    A bounded byte read makes the file hash and the deserialized bytes identical.
    Header names/shapes/dtypes are checked before tensor materialization. No
    optimizer state, Python pickle, foundation tensor or partial load is accepted.
    """
    path = Path(path).absolute()
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(char not in '0123456789abcdef' for char in expected_sha256)):
        raise ValueError("An exact lowercase checkpoint SHA256 is required")
    if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
        raise ValueError("Checkpoint must be a regular file without symlink components")
    with path.open('rb') as stream:
        payload = stream.read(MAX_CHECKPOINT_BYTES + 1)
    if not 8 < len(payload) <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Adapter checkpoint exceeds the 4 MiB bound or is truncated")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Adapter checkpoint SHA256 differs")
    header_size = struct.unpack('<Q', payload[:8])[0]
    if not 2 <= header_size <= MAX_HEADER_BYTES or 8 + header_size >= len(payload):
        raise ValueError("Invalid bounded safetensors header")
    header = json.loads(payload[8:8 + header_size], object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    metadata = header.pop('__metadata__', {})
    if (not isinstance(metadata, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())):
        raise ValueError("Safetensors metadata must contain string pairs")
    with torch.device('meta'):
        expected = PostBlockActionAdapter().state_dict()
    if set(header) != set(expected):
        raise ValueError("Checkpoint must contain exactly the original adapter parameters")
    offset = 0
    for record in header.values():
        if (not isinstance(record, dict) or set(record) != {'dtype', 'shape', 'data_offsets'}
                or not isinstance(record['data_offsets'], list) or len(record['data_offsets']) != 2
                or any(type(value) is not int for value in record['data_offsets'])):
            raise ValueError('Each adapter tensor requires exact integer byte offsets')
    for name, record in sorted(header.items(), key=lambda item: item[1]['data_offsets'][0]):
        shape = list(expected[name].shape)
        count = expected[name].numel()
        if (not isinstance(record, dict) or set(record) != {'dtype', 'shape', 'data_offsets'}
                or record['dtype'] != 'F32' or record['shape'] != shape
                or record['data_offsets'] != [offset, offset + 4 * count]):
            raise ValueError("Adapter tensor shape, dtype or contiguous data offsets differ")
        offset += 4 * count
    if offset != 4 * NATIVE_PARAMETER_COUNT or 8 + header_size + offset != len(payload):
        raise ValueError("Adapter parameter count or payload length differs")
    values = load_safetensors_bytes(payload)
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("Adapter checkpoint contains nonfinite values")
    # Preserve the caller's CPU RNG. All initialized values are then overwritten.
    with torch.random.fork_rng(devices=[]):
        adapter = PostBlockActionAdapter()
    adapter.load_state_dict(values, strict=True)
    # The unchanged bridge requires these flags even for track_grad=False.
    # Sampling disables graph construction without changing the module contract.
    adapter.requires_grad_(True).eval()
    return adapter, {'checkpoint_sha256': digest, 'parameter_count': NATIVE_PARAMETER_COUNT,
                     'dtype': 'float32', 'tensor_sha256': {k: tensor_sha(v) for k, v in values.items()}}


def _float(value, shape, name):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or value.layout != torch.strided or value.dtype != torch.float32
            or tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all()):
        raise ValueError(name + ' must be finite dense CPU FP32 at the declared shape')


def validate_inputs(values, contexts, commands, profile):
    """Reject target/state fields and malformed inputs before touching CUDA."""
    if profile not in PROFILES:
        raise ValueError('Only baseline and spatial profiles are supported')
    if not isinstance(values, dict) or set(values) != native_sampling.VALUE_KEYS:
        raise ValueError('Exactly the four saved native sampling inputs are required')
    if not isinstance(contexts, dict) or set(contexts) != native_sampling.CONTEXT_KEYS:
        raise ValueError('Exactly the saved positive and native-negative contexts are required')
    shape = PROFILES[profile]['shape']
    for key in ('initial_noise', 'initial_latent'):
        _float(values[key], shape, key)
    _float(values['observation'], (1, 48, 1, *shape[-2:]), 'observation')
    _float(commands, (1, 16, 6), 'commands')
    if not ((commands[..., 5] == 0) | (commands[..., 5] == 1)).all():
        raise ValueError('Interaction commands must be binary pulses')
    for key, context in contexts.items():
        if not isinstance(context, torch.Tensor) or context.ndim != 2 or not 1 <= context.shape[0] <= 512:
            raise ValueError('Text must have 1 through 512 tokens')
        _float(context, (context.shape[0], 4096), key)
    times = values['token_times']
    if (not isinstance(times, torch.Tensor) or times.device.type != 'cpu'
            or times.layout != torch.strided or times.dtype != torch.int64
            or not torch.equal(times, native_sampling.times_at(torch.tensor(999), shape))):
        raise ValueError('Saved first times must be zero on the prefix and 999 elsewhere')
    restored = values['initial_noise'].clone()
    restored[:, :1] = values['observation'][0]
    if not torch.equal(restored, values['initial_latent']):
        raise ValueError('Saved initial latent must be noise with the exact observation restored')
    return shape


def _sample_bridge(bridge, values, contexts, commands, profile, *, event=None, check=lambda: None):
    """Internal CPU-fixture seam; production constructs the unchanged CUDA bridge.

    The original sampler owns CFG, solver order and prefix restoration. Commands
    are identical on both text branches. Each callback receives private copies.
    """
    shape = validate_inputs(values, contexts, commands, profile)
    private_values = {key: value.clone() for key, value in values.items()}
    private_contexts = {key: value.clone() for key, value in contexts.items()}
    original_identity = {key: tensor_sha(value) for key, value in private_values.items()}
    original_text = {key: tensor_sha(value) for key, value in private_contexts.items()}
    command_identity = tensor_sha(commands)
    device = bridge.core.patch_embedding.weight.device
    device_commands = commands.clone().to(device)
    observation = private_values['observation'].clone().to(device)
    calls, steps, clean_prefix_calls = 0, 0, 0

    def predict(latent, times, context):
        nonlocal calls, clean_prefix_calls
        check()
        if not torch.equal(latent[:, :1], private_values['observation'][0]):
            raise ValueError('Sampler supplied a changed observed prefix')
        clean_prefix_calls += 1
        result = bridge(
            latent.unsqueeze(0).to(device), times.to(device), [context.to(device)],
            commands=device_commands.clone(), observation=observation.clone(), track_grad=False,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        result = result.detach().cpu()
        _float(result, (1, *shape), 'bridge velocity')
        calls += 1
        check()
        return result[0].clone()

    def completed(step, time, latent, velocities):
        nonlocal steps
        check()
        if step != steps or not torch.equal(latent[:, :1], private_values['observation'][0]):
            raise ValueError('Unexpected solver order or changed completed prefix')
        steps += 1
        if event is not None:
            event(step, time.clone(), latent.clone(), {k: v.clone() for k, v in velocities.items()})

    with torch.no_grad():
        final = native_sampling.sample(predict, private_values, private_contexts,
                                       latent_shape=shape, event=completed)
    if (calls != 100 or steps != 50 or clean_prefix_calls != 100
            or original_identity != {key: tensor_sha(value) for key, value in private_values.items()}
            or original_identity != {key: tensor_sha(value) for key, value in values.items()}
            or original_text != {key: tensor_sha(value) for key, value in private_contexts.items()}
            or original_text != {key: tensor_sha(value) for key, value in contexts.items()}
            or tensor_sha(commands) != command_identity):
        raise RuntimeError('Sampling count or input immutability check failed')
    check()
    return final, {'predictions': calls, 'solver_updates': steps, 'clean_prefix_calls': clean_prefix_calls,
                   'input_tensor_sha256': original_identity, 'text_tensor_sha256': original_text,
                   'commands_sha256': command_identity, 'final_latent_sha256': tensor_sha(final),
                   'profile': profile, 'settings': dict(native_sampling.SETTINGS),
                   'commands_on_cfg_branches': ['positive', 'native_negative'],
                   'negative_context_adapter_training': False, 'quality_assessed': False}


def sample_checkpoint(core, checkpoint, checkpoint_sha256, values, contexts, commands,
                      *, profile, event=None, check=lambda: None):
    """Called only by a separately admitted/guarded future parent runner.

    ``core`` is already loaded and independently verified. This function loads
    only a small adapter checkpoint, emits a latent and performs no decoding.
    ``check`` is called around every prediction and completed solver step.
    """
    validate_inputs(values, contexts, commands, profile)
    adapter, identity = load_adapter_checkpoint(checkpoint, checkpoint_sha256)
    if core.patch_embedding.weight.device.type != 'cuda':
        raise ValueError('Production sampling requires the already verified CUDA foundation')
    check()
    adapter.to(core.patch_embedding.weight.device)
    if {name: tensor_sha(value) for name, value in adapter.state_dict().items()} != identity['tensor_sha256']:
        raise RuntimeError('Adapter values changed during CUDA transfer')
    bridge = NativeCUDAActionBridge(core, adapter, profile=profile)
    final, record = _sample_bridge(bridge, values, contexts, commands, profile, event=event, check=check)
    if ({name: tensor_sha(value) for name, value in adapter.state_dict().items()} != identity['tensor_sha256']
            or any(parameter.grad is not None for parameter in adapter.parameters())):
        raise RuntimeError('Adapter values or gradients changed during inference')
    record['adapter'] = identity
    record['foundation_values_verified_by_this_kernel'] = False
    return final, record
