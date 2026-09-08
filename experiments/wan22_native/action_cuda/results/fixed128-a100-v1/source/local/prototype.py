"""Isolated fixed128 computation prototype. No loader, launcher or execution CLI."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from experiments.wan22_native.action_cuda import probe_math as numerical
from experiments.wan22_native.action_cuda.bridge import PROFILES
from experiments.wan22_native.action_training import objective as original
from experiments.wan22_native.official_cpu.streaming import sha, tensor_sha

UPDATES = 128
STARTS = (0, 8, 32, 49) * 32
CHECKPOINTS = tuple(range(0, 129, 16))


def protocol(profile):
    if profile not in PROFILES:
        raise ValueError('Declare baseline or spatial profile')
    return {'schema': 'worldline-action-cuda-fixed128-prototype-v1', 'profile': profile,
            'paired_updates': UPDATES, 'starts': list(STARTS), 'branch_order': ['closed', 'open'],
            'seed': original.SEED, 'shape': [1, *PROFILES[profile]['shape']],
            'optimizer': {**original.OPTIMIZER, 'betas': list(original.OPTIMIZER['betas'])},
            'gradient_clip_l2': 1., 'fresh_adapter_and_optimizer': True, 'resume_supported': False,
            'flow_equation': 'sigma=k/1000; noisy future=(1-sigma)*target+sigma*noise; velocity=noise-target',
            'loss': 'Half each branch future-only latent velocity MSE; initial latent excluded',
            'codec_prefix_limits': numerical.PREFIX, 'native_zero_adapter_limits': numerical.PARITY,
            'training_forwards': 2*UPDATES, 'initial_native_parity_forwards': 2, 'initial_bridge_parity_forwards': 2,
            'completed_checkpoints': list(CHECKPOINTS), 'model_execution': False,
            'production_admission_implemented': False, 'runtime_measured': False,
            'foundation_storage': 'Original FP32 native CUDA; unchanged BF16 autocast/FA2 bridge contract',
            'quality_assessed': False, 'image_generation': False}


def fresh_inputs(shape, *, prefix_schedule, prefix_draws, initial):
    """Load original cp0 bytes and continue the retained private CPU RNG once."""
    from extension import extend_draws
    schedule, draws = extend_draws(prefix_schedule, prefix_draws, shape)
    return {k:v.detach().cpu().clone() for k,v in initial.items()}, schedule, draws


def expected_commands(arm, start):
    commands = torch.zeros(1, 16, 6, dtype=torch.float32)
    if arm not in ('closed', 'open') or start not in (0, 8, 32, 49):
        raise ValueError('Unknown original capture window')
    if start == 0:
        commands[:, 1:, 3] = math.pi / 24
        commands[:, 0, 5] = int(arm == 'open')
    elif start == 8:
        commands[:, :, 3] = math.pi / 24
    elif start == 32:
        commands[:, 9:, 3] = -math.pi / 24
    else:
        commands[:, :, 3] = -math.pi / 24
    return commands


def validate_inputs(windows, schedule, draws, shape):
    keys = {f'{arm}-{start:04d}' for start in (0, 8, 32, 49) for arm in ('closed', 'open')}
    if set(windows) != keys or len(schedule) != UPDATES:
        raise ValueError('Require all eight windows and 128 paired updates')
    for identity, window in windows.items():
        arm, start = identity.split('-'); start = int(start)
        if set(window) != {'target', 'observation', 'commands'}:
            raise ValueError('Only target, independent observation and commands are inputs')
        if (tuple(window['target'].shape) != shape
                or tuple(window['observation'].shape) != (1, shape[1], 1, *shape[-2:])
                or window['commands'].dtype != torch.float32 or window['commands'].device.type != 'cpu'
                or not torch.equal(window['commands'], expected_commands(arm, start))):
            raise ValueError('Original window shape or destination-aligned command sequence differs')
        numerical.cross_length_prefix(window['target'][:, :, :1], window['observation'])
    if not torch.equal(windows['closed-0000']['observation'], windows['open-0000']['observation']):
        raise ValueError('Start0 must share the exact independent canonical observation')
    expected_draws = {'rng_initial'} | {f'{kind}_{i:04d}' for i in range(UPDATES) for kind in ('noise', 'rng_after')}
    if set(draws) != expected_draws:
        raise ValueError('Exactly 128 saved noises and generator states are required')
    for index, (row, start) in enumerate(zip(schedule, STARTS)):
        key = f'noise_{index:04d}'; state = f'rng_after_{index:04d}'
        if (row.get('update') != index + 1 or row.get('start') != start
                or row.get('branches') != [f'closed-{start:04d}', f'open-{start:04d}']
                or type(row.get('k')) is not int or not 50 <= row['k'] <= 950
                or row.get('sigma') != row['k'] / 1000. or row.get('noise_key') != key
                or tuple(draws[key].shape) != shape or draws[key].dtype != torch.float32
                or draws[key].device.type != 'cpu' or draws[key].requires_grad
                or not torch.isfinite(draws[key]).all()
                or row.get('noise_sha256') != tensor_sha(draws[key])
                or row.get('rng_after_sha256') != tensor_sha(draws[state])):
            raise ValueError('Saved chronological schedule or draw bytes differ')


def _save(path, values):
    path = Path(path)
    if path.exists():
        raise ValueError('Raw prototype evidence cannot be overwritten')
    save_file({key: value.detach().cpu().contiguous().clone() for key, value in values.items()}, str(path))
    return sha(path)


def _write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def initialization_hash(value):
    """Hash one explicit CPU verification copy, releasing its owner afterward."""
    copied = value.detach().to(device='cpu', copy=True).contiguous()
    try:
        return tensor_sha(copied)
    finally:
        del copied


def run_sequence(bridge, windows, schedule, draws, context, optimizer, output, *,
                 expected_initial, expected_core, native_predict, identity,
                 check=lambda: None, synchronize=lambda: None):
    """Computation component only. A future guarded owner must supply admission.

    Reuse the unchanged two-update routine, including its before-update parity
    checks, then the unchanged paired objective for the remaining 126.
    This function neither loads a core nor launches any child or CUDA process.
    """
    shape = (1, *bridge.shape)
    validate_inputs(windows, schedule, draws, shape)
    if optimizer.state:
        raise ValueError('Reject a resumed optimizer')
    for group in optimizer.param_groups:
        if any(group.get(key) != value for key, value in original.OPTIMIZER.items()):
            raise ValueError('Keep the original fixed128 optimizer settings')
    if set(expected_initial) != set(bridge.adapter.state_dict()):
        raise ValueError('Fresh initial parameter coverage differs')
    for name, value in bridge.adapter.state_dict().items():
        if initialization_hash(value) != tensor_sha(expected_initial[name]):
            raise ValueError('Adapter differs from the separately created fresh initialization')
    count = len(expected_core) if bridge.test_only else 825
    out = Path(output).absolute()
    repo = Path(numerical.__file__).resolve().parents[3]
    if (out.exists() or out.resolve().is_relative_to(repo)
            or any(path.is_symlink() for path in (out, *out.parents))):
        raise ValueError('A fresh prototype output outside the repository is required')
    out.mkdir(parents=True)
    report = {'schema': 'worldline-action-cuda-fixed128-prototype-result-v1', 'status': 'running',
              'identity': identity, 'completed_updates': 0, 'updates': [], 'schedule': schedule,
              'raw_files': {}, 'base_unchanged': None, 'resume_supported': False,
              'production_admission_implemented': False, 'quality_assessed': False}
    began = time.monotonic()

    def retain(name, values):
        report['raw_files'][name + '.safetensors'] = _save(out / (name + '.safetensors'), values)

    def progress(value):
        report.update(copy.deepcopy(value));report['status']='running'
        _write(out / 'metrics.json', report)

    def checkpoint(completed, current):
        check()
        if completed:
            if len(optimizer.state) != sum(1 for _ in bridge.adapter.parameters()):
                raise RuntimeError('Every adapter parameter must have an AdamW state')
            if any(float(value['step']) != completed for value in optimizer.state.values()):
                raise RuntimeError('AdamW step counters differ from the completed pair count')
            retain(f'gradients-after-clip-{completed:04d}',
                   {name: value.grad for name, value in bridge.adapter.named_parameters()})
        rng = draws['rng_initial'] if completed == 0 else draws[f'rng_after_{completed-1:04d}']
        if completed in CHECKPOINTS:
            record = original.save_checkpoint(out, bridge.adapter, optimizer, completed,
                                              identity=identity, draw_rng_state=rng)
            report['last_checkpoint'] = {'directory': record['directory'], 'manifest_sha256': record['manifest_sha256'],
                                         'completed_updates': completed}
        progress(current)

    class RetainingBridge:
        core, adapter = bridge.core, bridge.adapter
        training_calls = 0
        def __call__(self, *args, **kwargs):
            value = bridge(*args, **kwargs)
            if kwargs.get('track_grad', True):
                index, branch = divmod(self.training_calls, 2)
                retain(f'prediction-{index+1:04d}-' + ('closed', 'open')[branch], {'velocity': value})
                self.training_calls += 1
            return value

    proxy = RetainingBridge()
    try:
        before = original.parameter_records(bridge.core, check=check, expected_count=count, expected=expected_core)
        _write(out / 'core-before.json', before)
        retain('initial-adapter', expected_initial)
        # Prepared chunk files retain every exact draw once; no duplicate 422 MB draw file.
        _write(out / 'draw-inputs.json', identity.get('prepared_draw_files', {}))
        retain('positive', {'context': context})
        input_hashes = {key: {name: tensor_sha(value) for name, value in window.items()} for key, window in windows.items()}
        _write(out / 'window-tensor-hashes.json', input_hashes)
        current = numerical.execute_steps(proxy, windows, schedule[:2], draws, context, optimizer,
                                            native_predict=native_predict, retain=retain, checkpoint=checkpoint,
                                            progress=progress, check=check, synchronize=synchronize)
        for index, row in enumerate(schedule[2:], 3):
            check(); synchronize(); start_time = time.monotonic()
            result = numerical.paired_update(proxy, [windows[key] for key in row['branches']],
                                             draws[row['noise_key']], row['k'], context, optimizer, check=check)
            synchronize()
            result.update(update=index, start=row['start'], seconds=time.monotonic() - start_time)
            current['updates'].append(result)
            current['bridge_predictions'] += 2
            current['completed_updates'] = index
            checkpoint(index, current)
        after = original.parameter_records(bridge.core, check=check, expected_count=count, expected=before)
        _write(out / 'core-after.json', after)
        if input_hashes != {key: {name: tensor_sha(value) for name, value in window.items()} for key, window in windows.items()}:
            raise RuntimeError('A cached input changed during training')
        validate_inputs(windows, schedule, draws, shape)
        if (proxy.training_calls != 2*UPDATES or current['bridge_predictions'] != 2*UPDATES+2
                or current['native_reference_predictions'] != 2 or current['completed_updates'] != UPDATES):
            raise RuntimeError('Fixed128 forward or optimizer counts differ')
        report.update(current, status='passed', base_unchanged=True, inputs_unchanged=True,
                      training_forwards=proxy.training_calls)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - began
        _write(out / 'metrics.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(PROFILES), default='spatial')
    print(json.dumps(protocol(parser.parse_args().profile), indent=2))
