# SPDX-License-Identifier: Apache-2.0
"""Plan by default. Explicit native CUDA resource/gradient profile dispatch."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SCHEMA = 'worldline-command-attention-profile-v1'
SHAPE = [1, 48, 5, 44, 78]
ARMS = tuple(m + '_' + d for m in ('stationary', 'left', 'right') for d in ('closed', 'interact'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def protocol():
    return dict(schema=SCHEMA, scope='native-resource-and-gradient-profile-only',
        model_execution=False, image_generation=False, quality_assessed=False,
        automatic_training_promotion=False, latent_shape=SHAPE, rgb_size=[1248, 704],
        frames=17, tokens=4290, controller_blocks=list(range(24, 30)),
        controller_rank=32, controller_parameters=4_936_448,
        zero_gate=dict(contexts=['positive', 'negative'], arms=list(ARMS),
            paths=['full', 'cached'], comparisons=24, requirement='bit-exact FP32 native equality'),
        updates=[dict(update=1, motion='stationary', auxiliary=True, auxiliary_edge=0),
                 dict(update=2, motion='left', auxiliary=False)],
        training_math='Exact separately supplied math_steps module; one mixed FM/CFG auxiliary step and one FM step',
        limits=dict(seconds=900, host_rss_bytes=48*2**30, cuda_reserved_bytes=60*2**30,
                    minimum_host_available_bytes=8*2**30, minimum_cuda_available_bytes=8*2**30))


def execution_arguments(args, now=None):
    """Reject absent identity, paths or expired deadlines before any GPU import."""
    for name in ('repository', 'controller_source', 'training_source', 'prepared', 'weights', 'output', 'inputs_sha256', 'expected_gpu', 'deadline_utc'):
        if not getattr(args, name, None):
            raise ValueError('Explicit --' + name.replace('_', '-') + ' required for execution')
    if not args.expected_gpu.strip():
        raise ValueError('Exact nonempty GPU name required')
    end = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if end.tzinfo is None:
        raise ValueError('Deadline must include a UTC offset')
    now = now or datetime.now(timezone.utc)
    remaining = (end - now).total_seconds()
    if not 0 < remaining <= 900:
        raise ValueError('Deadline must be in the future and at most 900 seconds away')
    for name in ('repository', 'controller_source', 'training_source', 'prepared', 'weights'):
        path = Path(getattr(args, name))
        if not path.is_dir() or path.is_symlink():
            raise ValueError(name + ' must be an existing directory')
    if len(args.inputs_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.inputs_sha256):
        raise ValueError('Explicit lowercase SHA-256 input manifest identity required')
    if sha(Path(args.prepared)/'inputs.json') != args.inputs_sha256:
        raise ValueError('Prepared input manifest differs from the declared SHA-256')
    out = Path(args.output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError('Fresh output with regular ancestors required')
    for name in ('repository', 'controller_source', 'training_source', 'prepared', 'weights'):
        if out.resolve().is_relative_to(Path(getattr(args, name)).resolve()):
            raise ValueError('Output must be outside source and input directories')
    return remaining


def paths(args):
    repo = Path(args.repository).resolve()
    result = {'profiler/' + p.name: p for p in (HERE/'run_profile.py', HERE/'engine.py', HERE/'native_worker.py')}
    result.update({'controller/' + n: Path(args.controller_source).resolve()/n for n in ('controller.py', 'bridge.py')})
    for name in ('math_steps.py', 'packet.py'):
        result['training/'+name] = Path(args.training_source).resolve()/name
    for name in (
        'cuda_reference/native.py', 'cuda_reference/guards.py', 'spatial_reference/guards.py',
        'cuda_reference/config.json', 'cuda_reference/upstream-provenance.json', 'cuda_reference/expected-weights.json',
        'action_training/objective.py', 'action_cuda/probe.py', 'official_cpu/streaming.py'):
        result['repository/' + name] = repo/'experiments/wan22_native'/name
    return result


def dispatch(args):
    execution_arguments(args)
    end = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    deadline = time.monotonic() + (end-datetime.now(timezone.utc)).total_seconds()
    config = {name: str(Path(getattr(args, name)).resolve()) for name in
              ('repository', 'controller_source', 'training_source', 'prepared', 'weights', 'output')}
    config.update(schema=SCHEMA, expected_gpu=args.expected_gpu, deadline_utc=args.deadline_utc,
                  deadline=deadline, protocol=protocol(),
                  source_sha256={name: sha(path) for name, path in paths(args).items()},
                  inputs_sha256=args.inputs_sha256)
    sys.path.insert(0, config['repository'])
    from experiments.wan22_native.spatial_reference.guards import atomic, supervise
    out = Path(config['output']); out.mkdir(parents=True)
    atomic(out/'launch.json', config)
    with (out/'worker.log').open('x') as log:
        process = subprocess.Popen([sys.executable, str(HERE/'run_profile.py'), '--worker-config', str(out/'launch.json')],
                                   stdout=log, stderr=subprocess.STDOUT)
        supervise(process, out, config['deadline'], 'pair')
    terminal = json.loads((out/'terminal.json').read_text())
    result = json.loads((out/'result/metrics.json').read_text()) if (out/'result/metrics.json').exists() else {}
    if terminal.get('status') != 'complete' or result.get('status') != 'passed':
        raise RuntimeError('Profile did not complete; inspect retained terminal and metrics')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-config', type=Path, help=argparse.SUPPRESS)
    for name in ('repository', 'controller-source', 'training-source', 'prepared', 'weights', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--inputs-sha256')
    parser.add_argument('--expected-gpu')
    parser.add_argument('--deadline-utc')
    args = parser.parse_args(argv)
    if args.worker_config:
        from native_worker import worker
        value = worker(json.loads(args.worker_config.read_text()))
    elif args.execute:
        value = dispatch(args)
    else:
        value = protocol()
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
