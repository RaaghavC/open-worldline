# SPDX-License-Identifier: Apache-2.0
"""Plan-first, separately guarded original CUDA VAE action-data cache."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..official_cpu.streaming import sha
from ..spatial_reference.evidence import sources as spatial_sources
from ..spatial_reference.guards import atomic, limits, stop_child, supervise
from . import data

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCHEMA = 'worldline-wan22-action-cuda-cache-run-v1'
CPU_SCHEMA = 'worldline-wan22-action-cuda-cache-cpu-v1'


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sources():
    result = dict(spatial_sources())
    result.update(data.sources())
    for name in ('__init__.py', 'cache_run.py', 'cache_worker.py', 'cache_review.py'):
        path = HERE / name
        result[str(path.relative_to(REPO))] = sha(path)
    return dict(sorted(result.items()))


def tests():
    return {name: sha(HERE / name) for name in ('test_data.py', 'test_cache_run.py')}


def read_json(path):
    path = Path(path).absolute()
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)) or path.stat().st_size > 4 * 2**20:
        raise ValueError('A bounded regular JSON record is required')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('A JSON object is required')
    return value


def preflight(path):
    value = read_json(path)
    if (value.get('schema') != CPU_SCHEMA or value.get('status') != 'passed'
            or value.get('source_sha256') != sources() or value.get('test_sha256') != tests()
            or type(value.get('tests_run')) is not int or value['tests_run'] < 5
            or any(type(value.get(key)) is not int or value[key] != 0 for key in ('failures', 'errors', 'skipped'))
            or value.get('cuda_initialized') is not False):
        raise ValueError('Require a passing CPU report bound to all current cache sources and tests')
    return sha(path)


def make_plan(capture, profile, expected_gpu, cpu_report):
    if profile not in ('baseline', 'spatial') or expected_gpu != 'NVIDIA A100-SXM4-80GB':
        raise ValueError('Require a declared profile and the reviewed A100 model')
    review_sha = preflight(cpu_report)
    source_map = sources()
    prepared = data.plan(Path(capture), profile)
    if source_map != sources():
        raise ValueError('Cache sources changed during planning')
    return {'schema': SCHEMA, 'mode': 'cache', 'profile': profile,
            'expected_gpu': expected_gpu, 'limits': limits('codec'),
            'source_sha256': source_map, 'cpu_report_sha256': review_sha,
            'input_plan': prepared, 'input_plan_sha256': canonical_sha(prepared),
            'model_execution': False, 'quality_assessed': False}


def fresh_output(path, capture):
    path = Path(path).absolute()
    if (path.exists() or any(p.is_symlink() for p in (path, *path.parents))
            or path.resolve().is_relative_to(REPO)
            or path.resolve().is_relative_to(Path(capture).resolve())):
        raise ValueError('Fresh output outside source and original input directories is required')
    return path


def validate_completed(out, expected):
    out = Path(out)
    terminal = read_json(out / 'terminal.json')
    worker = read_json(out / 'worker/metrics.json')
    monitor = read_json(out / 'worker/monitor-terminal.json')
    completion = read_json(out / 'result/completion.json')
    if (terminal.get('status') != 'complete' or type(terminal.get('exit_code')) is not int
            or terminal['exit_code'] != 0 or terminal.get('mode') != 'codec'
            or terminal.get('cleanup_error') is not None
            or worker.get('status') != 'passed' or worker.get('mode') != 'cache'
            or worker.get('model_execution') is not True
            or monitor.get('status') != 'complete' or type(monitor.get('sample_count')) is not int
            or monitor['sample_count'] < 1 or completion.get('status') != 'passed'
            or list(out.rglob('watchdog-stop.json'))):
        raise RuntimeError('Cache worker, monitor or supervisor did not complete successfully')
    for key in ('profile', 'source_sha256', 'input_plan_sha256', 'cpu_report_sha256'):
        if worker.get(key) != expected[key]:
            raise ValueError('Completed cache identity differs: ' + key)
    if (worker.get('result_completion_sha256') != sha(out / 'result/completion.json')
            or completion.get('manifest_sha256') != sha(out / 'result/manifest.json')):
        raise ValueError('Completed cache manifest or completion bytes changed')
    for arm, start in data.SELECTION:
        data.read_window(out / 'result', f'{arm}-{start:04d}')
    return worker


def launch(config, out):
    out = Path(out)
    atomic(out / 'launch.json', config)
    proc = None
    try:
        with (out / 'worker.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-m', __package__ + '.cache_worker',
                                     '--config', str((out / 'launch.json').absolute())],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            supervise(proc, out, config['deadline'], 'codec')
    except BaseException as error:
        cleanup_error = None
        try:
            if proc is not None:
                stop_child(proc)
        except BaseException as cleanup:
            cleanup_error = str(cleanup)
        finally:
            if not (out / 'terminal.json').exists():
                atomic(out / 'terminal.json', {'status': 'failed', 'exit_code': proc.returncode if proc else None,
                    'mode': 'codec', 'error': str(error), 'cleanup_error': cleanup_error})
            elif cleanup_error:
                atomic(out / 'handoff-cleanup-error.json', {'error': str(error), 'cleanup_error': cleanup_error})
        raise
    return validate_completed(out, config)


def run(*, capture, profile, weights, cpu_report, output, expected_gpu,
        input_plan_sha256=None, execute=False):
    if type(execute) is not bool:
        raise ValueError('execute must be an explicit boolean')
    out = fresh_output(output, capture)
    out.mkdir(parents=True)
    began = time.monotonic()
    report = {'schema': SCHEMA, 'mode': 'cache', 'profile': profile, 'status': 'running',
              'execute_requested': execute, 'model_execution': False, 'quality_assessed': False,
              'limits': limits('codec')}
    atomic(out / 'metrics.json', report)
    try:
        plan = make_plan(capture, profile, expected_gpu, cpu_report)
        report.update(plan)
        atomic(out / 'plan.json', plan)
        for name, digest in plan['source_sha256'].items():
            destination = out / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, destination)
            if sha(destination) != digest:
                raise RuntimeError('Cache source changed during snapshot')
        shutil.copyfile(cpu_report, out / 'cpu-report.json')
        if sha(out / 'cpu-report.json') != plan['cpu_report_sha256']:
            raise ValueError('CPU review changed during snapshot')
        if not execute:
            report['status'] = 'planned'
            return report
        if input_plan_sha256 != plan['input_plan_sha256']:
            raise ValueError('Explicit matching input-plan SHA256 required before cache execution')
        deadline = began + limits('codec')['seconds']
        if time.monotonic() >= deadline:
            raise RuntimeError('Cache deadline reached before worker launch')
        config = {**plan, 'capture': str(Path(capture).absolute()), 'weights': str(Path(weights).absolute()),
                  'cpu_report': str((out / 'cpu-report.json').absolute()),
                  'output': str(out), 'deadline': deadline}
        report.update(worker_started=True, model_execution=None)
        atomic(out / 'metrics.json', report)
        worker = launch(config, out)
        if sources() != plan['source_sha256'] or canonical_sha(data.plan(Path(capture), profile)) != plan['input_plan_sha256']:
            raise ValueError('Cache sources or original capture changed during execution')
        if time.monotonic() >= deadline:
            raise RuntimeError('Cache deadline reached during parent final verification')
        report.update(status='passed', model_execution=True, hardware=worker['hardware'],
                      result_completion_sha256=worker['result_completion_sha256'],
                      worker_metrics_sha256=sha(out / 'worker/metrics.json'),
                      inputs_unchanged=True, sources_unchanged=True)
        return report
    except BaseException as error:
        if report.get('worker_started') and (out / 'worker/metrics.json').is_file():
            try:
                partial = read_json(out / 'worker/metrics.json')
                value = partial.get('model_execution')
                if type(value) is bool:
                    report['model_execution'] = value
            except (ValueError, OSError):
                pass
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - began
        atomic(out / 'metrics.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('baseline', 'spatial'), required=True)
    parser.add_argument('--expected-gpu', required=True)
    for name in ('capture', 'weights', 'cpu-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--input-plan-sha256')
    parser.add_argument('--execute', action='store_true')
    print(json.dumps(run(**vars(parser.parse_args(argv))), allow_nan=False))


if __name__ == '__main__':
    main()
