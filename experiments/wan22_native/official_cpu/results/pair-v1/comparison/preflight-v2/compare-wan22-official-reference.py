"""Read-only NumPy comparison of saved first-step velocities; no model imports."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


INPUT_SHA256 = '363ee00416161de834ab8213eb50a386abd42ce9cf3f34ff818e9213b9e9d2b5'
BASELINES = {
    'portable_cpu': ('cpu-pair-v1', '15b6993a7f5fe0057a41367208ea5f3d18fe49311fde28ce54c1f30a5d255c27'),
    'portable_mps': ('pair-v1', '70b6728371e5373b9e07b1055e2d23c59310d60f640b789240421ca6e42bc46d'),
}
VELOCITIES = ('positive_velocity', 'negative_velocity', 'guided_velocity')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensors(path):
    payload = path.read_bytes()
    if len(payload) < 8:
        raise ValueError('Missing safetensors header length')
    count = struct.unpack('<Q', payload[:8])[0]
    if count > 2**20 or count + 8 > len(payload):
        raise ValueError('Invalid bounded safetensors header')
    header = json.loads(payload[8:8+count])
    if not isinstance(header, dict):
        raise ValueError('Safetensors header must be an object')
    if '__metadata__' in header:
        metadata = header['__metadata__']
        if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
            raise ValueError('Safetensors metadata must map strings to strings')
    body = memoryview(payload)[8+count:]
    result = {}
    intervals = []
    for key, record in header.items():
        if key == '__metadata__':
            continue
        if record['dtype'] not in ('F32', 'I64'):
            raise ValueError('Only retained FP32 and int64 tensors are supported')
        dtype = np.dtype('<f4' if record['dtype'] == 'F32' else '<i8')
        start, end = record['data_offsets']
        shape = record['shape']
        size = int(np.prod(shape)) * dtype.itemsize
        if start < 0 or end < start or end > len(body) or end-start != size:
            raise ValueError('Invalid tensor byte range')
        result[key] = np.frombuffer(body[start:end], dtype=dtype).reshape(shape)
        intervals.append((start, end))
    intervals.sort()
    if not intervals or intervals[0][0] != 0 or intervals[-1][1] != len(body):
        raise ValueError('Unexpected unindexed payload')
    if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError('Overlapping or missing tensor bytes')
    return result


def statistics(actual, reference):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite, nonempty equal-size arrays required')
    a, b = a.reshape(-1), b.reshape(-1)
    delta = a - b
    rmse = float(np.sqrt(np.mean(delta * delta)))
    rms = float(np.sqrt(np.mean(b * b)))
    norm_product = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        'elements': int(a.size),
        'exact_value_equal': bool(np.array_equal(a, b)),
        'nonzero_difference_elements': int(np.count_nonzero(delta)),
        'maximum_absolute_difference': float(np.max(np.abs(delta))),
        'mean_absolute_difference': float(np.mean(np.abs(delta))),
        'rmse': rmse,
        'reference_rms': rms,
        'rmse_over_reference_rms': rmse / rms if rms else None,
        'cosine': float(np.dot(a, b) / norm_product) if norm_product else None,
    }


def guidance_residual(values):
    """Recompute three explicit eager FP32 operations; never replace saved data."""
    for name in VELOCITIES:
        value = values[name]
        if value.dtype != np.dtype('<f4') or value.shape != (48, 5, 18, 32) or not np.isfinite(value).all():
            raise ValueError('Guidance requires the declared finite FP32 velocity arrays')
    difference = np.subtract(values['positive_velocity'], values['negative_velocity'], dtype=np.float32)
    scaled = np.multiply(np.float32(5), difference, dtype=np.float32)
    expected = np.add(values['negative_velocity'], scaled, dtype=np.float32)
    return {
        'equation': 'negative + float32(5) * (positive - negative)',
        'operations': 'Separate NumPy subtract, multiply and add, each with dtype=float32',
        'saved_outputs_changed': False,
        'residual_denominator': 'Recomputed eager FP32 guidance; distinct from cross-model reference normalization',
        'all': statistics(values['guided_velocity'], expected),
        'observed': statistics(values['guided_velocity'][:, :1], expected[:, :1]),
        'future': statistics(values['guided_velocity'][:, 1:], expected[:, 1:]),
    }


def load_reference_run(repo, directory):
    """Bind output bytes to a completed run and retained/current source bytes."""
    directory = Path(directory)
    parent_path = directory / 'metrics.json'
    result_path = directory / 'result/metrics.json'
    terminal_path = directory / 'terminal.json'
    parent = json.loads(parent_path.read_text())
    result = json.loads(result_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    if (parent.get('status') != 'passed' or parent.get('weights_loaded') is not True
            or parent.get('model_execution') is not True
            or type(parent.get('pair_predictions')) is not int or parent['pair_predictions'] != 2
            or parent.get('solver_steps') != 0
            or terminal.get('status') != 'complete' or type(terminal.get('exit_code')) is not int or terminal['exit_code'] != 0
            or result.get('status') != 'passed' or result.get('device') != 'cpu'
            or type(result.get('completed_predictions')) is not int or result['completed_predictions'] != 2
            or result.get('finite_outputs') is not True or result.get('caller_inputs_unchanged') is not True
            or [p.get('prediction') for p in result.get('passes', [])] != ['positive', 'negative']
            or list(directory.rglob('*watchdog-stop*'))):
        raise ValueError('Completed, finite, exactly two-prediction CPU reference with no watchdog stop required')
    if digest(result_path) != parent.get('result_metrics_sha256'):
        raise ValueError('Reference worker report differs from the completed parent identity')
    if result.get('input_identity') != parent.get('input_identity') or not isinstance(parent.get('input_identity'), dict):
        raise ValueError('Reference parent and worker input identities differ')
    inputs_path = directory / 'inputs.safetensors'
    if digest(inputs_path) != INPUT_SHA256 or parent.get('input_file_sha256') != INPUT_SHA256:
        raise ValueError('Exact pinned reference input bytes required')
    official = Path(repo) / 'experiments/wan22_native/official_cpu'
    tree = ast.parse((official / 'run.py').read_text())
    declarations = [node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'NAMES' for target in node.targets)]
    if len(declarations) != 1:
        raise ValueError('A single literal official CPU source-name declaration is required')
    names = ast.literal_eval(declarations[0])
    if (not isinstance(names, tuple) or not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts for name in names)
            or not {'run.py', 'reference.py', 'streaming.py', 'inputs.py', 'vendor/model.py'}.issubset(names)):
        raise ValueError('Invalid complete official CPU source graph')
    sources = {name: digest(official / name) for name in names}
    if parent.get('source_sha256') != sources or result.get('source_sha256') != sources:
        raise ValueError('Reference source hashes differ from the supplied current official CPU source')
    for name, expected in sources.items():
        if digest(directory / 'measured-source' / (name + '.txt')) != expected:
            raise ValueError('Retained reference source snapshot changed: ' + name)
    outputs_path = directory / 'result/outputs.safetensors'
    outputs_sha256 = digest(outputs_path)
    if outputs_sha256 != result.get('output_sha256', {}).get('outputs.safetensors'):
        raise ValueError('Reference output bytes differ from the passed worker report')
    values = tensors(outputs_path)
    if set(values) != set(VELOCITIES):
        raise ValueError('Expected exactly the positive, negative and guided reference velocities')
    return values, {
        'parent_metrics_sha256': digest(parent_path), 'result_metrics_sha256': digest(result_path),
        'terminal_sha256': digest(terminal_path), 'inputs_sha256': INPUT_SHA256,
        'outputs_sha256': outputs_sha256, 'source_sha256': sources,
        'retained_and_current_sources_match': True, 'completed_predictions': 2,
        'no_watchdog_record': True,
    }


def checks():
    # Analytic cases exercise sign, normalization, zero denominator and FP32 input.
    assert statistics([1., -1.], [1., -1.])['rmse'] == 0
    reverse = statistics([-1., 1.], [1., -1.])
    assert reverse['rmse'] == 2 and reverse['rmse_over_reference_rms'] == 2
    assert abs(reverse['cosine'] + 1) < 1e-15
    assert statistics([2., 2.], [0., 0.])['rmse_over_reference_rms'] is None
    assert statistics([0.], [0.])['cosine'] is None
    pair = statistics(np.array([1, 3], dtype=np.float32), np.array([0, 1], dtype=np.float32))
    assert pair['maximum_absolute_difference'] == 2 and pair['mean_absolute_difference'] == 1.5
    assert abs(pair['rmse'] - np.sqrt(2.5)) < 1e-15
    try:
        statistics([float('nan')], [0.])
    except ValueError:
        return 7
    raise AssertionError('Nonfinite data accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('A fresh report path is required')
    reference, reference_identity = load_reference_run(args.repo, args.reference_run)
    for value in reference.values():
        if value.dtype != np.dtype('<f4') or value.shape != (48, 5, 18, 32) or not np.isfinite(value).all():
            raise ValueError('Reference velocity dtype, shape or values differ')
    result = {
        'schema': 'worldline-wan22-official-reference-comparison-v2',
        'scope': 'Descriptive first-step numerical comparison only; no visual quality or CUDA-equivalence conclusion',
        'normalization': 'The streamed official CPU output is the denominator in every cross-model comparison; guidance residuals use their own recomputed guidance denominator',
        'zero_denominators': 'Relative RMSE and cosine are null when their denominators are zero',
        'source_sha256': digest(Path(__file__)),
        'reference_inputs_sha256': reference_identity['inputs_sha256'],
        'reference_outputs_sha256': reference_identity['outputs_sha256'],
        'reference_run_identity': reference_identity,
        'numpy_version': np.__version__,
        'analytic_checks': checks(),
        'comparisons': {},
        'guidance_equation_residuals': {'official_reference': guidance_residual(reference)},
    }
    for label, (directory, expected) in BASELINES.items():
        root = args.repo / 'experiments/wan22_native/core-results' / directory
        assert digest(root / 'inputs.safetensors') == INPUT_SHA256
        assert digest(root / 'outputs.safetensors') == expected
        baseline = tensors(root / 'outputs.safetensors')
        result['guidance_equation_residuals'][label] = guidance_residual(baseline)
        comparisons = {}
        for name in VELOCITIES:
            comparisons[name] = {
                'all': statistics(baseline[name], reference[name]),
                'observed': statistics(baseline[name][:, :1], reference[name][:, :1]),
                'future': statistics(baseline[name][:, 1:], reference[name][:, 1:]),
            }
        result['comparisons'][label] = {'outputs_sha256': expected, 'velocities': comparisons}
    if not result['guidance_equation_residuals']['portable_cpu']['all']['exact_value_equal']:
        raise ValueError('Pinned eager CPU baseline no longer matches the specified FP32 guidance equation')
    if not result['guidance_equation_residuals']['official_reference']['all']['exact_value_equal']:
        raise ValueError('Official CPU reference guidance differs from its verified eager FP32 equation')
    result['reference_guidance_exact_required'] = True
    result['status'] = 'computed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'analytic_checks': result['analytic_checks'],
        'report_sha256': digest(args.output)}))


if __name__ == '__main__':
    main()
