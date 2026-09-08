# SPDX-License-Identifier: Apache-2.0
"""Independent CPU audit of saved native spatial prediction artifacts.

No spatial_reference validator or predictor is called. All tensor values are
read from bounded retained files; no model, CUDA context or weights are loaded.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct

import numpy as np
import torch
from safetensors import safe_open


SHAPES = {'baseline': (48, 5, 18, 32), 'spatial': (48, 5, 44, 78)}
HEADER_LIMIT = 2**20
TENSOR_FILE_LIMIT = 32 * 2**20
PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
CATALOG_NAME = 'experiments/wan22_native/cuda_reference/expected-weights.json'
SOLVER_NAME = 'experiments/wan22_native/cuda_reference/vendor/fm_solvers_unipc.py'
SOLVER_SHA256 = '0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9'
_UNBOUND = object()


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def _tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(x not in '0123456789abcdef' for x in value):
        raise ValueError('A lowercase SHA256 is required')
    return value


def _path(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
            or PurePosixPath(name).is_absolute() or str(PurePosixPath(name)) != name
            or any(x in ('', '.', '..') for x in name.split('/'))):
        raise ValueError('A canonical relative artifact path is required')
    root = Path(root).absolute()
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('Artifact roots must be real directories without symlink ancestors')
    result = root
    for component in name.split('/'):
        result = result / component
        if result.is_symlink():
            raise ValueError('Symlink artifacts are rejected')
    if not result.is_file():
        raise ValueError('A retained regular artifact is required: ' + name)
    return result


def _object(raw):
    def pairs(items):
        value = {}
        for name, item in items:
            if name in value:
                raise ValueError('Duplicate JSON keys are rejected')
            value[name] = item
        return value
    def invalid(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    def real(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=real)
    if not isinstance(value, dict):
        raise ValueError('A JSON object is required')
    return value


def _json(path):
    if path.stat().st_size > 4 * 2**20:
        raise ValueError('JSON artifact exceeds the audit bound')
    return _object(path.read_bytes())


def _read(path, expected, select=None):
    """Validate every header before materializing only the requested keys.

    expected maps names to (shape, 'F32' or 'I64'). Offsets must cover the exact
    data payload with no overlaps, gaps or trailing bytes. Metadata is strings.
    """
    size = path.stat().st_size
    if not 10 <= size <= TENSOR_FILE_LIMIT:
        raise ValueError('A bounded safetensors file is required')
    with path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if not 2 <= header_size <= HEADER_LIMIT or 8 + header_size > size:
            raise ValueError('Invalid safetensors header bound')
        header = _object(stream.read(header_size))
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ValueError('Safetensors metadata must contain strings only')
    if set(header) != set(expected):
        raise ValueError('Exact tensor keys required before materialization')
    intervals = []
    for name, (shape, dtype) in expected.items():
        row = header[name]
        if (not isinstance(row, dict) or set(row) != {'shape', 'dtype', 'data_offsets'}
                or row['shape'] != list(shape) or any(type(x) is not int for x in row['shape'])
                or row['dtype'] != dtype or dtype not in ('F32', 'I64')):
            raise ValueError('Tensor shape or dtype differs before materialization: ' + name)
        offsets = row['data_offsets']
        if (not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets)
                or offsets[0] < 0 or offsets[1] - offsets[0] != math.prod(shape) * (4 if dtype == 'F32' else 8)):
            raise ValueError('Invalid tensor offsets')
        intervals.append(tuple(offsets))
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError('Tensor payload must not overlap or contain gaps')
        end = stop
    if end != size - 8 - header_size:
        raise ValueError('Tensor payload has missing or trailing bytes')
    selected = tuple(expected) if select is None else tuple(select)
    if not selected or not set(selected) <= set(expected):
        raise ValueError('Only declared tensor keys may be materialized')
    before = _sha(path)
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        values = {name: handle.get_tensor(name) for name in selected}
    for name, value in values.items():
        shape, dtype = expected[name]
        if (value.device.type != 'cpu' or tuple(value.shape) != tuple(shape)
                or value.dtype != (torch.float32 if dtype == 'F32' else torch.int64)
                or not torch.isfinite(value).all().item()):
            raise ValueError('Finite declared CPU tensor values are required: ' + name)
    if _sha(path) != before:
        raise ValueError('Tensor artifact changed during audit')
    return values


def _equal(left, right, label):
    if (left.shape != right.shape or left.dtype != right.dtype
            or not torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))):
        raise ValueError('Exact tensor bytes differ: ' + label)


def _positive(value):
    if type(value) not in (int, float):
        raise ValueError('A finite positive timing is required')
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError('Timing overflow') from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError('A finite positive timing is required')
    return number


class _Audit:
    def __init__(self):
        self.files = {}

    def file(self, root, relative, label, expected=_UNBOUND):
        path = _path(root, relative)
        digest = _sha(path)
        if expected is not _UNBOUND and digest != _digest(expected):
            raise ValueError('Bound artifact hash differs: ' + label)
        self.files[label] = digest
        return path

    def completion(self, root, mode, profile, stage, label):
        root = Path(root)
        if any(root.rglob('watchdog-stop.json')):
            raise ValueError('A stopped run cannot be prediction evidence')
        parent = _json(self.file(root, 'metrics.json', label + '/metrics.json'))
        if (parent.get('status') != 'passed' or parent.get('model_execution') is not True
                or parent.get('mode') != mode or parent.get('profile') != profile
                or parent.get('settings') != {'steps': 50, 'shift': 5.0, 'guidance': 5.0}):
            raise ValueError('A completed explicitly executed matching run is required')
        reports, terminals = parent.get('child_reports'), parent.get('child_terminals')
        report_name, terminal_name = stage + '/result/metrics.json', stage + '/terminal.json'
        if not isinstance(reports, dict) or report_name not in reports or not isinstance(terminals, dict) or terminal_name not in terminals:
            raise ValueError('Hash-bound child report and terminal required')
        worker = _json(self.file(root, report_name, label + '/' + report_name, reports[report_name]))
        terminal = _json(self.file(root, terminal_name, label + '/' + terminal_name, terminals[terminal_name]))
        if (terminal.get('status') != 'complete' or type(terminal.get('exit_code')) is not int
                or terminal['exit_code'] != 0 or terminal.get('error') is not None
                or terminal.get('cleanup_error') is not None):
            raise ValueError('Successful worker completion is required')
        if (worker.get('status') != 'passed' or worker.get('stage') != stage
                or any(worker.get(k) != parent.get(k) for k in ('mode', 'profile', 'source_sha256', 'input_manifest_sha256'))
                or worker.get('finite_outputs') is not True or worker.get('inputs_unchanged') is not True):
            raise ValueError('Completed worker identity and input/output integrity required')
        expected_counts = (2, 0) if mode == 'pair' else (100, 50) if stage == 'core' else (0, 0)
        if any(type(worker.get(key)) is not int or worker[key] != count
               for key, count in zip(('predictions', 'solver_updates'), expected_counts)):
            raise ValueError('Exact prediction and solver counts required')
        mapping = parent.get('source_sha256')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('Source snapshots must be retained')
        for name, digest in mapping.items():
            self.file(root, 'source/' + name, label + '/source/' + name, digest)
        if not isinstance(worker.get('output_sha256'), dict) or not worker['output_sha256']:
            raise ValueError('Output hashes must be retained')
        return parent, worker

    def output(self, root, worker, name, label):
        hashes = worker['output_sha256']
        if name not in hashes:
            raise ValueError('Required output is absent from worker hash map: ' + name)
        return self.file(Path(root) / worker['stage'] / 'result', name,
                         label + '/' + worker['stage'] + '/result/' + name, hashes[name])


def _weight_check(path, mapping, audit):
    catalog_path = audit.file(REPO, CATALOG_NAME, 'original/' + CATALOG_NAME, mapping.get(CATALOG_NAME))
    if CATALOG_NAME not in mapping:
        raise ValueError('Original weight catalog must be source bound')
    catalog, weights = _json(catalog_path), _json(path)
    expected = catalog.get('tensors')
    if not isinstance(expected, dict) or len(expected) != 825:
        raise ValueError('Original 825-tensor catalog is required')
    counts = {'tensor_count': 825, 'parameter_count': 4999787712, 'parameter_bytes': 19999150848}
    if (any(type(weights.get(k)) is not int or weights[k] != n for k, n in counts.items())
            or weights.get('convert_model_dtype') is not False or weights.get('cuda_copy_exact') is not True
            or weights.get('all_shards_verified') is not True or not isinstance(weights.get('tensors'), dict)
            or set(weights['tensors']) != set(expected)):
        raise ValueError('Exact original FP32 core load evidence is required')
    for name, original in expected.items():
        row = weights['tensors'][name]
        if (not isinstance(row, dict) or row.get('shape') != original['shape'] or row.get('shard') != original['shard']
                or row.get('original_dtype') != 'float32' or row.get('loaded_dtype') != 'float32'
                or row.get('source_sha256') != original['original_sha256']
                or row.get('loaded_sha256') != original['original_sha256']
                or row.get('source_owner_released') is not True or row.get('cuda_copy_exact') is not True):
            raise ValueError('Original parameter identity differs: ' + name)
    return {'tensor_count': 825, 'parameter_count': 4999787712,
            'all_original_hashes_match': True, 'original_dtype': 'float32',
            'recorded_cuda_copy_checks_pass': True,
            'method': 'Compare all retained source/loaded hashes with the original catalog; no checkpoint values loaded'}


def _prepare(run_root, codec_root, packet_root, profile, mode):
    if not isinstance(profile, str) or profile not in SHAPES:
        raise ValueError('An explicit baseline or spatial profile is required')
    shape = SHAPES[profile]; audit = _Audit()
    parent, worker = audit.completion(run_root, mode, profile, 'core', 'run')
    codec_parent, codec_worker = audit.completion(codec_root, 'codec', 'both', 'codec', 'codec')
    manifest_path = audit.file(packet_root, 'manifest.json', 'packet/manifest.json', parent.get('input_manifest_sha256'))
    manifest = _json(manifest_path)
    if manifest.get('schema') != 'wan22-two-size-inputs-v1' or manifest.get('status') != 'prepared':
        raise ValueError('The declared prepared input packet is required')
    if (codec_parent.get('input_manifest_sha256') != _sha(manifest_path)
            or parent.get('source_sha256') != codec_parent.get('source_sha256')
            or parent.get('codec_result_report_sha256') != audit.files['codec/metrics.json']
            or worker.get('hardware') != codec_worker.get('hardware')):
        raise ValueError('Run, codec and prepared-packet identities differ')
    mapping = parent['source_sha256']
    audit.file(REPO, SOLVER_NAME, 'original/' + SOLVER_NAME, SOLVER_SHA256)
    if mapping.get(SOLVER_NAME) != SOLVER_SHA256:
        raise ValueError('The original native solver must be source bound')
    weight_report = _weight_check(audit.output(run_root, worker, 'weight-load.json', 'run'), mapping, audit)
    prefix = (shape[2] // 2) * (shape[3] // 2)
    saved_shapes = {'initial_noise': (shape, 'F32'), 'initial_latent': (shape, 'F32'),
                    'observation': ((1, 48, 1, shape[2], shape[3]), 'F32'), 'token_times': ((1, 5 * prefix), 'I64')}
    values = _read(audit.output(run_root, worker, 'sampling-inputs.safetensors', 'run'), saved_shapes)
    actual_hashes = {name: _tensor_sha(value) for name, value in values.items()}
    if worker.get('sampling_input_tensor_sha256') != actual_hashes:
        raise ValueError('Saved sampling tensor hashes differ from the worker record')
    prepared_shapes = {'reference_observation': ((1, 48, 1, 18, 32), 'F32')}
    for name, size in SHAPES.items():
        prepared_shapes[name + '_noise'] = (size, 'F32')
        prepared_shapes[name + '_rgb'] = ((1, 3, 1, size[2] * 16, size[3] * 16), 'F32')
    row = manifest.get('files', {}).get('prepared.safetensors')
    if not isinstance(row, dict):
        raise ValueError('Prepared tensor file must be manifest bound')
    packet_file = audit.file(packet_root, 'prepared.safetensors', 'packet/prepared.safetensors', row.get('sha256'))
    if type(row.get('bytes')) is not int or row['bytes'] != packet_file.stat().st_size:
        raise ValueError('Prepared file size differs')
    noise_key = profile + '_noise'
    noise = _read(packet_file, prepared_shapes, (noise_key,))[noise_key]
    if manifest.get('tensor_sha256', {}).get(noise_key) != _tensor_sha(noise):
        raise ValueError('Prepared noise tensor hash differs')
    _equal(values['initial_noise'], noise, 'initial noise versus prepared profile noise')
    observation_shapes = {name: ((1, 48, 1, s[2], s[3]), 'F32') for name, s in SHAPES.items()}
    combined = _read(audit.output(codec_root, codec_worker, 'observations.safetensors', 'codec'), observation_shapes, (profile,))[profile]
    separate = _read(audit.output(codec_root, codec_worker, profile + '/observation.safetensors', 'codec'),
                     {'observation': saved_shapes['observation']})['observation']
    _equal(combined, separate, 'aggregate versus separate codec observation')
    _equal(values['observation'], separate, 'saved sampling versus codec observation')
    if codec_worker.get('profiles', {}).get(profile, {}).get('observation_tensor_sha256') != _tensor_sha(separate):
        raise ValueError('Codec profile observation tensor hash differs')
    expected_initial = noise.clone(); expected_initial[:, :1] = separate[0]
    _equal(values['initial_latent'], expected_initial, 'initial latent with restored observation prefix')
    expected_times = torch.full((1, 5 * prefix), 999, dtype=torch.int64, device='cpu'); expected_times[:, :prefix] = 0
    _equal(values['token_times'], expected_times, 'observed time zero and future time 999')
    return audit, worker, values, shape, weight_report


def audit_pair(run_root, codec_root, packet_root, profile):
    """Check the saved first pair and explicit FP32 guidance, returning a report."""
    audit, worker, values, shape, weights = _prepare(run_root, codec_root, packet_root, profile, 'pair')
    positive = _read(audit.output(run_root, worker, 'completed-positive.safetensors', 'run'),
                     {'positive_velocity': (shape, 'F32')})
    negative = _read(audit.output(run_root, worker, 'completed-negative.safetensors', 'run'),
                     {name: (shape, 'F32') for name in ('positive_velocity', 'negative_velocity')})
    output = _read(audit.output(run_root, worker, 'outputs.safetensors', 'run'),
                   {name: (shape, 'F32') for name in ('positive_velocity', 'negative_velocity', 'guided_velocity')})
    _equal(positive['positive_velocity'], negative['positive_velocity'], 'positive result retained across partials')
    for name, value in negative.items():
        _equal(value, output[name], name + ' partial versus final')
    # NumPy's separate explicit FP32 operations are an independent arithmetic
    # oracle for the original eager CPU Torch expression, without fused math.
    delta = np.subtract(output['positive_velocity'].numpy(), output['negative_velocity'].numpy(), dtype=np.float32)
    expected = np.add(output['negative_velocity'].numpy(), np.multiply(np.float32(5), delta, dtype=np.float32), dtype=np.float32)
    if not np.isfinite(expected).all():
        raise ValueError('FP32 guidance arithmetic overflowed')
    expected = torch.from_numpy(expected)
    difference = output['guided_velocity'].double() - expected.double()
    residual = {'max_absolute': difference.abs().max().item(), 'rmse': difference.square().mean().sqrt().item(),
                'different_bytes': int((output['guided_velocity'].view(torch.uint8) != expected.view(torch.uint8)).sum().item())}
    _equal(output['guided_velocity'], expected, 'negative + float32(5) * (positive - negative)')
    return {'schema': 'wan22-spatial-prediction-audit-v1', 'status': 'passed', 'mode': 'pair', 'profile': profile,
            'latent_shape': list(shape), 'file_sha256': audit.files,
            'input_tensor_sha256': {name: _tensor_sha(value) for name, value in values.items()},
            'output_tensor_sha256': {name: _tensor_sha(value) for name, value in output.items()},
            'weights': weights, 'guidance_residual': residual,
            'checks': {'prepared_noise_exact': True, 'codec_observation_exact': True, 'initial_prefix_exact': True,
                       'initial_token_times_exact': True, 'partial_final_bytes_exact': True, 'fp32_guidance_bytes_exact': True},
            'limitations': ['No denoiser was rerun; the recorded velocities are not independently reproduced from weights.',
                            'No image-quality, action-control or world-memory score is computed.']}


def audit_clip(run_root, codec_root, packet_root, profile):
    """Check all retained solver states without claiming unavailable replay."""
    audit, worker, values, shape, weights = _prepare(run_root, codec_root, packet_root, profile, 'clip')
    from ..cuda_reference.vendor.fm_solvers_unipc import FlowUniPCMultistepScheduler
    solver = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    solver.set_timesteps(50, device='cpu', shift=5.0)
    times = [int(t) for t in solver.timesteps]
    path = audit.output(run_root, worker, 'steps.jsonl', 'run')
    if path.stat().st_size > 2**20:
        raise ValueError('Bounded 50-row step record required')
    rows = path.read_text().splitlines()
    if len(rows) != 50:
        raise ValueError('Exactly 50 saved solver records are required')
    records = []; last_seconds = 0.; latent = None
    for number, (raw, timestep) in enumerate(zip(rows, times), 1):
        row = _object(raw)
        if (type(row.get('step')) is not int or row['step'] != number
                or type(row.get('timestep')) is not int or row['timestep'] != timestep
                or row.get('prefix_exact') is not True):
            raise ValueError('Original UniPC schedule and step order must match exactly')
        seconds = _positive(row.get('seconds'))
        if seconds < last_seconds:
            raise ValueError('Cumulative solver timing must not decrease')
        last_seconds = seconds
        latent = _read(audit.output(run_root, worker, f'step-{number:02d}.safetensors', 'run'), {'latent': (shape, 'F32')})['latent']
        digest = _tensor_sha(latent)
        if row.get('latent_sha256') != digest:
            raise ValueError('Saved solver tensor differs from its recorded hash')
        _equal(latent[:, :1], values['observation'][0], 'observed prefix at step ' + str(number))
        records.append({'step': number, 'timestep': timestep, 'tensor_sha256': digest,
                        'prefix_bytes_exact': True, 'finite_fp32': True})
    final = _read(audit.output(run_root, worker, 'latents.safetensors', 'run'), {'latent': (shape, 'F32')})['latent']
    _equal(final, latent, 'final latent versus saved step 50')
    return {'schema': 'wan22-spatial-prediction-audit-v1', 'status': 'passed', 'mode': 'clip', 'profile': profile,
            'latent_shape': list(shape), 'file_sha256': audit.files,
            'input_tensor_sha256': {name: _tensor_sha(value) for name, value in values.items()},
            'final_tensor_sha256': _tensor_sha(final), 'weights': weights, 'steps': records,
            'checks': {'prepared_noise_exact': True, 'codec_observation_exact': True, 'initial_prefix_exact': True,
                       'initial_token_times_exact': True, 'schedule_and_order_exact': True,
                       'all_50_saved_prefixes_exact': True, 'final_equals_step_50': True},
            'solver_replayed': False,
            'limitations': ['Per-step positive, negative and guided velocities were not retained, so solver updates cannot be replayed.',
                            'The 100 denoiser calls were not rerun; this checks their reported count and retained solver states.',
                            'Decoded image pixels and visual quality are outside this prediction audit.']}
