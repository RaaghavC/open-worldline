# SPDX-License-Identifier: Apache-2.0
"""Read-only source, input, completion and timing checks. No model calls."""
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil

from ..official_cpu.streaming import sha
from ..cuda_reference.guards import validate_hardware
from ..cuda_reference.evidence import PRECISION
from .config import SETTINGS, SPECS, spec
from .guards import limits
from .inputs import load_prepared


REPO = Path(__file__).resolve().parents[3]
PREFIX = 'experiments/wan22_native/'
NAMES = tuple(sorted(
    [PREFIX + '__init__.py']
    + [PREFIX + 'spatial_reference/' + name for name in (
        '__init__.py', 'config.py', 'inputs.py', 'native.py', 'sampling.py',
        'codec.py', 'guards.py', 'output.py', 'evidence.py', 'run.py')]
    + [PREFIX + 'cuda_reference/' + name for name in (
        '__init__.py', 'native.py', 'sampling.py', 'guards.py', 'decode.py',
        'evidence.py', 'vendor/__init__.py', 'vendor/model.py', 'vendor/attention.py',
        'vendor/fm_solvers_unipc.py', 'vendor/vae2_2.py', 'config.json',
        'codec-source.json', 'upstream-provenance.json', 'expected-weights.json')]
    + [PREFIX + 'official_cpu/' + name for name in ('__init__.py', 'inputs.py', 'streaming.py')]
))
IGNORED_OUTPUTS = frozenset({'metrics.json', 'memory.jsonl', 'watchdog-stop.json'})


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('A lowercase SHA256 digest is required')
    return value


def _relative(name):
    if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
            or PurePosixPath(name).is_absolute() or str(PurePosixPath(name)) != name
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('A canonical relative file path is required')
    return name


def _root(directory):
    path = Path(directory).absolute()
    # Check the caller's spelling before resolve(), including symlink ancestors.
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_dir():
        raise ValueError('A regular directory without symlink ancestors is required')
    return path.resolve()


def _file(directory, name):
    root = _root(directory)
    path = root
    for part in _relative(name).split('/'):
        path = path / part
        if path.is_symlink():
            raise ValueError('Symlink artifacts are not accepted')
    if not path.is_file():
        raise ValueError('A retained regular file is required: ' + name)
    return path


def _files(directory):
    root = _root(directory)
    result = set()
    for folder, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(folder) / name
            if path.is_symlink():
                raise ValueError('Symlink artifacts are not accepted')
            if name == 'watchdog-stop.json':
                raise ValueError('Stopped runs do not qualify')
        for name in files:
            relative = (Path(folder) / name).relative_to(root).as_posix()
            _file(root, relative)
            result.add(relative)
    return result


def _json(path):
    if path.stat().st_size > 4 * 2**20:
        raise ValueError('Bounded JSON evidence required')
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate JSON keys are not accepted')
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('Nonfinite JSON values are not accepted: ' + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=invalid, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError('A JSON object is required')
    return value


def _mapping(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('A nonempty source or artifact hash map is required')
    for name, digest in value.items():
        _relative(name)
        _digest(digest)
    return value


def sources():
    """Stable repository-relative runtime sources and required pin/config files.

    Tests, results and external weights are excluded. The explicit list includes
    the unchanged official model, VAE and solver source imported by this runner.
    """
    return {name: sha(_file(REPO, name)) for name in NAMES}


def snapshot(out):
    """Copy exact source bytes to fresh out/source/<repository-relative name>."""
    root = _root(out)
    mapping = sources()
    target_root = root / 'source'
    target_root.mkdir()  # Never overwrite an earlier measured source bundle.
    for name, digest in mapping.items():
        target = target_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with _file(REPO, name).open('rb') as source, target.open('xb') as destination:
            shutil.copyfileobj(source, destination, length=2**20)
        if sha(target) != digest:
            raise RuntimeError('Source changed during snapshot: ' + name)
    if sources() != mapping:
        raise RuntimeError('Source changed during snapshot')
    return mapping


def validate_input_packet(directory, expected_sha):
    """Check the exact manifest and symlinks before any prepared tensor read."""
    expected_sha = _digest(expected_sha)
    root = _root(directory)
    manifest = _file(root, 'manifest.json')
    if sha(manifest) != expected_sha:
        raise ValueError('Prepared input manifest hash differs')
    _json(manifest)
    _files(root)
    result = load_prepared(root)
    if sha(manifest) != expected_sha:
        raise ValueError('Prepared input manifest changed during validation')
    return result


def _identity(report, mode, profile, mapping, packet_sha):
    if (report.get('status') != 'passed' or report.get('mode') != mode
            or report.get('profile') != profile or report.get('source_sha256') != mapping
            or report.get('input_manifest_sha256') != packet_sha
            or report.get('limits') != limits(mode)):
        raise ValueError('Completed report mode, profile, sources, inputs or limits differ')


def _core_weights(path):
    weights = _json(path)
    expected = _json(_file(REPO, PREFIX + 'cuda_reference/expected-weights.json'))['tensors']
    counts = {'tensor_count': 825, 'parameter_count': 4999787712, 'parameter_bytes': 19999150848}
    if (any(type(weights.get(key)) is not int or weights[key] != count for key, count in counts.items())
            or weights.get('convert_model_dtype') is not False or weights.get('all_shards_verified') is not True
            or weights.get('cuda_copy_exact') is not True or not isinstance(weights.get('tensors'), dict)
            or set(weights['tensors']) != set(expected)):
        raise ValueError('Complete 825-tensor original FP32 core weight records required')
    for name, original in expected.items():
        row = weights['tensors'][name]
        if (not isinstance(row, dict) or row.get('original_dtype') != 'float32'
                or row.get('loaded_dtype') != 'float32' or row.get('shape') != original['shape']
                or row.get('shard') != original['shard']
                or row.get('source_sha256') != original['original_sha256']
                or row.get('loaded_sha256') != original['original_sha256']
                or row.get('cuda_copy_exact') is not True or row.get('source_owner_released') is not True):
            raise ValueError('Original core parameter identity or copy evidence differs: ' + name)


def _codec_weights(path):
    weights = _json(path)
    pin = _json(_file(REPO, PREFIX + 'cuda_reference/codec-source.json'))
    config = dict(dim=160, dec_dim=256, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                  attn_scales=[], temperal_downsample=[False, True, True], dropout=0.)
    rows = weights.get('tensors')
    if (weights.get('weight_sha256') != pin['weight_sha256'] or weights.get('compute_dtype') != 'float32'
            or type(weights.get('parameters')) is not int or weights['parameters'] != 704688668
            or weights.get('config') != config or not isinstance(rows, dict) or len(rows) != 196):
        raise ValueError('Complete native 196-tensor VAE load records required')
    count = 0
    for name, row in rows.items():
        if not isinstance(name, str) or not name or not isinstance(row, dict) or row.get('cuda_copy_exact') is not True:
            raise ValueError('Every VAE copy record is required')
        shape = row.get('shape')
        if not isinstance(shape, list) or not shape or any(type(x) is not int or x < 1 for x in shape):
            raise ValueError('Every VAE tensor shape must be positive integers')
        _digest(row.get('sha256'))
        count += math.prod(shape)
    if count != 704688668:
        raise ValueError('VAE tensor shapes do not sum to the exact parameter count')


def _rgb_index(result, relative, selected, purpose, outputs):
    index = _json(_file(result, relative))
    if (index.get('schema') != 'wan22-rgb-frames-v1' or index.get('purpose') != purpose
            or index.get('shape') != [1, 3, 17, selected.height, selected.width]
            or index.get('dtype') != 'float32' or index.get('range') != [-1, 1]
            or not isinstance(index.get('frames'), list) or len(index['frames']) != 17):
        raise ValueError('Exact 17-frame raw RGB index required')
    folder = PurePosixPath(relative).parent
    for number, row in enumerate(index['frames']):
        if (not isinstance(row, dict) or type(row.get('index')) is not int or row['index'] != number
                or row.get('file') != f'{number:04d}.safetensors'):
            raise ValueError('All 17 raw frames must appear once in temporal order')
        name = (folder / row['file']).as_posix()
        path = _file(result, name)
        if (type(row.get('bytes')) is not int or row['bytes'] != path.stat().st_size
                or row.get('sha256') != outputs.get(name) or row.get('sha256') != sha(path)):
            raise ValueError('RGB frame bytes and hashes differ from the worker outputs')
        _digest(row.get('tensor_sha256'))


def _scientific_outputs(result, child, mode, profile, outputs):
    stage = child['stage']
    required = {'weight-load.json', 'monitor-terminal.json'}
    if stage == 'core':
        if child.get('precision') != PRECISION:
            raise ValueError('Unchanged native FP32 weights and CUDA BF16 execution required')
        _core_weights(_file(result, 'weight-load.json'))
        required.add('sampling-inputs.safetensors')
        if mode == 'pair':
            required.update({'outputs.safetensors', 'completed-positive.safetensors', 'completed-negative.safetensors'})
        else:
            required.update({'latents.safetensors', 'steps.jsonl'})
            required.update(f'step-{index:02d}.safetensors' for index in range(1, 51))
            path = _file(result, 'steps.jsonl')
            if path.stat().st_size > 2**20:
                raise ValueError('Bounded 50-row step record required')
            rows = path.read_text().splitlines()
            if len(rows) != 50:
                raise ValueError('All 50 solver step records required')
            from .sampling import scheduler
            expected_times = [int(t) for t in scheduler().timesteps]
            for number, (raw, timestep) in enumerate(zip(rows, expected_times), 1):
                row = json.loads(raw)
                if (not isinstance(row, dict) or type(row.get('step')) is not int or row['step'] != number
                        or type(row.get('timestep')) is not int or row['timestep'] != timestep
                        or row.get('prefix_exact') is not True):
                    raise ValueError('Exact solver order, times and prefix checks required')
                _digest(row.get('latent_sha256'))
                _positive(row.get('seconds'))
    else:
        if child.get('decoder_cache_clear') is not True:
            raise ValueError('Native VAE caches must be cleared')
        _codec_weights(_file(result, 'weight-load.json'))
        _positive(child.get('load_seconds'))
        if stage == 'codec':
            profiles = child.get('profiles')
            if not isinstance(profiles, dict) or set(profiles) != set(SPECS):
                raise ValueError('Both measured codec profiles are required')
            required.add('observations.safetensors')
            for name, selected in SPECS.items():
                row = profiles[name]
                if not isinstance(row, dict):
                    raise ValueError('A complete codec profile record is required')
                _positive(row.get('encode_seconds')); _positive(row.get('decode_seconds'))
                _digest(row.get('observation_tensor_sha256'))
                index = name + '/proxy-rgb/index.json'
                if row.get('raw_rgb_index') != index:
                    raise ValueError('Codec profile raw frame index differs')
                required.update({name + '/' + item for item in ('observation.safetensors',
                    'timing-proxy.safetensors', 'reconstructed-initial.png')})
                required.add(index)
                _rgb_index(result, index, selected, 'repeated_observation_codec_timing_proxy', outputs)
        else:
            selected = spec(profile)
            _positive(child.get('decode_seconds'))
            required.update({'rgb/index.json', 'preview.gif', 'comparison.png'})
            required.update(f'frames/{number:04d}.png' for number in range(17))
            _rgb_index(result, 'rgb/index.json', selected, 'generated_clip', outputs)
            images = child.get('images', {})
            if (not isinstance(images, dict) or images.get('frames') != 17
                    or images.get('height') != selected.height or images.get('width') != selected.width
                    or images.get('conditioned_initial_frames') != 1 or images.get('new_future_frames') != 16
                    or images.get('contact_images_resized') is not False):
                raise ValueError('Complete 17-frame native-size presentation record required')
    if not required <= set(outputs):
        raise ValueError('Missing required scientific artifacts: ' + ', '.join(sorted(required - set(outputs))))


def validate_completed(root, mode, profile, mapping, packet_sha, expected_gpu):
    """Validate one completed local run, returning its unchanged root report.

    This validates retained evidence, not visual quality. Callers supply the
    current sources() mapping and the exact input manifest hash to admit reuse.
    """
    limits(mode)
    if mode == 'codec':
        if profile != 'both':
            raise ValueError('Codec completion requires both profiles')
        stages = ('codec',)
    else:
        spec(profile)
        stages = ('core',) if mode == 'pair' else ('core', 'decode')
    _mapping(mapping); _digest(packet_sha)
    root = _root(root)
    _files(root)
    if {name for name in ('codec', 'core', 'decode') if (root / name).exists()} != set(stages):
        raise ValueError('Exact mode-specific child directories required')
    report = _json(_file(root, 'metrics.json'))
    _identity(report, mode, profile, mapping, packet_sha)
    if report.get('model_execution') is not True or report.get('settings') != SETTINGS:
        raise ValueError('An executed run with the fixed sampling settings is required')
    for name, digest in mapping.items():
        if sha(_file(root / 'source', name)) != digest:
            raise ValueError('Retained measured source differs: ' + name)
    if _files(root / 'source') != set(mapping):
        raise ValueError('Exact retained source file set required')
    # The packet is retained by the parent and independently checked before
    # admission. This call also prevents a changed copied input being accepted.
    validate_input_packet(root / 'inputs', packet_sha)
    reports = _mapping(report.get('child_reports'))
    terminals = _mapping(report.get('child_terminals'))
    if (set(reports) != {s + '/result/metrics.json' for s in stages}
            or set(terminals) != {s + '/terminal.json' for s in stages}):
        raise ValueError('Exact mode-specific child reports and terminals required')
    hardware = None
    for stage in stages:
        report_name = stage + '/result/metrics.json'
        terminal_name = stage + '/terminal.json'
        report_path = _file(root, report_name)
        terminal_path = _file(root, terminal_name)
        if sha(report_path) != reports[report_name] or sha(terminal_path) != terminals[terminal_name]:
            raise ValueError('Child report or terminal hash differs')
        child, terminal = _json(report_path), _json(terminal_path)
        _identity(child, mode, profile, mapping, packet_sha)
        if (terminal.get('status') != 'complete' or type(terminal.get('exit_code')) is not int
                or terminal['exit_code'] != 0 or terminal.get('error') is not None
                or terminal.get('cleanup_error') is not None or terminal.get('mode') != mode
                or terminal.get('limits') != limits(mode)):
            raise ValueError('A successful guarded worker exit is required')
        if (child.get('stage') != stage or child.get('finite_outputs') is not True
                or child.get('inputs_unchanged') is not True):
            raise ValueError('Completed finite outputs and unchanged inputs required')
        counts = (2, 0) if stage == 'core' and mode == 'pair' else (100, 50) if stage == 'core' else (0, 0)
        if any(type(child.get(key)) is not int or child[key] != count
               for key, count in zip(('predictions', 'solver_updates'), counts)):
            raise ValueError('Prescribed prediction and solver counts required')
        try:
            actual = validate_hardware(child['hardware'], expected_gpu)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('Complete measured hardware identity required') from error
        if hardware is not None and actual != hardware:
            raise ValueError('All child hardware and runtime identities must match')
        hardware = actual
        result = root / stage / 'result'
        outputs = _mapping(child.get('output_sha256'))
        retained = {name for name in _files(result) if PurePosixPath(name).name not in IGNORED_OUTPUTS}
        if set(outputs) != retained:
            raise ValueError('Every immutable worker output must be hash bound')
        for name, digest in outputs.items():
            if sha(_file(result, name)) != digest:
                raise ValueError('Retained worker output changed: ' + name)
        monitor = _json(_file(result, 'monitor-terminal.json'))
        if (monitor.get('status') != 'complete' or monitor.get('mode') != mode
                or monitor.get('limits') != limits(mode) or monitor.get('error') is not None):
            raise ValueError('A complete resource-monitor terminal is required')
        _scientific_outputs(result, child, mode, profile, outputs)
    return report


def _positive(value):
    if type(value) not in (int, float):
        raise ValueError('Every measured duration must be finite and positive')
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError('Measured duration is outside the finite range') from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError('Every measured duration must be finite and positive')
    return number


def admit_clip(pair_worker_report, codec_worker_report, profile):
    """Estimate after validate_completed has checked both report directories.

    This arithmetic gate does not authorize cloud spending or claim the clip
    will complete. The actual clip remains subject to its 1,800-second guard.
    """
    spec(profile)
    pair, codec = pair_worker_report, codec_worker_report
    if (pair.get('status') != 'passed' or pair.get('mode') != 'pair' or pair.get('stage') != 'core'
            or pair.get('profile') != profile or pair.get('predictions') != 2 or pair.get('solver_updates') != 0
            or codec.get('status') != 'passed' or codec.get('mode') != 'codec'
            or codec.get('stage') != 'codec' or codec.get('profile') != 'both'
            or set(codec.get('profiles', {})) != set(SPECS)):
        raise ValueError('Measured completed pair and both-profile codec reports required')
    if any(pair.get(key) is None or pair.get(key) != codec.get(key)
           for key in ('hardware', 'source_sha256', 'input_manifest_sha256')):
        raise ValueError('Measured pair and codec identities must match')
    try:
        timings = {'core_load_seconds': _positive(pair['load_seconds']),
                   'pair_seconds': _positive(pair['pair_seconds']),
                   'codec_load_seconds': _positive(codec['load_seconds']),
                   'decode_seconds': _positive(codec['profiles'][profile]['decode_seconds'])}
    except (KeyError, TypeError) as error:
        raise ValueError('Required measured timing is missing') from error
    estimate = (timings['core_load_seconds'] + 50 * timings['pair_seconds'] * 2.0
                + 1.2 * (timings['codec_load_seconds'] + timings['decode_seconds']) + 30)
    if not math.isfinite(estimate) or estimate >= 1800:
        raise ValueError('Estimated clip time must be strictly below 1,800 seconds')
    return {'estimated_seconds': estimate, 'limit_seconds': 1800, 'measured_seconds': timings,
            'factors': {'pairs': 50, 'pair_safety': 2.0, 'codec_safety': 1.2, 'artifact_seconds': 30},
            'formula': 'core.load_seconds + 50 * core.pair_seconds * 2.0 + 1.2 * (codec.load_seconds + codec.profiles[profile].decode_seconds) + 30',
            'profile': profile, 'estimate_is_not_a_completion_guarantee': True}
