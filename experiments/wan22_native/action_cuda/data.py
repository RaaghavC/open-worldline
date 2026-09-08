# SPDX-License-Identifier: Apache-2.0
"""Original RGB/action plans and separately encoded native CUDA training caches.

The CLI prepares CPU plans only. A guarded owner must explicitly load the
native VAE and call encode_cache; this module never loads weights or provisions
compute. Targets and independently encoded observations are never patched.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from ..action_data import data as original
from ..spatial_reference.config import spec
from ..spatial_reference.inputs import preprocess
from ..spatial_reference.evidence import sources as spatial_sources
from ..cuda_reference.native import verify_sources

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SELECTION = original.SELECTION
CHANNELS = original.CHANNELS
SCHEMA = 'worldline-wan22-native-cuda-action-cache-v1'
PLAN_SCHEMA = 'worldline-wan22-native-cuda-action-plan-v1'
PREFIX_MAX_ABS = 1e-5
PREFIX_RELATIVE_L2 = 1e-5
DIAGNOSTIC_NORM_REL_TOL = 1e-12
DIAGNOSTIC_NORM_ABS_TOL = 1e-15
VAE_SHA256 = '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'
VAE_CONFIG = dict(dim=160, dec_dim=256, z_dim=48, dim_mult=[1, 2, 4, 4],
                  num_res_blocks=2, attn_scales=[],
                  temperal_downsample=[False, True, True], dropout=0.)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sources():
    result = dict(spatial_sources())
    for path in [Path(__file__), Path(original.__file__),
                 Path(original.__file__).with_name('source-plan.md.txt'),
                 Path(original.original.__file__)]:
        result[str(path.resolve().relative_to(REPO))] = sha(path)
    return dict(sorted(result.items()))


source_hashes = sources


def shapes(profile):
    item = spec(profile)
    return {'target': (1, *item.latent_shape), 'observation': item.observation_shape,
            'commands': (1, 16, 6)}


def _fresh(output):
    output = Path(output)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(REPO):
        raise ValueError('Fresh output outside the repository is required')
    output.mkdir(parents=True)
    return output


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')
    temporary.replace(path)


def processed_rgb(window, profile):
    """Canonicalization already occurred in the verified raw RGB reader."""
    item = spec(profile)
    if window.rgb.shape != (17, 288, 512, 3) or window.rgb.dtype != np.uint8:
        raise ValueError('Exactly 17 original RGB frames are required')
    frames = []; transforms = []
    for frame in window.rgb:
        image, _, transform = preprocess(Image.fromarray(frame, mode='RGB'), profile)
        frames.append(np.array(image, dtype=np.uint8, copy=True)); transforms.append(transform)
    pixels = np.stack(frames)
    if pixels.shape != (17, item.height, item.width, 3) or any(t != transforms[0] for t in transforms):
        raise ValueError('Inconsistent native spatial preprocessing')
    return pixels, transforms[0]


def video_tensor(pixels):
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError('Byte RGB video required')
    # Literal arithmetic from the native image preprocessing, not /127.5 - 1.
    return torch.from_numpy(pixels.copy()).permute(3, 0, 1, 2).contiguous().float().div_(255).sub_(.5).div_(.5)[None]


def _inventory(windows):
    rows = {}
    for window in windows.values():
        for record in window.provenance['raw']['sources']:
            name = record['file']
            checksum = record['sha256']
            if name in rows and rows[name] != checksum:
                raise ValueError('One original file has inconsistent source hashes')
            rows[name] = checksum
    return dict(sorted(rows.items()))


def _plan(windows, profile):
    selected = spec(profile); entries = []
    for arm, start in SELECTION:
        window = windows[(arm, start)]; pixels, transform = processed_rgb(window, profile)
        entries.append({'id': f'{arm}-{start:04d}', 'arm': arm, 'start': start,
                        'source': window.provenance, 'preprocessing': transform,
                        'processed_rgb_shape': list(pixels.shape),
                        'processed_rgb_sha256': original.array_sha(pixels),
                        'processed_first_rgb_sha256': original.array_sha(pixels[0]),
                        'commands_sha256': original.array_sha(window.commands)})
    return {'schema': PLAN_SCHEMA, 'status': 'planned', 'profile': profile,
            'profile_spec': selected.record(), 'selection': entries,
            'capture_manifest_sha256': original.MANIFEST_SHA256,
            'original_file_inventory': _inventory(windows), 'source_sha256': sources(),
            'command_channels': list(CHANNELS),
            'command_alignment': 'commands[t] is raw records[start+t+1].action_from_previous; RGB[t] -> RGB[t+1]',
            'shapes': {name: list(value) for name, value in shapes(profile).items()},
            'canonicalization': 'Only derived raw closed-start0 RGB frame0 is replaced with pinned open/0000 before resizing and both encodes',
            'canonical_changes_scope': '18 one-level channel values in raw 512x288 RGB; Lanczos may spread changes over more processed pixels',
            'shared_start0_observation': True, 'target_prefix_replaced': False,
            'prefix_limits': {'max_abs': PREFIX_MAX_ABS, 'relative_l2': PREFIX_RELATIVE_L2,
                              'same_length_and_repeated_image': 'bit-exact equality required',
                              'measured_native_cuda_full_prefix_result_available': False},
            'serialized_diagnostic_norm_comparison': {
                'relative_tolerance': DIAGNOSTIC_NORM_REL_TOL,
                'absolute_tolerance': DIAGNOSTIC_NORM_ABS_TOL,
                'scope': 'Only descriptive float64 norms; recomputed model limits and tensor identities remain strict'},
            'data_license': 'CC0-1.0', 'split': 'development', 'independent_layouts': 1,
            'old_latent_cache_reused': False, 'model_execution': False,
            'limitations': 'One enlarged original layout, fixed-position yaw and remote door toggles. Later window branches have different observations and identical commands; only canonical start0 is an isolated outgoing-command contrast.'}


def plan(capture, profile):
    return _plan(original.load_selection(capture), profile)


def write_plan(capture, profile, output):
    result = plan(capture, profile)
    output = _fresh(output); _json(output / 'plan.json', result)
    return result


def _cpu_tensor(value, shape, label):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.device.type != 'cpu' or value.dtype != torch.float32 or value.requires_grad
            or not torch.isfinite(value).all().item()):
        raise ValueError('Detached finite CPU FP32 ' + label + ' required')


def _to_cuda(value):
    return value.to('cuda:0')


def native_encode(model, scale, video, profile):
    """Use only the unchanged native FP32 CUDA VAE encode and normalization."""
    item = spec(profile)
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[2] not in (1, 17):
        raise ValueError('An independent one-frame or original 17-frame RGB video is required')
    frames = video.shape[2]
    _cpu_tensor(video, (1, 3, frames, item.height, item.width), 'RGB')
    if video.min().item() < -1 or video.max().item() > 1:
        raise ValueError('Native normalized RGB range must be [-1,1]')
    verify_sources()
    try:
        model.clear_cache()
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            value = model.encode(_to_cuda(video), scale)
        torch.cuda.synchronize()
        result = value.detach().cpu().contiguous()
        expected = (1, 48, 1 if frames == 1 else 5, item.height // 16, item.width // 16)
        _cpu_tensor(result, expected, 'native VAE latent')
        return result
    finally:
        model.clear_cache()


def prefix_check(candidate, reference, name, relation, *, require_bit_exact=False):
    if not isinstance(reference, torch.Tensor) or reference.ndim != 5 or reference.shape[:3] != (1, 48, 1):
        raise ValueError('One reference latent frame required')
    _cpu_tensor(reference, reference.shape, 'reference prefix')
    _cpu_tensor(candidate, reference.shape, 'candidate prefix')
    delta = candidate.double() - reference.double()
    maximum = float(delta.abs().max()); numerator = float(torch.linalg.vector_norm(delta))
    denominator = float(torch.linalg.vector_norm(reference.double()))
    relative = numerator / denominator if denominator else (0. if numerator == 0 else None)
    candidate_sha = tensor_sha(candidate); reference_sha = tensor_sha(reference)
    bit_exact = candidate_sha == reference_sha
    return {'name': name, 'relation': relation, 'passed': maximum <= PREFIX_MAX_ABS and relative is not None and relative <= PREFIX_RELATIVE_L2 and (bit_exact or not require_bit_exact),
            'exact_equal': torch.equal(candidate, reference), 'max_abs': maximum,
            'bit_exact_equal': bit_exact, 'requires_bit_exact': require_bit_exact,
            'relative_l2': relative, 'relative_l2_reference_norm': denominator,
            'candidate_sha256': candidate_sha, 'reference_sha256': reference_sha,
            'limits': {'max_abs': PREFIX_MAX_ABS, 'relative_l2': PREFIX_RELATIVE_L2},
            'target_prefix_replaced': False}


def _validate_codec(value):
    if (not isinstance(value, dict) or value.get('weight_sha256') != VAE_SHA256
            or value.get('compute_dtype') != 'float32' or value.get('parameters') != 704688668
            or value.get('config') != VAE_CONFIG or len(value.get('tensors', {})) != 196):
        raise ValueError('Complete original FP32 native CUDA VAE provenance required')
    parameters = 0
    for name, row in value['tensors'].items():
        if (not isinstance(name, str) or row.get('cuda_copy_exact') is not True
                or not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['sha256'])
                or not isinstance(row.get('shape'), list)
                or any(type(n) is not int or n < 1 for n in row['shape'])):
            raise ValueError('Invalid native VAE loaded tensor record')
        parameters += math.prod(row['shape'])
    if parameters != value['parameters']:
        raise ValueError('Native VAE tensor counts disagree')


def _save(output, name, values):
    path = output / name; path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    values = {key: value.contiguous().clone() for key, value in values.items()}
    save_file(values, str(path))
    return {'file': name, 'bytes': path.stat().st_size, 'sha256': sha(path),
            'tensors': {key: {'shape': list(value.shape), 'dtype': 'float32', 'sha256': tensor_sha(value)}
                        for key, value in values.items()}}


def check_names():
    names = []
    for arm, start in SELECTION:
        key = f'{arm}-{start:04d}'; names.append(key + '_target_vs_independent')
        if (arm, start) in [('open', 0), ('open', 8)]:
            names.append(key + '_same_length_future_perturbation')
            if start == 0:
                names.append('shared_initial_after_full_perturbation')
    return names


def encode_cache(capture, profile, output, encoder, codec_provenance, check=None):
    """Called only by an independently guarded worker; no weight loading here."""
    _validate_codec(codec_provenance)
    check = check or (lambda: None)
    windows = original.load_selection(capture); prepared = _plan(windows, profile)
    output = _fresh(output); source_before = prepared['source_sha256']
    _json(output / 'plan.json', prepared)
    for name, checksum in source_before.items():
        destination = output / 'source' / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
        if sha(destination) != checksum:
            raise ValueError('Source changed during snapshot')
    manifest = {'schema': SCHEMA, 'status': 'running', 'profile': profile,
                'plan_sha256': canonical_sha(prepared), 'plan_file_sha256': sha(output / 'plan.json'),
                'source_sha256': source_before, 'capture_manifest_sha256': original.MANIFEST_SHA256,
                'command_channels': list(CHANNELS), 'shapes': prepared['shapes'],
                'codec_provenance': codec_provenance, 'codec_execution': 'native CUDA FP32',
                'observations': [], 'windows': [], 'causal_checks': [],
                'target_prefix_replaced': False, 'old_latent_cache_reused': False,
                'commands_provided_to_codec': False, 'raw_capture_modified': False,
                'runtime': {'torch': torch.__version__, 'numpy': np.__version__}}
    _json(output / 'manifest.json', manifest)
    completion = {'status': 'running', 'source_sha256': source_before,
                  'codec_provenance': codec_provenance,
                  'encoder_call_attempts': {'one_frame': 0, 'seventeen_frames': 0, 'total': 0},
                  'encoder_calls_completed': {'one_frame': 0, 'seventeen_frames': 0, 'total': 0}}
    _json(output / 'completion.json', completion)
    shared = None

    def encode(video):
        check(); before = tensor_sha(video)
        kind = 'one_frame' if video.shape[2] == 1 else 'seventeen_frames'
        completion['encoder_call_attempts'][kind] += 1
        completion['encoder_call_attempts']['total'] += 1
        _json(output / 'completion.json', completion)
        value = encoder(video)
        if tensor_sha(video) != before:
            raise RuntimeError('Encoder mutated original RGB input')
        item = spec(profile)
        _cpu_tensor(value, (1, 48, 1 if video.shape[2] == 1 else 5, item.height // 16, item.width // 16), 'encoded latent')
        check()
        completion['encoder_calls_completed'][kind] += 1
        completion['encoder_calls_completed']['total'] += 1
        _json(output / 'completion.json', completion)
        return value.contiguous().clone()

    def retain(candidate, reference, name, relation):
        row = prefix_check(candidate, reference, name, relation,
                           require_bit_exact=not name.endswith('_target_vs_independent'))
        row['evidence'] = _save(output, 'checks/' + name + '.safetensors',
                                {'candidate': candidate, 'reference': reference})
        manifest['causal_checks'].append(row); _json(output / 'manifest.json', manifest)
        if not row['passed']:
            raise RuntimeError('Predeclared prefix tolerance failed: ' + name)

    try:
        for arm, start in SELECTION:
            key = f'{arm}-{start:04d}'; pixels, _ = processed_rgb(windows[(arm, start)], profile)
            video = video_tensor(pixels)
            observation_id = 'start-0000-shared' if start == 0 else key
            if start == 0 and shared is not None:
                observation = shared.clone()
            else:
                observation = encode(video[:, :, :1].clone())
                if start == 0:
                    shared = observation.clone()
                row = _save(output, f'observations/{observation_id}.safetensors', {'observation': observation})
                row.update(id=observation_id, encoded_rgb_frames=1,
                           processed_first_rgb_sha256=original.array_sha(pixels[0]))
                manifest['observations'].append(row)
            target = encode(video)
            retain(target[:, :, :1].clone(), observation, key + '_target_vs_independent',
                   'Full 17-frame prefix versus separately encoded single RGB image')
            commands = torch.from_numpy(windows[(arm, start)].commands.copy())[None]
            _cpu_tensor(commands, (1, 16, 6), 'commands')
            row = _save(output, key + '.safetensors', {'target': target, 'observation': observation, 'commands': commands})
            row.update(id=key, observation_id=observation_id,
                       source=next(entry for entry in prepared['selection'] if entry['id'] == key),
                       target_prefix_sha256=tensor_sha(target[:, :, :1]),
                       encoded_rgb_tensor_sha256=tensor_sha(video))
            manifest['windows'].append(row)
            if (arm, start) in [('open', 0), ('open', 8)]:
                altered = video.clone(); altered[:, :, 1:] = -altered[:, :, 1:]
                if not torch.equal(altered[:, :, :1], video[:, :, :1]) or torch.equal(altered[:, :, 1:], video[:, :, 1:]):
                    raise ValueError('Future perturbation must preserve first RGB and change later RGB')
                changed = encode(altered)
                retain(changed[:, :, :1].clone(), target[:, :, :1].clone(),
                       key + '_same_length_future_perturbation',
                       'Same 17-frame length, exact first RGB, changed future RGB only')
                del altered, changed
                if start == 0:
                    repeated = encode(video[:, :, :1].clone())
                    retain(repeated, shared, 'shared_initial_after_full_perturbation',
                           'Repeated independent first-image encode after a different full-video encode')
            _json(output / 'manifest.json', manifest)
            del pixels, video, target
        if len(manifest['windows']) != 8 or len(manifest['observations']) != 7 or [c['name'] for c in manifest['causal_checks']] != check_names():
            raise RuntimeError('Incomplete prescribed cache')
        if sources() != source_before or _inventory(original.load_selection(capture)) != prepared['original_file_inventory']:
            raise RuntimeError('Original source or capture changed during encoding')
        check(); manifest['status'] = 'passed'; _json(output / 'manifest.json', manifest)
        completion.update(status='passed', manifest_sha256=sha(output / 'manifest.json'),
                          plan_sha256=canonical_sha(prepared), finite_output=True,
                          unique_observations=7, independent_observation_encodes=8,
                          original_target_encodes=8, future_perturbed_target_encodes=2,
                          windows=8, causal_checks=11,
                          sources_unchanged=True, original_capture_unchanged=True,
                          output_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*'))
                                         if p.is_file() and p.name != 'completion.json'})
        _json(output / 'completion.json', completion)
        return completion
    except BaseException as error:
        completion.update(status='failed', error_type=type(error).__name__, error=str(error))
        _json(output / 'completion.json', completion)
        raise


def _checked_file(directory, row):
    name = row['file']
    if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Cache artifact path or identity differs')
    relative = Path(name); path = directory / relative
    if (relative.is_absolute() or any(p.is_symlink() for p in [path, *path.parents] if p != directory)
            or not path.resolve().is_relative_to(directory) or not path.is_file()
            or ('bytes' in row and path.stat().st_size != row['bytes']) or sha(path) != row['sha256']):
        raise ValueError('Cache artifact path or identity differs')
    return path


def _matches_prefix_row(saved, measured):
    """Allow only CPU reduction roundoff in recorded descriptive L2 norms.

    The saved FP32 tensors are remeasured before this comparison. Their actual
    1e-5 gates must pass without this tolerance, as must bit-exact checks where
    required. Hashes, maximum error, equality flags and limits compare exactly.
    """
    if measured['passed'] is not True:
        return False
    for key, value in measured.items():
        recorded = saved.get(key)
        if key in ('relative_l2', 'relative_l2_reference_norm') and value is not None:
            if (type(recorded) not in (float, int) or not math.isfinite(recorded)
                    or not math.isclose(recorded, value, rel_tol=DIAGNOSTIC_NORM_REL_TOL,
                                        abs_tol=DIAGNOSTIC_NORM_ABS_TOL)):
                return False
        elif type(recorded) is not type(value) or recorded != value:
            return False
    return True


def _check_evidence(directory, row, shape):
    evidence = row['evidence']; path = _checked_file(directory, evidence)
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != {'candidate', 'reference'} or set(evidence['tensors']) != {'candidate', 'reference'}:
            raise ValueError('Exact prefix evidence pair required')
        for name in ['candidate', 'reference']:
            if tuple(handle.get_slice(name).get_shape()) != shape:
                raise ValueError('Prefix evidence has wrong native shape')
        values = {name: handle.get_tensor(name) for name in ['candidate', 'reference']}
    for name, value in values.items():
        _cpu_tensor(value, shape, 'prefix evidence')
        if evidence['tensors'][name] != {'shape': list(shape), 'dtype': 'float32', 'sha256': tensor_sha(value)}:
            raise ValueError('Prefix evidence tensor hash differs')
    measured = prefix_check(values['candidate'], values['reference'], row['name'], row['relation'],
                            require_bit_exact=not row['name'].endswith('_target_vs_independent'))
    if not _matches_prefix_row(row, measured):
        raise ValueError('Recomputed causal-prefix evidence differs')


def read_window(directory, identity, *, conditioning_only=False):
    directory = Path(directory).resolve()
    if (directory / 'watchdog-stop.json').exists():
        raise ValueError('Stopped cache is not admitted')
    completion = json.loads((directory / 'completion.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    prepared = json.loads((directory / 'plan.json').read_text())
    if (completion.get('status') != 'passed' or manifest.get('status') != 'passed'
            or manifest.get('schema') != SCHEMA or completion.get('manifest_sha256') != sha(directory / 'manifest.json')
            or completion.get('source_sha256') != sources() or manifest.get('source_sha256') != sources()
            or completion.get('plan_sha256') != canonical_sha(prepared)
            or manifest.get('plan_sha256') != canonical_sha(prepared)
            or manifest.get('plan_file_sha256') != sha(directory / 'plan.json')):
        raise ValueError('Complete source-matching native CUDA cache evidence required')
    if (completion.get('finite_output') is not True or completion.get('sources_unchanged') is not True
            or completion.get('original_capture_unchanged') is not True
            or completion.get('unique_observations') != 7 or completion.get('independent_observation_encodes') != 8
            or completion.get('original_target_encodes') != 8 or completion.get('future_perturbed_target_encodes') != 2
            or completion.get('windows') != 8 or completion.get('causal_checks') != 11
            or completion.get('encoder_call_attempts') != {'one_frame': 8, 'seventeen_frames': 10, 'total': 18}
            or completion.get('encoder_calls_completed') != completion['encoder_call_attempts']):
        raise ValueError('Incomplete successful native encoding counts/integrity evidence')
    if (manifest.get('codec_execution') != 'native CUDA FP32'
            or any(manifest.get(k) is not False for k in ['target_prefix_replaced', 'old_latent_cache_reused',
                                                        'commands_provided_to_codec', 'raw_capture_modified'])):
        raise ValueError('Wrong encoding/input boundaries')
    _validate_codec(completion['codec_provenance'])
    if manifest.get('codec_provenance') != completion['codec_provenance']:
        raise ValueError('Different codec provenance')
    if manifest.get('command_channels') != list(CHANNELS) or manifest.get('capture_manifest_sha256') != original.MANIFEST_SHA256:
        raise ValueError('Original command/capture contract differs')
    expected_shapes = shapes(manifest['profile'])
    if manifest.get('shapes') != {k: list(v) for k, v in expected_shapes.items()}:
        raise ValueError('Wrong native profile shapes')
    if (prepared.get('schema') != PLAN_SCHEMA or prepared.get('profile') != manifest['profile']
            or prepared.get('source_sha256') != sources()
            or prepared.get('capture_manifest_sha256') != original.MANIFEST_SHA256
            or prepared.get('command_channels') != list(CHANNELS)):
        raise ValueError('Wrong original plan identities')
    expected_ids = [f'{arm}-{start:04d}' for arm, start in SELECTION]
    if [row['id'] for row in manifest['windows']] != expected_ids or identity not in expected_ids:
        raise ValueError('Exactly the eight prescribed windows are required')
    if [row['name'] for row in manifest['causal_checks']] != check_names():
        raise ValueError('All eleven named prefix checks are required')
    for row in manifest['causal_checks']:
        relative = row.get('relative_l2')
        bit_exact_required = not row['name'].endswith('_target_vs_independent')
        if (row.get('passed') is not True or row.get('target_prefix_replaced') is not False
                or row.get('requires_bit_exact') is not bit_exact_required
                or (bit_exact_required and (row.get('bit_exact_equal') is not True or row.get('exact_equal') is not True
                                           or row.get('candidate_sha256') != row.get('reference_sha256')))
                or row.get('limits') != {'max_abs': PREFIX_MAX_ABS, 'relative_l2': PREFIX_RELATIVE_L2}
                or type(row.get('max_abs')) not in (int, float) or not math.isfinite(row['max_abs'])
                or not 0 <= row['max_abs'] <= PREFIX_MAX_ABS or type(relative) not in (int, float)
                or not math.isfinite(relative) or not 0 <= relative <= PREFIX_RELATIVE_L2):
            raise ValueError('Failed or changed predeclared causal tolerance')
        _check_evidence(directory, row, expected_shapes['observation'])
    observations = manifest['observations']
    expected_observations = ['start-0000-shared'] + [key for key in expected_ids if not key.endswith('-0000')]
    if [row['id'] for row in observations] != expected_observations:
        raise ValueError('Seven independent initial-image encodes are required')
    observation_map = {row['id']: row for row in observations}
    plans = {row['id']: row for row in prepared['selection']}
    if list(plans) != expected_ids or plans['closed-0000']['processed_first_rgb_sha256'] != plans['open-0000']['processed_first_rgb_sha256']:
        raise ValueError('Missing ordered source plan or shared canonical initial RGB')
    for row in observations:
        if row.get('encoded_rgb_frames') != 1 or set(row['tensors']) != {'observation'}:
            raise ValueError('Independent observation encoding record differs')
        _checked_file(directory, row)
    for row in manifest['windows']:
        observation_id = 'start-0000-shared' if row['id'].endswith('-0000') else row['id']
        if row['observation_id'] != observation_id or row['tensors']['observation'] != observation_map[observation_id]['tensors']['observation']:
            raise ValueError('Window observation differs from its independent image encoding')
        if (row.get('source') != plans[row['id']] or set(row['tensors']) != set(expected_shapes)
                or row['tensors']['commands']['sha256'] != plans[row['id']]['commands_sha256']
                or observation_map[observation_id]['processed_first_rgb_sha256'] != plans[row['id']]['processed_first_rgb_sha256']):
            raise ValueError('Window RGB/command source identity differs')
        prefix = next(c for c in manifest['causal_checks'] if c['name'] == row['id'] + '_target_vs_independent')
        if (prefix['candidate_sha256'] != row.get('target_prefix_sha256')
                or prefix['reference_sha256'] != row['tensors']['observation']['sha256']):
            raise ValueError('Cross-length check binds different target/observation prefixes')
        future_name = row['id'] + '_same_length_future_perturbation'
        for check_row in manifest['causal_checks']:
            if check_row['name'] == future_name and check_row['reference_sha256'] != row['target_prefix_sha256']:
                raise ValueError('Same-length comparison binds a different original target prefix')
    repeated = next(c for c in manifest['causal_checks'] if c['name'] == 'shared_initial_after_full_perturbation')
    if repeated['reference_sha256'] != observation_map['start-0000-shared']['tensors']['observation']['sha256']:
        raise ValueError('Repeated image check binds a different shared observation')
    expected_outputs = {'plan.json', 'manifest.json'} | {'source/' + name for name in sources()}
    expected_outputs |= {row['file'] for row in manifest['windows'] + observations}
    expected_outputs |= {row['evidence']['file'] for row in manifest['causal_checks']}
    if set(completion.get('output_sha256', {})) != expected_outputs:
        raise ValueError('Incomplete output/source hash inventory')
    for name, checksum in completion['output_sha256'].items():
        _checked_file(directory, {'file': name, 'sha256': checksum})
    for name, checksum in sources().items():
        if completion['output_sha256']['source/' + name] != checksum:
            raise ValueError('Retained source snapshot differs from its declared identity')
    row = next(row for row in manifest['windows'] if row['id'] == identity)
    path = _checked_file(directory, row)
    names = ['observation', 'commands'] if conditioning_only else ['target', 'observation', 'commands']
    values = {}
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(expected_shapes):
            raise ValueError('Unexpected cached tensor keys')
        for name, shape in expected_shapes.items():
            if tuple(handle.get_slice(name).get_shape()) != shape:
                raise ValueError('Malformed cache tensor shape')
        for name in names:
            value = handle.get_tensor(name)
            _cpu_tensor(value, expected_shapes[name], name)
            if row['tensors'][name] != {'shape': list(expected_shapes[name]), 'dtype': 'float32', 'sha256': tensor_sha(value)}:
                raise ValueError('Cached tensor identity differs')
            values[name] = value
    if not conditioning_only:
        measured = prefix_check(values['target'][:, :, :1], values['observation'], identity + '_target_vs_independent',
                                'Full 17-frame prefix versus separately encoded single RGB image')
        saved = next(c for c in manifest['causal_checks'] if c['name'] == measured['name'])
        if not _matches_prefix_row(saved, measured):
            raise ValueError('Recomputed target/observation prefix check differs')
    return values, {'manifest_sha256': sha(directory / 'manifest.json'),
                    'completion_sha256': sha(directory / 'completion.json'),
                    'plan_sha256': canonical_sha(prepared), 'source_sha256': sources(),
                    'codec_provenance': completion['codec_provenance'], 'profile': manifest['profile'],
                    'window_id': identity, 'file_sha256': row['sha256'],
                    'materialized_tensor_keys': names, 'source': row['source']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--profile', choices=['baseline', 'spatial'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = write_plan(args.capture, args.profile, args.output)
    print(json.dumps({'status': 'planned', 'profile': args.profile,
                      'plan_sha256': canonical_sha(result), 'model_execution': False}))


if __name__ == '__main__':
    main()
