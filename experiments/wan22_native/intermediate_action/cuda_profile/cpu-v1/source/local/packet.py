# SPDX-License-Identifier: Apache-2.0
"""Small exact public-input selection and CPU-only preparation."""
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from experiments.wan22_native.action_cuda import probe_evidence as old
from experiments.wan22_native.action_adapter.model import PostBlockActionAdapter
from experiments.wan22_native.official_cpu.streaming import sha, tensor_sha
from experiments.wan22_native.spatial_reference.guards import atomic, limits

HERE = Path(__file__).resolve().parent
REPO = old.REPO
SCHEMA = 'worldline-intermediate-cuda-profile-v1'
SCOPE = 'two-update-placement-numerical-profile-only'
PLACEMENTS = (28, 29)
SHAPE = (1, 48, 5, 44, 78)


def source_paths():
    paths = {'repo/'+name: path for name, path in old.source_paths().items()}
    for name in ('intermediate_action/bridge.py', 'intermediate_action/cached_intermediate.py',
                 'intermediate_action/test_bridge.py', 'action_effect/training/effect.py',
                 'spatial_reference/guards.py', 'spatial_reference/__init__.py'):
        path = REPO/'experiments/wan22_native'/name
        paths['repo/'+str(path.relative_to(REPO))] = path
    for name in ('packet.py', 'engine.py', 'run.py', 'test_cpu.py', 'selection.json'):
        paths['local/'+name] = HERE/name
    return dict(sorted(paths.items()))


def sources():
    return {name: sha(path) for name, path in source_paths().items()}


def protocol():
    return dict(scope=SCOPE, profile='spatial', placements=list(PLACEMENTS),
        zero_parity='Require bit-exact native/full/cached outputs before any update',
        schedule=old.read_json(HERE/'selection.json')['schedule'],
        updates_per_placement=2, main_predictions_per_placement=4,
        auxiliary_feature_extracts_per_placement=4, auxiliary_predictions_per_placement=8,
        optimizer=dict(lr=.0001, betas=[.9,.999], eps=1e-8, weight_decay=.01), clip_l2=1.,
        loss='Original two half-weight future FM losses plus original lambda1 auxiliary; one clip and AdamW step',
        limits=limits('pair'), image_generation=False, quality_assessed=False, automatic_promotion=False)


def read_selection(root):
    """Hash all eleven original files; materialize only selected CPU tensors."""
    selected = old.read_json(HERE/'selection.json')
    loaded = {}
    for name, record in selected['files'].items():
        path = old.relative_file(root, name)
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Original public input bytes differ: '+name)
        if 'header' not in record:
            continue
        with safe_open(path, framework='pt', device='cpu') as handle:
            if set(handle.keys()) != set(record['header']):
                raise ValueError('Input tensor names differ')
            for key, spec in record['header'].items():
                view = handle.get_slice(key)
                if view.get_shape() != spec['shape'] or view.get_dtype() != spec['dtype']:
                    raise ValueError('Input tensor header differs')
            values = {}
            for key, expected in record['selected'].items():
                value = handle.get_tensor(key)
                if not torch.isfinite(value).all() or tensor_sha(value) != expected:
                    raise ValueError('Selected tensor value differs')
                values[key] = value
            loaded[name] = values
    def one(suffix):
        matches = [values for name, values in loaded.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError('Unambiguous selected input required')
        return matches[0]
    windows = [one('/'+arm+'-0000.safetensors') for arm in ('closed', 'open')]
    initial = one('/checkpoint-0000/adapter.safetensors')
    draws = one('/draws-0000-0015.safetensors')
    positive, negative = (one('/'+name+'.safetensors')['context'] for name in ('positive', 'negative'))
    rng = one('/initial-cpu-rng.safetensors')['rng']
    with torch.device('meta'):
        specs = PostBlockActionAdapter().state_dict()
    if set(initial) != set(specs) or sum(v.numel() for v in initial.values()) != 947712:
        raise ValueError('Exact original adapter coverage required')
    if any(initial[k].shape != v.shape or initial[k].dtype != torch.float32 for k,v in specs.items()):
        raise ValueError('Original adapter shape/dtype differs')
    if any(torch.count_nonzero(initial[k]) for k in ('output.weight','output.bias')):
        raise ValueError('Exact zero output projection required')
    if any(tuple(w['target'].shape) != SHAPE or not torch.equal(w['target'][:,:,:1], w['observation']) for w in windows):
        raise ValueError('Selected target prefix must remain bit-exact')
    if not torch.equal(windows[0]['observation'], windows[1]['observation']):
        raise ValueError('Shared independent observation differs')
    expected = torch.zeros_like(windows[0]['commands']); expected[0,0,5] = 1.
    if not torch.equal(windows[1]['commands']-windows[0]['commands'], expected):
        raise ValueError('Only the first wait/interact command may differ')
    return dict(windows=windows, initial=initial, draws=draws, positive=positive, negative=negative, rng=rng,
                schedule=selected['schedule'])


def review(path):
    report = old.read_json(path)
    if (report.get('status') != 'passed' or report.get('source_sha256') != sources()
            or report.get('sources_unchanged') is not True or report.get('tests', 0) < 6
            or any(type(report.get(k)) is not int or report[k] != 0
                   for k in ('pytest_exit_code','failures','errors','skipped'))):
        raise ValueError('Current complete CPU evidence required')
    return sha(path)


def prepare(input_root, cpu_report, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh preparation output required')
    review_hash = review(cpu_report)
    read_selection(input_root)
    output.mkdir(parents=True)
    selected = old.read_json(HERE/'selection.json')
    for name in selected['files']:
        target = output/'inputs'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old.relative_file(input_root, name), target)
    mapping = sources()
    for name, path in source_paths().items():
        target = output/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(cpu_report, output/'cpu-report.json')
    plan = dict(schema=SCHEMA, status='prepared', model_execution=False, protocol=protocol(),
                source_sha256=mapping, input_selection_sha256=sha(HERE/'selection.json'),
                input_files=selected['files'], cpu_report_sha256=review_hash)
    atomic(output/'plan.json', plan)
    read_prepared(output)
    return plan


def read_prepared(root):
    root = Path(root)
    plan = old.read_json(old.relative_file(root,'plan.json'))
    if (plan.get('schema') != SCHEMA or plan.get('status') != 'prepared'
            or plan.get('protocol') != protocol() or plan.get('source_sha256') != sources()
            or plan.get('input_selection_sha256') != sha(HERE/'selection.json')
            or plan.get('input_files') != old.read_json(HERE/'selection.json')['files']):
        raise ValueError('Prepared protocol, inputs or source identity differs')
    for name, digest in plan['source_sha256'].items():
        if sha(old.relative_file(root/'source',name)) != digest:
            raise ValueError('Retained source snapshot differs')
    if review(root/'cpu-report.json') != plan['cpu_report_sha256']:
        raise ValueError('CPU evidence changed')
    return plan, read_selection(root/'inputs')


def admission(path, root, plan, expected_gpu):
    value = old.read_json(path)
    required = dict(schema=SCHEMA, scope=SCOPE, decision='admit', plan_sha256=sha(Path(root)/'plan.json'),
        source_sha256=plan['source_sha256'], input_selection_sha256=plan['input_selection_sha256'],
        cpu_report_sha256=plan['cpu_report_sha256'], expected_gpu=expected_gpu,
        limits=limits('pair'), quality_admitted=False, image_generation=False)
    if not expected_gpu or any(value.get(k) != v for k,v in required.items()):
        raise ValueError('Separate exact source/input-bound parent numerical admission required')
    return dict(sha256=sha(path), **required)
