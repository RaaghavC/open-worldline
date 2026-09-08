"""Work-only fixed128 preparation and single-use guarded execution.

Default CLI prepares CPU bytes. No resource provisioning or automatic admission.
Run from the verified repository root so its unchanged modules are importable.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import torch
from safetensors.torch import save_file

# The work-only script lives outside the verified repository. Its explicit
# working directory supplies the unchanged package, including in the child.
sys.path.insert(0, str(Path.cwd()))
import prototype
import extension
import effect_inputs
from experiments.wan22_native.action_cuda import probe, probe_evidence as evidence
from experiments.wan22_native.action_cuda.bridge import NativeCUDAActionBridge, PROFILES
from experiments.wan22_native.action_adapter.model import PostBlockActionAdapter
from experiments.wan22_native.cuda_reference.native import load_model
from experiments.wan22_native.cuda_reference.guards import LIMITS
from experiments.wan22_native.spatial_reference.guards import atomic, hardware, Monitor, supervise, stop_child

HERE = Path(__file__).resolve().parent
SCHEMA = 'worldline-action-effect128-run-v1'
SCOPE = 'effect128-fresh-cuda-training-only'
sha = evidence.sha


def sources():
    repository=evidence.source_hashes(independent=True)
    repository.update(extension.diagnostic().vi.sources()['repository'])
    names=('prototype.py','runner.py','extension.py','test_prototype.py','test_runner.py','reference-identities.json','effect.py','effect_inputs.py','test_effect.py','original128-plan.json','original-source.json','heldout-manifest.json')
    reference={str(p.relative_to(HERE)):sha(p) for folder in ('reference_fixed16','reference_diagnostic') for p in (HERE/folder).rglob('*.py')}
    return {'repository':repository,'local':{name:sha(HERE/name) for name in names},'reference':reference}


def cpu_review(path):
    report=evidence.read_json(path)
    if report.get('status')!='passed' or report.get('source_sha256')!=sources() or report.get('tests',0)<5 or any(report.get(k)!=0 for k in ('failures','errors','skipped')):
        raise ValueError('Current source-bound CPU review required')
    return sha(path)


def safe_root(path, *, fresh=False):
    root = Path(path).absolute()
    if (any(p.is_symlink() for p in (root, *root.parents)) or root.resolve().is_relative_to(evidence.REPO)
            or (root.exists() if fresh else not root.is_dir())):
        raise ValueError('Require a regular output outside the repository; preparation must be fresh')
    return root


def verify_probe(directory, cache, profile):
    root = safe_root(directory)
    prepared, *_ = probe.read_prepared(root, cache)
    parent = evidence.read_json(root / 'metrics.json')
    terminal = evidence.read_json(root / 'terminal.json')
    result = evidence.read_json(root / 'result/metrics.json')
    if (parent.get('schema') != probe.SCHEMA or parent.get('status') != 'passed'
            or parent.get('model_execution') is not True or result.get('status') != 'passed'
            or type(result.get('completed_updates')) is not int or result['completed_updates'] != 2
            or any(result.get(name) is not True for name in ('zero_adapter_gate_passed', 'base_unchanged',
                'all825_current_value_hashes_verified', 'second_update_gru_gradient_nonzero', 'inputs_unchanged', 'sources_unchanged'))
            or terminal.get('status') != 'complete' or type(terminal.get('exit_code')) is not int
            or terminal['exit_code'] != 0 or terminal.get('cleanup_error') is not None
            or parent.get('result_metrics_sha256') != sha(root / 'result/metrics.json')
            or parent.get('terminal_sha256') != sha(root / 'terminal.json')
            or parent.get('plan_sha256') != sha(root / 'plan.json')
            or result.get('plan_sha256') != sha(root / 'plan.json')
            or prepared['profile'] != profile or result.get('profile') != profile
            or result.get('source_sha256') != prepared['source_sha256']
            or result.get('input_identity') != prepared['input_identity']
            or result.get('schedule') != prepared['schedule'] or list(root.rglob('watchdog-stop.json'))):
        raise ValueError('A complete actual passed two-update probe on this cache/profile is required')
    required = {'core-before.json', 'core-after.json', 'weight-load.json', 'last-valid.json',
                'checkpoint-0002/adapter.safetensors', 'checkpoint-0002/optimizer-and-rng.pt',
                'checkpoint-0002/manifest.json'}
    if not required <= set(result.get('output_sha256', {})):
        raise ValueError('Prior probe final recovery and foundation evidence is required')
    for name, digest in result['output_sha256'].items():
        if sha(evidence.relative_file(root / 'result', name)) != digest:
            raise ValueError('Prior numerical evidence changed')
    before = evidence.read_json(root / 'result/core-before.json')
    after = evidence.read_json(root / 'result/core-after.json')
    if before != after or before != probe._original_weights(evidence.read_json(root / 'result/weight-load.json')):
        raise ValueError('Prior probe foundation value records disagree')
    return prepared, {'parent_sha256': sha(root / 'metrics.json'), 'result_sha256': sha(root / 'result/metrics.json'),
                      'terminal_sha256': sha(root / 'terminal.json'), 'plan_sha256': sha(root / 'plan.json'),
                      'source_sha256': prepared['source_sha256'], 'profile': profile,
                      'trained_adapter_loaded': False, 'warm_start': False}


def snapshot(out, mapping):
    for group, records in mapping.items():
        base = evidence.REPO if group == 'repository' else HERE
        for name, digest in records.items():
            destination = out / 'source' / group / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / name, destination)
            if sha(destination) != digest:
                raise ValueError('Source changed while copying')


def prepare(*, completed_probe, cache_run, text_directory, fixed128_prepared, diagnostic_prepared, cpu_report, profile, output):
    out = safe_root(output, fresh=True)
    for source in (completed_probe, cache_run, text_directory):
        if out.resolve().is_relative_to(Path(source).resolve()):
            raise ValueError('Preparation output cannot be inside an input directory')
    out.mkdir(parents=True)
    status = {'schema': SCHEMA, 'status': 'preparing', 'model_execution': False}
    try:
        mapping = sources();cpu_sha=cpu_review(cpu_report)
        diagnostic_plan,*_=extension.diagnostic().read_prepared(diagnostic_prepared)
        diagnostic_plan_sha=sha(Path(diagnostic_prepared)/'plan.json')
        prior, probe_identity = verify_probe(completed_probe, cache_run, profile)
        _, cache_identity = evidence.load_cache(cache_run, profile)
        context, text_identity = evidence.load_positive(text_directory)
        if cache_identity != prior['input_identity']['cache'] or text_identity != prior['input_identity']['text']:
            raise ValueError('Fresh preparation must use the actual probe cache/text')
        original128,initial,draws,rng,saved=effect_inputs.original_inputs(fixed128_prepared)
        schedule=original128['schedule']
        if original128['profile']!=profile:raise ValueError('Use the original128 profile unchanged')
        negative,negative_identity=effect_inputs.negative_context(text_directory)
        extension.prefix_identity(initial,schedule,draws)
        reference=extension.references()
        if reference['input_identity']['cache']!=cache_identity or reference['input_identity']['text']!=text_identity:raise ValueError('Keep original fixed16 cache and text')
        if diagnostic_plan['checkpoint_zero']['checkpoint_sha256']!=reference['checkpoint_zero_sha256']:raise ValueError('Diagnostic cp0 differs')
        if schedule[:2] != prior['schedule']:
            raise ValueError('Fresh fixed128 draw prefix differs from the completed probe')
        wanted_initial = prior['input_identity']['prepared_artifacts']['initial-adapter.safetensors']['tensors']
        if {name: prototype.numerical.tensor_sha(value) for name, value in initial.items()} != {name: row['sha256'] for name, row in wanted_initial.items()}:
            raise ValueError('Fresh initialized tensor bytes differ from the prior fresh preparation')
        saved['negative.safetensors']={'context':negative}
        artifacts = {}
        for name, values in saved.items():
            if name=='negative.safetensors':save_file(values,str(out/name))
            else:shutil.copyfile(evidence.relative_file(fixed128_prepared,name),out/name)
            artifacts[name] = probe.record_file(out / name, values)
        effect_inputs.original_records(artifacts)
        plan = {'schema': SCHEMA, 'status': 'prepared', 'scope': SCOPE, 'profile': profile,
                'source_sha256': mapping, 'protocol': prototype.protocol(profile), 'schedule': schedule,
                'input_identity': {'probe': probe_identity, 'cache': cache_identity, 'text': text_identity, 'negative_text':negative_identity, 'artifacts': artifacts},
                'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__), 'machine': platform.machine()},
                'limits': LIMITS, 'expected_gpu': 'NVIDIA A100-SXM4-80GB',
                'runner_admission_implemented': True,'cpu_report_sha256':cpu_sha,
                'diagnostic_plan_sha256':diagnostic_plan_sha,'diagnostic_execution_required':True,
                'fixed16_reference_sha256':sha(HERE/'reference-identities.json'),
                'original128_input_plan_sha256':effect_inputs.ORIGINAL_PLAN_SHA, 'heldout_assessment':effect_inputs.heldout_assessment(),
                'objective':'Original FM plus start0 guided clean-endpoint contrast MSE at ideal s1/lambda1. After128 use the unchanged matched visual pair and fixed-input diagnostic, final128 only',
                'checkpoint_selection':False,'negative_context_adapter_training':True,
                'model_execution': False, 'warm_start': False, 'resume_supported': False}
        snapshot(out, mapping)
        shutil.copyfile(cpu_report,out/'cpu-report.json')
        shutil.copyfile(Path(diagnostic_prepared)/'plan.json',out/'diagnostic-plan.json')
        if sources() != mapping:
            raise ValueError('Preparation sources changed')
        atomic(out / 'plan.json', plan)
        status.update(status='prepared', plan_sha256=sha(out / 'plan.json'))
        return plan
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic(out / 'metrics.json', status)


def read_prepared(root, completed_probe, cache_run, text_directory):
    root = safe_root(root); plan = evidence.read_json(root / 'plan.json')
    if (plan.get('schema') != SCHEMA or plan.get('status') != 'prepared' or plan.get('scope') != SCOPE
            or plan.get('source_sha256') != sources() or plan.get('protocol') != prototype.protocol(plan['profile'])
            or plan.get('limits') != LIMITS or plan.get('warm_start') is not False
            or plan.get('resume_supported') is not False or plan.get('runner_admission_implemented') is not True
            or plan.get('expected_gpu') != 'NVIDIA A100-SXM4-80GB' or plan.get('negative_context_adapter_training') is not True):
        raise ValueError('Exact current fixed128 preparation required')
    if plan.get('heldout_assessment')!=effect_inputs.heldout_assessment():raise ValueError('Predeclared held-out assessment differs')
    effect_inputs.original_schedule(plan['schedule'])
    for group, rows in plan['source_sha256'].items():
        for name, digest in rows.items():
            if sha(evidence.relative_file(root / 'source' / group, name)) != digest:
                raise ValueError('Retained source snapshot differs')
    if cpu_review(root/'cpu-report.json')!=plan['cpu_report_sha256'] or sha(root/'diagnostic-plan.json')!=plan['diagnostic_plan_sha256'] or plan.get('diagnostic_execution_required') is not True or plan.get('fixed16_reference_sha256')!=sha(HERE/'reference-identities.json'):raise ValueError('CPU/diagnostic/fixed16 source binding differs')
    prior, prior_identity = verify_probe(completed_probe, cache_run, plan['profile'])
    windows, cache_identity = evidence.load_cache(cache_run, plan['profile'])
    context, text_identity = evidence.load_positive(text_directory)
    if any(plan['input_identity'][key] != value for key, value in (
            ('probe', prior_identity), ('cache', cache_identity), ('text', text_identity))):
        raise ValueError('Prepared cache, probe or text identity changed')
    specs = {'initial-adapter.safetensors': probe._initial_specs(),
             'positive.safetensors': {'context': ((25,4096),'F32')},
             'negative.safetensors': {'context': ((126,4096),'F32')},
             'initial-cpu-rng.safetensors': {'rng': ((torch.get_rng_state().numel(),),'U8')}}
    for first in range(0,128,16):
        spec={**{f'noise_{i:04d}':((1,*PROFILES[plan['profile']]['shape']),'F32') for i in range(first,first+16)},
              **{f'rng_after_{i:04d}':((torch.get_rng_state().numel(),),'U8') for i in range(first,first+16)}}
        if first==0:spec['rng_initial']=((torch.get_rng_state().numel(),),'U8')
        specs[f'draws-{first:04d}-{first+15:04d}.safetensors']=spec
    records = plan['input_identity']['artifacts']
    if set(records) != set(specs):
        raise ValueError('All fresh artifact files are required')
    values = {}
    for name, shapes in specs.items():
        path = evidence.relative_file(root, name)
        values[name] = evidence.tensors(path, shapes, records[name]['sha256'])
        if probe.record_file(path, values[name]) != records[name]:
            raise ValueError('Saved fresh tensor identity differs')
    effect_inputs.original_records(records)
    negative,negative_identity=effect_inputs.negative_context(text_directory)
    if plan.get('original128_input_plan_sha256')!=effect_inputs.ORIGINAL_PLAN_SHA or plan['input_identity']['negative_text']!=negative_identity or not torch.equal(negative,values['negative.safetensors']['context']):raise ValueError('Original input plan or genuine negative context differs')
    initial = values['initial-adapter.safetensors']; draws = {k:v for name,rows in values.items() if name.startswith('draws-') for k,v in rows.items()}
    extension.prefix_identity(initial,plan['schedule'],draws)
    wanted = prior['input_identity']['prepared_artifacts']['initial-adapter.safetensors']['tensors']
    if (plan['schedule'][:2] != prior['schedule'] or torch.count_nonzero(initial['output.weight'])
            or torch.count_nonzero(initial['output.bias'])
            or {k: prototype.tensor_sha(v) for k, v in initial.items()} != {k: v['sha256'] for k, v in wanted.items()}
            or not torch.equal(context, values['positive.safetensors']['context'])):
        raise ValueError('Fresh initialization, prefix draws or genuine context differ')
    prototype.validate_inputs(windows, plan['schedule'], draws, (1, *PROFILES[plan['profile']]['shape']))
    return plan, windows, initial, draws, context, values['initial-cpu-rng.safetensors']['rng']


def admission(path, root, plan, diagnostic_record):
    value = evidence.read_json(path)
    expected = {'schema': 'worldline-action-effect128-admission-v1', 'decision': 'admit',
                'issued_by': 'parent-agent', 'scope': SCOPE, 'plan_sha256': sha(root / 'plan.json'),
                'source_sha256': plan['source_sha256'], 'input_identity': plan['input_identity'],
                'profile': plan['profile'], 'limits': LIMITS, 'expected_gpu': plan['expected_gpu'],
                'image_generation_admitted': False, 'resume_admitted': False,
                'diagnostic_evidence':extension.diagnostic_identity(diagnostic_record),'cfg_strategy_unchanged':True,'cpu_report_sha256':plan['cpu_report_sha256'], 'auxiliary_objective':{'lambda':1.0,'endpoint_scale':1.0,'updates':list(range(1,129,4))},'original128_input_plan_sha256':effect_inputs.ORIGINAL_PLAN_SHA,'heldout_assessment':effect_inputs.heldout_assessment()}
    if any(value.get(key) != wanted for key, wanted in expected.items()) or not isinstance(value.get('reason'), str) or not value['reason'].strip() or not isinstance(value.get('cfg_assessment'),str) or not value['cfg_assessment'].strip():
        raise ValueError('Exact-plan parent fixed128 admission required')
    return {'sha256': sha(path), 'scope': SCOPE}


def worker(config):
    root = safe_root(config['prepared']); out = root / 'worker'; out.mkdir()
    started = time.monotonic(); report = {'schema': SCHEMA, 'status': 'running', 'model_execution': False}
    try:
        plan, windows, initial, draws, context, rng = read_prepared(root, config['completed_probe'], config['cache_run'], config['text_directory'])
        diagnostic_record=extension.validate_diagnostic(config['diagnostic_result'],plan['diagnostic_plan_sha256'])
        admitted = admission(config['admission'], root, plan,diagnostic_record)
        if sha(root / 'plan.json') != config['plan_sha256'] or admitted != config['admission_record']:
            raise ValueError('Prepared plan or decision changed after launch')
        actual_hardware = hardware(plan['expected_gpu'])
        hardware_comparison=extension.diagnostic().visual.require_hardware(actual_hardware,plan['input_identity']['cache']['hardware'])
        flags = probe._runtime_flags()
        report.update(plan_sha256=config['plan_sha256'], source_sha256=plan['source_sha256'], hardware=actual_hardware,
                      admission=admitted, runtime_flags=flags, limits=LIMITS,hardware_comparison=hardware_comparison,diagnostic_evidence=diagnostic_record)
        atomic(out / 'metrics.json', report)
        with Monitor(out, config['deadline'], 'pair') as monitor:
            def check():
                monitor.check()
                if probe._runtime_flags() != flags:
                    raise RuntimeError('Precision/runtime flags changed')
            began = time.monotonic(); core, loaded = load_model(config['weights'], check)
            torch.cuda.synchronize(); report['load_seconds'] = time.monotonic() - began
            atomic(out / 'weight-load.json', loaded)
            before = probe._original_weights(loaded)
            adapter = PostBlockActionAdapter(); adapter.load_state_dict(initial, strict=True); adapter.to('cuda:0')
            bridge = NativeCUDAActionBridge(core, adapter, profile=plan['profile'])
            torch.set_rng_state(rng); torch.cuda.manual_seed_all(prototype.original.SEED)
            save_file({'rng': torch.cuda.get_rng_state(0)}, str(out / 'initial-cuda-rng.safetensors'))
            optimizer = torch.optim.AdamW(adapter.parameters(), **prototype.original.OPTIMIZER)
            report['model_execution'] = True; atomic(out / 'metrics.json', report)
            negative,_=effect_inputs.negative_context(config['text_directory'])
            result = prototype.run_sequence(bridge, windows, plan['schedule'], draws, context, optimizer, root / 'training',
                expected_initial=initial, expected_core=before, negative_context=negative, native_predict=lambda x, t, c: probe.native_reference(core, x, t, c),
                identity={'schema':SCHEMA,'plan_sha256':config['plan_sha256'],'admission':admitted,
                          'prepared_draw_files':{name:row['sha256'] for name,row in plan['input_identity']['artifacts'].items() if name.startswith('draws-')},
                          'diagnostic_result_sha256':diagnostic_record['result_sha256']},
                check=check, synchronize=torch.cuda.synchronize)
            save_file({'rng': torch.cuda.get_rng_state(0)}, str(out / 'final-cuda-rng.safetensors'))
            check()
        if result['status'] != 'passed' or result['completed_updates'] != 128 or result['base_unchanged'] is not True:
            raise RuntimeError('Fixed128 computation did not complete')
        read_prepared(root, config['completed_probe'], config['cache_run'], config['text_directory'])
        if time.monotonic() >= config['deadline']:
            raise RuntimeError('Deadline reached during final verification')
        report.update(status='passed', completed_updates=128, base_unchanged=True,negative_context_adapter_training=True,
                      training_metrics_sha256=sha(root / 'training/metrics.json'),
                      output_sha256={p.relative_to(root).as_posix(): sha(p) for directory in (out, root / 'training')
                                     for p in directory.rglob('*') if p.is_file() and p != out / 'metrics.json' and p.name != 'memory.jsonl'})
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started; atomic(out / 'metrics.json', report)


def execute(*, prepared, completed_probe, cache_run, text_directory, diagnostic_result, weights, decision):
    root = safe_root(prepared)
    if any((root / name).exists() for name in ('execution-attempt.json', 'launch.json', 'terminal.json', 'worker', 'training', 'worker.log')):
        raise ValueError('Fixed128 execution is single-use; retain partial evidence')
    started = time.monotonic()
    # Atomic exclusive marker also stops two callers racing before validation.
    with (root / 'execution-attempt.json').open('x') as handle:
        json.dump({'schema': SCHEMA, 'single_use': True}, handle)
    parent = {'schema': SCHEMA, 'status': 'running', 'model_execution': False, 'limits': LIMITS}
    proc = None
    try:
        plan, *_ = read_prepared(root, completed_probe, cache_run, text_directory)
        diagnostic_record=extension.validate_diagnostic(diagnostic_result,plan['diagnostic_plan_sha256'])
        admitted = admission(decision, root, plan,diagnostic_record)
        config = {'prepared': str(root),'diagnostic_result':str(Path(diagnostic_result).absolute()), 'completed_probe': str(Path(completed_probe).absolute()),
                  'cache_run': str(Path(cache_run).absolute()), 'text_directory': str(Path(text_directory).absolute()),
                  'weights': str(Path(weights).absolute()), 'admission': str(Path(decision).absolute()),
                  'admission_record': admitted, 'plan_sha256': sha(root / 'plan.json'), 'deadline': started + 900.}
        atomic(root / 'launch.json', config); shutil.copyfile(decision, root / 'executed-admission.json')
        if sha(root / 'executed-admission.json') != admitted['sha256']:
            raise ValueError('Admission changed while copying')
        if time.monotonic() >= config['deadline']:
            raise RuntimeError('Deadline reached before launch')
        with (root / 'worker.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, str(HERE / 'runner.py'), '--worker-config', str(root / 'launch.json')],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            supervise(proc, root, config['deadline'], 'pair')
        terminal = evidence.read_json(root / 'terminal.json'); result = evidence.read_json(root / 'worker/metrics.json')
        if (terminal.get('status') != 'complete' or terminal.get('exit_code') != 0 or terminal.get('cleanup_error') is not None
                or result.get('status') != 'passed' or result.get('completed_updates') != 128
                or result.get('base_unchanged') is not True or result.get('plan_sha256') != config['plan_sha256']
                or result.get('source_sha256') != plan['source_sha256'] or list(root.rglob('watchdog-stop.json'))):
            raise RuntimeError('Guarded fixed128 worker did not complete')
        for name, digest in result['output_sha256'].items():
            if sha(evidence.relative_file(root, name)) != digest:
                raise ValueError('Worker output changed')
        if time.monotonic() >= config['deadline']:
            raise RuntimeError('Deadline reached during parent verification')
        parent.update(status='passed', model_execution=True, completed_updates=128, worker_metrics_sha256=sha(root / 'worker/metrics.json'),
                      terminal_sha256=sha(root / 'terminal.json'), plan_sha256=config['plan_sha256'], admission=admitted)
        return parent
    except BaseException as error:
        cleanup_error = None
        try:
            if proc is not None: stop_child(proc)
        except BaseException as cleanup:
            cleanup_error = str(cleanup)
        finally:
            if not (root / 'terminal.json').exists():
                atomic(root / 'terminal.json', {'status': 'failed', 'mode': 'pair', 'exit_code': proc.returncode if proc else None,
                                               'error': str(error), 'cleanup_error': cleanup_error})
            elif cleanup_error:
                atomic(root / 'handoff-cleanup-error.json', {'error': str(error), 'cleanup_error': cleanup_error})
        parent['model_execution'] = None if proc is not None else False
        if proc is not None and (root / 'worker/metrics.json').is_file():
            try:
                partial = evidence.read_json(root / 'worker/metrics.json')
                if type(partial.get('model_execution')) is bool: parent['model_execution'] = partial['model_execution']
            except Exception as read_error:
                parent['partial_metrics_read_error'] = str(read_error)
        parent.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        parent['elapsed_seconds'] = time.monotonic() - started; atomic(root / 'metrics.json', parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-config', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--profile', choices=tuple(PROFILES), default='spatial')
    for name in ('completed-probe','cache-run','text-directory','fixed128-prepared','diagnostic-prepared','diagnostic-result','cpu-report','output','prepared','weights','decision'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.worker_config:
        result = worker(evidence.read_json(args.worker_config, maximum_bytes=4*2**20))
    else:
        fields = ('prepared','completed_probe','cache_run','text_directory','diagnostic_result','weights','decision') if args.execute else (
            'completed_probe','cache_run','text_directory','fixed128_prepared','diagnostic_prepared','cpu_report','profile','output')
        if any(getattr(args, name) is None for name in fields): parser.error('Supply every explicit input path')
        result = (execute if args.execute else prepare)(**{name: getattr(args, name) for name in fields})
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
