"""Saved command-attention profile audit. NumPy only; no model/backward replay."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
import numpy as np
from arrays import inventory, need, parse, path, read, record, sha, tensor_sha, tensors

HERE = Path(__file__).resolve().parent
RUN = 'action-results/command-attention-profile-v1'
SCHEMA = 'worldline-command-attention-profile-v1'
ARMS = tuple(m + '_' + d for m in ('stationary', 'left', 'right') for d in ('closed', 'interact'))
LIMITS = dict(seconds=900., host_rss_bytes=48 * 2**30, cuda_reserved_bytes=60 * 2**30,
              minimum_host_available_bytes=8 * 2**30, minimum_cuda_available_bytes=8 * 2**30,
              minimum_gpu_total_bytes=70 * 2**30)
PRECISION = {'parameter_storage': 'Original FP32', 'convert_model_dtype': False,
             'outer_autocast': 'Native CUDA BF16 with default cache setting',
             'inner_contexts': 'Unmodified upstream contexts',
             'attention': 'Unmodified upstream FlashAttention 2', 'rope': 'Unmodified upstream complex RoPE'}
REPOSITORY_SOURCES = ('cuda_reference/native.py', 'cuda_reference/guards.py', 'spatial_reference/guards.py',
    'cuda_reference/config.json', 'cuda_reference/upstream-provenance.json', 'cuda_reference/expected-weights.json',
    'action_training/objective.py', 'action_cuda/probe.py', 'official_cpu/streaming.py')


def close(actual, saved, label, *, precise=False):
    # FP32 GPU reduction is compared with an independent CPU reduction. This is
    # record-consistency rounding tolerance, not a prediction/quality threshold.
    need(type(saved) in (int, float) and math.isfinite(saved) and
         math.isclose(actual, saved, rel_tol=1e-10 if precise else 1e-5,
                      abs_tol=1e-12 if precise else 1e-7), 'Saved scalar differs: ' + label)
    return actual - saved


def seconds(value):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Finite nonnegative seconds')
    return value


def original_inputs(original, transfer, pins):
    def checked(name):
        p = path(original, name)
        need(record(p) == transfer[name], 'Original transfer file differs: ' + name)
        return p
    # All supplied producer dependencies are checked, without importing them.
    sources = {n: r for n, r in transfer.items() if not n.startswith('inputs/')}
    for name in sources:
        checked(name)
    p = checked('inputs/inputs.json')
    need(sha(p) == pins['input_manifest_sha256'], 'Pinned input identity')
    plan = read(p)
    need(plan['schema'] == 'worldline-command-attention-inputs-v1' and plan['updates'] == 512 and
         plan['shape'] == [1, 48, 5, 44, 78] and plan['controller_parameters'] == 4_936_448,
         'Actual native input contract')
    def values(name):
        p = checked('inputs/' + name)
        need(record(p) == {k: plan['files'][name][k] for k in ('bytes', 'sha256')}, 'Input plan file binding')
        return tensors(p, plan['files'][name]['tensors'])
    windows = {a: values('windows/' + a + '.safetensors') for a in ARMS}
    contexts = {k: values(k + '.safetensors')['context'] for k in ('positive', 'negative')}
    initial = values('initial-controller.safetensors')
    draws = values('draws/draws-0000-0015.safetensors')
    evaluation = values('evaluation-noises.safetensors')
    need({k: tensor_sha(v) for k, v in evaluation.items()} == plan['evaluation_tensor_sha256'], 'Evaluation identity loaded by producer')
    for row in plan['schedule'][:2]:
        need(tensor_sha(draws[row['noise_key']]) == row['noise_sha256'] and
             tensor_sha(draws[f"rng_after_{row['update']-1:04d}"]) == row['rng_after_sha256'], 'Two consumed draw identities')
    need(sum(x.size for x in initial.values()) == 4_936_448, 'Exact controller parameter count')
    for w in windows.values():
        need(tensor_sha(w['observation']) == tensor_sha(windows[ARMS[0]]['observation']), 'One shared observation')
    source_map = {'profiler/' + n: transfer['profile/' + n]['sha256']
                  for n in ('run_profile.py', 'engine.py', 'native_worker.py')}
    source_map.update({'controller/' + n: transfer['controller/' + n]['sha256'] for n in ('controller.py', 'bridge.py')})
    source_map.update({'training/' + n: transfer['training/' + n]['sha256'] for n in ('math_steps.py', 'packet.py')})
    source_map.update({'repository/' + n: transfer['repository/experiments/wan22_native/' + n]['sha256'] for n in REPOSITORY_SOURCES})
    return dict(plan=plan, windows=windows, contexts=contexts, initial=initial, draws=draws,
                source_map=source_map, verified_original_source_files=len(sources),
                transfer_source_sha256={n:r['sha256'] for n,r in transfer.items()
                    if not n.startswith('inputs/') or n == 'inputs/inputs.json'})


def source_bindings(config, report, data, protocol, pins):
    need(config['schema'] == SCHEMA and config['protocol'] == protocol, 'Exact profile launch protocol')
    need(config['inputs_sha256'] == pins['input_manifest_sha256'] and
         config['source_sha256'] == data['source_map'], 'Launch input/source bindings')
    expected_paths = dict(repository='/workspace/command-attention-v1/repository',
                          controller_source='/workspace/command-attention-v1/controller',
                          training_source='/workspace/command-attention-v1/training',
                          prepared='/workspace/command-attention-v1/inputs', output='/workspace/' + RUN)
    need(all(config[k] == v for k, v in expected_paths.items()), 'Exact actual runtime paths')
    need(config['expected_gpu'] == pins['expected_gpu'], 'Declared GPU identity')
    need(datetime.fromisoformat(config['deadline_utc'].replace('Z', '+00:00')).tzinfo is not None and
         type(config['deadline']) in (int, float) and math.isfinite(config['deadline']), 'Recorded explicit deadline')
    identity = dict(windows={a: {k: tensor_sha(v) for k, v in w.items()} for a, w in data['windows'].items()},
                    **{k: tensor_sha(v) for k, v in data['contexts'].items()})
    need(report['schema'] == SCHEMA and report['source_sha256'] == data['source_map'] and
         report['inputs_sha256'] == config['inputs_sha256'] and report['input_identity'] == identity and
         report['initial_controller_sha256'] == data['plan']['files']['initial-controller.safetensors']['sha256'] and
         report['rows'] == data['plan']['schedule'][:2] and report['precision'] == PRECISION,
         'Worker source, precision, input and row identities')
    return dict(source_files=len(data['source_map']), original_source_files=data['verified_original_source_files'],
                inputs_sha256=config['inputs_sha256'], input_identity=identity)


def core_records(result, expected, complete=True):
    loaded = read(result / 'weight-load.json')
    need(loaded['tensor_count'] == 825 and loaded['parameter_count'] == 4_999_787_712 and
         loaded['parameter_bytes'] == 19_999_150_848 and loaded['convert_model_dtype'] is False and
         loaded['all_shards_verified'] is True and loaded['cuda_copy_exact'] is True and
         set(loaded['tensors']) == set(expected), 'Complete original loader records')
    wanted = {}
    for name, e in expected.items():
        r = loaded['tensors'][name]
        need(r['shape'] == e['shape'] and r['shard'] == e['shard'] and
             r['original_dtype'] == r['loaded_dtype'] == 'float32' and
             r['source_sha256'] == r['loaded_sha256'] == e['original_sha256'] and
             r['source_owner_released'] is True and r['cuda_copy_exact'] is True, 'Original weight record: ' + name)
        wanted[name] = dict(shape=e['shape'], dtype='float32', sha256=e['original_sha256'])
    hashes = {}
    for label in ('before', 'after-parity', 'after-updates'):
        p = result / ('core-' + label + '.json')
        if not p.exists() and not complete:
            continue
        need(read(p) == wanted, 'All 825 current-value records: ' + label)
        hashes[label] = sha(p)
    return dict(records_per_phase=825, phases=len(hashes), complete_three_phases=len(hashes)==3, file_sha256=hashes,
                scope='Saved current-value hash records match pinned originals. Foundation tensors are not present for another hash computation.')


def check_memory(run, result, report, config):
    h = report['hardware']
    need(h['name'] == config['expected_gpu'] and type(h['total_memory_bytes']) is int and
         h['total_memory_bytes'] >= LIMITS['minimum_gpu_total_bytes'] and h['bf16_supported'] is True and
         len(h['capability']) == 2 and all(type(x) is int for x in h['capability']) and h['capability'][0] in (8, 9) and
         h['torch'].split('+')[0] == '2.5.1' and h['cuda'] == '12.4' and h['flash_attn'] == '2.7.4.post1' and
         h['flash_attention_2_available'] is True and h['flash_attention_3_available'] is False, 'Recorded admitted hardware/runtime')
    flags = report['runtime_flags']
    need(flags['flash_attention_2_available'] is True and flags['flash_attention_3_available'] is False,
         'Recorded native attention flags')
    for key in ('matmul_allow_tf32', 'cudnn_allow_tf32', 'cudnn_benchmark'):
        need(type(flags[key]) is bool and flags[key] == h[key], 'Runtime/hardware precision flag agreement')
    rows = {}
    for label, p in (('parent', run / 'parent-memory.jsonl'), ('worker', result / 'memory.jsonl')):
        need(record(p)['bytes'] <= 8 * 2**20, 'Bounded sampled memory log')
        values = [parse(line) for line in p.read_bytes().splitlines() if line]
        need(values, 'At least one memory sample')
        previous = -1.
        mismatches = 0
        for r in values:
            t = seconds(r['seconds'])
            need(previous <= t < 900, 'Sample chronology and deadline cap')
            previous = t
            rss = 'combined_rss_bytes' if label == 'parent' else 'host_rss_bytes'
            keys = (rss, 'host_available_bytes') + (() if label == 'parent' else
                       ('cuda_reserved_bytes', 'cuda_allocated_bytes', 'cuda_available_bytes'))
            need(all(type(r[k]) is int and r[k] >= 0 for k in keys), 'Memory byte integers')
            need(r[rss] <= LIMITS['host_rss_bytes'] and r['host_available_bytes'] >= LIMITS['minimum_host_available_bytes'], 'Original host caps')
            if label == 'worker':
                need(r['cuda_reserved_bytes'] <= LIMITS['cuda_reserved_bytes'] and
                     r['cuda_available_bytes'] >= LIMITS['minimum_cuda_available_bytes'], 'Original CUDA caps')
                mismatches += r['cuda_allocated_bytes'] > r['cuda_reserved_bytes']
        rows[label] = dict(samples=len(values), elapsed_last_sample=previous,
                           sequential_allocated_above_reserved_rows=mismatches)
    terminal = read(run / 'terminal.json')
    monitor = read(result / 'monitor-terminal.json')
    for r in (terminal, monitor):
        need(r['status'] == 'complete' and r['mode'] == 'pair' and r['limits'] == LIMITS and
             seconds(r['elapsed_seconds']) <= 900, 'Complete original guard terminal')
    need(terminal['exit_code'] == 0 and monitor['sample_count'] == rows['worker']['samples'], 'Terminal/count agreement')
    need(seconds(report['elapsed_seconds']) <= 900, 'Worker elapsed cap')
    for key in ('load_seconds', 'parity_seconds'):
        seconds(report[key])
    for value in report['native_seconds'].values():
        seconds(value)
    for value in report['foundation_hash_seconds'].values():
        seconds(value)
    for memory in [report['load_memory'], report['parity_memory'], report['memory']] + [r['memory'] for r in report['updates']]:
        need(all(type(v) is int and v >= 0 for v in memory.values()), 'Phase memory byte integers')
        for key in ('reserved_bytes', 'peak_reserved_bytes'):
            if key in memory:
                need(memory[key] <= LIMITS['cuda_reserved_bytes'], 'Recorded phase reserved memory cap')
    return dict(hardware=h, samples=rows, parent_seconds=terminal['elapsed_seconds'],
                worker_seconds=report['elapsed_seconds'],
                load_seconds=report['load_seconds'], parity_seconds=report['parity_seconds'],
                native_seconds=report['native_seconds'], foundation_hash_seconds=report['foundation_hash_seconds'],
                load_memory=report['load_memory'], parity_memory=report['parity_memory'],
                update_memory=[r['memory'] for r in report['updates']],
                limitation='Memory counters were read sequentially; allocated<=reserved is not an additional gate. Sampling does not establish every instantaneous allocation.')


def check_output_hashes(result, report):
    actual = {n: r['sha256'] for n, r in inventory(result).items() if Path(n).name != 'metrics.json'}
    need(actual == report['output_sha256'], 'Exact complete worker output inventory')
    return dict(output_files=len(actual), output_hashes_verified=True)


def dispatch_binding(recovered, run, config, data, pins, passed):
    root = path(recovered, 'action-results/command-attention-profile-dispatch-v1')
    request = read(root/'request.json')
    pid = read(root/'stage-pid.json')
    terminal = read(root/'stage-terminal.json')
    digest = sha(root/'request.json')
    need(request['schema'] == 'worldline-command-attention-stage-dispatch-v1' and request['stage'] == 'profile' and
         request['manifest_sha256'] == pins['transfer_manifest_sha256'] and
         request['inputs_sha256'] == pins['input_manifest_sha256'] and request['source_sha256'] == data['transfer_source_sha256'] and
         request['deadline_utc'] == config['deadline_utc'] and request['expected_gpu'] == config['expected_gpu'] and
         request['output'] == config['output'] and request['recovery_reserve_seconds'] == 600, 'Exact external dispatch binding')
    need(type(pid['pid']) is int and pid['pid'] > 0 and pid['pid'] == terminal['pid'] and
         pid['request_sha256'] == terminal['request_sha256'] == digest, 'Dispatch PID/request binding')
    need(sha(path(recovered, 'dispatch_command_attention.py')) == request['helper_sha256'], 'Retained executed dispatch source identity')
    records = {name:sha(run/name) for name in ('terminal.json','metrics.json','launch.json','result/metrics.json') if (run/name).is_file()}
    need(terminal['runner_records_sha256'] == records, 'External terminal binds exact runner records')
    def stamp(s):
        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        need(dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0, 'Explicit UTC lease timestamp')
        return dt.timestamp()
    lease = request['lease_plan']
    need(lease['schema'] == 'exact-pod-cleanup-plan-v1' and lease['pod_id'] == request['pod_id'] and
         lease['expected_network_volume'] is None and
         stamp(lease['cleanup_due_at']) == stamp(lease['created_at']) + 3600 and
         lease['cleanup_due_at'] == request['lease_deadline_utc'] and
         stamp(config['deadline_utc']) <= stamp(lease['cleanup_due_at']) - 600 and
         0 < stamp(config['deadline_utc']) - stamp(request['created_at']) <= 900, 'One-hour lease and 600-second reserve')
    if passed:
        need(terminal['status'] == 'complete' and type(terminal['exit_code']) is int and terminal['exit_code'] == 0 and
             terminal['runner_exited'] is True and stamp(terminal['finished_at']) <= stamp(config['deadline_utc']),
             'Completed external dispatch within recorded deadline')
    return dict(request_sha256=digest, terminal_sha256=sha(root/'stage-terminal.json'),
                pid_record_sha256=sha(root/'stage-pid.json'), helper_sha256=request['helper_sha256'],
                pod_id=request['pod_id'], lease_plan_sha256=request['lease_plan_sha256'],
                lease_deadline_utc=request['lease_deadline_utc'], stage_deadline_utc=config['deadline_utc'],
                limitation='Lease plan hash is a producer-verified receipt; its original standalone bytes are retained outside this run.')


def numeric_outputs(result, report, data):
    shape = list(next(iter(data['windows'].values()))['target'].shape)
    initial = data['initial']
    specifications = {k: {'shape': list(v.shape), 'dtype': 'F32'} for k, v in initial.items()}
    def velocity(name):
        return tensors(result / (name + '.safetensors'), {'velocity': {'shape': shape, 'dtype': 'F32'}})['velocity']
    native = {k: velocity('parity/native-' + k) for k in ('positive', 'negative')}
    expected_rows = [(context, arm, mode) for context in ('positive', 'negative')
                     for mode in ('full', 'cached') for arm in ARMS]
    need(len(report['parity']) == 24, 'Exactly 24 parity records')
    for row, (context, arm, mode) in zip(report['parity'], expected_rows):
        need((row['context'], row['arm'], row['path']) == (context, arm, mode) and
             row['exact_equal'] is True and row['max_abs'] == 0 and
             row['native_sha256'] == row['actual_sha256'] == tensor_sha(native[context]), 'Exact saved zero-parity hash record')
        seconds(row['seconds'])
        if mode == 'cached':
            seconds(row['feature_extract_seconds'])
    saved0 = tensors(result / 'controller-initial.safetensors', specifications)
    final = tensors(result / 'controller-final.safetensors', specifications)
    need(all(tensor_sha(saved0[n]) == tensor_sha(initial[n]) for n in initial), 'Exact original saved controller initialization')
    bnames = [n for n in initial if n.endswith('.b.weight')]
    need(bnames and all(not np.count_nonzero(initial[n]) for n in bnames), 'Zero B initialization')
    need(any(tensor_sha(final[n]) != tensor_sha(initial[n]) for n in initial), 'Controller values changed')
    need(report['completed_updates'] == 2 and len(report['updates']) == 2, 'Exactly two completed updates')
    summaries = []
    for index, update in enumerate(report['updates']):
        row = data['plan']['schedule'][index]
        noise = data['draws'][row['noise_key']]
        root = f"update-{index+1}/"
        need(update['update'] == index + 1 and update['optimizer_updates'] == 1 and update['main_predictions'] == 2 and
             update['all_controller_gradients_present_finite'] is True and update['foundation_gradients_absent'] is True,
             'Update identity and producer gradient ownership')
        losses = []
        for saved, arm in zip(update['main'], row['branches']):
            need(saved['arm'] == arm, 'Ordered main arms')
            value = velocity(root + 'main-' + arm)
            target = noise - data['windows'][arm]['target']
            delta = value[:, :, 1:] - target[:, :, 1:]
            mse = float(np.mean(np.square(delta), dtype=np.float64))
            close(mse, saved['future_flow_mse'], 'Main future loss')
            losses.append(mse)
        need(len(update['main']) == 2, 'Two main loss records')
        aux = update['auxiliary']
        aux_loss = 0.
        if row['auxiliary_edge'] is not None:
            need(aux['enabled'] is True and aux['edge'] == row['auxiliary_edge'] and aux['weight'] == 1. and
                 aux['feature_extracts'] == 2 and aux['predictions'] == 4, 'One four-path auxiliary update')
            p = {(i, label): velocity(root + f'aux-{i}-{label}') for label in ('positive', 'negative') for i in (0, 1)}
            guided = [p[i, 'negative'] + np.float32(5) * (p[i, 'positive'] - p[i, 'negative']) for i in (0, 1)]
            predicted = -(guided[1] - guided[0])
            a, b = row['auxiliary_edge']
            delta = predicted[:, :, 1:] - (data['windows'][b]['target'] - data['windows'][a]['target'])[:, :, 1:]
            aux_loss = float(np.mean(np.square(delta), dtype=np.float64))
        else:
            need(aux == dict(enabled=False, edge=None, weight=1., feature_extracts=0, predictions=0, loss=0.), 'FM-only second update')
        close(aux_loss, aux['loss'], 'Auxiliary future loss')
        total = sum(losses) / 2 + aux_loss
        close(total, update['total_objective'], 'Total objective')
        g = tensors(result / (root + 'gradients.safetensors'), specifications)
        norms = {n: float(np.linalg.norm(x.astype(np.float64).reshape(-1))) for n, x in g.items()}
        norm = math.sqrt(sum(v * v for v in norms.values()))
        s = update['gradients']
        need(s['all_present_finite'] is True and s['parameter_tensors'] == len(initial) and
             set(s['parameters']) == set(initial) and norm > 0, 'Complete finite gradient parameter coverage')
        for n, x in g.items():
            rr = s['parameters'][n]
            need(rr['elements'] == x.size and rr['max_abs'] == float(np.abs(x).max()), 'Gradient elements/max: ' + n)
            close(norms[n], rr['l2'], 'FP64 gradient norm: ' + n, precise=True)
        close(norm, s['l2_after_clip'], 'FP64 total norm', precise=True)
        close(norm, update['gradient_l2_after_clip'], 'FP32 clipped norm')
        need(math.isfinite(update['gradient_l2_before_clip']) and update['gradient_l2_before_clip'] > 0, 'Positive finite unclipped norm record')
        summaries.append(dict(update=index + 1, main_mse=losses, auxiliary_mse=aux_loss,
                              total_objective=total, gradient_l2_recomputed=norm,
                              gradient_tensors=len(g), gradient_file_sha256=sha(result / (root + 'gradients.safetensors')),
                              non_B_gradient_l2=math.sqrt(sum(v*v for n, v in norms.items() if n not in bnames)),
                              gradient_groups_l2={prefix: math.sqrt(sum(v*v for n,v in norms.items() if n.startswith(prefix)))
                                  for prefix in sorted({n.split('.')[0] for n in norms})},
                              profile_seconds=seconds(update['profile_seconds']), main_seconds=seconds(update['main_seconds'])))
    return dict(retained_native_reference_tensors=2, parity_hash_records_checked=24,
                raw_bridged_parity_tensors_retained=0, retained_main_predictions_checked=4,
                retained_auxiliary_predictions_checked=4, gradient_bundles_checked=2, updates=summaries,
                initial_controller_file_sha256=sha(result / 'controller-initial.safetensors'),
                final_controller_file_sha256=sha(result / 'controller-final.safetensors'),
                controller_parameters=sum(x.size for x in initial.values()), controller_parameter_tensors=len(initial),
                controller_change_l2=math.sqrt(sum(float(np.sum((final[n].astype(np.float64)-initial[n])**2)) for n in initial)),
                limitation='24 bridge outputs were discarded after producer comparison; their saved hashes are checked against two retained native tensors. No native forward, backward, optimizer-moment or CUDA replay was performed.')


def run_audit(recovered, original, output, recovery_verified=None):
    output.mkdir(parents=True, exist_ok=False)
    run = path(recovered, RUN)
    result = run / 'result'
    report = dict(schema='worldline-command-attention-profile-independent-v1', status='failed',
                  producer_status='unavailable', checks={}, errors=[], model_execution=False,
                  quality_assessed=False, scope='Saved profile artifacts and recorded contracts only')
    before = inventory(run)
    source_before = {p.name: sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.md')}
    def check(name, callback):
        try:
            value = callback()
            report['checks'][name] = value
            return value
        except Exception as error:
            report['errors'].append(dict(check=name, error_type=type(error).__name__, message=str(error)))
            return None
    pins = read(HERE / 'pins.json')
    def load_original():
        need(sha(HERE / 'original-transfer-manifest.json') == pins['transfer_manifest_sha256'] and
             sha(HERE / 'protocol.json') == pins['protocol_sha256'], 'Auditor frozen reference identity')
        manifest = read(HERE / 'original-transfer-manifest.json')
        return original_inputs(original, manifest['files'], pins)
    data = None
    try:
        data = load_original()
        report['checks']['original_transfer'] = dict(source_files=data['verified_original_source_files'],
            transfer_manifest_sha256=pins['transfer_manifest_sha256'], inputs_sha256=pins['input_manifest_sha256'])
    except Exception as error:
        report['errors'].append(dict(check='original_transfer', error_type=type(error).__name__, message=str(error)))
    config = check('launch_read', lambda: read(run / 'launch.json'))
    worker = check('worker_read', lambda: read(result / 'metrics.json'))
    terminal = check('parent_terminal_read', lambda: read(run / 'terminal.json'))
    # Keep only hashes and status in the public-facing audit, not operational paths.
    for key in ('launch_read', 'worker_read', 'parent_terminal_read'):
        report['checks'].pop(key, None)
    report['identity'] = dict(plan_sha256=sha(run / 'launch.json') if (run/'launch.json').is_file() else None,
                              parent_sha256=sha(run/'terminal.json') if (run/'terminal.json').is_file() else None,
                              worker_sha256=sha(result/'metrics.json') if (result/'metrics.json').is_file() else None,
                              inputs_sha256=pins['input_manifest_sha256'], transfer_manifest_sha256=pins['transfer_manifest_sha256'])
    if worker is not None:
        report['producer_status'] = worker.get('status', 'unavailable')
        report['producer_completed_updates'] = worker.get('completed_updates', 0)
        report['producer_failure'] = {k: worker.get(k) for k in ('error_type', 'error') if k in worker}
    if terminal is not None:
        report['producer_terminal'] = {k: terminal.get(k) for k in
            ('status', 'exit_code', 'error_type', 'error', 'cleanup_error_type', 'cleanup_error', 'elapsed_seconds')}
    for label, p in (('parent_stop', run/'watchdog-stop.json'), ('worker_stop', result/'watchdog-stop.json')):
        if p.is_file():
            check(label, lambda p=p: read(p))
    if data and config and worker:
        check('source_input_binding', lambda: source_bindings(config, worker, data, read(HERE/'protocol.json'), pins))
        if (recovered/'action-results/command-attention-profile-dispatch-v1').exists():
            check('dispatch', lambda: dispatch_binding(recovered, run, config, data, pins, worker.get('status') == 'passed'))
    if worker and worker.get('status') == 'passed':
        def completion():
            need(terminal is not None and terminal['status'] == 'complete' and terminal['exit_code'] == 0, 'Successful parent terminal')
            need(all(worker.get(k) is True for k in ('model_execution', 'zero_gate_passed', 'base_unchanged',
                 'rotary_unchanged', 'all825_current_value_hashes_verified', 'sources_unchanged', 'inputs_unchanged')),
                 'Required producer completion flags')
            need(worker['stage'] == 'complete' and worker['completed_updates'] == 2 and
                 all(worker.get(k) is False for k in ('quality_assessed', 'image_generation', 'automatic_training_promotion')),
                 'Bounded numerical profile scope')
            return check_output_hashes(result, worker)
        check('completion', completion)
        if config:
            check('resources', lambda: check_memory(run, result, worker, config))
        if data:
            check('numerical', lambda: numeric_outputs(result, worker, data))
            def core():
                e = read(original/'repository/experiments/wan22_native/cuda_reference/expected-weights.json')
                return core_records(result, e['tensors'])
            check('core', core)
    else:
        # An unsuccessful producer can still have valuable finite raw arrays.
        def partial():
            rows = {}
            for n, r in before.items():
                if n.endswith('.safetensors'):
                    try:
                        values = tensors(path(run, n))
                        rows[n] = dict(file_sha256=r['sha256'], finite=True,
                                       tensor_sha256={k: tensor_sha(v) for k, v in values.items()})
                    except Exception as error:
                        rows[n] = dict(file_sha256=r['sha256'], finite_or_complete=False, error_type=type(error).__name__)
            return dict(retained_tensor_files=rows, parent_status=terminal.get('status') if terminal else None,
                        meaning='Failure/partial bytes retained. Completion and model-call correctness are not certified.')
        check('partial_evidence', partial)
        if data and (result/'weight-load.json').is_file():
            check('available_core_records', lambda: core_records(result,
                read(original/'repository/experiments/wan22_native/cuda_reference/expected-weights.json')['tensors'], complete=False))
    if recovery_verified is not None:
        def recovery():
            v = read(recovery_verified)
            index_path = recovery_verified.parent/'index.json'
            need(v['status'] == 'verified' and sha(index_path) == v['index_sha256'], 'Verified original recovery index identity')
            idx = read(index_path)
            need(idx['stream_sha256'] == v['stream_sha256'], 'Original recovery stream identity')
            need(idx['original_transfer']['manifest_sha256'] == pins['transfer_manifest_sha256'] and
                 idx['original_transfer']['inputs_manifest_sha256'] == pins['input_manifest_sha256'], 'Recovery original-transfer binding')
            need(idx['input_comparison']['original_inventory_and_bytes_unchanged'] is True,
                 'Runtime source/input bytes differ from original transfer at recovery')
            for n, r in before.items():
                need(idx['files'][RUN+'/'+n] == r, 'Recovered run file map binding')
            return dict(verification_sha256=sha(recovery_verified), index_sha256=v['index_sha256'],
                        stream_sha256=v['stream_sha256'], profile_files=len(before))
        check('recovery', recovery)
    check('immutable_artifacts', lambda: need(inventory(run) == before, 'Run files changed during CPU audit') or True)
    report['audit_sources_unchanged'] = source_before == {p.name: sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md')}
    report['source_sha256'] = source_before
    report['retained_inventory'] = before
    if not report['errors'] and worker and worker.get('status') == 'passed' and report['audit_sources_unchanged']:
        report['status'] = 'passed'
    report['saved_failure_evidence_checked'] = report['producer_status'] != 'passed' and not report['errors']
    (output/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovered-root', type=Path, required=True)
    p.add_argument('--original-transfer', type=Path, required=True,
                   help='Verified extracted original transfer root containing repository/, profile/, training/, controller/, inputs/')
    p.add_argument('--recovery-verified', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = run_audit(a.recovered_root.resolve(), a.original_transfer.resolve(), a.output.resolve(), a.recovery_verified)
    print(json.dumps(dict(status=report['status'], producer_status=report['producer_status'], errors=report['errors'],
                         report_sha256=sha(a.output/'report.json'))))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
