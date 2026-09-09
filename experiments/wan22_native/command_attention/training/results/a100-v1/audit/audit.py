"""Read saved fixed512 evidence without Torch, model calls or pickle loading."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
import numpy as np
from arrays import inventory, need, parse, path, read, record, sha, tensor_sha, tensors
import profile_reference as profile

HERE = Path(__file__).resolve().parent
RUN = 'action-results/command-attention-training-v1'
PROFILE = 'action-results/command-attention-profile-v1'
SCHEMA = 'worldline-command-attention-training-v1'
SCOPE = 'fixed512-command-attention-seven-edge-fitting-only'
INPUT_SHA = '4f37fa59791d7b0645529e4b97978f08d928baf8987e79a9aad6da1b42f3046a'
TRANSFER_SHA = '4dfb2da33228ce58dd56646bb818fd93e6bf9455b685986d7bfc294e3e7cb932'
ARMS = profile.ARMS
EDGES = ((ARMS[0], ARMS[1]), (ARMS[2], ARMS[3]), (ARMS[4], ARMS[5]),
         (ARMS[0], ARMS[2]), (ARMS[0], ARMS[4]), (ARMS[1], ARMS[3]), (ARMS[1], ARMS[5]))
RAW = (1, 2, 5, 509, 512)
CHECKPOINTS = (0, 128, 256, 384, 512)
LIMITS = dict(profile.LIMITS, seconds=1800.)
close = profile.close
seconds = profile.seconds


def stamp(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    need(d.tzinfo is not None and d.utcoffset().total_seconds() == 0, 'Explicit UTC timestamp')
    return d.timestamp()


def load_inputs(original):
    need(sha(HERE/'original-transfer-manifest.json') == TRANSFER_SHA, 'Frozen original transfer manifest')
    transfer = read(HERE/'original-transfer-manifest.json')['files']
    data = profile.original_inputs(original, transfer, {'input_manifest_sha256': INPUT_SHA})
    plan = data['plan']
    need(len(plan['schedule']) == 512 and plan['edges'] == [list(e) for e in EDGES], 'Exactly512 schedule and seven edges')
    retained = {}
    # Stream one original shard at a time. Never regenerate Gaussian draws.
    for name, spec in plan['files'].items():
        p = path(original/'inputs', name)
        need(record(p) == {k:spec[k] for k in ('bytes', 'sha256')} == transfer['inputs/'+name], 'Exact original input file: '+name)
        if name.startswith('draws/'):
            values = tensors(p, spec['tensors'])
            for row in plan['schedule']:
                if row['noise_key'] not in values:
                    continue
                i = row['update'] - 1
                need(type(i) is int and 0 <= i < 512 and row == plan['schedule'][i], 'Ordered draw number')
                need(type(row['k']) is int and 50 <= row['k'] <= 950 and row['sigma'] == row['k']/1000 and
                     row['noise_key'] == f'noise_{i:04d}' and
                     row['branches'] == list(ARMS[2*(i%3):2*(i%3)+2]) and
                     row['auxiliary_edge'] == (list(EDGES[(i//4)%7]) if i%4 == 0 else None), 'Fixed noise/timestep/branch/edge schedule')
                need(tensor_sha(values[row['noise_key']]) == row['noise_sha256'] and
                     tensor_sha(values[f'rng_after_{i:04d}']) == row['rng_after_sha256'], 'Saved draw/RNG hashes')
                if i+1 in RAW:
                    retained[row['noise_key']] = values[row['noise_key']]
            del values
    need(len(retained) == 5, 'All five retained-update noise inputs')
    need(all(not np.any(v) for n,v in data['initial'].items() if n.endswith('.b.weight')), 'Zero B initialization')
    data['draws'] = retained
    data['evaluation'] = tensors(original/'inputs/evaluation-noises.safetensors', plan['files']['evaluation-noises.safetensors']['tensors'])
    freeze = read(HERE/'producer-freeze.json')
    need(sha(HERE/'protocol.json') == freeze['files']['protocol.json'], 'Pinned training protocol')
    data['training_sources'] = freeze['source_sha256']
    need(len(data['training_sources']) == 66, 'Exact frozen training source graph')
    for name, digest in data['training_sources'].items():
        need(transfer[name]['sha256'] == digest and sha(path(original, name)) == digest, 'Frozen training dependency: '+name)
    return data


def scalar_rows(worker, schedule, complete):
    rows = worker.get('updates', [])
    n = worker.get('completed_updates', 0)
    need(type(n) is int and 0 <= n <= 512 and len(rows) == n and (not complete or n == 512), 'Completed scalar row count')
    totals = dict(main_predictions=0, auxiliary_updates=0, auxiliary_predictions=0, auxiliary_feature_extracts=0)
    clipped = 0
    for i, r in enumerate(rows):
        s = schedule[i]
        need(r['schedule'] == s and r['update'] == i+1 and r['main_predictions'] == 2 and r['optimizer_updates'] == 1,
             'Exact scalar/schedule update binding')
        need(r['all_controller_gradients_present_finite'] is True and r['foundation_gradients_absent'] is True,
             'Recorded gradient ownership')
        need([v['arm'] for v in r['main']] == s['branches'], 'Two ordered main branches')
        losses = [seconds(v['future_flow_mse']) for v in r['main']]
        aux = r['auxiliary']; enabled = s['auxiliary_edge'] is not None
        need(aux['enabled'] is enabled and aux['edge'] == s['auxiliary_edge'] and aux['weight'] == 1. and
             aux['feature_extracts'] == 2*enabled and aux['predictions'] == 4*enabled, 'Declared auxiliary update contract')
        value = seconds(aux['loss'])
        need(enabled or value == 0., 'Disabled auxiliary loss is zero')
        close(sum(losses)/2+value, r['total_objective'], 'Scalar objective algebra', precise=True)
        pre, post = seconds(r['gradient_l2_before_clip']), seconds(r['gradient_l2_after_clip'])
        need(pre > 0 and post > 0 and post <= 1.00001 and post <= pre+1e-6, 'Positive clipped gradient norms')
        close(pre*min(1., 1./(pre+1e-6)), post, 'Recorded clip relation')
        clipped += pre > 1.
        need(seconds(r['main_seconds']) <= seconds(r['seconds']) <= 1800, 'Main/update duration ordering')
        totals['main_predictions'] += 2; totals['auxiliary_updates'] += enabled
        totals['auxiliary_predictions'] += 4*enabled; totals['auxiliary_feature_extracts'] += 2*enabled
    for k,v in totals.items():
        need(worker.get(k, 0) == v, 'Accumulated count: '+k)
    return dict(completed_updates=n, scalar_rows=len(rows), recorded_clipped_updates=clipped, **totals,
                limitation='Scalar algebra and saved schedule are checked for every completed row; gradients for other updates were not retained.')


def cfg(p, n):
    return n + np.float32(5)*(p-n)


def contrast(ap, an, bp, bn):
    return -(cfg(bp,bn)-cfg(ap,an))


def score(predicted, target):
    a, b = predicted[:,:,1:].astype(np.float64), target[:,:,1:].astype(np.float64)
    energy, magnitude = float(np.mean(b*b)), float(np.mean(a*a))
    mse = float(np.mean((a-b)**2))
    return dict(future_mse=mse, target_mean_square=energy, prediction_mean_square=magnitude,
                normalized_mse=mse/energy if energy>0 else None,
                cosine=float(np.sum(a*b)/math.sqrt(np.sum(a*a)*np.sum(b*b))) if energy>0 and magnitude>0 else None,
                target_rms=math.sqrt(energy), prediction_rms=math.sqrt(magnitude), target_informative=energy>0)


def numeric(result, worker, data, complete):
    shape = list(data['windows'][ARMS[0]]['target'].shape)
    specs = {n:dict(shape=list(v.shape), dtype='F32') for n,v in data['initial'].items()}
    def velocity(name):
        return tensors(result/(name+'.safetensors'), {'velocity':dict(shape=shape,dtype='F32')})['velocity']
    evaluations = {}; checked_files = []
    for phase in ('initial','final'):
        saved = worker.get(phase+'_evaluation')
        if saved is None and not complete:
            continue
        need(saved['label'] == phase and saved['predictions'] == 48 and saved['feature_extracts'] == 8 and
             saved['ideal_sigma'] == 1. and saved['native_time'] == 999 and saved['model_quality_assessed'] is False and
             saved['checkpoint_selection'] is False and len(saved['scores']) == 28, 'Fixed evaluation contract')
        seconds(saved['seconds']); rows=[]
        for k, noise in enumerate(sorted(data['evaluation'])):
            values = {}
            for arm in ARMS:
                for context in ('positive','negative'):
                    name=f'evaluation/{phase}/{noise}/{arm}-{context}'
                    values[arm,context] = velocity(name); checked_files.append(name+'.safetensors')
            for j,(a,b) in enumerate(EDGES):
                item = saved['scores'][k*7+j]
                need(item['noise'] == noise and item['edge'] == [a,b], 'Ordered evaluation noise/edge')
                actual = score(contrast(values[a,'positive'],values[a,'negative'],values[b,'positive'],values[b,'negative']),
                               data['windows'][b]['target']-data['windows'][a]['target'])
                for key,value in actual.items():
                    if value is None or type(value) is bool:
                        need(item[key] is value, 'Evaluation informative/zero-energy field')
                    else:
                        close(value,item[key],phase+' '+key,precise=True)
                rows.append(dict(noise=noise,edge=[a,b],**actual))
            values.clear()
        evaluations[phase] = dict(predictions=48, scores=rows)
    raw=[]
    for i in RAW:
        if i > worker.get('completed_updates',0) and not complete:
            continue
        r=worker['updates'][i-1]; row=data['plan']['schedule'][i-1]; base=f'raw/update-{i:04d}'
        noise=data['draws'][row['noise_key']]; losses=[]
        for j,a in enumerate(row['branches']):
            name=base+'/main-'+a; p=velocity(name); checked_files.append(name+'.safetensors')
            # FP32 subtraction matches the label and residual; FP64 sums bound CPU reduction error.
            residual=(p-(noise-data['windows'][a]['target']))[:,:,1:]
            value=float(np.mean(np.square(residual,dtype=np.float32),dtype=np.float64))
            close(value,r['main'][j]['future_flow_mse'],'Retained main MSE'); losses.append(value)
        auxiliary=None
        if row['auxiliary_edge'] is not None:
            values={}
            for a in (0,1):
                for c in ('positive','negative'):
                    name=f'{base}/aux-{a}-{c}'; values[a,c]=velocity(name); checked_files.append(name+'.safetensors')
            a,b=row['auxiliary_edge']
            prediction=contrast(values[0,'positive'],values[0,'negative'],values[1,'positive'],values[1,'negative'])
            residual=(prediction-(data['windows'][b]['target']-data['windows'][a]['target']))[:,:,1:]
            auxiliary=float(np.mean(np.square(residual,dtype=np.float32),dtype=np.float64))
            close(auxiliary,r['auxiliary']['loss'],'Retained auxiliary MSE')
        gradients=tensors(result/base/'gradients.safetensors',specs)
        norm=math.sqrt(sum(float(np.sum(v.astype(np.float64)**2)) for v in gradients.values()))
        close(norm,r['gradient_l2_after_clip'],'Retained post-clip gradient L2')
        raw.append(dict(update=i,main_future_mse=losses,auxiliary_future_mse=auxiliary,post_clip_l2=norm,
                        parameter_tensors=len(gradients),gradient_elements=sum(v.size for v in gradients.values())))
    return dict(evaluations=evaluations, retained_updates=raw, raw_prediction_files=len(checked_files),
                raw_gradient_bundles=len(raw),
                limitation='Saved FP32 predictions and combined post-clip gradients only. No backward graph, unretained gradients, optimizer update or CUDA model replay.')


def checkpoints(result, worker, plan, plan_sha, admission_sha, data, complete):
    wanted = dict(schema=SCHEMA,scope=SCOPE,plan_sha256=plan_sha,source_sha256=plan['source_sha256'],inputs_sha256=INPUT_SHA,
                  controller_parameters=4_936_448,admission_sha256=admission_sha,
                  note='Reused checkpoint helper calls the new controller file adapter.safetensors; original residual adapter is absent.')
    specs={n:dict(shape=list(v.shape),dtype='F32') for n,v in data['initial'].items()}
    rows=[]
    available={p.name for p in result.glob('checkpoint-*') if p.is_dir()}
    expected={f'checkpoint-{i:04d}' for i in CHECKPOINTS if i <= worker.get('completed_updates',0)}
    need(available == expected if complete else available <= {f'checkpoint-{i:04d}' for i in CHECKPOINTS}, 'Only declared checkpoint directories')
    for i in CHECKPOINTS:
        p=result/f'checkpoint-{i:04d}'
        if not p.exists() and not complete:
            continue
        m=read(p/'manifest.json')
        need(m['schema']=='worldline-wan22-action-checkpoint-v1' and m['completed_updates']==i and m['identity']==wanted and
             m['external_core_weights_included'] is False and m['resume_supported'] is False,
             'Checkpoint schema, source/input/admission identity')
        need(set(m['files']) == {'adapter.safetensors','optimizer-and-rng.pt'}, 'Exact checkpoint file names')
        for name,digest in m['files'].items():
            need(record(path(p,name))['sha256']==digest,'Checkpoint binary SHA')
        values=tensors(p/'adapter.safetensors',specs)
        actual={n:dict(shape=list(v.shape),dtype='float32',sha256=tensor_sha(v)) for n,v in values.items()}
        need(actual == m['tensors'],'Checkpoint complete tensor hashes')
        if i == 0:
            need(all(tensor_sha(values[n])==tensor_sha(v) for n,v in data['initial'].items()),'Original saved initial controller values')
        rng=tensors(result/f'cuda-rng-{i:04d}.safetensors')
        need(set(rng)=={'rng'} and rng['rng'].dtype==np.uint8 and rng['rng'].ndim==1 and 0<rng['rng'].size<=2**20,'Retained CUDA RNG byte tensor')
        rows.append(dict(completed_updates=i,manifest_sha256=sha(p/'manifest.json'),checkpoint_sha256=m['files']['adapter.safetensors'],
                         optimizer_and_rng_sha256=m['files']['optimizer-and-rng.pt'],cuda_rng_sha256=sha(result/f'cuda-rng-{i:04d}.safetensors'),
                         tensor_count=len(values),parameters=sum(v.size for v in values.values())))
    if rows:
        last=rows[-1]; pointer=dict(directory=f"checkpoint-{last['completed_updates']:04d}",manifest_sha256=last['manifest_sha256'],completed_updates=last['completed_updates'])
        need(read(result/'last-valid.json') == pointer and worker.get('last_checkpoint') == pointer,'Last valid checkpoint pointer')
    return dict(checkpoints=rows,optimizer_payloads_unpickled=False,
                limitation='Optimizer/CPU RNG pickle bytes are hash-verified against each manifest but are not loaded. CUDA RNG byte tensors and controller tensors are read. No AdamW or RNG-state semantic replay.')


def bindings(recovered, run, parent, worker, data, complete):
    plan=read(run/'executed-plan.json'); config=read(run/'launch.json'); admission=read(run/'executed-admission.json')
    plan_sha=sha(run/'executed-plan.json'); admitted=dict(sha256=sha(run/'executed-admission.json'),record=admission)
    need(plan['schema']==SCHEMA and plan['scope']==SCOPE and plan['status']=='prepared' and
         plan['protocol']==read(HERE/'protocol.json') and plan['inputs_sha256']==INPUT_SHA and
         plan['source_sha256']==data['training_sources'] and plan['input_schedule']==data['plan']['schedule'] and
         plan['input_initial_controller_sha256']==data['plan']['files']['initial-controller.safetensors']['sha256'], 'Exact executed source/input/protocol/schedule')
    actual_sources={n:r['sha256'] for n,r in inventory(run/'source').items()}
    need(actual_sources == plan['source_sha256'], 'Complete executed source snapshots')
    expected=dict(schema=SCHEMA,approved=True,scope=SCOPE,plan_sha256=plan_sha,inputs_sha256=INPUT_SHA,
                  source_sha256=plan['source_sha256'],resource_profile_sha256=plan['resource_profile']['result_sha256'],
                  expected_gpu=plan['expected_gpu'],lease_deadline_utc=plan['lease_deadline_utc'],recovery_reserve_seconds=600)
    need(all(admission.get(k)==v for k,v in expected.items()),'Exact source/input/profile/lease admission')
    need(config['schema']==SCHEMA and config['plan_sha256']==plan_sha and config['admission_record']==admitted and
         config['lease_deadline_utc']==plan['lease_deadline_utc'] and config['output']=='/workspace/'+RUN and
         math.isfinite(config['deadline']), 'Worker launch/admission binding')
    common=dict(plan_sha256=plan_sha,inputs_sha256=INPUT_SHA,source_sha256=plan['source_sha256'],admission=admitted,
                lease_deadline_utc=plan['lease_deadline_utc'])
    for label,r in (('parent',parent),('worker',worker)):
        need(r['schema']==SCHEMA and r['scope']==SCOPE and r['limits']==LIMITS,'Recorded training schema and limits')
        if complete or 'plan_sha256' in r:
            need(all(r.get(k)==v for k,v in common.items()),label+' exact identities')
    if complete or 'input_identity' in worker:
        identity=dict(windows={a:{k:tensor_sha(v) for k,v in w.items()} for a,w in data['windows'].items()},
                      positive=tensor_sha(data['contexts']['positive']),negative=tensor_sha(data['contexts']['negative']),
                      evaluation={k:tensor_sha(v) for k,v in data['evaluation'].items()})
        need(worker['input_identity']==identity and worker['resource_profile']==plan['resource_profile'] and
             worker['precision']==profile.PRECISION,'Exact worker input/profile/precision')
    if complete:
        need(parent['resource_profile_sha256']==plan['resource_profile']['result_sha256'] and
             parent['projection']==plan['resource_profile']['projection'],'Parent profiled forecast')
    return plan,plan_sha,admitted


def resource_profile(recovered, plan, data):
    p=path(recovered,PROFILE); r=read(p/'result/metrics.json'); launch=read(p/'launch.json')
    t=read(p/'terminal.json'); monitor=read(p/'result/monitor-terminal.json'); receipt=plan['resource_profile']
    need(r['status']=='passed' and r['completed_updates']==2 and all(r[k] is True for k in
         ('model_execution','zero_gate_passed','base_unchanged','all825_current_value_hashes_verified','rotary_unchanged','sources_unchanged','inputs_unchanged')),
         'Actual passed profile gates')
    need(r['source_sha256']==launch['source_sha256']==data['source_map'] and r['inputs_sha256']==INPUT_SHA and
         r['initial_controller_sha256']==plan['input_initial_controller_sha256'], 'Actual profile source/input/controller')
    for label,file in (('result_sha256','result/metrics.json'),('terminal_sha256','terminal.json'),
                       ('monitor_sha256','result/monitor-terminal.json'),('launch_sha256','launch.json')):
        need(receipt[label]==sha(p/file),'Actual profile receipt file hash')
    need(t['status']=='complete' and t['exit_code']==0 and t.get('cleanup_error') is None and
         monitor['status']=='complete' and monitor.get('error') is None and not list(p.rglob('watchdog-stop.json')),'Actual profile terminals')
    profile.check_output_hashes(p/'result',r)
    expected={(c,a,mode) for c in ('positive','negative') for a in ARMS for mode in ('full','cached')}
    need(len(r['parity'])==24 and {(v['context'],v['arm'],v['path']) for v in r['parity']}==expected and
         all(v['exact_equal'] is True and v['native_sha256']==v['actual_sha256'] for v in r['parity']), '24 exact recorded zero comparisons')
    for c in ('positive','negative'):
        v=tensors(p/f'result/parity/native-{c}.safetensors',{'velocity':dict(shape=data['plan']['shape'],dtype='F32')})['velocity']
        need(all(row['native_sha256']==tensor_sha(v) for row in r['parity'] if row['context']==c),'Retained native parity hash')
    updates=r['updates']; cached=[v for v in r['parity'] if v['path']=='cached']
    need(len(updates)==2 and [v['update'] for v in updates]==[1,2] and
         [v['auxiliary']['predictions'] for v in updates]==[4,0] and
         all(v['main_predictions']==2 and v['optimizer_updates']==1 and v['gradients']['all_present_finite'] is True for v in updates), 'Profile mixed and FM rows')
    components=dict(fm_step=max(updates[1]['seconds'],updates[1]['profile_seconds']),
        mixed_step=max(updates[0]['seconds'],updates[0]['profile_seconds']),
        prefix=max(v['feature_extract_seconds'] for v in cached),head=max(v['seconds'] for v in cached),
        load=r['load_seconds'],hash_before=r['foundation_hash_seconds']['before'],hash_after=r['foundation_hash_seconds']['after-updates'])
    need(all(seconds(v)>0 for v in components.values()),'Positive finite measured timing components')
    projected=1.2*(384*components['fm_step']+128*components['mixed_step']+16*components['prefix']+96*components['head']+
                   components['load']+components['hash_before']+components['hash_after'])+120
    need(0<projected<=1800 and receipt['projection']['components']==components,'Projection admission and components')
    close(projected,receipt['projection']['seconds'],'Profile projection',precise=True)
    for key in ('source_sha256','hardware','runtime_flags','initial_controller_sha256'):
        need(receipt[key]==r[key],'Profile receipt metadata: '+key)
    return dict(projection_seconds=projected,components=components,zero_parity_records=24,
                profile_sha256=sha(p/'result/metrics.json'),
                limitation='Actual retained profile files and zero-output hash records checked. Discarded bridged outputs and backward calls cannot be replayed here.')


def core(result, original, complete):
    expected=read(original/'repository/experiments/wan22_native/cuda_reference/expected-weights.json')['tensors']
    loaded=read(result/'weight-load.json')
    need(loaded['tensor_count']==825 and loaded['parameter_count']==4_999_787_712 and loaded['parameter_bytes']==19_999_150_848 and
         loaded['convert_model_dtype'] is False and loaded['all_shards_verified'] is True and loaded['cuda_copy_exact'] is True and
         set(loaded['tensors'])==set(expected), 'Original FP32 native loader')
    wanted={}
    for n,e in expected.items():
        r=loaded['tensors'][n]
        need(r['shape']==e['shape'] and r['shard']==e['shard'] and r['original_dtype']==r['loaded_dtype']=='float32' and
             r['source_sha256']==r['loaded_sha256']==e['original_sha256'] and r['cuda_copy_exact'] is True and
             r['source_owner_released'] is True,'Original foundation weight record')
        wanted[n]=dict(shape=e['shape'],dtype='float32',sha256=e['original_sha256'])
    result_hash={}
    for label in ('before','after'):
        p=result/f'core-{label}.json'
        if not p.exists() and not complete:continue
        need(read(p)==wanted,'All825 original current-value hashes '+label); result_hash[label]=sha(p)
    return dict(records_per_phase=825,phases=result_hash,
                limitation='Saved current-value records match pinned original values. Foundation tensors are not recovered, so no new model hash pass is performed.')


def resources(run, result, worker, parent, plan):
    h=worker['hardware']; old=plan['resource_profile']['hardware']
    need(set(h)==set(old),'Hardware field set')
    for k in h:
        if k=='total_memory_bytes':
            need(type(h[k]) is int and type(old[k]) is int and h[k]>=max(old[k],70*2**30),'Equal or greater GPU capacity')
        else:need(json.dumps(h[k],sort_keys=True)==json.dumps(old[k],sort_keys=True),'Exact profiled hardware/runtime: '+k)
    need(worker['runtime_flags']==plan['resource_profile']['runtime_flags'],'Unchanged native runtime flags')
    samples={}
    for label,p in (('parent',run/'parent-memory.jsonl'),('worker',result/'memory.jsonl')):
        need(record(p)['bytes']<=8*2**20,'Bounded memory record')
        rows=[parse(s) for s in p.read_bytes().splitlines() if s]; previous=-1.; mismatches=0
        need(rows,'Recorded memory samples')
        for r in rows:
            now=seconds(r['seconds']); need(previous<=now<1800,'Memory chronology/cap'); previous=now
            rss='combined_rss_bytes' if label=='parent' else 'host_rss_bytes'
            keys=(rss,'host_available_bytes') + (() if label=='parent' else ('cuda_reserved_bytes','cuda_allocated_bytes','cuda_available_bytes'))
            need(all(type(r[k]) is int and r[k]>=0 for k in keys),'Memory byte types')
            need(r[rss]<=48*2**30 and r['host_available_bytes']>=8*2**30,'Original host memory caps')
            if label=='worker':
                need(r['cuda_reserved_bytes']<=60*2**30 and r['cuda_available_bytes']>=8*2**30,'Original GPU memory caps')
                mismatches += r['cuda_allocated_bytes']>r['cuda_reserved_bytes']
        samples[label]=dict(count=len(rows),last_seconds=previous,sequential_allocated_above_reserved_rows=mismatches)
    terminal=read(run/'terminal.json'); monitor=read(result/'monitor-terminal.json')
    for r in (terminal,monitor):
        need(r['status']=='complete' and r['mode']=='clip' and r['limits']==LIMITS and seconds(r['elapsed_seconds'])<=1800,
             'Passed original time/memory guard terminal')
    need(terminal['exit_code']==0 and terminal.get('cleanup_error') is None and monitor.get('error') is None and
         monitor['sample_count']==samples['worker']['count'],'Guard exit/sample evidence')
    for r in (worker,parent):need(seconds(r['elapsed_seconds'])<=1800,'Recorded elapsed cap')
    for value in worker['foundation_hash_seconds'].values():seconds(value)
    seconds(worker['load_seconds']); seconds(worker['training_and_evaluation_seconds'])
    need(all(type(v) is int and v>=0 for v in worker['memory'].values()) and worker['memory']['peak_reserved_bytes']<=60*2**30,'Training peak reserved cap')
    return dict(hardware=h,samples=samples,parent_seconds=parent['elapsed_seconds'],worker_seconds=worker['elapsed_seconds'],
                training_and_evaluation_seconds=worker['training_and_evaluation_seconds'],
                limitation='Sampled counters do not establish every instant. Allocated/reserved counters are sequential reads and have no additional cross-counter inequality gate.')


def dispatch(recovered, run, plan, plan_sha, data, complete):
    d=path(recovered,'action-results/command-attention-training-dispatch-v1'); p=path(recovered,'action-results/command-attention-profile-dispatch-v1')
    request=read(d/'request.json'); terminal=read(d/'stage-terminal.json'); pid=read(d/'stage-pid.json'); prior=read(p/'request.json')
    digest=sha(d/'request.json'); lease=request['lease_plan']
    need(request['schema']=='worldline-command-attention-stage-dispatch-v1' and request['stage']=='training' and
         request['manifest_sha256']==TRANSFER_SHA and request['inputs_sha256']==INPUT_SHA and
         request['source_sha256']==data['transfer_source_sha256'] and request['training_plan']==plan and request['training_plan_sha256']==plan_sha and
         sha(d/'training-plan.json')==plan_sha and sha(d/'training-admission.json')==sha(run/'executed-admission.json'), 'External exact training request')
    need(request['profile_dispatch_sha256']=={n:sha(p/n) for n in ('request.json','stage-terminal.json','stage-pid.json')} and
         request['profile_sha256']==plan['resource_profile']['result_sha256'],'Exact preceding profile dispatch')
    need(all(request[k]==prior[k] for k in ('pod_id','lease_plan_sha256','lease_deadline_utc','manifest_sha256','source_sha256')),
         'Training and profile use the same immutable Pod lease')
    need(type(pid['pid']) is int and pid['pid']>0 and terminal['pid']==pid['pid'] and
         terminal['request_sha256']==pid['request_sha256']==digest,'External terminal/PID/request')
    need(sha(path(recovered,'dispatch_command_attention.py'))==request['helper_sha256'],'Executed dispatch source')
    need(lease['schema']=='exact-pod-cleanup-plan-v1' and lease['pod_id']==request['pod_id'] and lease['expected_network_volume'] is None and
         stamp(lease['cleanup_due_at'])==stamp(lease['created_at'])+3600 and
         lease['cleanup_due_at']==request['lease_deadline_utc']==plan['lease_deadline_utc'],'Immutable one-hour lease')
    projected=seconds(request['projection_seconds']); remaining=stamp(lease['cleanup_due_at'])-stamp(request['created_at'])
    need(request['recovery_reserve_seconds']==600 and 0<projected<=1800 and remaining<=3600 and
         remaining>=projected+600 and projected==plan['resource_profile']['projection']['seconds'],'Recorded forecast and 600-second reserve admission')
    files={n:sha(run/n) for n in ('terminal.json','metrics.json','launch.json','result/metrics.json') if (run/n).is_file()}
    need(terminal['runner_records_sha256']==files,'External terminal binds final runner files')
    if complete:
        need(terminal['status']=='complete' and terminal['exit_code']==0 and terminal['runner_exited'] is True and
             stamp(terminal['finished_at'])<=stamp(lease['cleanup_due_at'])-600,'Completed dispatcher before recovery reserve')
    return dict(request_sha256=digest,terminal_sha256=sha(d/'stage-terminal.json'),pod_id=request['pod_id'],
                lease_deadline_utc=request['lease_deadline_utc'],lease_plan_sha256=request['lease_plan_sha256'],
                limitation='The standalone cleanup-plan hash is a producer-verified receipt. This audit does not query provider lifecycle state.')


def run_audit(recovered, original, output, recovery_verified=None):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); run=path(recovered,RUN); result=run/'result'
    report=dict(schema='worldline-command-attention-training-independent-v1',status='failed',checks={},errors=[],
                producer_status='unavailable',model_execution=False,quality_assessed=False,
                scope='Saved fixed512 records and retained arrays; no model, backward, optimizer or omitted-gradient replay')
    source_before={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*')
                   if p.is_file() and p.suffix in ('.py','.json','.md') and not any(x.startswith('cpu-') for x in p.relative_to(HERE).parts)}
    def check(label, fn):
        try:
            value=fn(); report['checks'][label]=value; return value
        except Exception as e:
            report['errors'].append(dict(check=label,error_type=type(e).__name__,message=str(e))); return None
    before=check('inventory',lambda:inventory(run))
    data=check('original_inputs',lambda:load_inputs(original))
    if data is not None:
        report['checks']['original_inputs']=dict(manifest_sha256=INPUT_SHA,original_transfer_sha256=TRANSFER_SHA,
            original_source_files=data['verified_original_source_files'],input_files=len(data['plan']['files']),
            checked_draws=512,training_source_files=len(data['training_sources']),noise_regenerated=False)
    parent=check('parent_read',lambda:read(run/'metrics.json')); worker=check('worker_read',lambda:read(result/'metrics.json'))
    report['checks'].pop('parent_read',None); report['checks'].pop('worker_read',None)
    complete=bool(parent and worker and parent.get('status')==worker.get('status')=='passed')
    report['producer_status']=worker.get('status','unavailable') if worker else 'unavailable'
    report['producer_parent_status']=parent.get('status','unavailable') if parent else 'unavailable'
    report['completed_updates']=worker.get('completed_updates',0) if worker else 0
    report['producer_failure']={label:{k:r[k] for k in ('status','error_type','error','model_execution') if k in r}
                               for label,r in (('parent',parent),('worker',worker)) if r}
    report['identity']=dict(inputs_sha256=INPUT_SHA,transfer_manifest_sha256=TRANSFER_SHA,
        parent_sha256=sha(run/'metrics.json') if (run/'metrics.json').is_file() else None,
        worker_sha256=sha(result/'metrics.json') if (result/'metrics.json').is_file() else None,
        terminal_sha256=sha(run/'terminal.json') if (run/'terminal.json').is_file() else None,
        plan_sha256=sha(run/'executed-plan.json') if (run/'executed-plan.json').is_file() else None)
    bound=None
    if data and parent and worker and (run/'executed-plan.json').is_file():
        bound=check('source_input_admission',lambda:bindings(recovered,run,parent,worker,data,complete))
        if bound:
            plan,plan_sha,admitted=bound
            report['checks']['source_input_admission']=dict(source_files=len(plan['source_sha256']),plan_sha256=plan_sha,admission_sha256=admitted['sha256'])
            report['identity'].update(source_sha256=plan['source_sha256'],admission_sha256=admitted['sha256'],
                                      profile_sha256=plan['resource_profile']['result_sha256'])
            check('resource_profile',lambda:resource_profile(recovered,plan,data))
            check('dispatch_and_lease',lambda:dispatch(recovered,run,plan,plan_sha,data,complete))
            cp=check('checkpoints',lambda:checkpoints(result,worker,plan,plan_sha,admitted['sha256'],data,complete))
            if cp and cp['checkpoints'] and cp['checkpoints'][-1]['completed_updates']==512:
                last=cp['checkpoints'][-1]
                report['identity'].update(final_checkpoint_sha256=last['checkpoint_sha256'],final_checkpoint_manifest_sha256=last['manifest_sha256'])
    if data and worker:
        check('scalar_rows',lambda:scalar_rows(worker,data['plan']['schedule'],complete))
        check('retained_numerical',lambda:numeric(result,worker,data,complete))
        if (result/'weight-load.json').is_file():
            check('foundation_records',lambda:core(result,original,complete))
    if complete:
        def completion():
            need(worker['protocol']==read(HERE/'protocol.json') and
                 all(worker[k] is True for k in ('model_execution','base_unchanged','all825_current_value_hashes_verified','rotary_unchanged',
                    'sources_unchanged','inputs_unchanged','final_checkpoint_only','all512_scalar_records')) and
                 worker['raw_updates']==list(RAW) and worker['checkpoints']==list(CHECKPOINTS) and
                 worker['automatic_promotion'] is False and worker['quality_assessed'] is False and
                 parent['image_generation'] is False and parent['model_execution'] is True and
                 parent['result_metrics_sha256']==sha(result/'metrics.json') and parent['terminal_sha256']==sha(run/'terminal.json') and
                 not list(run.rglob('watchdog-stop.json')),'Producer completion, terminal hashes and bounded scope')
            return profile.check_output_hashes(result,worker)
        check('completion',completion)
        if bound:check('resources',lambda:resources(run,result,worker,parent,bound[0]))
    else:
        def partial():
            rows={}
            for name,r in (before or {}).items():
                if name.endswith('.safetensors'):
                    try:
                        value=tensors(path(run,name))
                        rows[name]=dict(complete_finite_arrays=True,file_sha256=r['sha256'],tensor_sha256={k:tensor_sha(v) for k,v in value.items()})
                    except Exception as e:
                        rows[name]=dict(complete_finite_arrays=False,file_sha256=r['sha256'],error_type=type(e).__name__)
            return dict(retained_tensors=rows,meaning='Partial and malformed files remain in the inventory; no successful512 result is certified.')
        check('partial_evidence',partial)
    if recovery_verified:
        def recovery():
            v=read(recovery_verified); ip=recovery_verified.parent/'index.json'; idx=read(ip)
            need(v['status']=='verified' and sha(ip)==v['index_sha256'] and idx['stream_sha256']==v['stream_sha256'],'Verified original recovery index/stream')
            need(idx['original_transfer']['manifest_sha256']==TRANSFER_SHA and idx['original_transfer']['inputs_manifest_sha256']==INPUT_SHA and
                 idx['input_comparison']['original_inventory_and_bytes_unchanged'] is True,'Recovery original source/input bytes')
            for name,r in (before or {}).items():need(idx['files'][RUN+'/'+name]==r,'Recovery saved training file map')
            return dict(verification_sha256=sha(recovery_verified),index_sha256=v['index_sha256'],stream_sha256=v['stream_sha256'])
        check('recovery',recovery)
    check('immutable_artifacts',lambda:need(inventory(run)==before,'Training artifacts changed during CPU audit') or True)
    after={n:sha(HERE/n) for n in source_before}
    need(source_before==after,'Auditor source changed')
    report['source_sha256']=source_before; report['retained_inventory']=before
    report['audit_sources_unchanged']=True; report['elapsed_seconds']=time.monotonic()-started
    report['status']='passed' if complete and not report['errors'] else 'failed'
    report['partial_evidence_checked']=not complete and bool(before)
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovered-root',type=Path,required=True)
    p.add_argument('--original-transfer',type=Path,required=True,help='Verified original extraction with inputs/, repository/, training/, controller/, profile/')
    p.add_argument('--recovery-verified',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); r=run_audit(a.recovered_root.resolve(),a.original_transfer.resolve(),a.output.resolve(),a.recovery_verified)
    print(json.dumps(dict(status=r['status'],producer_status=r['producer_status'],completed_updates=r['completed_updates'],
                          errors=r['errors'],report_sha256=sha(a.output/'report.json'))))
    return 0 if r['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
