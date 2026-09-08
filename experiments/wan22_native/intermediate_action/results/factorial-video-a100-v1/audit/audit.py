"""Six-video saved-artifact audit. NumPy/Pillow only; no model or solver replay."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import helpers as h
import rgb

HERE = Path(__file__).resolve().parent
PLAN_SHA = '07dc80a0daf69134e1985dc4649ad08f207fa6c2f85430cab23def82095d0a12'
CHECKPOINT_SHA = '9147ef7a53a01c4399e7073cab97a4ccdc7c8d1195305333560871eeff004dba'
SHAPE = (48, 5, 44, 78)
ARMS = tuple(m+'_'+d for m in ('stationary','left','right') for d in ('closed','interact'))
LIMITS = dict(seconds=1800.0,host_rss_bytes=48*2**30,cuda_reserved_bytes=60*2**30,
              minimum_host_available_bytes=8*2**30,minimum_cuda_available_bytes=8*2**30,
              minimum_gpu_total_bytes=70*2**30)
WORKER_LIMITS = dict(LIMITS,seconds=900.0)
need,read,sha,tensor_sha = h.need,h.read,h.sha,h.tensor_sha

def references():
    pins=read(HERE/'pins.json')
    for n,d in pins['reference_sha256'].items():need(sha(rgb._file(HERE/'reference',n))==d,'frozen audit reference '+n)
    for n in ('rgb','helpers'):need(sha(HERE/(n+'.py'))==pins[n+'_sha256'],'unchanged reused '+n)
    need(sha(HERE/'reference/plan.json')==PLAN_SHA,'exact reviewed actual plan')
    return read(HERE/'reference/plan.json')

def checked_map(base,rows):
    need(isinstance(rows,dict) and rows,'nonempty file hash map')
    for n,d in rows.items():need(sha(rgb._file(base,n))==rgb._digest(d),'file hash '+n)

def recovery_binding(root,path,inventory):
    record=read(path);index_path=path.parent/'index.json';index=read(index_path)
    need(record['status']=='verified' and record['index_sha256']==sha(index_path),'verified original recovery index')
    recovered=path.parent/'recovered'
    need(root.is_relative_to(recovered),'run belongs to exact original recovery')
    prefix=root.relative_to(recovered).as_posix()+'/'
    subset={n[len(prefix):]:r for n,r in index['files'].items() if n.startswith(prefix)}
    need(subset==inventory,'every run file matches original verified recovery inventory')
    return record

def schedule():
    # Literal pinned scheduler initialization and set_timesteps scalar equations.
    # This checks integer times, not UniPC state transitions.
    sigmas=(1.-np.linspace(1.,1/1000,1000)[::-1].copy()).astype(np.float32)
    raw=np.linspace(float(sigmas[0]),float(sigmas[-1]),51)[:-1]
    return (5*raw/(1+4*raw)*1000).astype(np.int64).tolist()

def commands_expected(arm):
    motion,door=arm.split('_');value=np.zeros((1,16,6),np.float32)
    value[0,:,3]=np.float32({'stationary':0.,'left':math.pi/120,'right':-math.pi/120}[motion])
    value[0,0,5]=int(door=='interact');return value

def conditions(root,plan):
    values=h.array_file(root/'sampling-inputs.safetensors',{'initial_noise':(SHAPE,'F32'),
        'initial_latent':(SHAPE,'F32'),'observation':((1,48,1,44,78),'F32'),'token_times':((1,4290),'I64')})
    contexts=h.array_file(root/'contexts.safetensors',{'atrium':((25,4096),'F32'),'native_negative':((126,4096),'F32')})
    commands=h.array_file(root/'commands.safetensors',{a:((1,16,6),'F32') for a in ARMS})
    restored=values['initial_noise'].copy();restored[:,:1]=values['observation'][0]
    need(restored.tobytes()==values['initial_latent'].tobytes(),'exact independent observation restored into shared noise')
    expected=np.full((1,4290),999,np.int64);expected[:,:858]=0
    need(np.array_equal(values['token_times'],expected),'exact observed zero/future999 token times')
    record=dict(values={k:tensor_sha(v) for k,v in values.items()},contexts={k:tensor_sha(v) for k,v in contexts.items()},
                commands={k:tensor_sha(v) for k,v in commands.items()})
    receipt=read(root/'fixed-receipt.json')
    need(record==receipt['tensor_identity'],'all actual fixed tensor identities')
    need(receipt['future_targets_materialized'] is False and receipt['fresh_noise_drawn'] is False,'fixed conditioning scope')
    checked_map(root,receipt['artifacts'])
    for arm in ARMS:
        need(np.array_equal(commands[arm],commands_expected(arm)),'destination-aligned commands '+arm)
        trained=plan['training_details']['input_identity']['windows'][arm]
        need(record['commands'][arm]==trained['commands'] and record['values']['observation']==trained['observation'],
             'exact trained commands and independent observation '+arm)
    return values,record

def prepared(root,plan):
    need(sha(root/'plan.json')==PLAN_SHA and read(root/'plan.json')==plan,'exact actual prepared plan')
    for scope,rows in plan['source_sha256'].items():checked_map(root/'source'/scope,rows)
    checked_map(root,plan['artifacts']);checked_map(root/'evidence/training',plan['evidence_sha256'])
    need(plan['artifacts']['adapter.safetensors']==CHECKPOINT_SHA==plan['training_identity']['final_checkpoint_sha256'],
         'exact final checkpoint only')
    cpu=read(root/'cpu-report.json');audit=read(root/'training-audit.json')
    need(cpu['status']=='passed' and cpu['tests']==17 and cpu['source_sha256']==plan['source_sha256']
         and cpu['sources_unchanged'] is True and all(type(cpu[k]) is int and cpu[k]==0 for k in ('pytest_exit_code','failures','errors','skipped')),
         'source-bound complete17 CPU review')
    need(audit['schema']=='worldline-factorial-intermediate128-actual-audit-v1' and audit['status']=='passed'
         and audit['identity']==plan['training_identity'] and audit['completed_updates']==128
         and audit['all825_unchanged'] is True and audit['final_checkpoint_only'] is True,'actual independent training audit')
    review=read(HERE/'reference/preparation-review.json')
    need(review['status']=='passed' and review['plan_sha256']==PLAN_SHA and review['source_sha256']==plan['source_sha256']
         and review['training_identity']==plan['training_identity'],'actual preparation review binding')
    manifest=read(root/'evidence/training/result/checkpoint-0128/manifest.json')
    adapter=h.array_file(root/'adapter.safetensors',{n:(tuple(r['shape']),'F32') for n,r in manifest['tensors'].items()},maximum=4*2**20)
    need(sum(v.size for v in adapter.values())==947712,'original947712 parameter adapter')
    hashes={k:tensor_sha(v) for k,v in adapter.items()}
    need(hashes=={k:v['sha256'] for k,v in manifest['tensors'].items()}==plan['training_details']['adapter']['tensor_sha256'],
         'final adapter raw tensor hashes')
    del adapter

def worker_sample(row):
    for k in ('host_rss_bytes','cuda_reserved_bytes','host_available_bytes','cuda_available_bytes','cuda_allocated_bytes'):
        need(type(row[k]) is int and row[k]>=0,'nonnegative integer resource sample '+k)
    need(row['host_rss_bytes']<=LIMITS['host_rss_bytes'] and row['cuda_reserved_bytes']<=LIMITS['cuda_reserved_bytes']
         and row['host_available_bytes']>=LIMITS['minimum_host_available_bytes']
         and row['cuda_available_bytes']>=LIMITS['minimum_cuda_available_bytes'],'declared sampled memory caps')
    # CUDA reserved/allocated counters are read sequentially; no invented relation.

def hardware(actual,trained,recorded):
    c='total_memory_bytes'
    need(set(actual)==set(trained) and type(actual[c]) is int and type(trained[c]) is int
         and actual[c]>=trained[c]>=LIMITS['minimum_gpu_total_bytes'],'GPU total equal or greater only')
    other=lambda v:json.dumps({k:x for k,x in v.items() if k!=c},sort_keys=True,allow_nan=False)
    need(other(actual)==other(trained),'all other runtime fields/types unchanged')
    need(recorded==dict(policy='All fields exact except reported total GPU bytes may be greater',
         training_total_memory_bytes=trained[c],actual_total_memory_bytes=actual[c],additional_reported_bytes=actual[c]-trained[c]),
         'recorded hardware comparison')

def completion(root,plan):
    parent=read(root/'metrics.json');term=read(root/'terminal.json');decision=read(root/'executed-admission.json')
    need(parent['schema']==plan['schema'] and parent['status']=='passed' and parent['model_execution'] is True
         and parent['model_frozen'] is True and parent['training'] is False and parent['quality_assessed'] is False
         and parent['native_control'] is False and parent['predictions']==600 and parent['solver_updates']==300
         and parent['frames_per_arm']==17 and parent['limits']==LIMITS and parent['worker_limits']==WORKER_LIMITS,
         'complete bounded six-video parent')
    need(parent['plan_sha256']==PLAN_SHA and parent['source_sha256']==plan['source_sha256'],'parent source and plan')
    need(term['status']=='complete' and type(term['exit_code']) is int and term['exit_code']==0
         and term['cleanup_error'] is None and term['plan_sha256']==PLAN_SHA and parent['terminal_sha256']==sha(root/'terminal.json')
         and 0<term['elapsed_seconds']==parent['elapsed_seconds']<1800,'complete exact parent terminal')
    required=dict(schema=plan['schema'],scope=plan['protocol']['scope'],decision='admit',issued_by='parent-agent',
        plan_sha256=PLAN_SHA,source_sha256=plan['source_sha256'],training_identity=plan['training_identity'],
        training_audit_sha256=plan['training_audit_sha256'],cpu_report_sha256=plan['cpu_report_sha256'],
        protocol=plan['protocol'],minimum_lease_remaining_seconds=2400,native_control=False,training_admitted=False)
    need(all(decision.get(k)==v for k,v in required.items()) and bool(decision['reason'].strip()),'exact actual video admission')
    admitted=dict(sha256=sha(root/'executed-admission.json'),scope=plan['protocol']['scope'],lease_deadline_utc=decision['lease_deadline_utc'])
    need(parent['admission']==admitted,'executed admission bytes')
    deadline=datetime.fromisoformat(admitted['lease_deadline_utc'].replace('Z','+00:00'))
    finish=datetime.fromisoformat(term['finished_at_utc'].replace('Z','+00:00'))
    remaining=(deadline-finish).total_seconds()
    need(deadline.tzinfo is not None and finish.tzinfo is not None and remaining>600
         and remaining+parent['elapsed_seconds']>2400,'recorded admission and final recovery reserve')
    need(not list(root.rglob('watchdog-stop.json')) and not list(root.rglob('*cleanup-error.json')),'no watchdog/cleanup failure')
    rows={};resources={};launches={}
    for stage in ('core','decode'):
        folder=root/stage;result=folder/'result';r=read(result/'metrics.json');t=read(folder/'terminal.json')
        m=read(result/'monitor-terminal.json');launch=read(folder/'launch.json');launches[stage]=launch
        need(r['status']=='passed' and r['stage']==stage and r['schema']==plan['schema']
             and r['model_execution'] is True and r['training'] is False and r['future_targets_materialized'] is False
             and r['inputs_unchanged'] is True and r['sources_unchanged'] is True,'complete worker scope')
        need(r['plan_sha256']==PLAN_SHA and r['source_sha256']==plan['source_sha256'] and r['admission']==admitted
             and r['combined_limits']==LIMITS,'worker identity and common cap')
        need(t['status']==m['status']=='complete' and type(t['exit_code']) is int and t['exit_code']==0
             and t['cleanup_error'] is None and t['error'] is None and m['error'] is None,'worker supervisor and monitor complete')
        for row in (r,t,m):need(row['limits']==WORKER_LIMITS and 0<row['elapsed_seconds']<900,'worker900s bound')
        need(parent['child_reports'][stage]==sha(result/'metrics.json') and parent['child_terminals'][stage]==sha(folder/'terminal.json'),
             'parent child report hashes')
        need(launch['stage']==stage and launch['plan_sha256']==PLAN_SHA and launch['admission']==admitted
             and launch['deadline']<=launch['parent_deadline'],'launch exact plan/admission/deadline')
        outputs=r['output_sha256'];checked_map(result,outputs)
        actual={str(p.relative_to(result)) for p in result.rglob('*') if p.is_file() and str(p.relative_to(result)) not in ('metrics.json','memory.jsonl')}
        need(set(outputs)==actual,'complete final output inventory')
        hardware(r['hardware'],plan['training_details']['hardware'],r['hardware_comparison'])
        need(r['runtime_flags']==read(root/'evidence/training/result/metrics.json')['runtime_flags'],
             'same retained native runtime precision flags as training')
        samples=h.jsonlines(result/'memory.jsonl');parents=h.jsonlines(folder/'parent-memory.jsonl')
        need(samples and parents and len(samples)==m['sample_count'],'complete resource samples')
        for s in samples:worker_sample(s)
        for s in parents:
            need(type(s['combined_rss_bytes']) is int and 0<=s['combined_rss_bytes']<=LIMITS['host_rss_bytes']
                 and type(s['host_available_bytes']) is int and s['host_available_bytes']>=LIMITS['minimum_host_available_bytes'],
                 'sampled combined host caps')
        need(max(s['combined_rss_bytes'] for s in parents)==t['peak_combined_rss_bytes']
             and min(s['host_available_bytes'] for s in parents)==t['minimum_host_available_bytes'],'supervisor resource extrema')
        resources[stage]=dict(worker_seconds=r['elapsed_seconds'],load_seconds=r['load_seconds'],
            worker_samples=len(samples),parent_samples=len(parents),peak_combined_rss_bytes=t['peak_combined_rss_bytes'],
            peak_cuda_reserved_bytes=max(s['cuda_reserved_bytes'] for s in samples),
            minimum_cuda_available_bytes=min(s['cuda_available_bytes'] for s in samples),
            sequential_cuda_counter_mismatches=sum(s['cuda_allocated_bytes']>s['cuda_reserved_bytes'] for s in samples),
            worker_log_sha256=sha(rgb._file(folder,'worker.log')))
        rows[stage]=r
    need(launches['core']['parent_deadline']==launches['decode']['parent_deadline'],'one common parent deadline')
    need(launches['decode']['core_metrics_sha256']==sha(root/'core/result/metrics.json'),'serialized decoder handoff')
    return rows,dict(parent_seconds=parent['elapsed_seconds'],final_recorded_lease_remaining_seconds=remaining,stages=resources)

def weights(root,plan,rows):
    base=root/'core/result';before=read(base/'core-before.json');after=read(base/'core-after.json');loaded=read(base/'weight-load.json')
    catalog=read(HERE/'reference/expected-weights.json')['tensors']
    need(before==after==plan['training_details']['core_records'] and len(before)==825,'all825 original core values unchanged')
    need(set(loaded['tensors'])==set(before)==set(catalog) and loaded['tensor_count']==825
         and loaded['parameter_count']==4999787712 and loaded['convert_model_dtype'] is False
         and loaded['all_shards_verified'] is True and loaded['cuda_copy_exact'] is True,'complete originalFP32 CUDA load')
    for n,r in catalog.items():
        need(before[n]==dict(shape=r['shape'],dtype='float32',sha256=r['original_sha256']),'original core identity '+n)
        v=loaded['tensors'][n]
        need(v['shape']==r['shape'] and v['source_sha256']==v['loaded_sha256']==r['original_sha256']
             and v['original_dtype']==v['loaded_dtype']=='float32' and v['source_owner_released'] is True
             and v['cuda_copy_exact'] is True,'exact source/copy/release record '+n)
    need(rows['core']['model_frozen'] is True and rows['core']['rotary_copy_exact'] is True
         and rows['core']['block_index']==28,'reported unchanged core, adapter and original rotary')
    vae=read(root/'decode/result/weight-load.json');reference=read(HERE/'reference/vae-load-reference.json')
    need(vae==reference and len(vae['tensors'])==196 and vae['parameters']==704688668 and vae['compute_dtype']=='float32'
         and vae['weight_sha256']==read(HERE/'reference/codec-source.json')['weight_sha256'],'original196-tensor FP32 codec load')
    expected={n:dict(shape=r['shape'],dtype='float32',sha256=r['sha256']) for n,r in vae['tensors'].items()}
    need(read(root/'decode/result/codec-before.json')==read(root/'decode/result/codec-after.json')==expected
         and all(r['cuda_copy_exact'] is True for r in vae['tensors'].values()),'all196 original codec values unchanged')
    need(rows['decode']['all196_unchanged'] is True and rows['decode']['normalization_unchanged'] is True
         and set(rows['decode']['normalization_sha256'])=={'0','1'},'reported unchanged native normalization')
    for d in rows['decode']['normalization_sha256'].values():rgb._digest(d)
    return dict(core_tensors=825,core_parameters=4999787712,vae_tensors=196,vae_parameters=704688668,
                adapter_parameters=947712,checkpoint_sha256=CHECKPOINT_SHA,
                value_hash_checks='Retained complete before/after hashes and loader records; no weight/model reloading.')

def state(directory,number,step,observation,shape=SHAPE):
    latent=h.array_file(directory/f'step-{number:02d}.safetensors',{'latent':(shape,'F32')})['latent']
    need(step['step']==number and step['timestep']==schedule()[number-1] and step['prefix_exact'] is True
         and step['latent_sha256']==tensor_sha(latent),'retained state order/time/tensor hash')
    need(np.ascontiguousarray(latent[:,:1]).tobytes()==observation.tobytes(),'exact saved independent prefix')
    return latent

def guided(velocities):
    expected=velocities['negative_velocity']+np.float32(5)*(velocities['positive_velocity']-velocities['negative_velocity'])
    need(expected.tobytes()==velocities['guided_velocity'].tobytes(),'exact first-update FP32 CFG arithmetic')

def arm_contract(row,arm,identity,adapter):
    need(row['arm']==row['commands_label']==arm and row['profile']=='spatial' and row['block_index']==28
         and row['predictions']==row['clean_prefix_calls']==100 and row['solver_updates']==50
         and row['settings']==dict(steps=50,shift=5.,guidance=5.),'complete declared arm count/settings')
    need(row['input_tensor_sha256']==identity['values'] and row['text_tensor_sha256']==identity['contexts']
         and row['commands_sha256']==identity['commands'][arm] and row['adapter']==adapter,'same fixed inputs and final adapter per arm')
    need(row['commands_on_cfg_branches']==['positive','native_negative'] and row['negative_context_adapter_training'] is True
         and row['negative_context_training_scope']=='Every fourth factorial paired update uses both CFG contexts; main FM uses positive text only'
         and row['training'] is False and row['quality_assessed'] is False and row['foundation_values_verified_by_this_kernel'] is False,
         'adapter commands on both CFG branches and actual training scope')

def arms(root,plan,rows,values,identity):
    need(set(rows['core']['arms'])==set(rows['decode']['arms'])==set(ARMS),'all six arms exactly')
    need(rows['core']['predictions']==600 and rows['core']['solver_updates']==300,'aggregate source-bound counters')
    answer={}
    for arm in ARMS:
        folder=root/'core/result'/arm;row=read(folder/'metrics.json')
        need(row==rows['core']['arms'][arm],'per-arm report binding')
        arm_contract(row,arm,identity,plan['training_details']['adapter'])
        steps=h.jsonlines(folder/'steps.jsonl')
        need(len(steps)==50 and {p.name for p in folder.glob('step-*.safetensors')}=={f'step-{i:02d}.safetensors' for i in range(1,51)},
             'all50 exact state files')
        hashes=[];last_seconds=-1
        for i,step in enumerate(steps,1):
            latent=state(folder,i,step,values['observation'][0]);hashes.append(tensor_sha(latent))
            need(last_seconds<=step['seconds']<=row['seconds'],'chronological step timing');last_seconds=step['seconds']
        final=h.array_file(folder/'latents.safetensors',{'latent':(SHAPE,'F32')})['latent']
        need(final.tobytes()==latent.tobytes() and tensor_sha(final)==row['final_latent_sha256'],'final exactly retained step50')
        velocities=h.array_file(folder/'initial-velocities.safetensors',{k:(SHAPE,'F32') for k in ('positive_velocity','negative_velocity','guided_velocity')})
        guided(velocities)
        decoded=rows['decode']['arms'][arm]
        need(decoded['latent_sha256']==tensor_sha(final) and decoded['cache_clear'] is True,'decoder exact final latent and cache-clear record')
        answer[arm]=dict(prediction_calls_recorded=100,states_checked=50,state_tensor_sha256=hashes,
            final_latent_sha256=tensor_sha(final),initial_velocity_tensor_sha256={k:tensor_sha(v) for k,v in velocities.items()},
            sampling_seconds=row['seconds'],rgb=h.check_rgb(root/'decode/result',arm,rows['decode']))
    return answer

def audit(root,recovery,out):
    began=time.monotonic();before=h.inventory(root);report=dict(schema='worldline-factorial-video-artifact-audit-v1',status='running',
        model_execution=False,model_replay=False,torch_imported='torch' in sys.modules,checks={})
    try:
        need('torch' not in sys.modules,'no Torch/model import')
        recovered=recovery_binding(root,recovery,before)
        plan=references();prepared(root,plan);values,identity=conditions(root,plan)
        report['checks']['conditions']=identity
        rows,resources=completion(root,plan);report['checks']['completion_resources']=resources
        report['checks']['weights']=weights(root,plan,rows)
        report['checks']['arms']=arms(root,plan,rows,values,identity)
        need(h.inventory(root)==before,'all recovered files unchanged by audit')
        report.update(status='passed',identity=dict(plan_sha256=PLAN_SHA,parent_sha256=sha(root/'metrics.json'),
            core_sha256=sha(root/'core/result/metrics.json'),decode_sha256=sha(root/'decode/result/metrics.json'),
            checkpoint_sha256=CHECKPOINT_SHA,admission_sha256=sha(root/'executed-admission.json'),
            training_audit_sha256=plan['training_audit_sha256'],source_sha256=plan['source_sha256']),
            arms=6,prediction_calls_recorded=600,raw_prediction_tensors_checked=12,raw_guided_velocity_tensors_checked=6,
            saved_solver_states_checked=300,raw_rgb_frames_checked=102,png_frames_checked=102,
            recovered_files=len(before),recovered_bytes=sum(r['bytes'] for r in before.values()),recovered_bytes_unchanged=True,
            recovery_verification_sha256=sha(recovery),limitations=[
                'Only the first positive/negative/guided velocity per arm is retained. The600 prediction calls are source-bound execution counters, not600 raw tensors independently checked.',
                'All300 retained solver states have checked hashes, finite FP32 values, order, integer times and clean prefixes. Final states match decoder inputs. Omitted velocities prevent independent UniPC trajectory replay.',
                'Initial guidance arithmetic is independently checked. No denoiser, VAE, gradient, codec-normalization or model output computation is replayed.',
                'Complete core/VAE before-and-after value hashes and unchanged adapter/rotary/normalization reports are checked. In-memory values cannot be independently remeasured after process completion.',
                'RawRGB/PNG/contact-frame pixel identity is distinct from visual quality, camera motion, door interaction or generalization. Scoring is separate.',
                'Resource caps hold at retained sampling instants. Reported lease timing does not establish provider deletion or a billing ceiling.',
                'Each17-frame clip includes one conditioned reconstruction and16 generated future frames. GIF playback duration is distinct from generation speed and palette colors are not tested for exact equality.'])
        report['recovery_index_sha256']=recovered['index_sha256']
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-began
        (out/'inventory.json').write_text(json.dumps(before,indent=2)+'\n')
        report['inventory_sha256']=sha(out/'inventory.json')
        report['audit_source_sha256']={n:sha(HERE/n) for n in ('audit.py','helpers.py','rgb.py','pins.json')}
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=report['status'],report_sha256=sha(out/'report.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('run-root','recovery-verified','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();root=a.run_root.resolve();out=a.output.resolve()
    need(root.is_dir() and not out.exists() and not out.is_relative_to(root),'fresh output outside recovered artifacts')
    out.mkdir(parents=True)
    for n in ('audit.py','helpers.py','rgb.py','pins.json'):shutil.copyfile(HERE/n,out/n)
    shutil.copytree(HERE/'reference',out/'reference')
    audit(root,a.recovery_verified.resolve(),out)
