# SPDX-License-Identifier: Apache-2.0
"""Explicit 512-update fitting run. Default command prints a model-free plan."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
SCHEMA='worldline-command-attention-training-v1'
SCOPE='fixed512-command-attention-seven-edge-fitting-only'
INPUT_SHA='4f37fa59791d7b0645529e4b97978f08d928baf8987e79a9aad6da1b42f3046a'
CHECKPOINTS=(0,128,256,384,512)
RAW_UPDATES=(1,2,5,509,512)
ARMS=tuple(m+'_'+d for m in ('stationary','left','right') for d in ('closed','interact'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def _pairs(values):
    d={}
    for k,v in values:
        if k in d:raise ValueError('Duplicate JSON key')
        d[k]=v
    return d


def read(path,maximum=16*2**20):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=maximum:raise ValueError('Bounded regular JSON required')
    value=json.loads(path.read_text(),object_pairs_hook=_pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(value,dict):raise ValueError('JSON object required')
    return value


def regular(root,name):
    root=Path(root).resolve();rel=PurePosixPath(name)
    if not name or rel.is_absolute() or '..' in rel.parts or str(rel)!=name:raise ValueError('Relative artifact required')
    p=root
    for part in rel.parts:
        p=p/part
        if p.is_symlink():raise ValueError('Artifact links are forbidden')
    if not p.is_file() or not p.resolve().is_relative_to(root):raise ValueError('Artifact missing or escaped')
    return p


def utc_seconds(deadline,now=None):
    d=datetime.fromisoformat(deadline.replace('Z','+00:00'))
    if d.tzinfo is None or d.utcoffset().total_seconds()!=0:raise ValueError('Explicit UTC lease deadline required')
    return (d-(now or datetime.now(timezone.utc))).total_seconds()


def lease_budget(deadline,projected,now=None):
    remaining=utc_seconds(deadline,now)
    if (isinstance(projected,bool) or not isinstance(projected,(int,float)) or not math.isfinite(projected)
            or not 0<projected<=1800 or not 600<remaining<=3600 or remaining<projected+600):
        raise ValueError('Projected training must fit 1800 seconds and the fixed lease with 600 seconds for recovery')
    return min(1800.,remaining-600.)


def protocol():
    return dict(schema=SCHEMA,scope=SCOPE,updates=512,controller_parameters=4_936_448,
        blocks=list(range(24,30)),rank=32,initialization_seed=2026090803,
        checkpoints=list(CHECKPOINTS),raw_updates=list(RAW_UPDATES),final_checkpoint_only=True,
        main_predictions=1024,auxiliary_updates=128,auxiliary_predictions=512,auxiliary_feature_extracts=256,
        evaluation=dict(initial_predictions=48,final_predictions=48,noises=4,arms=list(ARMS),contexts=['positive','negative'],
            time=999,ideal_sigma=1.,future_only=True,raw_payload_bytes=316293120),
        guard=dict(reused_mode='clip',purpose='Training time and memory limits only; no image generation',seconds=1800,recovery_reserve_seconds=600),
        optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=.01),
        retention_limit='Only five declared gradient/prediction updates are retained; omitted training gradients/velocities cannot be independently replayed.',
        automatic_sampling=False,automatic_checkpoint_selection=False,quality_assessed=False)


def bootstrap(repository,controller_source):
    sys.path[:0]=[str(HERE),str(Path(controller_source).resolve()),str(Path(repository).resolve())]
    import torch,packet,math_steps,controller,bridge
    for m,p in ((packet,HERE/'packet.py'),(math_steps,HERE/'math_steps.py'),(controller,Path(controller_source)/'controller.py'),(bridge,Path(controller_source)/'bridge.py')):
        if Path(m.__file__).resolve()!=p.resolve():raise ValueError('An imported module shadows its bound source')
    torch.set_num_threads(1)
    return torch,packet,math_steps,controller,bridge


def source_paths(repository,controller_source):
    bootstrap(repository,controller_source)
    from experiments.wan22_native.action_cuda.probe_evidence import source_paths as prior
    repo=Path(repository).resolve()
    result={'repository/'+n:p for n,p in prior().items()}
    for n in ('spatial_reference/guards.py','cuda_reference/guards.py','action_training/objective.py'):
        result['repository/experiments/wan22_native/'+n]=repo/'experiments/wan22_native'/n
    result.update({'training/'+n:HERE/n for n in ('run_training.py','packet.py','math_steps.py','prepare_inputs.py')})
    result.update({'controller/'+n:Path(controller_source).resolve()/n for n in ('controller.py','bridge.py')})
    if any(not p.is_file() or p.is_symlink() for p in result.values()):raise ValueError('Required source file missing or linked')
    return dict(sorted(result.items()))


def hashes(paths):return {n:sha(p) for n,p in paths.items()}


def profile_receipt(root,expected_sha,current_sources,input_sha):
    root=Path(root).resolve();r=read(root/'result/metrics.json');t=read(root/'terminal.json');m=read(root/'result/monitor-terminal.json');launch=read(root/'launch.json')
    if sha(root/'result/metrics.json')!=expected_sha:raise ValueError('Exact resource profile SHA required')
    if (r.get('schema')!='worldline-command-attention-profile-v1' or r.get('status')!='passed'
            or r.get('model_execution') is not True or r.get('completed_updates')!=2
            or r.get('inputs_sha256')!=input_sha or r.get('zero_gate_passed') is not True
            or r.get('base_unchanged') is not True or r.get('all825_current_value_hashes_verified') is not True
            or r.get('rotary_unchanged') is not True or r.get('sources_unchanged') is not True or r.get('inputs_unchanged') is not True
            or len(r.get('parity',[]))!=24 or any(v.get('exact_equal') is not True for v in r['parity'])
            or t.get('status')!='complete' or t.get('exit_code')!=0 or t.get('cleanup_error') is not None
            or m.get('status')!='complete' or m.get('error') is not None or list(root.rglob('watchdog-stop.json'))):
        raise ValueError('Completed native resource, zero-parity and unchanged-foundation gates required')
    expected_parity={(c,a,p) for c in ('positive','negative') for a in ARMS for p in ('full','cached')}
    if {(v.get('context'),v.get('arm'),v.get('path')) for v in r['parity']}!=expected_parity:raise ValueError('Profile parity coverage differs')
    if r.get('source_sha256')!=launch.get('source_sha256'):raise ValueError('Profile launch/source identity differs')
    for n,v in r['source_sha256'].items():
        current=n if not n.startswith('repository/') else 'repository/experiments/wan22_native/'+n.removeprefix('repository/')
        if n.startswith(('training/','controller/','repository/')) and current_sources.get(current)!=v:
            raise ValueError('Profile used different controller, math, packet or native sources: '+n)
    updates=r.get('updates',[])
    if (len(updates)!=2 or [v.get('update') for v in updates]!=[1,2]
            or [v.get('auxiliary',{}).get('predictions') for v in updates]!=[4,0]
            or any(v.get('optimizer_updates')!=1 or v.get('main_predictions')!=2 or v.get('gradients',{}).get('all_present_finite') is not True for v in updates)):
        raise ValueError('One four-graph mixed update and one FM-only update required')
    outputs=r.get('output_sha256',{})
    required={'weight-load.json','core-before.json','core-after-updates.json','monitor-terminal.json','memory.jsonl'}
    if not required<=set(outputs):raise ValueError('Required profile output identities missing')
    for name,v in outputs.items():
        if sha(regular(root/'result',name))!=v:raise ValueError('Resource profile output changed')
    # No allocated<=reserved assertion: the monitor samples those sequentially.
    from experiments.wan22_native.spatial_reference.guards import limits
    for row in updates:
        memory=row.get('memory',{})
        if type(memory.get('peak_reserved_bytes')) is not int or not 0<memory['peak_reserved_bytes']<=limits('clip')['cuda_reserved_bytes']:
            raise ValueError('Measured peak exceeds fixed memory limit')
    numeric=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0
    cached=[v for v in r['parity'] if v['path']=='cached']
    components=dict(fm_step=max(updates[1]['seconds'],updates[1]['profile_seconds']),
        mixed_step=max(updates[0]['seconds'],updates[0]['profile_seconds']),
        prefix=max(v['feature_extract_seconds'] for v in cached),head=max(v['seconds'] for v in cached),
        load=r['load_seconds'],hash_before=r['foundation_hash_seconds']['before'],hash_after=r['foundation_hash_seconds']['after-updates'])
    if not all(numeric(v) for v in components.values()):raise ValueError('Positive finite measured timings required')
    projected=1.2*(384*components['fm_step']+128*components['mixed_step']+16*components['prefix']+96*components['head']+
                   components['load']+components['hash_before']+components['hash_after'])+120
    return dict(result_sha256=expected_sha,terminal_sha256=sha(root/'terminal.json'),monitor_sha256=sha(root/'result/monitor-terminal.json'),
        launch_sha256=sha(root/'launch.json'),source_sha256=r['source_sha256'],hardware=r['hardware'],runtime_flags=r['runtime_flags'],
        initial_controller_sha256=r['initial_controller_sha256'],projection=dict(seconds=projected,components=components,
        formula='1.2*(384*FM +128*mixed +16*prefix +96*cached_head +load +before_hash +after_hash)+120',
        limitation='Measured two-step extrapolation with 20% time margin and120s I/O/checkpoint allowance; actual deadline and memory guards still apply.'))


def make_plan(args):
    _,packet,_,_,_=bootstrap(args.repository,args.controller_source)
    if args.inputs_sha256!=INPUT_SHA:raise ValueError('Use the exact fixed512 input packet')
    data=packet.Inputs(args.inputs,args.inputs_sha256);data.validate_all()
    paths=source_paths(args.repository,args.controller_source);sources=hashes(paths)
    receipt=profile_receipt(args.resource_profile,args.profile_sha256,sources,args.inputs_sha256)
    if receipt['initial_controller_sha256']!=data.plan['files']['initial-controller.safetensors']['sha256']:raise ValueError('Profile initial controller differs')
    if receipt['hardware']['name']!=args.expected_gpu:raise ValueError('Exact profiled GPU name required')
    lease_budget(args.lease_deadline_utc,receipt['projection']['seconds'])
    return dict(schema=SCHEMA,scope=SCOPE,status='prepared',protocol=protocol(),inputs_sha256=args.inputs_sha256,
        source_sha256=sources,resource_profile=receipt,expected_gpu=args.expected_gpu,lease_deadline_utc=args.lease_deadline_utc,
        input_initial_controller_sha256=receipt['initial_controller_sha256'],input_schedule=data.plan['schedule'],
        execution_paths={n:str(Path(getattr(args,n)).resolve()) for n in ('repository','controller_source','inputs','resource_profile')})


def read_plan(path,expected_sha):
    if sha(path)!=expected_sha:raise ValueError('Exact training plan hash required')
    plan=read(path)
    if plan.get('schema')!=SCHEMA or plan.get('scope')!=SCOPE or plan.get('protocol')!=protocol() or plan.get('inputs_sha256')!=INPUT_SHA:
        raise ValueError('Exact frozen training protocol required')
    paths=plan['execution_paths'];bootstrap(paths['repository'],paths['controller_source'])
    if hashes(source_paths(paths['repository'],paths['controller_source']))!=plan['source_sha256']:raise ValueError('Training sources changed')
    receipt=profile_receipt(paths['resource_profile'],plan['resource_profile']['result_sha256'],plan['source_sha256'],INPUT_SHA)
    if receipt!=plan['resource_profile']:raise ValueError('Resource profile receipt changed')
    import packet
    data=packet.Inputs(paths['inputs'],INPUT_SHA)
    if data.plan['schedule']!=plan['input_schedule'] or data.plan['files']['initial-controller.safetensors']['sha256']!=plan['input_initial_controller_sha256']:
        raise ValueError('Initial controller or all512 scheduled draws differ')
    return plan,data


def admission(path,plan,plan_sha):
    value=read(path,2**20)
    expected=dict(schema=SCHEMA,approved=True,scope=SCOPE,plan_sha256=plan_sha,inputs_sha256=INPUT_SHA,
        source_sha256=plan['source_sha256'],resource_profile_sha256=plan['resource_profile']['result_sha256'],
        expected_gpu=plan['expected_gpu'],lease_deadline_utc=plan['lease_deadline_utc'],recovery_reserve_seconds=600)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Exact source/input/profile/lease-bound admission required')
    return dict(sha256=sha(path),record=value)


def hardware_matches(actual,profiled):
    if set(actual)!=set(profiled):raise ValueError('Hardware/runtime fields differ')
    for key in actual:
        if key=='total_memory_bytes':
            if type(actual[key]) is not int or type(profiled[key]) is not int or actual[key]<profiled[key]:raise ValueError('GPU capacity is lower than profiled')
        elif json.dumps(actual[key],sort_keys=True)!=json.dumps(profiled[key],sort_keys=True):raise ValueError('Hardware/runtime differs from measured profile: '+key)


def score_contrast(predicted,target):
    import torch
    if predicted.shape!=target.shape or predicted.dtype!=torch.float32 or target.dtype!=torch.float32 or predicted.ndim!=5:
        raise ValueError('Aligned FP32 video contrasts required')
    a=predicted[:,:,1:].double();b=target[:,:,1:].double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise FloatingPointError('Nonfinite evaluation')
    energy=float(b.square().mean());actual=float(a.square().mean());mse=float((a-b).square().mean())
    cosine=float((a*b).sum()/(a.square().sum()*b.square().sum()).sqrt()) if energy>0 and actual>0 else None
    return dict(future_mse=mse,target_mean_square=energy,prediction_mean_square=actual,
        normalized_mse=mse/energy if energy>0 else None,cosine=cosine,
        target_rms=math.sqrt(energy),prediction_rms=math.sqrt(actual),target_informative=energy>0,
        definition='FP32 ideal endpoint contrast, then FP64 reductions over four future latent groups only')


def evaluate(bridge,data,label,retain,check=lambda:None,synchronize=lambda:None):
    import torch,math_steps
    device=next(bridge.controller.parameters()).device;results=[];predictions_count=0;extracts=0;started=time.monotonic()
    for key,noise in sorted(data.evaluation.items()):
        values={};w=data.windows[ARMS[0]]
        _,x,t,_=math_steps.endpoint_inputs(data.windows,math_steps.EDGES[0],noise)
        try:
            for context_name,context in (('positive',data.positive),('negative',data.negative)):
                check();features=bridge.extract_features(x.to(device),t.to(device),[context.to(device)]);extracts+=1
                for arm in ARMS:
                    check();window=data.windows[arm]
                    v=bridge.predict_from_features(features,window['commands'].to(device),window['observation'].to(device),track_grad=False)
                    # Retain each completed output before any later score/error.
                    retain(f'evaluation/{label}/{key}/{arm}-{context_name}',dict(velocity=v));predictions_count+=1
                    values[arm,context_name]=v.detach().float().cpu();del v
                del features
            for a,b in math_steps.EDGES:
                predicted=math_steps.contrast(values[a,'positive'],values[a,'negative'],values[b,'positive'],values[b,'negative'])
                target=data.windows[b]['target']-data.windows[a]['target']
                results.append(dict(noise=key,edge=[a,b],**score_contrast(predicted,target)))
            if not torch.equal(x[:,:,:1],w['observation']):raise RuntimeError('Evaluation prefix changed')
        finally:values.clear()
        del x,t
    synchronize()
    if predictions_count!=48 or extracts!=8 or len(results)!=28:raise ValueError('Exactly four noises, six arms and seven edge scores required')
    return dict(label=label,predictions=predictions_count,feature_extracts=extracts,scores=results,seconds=time.monotonic()-started,
        model_quality_assessed=False,checkpoint_selection=False,ideal_sigma=1.,native_time=999)


def train_loop(bridge,data,optimizer,*,retain,checkpoint,progress,check=lambda:None,synchronize=lambda:None,evaluator=evaluate):
    import math_steps
    schedule=data.plan['schedule']
    if len(schedule)!=512 or schedule!=math_steps.schedule([{k:v for k,v in r.items() if k not in ('branches','auxiliary_edge')} for r in schedule]):
        raise ValueError('Exact512 row/edge order required')
    report=dict(completed_updates=0,updates=[],main_predictions=0,auxiliary_updates=0,auxiliary_predictions=0,auxiliary_feature_extracts=0)
    checkpoint(0,None);report['initial_evaluation']=evaluator(bridge,data,'initial',retain,check,synchronize);progress(report)
    for row in schedule:
        check();i=row['update']
        def selected(name,values):
            if i in RAW_UPDATES:retain(f'raw/update-{i:04d}/'+name,values)
        result=math_steps.update(bridge,data.windows,row,data.noise(row),data.positive,data.negative,optimizer,
            check=check,retain=selected,synchronize=synchronize)
        expected_aux=row['auxiliary_edge'] is not None
        if (result.get('update')!=i or result.get('optimizer_updates')!=1 or result.get('main_predictions')!=2
                or result.get('auxiliary',{}).get('predictions')!=(4 if expected_aux else 0)
                or result.get('auxiliary',{}).get('feature_extracts')!=(2 if expected_aux else 0)):
            raise RuntimeError('Training step count contract changed')
        report['updates'].append(dict(result,schedule=row));report['completed_updates']=i
        report['main_predictions']+=2;report['auxiliary_updates']+=int(expected_aux)
        report['auxiliary_predictions']+=4*int(expected_aux);report['auxiliary_feature_extracts']+=2*int(expected_aux)
        if i in CHECKPOINTS:checkpoint(i,row)
        progress(report)
    optimizer.zero_grad(set_to_none=True)
    report['final_evaluation']=evaluator(bridge,data,'final',retain,check,synchronize)
    report.update(final_checkpoint_only=True,all512_scalar_records=True,raw_updates=list(RAW_UPDATES),checkpoints=list(CHECKPOINTS))
    return report


def worker(config):
    plan_path=Path(config['plan']);out=Path(config['output'])/'result';out.mkdir(exist_ok=False)
    paths=read(plan_path)['execution_paths'];torch,_,_,controller_module,bridge_module=bootstrap(paths['repository'],paths['controller_source'])
    from safetensors.torch import save_file
    from experiments.wan22_native.action_cuda.probe import _original_weights,_runtime_flags,PRECISION
    from experiments.wan22_native.action_training.objective import parameter_records,save_checkpoint
    from experiments.wan22_native.cuda_reference import native
    from experiments.wan22_native.official_cpu.streaming import tensor_sha
    from experiments.wan22_native.spatial_reference.guards import atomic,limits,hardware,Monitor,_deadline
    began=time.monotonic();report=dict(schema=SCHEMA,status='running',model_execution=False,completed_updates=0,
        scope=SCOPE,protocol=protocol(),limits=limits('clip'),guard_mode_reuse='Training1800s/memory only; no image generation',quality_assessed=False)
    atomic(out/'metrics.json',report)
    try:
        _deadline(config['deadline'],'clip')
        plan,data=read_plan(plan_path,config['plan_sha256'])
        admitted=admission(config['admission'],plan,config['plan_sha256'])
        if admitted!=config['admission_record']:raise ValueError('Admission changed after parent dispatch')
        if config['lease_deadline_utc']!=plan['lease_deadline_utc']:raise ValueError('Worker lease differs from admitted plan')
        lease_budget(plan['lease_deadline_utc'],plan['resource_profile']['projection']['seconds'])
        data.validate_all();_deadline(config['deadline'],'clip')
        actual_hardware=hardware(plan['expected_gpu']);hardware_matches(actual_hardware,plan['resource_profile']['hardware'])
        flags=_runtime_flags()
        if flags!=plan['resource_profile']['runtime_flags']:raise ValueError('Runtime flags differ from resource profile')
        identity=lambda:dict(windows={a:{k:tensor_sha(v) for k,v in w.items()} for a,w in data.windows.items()},
            positive=tensor_sha(data.positive),negative=tensor_sha(data.negative),evaluation={k:tensor_sha(v) for k,v in data.evaluation.items()})
        initial_identity=identity()
        report.update(source_sha256=plan['source_sha256'],inputs_sha256=INPUT_SHA,plan_sha256=config['plan_sha256'],
            admission=admitted,resource_profile=plan['resource_profile'],hardware=actual_hardware,runtime_flags=flags,
            precision=PRECISION,input_identity=initial_identity,lease_deadline_utc=plan['lease_deadline_utc'])
        atomic(out/'metrics.json',report)
        with Monitor(out,config['deadline'],'clip') as monitor:
            def check():
                monitor.check()
                if utc_seconds(plan['lease_deadline_utc'])<600:raise RuntimeError('Immutable lease recovery reserve reached')
                if _runtime_flags()!=flags:raise RuntimeError('Runtime flags changed during training')
            check();report['model_execution']=True;atomic(out/'metrics.json',report)
            started=time.monotonic();core,weights=native.load_model(config['weights'],check);torch.cuda.synchronize()
            report['load_seconds']=time.monotonic()-started;atomic(out/'weight-load.json',weights)
            expected=_original_weights(weights)
            def verify_core(label):
                start=time.monotonic();values=parameter_records(core,check=check,expected=expected)
                atomic(out/('core-'+label+'.json'),values);report.setdefault('foundation_hash_seconds',{})[label]=time.monotonic()-start
                if any(m._forward_hooks or m._forward_pre_hooks for m in core.modules()):raise RuntimeError('Leaked native projection hook')
            verify_core('before');rotary=core.freqs.detach().cpu().clone()
            with torch.inference_mode(False):
                torch.manual_seed(2026090803)
                controller=controller_module.CommandAttentionController();controller.load_state_dict(data.initial,strict=True);controller.to('cuda:0')
            if sum(p.numel() for p in controller.parameters())!=4_936_448 or any(torch.count_nonzero(p.b.weight) for p in controller.projections.values()):
                raise ValueError('Fresh default zero-B controller required')
            for name,value in controller.state_dict().items():
                copied=value.detach().cpu()
                if tensor_sha(copied)!=tensor_sha(data.initial[name]):raise ValueError('Saved controller CUDA copy differs')
                del copied
            bridge=bridge_module.NativeCommandAttentionBridge(core,controller,profile='spatial')
            import math_steps
            optimizer=torch.optim.AdamW(controller.parameters(),**math_steps.OPTIMIZER)
            if optimizer.state:raise ValueError('Fresh AdamW state required')
            torch.cuda.reset_peak_memory_stats(0)
            def retain(name,values):
                check();path=out/(name+'.safetensors');path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():raise ValueError('Refuse overwriting completed or partial evidence')
                save_file({k:v.detach().cpu().contiguous() for k,v in values.items()},str(path))
            checkpoint_identity=dict(schema=SCHEMA,scope=SCOPE,plan_sha256=config['plan_sha256'],source_sha256=plan['source_sha256'],
                inputs_sha256=INPUT_SHA,controller_parameters=4_936_448,admission_sha256=admitted['sha256'],
                note='Reused checkpoint helper calls the new controller file adapter.safetensors; original residual adapter is absent.')
            def checkpoint(completed,row):
                check()
                if completed==0:draw_rng=data.tensors('draws/draws-0000-0015.safetensors')['rng_initial']
                else:
                    data.noise(row);draw_rng=data.values[f'rng_after_{completed-1:04d}']
                retain(f'cuda-rng-{completed:04d}',dict(rng=torch.cuda.get_rng_state(0)))
                record=save_checkpoint(out,controller,optimizer,completed,identity=checkpoint_identity,draw_rng_state=draw_rng)
                report['last_checkpoint']=dict(directory=record['directory'],manifest_sha256=record['manifest_sha256'],completed_updates=completed)
            def progress(current):
                report.update(current);report['status']='running';atomic(out/'metrics.json',report)
            start=time.monotonic()
            numerical=train_loop(bridge,data,optimizer,retain=retain,checkpoint=checkpoint,progress=progress,check=check,synchronize=torch.cuda.synchronize)
            report.update(numerical,status='running',training_and_evaluation_seconds=time.monotonic()-start)
            verify_core('after')
            if core.freqs.is_inference() or not torch.equal(core.freqs.detach().cpu(),rotary):raise RuntimeError('Original rotary values changed')
            if identity()!=initial_identity:raise RuntimeError('Conditioning or evaluation values changed')
            for name,row in data.plan['files'].items():
                check();p=regular(paths['inputs'],name)
                if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Consumed input file changed')
            report.update(base_unchanged=True,all825_current_value_hashes_verified=True,rotary_unchanged=True,
                memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0)))
            check()
        if hashes(source_paths(paths['repository'],paths['controller_source']))!=plan['source_sha256']:raise ValueError('Executed sources changed')
        if sha(Path(paths['inputs'])/'inputs.json')!=INPUT_SHA:raise ValueError('Input manifest changed')
        report['output_sha256']={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='metrics.json'}
        _deadline(config['deadline'],'clip')
        if utc_seconds(plan['lease_deadline_utc'])<600:raise RuntimeError('Recovery reserve exhausted before completion')
        report.update(status='passed',sources_unchanged=True,inputs_unchanged=True,automatic_promotion=False)
        return report
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-began;atomic(out/'metrics.json',report)


def validate_result(root,plan,plan_sha,admitted):
    root=Path(root);r=read(root/'result/metrics.json');terminal=read(root/'terminal.json');monitor=read(root/'result/monitor-terminal.json')
    if (r.get('schema')!=SCHEMA or r.get('status')!='passed' or r.get('completed_updates')!=512
            or r.get('source_sha256')!=plan['source_sha256'] or r.get('plan_sha256')!=plan_sha or r.get('inputs_sha256')!=INPUT_SHA
            or r.get('admission')!=admitted or r.get('resource_profile')!=plan['resource_profile']
            or r.get('protocol')!=protocol() or r.get('main_predictions')!=1024 or r.get('auxiliary_updates')!=128
            or r.get('auxiliary_predictions')!=512 or r.get('auxiliary_feature_extracts')!=256
            or r.get('all825_current_value_hashes_verified') is not True or r.get('base_unchanged') is not True
            or r.get('rotary_unchanged') is not True or r.get('final_checkpoint_only') is not True
            or [v.get('schedule') for v in r.get('updates',[])]!=plan['input_schedule']
            or any(r.get(k,{}).get('predictions')!=48 or len(r[k].get('scores',[]))!=28 for k in ('initial_evaluation','final_evaluation'))
            or terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or terminal.get('cleanup_error') is not None
            or monitor.get('status')!='complete' or monitor.get('error') is not None or list(root.rglob('watchdog-stop.json'))):
        raise ValueError('Completed fixed512 training/evaluation/guard gates required')
    outputs=r.get('output_sha256',{});required={'weight-load.json','core-before.json','core-after.json','last-valid.json','monitor-terminal.json','memory.jsonl'}
    for i in CHECKPOINTS:
        required.add(f'cuda-rng-{i:04d}.safetensors')
        required.update(f'checkpoint-{i:04d}/{n}' for n in ('adapter.safetensors','optimizer-and-rng.pt','manifest.json'))
    for i in RAW_UPDATES:
        row=plan['input_schedule'][i-1];required.add(f'raw/update-{i:04d}/gradients.safetensors')
        required.update(f'raw/update-{i:04d}/main-{a}.safetensors' for a in row['branches'])
        if row['auxiliary_edge'] is not None:
            required.update(f'raw/update-{i:04d}/aux-{a}-{c}.safetensors' for a in (0,1) for c in ('positive','negative'))
    required.update(f'evaluation/{s}/noise_{n:02d}/{a}-{c}.safetensors' for s in ('initial','final') for n in range(4) for a in ARMS for c in ('positive','negative'))
    actual={p.relative_to(root/'result').as_posix() for p in (root/'result').rglob('*') if p.is_file() and p.name!='metrics.json'}
    if not required<=set(outputs) or actual!=set(outputs):raise ValueError('Complete retained output inventory required')
    for name,v in outputs.items():
        if sha(regular(root/'result',name))!=v:raise ValueError('Retained output hash changed')
    return r


def execute(plan_path,plan_sha,admission_path,weights,output):
    output=Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Fresh regular execution directory required')
    initial=read(plan_path);paths=initial['execution_paths']
    for p in (*paths.values(),str(weights)):
        if output.resolve().is_relative_to(Path(p).resolve()):raise ValueError('Output must be outside model/source/input/profile directories')
    bootstrap(paths['repository'],paths['controller_source'])
    from experiments.wan22_native.spatial_reference.guards import atomic,limits,supervise,stop_child,_deadline
    output.mkdir(parents=True);began=time.monotonic();proc=None
    parent=dict(schema=SCHEMA,status='running',scope=SCOPE,model_execution=False,limits=limits('clip'),image_generation=False)
    atomic(output/'attempt.json',dict(schema=SCHEMA,started_monotonic=began));atomic(output/'metrics.json',parent)
    try:
        budget=lease_budget(initial['lease_deadline_utc'],initial['resource_profile']['projection']['seconds']);deadline=began+budget
        plan,data=read_plan(plan_path,plan_sha);data.validate_all();del data
        admitted=admission(admission_path,plan,plan_sha)
        lease_budget(plan['lease_deadline_utc'],plan['resource_profile']['projection']['seconds'])
        config=dict(schema=SCHEMA,plan=str(Path(plan_path).resolve()),plan_sha256=plan_sha,admission=str(Path(admission_path).resolve()),
            admission_record=admitted,weights=str(Path(weights).resolve()),output=str(output),deadline=deadline,lease_deadline_utc=plan['lease_deadline_utc'])
        for label,p in (('executed-plan.json',plan_path),('executed-admission.json',admission_path)):shutil.copyfile(p,output/label)
        for name,p in source_paths(paths['repository'],paths['controller_source']).items():
            target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        atomic(output/'launch.json',config)
        parent.update(plan_sha256=plan_sha,inputs_sha256=INPUT_SHA,source_sha256=plan['source_sha256'],admission=admitted,
            resource_profile_sha256=plan['resource_profile']['result_sha256'],lease_deadline_utc=plan['lease_deadline_utc'],projection=plan['resource_profile']['projection'])
        with (output/'worker.log').open('x') as log:
            _deadline(deadline,'clip')
            if deadline-time.monotonic()<plan['resource_profile']['projection']['seconds']:raise ValueError('Remaining training budget fell below profiled projection before dispatch')
            lease_budget(plan['lease_deadline_utc'],plan['resource_profile']['projection']['seconds'])
            proc=subprocess.Popen([sys.executable,str(HERE/'run_training.py'),'--worker-config',str(output/'launch.json')],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            parent['model_execution']=None;atomic(output/'metrics.json',parent)
            supervise(proc,output,deadline,'clip')
        validate_result(output,plan,plan_sha,admitted)
        if hashes(source_paths(paths['repository'],paths['controller_source']))!=plan['source_sha256']:raise ValueError('Sources changed during supervised run')
        if admission(admission_path,plan,plan_sha)!=admitted:raise ValueError('Admission changed during supervised run')
        _deadline(deadline,'clip')
        if utc_seconds(plan['lease_deadline_utc'])<600:raise RuntimeError('Recovery reserve exhausted during parent validation')
        parent.update(status='passed',model_execution=True,result_metrics_sha256=sha(output/'result/metrics.json'),terminal_sha256=sha(output/'terminal.json'))
        return parent
    except BaseException as error:
        cleanup_error=None
        try:
            if proc is not None:stop_child(proc)
        except BaseException as cleanup:cleanup_error=str(cleanup)
        if not (output/'terminal.json').exists():atomic(output/'terminal.json',dict(status='failed',exit_code=proc.returncode if proc else None,error=str(error),cleanup_error=cleanup_error,mode='clip'))
        elif cleanup_error:atomic(output/'handoff-cleanup-error.json',dict(error=str(error),cleanup_error=cleanup_error))
        if proc is not None:
            parent['model_execution']=None
            try:
                partial=read(output/'result/metrics.json')
                if type(partial.get('model_execution')) is bool:parent['model_execution']=partial['model_execution']
            except (ValueError,OSError):pass
        parent.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        parent['elapsed_seconds']=time.monotonic()-began;atomic(output/'metrics.json',parent)


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--prepare-plan',action='store_true');mode.add_argument('--execute',action='store_true');mode.add_argument('--worker-config',type=Path,help=argparse.SUPPRESS)
    for n in ('repository','controller-source','inputs','resource-profile','plan','admission','weights','output'):p.add_argument('--'+n,type=Path)
    for n in ('inputs-sha256','profile-sha256','plan-sha256','expected-gpu','lease-deadline-utc'):p.add_argument('--'+n)
    a=p.parse_args()
    if a.worker_config:value=worker(read(a.worker_config,2**20))
    elif a.prepare_plan:
        required=('repository','controller_source','inputs','resource_profile','inputs_sha256','profile_sha256','expected_gpu','lease_deadline_utc','output')
        if any(getattr(a,n) is None for n in required):p.error('Complete source/input/profile/GPU/lease and fresh output-plan arguments required')
        if a.output.exists():p.error('Fresh plan path required')
        value=make_plan(a);a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
        value=dict(status='plan-prepared',model_execution=False,plan_sha256=sha(a.output),projection=value['resource_profile']['projection'])
    elif a.execute:
        if any(getattr(a,n) is None for n in ('plan','plan_sha256','admission','weights','output')):p.error('Exact plan, admission, weights and fresh output required')
        value=execute(a.plan,a.plan_sha256,a.admission,a.weights,a.output)
    else:value=dict(status='plan-only',model_execution=False,protocol=protocol())
    print(json.dumps(value,indent=2,allow_nan=False))

if __name__=='__main__':main()
