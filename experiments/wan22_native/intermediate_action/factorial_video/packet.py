# SPDX-License-Identifier: Apache-2.0
"""Fixed conditioning and later actual-checkpoint eligibility. CPU reads only."""
from pathlib import Path
import math
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import sampler
from video import ARMS
from experiments.wan22_native.action_cuda import probe,probe_evidence as pe
from experiments.wan22_native.spatial_reference import evidence as se,sampling
from experiments.wan22_native.spatial_reference.guards import atomic,limits

HERE=Path(__file__).resolve().parent;REPO=pe.REPO
sha,read,file=pe.sha,pe.read_json,pe.relative_file
SCHEMA='worldline-factorial-trained-video-v1';SCOPE='final128-six-arm-generated-video-only'
SAMPLER='374e2d1b68015c7ce0b436dcddb68c6f5d8b247584feaa9140b94ec1aae3c314'
TRAINING_PLAN='030f04106f588dc903d5f9e3de3078badd7bf870f0923dadf0f5b796eaae314e'
NOISE_FILE='0b1daf39df7630971ae67ae1d0db43ede1bc0fa20a893a295d4eed6e878faade'
NOISE='d3a4f22635d595df7ddc0b9f3bc7b7d621c23ea540c8f36f759ee6a145adeace'
OBS_FILE='dc97ee856cadc78520e9f87a9b7d3ba0534c6270239affce146d31281269785a'
OBS='222f3792096029fba5d4cd8c0ea5a4b5919f0f92a6046cba5acd87ffa153cba3'
TEXT_FILE='40f59cf54f2819555ff37a95a116e41f3fe1e247af14cf824704f6191df327ee'
TEXT={'atrium':'f152d177bf53650f4ac8658665fd6b7e86d4bad68106309c1ecd19e56a4bb093','native_negative':'aa6911f67cb7a4e5cf934131594b6fe264dff9983f84ab188e9199c706cf60cc'}
SHAPE=(48,5,44,78)
PROOFS=('plan.json','metrics.json','terminal.json','result/metrics.json','result/weight-load.json',
        'result/core-before.json','result/core-after.json','result/last-valid.json','result/checkpoint-0128/manifest.json')
FIXED=('sampling-inputs.safetensors','contexts.safetensors','commands.safetensors','fixed-receipt.json')

def contract():
    p=HERE/'training-contract.json'
    if sha(p)!=TRAINING_PLAN:raise ValueError('Exact reviewed training family required')
    return read(p)

def sources():
    repo={**pe.source_hashes(independent=True),**se.sources()}
    for name in ('intermediate_action/bridge.py','intermediate_action/test_bridge.py'):
        path='experiments/wan22_native/'+name;repo[path]=sha(REPO/path)
    local={n:sha(HERE/n) for n in ('sampler.py','video.py','packet.py','run.py','test_cpu.py','training-contract.json','criteria.md')}
    if local['sampler.py']!=SAMPLER or local['training-contract.json']!=TRAINING_PLAN:raise ValueError('Original sampler/training contract changed')
    return dict(repository=repo,local=local)

def protocol():
    return dict(scope=SCOPE,arms=list(ARMS),block_index=28,profile='spatial',settings=dict(sampling.SETTINGS),
        predictions=600,solver_updates=300,frames_per_arm=17,final_checkpoint_updates=128,
        checkpoint_selection=False,training=False,future_target_conditioning=False,native_control=False,
        negative_context_adapter_training=True,combined_limits=limits('clip'),worker_limits=limits('pair'),
        recovery_reserve_seconds=600,criteria_sha256=sha(HERE/'criteria.md'))

def root(path,fresh=False):
    p=Path(path).absolute()
    if any(x.is_symlink() for x in (p,*p.parents)) or p.resolve().is_relative_to(REPO):raise ValueError('Regular artifact directory outside repository required')
    if p.exists() if fresh else not p.is_dir():raise ValueError('Fresh output or existing input directory required')
    return p

def checked_outputs(base,rows):
    if not isinstance(rows,dict) or not rows:raise ValueError('Nonempty exact output inventory required')
    for n,digest in rows.items():
        if sha(file(base,n))!=digest:raise ValueError('Artifact bytes changed: '+n)

def validate_commands(commands):
    if set(commands)!=set(ARMS):raise ValueError('All six commands required')
    for arm,v in commands.items():
        motion,door=arm.split('_');expected=torch.zeros(1,16,6)
        expected[0,:,3]={'stationary':0.,'left':math.pi/120,'right':-math.pi/120}[motion]
        expected[0,0,5]=int(door=='interact')
        if not torch.equal(v,expected):raise ValueError('Only original destination-aligned factorial commands')

def read_fixed(folder):
    folder=root(folder);r=read(file(folder,'fixed-receipt.json'))
    if r.get('schema')!='worldline-factorial-video-fixed-inputs-v1' or r.get('status')!='prepared':raise ValueError('Fixed conditioning receipt required')
    if set(r['artifacts'])!=set(FIXED)-{'fixed-receipt.json'}:raise ValueError('Exact three conditioning artifacts required')
    checked_outputs(folder,r['artifacts'])
    v=pe.tensors(file(folder,'sampling-inputs.safetensors'),{'initial_noise':(SHAPE,'F32'),'initial_latent':(SHAPE,'F32'),'observation':((1,48,1,44,78),'F32'),'token_times':((1,4290),'I64')},r['artifacts']['sampling-inputs.safetensors'])
    c=pe.tensors(file(folder,'contexts.safetensors'),{'atrium':((25,4096),'F32'),'native_negative':((126,4096),'F32')},TEXT_FILE)
    commands=pe.tensors(file(folder,'commands.safetensors'),{a:((1,16,6),'F32') for a in ARMS},r['artifacts']['commands.safetensors'])
    if sampler.tensor_sha(v['initial_noise'])!=NOISE or sampler.tensor_sha(v['observation'])!=OBS or {k:sampler.tensor_sha(t) for k,t in c.items()}!=TEXT:raise ValueError('Original noise/text or new independent observation changed')
    validate_commands(commands)
    for arm in ARMS:
        if sampler.tensor_sha(commands[arm])!=contract()['files']['cache/result/'+arm+'.safetensors']['header']['commands']['sha256']:raise ValueError('Training command identity differs')
        sampler.validate_inputs(v,c,commands[arm],'spatial')
    identity=dict(values={k:sampler.tensor_sha(t) for k,t in v.items()},contexts=TEXT,
                  commands={k:sampler.tensor_sha(t) for k,t in commands.items()})
    if r['tensor_identity']!=identity:raise ValueError('Fixed input identity differs')
    return v,c,commands,r

def prepare_fixed(cache_result,previous_visual,output):
    out=root(output,fresh=True);cache_result=root(cache_result);previous_visual=root(previous_visual)
    if any(out.is_relative_to(x) for x in (cache_result,previous_visual)):raise ValueError('Output cannot be inside input')
    p=file(previous_visual,'sampling-inputs.safetensors')
    if sha(p)!=NOISE_FILE or sha(file(previous_visual,'contexts.safetensors'))!=TEXT_FILE or sha(file(cache_result,'observation.safetensors'))!=OBS_FILE:raise ValueError('Exact fixed source files required')
    with safe_open(p,framework='pt',device='cpu') as f:noise=f.get_tensor('initial_noise')
    with safe_open(file(cache_result,'observation.safetensors'),framework='pt',device='cpu') as f:obs=f.get_tensor('observation')
    commands={};provenance={}
    for arm in ARMS:
        p=file(cache_result,arm+'.safetensors');expected=contract()['files']['cache/result/'+arm+'.safetensors']
        if sha(p)!=expected['sha256'] or p.stat().st_size!=expected['bytes']:raise ValueError('Exact completed training cache file required')
        with safe_open(p,framework='pt',device='cpu') as f:
            value=f.get_tensor('observation');prefix=f.get_slice('target')[:,:,:1];commands[arm]=f.get_tensor('commands')
        if not torch.equal(value,obs) or not torch.equal(prefix,obs):raise ValueError('Independent raw prefix must already be exact')
        provenance[arm]=expected['sha256']
    initial=noise.clone();initial[:,:1]=obs[0]
    values=dict(initial_noise=noise,initial_latent=initial,observation=obs,token_times=sampling.times_at(torch.tensor(999),SHAPE))
    out.mkdir(parents=True)
    save_file(values,str(out/'sampling-inputs.safetensors'));save_file(commands,str(out/'commands.safetensors'))
    shutil.copyfile(previous_visual/'contexts.safetensors',out/'contexts.safetensors')
    atomic(out/'fixed-receipt.json',dict(schema='worldline-factorial-video-fixed-inputs-v1',status='prepared',model_execution=False,fresh_noise_drawn=False,
        previous_noise_file_sha256=NOISE_FILE,new_observation_file_sha256=OBS_FILE,cache_file_sha256=provenance,
        original_context_file_sha256=TEXT_FILE,future_targets_materialized=False,prefix_slices_read_for_verification=True,
        artifacts={n:sha(out/n) for n in FIXED if n!='fixed-receipt.json'},tensor_identity=dict(values={k:sampler.tensor_sha(v) for k,v in values.items()},contexts=TEXT,commands={k:sampler.tensor_sha(v) for k,v in commands.items()})))
    read_fixed(out);return read(out/'fixed-receipt.json')

def training_metadata(base,checkpoint=None,full=False):
    base=root(base);p=read(file(base,'plan.json'));parent=read(file(base,'metrics.json'));w=read(file(base,'result/metrics.json'));t=read(file(base,'terminal.json'))
    expected=contract()
    if sha(base/'plan.json')!=TRAINING_PLAN or p!=expected:raise ValueError('Exact reviewed 128 training plan required')
    if (parent.get('status')!='passed' or w.get('status')!='passed' or t.get('status')!='complete' or t.get('exit_code')!=0
        or t.get('cleanup_error') is not None or parent.get('model_execution') is not True or w.get('model_execution') is not True
        or parent.get('result_metrics_sha256')!=sha(base/'result/metrics.json') or parent.get('terminal_sha256')!=sha(base/'terminal.json')
        or parent.get('plan_sha256')!=TRAINING_PLAN or w.get('plan_sha256')!=TRAINING_PLAN
        or parent.get('source_sha256')!=p['source_sha256'] or w.get('source_sha256')!=p['source_sha256']
        or w.get('base_unchanged') is not True or w.get('all825_current_value_hashes_verified') is not True
        or w.get('completed_updates')!=128 or w.get('main_predictions')!=256 or w.get('auxiliary_predictions')!=128
        or w.get('auxiliary_feature_extracts')!=64 or w.get('auxiliary_updates')!=32 or w.get('zero_gate_passed') is not True
        or len(w.get('parity',[]))!=2 or any(r.get('exact_equal') is not True for r in w['parity'])
        or [r.get('schedule') for r in w.get('updates',[])]!=p['schedule'] or w.get('protocol')!=p['protocol']
        or parent.get('limits')!=limits('pair') or w.get('limits')!=limits('pair') or list(base.rglob('watchdog-stop.json'))):raise ValueError('Completed unchanged final128 training required')
    if full:
        checked_outputs(base/'source',p['source_sha256']);checked_outputs(base/'result',w['output_sha256'])
        required={f'main-{i:04d}-{a}.safetensors' for i in range(1,129) for a in ('closed','interact')}|{f'gradients-after-clip-{i:04d}.safetensors' for i in range(1,129)}
        required|={f'auxiliary-{i:04d}-{a}-{c}.safetensors' for i in range(1,129,4) for a in ('closed','open') for c in ('positive','negative')}
        if not required<=set(w['output_sha256']):raise ValueError('Complete raw training predictions/gradients required')
    before=read(file(base,'result/core-before.json'));after=read(file(base,'result/core-after.json'))
    if before!=after or before!=probe._original_weights(read(file(base,'result/weight-load.json'))):raise ValueError('Original825 identities differ')
    pointer=read(file(base,'result/last-valid.json'));m=read(file(base,'result/checkpoint-0128/manifest.json'))
    if (pointer.get('completed_updates')!=128 or pointer.get('directory')!='checkpoint-0128' or pointer.get('manifest_sha256')!=sha(base/'result/checkpoint-0128/manifest.json')
        or m.get('completed_updates')!=128 or m.get('identity',{}).get('plan_sha256')!=TRAINING_PLAN
        or m['identity'].get('source_sha256')!=p['source_sha256']):raise ValueError('Final128 checkpoint identity differs')
    path=Path(checkpoint) if checkpoint is not None else file(base,'result/checkpoint-0128/adapter.safetensors')
    adapter,record=sampler.load_adapter_checkpoint(path,m['files']['adapter.safetensors']);del adapter
    if record['tensor_sha256']!={k:v['sha256'] for k,v in m['tensors'].items()}:raise ValueError('Final adapter tensor records differ')
    identity=dict(parent_sha256=sha(base/'metrics.json'),result_sha256=sha(base/'result/metrics.json'),terminal_sha256=sha(base/'terminal.json'),plan_sha256=TRAINING_PLAN,
        final_checkpoint_manifest_sha256=sha(base/'result/checkpoint-0128/manifest.json'),final_checkpoint_sha256=record['checkpoint_sha256'],source_sha256=p['source_sha256'])
    return identity,dict(adapter=record,hardware=w['hardware'],core_records=before,input_identity=w['input_identity'])

def audit(path,identity):
    a=read(path)
    required=dict(schema='worldline-factorial-intermediate128-actual-audit-v1',status='passed',completed_updates=128,auxiliary_updates=32,main_predictions=256,auxiliary_predictions=128,auxiliary_feature_extracts=64,all825_unchanged=True,final_checkpoint_only=True)
    if any(a.get(k)!=v for k,v in required.items()) or a.get('identity')!=identity:raise ValueError('Passed exact final128 independent audit required')
    return sha(path)

def review(path):
    r=read(path)
    if r.get('status')!='passed' or r.get('source_sha256')!=sources() or r.get('sources_unchanged') is not True or type(r.get('tests')) is not int or r['tests']<8 or any(type(r.get(k)) is not int or r[k]!=0 for k in ('pytest_exit_code','failures','errors','skipped')):raise ValueError('Current complete CPU review required')
    return sha(path)

def prepare(training_run,training_audit,fixed_inputs,cpu_report,output):
    out=root(output,fresh=True);base=root(training_run);fixed=root(fixed_inputs)
    if out.is_relative_to(base) or out.is_relative_to(fixed):raise ValueError('Fresh output cannot be inside inputs')
    cpu=review(cpu_report);identity,details=training_metadata(base,full=True);audit_sha=audit(training_audit,identity)
    values,contexts,commands,fixed_record=read_fixed(fixed)
    for arm in ARMS:
        actual=details['input_identity']['windows'][arm]
        if actual['observation']!=OBS or actual['commands']!=sampler.tensor_sha(commands[arm]):raise ValueError('Evaluation differs from actual training conditions')
    out.mkdir(parents=True)
    for n in FIXED:shutil.copyfile(fixed/n,out/n)
    shutil.copyfile(base/'result/checkpoint-0128/adapter.safetensors',out/'adapter.safetensors')
    shutil.copyfile(training_audit,out/'training-audit.json');shutil.copyfile(cpu_report,out/'cpu-report.json')
    for n in PROOFS:
        target=out/'evidence/training'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/n,target)
    mapping=sources()
    for scope,rows in mapping.items():
        for n,d in rows.items():
            target=out/'source'/scope/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile((REPO if scope=='repository' else HERE)/n,target)
            if sha(target)!=d:raise ValueError('Source changed during preparation')
    plan=dict(schema=SCHEMA,status='prepared',protocol=protocol(),source_sha256=mapping,expected_gpu=details['hardware']['name'],
        training_identity=identity,training_details=details,training_audit_sha256=audit_sha,cpu_report_sha256=cpu,
        artifacts={n:sha(out/n) for n in (*FIXED,'adapter.safetensors','training-audit.json','cpu-report.json')},
        evidence_sha256={n:sha(out/'evidence/training'/n) for n in PROOFS},model_execution=False)
    atomic(out/'plan.json',plan);read_prepared(out);return plan

def read_prepared(folder):
    out=root(folder);p=read(file(out,'plan.json'))
    if p.get('schema')!=SCHEMA or p.get('status')!='prepared' or p.get('protocol')!=protocol() or p.get('source_sha256')!=sources():raise ValueError('Current exact six-arm video plan required')
    for scope,rows in p['source_sha256'].items():checked_outputs(out/'source'/scope,rows)
    if set(p['artifacts'])!=set((*FIXED,'adapter.safetensors','training-audit.json','cpu-report.json')) or set(p['evidence_sha256'])!=set(PROOFS):raise ValueError('Complete fixed input and prior proof set required')
    checked_outputs(out,p['artifacts']);checked_outputs(out/'evidence/training',p['evidence_sha256'])
    identity,details=training_metadata(out/'evidence/training',out/'adapter.safetensors')
    if identity!=p['training_identity'] or details!=p['training_details'] or audit(out/'training-audit.json',identity)!=p['training_audit_sha256'] or review(out/'cpu-report.json')!=p['cpu_report_sha256']:raise ValueError('Actual training/audit identity differs')
    values,contexts,commands,_=read_fixed(out)
    for arm in ARMS:
        v=details['input_identity']['windows'][arm]
        if v['observation']!=OBS or v['commands']!=sampler.tensor_sha(commands[arm]):raise ValueError('Actual training conditioning differs')
    if p['expected_gpu']!=details['hardware']['name']:raise ValueError('Expected GPU differs from actual training')
    return p,values,contexts,commands
