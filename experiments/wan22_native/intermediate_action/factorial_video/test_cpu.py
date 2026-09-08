# SPDX-License-Identifier: Apache-2.0
"""Focused CPU checks. No foundation weights, CUDA or generated videos."""
from datetime import datetime,timedelta,timezone
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file,load_file
import sampler
import packet
import video
import run

@pytest.fixture(autouse=True)
def bounded():
    old=torch.get_num_threads();torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized();torch.set_num_threads(old)

class FakeBridge:
    def __init__(self):
        self.core=torch.nn.Module();self.core.patch_embedding=torch.nn.Linear(1,1).requires_grad_(False)
        self.adapter=torch.nn.Linear(1,1);self.block_index=28;self.calls=[]
    def __call__(self,x,t,c,*,commands,observation,track_grad):
        assert not track_grad and torch.equal(x[:,:,:1],observation)
        self.calls.append((commands.clone(),c[0].clone(),t.clone()))
        return torch.ones_like(x)*float(c[0][0,0])*.001+commands[0,0,5]*.002

def values(shape=(48,5,18,32)):
    noise=torch.linspace(-.1,.1,int(torch.tensor(shape).prod())).reshape(shape);obs=torch.ones(1,shape[0],1,*shape[-2:])*.07
    initial=noise.clone();initial[:,:1]=obs[0]
    return dict(initial_noise=noise,initial_latent=initial,observation=obs,
        token_times=sampler.native_sampling.times_at(torch.tensor(999),shape))

def texts():return {'atrium':torch.ones(25,4096),'native_negative':-torch.ones(126,4096)}

def test_literal_sampler_50_steps_100_same_commands_and_clean_prefix():
    bridge=FakeBridge();v=values();c=texts();cmd=torch.zeros(1,16,6);cmd[0,0,5]=1;events=[]
    final,r=sampler._sample_bridge(bridge,v,c,cmd,'baseline',event=lambda i,t,x,vel:events.append((i,x[:,:1].clone())))
    assert r['solver_updates']==50 and r['predictions']==100 and len(bridge.calls)==100 and len(events)==50
    assert all(torch.equal(x,v['observation'][0]) for i,x in events) and torch.equal(final[:,:1],v['observation'][0])
    assert all(torch.equal(row[0],cmd) for row in bridge.calls)
    assert {sampler.tensor_sha(row[1]) for row in bridge.calls}=={sampler.tensor_sha(x) for x in c.values()}
    assert all((row[2][:,:144]==0).all() for row in bridge.calls)
    assert all(p.grad is None for p in bridge.adapter.parameters())

@pytest.mark.parametrize('kind',['nan','old_prefix','target_field'])
def test_malformed_conditioning_rejected_before_prediction(kind):
    bridge=FakeBridge();v=values();cmd=torch.zeros(1,16,6)
    if kind=='nan':cmd[0,0,0]=float('nan')
    elif kind=='old_prefix':v['initial_latent'][0,0,0,0]+=1
    else:v['target']=v['initial_latent'].clone()
    with pytest.raises(ValueError):sampler._sample_bridge(bridge,v,texts(),cmd,'baseline')
    assert not bridge.calls

def test_explicit_block28_constructor_never_patches_old_bridge(monkeypatch):
    old=sampler.NativeCUDAActionBridge;adapter=torch.nn.Linear(1,1);identity={'tensor_sha256':video.adapter_identity(adapter)}
    monkeypatch.setattr(adapter,'to',lambda device:adapter)
    monkeypatch.setattr(sampler,'load_adapter_checkpoint',lambda *a:(adapter,identity));seen={}
    def make(core,a,**kw):seen.update(kw);assert a is adapter;return SimpleNamespace(block_index=kw['block_index'])
    monkeypatch.setattr(video,'IntermediateActionBridge',make)
    core=SimpleNamespace(patch_embedding=SimpleNamespace(weight=SimpleNamespace(device=torch.device('cuda:0'))))
    b,r=video.load_bridge(core,'unused','a'*64)
    assert seen=={'block_index':28,'profile':'spatial'} and b.block_index==28 and sampler.NativeCUDAActionBridge is old
    with torch.inference_mode(),pytest.raises(ValueError,match='normal'):video.load_bridge(core,'unused','a'*64)

def test_six_arm_order_separate_solver_retention_and_metadata(tmp_path,monkeypatch):
    bridge=FakeBridge();identity={'tensor_sha256':video.adapter_identity(bridge.adapter)}
    noise=torch.zeros(2,5,2,2);obs=torch.zeros(1,2,1,2,2);v=dict(initial_noise=noise,initial_latent=noise.clone(),observation=obs,token_times=torch.zeros(1,5,dtype=torch.int64))
    commands={a:torch.full((1,16,6),i,dtype=torch.float32) for i,a in enumerate(video.ARMS)};calls=[]
    def sample(b,vals,ctx,cmd,profile,*,event,check):
        assert b is bridge and profile=='spatial';calls.append(int(cmd[0,0,0]))
        for i in range(50):event(i,torch.tensor(999-i),vals['initial_latent'].clone(),{'positive':noise.clone(),'negative':noise.clone(),'guided':noise.clone()})
        return noise.clone(),dict(predictions=100,solver_updates=50)
    monkeypatch.setattr(sampler,'_sample_bridge',sample)
    rows=video.sample_six(bridge,identity,tmp_path,v,{'text':torch.ones(1)},commands)
    assert calls==list(range(6)) and tuple(rows)==video.ARMS and len(list(tmp_path.glob('*/step-*.safetensors')))==300
    assert len(list(tmp_path.glob('*/initial-velocities.safetensors')))==6
    assert all(r['negative_context_adapter_training'] and r['block_index']==28 and not r['training'] for r in rows.values())
    with pytest.raises(FileExistsError):video.sample_six(bridge,identity,tmp_path,v,{'text':torch.ones(1)},commands)

def test_fixed_input_construction_uses_new_observation_not_old_clamp(tmp_path,monkeypatch):
    cache=tmp_path/'cache';old=tmp_path/'old';cache.mkdir();old.mkdir()
    noise=torch.arange(40,dtype=torch.float32).reshape(2,5,2,2);obs=torch.ones(1,2,1,2,2)*9
    save_file({'initial_noise':noise,'initial_latent':noise+20,'observation':obs-4},str(old/'sampling-inputs.safetensors'))
    save_file({'atrium':torch.zeros(25,4096),'native_negative':torch.zeros(126,4096)},str(old/'contexts.safetensors'))
    save_file({'observation':obs},str(cache/'observation.safetensors'));records={}
    for arm in video.ARMS:
        target=noise[None].clone();target[:,:,:1]=obs;p=cache/(arm+'.safetensors')
        save_file({'target':target,'observation':obs,'commands':torch.zeros(1,16,6)},str(p))
        records['cache/result/'+p.name]=dict(bytes=p.stat().st_size,sha256=packet.sha(p))
    for key,p in [('NOISE_FILE',old/'sampling-inputs.safetensors'),('TEXT_FILE',old/'contexts.safetensors'),('OBS_FILE',cache/'observation.safetensors')]:monkeypatch.setattr(packet,key,packet.sha(p))
    monkeypatch.setattr(packet,'contract',lambda:{'files':records});monkeypatch.setattr(packet,'read_fixed',lambda p:None)
    original=packet.safe_open
    class Reader:
        def __init__(self,*a,**kw):self.inner=original(*a,**kw)
        def __enter__(self):self.f=self.inner.__enter__();return self
        def __exit__(self,*a):return self.inner.__exit__(*a)
        def get_tensor(self,key):assert key!='target','Future target materialized';return self.f.get_tensor(key)
        def get_slice(self,key):return self.f.get_slice(key)
    monkeypatch.setattr(packet,'safe_open',Reader)
    out=tmp_path/'fixed';packet.prepare_fixed(cache,old,out);v=load_file(out/'sampling-inputs.safetensors')
    assert torch.equal(v['initial_noise'],noise) and torch.equal(v['initial_latent'][:,:1],obs[0])
    assert torch.equal(v['initial_latent'][:,1:],noise[:,1:])

def test_final_audit_requires_exact_family_counts_identity(tmp_path):
    identity={'final_checkpoint_sha256':'c'*64,'plan_sha256':packet.TRAINING_PLAN,'source_sha256':{'a':'b'}}
    good=dict(schema='worldline-factorial-intermediate128-actual-audit-v1',status='passed',completed_updates=128,auxiliary_updates=32,main_predictions=256,auxiliary_predictions=128,auxiliary_feature_extracts=64,all825_unchanged=True,final_checkpoint_only=True,identity=identity)
    p=tmp_path/'audit.json';run.atomic(p,good);assert packet.audit(p,identity)==packet.sha(p)
    for key,value in [('schema','worldline-action-effect128-actual-independent-v1'),('completed_updates',16),('all825_unchanged',False),('identity',{})]:
        run.atomic(p,{**good,key:value})
        with pytest.raises(ValueError):packet.audit(p,identity)

@pytest.mark.parametrize('delta,passed',[(2400,False),(2400.001,True),(2399,False),(600,False)])
def test_lease_boundary_includes_full_video_and_reserve(delta,passed):
    now=datetime(2026,9,8,tzinfo=timezone.utc);a={'lease_deadline_utc':(now+timedelta(seconds=delta)).isoformat()}
    if passed:run.lease_check(a,dispatch=True,now=now)
    else:
        with pytest.raises(RuntimeError):run.lease_check(a,dispatch=True,now=now)
    with pytest.raises(ValueError):run.lease_check({'lease_deadline_utc':'2026-09-09T10:00:00'},now=now)

def test_expired_parent_never_starts_worker(tmp_path,monkeypatch):
    out=tmp_path/'run';out.mkdir();monkeypatch.setattr(run.time,'monotonic',lambda:100)
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**kw:pytest.fail('Expired launch'))
    with pytest.raises(RuntimeError):run.launch(dict(prepared=str(out),stage='core',parent_deadline=99,admission={}))
    assert packet.read(out/'core/terminal.json')['status']=='failed'

def test_child_failure_keeps_original_error_and_cleanup_record(tmp_path,monkeypatch):
    out=tmp_path/'run';out.mkdir();now=run.time.monotonic();process=SimpleNamespace(returncode=1)
    monkeypatch.setattr(run,'lease_check',lambda *a,**kw:None)
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**kw:process)
    monkeypatch.setattr(run,'supervise',lambda *a:(_ for _ in ()).throw(RuntimeError('original failure')))
    monkeypatch.setattr(run,'stop_child',lambda *a:(_ for _ in ()).throw(RuntimeError('cleanup failure')))
    with pytest.raises(RuntimeError,match='original failure'):run.launch(dict(prepared=str(out),stage='core',parent_deadline=now+1799,admission={}))
    assert packet.read(out/'core/terminal.json')['cleanup_error']=='cleanup failure'

def test_hardware_only_equal_or_larger_memory_and_original_caps():
    old={'name':'A100','total_memory_bytes':80*2**30,'nested':{'flag':True}}
    assert run.require_hardware(old,old)['additional_reported_bytes']==0
    assert run.require_hardware({**old,'total_memory_bytes':old['total_memory_bytes']+2**21},old)
    for changed in ({**old,'total_memory_bytes':old['total_memory_bytes']-1},{**old,'nested':{'flag':1}}):
        with pytest.raises(ValueError):run.require_hardware(changed,old)
    assert run.limits('pair')['seconds']==900 and run.limits('clip')['seconds']==1800 and run.limits('pair')['cuda_reserved_bytes']==60*2**30

def test_exact_admission_and_single_use_exclude_native_and_training(tmp_path,monkeypatch):
    root=tmp_path/'prepared';root.mkdir();run.atomic(root/'plan.json',{})
    plan=dict(source_sha256={},training_identity={'final_checkpoint_sha256':'c'*64},training_audit_sha256='a'*64,cpu_report_sha256='b'*64)
    good=dict(schema=packet.SCHEMA,scope=packet.SCOPE,decision='admit',issued_by='parent-agent',plan_sha256=packet.sha(root/'plan.json'),
        **plan,protocol=packet.protocol(),minimum_lease_remaining_seconds=2400,native_control=False,training_admitted=False,reason='Reviewed complete result',lease_deadline_utc='2099-01-01T00:00:00Z')
    path=tmp_path/'admission.json';run.atomic(path,good);assert run.admit(path,root,plan)['sha256']==packet.sha(path)
    for field,value in [('schema','old-pair'),('native_control',True),('minimum_lease_remaining_seconds',1800),('training_admitted',True)]:
        run.atomic(path,{**good,field:value})
        with pytest.raises(ValueError):run.admit(path,root,plan)
    (root/'execution-attempt.json').write_text('prior failure')
    monkeypatch.setattr(packet,'read_prepared',lambda p:pytest.fail('Reused execution'))
    with pytest.raises(ValueError,match='Single-use'):run.execute(root,tmp_path,path)


def test_parent_failure_retains_terminal_for_complete_recovery(tmp_path,monkeypatch):
    root=tmp_path/'prepared';root.mkdir()
    monkeypatch.setattr(packet,'read_prepared',lambda p:(_ for _ in ()).throw(ValueError('invalid input')))
    with pytest.raises(ValueError,match='invalid input'):run.execute(root,tmp_path,tmp_path/'decision')
    terminal=packet.read(root/'terminal.json');metrics=packet.read(root/'metrics.json')
    assert terminal['status']=='failed' and terminal['exit_code']==1
    assert metrics['terminal_sha256']==packet.sha(root/'terminal.json')
