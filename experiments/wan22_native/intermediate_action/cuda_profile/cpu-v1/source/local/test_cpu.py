# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU wiring and failure tests, no CUDA or real foundation weights."""
import copy
from pathlib import Path
from unittest import mock
import pytest
import torch
from safetensors.torch import save_file
import engine
import packet
import run
from experiments.wan22_native.intermediate_action.test_bridge import fixture, cpu_attention
from experiments.wan22_native.intermediate_action.cached_intermediate import CachedIntermediateActionBridge
from experiments.wan22_native.cuda_reference.vendor import model as vendor
from experiments.wan22_native.spatial_reference.guards import atomic
from experiments.wan22_native.official_cpu.streaming import sha


@pytest.fixture(autouse=True)
def bounded(monkeypatch):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(vendor,'flash_attention',cpu_attention)
    yield
    torch.set_num_threads(previous)
    assert not torch.cuda.is_initialized()


def tiny():
    b,x,t,c,commands,observation=fixture()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(908)
        windows=[]
        for opened in (False,True):
            target=torch.randn_like(x);target[:,:,:1]=observation
            action=commands.clone();action[0,0,5]=float(opened)
            windows.append(dict(target=target,observation=observation.clone(),commands=action))
        draws={'noise_0000':torch.randn_like(x),'noise_0004':torch.randn_like(x)}
        negative=torch.randn_like(c[0])
    data=dict(windows=windows,initial={k:v.detach().clone() for k,v in b.adapter.state_dict().items()},
              draws=draws,positive=c[0],negative=negative,rng=torch.get_rng_state(),
              schedule=packet.protocol()['schedule'])
    def make(index):
        adapter=copy.deepcopy(b.adapter);adapter.load_state_dict(data['initial'])
        return CachedIntermediateActionBridge(b.core,adapter,block_index=index,test_only=True)
    def native(x,t,c):
        with torch.inference_mode(False),torch.no_grad():
            return torch.stack(b.core(list(x.unbind(0)),t,[c],t.shape[1]))
    return b.core,data,make,native


def test_literal_two_placement_original_updates_and_retention():
    core,data,make,native=tiny();saved={};checkpoints={};states=[]
    before={k:v.detach().clone() for k,v in core.state_dict().items()}
    def retain(name,values):
        assert name not in saved
        saved[name]={k:v.detach().clone() for k,v in values.items()}
    def cp(label,n,bridge,optimizer,current):
        assert current['zero_gate_passed']
        checkpoints[label,n]=copy.deepcopy(optimizer.state_dict())
        if n==0:
            assert not optimizer.state
            assert all(torch.equal(v,data['initial'][k]) for k,v in bridge.adapter.state_dict().items())
    result=engine.execute(data,placements=(0,1),make_bridge=make,native_predict=native,retain=retain,
        checkpoint=cp,progress=lambda r:states.append(copy.deepcopy(r)))
    assert result['completed_updates']==4 and result['status']=='passed'
    assert len(result['parity'])==16 and all(x['exact_equal'] for x in result['parity'])
    assert len(saved)==46  # 18 parity + 24 training predictions + 4 gradient bundles.
    assert len(checkpoints)==6
    for label in ('0','1'):
        updates=result['placements'][label]['updates']
        assert [r['command_gru_gradient_l2']==0 for r in updates]==[True,False]
        assert all(r['auxiliary']['head_predictions']==4 and r['auxiliary']['feature_extracts']==2 for r in updates)
        for n in (1,2):
            grads=saved[f'block{label}/gradients-after-clip-{n:04d}']
            assert len(grads)==18 and all(torch.isfinite(v).all() for v in grads.values())
            assert all(float(s['step'])==n for s in checkpoints[f'block{label}',n]['state'].values())
    assert all(torch.equal(v,before[k]) for k,v in core.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in core.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in core.modules())
    assert all(r['completed_updates']==0 for r in states if not r['zero_gate_passed'])


@pytest.mark.parametrize('kind',['shift','exception','nonfinite'])
def test_preupdate_failure_retains_completed_native_and_never_checkpoints(kind):
    core,data,make,native=tiny();saved={}
    def broken(index):
        bridge=make(index)
        class Proxy:
            adapter=bridge.adapter
            def __call__(self,*a,**kw):
                if kind=='exception':raise RuntimeError('injected bridge failure')
                value=bridge(*a,**kw).clone()
                value.reshape(-1)[0]=float('nan') if kind=='nonfinite' else value.reshape(-1)[0]+.001
                return value
        return Proxy()
    with pytest.raises((RuntimeError,ValueError,FloatingPointError)):
        engine.execute(data,placements=(0,1),make_bridge=broken,native_predict=native,
            retain=lambda name,v:saved.update({name:v}),checkpoint=lambda *a:pytest.fail('Updated before parity'),progress=lambda r:None)
    assert set(saved)>={'parity/native-positive','parity/native-negative'}
    if kind!='exception':assert 'parity/block0-positive-closed-full' in saved
    assert all(p.grad is None for p in core.parameters())


def test_exact_gate_is_stricter_than_old_numerical_tolerance():
    x=torch.ones(1,2,5,2,2);y=x.clone();y.reshape(-1)[0]+=torch.finfo(torch.float32).eps
    result=engine.exact_comparison(x,y)
    assert result['max_absolute']<1e-6 and not result['passed']
    assert engine.exact_comparison(x,x)['passed']


def test_wrong_schedule_rejected_before_predictions():
    _,data,make,native=tiny();data['schedule']=copy.deepcopy(data['schedule']);data['schedule'][1]['k']=378
    with pytest.raises(ValueError,match='draws 1 and 5'):
        engine.execute(data,placements=(0,1),make_bridge=make,native_predict=lambda *a:pytest.fail('Native called'),
                       retain=lambda *a:None,checkpoint=lambda *a:None,progress=lambda *a:None)


def test_hash_and_header_rejected_before_materializing(tmp_path,monkeypatch):
    path=tmp_path/'input.safetensors';save_file({'poison':torch.ones(1)},str(path))
    fake={'files':{'input.safetensors':{'bytes':path.stat().st_size,'sha256':sha(path),
              'header':{'expected':{'shape':[1],'dtype':'F32'}},'selected':{'expected':'0'*64}}}}
    monkeypatch.setattr(packet.old,'read_json',lambda p:fake)
    with pytest.raises(ValueError,match='names differ'):packet.read_selection(tmp_path)
    fake['files']['input.safetensors']['sha256']='0'*64
    monkeypatch.setattr(packet,'safe_open',lambda *a,**kw:pytest.fail('Opened hash-mismatched tensor'))
    with pytest.raises(ValueError,match='bytes differ'):packet.read_selection(tmp_path)


def parent_fixture(tmp_path,monkeypatch):
    root=tmp_path/'prepared';root.mkdir();atomic(root/'plan.json',{})
    admission=tmp_path/'admission.json';atomic(admission,{})
    plan={'source_sha256':{},'input_selection_sha256':'a'*64}
    monkeypatch.setattr(packet,'read_prepared',lambda p:(plan,{}))
    monkeypatch.setattr(packet,'admission',lambda *a:{'sha256':'b'*64})
    return root,admission


def test_exhausted_parent_deadline_never_spawns(tmp_path,monkeypatch):
    root,admission=parent_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(run,'_deadline',lambda *a:(_ for _ in ()).throw(RuntimeError('expired')))
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**kw:pytest.fail('Started expired child'))
    with pytest.raises(RuntimeError,match='expired'):run.execute(root,tmp_path,'fakeGPU',admission)
    assert packet.old.read_json(root/'metrics.json')['model_execution'] is False


def test_parent_preserves_primary_failure_and_partial_execution(tmp_path,monkeypatch):
    root,admission=parent_fixture(tmp_path,monkeypatch)
    class Process:returncode=1
    def launch(*args,**kwargs):
        assert kwargs['stdin']==run.subprocess.DEVNULL
        assert '--worker-config' in args[0]
        (root/'result').mkdir();atomic(root/'result/metrics.json',{'model_execution':True})
        return Process()
    monkeypatch.setattr(run.subprocess,'Popen',launch)
    monkeypatch.setattr(run,'supervise',lambda *a:(_ for _ in ()).throw(RuntimeError('original supervisor failure')))
    monkeypatch.setattr(run,'stop_child',lambda *a:(_ for _ in ()).throw(RuntimeError('cleanup failed')))
    with pytest.raises(RuntimeError,match='original supervisor failure'):run.execute(root,tmp_path,'fakeGPU',admission)
    report=packet.old.read_json(root/'metrics.json');terminal=packet.old.read_json(root/'terminal.json')
    assert report['model_execution'] is True and terminal['cleanup_error']=='cleanup failed'
    with pytest.raises(ValueError,match='Single-use'):run.execute(root,tmp_path,'fakeGPU',admission)


def test_admission_binds_plan_sources_inputs_limits_and_scope(tmp_path):
    atomic(tmp_path/'plan.json',{})
    plan={'source_sha256':{'x':'a'*64},'input_selection_sha256':'b'*64,'cpu_report_sha256':'c'*64}
    record=dict(schema=packet.SCHEMA,scope=packet.SCOPE,decision='admit',plan_sha256=sha(tmp_path/'plan.json'),
        **plan,expected_gpu='gpu',limits=run.limits('pair'),quality_admitted=False,image_generation=False)
    path=tmp_path/'admission.json';atomic(path,record)
    assert packet.admission(path,tmp_path,plan,'gpu')['sha256']==sha(path)
    for key,value in [('scope','quality'),('quality_admitted',True),('limits',{}),('plan_sha256','0'*64)]:
        bad=dict(record);bad[key]=value;atomic(path,bad)
        with pytest.raises(ValueError):packet.admission(path,tmp_path,plan,'gpu')


def test_review_rejects_contradictory_or_stale_results(tmp_path,monkeypatch):
    monkeypatch.setattr(packet,'sources',lambda:{'source':'a'*64})
    good=dict(status='passed',tests=10,source_sha256=packet.sources(),sources_unchanged=True,
              pytest_exit_code=0,failures=0,errors=0,skipped=0)
    path=tmp_path/'report.json';atomic(path,good)
    assert packet.review(path)==sha(path)
    for key,value in [('skipped',1),('pytest_exit_code',1),('source_sha256',{}),('sources_unchanged',False)]:
        atomic(path,dict(good,**{key:value}))
        with pytest.raises(ValueError):packet.review(path)
