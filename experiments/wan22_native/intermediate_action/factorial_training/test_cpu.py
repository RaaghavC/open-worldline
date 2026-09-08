# SPDX-License-Identifier: Apache-2.0
"""Small CPU checks of the new schedule, saved inputs and failure boundaries."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
import training_inputs as packet
import training_math as training
import training_run as run
from experiments.wan22_native.official_cpu.streaming import sha,tensor_sha
from experiments.wan22_native.spatial_reference.guards import atomic
from experiments.wan22_native.intermediate_action.test_bridge import fixture,cpu_attention
from experiments.wan22_native.intermediate_action.cached_intermediate import CachedIntermediateActionBridge
from experiments.wan22_native.cuda_reference.vendor import model as vendor


@pytest.fixture(autouse=True)
def bounded(monkeypatch):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(vendor,'flash_attention',cpu_attention)
    yield
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(previous)


def windows(shape=(1,2,5,2,2),observation=None):
    observation=torch.zeros(shape[0],shape[1],1,shape[3],shape[4]) if observation is None else observation
    result={}
    for i,motion in enumerate(training.MOTIONS):
        for pulse in (0,1):
            commands=torch.zeros(1,16,6);commands[0,:,3]=(0.,math.pi/120,-math.pi/120)[i];commands[0,0,5]=pulse
            target=torch.full(shape,(i+pulse+1)/10);target[:,:,:1]=observation
            result[motion+('_interact' if pulse else '_closed')]=dict(target=target,observation=observation.clone(),commands=commands)
    return result


def synthetic_rows(noise):
    original=packet.catalog()['schedule'];result=copy.deepcopy(original)
    for row in result:row['noise_sha256']=tensor_sha(noise)
    return training.schedule(result)


def test_original_draw_schedule_and_all_six_arm_mapping():
    original=packet.catalog()['schedule'];rows=training.schedule(original)
    assert len(rows)==128 and [r['update'] for r in rows if r['auxiliary']]==list(range(1,129,4))
    assert {m:sum(r['motion']==m for r in rows) for m in training.MOTIONS}=={'stationary':43,'left':43,'right':42}
    assert {m:sum(r['motion']==m and r['auxiliary'] for r in rows) for m in training.MOTIONS}=={'stationary':11,'left':11,'right':10}
    for i,(old,new) in enumerate(zip(original,rows)):
        assert new['branches']==[training.MOTIONS[i%3]+'_'+a for a in ('closed','interact')]
        assert all(new[k]==old[k] for k in ('k','sigma','noise_key','noise_sha256','rng_after_sha256'))
    training.validate_windows(windows())


@pytest.mark.parametrize('kind',['pulse','yaw','prefix','extra'])
def test_factorial_input_corruption_rejected(kind):
    values=windows()
    if kind=='pulse':values['left_interact']['commands'][0,1,5]=1
    elif kind=='yaw':values['right_closed']['commands'][0,3,3]*=-1
    elif kind=='prefix':values['stationary_closed']['target'][0,0,0,0,0]+=1e-8
    else:values['left_closed']['pose']=torch.zeros(1)
    with pytest.raises(ValueError):training.validate_windows(values)


class FakeBridge:
    def __init__(self):
        self.core=torch.nn.Linear(1,1).requires_grad_(False)
        self.adapter=torch.nn.Module();self.adapter.output=torch.nn.Linear(1,1);self.adapter.command_gru=torch.nn.Linear(1,1)
        torch.nn.init.zeros_(self.adapter.output.weight);torch.nn.init.zeros_(self.adapter.output.bias)
    def __call__(self,x,*args,**kwargs):return torch.zeros_like(x)


def test_128_schedule_invokes_original_update_and_retains_exact_counts(monkeypatch):
    bridge=FakeBridge();values=windows();noise=torch.zeros_like(values['stationary_closed']['target']);rows=synthetic_rows(noise)
    saved=set();cps=[];called=[];optimizer=torch.optim.AdamW(bridge.adapter.parameters(),**run.OPTIMIZER)
    def retain(name,values):
        assert name not in saved;saved.add(name)
    def stub(proxy,pair,epsilon,k,ctx,opt,*,auxiliary,negative_context,retain,update,check):
        row=rows[update-1];called.append(row)
        assert pair[0] is values[row['branches'][0]] and pair[1] is values[row['branches'][1]]
        assert epsilon is noise and k==row['k'] and opt is optimizer and auxiliary==row['auxiliary']
        for p in proxy.adapter.parameters():p.grad=torch.ones_like(p)
        for window in pair:proxy(noise,None,None)
        if auxiliary:
            for a in ('closed','open'):
                for label in ('positive','negative'):retain(f'auxiliary-{update:04d}-{a}-{label}',{'velocity':noise})
        return dict(command_gru_gradient_l2=0. if update==1 else .1,
                    auxiliary=dict(head_predictions=4 if auxiliary else 0,feature_extracts=2 if auxiliary else 0))
    monkeypatch.setattr(training.effect,'paired_update',stub)
    result=training.run(bridge,values,rows,lambda r:noise,torch.zeros(1,1),torch.ones(1,1),optimizer,
        initial={k:v.detach().clone() for k,v in bridge.adapter.state_dict().items()},retain=retain,
        checkpoint=lambda n,r:cps.append(n),progress=lambda r:None,native_predict=lambda x,t,c:torch.zeros_like(x))
    assert called==rows and cps==list(range(0,129,16))
    assert result['completed_updates']==128 and result['main_predictions']==256 and result['auxiliary_predictions']==128
    assert len(saved)==4+256+128+128


def test_two_literal_native_suffix_updates_reuse_unchanged_loss():
    b,x,t,c,commands,obs=fixture()
    bridge=CachedIntermediateActionBridge(b.core,b.adapter,block_index=0,test_only=True)
    values=windows(tuple(x.shape),obs);noise=torch.full_like(x,.3);rows=synthetic_rows(noise)
    optimizer=torch.optim.AdamW(bridge.adapter.parameters(),**run.OPTIMIZER);saved=set();latest={}
    before={k:v.detach().clone() for k,v in b.core.state_dict().items()}
    class StopAfterTwo(Exception):pass
    def getter(row):
        if row['update']==3:raise StopAfterTwo()
        return noise
    def progress(report):latest.update(copy.deepcopy(report))
    def native(x,t,c):
        with torch.inference_mode(False),torch.no_grad():return torch.stack(b.core(list(x.unbind(0)),t,[c],t.shape[1]))
    with pytest.raises(StopAfterTwo):
        training.run(bridge,values,rows,getter,c[0],c[0]+.2,optimizer,
            initial={k:v.detach().clone() for k,v in b.adapter.state_dict().items()},
            retain=lambda name,v:saved.add(name),checkpoint=lambda n,r:None,progress=progress,native_predict=native)
    assert latest['completed_updates']==2 and all(r['exact_equal'] for r in latest['parity'])
    assert latest['updates'][0]['command_gru_gradient_l2']==0 and latest['updates'][1]['command_gru_gradient_l2']>0
    assert [r['auxiliary']['head_predictions'] for r in latest['updates']]==[4,0]
    assert all(float(s['step'])==2 for s in optimizer.state.values())
    assert all(torch.equal(v,before[k]) for k,v in b.core.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in b.core.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in b.core.modules())


def test_parity_failure_retained_before_any_optimizer_update(monkeypatch):
    bridge=FakeBridge();values=windows();noise=torch.zeros_like(values['stationary_closed']['target']);saved=set()
    monkeypatch.setattr(training.effect,'paired_update',lambda *a,**kw:pytest.fail('Update before exact parity'))
    with pytest.raises(RuntimeError,match='parity'):
        training.run(bridge,values,synthetic_rows(noise),lambda r:noise,torch.zeros(1,1),torch.ones(1,1),
            torch.optim.AdamW(bridge.adapter.parameters(),**run.OPTIMIZER),
            initial={k:v.detach().clone() for k,v in bridge.adapter.state_dict().items()},
            retain=lambda name,v:saved.add(name),checkpoint=lambda *a:pytest.fail('Checkpoint before gate'),
            progress=lambda r:None,native_predict=lambda x,t,c:torch.full_like(x,1e-8))
    assert saved=={'parity-stationary_closed-native','parity-stationary_closed-bridge'}


def test_wrong_auxiliary_schedule_rejected_before_model():
    bridge=FakeBridge();values=windows();noise=torch.zeros_like(values['stationary_closed']['target']);rows=synthetic_rows(noise)
    rows[0]['auxiliary']=False;rows[1]['auxiliary']=True
    with pytest.raises(ValueError,match='schedule'):
        training.run(bridge,values,rows,lambda r:noise,torch.zeros(1,1),torch.ones(1,1),
            torch.optim.AdamW(bridge.adapter.parameters()),initial={},retain=lambda *a:None,checkpoint=lambda *a:None,
            progress=lambda r:None,native_predict=lambda *a:pytest.fail('Wrong schedule model call'))


def tensor_record(path,values):
    save_file({k:v.clone() for k,v in values.items()},str(path))
    return dict(bytes=path.stat().st_size,sha256=sha(path),header={k:dict(shape=list(v.shape),dtype='F32' if v.dtype==torch.float32 else 'U8') for k,v in values.items()})


def test_saved_noise_rng_loaded_without_regeneration_and_corruption_fails(tmp_path,monkeypatch):
    noise=torch.arange(40,dtype=torch.float32).reshape(1,2,5,2,2);rows=synthetic_rows(noise);rng=torch.tensor([1,2,3],dtype=torch.uint8)
    values={'rng_initial':rng}
    for row in rows[:16]:
        row['rng_after_sha256']=tensor_sha(rng);values[row['noise_key']]=noise;values[f'rng_after_{row["update"]-1:04d}']=rng
    name='draws-0000-0015.safetensors';record=tensor_record(tmp_path/name,values)
    monkeypatch.setattr(torch,'randn',lambda *a,**kw:pytest.fail('Regenerated Gaussian noise'))
    draws=packet.Draws(tmp_path,{name:record},rows)
    assert torch.equal(draws.noise(rows[0]),noise) and torch.equal(draws.rng(0),rng) and torch.equal(draws.rng(16),rng)
    bad=copy.deepcopy(rows);bad[0]['noise_sha256']='0'*64
    with pytest.raises(ValueError,match='noise'):packet.Draws(tmp_path,{name:record},bad).noise(bad[0])


def test_hash_and_tensor_header_checked_before_materialization(tmp_path,monkeypatch):
    path=tmp_path/'x.safetensors';record=tensor_record(path,{'actual':torch.ones(1)})
    wrong=dict(record,sha256='0'*64)
    with monkeypatch.context() as patch:
        patch.setattr(packet,'safe_open',lambda *a,**kw:pytest.fail('Read changed file'))
        with pytest.raises(ValueError):packet.checked_tensors(path,wrong)
    wrong=dict(record,header={'expected':dict(shape=[1],dtype='F32')})
    with pytest.raises(ValueError,match='names'):packet.checked_tensors(path,wrong)


def test_current_actual_receipts_and_input_catalog_bind_to_sources():
    assert packet.catalog()['original_plan']['sha256']=='bd6dc4082ddf1f19033f8b967eaf5cb3356d6e0484530e039e250d68a24c68e4'
    receipts=packet.receipts_at(packet.HERE)
    assert receipts['cache']['all196_unchanged'] and receipts['profile']['exact_zero_comparisons']==16


def test_admission_and_cpu_report_reject_stale_or_relaxed_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(packet,'sources',lambda:{'source':'a'*64})
    good=dict(status='passed',tests=14,source_sha256=packet.sources(),sources_unchanged=True,pytest_exit_code=0,failures=0,errors=0,skipped=0)
    path=tmp_path/'cpu.json';atomic(path,good);assert packet.review(path)==sha(path)
    atomic(path,dict(good,skipped=1))
    with pytest.raises(ValueError):packet.review(path)
    atomic(tmp_path/'plan.json',{})
    plan=dict(source_sha256={},prior_receipt_sha256=packet.RECEIPTS,cpu_report_sha256='c'*64)
    admission=dict(schema=packet.SCHEMA,scope=packet.SCOPE,decision='admit',plan_sha256=sha(tmp_path/'plan.json'),
                   **plan,expected_gpu='fakeGPU',limits=run.limits('pair'),image_generation=False)
    atomic(path,admission);assert packet.admission(path,tmp_path,plan,'fakeGPU')
    for key,value in [('limits',dict(run.limits('pair'),seconds=1000)),('prior_receipt_sha256',{}),('scope','quality')]:
        atomic(path,dict(admission,**{key:value}))
        with pytest.raises(ValueError):packet.admission(path,tmp_path,plan,'fakeGPU')


def parent_fixture(tmp_path,monkeypatch):
    root=tmp_path/'prepared';root.mkdir();atomic(root/'plan.json',{})
    admission=tmp_path/'admission.json';atomic(admission,{})
    plan={'source_sha256':{},'prior_receipt_sha256':packet.RECEIPTS}
    monkeypatch.setattr(packet,'read_prepared',lambda p:(plan,{}))
    monkeypatch.setattr(packet,'admission',lambda *a:{'sha256':'b'*64})
    return root,admission


def test_exhausted_deadline_never_launches_child(tmp_path,monkeypatch):
    root,admission=parent_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(run,'_deadline',lambda *a:(_ for _ in ()).throw(RuntimeError('expired')))
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**kw:pytest.fail('Started expired child'))
    with pytest.raises(RuntimeError,match='expired'):run.execute(root,tmp_path,'fakeGPU',admission)
    assert packet.evidence.read_json(root/'metrics.json')['model_execution'] is False


def test_parent_keeps_original_error_and_partial_model_status(tmp_path,monkeypatch):
    root,admission=parent_fixture(tmp_path,monkeypatch)
    class Process:returncode=1
    def launch(*args,**kwargs):
        assert kwargs['stdin']==run.subprocess.DEVNULL and '--worker-config' in args[0]
        (root/'result').mkdir();atomic(root/'result/metrics.json',{'model_execution':True});return Process()
    monkeypatch.setattr(run.subprocess,'Popen',launch)
    monkeypatch.setattr(run,'supervise',lambda *a:(_ for _ in ()).throw(RuntimeError('original failure')))
    monkeypatch.setattr(run,'stop_child',lambda *a:(_ for _ in ()).throw(RuntimeError('cleanup failed')))
    with pytest.raises(RuntimeError,match='original failure'):run.execute(root,tmp_path,'fakeGPU',admission)
    assert packet.evidence.read_json(root/'metrics.json')['model_execution'] is True
    assert packet.evidence.read_json(root/'terminal.json')['cleanup_error']=='cleanup failed'
    with pytest.raises(ValueError,match='Single-use'):run.execute(root,tmp_path,'fakeGPU',admission)
