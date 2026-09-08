# SPDX-License-Identifier: Apache-2.0
"""Small orchestration/corruption checks; no model weights or CUDA runtime."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from types import SimpleNamespace
import copy,json,os,sys
import pytest
import torch
CONTROLLER_SOURCE=Path(os.environ.get("WORLDLINE_COMMAND_CONTROLLER_SOURCE", Path(__file__).resolve().parent.parent/"command-attention-controller-v1")).resolve()
sys.path.insert(0,str(CONTROLLER_SOURCE))
import math_steps
import run_training as r


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_deadline_projection_and_recovery_boundary():
    now=datetime(2026,9,8,tzinfo=timezone.utc)
    stamp=lambda seconds:(now+timedelta(seconds=seconds)).isoformat()
    assert r.lease_budget(stamp(2400),1800,now)==1800
    assert r.lease_budget(stamp(1600),1000,now)==1000
    for seconds,estimate in ((2399,1800),(2000,1801),(3601,1000),(600,1),(2000,True),(2000,float('nan'))):
        with pytest.raises(ValueError):r.lease_budget(stamp(seconds),estimate,now)
    with pytest.raises(ValueError):r.lease_budget('2026-09-08T01:00:00',1000,now)


def profile_fixture(tmp_path):
    p=tmp_path/'profile';out=p/'result';out.mkdir(parents=True)
    for name in ('weight-load.json','core-before.json','core-after-updates.json'):
        write(out/name,{'fixture':name})
    write(out/'monitor-terminal.json',dict(status='complete',error=None))
    (out/'memory.jsonl').write_text('{}\n')
    write(p/'terminal.json',dict(status='complete',exit_code=0,cleanup_error=None))
    sources={'controller/controller.py':'a'*64,'training/math_steps.py':'b'*64,'training/packet.py':'c'*64}
    write(p/'launch.json',dict(source_sha256=sources))
    parity=[dict(context=c,arm=a,path=mode,exact_equal=True,seconds=.1,feature_extract_seconds=.2)
            for c in ('positive','negative') for a in r.ARMS for mode in ('full','cached')]
    updates=[dict(update=i,optimizer_updates=1,main_predictions=2,auxiliary=dict(predictions=4 if i==1 else 0),
                  gradients=dict(all_present_finite=True),seconds=1.,profile_seconds=1.1,memory=dict(peak_reserved_bytes=25*2**30)) for i in (1,2)]
    result=dict(schema='worldline-command-attention-profile-v1',status='passed',model_execution=True,completed_updates=2,
        inputs_sha256=r.INPUT_SHA,zero_gate_passed=True,base_unchanged=True,all825_current_value_hashes_verified=True,
        rotary_unchanged=True,sources_unchanged=True,inputs_unchanged=True,parity=parity,updates=updates,source_sha256=sources,
        output_sha256={f.name:r.sha(f) for f in out.iterdir()},hardware={'name':'CPU fixture'},runtime_flags={},
        initial_controller_sha256='d'*64,load_seconds=100.,foundation_hash_seconds=dict(before=40.,**{'after-updates':40.}))
    write(out/'metrics.json',result)
    return p,sources,result


def test_profile_receipt_projection_and_concrete_corruption(tmp_path):
    p,sources,original=profile_fixture(tmp_path)
    receipt=r.profile_receipt(p,r.sha(p/'result/metrics.json'),sources,r.INPUT_SHA)
    assert receipt['projection']['seconds']==pytest.approx(1.2*(512*1.1+16*.2+96*.1+100+40+40)+120)
    mutations=[lambda v:v['parity'].__setitem__(0,dict(v['parity'][0],exact_equal=False)),
        lambda v:v['parity'].__setitem__(0,v['parity'][1]),lambda v:v['updates'][0]['auxiliary'].__setitem__('predictions',2),
        lambda v:v['updates'][0]['memory'].__setitem__('peak_reserved_bytes',61*2**30),
        lambda v:v.__setitem__('inputs_sha256','wrong'),lambda v:v['source_sha256'].__setitem__('controller/controller.py','e'*64),
        lambda v:v['output_sha256'].__setitem__('core-after-updates.json','f'*64)]
    for mutate in mutations:
        value=copy.deepcopy(original);mutate(value);write(p/'result/metrics.json',value)
        with pytest.raises(ValueError):r.profile_receipt(p,r.sha(p/'result/metrics.json'),sources,r.INPUT_SHA)
    write(p/'result/metrics.json',original)
    with pytest.raises(ValueError):r.profile_receipt(p,'0'*64,sources,r.INPUT_SHA)


def test_admission_and_gpu_capacity_comparison(tmp_path):
    plan=dict(source_sha256={'a':'b'},resource_profile=dict(result_sha256='p'),expected_gpu='GPU',lease_deadline_utc='date')
    value=dict(schema=r.SCHEMA,approved=True,scope=r.SCOPE,plan_sha256='h',inputs_sha256=r.INPUT_SHA,
        source_sha256=plan['source_sha256'],resource_profile_sha256='p',expected_gpu='GPU',lease_deadline_utc='date',recovery_reserve_seconds=600)
    p=tmp_path/'admission.json';write(p,value)
    assert r.admission(p,plan,'h')['record']==value
    value['recovery_reserve_seconds']=599;write(p,value)
    with pytest.raises(ValueError):r.admission(p,plan,'h')
    old=dict(name='GPU',total_memory_bytes=80,flags={'enabled':False})
    r.hardware_matches(dict(old,total_memory_bytes=82),old)
    for changed in (dict(old,total_memory_bytes=79),dict(old,total_memory_bytes=80.),dict(old,name='other'),dict(old,flags={'enabled':0})):
        with pytest.raises(ValueError):r.hardware_matches(changed,old)


def test_score_contrast_future_only_normalization_and_sign():
    target=torch.tensor([0.,2.,4.,-2.,1.]).reshape(1,1,5,1,1)
    predicted=target*.25;predicted[:,:,0]=999
    actual=r.score_contrast(predicted,target)
    assert actual['normalized_mse']==pytest.approx(.75**2) and actual['cosine']==pytest.approx(1)
    assert r.score_contrast(-predicted,target)['normalized_mse']==pytest.approx(1.25**2)
    assert r.score_contrast(predicted,torch.zeros_like(target))['normalized_mse'] is None
    predicted[:,:,2]=float('nan')
    with pytest.raises(FloatingPointError):r.score_contrast(predicted,target)


def test_full512_orchestration_exact_counts_raw_selection_and_final_only(monkeypatch):
    schedule=math_steps.schedule([dict(update=i+1,k=506) for i in range(512)])
    data=SimpleNamespace(plan={'schedule':schedule},windows={},positive=None,negative=None,noise=lambda row:row['update'])
    kept=[];cp=[];evaluated=[];progress=[];calls=[]
    def step(bridge,windows,row,noise,p,n,opt,*,check,retain,synchronize):
        assert noise==row['update'];calls.append(row)
        for arm in row['branches']:retain('main-'+arm,{})
        enabled=row['auxiliary_edge'] is not None
        if enabled:
            for i in (0,1):
                for context in ('positive','negative'):retain('aux-'+str(i)+'-'+context,{})
        retain('gradients',{})
        return dict(update=row['update'],optimizer_updates=1,main_predictions=2,auxiliary=dict(predictions=4*enabled,feature_extracts=2*enabled))
    def evaluate(bridge,data,label,*args):evaluated.append(label);return dict(predictions=48)
    monkeypatch.setattr(math_steps,'update',step)
    result=r.train_loop(None,data,SimpleNamespace(zero_grad=lambda **kw:None),retain=lambda name,values:kept.append(name),
        checkpoint=lambda i,row:cp.append(i),progress=lambda v:progress.append(v['completed_updates']),evaluator=evaluate)
    assert len(calls)==512 and len(result['updates'])==512 and result['main_predictions']==1024
    assert result['auxiliary_predictions']==512 and result['auxiliary_feature_extracts']==256 and result['auxiliary_updates']==128
    assert cp==[0,128,256,384,512] and evaluated==['initial','final']
    assert sorted({int(n.split('/')[1].split('-')[1]) for n in kept})==list(r.RAW_UPDATES)
    assert len(kept)==27 and progress[-1]==512


def test_interrupted_loop_retains_completed_scalars_without_final_evaluation(monkeypatch):
    schedule=math_steps.schedule([dict(update=i+1,k=506) for i in range(512)])
    data=SimpleNamespace(plan={'schedule':schedule},windows={},positive=None,negative=None,noise=lambda row:None)
    records=[];evaluations=[]
    def step(bridge,windows,row,*args,**kwargs):
        if row['update']==3:raise KeyboardInterrupt('injected')
        return dict(update=row['update'],optimizer_updates=1,main_predictions=2,auxiliary=dict(predictions=4 if row['update']==1 else 0,feature_extracts=2 if row['update']==1 else 0))
    monkeypatch.setattr(math_steps,'update',step)
    with pytest.raises(KeyboardInterrupt):r.train_loop(None,data,None,retain=lambda *a:None,checkpoint=lambda *a:None,
        progress=lambda v:records.append(v['completed_updates']),evaluator=lambda b,d,label,*args:evaluations.append(label))
    assert records==[0,1,2] and evaluations==['initial']


def test_parent_cleanup_retains_partial_worker_status(tmp_path,monkeypatch):
    from experiments.wan22_native.spatial_reference import guards
    folders={k:str(tmp_path/k) for k in ('repository','controller_source','inputs','resource_profile')}
    for f in folders.values():Path(f).mkdir()
    plan=dict(execution_paths=folders,lease_deadline_utc=(datetime.now(timezone.utc)+timedelta(seconds=3000)).isoformat(),
        resource_profile=dict(projection=dict(seconds=900),result_sha256='p'),source_sha256={})
    p=tmp_path/'plan.json';write(p,plan);a=tmp_path/'admission.json';write(a,{})
    out=tmp_path/'execution';admitted={'sha256':r.sha(a),'record':{}}
    monkeypatch.setattr(r,'bootstrap',lambda *a:None);monkeypatch.setattr(r,'read_plan',lambda *a:(plan,SimpleNamespace(validate_all=lambda:None)))
    monkeypatch.setattr(r,'admission',lambda *a:admitted);monkeypatch.setattr(r,'source_paths',lambda *a:{})
    proc=SimpleNamespace(returncode=-15);stopped=[];starts=[]
    def popen(*args,**kwargs):
        assert kwargs['stdin']==r.subprocess.DEVNULL;starts.append(args)
        write(out/'result/metrics.json',{'model_execution':True,'status':'running'});return proc
    monkeypatch.setattr(r.subprocess,'Popen',popen)
    def fail(*args):raise KeyboardInterrupt('supervision interrupt')
    monkeypatch.setattr(guards,'supervise',fail);monkeypatch.setattr(guards,'stop_child',lambda p:stopped.append(p))
    with pytest.raises(KeyboardInterrupt):r.execute(p,r.sha(p),a,tmp_path/'weights',out)
    assert len(starts)==len(stopped)==1
    parent=r.read(out/'metrics.json');assert parent['status']=='interrupted' and parent['model_execution'] is True
    assert r.read(out/'terminal.json')['status']=='failed' and (out/'launch.json').exists()
    with pytest.raises(ValueError):r.execute(p,r.sha(p),a,tmp_path/'weights',out)


def test_expired_parent_prelaunch_never_starts_child(tmp_path,monkeypatch):
    from experiments.wan22_native.spatial_reference import guards
    folders={k:str(tmp_path/k) for k in ('repository','controller_source','inputs','resource_profile')}
    for f in folders.values():Path(f).mkdir()
    plan=dict(execution_paths=folders,lease_deadline_utc=(datetime.now(timezone.utc)+timedelta(seconds=3000)).isoformat(),
        resource_profile=dict(projection=dict(seconds=900),result_sha256='p'),source_sha256={})
    p=tmp_path/'plan.json';write(p,plan);a=tmp_path/'admission.json';write(a,{})
    monkeypatch.setattr(r,'bootstrap',lambda *a:None);monkeypatch.setattr(r,'read_plan',lambda *a:(plan,SimpleNamespace(validate_all=lambda:None)))
    monkeypatch.setattr(r,'admission',lambda *a:{});monkeypatch.setattr(r,'source_paths',lambda *a:{})
    monkeypatch.setattr(guards,'_deadline',lambda *a:(_ for _ in ()).throw(RuntimeError('expired')))
    monkeypatch.setattr(r.subprocess,'Popen',lambda *a,**kw:pytest.fail('worker must not start'))
    with pytest.raises(RuntimeError,match='expired'):r.execute(p,r.sha(p),a,tmp_path/'weights',tmp_path/'execution')
    assert r.read(tmp_path/'execution/metrics.json')['model_execution'] is False


def test_evaluator_real_tiny_native_all48_and_independent_scores(monkeypatch):
    # Reuse the root's literal tiny native fixture, no CUDA or weight load.
    import test_math
    old=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setattr(test_math.fixture.vendor,'flash_attention',test_math.fixture.cpu_attention)
    try:
        bridge,windows,noise,positive,negative=test_math.data()
        data=SimpleNamespace(windows=windows,positive=positive,negative=negative,evaluation={f'noise_{i:02d}':noise+i*.01 for i in range(4)})
        retained={}
        report=r.evaluate(bridge,data,'final',lambda name,values:retained.update({name:values['velocity'].detach().clone()}))
        assert len(retained)==48 and report['predictions']==48 and report['feature_extracts']==8 and len(report['scores'])==28
        for row in report['scores']:
            a,b=row['edge'];prefix='evaluation/final/'+row['noise']+'/'
            p=lambda arm,c:retained[prefix+arm+'-'+c]
            contrast=-((p(b,'negative')+5*(p(b,'positive')-p(b,'negative')))-(p(a,'negative')+5*(p(a,'positive')-p(a,'negative'))))
            target=windows[b]['target']-windows[a]['target']
            actual=((contrast[:,:,1:].double()-target[:,:,1:].double())**2).mean().item()
            assert row['future_mse']==actual
        assert all(not p.requires_grad and p.grad is None for p in bridge.core.parameters())
    finally:torch.set_num_threads(old)
