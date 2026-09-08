import copy
import importlib.util
import os
from pathlib import Path
import sys
import pytest
import torch
import math_steps as m

CONTROL = Path(os.environ.get('WORLDLINE_COMMAND_CONTROLLER_SOURCE',str(Path(__file__).resolve().parent.parent/'command-attention-controller-v1')))
sys.path.insert(0,str(CONTROL))
spec=importlib.util.spec_from_file_location('controller_fixtures',CONTROL/'test_cpu.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setattr(fixture.vendor,'flash_attention',fixture.cpu_attention)
    yield
    torch.set_num_threads(old)

def data():
    bridge,(noise,_,contexts,_,observation)=fixture.case(active=True,padding=0)
    windows={}
    for motion,rate in [('stationary',0.),('left',.02617993877991494),('right',-.02617993877991494)]:
        for state,pulse in [('closed',0),('interact',1)]:
            commands=torch.zeros(1,16,6);commands[:,:,3]=rate;commands[:,0,5]=pulse
            target=noise.clone();target[:,:,1:]+=rate*5+pulse*.17
            windows[motion+'_'+state]=dict(target=target,observation=observation.clone(),commands=commands)
    return bridge,windows,noise,contexts[0],contexts[0].flip(0)+.1

def test_fixed_512_schedule_and_balanced_edges():
    rows=m.schedule([dict(update=i+1,k=506) for i in range(512)])
    assert sum(r['auxiliary_edge'] is not None for r in rows)==128
    assert [rows[i]['auxiliary_edge'] for i in range(0,28,4)]==[list(e) for e in m.EDGES]
    assert [sum(r['auxiliary_edge']==list(e) for r in rows) for e in m.EDGES]==[19,19,18,18,18,18,18]
    assert [sum(r['branches'][0]==s+'_closed' for r in rows) for s in m.MOTIONS]==[171,171,170]
    with pytest.raises(ValueError):m.schedule([dict(update=1,k=999)])

def test_auxiliary_future_conditioning_independent_of_target():
    _,windows,noise,_,_=data();names=m.EDGES[3]
    _,x,t,d=m.endpoint_inputs(windows,names,noise)
    changed=copy.deepcopy(windows);changed[names[1]]['target'][:,:,1:]+=9
    _,other,ot,od=m.endpoint_inputs(changed,names,noise)
    assert torch.equal(x,other) and torch.equal(t,ot) and not torch.equal(d,od)
    assert torch.equal(x[:,:,:1],windows[names[0]]['observation'])
    assert torch.equal(x[:,:,1:],noise[:,:,1:]) and set(t.flatten().tolist())=={0,999}

def test_contrast_sign_guidance_and_target_orientation():
    a=torch.tensor([1.,2.]);n=torch.tensor([.2,.3]);b=torch.tensor([3.,-1.]);bn=torch.tensor([.1,.4])
    assert torch.equal(m.contrast(a,n,b,bn),(n+5*(a-n))-(bn+5*(b-bn)))
    assert torch.equal(m.contrast(b,bn,a,n),-m.contrast(a,n,b,bn))

@pytest.mark.parametrize('edge',[m.EDGES[0],m.EDGES[3]])
def test_actual_native_update_matches_independent_combined_loss(edge):
    bridge,windows,noise,positive,negative=data()
    second=fixture.NativeCommandAttentionBridge(bridge.core,copy.deepcopy(bridge.controller),test_only=True,padding_tokens=0)
    opts=[torch.optim.AdamW(b.controller.parameters(),**m.OPTIMIZER) for b in (bridge,second)]
    row=dict(update=1,k=506,branches=['left_closed','left_interact'],auxiliary_edge=list(edge))
    retained={}
    report=m.update(bridge,windows,row,noise,positive,negative,opts[0],retain=lambda n,v:retained.update({n:v}))
    opts[1].zero_grad(set_to_none=True)
    objective=0
    for name in row['branches']:
        w=windows[name];x,t,v=m.flow_inputs(w['target'],w['observation'],noise,506)
        p=second(x,t,[positive],commands=w['commands'],observation=w['observation'])
        objective=objective+.5*(p[:,:,1:]-v[:,:,1:]).square().mean()
    pair,x,t,target=m.endpoint_inputs(windows,edge,noise)
    predictions=[]
    for w in pair:
        p=second(x,t,[positive],commands=w['commands'],observation=w['observation'])
        n=second(x,t,[negative],commands=w['commands'],observation=w['observation'])
        predictions.append(n+5*(p-n))
    difference=predictions[0]-predictions[1]
    objective=objective+(difference[:,:,1:]-target[:,:,1:]).square().mean()
    objective.backward();torch.nn.utils.clip_grad_norm_(second.controller.parameters(),1.);opts[1].step()
    assert report['total_objective']==pytest.approx(float(objective.detach()),rel=2e-6)
    for a,b in zip(bridge.controller.parameters(),second.controller.parameters()):
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=2e-7)
    assert len(retained)==7 and report['auxiliary']['predictions']==4
    fixture.clean(bridge);fixture.clean(second)

def test_no_auxiliary_and_reversed_edge_rejected_before_update():
    bridge,windows,noise,p,n=data();opt=torch.optim.AdamW(bridge.controller.parameters(),**m.OPTIMIZER)
    row=dict(update=2,k=506,branches=['left_closed','left_interact'],auxiliary_edge=None)
    r=m.update(bridge,windows,row,noise,p,n,opt)
    assert not r['auxiliary']['enabled'] and r['auxiliary']['predictions']==0
    before=[v.detach().clone() for v in bridge.controller.parameters()]
    row['auxiliary_edge']=list(reversed(m.EDGES[0]))
    with pytest.raises(ValueError,match='orientation'):m.update(bridge,windows,row,noise,p,n,opt)
    assert all(torch.equal(a,b) for a,b in zip(before,bridge.controller.parameters()))
