"""Tiny CPU derivative and update checks. No pretrained values or CUDA."""
import copy
from unittest.mock import patch
import torch
import pytest
import effect as e
import prototype as p
from experiments.wan22_native.action_cuda.test_cpu import fixture


def case():
    torch.set_num_threads(1)
    bridge,target,_,contexts,_,observation=fixture()
    adapter=p.original.fresh_adapter(test_configuration={'hidden_dim':8,'observation_channels':2,'width':8})
    bridge.adapter=adapter
    windows=[]
    for arm in ('closed','open'):
        value=target.clone()
        if arm=='open':value[:,:,1:]*=.7
        windows.append({'target':value,'observation':observation.clone(),'commands':p.expected_commands(arm,0)})
    noise=torch.randn(target.shape,generator=torch.Generator().manual_seed(812))
    negative=contexts[0].flip(0).clone()+.02
    return bridge,windows,noise,contexts[0],negative


def test_endpoint_signs_prefix_and_loss_scale():
    shape=(1,2,5,2,2)
    pc,nc,po,no=[torch.full(shape,float(i),requires_grad=True) for i in (1,2,3,4)]
    predicted=e.endpoint_contrast(pc,nc,po,no)
    assert torch.equal(predicted,torch.full(shape,-2.))
    loss=e.endpoint_loss(predicted,torch.zeros(shape))
    assert loss.item()==4.
    predicted[:,:,1:].sum().backward()
    for value,sign in zip((pc,nc,po,no),(5.,-4.,-5.,4.)):
        assert torch.count_nonzero(value.grad[:,:,:1])==0
        assert torch.equal(value.grad[:,:,1:],torch.full_like(value.grad[:,:,1:],sign))
    changed=predicted.detach().clone();changed[:,:,:1]=123456.
    assert e.endpoint_loss(changed,torch.zeros(shape)).item()==4.
    assert e.LAMBDA==1. and e.ENDPOINT_SCALE==1.


def test_auxiliary_has_no_future_target_in_condition_and_no_new_draw():
    bridge,windows,noise,_,_=case();before=noise.clone();rng=torch.get_rng_state().clone()
    x,t,d=e.auxiliary_inputs(windows,noise)
    assert torch.equal(x[:,:,1:],noise[:,:,1:]) and torch.equal(x[:,:,:1],windows[0]['observation'])
    prefix=bridge.tokens//5
    assert torch.all(t[:,:prefix]==0) and torch.all(t[:,prefix:]==999)
    changed=copy.deepcopy(windows);changed[1]['target'][:,:,1:]+=7
    y,ty,dy=e.auxiliary_inputs(changed,noise)
    assert torch.equal(x,y) and torch.equal(t,ty) and not torch.equal(d,dy)
    assert torch.equal(noise,before) and torch.equal(torch.get_rng_state(),rng)
    changed[1]['observation']+=.1
    with pytest.raises(ValueError):e.auxiliary_inputs(changed,noise)


def test_four_head_paths_use_two_frozen_feature_extracts():
    bridge,windows,noise,positive,negative=case();before=p.original.parameter_records(bridge.core,expected_count=len(list(bridge.core.parameters())))
    retained={};features=[];heads=[]
    original_extract=bridge.extract_features;original_predict=bridge.predict_from_features
    def extract(*args,**kwargs):
        f=original_extract(*args,**kwargs);features.append(f)
        assert not f.hidden.requires_grad and not f.time_embedding.requires_grad
        return f
    def head(f,*args,**kwargs):heads.append(f);return original_predict(f,*args,**kwargs)
    with patch.object(bridge,'extract_features',side_effect=extract),patch.object(bridge,'predict_from_features',side_effect=head):
        row=e.auxiliary_backward(bridge,windows,noise,positive,negative,retain=lambda n,v:retained.update({n:v['velocity'].detach().clone()}),update=1,check=lambda:None)
    assert len(features)==2 and len(heads)==4 and heads[0] is heads[1] and heads[2] is heads[3] and heads[0] is not heads[2]
    assert len(retained)==4 and row['feature_extracts']==2 and row['head_predictions']==4
    assert row['future_clean_difference_mse']>0 and row['target_conditioning'] is False
    assert all(v.grad is None and not v.requires_grad for v in bridge.core.parameters())
    assert before==p.original.parameter_records(bridge.core,expected_count=len(before))
    assert sum(float(v.grad.square().sum()) for v in bridge.adapter.parameters() if v.grad is not None)>0


def test_disabled_auxiliary_matches_original_update_bit_exact():
    first=case();second=case();second[0].core.load_state_dict(first[0].core.state_dict());second[0].adapter.load_state_dict(first[0].adapter.state_dict())
    bridge,windows,noise,positive,_=first;other=second[0]
    one=torch.optim.AdamW(bridge.adapter.parameters(),**p.original.OPTIMIZER);two=torch.optim.AdamW(other.adapter.parameters(),**p.original.OPTIMIZER)
    a=p.numerical.paired_update(bridge,windows,noise,506,positive,one)
    b=e.paired_update(other,windows,noise,506,positive,two,auxiliary=False)
    assert all(a[k]==b[k] for k in a)
    for x,y in zip(bridge.adapter.parameters(),other.adapter.parameters()):assert torch.equal(x,y) and torch.equal(x.grad,y.grad)
    for x,y in zip(one.state.values(),two.state.values()):assert all(torch.equal(x[k],y[k]) for k in x)


def test_auxiliary_accumulates_before_one_clip_and_one_step():
    bridge,windows,noise,positive,negative=case();optimizer=torch.optim.AdamW(bridge.adapter.parameters(),**p.original.OPTIMIZER)
    calls=[];clip=torch.nn.utils.clip_grad_norm_;step=optimizer.step
    def clipping(*args,**kwargs):calls.append('clip');return clip(*args,**kwargs)
    def stepping(*args,**kwargs):calls.append('step');return step(*args,**kwargs)
    def retained(name,value):calls.append(name)
    with patch.object(torch.nn.utils,'clip_grad_norm_',side_effect=clipping),patch.object(optimizer,'step',side_effect=stepping):
        row=e.paired_update(bridge,windows,noise,506,positive,optimizer,auxiliary=True,negative_context=negative,retain=retained,update=1)
    assert len(calls)==6 and calls[-2:]==['clip','step'] and all(x.startswith('auxiliary-0001-') for x in calls[:4])
    assert row['optimizer_updates']==1 and all(v['step']==1 for v in optimizer.state.values())
    assert row['total_objective']==row['paired_mean_future_flow_mse']+row['auxiliary']['future_clean_difference_mse']
    assert row['gradient_l2_after_clip']<=1.000001 and all(v.grad is None for v in bridge.core.parameters())
