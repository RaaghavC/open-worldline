"""Tiny CPU causal-state fixtures; no real checkpoint execution."""
import inspect
from unittest import mock

import pytest
import torch
from torch import nn

from experiments.room_world import memory_state_diagnostic as diagnostic


class StateFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker=nn.Parameter(torch.zeros(()))
        self.calls=[]
    def initial_state(self,batch):
        return torch.zeros(batch,32)
    def forward(self,history,action,state,*,mode='carry'):
        incoming=state.clone()
        used=torch.zeros_like(state) if mode=='reset' else state
        following=used*.5+.04+action[:,None].float()*.01
        prediction=(history[:,-1]+following[:,0,None,None,None]*.03).clamp(-1,1)
        self.calls.append((history.clone(),action.clone(),incoming,prediction.clone(),following.clone(),mode))
        return prediction,following


def inputs():
    prefix=torch.zeros(2,7,3,4,4)
    commands=torch.zeros(2,6,dtype=torch.int64);commands[1,0]=5
    return prefix,commands,torch.full((2,4),4,dtype=torch.int64)


def test_boundary_actions_are_applied_once_before_return_and_inputs_remain_owned():
    model=StateFixture();prefix,commands,returns=inputs();saved=prefix.clone()
    values=diagnostic.intervene(model,prefix,commands,returns)
    assert len(model.calls)==6+3*4
    state=torch.zeros(2,32)
    for t in range(6):
        h,a,s,_,after,_=model.calls[t]
        assert torch.equal(a,commands[:,t]) and torch.equal(s,state)
        assert torch.equal(h,diagnostic.history_at(prefix,t))
        state=state*.5+.04+commands[:,t,None]*.01
        assert torch.equal(after,state)
    assert torch.equal(values['state_normal'],state)
    assert torch.equal(values['state_swapped'],state.flip(0))
    assert torch.equal(values['state_zero'],torch.zeros_like(state))
    assert torch.equal(prefix,saved)
    for name in ('normal','swapped','zero'):
        assert values[f'state_{name}'].data_ptr()!=values['boundary_history'].data_ptr()
    assert values['state_normal'].data_ptr()!=values['state_swapped'].data_ptr()


def test_every_return_recursively_uses_generated_rgb_with_same_actions():
    model=StateFixture();prefix,commands,returns=inputs()
    with mock.patch.object(diagnostic,'score_return',side_effect=AssertionError('Truth scoring cannot run inside generation')):
        values=diagnostic.intervene(model,prefix,commands,returns)
    for arm,name in enumerate(diagnostic.INTERVENTIONS):
        history=values['boundary_history'].clone()
        state=values[f'state_{name}'].clone()
        for t in range(4):
            h,a,s,prediction,next_state,_=model.calls[6+arm*4+t]
            assert torch.equal(h,history) and torch.equal(a,returns[:,t]) and torch.equal(s,state)
            assert torch.equal(prediction,values[f'prediction_{name}'][:,t])
            history=torch.cat((history[:,1:],prediction[:,None]),dim=1);state=next_state
    assert torch.equal(values['prediction_swapped'],values['prediction_normal'].flip(0))
    assert not torch.equal(values['prediction_normal'],values['prediction_zero'])
    assert torch.equal(values['prediction_zero'][0],values['prediction_zero'][1])
    assert 'truth' not in inspect.signature(diagnostic.intervene).parameters


def test_reset_discards_nonzero_state_under_both_interventions():
    prefix,commands,returns=inputs();values=diagnostic.intervene(StateFixture(),prefix,commands,returns,mode='reset')
    assert torch.count_nonzero(values['state_normal'])>0
    assert torch.equal(values['prediction_normal'],values['prediction_swapped'])
    assert torch.equal(values['prediction_normal'],values['prediction_zero'])


def test_alias_mismatch_is_rejected_and_truth_is_used_only_for_postgeneration_scores():
    prefix,commands,returns=inputs();bad=prefix.clone();bad[1,-1,0,0,0]=.1
    with pytest.raises(ValueError,match='Recent observed RGB'):
        diagnostic.intervene(StateFixture(),bad,commands,returns)
    bad=returns.clone();bad[1,0]=0
    with pytest.raises(ValueError,match='same CPU return-command'):
        diagnostic.intervene(StateFixture(),prefix,commands,bad)
    values=diagnostic.intervene(StateFixture(),prefix,commands,returns)
    truth=torch.zeros_like(values['prediction_normal']);truth[1,:,:,1:3,1:3]=.4
    report=diagnostic.summarize(values,truth)
    assert report['state_branch_l2_difference']>0
    assert report['prediction_changes']['swapped']['paired_region_prediction_change_mae']>0
    assert set(report['intervention_scores'])==set(diagnostic.INTERVENTIONS)
    assert all(row['return_frames']==4 for row in report['intervention_scores'].values())
