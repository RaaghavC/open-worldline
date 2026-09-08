# SPDX-License-Identifier: Apache-2.0
"""Bounded literal native CPU fixture; no CUDA, FA2, checkpoints or weights."""
import copy
from dataclasses import replace
import gc
from types import MappingProxyType
from unittest import mock
import weakref

import pytest
import torch

from experiments.wan22_native.intermediate_action.test_bridge import fixture, cpu_attention
from experiments.wan22_native.cuda_reference.vendor import model as vendor
from experiments.wan22_native.action_adapter.model import PostBlockActionAdapter
from experiments.wan22_native.intermediate_action import cached_intermediate as implementation
from experiments.wan22_native.intermediate_action.cached_intermediate import CachedIntermediateActionBridge


@pytest.fixture(autouse=True)
def bounded_cpu(monkeypatch):
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(vendor, 'flash_attention', cpu_attention)
    yield
    torch.set_num_threads(previous)
    assert not torch.cuda.is_initialized()


def case(index=0, activated=False):
    full, *inputs = fixture(index, activated)
    cached = CachedIntermediateActionBridge(full.core, copy.deepcopy(full.adapter), block_index=index, test_only=True)
    return full, cached, inputs


def check_no_hooks(bridge):
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in bridge.core.modules())
    assert not bridge._contract._call_lock.locked()


def all_grads_equal(full, cached):
    for (name, expected), (other, actual) in zip(full.adapter.named_parameters(), cached.adapter.named_parameters()):
        assert name == other and expected.grad is not None and actual.grad is not None
        assert torch.isfinite(actual.grad).all()
        assert torch.equal(expected.grad, actual.grad), name
    assert all(not p.requires_grad and p.grad is None for p in cached.core.parameters())


def tensors(features):
    return [features.hidden, features.time_embedding, features.observed_prefix] + [v for v in features.block_kwargs.values() if isinstance(v, torch.Tensor)]


@pytest.mark.parametrize('index', [0, 1])
def test_zero_cache_equals_native_and_full_forward_with_literal_capture(index):
    full, cached, (noisy, times, contexts, commands, observation) = case(index)
    expected = {}
    def block(module, args, kwargs, hidden):
        expected['hidden'] = hidden.clone(); expected['kwargs'] = kwargs
    def time(module, args, output): expected['time'] = output.clone()
    a=full.core.blocks[index].register_forward_hook(block, with_kwargs=True)
    b=full.core.time_embedding.register_forward_hook(time)
    try:
        with torch.no_grad(): native=torch.stack(full.core(list(noisy.unbind(0)), times, contexts, full.tokens))
    finally:a.remove();b.remove()
    features = cached.extract_features(noisy, times, contexts)
    assert torch.equal(features.hidden, expected['hidden'])
    assert torch.equal(features.time_embedding, expected['time'])
    for name, value in features.block_kwargs.items():
        if value is None: assert expected['kwargs'][name] is None
        else: assert torch.equal(value, expected['kwargs'][name]), name
    assert features.time_embedding.shape == (1,720,8)
    assert features.block_kwargs['e'].shape == (1,720,6,8)
    actual=cached.predict_from_features(features, commands, observation)
    assert torch.equal(actual, native)
    assert torch.equal(actual, full(noisy,times,contexts,commands=commands,observation=observation))
    assert actual.requires_grad and actual.dtype == torch.float32
    assert all(not x.requires_grad and x.grad_fn is None and not x.is_inference() for x in tensors(features))
    check_no_hooks(cached)


@pytest.mark.parametrize('index', [0, 1])
def test_four_cfg_paths_match_full_forward_and_all_gradients(index):
    full, cached, (noisy, times, contexts, commands, observation) = case(index, activated=True)
    alternate=commands.clone();alternate[:,0,5]=0
    contexts=[contexts[0],contexts[0].flip(0)+.02]
    initial={name:p.clone() for name,p in full.core.named_parameters()}
    expected={};actual={}
    for text_index, context in enumerate(contexts):
        features=cached.extract_features(noisy,times,[context])
        for action_index, action in enumerate((commands,alternate)):
            expected[text_index,action_index]=full(noisy,times,[context],commands=action,observation=observation)
            actual[text_index,action_index]=cached.predict_from_features(features,action,observation)
            assert torch.equal(expected[text_index,action_index],actual[text_index,action_index])
    def loss(values):
        closed=values[1,0]+5*(values[0,0]-values[1,0])
        opened=values[1,1]+5*(values[0,1]-values[1,1])
        return (-(opened-closed)[:,:,1:]-.125).square().mean()
    reference_loss=loss(expected);actual_loss=loss(actual)
    assert torch.equal(reference_loss,actual_loss)
    reference_loss.backward();actual_loss.backward();all_grads_equal(full,cached)
    assert sum(p.grad.abs().sum() for p in cached.adapter.command_gru.parameters())>0
    assert cached.adapter.query.weight.grad.abs().sum()>0
    assert all(torch.equal(p,initial[name]) for name,p in full.core.named_parameters())
    check_no_hooks(cached)


@pytest.mark.parametrize('index', [0, 1])
def test_cache_reuse_across_three_adapter_updates_has_fresh_graphs(index):
    full,cached,(noisy,times,contexts,commands,observation)=case(index)
    features=cached.extract_features(noisy,times,contexts)
    snapshots=[x.clone() for x in tensors(features)]
    optimizers=[torch.optim.AdamW(x.adapter.parameters(),lr=.005)for x in(full,cached)]
    recurrent=[];outputs=[]
    for _ in range(3):
        for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
        expected=full(noisy,times,contexts,commands=commands,observation=observation)
        actual=cached.predict_from_features(features,commands,observation)
        assert torch.equal(expected,actual);outputs.append(actual.detach().clone())
        expected[:,:,1:].square().mean().backward();actual[:,:,1:].square().mean().backward()
        all_grads_equal(full,cached)
        recurrent.append(sum(p.grad.square().sum()for p in cached.adapter.command_gru.parameters()).sqrt().item())
        for optimizer in optimizers:optimizer.step()
        for a,b in zip(full.adapter.parameters(),cached.adapter.parameters()):assert torch.equal(a,b)
        for a,b in zip(optimizers[0].state.values(),optimizers[1].state.values()):
            assert set(a)==set(b)
            assert all(torch.equal(a[k],b[k]) for k in a)
    assert recurrent[0]==0 and recurrent[1]>0 and recurrent[2]>0
    assert not torch.equal(outputs[0],outputs[1])
    assert all(torch.equal(x,y) for x,y in zip(tensors(features),snapshots))


def test_only_prefix_executes_once_and_suffix_repeats_for_commands():
    full,cached,(noisy,times,contexts,commands,observation)=case(0,activated=True)
    calls=[];original=vendor.WanAttentionBlock.forward
    def inspect(module,*args,**kwargs):
        calls.append((list(full.core.blocks).index(module),args[0].requires_grad,torch.is_grad_enabled()))
        return original(module,*args,**kwargs)
    with mock.patch.object(vendor.WanAttentionBlock,'forward',inspect):
        features=cached.extract_features(noisy,times,contexts)
        assert calls==[(0,False,False)]
        for _ in range(2):cached.predict_from_features(features,commands,observation).sum().backward()
    assert calls==[(0,False,False),(1,True,True),(1,True,True)]
    assert all(p.grad is None for p in full.core.parameters())


def test_outer_inference_context_and_input_storage_are_preserved():
    _,cached,(noisy,times,contexts,commands,observation)=case(0,activated=True)
    before=[x.clone()for x in(noisy,times,*contexts,commands,observation)]
    with torch.inference_mode():
        features=cached.extract_features(noisy,times,contexts)
        actual=cached.predict_from_features(features,commands,observation)
        assert torch.is_inference_mode_enabled()
    assert actual.requires_grad and not actual.is_inference()
    actual[:,:,1:].sum().backward()
    inference=cached.predict_from_features(features,commands,observation,track_grad=False)
    assert not inference.requires_grad and inference.grad_fn is None
    assert all(torch.equal(a,b)for a,b in zip((noisy,times,*contexts,commands,observation),before))


def test_sentinel_traceback_cleared_and_native_temporary_released_without_gc():
    _,cached,(noisy,times,contexts,_,_)=case()
    records=[];temporaries=[];original=vendor.WanModel.forward
    class Recorded(implementation._PrefixBoundary):
        def __init__(self):records.append(self)
    def inspect(module,*args,**kwargs):
        temporary=torch.ones(16);temporaries.append(weakref.ref(temporary))
        return original(module,*args,**kwargs)
    enabled=gc.isenabled();gc.disable()
    try:
        with mock.patch.object(implementation,'_PrefixBoundary',Recorded),mock.patch.object(vendor.WanModel,'forward',inspect):
            features=cached.extract_features(noisy,times,contexts)
        assert len(records)==1 and records[0].__traceback__ is None
        assert temporaries[0]() is None
        assert features.hidden.shape==(1,720,8)
    finally:
        if enabled:gc.enable()
    check_no_hooks(cached)


@pytest.mark.parametrize('where',['before-time','time','selected','second-hook-registration','foreign-sentinel'])
def test_capture_failure_cleans_both_hooks_and_preserves_error(where):
    _,cached,(noisy,times,contexts,_,_)=case()
    failure=implementation._PrefixBoundary()if where=='foreign-sentinel'else KeyboardInterrupt('capture fixture')
    def fail(*args,**kwargs):raise failure
    if where in('before-time','foreign-sentinel'):patcher=mock.patch.object(vendor.WanModel,'forward',fail)
    elif where=='time':patcher=mock.patch.object(cached.core.time_embedding,'forward',fail)
    elif where=='selected':patcher=mock.patch.object(vendor.WanAttentionBlock,'forward',fail)
    else:patcher=mock.patch.object(cached.core.blocks[0],'register_forward_hook',fail)
    # Core instance forward overrides are correctly rejected by the inherited
    # contract; target the time module class instead for a native-stage error.
    if where=='time':
        original=torch.nn.Sequential.forward
        def time_fail(module,*args,**kwargs):
            if module is cached.core.time_embedding:raise failure
            return original(module,*args,**kwargs)
        patcher=mock.patch.object(torch.nn.Sequential,'forward',time_fail)
    with patcher:
        with pytest.raises(type(failure))as caught:cached.extract_features(noisy,times,contexts)
    assert caught.value is failure;check_no_hooks(cached)
    assert cached.extract_features(noisy,times,contexts).hidden.shape==(1,720,8)


@pytest.mark.parametrize('where',['adapter','suffix','head','unpatchify'])
def test_suffix_failure_releases_lock_and_cache_can_be_reused(where):
    _,cached,(noisy,times,contexts,commands,observation)=case()
    features=cached.extract_features(noisy,times,contexts);failure=KeyboardInterrupt('suffix fixture')
    def fail(*args,**kwargs):raise failure
    owner,name={'adapter':(PostBlockActionAdapter,'forward'),'suffix':(vendor.WanAttentionBlock,'forward'),
                'head':(vendor.Head,'forward'),'unpatchify':(vendor.WanModel,'unpatchify')}[where]
    with mock.patch.object(owner,name,fail):
        with pytest.raises(KeyboardInterrupt)as caught:cached.predict_from_features(features,commands,observation)
    assert caught.value is failure;check_no_hooks(cached)
    assert cached.predict_from_features(features,commands,observation).shape==noisy.shape


@pytest.mark.parametrize('change',['owner','profile','index','hidden-shape','time-dtype','nonfinite-context','keyword','time-projection-shape'])
def test_invalid_feature_identity_shapes_and_constants_rejected(change):
    _,cached,(noisy,times,contexts,commands,observation)=case()
    features=cached.extract_features(noisy,times,contexts)
    if change=='owner':features=replace(features,owner=object())
    elif change=='profile':features=replace(features,profile='spatial')
    elif change=='index':features=replace(features,block_index=1)
    elif change=='hidden-shape':features=replace(features,hidden=features.hidden[:,:-1])
    elif change=='time-dtype':features=replace(features,time_embedding=features.time_embedding.double())
    else:
        kwargs=dict(features.block_kwargs)
        if change=='nonfinite-context':kwargs['context']=torch.full_like(kwargs['context'],float('nan'))
        elif change=='keyword':kwargs['target']=torch.ones(1)
        else:kwargs['e']=kwargs['e'][:,:,0]
        features=replace(features,block_kwargs=MappingProxyType(kwargs))
    with pytest.raises(ValueError):cached.predict_from_features(features,commands,observation)
    check_no_hooks(cached)


def test_inplace_constant_and_core_changes_invalidate_cache_adapter_updates_do_not():
    _,cached,(noisy,times,contexts,commands,observation)=case()
    features=cached.extract_features(noisy,times,contexts)
    with pytest.raises(TypeError):features.block_kwargs['target']=torch.ones(1)
    features.hidden.add_(0)
    with pytest.raises(ValueError,match='mutated'):cached.predict_from_features(features,commands,observation)
    features=cached.extract_features(noisy,times,contexts)
    with torch.no_grad():next(cached.core.parameters()).add_(0)
    with pytest.raises(ValueError,match='core parameters'):cached.predict_from_features(features,commands,observation)
    features=cached.extract_features(noisy,times,contexts)
    with torch.no_grad():cached.adapter.output.bias.add_(.01)
    assert cached.predict_from_features(features,commands,observation).shape==noisy.shape


def test_native_lazy_rotary_storage_replacement_is_bound_after_extraction():
    _,cached,(noisy,times,contexts,commands,observation)=case()
    original=vendor.WanModel.forward;old=cached.core.freqs
    def replace_storage(module,*args,**kwargs):
        module.freqs=module.freqs.clone()
        return original(module,*args,**kwargs)
    # CPU-only identity analogue of the official first-call device transfer.
    with mock.patch.object(vendor.WanModel,'forward',replace_storage):
        features=cached.extract_features(noisy,times,contexts)
    assert cached.core.freqs is not old and torch.equal(cached.core.freqs,old)
    assert features.block_kwargs['freqs'] is cached.core.freqs
    assert cached.predict_from_features(features,commands,observation).shape==noisy.shape


def test_foreign_hooks_gradient_inputs_and_changed_observation_fail_before_execution():
    _,cached,(noisy,times,contexts,commands,observation)=case()
    h=cached.core.head.register_forward_pre_hook(lambda module,args:None)
    try:
        with pytest.raises(ValueError,match='no hooks'):cached.extract_features(noisy,times,contexts)
        assert h.id in cached.core.head._forward_pre_hooks
    finally:h.remove()
    noisy.requires_grad_()
    with pytest.raises(ValueError,match='non-gradient'):cached.extract_features(noisy,times,contexts)
    noisy.requires_grad_(False);features=cached.extract_features(noisy,times,contexts)
    changed=observation.clone();changed.flatten()[0]+=.1
    with pytest.raises(ValueError,match='exact clean initial prefix'):cached.predict_from_features(features,commands,changed)
    commands.requires_grad_()
    with pytest.raises(ValueError,match='non-gradient'):cached.predict_from_features(features,commands,observation)
    check_no_hooks(cached)
