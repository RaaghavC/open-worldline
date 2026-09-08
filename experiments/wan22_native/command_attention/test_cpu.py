# SPDX-License-Identifier: Apache-2.0
"""Tiny literal native Wan tests. CPU attention is scoped to these fixtures."""
import copy
from dataclasses import replace
from unittest import mock
import pytest
import torch
from controller import CommandAttentionController, DEFAULT_PARAMETER_COUNT
from bridge import NativeCommandAttentionBridge, _Boundary
from experiments.wan22_native.cuda_reference.vendor import model as vendor
from experiments.wan22_native.intermediate_action.test_bridge import cpu_attention

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(vendor,'flash_attention',cpu_attention)
    yield
    torch.set_num_threads(previous)
    assert not torch.cuda.is_initialized()


def case(active=False,padding=2):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026090802)
        core=vendor.WanModel(model_type='ti2v',in_dim=2,out_dim=2,dim=8,ffn_dim=16,
            freq_dim=8,text_dim=4,text_len=8,num_heads=1,num_layers=3)
        with torch.no_grad():core.head.head.weight.normal_(0,.12);core.head.head.bias.normal_(0,.02)
        core.requires_grad_(False).eval()
        control=CommandAttentionController(8,2,4,(1,2))
        if active:
            with torch.no_grad():
                for value in control.projections.values():value.b.weight.normal_(0,.07)
        bridge=NativeCommandAttentionBridge(core,control,test_only=True,padding_tokens=padding)
        noisy=torch.randn(1,*bridge.shape);times=torch.full((1,bridge.tokens),506,dtype=torch.int64)
        times[:,:bridge.prefix]=0;times[:,bridge.valid_tokens:]=0
        contexts=[torch.randn(3,4)]
        commands=torch.zeros(1,16,6);commands[:,:,3]=.02;commands[:,0,5]=1
        observed=noisy[:,:,:1].clone()
    return bridge,(noisy,times,contexts,commands,observed)


def full(bridge,data,**kwargs):
    n,t,c,a,o=data
    return bridge(n,t,c,commands=a,observation=o,**kwargs)


def clean(bridge):
    assert not bridge._lock.locked()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in bridge.core.modules())
    assert all(not p.requires_grad and p.grad is None for p in bridge.core.parameters())


def test_default_exact_parameter_count_and_zero_b():
    c=CommandAttentionController()
    assert sum(p.numel() for p in c.parameters())==DEFAULT_PARAMETER_COUNT==4_936_448
    assert sum(p.numel() for p in c.projections.parameters())==4_718_592
    assert all(torch.count_nonzero(p.b.weight)==0 for p in c.projections.values())


@pytest.mark.parametrize('padding',[0,3])
def test_zero_native_full_cache_identity_and_native_call_counts(padding):
    bridge,data=case(padding=padding);n,t,c,a,o=data
    with torch.no_grad():native=torch.stack(bridge.core(list(n.unbind(0)),t,c,bridge.tokens))
    calls=[];original=vendor.WanAttentionBlock.forward
    def count(module,*args,**kwargs):
        calls.append(list(bridge.core.blocks).index(module));return original(module,*args,**kwargs)
    with mock.patch.object(vendor.WanAttentionBlock,'forward',count):
        features=bridge.extract_features(n,t,c);assert calls==[0]
        result=bridge.predict_from_features(features,a,o);assert calls==[0,1,2]
    assert torch.equal(result,native) and torch.equal(full(bridge,data),native)
    assert result.requires_grad and not features.hidden.requires_grad
    clean(bridge)


def test_mask_and_destination_aligned_history():
    bridge,data=case(active=True);c=bridge.controller;commands=data[3]
    gates=c.gates(commands,bridge.grid,bridge.tokens)
    assert torch.count_nonzero(gates[:,:,:bridge.prefix])==0
    assert torch.count_nonzero(gates[:,:,bridge.valid_tokens:])==0
    assert torch.count_nonzero(gates[:,:,bridge.prefix:bridge.valid_tokens])>0
    changed=commands.clone();changed[:,12:,0]=.8
    states=c.command_states(commands);changed_states=c.command_states(changed)
    assert torch.equal(states[:,:4],changed_states[:,:4])
    assert not torch.equal(states[:,4],changed_states[:,4])
    changed=commands.clone();changed[:,0,0]=.8
    assert not torch.equal(states[:,1:],c.command_states(changed)[:,1:])
    assert torch.equal(states,c.command_states(commands)) # no persistent state
    value=torch.randn(1,bridge.tokens,c.hidden_dim)
    delta=c.residual(1,'q',value,gates[0])
    assert torch.count_nonzero(delta[:,:bridge.prefix])==0 and torch.count_nonzero(delta[:,bridge.valid_tokens:])==0


def test_direct_linear_oracle_and_gate_derivative():
    bridge,data=case(active=True);c=bridge.controller;gate=c.gates(data[3],bridge.grid,bridge.tokens)[0]
    x=torch.randn(1,bridge.tokens,c.hidden_dim,requires_grad=True)
    factors=c.projections['1_q']
    actual=c.residual(1,'q',x,gate)
    expected=((x@factors.a.weight.T)*gate)@factors.b.weight.T
    assert torch.equal(actual,expected)
    weights=torch.randn_like(actual)
    args=(x,gate,factors.a.weight,factors.b.weight)
    a=torch.autograd.grad((actual*weights).sum(),args,retain_graph=True)
    b=torch.autograd.grad((expected*weights).sum(),args)
    assert all(torch.equal(u,v) for u,v in zip(a,b))


def test_full_cached_four_paths_all_gradients_and_foundation_unchanged():
    full_bridge,data=case(active=True);n,t,contexts,commands,observed=data
    cached=NativeCommandAttentionBridge(full_bridge.core,copy.deepcopy(full_bridge.controller),test_only=True,padding_tokens=2)
    original={name:p.clone() for name,p in full_bridge.core.named_parameters()}
    actions=[commands,commands.clone()];actions[1][:,0,5]=0
    predictions=[{},{}]
    for text_index,context in enumerate([contexts[0],contexts[0].flip(0)+.07]):
        bundle=cached.extract_features(n,t,[context])
        for action_index,action in enumerate(actions):
            predictions[0][text_index,action_index]=full_bridge(n,t,[context],commands=action,observation=observed)
            predictions[1][text_index,action_index]=cached.predict_from_features(bundle,action,observed)
            assert torch.equal(predictions[0][text_index,action_index],predictions[1][text_index,action_index])
    def loss(values):
        closed=values[1,0]+5*(values[0,0]-values[1,0])
        opened=values[1,1]+5*(values[0,1]-values[1,1])
        return (-(opened-closed)[:,:,1:]-.13).square().mean()
    for values in predictions:loss(values).backward()
    for (name,a),(other,b) in zip(full_bridge.controller.named_parameters(),cached.controller.named_parameters()):
        assert name==other and a.grad is not None and b.grad is not None
        assert torch.equal(a.grad,b.grad) and torch.isfinite(a.grad).all(),name
    assert sum(p.grad.abs().sum() for p in cached.controller.command_gru.parameters())>0
    assert all(torch.equal(p,original[name]) for name,p in cached.core.named_parameters())
    clean(full_bridge);clean(cached)


def test_two_adamw_updates_reuse_cache_and_reach_encoder():
    bridge,data=case();n,t,c,a,o=data
    cached=NativeCommandAttentionBridge(bridge.core,copy.deepcopy(bridge.controller),test_only=True,padding_tokens=2)
    features=cached.extract_features(n,t,c)
    optimizers=[torch.optim.AdamW(x.controller.parameters(),lr=.003) for x in (bridge,cached)]
    norms=[]
    for step in range(2):
        outputs=[full(bridge,data),cached.predict_from_features(features,a,o)]
        assert torch.equal(*outputs)
        for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
        for output in outputs:(output[:,:,1:]-.17).square().mean().backward()
        norms.append(sum(p.grad.square().sum() for p in cached.controller.command_gru.parameters()).sqrt().item())
        for x,y in zip(bridge.controller.parameters(),cached.controller.parameters()):assert torch.equal(x.grad,y.grad)
        for optimizer in optimizers:optimizer.step()
        for x,y in zip(bridge.controller.parameters(),cached.controller.parameters()):assert torch.equal(x,y)
    assert norms[0]==0 and norms[1]>0
    clean(bridge);clean(cached)


@pytest.mark.parametrize('bad',['noise-nan','time-prefix','time-padding','commands-shape','command-nan','interaction','observation','context','grad-input','inference-input'])
def test_malformed_inputs_fail_before_native_execution(bad):
    bridge,data=case();n,t,c,a,o=data
    if bad=='noise-nan':n[0,0,1,0,0]=float('nan')
    if bad=='time-prefix':t[0,0]=1
    if bad=='time-padding':t[0,-1]=1
    if bad=='commands-shape':a=a[:,:15]
    if bad=='command-nan':a[0,0,0]=float('nan')
    if bad=='interaction':a[0,0,5]=.5
    if bad=='observation':o=o+.1
    if bad=='context':c=[]
    if bad=='grad-input':n.requires_grad_(True)
    if bad=='inference-input':
        with torch.inference_mode():n=n.clone()
    with mock.patch.object(vendor.WanModel,'forward',side_effect=AssertionError('must not execute')):
        with pytest.raises(ValueError):bridge(n,t,c,commands=a,observation=o)
    clean(bridge)


@pytest.mark.parametrize('where',['prefix','projection','suffix','head','registration','reentrant'])
def test_failure_cleanup_and_original_exception(where):
    bridge,data=case(active=True);failure=KeyboardInterrupt('controlled failure')
    native=vendor.WanAttentionBlock.forward
    def blocks(module,*args,**kwargs):
        index=list(bridge.core.blocks).index(module)
        if (where=='prefix' and index==0) or (where=='suffix' and index==2):raise failure
        return native(module,*args,**kwargs)
    def recurse(*args,**kwargs):return full(bridge,data)
    if where in ('prefix','suffix'):patcher=mock.patch.object(vendor.WanAttentionBlock,'forward',blocks)
    elif where=='projection':patcher=mock.patch.object(bridge.controller,'residual',side_effect=failure)
    elif where=='head':patcher=mock.patch.object(vendor.Head,'forward',side_effect=failure)
    elif where=='registration':patcher=mock.patch.object(bridge.core.blocks[1].self_attn.k,'register_forward_hook',side_effect=failure)
    else:patcher=mock.patch.object(bridge.controller,'residual',side_effect=recurse)
    with patcher:
        with pytest.raises(RuntimeError if where=='reentrant' else KeyboardInterrupt) as e:full(bridge,data)
        if where!='reentrant':assert e.value is failure
    clean(bridge)
    assert torch.isfinite(full(bridge,data)).all()


@pytest.mark.parametrize('mode',['mutated-hidden','wrong-owner','changed-core','changed-rotary'])
def test_cache_identity_rejects_changes(mode):
    bridge,data=case();n,t,c,a,o=data;features=bridge.extract_features(n,t,c)
    if mode=='mutated-hidden':features.hidden.add_(.1)
    if mode=='wrong-owner':features=replace(features,owner=object())
    if mode=='changed-core':
        with torch.no_grad():next(bridge.core.parameters()).add_(.1)
    if mode=='changed-rotary':bridge.core.freqs=bridge.core.freqs.clone()
    with pytest.raises(ValueError):bridge.predict_from_features(features,a,o)
    clean(bridge)


def test_capture_registration_failure_foreign_sentinel_and_context_restore():
    bridge,data=case(active=True);n,t,c,a,o=data
    foreign=_Boundary()
    for patcher in [mock.patch.object(bridge.core.blocks[1],'register_forward_pre_hook',side_effect=KeyboardInterrupt('register')),
                    mock.patch.object(vendor.WanModel,'forward',side_effect=foreign)]:
        with patcher,pytest.raises(BaseException):bridge.extract_features(n,t,c)
        clean(bridge)
    with torch.inference_mode():
        features=bridge.extract_features(n,t,c)
        output=bridge.predict_from_features(features,a,o)
        assert torch.is_inference_mode_enabled()
    assert output.requires_grad and not output.is_inference()
    output.sum().backward()
    assert not bridge.predict_from_features(features,a,o,track_grad=False).requires_grad
    clean(bridge)


def test_production_rejects_cpu_and_nonboolean_modes():
    bridge,data=case()
    with pytest.raises(ValueError):NativeCommandAttentionBridge(bridge.core,bridge.controller)
    with pytest.raises(ValueError):full(bridge,data,track_grad=1)
    with pytest.raises(ValueError):CommandAttentionController(blocks=(2,1))


def test_independent_native_projection_oracle_output_and_all_parameter_gradients():
    bridge,data=case(active=True);n,t,contexts,commands,observation=data
    expected_controller=copy.deepcopy(bridge.controller)
    gates=expected_controller.gates(commands,bridge.grid,bridge.tokens)
    sites={id(getattr(bridge.core.blocks[b].self_attn,p)):(i,b,p)
           for i,(b,p) in enumerate((b,p) for b in (1,2) for p in ('q','k','v','o'))}
    native_linear=torch.nn.Linear.forward
    def oracle(module,value):
        original=native_linear(module,value)
        if id(module) not in sites:return original
        i,b,p=sites[id(module)];f=expected_controller.projections[f'{b}_{p}']
        # Independent matrix formula inserted through the class-level test
        # fixture, not through production hook/residual/cached suffix methods.
        addition=((value.float()@f.a.weight.T)*gates[i])@f.b.weight.T
        return original+addition.to(original.dtype)
    with mock.patch.object(torch.nn.Linear,'forward',oracle):
        expected=torch.stack(bridge.core(list(n.unbind(0)),t,contexts,bridge.tokens))
    actual=full(bridge,data)
    assert torch.equal(actual,expected)
    weights=torch.linspace(-.3,.7,actual.numel()).reshape_as(actual)
    (actual*weights).sum().backward();(expected*weights).sum().backward()
    for (name,a),(other,b) in zip(bridge.controller.named_parameters(),expected_controller.named_parameters()):
        assert name==other and a.grad is not None and b.grad is not None
        assert torch.equal(a.grad,b.grad),name
    clean(bridge)
