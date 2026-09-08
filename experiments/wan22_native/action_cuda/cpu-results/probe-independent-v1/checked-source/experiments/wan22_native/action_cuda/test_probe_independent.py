# SPDX-License-Identifier: Apache-2.0
"""Independent CPU checks for the numerical probe. No CUDA or real weights."""
import copy

import pytest
import torch

from . import probe_math
from .test_independent import fixture


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_parity_requires_both_predeclared_bounds_and_handles_zero_reference():
    assert probe_math.PARITY == {'max_absolute': 1e-6, 'relative_l2': 1e-6}
    shape = (1, 2, 5, 2, 2)
    x = torch.full(shape, 2., dtype=torch.float32)
    assert probe_math.compare_velocity(x, x + 5e-7)['passed']
    # Large reference: relative error passes, absolute error fails.
    x = torch.full(shape, 1000., dtype=torch.float32)
    report = probe_math.compare_velocity(x, x + 1e-4)
    assert not report['passed'] and report['relative_l2'] < 1e-6 and report['max_absolute'] > 1e-6
    # Small reference: absolute error passes, relative error fails.
    x = torch.full(shape, 1e-5, dtype=torch.float32)
    report = probe_math.compare_velocity(x, x + 5e-7)
    assert not report['passed'] and report['relative_l2'] > 1e-6 and report['max_absolute'] < 1e-6
    zero = torch.zeros(shape)
    assert probe_math.compare_velocity(zero, zero)['passed']
    report = probe_math.compare_velocity(zero, torch.full(shape, 1e-8))
    assert not report['passed'] and report['relative_l2'] is None
    bad = zero.clone(); bad.flatten()[0] = float('nan')
    with pytest.raises(ValueError): probe_math.compare_velocity(zero, bad)


def assert_tree_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_tree_equal(a, b)
    else:
        assert left == right


def test_two_paired_adamw_updates_match_independent_future_flow_objective():
    actual, noisy, _, contexts, commands, observation = fixture()
    reference, *_ = fixture()
    parameters_before = {name: p.detach().clone() for name, p in actual.core.named_parameters()}
    # Each branch has its own future truth, while both share the exact observed
    # image. These differing corruptions cannot establish action control.
    target_closed = noisy.clone()
    target_open = noisy.clone(); target_open[:, :, 1:] *= .6
    action_open = commands.clone(); action_open[:, 0, 5] = 1
    windows = [dict(target=target_closed, observation=observation, commands=commands),
               dict(target=target_open, observation=observation, commands=action_open)]
    immutable = [{name: value.clone() for name, value in row.items()} for row in windows]
    optim = torch.optim.AdamW(actual.adapter.parameters(), **probe_math.OPTIMIZER)
    ref_optim = torch.optim.AdamW(reference.adapter.parameters(), **probe_math.OPTIMIZER)
    rng = torch.Generator(device='cpu').manual_seed(398231)
    norms = []
    for update, k in enumerate((317, 719), 1):
        noise = torch.randn(noisy.shape, generator=rng)
        saved_noise = noise.clone()
        result = probe_math.paired_update(actual, windows, noise, k, contexts[0], optim)
        norms.append(result['command_gru_gradient_l2'])
        ref_optim.zero_grad(set_to_none=True)
        losses = []
        for row in windows:
            sigma = k / 1000.
            corruption = (1 - sigma) * row['target'] + sigma * noise
            corruption[:, :, :1] = row['observation']
            velocity = noise - row['target']
            times = torch.full((1, 720), k, dtype=torch.int64)
            times[:, :144] = 0
            prediction = reference(corruption, times, contexts,
                                   commands=row['commands'], observation=row['observation'])
            loss = (prediction[:, :, 1:] - velocity[:, :, 1:]).square().mean()
            (.5 * loss).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(reference.adapter.parameters(), 1., error_if_nonfinite=True)
        ref_optim.step()
        assert result['paired_mean_future_flow_mse'] == sum(losses) / 2
        assert result['live_sequential_forwards'] == 2 and result['optimizer_updates'] == 1
        assert [row['branch'] for row in result['branches']] == ['closed', 'open']
        assert all(row['observed_input_prefix_exact'] for row in result['branches'])
        assert_tree_equal(actual.adapter.state_dict(), reference.adapter.state_dict())
        assert_tree_equal(optim.state_dict(), ref_optim.state_dict())
        assert all(float(state['step']) == update for state in optim.state.values())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actual.adapter.parameters())
        assert all(p.grad is None and torch.equal(p, parameters_before[name]) for name, p in actual.core.named_parameters())
        assert torch.equal(noise, saved_noise)
        for before, after in zip(immutable, windows):
            assert all(torch.equal(before[key], after[key]) for key in before)
    assert norms[0] == 0. and norms[1] > 0.


def test_nonexact_cross_length_prefix_is_bounded_without_changing_target_or_loss():
    observation=torch.ones(1,2,1,4,4)
    target=torch.full((1,2,5,4,4),.2); target[:,:,:1]=observation+5e-6
    noise=torch.full_like(target,.7)
    before=target.clone(); saved_observation=observation.clone()
    noisy,times,velocity=probe_math.flow_inputs(target,observation,noise,317)
    assert not torch.equal(target[:,:,:1],observation)
    assert torch.equal(target,before) and torch.equal(observation,saved_observation)
    assert torch.equal(noisy[:,:,:1],observation)
    assert torch.equal(noisy[:,:,1:],(1-.317)*target[:,:,1:]+.317*noise[:,:,1:])
    assert torch.equal(velocity,noise-target)
    assert torch.equal(times,torch.tensor([[0]*4+[317]*16],dtype=torch.int64))
    second=target.clone(); second[:,:,:1]=observation+2e-6
    noisy2,times2,velocity2=probe_math.flow_inputs(second,observation,noise,317)
    assert torch.equal(noisy,noisy2) and torch.equal(times,times2)
    assert torch.equal(velocity[:,:,1:],velocity2[:,:,1:])
    prediction=torch.full_like(target,.4)
    assert probe_math.future_flow_mse(prediction,velocity)==probe_math.future_flow_mse(prediction,velocity2)
    bad=target.clone(); bad[:,:,:1]=observation+2e-5
    with pytest.raises(ValueError,match='Cross-length'):
        probe_math.flow_inputs(bad,observation,noise,317)


@pytest.mark.parametrize('failure',['parity','bridge'])
def test_pretraining_failures_preserve_native_output_and_never_update(failure):
    from unittest import mock
    wrapped,noisy,_,contexts,commands,observation=fixture()
    initial={name:value.clone()for name,value in wrapped.adapter.state_dict().items()}
    alternate=commands.clone(); alternate[:,0,5]=1
    windows={'closed-0000':dict(target=noisy,observation=observation,commands=commands),
             'open-0000':dict(target=noisy,observation=observation,commands=alternate)}
    schedule=[{'start':0,'branches':['closed-0000','open-0000'],'k':317,'noise_key':'noise0'},
              {'start':8,'branches':['closed-0008','open-0008'],'k':719,'noise_key':'noise1'}]
    draws={'noise0':torch.full_like(noisy,.3),'noise1':torch.full_like(noisy,.7)}
    optimizer=torch.optim.AdamW(wrapped.adapter.parameters(),**probe_math.OPTIMIZER)
    retained=[]; checkpoints=[]
    def native(x,t,context):
        with torch.no_grad():
            value=torch.stack(wrapped.core(list(x.unbind(0)),t,[context],720))
        return value+.01 if failure=='parity' else value
    def retain(name,values):
        retained.append((name,{key:value.clone()for key,value in values.items()}))
    patch=mock.patch.object(wrapped,'forward',side_effect=RuntimeError('injected bridge failure'))
    import contextlib
    with patch if failure=='bridge' else contextlib.nullcontext():
        with pytest.raises(RuntimeError):
            probe_math.execute_steps(wrapped,windows,schedule,draws,contexts[0],optimizer,
                native_predict=native,retain=retain,checkpoint=lambda step,report:checkpoints.append(step),
                progress=lambda report:None)
    assert checkpoints==[0]
    assert not optimizer.state
    assert all(torch.equal(value,initial[name])for name,value in wrapped.adapter.state_dict().items())
    assert all(parameter.grad is None for parameter in wrapped.adapter.parameters())
    native_values=[values['native_velocity']for _,values in retained if 'native_velocity'in values]
    assert native_values and all(torch.isfinite(value).all()for value in native_values)
