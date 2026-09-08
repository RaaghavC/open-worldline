# SPDX-License-Identifier: Apache-2.0
"""Small callback-driven parity and two-step profiler; no loading or dispatch."""
import math
import time
import torch

ARMS = tuple(m+'_'+d for m in ('stationary','left','right') for d in ('closed','interact'))


def comparison(reference, actual, tensor_sha):
    if reference.shape != actual.shape or reference.dtype != torch.float32 or actual.dtype != torch.float32:
        raise ValueError('Aligned FP32 native velocities required')
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise FloatingPointError('Finite native velocities required')
    reference_sha, actual_sha = tensor_sha(reference), tensor_sha(actual)
    return dict(exact_equal=bool(torch.equal(reference, actual)) and reference_sha==actual_sha,
                max_abs=float((reference-actual).abs().max()),
                native_sha256=reference_sha, actual_sha256=actual_sha)


def gradient_summary(controller):
    rows = {}
    for name, parameter in controller.named_parameters():
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('Missing or nonfinite controller gradient: '+name)
        rows[name] = dict(elements=gradient.numel(), l2=float(gradient.detach().double().norm()),
                          max_abs=float(gradient.detach().abs().max()))
    total = math.sqrt(sum(row['l2']**2 for row in rows.values()))
    if not rows or not math.isfinite(total) or total <= 0:
        raise FloatingPointError('Positive finite controller gradient required')
    return dict(parameters=rows, parameter_tensors=len(rows), l2_after_clip=total,
                all_present_finite=True)


def execute(bridge, data, rows, *, noise, flow_inputs, native_predict, tensor_sha,
            make_optimizer, run_step, retain, progress, verify_core, check=lambda:None,
            synchronize=lambda:None, reset_peak=lambda:None, memory=lambda:{}):
    if set(data['windows']) != set(ARMS) or len(rows) != 2:
        raise ValueError('Six native windows and exactly two profile steps required')
    if (rows[0].get('update') != 1 or rows[0].get('auxiliary_edge') is None
            or rows[1].get('update') != 2 or rows[1].get('auxiliary_edge') is not None):
        raise ValueError('Mixed first step and FM-only second step required')
    report = dict(stage='zero-parity', completed_updates=0, parity=[], updates=[],
                  quality_assessed=False, image_generation=False, zero_gate_passed=False)
    device = next(bridge.controller.parameters()).device
    window = data['windows']['stationary_closed']
    noisy,times,_ = flow_inputs(window['target'],window['observation'],noise(0),rows[0]['k'])
    reset_peak(); start = time.monotonic()
    for context_name in ('positive','negative'):
        context = data[context_name]
        check(); synchronize(); began = time.monotonic()
        reference = native_predict(noisy,times,context)
        synchronize()
        report.setdefault('native_seconds',{})[context_name] = time.monotonic()-began
        retain('parity/native-'+context_name,{'velocity':reference})
        x,t,c = noisy.to(device),times.to(device),context.to(device)
        for arm in ARMS:
            w = data['windows'][arm]
            check(); synchronize(); began = time.monotonic()
            actual = bridge(x,t,[c],commands=w['commands'].to(device),
                            observation=w['observation'].to(device),track_grad=False).detach().cpu()
            synchronize()
            row = dict(comparison(reference,actual,tensor_sha),context=context_name,arm=arm,
                       path='full',seconds=time.monotonic()-began)
            report['parity'].append(row); progress(report)
            if not row['exact_equal']:
                raise RuntimeError('Full zero equality failed before optimizer creation')
            del actual
        check(); synchronize(); began = time.monotonic()
        features = bridge.extract_features(x,t,[c]); synchronize()
        extract_seconds = time.monotonic()-began
        for arm in ARMS:
            w = data['windows'][arm]
            check(); synchronize(); began = time.monotonic()
            actual = bridge.predict_from_features(features,w['commands'].to(device),
                                                   w['observation'].to(device),track_grad=False).detach().cpu()
            synchronize()
            row = dict(comparison(reference,actual,tensor_sha),context=context_name,arm=arm,
                       path='cached',seconds=time.monotonic()-began,feature_extract_seconds=extract_seconds)
            report['parity'].append(row); progress(report)
            if not row['exact_equal']:
                raise RuntimeError('Cached zero equality failed before optimizer creation')
            del actual
        del features,reference,x,t,c
    synchronize()
    report.update(zero_gate_passed=True,parity_seconds=time.monotonic()-start,parity_memory=memory())
    verify_core('after-parity'); progress(report)
    optimizer = make_optimizer()
    if optimizer.state:
        raise ValueError('Fresh profile optimizer required')
    retain('controller-initial',bridge.controller.state_dict())
    for index,row in enumerate(rows):
        report['stage'] = 'update-'+str(row['update']); check(); reset_peak(); synchronize()
        start = time.monotonic()
        result = run_step(row,noise(index),optimizer,
                          lambda name,values:retain('update-'+str(row['update'])+'/'+name,values))
        synchronize()
        result.update(profile_seconds=time.monotonic()-start,memory=memory(),
                      gradients=gradient_summary(bridge.controller))
        if (result.get('optimizer_updates') != 1 or result.get('main_predictions') != 2
                or result.get('auxiliary',{}).get('predictions') != (4 if index==0 else 0)):
            raise RuntimeError('Profile did not execute the declared FM/CFG update counts')
        report['updates'].append(result); report['completed_updates'] = index+1; progress(report)
    retain('controller-final',bridge.controller.state_dict())
    optimizer.zero_grad(set_to_none=True)
    verify_core('after-updates')
    report.update(status='passed',stage='complete',base_unchanged=True)
    progress(report)
    return report
