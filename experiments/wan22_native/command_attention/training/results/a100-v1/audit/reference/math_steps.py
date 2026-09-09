# SPDX-License-Identifier: Apache-2.0
"""One declared fitting package: conditional attention and seven action edges."""
import math
import time
import torch
from experiments.wan22_native.action_cuda.probe_math import flow_inputs, future_flow_mse
from experiments.wan22_native.action_training.objective import grad_norm, require_finite_tree

MOTIONS = ('stationary', 'left', 'right')
EDGES = (
    ('stationary_closed', 'stationary_interact'),
    ('left_closed', 'left_interact'),
    ('right_closed', 'right_interact'),
    ('stationary_closed', 'left_closed'),
    ('stationary_closed', 'right_closed'),
    ('stationary_interact', 'left_interact'),
    ('stationary_interact', 'right_interact'),
)
OPTIMIZER = dict(lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)

def schedule(draws):
    if not draws:
        raise ValueError('At least one predeclared draw required')
    result = []
    for i, draw in enumerate(draws):
        if draw['update'] != i + 1 or type(draw['k']) is not int or not 50 <= draw['k'] <= 950:
            raise ValueError('Sequential draw rows with integer timesteps in [50,950] required')
        motion = MOTIONS[i % 3]
        result.append(dict(draw, branches=[motion+'_closed', motion+'_interact'],
                           auxiliary_edge=list(EDGES[(i//4) % len(EDGES)]) if i % 4 == 0 else None))
    return result

def validate_pair(windows, names, noise):
    if len(names) != 2 or names[0] == names[1]:
        raise ValueError('Two distinct ordered arms required')
    pair = [windows[name] for name in names]
    for w in pair:
        if set(w) != {'target', 'observation', 'commands'}:
            raise ValueError('Windows contain only target, observation and commands')
        # Keep the existing native target, prefix, dtype and no-gradient checks.
        flow_inputs(w['target'], w['observation'], noise, 506)
        c = w['commands']
        if c.shape != (1,16,6) or c.dtype != torch.float32 or c.device.type != 'cpu' or c.requires_grad or not torch.isfinite(c).all():
            raise ValueError('Finite frozen CPU command arrays required')
    if not torch.equal(pair[0]['observation'], pair[1]['observation']):
        raise ValueError('A paired contrast requires one identical starting observation')
    return pair

def endpoint_inputs(windows, names, noise):
    pair = validate_pair(windows, names, noise)
    x = noise.clone()
    x[:,:,:1] = pair[0]['observation']
    prefix = (x.shape[-2]//2)*(x.shape[-1]//2)
    times = torch.full((1,5*prefix),999,dtype=torch.int64)
    times[:,:prefix] = 0
    return pair, x, times, pair[1]['target']-pair[0]['target']

def cfg(positive, negative):
    return negative + 5*(positive-negative)

def contrast(a_positive, a_negative, b_positive, b_negative):
    return -(cfg(b_positive,b_negative)-cfg(a_positive,a_negative))

def update(bridge, windows, row, noise, positive, negative, optimizer, *,
           check=lambda: None, retain=lambda name, values: None, synchronize=lambda: None):
    """Two sequential main backwards, optional four-branch contrast, one step.

    All targets are loss labels. Auxiliary future inputs contain only saved
    Gaussian noise. Positive and negative contexts receive the same command.
    No persistent hidden state is carried from one training update to another.
    """
    parameters = list(bridge.controller.parameters())
    if not parameters or {id(p) for g in optimizer.param_groups for p in g['params']} != {id(p) for p in parameters}:
        raise ValueError('Optimizer must own exactly controller parameters')
    if any(p.requires_grad or p.grad is not None for p in bridge.core.parameters()):
        raise ValueError('Foundation must be frozen and gradient-free')
    for c in (positive, negative):
        if c.ndim != 2 or c.dtype != torch.float32 or c.device.type != 'cpu' or c.requires_grad or not torch.isfinite(c).all():
            raise ValueError('Finite frozen CPU text contexts required')
    if row['auxiliary_edge'] is not None and tuple(row['auxiliary_edge']) not in EDGES:
        raise ValueError('Auxiliary edge must be predeclared with its exact orientation')
    pair = validate_pair(windows,row['branches'],noise)
    device = parameters[0].device
    optimizer.zero_grad(set_to_none=True)
    main = []
    synchronize(); started = time.monotonic()
    for name,w in zip(row['branches'],pair):
        check()
        x,t,v = flow_inputs(w['target'],w['observation'],noise,row['k'])
        x = x.to(device)
        prediction = bridge(x,t.to(device),[positive.to(device)],commands=w['commands'].to(device),
                            observation=w['observation'].to(device),track_grad=True)
        loss = future_flow_mse(prediction,v.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite main loss')
        retain('main-'+name,dict(velocity=prediction))
        (.5*loss).backward()
        if not torch.equal(x[:,:,:1].detach().cpu(),w['observation']):
            raise RuntimeError('Model modified the observed prefix input')
        main.append(dict(arm=name,future_flow_mse=float(loss.detach().cpu())))
        del prediction,loss,x,t,v
    synchronize(); main_seconds = time.monotonic()-started
    aux = dict(enabled=False,edge=None,weight=1.,feature_extracts=0,predictions=0,loss=0.)
    if row['auxiliary_edge'] is not None:
        pair,x,t,target = endpoint_inputs(windows,row['auxiliary_edge'],noise)
        predictions = {}
        try:
            for label,context in (('positive',positive),('negative',negative)):
                check()
                features = bridge.extract_features(x.to(device),t.to(device),[context.to(device)])
                for i,w in enumerate(pair):
                    check()
                    p = bridge.predict_from_features(features,w['commands'].to(device),w['observation'].to(device),track_grad=True)
                    retain('aux-'+str(i)+'-'+label,dict(velocity=p))
                    predictions[i,label] = p
                del features,p
            predicted = contrast(predictions[0,'positive'],predictions[0,'negative'],predictions[1,'positive'],predictions[1,'negative'])
            loss = future_flow_mse(predicted,target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite contrast loss')
            aux = dict(enabled=True,edge=list(row['auxiliary_edge']),weight=1.,feature_extracts=2,predictions=4,
                       loss=float(loss.detach().cpu()),definition='Future mean squared error of -(G_b-G_a) against z_b-z_a; G=N+5*(P-N); ideal s=1, native time999')
            loss.backward()
            del predicted,loss,target,x,t
        finally:
            predictions.clear()
    check()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError('Every controller gradient must be present and finite')
    norm = grad_norm(parameters)
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError('Positive finite aggregate gradient required')
    torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    clipped = grad_norm(parameters)
    retain('gradients',dict((n,p.grad) for n,p in bridge.controller.named_parameters()))
    optimizer.step()
    require_finite_tree(optimizer.state_dict())
    if any(not torch.isfinite(p).all() for p in parameters) or any(p.requires_grad or p.grad is not None for p in bridge.core.parameters()):
        raise FloatingPointError('Finite controller and unchanged gradient ownership required')
    synchronize()
    return dict(update=row['update'],main=main,auxiliary=aux,total_objective=sum(x['future_flow_mse'] for x in main)/2+aux['loss'],
                gradient_l2_before_clip=norm,gradient_l2_after_clip=clipped,all_controller_gradients_present_finite=True,
                foundation_gradients_absent=True,main_predictions=2,optimizer_updates=1,
                main_seconds=main_seconds,seconds=time.monotonic()-started)
