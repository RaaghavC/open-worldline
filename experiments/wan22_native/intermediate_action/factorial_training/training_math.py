# SPDX-License-Identifier: Apache-2.0
"""Prospective six-arm schedule; unchanged original main/auxiliary updates."""
import time
import math
import torch
from experiments.wan22_native.action_effect.training import effect
from experiments.wan22_native.action_cuda import probe_math
from experiments.wan22_native.official_cpu.streaming import tensor_sha

MOTIONS=('stationary','left','right')
CHECKPOINTS=tuple(range(0,129,16))


def schedule(original):
    if len(original)!=128:raise ValueError('All128 original saved draw rows required')
    result=[]
    for i,row in enumerate(original):
        if (row['update']!=i+1 or row['noise_key']!=f'noise_{i:04d}' or type(row['k']) is not int
                or not 50<=row['k']<=950 or row['sigma']!=row['k']/1000):raise ValueError('Original noise/time schedule malformed')
        motion=MOTIONS[i%3]
        result.append(dict(update=i+1,motion=motion,branches=[motion+'_closed',motion+'_interact'],
            auxiliary=i%4==0,noise_key=row['noise_key'],noise_sha256=row['noise_sha256'],
            k=row['k'],sigma=row['sigma'],rng_after_sha256=row['rng_after_sha256']))
    return result


def validate_windows(windows):
    if set(windows)!={m+'_'+a for m in MOTIONS for a in ('closed','interact')}:raise ValueError('Exactly six native arms required')
    first=None
    for motion in MOTIONS:
        pair=[windows[motion+'_'+a] for a in ('closed','interact')]
        for window in pair:
            if set(window)!={'target','observation','commands'}:raise ValueError('Only target,observation,commands allowed')
            target=window['target'];observation=window['observation']
            if (target.ndim!=5 or target.shape[0]!=1 or target.shape[2]!=5
                    or tuple(observation.shape)!=tuple(target[:,:,:1].shape)
                    or tuple(window['commands'].shape)!=(1,16,6)):
                raise ValueError('B1 five-latent target, one observation and sixteen commands required')
            for value in window.values():
                if value.dtype!=torch.float32 or value.device.type!='cpu' or value.requires_grad or not torch.isfinite(value).all():raise ValueError('Finite frozen CPU inputs required')
            if not torch.equal(window['target'][:,:,:1],window['observation']) or tensor_sha(window['target'][:,:,:1])!=tensor_sha(window['observation']):
                raise ValueError('Bit-exact raw target prefix required; no patching')
            obs=tensor_sha(window['observation'])
            if first is not None and obs!=first:raise ValueError('Shared original observation required')
            first=obs
        difference=pair[1]['commands']-pair[0]['commands'];expected=torch.zeros_like(difference);expected[0,0,5]=1.
        if not torch.equal(difference,expected):raise ValueError('Only first interaction pulse may differ within a motion pair')
        expected_yaw={'stationary':0.,'left':math.pi/120,'right':-math.pi/120}[motion]
        for window in pair:
            expected=torch.zeros_like(window['commands']);expected[0,:,3]=expected_yaw
            expected[0,0,5]=float(window is pair[1])
            if not torch.equal(window['commands'],expected):raise ValueError('Exact published factorial command units/order required')


def run(bridge,windows,rows,noise,context,negative,optimizer,*,initial,retain,checkpoint,progress,
        native_predict,check=lambda:None,synchronize=lambda:None):
    validate_windows(windows)
    if rows!=schedule(rows) or optimizer.state:raise ValueError('Fresh exact 128/32 motion schedule required')
    if any(tensor_sha(p.detach().cpu())!=tensor_sha(initial[n]) for n,p in bridge.adapter.named_parameters()):raise ValueError('Fresh saved zero adapter required')
    if any(torch.count_nonzero(p) for p in (bridge.adapter.output.weight,bridge.adapter.output.bias)):raise ValueError('Zero initial output required')
    report=dict(status='running',completed_updates=0,updates=[],parity=[],zero_gate_passed=False,
                main_predictions=0,auxiliary_predictions=0,auxiliary_feature_extracts=0,auxiliary_updates=0,
                quality_assessed=False,image_generation=False,block_index=28)
    device=next(bridge.adapter.parameters()).device
    first=rows[0];epsilon=noise(first)
    for arm in first['branches']:
        window=windows[arm];x,times,_=effect.flow_inputs(window['target'],window['observation'],epsilon,first['k'])
        before={k:tensor_sha(v) for k,v in window.items()}
        before.update(noisy=tensor_sha(x),times=tensor_sha(times),context=tensor_sha(context))
        check();reference=native_predict(x,times,context)
        retain('parity-'+arm+'-native',{'velocity':reference})
        noisy=x.to(device)
        actual=bridge(noisy,times.to(device),[context.to(device)],commands=window['commands'].to(device),
                      observation=window['observation'].to(device),track_grad=False).detach().cpu()
        retain('parity-'+arm+'-bridge',{'velocity':actual})
        result=probe_math.compare_velocity(reference,actual);result['passed']=result['exact_equal'];result['arm']=arm
        after={k:tensor_sha(v) for k,v in window.items()}
        after.update(noisy=tensor_sha(x),times=tensor_sha(times),context=tensor_sha(context))
        result['input_identity']=before
        result['inputs_unchanged']=before==after
        result['observed_prefix_exact']=torch.equal(noisy[:,:,:1].detach().cpu(),window['observation'])
        report['parity'].append(result);progress(report)
        if not result['passed'] or not result['inputs_unchanged'] or not result['observed_prefix_exact']:
            raise RuntimeError('Exact new-data zero parity or input preservation failed before updates')
    report['zero_gate_passed']=True;checkpoint(0,report);progress(report)
    class RetainingBridge:
        core,adapter=bridge.core,bridge.adapter
        calls=0
        def __call__(self,*a,**kw):
            value=bridge(*a,**kw);i,arm=divmod(self.calls,2)
            retain(f'main-{i+1:04d}-'+('closed','interact')[arm],{'velocity':value});self.calls+=1
            return value
        def extract_features(self,*a,**kw):return bridge.extract_features(*a,**kw)
        def predict_from_features(self,*a,**kw):return bridge.predict_from_features(*a,**kw)
    proxy=RetainingBridge()
    for row in rows:
        check();synchronize();start=time.monotonic();epsilon=noise(row)
        if tensor_sha(epsilon)!=row['noise_sha256']:raise ValueError('Saved original noise changed')
        result=effect.paired_update(proxy,[windows[a] for a in row['branches']],epsilon,row['k'],context,optimizer,
            auxiliary=row['auxiliary'],negative_context=negative,retain=retain,update=row['update'],check=check)
        synchronize();result.update(seconds=time.monotonic()-start,schedule=row)
        retain(f'gradients-after-clip-{row["update"]:04d}',{n:p.grad for n,p in bridge.adapter.named_parameters()})
        report['updates'].append(result)
        recurrent=result['command_gru_gradient_l2']
        if (not math.isfinite(recurrent) or (row['update']==1 and recurrent!=0)
                or (row['update']==2 and recurrent<=0)):
            progress(report);raise RuntimeError('Expected zero then positive recurrent gradient missing')
        report['completed_updates']=row['update'];report['main_predictions']+=2
        report['auxiliary_updates']+=int(row['auxiliary'])
        report['auxiliary_predictions']+=result['auxiliary']['head_predictions']
        report['auxiliary_feature_extracts']+=result['auxiliary']['feature_extracts']
        if row['update'] in CHECKPOINTS:checkpoint(row['update'],report)
        progress(report)
    if (proxy.calls!=256 or report['auxiliary_updates']!=32 or report['auxiliary_predictions']!=128
            or report['auxiliary_feature_extracts']!=64):raise RuntimeError('Original main/auxiliary counts differ')
    validate_windows(windows)
    return report
