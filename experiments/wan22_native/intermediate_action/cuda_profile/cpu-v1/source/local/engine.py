# SPDX-License-Identifier: Apache-2.0
"""Two placements, with all zero gates completed before any original update."""
import math
import time
import torch
from experiments.wan22_native.action_effect.training import effect
from experiments.wan22_native.action_cuda import probe_math
from experiments.wan22_native.official_cpu.streaming import tensor_sha


def identities(data):
    return dict(windows=[{k:tensor_sha(v) for k,v in w.items()} for w in data['windows']],
        draws={k:tensor_sha(v) for k,v in data['draws'].items()},
        positive=tensor_sha(data['positive']), negative=tensor_sha(data['negative']),
        initial={k:tensor_sha(v) for k,v in data['initial'].items()}, rng=tensor_sha(data['rng']))


def exact_comparison(reference, actual):
    result = probe_math.compare_velocity(reference, actual)
    result['passed'] = result['exact_equal']
    result['gate'] = 'Bit-exact FP32 equality; descriptive errors do not relax this requirement'
    return result


def execute(data, *, placements, make_bridge, native_predict, retain, checkpoint, progress,
            check=lambda:None, synchronize=lambda:None, reset_peak=lambda:None,
            memory=lambda:{}, release=lambda:None, verify_core=lambda label:None):
    """Callbacks provide storage/guards; original paired_update supplies math.

    make_bridge must load the same saved initial adapter every time. Two
    parity-only owners are released before two fresh training owners begin.
    No feature from an adapter-dependent suffix is reused across commands.
    """
    if len(placements) != 2 or placements[0] == placements[1]:
        raise ValueError('Two distinct ordered placement indices required')
    if [(r['original_update'],r['k'],r['noise_key']) for r in data['schedule']] != [(1,506,'noise_0000'),(5,265,'noise_0004')]:
        raise ValueError('Only the fixed original draws 1 and 5 are selected')
    original = identities(data)
    report = dict(stage='parity', completed_updates=0, placements={}, parity=[], native_predictions=0,
                  zero_gate_passed=False, image_generation=False, quality_assessed=False)
    def emit():
        progress(report)
    def fresh(index):
        bridge = make_bridge(index)
        values = {name:tensor_sha(p.detach().cpu()) for name,p in bridge.adapter.named_parameters()}
        if values != original['initial'] or any(torch.count_nonzero(p) for p in (bridge.adapter.output.weight,bridge.adapter.output.bias)):
            raise ValueError('Placement did not start from exact saved zero adapter')
        return bridge
    x,times,_ = effect.auxiliary_inputs(data['windows'],data['draws']['noise_0000'])
    refs = {}
    for context_name in ('positive','negative'):
        check(); synchronize(); start=time.monotonic()
        value=native_predict(x,times,data[context_name]); synchronize()
        retain('parity/native-'+context_name,{'velocity':value})
        report['native_predictions'] += 1
        refs[context_name]=value
        report.setdefault('native_seconds',{})[context_name]=time.monotonic()-start
        emit()
    for index in placements:
        bridge=fresh(index); device=next(bridge.adapter.parameters()).device
        reset_peak(); begin=time.monotonic()
        for context_name in ('positive','negative'):
            context=data[context_name].to(device)
            for arm,window in zip(('closed','open'),data['windows']):
                check(); synchronize(); start=time.monotonic()
                value=bridge(x.to(device),times.to(device),[context],commands=window['commands'].to(device),
                             observation=window['observation'].to(device),track_grad=False)
                synchronize(); value=value.detach().cpu()
                retain(f'parity/block{index}-{context_name}-{arm}-full',{'velocity':value})
                result=exact_comparison(refs[context_name],value)
                result.update(block_index=index,context=context_name,arm=arm,path='full',seconds=time.monotonic()-start)
                report['parity'].append(result);emit()
                if not result['passed']: raise RuntimeError('Full zero parity failed before optimizer creation')
            check(); synchronize(); start=time.monotonic()
            features=bridge.extract_features(x.to(device),times.to(device),[context]); synchronize()
            extract_seconds=time.monotonic()-start
            for arm,window in zip(('closed','open'),data['windows']):
                check(); synchronize();start=time.monotonic()
                value=bridge.predict_from_features(features,window['commands'].to(device),window['observation'].to(device),track_grad=False)
                synchronize();value=value.detach().cpu()
                retain(f'parity/block{index}-{context_name}-{arm}-cached',{'velocity':value})
                result=exact_comparison(refs[context_name],value)
                result.update(block_index=index,context=context_name,arm=arm,path='cached',seconds=time.monotonic()-start,
                              feature_extract_seconds=extract_seconds)
                report['parity'].append(result);emit()
                if not result['passed']:raise RuntimeError('Cached zero parity failed before optimizer creation')
            del features
        synchronize();report['placements'][str(index)]={'parity_seconds':time.monotonic()-begin,'parity_memory':memory()}
        del bridge,context,value
        release()
    report['zero_gate_passed']=True;emit()
    verify_core('after-parity')
    for index in placements:
        bridge=fresh(index);optimizer=torch.optim.AdamW(bridge.adapter.parameters(),**probe_math.OPTIMIZER)
        if optimizer.state:raise ValueError('Fresh AdamW required')
        label=f'block{index}'
        checkpoint(label,0,bridge,optimizer,report)
        rows=[];reset_peak();begin=time.monotonic()
        class RetainingBridge:
            core,adapter=bridge.core,bridge.adapter
            calls=0
            def __call__(self,*args,**kwargs):
                value=bridge(*args,**kwargs)
                update,arm=divmod(self.calls,2)
                retain(f'{label}/main-{update+1:04d}-'+('closed','open')[arm],{'velocity':value})
                self.calls+=1
                return value
            def extract_features(self,*a,**kw):return bridge.extract_features(*a,**kw)
            def predict_from_features(self,*a,**kw):return bridge.predict_from_features(*a,**kw)
        proxy=RetainingBridge()
        for number,row in enumerate(data['schedule'],1):
            report['stage']=f'{label}/update{number}';emit()
            check();synchronize();start=time.monotonic()
            result=effect.paired_update(proxy,data['windows'],data['draws'][row['noise_key']],row['k'],data['positive'],optimizer,
                auxiliary=True,negative_context=data['negative'],update=number,check=check,
                retain=lambda name,values:retain(label+'/'+name,values))
            synchronize();result['seconds']=time.monotonic()-start
            retain(f'{label}/gradients-after-clip-{number:04d}',{name:p.grad for name,p in bridge.adapter.named_parameters()})
            rows.append(result);report['placements'][str(index)]['updates']=rows
            recurrent=result['command_gru_gradient_l2']
            if (not math.isfinite(recurrent) or (number==1 and recurrent!=0) or (number==2 and recurrent<=0)):
                emit()
                raise RuntimeError('Expected zero then finite positive recurrent gradient was not observed')
            report['completed_updates']+=1
            checkpoint(label,number,bridge,optimizer,report);emit()
        if proxy.calls!=4 or any(r['auxiliary']['feature_extracts']!=2 or r['auxiliary']['head_predictions']!=4 for r in rows):
            raise RuntimeError('Original main/auxiliary prediction counts changed')
        synchronize();report['placements'][str(index)].update(training_seconds=time.monotonic()-begin,training_memory=memory())
        verify_core('after-'+label)
        optimizer.zero_grad(set_to_none=True)
        if any(p.grad is not None or p.requires_grad for p in bridge.core.parameters()):raise RuntimeError('Foundation gradients present')
        if any(m._forward_hooks or m._forward_pre_hooks for m in bridge.core.modules()):raise RuntimeError('Leaked model hooks')
        del proxy,optimizer,bridge,RetainingBridge
        release();emit()
    if identities(data)!=original:raise RuntimeError('Saved input tensors changed')
    report.update(status='passed',inputs_unchanged=True,base_unchanged=True,stage='complete')
    emit()
    return report
