# SPDX-License-Identifier: Apache-2.0
"""Train an original small Wan adapter on eight original Atrium development windows.

This is real-data adapter training on an attributed frozen foundation. It is
not a new foundation model, scene generalization test, or streaming system.
"""
import argparse
import math
from pathlib import Path
import torch
from safetensors.torch import save_file
from training_common import (ActionObservationAdapter,CaptureCache,Evidence,ORDER,PROTOCOL,
    flow_training_pair,future_mse,load_core,load_texts,module_sha,predict,sha,sync,tensor_sha,to_device)


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--capture-cache',type=Path,required=True)
    p.add_argument('--text-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--seed',type=int,default=20260907)
    p.add_argument('--max-seconds',type=float,default=900)
    a=p.parse_args()
    if not 1<=a.steps<=10:p.error('This bounded pilot permits 1 through10 updates')
    # All real-data integrity gates run on CPU before any GPU reservation.
    cache=CaptureCache(a.capture_cache)
    windows={key:cache.read(key,include_target=True) for key in ORDER}
    positive,negative,text_info=load_texts(a.text_cache,'cpu')
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    evidence=Evidence(a.output,a.device,a.max_seconds,'Real flow-matching updates of original adapter on single-layout development data')
    report=evidence.report;error=None
    report.update(seed=a.seed,requested_updates=a.steps,batch_size=1,
        window_order=[ORDER[i%len(ORDER)] for i in range(a.steps)],
        capture_cache_manifest_sha256=cache.manifest_sha256,text=text_info,
        optimizer={'type':'AdamW','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'gradient_clip_norm':1.},
        flow={'noising':'x_sigma=(1-sigma)*target+sigma*noise; overwrite latent0 with observation',
              'prediction_target':'noise-target','sigma_distribution':'Uniform[.05,.95], CPU generator seed',
              'model_timestep':'1000*sigma','loss':'Mean squared error on latent frames1..4 only; observed frame0 excluded'},
        conditioning={'observation':'separately encoded first RGB, initial latent clamped and 32 observation tokens',
                      'actions':'16 ordered six-channel raw commands','text':'same atrium prompt for both arms',
                      'training_cfg':False,'clean_future_conditioning':False},
        evaluation={'window':'open-0000','sigma':.5,'noise_seed':a.seed+1000,
                    'controls':['untrained_before','trained_after','shuffled_actions','masked_observation_tokens'],
                    'masked_observation_tokens':'Adapter observation input zeroed; learned biases can create nonzero tokens; observed latent0 remains clamped',
                    'use':'fixed development flow predictions, not semantic control or generalization evidence'},
        source_sha256={f:sha(Path(__file__).with_name(f)) for f in ['train_clip.py','training_common.py','adapter.py','compat.py','PROTOCOL.md']},
        updates=[])
    evidence.save()
    try:
        core,base=evidence.measure('load_frozen_core',lambda:load_core(a.weights,a.device));report['foundation']=base
        positive=positive.to(a.device,torch.float16)
        adapter=ActionObservationAdapter(core.dim).to(a.device)
        report['adapter_parameters']=sum(p.numel() for p in adapter.parameters())
        report['initial_adapter_sha256']=module_sha(adapter)
        save_file(cpu_tree(adapter.state_dict()),str(a.output/'untrained-adapter.safetensors'))
        evaluation_window=to_device(windows['open-0000'],a.device)
        eval_generator=torch.Generator(device='cpu').manual_seed(a.seed+1000)
        fixed_noise=torch.randn(windows['open-0000']['target'].shape,generator=eval_generator).to(a.device,torch.float16)
        eval_input,eval_target=flow_training_pair(evaluation_window['target'],fixed_noise,evaluation_window['observation'],.5)
        permutation=torch.randperm(16,generator=torch.Generator().manual_seed(a.seed+2000))
        if torch.equal(evaluation_window['actions'][:,permutation.to(evaluation_window['actions'].device)],evaluation_window['actions']):
            raise ValueError('Shuffled-action control must change the actual command tensor')
        report['evaluation']['shuffled_action_permutation']=permutation.tolist()
        report['evaluation']['noise_tensor_sha256']=tensor_sha(fixed_noise)
        report['evaluation']['fixed_noised_input_sha256']=tensor_sha(eval_input)
        predictions={};eval_records={}
        def evaluate(label):
            commands=evaluation_window['actions']
            observation=evaluation_window['observation']
            if label=='shuffled_actions':commands=commands[:,permutation.to(commands.device)]
            if label=='masked_observation_tokens':observation=torch.zeros_like(observation)
            prediction=predict(core,adapter,eval_input,observation,commands,[positive],500.)
            if not torch.isfinite(prediction).all().item():raise RuntimeError('Non-finite fixed development prediction')
            if label=='untrained_before':
                with torch.inference_mode():
                    untouched=core(list(eval_input.unbind(0)),torch.tensor([500.],device=a.device),[positive],2880)
                if not torch.equal(untouched,prediction):raise RuntimeError('Actual zero-init adapter differs from frozen core')
                report['actual_zero_init_equal']=True
            loss=future_mse(prediction,eval_target).item()
            predictions[label]=prediction.detach().float().cpu().contiguous()
            eval_records[label]={'future_latent_mse':loss,'prediction_sha256':tensor_sha(predictions[label])}
            return loss
        evidence.measure('fixed_prediction_untrained',lambda:evaluate('untrained_before'))
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        generator=torch.Generator(device='cpu').manual_seed(a.seed)
        for index in range(a.steps):
            key=ORDER[index%len(ORDER)];window=to_device(windows[key],a.device)
            sigma=.05+.90*torch.rand((),generator=generator).item()
            noise=torch.randn(windows[key]['target'].shape,generator=generator).to(a.device,torch.float16)
            noisy,target=flow_training_pair(window['target'],noise,window['observation'],sigma)
            optimizer.zero_grad(set_to_none=True)
            adapter.attach(core,window['actions'],window['observation'][:,:,0],(5,18,32))
            start_time=__import__('time').perf_counter();sync(a.device)
            try:
                evidence.stage=f'update-{index}-forward'
                prediction=core(list(noisy.unbind(0)),torch.tensor([1000*sigma],device=a.device),[positive],2880)
                loss=future_mse(prediction,target)
                if not torch.isfinite(loss).item():raise RuntimeError('Non-finite real-data training loss')
                evidence.stage=f'update-{index}-backward';loss.backward()
                gradients=[p.grad for p in adapter.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):raise RuntimeError('Absent or non-finite adapter gradients')
                norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.).item()
                if not math.isfinite(norm) or norm<=0:raise RuntimeError('Adapter global gradient norm must be finite and positive')
                if any(p.grad is not None for p in core.parameters()):raise RuntimeError('Frozen foundation received parameter gradients')
                evidence.stage=f'update-{index}-optimizer';optimizer.step()
                if not all(torch.isfinite(p).all().item() for p in adapter.parameters()):raise RuntimeError('Non-finite updated adapter parameters')
                sync(a.device)
                item={'step':index,'window_id':key,'sigma':sigma,'future_latent_mse':loss.item(),
                      'gradient_norm_before_clip':norm,'gradient_tensors':len(gradients),
                      'seconds':__import__('time').perf_counter()-start_time}
                report['updates'].append(item);evidence.save();print(item,flush=True)
            finally:adapter.detach()
        for label in ['trained_after','shuffled_actions','masked_observation_tokens']:
            evidence.measure('fixed_prediction_'+label,lambda label=label:evaluate(label))
        report['fixed_development_predictions']=eval_records
        save_file(predictions,str(a.output/'development-flow-predictions.safetensors'))
        report['updated_adapter_sha256']=module_sha(adapter)
        save_file(cpu_tree(adapter.state_dict()),str(a.output/'trained-adapter.safetensors'))
        report['adapter_checkpoint_sha256']=sha(a.output/'trained-adapter.safetensors')
        report['base_sha256_after']=evidence.measure('verify_frozen_core',lambda:module_sha(core))
        if report['base_sha256_after']!=base['converted_base_sha256']:raise RuntimeError('Frozen foundation changed')
        report['base_parameters_unchanged']=True
        torch.save({'optimizer':cpu_tree(optimizer.state_dict()),'generator_state':generator.get_state(),
                    'completed_updates':len(report['updates']),'seed':a.seed,'protocol':PROTOCOL},a.output/'training-state.pt')
        report['training_state_sha256']=sha(a.output/'training-state.pt')
    except BaseException as e:error=e;raise
    finally:evidence.close(error)

if __name__=='__main__':main()
