# SPDX-License-Identifier: Apache-2.0
"""One guarded real-data FP32 adapter update, with explicit tokenwise times.

Running this command uses MPS. This is a mechanics/resource profile on one
original development clip, not a quality study, sampler or trained I2V release.
No downloads occur, no previous adapter weights are loaded, and no video is made.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from safetensors.torch import save_file
from fetch_weights import sha,FILES
from text_cache.cache import load_context
from native_control.evidence import Evidence
from native_control.profile_pair import load_core
from tokenwise_time.portable import extend_model
from tokenwise_time.training import ActionObservationAdapter,training_pair,sample_training_inputs,predict,optimizer_update
from tokenwise_time.training_data import read_training_window,tensor_sha

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent


def module_sha(model):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        result.update(name.encode());result.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def validate_report(path,required,min_tests=None,min_rows=None):
    report=json.loads(Path(path).read_text())
    if report.get('status')!='passed':raise ValueError('Passed CPU report required: '+str(path))
    if min_tests is not None and report.get('tests',0)<min_tests:raise ValueError('Insufficient CPU test coverage')
    if min_rows is not None and len(report.get('rows',[]))<min_rows:raise ValueError('Insufficient CPU parity coverage')
    for name in required:
        if report.get('source_sha256',{}).get(name)!=sha(PARENT/name):
            raise ValueError('Source changed since CPU review: '+name)
    return sha(path)


def validate_preflight(parity,training,review):
    first=validate_report(parity,['tokenwise_time/portable.py','tokenwise_time/test_parity.py',
        'native_control/portable.py','native_control/vendor/model.py'],min_rows=56)
    second=validate_report(training,['tokenwise_time/training.py','tokenwise_time/test_training.py',
        'tokenwise_time/portable.py','adapter.py','native_control/portable.py','native_control/vendor/model.py'],min_tests=5)
    third=validate_report(review,['tokenwise_time/training.py','tokenwise_time/training_data.py',
        'tokenwise_time/profile_update.py','tokenwise_time/test_training_independent.py',
        'tokenwise_time/portable.py','adapter.py'],min_tests=4)
    return {'uniform_time_parity_sha256':first,'training_mechanics_sha256':second,'independent_training_review_sha256':third}


def snapshot_sources(output):
    names=['tokenwise_time/'+name for name in ('portable.py','training.py','training_data.py','profile_update.py',
        'test_parity.py','test_training.py','test_training_independent.py')]
    names+=['adapter.py','fetch_weights.py','PROTOCOL.md','requirements-real.txt','requirements.txt',
        'text_cache/cache.py','text_cache/prompts.json','native_control/portable.py',
        'native_control/profile_pair.py','native_control/evidence.py','native_control/sampling.py',
        'native_control/clamp_cache.py','native_control/vendor/model.py','native_control/vendor/attention.py',
        'native_control/vendor/fm_solvers_unipc.py','native_control/vendor/source-manifest.json',
        'native_control/vendor/LICENSE-APACHE-2.0.txt']
    source=output/'measured-source';source.mkdir()
    hashes={}
    for name in names:
        path=PARENT/name;dest=source/(name+'.txt');dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);hashes[name]=sha(path)
        if sha(dest)!=hashes[name]:raise RuntimeError('Source snapshot changed during copy')
    return hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--capture-cache',type=Path,required=True)
    p.add_argument('--text-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',choices=[f'{arm}-{start:04d}' for start in (0,8,32,49) for arm in ('closed','open')],default='open-0000')
    p.add_argument('--seed',type=int,default=20260911)
    p.add_argument('--parity-report',type=Path,default=HERE/'results/cpu-parity.json')
    p.add_argument('--training-report',type=Path,default=HERE/'results/training-cpu-tests.json')
    p.add_argument('--independent-report',type=Path,default=HERE/'results/training-independent-tests.json')
    args=p.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.output.resolve().is_relative_to(HERE):
        p.error('Output must be a new directory outside tokenwise_time')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':
        raise RuntimeError('Automatic MPS CPU fallback must be disabled for this profile')
    if not torch.backends.mps.is_available():raise RuntimeError('The declared actual-shape profile requires MPS')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    evidence=Evidence(args.output);report=evidence.report;error=None;completed=False
    report.update(experiment='One real-data tokenwise-time adapter update',requested_updates=1,completed_updates=0,
        split='Single-layout development mechanics',device='mps',core_dtype='float32',adapter_dtype='float32',
        training_mixture={'image_conditioned':1.0,'uniform_t2v':0.0},
        image_conditioning='Independent clean observed latent has time 0; future has matching k/1000 noise and exact integer time k',
        time_distribution='Integer k uniformly sampled from 50..950 inclusive; sigma=k/1000',
        loss='MSE on future latent frames 1..4 only, target=noise-target; no observed loss',
        actions='Actual ordered 16 six-channel commands from original capture; mechanics only, no control-quality claim',
        observation_adapter='32 pooled tokens from independently encoded initial observation; not persistent memory',
        training_cfg=False,pretrained_i2v_reproduced=False,video_generation=False,novelty_claim=False,
        original_adapter_initialization='Fresh CPU seed initialization, no previously trained checkpoint',
        original_adapter_seed=args.seed,noise_seed=args.seed+1,
        optimizer={'type':'AdamW','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'clip_norm':1.},
        checkpointing='Non-reentrant frozen block forward; adapter forward hooks outside replay; retained through backward',
        platform=platform.platform(),torch=torch.__version__,automatic_mps_cpu_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0'))
    try:
        report['cpu_checks']=validate_preflight(args.parity_report,args.training_report,args.independent_report)
        report['source_sha256']=snapshot_sources(args.output)
        window,provenance=read_training_window(args.capture_cache,args.window);report['capture']=provenance
        prompts=json.loads((PARENT/'text_cache/prompts.json').read_text())['prompts']
        positive_text=next(row['text'] for row in prompts if row['id']=='atrium')
        context=load_context(args.text_cache,'atrium',expected_text=positive_text,device='cpu',dtype=torch.float32)
        report['text']={'id':'atrium','text_sha256':hashlib.sha256(positive_text.encode()).hexdigest(),
            'tensor_sha256':tensor_sha(context),'manifest_sha256':sha(args.text_cache/'manifest.json'),
            'file_sha256':sha(args.text_cache/'embeddings.safetensors'),'shape':list(context.shape),
            'training_cfg':False,'native_padding':'Unpadded context, then 512-token padding before projection, no context mask'}
        adapter=ActionObservationAdapter(1536)
        report['trainable_parameters']=sum(p.numel() for p in adapter.parameters())
        if report['trainable_parameters']!=1349376:raise RuntimeError('Unexpected original adapter size')
        report['adapter_sha256_before']=module_sha(adapter)
        save_file({k:v.detach().contiguous() for k,v in adapter.state_dict().items()},str(args.output/'initial-adapter.safetensors'))
        generator=torch.Generator(device='cpu').manual_seed(args.seed+1)
        k,noise=sample_training_inputs(window['target'],generator)
        noisy,velocity,times=training_pair(window['target'],noise,window['observation'],k)
        save_file({'noise':noise,'integer_time':k,'token_times':times,'noisy_training_latent':noisy},str(args.output/'sampled-training-inputs.safetensors'))
        report['training_sample']={'k':k.tolist(),'sigma_float32':(k.float()/1000).tolist(),'sigma_exact_rational':[str(int(value))+'/1000' for value in k],'observed_token_times':0,
            'observed_tokens_per_example':576,'future_tokens_per_example':2304,'token_times_sha256':tensor_sha(times),
            'noise_sha256':tensor_sha(noise),'noisy_latent_sha256':tensor_sha(noisy),'prediction_target_sha256':tensor_sha(velocity),
            'latent_shape':list(noisy.shape),'artifact_sha256':sha(args.output/'sampled-training-inputs.safetensors')}
        core,base=evidence.measure('load_verified_fp32_base',lambda:load_core(args.weights,'mps'))
        extend_model(core)
        # This is a plain attribute, not a parameter/buffer. Prepare outside the
        # following inference-mode check so backward never saves inference tensors.
        core.freqs=core.freqs.to(core.patch_embedding.weight.device).clone()
        if core.freqs.is_inference():raise RuntimeError('Training RoPE table must not be an inference tensor')
        report['training_rope_table_inference_tensor']=False
        report['foundation']=base
        report['base_tensor_sha256_before']=evidence.measure('hash_frozen_base_before',lambda:module_sha(core))
        adapter=adapter.to('mps');context=context.to('mps')
        values=[v.to('mps') for v in (noisy,velocity,times,window['observation'],window['actions'])]
        noisy,velocity,times,observation,actions=values
        def zero_check():
            with torch.inference_mode():
                native=torch.stack(core(list(noisy.unbind(0)),times,[context],2880))
                attached=predict(core,adapter,noisy,times,observation,actions,[context])
                if not torch.equal(native,attached):raise RuntimeError('Actual zero-init adapter differs from extended base')
            return True
        report['actual_zero_init_equal']=evidence.measure('zero_initialized_adapter_equality',zero_check)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        report['update']=evidence.measure('one_optimizer_update',lambda:optimizer_update(core,adapter,optimizer,noisy,velocity,times,observation,actions,[context]))
        report['completed_updates']=1
        report['adapter_sha256_after']=module_sha(adapter)
        if report['adapter_sha256_after']==report['adapter_sha256_before']:raise RuntimeError('Adapter did not change')
        if adapter._hooks:raise RuntimeError('Adapter hooks were not removed')
        save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},str(args.output/'updated-adapter.safetensors'))
        report['base_tensor_sha256_after']=evidence.measure('hash_frozen_base_after',lambda:module_sha(core))
        if report['base_tensor_sha256_after']!=report['base_tensor_sha256_before']:raise RuntimeError('Frozen core tensor bytes changed')
        for name in ('diffusion_pytorch_model.safetensors','config.json'):
            if sha(args.weights/name)!=FILES[name][1]:raise RuntimeError('External base/config changed during run')
        report['base_parameters_unchanged']=True;report['hooks_removed']=True
        report['output_sha256']={f:sha(args.output/f) for f in ['initial-adapter.safetensors','updated-adapter.safetensors','sampled-training-inputs.safetensors']}
        completed=True
    except BaseException as problem:error=problem;raise
    finally:
        if not completed and error is None:error=RuntimeError('Profile did not complete')
        evidence.close(error)
    print(json.dumps({'status':report['status'],'update':report['update'],'timings':report['timings'],'peaks':report['peaks_sampled']},indent=2))

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='.*torch.cuda.amp.autocast.*')
        warnings.filterwarnings('ignore',message="User provided device_type of 'cuda'.*")
        main()
