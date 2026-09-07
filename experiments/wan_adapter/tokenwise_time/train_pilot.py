# SPDX-License-Identifier: Apache-2.0
"""Exactly 16 original adapter updates on eight development windows, twice.

No new foundation weights, text encoding, GPU run or download occurs on import.
Use only after the declared GPU reservation and independent CPU review.
"""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from fetch_weights import sha,FILES
from text_cache.cache import load_context
from native_control.evidence import Evidence
from native_control.profile_pair import load_core
from tokenwise_time.portable import extend_model
from tokenwise_time.training import optimizer_update,predict
from tokenwise_time.training_data import read_training_window,tensor_sha
from tokenwise_time import pilot_common as pc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','capture-cache','text-cache','output','cpu-report','independent-report'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();pc.reject_output(args.output)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':raise RuntimeError('Automatic CPU fallback must be disabled')
    if not torch.backends.mps.is_available():raise RuntimeError('This declared run requires MPS')
    torch.set_num_threads(4);torch.manual_seed(pc.SEED)
    evidence=Evidence(args.output);r=evidence.report;error=None;completed=False
    r.update(experiment='Fixed 16-update tokenwise-time development pilot',requested_updates=16,completed_updates=0,
        schedule=list(pc.SCHEDULE),adapter_seed=pc.SEED,training_noise_seed=pc.SEED+1,diagnostic_noise_seed=pc.SEED+2,
        device='mps',core_dtype='float32',adapter_dtype='float32',training_mixture={'image_conditioned':1.,'uniform_t2v':0.},
        time_distribution='k uniformly sampled from integers 50..950; sigma=k/1000; independent observed prefix time 0',
        loss='Future-only latent flow MSE; target velocity=noise-target; no observed loss',training_cfg=False,
        optimizer={'type':'AdamW','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'clip_norm':1.},
        checkpoint_selection='Only the fixed final update16; no loss-based selection, early stopping or hyperparameter sweep',
        diagnostic_scope='All eight training-layout development windows, fixed saved noises and times; not generated-image quality or held-out evaluation',
        pretrained_i2v_reproduced=False,persistent_memory_trained=False,video_generated=False,
        automatic_mps_cpu_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0'),
        platform=platform.platform(),dependencies={n:importlib.metadata.version(n) for n in ('torch','numpy','safetensors','psutil','diffusers')},
        updates=[],diagnostics={'before':[],'after':[]})
    try:
        r['cpu_checks']=pc.validate_preflight(pc.HERE/'results/cpu-parity.json',pc.HERE/'results/training-cpu-tests.json',pc.HERE/'results/training-independent-tests.json')
        r['pilot_cpu_sha256']=pc.check_pilot_report(args.cpu_report)
        r['pilot_independent_sha256']=pc.check_pilot_report(args.independent_report,independent=True)
        r['source_sha256']=pc.snapshot_sources(args.output)
        review_paths={'cpu-tests.json':args.cpu_report,'independent-review.json':args.independent_report,
            'uniform-time-parity.json':pc.HERE/'results/cpu-parity.json',
            'training-mechanics.json':pc.HERE/'results/training-cpu-tests.json',
            'training-independent-review.json':pc.HERE/'results/training-independent-tests.json'}
        for name,path in review_paths.items():shutil.copyfile(path,args.output/name)
        windows={};r['capture']={}
        for name in pc.WINDOWS:windows[name],r['capture'][name]=read_training_window(args.capture_cache,name)
        context=load_context(args.text_cache,'atrium',expected_text=pc.POSITIVE_TEXT,device='cpu',dtype=torch.float32)
        r['text']={'manifest_sha256':sha(args.text_cache/'manifest.json'),'file_sha256':sha(args.text_cache/'embeddings.safetensors'),
            'id':'atrium','tensor_sha256':tensor_sha(context),'shape':list(context.shape),'text':pc.POSITIVE_TEXT}
        adapter=pc.ActionObservationAdapter(1536)
        if sum(v.numel() for v in adapter.parameters())!=1349376:raise RuntimeError('Unexpected adapter parameter count')
        r['adapter_sha256_before']=pc.module_sha(adapter)
        pc.atomic_tensors(args.output/'initial-adapter.safetensors',adapter.state_dict())
        inputs,records=pc.prepare_inputs(windows)
        r['input_records']=records
        r['fixed_inputs_sha256']=pc.atomic_tensors(args.output/'fixed-inputs.safetensors',inputs)
        core,r['foundation']=evidence.measure('load_verified_fp32_base',lambda:load_core(args.weights,'mps'))
        extend_model(core)
        core.freqs=core.freqs.to(core.patch_embedding.weight.device).clone()
        if core.freqs.is_inference():raise RuntimeError('Training RoPE table must not be an inference tensor')
        r['base_tensor_sha256_before']=evidence.measure('hash_frozen_base_before',lambda:pc.module_sha(core))
        adapter=adapter.to('mps');context=context.to('mps')
        with torch.inference_mode():
            x,v,t,obs,actions=pc.example(windows[pc.WINDOWS[0]],inputs,'train.00','mps')
            plain=torch.stack(core(list(x.unbind(0)),t,[context],2880))
            attached=predict(core,adapter,x,t,obs,actions,[context])
            r['actual_zero_init_equal']=torch.equal(plain,attached)
            if not r['actual_zero_init_equal']:raise RuntimeError('Zero adapter changed the tokenwise base output')
        del x,v,t,obs,actions,plain,attached
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        r['last_recovery_sha256']=pc.save_recovery(args.output,adapter,optimizer,0)
        for stage in ('before','after'):
            if stage=='after':
                for index,name in enumerate(pc.SCHEDULE):
                    values=pc.example(windows[name],inputs,f'train.{index:02d}','mps')
                    row=evidence.measure(f'optimizer_update_{index+1:02d}',lambda:optimizer_update(core,adapter,optimizer,*values,[context]))
                    if adapter._hooks:raise RuntimeError('Adapter hook survived optimizer update')
                    digest=pc.save_recovery(args.output,adapter,optimizer,index+1)
                    r['updates'].append({'update':index+1,'window':name,**row,'recovery_sha256':digest})
                    r['completed_updates']=index+1;r['last_recovery_sha256']=digest;evidence.save()
                    del values;gc.collect();torch.mps.empty_cache()
                pc.atomic_tensors(args.output/'final-adapter.safetensors',adapter.state_dict())
            for index,name in enumerate(pc.WINDOWS):
                row,prediction=evidence.measure(f'{stage}_fixed_diagnostic_{index+1}',lambda:pc.diagnostic(core,adapter,windows[name],inputs,f'diagnostic.{index:02d}',context,'mps'))
                diagnostic_file=f'{stage}-prediction-{index:02d}.safetensors'
                r['diagnostics'][stage].append({'window':name,**row,'prediction_file':diagnostic_file,
                    'prediction_file_sha256':pc.atomic_tensors(args.output/diagnostic_file,{'prediction':prediction})})
                evidence.save();del prediction;gc.collect();torch.mps.empty_cache()
        for before,after in zip(r['diagnostics']['before'],r['diagnostics']['after']):
            for key in ('window','noisy_latent_sha256','target_velocity_sha256','token_times_sha256'):
                if before[key]!=after[key]:raise RuntimeError('Fixed diagnostic inputs changed')
        r['adapter_sha256_after']=pc.module_sha(adapter)
        if r['adapter_sha256_before']==r['adapter_sha256_after']:raise RuntimeError('Adapter did not change')
        r['base_tensor_sha256_after']=evidence.measure('hash_frozen_base_after',lambda:pc.module_sha(core))
        if r['base_tensor_sha256_before']!=r['base_tensor_sha256_after']:raise RuntimeError('Frozen base changed')
        for name in ('diffusion_pytorch_model.safetensors','config.json'):
            if sha(args.weights/name)!=FILES[name][1]:raise RuntimeError('External foundation file changed')
        r['base_parameters_unchanged']=True;r['hooks_removed']=not adapter._hooks
        r['output_sha256']={n:sha(args.output/n) for n in ('initial-adapter.safetensors','final-adapter.safetensors','fixed-inputs.safetensors','recovery-last.pt')}
        completed=True
    except BaseException as problem:error=problem;raise
    finally:
        if not completed and error is None:error=RuntimeError('Fixed training pilot did not complete')
        evidence.close(error)
    print(json.dumps({'status':r['status'],'completed_updates':r['completed_updates'],'elapsed_seconds':r['elapsed_seconds'],'peaks':r['peaks_sampled']},indent=2))

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='.*torch.cuda.amp.autocast.*')
        warnings.filterwarnings('ignore',message="User provided device_type of 'cuda'.*")
        main()
