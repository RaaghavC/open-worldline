# SPDX-License-Identifier: Apache-2.0
"""One prescribed zero-adapter or update16 clip, in a separate guarded process.

Both arms consume the same original observation, ordered commands, retained
Gaussian noise and native contexts. No future target latent is materialized.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image,ImageDraw
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from fetch_weights import sha,FILES
from text_cache.cache import load_context
from native_control.evidence import Evidence
from native_control.profile_pair import load_core,native_negative_prompt
from native_control.sampling import initial_noise,make_scheduler,SEED,STEPS,SHIFT,GUIDANCE
from codec.helper import OfficialWanCodec
from codec.decode_policy import cleanup_after_temporal_chunk
from tokenwise_time.portable import extend_model
from tokenwise_time.training_data import tensor_sha
from tokenwise_time import pilot_common as pc

CANONICAL_NOISE_FILE_SHA256 = '737ae2a38575fde3d069008c06a66849eecc6e04ec3742b7ac8a6871fcfec5be'
CANONICAL_NOISE_TENSOR_SHA256 = 'ce38d20bcf6ca132aeac8259e0f085fdfe80c64db7745df1c561b294dc201eeb'


def estimate_total(elapsed,first_step,*,remaining_steps=49,decode_and_artifact_allowance=60.):
    values=(elapsed,first_step,decode_and_artifact_allowance)
    if any(not math.isfinite(v) or v<=0 for v in values) or remaining_steps!=49:raise ValueError('Invalid prescribed runtime estimate')
    estimated=elapsed+remaining_steps*first_step+decode_and_artifact_allowance
    if estimated>900:raise RuntimeError(f'Projected total {estimated:.3f} seconds exceeds the unchanged 900-second cap')
    return estimated


def retained_noise(directory):
    directory=Path(directory);r=json.loads((directory/'metrics.json').read_text())
    if r.get('status')!='passed' or r.get('seed')!=SEED or r.get('generated_frames')!=17 or r.get('observed_frames')!=0:
        raise ValueError('The completed pure T2V control is required for its retained noise')
    path=directory/'initial-noise.safetensors'
    file_hash=sha(path)
    if file_hash!=r.get('initial_noise_sha256') or file_hash!=CANONICAL_NOISE_FILE_SHA256:
        raise ValueError('Retained control noise hash mismatch')
    with safe_open(path,framework='pt',device='cpu') as f:
        if set(f.keys())!={'initial_noise'}:raise ValueError('Unexpected noise artifact keys')
        noise=f.get_tensor('initial_noise')
    if noise.shape!=(16,5,36,64) or noise.dtype!=torch.float32 or not torch.isfinite(noise).all():
        raise ValueError('Invalid retained control noise tensor')
    tensor_hash=tensor_sha(noise)
    if tensor_hash!=CANONICAL_NOISE_TENSOR_SHA256 or tensor_hash!=r.get('initial_noise_tensor_sha256'):
        raise ValueError('Retained control noise tensor hash mismatch')
    # The published tensor defines the matched comparison. Torch's CPU normal
    # sampler need not produce identical bits on ARM and x86 for the same seed.
    return noise,{'control_metrics_sha256':sha(directory/'metrics.json'),'noise_file_sha256':file_hash,
        'noise_tensor_sha256':tensor_hash,'source_seed':SEED,
        'identity_rule':'Exact published tensor and file hashes; no cross-platform RNG regeneration requirement'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','capture-cache','text-cache','training-run','output','cpu-report','independent-report'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--control-run',type=Path,default=pc.PARENT/'native_control/results/clip50')
    p.add_argument('--arm',choices=('zero','trained'),required=True)
    args=p.parse_args();pc.reject_output(args.output)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':raise RuntimeError('Automatic CPU fallback must be disabled')
    if not torch.backends.mps.is_available():raise RuntimeError('The declared sampler requires MPS')
    torch.set_num_threads(4);torch.manual_seed(pc.SEED)
    evidence=Evidence(args.output);r=evidence.report;error=None;completed=False
    r.update(experiment='Fixed tokenwise pilot paired sampling arm',arm=args.arm,training_updates=0 if args.arm=='zero' else 16,
        window=pc.SAMPLE_WINDOW,seed=SEED,shift=SHIFT,guidance=GUIDANCE,declared_solver_steps=STEPS,declared_model_calls=2*STEPS,
        observed_frames=1,generated_future_frames=16,total_decoded_frames=17,size=[512,288],preview_playback_fps=10,
        denoiser_device='mps',denoiser_dtype='float32',decoder_dtype='float32',scheduler_device='cpu',scheduler_dtype='float32',
        conditioning='Independent clean initial observation; prefix tokens time0, future tokens exact native solver time; identical ordered actions in both CFG calls',
        sampler='Unchanged official CPU UniPC50, shift8, CFG6; prefix clamped before both model calls and after all updates',
        source_scope='Original small adapter over attributed frozen Wan2.1 T2V foundation; does not reproduce a native pretrained I2V model',
        development_only=True,persistent_memory_claim=False,automatic_mps_cpu_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0'),
        platform=platform.platform(),torch=torch.__version__)
    try:
        r['cpu_sha256']=pc.check_pilot_report(args.cpu_report)
        r['independent_sha256']=pc.check_pilot_report(args.independent_report,independent=True)
        training=pc.validate_training_run(args.training_run)
        r['training_metrics_sha256']=sha(args.training_run/'metrics.json')
        r['source_sha256']=pc.snapshot_sources(args.output)
        shutil.copyfile(args.cpu_report,args.output/'cpu-tests.json')
        shutil.copyfile(args.independent_report,args.output/'independent-review.json')
        observation,actions,r['capture']=pc.read_conditions(args.capture_cache)
        noise,r['noise_provenance']=retained_noise(args.control_run)
        pc.atomic_tensors(args.output/'initial-noise.safetensors',{'initial_noise':noise})
        pc.atomic_tensors(args.output/'conditions.safetensors',{'observation':observation,'actions':actions})
        positive=load_context(args.text_cache,'atrium',expected_text=pc.POSITIVE_TEXT,device='cpu',dtype=torch.float32)
        negative=load_context(args.text_cache,'native_negative',expected_text=native_negative_prompt(),device='cpu',dtype=torch.float32)
        if tensor_sha(positive)!=training['text']['tensor_sha256']:raise ValueError('Sampling positive text differs from training')
        r['text']={'manifest_sha256':sha(args.text_cache/'manifest.json'),'file_sha256':sha(args.text_cache/'embeddings.safetensors'),
            'positive_tensor_sha256':tensor_sha(positive),'negative_tensor_sha256':tensor_sha(negative)}
        adapter=pc.ActionObservationAdapter(1536)
        if pc.module_sha(adapter)!=training['adapter_sha256_before']:raise ValueError('Initial adapter differs from the prescribed fresh seed')
        checkpoint='initial-adapter.safetensors' if args.arm=='zero' else 'final-adapter.safetensors'
        state=load_file(str(args.training_run/checkpoint),device='cpu')
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Non-finite adapter checkpoint')
        adapter.load_state_dict(state,strict=True)
        expected=training['adapter_sha256_before' if args.arm=='zero' else 'adapter_sha256_after']
        if pc.module_sha(adapter)!=expected:raise ValueError('Loaded adapter tensor hash mismatch')
        r['adapter_checkpoint_sha256']=sha(args.training_run/checkpoint);r['adapter_tensor_sha256']=expected
        core,r['foundation']=evidence.measure('load_verified_fp32_base',lambda:load_core(args.weights,'mps'))
        if r['foundation']!=training['foundation']:raise ValueError('Sampling foundation differs from training')
        extend_model(core);adapter.to('mps').eval().requires_grad_(False)
        schedule=make_scheduler('cpu');r['scheduler']={'timesteps':schedule.timesteps.tolist(),'sigmas':schedule.sigmas.tolist()}
        def record(index,timestep,latent,timing):
            r['steps'].append({'step':index+1,'timestep':int(timestep),'min':float(latent.min()),'max':float(latent.max()),**timing})
            if index==0:
                step_seconds=sum(timing[key] for key in ('input_transfer_seconds','negative_positive_seconds','output_transfer_seconds','cpu_solver_seconds'))
                r['first_step_projection']={'measured_step_seconds':step_seconds,'decode_and_artifact_allowance_seconds':60.,'remaining_steps':49}
                evidence.save()
                r['first_step_projection']['estimated_total_seconds']=estimate_total(time.perf_counter()-evidence.started,step_seconds)
            evidence.stage=f'{args.arm} step {index+1}/50 complete';evidence.save()
        latent,r['prefix_checks']=evidence.measure('50_step_tokenwise_unipc',lambda:pc.sample_latents(core,adapter,noise,observation,actions,negative,positive,device='mps',callback=record))
        if len(r['steps'])!=50 or any(s['clamped_prefix_max_abs_difference']!=0 for s in r['steps']):raise RuntimeError('Incomplete or drifting observed prefix')
        if not torch.equal(noise,initial_noise()):raise RuntimeError('Sampling mutated retained caller noise')
        pc.atomic_tensors(args.output/'generated-latents.safetensors',{'generated':latent})
        if pc.module_sha(adapter)!=expected:raise RuntimeError('Sampling changed adapter parameters')
        del adapter,core;gc.collect();torch.mps.empty_cache()
        codec=evidence.measure('load_fp32_vae',lambda:OfficialWanCodec(args.weights/'Wan2.1_VAE.pth','mps',torch.float32))
        torch.mps.empty_cache()
        with cleanup_after_temporal_chunk(codec) as cleanup:
            video=evidence.measure('decode_generated_clip',lambda:codec.decode(latent.unsqueeze(0)))
        r['decoder_allocator_hook']=cleanup
        if list(video.shape)!=[1,3,17,288,512] or not torch.isfinite(video).all():raise RuntimeError('Invalid decoded video')
        pixels=np.rint(((video[0].permute(1,2,3,0).cpu().numpy()+1)/2).clip(0,1)*255).astype(np.uint8)
        del video,codec;gc.collect();torch.mps.empty_cache()
        frames=args.output/'frames';frames.mkdir();images=[]
        for index,array in enumerate(pixels):
            im=Image.fromarray(array);im.save(frames/f'{index:04d}.png');images.append(im)
        images[0].save(args.output/'preview.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
        sheet=Image.new('RGB',(1024,624),(239,238,230));draw=ImageDraw.Draw(sheet)
        for position,index in enumerate((0,5,10,16)):
            x=(position%2)*512;y=(position//2)*312
            label='Observed reconstruction' if index==0 else f'{args.arm} adapter: generated frame {index}'
            draw.text((x+8,y+5),label,fill=(20,20,20));sheet.paste(images[index],(x,y+24))
        sheet.save(args.output/'comparison.png')
        for name in ('diffusion_pytorch_model.safetensors','config.json','Wan2.1_VAE.pth'):
            if sha(args.weights/name)!=FILES[name][1]:raise RuntimeError('External weights/config changed')
        r['output_sha256']={str(path.relative_to(args.output)):sha(path) for path in [*sorted(frames.glob('*.png')),*sorted(args.output.glob('*.safetensors')),args.output/'preview.gif',args.output/'comparison.png']}
        completed=True
    except BaseException as problem:error=problem;raise
    finally:
        if not completed and error is None:error=RuntimeError('Prescribed sampling arm did not complete')
        evidence.close(error)
    print(json.dumps({'status':r['status'],'arm':args.arm,'elapsed_seconds':r['elapsed_seconds'],'peaks':r['peaks_sampled']},indent=2))

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='.*torch.cuda.amp.autocast.*')
        warnings.filterwarnings('ignore',message="User provided device_type of 'cuda'.*")
        main()
