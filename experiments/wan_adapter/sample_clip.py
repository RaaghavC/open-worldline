# SPDX-License-Identifier: Apache-2.0
"""Generate matched base/adapter action-planned clips from one initial observation.

Only observation/actions are loaded from the capture cache. Future target
latents and renderer truth are never read for generation. The real empty-string
UMT5 context supplies text CFG with identical commands and image conditioning.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from safetensors.torch import load_file,save_file
from training_common import (ActionObservationAdapter,CaptureCache,Evidence,euler_update,
    load_core,load_texts,module_sha,predict,sha,shifted_schedule,sync,to_device)
from codec.helper import OfficialWanCodec


@torch.inference_mode()
def sample_latents(core,adapter,observation,actions,positive,unconditional,noise,*,steps=20,shift=5.,cfg=5.,progress=None):
    schedule=shifted_schedule(steps,shift)
    # The solver accumulates in float32; only denoiser input/output use fp16.
    latent=torch.cat((observation.float(),noise[:,:,1:].float()),dim=2)
    both_observed=observation.repeat(2,1,1,1,1)
    both_actions=actions.repeat(2,1,1)
    for index,(sigma,next_sigma) in enumerate(zip(schedule,schedule[1:])):
        latent=torch.cat((observation.float(),latent[:,:,1:]),dim=2)
        model_input=latent.to(torch.float16).repeat(2,1,1,1,1)
        velocities=predict(core,adapter,model_input,both_observed,both_actions,[unconditional,positive],1000*sigma).float()
        if not torch.isfinite(velocities).all().item():raise RuntimeError('Non-finite denoiser velocity')
        negative,positive_velocity=velocities.chunk(2,dim=0)
        velocity=negative+cfg*(positive_velocity-negative)
        latent=euler_update(latent,velocity,sigma,next_sigma,observation.float())
        if not torch.isfinite(latent).all().item():raise RuntimeError('Non-finite generated latent')
        if not torch.equal(latent[:,:,:1],observation.float()):raise RuntimeError('Initial latent clamp changed')
        if progress:progress(index,sigma,next_sigma)
    return latent


def save_preview(directory,video):
    pixels=np.rint(((video[0].permute(1,2,3,0).cpu().numpy()+1)/2).clip(0,1)*255).astype(np.uint8)
    directory.mkdir()
    frames=[]
    for i,array in enumerate(pixels):
        image=Image.fromarray(array);image.save(directory/f'{i:04d}.png');frames.append(image)
    # A preview loop displays the 17 generated frames; it is not an interactive FPS benchmark.
    frames[0].save(directory/'preview.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    return pixels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--capture-cache',type=Path,required=True)
    p.add_argument('--text-cache',type=Path,required=True)
    p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    p.add_argument('--seed',type=int,default=20260908)
    p.add_argument('--denoise-steps',type=int,default=20)
    p.add_argument('--shift',type=float,default=5.)
    p.add_argument('--cfg',type=float,default=5.)
    p.add_argument('--max-seconds',type=float,default=900)
    a=p.parse_args()
    if not 1<=a.denoise_steps<=20 or not np.isfinite(a.cfg) or not 0<=a.cfg<=10:p.error('Invalid bounded sampling settings')
    schedule=shifted_schedule(a.denoise_steps,a.shift)
    cache=CaptureCache(a.capture_cache)
    window=cache.read('open-0000',include_target=False)
    positive,negative,text_info=load_texts(a.text_cache,'cpu')
    training=json.loads((a.training_run/'metrics.json').read_text())
    if training.get('status')!='passed' or training.get('capture_cache_manifest_sha256')!=cache.manifest_sha256:
        raise ValueError('Completed matching real-data training run required')
    if training.get('text')!=text_info:raise ValueError('Text context differs from training')
    checkpoint=a.training_run/'trained-adapter.safetensors'
    if sha(checkpoint)!=training['adapter_checkpoint_sha256']:raise ValueError('Trained adapter hash mismatch')
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    evidence=Evidence(a.output,a.device,a.max_seconds,'Matched base and trained-adapter planned clips from initial observed RGB only')
    report=evidence.report;error=None
    report.update(window='open-0000',capture_cache_manifest_sha256=cache.manifest_sha256,text=text_info,
        training_report_sha256=sha(a.training_run/'metrics.json'),training_updates=len(training['updates']),
        source_sha256={f:sha(Path(__file__).with_name(f)) for f in ['sample_clip.py','training_common.py','adapter.py','compat.py','PROTOCOL.md']},
        seed=a.seed,sampler={'method':'Euler flow matching','steps':a.denoise_steps,'shift':a.shift,'cfg':a.cfg,'sigmas':schedule,
                            'solver_dtype':'float32','denoiser_dtype':'float16','timestep':'1000*sigma',
                            'velocity':'v_empty + cfg*(v_atrium-v_empty)','update':'x_next=x+(sigma_next-sigma)*velocity',
                            'prefix':'first observed latent clamped before every denoiser and after every Euler update',
                            'cfg_conditioning':'identical observed image and ordered actions on both text passes'},
        generation_inputs=['independently encoded first observed RGB','16 planned actual commands','genuine text context','seeded Gaussian noise'],
        future_target_values_loaded=False,renderer_after_initialization=False,
        claims='Single-layout development comparison only; not scene generalization, causal streaming or persistent memory',clips=[])
    evidence.save()
    try:
        core,foundation=evidence.measure('load_frozen_core',lambda:load_core(a.weights,a.device));report['foundation']=foundation
        if foundation['converted_base_sha256']!=training['foundation']['converted_base_sha256']:raise ValueError('Foundation differs from training')
        window=to_device(window,a.device)
        positive=positive.to(a.device,torch.float16);negative=negative.to(a.device,torch.float16)
        noise=torch.randn((1,16,5,36,64),generator=torch.Generator().manual_seed(a.seed)).to(a.device)
        latents={}
        for label,name,expected in [('base','untrained-adapter.safetensors',training['initial_adapter_sha256']),
                                    ('trained','trained-adapter.safetensors',training['updated_adapter_sha256'])]:
            adapter=ActionObservationAdapter(core.dim)
            adapter.load_state_dict(load_file(str(a.training_run/name),device='cpu'),strict=True)
            if not all(torch.isfinite(p).all().item() for p in adapter.parameters()) or module_sha(adapter)!=expected:
                raise ValueError('Invalid adapter state or hash')
            adapter=adapter.to(a.device)
            start=time.perf_counter()
            def progress(i,sigma,next_sigma):
                evidence.stage=f'{label}-denoise-{i}'
                print(json.dumps({'clip':label,'step':i,'sigma':sigma,'next_sigma':next_sigma}),flush=True)
            latent=evidence.measure(label+'_denoise',lambda:sample_latents(core,adapter,window['observation'],window['actions'],
                positive,negative,noise,steps=a.denoise_steps,shift=a.shift,cfg=a.cfg,progress=progress))
            latents[label]=latent.detach().cpu().contiguous()
            report['clips'].append({'label':label,'denoise_seconds':time.perf_counter()-start,'adapter_sha256':expected})
            del adapter,latent;gc.collect()
            if a.device=='mps':torch.mps.empty_cache()
        save_file(latents,str(a.output/'generated-latents.safetensors'))
        report['generated_latents_sha256']=sha(a.output/'generated-latents.safetensors')
        report['base_sha256_after']=evidence.measure('verify_frozen_core',lambda:module_sha(core))
        if report['base_sha256_after']!=foundation['converted_base_sha256']:raise RuntimeError('Frozen core changed during generation')
        del core,noise,window,positive,negative;gc.collect()
        if a.device=='mps':torch.mps.empty_cache()
        codec=evidence.measure('load_official_vae',lambda:OfficialWanCodec(a.weights/'Wan2.1_VAE.pth',a.device,torch.float16))
        decoded={}
        for label,latent in latents.items():
            video=evidence.measure(label+'_decode',lambda latent=latent:codec.decode(latent))
            if not torch.isfinite(video).all().item():raise RuntimeError('Non-finite decoded generated video')
            decoded[label]=save_preview(a.output/label,video)
            del video;gc.collect()
            if a.device=='mps':torch.mps.empty_cache()
        sheet=Image.new('RGB',(1024,936),(240,239,232));draw=ImageDraw.Draw(sheet)
        for row,index in enumerate([0,8,16]):
            y=row*312;draw.text((8,y+5),f'Frozen Wan + observed clamp, frame {index}',fill=(20,20,20))
            draw.text((520,y+5),f'After {len(training["updates"])} adapter updates, frame {index}',fill=(20,20,20))
            sheet.paste(Image.fromarray(decoded['base'][index]),(0,y+24));sheet.paste(Image.fromarray(decoded['trained'][index]),(512,y+24))
        sheet.save(a.output/'comparison.png')
        report['comparison_sha256']=sha(a.output/'comparison.png')
    except BaseException as e:error=e;raise
    finally:evidence.close(error)

if __name__=='__main__':main()
