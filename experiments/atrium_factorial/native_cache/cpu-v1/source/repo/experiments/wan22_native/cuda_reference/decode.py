# SPDX-License-Identifier: Apache-2.0
"""Separate process FP32 native VAE decode; no training or target access."""
import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from ..official_cpu.streaming import sha,tensor_sha
from .native import HERE,verify_sources
from .vendor.vae2_2 import WanVAE_


def load_codec(path,check):
    verify_sources();info=json.loads((HERE/'codec-source.json').read_text())
    path=Path(path)
    if path.stat().st_size!=info['weight_bytes']or sha(path)!=info['weight_sha256']:raise ValueError('Pinned original VAE file required')
    config=dict(dim=160,dec_dim=256,z_dim=48,dim_mult=[1,2,4,4],num_res_blocks=2,
                attn_scales=[],temperal_downsample=[False,True,True],dropout=0.)
    with torch.device('meta'):model=WanVAE_(**config)
    state=torch.load(path,map_location='cpu',mmap=True,weights_only=True);expected=model.state_dict()
    if set(state)!=set(expected):raise ValueError('Exact native VAE keys required')
    rows={}
    for name,value in state.items():
        check()
        if value.dtype!=torch.float32 or value.shape!=expected[name].shape or not torch.isfinite(value).all():raise ValueError('Native finite FP32 VAE weights required')
        owner,key=name.rsplit('.',1);loaded=value.to('cuda:0')
        verification=loaded.detach().cpu()
        if tensor_sha(verification)!=tensor_sha(value):raise RuntimeError('FP32 VAE CUDA copy differs')
        del verification
        setattr(model.get_submodule(owner),key,torch.nn.Parameter(loaded,requires_grad=False))
        rows[name]={'shape':list(value.shape),'sha256':tensor_sha(value),'cuda_copy_exact':True}
        del loaded
    del value,state,expected;gc.collect()
    model.eval().requires_grad_(False);model.clear_cache()
    mean=torch.tensor(info['normalization']['mean'],dtype=torch.float32,device='cuda:0')
    std=torch.tensor(info['normalization']['std'],dtype=torch.float32,device='cuda:0')
    if mean.shape!=(48,)or std.shape!=(48,)or not (std>0).all():raise ValueError('Exact 48-channel scales required')
    return model,[mean,1./std],{'weight_sha256':info['weight_sha256'],'compute_dtype':'float32',
        'parameters':sum(p.numel()for p in model.parameters()),'tensors':rows,'config':config}


def decode(model,scale,latent):
    if latent.shape!=(48,5,18,32)or latent.dtype!=torch.float32 or not torch.isfinite(latent).all():raise ValueError('Saved finite native latent required')
    model.clear_cache()
    try:
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            video=model.decode(latent.unsqueeze(0).to('cuda:0'),scale).float().clamp(-1,1)
        torch.cuda.synchronize()
        if video.shape!=(1,3,17,288,512)or not torch.isfinite(video).all():raise FloatingPointError('Invalid native decoded clip')
        return video.cpu()
    finally:model.clear_cache()


def images(video,out):
    out=Path(out);frames=out/'frames';frames.mkdir()
    raw=video[0].permute(1,2,3,0).numpy();pixels=np.rint((raw+1)*127.5).clip(0,255).astype(np.uint8)
    sequence=[Image.fromarray(x)for x in pixels]
    for i,im in enumerate(sequence):im.save(frames/f'{i:04d}.png')
    sequence[0].save(out/'preview.gif',save_all=True,append_images=sequence[1:],duration=125,loop=0)
    canvas=Image.new('RGB',(1024,5*316),(245,242,234));draw=ImageDraw.Draw(canvas)
    for j,i in enumerate([0,1,2,4,6,8,10,12,14,16]):
        x=j%2*512;y=j//2*316;draw.text((x+8,y+6),f'{"Conditioned reconstruction"if i==0 else"Generated future"} {i}',fill=(20,20,20));canvas.paste(sequence[i],(x,y+28))
    canvas.save(out/'comparison.png')
    return {'pixel_rule':'rint((clamped FP32 RGB + 1) * 127.5), uint8','frames':17,
        'conditioned_initial_frames':1,'new_future_frames':16,'preview_playback_fps':8,
        'quality_assessment':'Not assessed automatically; all frames retained for review'}
