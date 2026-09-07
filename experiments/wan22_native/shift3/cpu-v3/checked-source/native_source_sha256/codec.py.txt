# SPDX-License-Identifier: Apache-2.0
"""FP32 execution wrapper for the unchanged official Wan2.2 48-channel VAE.

No CUDA autocast, suppressed errors, inference API or model download. The
underlying neural equations, spatial patch order and temporal caches are native.
"""
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import torch
from .vendor.vae2_2 import WanVAE_

HERE=Path(__file__).resolve().parent
SOURCE_SHA256='eab5ce4aa1ce03af2978f2a8e8364c419f5fbb8535d265ac86b0b02ddcf0c1f6'
WEIGHT_SHA256='20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'
WEIGHT_BYTES=2818839170
CONFIG=dict(dim=160,dec_dim=256,z_dim=48,dim_mult=[1,2,4,4],num_res_blocks=2,
            attn_scales=[],temperal_downsample=[False,True,True],dropout=0.)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def scales(device='cpu'):
    if sha(HERE/'vendor/vae2_2.py')!=SOURCE_SHA256:raise ValueError('Pinned official VAE source changed')
    source=json.loads((HERE/'codec-source.json').read_text())
    if source['vendor_sha256']!=SOURCE_SHA256 or source['weight_sha256']!=WEIGHT_SHA256:raise ValueError('Codec provenance changed')
    mean=torch.tensor(source['normalization']['mean'],dtype=torch.float32,device=device)
    std=torch.tensor(source['normalization']['std'],dtype=torch.float32,device=device)
    if mean.shape!=(48,) or std.shape!=(48,) or not torch.isfinite(mean).all() or not (std>0).all():raise ValueError('Invalid native scales')
    return [mean,1./std]


def make_model(*,small=False,device='meta'):
    config=dict(CONFIG)
    if small:config.update(dim=4,dec_dim=4,num_res_blocks=1)
    with torch.device(device):model=WanVAE_(**config)
    return model


def verify_weight_file(path):
    path=Path(path)
    if path.stat().st_size!=WEIGHT_BYTES or sha(path)!=WEIGHT_SHA256:raise ValueError('Pinned official Wan2.2 VAE file differs')


def inspect_metadata(path):
    """Inspect checkpoint keys/shapes using meta tensors, without weight values."""
    verify_weight_file(path)
    model=make_model()
    state=torch.load(path,map_location='meta',weights_only=True,mmap=True)
    expected=model.state_dict()
    if set(state)!=set(expected):raise ValueError('Official VAE key set differs from architecture')
    rows=[]
    for name,value in state.items():
        if not isinstance(value,torch.Tensor) or value.device.type!='meta' or value.dtype!=torch.float32 or value.shape!=expected[name].shape:
            raise ValueError('Unexpected VAE meta tensor: '+name)
        rows.append({'name':name,'shape':list(value.shape),'dtype':str(value.dtype),'elements':value.numel()})
    loaded=model.load_state_dict(state,strict=True,assign=True)
    return {'weight_sha256':WEIGHT_SHA256,'weight_bytes':WEIGHT_BYTES,'loaded_keys':len(rows),
        'missing_keys':loaded.missing_keys,'unexpected_keys':loaded.unexpected_keys,
        'parameters':sum(p.numel() for p in model.parameters()),'rows':rows,
        'tensor_values_materialized':False,'source_sha256':SOURCE_SHA256,'config':CONFIG}


class Wan22Codec:
    """One codec process; accepted compute/storage dtype is explicitly FP32."""
    def __init__(self,weights,device='cpu',check=None):
        if device not in ('cpu','mps'):raise ValueError('Only reviewed CPU/MPS device paths are available')
        verify_weight_file(weights)
        model=make_model()
        state=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
        expected=model.state_dict()
        if set(state)!=set(expected):raise ValueError('Official codec checkpoint keys differ')
        for name,value in state.items():
            if value.shape!=expected[name].shape or value.dtype!=torch.float32:raise ValueError('Unexpected codec tensor: '+name)
            if check:check()
            if not torch.isfinite(value).all():raise ValueError('Non-finite codec weights: '+name)
        loaded=model.load_state_dict(state,strict=True,assign=True)
        del expected,state;gc.collect()
        model.to(device=device,dtype=torch.float32)
        self._configure(model,device)
        self.provenance={'external_pretrained_model':True,'weight_sha256':WEIGHT_SHA256,'weight_bytes':WEIGHT_BYTES,
            'loaded_keys':len(model.state_dict()),'missing_keys':loaded.missing_keys,'unexpected_keys':loaded.unexpected_keys,
            'parameters':sum(p.numel() for p in model.parameters()),'source_sha256':SOURCE_SHA256,'config':CONFIG,
            'load':'Meta architecture; weights_only=True, mmap=True; strict FP32 state; no full dtype conversion',
            'compute_dtype':'float32','device':device}
        if check:check()

    @classmethod
    def from_model(cls,model):
        """CPU-test injection only; it makes no pretrained-weight claim."""
        obj=cls.__new__(cls)
        obj._configure(model,str(next(model.parameters()).device))
        obj.provenance={'external_pretrained_model':False,'scope':'Injected test model'}
        return obj

    def _configure(self,model,device):
        if model.z_dim!=48 or model.dim_mult!=[1,2,4,4] or model.temperal_downsample!=[False,True,True]:raise ValueError('Expected native 48-channel stride configuration')
        if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Only FP32 codec parameters are reviewed')
        self.model=model.eval().requires_grad_(False);self.device=torch.device(device)
        self.scale=scales(device);self.model.clear_cache()

    def cache_is_clear(self):
        return all(x is None for x in self.model._feat_map+self.model._enc_feat_map) and self.model._conv_idx==[0] and self.model._enc_conv_idx==[0]

    @torch.inference_mode()
    def encode(self,video):
        """RGB [-1,1], [B,3,1+4n,H,W] -> normalized [B,48,1+n,H/16,W/16]."""
        self.model.clear_cache()
        try:
            if video.ndim!=5 or video.shape[0]<1 or video.shape[1]!=3 or video.shape[2]%4!=1 or min(video.shape[3:])<16 or any(x%16 for x in video.shape[3:]):raise ValueError('Expected nonempty [B,3,1+4n,H,W] with H/W multiples of16')
            if video.dtype!=torch.float32 or not torch.isfinite(video).all() or video.min() < -1 or video.max()>1:raise ValueError('RGB must be finite FP32 in [-1,1]')
            with torch.autocast(device_type=self.device.type,enabled=False):
                z=self.model.encode(video.to(self.device),self.scale).float()
            shape=(video.shape[0],48,1+(video.shape[2]-1)//4,video.shape[3]//16,video.shape[4]//16)
            if z.shape!=shape or not torch.isfinite(z).all():raise RuntimeError('Invalid native encoded latent')
            return z
        finally:self.model.clear_cache()

    @torch.inference_mode()
    def decode(self,latent):
        """Normalized [B,48,F,H,W] -> native clamped FP32 RGB reconstruction."""
        self.model.clear_cache()
        try:
            if latent.ndim!=5 or latent.shape[0]<1 or latent.shape[1]!=48 or min(latent.shape[2:])<1:raise ValueError('Expected nonempty [B,48,F,H,W] latent')
            if latent.dtype!=torch.float32 or not torch.isfinite(latent).all():raise ValueError('Latent must be finite FP32')
            with torch.autocast(device_type=self.device.type,enabled=False):
                video=self.model.decode(latent.to(self.device),self.scale).float().clamp(-1,1)
            shape=(latent.shape[0],3,1+4*(latent.shape[2]-1),latent.shape[3]*16,latent.shape[4]*16)
            if video.shape!=shape or not torch.isfinite(video).all():raise RuntimeError('Invalid native decoded video')
            return video
        finally:self.model.clear_cache()


@contextmanager
def cleanup_temporal_chunks(codec,callback=None):
    """Optional allocator cleanup after encoder/decoder chunk return; no math edits."""
    if callback is None:
        callback=torch.mps.empty_cache if codec.device.type=='mps' else lambda:None
    counts={'encoder':0,'decoder':0};handles=[]
    def hook(name):
        def called(module,args,output):
            counts[name]+=1;callback()
        return called
    try:
        handles.append(codec.model.encoder.register_forward_hook(hook('encoder')))
        handles.append(codec.model.decoder.register_forward_hook(hook('decoder')))
        yield counts
    finally:
        for handle in handles:handle.remove()
        codec.model.clear_cache()
