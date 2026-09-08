# SPDX-License-Identifier: Apache-2.0
"""Literal upstream core, original FP32 values, CUDA BF16 autocast."""
import gc
import json
from pathlib import Path
import torch
from ..official_cpu.streaming import ShardSource,sha,tensor_sha

HERE=Path(__file__).resolve().parent
PINS={'vendor/model.py':'8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432',
      'vendor/attention.py':'23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc',
      'vendor/fm_solvers_unipc.py':'0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9',
      'vendor/vae2_2.py':'eab5ce4aa1ce03af2978f2a8e8364c419f5fbb8535d265ac86b0b02ddcf0c1f6',
      'config.json':'d1fea36899d00c2501b836c13ad65af56e2f9529ba622e50886d3f5c3e6c02bc'}


def verify_sources():
    for name,digest in PINS.items():
        if sha(HERE/name)!=digest:raise ValueError('Pinned native source changed: '+name)


def meta_model():
    verify_sources()
    from .vendor.model import WanModel,rope_params
    config={k:v for k,v in json.loads((HERE/'config.json').read_text()).items()if not k.startswith('_')}
    with torch.device('meta'):model=WanModel(**config)
    d=model.dim//model.num_heads
    model.freqs=torch.cat([rope_params(1024,d-4*(d//6)),rope_params(1024,2*(d//6)),rope_params(1024,2*(d//6))],dim=1)
    return model.eval().requires_grad_(False)


def load_model(directory,check):
    model=meta_model()
    source=ShardSource(directory,model,json.loads((HERE/'upstream-provenance.json').read_text()),check=check)
    rows={}
    for name,p in list(model.named_parameters()):
        check();value=source.load(name,p.shape)
        loaded=value.to(device='cuda:0',dtype=torch.float32)
        verification=loaded.detach().cpu()
        if tensor_sha(verification)!=source.records[name]['source_sha256']:raise RuntimeError('CUDA copy differs from original FP32')
        del verification
        owner,key=name.rsplit('.',1);setattr(model.get_submodule(owner),key,torch.nn.Parameter(loaded,requires_grad=False))
        rows[name]=dict(source.records[name],cuda_copy_exact=True)
        del value,loaded
    if len(rows)!=825 or any(p.dtype!=torch.float32 or p.device.type!='cuda'for p in model.parameters()):
        raise RuntimeError('Exactly 825 original FP32 CUDA parameters required')
    gc.collect();check()
    return model,{'tensors':rows,'tensor_count':825,'parameter_count':sum(p.numel()for p in model.parameters()),
        'parameter_bytes':sum(p.numel()*p.element_size()for p in model.parameters()),'convert_model_dtype':False,
        'all_shards_verified':True,'cuda_copy_exact':True}


def predict(model,latent,times,context):
    from .vendor import attention
    if not attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError('This declared reference requires FlashAttention 2 only; SDPA/FA3 are not substituted')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=model([latent.to('cuda:0')],times.to('cuda:0'),[context.to('cuda:0')],720)[0]
    torch.cuda.synchronize()
    return output.float().cpu()
