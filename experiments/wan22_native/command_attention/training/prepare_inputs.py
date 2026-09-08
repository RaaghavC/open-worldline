# SPDX-License-Identifier: Apache-2.0
"""Make a fresh portable packet from previously verified native inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def record(p):
    with safe_open(p,framework='pt',device='cpu') as handle:
        tensors={n:dict(shape=handle.get_slice(n).get_shape(),dtype=handle.get_slice(n).get_dtype()) for n in handle.keys()}
    return dict(sha256=digest(p),bytes=p.stat().st_size,tensors=tensors)

def prepare(old,out,repository=None,controller_source=None,prior_training_source=None):
    repository=Path(repository or ROOT/'outputs/open-worldline').resolve()
    controller_source=Path(controller_source or HERE.parent/'command-attention-controller-v1').resolve()
    prior_training_source=Path(prior_training_source or HERE.parent/'factorial-intermediate-training-v1').resolve()
    sys.path[:0]=[str(repository),str(controller_source),str(prior_training_source)]
    import math_steps
    from controller import CommandAttentionController,DEFAULT_PARAMETER_COUNT
    import training_inputs as original
    from experiments.wan22_native.official_cpu.streaming import tensor_sha
    if Path(original.__file__).resolve()!=prior_training_source/'training_inputs.py':
        raise ValueError('Original training reader is shadowed by another module')
    old=Path(old).resolve();out=Path(out).absolute()
    if out.exists():raise ValueError('New output directory required')
    prior,data=original.read_prepared(old)
    out.mkdir(parents=True)
    rows=[]
    for source in prior['schedule']:
        rows.append({k:source[k] for k in ('update','k','sigma','noise_key','noise_sha256','rng_after_sha256')})
    files={}
    def copy(src,relative):
        p=out/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p);files[relative]=record(p)
    for arm in data['windows']:
        copy(old/'cache/result'/f'{arm}.safetensors',f'windows/{arm}.safetensors')
    for name in ('positive.safetensors','negative.safetensors'):
        copy(old/'original'/name,name)
    for i in range(0,128,16):
        name=f'draws-{i:04d}-{i+15:04d}.safetensors';copy(old/'original'/name,'draws/'+name)
    g=torch.Generator(device='cpu');g.set_state(data['draw_rng'](128))
    shape=tuple(data['windows']['stationary_closed']['target'].shape)
    for start in range(128,512,16):
        values={}
        for i in range(start,start+16):
            k=int(torch.randint(50,951,(1,),generator=g).item())
            noise=torch.randn(shape,generator=g,dtype=torch.float32,device='cpu')
            state=g.get_state().clone();key=f'noise_{i:04d}'
            values[key]=noise;values[f'rng_after_{i:04d}']=state
            rows.append(dict(update=i+1,k=k,sigma=k/1000,noise_key=key,noise_sha256=tensor_sha(noise),rng_after_sha256=tensor_sha(state)))
        name=f'draws/draws-{start:04d}-{start+15:04d}.safetensors';save_file(values,str(out/name));files[name]=record(out/name)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026090803);control=CommandAttentionController()
        assert sum(p.numel() for p in control.parameters())==DEFAULT_PARAMETER_COUNT
        assert all(torch.count_nonzero(p.b.weight)==0 for p in control.projections.values())
        name='initial-controller.safetensors';save_file(control.state_dict(),str(out/name));files[name]=record(out/name)
    eval_generator=torch.Generator(device='cpu').manual_seed(2026090804)
    evaluation={f'noise_{i:02d}':torch.randn(shape,generator=eval_generator,dtype=torch.float32) for i in range(4)}
    name='evaluation-noises.safetensors';save_file(evaluation,str(out/name));files[name]=record(out/name)
    evaluation_hash={k:tensor_sha(v) for k,v in evaluation.items()}
    plan=dict(schema='worldline-command-attention-inputs-v1',status='prepared',shape=list(shape),updates=512,
              initialization_seed=2026090803,controller_parameters=DEFAULT_PARAMETER_COUNT,evaluation_seed=2026090804,
              evaluation_tensor_sha256=evaluation_hash,files=files,schedule=math_steps.schedule(rows),edges=[list(e) for e in math_steps.EDGES],
              original_128_plan_sha256=digest(old/'plan.json'),original_128_draws='Eight exact original shards; first 128 noise/time/RNG rows retained',
              continuation='Private CPU generator resumes original rng_after_0127; integer k in [50,950], then full FP32 Gaussian noise',
              model_execution=False,training=False)
    (out/'inputs.json').write_text(json.dumps(plan,indent=2)+'\n')
    return dict(status='prepared',files=len(files),bytes=sum(v['bytes'] for v in files.values()),inputs_sha256=digest(out/'inputs.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for n in ('repository','controller-source','prior-training-source'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    print(json.dumps(prepare(a.original,a.output,a.repository,a.controller_source,a.prior_training_source)))
