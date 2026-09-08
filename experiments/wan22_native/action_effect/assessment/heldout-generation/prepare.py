"""Generate four canonical evaluation noises once; never run a model."""
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[2]
TRAIN=ROOT/'work/wan22-action-cuda-fixed128-recovered-final-v1/recovered/action-results/fixed128-spatial-v1'
OUT=ROOT/'work/wan22-action-effect-heldout-v1'
PLAN_SHA='94c2df86e6d62de09fc9fa469b2fca7e04fd7f5fe894b4182349fd36e6af833a'
AUDIT=ROOT/'work/wan22-action-cuda-fixed128-actual-audit-v1/report.json'
AUDIT_SHA='e987d7b31ce754977ce8ee547aa9e2cf4a2b31afc136a4f57b893de6950caa7b'
SEED=2026090801
SHAPE=(1,48,5,44,78)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ahash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def array_record(x):return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':ahash(x)}
def file_record(path):return {'bytes':path.stat().st_size,'sha256':sha(path)}

def main():
    if OUT.exists():raise FileExistsError('Canonical output already exists; generation is single-use')
    if sha(TRAIN/'plan.json')!=PLAN_SHA or sha(AUDIT)!=AUDIT_SHA:raise ValueError('Original audited training identity differs')
    plan=json.loads((TRAIN/'plan.json').read_text());audit=json.loads(AUDIT.read_text())
    if audit['status']!='passed' or audit['completed_updates']!=128:raise ValueError('Completed original128 audit required')
    shards={}
    for first in range(0,128,16):
        name=f'draws-{first:04d}-{first+15:04d}.safetensors';path=TRAIN/name;expected=plan['input_identity']['artifacts'][name]
        if path.is_symlink() or sha(path)!=expected['sha256']:raise ValueError('Original draw shard differs')
        shards[name]={**file_record(path),'noise_tensors':{k:v for k,v in expected['tensors'].items() if k.startswith('noise_')}}
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only preparation requires CUDA uninitialized')
    global_before=torch.random.get_rng_state().clone()
    generator=torch.Generator(device='cpu');generator.manual_seed(SEED)
    states={'initial':generator.get_state().clone()};noises=[];records={}
    OUT.mkdir()
    (OUT/'prepare.py').write_bytes(Path(__file__).read_bytes())
    write(OUT/'generation-status.json',{'status':'running','seed':SEED,'no_regeneration_allowed':True})
    for i in range(4):
        x=torch.randn(SHAPE,dtype=torch.float32,device='cpu',generator=generator)
        if not torch.isfinite(x).all():raise ValueError('Nonfinite Gaussian output')
        states[f'after_{i:04d}']=generator.get_state().clone()
        path=OUT/f'noise-{i:04d}.safetensors';save_file({'noise':x},str(path))
        noises.append(x.numpy().copy());records[path.name]={**file_record(path),'tensors':{'noise':array_record(noises[-1])}}
        save_file(states,str(OUT/'generator-states.safetensors'))
    if not torch.equal(global_before,torch.random.get_rng_state()):raise RuntimeError('Dedicated generator changed global CPU RNG')
    equal=[];future_equal=[];training_records=[]
    for name,metadata in shards.items():
        with safe_open(TRAIN/name,framework='np') as f:
            if set(f.keys())!=set(plan['input_identity']['artifacts'][name]['tensors']):raise ValueError('Draw shard key set differs')
            for key,expected in metadata['noise_tensors'].items():
                a=f.get_tensor(key)
                if a.dtype!=np.float32 or a.shape!=SHAPE or not np.isfinite(a).all() or ahash(a)!=expected['sha256']:raise ValueError('Original noise bytes differ')
                training_records.append({'shard':name,'key':key,**array_record(a)})
                for i,b in enumerate(noises):
                    if np.array_equal(a,b):equal.append([i,key])
                    if np.array_equal(a[:,:,1:],b[:,:,1:]):future_equal.append([i,key])
                del a
    if len(training_records)!=128 or equal or future_equal:raise ValueError('Evaluation noise overlaps original training noise')
    for i in range(4):
        for j in range(i):
            if np.array_equal(noises[i],noises[j]) or np.array_equal(noises[i][:,:,1:],noises[j][:,:,1:]):raise ValueError('Evaluation noises repeat')
    write(OUT/'training-comparison.json',{'status':'passed','training_plan_sha256':PLAN_SHA,'training_audit_sha256':AUDIT_SHA,
        'shards':shards,'training_noise_records':training_records,'full_tensor_comparisons':512,'future_only_comparisons':512,
        'equal_training_pairs':equal,'equal_future_pairs':future_equal,'heldout_pairwise_checks':6,
        'training_noises_regenerated':0,'comparison':'NumPy exact array equality after all original file/tensor hash checks'})
    write(OUT/'environment.json',{'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
        'byteorder':sys.byteorder,'torch':torch.__version__,'numpy':np.__version__,'safetensors':safetensors.__version__,
        'torch_build_config':torch.__config__.show(),'threads':torch.get_num_threads(),'seed':SEED,'device':'cpu',
        'generator':'dedicated torch.Generator(device=cpu)','draw_order':[0,1,2,3],
        'global_cpu_rng_unchanged':True,'cuda_initialized':torch.cuda.is_initialized(),'model_execution':False})
    states_path=OUT/'generator-states.safetensors'
    records[states_path.name]={**file_record(states_path),'tensors':{k:array_record(v.numpy()) for k,v in states.items()}}
    write(OUT/'generation-status.json',{'status':'complete','seed':SEED,'draws':4,'no_regeneration_allowed':True})
    for name in ('prepare.py','environment.json','training-comparison.json','generation-status.json'):
        records[name]=file_record(OUT/name)
    manifest={'schema':'worldline-action-effect-heldout-noise-v1','status':'prepared','evaluation_only':True,
        'training_use_prohibited':True,'seed':SEED,'shape':list(SHAPE),'dtype':'float32','count':4,
        'noise_files':[f'noise-{i:04d}.safetensors' for i in range(4)],'files':records,
        'training_plan_sha256':PLAN_SHA,'training_audit_sha256':AUDIT_SHA,
        'training_shard_sha256':{k:v['sha256'] for k,v in shards.items()},
        'cross_architecture_rule':'Saved file and tensor hashes are canonical; do not regenerate from seed at evaluation.',
        'generation_completed_once':True,'model_execution':False}
    write(OUT/'manifest.json',manifest)
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'manifest_sha256':sha(OUT/'manifest.json'),
        'files':len(records)+1,'bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),'cuda_initialized':torch.cuda.is_initialized()}))

if __name__=='__main__':main()
