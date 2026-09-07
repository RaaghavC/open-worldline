"""Measure the official DIAMOND CSGO checkpoint; this is not Worldline training."""
from pathlib import Path
import argparse, hashlib, json, os, platform, statistics, subprocess, sys, time
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('WANDB_DISABLED','true')
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--assets',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--frames',type=int,default=12)
p.add_argument('--spawn',type=int,default=0)
p.add_argument('--device',choices=['mps','cpu','cuda'],default='mps')
p.add_argument('--quality',choices=['fast','higher_quality'],default='fast')
p.add_argument('--arms',nargs='+',choices=['idle','forward','left','right','replay'],default=['idle','forward','left'])
args=p.parse_args()
if not 1<=args.frames<=200: p.error('frames must be1..200')
args.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(args.source.resolve()/'src'))
from models.diffusion import Denoiser,DenoiserConfig,InnerModelConfig,DiffusionSampler,DiffusionSamplerConfig

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def synchronize():
    if args.device=='mps':torch.mps.synchronize()
    elif args.device=='cuda':torch.cuda.synchronize()

def make_model(raw):
    raw={k:v for k,v in raw.items() if k!='_target_'}
    inner={k:v for k,v in raw.pop('inner_model').items() if k!='_target_'}
    inner['num_actions']=51
    return Denoiser(DenoiserConfig(inner_model=InnerModelConfig(**inner),**raw))

def make_sampler(model,raw):
    raw={k:v for k,v in raw.items() if k!='_target_'}
    if isinstance(raw.get('s_tmax'),str) and raw['s_tmax'].startswith('${'):raw['s_tmax']=float('inf')
    for key in ('sigma_min','sigma_max','s_churn','s_tmin','s_tmax','s_noise','s_cond'):
        if key in raw:raw[key]=float(raw[key])
    return DiffusionSampler(model,DiffusionSamplerConfig(**raw))

def array(tensor):
    return tensor.detach().clamp(-1,1).add(1).mul(127.5).round().byte().cpu().numpy()[0].transpose(1,2,0)

def action(name):
    value=torch.zeros((1,51),device=args.device,dtype=torch.long)
    value[0,24]=1 # no horizontal mouse displacement
    value[0,43]=1 # no vertical mouse displacement
    if name=='forward':value[0,0]=1
    elif name in ('left','right'):
        value[0,24]=0
        value[0,18 if name=='left' else 30]=1 # -60 or +60 author mouse bin
    return value

torch.set_num_threads(4)
load_start=time.perf_counter()
config=yaml.safe_load((args.assets/'csgo/config/agent/csgo.yaml').read_text())
sampling=yaml.safe_load((args.source/f'config/world_model_env/{args.quality}.yaml').read_text())
models=[make_model(config[name]) for name in ('denoiser','upsampler')]
checkpoint=args.assets/'csgo/model/csgo.pt'
checkpoint_sha=file_hash(checkpoint)
if checkpoint_sha!='9a56a599cec69863717001660871418af1ac3598762167a5cbda73076951bcb6':
    raise ValueError('This runner expects the verified official DIAMOND checkpoint')
weights=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
for model,name in zip(models,('denoiser','upsampler')):
    sd={k[len(name)+1:]:v for k,v in weights.items() if k.startswith(name+'.')}
    model.load_state_dict(sd,strict=True)
    model.eval().to(args.device)
del sd,weights
sampler=make_sampler(models[0],sampling['diffusion_sampler_next_obs'])
upsampler=make_sampler(models[1],sampling['diffusion_sampler_upsampling'])
synchronize()
print(json.dumps({'event':'loaded','seconds':time.perf_counter()-load_start,'parameters':[sum(p.numel() for p in m.parameters()) for m in models]}),flush=True)
spawn=args.assets/f'csgo/spawn/{args.spawn}'
low=np.load(spawn/'low_res.npy',allow_pickle=False)
full=np.load(spawn/'full_res.npy',allow_pickle=False)
act=np.load(spawn/'act.npy',allow_pickle=False)
future=np.load(spawn/'next_act.npy',allow_pickle=False)
report={'baseline':'DIAMOND CSGO external author checkpoint','device':args.device,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'quality':args.quality,'frames_per_arm':args.frames,'spawn':args.spawn,'checkpoint_sha256':checkpoint_sha,'seed':20260907,'input_sha256':{name:file_hash(spawn/name) for name in ('act.npy','full_res.npy','low_res.npy','next_act.npy')},'sampling_config':sampling,'arms':{},'limitations':['No future ground-truth frames in the released spawn bundle; no prediction accuracy or FVD is measured.','Pixel differences show input sensitivity, not correct action following.','Output is150x280, not720p; saved animation playback timing is not neural throughput.','External checkpoint is not an original Worldline model or redistributed here.']}
outputs={}
with torch.inference_mode():
    for name in args.arms:
        torch.manual_seed(20260907)
        if args.device=='mps':torch.mps.manual_seed(20260907)
        obs=torch.from_numpy(low).to(args.device).float().div(255).mul(2).sub(1).unsqueeze(0)
        history=torch.from_numpy(full).to(args.device).float().div(255).mul(2).sub(1).unsqueeze(0)
        actions=torch.from_numpy(act).to(args.device).long().unsqueeze(0)
        frames=[array(history[:,-1])]
        timings=[]
        for index in range(args.frames):
            if name=='replay':
                control=actions[:,-1].clone() if index==0 else torch.from_numpy(future[index-1]).to(args.device).long().unsqueeze(0)
            else:control=action(name)
            actions[:,-1]=control
            synchronize();started=time.perf_counter()
            next_obs,_=sampler.sample(obs,actions)
            target=F.interpolate(next_obs,scale_factor=5,mode='bicubic').unsqueeze(1)
            next_full,_=upsampler.sample(torch.cat([history[:,-1:],target],dim=1),None)
            synchronize();timings.append((time.perf_counter()-started)*1000)
            obs=torch.cat([obs[:,1:],next_obs.unsqueeze(1)],dim=1)
            actions=actions.roll(-1,dims=1)
            history=torch.cat([history[:,1:],next_full.unsqueeze(1)],dim=1)
            frames.append(array(next_full))
            print(json.dumps({'event':'frame','arm':name,'index':index+1,'ms':round(timings[-1],2)}),flush=True)
        packed=np.stack(frames)
        outputs[name]=packed
        np.save(args.output/f'{name}.npy',packed,allow_pickle=False)
        Image.fromarray(packed[-1]).save(args.output/f'{name}-final.png')
        imgs=[Image.fromarray(frame) for frame in packed]
        # Use measured mean generation time for preview pacing; repeated GIF frames may be coalesced.
        imgs[0].save(args.output/f'{name}.gif',save_all=True,append_images=imgs[1:],duration=max(20,int(statistics.mean(timings))),loop=0)
        report['arms'][name]={'inference_ms':timings,'median_ms':statistics.median(timings),'p95_ms':float(np.percentile(timings,95)),'mean_model_inference_fps':1000/statistics.mean(timings),'timing_scope':'Denoiser plus neural upsampler, synchronized; includes first cold call, excludes history updates, image copies, storage and display','frames_sha256':hashlib.sha256(packed.tobytes()).hexdigest()}
if 'idle' in outputs:
    report['paired_action_mae_0_to_255']={name:float(np.abs(value[1:].astype(float)-outputs['idle'][1:].astype(float)).mean()) for name,value in outputs.items() if name!='idle'}
if args.device=='mps':
    report['final_mps_current_allocated_bytes']=torch.mps.current_allocated_memory()
    report['final_mps_driver_allocated_bytes']=torch.mps.driver_allocated_memory()
report['upstream_commit']=subprocess.check_output(['git','-C',str(args.source.resolve()),'rev-parse','HEAD'],text=True).strip()
(args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'event':'finished','output':str(args.output)}),flush=True)
