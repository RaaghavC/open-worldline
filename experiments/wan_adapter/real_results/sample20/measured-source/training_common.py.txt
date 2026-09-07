# SPDX-License-Identifier: Apache-2.0
"""Shared validation, frozen-core loading and evidence for the Atrium pilot."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import psutil
import torch
from safetensors import safe_open
from adapter import ActionObservationAdapter
from compat import install
from fetch_weights import FILES, REVISION, sha
from vendor.wan21.model import Wan21Model
from text_cache.cache import load_context

GIB = 1024**3
PROTOCOL = 'worldline-wan-atrium-17-v1'
ORDER = [f'{arm}-{start:04d}' for start in (0,8,32,49) for arm in ('closed','open')]


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def module_sha(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def sync(device):
    if torch.device(device).type=='mps':torch.mps.synchronize()


class Evidence:
    def __init__(self, output, device, seconds, purpose):
        self.path=Path(output)
        if self.path.exists() or self.path.is_symlink():
            raise ValueError('Output directory must be new; evidence is not resumed or overwritten')
        self.path.mkdir(parents=True)
        self.device=torch.device(device)
        if not 0 < seconds <= 900:raise ValueError('Time cap must be in (0,900] seconds')
        if self.device.type=='mps':
            if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
            torch.mps.set_per_process_memory_fraction(min(1.,18*GIB/torch.mps.recommended_max_memory()))
        self.started=time.perf_counter();self.stage='validation';self.stop=threading.Event()
        self.peak={'rss_bytes':0,'mps_active_bytes':0,'mps_driver_bytes':0}
        self.report={'status':'running','purpose':purpose,'protocol':PROTOCOL,'split':'single-layout development',
            'independent_layouts':1,'causal_streaming':False,'persistent_memory':False,'torch':torch.__version__,
            'platform':platform.platform(),'device':str(self.device),'max_seconds':seconds,'max_memory_gib':18,
            'minimum_available_gib':2,'automatic_mps_cpu_fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0'),
            'core_dtype':'float16','adapter_dtype':'float32','timings':[]}
        self.save()
        def monitor():
            proc=psutil.Process()
            with (self.path/'memory.jsonl').open('x') as log:
                while not self.stop.is_set():
                    row={'seconds':time.perf_counter()-self.started,'stage':self.stage,
                         'rss_bytes':proc.memory_info().rss,'available_system_bytes':psutil.virtual_memory().available}
                    if self.device.type=='mps':
                        row.update(mps_active_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory())
                    for key in self.peak:self.peak[key]=max(self.peak[key],row.get(key,0))
                    log.write(json.dumps(row)+'\n');log.flush()
                    reason=None
                    if row['seconds']>seconds:reason='time-cap'
                    elif max(row['rss_bytes'],row.get('mps_driver_bytes',0))>18*GIB:reason='memory-cap'
                    elif row['available_system_bytes']<2*GIB:reason='available-system-memory-below-2-GiB'
                    if reason:
                        write_json(self.path/'watchdog-stop.json',{'status':'stopped','reason':reason,'last_sample':row})
                        os._exit(124)
                    self.stop.wait(.5)
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()

    def save(self):
        self.report['peaks_sampled']=dict(self.peak)
        write_json(self.path/'metrics.json',self.report)

    def measure(self,label,func):
        self.stage=label;sync(self.device);start=time.perf_counter();result=func();sync(self.device)
        self.report['timings'].append({'stage':label,'seconds':time.perf_counter()-start});self.save()
        return result

    def close(self,error=None):
        self.stop.set();self.thread.join(timeout=2)
        self.report['status']='failed' if error else 'passed'
        if error:self.report.update(error_type=type(error).__name__,error=str(error))
        self.report['elapsed_seconds']=time.perf_counter()-self.started;self.save()


def load_core(weights,device):
    root=Path(weights)
    for name in ['diffusion_pytorch_model.safetensors','config.json']:
        size,digest=FILES[name]
        if (root/name).stat().st_size!=size or sha(root/name)!=digest:
            raise ValueError(f'Pinned Wan integrity failure: {name}')
    config={k:v for k,v in json.loads((root/'config.json').read_text()).items() if not k.startswith('_')}
    install()
    with torch.device('meta'):core=Wan21Model(**config)
    values={}
    with safe_open(root/'diffusion_pytorch_model.safetensors',framework='pt',device='cpu') as f:
        for key in f.keys():values[key]=f.get_tensor(key).to(torch.float16)
    loaded=core.load_state_dict(values,strict=True,assign=True);del values
    core.requires_grad_(False).eval()
    before=module_sha(core)
    core=core.to(device);core.gradient_checkpointing=True
    details={'repository':'Wan-AI/Wan2.1-T2V-1.3B','revision':REVISION,
             'source_sha256':FILES['diffusion_pytorch_model.safetensors'][1],'config_sha256':FILES['config.json'][1],
             'converted_base_sha256':before,'loaded_keys':len(core.state_dict()),'missing_keys':loaded.missing_keys,
             'unexpected_keys':loaded.unexpected_keys,'parameters':sum(p.numel() for p in core.parameters())}
    gc.collect()
    return core,details


class CaptureCache:
    def __init__(self,directory):
        self.root=Path(directory).resolve();path=self.root/'manifest.json'
        self.manifest=json.loads(path.read_text());m=self.manifest
        if m.get('schema')!='worldline-wan-atrium-cache-v1' or m.get('status')!='passed' or m.get('protocol')!=PROTOCOL:
            raise ValueError('Completed exact-protocol VAE cache required')
        if m.get('protocol_sha256')!=sha(Path(__file__).with_name('PROTOCOL.md')):
            raise ValueError('Cache protocol hash does not match current declared protocol')
        if m.get('vae_weights_sha256')!=FILES['Wan2.1_VAE.pth'][1]:raise ValueError('Cache used different VAE')
        if len(m.get('causal_checks',[]))!=11 or not all(x.get('passed') is True for x in m['causal_checks']):
            raise ValueError('All 11 declared VAE causality/cache checks must pass')
        if [r['id'] for r in m.get('windows',[])]!=ORDER:raise ValueError('Cache window order differs from fixed selection')
        self.entries={r['id']:r for r in m['windows']};self.manifest_sha256=sha(path)

    def read(self,window_id,*,include_target):
        r=self.entries[window_id];relative=Path(r['file']);path=(self.root/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(self.root):raise ValueError('Cache file escapes cache directory')
        if sha(path)!=r['sha256']:raise ValueError('Cache file hash mismatch')
        keys=['observation','actions']+(['target'] if include_target else [])
        out={};shapes={'target':(1,16,5,36,64),'observation':(1,16,1,36,64),'actions':(1,16,6)}
        with safe_open(path,framework='pt',device='cpu') as f:
            if set(f.keys())!=set(shapes):raise ValueError('Unexpected cache tensor keys')
            for key in keys:
                value=f.get_tensor(key)
                metadata=r['tensors'][key]
                if value.shape!=shapes[key] or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid {key} tensor')
                if list(value.shape)!=metadata['shape'] or str(value.dtype)!=metadata['dtype'] or tensor_sha(value)!=metadata['sha256']:
                    raise ValueError(f'Tensor provenance mismatch: {key}')
                out[key]=value
        return out


def load_texts(directory,device):
    prompts=json.loads((Path(__file__).parent/'text_cache/prompts.json').read_text())['prompts']
    expected={entry['id']:entry['text'] for entry in prompts}
    positive=load_context(directory,'atrium',expected_text=expected['atrium'],device=device,dtype=torch.float16)
    negative=load_context(directory,'unconditional',expected_text='',device=device,dtype=torch.float16)
    return positive,negative,{'manifest_sha256':sha(Path(directory)/'manifest.json'),
        'embeddings_file_sha256':sha(Path(directory)/'embeddings.safetensors'),
        'positive_id':'atrium','negative_id':'unconditional','token_lengths':[positive.shape[0],negative.shape[0]],
        'padding':'Pass unpadded context; frozen Wan pads to512 before projection, context_lens=None'}


def to_device(window,device):
    return {k:v.to(device=device,dtype=torch.float32 if k=='actions' else torch.float16) for k,v in window.items()}


def flow_training_pair(clean,noise,observation,sigma):
    """Noised future targets; first observed latent remains fixed and clean."""
    noisy=(1-sigma)*clean+sigma*noise
    noisy=torch.cat((observation,noisy[:,:,1:]),dim=2)
    return noisy,noise-clean


def future_mse(predicted,target):
    return torch.nn.functional.mse_loss(predicted[:,:,1:].float(),target[:,:,1:].float())


@torch.inference_mode()
def predict(core,adapter,latent,observation,actions,context,time_value):
    adapter.attach(core,actions,observation[:,:,0],(5,18,32))
    try:
        return core(list(latent.unbind(0)),torch.full((latent.shape[0],),time_value,device=latent.device),context,2880)
    finally:adapter.detach()


def shifted_schedule(steps=20,shift=5.):
    if steps<1 or not math.isfinite(shift) or shift<=0:raise ValueError('Invalid Euler schedule')
    sigma=torch.linspace(1.,0.,steps+1,dtype=torch.float64)
    return (shift*sigma/(1+(shift-1)*sigma)).tolist()


def euler_update(latent,velocity,sigma,next_sigma,observation):
    updated=latent+(next_sigma-sigma)*velocity
    return torch.cat((observation,updated[:,:,1:]),dim=2)
