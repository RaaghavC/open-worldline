# SPDX-License-Identifier: Apache-2.0
"""Fixed process/GPU limits, with parent kill escalation."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import psutil
import torch

LIMITS={'seconds':900.,'host_rss_bytes':48*2**30,'cuda_reserved_bytes':60*2**30,
        'minimum_host_available_bytes':8*2**30,'minimum_cuda_available_bytes':8*2**30,
        'minimum_gpu_total_bytes':70*2**30}


def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def validate_hardware(value,expected):
    if not expected or value['name']!=expected:raise ValueError('Exact caller-declared GPU name required')
    if (type(value['total_memory_bytes'])is not int or value['total_memory_bytes']<LIMITS['minimum_gpu_total_bytes']
            or value['bf16_supported']is not True):
        raise ValueError('At least 70 GiB GPU memory and native BF16 support required')
    capability=value['capability']
    if not isinstance(capability,list)or len(capability)!=2 or any(type(x)is not int for x in capability)or capability[0]not in (8,9):
        raise ValueError('Reviewed FlashAttention 2 Ampere/Ada/Hopper hardware required')
    if value['torch'].split('+')[0]!='2.5.1'or value['cuda']!='12.4'or value['flash_attn']!='2.7.4.post1':
        raise ValueError('The declared Torch 2.5.1 / CUDA 12.4 / FlashAttention 2.7.4.post1 environment is required')
    if value['flash_attention_2_available']is not True or value['flash_attention_3_available']is not False:
        raise ValueError('FlashAttention 2 must be available, with no FA3 override')
    return value


def hardware(expected):
    if not torch.cuda.is_available()or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible CUDA GPU required')
    from .vendor import attention
    p=torch.cuda.get_device_properties(0)
    r={'name':p.name,'total_memory_bytes':int(p.total_memory),'capability':list(torch.cuda.get_device_capability(0)),
       'bf16_supported':torch.cuda.is_bf16_supported(including_emulation=False),'torch':torch.__version__,
       'cuda':torch.version.cuda,'flash_attn':importlib.metadata.version('flash_attn'),
       'flash_attention_2_available':attention.FLASH_ATTN_2_AVAILABLE,'flash_attention_3_available':attention.FLASH_ATTN_3_AVAILABLE,
       'cudnn':torch.backends.cudnn.version(),'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
       'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,
       'dependencies':{k:importlib.metadata.version(k)for k in ('diffusers','numpy','safetensors','psutil','einops','Pillow')},
       'environment':{k:os.environ.get(k)for k in ('CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE')}}
    return validate_hardware(r,expected)


def check_sample(row,deadline,now=None):
    now=time.monotonic()if now is None else now
    if not math.isfinite(deadline)or now>=deadline:raise RuntimeError('900-second combined deadline reached')
    for key in ('host_rss_bytes','cuda_reserved_bytes','host_available_bytes','cuda_available_bytes'):
        if type(row[key])is not int or row[key]<0:raise RuntimeError('Invalid memory sample')
    if row['host_rss_bytes']>LIMITS['host_rss_bytes']or row['cuda_reserved_bytes']>LIMITS['cuda_reserved_bytes']:
        raise RuntimeError('Fixed host/GPU memory cap exceeded')
    if row['host_available_bytes']<LIMITS['minimum_host_available_bytes']or row['cuda_available_bytes']<LIMITS['minimum_cuda_available_bytes']:
        raise RuntimeError('Available host/GPU memory fell below 8 GiB')


class Monitor:
    def __init__(self,out,deadline):
        self.out=Path(out);self.deadline=deadline;self.started=time.monotonic();self.stop=threading.Event();self.error=None
        self.thread=threading.Thread(target=self.watch,daemon=True)
    def sample(self):
        free,_=torch.cuda.mem_get_info(0)
        return {'seconds':time.monotonic()-self.started,'host_rss_bytes':psutil.Process().memory_info().rss,
                'host_available_bytes':psutil.virtual_memory().available,'cuda_reserved_bytes':torch.cuda.memory_reserved(0),
                'cuda_allocated_bytes':torch.cuda.memory_allocated(0),'cuda_available_bytes':free}
    def check(self):
        if self.error:raise RuntimeError(self.error)
        row=self.sample();check_sample(row,self.deadline)
    def watch(self):
        with (self.out/'memory.jsonl').open('x')as f:
            while not self.stop.is_set():
                try:
                    row=self.sample();f.write(json.dumps(row)+'\n');f.flush();check_sample(row,self.deadline)
                except BaseException as e:
                    self.error=str(e);atomic(self.out/'watchdog-stop.json',{'status':'stopped','reason':str(e)})
                    os._exit(124)
                self.stop.wait(.25)
    def __enter__(self):self.check();self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=3)


def stop_child(proc):
    if proc.poll()is None:
        proc.terminate()
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)


def supervise(proc,out,deadline):
    out=Path(out);began=time.monotonic();error=None;peak=0;minimum=None
    try:
        with (out/'parent-memory.jsonl').open('x')as f:
            while proc.poll()is None:
                root=psutil.Process()
                try:rss=root.memory_info().rss+sum(p.memory_info().rss for p in root.children(recursive=True)if p.is_running())
                except psutil.NoSuchProcess:continue
                available=psutil.virtual_memory().available
                peak=max(peak,rss);minimum=available if minimum is None else min(minimum,available)
                row={'seconds':time.monotonic()-began,'combined_rss_bytes':rss,'host_available_bytes':available}
                f.write(json.dumps(row)+'\n');f.flush()
                if time.monotonic()>=deadline or rss>LIMITS['host_rss_bytes']or available<LIMITS['minimum_host_available_bytes']:
                    atomic(out/'watchdog-stop.json',{'status':'stopped','sample':row});raise RuntimeError('Parent resource guard stopped child')
                time.sleep(.25)
    except BaseException as e:error=e;raise
    finally:
        stop_child(proc)
        atomic(out/'terminal.json',{'status':'complete'if proc.returncode==0 and error is None else'failed',
            'exit_code':proc.returncode,'error':str(error)if error else None,'elapsed_seconds':time.monotonic()-began,
            'peak_combined_rss_bytes':peak,'minimum_host_available_bytes':minimum})
