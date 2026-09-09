# SPDX-License-Identifier: Apache-2.0
"""Portable, file-hash-checked inputs; loads one saved noise shard at a time."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import torch
from safetensors import safe_open
from experiments.wan22_native.official_cpu.streaming import tensor_sha
import math_steps

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

class Inputs:
    def __init__(self,root,expected_sha):
        self.root=Path(root).resolve()
        if not isinstance(expected_sha,str) or len(expected_sha)!=64 or sha(self.root/'inputs.json')!=expected_sha:
            raise ValueError('Exact declared input manifest hash required')
        if (self.root/'inputs.json').stat().st_size>2**20:raise ValueError('Input manifest too large')
        self.plan=json.loads((self.root/'inputs.json').read_text())
        p=self.plan
        if p['schema']!='worldline-command-attention-inputs-v1' or p['updates']!=512 or p['shape']!=[1,48,5,44,78]:
            raise ValueError('Exact native 512-update input format required')
        rows=[{k:r[k] for k in ('update','k','sigma','noise_key','noise_sha256','rng_after_sha256')} for r in p['schedule']]
        if p['schedule']!=math_steps.schedule(rows) or p['edges']!=[list(e) for e in math_steps.EDGES] or len(rows)!=512:
            raise ValueError('Declared seven-edge schedule differs')
        for i,row in enumerate(rows):
            if row['noise_key']!=f'noise_{i:04d}' or row['sigma']!=row['k']/1000:
                raise ValueError('Noise key or timestep inconsistent')
        self.windows={arm:self.tensors(f'windows/{arm}.safetensors') for arm in (m+'_'+d for m in math_steps.MOTIONS for d in ('closed','interact'))}
        self.positive=self.tensors('positive.safetensors')['context']
        self.negative=self.tensors('negative.safetensors')['context']
        self.initial=self.tensors('initial-controller.safetensors')
        self.evaluation=self.tensors('evaluation-noises.safetensors')
        if {k:tensor_sha(v) for k,v in self.evaluation.items()}!=p['evaluation_tensor_sha256']:
            raise ValueError('Fixed evaluation noise differs')
        self.current=None;self.values={}
        sample=torch.zeros(p['shape'],dtype=torch.float32)
        for pair in math_steps.EDGES:math_steps.validate_pair(self.windows,pair,sample)

    def tensors(self,name):
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or str(rel)!=name:raise ValueError('Regular relative file required')
        path=self.root/name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):raise ValueError('No file links or escapes')
        row=self.plan['files'][name]
        if not 0<row['bytes']<=64*2**20 or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Input file identity differs')
        with safe_open(path,framework='pt',device='cpu') as f:
            if set(f.keys())!=set(row['tensors']):raise ValueError('Saved tensor names differ')
            values={}
            for k,v in row['tensors'].items():
                view=f.get_slice(k)
                if view.get_shape()!=v['shape'] or view.get_dtype()!=v['dtype']:raise ValueError('Saved tensor dimensions/dtype differ')
                value=f.get_tensor(k)
                if not torch.isfinite(value).all():raise ValueError('Nonfinite input tensor')
                values[k]=value
        return values

    def noise(self,row):
        index=row['update']-1
        if row!=self.plan['schedule'][index]:raise ValueError('Requested draw is not the declared schedule row')
        start=index//16*16
        name=f'draws/draws-{start:04d}-{start+15:04d}.safetensors'
        if self.current!=name:
            self.values={};self.current=None
            self.values=self.tensors(name);self.current=name
        noise=self.values[row['noise_key']]
        if tensor_sha(noise)!=row['noise_sha256'] or tensor_sha(self.values[f'rng_after_{index:04d}'])!=row['rng_after_sha256']:
            raise ValueError('Saved noise or RNG bytes differ')
        return noise

    def validate_all(self):
        for row in self.plan['schedule']:self.noise(row)
        self.current=None;self.values={}
