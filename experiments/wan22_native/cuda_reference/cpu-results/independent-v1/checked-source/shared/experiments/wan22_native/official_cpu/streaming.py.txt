# SPDX-License-Identifier: Apache-2.0
"""Original FP32 parameter ownership, one literal module invocation at a time."""
import gc
import hashlib
import json
from pathlib import Path
import struct
import time
import weakref

import torch
from safetensors import safe_open


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def tensor_sha(t):
    if t.device.type!='cpu':raise ValueError('CPU tensor hash only')
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def bounded_file(root,name):
    root=Path(root).resolve();p=root/name
    if Path(name).is_absolute()or '..'in Path(name).parts or not p.resolve().is_relative_to(root)or not p.is_file():
        raise ValueError('File must remain inside declared root')
    return p


def header(path):
    with Path(path).open('rb')as f:
        size=struct.unpack('<Q',f.read(8))[0]
        if not 2<=size<=16*2**20:raise ValueError('Invalid safetensors header size')
        result=json.loads(f.read(size))
    return {k:v for k,v in result.items()if k!='__metadata__'}


class ShardSource:
    """Check every shard before any value read. Return independent FP32 copies."""
    def __init__(self,directory,model,provenance,*,check=None,verify_bytes=True):
        self.root=Path(directory).resolve();self.check=check or (lambda:None)
        index=bounded_file(self.root,'diffusion_pytorch_model.safetensors.index.json')
        config=bounded_file(self.root,'config.json')
        if sha(index)!=provenance['index_sha256']or sha(config)!=provenance['config_sha256']:
            raise ValueError('Pinned original index/config differs')
        self.mapping=json.loads(index.read_text())['weight_map'];expected={n:tuple(v.shape)for n,v in model.named_parameters()}
        if set(expected)!=set(self.mapping):raise ValueError('Exact parameter keys required')
        artifacts={r['file']:r for r in provenance['weights']}
        if set(artifacts)!=set(self.mapping.values()):raise ValueError('Every shard must be pinned')
        seen=set();self.records={};self.verified=verify_bytes
        for name,row in artifacts.items():
            self.check();path=bounded_file(self.root,name)
            if path.stat().st_size!=row['bytes']:raise ValueError('Shard size differs')
            if verify_bytes and sha(path)!=row['sha256']:raise ValueError('Shard hash differs')
            for key,spec in header(path).items():
                if key in seen or self.mapping.get(key)!=name or tuple(spec['shape'])!=expected.get(key)or spec['dtype']!='F32':
                    raise ValueError('Wrong, duplicate or non-FP32 source tensor')
                seen.add(key)
        if seen!=set(expected):raise ValueError('Missing original parameters')
        self.artifacts=artifacts

    def load(self,name,shape):
        if not self.verified:raise RuntimeError('Plan-only headers do not authorize tensor reads')
        self.check();path=bounded_file(self.root,self.mapping[name])
        with safe_open(str(path),framework='pt',device='cpu')as handle:
            source=handle.get_tensor(name)
            if source.dtype!=torch.float32 or tuple(source.shape)!=tuple(shape)or not torch.isfinite(source).all():
                raise ValueError('Expected original finite FP32 values')
            value=source.clone();digest=tensor_sha(source)
            if value.untyped_storage().data_ptr()==source.untyped_storage().data_ptr()or tensor_sha(value)!=digest:
                raise RuntimeError('Loaded value must own an exact independent copy')
            source_ref=weakref.ref(source);del source
        del handle
        if source_ref()is not None:raise RuntimeError('Source tensor mapping owner retained')
        row={'shape':list(shape),'original_dtype':'float32','loaded_dtype':'float32','source_sha256':digest,
             'loaded_sha256':digest,'source_owner_released':True,'shard':self.mapping[name]}
        if name in self.records and self.records[name]!=row:raise RuntimeError('Source tensor changed between passes')
        self.records[name]=row;self.check();return value


def groups(model):
    return ['patch_embedding','time_embedding','time_projection','text_embedding',
            *[f'blocks.{i}'for i in range(len(model.blocks))],'head']


def summary(value):
    if not isinstance(value,torch.Tensor):raise TypeError('Expected module tensor output')
    if value.device.type!='cpu' or value.grad_fn is not None or not torch.isfinite(value).all():
        raise RuntimeError('Finite CPU inference output required')
    f=value.float()
    return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':tensor_sha(value),
            'min':float(f.min()),'max':float(f.max()),'rms':float(f.square().mean().sqrt())}


class StreamedModules:
    def __init__(self,model,source,*,event=None,check=None):
        if any(p.device.type!='meta' or p.dtype!=torch.float32 for p in model.parameters()):
            raise ValueError('Wholly meta original-FP32 model required')
        self.model=model;self.source=source;self.event=event or (lambda row:None);self.check=check or (lambda:None)
        self.order=groups(model);self.names={g:[n for n,_ in model.named_parameters()if n.startswith(g+'.')]for g in self.order}
        if sum(map(len,self.names.values()))!=len(list(model.parameters())):raise ValueError('Parameter grouping incomplete')
        self.handles=[];self.active=None;self.seen=[];self.rows=[];self.peak_live_bytes=0;self.released_owners=0;self.primed_patch=False

    def live(self):
        return [n for n,p in self.model.named_parameters()if p.device.type!='meta']

    def _emit(self,row):
        self.rows.append(row);self.event(dict(row))

    def _before(self,group,module,args):
        self.check()
        if group=='patch_embedding'and self.primed_patch:
            if self.active!=group or set(self.live())!=set(self.names[group]):raise RuntimeError('Primed patch ownership differs')
            self.primed_patch=False
            return
        if self.active is not None or self.live():raise RuntimeError('Previous module storage retained')
        if len(self.seen)>=len(self.order)or group!=self.order[len(self.seen)]:raise RuntimeError('Unexpected native module order')
        self.active=group;started=time.monotonic()
        try:
            for name in self.names[group]:
                parent,key=name.rsplit('.',1);owner=self.model.get_submodule(parent);old=getattr(owner,key)
                value=self.source.load(name,old.shape)
                if value.dtype!=torch.float32 or value.device.type!='cpu':raise RuntimeError('Only original FP32 CPU parameters')
                setattr(owner,key,torch.nn.Parameter(value,requires_grad=False));del old,value
            if set(self.live())!=set(self.names[group]):raise RuntimeError('Live parameter ownership differs')
            size=sum(p.numel()*p.element_size()for p in module.parameters());self.peak_live_bytes=max(self.peak_live_bytes,size)
            self._emit({'event':'loaded','group':group,'live_parameter_names':self.live(),'live_parameter_bytes':size,
                        'seconds':time.monotonic()-started})
        except BaseException:
            self._evict(group);raise

    def _evict(self,group):
        refs=[]
        for name in self.names[group]:
            parent,key=name.rsplit('.',1);owner=self.model.get_submodule(parent);p=getattr(owner,key)
            if p.device.type!='meta':refs.append(weakref.ref(p))
            setattr(owner,key,torch.nn.Parameter(torch.empty(p.shape,device='meta',dtype=torch.float32),requires_grad=False))
            del p
        gc.collect()
        if any(r()is not None for r in refs):raise RuntimeError('Evicted parameter owner remains alive')
        self.released_owners+=len(refs);self.active=None
        if self.live():raise RuntimeError('Parameter storage remains after eviction')

    def _after(self,group,module,args,output):
        # always_call=True reaches this handler when the native forward raises.
        if self.active!=group:return
        try:record=summary(output)if output is not None else None
        finally:self._evict(group)
        if output is not None:self.seen.append(group)
        self._emit({'event':'evicted','group':group,'live_parameter_names':self.live(),'output':record,
                    'released_parameter_owners_total':self.released_owners})
        self.check()

    def __enter__(self):
        # Native root forward reads patch.weight.device before calling the patch.
        # Prime only that group before native code starts; its own hook consumes
        # this one load and its normal post-hook evicts it after the convolution.
        def prime(module,args):
            self._before('patch_embedding',self.model.patch_embedding,())
            self.primed_patch=True
        def root_after(module,args,output):
            if self.active is not None:self._evict(self.active)
            self.primed_patch=False
        try:
            self.handles.append(self.model.register_forward_pre_hook(prime))
            self.handles.append(self.model.register_forward_hook(root_after,always_call=True))
            for group in self.order:
                module=self.model.get_submodule(group)
                self.handles.append(module.register_forward_pre_hook(lambda m,a,g=group:self._before(g,m,a)))
                self.handles.append(module.register_forward_hook(lambda m,a,o,g=group:self._after(g,m,a,o),always_call=True))
        except BaseException:
            for handle in self.handles:handle.remove()
            self.handles=[]
            raise
        return self

    def completed(self):
        if self.seen!=self.order or self.live()or self.active is not None:raise RuntimeError('Incomplete native pass or retained storage')
        return {'groups':self.seen,'peak_live_parameter_bytes':self.peak_live_bytes,
                'released_parameter_owners':self.released_owners,'all_parameters_meta_after_pass':True}

    def __exit__(self,typ,value,tb):
        try:
            if self.active is not None:self._evict(self.active)
        finally:
            for h in self.handles:h.remove()
            self.handles=[]
        if self.live():raise RuntimeError('Dangling native parameter storage on context exit')
