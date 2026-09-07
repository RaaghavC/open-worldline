# SPDX-License-Identifier: Apache-2.0
"""Exact retained CPU-pair tensors and genuine text; no target/action access."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from .streaming import bounded_file,sha,tensor_sha

PAIR_PINS={'metrics.json':'b8f1b035498278a1f5bc273b0526b7a17e70f09244f49b792952f2bd68df6b57',
 'terminal.json':'bf7c395ff1544bf9195c0ccc949bcaf13e54f57bc10ca9abd17b7f61eb4943a5',
 'inputs.safetensors':'363ee00416161de834ab8213eb50a386abd42ce9cf3f34ff818e9213b9e9d2b5'}
TEXT_PINS={'manifest.json':'03936c8122c5a902062fdf6a555552a45d861630d0ee2012a712a6a3caf5c11c',
 'embeddings.safetensors':'2f00251cd8ffbbd72f8cee232feeac49df8b6645c204dfc07667748cff36c406'}
SHAPES={'initial_latent':(48,5,18,32),'initial_noise':(48,5,18,32),
        'observation':(1,48,1,18,32),'token_times':(1,720)}


def read_exact(path,shapes):
    if Path(path).stat().st_size>8*2**20:raise ValueError('Bounded saved tensor file required')
    with safe_open(str(path),framework='pt',device='cpu')as h:
        if set(h.keys())!=set(shapes):raise ValueError('Unexpected keys before materialization')
        for key,shape in shapes.items():
            if tuple(h.get_slice(key).get_shape())!=tuple(shape):raise ValueError('Unexpected tensor shape')
        values={k:h.get_tensor(k)for k in shapes}
    for k,v in values.items():
        if v.dtype!=(torch.int64 if k=='token_times'else torch.float32)or not torch.isfinite(v).all():
            raise ValueError('Finite declared CPU dtype required')
    return values


def load_inputs(pair_directory,text_directory):
    for root,pins in [(pair_directory,PAIR_PINS),(text_directory,TEXT_PINS)]:
        for name,digest in pins.items():
            if sha(bounded_file(root,name))!=digest:raise ValueError('Exact retained input evidence required: '+name)
    pair=Path(pair_directory);text=Path(text_directory)
    if list(pair.glob('*watchdog-stop*')):raise ValueError('Stopped pair rejected')
    report=json.loads((pair/'metrics.json').read_text());terminal=json.loads((pair/'terminal.json').read_text())
    if report['status']!='passed'or terminal['status']!='complete'or terminal['exit_code']!=0:raise ValueError('Completed pair required')
    values=read_exact(pair/'inputs.safetensors',SHAPES)
    observed=values['observation'][0];x=values['initial_noise'].clone();x[:,:1]=observed
    if not torch.equal(x,values['initial_latent'])or tensor_sha(values['observation'])!=report['observation']['observation_tensor_sha256']:
        raise ValueError('Exact independently encoded prefix required')
    times=values['token_times']
    if not torch.equal(times[:,:144],torch.zeros(1,144,dtype=torch.int64))or not torch.equal(times[:,144:],torch.full((1,576),999,dtype=torch.int64)):
        raise ValueError('Native initial token times required')
    contexts=read_exact(text/'embeddings.safetensors',{'atrium':(25,4096),'native_negative':(126,4096)})
    for key,label in [('atrium','positive'),('native_negative','negative')]:
        if tensor_sha(contexts[key])!=report['text'][label+'_tensor_sha256']:raise ValueError('Exact genuine context required')
    return values,contexts,{'pair_files':PAIR_PINS,'text_files':TEXT_PINS,'observation':report['observation'],
        'text':report['text'],'input_tensor_sha256':{k:tensor_sha(v)for k,v in values.items()},
        'noise_regenerated':False,'actions_read':False,'future_target_read':False}
