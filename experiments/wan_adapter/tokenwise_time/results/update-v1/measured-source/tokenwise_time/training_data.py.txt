# SPDX-License-Identifier: Apache-2.0
"""Read one verified original development window for training, not inference."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from fetch_weights import FILES,sha
from native_control.clamp_cache import MANIFEST_SHA256


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def read_training_window(directory,window):
    root=Path(directory).resolve();path=root/'manifest.json'
    if sha(path)!=MANIFEST_SHA256:raise ValueError('Require the fixed original Atrium development cache')
    m=json.loads(path.read_text())
    if m.get('schema')!='worldline-wan-atrium-cache-v1' or m.get('status')!='passed':
        raise ValueError('Completed original cache required')
    if m['protocol_sha256']!=sha(Path(__file__).parent.parent/'PROTOCOL.md') or m['vae_weights_sha256']!=FILES['Wan2.1_VAE.pth'][1]:
        raise ValueError('Different protocol or observed-image codec')
    checks=m.get('causal_checks',[])
    if len(checks)!=11 or not all(x.get('passed') is True for x in checks):
        raise ValueError('All 11 independent observation checks must pass')
    rows=[r for r in m['windows'] if r['id']==window]
    if len(rows)!=1:raise ValueError('Unknown or duplicate development window')
    row=rows[0];relative=Path(row['file']);file=(root/relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not file.is_relative_to(root) or sha(file)!=row['sha256']:
        raise ValueError('Cache path or file hash mismatch')
    shapes={'target':(1,16,5,36,64),'observation':(1,16,1,36,64),'actions':(1,16,6)}
    values={}
    with safe_open(file,framework='pt',device='cpu') as f:
        if set(f.keys())!=set(shapes):raise ValueError('Unexpected cache tensor keys')
        for key,shape in shapes.items():
            value=f.get_tensor(key);meta=row['tensors'][key]
            if value.shape!=shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Invalid training tensor: '+key)
            if meta['shape']!=list(shape) or meta['dtype']!=str(value.dtype) or meta['sha256']!=tensor_sha(value):
                raise ValueError('Training tensor hash mismatch: '+key)
            values[key]=value
    return values,{'manifest_sha256':MANIFEST_SHA256,'window':window,'file_sha256':row['sha256'],
        'tensors':row['tensors'],'source':row['source'],'causal_checks_passed':11,
        'observation':'Independently encoded first RGB image, not selected from full target encoding',
        'target_use':'Training corruption and future-only loss; never adapter observation conditioning',
        'data_scope':'One original layout, development mechanics only'}
