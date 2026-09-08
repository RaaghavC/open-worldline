"""Exact original128 training inputs and pinned genuine negative text only."""
from pathlib import Path
import torch
from safetensors import safe_open
from experiments.wan22_native.action_cuda import probe,probe_evidence as evidence
from experiments.wan22_native.official_cpu.inputs import TEXT_PINS
HERE=Path(__file__).resolve().parent
ORIGINAL_PLAN_SHA='94c2df86e6d62de09fc9fa469b2fca7e04fd7f5fe894b4182349fd36e6af833a'


def original_plan():
    path=HERE/'original128-plan.json'
    if evidence.sha(path)!=ORIGINAL_PLAN_SHA:raise ValueError('Exact original128 input plan required')
    return evidence.read_json(path)


def negative_context(directory):
    root=Path(directory)
    for name,digest in TEXT_PINS.items():
        if evidence.sha(evidence.relative_file(root,name))!=digest:raise ValueError('Pinned genuine text differs')
    with safe_open(root/'embeddings.safetensors',framework='pt',device='cpu') as h:
        if set(h.keys())!={'atrium','native_negative'}:raise ValueError('Exact genuine context keys required')
        view=h.get_slice('native_negative')
        if view.get_shape()!=[126,4096] or view.get_dtype()!='F32':raise ValueError('Exact negative context shape/dtype required')
        value=h.get_tensor('native_negative')
    if not torch.isfinite(value).all():raise ValueError('Finite negative context required')
    return value,{'text_files':dict(TEXT_PINS),'text_id':'native_negative','negative_tensor_sha256':probe.tensor_sha(value)}


def original_inputs(directory):
    root=Path(directory);plan=original_plan()
    if evidence.sha(evidence.relative_file(root,'plan.json'))!=ORIGINAL_PLAN_SHA:raise ValueError('Use original128 prepared inputs, never a trained checkpoint')
    rows={}
    dtype={'float32':'F32','uint8':'U8'}
    for name,record in plan['input_identity']['artifacts'].items():
        path=evidence.relative_file(root,name)
        specs={k:(tuple(v['shape']),dtype[v['dtype']]) for k,v in record['tensors'].items()}
        rows[name]=evidence.tensors(path,specs,record['sha256'])
        if probe.record_file(path,rows[name])!=record:raise ValueError('Exact original128 tensor/file bytes required')
    initial=rows['initial-adapter.safetensors']
    draws={k:v for n,row in rows.items() if n.startswith('draws-') for k,v in row.items()}
    return plan,initial,draws,rows['initial-cpu-rng.safetensors']['rng'],rows


def original_records(records):
    wanted=original_plan()['input_identity']['artifacts']
    if any(records.get(k)!=v for k,v in wanted.items()):raise ValueError('Original128 complete draw/initial/context file records changed')
    return ORIGINAL_PLAN_SHA

HELDOUT_MANIFEST_SHA='058e8c34570032f9a76a34ed040d89490fc25aede5bcaa9ffe40559c43111c19'


def heldout_assessment():
    """Read metadata only. Evaluation Gaussian values never enter training."""
    path=HERE/'heldout-manifest.json'
    if evidence.sha(path)!=HELDOUT_MANIFEST_SHA:raise ValueError('Predeclared held-out manifest differs')
    value=evidence.read_json(path)
    if (value['evaluation_only'] is not True or value['training_use_prohibited'] is not True
            or value['count']!=4 or value['seed']!=2026090801 or value['shape']!=[1,48,5,44,78]
            or value['training_plan_sha256']!=ORIGINAL_PLAN_SHA):raise ValueError('Exact four evaluation-only noises required')
    return {'manifest_sha256':HELDOUT_MANIFEST_SHA,'seed':value['seed'],'count':4,'training_uses_noise_values':False,
            'noises':{name:value['files'][name] for name in value['noise_files']},
            'assessment':'Final new versus zero and original128 on all four held-out noises for this seen scene; unchanged original visual and k506 checks retained separately.'}


def original_schedule(schedule):
    if schedule!=original_plan()['schedule']:
        raise ValueError('Every original128 timestep, sigma, window and noise/RNG record must remain exact')
