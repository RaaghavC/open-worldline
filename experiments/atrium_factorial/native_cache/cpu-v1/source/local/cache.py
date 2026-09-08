# SPDX-License-Identifier: Apache-2.0
"""Eight sequential native encode calls with retained prefix evidence."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from experiments.atrium_factorial import reader
from experiments.wan22_native.action_cuda.data import prefix_check
from experiments.wan22_native.official_cpu.streaming import sha,tensor_sha
from experiments.wan22_native.spatial_reference.guards import atomic
import packet


def video(window):
    """Only reorder the public reader's owned FP32 axes; no pixel arithmetic."""
    a,b,c=window.initial_rgb,window.future_rgb_targets,window.commands
    if (any(not isinstance(v,np.ndarray) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in (a,b,c))
            or a.ndim!=3 or a.shape[0]!=3 or b.shape!=(16,*a.shape) or c.shape!=(16,6)
            or min(a.min(),b.min()) < -1 or max(a.max(),b.max())>1):
        raise ValueError('Published finite FP32 RGB/action arrays required')
    combined=torch.cat((torch.from_numpy(a.copy())[None],torch.from_numpy(b.copy())),dim=0)
    return combined.permute(1,0,2,3).contiguous()[None],torch.from_numpy(c.copy())[None]


def _latent(value,frames):
    if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.device.type!='cpu'
            or value.requires_grad or tuple(value.shape)!=(1,48,frames,44,78)
            or not torch.isfinite(value).all()):raise ValueError('Finite detached native FP32 latent required')


def save(path,values):
    path=Path(path)
    if path.exists():raise ValueError('Refuse overwriting encoded evidence')
    save_file({k:v.detach().cpu().contiguous() for k,v in values.items()},str(path))
    return dict(bytes=path.stat().st_size,sha256=sha(path),
                tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=tensor_sha(v.detach().cpu())) for k,v in values.items()})


def encode_sequence(output,read_arm,encoder,*,identity,check=lambda:None):
    """read_arm is the published reader; tests supply small explicit arrays."""
    out=Path(output);out.mkdir(exist_ok=False)
    report=dict(schema=packet.SCHEMA,status='running',identity=identity,arms=list(reader.ARMS),
        encoder_attempts=0,encoder_completed=0,one_frame_calls=0,seventeen_frame_calls=0,
        files={},checks=[],target_prefix_replaced=False,commands_provided_to_codec=False,
        core_model_loaded=False,quality_assessed=False)
    atomic(out/'completion.json',report)
    def emit():atomic(out/'completion.json',report)
    def encode(name,inputs,values):
        check();before=tensor_sha(inputs);report['encoder_attempts']+=1;emit();start=time.monotonic()
        value=encoder(inputs)
        report['encoder_completed']+=1
        report['one_frame_calls' if inputs.shape[2]==1 else 'seventeen_frame_calls']+=1
        # Retain any returned tensor before testing finiteness or prefix bounds.
        if isinstance(value,torch.Tensor):
            values=dict(values);values['observation' if inputs.shape[2]==1 else 'target']=value
            report['files'][name]=save(out/name,values)
        report.setdefault('calls',[]).append(dict(file=name,rgb_sha256=before,frames=inputs.shape[2],seconds=time.monotonic()-start))
        emit()
        if tensor_sha(inputs)!=before:raise RuntimeError('Encoder mutated original RGB')
        _latent(value,1 if inputs.shape[2]==1 else 5);check()
        return value
    def compare(candidate,reference,name,exact):
        row=prefix_check(candidate,reference,name,
            'Exact same-length/cache-isolation comparison' if exact else 'Independent one-frame versus17-frame prefix',
            require_bit_exact=exact)
        report['checks'].append(row);emit()
        if not row['passed']:raise RuntimeError('Predeclared prefix check failed: '+name)
    first_video=None;first_full=None
    try:
        for index,arm in enumerate(reader.ARMS):
            check();window=read_arm(arm);rgb,commands=video(window)
            initial=rgb[:,:,:1].clone()
            if first_video is None:
                first_video=initial.clone()
                observation=encode('observation.safetensors',first_video,{})
            if not torch.equal(initial,first_video):raise ValueError('Original initial RGB is not shared exactly')
            target=encode(arm+'.safetensors',rgb,{'observation':observation,'commands':commands})
            compare(target[:,:,:1],observation,arm+'_cross_length',False)
            if first_full is None:first_full=target[:,:,:1].clone()
            else:compare(target[:,:,:1],first_full,arm+'_same_length',True)
            del window,rgb,commands,initial,target
        repeated=encode('observation-repeat.safetensors',first_video,{})
        compare(repeated,observation,'shared_image_after_six_videos',True)
        if (report['encoder_completed']!=8 or report['one_frame_calls']!=2 or report['seventeen_frame_calls']!=6
                or len(report['checks'])!=12):raise RuntimeError('Fixed encode/check count changed')
        report.update(status='passed',model_execution=True,inputs_unchanged=True)
        emit();return report
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__,error=str(error));emit();raise


def read_window(root,arm,*,conditioning_only=False):
    """Read passed encoded artifacts; parent terminal admission is separate."""
    if arm not in reader.ARMS or type(conditioning_only) is not bool:raise ValueError('Explicit arm/read mode required')
    root=Path(root);record=packet.evidence.read_json(root/'completion.json')
    identity=record.get('identity',{})
    if (record.get('schema')!=packet.SCHEMA or record.get('status')!='passed'
            or record.get('encoder_attempts')!=8 or record.get('encoder_completed')!=8
            or record.get('one_frame_calls')!=2 or record.get('seventeen_frame_calls')!=6
            or record.get('arms')!=list(reader.ARMS) or identity.get('source_sha256')!=packet.sources()
            or identity.get('manifest_sha256')!=packet.MANIFEST
            or set(record.get('files',{}))!={'observation.safetensors','observation-repeat.safetensors',*(a+'.safetensors' for a in reader.ARMS)}):
        raise ValueError('Complete source-bound native cache required')
    expected_checks=[a+'_cross_length' for a in reader.ARMS]+[a+'_same_length' for a in reader.ARMS[1:]]+['shared_image_after_six_videos']
    rows=record.get('checks',[])
    if len(rows)!=12 or {r.get('name') for r in rows}!=set(expected_checks):raise ValueError('Complete prefix checks required')
    for row in rows:
        exact=not row['name'].endswith('_cross_length')
        if (row.get('passed') is not True or row.get('requires_bit_exact') is not exact
                or row.get('target_prefix_replaced') is not False
                or row.get('limits')!={'max_abs':1e-5,'relative_l2':1e-5}
                or (exact and row.get('bit_exact_equal') is not True)):
            raise ValueError('Prefix contract differs')
    prefixes={};observations={}
    for name,value in record['files'].items():
        file=packet.evidence.relative_file(root,name)
        if not 0<file.stat().st_size<=8*2**20 or file.stat().st_size!=value['bytes'] or sha(file)!=value['sha256']:
            raise ValueError('Encoded artifact differs')
        window_name=name.removesuffix('.safetensors')
        expected={'target':([1,48,5,44,78],'F32'),'observation':([1,48,1,44,78],'F32'),
                  'commands':([1,16,6],'F32')} if window_name in reader.ARMS else {'observation':([1,48,1,44,78],'F32')}
        with safe_open(file,framework='pt',device='cpu') as handle:
            if set(handle.keys())!=set(expected):raise ValueError('Encoded file keys differ')
            for key,(shape,dtype) in expected.items():
                view=handle.get_slice(key)
                if view.get_shape()!=shape or view.get_dtype()!=dtype:raise ValueError('Encoded header differs')
            observation=handle.get_tensor('observation');_latent(observation,1)
            if tensor_sha(observation)!=value['tensors']['observation']['sha256']:raise ValueError('Observation hash differs')
            observations[window_name]=observation
            if window_name in reader.ARMS:
                # Read only the initial target latent for the causal checks.
                # Conditioning-only reads never materialize future targets.
                prefix=handle.get_slice('target')[:,:,:1];_latent(prefix,1)
                prefixes[window_name]=prefix
                commands=handle.get_tensor('commands')
                if (not torch.isfinite(commands).all()
                        or tensor_sha(commands)!=value['tensors']['commands']['sha256']
                        or tensor_sha(commands)!=identity.get('commands_sha256',{}).get(window_name)):
                    raise ValueError('Commands differ from exact prepared reader receipt')
    shared=observations['observation']
    if any(tensor_sha(observations[a])!=tensor_sha(shared) for a in reader.ARMS):raise ValueError('Independent observation mapping changed')
    recomputed=[]
    for a in reader.ARMS:
        recomputed.append(prefix_check(prefixes[a],shared,a+'_cross_length','recomputed',require_bit_exact=False))
    for a in reader.ARMS[1:]:
        recomputed.append(prefix_check(prefixes[a],prefixes[reader.ARMS[0]],a+'_same_length','recomputed',require_bit_exact=True))
    recomputed.append(prefix_check(observations['observation-repeat'],shared,'shared_image_after_six_videos','recomputed',require_bit_exact=True))
    recorded={r['name']:r for r in rows}
    for row in recomputed:
        previous=recorded[row['name']]
        if not row['passed'] or any(previous.get(k)!=row[k] for k in
                ('candidate_sha256','reference_sha256','exact_equal','bit_exact_equal','requires_bit_exact')):
            raise ValueError('Saved prefix evidence fails independent recomputation')
        for key in ('max_abs','relative_l2','relative_l2_reference_norm'):
            a,b=previous.get(key),row[key]
            if a is None or b is None:
                if a!=b:raise ValueError('Prefix diagnostic differs')
            elif (type(a) not in (int,float) or not math.isfinite(a)
                  or not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15)):
                raise ValueError('Prefix diagnostic differs beyond declared CPU reduction bound')
    path=root/(arm+'.safetensors');spec={'target':([1,48,5,44,78],'F32'),
        'observation':([1,48,1,44,78],'F32'),'commands':([1,16,6],'F32')}
    selected=('observation','commands') if conditioning_only else tuple(spec)
    with safe_open(path,framework='pt',device='cpu') as handle:
        if set(handle.keys())!=set(spec):raise ValueError('Exact window tensor names required')
        for key,(shape,dtype) in spec.items():
            view=handle.get_slice(key)
            if view.get_shape()!=shape or view.get_dtype()!=dtype:raise ValueError('Window header differs')
        values={}
        for key in selected:
            value=handle.get_tensor(key)
            if not torch.isfinite(value).all() or tensor_sha(value)!=record['files'][path.name]['tensors'][key]['sha256']:
                raise ValueError('Selected latent or command tensor differs')
            values[key]=value
    return values,dict(completion_sha256=sha(root/'completion.json'),file_sha256=sha(path),
                       identity=identity,materialized_tensor_keys=list(selected))
