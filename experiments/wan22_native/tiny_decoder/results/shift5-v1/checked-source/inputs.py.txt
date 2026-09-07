# SPDX-License-Identifier: Apache-2.0
"""Verified saved native48 latents. Reference RGB is loaded only after decoding."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from ..codec import WEIGHT_SHA256 as FULL_VAE_SHA256
from .codec import sha,tensor_sha


def local(root,name):
    root=Path(root).resolve();path=root/name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('Evidence file must stay inside source run')
    return path


def checked(root,name,digest):
    path=local(root,name)
    if not isinstance(digest,str) or sha(path)!=digest:raise ValueError('Saved evidence file changed: '+name)
    return path


def read_tensor(path,key,keys,shape,dtype,max_bytes):
    path=Path(path)
    if path.stat().st_size>max_bytes:raise ValueError('Input exceeds bounded artifact size')
    with safe_open(path,framework='pt',device='cpu') as handle:
        if set(handle.keys())!=set(keys):raise ValueError('Unexpected tensor keys')
        if tuple(handle.get_slice(key).get_shape())!=tuple(shape):raise ValueError('Saved tensor shape differs')
        value=handle.get_tensor(key)
    if value.dtype!=dtype or not torch.isfinite(value).all():raise ValueError('Saved tensor must have declared finite dtype')
    return value


def describe_run(directory,kind):
    root=Path(directory).resolve()
    if kind not in ('reconstruction','shift5','shift3'):raise ValueError('Only original reconstruction or declared generated controls')
    if (root/'watchdog-stop.json').exists():raise ValueError('Stopped source run')
    if kind=='reconstruction':
        terminal=json.loads(local(root,'terminal.json').read_text())
        report_path=local(root,'result/metrics.json');report=json.loads(report_path.read_text())
        if (terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or report.get('status')!='passed'
                or report.get('mode')!='roundtrip' or report.get('reconstructed_frames')!=17
                or report.get('newly_generated_frames')!=0 or report.get('finite_output') is not True
                or report.get('codec',{}).get('weight_sha256')!=FULL_VAE_SHA256
                or report.get('target_latents_from_old_cache') is not False
                or (root/'result/watchdog-stop.json').exists()):
            raise ValueError('Completed original native48 full17 codec reconstruction required')
        hashes=report['output_sha256'];latent_name='result/encoded.safetensors';key='target';keys={'target','observation'}
        checked(root,latent_name,hashes['encoded.safetensors'])
        baseline_name='result/reconstruction.safetensors';checked(root,baseline_name,hashes['reconstruction.safetensors'])
        truth_name='result/rgb-input.safetensors';checked(root,truth_name,hashes['rgb-input.safetensors'])
        extra={'baseline_key':'reconstruction','truth_file':truth_name,'truth_sha256':hashes['rgb-input.safetensors']}
        baseline_sha=hashes['reconstruction.safetensors'];latent_sha=hashes['encoded.safetensors']
    else:
        report_path=local(root,'metrics.json');report=json.loads(report_path.read_text())
        expected={'steps':50,'shift':5. if kind=='shift5' else 3.,'guidance':5.}
        if (report.get('status')!='passed' or report.get('sampling_configuration')!=expected
                or report.get('frame_count')!=17 or report.get('generated_future_frames')!=16):
            raise ValueError('Completed declared generated clip required')
        stages={}
        for stage in ('core','decode'):
            terminal=json.loads(local(root,stage+'/terminal.json').read_text())
            path=checked(root,stage+'/result/metrics.json',report[stage+'_metrics_sha256'])
            stages[stage]=json.loads(path.read_text())
            if terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or stages[stage].get('status')!='passed' or (root/stage/'result/watchdog-stop.json').exists():
                raise ValueError('Incomplete generated source stage')
        if (stages['core'].get('completed_steps')!=50 or stages['core'].get('completed_core_calls')!=100
                or stages['core'].get('all_prefix_checks_passed') is not True
                or stages['decode'].get('codec',{}).get('weight_sha256')!=FULL_VAE_SHA256):
            raise ValueError('Source is not the full native48 generated control')
        latent_name='core/result/latents.safetensors';key='latent';keys={'latent'}
        latent_sha=stages['core']['output_sha256']['latents.safetensors'];checked(root,latent_name,latent_sha)
        if report.get('output_latent_sha256')!=latent_sha:raise ValueError('Source parent latent identity differs')
        baseline_name='decode/result/decoded.safetensors';baseline_sha=stages['decode']['output_sha256']['decoded.safetensors']
        checked(root,baseline_name,baseline_sha);extra={'baseline_key':'video','truth_file':None}
    return {'kind':kind,'source_metrics_sha256':sha(report_path),'latent_file':latent_name,'latent_file_sha256':latent_sha,
        'latent_key':key,'latent_keys':sorted(keys),'latent_shape':[1,48,5,18,32],
        'baseline_file':baseline_name,'baseline_sha256':baseline_sha,**extra,
        'input_convention':'Native48 normalized diffusion latent, passed directly without inverse mean/std',
        'raw_reference_loaded_before_decode':False,'source_original_run_unchanged':True}


def load_latent(directory,spec):
    path=checked(directory,spec['latent_file'],spec['latent_file_sha256'])
    value=read_tensor(path,spec['latent_key'],spec['latent_keys'],(1,48,5,18,32),torch.float32,2**21)
    return value,{'tensor_sha256':tensor_sha(value),'materialized_keys':[spec['latent_key']]}


def load_references_after_decode(directory,spec):
    path=checked(directory,spec['baseline_file'],spec['baseline_sha256'])
    baseline=read_tensor(path,spec['baseline_key'],{spec['baseline_key']},(1,3,17,288,512),torch.float32,40*2**20)
    truth=None
    if spec['truth_file'] is not None:
        path=checked(directory,spec['truth_file'],spec['truth_sha256'])
        truth=read_tensor(path,'rgb',{'rgb'},(17,288,512,3),torch.uint8,9*2**20)
    return baseline,truth
