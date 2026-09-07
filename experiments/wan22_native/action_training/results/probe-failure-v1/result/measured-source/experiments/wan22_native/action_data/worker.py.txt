# SPDX-License-Identifier: Apache-2.0
"""The only process that loads the native VAE; bounded by parent and monitor."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors.torch import save_file

from ..codec import Wan22Codec, cleanup_temporal_chunks
from ..codec_memory import cleanup_causal_convolutions
from ..codec_profile import Guard, atomic_json
from ..codec_decode_profile import runtime_environment, chunk_timings
from . import data
from .cache import SCHEMA, SHAPES
from .operations import tensor_sha, exact_prefix, observation_only, reconstruct_target, score_reconstruction
from .runtime import ENVIRONMENT, gates, snapshot


def save_tensors(path, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    values = {k:v.detach().cpu().contiguous() for k,v in values.items()}
    temporary = path.with_name(path.name+'.tmp')
    save_file(values,str(temporary));temporary.replace(path)
    return {'file':str(path.name),'sha256':data.sha(path),'tensors':{
        k:{'shape':list(v.shape),'dtype':str(v.dtype).removeprefix('torch.'),'sha256':tensor_sha(v)} for k,v in values.items()}}


def retain_check(guard, row):
    guard.report.setdefault('causal_checks',[]).append(row);guard.save()
    if row['passed'] is not True:
        raise RuntimeError('Exact native prefix causality failed: '+row['name'])


def encode_cache(codec, guard, capture, output):
    windows = data.load_selection(capture)
    manifest = {'schema':SCHEMA,'status':'running','command_channels':list(data.CHANNELS),
        'split':'development','independent_layouts':1,'data_license':'CC0-1.0',
        'target_shape':list(SHAPES['target']),'observation_shape':list(SHAPES['observation']),
        'commands_shape':list(SHAPES['commands']),'windows':[], 'observations':[],
        'source_sha256':guard.report['source_sha256'],'codec':codec.provenance,
        'runtime_environment':guard.report['runtime_environment'],
        'canonicalization':'Only derived closed start0 RGB frame0 replaced with original open/0000 before full encoding',
        'shared_start0_observation':True,'target_prefix_overwritten_after_encoding':False}
    atomic_json(output/'manifest.json',manifest)
    shared = None
    for arm,start in data.SELECTION:
        identity = f'{arm}-{start:04d}';window = windows[(arm,start)]
        video = torch.from_numpy(window.video_array())
        observation_id = 'start-0000-shared' if start==0 else identity
        if start==0 and shared is not None:
            observation = shared
        else:
            observation = guard.measure(identity+'_encode_initial_alone',lambda:observation_only(codec,video)).cpu().contiguous()
            if start==0: shared = observation
            spec = save_tensors(output/'observations'/(observation_id+'.safetensors'),{'observation':observation})
            spec.update(id=observation_id,file='observations/'+spec['file'],
                source_first_rgb_sha256=data.array_sha(window.rgb[0]),encoded_rgb_frames=1)
            manifest['observations'].append(spec)
        target = guard.measure(identity+'_encode_full_17_rgb',lambda:codec.encode(video)).cpu().contiguous()
        retain_check(guard,exact_prefix(target,observation,name=identity+'_target_prefix_vs_independent_initial'))
        if not codec.cache_is_clear(): raise RuntimeError('Native encode cache retained')
        values = {'target':target,'observation':observation,'commands':torch.from_numpy(window.commands.copy())[None]}
        for name,value in values.items():
            if tuple(value.shape)!=SHAPES[name] or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                raise RuntimeError('Invalid cache shape/dtype/finite values: '+name)
        spec = save_tensors(output/(identity+'.safetensors'),values)
        spec.update(id=identity,observation_id=observation_id,source=window.provenance)
        manifest['windows'].append(spec)
        manifest['causal_checks']=guard.report['causal_checks'];atomic_json(output/'manifest.json',manifest)
        if (arm,start) in (('open',0),('open',8)):
            # Perturb only future RGB; the initial-image input is kept exact.
            altered=video.clone();altered[:,:,1:]=-altered[:,:,1:]
            changed=guard.measure(identity+'_future_rgb_perturbation',lambda:codec.encode(altered)).cpu()
            retain_check(guard,exact_prefix(changed,observation,name=identity+'_future_rgb_cannot_change_initial_latent'))
            if start==0:
                repeated=guard.measure('shared_initial_repeat_after_other_full_encode',lambda:observation_only(codec,video)).cpu()
                retain_check(guard,exact_prefix(repeated,shared,name='shared_initial_A_after_full_B_is_identical'))
        del target,video,values
    if len(manifest['windows'])!=8 or len(manifest['observations'])!=7 or len(guard.report['causal_checks'])!=11:
        raise RuntimeError('Incomplete prescribed windows, observations or causal checks')
    manifest.update(status='passed',causal_checks=guard.report['causal_checks'])
    atomic_json(output/'manifest.json',manifest)
    guard.report.update(cache_manifest_sha256=data.sha(output/'manifest.json'),cache_windows=8,
        independent_observation_encodes=7,finite_output=True,commands_provided_to_codec=False)


def roundtrip(codec, guard, capture, output):
    rgb, provenance=data.load_roundtrip_rgb(capture)
    video=torch.from_numpy(np.ascontiguousarray(rgb.transpose(3,0,1,2)[None],dtype=np.float32)/127.5-1.)
    inputs=save_tensors(output/'rgb-input.safetensors',{'rgb':torch.from_numpy(rgb.copy())})
    guard.report.update(input=provenance,input_file=inputs,action_values_materialized=False,
        commands_provided_to_codec=False,target_latents_from_old_cache=False,
        reconstructed_frames=17,newly_generated_frames=0,quality_scope='Codec reconstruction of supplied original RGB only')
    observation=guard.measure('encode_initial_rgb_alone',lambda:observation_only(codec,video)).cpu().contiguous()
    target=guard.measure('encode_original_17_rgb',lambda:codec.encode(video)).cpu().contiguous()
    save_tensors(output/'encoded.safetensors',{'target':target,'observation':observation})
    retain_check(guard,exact_prefix(target,observation,name='full_target_prefix_vs_independent_original_initial'))
    if target.shape!=(1,48,5,18,32):raise RuntimeError('Invalid native 48-channel encoded target')
    guard.report['encoded_target_sha256']=tensor_sha(target)
    # Decoder receives only this target tensor. RGB stays on CPU for later scoring.
    with chunk_timings(codec,guard.device) as chunks:
        prediction=guard.measure('decode_original_target_17_frames',lambda:reconstruct_target(codec,target)).cpu().contiguous()
    guard.report['decoder_chunks']=chunks
    if [r['output_shape'][2] for r in chunks]!=[1,4,4,4,4] or [r['first_chunk'] for r in chunks]!=[True,False,False,False,False]:
        raise RuntimeError('Different native decoder temporal chunks')
    if tensor_sha(target)!=guard.report['encoded_target_sha256']:raise RuntimeError('Decoder mutated saved target')
    saved=save_tensors(output/'reconstruction.safetensors',{'reconstruction':prediction})
    metrics,pixels=score_reconstruction(rgb,prediction)
    guard.report.update(reconstruction=metrics,reconstruction_file=saved,finite_output=True,
        original_pixels_sha256=data.array_sha(rgb),reconstructed_pixels_sha256=data.array_sha(pixels))
    truth_dir=output/'truth';recon_dir=output/'reconstruction';truth_dir.mkdir();recon_dir.mkdir()
    panels=[]
    for index in range(17):
        Image.fromarray(rgb[index]).save(truth_dir/f'{index:04d}.png')
        Image.fromarray(pixels[index]).save(recon_dir/f'{index:04d}.png')
        panel=Image.new('RGB',(1024,320),(240,239,235));draw=ImageDraw.Draw(panel)
        draw.text((8,6),f'Original RGB | frame {index}/16',fill=(25,25,25))
        draw.text((520,6),'Native FP32 codec reconstruction | no generated future',fill=(25,25,25))
        panel.paste(Image.fromarray(rgb[index]),(0,32));panel.paste(Image.fromarray(pixels[index]),(512,32));panels.append(panel)
    panels[0].save(output/'reconstruction-preview.gif',save_all=True,append_images=panels[1:],duration=100,loop=0)
    sheet=Image.new('RGB',(1024,1280))
    for position,index in enumerate((0,5,10,16)):sheet.paste(panels[index],(0,position*320))
    sheet.save(output/'comparison.png')
    guard.report['preview_playback_fps']=10
    guard.report['output_sha256']={str(p.relative_to(output)):data.sha(p) for p in [*truth_dir.glob('*.png'),*recon_dir.glob('*.png'),output/'comparison.png',output/'reconstruction-preview.gif',output/'encoded.safetensors',output/'rgb-input.safetensors',output/'reconstruction.safetensors']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text());device=request['device']
    if {n:os.environ.get(n) for n in ENVIRONMENT}!=ENVIRONMENT:raise RuntimeError('Measured decoder environment must be exact before worker startup')
    if device not in ('cpu','mps'):raise ValueError('Only CPU or MPS')
    torch.set_num_threads(4);guard=Guard(args.output,device);error=None;completed=False
    guard.report.update(schema='worldline-wan22-action-data-run-v1',mode=request['mode'],device=device,dtype='float32',
        runtime_environment=runtime_environment(device),source_sha256=snapshot(args.output),
        external_pretrained_codec=True,original_model=False,raw_capture_modified=False,
        dependencies={n:importlib.metadata.version(n) for n in ('torch','einops','numpy','pillow','psutil','safetensors')})
    try:
        guard.report['gates']=gates(request['cpu_report'],request['decoder_run'])
        codec=guard.measure('verify_and_load_fp32_codec',lambda:Wan22Codec(request['weights'],device));guard.report['codec']=codec.provenance
        with (args.output/'per-convolution-cleanup.jsonl').open('x') as log:
            def event(row):log.write(json.dumps(row)+'\n');log.flush()
            with cleanup_temporal_chunks(codec) as chunks,cleanup_causal_convolutions(codec,event_callback=event) as layers:
                guard.report['per_convolution_cleanup_report']=layers
                if request['mode']=='roundtrip':roundtrip(codec,guard,request['capture'],args.output)
                elif request['mode']=='cache':encode_cache(codec,guard,request['capture'],args.output)
                else:raise ValueError('Unsupported worker mode')
        guard.report.update(allocator_chunk_counts=chunks,cache_clear_after_operations=codec.cache_is_clear(),
            per_convolution_cleanup_log_sha256=data.sha(args.output/'per-convolution-cleanup.jsonl'))
        if not codec.cache_is_clear() or not layers['hooks_removed'] or layers['cleanup_calls']!=layers['completed_cleanups']:
            raise RuntimeError('Incomplete cleanup or retained native cache')
        completed=True
    except BaseException as problem:error=problem;raise
    finally:
        if not completed and error is None:error=RuntimeError('Codec work did not complete')
        guard.close(error)

if __name__=='__main__':main()
