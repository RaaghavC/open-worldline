# SPDX-License-Identifier: Apache-2.0
"""One-factor initial-observation clamp, with successful native settings fixed."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from fetch_weights import FILES, sha
from text_cache.cache import load_context
from codec.helper import OfficialWanCodec
from codec.decode_policy import cleanup_after_temporal_chunk
from native_control.profile_pair import load_core, native_negative_prompt
from native_control.sampling import initial_noise, make_scheduler, STEPS, SHIFT, GUIDANCE, SEED
from native_control.clamp_loop import integrate_clamped, clamp_first
from native_control.clamp_cache import read_observation
from native_control.evidence import Evidence


from native_control.run_clip import validate_profile


def validate_control(directory, profile, text_cache):
    directory=Path(directory)
    control=json.loads((directory/'metrics.json').read_text())
    if control.get('status') != 'passed' or len(control.get('steps',[])) != 50 or control.get('observed_frames') != 0 or control.get('generated_frames') != 17:
        raise ValueError('Completed exact pure T2V control required')
    for key,expected in [('seed',SEED),('shift',SHIFT),('guidance',GUIDANCE),('declared_solver_steps',50),('declared_model_calls',100)]:
        if control.get(key) != expected: raise ValueError('Completed control settings differ')
    if control['base'] != profile['base'] or control['text'] != profile['text'] or control['scheduler'] != profile['scheduler']:
        raise ValueError('Base, text or scheduler identity differs from successful control')
    for relative in ('portable.py','sampling.py','loop.py','run_clip.py','profile_pair.py','evidence.py','vendor/model.py','vendor/fm_solvers_unipc.py','../codec/helper.py','../codec/decode_policy.py','../codec/vendor/wan_vae.py'):
        if control['source_sha256'][relative] != sha(Path(__file__).parent/relative):
            raise ValueError(f'Successful source changed: {relative}')
    if sha(Path(text_cache)/'manifest.json') != control['text']['manifest_sha256']:
        raise ValueError('Native text cache changed')
    return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--text-cache', type=Path, required=True)
    parser.add_argument('--profile-run', type=Path, required=True)
    parser.add_argument('--solver-profile', type=Path, required=True)
    parser.add_argument('--decoder-profile',type=Path,default=Path(__file__).parent.parent/'codec/results/generated-base-fp32/metrics.json')
    parser.add_argument('--control-run',type=Path,required=True)
    parser.add_argument('--observation-cache',type=Path,required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    evidence = Evidence(args.output)
    report = evidence.report
    error = None
    try:
        profile, estimated = validate_profile(args.profile_run, args.text_cache, args.solver_profile,args.decoder_profile)
        control=validate_control(args.control_run,profile,args.text_cache)
        observation,observation_provenance=read_observation(args.observation_cache)
        report.update(experiment='One-factor initial-image clamp with successful native-style settings',
            image_conditioning=True, action_conditioning=False, adapter_loaded=False, capture_cache_access='observation only',
            observed_frames=1, generated_future_frames=16, total_decoded_frames=17, size=[512,288], seed=SEED, guidance=GUIDANCE, shift=SHIFT,
            declared_solver_steps=STEPS, declared_model_calls=STEPS*2, projected_seconds=estimated,
            denoiser_device='mps', denoiser_precision='float32 parameters, all activations and attention',
            scheduler_device='cpu', scheduler_precision='float32 unchanged official UniPC',
            decoder_precision='float32 official VAE, allocator-only cleanup after temporal chunks',
            native_cuda_bf16_flashattention_reproduced=False,
            preview_playback_fps=10, torch=torch.__version__, platform=platform.platform(),
            profile_metrics_sha256=sha(args.profile_run/'metrics.json'), solver_profile_sha256=sha(args.solver_profile),
            decoder_profile_sha256=sha(args.decoder_profile),
            text=profile['text'], scheduler=profile['scheduler'],
            observation=observation_provenance, control_metrics_sha256=sha(args.control_run/'metrics.json'),
            sole_treatment='Clamp independently observed frame0 before each denoiser and after each CPU UniPC update; official solver history unchanged',
            automatic_mps_cpu_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0'))
        tensor_path = args.control_run/'initial-noise.safetensors'
        if sha(tensor_path) != control['initial_noise_sha256']:
            raise ValueError('Successful initial-noise artifact changed')
        with safe_open(tensor_path, framework='pt', device='cpu') as tensors:
            noise = tensors.get_tensor('initial_noise')
        if not torch.equal(noise, initial_noise()):
            raise ValueError('Noise differs from declared exact seed artifact')
        save_file({'initial_noise':noise}, str(args.output/'initial-noise.safetensors'))
        report['initial_noise_sha256'] = sha(args.output/'initial-noise.safetensors')
        if report['initial_noise_sha256'] != control['initial_noise_sha256']:
            raise RuntimeError('Retained original noise serialization changed')
        save_file({'observation':observation},str(args.output/'observation.safetensors'))
        save_file({'clamped_initial':clamp_first(noise,observation)},str(args.output/'clamped-initial.safetensors'))
        report['observation_file_sha256']=sha(args.output/'observation.safetensors')
        report['clamped_initial_sha256']=sha(args.output/'clamped-initial.safetensors')
        report['initial_noise_tensor_sha256'] = hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest()
        positive_text = 'A sunlit interior with warm plaster walls, a wooden door, limestone flooring, brass details, and green plants.'
        positive = load_context(args.text_cache,'atrium',expected_text=positive_text,dtype=torch.float32)
        negative = load_context(args.text_cache,'native_negative',expected_text=native_negative_prompt(),dtype=torch.float32)
        source_files = [p for p in sorted(Path(__file__).parent.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.txt') and 'results' not in p.parts]
        snapshot = args.output/'measured-source'; snapshot.mkdir()
        report['source_sha256'] = {}
        for path in source_files:
            relative = str(path.relative_to(Path(__file__).parent))
            destination = snapshot/(relative+'.txt'); destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination); report['source_sha256'][relative] = sha(path)
        for relative in ('fetch_weights.py','text_cache/cache.py','codec/helper.py','codec/decode_policy.py','codec/vendor/wan_vae.py','codec/vendor/source-manifest.json','requirements-real.txt','requirements.txt','codec/requirements.txt'):
            path = Path(__file__).parent.parent/relative
            destination = snapshot/'shared-source'/(relative+'.txt'); destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination); report['source_sha256']['../'+relative] = sha(path)
        model, report['base'] = evidence.measure('load_fp32_core',lambda:load_core(args.weights,'mps'))
        torch.mps.empty_cache()
        def step_record(index,timestep,latent,times):
            report['steps'].append({'step':index+1,'timestep':int(timestep),'min':float(latent.min()),'max':float(latent.max()),**times})
            evidence.stage = f'denoise {index+1}/50 complete'
            evidence.save()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='.*torch.cuda.amp.autocast.*')
            warnings.filterwarnings('ignore',message="User provided device_type of 'cuda'.*")
            latent = evidence.measure('50_step_unipc',lambda:integrate_clamped(model,noise,observation,negative,positive,device='mps',callback=step_record))
        if len(report['steps']) != 50 or any(step['clamped_prefix_max_abs_difference'] != 0 for step in report['steps']):
            raise RuntimeError('Incomplete sequence or observed-prefix drift')
        if not torch.equal(noise,initial_noise()):
            raise RuntimeError('Clamp mutated the caller original noise')
        save_file({'generated':latent.contiguous()},str(args.output/'generated-latents.safetensors'))
        report['generated_latents_sha256'] = sha(args.output/'generated-latents.safetensors')
        del model; gc.collect(); torch.mps.empty_cache()
        codec = evidence.measure('load_official_vae',lambda:OfficialWanCodec(args.weights/'Wan2.1_VAE.pth','mps',torch.float32))
        torch.mps.empty_cache()
        with cleanup_after_temporal_chunk(codec) as cleanup:
            video = evidence.measure('decode_generated_clip',lambda:codec.decode(latent.unsqueeze(0)))
        report['decoder_allocator_hook'] = cleanup
        if list(video.shape) != [1,3,17,288,512] or not torch.isfinite(video).all():
            raise RuntimeError('Invalid decoded output')
        pixels = np.rint(((video[0].permute(1,2,3,0).cpu().numpy()+1)/2).clip(0,1)*255).astype(np.uint8)
        del video,codec; gc.collect(); torch.mps.empty_cache()
        evidence.stage = 'save-frames'
        frames = args.output/'frames'; frames.mkdir()
        images = []
        for index,array in enumerate(pixels):
            image = Image.fromarray(array); image.save(frames/f'{index:04d}.png'); images.append(image)
        images[0].save(args.output/'preview.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
        sheet = Image.new('RGB',(1024,624),(239,238,230)); draw=ImageDraw.Draw(sheet)
        for position,index in enumerate((0,5,10,16)):
            x=(position%2)*512; y=(position//2)*312
            draw.text((x+8,y+5),('Observed reconstruction' if index == 0 else f'Clamped control generated frame {index}'),fill=(20,20,20));sheet.paste(images[index],(x,y+24))
        sheet.save(args.output/'comparison.png')
        report['output_sha256'] = {str(path.relative_to(args.output)):sha(path) for path in [*sorted(frames.glob('*.png')),args.output/'comparison.png',args.output/'preview.gif']}
        report['decoded_frames_finite'] = True
        if sha(args.weights/'diffusion_pytorch_model.safetensors') != FILES['diffusion_pytorch_model.safetensors'][1]:
            raise RuntimeError('External source file changed')
    except BaseException as exc:
        error = exc
        raise
    finally:
        evidence.close(error)
    print(json.dumps({key:value for key,value in report.items() if key not in ('steps','source_sha256','scheduler','output_sha256')},indent=2))


if __name__ == '__main__':
    main()
