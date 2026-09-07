# SPDX-License-Identifier: Apache-2.0
"""One reviewed 50-step pure T2V control with the official FP32 Wan model."""
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
from native_control.loop import integrate
from native_control.evidence import Evidence


def validate_profile(directory, text_cache, solver_profile, decoder_profile=None):
    directory = Path(directory)
    profile = json.loads((directory/'metrics.json').read_text())
    estimate_key = 'non_pair_overhead_plus_50_pairs_plus_decode30s_and_artifacts10s_seconds'
    estimate = profile.get('runtime_estimate', {})
    if profile.get('status') != 'passed':
        raise ValueError('Completed FP32 profile required')
    for key in ('measured_first_pair_seconds', estimate_key):
        value = estimate.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= 900:
            raise ValueError('Invalid or over-budget runtime estimate')
    if profile.get('seed') != SEED or profile.get('steps_for_estimate') != STEPS or profile.get('shift') != SHIFT or profile.get('guidance') != GUIDANCE:
        raise ValueError('Profile settings differ from declared control')
    if profile['base']['weight_sha256'] != FILES['diffusion_pytorch_model.safetensors'][1] or profile['base']['config_sha256'] != FILES['config.json'][1]:
        raise ValueError('Profile foundation differs')
    for relative in ('portable.py', 'sampling.py', 'profile_pair.py', 'vendor/model.py', 'vendor/fm_solvers_unipc.py', '../fetch_weights.py', '../text_cache/cache.py'):
        if profile['source_sha256'][relative] != sha(Path(__file__).parent/relative):
            raise ValueError(f'Profile source no longer matches: {relative}')
    if sha(Path(text_cache)/'manifest.json') != profile['text']['manifest_sha256'] or sha(Path(text_cache)/'embeddings.safetensors') != profile['text']['embeddings_sha256']:
        raise ValueError('Text identities differ from measured profile')
    scheduler = make_scheduler('cpu')
    if profile['scheduler']['timesteps'] != scheduler.timesteps.tolist() or profile['scheduler']['sigmas'] != scheduler.sigmas.tolist():
        raise ValueError('Official schedule differs from measured profile')
    overhead = json.loads(Path(solver_profile).read_text())
    if overhead.get('status') != 'passed' or overhead.get('steps') != 50 or overhead.get('loop_sha256') != sha(Path(__file__).with_name('loop.py')):
        raise ValueError('Current measured solver/transfer overhead required')
    extra = overhead.get('solver_and_transfers_seconds')
    if not isinstance(extra, (float, int)) or not math.isfinite(extra) or extra < 0:
        raise ValueError('Invalid measured solver/transfer overhead')
    if decoder_profile is None:
        decoder_profile = Path(__file__).parent.parent/'codec/results/generated-base-fp32/metrics.json'
    decoder = json.loads(Path(decoder_profile).read_text())
    if decoder.get('status') != 'passed' or decoder.get('dtype') != 'float32' or decoder.get('weights_sha256') != FILES['Wan2.1_VAE.pth'][1]:
        raise ValueError('Completed official FP32 decoder profile required')
    for name,relative in (('helper.py','helper.py'),('decode_policy.py','decode_policy.py'),('wan_vae.py','vendor/wan_vae.py')):
        if decoder['source_sha256'][name] != sha(Path(__file__).parent.parent/'codec'/relative):
            raise ValueError('Decoder source differs from measured profile')
    decoder_times = {row['stage']:row['seconds'] for row in decoder['timings']}
    decoder_seconds = sum(decoder_times[name] for name in ('load_official_vae','base_decode_fp32'))
    if not math.isfinite(decoder_seconds) or decoder_seconds <= 0:
        raise ValueError('Invalid measured FP32 decoder time')
    combined = estimate[estimate_key]-30+decoder_seconds+extra
    if combined > 900:
        raise ValueError('Combined observed estimate exceeds the fixed900-second cap')
    return profile, combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--text-cache', type=Path, required=True)
    parser.add_argument('--profile-run', type=Path, required=True)
    parser.add_argument('--solver-profile', type=Path, required=True)
    parser.add_argument('--decoder-profile',type=Path,default=Path(__file__).parent.parent/'codec/results/generated-base-fp32/metrics.json')
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
        report.update(experiment='One pure T2V reduced-resolution native-style control',
            image_conditioning=False, action_conditioning=False, adapter_loaded=False, capture_cache_access=False,
            observed_frames=0, generated_frames=17, size=[512,288], seed=SEED, guidance=GUIDANCE, shift=SHIFT,
            declared_solver_steps=STEPS, declared_model_calls=STEPS*2, projected_seconds=estimated,
            denoiser_device='mps', denoiser_precision='float32 parameters, all activations and attention',
            scheduler_device='cpu', scheduler_precision='float32 unchanged official UniPC',
            decoder_precision='float32 official VAE, allocator-only cleanup after temporal chunks',
            native_cuda_bf16_flashattention_reproduced=False,
            preview_playback_fps=10, torch=torch.__version__, platform=platform.platform(),
            profile_metrics_sha256=sha(args.profile_run/'metrics.json'), solver_profile_sha256=sha(args.solver_profile),
            decoder_profile_sha256=sha(args.decoder_profile),
            text=profile['text'], scheduler=profile['scheduler'],
            automatic_mps_cpu_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0'))
        tensor_path = args.profile_run/'runtime-tensors.safetensors'
        if sha(tensor_path) != profile['runtime_tensors_sha256']:
            raise ValueError('Profile initial-noise artifact changed')
        with safe_open(tensor_path, framework='pt', device='cpu') as tensors:
            noise = tensors.get_tensor('initial_noise')
        if not torch.equal(noise, initial_noise()):
            raise ValueError('Noise differs from declared exact seed artifact')
        save_file({'initial_noise':noise}, str(args.output/'initial-noise.safetensors'))
        report['initial_noise_sha256'] = sha(args.output/'initial-noise.safetensors')
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
            latent = evidence.measure('50_step_unipc',lambda:integrate(model,noise,negative,positive,device='mps',callback=step_record))
        if len(report['steps']) != 50:
            raise RuntimeError('Incomplete scheduler sequence')
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
            draw.text((x+8,y+5),f'Pure T2V generated frame {index}',fill=(20,20,20));sheet.paste(images[index],(x,y+24))
        sheet.save(args.output/'comparison.png')
        report['output_sha256'] = {str(path.relative_to(args.output)):sha(path) for path in [*sorted(frames.glob('*.png')),args.output/'comparison.png',args.output/'preview.gif']}
        report['generated_frames_finite'] = True
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
