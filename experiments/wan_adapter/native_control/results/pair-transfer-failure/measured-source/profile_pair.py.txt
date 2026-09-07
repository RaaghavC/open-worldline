# SPDX-License-Identifier: Apache-2.0
"""One FP32 negative/positive forward pair, after parity and real text checks.

This profiler cannot run a generation loop. It estimates the declared 50-step
cost and saves its initial noise and one velocity for runtime evidence only.
"""
import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import psutil
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from fetch_weights import FILES, REVISION, sha
from text_cache.cache import load_context
from native_control.portable import create_model
from native_control.sampling import initial_noise, make_scheduler, negative_positive_pair, STEPS, SHIFT, GUIDANCE, SEED

GIB = 1024**3


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def native_negative_prompt():
    tree = ast.parse((Path(__file__).parent/'vendor/shared_config.py.txt').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == 'sample_neg_prompt' for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Pinned official negative prompt not found')


def load_core(weights, device):
    weights = Path(weights)
    for name in ('diffusion_pytorch_model.safetensors', 'config.json'):
        expected_size, expected_sha = FILES[name]
        if (weights/name).stat().st_size != expected_size or sha(weights/name) != expected_sha:
            raise ValueError(f'Official source integrity mismatch: {name}')
    configuration = {key:value for key,value in json.loads((weights/'config.json').read_text()).items() if not key.startswith('_')}
    with torch.device('meta'):
        model = create_model(**configuration)
    state = {}
    with safe_open(weights/'diffusion_pytorch_model.safetensors', framework='pt', device='cpu') as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            if value.dtype != torch.float32:
                raise TypeError('Official source is expected to be float32')
            state[key] = value
    loaded = model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    del state
    gc.collect()
    return model, {'weight_revision': REVISION, 'weight_sha256': FILES['diffusion_pytorch_model.safetensors'][1],
                   'config_sha256': FILES['config.json'][1], 'parameters': sum(p.numel() for p in model.parameters()),
                   'loaded_keys': len(model.state_dict()), 'missing_keys': loaded.missing_keys, 'unexpected_keys': loaded.unexpected_keys}


def validate_parity(path):
    report = json.loads(Path(path).read_text())
    if report.get('status') != 'passed' or report.get('timesteps') != [999, 500, 50]:
        raise ValueError('Completed official-source CPU parity report required')
    for name in ('portable.py', 'vendor/model.py', 'test_parity.py'):
        if report['source_sha256'][name] != sha(Path(__file__).parent/name):
            raise ValueError(f'Parity source changed after measurement: {name}')
    return sha(path)


def validate_independent(path):
    report = json.loads(Path(path).read_text())
    if report.get('status') != 'passed' or report.get('tests_run', 0) < 8:
        raise ValueError('Completed independent parity/solver checks required')
    for name in ('portable.py', 'sampling.py', 'test_independent.py', 'vendor/model.py', 'vendor/fm_solvers_unipc.py'):
        if report['source_sha256'][name] != sha(Path(__file__).parent/name):
            raise ValueError(f'Independent test source changed: {name}')
    return sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--text-cache', type=Path, required=True)
    parser.add_argument('--parity-report', type=Path, required=True)
    parser.add_argument('--independent-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):
        parser.error('Run output must be outside the native_control source tree')
    if args.output.exists() or args.output.is_symlink():
        parser.error('Output directory must be new')
    args.output.mkdir(parents=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(min(1., 18*GIB/torch.mps.recommended_max_memory()))
    started = time.perf_counter()
    stage = ['validation']
    stop = threading.Event()
    peaks = {'rss_bytes': 0, 'mps_active_bytes': 0, 'mps_driver_bytes': 0}
    report = {'status': 'running', 'experiment': 'One native-style FP32 T2V CFG pair at reduced 512x288/17-frame shape',
              'device': 'mps', 'precision': 'All parameters, attention Q/K/V and activations float32; no autocast',
              'torch': torch.__version__, 'platform': platform.platform(),
              'automatic_mps_cpu_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0'),
              'native_cuda_bf16_flashattention_reproduced': False,
              'image_conditioning': False, 'action_conditioning': False, 'adapter_loaded': False,
              'capture_cache_access': False, 'completed_generation': False,
              'max_seconds': 900, 'max_memory_gib': 18, 'minimum_available_gib': 2,
              'timings': [], 'seed': SEED, 'steps_for_estimate': STEPS, 'shift': SHIFT, 'guidance': GUIDANCE,
              'noise_device': 'CPU generator, then MPS transfer; seed alone is not CUDA RNG bit-equivalence',
              'elapsed_scope': 'After interpreter/import startup; includes validation, loading, forwards, hashes and artifacts'}
    report['dependencies'] = {name: importlib.metadata.version(name) for name in ('torch', 'diffusers', 'numpy', 'safetensors', 'psutil', 'scipy')}
    write_json(args.output/'metrics.json', report)

    def monitor():
        process = psutil.Process()
        with (args.output/'memory.jsonl').open('x') as log:
            while not stop.is_set():
                row = {'seconds': time.perf_counter()-started, 'stage': stage[0], 'rss_bytes': process.memory_info().rss,
                       'available_system_bytes': psutil.virtual_memory().available,
                       'mps_active_bytes': torch.mps.current_allocated_memory(), 'mps_driver_bytes': torch.mps.driver_allocated_memory()}
                for key in peaks:
                    peaks[key] = max(peaks[key], row[key])
                log.write(json.dumps(row)+'\n'); log.flush()
                reason = None
                if row['seconds'] > 900:
                    reason = 'time-cap-900-seconds'
                elif max(row['rss_bytes'], row['mps_driver_bytes']) > 18*GIB:
                    reason = 'memory-cap-18-GiB'
                elif row['available_system_bytes'] < 2*GIB:
                    reason = 'available-system-memory-below-2-GiB'
                if reason:
                    write_json(args.output/'watchdog-stop.json', {'status': 'stopped', 'reason': reason, 'last_sample': row})
                    os._exit(124)
                stop.wait(.5)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def timed(label, function):
        stage[0] = label
        torch.mps.synchronize()
        start = time.perf_counter()
        result = function()
        torch.mps.synchronize()
        report['timings'].append({'stage': label, 'seconds': time.perf_counter()-start})
        write_json(args.output/'metrics.json', report)
        return result
    try:
        report['parity_report_sha256'] = validate_parity(args.parity_report)
        report['independent_report_sha256'] = validate_independent(args.independent_report)
        positive_text = 'A sunlit interior with warm plaster walls, a wooden door, limestone flooring, brass details, and green plants.'
        negative_text = native_negative_prompt()
        positive = load_context(args.text_cache, 'atrium', expected_text=positive_text, device='cpu', dtype=torch.float32)
        negative = load_context(args.text_cache, 'native_negative', expected_text=negative_text, device='cpu', dtype=torch.float32)
        report['text'] = {'manifest_sha256': sha(args.text_cache/'manifest.json'),
                          'embeddings_sha256': sha(args.text_cache/'embeddings.safetensors'),
                          'positive_text_sha256': hashlib.sha256(positive_text.encode()).hexdigest(),
                          'negative_text_sha256': hashlib.sha256(negative_text.encode()).hexdigest(),
                          'positive_tokens': len(positive), 'negative_tokens': len(negative)}
        source_files = [path for path in sorted(Path(__file__).parent.rglob('*'))
                        if path.is_file() and path.suffix in ('.py', '.txt', '.json') and 'results' not in path.parts]
        source = args.output/'measured-source'
        source.mkdir()
        report['source_sha256'] = {}
        for path in source_files:
            relative = path.relative_to(Path(__file__).parent)
            destination = source/(str(relative)+'.txt')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            report['source_sha256'][str(relative)] = sha(path)
        for relative in ('fetch_weights.py', 'text_cache/cache.py', 'requirements-real.txt', 'requirements.txt', 'codec/requirements.txt'):
            path = Path(__file__).parent.parent/relative
            destination = source/'shared-source'/(relative+'.txt')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            report['source_sha256']['../'+relative] = sha(path)
        model, report['base'] = timed('load_fp32_core', lambda: load_core(args.weights, 'mps'))
        positive = positive.to('mps')
        negative = negative.to('mps')
        noise = initial_noise(device='mps')
        saved_noise = noise.cpu().contiguous()
        schedule = make_scheduler('mps')
        report['scheduler'] = {'class': type(schedule).__name__, 'configuration': dict(schedule.config),
                               'timesteps': schedule.timesteps.cpu().tolist(), 'sigmas': schedule.sigmas.cpu().tolist(),
                               'timestep_dtype': str(schedule.timesteps.dtype), 'sigma_dtype': str(schedule.sigmas.dtype)}
        torch.mps.empty_cache()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')
            warnings.filterwarnings('ignore', message="User provided device_type of 'cuda'.*")
            velocity = timed('negative_positive_pair', lambda: negative_positive_pair(model, noise, schedule.timesteps[0], negative, positive))
        if not torch.equal(noise.cpu(), saved_noise):
            raise RuntimeError('Forward modified original Gaussian input')
        pair_seconds = report['timings'][-1]['seconds']
        report['runtime_estimate'] = {'measured_first_pair_seconds': pair_seconds,
            'denoise_50_pairs_seconds': pair_seconds*50,
            'load_plus_50_pairs_plus_decode_30s_seconds': report['timings'][0]['seconds']+pair_seconds*50+30,
            'method': '50 times the first actual-shape sequential FP32 pair; warm performance unmeasured; decoder allowance30s from prior run',
            'full_generation_started': False}
        report['velocity'] = {'shape': list(velocity.shape), 'dtype': str(velocity.dtype),
                              'finite': bool(torch.isfinite(velocity).all()), 'max_abs': float(velocity.abs().max())}
        stage[0] = 'save-and-verify'
        save_file({'initial_noise': saved_noise, 'one_cfg_velocity': velocity.cpu().contiguous()}, str(args.output/'runtime-tensors.safetensors'))
        report['runtime_tensors_sha256'] = sha(args.output/'runtime-tensors.safetensors')
        if sha(args.weights/'diffusion_pytorch_model.safetensors') != FILES['diffusion_pytorch_model.safetensors'][1]:
            raise RuntimeError('External source weight file changed')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stop.set(); thread.join(timeout=2)
        report.update(peaks_sampled=peaks, elapsed_seconds=time.perf_counter()-started)
        if report['status'] == 'passed':
            estimate = report['runtime_estimate']
            overhead = report['elapsed_seconds']-estimate['measured_first_pair_seconds']
            estimate['measured_non_pair_overhead_seconds'] = overhead
            estimate['non_pair_overhead_plus_50_pairs_plus_decode30s_and_artifacts10s_seconds'] = overhead+estimate['denoise_50_pairs_seconds']+40
            estimate['generation_requires_separate_review'] = True
        write_json(args.output/'metrics.json', report)
    print(json.dumps({key:value for key,value in report.items() if key not in ('source_sha256', 'scheduler')}, indent=2))


if __name__ == '__main__':
    main()
