"""Bounded frozen-Wan adapter runtime probe using synthetic tensors only.

No quality, prompt following, persistent memory, or streaming claim follows
from this test. The VAE and text encoder are not loaded. Shape equivalence is
17 decoded frames at 512x288, represented by 5x36x64 latent grids.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import threading
import time
import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from adapter import ActionObservationAdapter
from compat import install
from fetch_weights import sha, REVISION, FILES
from vendor.wan21.model import Wan21Model

GiB = 1024 ** 3
EXPECTED = '96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f'


def model_digest(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, data):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'mps'], required=True)
    p.add_argument('--synthetic-runtime-only', action='store_true', required=True)
    p.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--max-seconds', type=int, default=600)
    p.add_argument('--max-memory-gib', type=float, default=18.)
    p.add_argument('--seed', type=int, default=20260907)
    a = p.parse_args()
    if not 1 <= a.steps <= 10 or not 1 <= a.max_seconds <= 600 or not 1 <= a.max_memory_gib <= 18:
        p.error('Runtime probe limits: 1-10 updates, at most 600 seconds and 18 GiB')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output directory must be empty; this experiment does not resume or overwrite evidence')
    a.output.mkdir(parents=True, exist_ok=True)
    install()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    dtype = getattr(torch, a.dtype)
    if a.device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable')
        torch.mps.set_per_process_memory_fraction(min(1., a.max_memory_gib * GiB / torch.mps.recommended_max_memory()))
    started = time.perf_counter()
    report = {'status': 'running', 'purpose': 'synthetic runtime and adapter-gradient probe only',
        'quality_evidence': False, 'causal_streaming': False, 'persistent_memory': False,
        'observation': 'first latent frame only, pooled to 32 conditioning tokens',
        'pose_conditioning': False, 'text_embeddings': 'synthetic normal tensors, not UMT5 outputs',
        'latent_input': 'synthetic normal tensors, not encoded video',
        'weights_repository': 'Wan-AI/Wan2.1-T2V-1.3B', 'weights_revision': REVISION,
        'weights_source_sha256': EXPECTED, 'source_dtype': 'float32', 'compute_dtype': a.dtype,
        'adapter_dtype': 'float32', 'device': a.device, 'torch': torch.__version__,
        'platform': platform.platform(), 'mps_automatic_cpu_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0'), 'seed': a.seed, 'pixel_equivalent_shape': [1, 17, 288, 512],
        'latent_shape': [1, 16, 5, 36, 64], 'patch_tokens': 2880, 'text_tokens': 512,
        'action_shape': [1, 16, 6], 'action_order': 'four ordered intervals per later latent frame',
        'checkpointing': 'per Wan block, non-reentrant', 'adapter_blocks': [9, 19, 29],
        'max_seconds': a.max_seconds, 'max_memory_gib': a.max_memory_gib, 'steps': []}
    metrics_path = a.output / 'metrics.json'
    write_json(metrics_path, report)
    stopped = threading.Event()
    process = psutil.Process()
    peak = {'rss_bytes': 0, 'mps_allocated_bytes': 0, 'mps_driver_bytes': 0}
    stage = ['load']
    sample_lock = threading.Lock()

    def sample():
        row = {'seconds': time.perf_counter() - started, 'stage': stage[0],
               'rss_bytes': process.memory_info().rss, 'available_system_bytes': psutil.virtual_memory().available}
        if a.device == 'mps':
            row['mps_allocated_bytes'] = torch.mps.current_allocated_memory()
            row['mps_driver_bytes'] = torch.mps.driver_allocated_memory()
        for key in peak:
            peak[key] = max(peak[key], row.get(key, 0))
        return row

    def monitor():
        with (a.output / 'memory.jsonl').open('w') as log:
            while not stopped.is_set():
                row = sample()
                log.write(json.dumps(row) + '\n'); log.flush()
                reason = None
                if row['seconds'] > a.max_seconds:
                    reason = '600-second-or-requested-deadline'
                elif max(row['rss_bytes'], row.get('mps_driver_bytes', 0)) > a.max_memory_gib * GiB:
                    reason = 'memory-cap'
                elif row['available_system_bytes'] < 2 * GiB:
                    reason = 'system-memory-below-2-GiB'
                if reason:
                    write_json(a.output / 'watchdog-stop.json', {'status': 'stopped', 'reason': reason, 'last_sample': row})
                    os._exit(124)
                stopped.wait(.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    def synchronize():
        if a.device == 'mps':
            torch.mps.synchronize()

    try:
        checkpoint = a.weights / 'diffusion_pytorch_model.safetensors'
        if checkpoint.stat().st_size != 5676070424 or sha(checkpoint) != EXPECTED:
            raise RuntimeError('Pinned original transformer SHA256 mismatch')
        config_path = a.weights / 'config.json'
        config_size, config_sha = FILES['config.json']
        if config_path.stat().st_size != config_size or sha(config_path) != config_sha:
            raise RuntimeError('Pinned model config SHA256 mismatch')
        report['config_sha256'] = config_sha
        config = json.loads(config_path.read_text())
        config = {k: v for k, v in config.items() if not k.startswith('_')}
        with torch.device('meta'):
            core = Wan21Model(**config)
        state = {}
        with safe_open(checkpoint, framework='pt', device='cpu') as f:
            for name in f.keys():
                state[name] = f.get_tensor(name).to(dtype)
        result = core.load_state_dict(state, strict=True, assign=True)
        del state
        core.requires_grad_(False)
        core.eval()
        report['loaded_keys'] = len(core.state_dict())
        report['missing_keys'] = result.missing_keys
        report['unexpected_keys'] = result.unexpected_keys
        report['base_parameters'] = sum(p.numel() for p in core.parameters())
        report['converted_base_sha256_before'] = model_digest(core)
        core = core.to(device)
        core.gradient_checkpointing = True
        adapter = ActionObservationAdapter(core.dim).to(device)
        report['adapter_parameters'] = sum(p.numel() for p in adapter.parameters())
        gc.collect()
        synchronize()
        report['load_seconds'] = time.perf_counter() - started
        write_json(metrics_path, report)
        clean = torch.randn(1, 16, 5, 36, 64).to(device, dtype)
        noise = torch.randn_like(clean)
        actions = torch.randn(1, 16, 6).to(device)
        observed = clean[:, :, 0].detach()
        context = [torch.randn(512, 4096).mul_(.02).to(device, dtype)]
        sigma = .5
        input_latent = (1 - sigma) * clean + sigma * noise
        target = (noise - clean).float()
        t = torch.tensor([500.], device=device, dtype=torch.float32)
        adapter.attach(core, actions, observed, (5, 18, 32))
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
        initial_adapter_hash = model_digest(adapter)
        for index in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            stage[0] = f'update-{index}-forward'
            synchronize(); step_start = time.perf_counter()
            output = core(list(input_latent.unbind(0)), t, context, 2880)
            synchronize(); forward_end = time.perf_counter()
            if not torch.isfinite(output).all().item():
                raise RuntimeError('Non-finite core output; no optimizer update performed')
            loss = torch.nn.functional.mse_loss(output.float(), target)
            stage[0] = f'update-{index}-backward'
            loss.backward()
            synchronize(); backward_end = time.perf_counter()
            grads = [p.grad for p in adapter.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise RuntimeError('Adapter gradients absent or non-finite; optimizer update rejected')
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.).item()
            if not math.isfinite(grad_norm) or not grad_norm > 0:
                raise RuntimeError('Adapter gradient norm must be finite and positive')
            if any(p.grad is not None for p in core.parameters()):
                raise RuntimeError('Frozen core unexpectedly received parameter gradients')
            stage[0] = f'update-{index}-optimizer'
            optimizer.step()
            if not all(torch.isfinite(p).all().item() for p in adapter.parameters()):
                raise RuntimeError('Non-finite adapter parameter after optimizer update')
            synchronize(); step_end = time.perf_counter()
            record = {'index': index, 'forward_seconds': forward_end-step_start,
                      'backward_seconds': backward_end-forward_end,
                      'optimizer_and_gradient_validation_seconds': step_end-backward_end,
                      'total_seconds': step_end-step_start, 'synthetic_mse': loss.item(),
                      'adapter_gradient_norm_before_clipping': grad_norm,
                      'adapter_gradient_tensors': len(grads)}
            report['steps'].append(record)
            report['peaks_sampled'] = dict(peak)
            write_json(metrics_path, report)
            print(json.dumps(record), flush=True)
        adapter.detach()
        stage[0] = 'verify-and-save'
        report['converted_base_sha256_after'] = model_digest(core)
        if report['converted_base_sha256_before'] != report['converted_base_sha256_after']:
            raise RuntimeError('Frozen base changed')
        report['base_parameters_unchanged'] = True
        report['initial_adapter_sha256'] = initial_adapter_hash
        report['updated_adapter_sha256'] = model_digest(adapter)
        save_file({k: v.detach().cpu().contiguous() for k, v in adapter.state_dict().items()}, str(a.output / 'runtime-only-adapter.safetensors'))
        report['adapter_checkpoint_sha256'] = sha(a.output / 'runtime-only-adapter.safetensors')
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        report['error'] = str(error)
        raise
    finally:
        stopped.set(); monitor_thread.join(timeout=2)
        report['peaks_sampled'] = dict(peak)
        report['total_seconds'] = time.perf_counter()-started
        write_json(metrics_path, report)

if __name__ == '__main__':
    main()
