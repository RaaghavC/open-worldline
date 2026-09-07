# SPDX-License-Identifier: Apache-2.0
"""Compare cleanup hooks with official CPU decode on saved small latents."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import psutil
from safetensors import safe_open
import torch
from codec.helper import OfficialWanCodec, sha, VAE_SHA256
from codec.decode_policy import cleanup_after_temporal_chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--latents', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output file must be new')
    started = time.perf_counter()
    stop = threading.Event()
    def watchdog():
        while not stop.wait(.5):
            if time.perf_counter()-started > 600 or psutil.virtual_memory().available < 2*1024**3 or max(psutil.Process().memory_info().rss, torch.mps.driver_allocated_memory()) > 18*1024**3:
                args.output.write_text(json.dumps({'status': 'guard-stopped', 'seconds': time.perf_counter()-started})+'\n')
                os._exit(124)
    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    torch.set_num_threads(4)
    report = {'status': 'running', 'purpose': 'Small spatial crop validates allocator hook and FP32 MPS against unmodified CPU VAE',
              'weights_sha256': VAE_SHA256, 'latents_sha256': sha(args.latents),
              'new_latent_generation': False, 'crop': 'base[:, :, :, :4, :4]',
              'source_sha256': {path.name: sha(path) for path in [Path(__file__), Path(__file__).with_name('decode_policy.py'), Path(__file__).with_name('helper.py')]}}
    try:
        with safe_open(args.latents, framework='pt', device='cpu') as f:
            latent = f.get_tensor('base')[:, :, :, :4, :4].contiguous()
        codec = OfficialWanCodec(args.weights, 'cpu', torch.float32)
        reference = codec.decode(latent)
        with cleanup_after_temporal_chunk(codec) as cpu_hook:
            hooked = codec.decode(latent)
        report.update(input_shape=list(latent.shape), output_shape=list(reference.shape),
                      cpu_hook_max_abs=float((reference-hooked).abs().max()), cpu_hook=cpu_hook)
        if not torch.equal(reference, hooked):
            raise RuntimeError('Hook changed CPU decoder output')
        codec.model.to('mps')
        codec.device = torch.device('mps')
        codec.scale = [v.to('mps') for v in codec.scale]
        torch.mps.empty_cache()
        with cleanup_after_temporal_chunk(codec) as mps_hook:
            mps = codec.decode(latent).cpu()
        error = (reference-mps).abs()
        report.update(mps_cpu_max_abs=float(error.max()), mps_cpu_mean_abs=float(error.mean()),
                      mps_hook=mps_hook, mps_cpu_tolerance={'atol': .002, 'rtol': .002},
                      finite=bool(torch.isfinite(mps).all()), hooks_remaining=len(codec.model.decoder._forward_hooks))
        if not torch.isfinite(mps).all() or not torch.allclose(reference, mps, atol=.002, rtol=.002):
            raise RuntimeError('FP32 MPS decoder differs from CPU beyond declared tolerance')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        report['seconds'] = time.perf_counter()-started
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
