# SPDX-License-Identifier: Apache-2.0
"""Owned CUDA VAE cache worker; invoked only by the guarded cache parent."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from ..cuda_reference.decode import load_codec
from ..official_cpu.streaming import sha
from ..spatial_reference.guards import atomic, hardware, Monitor, limits
from . import data
from .cache_run import canonical_sha, preflight, read_json, sources


def worker(config):
    out = Path(config['output'])
    if not out.is_dir() or (out / 'worker').exists() or (out / 'result').exists():
        raise ValueError('Require a fresh worker/result within the owned parent output')
    destination = out / 'worker'
    destination.mkdir()
    began = time.monotonic()
    report = {key: config[key] for key in ('profile', 'source_sha256', 'input_plan_sha256', 'cpu_report_sha256')}
    report.update(status='running', mode='cache', model_execution=False, quality_assessed=False,
                  model='External pretrained Wan2.2 original 48-channel VAE', limits=limits('codec'))
    atomic(destination / 'metrics.json', report)
    try:
        if sources() != config['source_sha256'] or preflight(config['cpu_report']) != config['cpu_report_sha256']:
            raise ValueError('Cache source or CPU review changed before worker execution')
        if canonical_sha(data.plan(Path(config['capture']), config['profile'])) != config['input_plan_sha256']:
            raise ValueError('Original capture or preprocessing plan changed')
        report['hardware'] = hardware(config['expected_gpu'])
        atomic(destination / 'metrics.json', report)
        with Monitor(destination, config['deadline'], 'codec') as guard:
            loaded_at = time.monotonic()
            model, scale, provenance = load_codec(Path(config['weights']) / 'Wan2.2_VAE.pth', guard.check)
            report['load_seconds'] = time.monotonic() - loaded_at
            atomic(destination / 'weight-load.json', provenance)
            report['model_execution'] = True
            atomic(destination / 'metrics.json', report)
            try:
                data.encode_cache(Path(config['capture']), config['profile'], out / 'result',
                    lambda video: data.native_encode(model, scale, video, config['profile']),
                    provenance, guard.check)
                report['codec_caches_clear'] = all(v is None for v in model._feat_map + model._enc_feat_map)
                if not report['codec_caches_clear']:
                    raise RuntimeError('Original codec caches remain populated')
            finally:
                model.clear_cache()
                del model, scale
                gc.collect()
                torch.cuda.empty_cache()
            guard.check()
        if sources() != config['source_sha256'] or canonical_sha(data.plan(Path(config['capture']), config['profile'])) != config['input_plan_sha256']:
            raise ValueError('Cache source or original data changed during worker execution')
        if time.monotonic() >= config['deadline']:
            raise RuntimeError('Cache deadline reached during final verification')
        completion = json.loads((out / 'result/completion.json').read_text())
        if completion.get('status') != 'passed' or completion.get('manifest_sha256') != sha(out / 'result/manifest.json'):
            raise ValueError('Cache result did not complete successfully')
        report.update(status='passed', result_completion_sha256=sha(out / 'result/completion.json'),
                      sources_unchanged=True, inputs_unchanged=True)
        return report
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - began
        atomic(destination / 'metrics.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    worker(read_json(parser.parse_args().config))
