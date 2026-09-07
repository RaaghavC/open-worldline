# SPDX-License-Identifier: Apache-2.0
"""Diagnose CPU tiny-codec prefix equality without changing production gates.

Run before the frozen test suite in CI. This records all four oneDNN/thread
configurations and does not choose a backend or treat unequal values as passing.
No external weights, GPU operations, data-cache edits or training occur.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import torch

from .codec import Wan22Codec, make_model, cleanup_temporal_chunks
from .codec_memory import cleanup_causal_convolutions
from .action_data.operations import observation_only

HERE=Path(__file__).resolve().parent
SOURCES=('codec_cpu_diagnostic.py','codec.py','codec-source.json','codec_memory.py','vendor/vae2_2.py',
         'action_data/operations.py','action_data/test_codec_path.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def comparison(left,right):
    if left.shape!=right.shape or left.dtype!=right.dtype:raise ValueError('Matching tensors required')
    different=left!=right;indices=torch.nonzero(different,as_tuple=False)
    absolute=(left.double()-right.double()).abs()
    return {'torch_equal':bool(torch.equal(left,right)),
        'bytes_equal':tensor_sha(left)==tensor_sha(right),
        'left_sha256':tensor_sha(left),'right_sha256':tensor_sha(right),
        'shape':list(left.shape),'dtype':str(left.dtype),'total_elements':left.numel(),
        'different_elements':int(different.sum()),'max_abs':float(absolute.max()),
        'mean_abs':float(absolute.mean()),'both_finite':bool(torch.isfinite(left).all()and torch.isfinite(right).all()),
        'first_differences':[{'index':index.tolist(),'left':float(left[tuple(index)]),
            'right':float(right[tuple(index)]),'abs':float(absolute[tuple(index)])}for index in indices[:8]]}


def one_configuration(*,mkldnn,threads):
    started=time.monotonic();torch.set_num_threads(threads)
    with torch.backends.mkldnn.flags(enabled=mkldnn):
        # Match the frozen fixture's seed, model construction and RGB draw order.
        torch.manual_seed(762)
        codec=Wan22Codec.from_model(make_model(small=True,device='cpu'))
        video=torch.rand(1,3,17,32,32)*2-1
        before=state_sha(codec.model);video_hash=tensor_sha(video)
        with cleanup_temporal_chunks(codec,lambda:None)as chunks,cleanup_causal_convolutions(codec,lambda:None)as layers:
            first=observation_only(codec,video)
            full=codec.encode(video)
            altered_rgb=video.clone();altered_rgb[:,:,1:]=-altered_rgb[:,:,1:]
            perturbed=codec.encode(altered_rgb)
            repeated=observation_only(codec,video)
        after=state_sha(codec.model)
        checks={'full_prefix_vs_standalone':comparison(full[:,:,:1],first),
            'perturbed_future_prefix_vs_standalone':comparison(perturbed[:,:,:1],first),
            'full_prefix_vs_perturbed_future_prefix':comparison(full[:,:,:1],perturbed[:,:,:1]),
            'standalone_A_before_and_after_other_encodes':comparison(first,repeated)}
        return {'status':'completed','requested_mkldnn_enabled':mkldnn,
            'mkldnn_available':torch.backends.mkldnn.is_available(),'recorded_mkldnn_enabled':torch.backends.mkldnn.enabled,
            'requested_threads':threads,'recorded_threads':torch.get_num_threads(),
            'interop_threads':torch.get_num_interop_threads(),'seconds':time.monotonic()-started,
            'fixture_seed':762,'fixture_state_sha256':before,'fixture_state_unchanged':before==after,
            'rgb_sha256':video_hash,'input_rgb_unchanged':video_hash==tensor_sha(video),
            'all_four_comparisons_exact':all(c['torch_equal']and c['bytes_equal']for c in checks.values()),
            'comparisons':checks,'allocator_chunk_counts':chunks,'cleanup_calls':layers['cleanup_calls'],
            'cleanup_hooks_removed':layers['hooks_removed'],'cache_clear':codec.cache_is_clear()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new diagnostic output file')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    before={n:sha(HERE/n)for n in SOURCES};old_threads=torch.get_num_threads();started=time.monotonic()
    result={'schema':'worldline-wan22-cpu-prefix-backend-diagnostic-v1','status':'running',
        'source_sha256':before,'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
        'torch_version':torch.__version__,'torch_config':torch.__config__.show(),
        'initial_num_threads':old_threads,'initial_mkldnn_enabled':torch.backends.mkldnn.enabled,
        'mkldnn_available':torch.backends.mkldnn.is_available(),
        'environment':{n:os.environ.get(n)for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','DNNL_MAX_CPU_ISA','ONEDNN_MAX_CPU_ISA','DNNL_VERBOSE','ONEDNN_VERBOSE')},
        'external_weights_loaded':False,'gpu_operations':False,'production_exact_prefix_changed':False,
        'selects_test_backend':False,'results':[]}
    try:
        for enabled in (True,False):
            for threads in (1,2):
                row=one_configuration(mkldnn=enabled,threads=threads);result['results'].append(row)
                print(json.dumps({'mkldnn_enabled':enabled,'threads':threads,
                    'max_abs':row['comparisons']['full_prefix_vs_standalone']['max_abs'],
                    'different_elements':row['comparisons']['full_prefix_vs_standalone']['different_elements'],
                    'all_four_comparisons_exact':row['all_four_comparisons_exact']}),flush=True)
        result['source_unchanged']=before=={n:sha(HERE/n)for n in SOURCES}
        result['status']='completed'if result['source_unchanged']else'source_changed'
    except BaseException as problem:
        result.update(status='failed',error_type=type(problem).__name__,error=str(problem));raise
    finally:
        torch.set_num_threads(old_threads);result['seconds']=time.monotonic()-started
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
