# SPDX-License-Identifier: Apache-2.0
"""Separate read-only tracing of the tiny CPU codec's final 1x1x1 projection.

No frozen source or tensor is edited. Hooks copy intermediate values and return
None. Their output neutrality is checked against an unhooked full encode.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from .codec import cleanup_temporal_chunks
from .codec_memory import cleanup_causal_convolutions
from .action_data.test_codec_path import fixture
from .codec_cpu_diagnostic import comparison, state_sha

HERE=Path(__file__).resolve().parent
SOURCES=('codec_cpu_projection_diagnostic.py','codec_cpu_diagnostic.py','codec.py','codec-source.json',
         'codec_memory.py','vendor/vae2_2.py','action_data/operations.py','action_data/test_codec_path.py')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def traced_encode(codec,video):
    observed={};handles=[]
    def encoder_after(module,args,output):
        if 'first_encoder_output'not in observed:observed['first_encoder_output']=output.detach().clone()
    def before(module,args):
        if 'projection_input'in observed:raise RuntimeError('Unexpected repeated final projection')
        observed['projection_input']=args[0].detach().clone()
        observed['projection_input_original_stride']=list(args[0].stride())
        observed['projection_input_original_contiguous']=args[0].is_contiguous()
    def after(module,args,output):observed['projection_output']=output.detach().clone()
    try:
        handles.append(codec.model.encoder.register_forward_hook(encoder_after))
        handles.append(codec.model.conv1.register_forward_pre_hook(before))
        handles.append(codec.model.conv1.register_forward_hook(after))
        latent=codec.encode(video)
    finally:
        for handle in handles:handle.remove()
    observed['latent']=latent
    return observed


def configuration(enabled,threads):
    torch.set_num_threads(threads);started=time.monotonic()
    with torch.backends.mkldnn.flags(enabled=enabled):
        codec,video=fixture();before_state=state_sha(codec.model)
        with cleanup_temporal_chunks(codec,lambda:None),cleanup_causal_convolutions(codec,lambda:None):
            ordinary=codec.encode(video)
            one=traced_encode(codec,video[:,:,:1].clone())
            full=traced_encode(codec,video)
        projection=codec.model.conv1
        if tuple(projection.kernel_size)!=(1,1,1)or tuple(projection.stride)!=(1,1,1):
            raise ValueError('Expected the literal native final pointwise projection')
        # Replay the exact saved projection input, changing only temporal length
        # or later temporal values. These are diagnostic calls, not substitutions.
        long=full['projection_input'].clone();changed=long.clone();changed[:,:,1:]=-changed[:,:,1:]
        with torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):
            short_output=projection(long[:,:,:1].clone())
            long_output=projection(long)
            changed_output=projection(changed)
        checks={'hooked_full_encode_vs_unhooked_full_encode':comparison(full['latent'],ordinary),
            'first_encoder_output_full_vs_standalone':comparison(full['first_encoder_output'],one['first_encoder_output']),
            'projection_input_prefix_full_vs_standalone':comparison(full['projection_input'][:,:,:1],one['projection_input']),
            'projection_output_prefix_full_vs_standalone':comparison(full['projection_output'][:,:,:1],one['projection_output']),
            'normalized_latent_prefix_full_vs_standalone':comparison(full['latent'][:,:,:1],one['latent']),
            'replayed_identical_input_short_vs_long_projection':comparison(short_output,long_output[:,:,:1]),
            'same_length_projection_future_perturbation':comparison(long_output[:,:,:1],changed_output[:,:,:1])}
        hooks_clear=all(not m._forward_hooks and not m._forward_pre_hooks for m in codec.model.modules())
        neutral=checks['hooked_full_encode_vs_unhooked_full_encode']['bytes_equal']
        if not neutral or not hooks_clear:location='Trace neutrality or hook cleanup failed; no localization claim'
        elif not checks['projection_input_prefix_full_vs_standalone']['bytes_equal']:
            location='Difference already present before the final projection'
        elif not checks['projection_output_prefix_full_vs_standalone']['bytes_equal']:
            location='First compared discrepancy appears at the final 1x1x1 projection'
        elif not checks['normalized_latent_prefix_full_vs_standalone']['bytes_equal']:
            location='Compared projection output is exact; discrepancy appears during normalization'
        else:location='No cross-length discrepancy reproduced on this configuration'
        return {'status':'completed','mkldnn_enabled':enabled,'mkldnn_available':torch.backends.mkldnn.is_available(),
            'threads':threads,'seconds':time.monotonic()-started,'fixture_state_sha256':before_state,
            'fixture_state_unchanged':before_state==state_sha(codec.model),'hook_neutrality_passed':neutral,
            'hooks_removed':hooks_clear,'cache_clear':codec.cache_is_clear(),'observed_location':location,
            'projection_kernel_size':list(projection.kernel_size),'projection_stride':list(projection.stride),
            'standalone_projection_input_shape':list(one['projection_input'].shape),
            'full_projection_input_shape':list(full['projection_input'].shape),
            'standalone_projection_original_stride':one['projection_input_original_stride'],
            'full_projection_original_stride':full['projection_input_original_stride'],
            'checks':checks,'future_dependency_scope':'Only the recorded same-length projection perturbation; no universal proof from one fixture'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('New diagnostic output required')
    a.output.parent.mkdir(parents=True,exist_ok=True);old_threads=torch.get_num_threads();before={n:sha(HERE/n)for n in SOURCES}
    result={'schema':'worldline-wan22-cpu-projection-diagnostic-v2','status':'running','source_sha256':before,
        'platform':platform.platform(),'torch_version':torch.__version__,'torch_config':torch.__config__.show(),
        'external_weights_loaded':False,'gpu_operations':False,'production_source_or_values_changed':False,'results':[]}
    try:
        for enabled in (True,False):
            for threads in (1,2):
                row=configuration(enabled,threads);result['results'].append(row)
                print(json.dumps({'mkldnn_enabled':enabled,'threads':threads,'observed_location':row['observed_location'],
                    'hook_neutrality_passed':row['hook_neutrality_passed'],
                    'projection_future_perturbation_exact':row['checks']['same_length_projection_future_perturbation']['bytes_equal']}),flush=True)
        result['source_unchanged']=before=={n:sha(HERE/n)for n in SOURCES};result['status']='completed'if result['source_unchanged']else'source_changed'
    except BaseException as problem:result.update(status='failed',error_type=type(problem).__name__,error=str(problem));raise
    finally:
        torch.set_num_threads(old_threads);a.output.write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
