# SPDX-License-Identifier: Apache-2.0
"""Six explicit block-28 calls to the byte-exact reviewed sampling kernel."""
from pathlib import Path
import json
import time
import torch
from safetensors.torch import save_file
import sampler
from experiments.wan22_native.intermediate_action.bridge import IntermediateActionBridge
from experiments.wan22_native.spatial_reference.guards import atomic

ARMS=tuple(m+'_'+d for m in ('stationary','left','right') for d in ('closed','interact'))

def adapter_identity(adapter):
    return {n:sampler.tensor_sha(v) for n,v in adapter.state_dict().items()}

def load_bridge(core,checkpoint,digest):
    if torch.is_inference_mode_enabled():raise ValueError('Bridge requires normal constants, not inference tensors')
    if core.patch_embedding.weight.device.type!='cuda':raise ValueError('Verified CUDA foundation required')
    adapter,identity=sampler.load_adapter_checkpoint(checkpoint,digest)
    adapter.to(core.patch_embedding.weight.device)
    if adapter_identity(adapter)!=identity['tensor_sha256']:raise ValueError('Adapter CUDA copy changed')
    return IntermediateActionBridge(core,adapter,block_index=28,profile='spatial'),identity

def require_unchanged(bridge,identity):
    if (adapter_identity(bridge.adapter)!=identity['tensor_sha256']
            or any(p.grad is not None for p in bridge.adapter.parameters())
            or any(p.grad is not None or p.requires_grad for p in bridge.core.parameters())):
        raise RuntimeError('Sampling changed adapter values or foundation/adapter gradients')

def sample_six(bridge,identity,out,values,contexts,commands,*,check=lambda:None,progress=lambda row:None):
    """The caller owns one loaded bridge; each arm gets a fresh native solver."""
    if bridge.block_index!=28 or set(commands)!=set(ARMS):raise ValueError('Only six declared block-28 arms')
    initial={k:sampler.tensor_sha(v) for k,v in values.items()}
    texts={k:sampler.tensor_sha(v) for k,v in contexts.items()};rows={}
    require_unchanged(bridge,identity)
    for arm in ARMS:
        check();folder=Path(out)/arm;folder.mkdir(exist_ok=False);began=time.monotonic();count=0
        with (folder/'steps.jsonl').open('x') as stream:
            def event(index,t,latent,velocities):
                nonlocal count
                if index!=count or not torch.equal(latent[:,:1],values['observation'][0]):raise ValueError('Step order or prefix changed')
                save_file({'latent':latent},str(folder/f'step-{index+1:02d}.safetensors'))
                if index==0:save_file(velocities,str(folder/'initial-velocities.safetensors'))
                stream.write(json.dumps(dict(step=index+1,timestep=int(t),prefix_exact=True,
                    latent_sha256=sampler.tensor_sha(latent),seconds=time.monotonic()-began))+'\n');stream.flush()
                count+=1;check();progress(dict(arm=arm,completed_steps=count,completed_arms=list(rows)))
            latent,row=sampler._sample_bridge(bridge,values,contexts,commands[arm],'spatial',event=event,check=check)
        if count!=50 or row['predictions']!=100 or row['solver_updates']!=50:raise RuntimeError('Incomplete six-arm solver')
        require_unchanged(bridge,identity)
        save_file({'latent':latent},str(folder/'latents.safetensors'))
        row.update(adapter=identity,block_index=28,negative_context_adapter_training=True,
            negative_context_training_scope='Every fourth factorial paired update uses both CFG contexts; main FM uses positive text only',
            foundation_values_verified_by_this_kernel=False,training=False,arm=arm,
            commands_label=arm,seconds=time.monotonic()-began)
        atomic(folder/'metrics.json',row);rows[arm]=row;progress(dict(arm=arm,completed_steps=50,completed_arms=list(rows)))
        if initial!={k:sampler.tensor_sha(v) for k,v in values.items()} or texts!={k:sampler.tensor_sha(v) for k,v in contexts.items()}:
            raise RuntimeError('Shared fixed inputs changed between arms')
    return rows
