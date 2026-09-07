# SPDX-License-Identifier: Apache-2.0
"""Fixed development-pilot inputs, checkpoints and tokenwise sampling boundary.

Original Worldline experiment code. External Wan modules retain their separate
attribution. This module performs no model loading, download or GPU operation.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from adapter import ActionObservationAdapter
from fetch_weights import sha, FILES
from native_control.clamp_cache import read_observation, MANIFEST_SHA256
from native_control.clamp_loop import integrate_clamped
from tokenwise_time.portable import token_times
from tokenwise_time.profile_update import module_sha, validate_preflight
from tokenwise_time.training import sample_training_inputs, training_pair, predict, future_loss
from tokenwise_time.training_data import tensor_sha

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
WINDOWS=tuple(f'{arm}-{start:04d}' for start in (0,8,32,49) for arm in ('closed','open'))
SCHEDULE=WINDOWS*2
SEED=20260912
SAMPLE_WINDOW='open-0000'
POSITIVE_TEXT='A sunlit interior with warm plaster walls, a wooden door, limestone flooring, brass details, and green plants.'
PILOT_NAMES=['tokenwise_time/'+n for n in ('pilot_common.py','train_pilot.py','sample_pilot.py','test_pilot.py')]
SHARED_NAMES=['tokenwise_time/'+n for n in ('portable.py','training.py','training_data.py','profile_update.py',
    'test_parity.py','test_independent.py','test_training.py','test_training_independent.py')]+[
    'adapter.py','fetch_weights.py','PROTOCOL.md','requirements-real.txt','requirements.txt','text_cache/cache.py',
    'text_cache/prompts.json','native_control/portable.py','native_control/profile_pair.py','native_control/evidence.py',
    'native_control/sampling.py','native_control/clamp_loop.py','native_control/clamp_cache.py','native_control/loop.py',
    'native_control/vendor/model.py','native_control/vendor/attention.py','native_control/vendor/fm_solvers_unipc.py',
    'native_control/vendor/shared_config.py.txt','native_control/vendor/source-manifest.json',
    'native_control/vendor/LICENSE-APACHE-2.0.txt','codec/helper.py','codec/decode_policy.py',
    'codec/vendor/wan_vae.py','codec/vendor/source-manifest.json','codec/requirements.txt']


def reject_output(path):
    path=Path(path)
    if path.exists() or path.is_symlink() or path.resolve().is_relative_to(PARENT):
        raise ValueError('Output must be a new directory outside the Wan experiment source tree')


def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def atomic_tensors(path,values):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    tensors={k:v.detach().cpu().contiguous() for k,v in values.items()}
    if not all(torch.isfinite(v).all().item() for v in tensors.values()):raise ValueError('Non-finite saved tensor')
    save_file(tensors,str(temp));temp.replace(path)
    return sha(path)


def snapshot_sources(output):
    names=SHARED_NAMES+PILOT_NAMES+['tokenwise_time/test_pilot_independent.py']
    result={}
    for name in names:
        source=PARENT/name;dest=Path(output)/'measured-source'/(name+'.txt')
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest);result[name]=sha(dest)
        if sha(source)!=result[name]:raise RuntimeError('Source changed while snapshotting')
    return result


def check_pilot_report(path,*,independent=False):
    r=json.loads(Path(path).read_text())
    if r.get('status')!='passed' or r.get('tests',0)<5:raise ValueError('Passed pilot CPU report with at least five tests required')
    required=PILOT_NAMES+SHARED_NAMES
    if independent:required+=['tokenwise_time/test_pilot_independent.py']
    for name in required:
        if r.get('source_sha256',{}).get(name)!=sha(PARENT/name):raise ValueError('Pilot source changed since CPU report: '+name)
    return sha(path)


def prepare_inputs(windows,seed=SEED):
    """Save all 16 training draws and eight separate fixed diagnostic draws."""
    if tuple(windows)!=WINDOWS:raise ValueError('Exactly eight windows in the predefined order are required')
    tensors={};records=[]
    for kind,order,rng_seed in [('train',SCHEDULE,seed+1),('diagnostic',WINDOWS,seed+2)]:
        rng=torch.Generator(device='cpu').manual_seed(rng_seed)
        for index,name in enumerate(order):
            k,noise=sample_training_inputs(windows[name]['target'],rng)
            prefix=f'{kind}.{index:02d}'
            tensors[prefix+'.noise']=noise;tensors[prefix+'.k']=k
            records.append({'kind':kind,'index':index,'window':name,'prefix':prefix,'k':k.tolist(),
                'sigma_rational':[str(int(v))+'/1000' for v in k], 'noise_sha256':tensor_sha(noise)})
    return tensors,records


def example(window,inputs,prefix,device='cpu'):
    noisy,velocity,times=training_pair(window['target'],inputs[prefix+'.noise'],window['observation'],inputs[prefix+'.k'])
    return tuple(v.to(device) for v in (noisy,velocity,times,window['observation'],window['actions']))


@torch.inference_mode()
def diagnostic(core,adapter,window,inputs,prefix,context,device='cpu'):
    noisy,velocity,times,observation,actions=example(window,inputs,prefix,device)
    prediction=predict(core,adapter,noisy,times,observation,actions,[context])
    if not torch.isfinite(prediction).all():raise RuntimeError('Non-finite fixed diagnostic prediction')
    mse=float(future_loss(prediction,velocity))
    if not math.isfinite(mse):raise RuntimeError('Non-finite diagnostic MSE')
    return {'future_flow_mse':mse,'prediction_sha256':tensor_sha(prediction),
        'noisy_latent_sha256':tensor_sha(noisy),'target_velocity_sha256':tensor_sha(velocity),
        'token_times_sha256':tensor_sha(times),'scope':'Fixed-noise development FM diagnostic, not generated-image accuracy'},prediction.cpu()


def save_recovery(directory,adapter,optimizer,completed):
    """One atomic bundle: last fully checked update, optimizer and RNG state."""
    if type(completed)!=int or not 0<=completed<=16:raise ValueError('Invalid completed-update count')
    state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Recovery adapter is not finite')
    def cpu(value):
        if isinstance(value,torch.Tensor):
            value=value.detach().cpu().clone()
            if not torch.isfinite(value).all():raise ValueError('Recovery optimizer contains non-finite state')
            return value
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return type(value)(cpu(v) for v in value)
        return value
    bundle={'schema':'worldline-tokenwise-pilot-recovery-v1','completed_updates':completed,
        'schedule_completed':list(SCHEDULE[:completed]),'adapter':state,'optimizer':cpu(optimizer.state_dict()),
        'cpu_rng_state':torch.random.get_rng_state(),'adapter_seed':SEED,
        'limit':'Recovery evidence only; no automatic resume is implemented'}
    path=Path(directory)/'recovery-last.pt';temp=path.with_name(path.name+'.tmp')
    torch.save(bundle,temp);temp.replace(path)
    return sha(path)


def read_conditions(directory):
    """Materialize original initial observation and actual commands, never target."""
    observation,provenance=read_observation(directory)
    root=Path(directory).resolve();m=json.loads((root/'manifest.json').read_text())
    row=next(r for r in m['windows'] if r['id']==SAMPLE_WINDOW)
    path=(root/row['file']).resolve()
    if not path.is_relative_to(root) or sha(path)!=row['sha256']:raise ValueError('Action cache file integrity changed')
    with safe_open(path,framework='pt',device='cpu') as f:actions=f.get_tensor('actions')
    meta=row['tensors']['actions']
    if actions.shape!=(1,16,6) or actions.dtype!=torch.float32 or not torch.isfinite(actions).all():raise ValueError('Invalid actions')
    if meta['sha256']!=tensor_sha(actions) or meta['shape']!=list(actions.shape) or meta['dtype']!=str(actions.dtype):raise ValueError('Action metadata mismatch')
    provenance.update(materialized_tensor_keys=['observation','actions'],actions_materialized=True,
                      action_tensor_sha256=tensor_sha(actions),action_source=row['source'])
    return observation,actions,provenance


class TokenwiseConditionedModel:
    """Map scalar native solver times to exact prefix-zero/future-time fields."""
    def __init__(self,core,adapter,observation,actions):
        self.core,self.adapter,self.observation,self.actions=core,adapter,observation,actions
        self.calls=0;self.prefix_differences=[]
    def __call__(self,values,timestep,context,seq_len):
        if len(values)!=1 or seq_len!=2880:raise ValueError('Only the prescribed single clip is supported')
        value=values[0]
        if not torch.equal(value[:,:1],self.observation):raise ValueError('Every CFG input must have the exact observed prefix')
        times=token_times(torch.tensor([[5,18,32]],dtype=torch.int64),timestep,seq_len)
        result=predict(self.core,self.adapter,value.unsqueeze(0),times,self.observation.unsqueeze(0),self.actions,context)
        self.calls+=1;self.prefix_differences.append(float((value[:,:1]-self.observation).abs().max()))
        return list(result.unbind(0))


@torch.inference_mode()
def sample_latents(core,adapter,noise,observation,actions,negative,positive,*,device='mps',callback=None):
    """Same official CPU UniPC50/shift8/CFG6, with explicit tokenwise times."""
    wrapped=TokenwiseConditionedModel(core,adapter,observation.to(device),actions.to(device))
    latent=integrate_clamped(wrapped,noise,observation,negative,positive,device=device,callback=callback)
    if wrapped.calls!=100 or any(wrapped.prefix_differences):raise RuntimeError('Incomplete or unclamped denoiser inputs')
    if adapter._hooks:raise RuntimeError('Sampling retained adapter hooks')
    return latent,{'model_calls':wrapped.calls,'denoiser_prefix_max_abs_differences':wrapped.prefix_differences}


def validate_training_run(directory):
    directory=Path(directory);r=json.loads((directory/'metrics.json').read_text())
    if r.get('status')!='passed' or r.get('requested_updates')!=16 or r.get('completed_updates')!=16 or r.get('schedule')!=list(SCHEDULE):raise ValueError('Completed exact 16-update pilot required')
    if r.get('adapter_seed')!=SEED or r.get('base_parameters_unchanged') is not True:raise ValueError('Wrong initialization or changed base')
    base_before=r.get('base_tensor_sha256_before')
    if not isinstance(base_before,str) or len(base_before)!=64 or base_before!=r.get('base_tensor_sha256_after'):
        raise ValueError('Frozen base before/after tensor hashes must match')
    if r.get('actual_zero_init_equal') is not True or r.get('hooks_removed') is not True:raise ValueError('Missing adapter integration checks')
    rows=r.get('updates',[])
    if len(rows)!=16:raise ValueError('Exactly 16 completed optimizer records required')
    for index,(row,window) in enumerate(zip(rows,SCHEDULE)):
        if row.get('update')!=index+1 or row.get('window')!=window or row.get('gradient_tensors')!=42:
            raise ValueError('Optimizer order or gradient coverage differs')
        loss,norm=row.get('future_flow_mse'),row.get('gradient_norm_before_clip')
        if type(loss) not in (float,int) or not math.isfinite(loss) or loss<0:
            raise ValueError('Retained update loss must be finite and nonnegative')
        if type(norm) not in (float,int) or not math.isfinite(norm) or norm<=0:
            raise ValueError('Retained update gradient norm must be finite and positive')
        norms=row.get('residual_gradient_norms',[])
        if len(norms)!=3 or any(not math.isfinite(v) or v<=0 for v in norms):raise ValueError('Invalid residual gradient evidence')
    diagnostics=r.get('diagnostics',{})
    for stage in ('before','after'):
        if [row.get('window') for row in diagnostics.get(stage,[])]!=list(WINDOWS):raise ValueError('Incomplete fixed development diagnostic')
    for before,after in zip(diagnostics['before'],diagnostics['after']):
        for key in ('noisy_latent_sha256','target_velocity_sha256','token_times_sha256'):
            if not before.get(key) or before[key]!=after.get(key):raise ValueError('Development diagnostic inputs changed')
    for name in SHARED_NAMES+['tokenwise_time/pilot_common.py','tokenwise_time/train_pilot.py']:
        if r.get('source_sha256',{}).get(name)!=sha(PARENT/name):raise ValueError('Training source changed: '+name)
    for name in ('initial-adapter.safetensors','final-adapter.safetensors','fixed-inputs.safetensors'):
        if r.get('output_sha256',{}).get(name)!=sha(directory/name):raise ValueError('Training artifact integrity changed: '+name)
    return r
