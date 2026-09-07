# SPDX-License-Identifier: Apache-2.0
"""Plan-only by default. A separate guarded child can run an admitted pilot."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from experiments.room_world.memory_train import atomic_write, new_directory, handoff_and_guard
from ..action_adapter.wrapper import NativeActionWrapper
from ..action_data.cache import load_training_window
from ..codec_profile import Guard
from ..load_weights import load_core, sha256, tensor_sha256
from ..profile_core import runtime_environment, RUNTIME_VARIABLES
from . import runtime as gate
from .objective import (SEED, SHAPE, OPTIMIZER, fresh_adapter, make_draws, paired_update,
                       parameter_records, loaded_records, save_checkpoint)

ENVIRONMENT={name:('0' if name=='PYTORCH_ENABLE_MPS_FALLBACK' else None) for name in RUNTIME_VARIABLES}


def admitted(config):
    inputs,positive=gate.validate_inputs(config['capture'],config['cache'],config['roundtrip_run'],config['text_cache'])
    evidence=dict(cpu_report_sha256=gate.validate_cpu(config['cpu_report']),
        independent_report_sha256=gate.validate_cpu(config['independent_report'],independent=True),
        visual_admission=gate.validate_admission(config['admission'],config['mode'],inputs))
    if config['mode']=='fixed16':evidence['probe']=gate.validate_probe(config['probe_run'],inputs)
    return inputs,positive,evidence


def verify_saved_draws(run,mode,expected):
    rows,tensors=make_draws(mode)
    if rows!=expected:raise ValueError('Schedule differs from fresh prescribed draw sequence')
    path=Path(run)/'draws.safetensors'
    with safe_open(path,framework='pt',device='cpu')as handle:
        if set(handle.keys())!=set(tensors):raise ValueError('Saved noise/RNG keys differ')
        for name,wanted in tensors.items():
            if not torch.equal(handle.get_tensor(name),wanted):raise ValueError('Saved draw differs: '+name)
    return rows,tensors


def worker(config):
    if config.get('device')!='mps':raise ValueError('Actual training has only an explicit MPS path; CPU is fixture-only')
    if {n:os.environ.get(n)for n in ENVIRONMENT}!=ENVIRONMENT:
        raise RuntimeError('Exact declared core environment required; no automatic CPU fallback')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(2)
    out=Path(config['output']);guard=Guard(out,'mps');report=guard.report;error=None;completed=False;model=None;before=None
    report.update(schema='worldline-wan22-action-training-run-v1',protocol=gate.protocol(config['mode']),
        source_sha256=gate.snapshot(out),device='mps',adapter_parameters=947712,adapter_dtype='float32',
        external_frozen_core=True,original_trainable_adapter=True,generated_images=0,quality_evaluation=False,
        completed_updates=0,updates=[],base_unchanged=None,second_update_gru_gradient_nonzero=False,
        runtime_environment=runtime_environment('mps'),dependencies={n:importlib.metadata.version(n)for n in ('torch','numpy','safetensors','psutil','diffusers')})
    try:
        def check():
            if time.monotonic()>=config['deadline']:raise TimeoutError('900-second child deadline reached')
        if config['source_sha256']!=gate.source_hashes():raise ValueError('Source changed after parent plan')
        check()
        inputs,positive,evidence=admitted(config)
        if inputs!=config['input_evidence'] or evidence!=config['admission_evidence']:
            raise ValueError('Evidence changed between parent validation and worker')
        report.update(input_evidence=inputs,admission_evidence=evidence)
        rows,draws=verify_saved_draws(config['run'],config['mode'],config['schedule'])
        report['schedule']=rows;report['draw_file_sha256']=config['draw_file_sha256']
        if sha256(Path(config['run'])/'draws.safetensors')!=config['draw_file_sha256']:
            raise ValueError('Parent draw file changed')
        model,loaded=guard.measure('verify_and_load_external_frozen_core',lambda:load_core(config['weights'],device='mps',check=check))
        atomic_write(out/'weight-load.json',loaded)
        before=guard.measure('hash_all825_current_core_parameters_before_training',
            lambda:parameter_records(model,expected=loaded_records(loaded),check=check))
        atomic_write(out/'core-before.json',before)
        adapter=fresh_adapter().to('mps')
        if sum(p.numel()for p in adapter.parameters())!=947712 or any(p.dtype!=torch.float32 for p in adapter.parameters()):
            raise ValueError('Exact original FP32 adapter required')
        if torch.count_nonzero(adapter.output.weight).item() or torch.count_nonzero(adapter.output.bias).item():
            raise ValueError('Fresh zero output initialization required')
        wrapper=NativeActionWrapper(model,adapter).train()
        optimizer=torch.optim.AdamW(adapter.parameters(),**OPTIMIZER)
        identity=dict(mode=config['mode'],seed=SEED,source_sha256=report['source_sha256'],
            core_before_sha256=sha256(out/'core-before.json'),cache_manifest_sha256=inputs['cache_manifest_sha256'],
            positive_tensor_sha256=inputs['text']['positive_tensor_sha256'])
        initial=save_checkpoint(out,adapter,optimizer,0,identity=identity,
            draw_rng_state=torch.Generator(device='cpu').manual_seed(SEED).get_state())
        report['initial_adapter_sha256']=initial['files']['adapter.safetensors']
        if config['mode']=='fixed16':
            probe=evidence['probe']
            if probe['initial_adapter_sha256']!=report['initial_adapter_sha256'] or probe['schedule_rows']!=rows[:2]:
                raise ValueError('Fresh fixed16 initialization/draws differ from the probe protocol')
        report['last_checkpoint']=initial;guard.save()
        for index,row in enumerate(rows):
            check();report['active_update']=index+1;guard.save()
            windows=[]
            for name in row['branches']:
                values,provenance=load_training_window(config['cache'],name)
                expected=inputs['windows'][name]
                if provenance!=expected['provenance'] or {n:tensor_sha256(v)for n,v in values.items()}!=expected['tensors']:
                    raise ValueError('Training window changed after input validation')
                windows.append(values)
            record=guard.measure(f'paired_update_{index+1:04d}',lambda:paired_update(wrapper,windows,draws[row['noise_key']],row['k'],positive,optimizer,check=check))
            if index==0 and record['command_gru_gradient_l2']!=0.:
                raise RuntimeError('Zero-output initialization should block first-update GRU gradient')
            if index==1:
                if record['command_gru_gradient_l2']<=0.:raise RuntimeError('Second paired update must reach the upstream command GRU')
                report['second_update_gru_gradient_nonzero']=True
            # Never save a failed or partially applied optimizer update as last-valid.
            checkpoint=save_checkpoint(out,adapter,optimizer,index+1,identity=identity,draw_rng_state=draws[f'rng_after_{index:04d}'])
            record.update(update=index+1,start=row['start'],k=row['k'],sigma=row['sigma'],noise_sha256=row['noise_sha256'])
            report['updates'].append(record);report.update(completed_updates=index+1,last_checkpoint=checkpoint);guard.save()
            del windows
        after=guard.measure('hash_all825_current_core_parameters_after_training',lambda:parameter_records(model,expected=before,check=check))
        atomic_write(out/'core-after.json',after)
        report.update(base_unchanged=True,all825_current_value_hashes_verified=True,final_checkpoint=report['last_checkpoint'],
            automatic_promotion_to_fixed16=False,training_only=True)
        if config['source_sha256']!=gate.source_hashes():raise ValueError('Source changed during training')
        report['output_sha256']={name:sha256(out/name)for name in ('weight-load.json','core-before.json','core-after.json','last-valid.json')}
        check();completed=True
    except BaseException as caught:
        error=caught
        report['failure_boundary']='Last-valid checkpoint is only the last fully checked optimizer update; no failure state overwrites it'
        raise
    finally:
        if not completed and error is None:error=RuntimeError('Training did not finish')
        if before is not None and not completed:
            report['base_unchanged']=None
            report['base_after_unavailable']='Complete after-training hash verification did not finish; no unchanged-base claim'
        guard.close(error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--mode',choices=('probe','fixed16'),default='probe')
    parser.add_argument('--execute',action='store_true')
    for name in ('capture','cache','roundtrip-run','text-cache','weights','cpu-report','independent-report','admission','probe-run','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.worker:worker(json.loads(sys.stdin.read(131072)));return
    required=('capture','cache','roundtrip_run','text_cache','output')
    if any(getattr(args,n)is None for n in required):parser.error('Capture, completed cache, roundtrip, genuine text and fresh output are required')
    if args.output.resolve().is_relative_to(gate.HERE.parent.resolve()):raise ValueError('Output must be outside the source package')
    out=new_directory(args.output);report=dict(status='running',protocol=gate.protocol(args.mode),model_execution=False,gpu_execution=False)
    try:
        report['source_sha256']=gate.snapshot(out)
        inputs,positive=gate.validate_inputs(args.capture,args.cache,args.roundtrip_run,args.text_cache)
        report['input_evidence']=inputs
        rows,draws=make_draws(args.mode);save_file(draws,str(out/'draws.safetensors'))
        save_file({'atrium':positive.contiguous()},str(out/'positive.safetensors'))
        report.update(schedule=rows,draw_file_sha256=sha256(out/'draws.safetensors'),positive_file_sha256=sha256(out/'positive.safetensors'),
            visual_admission='Not inferred from finite outputs, CPU tests, codec reconstruction or a probe')
        atomic_write(out/'plan.json',report)
        if not args.execute:
            report.update(status='planned',execution_requires=['current CPU report','independent review','explicit parent visual admission for this mode',
                'completed separate probe and fresh initialization for fixed16'])
            return
        required=('weights','cpu_report','independent_report','admission')
        if any(getattr(args,n)is None for n in required):raise ValueError('Execution requires weights and every admission gate')
        config={n:str(getattr(args,n).resolve()) if getattr(args,n)is not None else None for n in (*required,'capture','cache','roundtrip_run','text_cache','probe_run')}
        config.update(mode=args.mode,device='mps',run=str(out.resolve()),output=str((out/'result').resolve()),
            source_sha256=report['source_sha256'],input_evidence=inputs,schedule=rows,draw_file_sha256=report['draw_file_sha256'])
        repeated,unused,evidence=admitted(config)
        if repeated!=inputs:raise ValueError('Input changed during preparation')
        del unused
        config['admission_evidence']=evidence
        config['deadline']=time.monotonic()+900
        env=os.environ.copy()
        for name,value in ENVIRONMENT.items():
            if value is None:env.pop(name,None)
            else:env[name]=value
        env['PYTHONUNBUFFERED']='1'
        atomic_write(out/'launch.json',dict(config,worker_environment=ENVIRONMENT))
        report['model_execution']=True;report['gpu_execution']=True
        with (out/'worker.log').open('x')as log:
            process=subprocess.Popen([sys.executable,'-m','experiments.wan22_native.action_training.train','--worker'],
                stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,cwd=gate.REPO,env=env)
            terminal=handoff_and_guard(process,config,out,max_seconds=900,max_rss_gib=18,minimum_available_gib=2)
        report.update(status='complete' if terminal['status']=='complete' else terminal['status'],terminal_sha256=sha256(out/'terminal.json'))
        if terminal['status']!='complete':raise RuntimeError('Training child failed or stopped; retained evidence is incomplete')
        result=json.loads((out/'result/metrics.json').read_text())
        if result.get('status')!='passed' or result.get('completed_updates')!=report['protocol']['paired_updates'] or result.get('base_unchanged')is not True:
            raise RuntimeError('Worker lacks complete prescribed update and base verification')
    except BaseException as caught:
        report.update(status='interrupted' if isinstance(caught,KeyboardInterrupt)else'failed',error_type=type(caught).__name__,error=str(caught));raise
    finally:
        atomic_write(out/'plan.json',report)

if __name__=='__main__':main()
