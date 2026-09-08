# SPDX-License-Identifier: Apache-2.0
"""Plan-only default; explicit CPU preparation or separately admitted CUDA run."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from safetensors.torch import save_file
import training_inputs as packet
import training_math
from experiments.wan22_native.action_adapter.model import PostBlockActionAdapter
from experiments.wan22_native.action_cuda.probe import _original_weights,_runtime_flags,PRECISION
from experiments.wan22_native.action_cuda.probe_math import OPTIMIZER
from experiments.wan22_native.action_training.objective import parameter_records,save_checkpoint
from experiments.wan22_native.cuda_reference import native
from experiments.wan22_native.intermediate_action.cached_intermediate import CachedIntermediateActionBridge
from experiments.wan22_native.official_cpu.streaming import sha,tensor_sha
from experiments.wan22_native.spatial_reference.guards import (
    atomic,limits,hardware,Monitor,supervise,stop_child,_deadline)


def native_reference(core,noisy,times,context):
    """Literal native forward, creating normal constants for suffix backward."""
    if tuple(noisy.shape)!=(1,48,5,44,78) or tuple(times.shape)!=(1,4290):
        raise ValueError('Only native spatial inputs are admitted')
    with torch.inference_mode(False),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        values=core([noisy[0].to('cuda:0')],times.to('cuda:0'),[context.to('cuda:0')],4290)
    torch.cuda.synchronize()
    if not isinstance(values,list) or len(values)!=1 or not isinstance(values[0],torch.Tensor):
        raise RuntimeError('One original native output required')
    return values[0].detach().float().cpu().unsqueeze(0)


def input_identity(data):
    return dict(windows={arm:{k:tensor_sha(v) for k,v in window.items()} for arm,window in data['windows'].items()},
                positive=tensor_sha(data['positive']),negative=tensor_sha(data['negative']),
                initial={k:tensor_sha(v) for k,v in data['initial'].items()},initial_cpu_rng=tensor_sha(data['rng']))


def worker(config):
    root=Path(config['prepared_directory']);out=root/'result';out.mkdir(exist_ok=False)
    started=time.monotonic()
    report=dict(schema=packet.SCHEMA,status='running',model_execution=False,completed_updates=0,
                scope=packet.SCOPE,quality_assessed=False,image_generation=False,limits=limits('pair'))
    atomic(out/'metrics.json',report)
    try:
        _deadline(config['deadline'],'pair')
        plan,data=packet.read_prepared(root)
        if sha(root/'plan.json')!=config['plan_sha256']:raise ValueError('Plan changed since dispatch')
        admitted=packet.admission(config['admission'],root,plan,config['expected_gpu'])
        if admitted!=config['admission_record']:raise ValueError('Admission changed since dispatch')
        torch.set_num_threads(1)
        identity=input_identity(data)
        report.update(source_sha256=plan['source_sha256'],plan_sha256=config['plan_sha256'],
            prior_receipt_sha256=plan['prior_receipt_sha256'],input_identity=identity,
            admission=admitted,protocol=plan['protocol'],precision=PRECISION)
        _deadline(config['deadline'],'pair')
        report['hardware']=hardware(config['expected_gpu']);report['runtime_flags']=_runtime_flags()
        atomic(out/'metrics.json',report)
        with Monitor(out,config['deadline'],'pair') as monitor:
            def check():
                monitor.check()
                if _runtime_flags()!=report['runtime_flags']:raise RuntimeError('Precision/runtime settings changed')
            report['model_execution']=True;atomic(out/'metrics.json',report)
            begin=time.monotonic();core,weights=native.load_model(config['weights'],check);torch.cuda.synchronize()
            report['load_seconds']=time.monotonic()-begin;atomic(out/'weight-load.json',weights)
            expected=_original_weights(weights)
            def verify_core(label):
                begin=time.monotonic();values=parameter_records(core,check=check,expected=expected)
                atomic(out/('core-'+label+'.json'),values)
                report.setdefault('base_hash_seconds',{})[label]=time.monotonic()-begin
                atomic(out/'metrics.json',report)
            verify_core('before')
            rotary_before=core.freqs.detach().cpu().clone()
            report['rotary_before_sha256']=tensor_sha(rotary_before)
            with torch.inference_mode(False):
                adapter=PostBlockActionAdapter();adapter.load_state_dict(data['initial'],strict=True);adapter.to('cuda:0')
            torch.set_rng_state(data['rng']);torch.cuda.manual_seed_all(20260907)
            for name,value in adapter.state_dict().items():
                copied=value.detach().cpu()
                if tensor_sha(copied)!=identity['initial'][name]:raise RuntimeError('Saved initial adapter CUDA copy differs')
                del copied
            bridge=CachedIntermediateActionBridge(core,adapter,block_index=28,profile='spatial')
            optimizer=torch.optim.AdamW(adapter.parameters(),**OPTIMIZER)
            torch.cuda.reset_peak_memory_stats(0)
            def memory():
                return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                            allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0))
            def retain(name,values):
                path=out/(name+'.safetensors');path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():raise ValueError('Refuse overwriting partial numerical output')
                save_file({k:v.detach().cpu().contiguous() for k,v in values.items()},str(path))
            checkpoint_identity=dict(schema=packet.SCHEMA,source_sha256=plan['source_sha256'],
                plan_sha256=config['plan_sha256'],admission_sha256=admitted['sha256'],
                prior_receipt_sha256=plan['prior_receipt_sha256'],protocol=plan['protocol'])
            def checkpoint(completed,current):
                check();retain(f'cuda-rng-{completed:04d}',{'rng':torch.cuda.get_rng_state(0)})
                record=save_checkpoint(out,adapter,optimizer,completed,identity=checkpoint_identity,
                                       draw_rng_state=data['draw_rng'](completed))
                report['last_checkpoint']=dict(directory=record['directory'],manifest_sha256=record['manifest_sha256'],completed_updates=completed)
            def progress(current):
                report.update(current);report.update(status='running',memory=memory())
                atomic(out/'metrics.json',report)
            numerical=training_math.run(bridge,data['windows'],plan['schedule'],data['noise'],data['positive'],data['negative'],optimizer,
                initial=data['initial'],retain=retain,checkpoint=checkpoint,progress=progress,check=check,
                synchronize=torch.cuda.synchronize,native_predict=lambda x,t,c:native_reference(core,x,t,c))
            report.update(numerical);report['status']='running'
            verify_core('after')
            if (core.freqs.is_inference() or core.freqs.requires_grad or not torch.equal(core.freqs.detach().cpu(),rotary_before)):
                raise RuntimeError('Original rotary constant changed')
            if input_identity(data)!=identity:raise RuntimeError('Training mutated an input tensor')
            report.update(rotary_after_sha256=tensor_sha(core.freqs.detach().cpu()),rotary_copy_exact=True,
                          all825_current_value_hashes_verified=True,base_unchanged=True,memory=memory())
            check()
        packet.read_prepared(root);_deadline(config['deadline'],'pair')
        report['output_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
                                if p.is_file() and p.name not in ('metrics.json','memory.jsonl')}
        _deadline(config['deadline'],'pair')
        report.update(status='passed',sources_unchanged=True,inputs_unchanged=True,automatic_promotion=False)
        return report
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;atomic(out/'metrics.json',report)


def validate_result(root,plan,admitted):
    root=Path(root);terminal=packet.evidence.read_json(root/'terminal.json');result=packet.evidence.read_json(root/'result/metrics.json')
    if (terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or terminal.get('cleanup_error') is not None
            or result.get('status')!='passed' or result.get('model_execution') is not True
            or result.get('completed_updates')!=128 or result.get('zero_gate_passed') is not True
            or result.get('base_unchanged') is not True or result.get('all825_current_value_hashes_verified') is not True
            or result.get('source_sha256')!=plan['source_sha256'] or result.get('plan_sha256')!=sha(root/'plan.json')
            or result.get('prior_receipt_sha256')!=plan['prior_receipt_sha256'] or result.get('admission')!=admitted
            or len(result.get('parity',[]))!=2 or any(not r.get('passed') or not r.get('exact_equal') for r in result.get('parity',[]))
            or result.get('main_predictions')!=256 or result.get('auxiliary_predictions')!=128
            or result.get('auxiliary_feature_extracts')!=64 or result.get('auxiliary_updates')!=32
            or [r.get('schedule') for r in result.get('updates',[])]!=plan['schedule']
            or result.get('precision')!=PRECISION or result.get('protocol')!=packet.protocol()
            or list(root.rglob('watchdog-stop.json'))):raise RuntimeError('128-update numerical gates incomplete')
    outputs=result.get('output_sha256',{})
    required={'weight-load.json','core-before.json','core-after.json','last-valid.json','monitor-terminal.json'}
    for arm in ('stationary_closed','stationary_interact'):
        required.update(f'parity-{arm}-{label}.safetensors' for label in ('native','bridge'))
    for row in plan['schedule']:
        i=row['update'];required.add(f'gradients-after-clip-{i:04d}.safetensors')
        required.update(f'main-{i:04d}-{arm}.safetensors' for arm in ('closed','interact'))
        if row['auxiliary']:required.update(f'auxiliary-{i:04d}-{arm}-{label}.safetensors' for arm in ('closed','open') for label in ('positive','negative'))
    for i in training_math.CHECKPOINTS:
        required.add(f'cuda-rng-{i:04d}.safetensors')
        required.update(f'checkpoint-{i:04d}/{n}' for n in ('adapter.safetensors','optimizer-and-rng.pt','manifest.json'))
    if not required<=set(outputs):raise ValueError('Required scientific output files missing')
    actual={str(p.relative_to(root/'result')) for p in (root/'result').rglob('*') if p.is_file() and p.name not in ('metrics.json','memory.jsonl')}
    if actual!=set(outputs):raise ValueError('Output inventory incomplete')
    for name,digest in outputs.items():
        if sha(packet.evidence.relative_file(root/'result',name))!=digest:raise ValueError('Retained output changed')
    return result


def execute(prepared_directory,weights,expected_gpu,admission):
    root=Path(prepared_directory).absolute()
    if (not root.is_dir() or root.resolve().is_relative_to(packet.REPO) or any(p.is_symlink() for p in (root,*root.parents))):
        raise ValueError('Regular prepared output outside repository required')
    if any((root/name).exists() for name in ('attempt.json','launch.json','result','terminal.json','worker.log')):
        raise ValueError('Single-use attempt; retain failures and prepare a fresh output')
    started=time.monotonic();deadline=started+limits('pair')['seconds']
    with (root/'attempt.json').open('x') as stream:json.dump({'schema':packet.SCHEMA,'started_monotonic':started},stream)
    parent=dict(schema=packet.SCHEMA,status='running',scope=packet.SCOPE,model_execution=False,
                limits=limits('pair'),quality_assessed=False,image_generation=False)
    proc=None
    try:
        plan,data=packet.read_prepared(root);del data
        admitted=packet.admission(admission,root,plan,expected_gpu)
        config=dict(prepared_directory=str(root),weights=str(Path(weights).absolute()),expected_gpu=expected_gpu,
                    admission=str(Path(admission).absolute()),admission_record=admitted,plan_sha256=sha(root/'plan.json'),deadline=deadline)
        atomic(root/'launch.json',config)
        if (root/'launch.json').stat().st_size>4*2**20:raise ValueError('Bounded worker configuration required')
        shutil.copyfile(admission,root/'executed-admission.json')
        parent.update(plan_sha256=config['plan_sha256'],source_sha256=plan['source_sha256'],prior_receipt_sha256=plan['prior_receipt_sha256'],admission=admitted)
        with (root/'worker.log').open('x') as log:
            _deadline(deadline,'pair')
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker-config',str(root/'launch.json')],
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            parent['model_execution']=None;atomic(root/'metrics.json',parent)
            supervise(proc,root,deadline,'pair')
        validate_result(root,plan,admitted);packet.read_prepared(root)
        if packet.admission(admission,root,plan,expected_gpu)!=admitted:raise ValueError('Admission changed')
        _deadline(deadline,'pair')
        parent.update(status='passed',model_execution=True,result_metrics_sha256=sha(root/'result/metrics.json'),terminal_sha256=sha(root/'terminal.json'))
        return parent
    except BaseException as error:
        cleanup_error=None
        try:
            if proc is not None:stop_child(proc)
        except BaseException as cleanup:cleanup_error=str(cleanup)
        if not (root/'terminal.json').exists():
            atomic(root/'terminal.json',dict(status='failed',exit_code=proc.returncode if proc else None,error=str(error),cleanup_error=cleanup_error,mode='pair'))
        elif cleanup_error:atomic(root/'handoff-cleanup-error.json',dict(error=str(error),cleanup_error=cleanup_error))
        if proc is not None:
            parent['model_execution']=None
            try:
                partial=packet.evidence.read_json(root/'result/metrics.json')
                if type(partial.get('model_execution')) is bool:parent['model_execution']=partial['model_execution']
            except (OSError,ValueError):pass
        parent.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise
    finally:
        parent['elapsed_seconds']=time.monotonic()-started;atomic(root/'metrics.json',parent)


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    for name in ('prepare','preflight','execute'):modes.add_argument('--'+name,action='store_true')
    modes.add_argument('--worker-config',type=Path)
    for name in ('original-run','cache-run','profile-run','cpu-report','output','prepared-directory','weights','admission'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--expected-gpu');args=parser.parse_args()
    if args.worker_config:result=worker(packet.evidence.read_json(args.worker_config,maximum_bytes=4*2**20))
    elif args.execute:
        keys=('prepared_directory','weights','expected_gpu','admission')
        if any(getattr(args,k) is None for k in keys):parser.error('Complete explicit execution arguments required')
        result=execute(**{k:getattr(args,k) for k in keys})
    elif args.prepare:
        keys=('original_run','cache_run','profile_run','cpu_report','output')
        if any(getattr(args,k) is None for k in keys):parser.error('Exact prior inputs, CPU report and fresh output required')
        result=packet.prepare(**{k:getattr(args,k) for k in keys})
    elif args.preflight:
        if args.prepared_directory is None:parser.error('Prepared directory required')
        plan,_=packet.read_prepared(args.prepared_directory)
        result=dict(status='passed',model_execution=False,plan_sha256=sha(args.prepared_directory/'plan.json'),source_sha256=plan['source_sha256'])
    else:result=dict(schema=packet.SCHEMA,status='plan-only',model_execution=False,protocol=packet.protocol(),source_sha256=packet.sources())
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
