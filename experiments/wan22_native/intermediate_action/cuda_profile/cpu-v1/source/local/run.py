# SPDX-License-Identifier: Apache-2.0
"""Plan by default. Explicit preparation, preflight or admitted guarded worker."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from safetensors.torch import save_file
import packet
import engine
from experiments.wan22_native.action_adapter.model import PostBlockActionAdapter
from experiments.wan22_native.action_cuda.probe import _original_weights, _runtime_flags, PRECISION
from experiments.wan22_native.action_training.objective import parameter_records, save_checkpoint
from experiments.wan22_native.cuda_reference import native
from experiments.wan22_native.intermediate_action.cached_intermediate import CachedIntermediateActionBridge
from experiments.wan22_native.official_cpu.streaming import sha, tensor_sha
from experiments.wan22_native.spatial_reference.guards import (
    atomic, limits, hardware, Monitor, supervise, stop_child, _deadline)


def native_reference(core, noisy, times, context):
    """Literal native equations; normal constants survive the lazy RoPE move."""
    if tuple(noisy.shape) != packet.SHAPE or tuple(times.shape) != (1,4290):
        raise ValueError('Only the fixed spatial profile is admitted')
    with torch.inference_mode(False), torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        values=core([noisy[0].to('cuda:0')],times.to('cuda:0'),[context.to('cuda:0')],4290)
    torch.cuda.synchronize()
    if not isinstance(values,list) or len(values)!=1 or not isinstance(values[0],torch.Tensor):
        raise RuntimeError('One literal native velocity required')
    return values[0].detach().float().cpu().unsqueeze(0)


def worker(config):
    root=Path(config['prepared_directory']); out=root/'result';out.mkdir(exist_ok=False)
    started=time.monotonic()
    report=dict(schema=packet.SCHEMA,status='running',model_execution=False,completed_updates=0,
                scope=packet.SCOPE,quality_assessed=False,image_generation=False,limits=limits('pair'))
    atomic(out/'metrics.json',report)
    try:
        _deadline(config['deadline'],'pair')
        plan,data=packet.read_prepared(root)
        if sha(root/'plan.json')!=config['plan_sha256']:
            raise ValueError('Plan changed since dispatch')
        admitted=packet.admission(config['admission'],root,plan,config['expected_gpu'])
        if admitted!=config['admission_record']:
            raise ValueError('Admission changed since dispatch')
        torch.set_num_threads(1)
        report.update(source_sha256=plan['source_sha256'],plan_sha256=config['plan_sha256'],
            input_selection_sha256=plan['input_selection_sha256'],input_identity=engine.identities(data),
            admission=admitted,protocol=plan['protocol'],precision=PRECISION)
        _deadline(config['deadline'],'pair')
        report['hardware']=hardware(config['expected_gpu'])
        report['runtime_flags']=_runtime_flags()
        atomic(out/'metrics.json',report)
        with Monitor(out,config['deadline'],'pair') as monitor:
            def check():
                monitor.check()
                if _runtime_flags()!=report['runtime_flags']:
                    raise RuntimeError('Precision, attention or deterministic settings changed')
            begin=time.monotonic()
            core,weights=native.load_model(config['weights'],check)
            torch.cuda.synchronize()
            report['load_seconds']=time.monotonic()-begin
            atomic(out/'weight-load.json',weights)
            expected=_original_weights(weights)
            def verify_core(label):
                begin=time.monotonic()
                values=parameter_records(core,check=check,expected=expected)
                atomic(out/('core-'+label+'.json'),values)
                report.setdefault('base_hash_seconds',{})[label]=time.monotonic()-begin
                atomic(out/'metrics.json',report)
            verify_core('before')
            # Record the native CPU complex constant before its literal move.
            rotary_before=core.freqs.detach().cpu().clone()
            report['rotary_before_sha256']=tensor_sha(rotary_before)
            def make_bridge(index):
                with torch.inference_mode(False):
                    adapter=PostBlockActionAdapter()
                    adapter.load_state_dict(data['initial'],strict=True)
                    adapter.to('cuda:0')
                torch.set_rng_state(data['rng'])
                torch.cuda.manual_seed_all(20260907)
                for name,value in adapter.state_dict().items():
                    copied=value.detach().cpu()
                    if tensor_sha(copied)!=tensor_sha(data['initial'][name]):
                        raise RuntimeError('Saved adapter CUDA copy differs')
                    del copied
                return CachedIntermediateActionBridge(core,adapter,block_index=index,profile='spatial')
            def retain(name,values):
                path=out/(name+'.safetensors')
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():raise ValueError('Refuse overwriting partial numerical output')
                save_file({key:value.detach().cpu().contiguous() for key,value in values.items()},str(path))
            identity=dict(schema=packet.SCHEMA,source_sha256=plan['source_sha256'],
                plan_sha256=config['plan_sha256'],admission_sha256=admitted['sha256'],
                input_selection_sha256=plan['input_selection_sha256'],protocol=plan['protocol'])
            def checkpoint(label,completed,bridge,optimizer,current):
                check(); directory=out/label;directory.mkdir(exist_ok=True)
                retain(f'{label}/cuda-rng-{completed:04d}',{'rng':torch.cuda.get_rng_state(0)})
                record=save_checkpoint(directory,bridge.adapter,optimizer,completed,
                    identity=dict(identity,placement=label),draw_rng_state=data['rng'])
                report.setdefault('last_checkpoint',{})[label]=dict(directory=label+'/'+record['directory'],
                    manifest_sha256=record['manifest_sha256'],completed_updates=completed)
            def progress(current):
                report.update(current);report['status']='running'
                atomic(out/'metrics.json',report)
            def release():
                gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();check()
            report['model_execution']=True
            atomic(out/'metrics.json',report)
            numerical=engine.execute(data,placements=packet.PLACEMENTS,make_bridge=make_bridge,
                native_predict=lambda x,t,c:native_reference(core,x,t,c),retain=retain,checkpoint=checkpoint,
                progress=progress,check=check,synchronize=torch.cuda.synchronize,
                reset_peak=torch.cuda.reset_peak_memory_stats,release=release,verify_core=verify_core,
                memory=lambda:dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                                   allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0)))
            report.update(numerical);report['status']='running'
            if (core.freqs.is_inference() or core.freqs.requires_grad
                    or not torch.equal(core.freqs.detach().cpu(),rotary_before)):
                raise RuntimeError('Literal rotary transfer changed its value or normal constant status')
            report.update(rotary_after_sha256=tensor_sha(core.freqs.detach().cpu()),rotary_copy_exact=True,
                          all825_current_value_hashes_verified=True)
            check()
        packet.read_prepared(root)
        _deadline(config['deadline'],'pair')
        report['output_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
                                if p.is_file() and p.name not in ('metrics.json','memory.jsonl')}
        report.update(status='passed',sources_unchanged=True,inputs_unchanged=True,automatic_promotion=False)
        return report
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        atomic(out/'metrics.json',report)


def execute(prepared_directory,weights,expected_gpu,admission):
    root=Path(prepared_directory).absolute()
    if (not root.is_dir() or root.resolve().is_relative_to(packet.REPO)
            or any(p.is_symlink() for p in (root,*root.parents))):
        raise ValueError('Regular prepared output outside the repository required')
    if any((root/name).exists() for name in ('attempt.json','launch.json','result','terminal.json','worker.log')):
        raise ValueError('Single-use attempt; retain failures and prepare a fresh output')
    started=time.monotonic();deadline=started+limits('pair')['seconds']
    with (root/'attempt.json').open('x') as stream:
        json.dump({'schema':packet.SCHEMA,'started_monotonic':started},stream)
    parent=dict(schema=packet.SCHEMA,status='running',scope=packet.SCOPE,model_execution=False,
                limits=limits('pair'),quality_assessed=False,image_generation=False)
    proc=None
    try:
        plan,_=packet.read_prepared(root)
        admitted=packet.admission(admission,root,plan,expected_gpu)
        config=dict(prepared_directory=str(root),weights=str(Path(weights).absolute()),expected_gpu=expected_gpu,
                    admission=str(Path(admission).absolute()),admission_record=admitted,
                    plan_sha256=sha(root/'plan.json'),deadline=deadline)
        atomic(root/'launch.json',config)
        if (root/'launch.json').stat().st_size>4*2**20:raise ValueError('Bounded worker configuration required')
        shutil.copyfile(admission,root/'executed-admission.json')
        parent.update(plan_sha256=config['plan_sha256'],source_sha256=plan['source_sha256'],
                      input_selection_sha256=plan['input_selection_sha256'],admission=admitted)
        with (root/'worker.log').open('x') as log:
            _deadline(deadline,'pair')
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker-config',str(root/'launch.json')],
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            parent['model_execution']=None
            atomic(root/'metrics.json',parent)
            supervise(proc,root,deadline,'pair')
        terminal=packet.old.read_json(root/'terminal.json')
        result=packet.old.read_json(root/'result/metrics.json')
        if (terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or terminal.get('cleanup_error') is not None
                or result.get('status')!='passed' or result.get('completed_updates')!=4 or result.get('zero_gate_passed') is not True
                or result.get('base_unchanged') is not True or result.get('all825_current_value_hashes_verified') is not True
                or result.get('source_sha256')!=plan['source_sha256'] or result.get('plan_sha256')!=config['plan_sha256']
                or result.get('input_selection_sha256')!=plan['input_selection_sha256']
                or len(result.get('parity',[]))!=16 or any(not r.get('passed') for r in result.get('parity',[]))
                or list(root.rglob('watchdog-stop.json'))):
            raise RuntimeError('Both placements did not complete all bounded numerical gates')
        for name,digest in result['output_sha256'].items():
            if sha(packet.old.relative_file(root/'result',name))!=digest:
                raise ValueError('Retained output changed')
        packet.read_prepared(root)
        if packet.admission(admission,root,plan,expected_gpu)!=admitted:raise ValueError('Admission changed')
        _deadline(deadline,'pair')
        parent.update(status='passed',model_execution=True,result_metrics_sha256=sha(root/'result/metrics.json'),
                      terminal_sha256=sha(root/'terminal.json'))
        return parent
    except BaseException as error:
        cleanup_error=None
        try:
            if proc is not None:stop_child(proc)
        except BaseException as cleanup:cleanup_error=str(cleanup)
        if not (root/'terminal.json').exists():
            atomic(root/'terminal.json',dict(status='failed',exit_code=proc.returncode if proc else None,
                                           error=str(error),cleanup_error=cleanup_error,mode='pair'))
        elif cleanup_error:
            atomic(root/'handoff-cleanup-error.json',dict(error=str(error),cleanup_error=cleanup_error))
        if proc is not None:
            parent['model_execution']=None
            try:
                partial=packet.old.read_json(root/'result/metrics.json')
                if type(partial.get('model_execution')) is bool:parent['model_execution']=partial['model_execution']
            except (OSError,ValueError):pass
        parent.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise
    finally:
        parent['elapsed_seconds']=time.monotonic()-started
        atomic(root/'metrics.json',parent)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    for name in ('prepare','preflight','execute'):modes.add_argument('--'+name,action='store_true')
    modes.add_argument('--worker-config',type=Path)
    for name in ('input-root','cpu-report','output','prepared-directory','weights','admission'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--expected-gpu')
    args=parser.parse_args()
    if args.worker_config:
        result=worker(packet.old.read_json(args.worker_config,maximum_bytes=4*2**20))
    elif args.execute:
        keys=('prepared_directory','weights','expected_gpu','admission')
        if any(getattr(args,k) is None for k in keys):parser.error('Complete explicit execution arguments required')
        result=execute(**{k:getattr(args,k) for k in keys})
    elif args.prepare:
        if any(getattr(args,k) is None for k in ('input_root','cpu_report','output')):parser.error('Input root, CPU report and output required')
        result=packet.prepare(args.input_root,args.cpu_report,args.output)
    elif args.preflight:
        if args.prepared_directory is None:parser.error('Prepared directory required')
        plan,_=packet.read_prepared(args.prepared_directory)
        result=dict(status='passed',model_execution=False,plan_sha256=sha(args.prepared_directory/'plan.json'),
                    source_sha256=plan['source_sha256'])
    else:
        result=dict(schema=packet.SCHEMA,status='plan-only',model_execution=False,protocol=packet.protocol(),source_sha256=packet.sources())
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
