# SPDX-License-Identifier: Apache-2.0
"""Separate guarded native VAE cache. Plan/preflight are CPU only."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
import packet
import cache
from experiments.atrium_factorial import reader
from experiments.wan22_native.action_cuda import data
from experiments.wan22_native.action_training.objective import parameter_records
from experiments.wan22_native.cuda_reference.decode import load_codec
from experiments.wan22_native.official_cpu.streaming import sha,tensor_sha
from experiments.wan22_native.spatial_reference.guards import (
    atomic,limits,hardware,Monitor,supervise,stop_child,_deadline)


def codec_records(provenance):
    data._validate_codec(provenance)
    return {name:dict(shape=row['shape'],dtype='float32',sha256=row['sha256'])
            for name,row in provenance['tensors'].items()}


def worker(config):
    root=Path(config['prepared_directory']);out=root/'worker';out.mkdir(exist_ok=False)
    started=time.monotonic()
    report=dict(schema=packet.SCHEMA,status='running',model_execution=False,scope=packet.SCOPE,
                limits=limits('codec'),core_model_loaded=False,quality_assessed=False)
    atomic(out/'metrics.json',report)
    try:
        _deadline(config['deadline'],'codec')
        plan=packet.read_prepared(root)
        if sha(root/'plan.json')!=config['plan_sha256']:raise ValueError('Prepared plan changed')
        admitted=packet.admission(config['admission'],root,plan,config['expected_gpu'])
        if admitted!=config['admission_record']:raise ValueError('Admission changed')
        torch.set_num_threads(1)
        report.update(source_sha256=plan['source_sha256'],plan_sha256=config['plan_sha256'],
            manifest_sha256=packet.MANIFEST,cpu_report_sha256=plan['cpu_report_sha256'],admission=admitted,
            protocol=plan['protocol'])
        _deadline(config['deadline'],'codec');report['hardware']=hardware(config['expected_gpu'])
        atomic(out/'metrics.json',report)
        with Monitor(out,config['deadline'],'codec') as guard:
            loaded=time.monotonic()
            model,scale,provenance=load_codec(Path(config['weights'])/'Wan2.2_VAE.pth',guard.check)
            torch.cuda.synchronize();report['load_seconds']=time.monotonic()-loaded
            atomic(out/'weight-load.json',provenance)
            expected=codec_records(provenance)
            try:
                begin=time.monotonic()
                before=parameter_records(model,check=guard.check,expected_count=196,expected=expected)
                atomic(out/'codec-before.json',before);report['before_hash_seconds']=time.monotonic()-begin
                scales={str(i):tensor_sha(v.detach().cpu()) for i,v in enumerate(scale)}
                report['normalization_sha256']=scales
                report['model_execution']=True;atomic(out/'metrics.json',report)
                identity=dict(source_sha256=plan['source_sha256'],plan_sha256=config['plan_sha256'],
                    manifest_sha256=packet.MANIFEST,admission_sha256=admitted['sha256'],
                    codec_provenance=provenance,normalization_sha256=scales,
                    commands_sha256={arm:plan['input_plan']['arms'][arm]['arrays']['commands']['sha256'] for arm in reader.ARMS})
                def encode(video):
                    if not all(v is None for v in model._feat_map+model._enc_feat_map):
                        raise RuntimeError('Native cache was populated before encode')
                    value=data.native_encode(model,scale,video,'spatial')
                    if not all(v is None for v in model._feat_map+model._enc_feat_map):
                        raise RuntimeError('Native cache was not cleared after encode')
                    return value
                torch.cuda.reset_peak_memory_stats()
                cache.encode_sequence(root/'result',
                    lambda arm:reader.read_window(root/'capture',arm,expected_manifest_sha256=packet.MANIFEST),
                    encode,identity=identity,check=guard.check)
                torch.cuda.synchronize()
                report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
                report['peak_reserved_bytes']=torch.cuda.max_memory_reserved(0)
                begin=time.monotonic()
                after=parameter_records(model,check=guard.check,expected_count=196,expected=before)
                atomic(out/'codec-after.json',after);report['after_hash_seconds']=time.monotonic()-begin
                if scales!={str(i):tensor_sha(v.detach().cpu()) for i,v in enumerate(scale)}:
                    raise RuntimeError('Original native normalization changed')
                report.update(all196_current_values_unchanged=True,normalization_unchanged=True,
                              codec_caches_clear=all(v is None for v in model._feat_map+model._enc_feat_map))
                if not report['codec_caches_clear']:raise RuntimeError('Native codec cache remained populated')
            finally:
                model.clear_cache();del model,scale;gc.collect();torch.cuda.empty_cache()
            guard.check()
        packet.read_prepared(root);_deadline(config['deadline'],'codec')
        report.update(result_completion_sha256=sha(root/'result/completion.json'),
            output_sha256={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
                if p.is_file() and (p.is_relative_to(root/'result') or (p.parent==out and p.name in
                    ('weight-load.json','codec-before.json','codec-after.json','monitor-terminal.json')))},
            status='passed',sources_unchanged=True,inputs_unchanged=True)
        return report
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;atomic(out/'metrics.json',report)


def validate_completed(root):
    root=Path(root);plan=packet.read_prepared(root)
    terminal=packet.evidence.read_json(root/'terminal.json')
    report=packet.evidence.read_json(root/'worker/metrics.json')
    completion=packet.evidence.read_json(root/'result/completion.json')
    if (terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or terminal.get('cleanup_error') is not None
            or report.get('status')!='passed' or report.get('all196_current_values_unchanged') is not True
            or report.get('normalization_unchanged') is not True or report.get('codec_caches_clear') is not True
            or report.get('model_execution') is not True or report.get('source_sha256')!=plan['source_sha256']
            or report.get('plan_sha256')!=sha(root/'plan.json') or report.get('manifest_sha256')!=packet.MANIFEST
            or report.get('result_completion_sha256')!=sha(root/'result/completion.json')
            or list(root.rglob('watchdog-stop.json'))):raise ValueError('Completed native cache/terminal required')
    if (completion.get('identity',{}).get('plan_sha256')!=sha(root/'plan.json')
            or completion['identity'].get('commands_sha256')!={arm:plan['input_plan']['arms'][arm]['arrays']['commands']['sha256'] for arm in reader.ARMS}
            or completion['identity'].get('source_sha256')!=plan['source_sha256']
            or completion['identity'].get('manifest_sha256')!=packet.MANIFEST
            or completion['identity'].get('admission_sha256')!=report.get('admission',{}).get('sha256')
            or completion['identity'].get('normalization_sha256')!=report.get('normalization_sha256')):
        raise ValueError('Encoded input/command identity differs from prepared capture')
    names={'result/completion.json','result/observation.safetensors','result/observation-repeat.safetensors',
           *(f'result/{a}.safetensors' for a in reader.ARMS),
           *(f'worker/{n}.json' for n in ('weight-load','codec-before','codec-after','monitor-terminal'))}
    if set(report.get('output_sha256',{}))!=names:raise ValueError('Complete output coverage required')
    for name,digest in report['output_sha256'].items():
        if sha(packet.evidence.relative_file(root,name))!=digest:raise ValueError('Worker result bytes changed')
    before=packet.evidence.read_json(root/'worker/codec-before.json')
    after=packet.evidence.read_json(root/'worker/codec-after.json')
    loaded=packet.evidence.read_json(root/'worker/weight-load.json')
    if before!=after or before!=codec_records(loaded) or completion['identity'].get('codec_provenance')!=loaded:
        raise ValueError('All196 unchanged original codec values required')
    monitor=packet.evidence.read_json(root/'worker/monitor-terminal.json')
    if monitor.get('status')!='complete' or monitor.get('mode')!='codec' or monitor.get('sample_count',0)<1:
        raise ValueError('Completed codec monitor required')
    for arm in reader.ARMS:cache.read_window(root/'result',arm,conditioning_only=True)
    return report


def read_completed(root,arm,*,conditioning_only=False):
    """Require completed parent/worker evidence before exposing a window."""
    root=Path(root);report=validate_completed(root)
    parent=packet.evidence.read_json(root/'metrics.json')
    if (parent.get('status')!='passed' or parent.get('worker_metrics_sha256')!=sha(root/'worker/metrics.json')
            or parent.get('terminal_sha256')!=sha(root/'terminal.json') or parent.get('source_sha256')!=report['source_sha256']):
        raise ValueError('Completed parent identity required')
    values,provenance=cache.read_window(root/'result',arm,conditioning_only=conditioning_only)
    provenance.update(parent_sha256=sha(root/'metrics.json'),worker_sha256=sha(root/'worker/metrics.json'),
                      terminal_sha256=sha(root/'terminal.json'))
    return values,provenance


def execute(prepared_directory,weights,expected_gpu,admission):
    root=Path(prepared_directory).absolute()
    if (not root.is_dir() or root.resolve().is_relative_to(packet.REPO)
            or any(p.is_symlink() for p in (root,*root.parents))):raise ValueError('Regular prepared directory outside repo required')
    if any((root/name).exists() for name in ('attempt.json','launch.json','worker','result','terminal.json','worker.log')):
        raise ValueError('Single-use attempt; retain any previous failure')
    started=time.monotonic();deadline=started+limits('codec')['seconds']
    with (root/'attempt.json').open('x') as stream:json.dump(dict(schema=packet.SCHEMA,started_monotonic=started),stream)
    parent=dict(schema=packet.SCHEMA,status='running',model_execution=False,scope=packet.SCOPE,
                limits=limits('codec'),core_model_loaded=False,training_admitted=False)
    proc=None
    try:
        plan=packet.read_prepared(root);admitted=packet.admission(admission,root,plan,expected_gpu)
        config=dict(prepared_directory=str(root),weights=str(Path(weights).absolute()),expected_gpu=expected_gpu,
                    admission=str(Path(admission).absolute()),admission_record=admitted,
                    plan_sha256=sha(root/'plan.json'),deadline=deadline)
        atomic(root/'launch.json',config)
        if (root/'launch.json').stat().st_size>4*2**20:raise ValueError('Bounded worker config required')
        shutil.copyfile(admission,root/'executed-admission.json')
        parent.update(plan_sha256=config['plan_sha256'],source_sha256=plan['source_sha256'],
                      manifest_sha256=packet.MANIFEST,admission=admitted)
        with (root/'worker.log').open('x') as log:
            _deadline(deadline,'codec')
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker-config',str(root/'launch.json')],
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            parent['model_execution']=None;atomic(root/'metrics.json',parent)
            supervise(proc,root,deadline,'codec')
        validate_completed(root)
        if packet.admission(admission,root,plan,expected_gpu)!=admitted:raise ValueError('Admission changed')
        _deadline(deadline,'codec')
        parent.update(status='passed',model_execution=True,worker_metrics_sha256=sha(root/'worker/metrics.json'),
                      terminal_sha256=sha(root/'terminal.json'))
        return parent
    except BaseException as error:
        cleanup_error=None
        try:
            if proc is not None:stop_child(proc)
        except BaseException as cleanup:cleanup_error=str(cleanup)
        if not (root/'terminal.json').exists():
            atomic(root/'terminal.json',dict(status='failed',exit_code=proc.returncode if proc else None,
                                           error=str(error),cleanup_error=cleanup_error,mode='codec'))
        elif cleanup_error:atomic(root/'handoff-cleanup-error.json',dict(error=str(error),cleanup_error=cleanup_error))
        if proc is not None:
            parent['model_execution']=None
            try:
                partial=packet.evidence.read_json(root/'worker/metrics.json')
                if type(partial.get('model_execution')) is bool:parent['model_execution']=partial['model_execution']
            except (OSError,ValueError):pass
        parent.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        parent['elapsed_seconds']=time.monotonic()-started;atomic(root/'metrics.json',parent)


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    for name in ('prepare','preflight','execute'):modes.add_argument('--'+name,action='store_true')
    modes.add_argument('--worker-config',type=Path)
    for name in ('capture','cpu-report','output','prepared-directory','weights','admission'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--expected-gpu');args=parser.parse_args()
    if args.worker_config:result=worker(packet.evidence.read_json(args.worker_config,maximum_bytes=4*2**20))
    elif args.execute:
        names=('prepared_directory','weights','expected_gpu','admission')
        if any(getattr(args,k) is None for k in names):parser.error('Complete explicit execution arguments required')
        result=execute(**{k:getattr(args,k) for k in names})
    elif args.prepare:
        if any(getattr(args,k) is None for k in ('capture','cpu_report','output')):parser.error('Capture, CPU report and fresh output required')
        result=packet.prepare(args.capture,args.cpu_report,args.output)
    elif args.preflight:
        if args.prepared_directory is None:parser.error('Prepared directory required')
        plan=packet.read_prepared(args.prepared_directory)
        result=dict(status='passed',model_execution=False,plan_sha256=sha(args.prepared_directory/'plan.json'),source_sha256=plan['source_sha256'])
    else:result=dict(status='plan-only',model_execution=False,protocol=packet.protocol(),source_sha256=packet.sources())
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
