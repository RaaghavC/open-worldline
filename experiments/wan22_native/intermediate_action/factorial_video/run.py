# SPDX-License-Identifier: Apache-2.0
"""One 1800-second six-video parent, two workers capped at 900 seconds each."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
import packet
import video
import sampler
from experiments.wan22_native.action_cuda import probe
from experiments.wan22_native.action_training.objective import parameter_records
from experiments.wan22_native.cuda_reference.native import load_model
from experiments.wan22_native.spatial_reference.codec import load_codec,decode,images
from experiments.wan22_native.spatial_reference.output import save_rgb,read_tensors
from experiments.wan22_native.spatial_reference.config import SPECS
from experiments.wan22_native.spatial_reference.guards import atomic,hardware,Monitor,supervise,stop_child,limits,_deadline

def require_hardware(actual,training):
    """Keep the runtime exact, allowing only equal or greater reported capacity.

    The unchanged hardware() validator and Monitor enforce the 70 GiB minimum,
    60 GiB reserved cap and 8 GiB free-memory floor independently of this check.
    """
    if not isinstance(actual,dict) or not isinstance(training,dict) or set(actual)!=set(training):
        raise ValueError('Native hardware/runtime fields differ from training')
    capacity='total_memory_bytes'
    if (type(actual.get(capacity)) is not int or type(training.get(capacity)) is not int
        or actual[capacity]<training[capacity]):
        raise ValueError('Reported GPU capacity must be at least the training capacity')
    other=lambda value:json.dumps({k:v for k,v in value.items() if k!=capacity},sort_keys=True,allow_nan=False)
    if other(actual)!=other(training):raise ValueError('Native hardware/runtime differs beyond total capacity')
    return {'policy':'All fields exact except reported total GPU bytes may be greater',
            'training_total_memory_bytes':training[capacity],'actual_total_memory_bytes':actual[capacity],
            'additional_reported_bytes':actual[capacity]-training[capacity]}

def admit(path,root,plan):
    a=packet.read(path)
    required=dict(schema=packet.SCHEMA,scope=packet.SCOPE,decision='admit',issued_by='parent-agent',
        plan_sha256=packet.sha(root/'plan.json'),source_sha256=plan['source_sha256'],training_identity=plan['training_identity'],
        training_audit_sha256=plan['training_audit_sha256'],cpu_report_sha256=plan['cpu_report_sha256'],protocol=packet.protocol(),
        minimum_lease_remaining_seconds=2400,native_control=False,training_admitted=False)
    if any(a.get(k)!=v for k,v in required.items()) or not isinstance(a.get('reason'),str) or not a['reason'].strip():raise ValueError('Explicit current six-arm video admission required')
    result=dict(sha256=packet.sha(path),scope=packet.SCOPE,lease_deadline_utc=a.get('lease_deadline_utc'))
    lease_check(result);return result

def lease_check(admitted,*,dispatch=False,now=None):
    stamp=admitted.get('lease_deadline_utc')
    if not isinstance(stamp,str):raise ValueError('Explicit external UTC lease deadline required')
    deadline=datetime.fromisoformat(stamp.replace('Z','+00:00'))
    now=datetime.now(timezone.utc) if now is None else now
    if deadline.tzinfo is None or deadline.utcoffset() is None or now.tzinfo is None:raise ValueError('Timezone-aware lease deadline required')
    required=2400 if dispatch else 600
    if (deadline-now).total_seconds()<=required:raise RuntimeError('Insufficient external lease time including recovery reserve')

def worker(config):
    root=packet.root(config['prepared']);out=root/config['stage']/'result';out.mkdir(exist_ok=False)
    began=time.monotonic();report=dict(schema=packet.SCHEMA,status='running',stage=config['stage'],model_execution=False,
        training=False,future_targets_materialized=False,limits=limits('pair'),combined_limits=limits('clip'))
    atomic(out/'metrics.json',report)
    try:
        _deadline(config['deadline'],'pair');_deadline(config['parent_deadline'],'clip')
        plan,values,contexts,commands=packet.read_prepared(root);admitted=admit(config['decision'],root,plan)
        if packet.sha(root/'plan.json')!=config['plan_sha256'] or admitted!=config['admission']:raise ValueError('Dispatch identity changed')
        torch.set_num_threads(1);actual=hardware(plan['expected_gpu']);comparison=require_hardware(actual,plan['training_details']['hardware']);flags=probe._runtime_flags()
        report.update(plan_sha256=config['plan_sha256'],source_sha256=plan['source_sha256'],admission=admitted,hardware=actual,
            hardware_comparison=comparison,runtime_flags=flags)
        atomic(out/'metrics.json',report)
        with Monitor(out,config['deadline'],'pair') as monitor:
            def check():
                monitor.check();_deadline(config['parent_deadline'],'clip');lease_check(admitted)
                if probe._runtime_flags()!=flags:raise RuntimeError('Native runtime settings changed')
            if config['stage']=='core':
                begin=time.monotonic();core,loaded=load_model(config['weights'],check);torch.cuda.synchronize()
                report.update(load_seconds=time.monotonic()-begin,model_execution=True);atomic(out/'weight-load.json',loaded);atomic(out/'metrics.json',report)
                expected=probe._original_weights(loaded)
                if expected!=plan['training_details']['core_records']:raise ValueError('Foundation differs from actual training')
                begin=time.monotonic();before=parameter_records(core,check=check,expected=expected)
                report['before_hash_seconds']=time.monotonic()-begin;atomic(out/'core-before.json',before)
                rotary=core.freqs.detach().cpu().clone()
                bridge,identity=video.load_bridge(core,root/'adapter.safetensors',plan['training_identity']['final_checkpoint_sha256'])
                def progress(row):report['sampling_progress']=row;atomic(out/'metrics.json',report)
                report['arms']=video.sample_six(bridge,identity,out,values,contexts,commands,check=check,progress=progress)
                video.require_unchanged(bridge,identity)
                begin=time.monotonic();after=parameter_records(core,check=check,expected=before)
                report['after_hash_seconds']=time.monotonic()-begin;atomic(out/'core-after.json',after)
                if core.freqs.is_inference() or core.freqs.requires_grad or not torch.equal(core.freqs.detach().cpu(),rotary):raise RuntimeError('Original rotary constant changed')
                report.update(model_frozen=True,predictions=600,solver_updates=300,block_index=28,rotary_copy_exact=True,
                    model_frozen_definition='All825 core values and loaded adapter values unchanged, every parameter gradient None; no optimizer')
                del bridge,core;gc.collect();torch.cuda.empty_cache()
            elif config['stage']=='decode':
                core_report=packet.read(packet.file(root,'core/result/metrics.json'))
                if packet.sha(root/'core/result/metrics.json')!=config['core_metrics_sha256'] or core_report.get('model_frozen') is not True:raise ValueError('Completed frozen core stage required')
                packet.checked_outputs(root/'core/result',core_report['output_sha256'])
                begin=time.monotonic();model,scale,loaded=load_codec(Path(config['weights'])/'Wan2.2_VAE.pth',check)
                report.update(load_seconds=time.monotonic()-begin,model_execution=True);atomic(out/'weight-load.json',loaded);packet.se._codec_weights(out/'weight-load.json')
                expected={n:dict(shape=r['shape'],dtype='float32',sha256=r['sha256']) for n,r in loaded['tensors'].items()}
                before=parameter_records(model,check=check,expected_count=196,expected=expected);atomic(out/'codec-before.json',before)
                scales={str(i):sampler.tensor_sha(v) for i,v in enumerate(scale)};report['arms']={};spec=SPECS['spatial']
                try:
                    for arm in video.ARMS:
                        check();folder=out/arm;folder.mkdir();path=root/'core/result'/arm/'latents.safetensors'
                        if packet.sha(path)!=core_report['output_sha256'][arm+'/latents.safetensors']:raise ValueError('Generated latent changed')
                        latent=read_tensors(path,{'latent':packet.SHAPE})['latent']
                        begin=time.monotonic();rgb=decode(model,scale,latent,spec.height,spec.width);check()
                        row=dict(decode_seconds=time.monotonic()-begin,latent_sha256=sampler.tensor_sha(latent))
                        save_rgb(rgb,folder/'rgb',spec,purpose='generated_clip');row['images']=images(rgb,folder,spec.height,spec.width)
                        row['cache_clear']=all(v is None for v in model._feat_map+model._enc_feat_map)
                        if not row['cache_clear']:raise RuntimeError('VAE state leaked across clips')
                        report['arms'][arm]=row;atomic(out/'metrics.json',report);del rgb,latent
                    after=parameter_records(model,check=check,expected_count=196,expected=before);atomic(out/'codec-after.json',after)
                    if scales!={str(i):sampler.tensor_sha(v) for i,v in enumerate(scale)}:raise RuntimeError('Native normalization changed')
                    report.update(all196_unchanged=True,normalization_unchanged=True,normalization_sha256=scales)
                finally:model.clear_cache();del model,scale;gc.collect();torch.cuda.empty_cache()
            else:raise ValueError('Only core and decode workers exist')
            check()
        packet.read_prepared(root);_deadline(config['deadline'],'pair');_deadline(config['parent_deadline'],'clip');lease_check(admitted)
        report['output_sha256']={str(p.relative_to(out)):packet.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('metrics.json','memory.jsonl')}
        # Per-arm metrics are evidence too; only the worker's own mutable metrics are excluded.
        for p in out.glob('*/metrics.json'):report['output_sha256'][str(p.relative_to(out))]=packet.sha(p)
        _deadline(config['deadline'],'pair');_deadline(config['parent_deadline'],'clip')
        report.update(status='passed',sources_unchanged=True,inputs_unchanged=True);return report
    except BaseException as e:
        report.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error_type=type(e).__name__,error=str(e));raise
    finally:report['elapsed_seconds']=time.monotonic()-began;atomic(out/'metrics.json',report)

def launch(config):
    root=Path(config['prepared']);out=root/config['stage'];out.mkdir(exist_ok=False);proc=None
    # The child gets at most900s, including its preflight, within the common1800s parent.
    config=dict(config,deadline=min(config['parent_deadline'],time.monotonic()+900))
    atomic(out/'launch.json',config)
    try:
        _deadline(config['deadline'],'pair');_deadline(config['parent_deadline'],'clip');lease_check(config['admission'])
        with (out/'worker.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,str(packet.HERE/'run.py'),'--worker-config',str(out/'launch.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            supervise(proc,out,config['deadline'],'pair')
    except BaseException as e:
        cleanup=None
        try:
            if proc is not None:stop_child(proc)
        except BaseException as c:cleanup=str(c)
        finally:
            if not (out/'terminal.json').exists():atomic(out/'terminal.json',dict(status='failed',exit_code=proc.returncode if proc else None,error=str(e),cleanup_error=cleanup))
            elif cleanup:atomic(out/'handoff-cleanup-error.json',dict(error=str(e),cleanup_error=cleanup))
        raise
    t=packet.read(out/'terminal.json');r=packet.read(out/'result/metrics.json')
    if (t.get('status')!='complete' or type(t.get('exit_code')) is not int or t['exit_code']!=0 or t.get('cleanup_error') is not None
        or r.get('status')!='passed' or r.get('plan_sha256')!=config['plan_sha256'] or r.get('admission')!=config['admission']
        or list(out.rglob('watchdog-stop.json'))):raise RuntimeError('Incomplete guarded worker')
    packet.checked_outputs(out/'result',r['output_sha256']);return r

def execute(prepared,weights,decision):
    out=packet.root(prepared)
    if any((out/n).exists() for n in ('execution-attempt.json','core','decode')):raise ValueError('Single-use prepared video directory')
    began=time.monotonic();deadline=began+1800
    with (out/'execution-attempt.json').open('x') as f:f.write('single-use\n')
    report=dict(schema=packet.SCHEMA,status='running',model_execution=False,training=False,limits=limits('clip'),worker_limits=limits('pair'))
    try:
        plan,*_=packet.read_prepared(out);admitted=admit(decision,out,plan);lease_check(admitted,dispatch=True);_deadline(deadline,'clip')
        shutil.copyfile(decision,out/'executed-admission.json')
        if packet.sha(out/'executed-admission.json')!=admitted['sha256']:raise ValueError('Admission changed during copy')
        config=dict(prepared=str(out),weights=str(Path(weights).absolute()),decision=str(Path(decision).absolute()),
            parent_deadline=deadline,plan_sha256=packet.sha(out/'plan.json'),admission=admitted)
        report.update(plan_sha256=config['plan_sha256'],source_sha256=plan['source_sha256'],admission=admitted,model_execution=None)
        atomic(out/'metrics.json',report)
        core=launch(dict(config,stage='core'))
        if core.get('model_frozen') is not True or core.get('predictions')!=600 or core.get('solver_updates')!=300 or set(core.get('arms',{}))!=set(video.ARMS):raise ValueError('All six complete frozen-model clips required')
        decoded=launch(dict(config,stage='decode',core_metrics_sha256=packet.sha(out/'core/result/metrics.json')))
        if (set(decoded.get('arms',{}))!=set(video.ARMS) or any(v['images']['frames']!=17 for v in decoded['arms'].values())
            or decoded.get('all196_unchanged') is not True):raise ValueError('All102 decoded frames and original codec records required')
        packet.read_prepared(out);_deadline(deadline,'clip');lease_check(admitted)
        report.update(status='passed',model_execution=True,model_frozen=True,predictions=600,solver_updates=300,frames_per_arm=17,
            quality_assessed=False,native_control=False,child_reports={s:packet.sha(out/s/'result/metrics.json') for s in ('core','decode')},
            child_terminals={s:packet.sha(out/s/'terminal.json') for s in ('core','decode')});return report
    except BaseException as e:
        report.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-began
        terminal=dict(status='complete' if report['status']=='passed' else report['status'],
            exit_code=0 if report['status']=='passed' else (130 if report['status']=='interrupted' else 1),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=report['elapsed_seconds'],
            plan_sha256=report.get('plan_sha256'),cleanup_error=None)
        atomic(out/'terminal.json',terminal);report['terminal_sha256']=packet.sha(out/'terminal.json')
        atomic(out/'metrics.json',report)

def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group()
    for n in ('prepare-fixed','prepare','preflight','execute'):m.add_argument('--'+n,action='store_true')
    m.add_argument('--worker-config',type=Path)
    for n in ('cache-result','previous-visual','fixed-inputs','training-run','training-audit','cpu-report','output','prepared','weights','decision'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.worker_config:return worker(packet.read(a.worker_config))
    if a.prepare_fixed:fn=packet.prepare_fixed;names=('cache_result','previous_visual','output')
    elif a.prepare:fn=packet.prepare;names=('training_run','training_audit','fixed_inputs','cpu_report','output')
    elif a.execute:fn=execute;names=('prepared','weights','decision')
    elif a.preflight:
        plan,*_=packet.read_prepared(a.prepared);print(json.dumps(dict(status='passed',plan_sha256=packet.sha(a.prepared/'plan.json'),model_execution=False)));return
    else:print(json.dumps(dict(status='plan-only',model_execution=False,protocol=packet.protocol(),source_sha256=packet.sources())));return
    if any(getattr(a,n) is None for n in names):p.error('Every explicit input path is required')
    result=fn(**{n:getattr(a,n) for n in names});print(json.dumps(dict(status=result['status'],model_execution=result.get('model_execution',False))))

if __name__=='__main__':main()
