"""Work-only plan-default matched trained-adapter visual evaluation. No cloud API."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path.cwd()))
import torch
from safetensors.torch import save_file
import inputs as inp
import sampler
from experiments.wan22_native.action_cuda import probe
from experiments.wan22_native.action_training.objective import parameter_records
from experiments.wan22_native.cuda_reference.native import load_model
from experiments.wan22_native.spatial_reference.codec import load_codec, decode, images
from experiments.wan22_native.spatial_reference.output import save_rgb, read_tensors
from experiments.wan22_native.spatial_reference.config import SPECS
from experiments.wan22_native.spatial_reference.guards import atomic, hardware, Monitor, supervise, stop_child, limits

# The new plan binds a later independently reviewed actual128 audit.


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


def admit(path,out,plan):
    value=inp.read(path)
    required={'schema':'worldline-action-cuda-visual-admission-v1','decision':'admit','issued_by':'parent-agent',
        'scope':inp.SCOPE,'plan_sha256':inp.sha(out/'plan.json'),'source_sha256':plan['source_sha256'],
        'input_identity':plan['input_identity'],'artifacts':plan['artifacts'],'limits':limits('clip'),
        'settings':plan['settings'],'profile':'spatial','arms':['closed','open'],'training_admitted':False,
        'final128_audit_sha256':plan['training_audit_sha256'],'cpu_report_sha256':plan['cpu_report_sha256']}
    if any(value.get(k)!=v for k,v in required.items()) or not isinstance(value.get('reason'),str) or not value['reason'].strip():
        raise ValueError('Root must admit this exact matched visual plan and passed training audit')
    return {'sha256':inp.sha(path),'scope':inp.SCOPE}


def sample_arms(core,out,checkpoint,checkpoint_sha,values,contexts,commands,check):
    """One loaded frozen foundation, two isolated calls to the exact reviewed sampler."""
    rows={}; initial={k:sampler.tensor_sha(v) for k,v in values.items()}
    texts={k:sampler.tensor_sha(v) for k,v in contexts.items()}
    for arm in ('closed','open'):
        folder=out/arm; folder.mkdir(); began=time.monotonic(); count=0
        with (folder/'steps.jsonl').open('x') as stream:
            def event(index,t,latent,velocities):
                nonlocal count
                if index!=count or not torch.equal(latent[:,:1],values['observation'][0]):
                    raise ValueError('Step order or exact observed prefix changed')
                save_file({'latent':latent},str(folder/f'step-{index+1:02d}.safetensors'))
                if index==0: save_file(velocities,str(folder/'initial-velocities.safetensors'))
                row={'step':index+1,'timestep':int(t),'prefix_exact':True,'latent_sha256':sampler.tensor_sha(latent),
                     'seconds':time.monotonic()-began}
                stream.write(json.dumps(row)+'\n');stream.flush();count+=1;check()
            latent,record=sampler.sample_checkpoint(core,checkpoint,checkpoint_sha,
                {k:v.clone() for k,v in values.items()},{k:v.clone() for k,v in contexts.items()},commands[arm].clone(),
                profile='spatial',event=event,check=check)
        if count!=50 or record['predictions']!=100 or record['solver_updates']!=50:
            raise RuntimeError('Every complete arm requires 100 predictions and 50 saved updates')
        save_file({'latent':latent},str(folder/'latents.safetensors'))
        record.update(seconds=time.monotonic()-began,commands_label='wait then 15 left turns' if arm=='closed' else 'interact then 15 left turns')
        atomic(folder/'metrics.json',record); rows[arm]=record
        if initial!={k:sampler.tensor_sha(v) for k,v in values.items()} or texts!={k:sampler.tensor_sha(v) for k,v in contexts.items()}:
            raise RuntimeError('Shared native inputs changed between arms')
    return rows


def worker(config):
    root=inp.root(config['prepared']); out=root/config['stage']/'result'; out.mkdir()
    began=time.monotonic(); report={'schema':inp.SCHEMA,'status':'running','stage':config['stage'],
        'model_execution':False,'training':False,'future_targets_materialized':False,'limits':limits('clip')}
    try:
        plan,values,contexts,commands=inp.read_prepared(root)
        decision=admit(config['decision'],root,plan)
        if inp.sha(root/'plan.json')!=config['plan_sha256'] or decision!=config['admission']: raise ValueError('Launch identity changed')
        actual=hardware(plan['expected_gpu'])
        comparison=require_hardware(actual,plan['input_identity']['training']['hardware'])
        flags=probe._runtime_flags()
        report.update(plan_sha256=config['plan_sha256'],source_sha256=plan['source_sha256'],hardware=actual,
                      hardware_comparison=comparison,runtime_flags=flags)
        atomic(out/'metrics.json',report)
        with Monitor(out,config['deadline'],'clip') as monitor:
            def check():
                monitor.check()
                if probe._runtime_flags()!=flags: raise RuntimeError('Native runtime flags changed')
            if config['stage']=='core':
                # No ambient inference_mode: the unchanged bridge requires normal tensors.
                started=time.monotonic(); core,loaded=load_model(config['weights'],check); torch.cuda.synchronize()
                report['load_seconds']=time.monotonic()-started; report['model_execution']=True
                atomic(out/'weight-load.json',loaded); atomic(out/'metrics.json',report)
                expected=probe._original_weights(loaded)
                if expected!=plan['input_identity']['training']['core_records']: raise ValueError('Actual original foundation differs from training')
                started=time.monotonic(); before=parameter_records(core,check=check,expected=expected)
                report['before_hash_seconds']=time.monotonic()-started; atomic(out/'core-before.json',before)
                report['arms']=sample_arms(core,out,root/'adapter.safetensors',plan['artifacts']['adapter.safetensors'],values,contexts,commands,check)
                started=time.monotonic(); after=parameter_records(core,check=check,expected=before)
                report['after_hash_seconds']=time.monotonic()-started; atomic(out/'core-after.json',after)
                report.update(model_frozen=True,predictions=200,solver_updates=100,
                    model_frozen_definition='All 825 foundation parameter values unchanged; adapter bytes unchanged and every parameter gradient None; no optimizer')
                del core;gc.collect();torch.cuda.empty_cache()
            elif config['stage']=='decode':
                core_report=inp.read(inp.file(root,'core/result/metrics.json'))
                if inp.sha(root/'core/result/metrics.json')!=config['core_metrics_sha256'] or core_report.get('model_frozen') is not True:
                    raise ValueError('Completed frozen core result required before decoding')
                inp.checked_outputs(root/'core/result',core_report['output_sha256'])
                started=time.monotonic(); model,scale,loaded=load_codec(Path(config['weights'])/'Wan2.2_VAE.pth',check)
                report['load_seconds']=time.monotonic()-started;report['model_execution']=True
                atomic(out/'weight-load.json',loaded); inp.se._codec_weights(out/'weight-load.json'); atomic(out/'metrics.json',report)
                report['arms']={}; selected=SPECS['spatial']
                for arm in ('closed','open'):
                    folder=out/arm;folder.mkdir(); path=root/'core/result'/arm/'latents.safetensors'
                    if inp.sha(path)!=core_report['output_sha256'][arm+'/latents.safetensors']: raise ValueError('Generated latent changed')
                    latent=read_tensors(path,{'latent':selected.latent_shape})['latent']
                    started=time.monotonic();video=decode(model,scale,latent,selected.height,selected.width);check()
                    row={'decode_seconds':time.monotonic()-started,'latent_sha256':sampler.tensor_sha(latent)}
                    save_rgb(video,folder/'rgb',selected,purpose='generated_clip')
                    row['images']=images(video,folder,selected.height,selected.width)
                    row['cache_clear']=all(v is None for v in model._feat_map+model._enc_feat_map)
                    if not row['cache_clear']: raise RuntimeError('Native VAE retained state across branches')
                    report['arms'][arm]=row;atomic(out/'metrics.json',report);del video,latent
                del model;gc.collect();torch.cuda.empty_cache()
            else: raise ValueError('Only core and decode stages exist')
            check()
        inp.read_prepared(root)
        if time.monotonic()>=config['deadline']: raise RuntimeError('Deadline reached during output verification')
        report.update(status='passed',output_sha256={str(p.relative_to(out)):inp.sha(p) for p in sorted(out.rglob('*'))
                      if p.is_file() and p != out/'metrics.json' and p.name != 'memory.jsonl'})
        return report
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-began;atomic(out/'metrics.json',report)


def launch(config):
    root=Path(config['prepared']);out=root/config['stage'];out.mkdir();atomic(out/'launch.json',config);proc=None
    try:
        if time.monotonic()>=config['deadline']: raise RuntimeError('Deadline reached before child launch')
        with (out/'worker.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,str(inp.HERE/'run.py'),'--worker-config',str(out/'launch.json')],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            supervise(proc,out,config['deadline'],'clip')
    except BaseException as error:
        cleanup_error=None
        try:
            if proc is not None:stop_child(proc)
        except BaseException as cleanup:cleanup_error=str(cleanup)
        finally:
            if not (out/'terminal.json').exists():
                atomic(out/'terminal.json',{'status':'failed','exit_code':proc.returncode if proc else None,
                       'error':str(error),'cleanup_error':cleanup_error})
            elif cleanup_error:atomic(out/'handoff-cleanup-error.json',{'error':str(error),'cleanup_error':cleanup_error})
        raise
    terminal=inp.read(out/'terminal.json');report=inp.read(out/'result/metrics.json')
    if (terminal.get('status')!='complete' or type(terminal.get('exit_code')) is not int or terminal['exit_code']!=0
        or terminal.get('cleanup_error') is not None or report.get('status')!='passed'
        or report.get('plan_sha256')!=config['plan_sha256'] or list(out.rglob('watchdog-stop.json'))):
        raise RuntimeError('Guarded visual worker failed; retain partial evidence')
    inp.checked_outputs(out/'result',report['output_sha256'])
    return report


def execute(prepared,weights,decision):
    out=inp.root(prepared)
    if any((out/name).exists() for name in ('execution-attempt.json','core','decode')): raise ValueError('Visual execution is single-use')
    with (out/'execution-attempt.json').open('x') as stream:stream.write('single-use\n')
    began=time.monotonic();report={'schema':inp.SCHEMA,'status':'running','model_execution':False,'training':False,'limits':limits('clip')}
    try:
        plan,*_=inp.read_prepared(out);admitted=admit(decision,out,plan)
        shutil.copyfile(decision,out/'executed-admission.json')
        if inp.sha(out/'executed-admission.json')!=admitted['sha256']:raise ValueError('Admission changed during copy')
        config={'prepared':str(out),'weights':str(Path(weights).absolute()),'decision':str(Path(decision).absolute()),
                'deadline':began+limits('clip')['seconds'],'plan_sha256':inp.sha(out/'plan.json'),'admission':admitted}
        report.update(plan_sha256=config['plan_sha256'],source_sha256=plan['source_sha256'],admission=admitted,model_execution=None)
        atomic(out/'metrics.json',report)
        core=launch(dict(config,stage='core'))
        if core.get('model_frozen') is not True or core.get('predictions')!=200 or core.get('solver_updates')!=100:
            raise ValueError('Both full frozen-model arms must complete before decoding')
        decoded=launch(dict(config,stage='decode',core_metrics_sha256=inp.sha(out/'core/result/metrics.json')))
        if set(decoded.get('arms',{}))!={'closed','open'} or any(r['images']['frames']!=17 for r in decoded['arms'].values()):
            raise ValueError('Retain every frame of both complete arms')
        inp.read_prepared(out)
        if time.monotonic()>=config['deadline']:raise RuntimeError('Combined deadline reached')
        report.update(status='passed',model_execution=True,model_frozen=True,predictions=200,solver_updates=100,
            frames_per_arm=17,quality_assessed=False,
            child_reports={stage:inp.sha(out/stage/'result/metrics.json') for stage in ('core','decode')},
            child_terminals={stage:inp.sha(out/stage/'terminal.json') for stage in ('core','decode')})
        return report
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:report['elapsed_seconds']=time.monotonic()-began;atomic(out/'metrics.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    for name in ('worker-config','training-run','cache-run','baseline-run','cpu-report','training-audit','output','prepared','weights','decision'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.worker_config:return worker(inp.read(args.worker_config))
    names=('prepared','weights','decision') if args.execute else ('training_run','cache_run','baseline_run','cpu_report','training_audit','output')
    if any(getattr(args,n) is None for n in names):parser.error('Supply each explicit stage input path')
    result=(execute if args.execute else inp.prepare)(**{n:getattr(args,n) for n in names});print(json.dumps(result))


if __name__=='__main__':main()
