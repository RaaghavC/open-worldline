# SPDX-License-Identifier: Apache-2.0
"""Plan or run exactly one independent official-equation CPU prediction pair."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
import torch
from safetensors.torch import save_file
from .inputs import load_inputs
from .reference import HERE,create_meta,forward,translated_source
from .streaming import ShardSource,StreamedModules,sha,tensor_sha

NAMES=('__init__.py','inputs.py','reference.py','streaming.py','run.py','test_cpu.py',
       'vendor/__init__.py','vendor/model.py','vendor/attention.py','vendor/attention-original.py.txt',
       'vendor/shared_config-original.py.txt','config.json','upstream-provenance.json','source-map.json','LICENSE-APACHE-2.0.txt')
MAX_SECONDS=900.;MAX_BYTES=18*2**30;MIN_AVAILABLE=2*2**30


def hashes():return {name:sha(HERE/name)for name in NAMES}


def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def preflight(path,*,independent=False):
    r=json.loads(Path(path).read_text())
    expected=hashes()
    if independent:expected['test_independent.py']=sha(HERE/'test_independent.py')
    if r.get('status')!='passed'or r.get('tests',0)<(4 if independent else 8)or r.get('source_sha256')!=expected:
        raise ValueError('Completed source-matching CPU report required')
    return sha(path)


def snapshot(out):
    sources=hashes()
    for name,digest in sources.items():
        dst=out/'measured-source'/(name+'.txt');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/name,dst)
        if sha(dst)!=digest:raise RuntimeError('Source changed during snapshot')
    code,translation=translated_source();(out/'translated-model.py.txt').write_text(code);atomic(out/'ast-translation.json',translation)
    return sources


def check_limits(deadline):
    rss=psutil.Process().memory_info().rss;available=psutil.virtual_memory().available
    if time.monotonic()>=deadline or rss>MAX_BYTES or available<MIN_AVAILABLE:
        raise RuntimeError('900 s / 18 GiB / 2 GiB available resource guard reached')


def worker(config):
    out=Path(config['output']);out.mkdir();started=time.monotonic();done=False;error=None
    report={'status':'running','device':'cpu','experiment':'Independent literal-official CPU initial prediction pair',
        'external_model':'Wan2.2 TI2V-5B','original_worldline_model':False,'full_video_generated':False,
        'actions_read':False,'future_target_read':False,'automatic_gpu_execution':False,'no_solver_steps':True,
        'source_sha256':config['source_sha256'],'timings':[],'completed_predictions':0,
        'max_seconds':MAX_SECONDS,'max_rss_bytes':MAX_BYTES,'minimum_available_bytes':MIN_AVAILABLE,
        'outer_autocast':'CPU BF16, cache_enabled=False','parameter_storage':'Original FP32, one module at a time',
        'attention':'CPU BF16 SDPA; not CUDA FlashAttention parity',
        'dependencies':{k:importlib.metadata.version(k)for k in ('torch','diffusers','numpy','safetensors','psutil')}}
    atomic(out/'metrics.json',report)
    check=lambda:check_limits(config['deadline'])
    def measured(label,fn):
        check();begin=time.monotonic();value=fn();check();report['timings'].append({'stage':label,'seconds':time.monotonic()-begin});atomic(out/'metrics.json',report);return value
    try:
        if config['source_sha256']!=hashes():raise ValueError('Sources changed after parent gate')
        torch.set_num_threads(2)
        values,contexts,identity=load_inputs(config['pair_directory'],config['text_directory'])
        if identity!=config['input_identity']:raise ValueError('Inputs changed after plan')
        model,translation=create_meta();report['ast_translation']=translation
        source=measured('verify original shards before any parameter values',lambda:ShardSource(config['weights'],model,
            json.loads((HERE/'upstream-provenance.json').read_text()),check=check))
        velocities={};report['passes']=[]
        with (out/'ownership.jsonl').open('x')as events:
            for label,key in [('positive','atrium'),('negative','native_negative')]:
                def event(row):events.write(json.dumps(dict(prediction=label,**row))+'\n');events.flush()
                def one():
                    with StreamedModules(model,source,event=event,check=check)as stream:
                        value=forward(model,values['initial_latent'],values['token_times'],contexts[key],720)
                        summary=stream.completed()
                    if stream.handles or any(p.device.type!='meta'for p in model.parameters()):raise RuntimeError('Prediction cleanup incomplete')
                    return value,summary
                value,summary=measured(label+' official CPU forward with module weight reads',one)
                if value.dtype!=torch.float32 or value.shape!=(48,5,18,32)or not torch.isfinite(value).all():raise FloatingPointError('Invalid official prediction')
                velocities[label+'_velocity']=value.contiguous();report['passes'].append(dict(prediction=label,**summary));report['completed_predictions']+=1
                save_file(velocities,str(out/('completed-'+label+'.safetensors')))
                atomic(out/'metrics.json',report)
        guided=velocities['negative_velocity']+5.*(velocities['positive_velocity']-velocities['negative_velocity'])
        if not torch.isfinite(guided).all():raise FloatingPointError('Guided velocity is nonfinite')
        velocities['guided_velocity']=guided.contiguous();save_file(velocities,str(out/'outputs.safetensors'))
        for k,v in values.items():
            if tensor_sha(v)!=identity['input_tensor_sha256'][k]:raise RuntimeError('Retained caller input was mutated')
        if len(source.records)!=825:raise RuntimeError('Every official tensor must have been used')
        atomic(out/'weights-used.json',{'original_fp32_tensors':source.records,'count':len(source.records),'all_shards_verified':True})
        report.update(finite_outputs=True,input_identity=identity,caller_inputs_unchanged=True,
            output_sha256={name:sha(out/name)for name in ['outputs.safetensors','completed-positive.safetensors','completed-negative.safetensors','ownership.jsonl','weights-used.json']})
        check();done=True
    except BaseException as e:error=e;raise
    finally:
        report.update(status='passed'if done else'interrupted'if isinstance(error,KeyboardInterrupt)else'failed',
                      error_type=type(error).__name__ if error else None,error=str(error)if error else None,
                      elapsed_seconds=time.monotonic()-started)
        atomic(out/'metrics.json',report)


def stop_child(proc):
    if proc.poll()is None:
        proc.terminate()
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)


def supervise(proc,out,deadline,*,interval=.25):
    """Parent enforces combined process RSS and kills the child on every failure."""
    started=time.monotonic();error=None;stopped=False;peak=0;minimum=None
    try:
        with (out/'memory.jsonl').open('x')as stream:
            while proc.poll()is None:
                try:
                    root=psutil.Process();rss=root.memory_info().rss+sum(p.memory_info().rss for p in root.children(recursive=True)if p.is_running())
                except psutil.NoSuchProcess:continue
                available=psutil.virtual_memory().available;peak=max(peak,rss);minimum=available if minimum is None else min(minimum,available)
                row={'seconds':time.monotonic()-started,'combined_rss_bytes':rss,'available_bytes':available}
                stream.write(json.dumps(row)+'\n');stream.flush()
                if time.monotonic()>=deadline or rss>MAX_BYTES or available<MIN_AVAILABLE:
                    atomic(out/'watchdog-stop.json',{'status':'stopped','sample':row});stopped=True;raise RuntimeError('Parent resource guard stopped worker')
                time.sleep(interval)
    except BaseException as e:error=e;raise
    finally:
        stop_child(proc)
        atomic(out/'terminal.json',{'status':'complete'if proc.returncode==0 and error is None and not stopped else'failed',
             'exit_code':proc.returncode,'error_type':type(error).__name__ if error else None,'error':str(error)if error else None,
             'elapsed_seconds':time.monotonic()-started,'peak_combined_rss_bytes':peak,'minimum_available_bytes':minimum})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--execute',action='store_true')
    for name in ('weights','pair-directory','text-directory','cpu-report','independent-report','output'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.worker:worker(json.loads(sys.stdin.read(65536)));return
    if any(getattr(a,n)is None for n in ('weights','pair_directory','text_directory','cpu_report','independent_report','output')):p.error('All paths and both source-bound reviews are required')
    if a.output.exists()or a.output.resolve().is_relative_to(HERE):raise ValueError('Fresh output outside source package required')
    a.output.mkdir(parents=True);started=time.monotonic();report={'status':'running','execute_requested':a.execute,'weights_loaded':False,'model_execution':False}
    atomic(a.output/'metrics.json',report)
    try:
        gates={'author_cpu_report_sha256':preflight(a.cpu_report),'independent_report_sha256':preflight(a.independent_report,independent=True)}
        values,contexts,identity=load_inputs(a.pair_directory,a.text_directory)
        sources=snapshot(a.output)
        for src,name in [(a.pair_directory/'inputs.safetensors','inputs.safetensors'),(a.text_directory/'embeddings.safetensors','contexts.safetensors'),
                         (a.text_directory/'manifest.json','text-manifest.json'),(a.cpu_report,'cpu-report.json'),(a.independent_report,'independent-report.json'),
                         (HERE/'test_independent.py','independent-test.py.txt')]:shutil.copyfile(src,a.output/name)
        model,_=create_meta();ShardSource(a.weights,model,json.loads((HERE/'upstream-provenance.json').read_text()),verify_bytes=False);del model
        config={'output':str((a.output/'result').resolve()),'deadline':started+MAX_SECONDS,'source_sha256':sources,'input_identity':identity,
                **{n:str(getattr(a,n).resolve())for n in ('weights','pair_directory','text_directory')}}
        report.update(status='planned',source_sha256=sources,input_identity=identity,gates=gates,
            pair_predictions=2,solver_steps=0,input_file_sha256=sha(a.output/'inputs.safetensors'),
            plan_checks='Pinned config/index and shard sizes/headers only; all shard bytes are verified in worker before tensor values')
        atomic(a.output/'metrics.json',report)
        if not a.execute:return
        atomic(a.output/'launch.json',config)
        with (a.output/'worker.log').open('x')as log:
            proc=subprocess.Popen([sys.executable,'-m',__package__+'.run','--worker'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
            try:
                proc.stdin.write(json.dumps(config).encode());proc.stdin.close();supervise(proc,a.output,config['deadline'])
            except BaseException as error:
                stop_child(proc)
                if not (a.output/'terminal.json').exists():
                    atomic(a.output/'terminal.json',{'status':'failed','exit_code':proc.returncode,
                        'error_type':type(error).__name__,'error':str(error),'stage':'worker handoff'})
                raise
        terminal=json.loads((a.output/'terminal.json').read_text());result=json.loads((a.output/'result/metrics.json').read_text())
        if terminal['status']!='complete'or result['status']!='passed'or result['completed_predictions']!=2:raise RuntimeError('Reference pair failed; evidence retained')
        report.update(status='passed',weights_loaded=True,model_execution=True,result_metrics_sha256=sha(a.output/'result/metrics.json'))
    except BaseException as e:report.update(status='interrupted'if isinstance(e,KeyboardInterrupt)else'failed',error_type=type(e).__name__,error=str(e));raise
    finally:report['elapsed_seconds']=time.monotonic()-started;atomic(a.output/'metrics.json',report)


if __name__=='__main__':main()
