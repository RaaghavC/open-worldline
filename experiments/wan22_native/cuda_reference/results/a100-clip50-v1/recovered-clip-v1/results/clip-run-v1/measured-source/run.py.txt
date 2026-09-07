# SPDX-License-Identifier: Apache-2.0
"""Plan by default. Explicit separate pair/clip modes, with no cloud control."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from safetensors.torch import save_file
from ..official_cpu.inputs import load_inputs,read_exact
from ..official_cpu.streaming import sha,tensor_sha
from .evidence import sources,snapshot,preflight,validate_pair,HERE,PRECISION
from .guards import LIMITS,Monitor,atomic,hardware,stop_child,supervise
from .sampling import SETTINGS,SHAPE,pair,sample


def worker(config):
    out=Path(config['output']);out.mkdir();started=time.monotonic();complete=False;error=None
    report={'status':'running','mode':config['mode'],'stage':config['stage'],'source_sha256':config['source_sha256'],
        'input_identity':config['input_identity'],'settings':SETTINGS,'limits':LIMITS,'predictions':0,'solver_updates':0,
        'actions_read':False,'future_target_read':False,'original_worldline_model':False,
        'model':'External pretrained Wan2.2 TI2V-5B','quality_assessed':False}
    atomic(out/'metrics.json',report)
    try:
        if sources()!=config['source_sha256']:raise ValueError('Source changed after plan')
        report['hardware']=hardware(config['expected_gpu']);atomic(out/'metrics.json',report)
        if config.get('pair_admission')and report['hardware']!=config['pair_admission']['hardware']:
            raise ValueError('Measured CUDA hardware/environment differs from admitted pair')
        with Monitor(out,config['deadline'])as guard:
            values,contexts,identity=load_inputs(config['pair_directory'],config['text_directory'])
            if identity!=config['input_identity']:raise ValueError('Input changed after plan')
            if config['stage']=='core':
                from .native import load_model,predict
                begin=time.monotonic();model,weights=load_model(config['weights'],guard.check)
                torch.cuda.synchronize();report['load_seconds']=time.monotonic()-begin
                atomic(out/'weight-load.json',weights)
                report['precision']=PRECISION
                def call(x,t,context):
                    guard.check();value=predict(model,x,t,context);guard.check();report['predictions']+=1;return value
                if config['mode']=='pair':
                    def partial(label,result):
                        save_file(result,str(out/f'completed-{label}.safetensors'));atomic(out/'metrics.json',report)
                    begin=time.monotonic()
                    output=pair(call,values['initial_latent'],values['token_times'],contexts['atrium'],contexts['native_negative'],values['observation'],partial)
                    report['pair_seconds']=time.monotonic()-begin
                    save_file(output,str(out/'outputs.safetensors'))
                else:
                    with (out/'steps.jsonl').open('x')as stream:
                        def event(i,t,x,result):
                            row={'step':i+1,'timestep':int(t),'prefix_exact':torch.equal(x[:,:1],values['observation'][0]),
                                'latent_sha256':tensor_sha(x),'seconds':time.monotonic()-begin}
                            if not row['prefix_exact']:raise RuntimeError('Clean observed prefix changed')
                            # Each completed step is retained even if a later prediction fails.
                            save_file({'latent':x},str(out/f'step-{i+1:02d}.safetensors'))
                            stream.write(json.dumps(row)+'\n');stream.flush();report['solver_updates']=i+1;atomic(out/'metrics.json',report)
                        begin=time.monotonic();latent=sample(call,values,contexts,event=event)
                    report['sampling_seconds']=time.monotonic()-begin
                    save_file({'latent':latent},str(out/'latents.safetensors'))
                del model;gc.collect();torch.cuda.empty_cache()
                report['input_tensors_unchanged']=all(tensor_sha(v)==identity['input_tensor_sha256'][k]for k,v in values.items())
                if not report['input_tensors_unchanged']:raise RuntimeError('Saved caller tensors changed')
            elif config['stage']=='decode':
                from .decode import load_codec,decode,images
                latent_path=Path(config['latent_file'])
                if sha(latent_path)!=config['latent_sha256']:raise ValueError('Generated latent changed before decode')
                latent=read_exact(latent_path,{'latent':SHAPE})['latent']
                begin=time.monotonic();codec,scales,weights=load_codec(Path(config['weights'])/'Wan2.2_VAE.pth',guard.check)
                report['load_seconds']=time.monotonic()-begin;atomic(out/'weight-load.json',weights)
                begin=time.monotonic();video=decode(codec,scales,latent);guard.check();report['decode_seconds']=time.monotonic()-begin
                save_file({'decoded_rgb':video},str(out/'decoded.safetensors'))
                report['images']=images(video,out)
                report['decoder_cache_clear']=all(v is None for v in codec._feat_map+codec._enc_feat_map)
                if not report['decoder_cache_clear']:raise RuntimeError('Decoder caches retained')
                del codec;gc.collect();torch.cuda.empty_cache()
            else:raise ValueError('Unknown worker stage')
            report['finite_outputs']=True
            report['output_sha256']={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name not in ('metrics.json','memory.jsonl')}
            guard.check();complete=True
    except BaseException as e:error=e;raise
    finally:
        report.update(status='passed'if complete else'interrupted'if isinstance(error,KeyboardInterrupt)else'failed',
            elapsed_seconds=time.monotonic()-started,error_type=type(error).__name__ if error else None,error=str(error)if error else None)
        atomic(out/'metrics.json',report)


def launch(config,out):
    out=Path(out);out.mkdir();atomic(out/'launch.json',config)
    with (out/'worker.log').open('x')as log:
        proc=subprocess.Popen([sys.executable,'-m',__package__+'.run','--worker'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        try:
            proc.stdin.write(json.dumps(config).encode());proc.stdin.close();supervise(proc,out,config['deadline'])
        except BaseException as e:
            stop_child(proc)
            if not (out/'terminal.json').exists():atomic(out/'terminal.json',{'status':'failed','exit_code':proc.returncode,'error':str(e),'stage':'handoff'})
            raise
    terminal=json.loads((out/'terminal.json').read_text());report=json.loads((out/'result/metrics.json').read_text())
    if terminal['status']!='complete'or terminal['exit_code']!=0 or report['status']!='passed'or list(out.rglob('watchdog-stop.json')):
        raise RuntimeError('Child failed; complete and partial evidence retained')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--mode',choices=('pair','clip'));p.add_argument('--expected-gpu')
    for name in ('weights','pair-directory','text-directory','cpu-report','independent-report','output','cuda-pair'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.worker:worker(json.loads(sys.stdin.read(131072)));return
    if any(getattr(a,n)is None for n in ('mode','expected_gpu','weights','pair_directory','text_directory','cpu_report','independent_report','output')):
        p.error('Explicit mode, expected GPU and all input/review/output paths are required')
    if (a.mode=='clip')!=(a.cuda_pair is not None):p.error('Only clip mode requires a completed --cuda-pair')
    if a.output.exists()or a.output.resolve().is_relative_to(HERE):raise ValueError('Fresh output outside the source package required')
    a.output.mkdir(parents=True);started=time.monotonic();report={'status':'running','mode':a.mode,'execute_requested':a.execute,'model_execution':False,'limits':LIMITS}
    atomic(a.output/'metrics.json',report)
    try:
        gates={'cpu_report_sha256':preflight(a.cpu_report),'independent_report_sha256':preflight(a.independent_report,True)}
        values,contexts,identity=load_inputs(a.pair_directory,a.text_directory)
        mapping=snapshot(a.output);report.update(source_sha256=mapping,input_identity=identity,gates=gates)
        for src,name in [(a.pair_directory/'inputs.safetensors','inputs.safetensors'),(a.text_directory/'embeddings.safetensors','contexts.safetensors'),
                         (a.text_directory/'manifest.json','text-manifest.json'),(a.cpu_report,'cpu-report.json'),(a.independent_report,'independent-report.json'),
                         (HERE/'test_independent.py','independent-test.py.txt')]:shutil.copyfile(src,a.output/name)
        report['copied_input_sha256']={name:sha(a.output/name)for name in ('inputs.safetensors','contexts.safetensors','text-manifest.json')}
        admission=validate_pair(a.cuda_pair,mapping,identity,a.expected_gpu)if a.mode=='clip'else None
        report.update(status='planned',pair_admission=admission,plan_checks='Input/source/report hashes and CPU schedule only; no CUDA, parameter values or account access')
        atomic(a.output/'metrics.json',report)
        if not a.execute:return
        config={'mode':a.mode,'expected_gpu':a.expected_gpu,'deadline':started+LIMITS['seconds'],'source_sha256':mapping,
            'input_identity':identity,'pair_admission':admission,**{n:str(getattr(a,n).resolve())for n in ('weights','pair_directory','text_directory')}}
        core=launch(dict(config,stage='core',output=str((a.output/'core/result').resolve())),a.output/'core')
        expected=2 if a.mode=='pair'else 100
        if core['predictions']!=expected or core['solver_updates']!=(0 if a.mode=='pair'else 50):raise RuntimeError('Incomplete prescribed prediction count')
        report['core_report_sha256']=sha(a.output/'core/result/metrics.json')
        report['core_terminal_sha256']=sha(a.output/'core/terminal.json')
        if a.mode=='clip':
            latent=a.output/'core/result/latents.safetensors'
            launch(dict(config,stage='decode',output=str((a.output/'decode/result').resolve()),latent_file=str(latent.resolve()),latent_sha256=sha(latent)),a.output/'decode')
            report['decode_report_sha256']=sha(a.output/'decode/result/metrics.json')
        report.update(status='passed',model_execution=True)
    except BaseException as e:report.update(status='interrupted'if isinstance(e,KeyboardInterrupt)else'failed',error_type=type(e).__name__,error=str(e));raise
    finally:report['elapsed_seconds']=time.monotonic()-started;atomic(a.output/'metrics.json',report)


if __name__=='__main__':main()
