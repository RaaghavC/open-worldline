"""Fourteen fixed-input denoiser predictions. No optimizer, clip or cloud call."""
import argparse,json,shutil,sys,time,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(HERE/'visual'))
import inputs as vi
import run as visual
import sampler
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from experiments.wan22_native.action_cuda import probe,probe_math as math,bridge as bridge_module
from experiments.wan22_native.action_training.objective import parameter_records
from experiments.wan22_native.cuda_reference.native import load_model
from experiments.wan22_native.spatial_reference.guards import atomic,hardware,Monitor,supervise,stop_child,limits

SCHEMA='worldline-action-fixed-input-diagnostic-v1'


def sources():
    return {'visual':vi.sources(),'diagnostic':{n:vi.sha(HERE/n) for n in ('diagnostic.py','test_diagnostic.py')}}


def review(path):
    r=vi.read(path)
    if r.get('status')!='passed' or r.get('source_sha256')!=sources() or r.get('tests',0)<3 or any(r.get(k)!=0 for k in ('failures','errors','skipped')):
        raise ValueError('Current diagnostic CPU review required')
    return vi.sha(path)


def prepare(visual_prepared,training_run,cache_run,cpu_report,output):
    out=vi.root(output,fresh=True);out.mkdir(parents=True)
    status={'schema':SCHEMA,'status':'preparing','model_execution':False}
    try:
        cpu_sha=review(cpu_report); mapping=sources()
        old,values,contexts,commands=vi.read_prepared(visual_prepared)
        trained,_,identity=vi.training(training_run)
        if identity!=old['input_identity']['training']:raise ValueError('Visual/training identity mismatch')
        root=Path(training_run); measured=vi.read(root/'training/metrics.json');first=measured['schedule'][0]
        if first['k']!=506 or first['start']!=0:raise ValueError('Use the retained first training draw k506')
        with safe_open(str(root/'training/fresh-draws.safetensors'),framework='pt',device='cpu') as f:noise=f.get_tensor('noise_0000')
        if sampler.tensor_sha(noise)!=first['noise_sha256']:raise ValueError('Exact training noise required')
        cases={};provenance={};raw={'noise':noise,'observation':values['observation']}
        for arm in ('closed','open'):
            window,source=vi.data.read_window(Path(cache_run)/'result',arm+'-0000')
            expected=trained['input_identity']['cache']['windows'][arm+'-0000']['tensor_sha256']
            if any(sampler.tensor_sha(v)!=expected[k] for k,v in window.items()):raise ValueError('Original target cache differs')
            noisy,times,target=math.flow_inputs(window['target'],window['observation'],noise,506)
            if math.condition_identity(noisy,times,target,window)!=next(r['input_sha256'] for r in measured['updates'][0]['branches'] if r['branch']==arm):raise ValueError('Recreated first training input differs bytewise')
            cases.update({arm+'_noisy':noisy,arm+'_times':times,arm+'_target':target});provenance[arm]=source;raw[arm+'_target']=window['target']
            if not torch.equal(window['observation'],values['observation']) or not torch.equal(window['commands'],commands[arm]):raise ValueError('Training and visual conditioning differ')
        save_file(cases,str(out/'objective-inputs.safetensors'))
        save_file(raw,str(out/'original-training-inputs.safetensors'))
        initial=root/'training/checkpoint-0000/adapter.safetensors'
        init_record=vi.read(root/'training/checkpoint-0000/manifest.json')
        if init_record['completed_updates']!=0:raise ValueError('Only original checkpoint zero is accepted')
        zero,zero_id=sampler.load_adapter_checkpoint(initial,init_record['files']['adapter.safetensors'])
        if torch.count_nonzero(zero.output.weight) or torch.count_nonzero(zero.output.bias):raise ValueError('Checkpoint zero output must be zero')
        del zero;shutil.copyfile(initial,out/'checkpoint-0000.safetensors')
        shutil.copytree(visual_prepared,out/'visual-inputs',ignore=shutil.ignore_patterns('__pycache__'))
        # The original prepared metadata is sufficient, independent of any later visual attempt.
        shutil.copyfile(cpu_report,out/'cpu-report.json')
        for n in ('diagnostic.py','test_diagnostic.py'):shutil.copyfile(HERE/n,out/(n+'.txt'))
        plan={'schema':SCHEMA,'status':'prepared','source_sha256':mapping,'cpu_report_sha256':cpu_sha,
            'visual_plan_sha256':vi.sha(Path(visual_prepared)/'plan.json'),'training_parent_sha256':identity['parent_sha256'],
            'objective_input_sha256':vi.sha(out/'objective-inputs.safetensors'),'original_training_input_sha256':vi.sha(out/'original-training-inputs.safetensors'),'checkpoint_zero':zero_id,
            'first_draw':first,'first_training_update':measured['updates'][0],'cache_provenance':provenance,'predictions':14,'limits':limits('pair'),
            'scope':'Fixed input learning and separate pure-noise action/CFG diagnostic; no training or generation',
            'future_target_use':'Only own-corruption objective panel; causal k999 panel uses no future target',
            'same_input_causal_panel':True}
        atomic(out/'plan.json',plan);read_prepared(out);status.update(status='prepared',plan_sha256=vi.sha(out/'plan.json'));return status
    except BaseException as e:status.update(status='failed',error=str(e));raise
    finally:atomic(out/'metrics.json',status)


def read_prepared(out):
    out=vi.root(out);plan=vi.read(out/'plan.json')
    if plan.get('schema')!=SCHEMA or plan.get('status')!='prepared' or plan.get('source_sha256')!=sources() or plan.get('limits')!=limits('pair'):
        raise ValueError('Exact current diagnostic plan required')
    if review(out/'cpu-report.json')!=plan['cpu_report_sha256']:raise ValueError('CPU report differs')
    for n,h in plan['source_sha256']['diagnostic'].items():
        if vi.sha(out/(n+'.txt'))!=h:raise ValueError('Diagnostic snapshot differs')
    old,values,contexts,commands=vi.read_prepared(out/'visual-inputs')
    if vi.sha(out/'visual-inputs/plan.json')!=plan['visual_plan_sha256']:raise ValueError('Visual input plan differs')
    shape=(1,48,5,44,78);spec={}
    for arm in ('closed','open'):
        spec.update({arm+'_noisy':(shape,'F32'),arm+'_target':(shape,'F32'),arm+'_times':((1,4290),'I64')})
    cases=vi.pe.tensors(out/'objective-inputs.safetensors',spec,plan['objective_input_sha256'])
    raw=vi.pe.tensors(out/'original-training-inputs.safetensors',{'noise':(shape,'F32'),'closed_target':(shape,'F32'),'open_target':(shape,'F32'),'observation':((1,48,1,44,78),'F32')},plan['original_training_input_sha256'])
    if sampler.tensor_sha(raw['noise'])!=plan['first_draw']['noise_sha256'] or not torch.equal(raw['observation'],values['observation']):raise ValueError('Exact original draw or observation differs')
    for arm in ('closed','open'):
        expected=old['input_identity']['training']['training_input_identity']['cache']['windows'][arm+'-0000']['tensor_sha256']
        if sampler.tensor_sha(raw[arm+'_target'])!=expected['target']:raise ValueError('Original target identity differs')
        formed=math.flow_inputs(raw[arm+'_target'],raw['observation'],raw['noise'],506)
        if any(not torch.equal(cases[arm+'_'+key],value) for key,value in zip(('noisy','times','target'),formed)):raise ValueError('Original fixed corruption differs')
    zero,record=sampler.load_adapter_checkpoint(out/'checkpoint-0000.safetensors',plan['checkpoint_zero']['checkpoint_sha256']);del zero
    if record!=plan['checkpoint_zero']:raise ValueError('Checkpoint zero values differ')
    return plan,old,values,contexts,commands,cases


def contrast(a,b):
    # Only generated future latent frames; retained raw predictions include the known prefix.
    a=a[:,:,1:];b=b[:,:,1:]
    delta=a.double()-b.double();den=float(b.double().norm());norm=float(delta.norm())
    return {'max_absolute':float(delta.abs().max()),'rms':float(delta.square().mean().sqrt()),'relative_l2':norm/den if den else None}


def evaluate(native,adapt,cases,values,contexts,commands,retain,check):
    """Injectable predictor seam; every call is a retained fixed CPU tensor input."""
    report={'predictions':0,'objective':{},'causal':{},'calls':[],'contrast_region':'four future latent frames; known prefix excluded'}
    def record(name,v,x,t,c,a=None,checkpoint='native'):
        retain(name,v)
        report['calls'].append({'name':name,'checkpoint':checkpoint,'input_sha256':sampler.tensor_sha(x),'times_sha256':sampler.tensor_sha(t),'context_sha256':sampler.tensor_sha(c),'commands_sha256':sampler.tensor_sha(a) if a is not None else None,'prediction_sha256':sampler.tensor_sha(v)})
        if v.shape!=x.shape or v.dtype!=torch.float32 or v.device.type!='cpu' or not torch.isfinite(v).all():raise FloatingPointError('Malformed or nonfinite prediction')
    initial=values['initial_latent'][None];times=values['token_times'];outputs={}
    for text in ('atrium','native_negative'):
        check();v=native(initial.clone(),times.clone(),contexts[text].clone());record('native-'+text,v,initial,times,contexts[text])
        if not torch.isfinite(v).all():raise FloatingPointError('Native prediction nonfinite')
        outputs['native-'+text]=v;report['predictions']+=1
    for checkpoint in ('0','16'):
        report['objective'][checkpoint]={}
        for arm in ('closed','open'):
            check();v=adapt(checkpoint,cases[arm+'_noisy'].clone(),cases[arm+'_times'].clone(),contexts['atrium'].clone(),commands[arm].clone())
            record('objective-'+checkpoint+'-'+arm,v,cases[arm+'_noisy'],cases[arm+'_times'],contexts['atrium'],commands[arm],checkpoint);report['predictions']+=1
            loss=math.future_flow_mse(v,cases[arm+'_target'])
            if not torch.isfinite(loss):raise FloatingPointError('Objective nonfinite')
            report['objective'][checkpoint][arm]=float(loss)
        for arm in ('closed','open'):
            for text in ('atrium','native_negative'):
                check();v=adapt(checkpoint,initial.clone(),times.clone(),contexts[text].clone(),commands[arm].clone())
                record('causal-'+checkpoint+'-'+arm+'-'+text,v,initial,times,contexts[text],commands[arm],checkpoint);report['predictions']+=1
                if not torch.isfinite(v).all():raise FloatingPointError('Causal prediction nonfinite')
                if checkpoint=='0' and not math.compare_velocity(outputs['native-'+text],v)['passed']:
                    raise ValueError('Zero checkpoint no longer matches native on the causal input')
                outputs[checkpoint+'-'+arm+'-'+text]=v
        dp=outputs[checkpoint+'-open-atrium']-outputs[checkpoint+'-closed-atrium']
        dn=outputs[checkpoint+'-open-native_negative']-outputs[checkpoint+'-closed-native_negative']
        gopen=outputs[checkpoint+'-open-native_negative']+5*(outputs[checkpoint+'-open-atrium']-outputs[checkpoint+'-open-native_negative'])
        gclosed=outputs[checkpoint+'-closed-native_negative']+5*(outputs[checkpoint+'-closed-atrium']-outputs[checkpoint+'-closed-native_negative'])
        retain('command-delta-'+checkpoint+'-positive',dp);retain('command-delta-'+checkpoint+'-negative',dn)
        report['causal'][checkpoint]={'positive_command_effect':contrast(outputs[checkpoint+'-open-atrium'],outputs[checkpoint+'-closed-atrium']),
            'negative_command_effect':contrast(outputs[checkpoint+'-open-native_negative'],outputs[checkpoint+'-closed-native_negative']),
            'guided_command_effect':contrast(gopen,gclosed),
            'positive_residual_vs_native':contrast(outputs[checkpoint+'-closed-atrium'],outputs['native-atrium']),
            'negative_residual_vs_native':contrast(outputs[checkpoint+'-closed-native_negative'],outputs['native-native_negative']),
            'guided_delta_decomposition_max_roundoff':float(((gopen-gclosed)-(5*dp-4*dn)).abs().max())}
    report['relative_objective_improvement']={arm:1-report['objective']['16'][arm]/report['objective']['0'][arm] if report['objective']['0'][arm] else None for arm in ('closed','open')}
    if report['predictions']!=14:raise RuntimeError('Exactly fourteen predictions required')
    return report


def admission(path,out,plan):
    record=vi.read(path)
    wanted={'schema':'worldline-action-fixed-input-diagnostic-admission-v1','decision':'admit','issued_by':'parent-agent',
        'plan_sha256':vi.sha(out/'plan.json'),'source_sha256':plan['source_sha256'],'limits':limits('pair'),
        'training_admitted':False,'generation_admitted':False,'predictions':14,'cpu_report_sha256':plan['cpu_report_sha256']}
    if any(record.get(k)!=v for k,v in wanted.items()):raise ValueError('Exact-plan root diagnostic admission required')
    return vi.sha(path)


def worker(config):
    out=Path(config['prepared']);result=out/'result';result.mkdir();began=time.monotonic();report={'schema':SCHEMA,'status':'running','model_execution':False}
    try:
        plan,old,values,contexts,commands,cases=read_prepared(out)
        if admission(config['decision'],out,plan)!=config['admission_sha256'] or vi.sha(out/'plan.json')!=config['plan_sha256']:raise ValueError('Launch changed')
        actual=hardware(old['expected_gpu']);compatibility=visual.require_hardware(actual,old['input_identity']['training']['hardware'])
        report.update(hardware=actual,hardware_comparison=compatibility,plan_sha256=config['plan_sha256'],source_sha256=sources(),limits=limits('pair'),completed_prediction_files=0)
        atomic(result/'metrics.json',report)
        with Monitor(result,config['deadline'],'pair') as guard:
            flags=probe._runtime_flags();report['runtime_flags']=flags
            def check():
                guard.check()
                if probe._runtime_flags()!=flags:raise ValueError('Runtime flags changed')
            start=time.monotonic();core,loaded=load_model(config['weights'],check);report['load_seconds']=time.monotonic()-start;report['model_execution']=True
            atomic(result/'weight-load.json',loaded);atomic(result/'metrics.json',report)
            start=time.monotonic();before=parameter_records(core,check=check,expected=probe._original_weights(loaded));report['before_hash_seconds']=time.monotonic()-start;atomic(result/'core-before.json',before)
            bridges={};identities={}
            for name,path,digest in [('0',out/'checkpoint-0000.safetensors',plan['checkpoint_zero']['checkpoint_sha256']),('16',out/'visual-inputs/adapter.safetensors',old['artifacts']['adapter.safetensors'])]:
                adapter,identity=sampler.load_adapter_checkpoint(path,digest);adapter.to('cuda:0');identities[name]=identity
                bridges[name]=bridge_module.NativeCUDAActionBridge(core,adapter,profile='spatial')
            def native(x,t,c):return probe.native_reference(core,x,t,c)
            def adapt(name,x,t,c,a):
                with torch.no_grad():v=bridges[name](x.to('cuda:0'),t.to('cuda:0'),[c.to('cuda:0')],commands=a.to('cuda:0'),observation=values['observation'].to('cuda:0'),track_grad=False)
                torch.cuda.synchronize();return v.detach().cpu()
            def retain(name,value):
                save_file({'velocity':value.detach().cpu().contiguous()},str(result/(name+'.safetensors')))
                if not name.startswith('command-delta-'):report['completed_prediction_files']+=1
                report['last_retained']=name;atomic(result/'metrics.json',report)
            report.update(evaluate(native,adapt,cases,values,contexts,commands,retain,check))
            start=time.monotonic();after=parameter_records(core,check=check,expected=before);report['after_hash_seconds']=time.monotonic()-start;atomic(result/'core-after.json',after)
            for name,b in bridges.items():
                if any(p.grad is not None for p in b.adapter.parameters()) or {k:sampler.tensor_sha(v) for k,v in b.adapter.state_dict().items()}!=identities[name]['tensor_sha256']:
                    raise ValueError('Adapter changed during inference')
            report['model_frozen']=True;check()
        read_prepared(out)
        if time.monotonic()>=config['deadline']:raise RuntimeError('Worker deadline reached')
        report.update(status='passed',output_sha256={str(p.relative_to(result)):vi.sha(p) for p in result.rglob('*') if p.is_file() and p.name not in ('metrics.json','memory.jsonl')})
    except BaseException as e:report.update(status='failed',error=str(e));raise
    finally:report['elapsed_seconds']=time.monotonic()-began;atomic(result/'metrics.json',report)


def execute(prepared,weights,decision):
    out=vi.root(prepared)
    with (out/'execution-attempt.json').open('x') as f:f.write('single-use\n')
    began=time.monotonic();proc=None;parent={'schema':SCHEMA,'status':'running','model_execution':False}
    try:
        plan,*_=read_prepared(out);digest=admission(decision,out,plan)
        config={'prepared':str(out),'weights':str(Path(weights).absolute()),'decision':str(Path(decision).absolute()),'admission_sha256':digest,'plan_sha256':vi.sha(out/'plan.json'),'deadline':began+900.}
        atomic(out/'launch.json',config);shutil.copyfile(decision,out/'executed-admission.json')
        if vi.sha(out/'executed-admission.json')!=digest:raise ValueError('Admission copy changed')
        with (out/'worker.log').open('x') as log:
            if time.monotonic()>=config['deadline']:raise RuntimeError('Deadline before launch')
            proc=subprocess.Popen([sys.executable,str(HERE/'diagnostic.py'),'--worker-config',str(out/'launch.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            supervise(proc,out,config['deadline'],'pair')
        terminal=vi.read(out/'terminal.json');report=vi.read(out/'result/metrics.json')
        if terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or terminal.get('cleanup_error') is not None or report.get('status')!='passed' or report.get('predictions')!=14 or report.get('model_frozen') is not True or list(out.rglob('watchdog-stop.json')):raise ValueError('Diagnostic did not complete')
        vi.checked_outputs(out/'result',report['output_sha256'])
        if report.get('plan_sha256')!=config['plan_sha256'] or report.get('source_sha256')!=sources():raise ValueError('Worker identity differs')
        if time.monotonic()>=config['deadline']:raise RuntimeError('Parent deadline reached')
        parent.update(status='passed',model_execution=True,result_sha256=vi.sha(out/'result/metrics.json'),terminal_sha256=vi.sha(out/'terminal.json'),plan_sha256=config['plan_sha256'])
    except BaseException as e:
        cleanup=None
        try:
            if proc is not None:stop_child(proc)
        except BaseException as problem:cleanup=str(problem)
        finally:
            if not (out/'terminal.json').exists():atomic(out/'terminal.json',{'status':'failed','exit_code':proc.returncode if proc else None,'error':str(e),'cleanup_error':cleanup})
            elif cleanup:atomic(out/'cleanup-error.json',{'error':str(e),'cleanup_error':cleanup})
        parent.update(status='failed',model_execution=None if proc else False,error=str(e));raise
    finally:parent['elapsed_seconds']=time.monotonic()-began;atomic(out/'metrics.json',parent)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    for n in ('worker-config','visual-prepared','training-run','cache-run','cpu-report','output','prepared','weights','decision'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.worker_config:return worker(vi.read(a.worker_config))
    names=('prepared','weights','decision') if a.execute else ('visual_prepared','training_run','cache_run','cpu_report','output')
    if any(getattr(a,n) is None for n in names):p.error('Every explicit input is required')
    r=(execute if a.execute else prepare)(**{n:getattr(a,n) for n in names});print(json.dumps(r))
if __name__=='__main__':main()
