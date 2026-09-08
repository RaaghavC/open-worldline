"""Twenty fixed-input denoiser predictions. No optimizer, clip or cloud call."""
import argparse,json,shutil,sys,time,subprocess,importlib.util
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

SCHEMA='worldline-action-fixed-input-comparison128-v1'


def sources():
    return {'visual':vi.sources(),'diagnostic':{n:vi.sha(HERE/n) for n in ('diagnostic.py','test_diagnostic.py')},'reference14':{n:vi.sha(HERE/'reference14'/n) for n in ('diagnostic.py','test_diagnostic.py')},'training_reader':{n:vi.sha(HERE/'training_reader'/n) for n in ('inputs.py','training-source.json')}}


def review(path):
    r=vi.read(path)
    if r.get('status')!='passed' or r.get('source_sha256')!=sources() or r.get('tests',0)<3 or any(r.get(k)!=0 for k in ('failures','errors','skipped')):
        raise ValueError('Current diagnostic CPU review required')
    return vi.sha(path)


def module(name,path):
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]


def reference():return module('_comparison128_original14',HERE/'reference14/diagnostic.py')
def training_reader():return module('_comparison128_training_reader',HERE/'training_reader/inputs.py')


def training_matches_original(trained,original):
    old=original['input_identity']['training']
    if trained['core_records']!=old['core_records'] or any(trained['training_input_identity'][key]!=old['training_input_identity'][key] for key in ('cache','text')):
        raise ValueError('Final128 foundation, cache or text differs from the original diagnostic')
    visual.require_hardware(trained['hardware'],old['hardware'])


def prepare(diagnostic_prepared,training_run,training_audit,cpu_report,output):
    out=vi.root(output,fresh=True);out.mkdir(parents=True)
    status={'schema':SCHEMA,'status':'preparing','model_execution':False}
    try:
        cpu_sha=review(cpu_report);mapping=sources();prior=vi.root(diagnostic_prepared)
        original,old,values,contexts,commands,cases=reference().read_prepared(prior)
        trained_plan,checkpoint,trained=training_reader().training(training_run)
        audit_sha=training_reader().validate_training_audit(training_audit,trained)
        training_matches_original(trained,old)
        prior_copy=out/'original14';prior_copy.mkdir()
        for name in ('plan.json','cpu-report.json','objective-inputs.safetensors','original-training-inputs.safetensors','checkpoint-0000.safetensors','diagnostic.py.txt','test_diagnostic.py.txt'):
            shutil.copyfile(vi.file(prior,name),prior_copy/name)
        shutil.copytree(prior/'visual-inputs',prior_copy/'visual-inputs',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(checkpoint,out/'checkpoint-0128.safetensors');shutil.copyfile(training_audit,out/'training-audit.json');shutil.copyfile(cpu_report,out/'cpu-report.json')
        # The complete training run is checked before this compact exact evidence copy.
        selected=('metrics.json','plan.json','terminal.json','worker/metrics.json','worker/weight-load.json','training/metrics.json','training/core-before.json','training/core-after.json','training/last-valid.json','training/checkpoint-0128/manifest.json')
        evidence={}
        for name in selected:
            target=out/'training-evidence'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(vi.file(training_run,name),target);evidence[str(target.relative_to(out))]=vi.sha(target)
        for group in ('diagnostic','reference14','training_reader'):
            for name,h in mapping[group].items():
                source=(HERE if group=='diagnostic' else HERE/group)/name;target=out/'source'/group/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                if vi.sha(target)!=h:raise ValueError('Source changed during copy')
        plan={'schema':SCHEMA,'status':'prepared','source_sha256':mapping,'cpu_report_sha256':cpu_sha,
              'original14_plan_sha256':vi.sha(prior/'plan.json'),'checkpoint_128':trained['adapter'],'training_identity':trained,
              'training_audit_sha256':audit_sha,'training_evidence_sha256':evidence,'predictions':20,'limits':limits('pair'),
              'checkpoint_order':['0','16','128'],'native_predictions':2,'training':False,'generation':False,
              'scope':'Exact fixed k506 objective and held-fixed k999 command/CFG comparison at cp0,cp16,cp128; no new training or generation'}
        atomic(out/'plan.json',plan);read_prepared(out);status.update(status='prepared',plan_sha256=vi.sha(out/'plan.json'));return status
    except BaseException as e:status.update(status='failed',error=str(e));raise
    finally:atomic(out/'metrics.json',status)


def read_prepared(out):
    out=vi.root(out);plan=vi.read(out/'plan.json')
    if plan.get('schema')!=SCHEMA or plan.get('status')!='prepared' or plan.get('source_sha256')!=sources() or plan.get('limits')!=limits('pair') or plan.get('predictions')!=20 or plan.get('checkpoint_order')!=['0','16','128'] or plan.get('training') is not False or plan.get('generation') is not False:
        raise ValueError('Exact current20 comparison plan required')
    if review(out/'cpu-report.json')!=plan['cpu_report_sha256']:raise ValueError('CPU report differs')
    for group in ('diagnostic','reference14','training_reader'):
        for name,h in plan['source_sha256'][group].items():
            if vi.sha(vi.file(out/'source'/group,name))!=h:raise ValueError('Source snapshot differs')
    original,old,values,contexts,commands,cases=reference().read_prepared(out/'original14')
    if vi.sha(out/'original14/plan.json')!=plan['original14_plan_sha256']:raise ValueError('Original14 input plan differs')
    trained=plan['training_identity'];training_matches_original(trained,old)
    if training_reader().validate_training_audit(out/'training-audit.json',trained)!=plan['training_audit_sha256']:raise ValueError('Actual128 audit differs')
    vi.checked_outputs(out,plan['training_evidence_sha256'])
    links={'metrics.json':'parent_sha256','worker/metrics.json':'worker_sha256','terminal.json':'terminal_sha256','training/metrics.json':'training_sha256','plan.json':'plan_sha256','training/checkpoint-0128/manifest.json':'checkpoint_manifest_sha256'}
    if any(plan['training_evidence_sha256'].get('training-evidence/'+name)!=trained[key] for name,key in links.items()):raise ValueError('Actual training evidence binding differs')
    before=vi.read(out/'training-evidence/training/core-before.json');after=vi.read(out/'training-evidence/training/core-after.json')
    if before!=after or before!=trained['core_records']:raise ValueError('Retained128 foundation evidence differs')
    adapter,record=sampler.load_adapter_checkpoint(out/'checkpoint-0128.safetensors',trained['checkpoint_sha256']);del adapter
    if record!=plan['checkpoint_128'] or record!=trained['adapter']:raise ValueError('Exact final128 checkpoint required')
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
    for checkpoint in ('0','16','128'):
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
    report['relative_objective_improvement_128_vs16']={arm:1-report['objective']['128'][arm]/report['objective']['16'][arm] if report['objective']['16'][arm] else None for arm in ('closed','open')}
    report['relative_objective_improvement_128_vs0']={arm:1-report['objective']['128'][arm]/report['objective']['0'][arm] if report['objective']['0'][arm] else None for arm in ('closed','open')}
    if report['predictions']!=20:raise RuntimeError('Exactly twenty predictions required')
    return report


def admission(path,out,plan):
    record=vi.read(path)
    wanted={'schema':'worldline-action-fixed-input-comparison128-admission-v1','decision':'admit','issued_by':'parent-agent',
        'plan_sha256':vi.sha(out/'plan.json'),'source_sha256':plan['source_sha256'],'limits':limits('pair'),
        'training_admitted':False,'generation_admitted':False,'predictions':20,'cpu_report_sha256':plan['cpu_report_sha256'],'training_audit_sha256':plan['training_audit_sha256'],'checkpoint128_sha256':plan['checkpoint_128']['checkpoint_sha256']}
    if any(record.get(k)!=v for k,v in wanted.items()):raise ValueError('Exact-plan root diagnostic admission required')
    return vi.sha(path)


def worker(config):
    out=Path(config['prepared']);result=out/'result';result.mkdir();began=time.monotonic();report={'schema':SCHEMA,'status':'running','model_execution':False}
    try:
        plan,old,values,contexts,commands,cases=read_prepared(out)
        if admission(config['decision'],out,plan)!=config['admission_sha256'] or vi.sha(out/'plan.json')!=config['plan_sha256']:raise ValueError('Launch changed')
        actual=hardware(old['expected_gpu']);compatibility=visual.require_hardware(actual,plan['training_identity']['hardware'])
        report.update(hardware=actual,hardware_comparison=compatibility,plan_sha256=config['plan_sha256'],source_sha256=sources(),limits=limits('pair'),completed_prediction_files=0)
        atomic(result/'metrics.json',report)
        with Monitor(result,config['deadline'],'pair') as guard:
            flags=probe._runtime_flags();report['runtime_flags']=flags
            def check():
                guard.check()
                if probe._runtime_flags()!=flags:raise ValueError('Runtime flags changed')
            start=time.monotonic();core,loaded=load_model(config['weights'],check);report['load_seconds']=time.monotonic()-start;report['model_execution']=True
            atomic(result/'weight-load.json',loaded);atomic(result/'metrics.json',report)
            start=time.monotonic();before=parameter_records(core,check=check,expected=plan['training_identity']['core_records']);
            if before!=probe._original_weights(loaded):raise ValueError('Original foundation loader differs from training')
            report['before_hash_seconds']=time.monotonic()-start;atomic(result/'core-before.json',before)
            bridges={};identities={}
            for name,path,digest in [('0',out/'original14/checkpoint-0000.safetensors',vi.read(out/'original14/plan.json')['checkpoint_zero']['checkpoint_sha256']),('16',out/'original14/visual-inputs/adapter.safetensors',old['artifacts']['adapter.safetensors']),('128',out/'checkpoint-0128.safetensors',plan['checkpoint_128']['checkpoint_sha256'])]:
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
        if terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or terminal.get('cleanup_error') is not None or report.get('status')!='passed' or report.get('predictions')!=20 or report.get('model_frozen') is not True or list(out.rglob('watchdog-stop.json')):raise ValueError('Diagnostic did not complete')
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
    for n in ('worker-config','diagnostic-prepared','training-run','training-audit','cpu-report','output','prepared','weights','decision'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.worker_config:return worker(vi.read(a.worker_config))
    names=('prepared','weights','decision') if a.execute else ('diagnostic_prepared','training_run','training_audit','cpu_report','output')
    if any(getattr(a,n) is None for n in names):p.error('Every explicit input is required')
    r=(execute if a.execute else prepare)(**{n:getattr(a,n) for n in names});print(json.dumps(r))
if __name__=='__main__':main()
