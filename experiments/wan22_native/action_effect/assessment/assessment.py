"""Guarded fixed effect128 assessment. Preparation is CPU-only; no default execution."""
import argparse,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import reader,bindings
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path.cwd()))
spec=importlib.util.spec_from_file_location('_effect_assessment_reference20',HERE/'reference20/diagnostic.py')
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
vi=ref.vi;torch=ref.torch
SCHEMA='worldline-action-effect-fixed-assessment-v1'
OLD_PLAN_SHA='6622575e7c3e66aeb283834bd2b43d7567cd28d66d11be6e4f7b016aaaaee9e6'
ZERO_SHA='8f7751f08435eed06d4e412357b96713dbecc43771a89594dfd2f9557afd14e0'
OLD128_SHA='ac1160b22531804c473746d2fb8ee9edd75710e0e43d6f1eabc86d268c864996'
NAMES=('assessment.py','bindings.py','reader.py','test_reader.py','test_assessment.py','effect-training-source.json','effect-training-plan.json')

def sources():
    return {'repository':vi.sources()['repository'],'local':{n:vi.sha(HERE/n) for n in NAMES},
        'reference':{str(p.relative_to(HERE/'reference20')):vi.sha(p) for p in (HERE/'reference20').rglob('*') if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts}}

def cpu_review(path):
    r=vi.read(path)
    if (r.get('status')!='passed' or r.get('source_sha256')!=sources() or type(r.get('tests'))is not int or r['tests']<8
        or any(type(r.get(k))is not int or r[k]!=0 for k in ('failures','errors','skipped','exit_code'))
        or r.get('source_unchanged')is not True or r.get('cuda_initialized')is not False):
        raise ValueError('Current source-bound assessment CPU review required')
    return vi.sha(path)

def counts():return {'heldout_heads':48,'seen506_heads':6,'original999_heads':12,'head_predictions':66,'feature_extracts':12}

def prepare(comparison_prepared,heldout,training_run,training_audit,cpu_report,output):
    out=vi.root(output,fresh=True);out.mkdir();status={'schema':SCHEMA,'status':'preparing','model_execution':False}
    try:
        mapping=sources();review=cpu_review(cpu_report);prior=vi.root(comparison_prepared)
        if vi.sha(prior/'plan.json')!=OLD_PLAN_SHA:raise ValueError('Exact original20 prepared input required')
        oldplan,old,_,_,_,_=ref.read_prepared(prior)
        reader.read_heldout(heldout)
        checkpoint,trained=bindings.completed(training_run,training_audit,ref,HERE)
        ref.training_matches_original(trained,old)
        if oldplan['checkpoint_128']['checkpoint_sha256']!=OLD128_SHA:raise ValueError('Exact old128 required')
        copied=out/'reference-inputs';copied.mkdir()
        for n in ('plan.json','cpu-report.json','checkpoint-0128.safetensors','training-audit.json'):shutil.copyfile(vi.file(prior,n),copied/n)
        for n in ('original14','source','training-evidence'):shutil.copytree(prior/n,copied/n,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(heldout,out/'heldout')
        shutil.copyfile(checkpoint,out/'new128.safetensors');shutil.copyfile(training_audit,out/'new-training-audit.json');shutil.copyfile(cpu_report,out/'cpu-report.json')
        selected={}
        for n in bindings.SELECTED:
            p=out/'new-training-evidence'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(vi.file(training_run,n),p);selected[str(p.relative_to(out))]=vi.sha(p)
        for group,rows in mapping.items():
            base=vi.REPO if group=='repository' else HERE/'reference20' if group=='reference' else HERE
            for n,digest in rows.items():
                p=out/'source'/group/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/n,p)
                if vi.sha(p)!=digest:raise ValueError('Source changed during preparation')
        plan={'schema':SCHEMA,'status':'prepared','source_sha256':mapping,'cpu_report_sha256':review,
            'comparison_plan_sha256':OLD_PLAN_SHA,'heldout_manifest_sha256':reader.MANIFEST_SHA,
            'checkpoint_order':list(reader.CHECKPOINTS),'checkpoint_sha256':{'zero':ZERO_SHA,'old128':OLD128_SHA,'new128':trained['checkpoint_sha256']},
            'new_training_identity':trained,'new_training_audit_sha256':vi.sha(training_audit),'new_training_evidence':selected,
            'counts':counts(),'limits':ref.limits('pair'),'training':False,'sampling':False,
            'endpoint_scale':1.0,'guidance':5.0,'model_time':999,'seen_objective_time':506,
            'scope':'Four held-out noises on one seen scene, plus distinct original seen506 and original visual999 checks; final checkpoint only'}
        ref.atomic(out/'plan.json',plan);read_prepared(out);status.update(status='prepared',plan_sha256=vi.sha(out/'plan.json'));return status
    except BaseException as e:status.update(status='failed',error=str(e));raise
    finally:ref.atomic(out/'metrics.json',status)

def read_prepared(out):
    out=vi.root(out);p=vi.read(out/'plan.json')
    if (p.get('schema')!=SCHEMA or p.get('status')!='prepared' or p.get('source_sha256')!=sources() or p.get('limits')!=ref.limits('pair')
        or p.get('counts')!=counts() or p.get('checkpoint_order')!=list(reader.CHECKPOINTS) or p.get('training')is not False or p.get('sampling')is not False
        or p.get('endpoint_scale')!=1.0 or p.get('guidance')!=5.0 or p.get('model_time')!=999 or p.get('seen_objective_time')!=506):raise ValueError('Exact assessment protocol required')
    if cpu_review(out/'cpu-report.json')!=p['cpu_report_sha256']:raise ValueError('Assessment CPU review differs')
    for group,rows in p['source_sha256'].items():
        for n,h in rows.items():
            if vi.sha(vi.file(out/'source'/group,n))!=h:raise ValueError('Retained assessment source differs')
    if vi.sha(out/'reference-inputs/plan.json')!=OLD_PLAN_SHA or p['comparison_plan_sha256']!=OLD_PLAN_SHA:raise ValueError('Original input plan differs')
    oldplan,old,values,contexts,commands,cases=ref.read_prepared(out/'reference-inputs')
    if p['heldout_manifest_sha256']!=reader.MANIFEST_SHA:raise ValueError('Held-out identity changed')
    _,noises=reader.read_heldout(out/'heldout')
    vi.checked_outputs(out,p['new_training_evidence']);trained=p['new_training_identity']
    if vi.sha(out/'new-training-audit.json')!=p['new_training_audit_sha256']:raise ValueError('Completed training audit differs')
    bindings.validate_audit(vi.read(out/'new-training-audit.json'),trained,vi.read(HERE/'effect-training-source.json'))
    links={'metrics.json':'parent_sha256','worker/metrics.json':'worker_sha256','training/metrics.json':'training_sha256','terminal.json':'terminal_sha256','plan.json':'plan_sha256','training/checkpoint-0128/manifest.json':'checkpoint_manifest_sha256'}
    if any(p['new_training_evidence'].get('new-training-evidence/'+n)!=trained[k] for n,k in links.items()):raise ValueError('Completed training evidence links differ')
    saved={n:vi.read(out/'new-training-evidence'/n) for n in bindings.SELECTED}
    bindings.validate_reports(saved['plan.json'],saved['metrics.json'],saved['worker/metrics.json'],saved['training/metrics.json'],saved['terminal.json'],saved['training/last-valid.json'],saved['training/checkpoint-0128/manifest.json'],vi.read(HERE/'effect-training-plan.json'),vi.read(HERE/'effect-training-source.json'))
    if saved['training/core-before.json']!=saved['training/core-after.json'] or saved['training/core-before.json']!=trained['core_records']:raise ValueError('Retained frozen base differs')
    ref.training_matches_original(trained,old)
    expected={'zero':ZERO_SHA,'old128':OLD128_SHA,'new128':trained['checkpoint_sha256']}
    if p['checkpoint_sha256']!=expected or oldplan['checkpoint_128']['checkpoint_sha256']!=OLD128_SHA:raise ValueError('Exact three checkpoint identities required')
    for name,path in checkpoint_paths(out).items():
        adapter,record=ref.sampler.load_adapter_checkpoint(path,expected[name]);del adapter
        if name=='new128' and record!=trained['adapter']:raise ValueError('New final checkpoint parameter identity differs')
    raw=vi.pe.tensors(out/'reference-inputs/original14/original-training-inputs.safetensors',{'noise':(reader.SHAPE,'F32'),'closed_target':(reader.SHAPE,'F32'),'open_target':(reader.SHAPE,'F32'),'observation':((1,48,1,44,78),'F32')},vi.read(out/'reference-inputs/original14/plan.json')['original_training_input_sha256'])
    target_difference=(raw['open_target']-raw['closed_target']).numpy()
    return p,noises,values,contexts,commands,cases,target_difference

def checkpoint_paths(out):return {'zero':out/'reference-inputs/original14/checkpoint-0000.safetensors','old128':out/'reference-inputs/checkpoint-0128.safetensors','new128':out/'new128.safetensors'}

def activate_checkpoint(adapter,states,identities,name):
    """Retain the same bridge owner while replacing only its small adapter."""
    adapter.load_state_dict(states[name],strict=True)
    if {k:ref.sampler.tensor_sha(v) for k,v in adapter.state_dict().items()}!=identities[name]['tensor_sha256']:
        raise ValueError('Checkpoint copy differs')

def admission(path,out,plan):
    r=vi.read(path);wanted={'schema':'worldline-action-effect-assessment-admission-v1','decision':'admit','issued_by':'parent-agent',
        'plan_sha256':vi.sha(out/'plan.json'),'source_sha256':plan['source_sha256'],'limits':ref.limits('pair'),'counts':counts(),
        'checkpoint_sha256':plan['checkpoint_sha256'],'heldout_manifest_sha256':reader.MANIFEST_SHA,
        'new_training_audit_sha256':plan['new_training_audit_sha256'],'cpu_report_sha256':plan['cpu_report_sha256'],'training_admitted':False,'sampling_admitted':False}
    if any(r.get(k)!=v for k,v in wanted.items()):raise ValueError('Exact-plan parent assessment admission required')
    return vi.sha(path)

def worker(config):
    out=Path(config['prepared']);result=out/'result';result.mkdir();began=time.monotonic();report={'schema':SCHEMA,'status':'running','model_execution':False,'completed_prediction_files':0}
    try:
        plan,noises,values,contexts,commands,cases,target=read_prepared(out)
        if admission(config['decision'],out,plan)!=config['admission_sha256'] or vi.sha(out/'plan.json')!=config['plan_sha256']:raise ValueError('Launch identity changed')
        actual=ref.hardware('NVIDIA A100-SXM4-80GB');compat=ref.visual.require_hardware(actual,plan['new_training_identity']['hardware'])
        report.update(hardware=actual,hardware_comparison=compat,plan_sha256=config['plan_sha256'],source_sha256=sources(),limits=ref.limits('pair'));ref.atomic(result/'metrics.json',report)
        with ref.Monitor(result,config['deadline'],'pair') as guard:
            flags=ref.probe._runtime_flags();report['runtime_flags']=flags
            def check():
                guard.check()
                if ref.probe._runtime_flags()!=flags:raise ValueError('Runtime flags changed')
            start=time.monotonic();core,loaded=ref.load_model(config['weights'],check);report['load_seconds']=time.monotonic()-start;report['model_execution']=True
            ref.atomic(result/'weight-load.json',loaded);ref.atomic(result/'metrics.json',report)
            before=ref.parameter_records(core,check=check,expected=plan['new_training_identity']['core_records']);ref.atomic(result/'core-before.json',before)
            if before!=ref.probe._original_weights(loaded):raise ValueError('Loaded original base differs')
            states={};identities={};adapter=None
            for name,path in checkpoint_paths(out).items():
                current,identities[name]=ref.sampler.load_adapter_checkpoint(path,plan['checkpoint_sha256'][name])
                states[name]={k:v.detach().cpu().clone() for k,v in current.state_dict().items()}
                if name=='zero':adapter=current
            adapter.to('cuda:0');activate_checkpoint(adapter,states,identities,'zero')
            bridge=ref.bridge_module.NativeCUDAActionBridge(core,adapter,profile='spatial');active=['zero'];feature_count=[0]
            def extract(name,text,x,t,c):
                check();feature_count[0]+=1
                return bridge.extract_features(torch.from_numpy(x).to('cuda:0'),torch.from_numpy(t).to('cuda:0'),[torch.from_numpy(c).to('cuda:0')])
            def predict(cp,features,a,o):
                check()
                if active[0]!=cp:
                    activate_checkpoint(adapter,states,identities,cp);active[0]=cp
                with torch.no_grad():v=bridge.predict_from_features(features,torch.from_numpy(a).to('cuda:0'),torch.from_numpy(o).to('cuda:0'),track_grad=False)
                torch.cuda.synchronize();return v.detach().cpu().numpy()
            output_files={}
            def retain(name,items):
                path=result/(name+'.safetensors');ref.save_file({k:torch.from_numpy(v).contiguous() for k,v in items.items()},str(path));output_files[path.name]=vi.sha(path)
                report['completed_prediction_files']+=1;report['last_retained']=name;ref.atomic(result/'metrics.json',report)
            obs=values['observation'].numpy();acts={a:v.numpy() for a,v in commands.items()};text={'positive':contexts['atrium'].numpy(),'negative':contexts['native_negative'].numpy()}
            held=reader.evaluate_matrix(noises,obs,acts,text,extract,predict,retain,check)
            report['heldout']=held;report['heldout_scores']=reader.read_saved_scores(result,{n:h for n,h in output_files.items() if n.startswith('heldout-')},held['calls'],target)
            report['additional_checks']=additional_checks(values,cases,obs,acts,text,target,extract,predict,retain,check)
            if feature_count[0]!=12 or report['completed_prediction_files']!=66:raise ValueError('Exact assessment feature/head counts differ')
            after=ref.parameter_records(core,check=check,expected=before);ref.atomic(result/'core-after.json',after)
            if any(p.grad is not None for p in adapter.parameters()) or {k:ref.sampler.tensor_sha(v) for k,v in adapter.state_dict().items()}!=identities[active[0]]['tensor_sha256']:raise ValueError('Adapter acquired gradients or changed')
            report.update(model_frozen=True,counts=counts(),checkpoint_records=identities,raw_prediction_sha256=output_files);check()
        read_prepared(out)
        if time.monotonic()>=config['deadline']:raise RuntimeError('Worker deadline reached')
        report.update(status='passed',output_sha256={str(p.relative_to(result)):vi.sha(p) for p in result.rglob('*') if p.is_file() and p.name not in ('metrics.json','memory.jsonl')})
    except BaseException as e:report.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:report['elapsed_seconds']=time.monotonic()-began;ref.atomic(result/'metrics.json',report)

def additional_checks(values,cases,obs,commands,contexts,target,extract,predict,retain,check):
    result={'seen506':{},'original999':{},'calls':[]}
    def keep(name,cp,arm,text,v,x,t):
        reader.finite(v);retain(name,{'velocity':v.copy()});result['calls'].append({'name':name,'checkpoint':cp,'arm':arm,'text':text,
            'input_sha256':reader.tensor_sha(x),'time_sha256':reader.tensor_sha(t),'commands_sha256':reader.tensor_sha(commands[arm]),'context_sha256':reader.tensor_sha(contexts[text]),'prediction_sha256':reader.tensor_sha(v)})
    for arm in reader.ARMS:
        x=cases[arm+'_noisy'].numpy();t=cases[arm+'_times'].numpy();features=extract('seen506-'+arm,'positive',x.copy(),t.copy(),contexts['positive'].copy())
        try:
            for cp in reader.CHECKPOINTS:
                check();v=predict(cp,features,commands[arm].copy(),obs.copy());keep(f'seen506-{cp}-{arm}-positive',cp,arm,'positive',v,x,t)
                loss=float(np.mean((v[:,:,1:].astype(np.float64)-cases[arm+'_target'].numpy()[:,:,1:].astype(np.float64))**2))
                result['seen506'].setdefault(cp,{})[arm]={'future_flow_mse_fp64':loss,'seen_training_draw':True}
        finally:del features
    x=values['initial_latent'].numpy()[None];t=values['token_times'].numpy();features={}
    try:
        for text in reader.TEXTS:features[text]=extract('original999',text,x.copy(),t.copy(),contexts[text].copy())
        for cp in reader.CHECKPOINTS:
            outputs={}
            for arm in reader.ARMS:
                for text in reader.TEXTS:
                    check();v=predict(cp,features[text],commands[arm].copy(),obs.copy());keep(f'original999-{cp}-{arm}-{text}',cp,arm,text,v,x,t);outputs[arm+'-'+text]=v
            result['original999'][cp]=reader.score(outputs,target)
    finally:features.clear()
    if len(result['calls'])!=18:raise ValueError('Exactly18 distinct original-input predictions required')
    return result

def execute(prepared,weights,decision):
    out=vi.root(prepared)
    with (out/'execution-attempt.json').open('x') as f:f.write('single-use\n')
    began=time.monotonic();proc=None;parent={'schema':SCHEMA,'status':'running','model_execution':False}
    try:
        plan,*_=read_prepared(out);digest=admission(decision,out,plan)
        config={'prepared':str(out),'weights':str(Path(weights).absolute()),'decision':str(Path(decision).absolute()),'admission_sha256':digest,'plan_sha256':vi.sha(out/'plan.json'),'deadline':began+900.}
        ref.atomic(out/'launch.json',config);shutil.copyfile(decision,out/'executed-admission.json')
        if vi.sha(out/'executed-admission.json')!=digest:raise ValueError('Admission copy changed')
        with (out/'worker.log').open('x') as log:
            if time.monotonic()>=config['deadline']:raise RuntimeError('Deadline before launch')
            proc=subprocess.Popen([sys.executable,str(HERE/'assessment.py'),'--worker-config',str(out/'launch.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            ref.supervise(proc,out,config['deadline'],'pair')
        terminal=vi.read(out/'terminal.json');r=vi.read(out/'result/metrics.json')
        if terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or terminal.get('cleanup_error')is not None or r.get('status')!='passed' or r.get('counts')!=counts() or r.get('model_frozen')is not True or list(out.rglob('watchdog-stop.json')):raise ValueError('Assessment did not complete')
        vi.checked_outputs(out/'result',r['output_sha256'])
        if r.get('plan_sha256')!=config['plan_sha256'] or r.get('source_sha256')!=sources():raise ValueError('Worker identity differs')
        if time.monotonic()>=config['deadline']:raise RuntimeError('Parent deadline reached')
        parent.update(status='passed',model_execution=True,result_sha256=vi.sha(out/'result/metrics.json'),terminal_sha256=vi.sha(out/'terminal.json'),plan_sha256=config['plan_sha256'])
    except BaseException as e:
        cleanup=None
        try:
            if proc is not None:ref.stop_child(proc)
        except BaseException as err:cleanup=str(err)
        if not (out/'terminal.json').exists():ref.atomic(out/'terminal.json',{'status':'failed','exit_code':proc.returncode if proc else None,'error':str(e),'cleanup_error':cleanup})
        elif cleanup:ref.atomic(out/'cleanup-error.json',{'error':str(e),'cleanup_error':cleanup})
        parent.update(status='failed',model_execution=None if proc else False,error=str(e));raise
    finally:parent['elapsed_seconds']=time.monotonic()-began;ref.atomic(out/'metrics.json',parent)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    for n in ('worker-config','comparison-prepared','heldout','training-run','training-audit','cpu-report','output','prepared','weights','decision'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.worker_config:return worker(vi.read(a.worker_config))
    names=('prepared','weights','decision') if a.execute else ('comparison_prepared','heldout','training_run','training_audit','cpu_report','output')
    if any(getattr(a,n)is None for n in names):p.error('All explicit inputs are required')
    print(json.dumps((execute if a.execute else prepare)(**{n:getattr(a,n) for n in names})))
if __name__=='__main__':main()
