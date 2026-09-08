"""Review later actual assessment inputs and optionally bind them. No model import."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
REPO=BASE/'outputs/open-worldline'
SCHEMA='worldline-action-effect-assessment-actual-plan-review-v1'

def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def need(value,message):
    if not value:raise ValueError(message)

def exact(value,expected):
    if type(value)is not type(expected):return False
    if isinstance(expected,dict):return set(value)==set(expected) and all(exact(value[k],v)for k,v in expected.items())
    if isinstance(expected,list):return len(value)==len(expected) and all(exact(x,y)for x,y in zip(value,expected))
    return value==expected

def audit_module():
    pins=json.loads((HERE/'pins.json').read_text());root=BASE/pins['auditor_directory']
    for name,digest in pins['auditor_files'].items():need(sha(root/name)==digest,'Frozen audit dependency changed: '+name)
    need(sha(root/'cpu-v1.json')==pins['auditor_cpu_sha256'],'Auditor CPU report changed')
    need(sha(BASE/'work/wan22-action-effect-assessment-audit-independent-review-v1/report.json')==pins['auditor_independent_review_sha256'],'Independent auditor review changed')
    spec=importlib.util.spec_from_file_location('_effect_actual_plan_auditor',root/'audit.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);return a

def protocol(plan):
    expected={'schema':'worldline-action-effect-fixed-assessment-v1','status':'prepared','checkpoint_order':['zero','old128','new128'],
              'counts':{'heldout_heads':48,'seen506_heads':6,'original999_heads':12,'head_predictions':66,'feature_extracts':12},
              'training':False,'sampling':False,'endpoint_scale':1.0,'guidance':5.0,'model_time':999,'seen_objective_time':506,
              'scope':'Four held-out noises on one seen scene, plus distinct original seen506 and original visual999 checks; final checkpoint only'}
    for key,value in expected.items():need(exact(plan.get(key),value),'Predeclared assessment protocol differs: '+key)
    need(set(plan['checkpoint_sha256'])==set(expected['checkpoint_order']),'Exactly the three declared checkpoint identities required')
    return expected

def output_path(path,prepared):
    p=Path(path).absolute();need(not p.exists() and p.parent.is_dir() and not any(q.is_symlink()for q in(p,*p.parents)),'Fresh regular output file required')
    need(not any(p.resolve().is_relative_to(root)for root in(prepared.resolve(),HERE.resolve(),REPO.resolve())),'Output must be separate from inputs and frozen sources');return p

def review(prepared,output,binding_output=None):
    prepared=Path(prepared).absolute();out=output_path(output,prepared)
    bind=output_path(binding_output,prepared)if binding_output is not None else None
    need(bind is None or bind!=out,'Review and binding must be separate files')
    report={'schema':SCHEMA,'status':'validating','model_execution':False,'cloud_actions':False,'training_admission':False,'sampling_admission':False,'helper_sha256':sha(__file__),'helper_pins_sha256':sha(HERE/'pins.json')}
    a=None
    try:
        a=audit_module();need(not any((prepared/n).exists()for n in('execution-attempt.json','result','terminal.json','launch.json')),'Review a fresh, unexecuted prepared input copy')
        before=a.inventory(prepared);plan=a.validate_prepared(prepared);protocol(plan)
        completed=a.js(prepared/'new-training-audit.json')
        for key,value in {'completed_updates':128,'auxiliary_updates':32,'auxiliary_feature_extracts':64,'auxiliary_head_predictions':128,'main_predictions':256}.items():
            need(exact(completed.get(key),value),'Completed training count differs: '+key)
        values,texts,commands,observation,cases,target=a.source_inputs(prepared)
        need(a.np.isfinite(target).all() and a.np.any(target[:,:,1:]),'Finite nonzero original target contrast required')
        for arm in a.reader.ARMS:need(a.np.array_equal(cases[arm][0][:,:,:1],observation),'Exact independent observed prefix')
        need(a.inventory(prepared)==before,'Prepared bytes changed during review')
        report.update(status='passed',plan_sha256=a.sha(prepared/'plan.json'),source_sha256=plan['source_sha256'],checkpoint_sha256=plan['checkpoint_sha256'],
            new_training_audit_sha256=plan['new_training_audit_sha256'],counts=plan['counts'],limits=plan['limits'],heldout_manifest_sha256=plan['heldout_manifest_sha256'],
            cpu_report_sha256=plan['cpu_report_sha256'],files_verified=len(before),file_bytes=sum(v['bytes']for v in before.values()),
            exact_original20_input_files=len(a.pins()['original20_input_files']),training_plan_sha256=a.pins()['training_plan_sha256'],
            input_tensor_sha256={'observation':a.ts(observation),'original999':a.ts(values['initial_latent']),'positive_text':a.ts(texts['positive']),'negative_text':a.ts(texts['negative']),
                                 'commands':{k:a.ts(v)for k,v in commands.items()},'seen506':{k:a.ts(v[0])for k,v in cases.items()},'target_difference':a.ts(target)},
            conditions='Held-out and999 model calls share each declared noise and observation across checkpoint/command comparisons. Seen506 preserves each arm own original corruption. Targets are scored afterward.',
            numeric_gate='New final must reduce held-out contrast MSE versus zero and old128 and have positive alignment on all four fixed noises. This is evaluated only after actual output recovery.',
            limitations=['This is saved-input eligibility, not model execution or numerical success.','One seen scene with four held-out Gaussian tensors does not establish generalization.'])
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        with out.open('x')as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    result={'status':'passed','review_sha256':sha(out),'review':str(out),'model_execution':False}
    if bind is not None:result['binding']=a.create_binding(prepared,out,bind)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--binding-output',type=Path);args=p.parse_args()
    print(json.dumps(review(args.prepared,args.output,args.binding_output)))
