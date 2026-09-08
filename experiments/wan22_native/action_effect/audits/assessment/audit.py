"""Read-only66-output assessment audit. No Torch, model, optimizer or network call."""
import argparse,hashlib,importlib.util,json,math,re,shutil,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]/'outputs/open-worldline'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

reader=module('_saved_effect_reader',HERE/'reference/reader.py')
checks=module('_saved_effect_arrays',HERE/'reference/array_checks.py')
sha=checks.sha;read=checks.parse;need=checks.need;arrays=checks.arrays;ts=checks.tensor_sha
SCHEMA='worldline-action-effect-assessment-actual-audit-v1'
BINDING_SCHEMA='worldline-action-effect-assessment-audit-binding-v1'
PLAN_REVIEW_SCHEMA='worldline-action-effect-assessment-actual-plan-review-v1'
COUNTS={'heldout_heads':48,'seen506_heads':6,'original999_heads':12,'head_predictions':66,'feature_extracts':12}
LIMITS={'seconds':900.,'host_rss_bytes':48*2**30,'cuda_reserved_bytes':60*2**30,'minimum_host_available_bytes':8*2**30,'minimum_cuda_available_bytes':8*2**30,'minimum_gpu_total_bytes':70*2**30}
ROUNDING={'relative':1e-12,'absolute':1e-15}

def regular(path):
    p=Path(path).absolute();need(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'Regular file without symlink ancestors required');return p

def js(path):
    p=regular(path);need(p.stat().st_size<=16*2**20,'Bounded JSON required');return read(p.read_bytes())

def member(root,name):
    need(isinstance(name,str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts,'Safe relative path required');return regular(Path(root)/name)

def digest(value):need(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'Exact SHA256 required');return value

def inventory(root):
    root=Path(root).absolute();need(root.is_dir() and not any(q.is_symlink() for q in(root,*root.parents)),'Regular directory required');need(not any(p.is_symlink() for p in root.rglob('*')),'No symlink artifacts')
    return {p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(root.rglob('*'))if p.is_file() and '__pycache__' not in p.parts}

def file_map(root,rows):
    need(isinstance(rows,dict) and rows,'Nonempty file inventory required')
    for name,row in rows.items():
        p=member(root,name);value=row['sha256'] if isinstance(row,dict) else row
        need(sha(p)==digest(value),'File identity '+name)
        if isinstance(row,dict):need(p.stat().st_size==row['bytes'],'File size '+name)

def pins():
    value=js(HERE/'pins.json')
    for name,value_sha in value['reference_sha256'].items():need(sha(HERE/'reference'/name)==value_sha,'Unchanged copied reader source')
    return value

def identity(plan):
    trained=plan['new_training_identity'];result={k:trained[k]for k in('parent_sha256','worker_sha256','training_sha256','terminal_sha256','plan_sha256')}
    return result|{'final_checkpoint_manifest_sha256':trained['checkpoint_manifest_sha256'],'final_checkpoint_sha256':trained['checkpoint_sha256'],'source_sha256':pins()['training_source_sha256']}

def validate_prepared(root):
    root=Path(root);p=js(root/'plan.json');fixed=pins()
    need(p.get('schema')=='worldline-action-effect-fixed-assessment-v1' and p.get('status')=='prepared' and p.get('source_sha256')==fixed['source_sha256'],'Exact assessment source family')
    need(p.get('counts')==COUNTS and p.get('limits')==LIMITS and p.get('training') is False and p.get('sampling') is False,'Exact counts and900-second limits')
    need(p.get('checkpoint_order')==list(reader.CHECKPOINTS) and p.get('endpoint_scale')==1. and p.get('guidance')==5. and p.get('model_time')==999 and p.get('seen_objective_time')==506,'Exact assessment protocol')
    need(p['comparison_plan_sha256']==fixed['original20_plan_sha256'] and p['heldout_manifest_sha256']==fixed['heldout_manifest_sha256'],'Original comparison and held-out identities')
    for group,rows in fixed['source_sha256'].items():file_map(root/'source'/group,rows)
    need(sha(root/'cpu-report.json')==fixed['cpu_report_sha256']==p['cpu_report_sha256'],'Exact assessment CPU report')
    cpu=js(root/'cpu-report.json');need(cpu['status']=='passed' and cpu['tests']==14 and cpu['source_sha256']==fixed['source_sha256'] and cpu['source_unchanged'] is True and cpu['cuda_initialized'] is False and all(cpu[k]==0 for k in('failures','errors','skipped','exit_code')),'Passed source-bound CPU review')
    file_map(root/'reference-inputs',fixed['original20_input_files']);reader.read_heldout(root/'heldout')
    file_map(root,p['new_training_evidence']);completed=js(root/'new-training-audit.json')
    need(sha(root/'new-training-audit.json')==p['new_training_audit_sha256'],'Actual training audit hash')
    need(completed.get('schema')=='worldline-action-effect128-actual-independent-v1' and completed.get('status')=='passed' and completed.get('identity')==identity(p),'Actual final128 training audit identity')
    need(completed.get('completed_updates')==128 and completed.get('foundation_values_unchanged') is True and completed.get('final_checkpoint_only') is True and completed.get('auxiliary_updates')==32 and completed.get('auxiliary_head_predictions')==128,'Completed fixed auxiliary training counts')
    need(sha(root/'new-training-evidence/plan.json')==fixed['training_plan_sha256']==p['new_training_identity']['plan_sha256'],'Frozen effect training plan')
    paths=checkpoint_paths(root)
    expected={'zero':'8f7751f08435eed06d4e412357b96713dbecc43771a89594dfd2f9557afd14e0','old128':'ac1160b22531804c473746d2fb8ee9edd75710e0e43d6f1eabc86d268c864996','new128':p['new_training_identity']['checkpoint_sha256']}
    need(p['checkpoint_sha256']==expected,'Exact three checkpoint identities')
    for key,path in paths.items():need(sha(path)==digest(expected[key]),'Checkpoint bytes '+key)
    return p

def checkpoint_paths(root):return {'zero':root/'reference-inputs/original14/checkpoint-0000.safetensors','old128':root/'reference-inputs/checkpoint-0128.safetensors','new128':root/'new128.safetensors'}

def create_binding(prepared,plan_review,output):
    prepared=Path(prepared).absolute();out=Path(output).absolute();need(not out.exists() and not out.resolve().is_relative_to(prepared) and not out.resolve().is_relative_to(HERE) and not out.resolve().is_relative_to(REPO) and not any(q.is_symlink() for q in(out,*out.parents)),'Fresh separate binding output')
    need(not any((prepared/n).exists() for n in ('execution-attempt.json','result','terminal.json','launch.json')),'Bind a fresh prepared input copy before execution')
    p=validate_prepared(prepared);review=js(plan_review)
    expected={'schema':PLAN_REVIEW_SCHEMA,'status':'passed','plan_sha256':sha(prepared/'plan.json'),'source_sha256':p['source_sha256'],'checkpoint_sha256':p['checkpoint_sha256'],'new_training_audit_sha256':p['new_training_audit_sha256'],'counts':COUNTS}
    need(all(review.get(k)==v for k,v in expected.items()),'Passed actual-plan review required')
    records=inventory(prepared);records.pop('metrics.json',None)
    value={'schema':BINDING_SCHEMA,'status':'bound','plan_sha256':sha(prepared/'plan.json'),'prepared_files':records,'plan_review_sha256':sha(plan_review),'pins_sha256':sha(HERE/'pins.json'),'auditor_sha256':sha(__file__),'actual_plan_review':review,'model_execution':False}
    with out.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return {'status':'bound','binding_sha256':sha(out),'input_files':len(records)}

def close_tree(actual,recorded,path='score'):
    if type(actual)is float:
        need(type(recorded)in(int,float) and math.isfinite(actual) and math.isfinite(recorded) and math.isclose(actual,recorded,rel_tol=ROUNDING['relative'],abs_tol=ROUNDING['absolute']),'Descriptive FP64 scalar mismatch '+path)
    elif isinstance(actual,dict):
        need(isinstance(recorded,dict) and set(actual)==set(recorded),'Score fields '+path)
        for key,value in actual.items():close_tree(value,recorded[key],path+'.'+key)
    elif isinstance(actual,list):
        need(isinstance(recorded,list) and len(actual)==len(recorded),'Score list '+path)
        for i,(x,y)in enumerate(zip(actual,recorded)):close_tree(x,y,path+str(i))
    else:need(type(actual)is type(recorded) and actual==recorded,'Exact descriptive value '+path)

def resource_samples(worker,parent):
    mismatch=[]
    need(worker and parent,'Both resource sample streams required')
    for i,row in enumerate(worker):
        for k in ('host_rss_bytes','cuda_reserved_bytes','host_available_bytes','cuda_available_bytes'):need(type(row[k])is int and row[k]>=0,'Strict worker resource value')
        need(row['host_rss_bytes']<=LIMITS['host_rss_bytes'] and row['cuda_reserved_bytes']<=LIMITS['cuda_reserved_bytes'] and row['host_available_bytes']>=LIMITS['minimum_host_available_bytes'] and row['cuda_available_bytes']>=LIMITS['minimum_cuda_available_bytes'],'Sampled worker cap/floor')
        if 'cuda_allocated_bytes'in row:
            need(type(row['cuda_allocated_bytes'])is int and row['cuda_allocated_bytes']>=0,'Strict allocated counter')
            if row['cuda_allocated_bytes']>row['cuda_reserved_bytes']:mismatch.append({'sample':i,'allocated':row['cuda_allocated_bytes'],'reserved':row['cuda_reserved_bytes']})
    for row in parent:
        need(all(type(row[k])is int and row[k]>=0 for k in('combined_rss_bytes','host_available_bytes')),'Strict aggregate host values')
        need(row['combined_rss_bytes']<=LIMITS['host_rss_bytes'] and row['host_available_bytes']>=LIMITS['minimum_host_available_bytes'],'Sampled parent caps')
    return {'worker_samples':len(worker),'parent_samples':len(parent),'peak_cuda_reserved_bytes':max(x['cuda_reserved_bytes']for x in worker),'peak_combined_rss_bytes':max(x['combined_rss_bytes']for x in parent),'separately_sampled_allocated_above_reserved':mismatch,'counter_limit':'Sequential allocation/reservation reads do not form one instantaneous pair.'}

def heldout_gate(scores):
    need(len(scores)==12,'Twelve held-out checkpoint scores required');rows=[]
    for i in range(4):
        selected=scores[3*i:3*i+3];need([r['checkpoint']for r in selected]==list(reader.CHECKPOINTS) and all(r['noise_index']==i for r in selected),'Fixed held-out grouping')
        zero,old,new=selected;finite=all(math.isfinite(r['future_contrast_mse']) for r in selected)
        row={'noise_index':i,'lower_than_zero':finite and new['future_contrast_mse']<zero['future_contrast_mse'],'lower_than_old128':finite and new['future_contrast_mse']<old['future_contrast_mse'],'positive_alignment':type(new['cosine'])in(int,float) and math.isfinite(new['cosine']) and new['cosine']>0};row['passed']=all(row[k]for k in('lower_than_zero','lower_than_old128','positive_alignment'));rows.append(row)
    return {'passed':all(r['passed']for r in rows),'per_noise':rows,'criterion':'Strict lower future contrast MSE than zero and old128, and positive target alignment, for every one of four fixed held-out noises.','scope':'Held-out noise on one seen scene; no visual or generalization claim.'}

def source_inputs(root):
    original=root/'reference-inputs/original14';v=original/'visual-inputs'
    values=arrays(v/'sampling-inputs.safetensors');texts=arrays(v/'contexts.safetensors');commands=arrays(v/'commands.safetensors');raw=arrays(original/'original-training-inputs.safetensors')
    expected=np.zeros((1,16,6),np.float32);expected[:,1:,3]=np.float32(math.pi/24)
    need(np.array_equal(commands['closed'],expected),'Original15 left-turn commands');expected[:,0,5]=1;need(np.array_equal(commands['open'],expected),'Original initial interact pulse')
    obs=raw['observation'];need(np.array_equal(obs,values['observation']),'Independent observation identity')
    x=values['initial_noise'].copy();x[:,:1]=obs[0];need(np.array_equal(x,values['initial_latent']),'Exact pure-noise initial input')
    times=np.full((1,4290),999,np.int64);times[:,:858]=0;need(np.array_equal(times,values['token_times']),'Original999 token mask')
    contexts={'positive':texts['atrium'],'negative':texts['native_negative']};cases={}
    for arm in reader.ARMS:
        target=raw[arm+'_target'];noisy=np.float32(.494)*target+np.float32(.506)*raw['noise'];noisy[:,:,:1]=obs
        t=np.full((1,4290),506,np.int64);t[:,:858]=0;cases[arm]=(noisy,t,raw['noise']-target)
    return values,contexts,commands,obs,cases,raw['open_target']-raw['closed_target']

def verify_calls(root,report,values,texts,commands,obs,cases,noises):
    held=report['heldout']['calls'];additional=report['additional_checks']['calls'];expected=[]
    for i,(noise_name,noise)in enumerate(noises.items()):
        x=noise.copy();x[:,:,:1]=obs;t=values['token_times']
        for cp in reader.CHECKPOINTS:
            for arm in reader.ARMS:
                for text in reader.TEXTS:expected.append((f'heldout-{i:04d}-{cp}-{arm}-{text}',cp,arm,text,x,t,noise_name))
    for arm in reader.ARMS:
        x,t,_=cases[arm]
        for cp in reader.CHECKPOINTS:expected.append((f'seen506-{cp}-{arm}-positive',cp,arm,'positive',x,t,None))
    for cp in reader.CHECKPOINTS:
        for arm in reader.ARMS:
            for text in reader.TEXTS:expected.append((f'original999-{cp}-{arm}-{text}',cp,arm,text,values['initial_latent'][None],values['token_times'],None))
    need(len(held)==48 and len(additional)==18 and len(expected)==66,'All66 raw call identities')
    need(report['heldout']['feature_extracts']==8 and report['heldout']['head_predictions']==48 and report['heldout']['target_conditioning'] is False,'Declared held-out helper counts and target isolation')
    need(set(report['raw_prediction_sha256'])=={row[0]+'.safetensors'for row in expected}=={p.name for p in(root/'result').glob('*.safetensors')},'Exact66 raw output filenames')
    calls={}
    for saved,(name,cp,arm,text,x,t,noise)in zip(held+additional,expected):
        actual={'name':name,'checkpoint':cp,'arm':arm,'text':text,'input_sha256':ts(x),'time_sha256':ts(t),'context_sha256':ts(texts[text]),'commands_sha256':ts(commands[arm]),'prediction_sha256':saved['prediction_sha256']}
        if noise is not None:actual['noise']=noise
        need(saved==actual,'Exact call/input/checkpoint/context record '+name)
        path=member(root/'result',name+'.safetensors');need(sha(path)==report['raw_prediction_sha256'][name+'.safetensors'],'Raw prediction file hash')
        value=reader.read_tensor_file(path,{'velocity':(reader.SHAPE,'F32')},sha(path))['velocity'];need(ts(value)==saved['prediction_sha256'],'Raw prediction tensor hash')
        calls[name]=saved
    return calls

def audit(root,recovery_verified,old_comparison,binding_path,binding_sha256,output):
    root=Path(root).absolute();out=Path(output).absolute();need(not out.exists() and not out.resolve().is_relative_to(root) and not out.resolve().is_relative_to(HERE) and not out.resolve().is_relative_to(REPO) and not out.resolve().is_relative_to(Path(old_comparison).resolve()) and not out.resolve().is_relative_to(Path(recovery_verified).resolve().parent) and not any(q.is_symlink() for q in(out,*out.parents)),'Fresh external audit output')
    out.mkdir(parents=True);began=time.monotonic();report={'schema':SCHEMA,'status':'running','model_execution':False,'model_replay':False,'source_sha256':{n:sha(HERE/n)for n in('audit.py','pins.json','reference/reader.py','reference/array_checks.py')},'checks':{}};before={}
    try:
        need(sha(binding_path)==digest(binding_sha256),'Explicit reviewed binding checksum');bound=js(binding_path);fixed=pins()
        need(bound['schema']==BINDING_SCHEMA and bound['status']=='bound' and bound['pins_sha256']==sha(HERE/'pins.json') and bound['auditor_sha256']==sha(__file__),'Frozen auditor binding')
        file_map(root,bound['prepared_files']);plan=validate_prepared(root);need(sha(root/'plan.json')==bound['plan_sha256'],'Bound actual plan')
        before=inventory(root);rec=js(recovery_verified);need(rec['status']=='verified','Verified complete recovery')
        recdir=Path(recovery_verified).resolve().parent;index=js(recdir/'index.json');need(sha(recdir/'index.json')==rec['index_sha256'],'Recovery index identity')
        relative=root.resolve().relative_to(recdir/'recovered').as_posix()
        for name,row in before.items():need(index['files'].get(relative+'/'+name)==row,'Every recovered run file '+name)
        parent=js(root/'metrics.json');worker=js(root/'result/metrics.json');terminal=js(root/'terminal.json');monitor=js(root/'result/monitor-terminal.json');launch=js(root/'launch.json');admit=js(root/'executed-admission.json')
        need(parent['schema']==worker['schema']=='worldline-action-effect-fixed-assessment-v1' and parent['status']==worker['status']=='passed' and parent['model_execution'] is True and worker['model_execution'] is True and worker['model_frozen'] is True,'Complete frozen assessment')
        need(worker['counts']==COUNTS and worker['completed_prediction_files']==66 and worker['source_sha256']==plan['source_sha256'],'Exact counters and sources')
        need(parent['result_sha256']==sha(root/'result/metrics.json') and parent['terminal_sha256']==sha(root/'terminal.json') and parent['plan_sha256']==worker['plan_sha256']==launch['plan_sha256']==bound['plan_sha256'],'Complete result hash links')
        expected={'schema':'worldline-action-effect-assessment-admission-v1','decision':'admit','issued_by':'parent-agent','plan_sha256':bound['plan_sha256'],'source_sha256':plan['source_sha256'],'limits':LIMITS,'counts':COUNTS,'checkpoint_sha256':plan['checkpoint_sha256'],'heldout_manifest_sha256':fixed['heldout_manifest_sha256'],'new_training_audit_sha256':plan['new_training_audit_sha256'],'cpu_report_sha256':plan['cpu_report_sha256'],'training_admitted':False,'sampling_admitted':False}
        need(all(admit.get(k)==v for k,v in expected.items()) and launch['admission_sha256']==sha(root/'executed-admission.json'),'Exact executed assessment admission')
        need(terminal['status']==monitor['status']=='complete' and type(terminal['exit_code'])is int and terminal['exit_code']==0 and terminal['cleanup_error'] is None and not list(root.rglob('watchdog-stop.json')) and not list(root.rglob('*cleanup-error.json')),'Successful process and cleanup')
        for record in(parent,worker,terminal,monitor):need(type(record['elapsed_seconds'])in(int,float) and math.isfinite(record['elapsed_seconds']) and 0<record['elapsed_seconds']<900,'Every stage within900 seconds')
        for record in(worker,terminal,monitor):need(record['limits']==LIMITS,'Unchanged declared memory limits')
        need(type(launch['deadline'])in(int,float) and math.isfinite(launch['deadline']) and launch['deadline']>0,'Finite monotonic deadline')
        expected_outputs={p.relative_to(root/'result').as_posix()for p in(root/'result').rglob('*')if p.is_file()and p.name not in('metrics.json','memory.jsonl')};need(set(worker['output_sha256'])==expected_outputs,'Full result output inventory');file_map(root/'result',worker['output_sha256'])
        ws=[read(x)for x in(root/'result/memory.jsonl').read_bytes().splitlines()];ps=[read(x)for x in(root/'parent-memory.jsonl').read_bytes().splitlines()]
        resources=resource_samples(ws,ps);need(len(ws)==monitor['sample_count'] and resources['peak_combined_rss_bytes']==terminal['peak_combined_rss_bytes'],'Resource counters and peak');report['checks']['resources']=resources|{'parent_seconds':parent['elapsed_seconds'],'worker_seconds':worker['elapsed_seconds'],'limits':LIMITS}
        actual=worker['hardware'];trained=plan['new_training_identity']['hardware'];capacity='total_memory_bytes'
        need(set(actual)==set(trained) and type(actual[capacity])is int and actual[capacity]>=trained[capacity]>=70*2**30 and all(json.dumps(actual[k],sort_keys=True)==json.dumps(trained[k],sort_keys=True)for k in actual if k!=capacity),'Only approved equal-or-greater capacity difference')
        comparison={'policy':'All fields exact except reported total GPU bytes may be greater','training_total_memory_bytes':trained[capacity],'actual_total_memory_bytes':actual[capacity],'additional_reported_bytes':actual[capacity]-trained[capacity]};need(worker['hardware_comparison']==comparison,'Recorded hardware compatibility')
        beforecore=js(root/'result/core-before.json');aftercore=js(root/'result/core-after.json');loaded=js(root/'result/weight-load.json');catalog=js(root/'source/repository/experiments/wan22_native/cuda_reference/expected-weights.json')['tensors']
        need(beforecore==aftercore==plan['new_training_identity']['core_records'] and set(loaded['tensors'])==set(catalog) and len(beforecore)==len(catalog)==825,'Unchanged825 original foundation values')
        for name,row in catalog.items():
            need(beforecore[name]=={'shape':row['shape'],'dtype':'float32','sha256':row['original_sha256']},'Original base tensor '+name);item=loaded['tensors'][name]
            need(item['shape']==row['shape'] and item['source_sha256']==item['loaded_sha256']==row['original_sha256'] and item['original_dtype']==item['loaded_dtype']=='float32' and item['source_owner_released'] is True and item['cuda_copy_exact'] is True,'Original loader record '+name)
        checkpoint_records={};zero_shapes={n:v.shape for n,v in arrays(checkpoint_paths(root)['zero']).items()}
        for cp,path in checkpoint_paths(root).items():
            values=arrays(path);need(len(values)==18 and {n:v.shape for n,v in values.items()}==zero_shapes and all(v.dtype==np.float32 for v in values.values()),'Exact adapter architecture and FP32 dtype');record=worker['checkpoint_records'][cp];need(record['checkpoint_sha256']==sha(path)==plan['checkpoint_sha256'][cp] and record['parameter_count']==sum(x.size for x in values.values())==947712 and record['dtype']=='float32' and record['tensor_sha256']=={n:ts(v)for n,v in values.items()},'Exact retained checkpoint record '+cp);checkpoint_records[cp]=record
        report['checks']['weights']={'original_core_tensors':825,'checkpoint_records':checkpoint_records,'original_fp32_values_unchanged':True}
        values,texts,commands,obs,cases,target=source_inputs(root);_,noises=reader.read_heldout(root/'heldout');calls=verify_calls(root,worker,values,texts,commands,obs,cases,noises)
        heldfiles={n:h for n,h in worker['raw_prediction_sha256'].items()if n.startswith('heldout-')};scores=reader.read_saved_scores(root/'result',heldfiles,worker['heldout']['calls'],target);close_tree(scores,worker['heldout_scores'])
        report['checks']['heldout_scores']=scores;report['numerical_gate']=heldout_gate(scores)
        seen={};original={}
        for cp in reader.CHECKPOINTS:
            seen[cp]={};predictions={}
            for arm in reader.ARMS:
                v=arrays(root/'result'/f'seen506-{cp}-{arm}-positive.safetensors',['velocity'])['velocity'];loss=float(np.mean((v[:,:,1:].astype(np.float64)-cases[arm][2][:,:,1:].astype(np.float64))**2));seen[cp][arm]={'future_flow_mse_fp64':loss,'seen_training_draw':True}
                for text in reader.TEXTS:predictions[arm+'-'+text]=arrays(root/'result'/f'original999-{cp}-{arm}-{text}.safetensors',['velocity'])['velocity']
            original[cp]=reader.score(predictions,target)
        close_tree(seen,worker['additional_checks']['seen506']);close_tree(original,worker['additional_checks']['original999']);report['checks']['seen506']=seen;report['checks']['original999']=original
        old=Path(old_comparison);need(sha(old/'metrics.json')==fixed['original20_parent_sha256'] and sha(old/'result/metrics.json')==fixed['original20_result_sha256'] and sha(old/'plan.json')==fixed['original20_plan_sha256'],'Exact recovered original20 baseline')
        oldreport=js(old/'result/metrics.json');need(worker['runtime_flags']==oldreport['runtime_flags'],'Exact previously measured runtime flags');oldcalls={r['name']:r for r in oldreport['calls']};repeated=[]
        for newname,oldname in fixed['repeat_mapping'].items():
            path=old/'result'/(oldname+'.safetensors');need(sha(path)==oldreport['output_sha256'][path.name],'Original20 raw output hash');a=arrays(root/'result'/(newname+'.safetensors'),['velocity'])['velocity'];b=arrays(path,['velocity'])['velocity'];need(ts(b)==oldcalls[oldname]['prediction_sha256'],'Old raw prediction identity')
            for key,oldkey in [('input_sha256','input_sha256'),('time_sha256','times_sha256'),('commands_sha256','commands_sha256'),('context_sha256','context_sha256')]:need(calls[newname][key]==oldcalls[oldname][oldkey],'Repeated exact input identity')
            d=a.astype(np.float64)-b.astype(np.float64);repeated.append({'new':newname,'old':oldname,'tensor_bytes_equal':a.tobytes()==b.tobytes(),'file_bytes_equal':sha(root/'result'/(newname+'.safetensors'))==sha(path),'maximum_absolute_difference':float(abs(d).max()),'future_difference_rms':float(np.sqrt(np.mean(d[:,:,1:]**2))),'new_tensor_sha256':ts(a),'old_tensor_sha256':ts(b)})
        report['checks']['repeated_original20']={'all12_exact':all(r['tensor_bytes_equal']for r in repeated),'records':repeated,'equivalence_status':'exact' if all(r['tensor_bytes_equal']for r in repeated) else 'unresolved finite cross-run differences','original20_independent_audit_sha256':fixed['original20_audit_sha256'],'claim':'Descriptive whole-core-versus-reused-feature equivalence check; no numerical tolerance or silent substitution.'}
        report['checks']['calls']={'raw_predictions_verified':66,'heldout_feature_extracts_recorded':worker['heldout']['feature_extracts'],'total_feature_extracts_recorded':worker['counts']['feature_extracts'],'identities':list(calls.values())}
        need(inventory(root)==before,'All recovered bytes unchanged')
        report.update(status='passed',identity={'plan_sha256':bound['plan_sha256'],'parent_sha256':sha(root/'metrics.json'),'result_sha256':sha(root/'result/metrics.json'),'checkpoint_sha256':plan['checkpoint_sha256'],'training_audit_sha256':plan['new_training_audit_sha256'],'source_sha256':plan['source_sha256'],'binding_sha256':binding_sha256},recovered_files=len(before),recovered_bytes_unchanged=True,roundoff_record_tolerance=ROUNDING,limitations=['No model, backward or native feature replay. Source and saved counts support12 feature extractions; only66 head outputs are retained.','Numeric gate is separate from raw artifact integrity and from descriptive cross-run equality.','Four held-out noises on one seen scene do not establish successful visual actions, camera control or scene generalization.','FP64 record roundoff tolerance never changes the strict numerical gate inequalities.'])
    except BaseException as error:report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['seconds']=time.monotonic()-began;(out/'inventory.json').write_text(json.dumps(before,indent=2)+'\n');report['inventory_sha256']=sha(out/'inventory.json');(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for name in ('audit.py','pins.json'):shutil.copyfile(HERE/name,out/name)
        shutil.copytree(HERE/'reference',out/'reference');shutil.copyfile(binding_path,out/'binding.json')
        print(json.dumps({'status':report['status'],'report_sha256':sha(out/'report.json'),'seconds':report['seconds']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bind',action='store_true')
    for n in('prepared','plan-review','run-root','recovery-verified','old-comparison','binding','output'):p.add_argument('--'+n,type=Path)
    p.add_argument('--binding-sha256');a=p.parse_args()
    if a.bind:
        if any(getattr(a,n)is None for n in('prepared','plan_review','output')):p.error('Binding requires actual prepared/review/fresh output')
        print(json.dumps(create_binding(a.prepared,a.plan_review,a.output)))
    else:
        if any(getattr(a,n)is None for n in('run_root','recovery_verified','old_comparison','binding','binding_sha256','output')):p.error('Audit requires exact recovered inputs and binding')
        audit(a.run_root,a.recovery_verified,a.old_comparison,a.binding,a.binding_sha256,a.output)
