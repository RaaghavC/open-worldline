from pathlib import Path
p=Path(__file__).with_name('audit.py');s=p.read_text()
def rep(a,b,count=1):
 global s
 assert s.count(a)==count,(a[:70],s.count(a),count)
 s=s.replace(a,b)
rep('fourteen retained predictions','twenty retained predictions')
rep('import argparse,hashlib,json,math,struct','import argparse,hashlib,json,math,struct,shutil,time')
rep("PREPARED=BASE/'work/wan22-action-cuda-diagnostic-prepared-v3'", "PREPARED=BASE/'work/wan22-action-cuda-comparison128-prepared-v1'")
rep("PROGRAM=BASE/'work/wan22-action-cuda-diagnostic-prep-v1'", "PROGRAM=BASE/'work/wan22-action-cuda-comparison128-prep-v1'\nORIGINAL_PROGRAM=BASE/'work/wan22-action-cuda-diagnostic-prep-v1'")
rep("PLAN_SHA='c17ad42baf1dcd9898c7fcecb1aa3458113576ad92dd977c9ee7376e7f50c1e6'", "PLAN_SHA='6622575e7c3e66aeb283834bd2b43d7567cd28d66d11be6e4f7b016aaaaee9e6'\nORIGINAL_PLAN_SHA='c17ad42baf1dcd9898c7fcecb1aa3458113576ad92dd977c9ee7376e7f50c1e6'\nTRAINING_AUDIT_SHA='e987d7b31ce754977ce8ee547aa9e2cf4a2b31afc136a4f57b893de6950caa7b'\nCHECKPOINT_SHA='ac1160b22531804c473746d2fb8ee9edd75710e0e43d6f1eabc86d268c864996'\nCPU_SHA='9d902564fe0746ec81a727ec22f1e90a508e1300cd9414ff9045b7edd5e9cff2'\nCPS=('0','16','128')\nLIMITS={'seconds':900.,'host_rss_bytes':48*2**30,'cuda_reserved_bytes':60*2**30,\n        'minimum_host_available_bytes':8*2**30,'minimum_cuda_available_bytes':8*2**30,'minimum_gpu_total_bytes':70*2**30}")
rep('def prepared_inputs(root):','def original_inputs(root):')
rep("sha(plan_path)==PLAN_SHA,'Exact prepared-v3 plan required'", "sha(plan_path)==ORIGINAL_PLAN_SHA,'Exact original14 prepared-v3 plan required'")
rep("sha(PROGRAM/'diagnostic.py')==plan['source_sha256']['diagnostic']['diagnostic.py']", "sha(ORIGINAL_PROGRAM/'diagnostic.py')==plan['source_sha256']['diagnostic']['diagnostic.py']")
rep("worker.get('predictions')==14", "worker.get('predictions')==20")
rep("worker.get('completed_prediction_files')==14", "worker.get('completed_prediction_files')==20")
rep("'Fourteen completed current predictions required'", "'Twenty completed current predictions required'")
rep("    names=['native-'+text for text in TEXTS]\n    for cp in ('0','16'):\n        names += ['objective-'+cp+'-'+arm for arm in ARMS]\n        names += ['causal-'+cp+'-'+arm+'-'+text for arm in ARMS for text in TEXTS]", "    names=prediction_names()")
rep("'Fourteen call names/order differ'", "'Twenty call names/order differ'")
rep("for cp in ('0','16')", "for cp in CPS",count=2)
rep("'Exact 18 saved tensor files required'", "'Exact 26 saved tensor files required'")
rep("'worldline-fourteen-prediction-independent-audit-v1'", "'worldline-twenty-prediction-independent-audit-v1'")
rep("'predictions_verified':14,'command_delta_files_verified':4", "'predictions_verified':20,'command_delta_files_verified':6")
rep("'Four losses use", "'Six losses use")
rep("'predictions':14,'model_execution':False", "'predictions':20,'model_execution':False")
rep("'loss_reduction_comparison_tolerance':LOSS_TOL", "'relative_objective_improvement_128_vs16_float64':{a:1-objective['128'][a]['float64_mse']/objective['16'][a]['float64_mse'] if objective['16'][a]['float64_mse'] else None for a in ARMS},\n        'relative_objective_improvement_128_vs0_float64':{a:1-objective['128'][a]['float64_mse']/objective['0'][a]['float64_mse'] if objective['0'][a]['float64_mse'] else None for a in ARMS},\n        'checkpoint128_sha256':CHECKPOINT_SHA,'training_audit_sha256':TRAINING_AUDIT_SHA,\n        'zero_native_bit_exact_comparisons':sum(v['exact_equal'] for v in parity.values()),\n        'resource_checks':resources,'executed_admission_sha256':sha(root/'executed-admission.json'),\n        'loss_reduction_comparison_tolerance':LOSS_TOL")
rep("    inventory={}\n", "    resources=check_runtime(root,plan,parent,worker,terminal)\n    inventory={}\n")
rep("'Original 825 before/after records differ')", "'Original 825 before/after records differ')\n    require(expected==plan['training_identity']['core_records'], 'Same actual128 frozen foundation required')")
# Validate each reported improvement from exactly its own recorded loss scalars.
rep("    return {'schema':'worldline-twenty", "    for key,new,prior in [('relative_objective_improvement','16','0'),('relative_objective_improvement_128_vs16','128','16'),('relative_objective_improvement_128_vs0','128','0')]:\n        for arm in ARMS:\n            x=worker['objective'][prior][arm]; expected_value=1-worker['objective'][new][arm]/x if x else None\n            require(worker[key][arm]==expected_value,'Recorded matched-loss improvement arithmetic differs')\n    return {'schema':'worldline-twenty")
# Keep report writes outside measured directories and retain unchanged inventory.
rep("    require(not a.output.exists(),'Fresh report directory required');a.output.mkdir(parents=True)\n    try:report=audit(a.run)", "    a.run=a.run.resolve();a.output=a.output.resolve()\n    require(not a.output.exists() and not a.output.is_relative_to(a.run) and not a.output.is_relative_to(REPO.resolve()),'Fresh report outside measured run/repository required');a.output.mkdir(parents=True)\n    shutil.copyfile(__file__,a.output/'audit.py');before=inventory_tree(a.run);started=time.monotonic()\n    try:\n        report=audit(a.run)\n        require(inventory_tree(a.run)==before,'Measured files changed during audit')\n        report.update(recovered_bytes_unchanged=True,recovered_files=len(before),recovered_bytes=sum(x['bytes'] for x in before.values()),elapsed_seconds=time.monotonic()-started)\n        (a.output/'inventory.json').write_text(json.dumps(before,indent=2)+'\\n')\n        report['full_inventory_sha256']=sha(a.output/'inventory.json')")
anchor='def audit(root):'
helpers='''def prediction_names():
    names=['native-'+text for text in TEXTS]
    for cp in CPS:
        names += ['objective-'+cp+'-'+arm for arm in ARMS]
        names += ['causal-'+cp+'-'+arm+'-'+text for arm in ARMS for text in TEXTS]
    return names


def training_binding(plan,completed,training_sources):
    trained=plan['training_identity']
    require(trained['checkpoint_sha256']==CHECKPOINT_SHA==plan['checkpoint_128']['checkpoint_sha256'], 'Exact final128 checkpoint required')
    identity={k:trained[k] for k in ('parent_sha256','worker_sha256','training_sha256','terminal_sha256','plan_sha256')}
    identity.update(final_checkpoint_manifest_sha256=trained['checkpoint_manifest_sha256'],final_checkpoint_sha256=CHECKPOINT_SHA,source_sha256=training_sources)
    require(completed['schema']=='worldline-action-cuda-fixed128-actual-independent-v1' and completed['status']=='passed'
            and completed['completed_updates']==128 and completed['foundation_values_unchanged'] is True
            and completed['final_checkpoint_only'] is True and completed['identity']==identity, 'Completed128 audit identity differs')


def prepared_inputs(root):
    root=Path(root);require(sha(file(root,'plan.json'))==PLAN_SHA,'Exact actual20 plan required');plan=read_json(root/'plan.json')
    require(plan['schema']=='worldline-action-fixed-input-comparison128-v1' and plan['status']=='prepared'
            and plan['predictions']==20 and plan['native_predictions']==2 and plan['checkpoint_order']==list(CPS)
            and plan['training'] is False and plan['generation'] is False and plan['limits']==LIMITS,'Fixed comparison contract differs')
    for group in ('diagnostic','reference14','training_reader'):
        for name,digest in plan['source_sha256'][group].items():
            require(sha(file(root/'source'/group,name))==digest,'Retained comparison source differs')
            source=(PROGRAM if group=='diagnostic' else PROGRAM/group)/name
            require(sha(source)==digest,'Reviewed comparison source differs')
    require(sha(file(root,'cpu-report.json'))==CPU_SHA==plan['cpu_report_sha256'],'Exact comparison CPU report required')
    cpu=read_json(root/'cpu-report.json')
    require(cpu['status']=='passed' and cpu['tests']==5 and cpu['source_sha256']==plan['source_sha256']
            and all(cpu[k]==0 for k in ('failures','errors','skipped')),'Passed current comparison CPU checks required')
    original,old,cases,values,contexts,commands=original_inputs(root/'original14')
    require(plan['original14_plan_sha256']==ORIGINAL_PLAN_SHA and plan['source_sha256']['visual']==original['source_sha256']['visual'],'Unchanged original14 input/source identity')
    require(sha(file(root,'training-audit.json'))==TRAINING_AUDIT_SHA==plan['training_audit_sha256'],'Actual128 audit bytes differ')
    require(sha(file(root,'checkpoint-0128.safetensors'))==CHECKPOINT_SHA,'Actual128 checkpoint bytes differ')
    training_binding(plan,read_json(root/'training-audit.json'),read_json(root/'source/training_reader/training-source.json'))
    for name,digest in plan['training_evidence_sha256'].items():require(sha(file(root,name))==digest,'Retained actual128 evidence differs')
    trained=plan['training_identity']
    links={'metrics.json':'parent_sha256','worker/metrics.json':'worker_sha256','terminal.json':'terminal_sha256','training/metrics.json':'training_sha256','plan.json':'plan_sha256','training/checkpoint-0128/manifest.json':'checkpoint_manifest_sha256'}
    require(all(plan['training_evidence_sha256']['training-evidence/'+n]==trained[k] for n,k in links.items()),'Actual128 evidence hash links differ')
    require(read_json(root/'training-evidence/training/core-before.json')==read_json(root/'training-evidence/training/core-after.json')==trained['core_records']==old['input_identity']['training']['core_records'],'Unchanged original/128 foundation')
    require(all(trained['training_input_identity'][k]==old['input_identity']['training']['training_input_identity'][k] for k in ('cache','text')),'Unchanged original/128 cache and text')
    return plan,old,cases,values,contexts,commands


def inventory_tree(root):
    result={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'No linked artifacts')
        if p.is_file():result[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size}
    return result


def check_runtime(root,plan,parent,worker,terminal):
    admission=read_json(file(root,'executed-admission.json'));launch=read_json(file(root,'launch.json'))
    wanted={'schema':'worldline-action-fixed-input-comparison128-admission-v1','decision':'admit','issued_by':'parent-agent',
        'plan_sha256':PLAN_SHA,'source_sha256':plan['source_sha256'],'limits':LIMITS,'training_admitted':False,
        'generation_admitted':False,'predictions':20,'cpu_report_sha256':CPU_SHA,
        'training_audit_sha256':TRAINING_AUDIT_SHA,'checkpoint128_sha256':CHECKPOINT_SHA}
    require(all(admission.get(k)==v for k,v in wanted.items()),'Exact parent comparison admission required')
    require(launch['admission_sha256']==sha(root/'executed-admission.json') and launch['plan_sha256']==PLAN_SHA
            and type(launch['deadline']) in (int,float) and math.isfinite(launch['deadline']),'Launch admission/deadline binding differs')
    monitor=read_json(file(root/'result','monitor-terminal.json'))
    require(monitor['status']=='complete' and not list(root.rglob('*cleanup-error.json')),'Completed monitor and cleanup required')
    for row in (worker,terminal,monitor):
        require(row['limits']==LIMITS and type(row['elapsed_seconds']) in (int,float) and 0<row['elapsed_seconds']<900,'Unchanged measured stage caps')
    require(type(parent['elapsed_seconds']) in (int,float) and 0<parent['elapsed_seconds']<900,'Parent within fixed900 cap')
    actual=worker['hardware'];trained=plan['training_identity']['hardware'];key='total_memory_bytes'
    require(set(actual)==set(trained) and type(actual[key]) is int and type(trained[key]) is int
            and actual[key]>=trained[key]>=LIMITS['minimum_gpu_total_bytes'],'Actual/trained GPU capacity contract')
    other=lambda x:json.dumps({k:v for k,v in x.items() if k!=key},sort_keys=True,allow_nan=False)
    require(other(actual)==other(trained),'All other hardware/runtime values and types exact')
    require(worker['hardware_comparison']=={'policy':'All fields exact except reported total GPU bytes may be greater',
        'training_total_memory_bytes':trained[key],'actual_total_memory_bytes':actual[key],'additional_reported_bytes':actual[key]-trained[key]},'Recorded GPU compatibility differs')
    rows=lambda p:[parse(x) for x in file(root,p).read_bytes().splitlines() if x]
    samples=rows('result/memory.jsonl');parents=rows('parent-memory.jsonl')
    require(samples and parents and len(samples)==monitor['sample_count'],'Complete retained resource samples')
    for row in samples:
        for field in ('host_rss_bytes','cuda_reserved_bytes','cuda_allocated_bytes','host_available_bytes','cuda_available_bytes'):
            require(type(row[field]) is int and row[field]>=0,'Finite nonnegative integer resource value')
        require(row['host_rss_bytes']<=LIMITS['host_rss_bytes'] and row['cuda_reserved_bytes']<=LIMITS['cuda_reserved_bytes']
                and row['cuda_allocated_bytes']<=row['cuda_reserved_bytes'] and row['host_available_bytes']>=LIMITS['minimum_host_available_bytes']
                and row['cuda_available_bytes']>=LIMITS['minimum_cuda_available_bytes'],'Worker sample beyond unchanged limits')
    for row in parents:
        require(type(row['combined_rss_bytes']) is int and 0<=row['combined_rss_bytes']<=LIMITS['host_rss_bytes']
                and type(row['host_available_bytes']) is int and row['host_available_bytes']>=LIMITS['minimum_host_available_bytes'],'Combined process sample beyond limits')
    require(max(r['combined_rss_bytes'] for r in parents)==terminal['peak_combined_rss_bytes']
            and min(r['host_available_bytes'] for r in parents)==terminal['minimum_host_available_bytes'],'Parent extrema differ')
    return {'passed':True,'worker_samples':len(samples),'parent_samples':len(parents),'parent_seconds':parent['elapsed_seconds'],
            'worker_seconds':worker['elapsed_seconds'],'peak_cuda_reserved_bytes':max(r['cuda_reserved_bytes'] for r in samples),
            'limits':LIMITS,'sampled_limits_only':True,'provider_billing_termination_proved':False}


'''
rep(anchor,helpers+anchor)
rep("shutil.copyfile(__file__,a.output/'audit.py');before=inventory_tree(a.run);started=time.monotonic()", "shutil.copyfile(__file__,a.output/'audit.py');before=inventory_tree(a.run);source_before=sources();started=time.monotonic()")
rep("        require(inventory_tree(a.run)==before,'Measured files changed during audit')", "        require(inventory_tree(a.run)==before,'Measured files changed during audit')\n        require(sources()==source_before==report['source_sha256'],'Audit or bound source changed during inspection')")
p.write_text(s)
