"""Independent tiny analytic/parser checks for the post-run comparator."""
from pathlib import Path
import hashlib,importlib.util,json,math,struct,tempfile,time,shutil,sys,copy
from unittest import mock
import numpy as np
from safetensors.numpy import load_file,save_file
base=Path(__file__).resolve().parent.parent;source=base/'work/compare-wan22-official-reference.py';start=time.perf_counter()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before=sha(source)
spec=importlib.util.spec_from_file_location('comparator_under_review',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
records=[]
def oracle(a,b):
    a=[float(v)for v in a];b=[float(v)for v in b];delta=[x-y for x,y in zip(a,b)]
    mse=math.fsum(x*x for x in delta)/len(delta);den=math.fsum(x*x for x in b)/len(b)
    norm=math.sqrt(math.fsum(x*x for x in a))*math.sqrt(math.fsum(x*x for x in b))
    return dict(elements=len(a),exact_value_equal=a==b,nonzero_difference_elements=sum(x!=y for x,y in zip(a,b)),
        maximum_absolute_difference=max(abs(x)for x in delta),mean_absolute_difference=math.fsum(abs(x)for x in delta)/len(delta),
        rmse=math.sqrt(mse),reference_rms=math.sqrt(den),rmse_over_reference_rms=math.sqrt(mse/den)if den else None,
        cosine=math.fsum(x*y for x,y in zip(a,b))/norm if norm else None)
for name,a,b in [('signed',[1,-2,4],[2,-1,-4]),('orthogonal',[1,0],[0,2]),('zero_actual',[0,0],[3,4]),('zero_reference',[2,-2],[0,0]),('equal',[1,-2,3],[1,-2,3])]:
    a=np.array(a,dtype=np.float32);b=np.array(b,dtype=np.float32);actual=module.statistics(a,b);expected=oracle(a,b)
    for key,value in expected.items():
        if value is None or type(value)in (bool,int):assert actual[key]==value,(name,key)
        else:assert math.isclose(actual[key],value,rel_tol=2e-15,abs_tol=2e-15),(name,key)
    records.append(dict(check=name,status='passed',metrics=actual))
for a,b in [([],[]),([np.inf],[1.]),([np.nan],[0.]),([[1,2]],[1,2])]:
    try:module.statistics(a,b)
    except ValueError:pass
    else:raise AssertionError('Invalid metric input accepted')
records.append(dict(check='nonfinite_empty_shape_rejection',status='passed',cases=4))
files=[]
for label,(directory,expected)in module.BASELINES.items():
    root=base/'outputs/open-worldline/experiments/wan22_native/core-results'/directory
    report=json.loads((root/'metrics.json').read_text());assert report['status']=='passed'
    assert report['device']==('cpu'if label=='portable_cpu'else'mps')
    for filename,digest in [('inputs.safetensors',module.INPUT_SHA256),('outputs.safetensors',expected)]:
        p=root/filename;assert sha(p)==digest==report['output_sha256'][filename]
        actual=module.tensors(p);reference=load_file(p)
        assert set(actual)==set(reference)
        for name in actual:assert actual[name].dtype==reference[name].dtype and np.array_equal(actual[name],reference[name])
        files.append(dict(label=label,file=filename,sha256=digest,tensors=len(actual),parser_equals_safetensors_numpy=True))
    velocity=module.tensors(root/'outputs.safetensors')['positive_velocity']
    assert velocity.shape==(48,5,18,32) and velocity[:,:1].size==27648 and velocity[:,1:].size==110592
records.append(dict(check='native_time_axis_partition',status='passed',observed_elements=27648,future_elements=110592,total_elements=138240))
parser_cases=[]
with tempfile.TemporaryDirectory()as tmp:
    root=Path(tmp)
    variants={
      'valid_scalar':({'x':{'dtype':'F32','shape':[],'data_offsets':[0,4]}},np.array([3.],dtype='<f4').tobytes()),
      'valid_zero_size':({'x':{'dtype':'F32','shape':[0],'data_offsets':[0,0]}},b''),
      'gap':({'x':{'dtype':'F32','shape':[1],'data_offsets':[4,8]}},b'\0'*8),
      'trailing':({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},b'\0'*8),
      'shape_mismatch':({'x':{'dtype':'F32','shape':[2],'data_offsets':[0,4]}},b'\0'*4),
      'overlap':({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]},'y':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},b'\0'*4),
      'unsupported_dtype':({'x':{'dtype':'F16','shape':[2],'data_offsets':[0,4]}},b'\0'*4),
      'invalid_metadata':({'__metadata__':{'bad':12},'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},b'\0'*4),
    }
    for name,(header,body)in variants.items():
        raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
        p=root/(name+'.safetensors');p.write_bytes(struct.pack('<Q',len(raw))+raw+body)
        outcomes={}
        for key,reader in [('comparator',module.tensors),('library',load_file)]:
            try:reader(p);outcomes[key]='accepted'
            except Exception as error:outcomes[key]='rejected:'+type(error).__name__
        if name in ('gap','trailing','shape_mismatch','overlap','unsupported_dtype','invalid_metadata'):assert outcomes['comparator'].startswith('rejected')
        if name in ('valid_scalar','valid_zero_size'):assert outcomes['comparator']=='accepted'
        parser_cases.append(dict(case=name,**outcomes))
    record='"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}'
    raw=('{'+record+','+record+'}').encode();raw+=b' '*((-len(raw))%8);p=root/'duplicate.safetensors';p.write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*4)
    outcomes={}
    for key,reader in [('comparator',module.tensors),('library',load_file)]:
        try:reader(p);outcomes[key]='accepted'
        except Exception as error:outcomes[key]='rejected:'+type(error).__name__
    parser_cases.append(dict(case='duplicate_json_key',**outcomes))
coupling_checks={}
for label,(directory,_)in module.BASELINES.items():
    values=module.tensors(base/'outputs/open-worldline/experiments/wan22_native/core-results'/directory/'outputs.safetensors')
    coupling=module.guidance_residual(values)
    assert coupling['all']['exact_value_equal']
    coupling_checks[label]=coupling

# Complete metadata fixtures bind the actual pinned inputs and existing CPU
# outputs, but are not an executed official-model reference.
gates=[]
with tempfile.TemporaryDirectory()as temporary:
    temp=Path(temporary);fake_repo=temp/'repo';official=fake_repo/'experiments/wan22_native/official_cpu'
    real_official=base/'outputs/open-worldline/experiments/wan22_native/official_cpu'
    official.mkdir(parents=True)
    for p in real_official.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            target=official/p.relative_to(real_official);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    declarations=[node.value for node in module.ast.parse((official/'run.py').read_text()).body if isinstance(node,module.ast.Assign)
        and any(isinstance(target,module.ast.Name)and target.id=='NAMES'for target in node.targets)]
    names=module.ast.literal_eval(declarations[0]);sources={n:sha(official/n)for n in names}
    baseline_root=fake_repo/'experiments/wan22_native/core-results'
    for _,(directory,_)in module.BASELINES.items():
        target=baseline_root/directory;target.mkdir(parents=True)
        for name in ('inputs.safetensors','outputs.safetensors'):
            shutil.copyfile(base/'outputs/open-worldline/experiments/wan22_native/core-results'/directory/name,target/name)
    run=temp/'run';(run/'result').mkdir(parents=True)
    shutil.copyfile(baseline_root/'cpu-pair-v1/inputs.safetensors',run/'inputs.safetensors')
    values={n:v for n,v in load_file(baseline_root/'cpu-pair-v1/outputs.safetensors').items()if n in module.VELOCITIES}
    save_file(values,str(run/'result/outputs.safetensors'))
    for name in names:
        dest=run/'measured-source'/(name+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(official/name,dest)
    worker=dict(status='passed',device='cpu',completed_predictions=2,finite_outputs=True,caller_inputs_unchanged=True,
        passes=[dict(prediction='positive'),dict(prediction='negative')],source_sha256=sources,input_identity={'fixture':'no official inference'},
        output_sha256={'outputs.safetensors':sha(run/'result/outputs.safetensors')})
    (run/'result/metrics.json').write_text(json.dumps(worker))
    parent=dict(status='passed',weights_loaded=True,model_execution=True,pair_predictions=2,solver_steps=0,
        source_sha256=sources,input_identity=worker['input_identity'],input_file_sha256=module.INPUT_SHA256,
        result_metrics_sha256=sha(run/'result/metrics.json'))
    terminal=dict(status='complete',exit_code=0)
    (run/'metrics.json').write_text(json.dumps(parent));(run/'terminal.json').write_text(json.dumps(terminal))
    base_files={p:p.read_bytes()for p in run.rglob('*')if p.is_file()}
    official_files={p:p.read_bytes()for p in official.rglob('*')if p.is_file()}
    def restore():
        for p,content in base_files.items():p.write_bytes(content)
        for p,content in official_files.items():p.write_bytes(content)
        stop=run/'result/watchdog-stop.json'
        if stop.exists():stop.unlink()
    def update_json(path,change):
        value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value))
    for case in ('valid','parent_planned','terminal_failed','terminal_nonzero','one_prediction','extra_pass','worker_hash',
                 'worker_input_mismatch','input_bytes','output_bytes','snapshot_bytes','current_source_bytes','source_map','watchdog','extra_output'):
        restore()
        if case=='parent_planned':update_json(run/'metrics.json',lambda r:r.update(status='planned'))
        elif case=='terminal_failed':update_json(run/'terminal.json',lambda r:r.update(status='failed'))
        elif case=='terminal_nonzero':update_json(run/'terminal.json',lambda r:r.update(exit_code=1))
        elif case=='one_prediction':update_json(run/'result/metrics.json',lambda r:r.update(completed_predictions=1))
        elif case=='extra_pass':update_json(run/'result/metrics.json',lambda r:r['passes'].append(dict(prediction='extra')))
        elif case=='worker_hash':update_json(run/'metrics.json',lambda r:r.update(result_metrics_sha256='0'*64))
        elif case=='worker_input_mismatch':update_json(run/'result/metrics.json',lambda r:r.update(input_identity={'changed':True}))
        elif case=='input_bytes':(run/'inputs.safetensors').write_bytes(b'changed')
        elif case=='output_bytes':(run/'result/outputs.safetensors').write_bytes(b'changed')
        elif case=='snapshot_bytes':(run/'measured-source/reference.py.txt').write_text('changed')
        elif case=='current_source_bytes':(official/'reference.py').write_text('changed')
        elif case=='source_map':update_json(run/'metrics.json',lambda r:r['source_sha256'].pop('reference.py'))
        elif case=='watchdog':(run/'result/watchdog-stop.json').write_text('{}')
        elif case=='extra_output':
            extra=dict(values,one_step_latent=values['guided_velocity']);save_file(extra,str(run/'result/outputs.safetensors'))
            update_json(run/'result/metrics.json',lambda r:r['output_sha256'].update({'outputs.safetensors':sha(run/'result/outputs.safetensors')}))
            update_json(run/'metrics.json',lambda r:r.update(result_metrics_sha256=sha(run/'result/metrics.json')))
        try:got,identity=module.load_reference_run(fake_repo,run);outcome='accepted'
        except (ValueError,KeyError):outcome='rejected'
        assert outcome==('accepted'if case=='valid'else'rejected'),(case,outcome)
        gates.append(dict(case=case,outcome=outcome))
    restore();output=temp/'comparison.json'
    with mock.patch.object(sys,'argv',[str(source),'--repo',str(fake_repo),'--reference-run',str(run),'--output',str(output)]):module.main()
    completed=json.loads(output.read_text());assert completed['status']=='computed' and completed['reference_guidance_exact_required']
    assert completed['comparisons']['portable_cpu']['velocities']['positive_velocity']['all']['exact_value_equal']
    assert set(completed['guidance_equation_residuals'])=={'official_reference','portable_cpu','portable_mps'}
    gates.append(dict(case='end_to_end_fixture_only',outcome='computed with exact coupling'))
    changed={k:v.copy()for k,v in values.items()};changed['guided_velocity'].flat[0]+=1
    save_file(changed,str(run/'result/outputs.safetensors'))
    update_json(run/'result/metrics.json',lambda r:r['output_sha256'].update({'outputs.safetensors':sha(run/'result/outputs.safetensors')}))
    update_json(run/'metrics.json',lambda r:r.update(result_metrics_sha256=sha(run/'result/metrics.json')))
    rejected=temp/'must-not-exist.json'
    with mock.patch.object(sys,'argv',[str(source),'--repo',str(fake_repo),'--reference-run',str(run),'--output',str(rejected)]):
        try:module.main()
        except ValueError as error:assert 'guidance differs'in str(error)
        else:raise AssertionError('Inconsistent reference guidance accepted')
    assert not rejected.exists();gates.append(dict(case='inconsistent_reference_guidance',outcome='rejected before report write'))
assert sha(source)==before
result=dict(schema='worldline-reference-comparator-independent-v2',status='passed',source_sha256=before,
    test_source_sha256=sha(Path(__file__)),source_unchanged_after_checks=True,seconds=time.perf_counter()-start,numpy_version=np.__version__,
    analytic=records,known_saved_files=files,parser_cases=parser_cases,reference_run_gate_cases=gates,guidance_checks=coupling_checks,
    limitations=['No forthcoming official reference run was executed or certified.','End-to-end reference-run metadata uses an explicitly synthetic fixture containing known portable CPU outputs, not a newly executed official reference.','No official-model comparison result or image-quality result was produced by these checks.'])
out=base/'work/wan22-reference-comparator-independent-v2.json'
if out.exists():raise ValueError('No overwrite')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],source_sha256=before,report_sha256=sha(out),analytic_cases=len(records),known_files=len(files),reference_gate_cases=len(gates),guidance_exact={k:v['all']['exact_value_equal']for k,v in coupling_checks.items()}),indent=2))
