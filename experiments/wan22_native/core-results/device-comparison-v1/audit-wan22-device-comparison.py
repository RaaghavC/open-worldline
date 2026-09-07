"""Independent NumPy verification of saved CPU/MPS pair evidence; no model code."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def score(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1)
    assert a.shape==b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    d=b-a;rr=float(np.sqrt(np.mean(a*a)));err=float(np.sqrt(np.mean(d*d)))
    denom=math.sqrt(float(np.sum(a*a))*float(np.sum(b*b)))
    return {'elements':a.size,'mean_absolute_difference':float(np.mean(abs(d))),
        'root_mean_square_difference':err,'maximum_absolute_difference':float(np.max(abs(d))),
        'cpu_root_mean_square':rr,'relative_rms_difference':err/rr if rr else None,
        'cosine_similarity':float(np.sum(a*b))/denom if denom else None,
        'exactly_equal_elements':int(np.count_nonzero(a==b))}

def main():
    parser=argparse.ArgumentParser()
    for name in ('cpu','mps','comparison','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    reference=json.loads(args.comparison.read_text());roots={'cpu':args.cpu,'mps':args.mps};runs={};loads={};inputs={};outputs={};files={}
    for name,root in roots.items():
        runs[name]=r=json.loads((root/'metrics.json').read_text());terminal=json.loads((root/'terminal.json').read_text())
        assert r['status']=='passed' and r['device']==name and terminal['status']=='complete' and terminal['exit_code']==0
        assert not (root/'watchdog-stop.json').exists()
        assert (r['max_seconds'],r['max_memory_gib'],r['minimum_available_gib'])==(900,18,2)
        assert r['full_video_generated'] is False and r['quality_measured'] is False
        assert sha(root/'metrics.json')==reference[name+'_metrics_sha256']
        files[name]={f:sha(root/f) for f in ('metrics.json','terminal.json','inputs.safetensors','outputs.safetensors','weight-load.json')}
        for filename,digest in r['output_sha256'].items():assert sha(root/filename)==digest
        loads[name]=json.loads((root/'weight-load.json').read_text());assert sha(root/'weight-load.json')==r['weight_load_sha256']
        inputs[name]=load_file(root/'inputs.safetensors');outputs[name]=load_file(root/'outputs.safetensors')
        assert set(outputs[name])=={'positive_velocity','negative_velocity','guided_velocity','one_step_latent'}
        for key,array in outputs[name].items():
            assert array.shape==(48,5,18,32) and array.dtype==np.float32 and np.isfinite(array).all()
            assert tensor_sha(array)==reference['output_tensor_hashes'][name][key]
        assert np.array_equal(outputs[name]['one_step_latent'][:,:1],inputs[name]['observation'][0])
    identities=('source_sha256','reused_source_sha256','sampling_configuration','seed','latent_shape','tokens','observed_tokens','text','observation')
    assert all(runs['cpu'][key]==runs['mps'][key] for key in identities)
    assert loads['cpu']==loads['mps'] and len(loads['cpu']['tensors'])==825
    assert sum(math.prod(t['shape']) for t in loads['cpu']['tensors'].values())==4_999_787_712
    assert all(t['converted_values_exact'] is True for t in loads['cpu']['tensors'].values())
    assert inputs['cpu'].keys()==inputs['mps'].keys()
    for key,array in inputs['cpu'].items():assert np.array_equal(array,inputs['mps'][key])
    scalar_count=0;max_delta=0.;details={}
    for key in sorted(outputs['cpu']):
        a,b=outputs['cpu'][key],outputs['mps'][key]
        sections={'all_latents':(a,b),'observed_latent':(a[:,:1],b[:,:1]),'future_latents':(a[:,1:],b[:,1:])}
        checked={name:score(*pair) for name,pair in sections.items()}
        checked['per_latent_frame']=[score(a[:,i],b[:,i]) for i in range(5)]
        for section in ('all_latents','observed_latent','future_latents','per_latent_frame'):
            actual=checked[section];expected=reference['comparison'][key][section]
            actual_rows=actual if isinstance(actual,list) else [actual];expected_rows=expected if isinstance(expected,list) else [expected]
            for actual_row,expected_row in zip(actual_rows,expected_rows):
                assert actual_row.keys()==expected_row.keys()
                for metric,value in actual_row.items():
                    known=expected_row[metric];scalar_count+=1
                    if isinstance(value,int) or value is None:assert value==known
                    else:
                        delta=abs(value-known);max_delta=max(delta,max_delta)
                        assert math.isclose(value,known,rel_tol=1e-12,abs_tol=2e-13),(key,section,metric,value,known)
        details[key]=checked['future_latents']
    guidance=[]
    for name in roots:
        pos=outputs[name]['positive_velocity'];neg=outputs[name]['negative_velocity']
        equation=neg+np.float32(5)*(pos-neg)
        guidance.append({'device':name,'saved_guidance_vs_explicit_np_float32_max_abs':float(np.max(abs(equation-outputs[name]['guided_velocity'])))})
    report={'schema':'wan22-device-comparison-independent-v1','status':'passed_readonly_arithmetic_audit',
      'scalar_values_checked':scalar_count,'maximum_scalar_difference_from_original':max_delta,
      'numeric_audit_tolerance':{'absolute':2e-13,'relative':1e-12,'scope':'float64 reduction-order differences only; no model quality acceptance threshold'},
      'all_input_tensors_exact':True,'all825_weight_metadata_records_equal':True,'parameter_count_from_shapes':4_999_787_712,
      'source_text_observation_sampler_identities_equal':True,'known_prefix_exact_both':True,
      'future_latent_comparison':details,'guidance_arithmetic':guidance,'input_file_sha256':files,
      'original_comparison_sha256':sha(args.comparison),'original_comparison_source_sha256':reference['source_sha256'],
      'independent_audit_source_sha256':sha(Path(__file__)),'model_code_imported':False,'official_weight_values_read':False,
      'model_inference_executed':False,'gpu_used':False,
      'limits':'One initial positive/negative pair and one solver step. CPU also uses the declared selective-BF16 policy; neither execution is full-FP32 or official CUDA ground truth. These differences neither establish full-rollout equivalence nor identify the cause of failed video quality.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'scalars':scalar_count,'max_scalar_delta':max_delta,'guidance':guidance,'report_sha256':sha(args.output)},indent=2))
if __name__=='__main__':main()
