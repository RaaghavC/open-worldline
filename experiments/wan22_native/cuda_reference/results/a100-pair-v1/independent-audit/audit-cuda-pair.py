"""Independent retained CUDA-pair audit. Standard library and NumPy only."""
import os
import sys
_HERE=os.path.dirname(os.path.realpath(__file__))
sys.path[:]=[p for p in sys.path if os.path.realpath(p or os.getcwd())!=_HERE]
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/'recovered-pair-v1/results/pair-run-v1'
REPO=HERE.parents[1]/'outputs/open-worldline'
COMPARISON=HERE/'cuda-pair-comparison-v1.json'
COMPARISON_SHA='2dd5328841b3f1c424ce50f86e00c645fbed40afde0edbf6533557c9bc8878b3'
OUT=HERE/'pair-independent-audit-v1.json'
KEYS=('positive_velocity','negative_velocity','guided_velocity')
SHAPE=(48,5,18,32)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def data_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def obj(path):return json.loads(Path(path).read_text())


def arrays(path):
    raw=Path(path).read_bytes();assert 8<=len(raw)<=8*2**20
    count=struct.unpack('<Q',raw[:8])[0];assert 2<=count<=2**20 and count+8<=len(raw)
    header=json.loads(raw[8:count+8]);body=memoryview(raw)[count+8:];values={};intervals=[]
    for name,row in header.items():
        if name=='__metadata__':continue
        assert row['dtype']in ('F32','I64')
        dtype=np.dtype('<f4'if row['dtype']=='F32'else'<i8');lo,hi=row['data_offsets']
        assert 0<=lo<hi<=len(body)and hi-lo==math.prod(row['shape'])*dtype.itemsize
        a=np.frombuffer(body[lo:hi],dtype=dtype).reshape(row['shape']);assert np.isfinite(a).all()
        values[name]=a;intervals.append((lo,hi))
    intervals.sort();assert intervals[0][0]==0 and intervals[-1][1]==len(body)
    assert all(a[1]==b[0]for a,b in zip(intervals,intervals[1:]))
    return values


def stats(actual,baseline,official):
    a=np.array(actual,dtype=np.float64).ravel();b=np.array(baseline,dtype=np.float64).ravel();o=np.array(official,dtype=np.float64).ravel()
    assert a.shape==b.shape==o.shape and a.size
    d=a-b;rmse=float(np.sqrt(np.mean(d*d)));normalizer=float(np.sqrt(np.mean(o*o)))
    product=float(np.linalg.norm(a)*np.linalg.norm(b))
    return {'elements':int(a.size),'exact_value_equal':bool(np.array_equal(a,b)),
        'nonzero_difference_elements':int(np.count_nonzero(d)),'maximum_absolute_difference':float(np.max(np.abs(d))),
        'mean_absolute_difference':float(np.mean(np.abs(d))),'rmse':rmse,
        'comparison_baseline_rms':float(np.sqrt(np.mean(b*b))),
        'official_cpu_normalization_rms':normalizer,'rmse_over_official_cpu_rms':rmse/normalizer if normalizer else None,
        'cosine':float(np.dot(a,b)/product)if product else None}


def main():
    if OUT.exists():raise ValueError('Fresh independent audit output required')
    assert sha(COMPARISON)==COMPARISON_SHA
    raw_before={str(p.relative_to(ROOT)):sha(p)for p in ROOT.rglob('*')if p.is_file()}
    parent=obj(ROOT/'metrics.json');worker=obj(ROOT/'core/result/metrics.json');terminal=obj(ROOT/'core/terminal.json')
    assert parent['status']==worker['status']=='passed'and parent['mode']==worker['mode']=='pair'
    assert parent['execute_requested']is True and parent['model_execution']is True
    assert terminal['status']=='complete'and terminal['exit_code']==0 and worker['predictions']==2 and worker['solver_updates']==0
    assert worker['settings']=={'steps':50,'shift':5.,'guidance':5.}
    assert worker['finite_outputs']and worker['input_tensors_unchanged']and not worker['actions_read']and not worker['future_target_read']
    assert not list(ROOT.rglob('*watchdog-stop*'))
    assert sha(ROOT/'core/result/metrics.json')==parent['core_report_sha256']
    assert sha(ROOT/'core/terminal.json')==parent['core_terminal_sha256']
    package=REPO/'experiments/wan22_native/cuda_reference'
    expected_sources={}
    for node in ast.parse((package/'evidence.py').read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name)and target.id in ('NAMES','SHARED'):
                    names=ast.literal_eval(node.value)
                    expected_sources.update({('shared/'+n if target.id=='SHARED'else n):sha((REPO if target.id=='SHARED'else package)/n)for n in names})
    assert parent['source_sha256']==worker['source_sha256']==expected_sources
    for name,digest in expected_sources.items():assert sha(ROOT/'measured-source'/(name+'.txt'))==digest
    assert sha(ROOT/'cpu-report.json')==parent['gates']['cpu_report_sha256']
    assert sha(ROOT/'independent-report.json')==parent['gates']['independent_report_sha256']
    author=obj(ROOT/'cpu-report.json');review=obj(ROOT/'independent-report.json')
    assert author['status']==review['status']=='passed'and author['tests']==9 and review['tests']==12
    assert author['source_sha256']==expected_sources
    assert review['source_sha256']==dict(expected_sources,**{'test_independent.py':sha(ROOT/'independent-test.py.txt')})
    assert sha(ROOT/'independent-test.py.txt')==sha(package/'test_independent.py')
    official_root=REPO/'experiments/wan22_native/official_cpu/results/pair-v1'
    portable_root=REPO/'experiments/wan22_native/core-results'
    for name,digest in parent['copied_input_sha256'].items():
        assert sha(ROOT/name)==digest==sha(official_root/name)
    assert set(parent['copied_input_sha256'])=={'inputs.safetensors','contexts.safetensors','text-manifest.json'}
    assert parent['input_identity']==worker['input_identity']==obj(official_root/'metrics.json')['input_identity']
    inputs=arrays(ROOT/'inputs.safetensors');contexts=arrays(ROOT/'contexts.safetensors')
    assert set(inputs)=={'initial_noise','initial_latent','observation','token_times'}and set(contexts)=={'atrium','native_negative'}
    assert inputs['initial_noise'].shape==inputs['initial_latent'].shape==SHAPE
    assert inputs['observation'].shape==(1,48,1,18,32)and inputs['token_times'].shape==(1,720)
    assert contexts['atrium'].shape==(25,4096)and contexts['native_negative'].shape==(126,4096)
    assert np.array_equal(inputs['initial_latent'][:,:1],inputs['observation'][0])
    assert np.array_equal(inputs['initial_latent'][:,1:],inputs['initial_noise'][:,1:])
    assert (inputs['token_times'][:,:144]==0).all()and (inputs['token_times'][:,144:]==999).all()
    for name,a in inputs.items():assert data_sha(a)==parent['input_identity']['input_tensor_sha256'][name]
    for name,label in [('atrium','positive'),('native_negative','negative')]:assert data_sha(contexts[name])==parent['input_identity']['text'][label+'_tensor_sha256']
    loaded=obj(ROOT/'core/result/weight-load.json');original=obj(portable_root/'cpu-pair-v1/weight-load.json')['tensors']
    assert set(loaded['tensors'])==set(original)and len(original)==loaded['tensor_count']==825
    assert loaded['all_shards_verified']is True and loaded['cuda_copy_exact']is True and loaded['convert_model_dtype']is False
    total=0
    for name,row in loaded['tensors'].items():
        saved=original[name];assert row['source_sha256']==row['loaded_sha256']==saved['original_sha256']
        assert row['shape']==saved['shape']and row['shard']==saved['shard']
        assert row['original_dtype']==row['loaded_dtype']==saved['original_dtype']=='float32'
        assert row['cuda_copy_exact']and row['source_owner_released'];total+=math.prod(row['shape'])
    assert total==loaded['parameter_count']==4999787712 and total*4==loaded['parameter_bytes']==19999150848
    for name,digest in worker['output_sha256'].items():assert sha(ROOT/'core/result'/name)==digest
    cuda=arrays(ROOT/'core/result/outputs.safetensors');assert set(cuda)==set(KEYS)
    saved={'official_cpu':arrays(official_root/'result/outputs.safetensors'),
           'portable_cpu':arrays(portable_root/'cpu-pair-v1/outputs.safetensors'),
           'portable_mps':arrays(portable_root/'pair-v1/outputs.safetensors')}
    for label,subset in [('positive',{'positive_velocity'}),('negative',{'positive_velocity','negative_velocity'})]:
        partial=arrays(ROOT/f'core/result/completed-{label}.safetensors');assert set(partial)==subset
        assert all(np.array_equal(partial[k],cuda[k])for k in subset)
    for values in [cuda,*saved.values()]:
        assert all(values[k].shape==SHAPE and values[k].dtype==np.dtype('<f4')and np.isfinite(values[k]).all()for k in KEYS)
    comparison=obj(COMPARISON);assert comparison['source_sha256']==sha(HERE/'compare-cuda.py')
    checked=0;future={}
    for label,baseline in saved.items():
        future[label]={}
        for key in KEYS:
            for region,cut in [('all',slice(None)),('observed',slice(0,1)),('future',slice(1,5))]:
                row=stats(cuda[key][:,cut],baseline[key][:,cut],saved['official_cpu'][key][:,cut])
                assert row==comparison['cuda_minus_saved_baseline'][label][key][region],(label,key,region)
                checked+=1
                if region=='future':assert row['elements']==110592;future[label][key]=row['rmse_over_official_cpu_rms']
    guidance={}
    for label,values in {'cuda':cuda,**saved}.items():
        recomputed=np.add(values['negative_velocity'],np.multiply(np.float32(5),np.subtract(values['positive_velocity'],values['negative_velocity'],dtype=np.float32),dtype=np.float32),dtype=np.float32)
        guidance[label]={'exact':bool(np.array_equal(recomputed,values['guided_velocity'])),
            'maximum_absolute_residual':float(np.abs(recomputed.astype(np.float64)-values['guided_velocity'].astype(np.float64)).max())}
        for region,cut in [('all',slice(None)),('observed',slice(0,1)),('future',slice(1,5))]:
            row=stats(values['guided_velocity'][:,cut],recomputed[:,cut],recomputed[:,cut])
            row['reference_rms']=row.pop('official_cpu_normalization_rms');row['rmse_over_reference_rms']=row.pop('rmse_over_official_cpu_rms');row.pop('comparison_baseline_rms')
            assert row==comparison['guidance_equation_residuals'][label][region]
    gpu=[json.loads(line)for line in (ROOT/'core/result/memory.jsonl').read_text().splitlines()]
    host=[json.loads(line)for line in (ROOT/'core/parent-memory.jsonl').read_text().splitlines()]
    limits=worker['limits'];assert limits==parent['limits']and limits['seconds']==900
    assert max(r['combined_rss_bytes']for r in host)==terminal['peak_combined_rss_bytes']<=limits['host_rss_bytes']
    assert min(r['host_available_bytes']for r in host)==terminal['minimum_host_available_bytes']>=limits['minimum_host_available_bytes']
    assert max(r['cuda_reserved_bytes']for r in gpu)<=limits['cuda_reserved_bytes']
    assert min(r['cuda_available_bytes']for r in gpu)>=limits['minimum_cuda_available_bytes']
    assert parent['elapsed_seconds']<900 and worker['elapsed_seconds']<900
    assert raw_before=={str(p.relative_to(ROOT)):sha(p)for p in ROOT.rglob('*')if p.is_file()}
    result={'status':'passed','scope':'Independent saved-file audit only; no model, CUDA, cloud or credential calls',
        'source_sha256':sha(__file__),'comparison_sha256':COMPARISON_SHA,'raw_files_unchanged':True,'raw_file_sha256':raw_before,
        'matched_original_fp32_tensors':825,'matched_parameter_count':total,'input_and_context_files_exact':True,
        'clean_observed_prefix_and_144_zero_times_exact':True,'finite_velocity_arrays':12,
        'cross_run_metric_rows_recomputed_exact':checked,'guidance_metric_rows_recomputed_exact':12,
        'normalization':'Fixed streamed official CPU RMS for each region and velocity, including CUDA versus portable CPU and MPS.',
        'future_values_per_velocity':110592,'future_rmse_over_official_cpu_rms':future,'guidance_residuals':guidance,
        'resources':{'parent_seconds':parent['elapsed_seconds'],'worker_seconds':worker['elapsed_seconds'],'load_seconds':worker['load_seconds'],
            'pair_seconds':worker['pair_seconds'],'peak_combined_host_rss_bytes':terminal['peak_combined_rss_bytes'],
            'peak_sampled_cuda_reserved_bytes':max(r['cuda_reserved_bytes']for r in gpu),
            'peak_sampled_cuda_allocated_bytes':max(r['cuda_allocated_bytes']for r in gpu),
            'minimum_sampled_cuda_free_bytes':min(r['cuda_available_bytes']for r in gpu),'sampled_guards_passed':True},
        'limitations':'First-step predictions only. No new numerical acceptance threshold, CUDA equivalence, full-trajectory agreement, image-quality or failure-cause conclusion. Host available memory is the psutil-reported system figure, not a cgroup quota measurement.'}
    with OUT.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'passed','audit_sha256':sha(OUT),'metric_rows':checked+12,'future_relative_rmse':future},indent=2))


if __name__=='__main__':main()
