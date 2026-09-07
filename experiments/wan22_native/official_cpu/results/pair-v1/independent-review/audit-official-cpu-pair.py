"""Independent completed-reference audit. NumPy/stdlib only, no Torch or models."""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def tensor_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensors(path,contract):
    raw=Path(path).read_bytes();assert len(raw)<8*2**20
    n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n]);offset=8+n
    assert set(header)-{'__metadata__'}==set(contract)
    result={}
    for name,(dtype,shape)in contract.items():
        h=header[name];assert h['dtype']==dtype and h['shape']==list(shape)
        lo,hi=h['data_offsets'];assert 0<=lo<hi<=len(raw)-offset
        a=np.frombuffer(raw[offset+lo:offset+hi],dtype={'F32':'<f4','I64':'<i8'}[dtype]).reshape(shape)
        assert np.isfinite(a).all();result[name]=a
    return result


def compare(saved,current):
    a=saved.astype(np.float64).ravel();b=current.astype(np.float64).ravel();delta=b-a
    rms=math.sqrt(float(np.mean(a*a)));rmse=math.sqrt(float(np.mean(delta*delta)))
    denom=math.sqrt(float(np.sum(a*a)*np.sum(b*b)))
    return {'elements':a.size,'different_elements':int(np.count_nonzero(a!=b)),
        'saved_portable_cpu_rms':rms,'current_official_reference_rms':math.sqrt(float(np.mean(b*b))),
        'difference_rmse':rmse,'rmse_over_saved_portable_cpu_rms':rmse/rms if rms else None,
        'max_abs':float(abs(delta).max()),'cosine':float(np.sum(a*b))/denom if denom else None,
        'exact':bool(np.array_equal(a,b))}


def ast_audit(run):
    original=run/'measured-source/vendor/model.py.txt';translated=run/'translated-model.py.txt'
    record=json.loads((run/'ast-translation.json').read_text())
    assert sha(original)==record['original_source_sha256'] and sha(translated)==record['translated_source_sha256']
    left,right=ast.parse(original.read_text()),ast.parse(translated.read_text())
    calls=lambda tree:[n for n in ast.walk(tree)if isinstance(n,ast.Call)and ast.unparse(n.func)=='torch.amp.autocast']
    a,b=calls(left),calls(right);assert len(a)==len(b)==len(record['changed_calls'])==7
    for old,new in zip(a,b):
        assert ast.literal_eval(old.args[0])=='cuda'
        assert ast.unparse(new)=="torch.amp.autocast('cpu', enabled=False)"
        new.args=copy.deepcopy(old.args);new.keywords=copy.deepcopy(old.keywords)
    assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)
    return {'seven_context_changes_only':True,'complete_reversed_ast_exact':True,
        'original_source_sha256':sha(original),'translated_source_sha256':sha(translated)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--saved-cpu-pair',type=Path,required=True)
    p.add_argument('--source-package',type=Path,required=True);p.add_argument('--text-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh audit output required')
    root=a.run;result=root/'result';old=a.saved_cpu_pair
    originals={str(f.relative_to(root)):sha(f)for f in sorted(root.rglob('*'))if f.is_file()}
    parent=json.loads((root/'metrics.json').read_text());worker=json.loads((result/'metrics.json').read_text())
    terminal=json.loads((root/'terminal.json').read_text());launch=json.loads((root/'launch.json').read_text())
    assert parent['status']==worker['status']=='passed' and terminal['status']=='complete' and terminal['exit_code']==0
    assert worker['completed_predictions']==2 and parent['pair_predictions']==2 and parent['solver_steps']==0
    assert not list(root.rglob('watchdog-stop.json'))
    assert parent['result_metrics_sha256']==sha(result/'metrics.json')
    assert parent['source_sha256']==worker['source_sha256']==launch['source_sha256']
    for name,digest in parent['source_sha256'].items():
        assert sha(root/'measured-source'/(name+'.txt'))==digest==sha(a.source_package/name)
    author=json.loads((root/'cpu-report.json').read_text());independent=json.loads((root/'independent-report.json').read_text())
    assert author['status']==independent['status']=='passed' and author['tests']>=10 and independent['tests']>=12
    assert author['source_sha256']==parent['source_sha256']
    independent_sources=dict(parent['source_sha256']);independent_sources['test_independent.py']=sha(root/'independent-test.py.txt')
    assert independent['source_sha256']==independent_sources
    assert sha(a.source_package/'test_independent.py')==independent_sources['test_independent.py']
    assert sha(root/'cpu-report.json')==parent['gates']['author_cpu_report_sha256']
    assert sha(root/'independent-report.json')==parent['gates']['independent_report_sha256']
    identity=parent['input_identity'];assert identity==worker['input_identity']==launch['input_identity']
    for name,digest in identity['pair_files'].items():assert sha(old/name)==digest
    for name,digest in identity['text_files'].items():assert sha(a.text_cache/name)==digest
    assert sha(root/'inputs.safetensors')==sha(old/'inputs.safetensors')==parent['input_file_sha256']
    assert sha(root/'contexts.safetensors')==sha(a.text_cache/'embeddings.safetensors')
    assert sha(root/'text-manifest.json')==sha(a.text_cache/'manifest.json')
    shape=(48,5,18,32)
    values=tensors(root/'inputs.safetensors',{'initial_noise':('F32',shape),'initial_latent':('F32',shape),
        'observation':('F32',(1,48,1,18,32)),'token_times':('I64',(1,720))})
    for name,v in values.items():assert tensor_sha(v)==identity['input_tensor_sha256'][name]
    assert np.array_equal(values['initial_latent'][:,:1],values['observation'][0])
    assert np.array_equal(values['initial_latent'][:,1:],values['initial_noise'][:,1:])
    assert np.count_nonzero(values['token_times'][:,:144])==0 and np.all(values['token_times'][:,144:]==999)
    contexts=tensors(root/'contexts.safetensors',{'atrium':('F32',(25,4096)),'native_negative':('F32',(126,4096))})
    assert tensor_sha(contexts['atrium'])==identity['text']['positive_tensor_sha256']
    assert tensor_sha(contexts['native_negative'])==identity['text']['negative_tensor_sha256']
    weights=json.loads((result/'weights-used.json').read_text());prior=json.loads((old/'weight-load.json').read_text())
    new=weights['original_fp32_tensors'];saved=prior['tensors']
    assert len(new)==len(saved)==weights['count']==825 and set(new)==set(saved) and weights['all_shards_verified']
    for name,row in new.items():
        baseline=saved[name]
        assert row['source_sha256']==row['loaded_sha256']==baseline['original_sha256']
        assert row['original_dtype']==row['loaded_dtype']==baseline['original_dtype']=='float32'
        assert row['shape']==baseline['shape'] and row['shard']==baseline['shard'] and row['source_owner_released']
    groups=['patch_embedding','time_embedding','time_projection','text_embedding',*[f'blocks.{i}'for i in range(30)],'head']
    names={g:[n for n in saved if n.startswith(g+'.')]for g in groups}
    groupbytes={g:sum(math.prod(saved[n]['shape'])*4 for n in names[g])for g in groups}
    assert len(groups)==35 and sum(map(len,names.values()))==825
    ownership=[json.loads(line)for line in (result/'ownership.jsonl').read_text().splitlines()]
    assert len(ownership)==140
    passes=[]
    for j,label in enumerate(['positive','negative']):
        rows=ownership[j*70:(j+1)*70];released=0
        for i,group in enumerate(groups):
            loaded,evicted=rows[2*i:2*i+2]
            assert loaded['prediction']==evicted['prediction']==label and loaded['group']==evicted['group']==group
            assert loaded['event']=='loaded' and evicted['event']=='evicted'
            assert set(loaded['live_parameter_names'])==set(names[group]) and len(loaded['live_parameter_names'])==len(names[group])
            assert loaded['live_parameter_bytes']==groupbytes[group] and not evicted['live_parameter_names']
            assert math.isfinite(loaded['seconds']) and loaded['seconds']>=0
            released+=len(names[group]);assert evicted['released_parameter_owners_total']==released
            output=evicted['output'];assert output['dtype']==('torch.bfloat16'if group in ('patch_embedding','text_embedding')else'torch.float32')
            assert all(math.isfinite(output[k])for k in ('min','max','rms')) and output['min']<=output['max'] and output['rms']>=0
        declared=worker['passes'][j]
        assert declared['prediction']==label and declared['groups']==groups and declared['all_parameters_meta_after_pass']
        assert declared['released_parameter_owners']==825 and declared['peak_live_parameter_bytes']==max(groupbytes.values())
        passes.append({'prediction':label,'groups':35,'loaded_events':35,'evicted_events':35,'released_parameter_owners':825,
            'source_owner_release_records':825,'peak_live_parameter_bytes':max(groupbytes.values()),'recorded_live_parameter_names_empty_after_every_group':True})
    for name,digest in worker['output_sha256'].items():assert sha(result/name)==digest
    keys=['positive_velocity','negative_velocity','guided_velocity']
    output=tensors(result/'outputs.safetensors',{k:('F32',shape)for k in keys})
    original=tensors(old/'outputs.safetensors',{k:('F32',shape)for k in [*keys,'one_step_latent']})
    for label,subset in [('positive',['positive_velocity']),('negative',['positive_velocity','negative_velocity'])]:
        part=tensors(result/f'completed-{label}.safetensors',{k:('F32',shape)for k in subset})
        assert all(np.array_equal(part[k],output[k])for k in subset)
    assert np.array_equal(output['guided_velocity'],output['negative_velocity']+np.float32(5)*(output['positive_velocity']-output['negative_velocity']))
    comparisons={k:{'all':compare(original[k],output[k]),'observed':compare(original[k][:,:1],output[k][:,:1]),
        'future':compare(original[k][:,1:],output[k][:,1:]),'per_latent_frame':[compare(original[k][:,i],output[k][:,i])for i in range(5)]}for k in keys}
    memory=[json.loads(line)for line in(root/'memory.jsonl').read_text().splitlines()]
    assert memory and max(r['combined_rss_bytes']for r in memory)==terminal['peak_combined_rss_bytes']<=18*2**30
    assert min(r['available_bytes']for r in memory)==terminal['minimum_available_bytes']>=2*2**30
    assert parent['elapsed_seconds']<900 and worker['elapsed_seconds']<900 and terminal['elapsed_seconds']<900
    assert worker['max_seconds']==900 and worker['max_rss_bytes']==18*2**30 and worker['minimum_available_bytes']==2*2**30
    assert worker['caller_inputs_unchanged'] and worker['finite_outputs'] and worker['no_solver_steps']
    assert all(worker[k]is False for k in ('actions_read','future_target_read','automatic_gpu_execution','original_worldline_model','full_video_generated'))
    assert originals=={str(f.relative_to(root)):sha(f)for f in sorted(root.rglob('*'))if f.is_file()}
    record={'status':'passed','method':'Independent NumPy/stdlib artifact reads only. No Torch import, full-weight reads or model execution.',
        'audit_source_sha256':sha(__file__),'raw_run_unchanged':True,'raw_file_sha256':originals,
        'original_fp32_source_tensors_matching_saved_official_shards':825,'total_original_fp32_parameter_bytes':sum(groupbytes.values()),
        'shared_input_file_byte_exact':True,'shared_context_file_byte_exact':True,'native_initial_times_and_clean_prefix_exact':True,
        'ast_translation':ast_audit(root),'passes':passes,'ownership_events':140,
        'guidance_recomputed_fp32_exact':True,'completed_partial_files_match_final':True,
        'all_output_file_hashes_checked':True,'source_report_and_review_hashes_checked':True,
        'reference_vs_saved_portable_cpu':comparisons,'no_numeric_acceptance_threshold':True,
        'resources':{'parent_seconds':parent['elapsed_seconds'],'worker_seconds':worker['elapsed_seconds'],
            'peak_sampled_combined_rss_bytes':terminal['peak_combined_rss_bytes'],'minimum_sampled_available_bytes':terminal['minimum_available_bytes'],
            'memory_samples':len(memory),'timings':worker['timings']},
        'limitations':'Ownership checks audit recorded events and runtime source; no post hoc direct inspection of exited-process memory. Initial-pair differences do not establish CUDA parity, all-step agreement, visual quality or cause of prior image failure.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'passed','audit_sha256':sha(a.output),'matched_original_tensors':825,'ownership_events':140}))


if __name__=='__main__':main()
