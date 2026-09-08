"""Bounded saved-byte and synthetic failure fixtures; no Torch/model import."""
import copy,json,math,shutil
from pathlib import Path
import numpy as np
import unittest,tempfile
from unittest.mock import patch
import audit as a
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]

def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def score_rows():
    return [{'noise_index':i,'checkpoint':cp,'future_contrast_mse':loss,'cosine':.5}for i in range(4)for cp,loss in zip(a.reader.CHECKPOINTS,(1.,.8,.4))]

class AuditTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup);self.root=Path(folder.name).resolve()

    def test_strict_predeclared_gate_requires_every_noise(self):
        rows=score_rows();assert a.heldout_gate(rows)['passed']
        for key,value in [('future_contrast_mse',.8),('cosine',0.),('cosine',None),('future_contrast_mse',float('nan'))]:
            bad=copy.deepcopy(rows);bad[-1][key]=value;assert not a.heldout_gate(bad)['passed']
        bad=copy.deepcopy(rows);bad[0]['checkpoint']='new128'
        with self.assertRaises(ValueError):a.heldout_gate(bad)
        # Descriptive serialization tolerance is not the model gate's inequality.
        near=copy.deepcopy(rows);near[-1]['future_contrast_mse']=.8+1e-14
        a.close_tree(.8,near[-1]['future_contrast_mse']);assert not a.heldout_gate(near)['passed']

    def test_resource_caps_reject_corruption_without_instantaneous_counter_assumption(self):
        w={'host_rss_bytes':100,'cuda_reserved_bytes':1000,'cuda_allocated_bytes':1010,'host_available_bytes':9*2**30,'cuda_available_bytes':9*2**30};p={'combined_rss_bytes':110,'host_available_bytes':9*2**30}
        report=a.resource_samples([w],[p]);assert len(report['separately_sampled_allocated_above_reserved'])==1
        for key,value in [('host_rss_bytes',49*2**30),('cuda_reserved_bytes',61*2**30),('host_available_bytes',8*2**30-1),('cuda_available_bytes',8*2**30-1),('cuda_reserved_bytes',True),('host_rss_bytes',float('nan'))]:
            bad=dict(w);bad[key]=value
            with self.assertRaises(ValueError):a.resource_samples([bad],[p])
        with self.assertRaises(ValueError):a.resource_samples([w],[{'combined_rss_bytes':49*2**30,'host_available_bytes':9*2**30}])

    def test_fp32_guidance_endpoint_and_future_scoring_oracle(self):
        shape=a.reader.SHAPE
        predictions={k:np.full(shape,v,np.float32)for k,v in [('closed-positive',.125),('closed-negative',.25),('open-positive',.25),('open-negative',.125)]}
        target=np.full(shape,.125,np.float32);result=a.reader.score(predictions,target)
        assert result['future_contrast_mse']==1.5625 and result['normalized_contrast_mse']==100 and math.isclose(result['cosine'],-1.,rel_tol=1e-12)
        predictions['closed-positive'][:,:,:1]=10000;assert a.reader.score(predictions,target)['future_contrast_mse']==1.5625
        bad=dict(result);bad['future_contrast_mse']+=.01
        with self.assertRaises(ValueError):a.close_tree(result,bad)

    def test_copied_safe_reader_source_and_original12_mapping(self):
        fixed=a.pins();program=BASE/'work/wan22-action-effect-evaluation-prep-v1'
        assert a.sha(HERE/'reference/reader.py')==a.sha(program/'reader.py')
        assert len(fixed['repeat_mapping'])==12 and len(set(fixed['repeat_mapping'].values()))==12
        actual=BASE/'work/wan22-action-cuda-post128-recovered-final-v1/recovered/action-results/comparison128-spatial-v1'
        assert a.sha(actual/'result/metrics.json')==fixed['original20_result_sha256']
        report=a.js(actual/'result/metrics.json');calls={r['name']:r for r in report['calls']}
        for new,old in fixed['repeat_mapping'].items():
            path=actual/'result'/(old+'.safetensors');assert a.sha(path)==report['output_sha256'][path.name]
            v=a.arrays(path,['velocity'])['velocity'];assert a.ts(v)==calls[old]['prediction_sha256']
            assert old.startswith('objective-') if new.startswith('seen506-') else old.startswith('causal-')

    def test_actual_original_inputs_reconstruct_exact506_and999_hashes(self):
        tmp_path=self.root
        original=BASE/'work/wan22-action-cuda-comparison128-prepared-v1/original14';root=tmp_path/'fixture'
        for name in ['original-training-inputs.safetensors','visual-inputs/sampling-inputs.safetensors','visual-inputs/contexts.safetensors','visual-inputs/commands.safetensors']:
            dst=root/'reference-inputs/original14'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/name,dst)
        values,texts,commands,obs,cases,target=a.source_inputs(root)
        actual=BASE/'work/wan22-action-cuda-post128-recovered-final-v1/recovered/action-results/comparison128-spatial-v1';calls={r['name']:r for r in a.js(actual/'result/metrics.json')['calls']}
        for arm in a.reader.ARMS:
            row=calls['objective-0-'+arm];assert row['input_sha256']==a.ts(cases[arm][0]) and row['times_sha256']==a.ts(cases[arm][1]);assert row['commands_sha256']==a.ts(commands[arm])
            assert np.array_equal(cases[arm][0][:,:,:1],obs)
            for text,old in [('positive','atrium'),('negative','native_negative')]:
                row=calls['causal-0-'+arm+'-'+old];assert row['input_sha256']==a.ts(values['initial_latent'][None]) and row['context_sha256']==a.ts(texts[text])
        assert not np.array_equal(cases['closed'][0][:,:,1:],cases['open'][0][:,:,1:]) and np.any(target[:,:,1:])

    def test_later_binding_needs_actual_review_and_fresh_output(self):
        tmp_path=self.root
        prepared=tmp_path/'prepared';prepared.mkdir();put(prepared/'plan.json',{'fixture':'not an actual run'})
        plan={'source_sha256':{},'checkpoint_sha256':{'new128':'a'*64},'new_training_audit_sha256':'b'*64}
        review={'schema':a.PLAN_REVIEW_SCHEMA,'status':'passed','plan_sha256':a.sha(prepared/'plan.json'),'source_sha256':{},'checkpoint_sha256':plan['checkpoint_sha256'],'new_training_audit_sha256':'b'*64,'counts':a.COUNTS}
        path=tmp_path/'review.json';put(path,review);output=tmp_path/'binding.json'
        patcher=patch.object(a,'validate_prepared',return_value=plan);patcher.start();self.addCleanup(patcher.stop)
        result=a.create_binding(prepared,path,output);assert result['status']=='bound' and a.js(output)['model_execution'] is False
        with self.assertRaises(ValueError):a.create_binding(prepared,path,output)
        bad=dict(review);bad['new_training_audit_sha256']='c'*64;put(path,bad)
        with self.assertRaises(ValueError):a.create_binding(prepared,path,tmp_path/'wrong.json')
        assert not(tmp_path/'wrong.json').exists()
        (prepared/'execution-attempt.json').write_text('already used')
        with self.assertRaises(ValueError):a.create_binding(prepared,path,tmp_path/'late.json')

    def test_safe_relative_paths_and_serialized_numbers(self):
        tmp_path=self.root
        for name in ['../outside','/absolute']:
            with self.assertRaises(ValueError):a.member(tmp_path,name)
        for digest in [None,True,'0'*63,'A'*64]:
            with self.assertRaises(ValueError):a.digest(digest)
        with self.assertRaises(ValueError):a.close_tree({'score':.5},{'score':float('inf')})
        with self.assertRaises(ValueError):a.close_tree({'flag':True},{'flag':1})

if __name__=='__main__':unittest.main(verbosity=2)
