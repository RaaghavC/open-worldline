"""Small synthetic file tests; no new training, assessment, or model execution."""
import copy, importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('actual_plan_review',HERE/'review.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def put(path,value):path.write_text(json.dumps(value))

def plan():
    return {'schema':'worldline-action-effect-fixed-assessment-v1','status':'prepared',
        'checkpoint_order':['zero','old128','new128'],'counts':{'heldout_heads':48,'seen506_heads':6,'original999_heads':12,'head_predictions':66,'feature_extracts':12},
        'training':False,'sampling':False,'endpoint_scale':1.0,'guidance':5.0,'model_time':999,'seen_objective_time':506,
        'scope':'Four held-out noises on one seen scene, plus distinct original seen506 and original visual999 checks; final checkpoint only',
        'checkpoint_sha256':{'zero':'a'*64,'old128':'b'*64,'new128':'c'*64},'source_sha256':{},'new_training_audit_sha256':'d'*64,
        'limits':{},'heldout_manifest_sha256':'e'*64,'cpu_report_sha256':'f'*64}

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
    def test_exact_protocol_rejects_changes_and_boolean_numbers(self):
        r.protocol(plan())
        for key,value in [('guidance',4.0),('model_time',998),('training',0),('endpoint_scale',True),('scope','other'),('checkpoint_order',['new128','old128','zero'])]:
            p=plan();p[key]=value
            with self.assertRaises(ValueError):r.protocol(p)
        p=plan();p['counts']['head_predictions']=65
        with self.assertRaises(ValueError):r.protocol(p)
    def test_current_frozen_dependencies_import_without_torch(self):
        import sys
        self.assertNotIn('torch',sys.modules)
        a=r.audit_module();self.assertEqual(a.SCHEMA,'worldline-action-effect-assessment-actual-audit-v1')
        self.assertNotIn('torch',sys.modules)
    def test_failed_input_retains_failure_and_no_binding(self):
        prepared=self.root/'prepared';prepared.mkdir();put(prepared/'plan.json',{'synthetic':True})
        out=self.root/'failed.json';binding=self.root/'binding.json'
        with self.assertRaises(ValueError):r.review(prepared,out,binding)
        report=json.loads(out.read_text());self.assertEqual(report['status'],'failed');self.assertFalse(binding.exists())
        with self.assertRaises(ValueError):r.review(prepared,out,binding)
    def test_successful_synthetic_review_handoff_and_used_input_rejection(self):
        a=r.audit_module();prepared=self.root/'prepared';prepared.mkdir();p=plan();put(prepared/'plan.json',p)
        put(prepared/'new-training-audit.json',{'completed_updates':128,'auxiliary_updates':32,'auxiliary_feature_extracts':64,'auxiliary_head_predictions':128,'main_predictions':256})
        obs=np.zeros((1,1,1,1,1),dtype=np.float32);x=np.zeros((1,1,5,1,1),dtype=np.float32)
        target=x.copy();target[:,:,1:]=1
        inputs=({'initial_latent':x[0]}, {'positive':x,'negative':x}, {'closed':x,'open':x},obs,{'closed':(x,None,None),'open':(x,None,None)},target)
        def bind(_prepared,review,out):
            saved=json.loads(review.read_text());self.assertEqual(saved['schema'],r.SCHEMA);self.assertEqual(saved['status'],'passed');self.assertEqual(saved['checkpoint_sha256'],p['checkpoint_sha256']);put(out,{'review_sha256':r.sha(review)});return {'status':'bound'}
        with patch.object(r,'audit_module',return_value=a),patch.object(a,'validate_prepared',return_value=p),patch.object(a,'source_inputs',return_value=inputs),patch.object(a,'create_binding',side_effect=bind):
            result=r.review(prepared,self.root/'passed.json',self.root/'bound.json');self.assertEqual(result['binding']['status'],'bound')
            put(prepared/'execution-attempt.json',{})
            with self.assertRaises(ValueError):r.review(prepared,self.root/'used.json',self.root/'unused.json')
            self.assertFalse((self.root/'unused.json').exists())
        with self.assertRaises(ValueError):r.output_path(prepared/'nested.json',prepared)

if __name__=='__main__':unittest.main(verbosity=2)
