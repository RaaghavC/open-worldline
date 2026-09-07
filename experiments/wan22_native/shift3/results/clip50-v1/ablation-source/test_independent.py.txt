# SPDX-License-Identifier: Apache-2.0
"""Independent CPU control, solver and interrupted worker checks."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock
import torch
from . import sample as s

CONTROL=s.native.HERE/'sample-results/clip50-v1'

class FakeGuard:
    def __init__(self,out,device):
        self.output=Path(out);self.output.mkdir();self.report={'status':'running','timings':[]}
    def save(self):s.native.atomic_write(self.output/'metrics.json',self.report)
    def measure(self,name,fn):return fn()
    def close(self,error):self.report['status']='failed' if error else 'passed';self.save()

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def inputs(self):return s.native.read_tensors(CONTROL/'inputs.safetensors',s.native.RUN_KEYS)
    def config(self,root):
        for n in ('inputs.safetensors','profile-weight-load.json'):shutil.copyfile(CONTROL/n,root/n)
        return dict(stage='core',device='cpu',output=str(root/'result'),run=str(root),weights='unused',deadline=time.monotonic()+900,
            ablation_settings=s.SETTINGS,ablation_source_sha256=s.source_hashes(independent=True),
            source_sha256={n:s.native.sha256(s.native.HERE/n)for n in s.native.SOURCES},
            reused_source_sha256={n:s.native.sha256(s.native.REPO/n)for n in s.native.REUSED_NAMES},
            input_sha256=s.native.sha256(root/'inputs.safetensors'),weight_load_sha256=s.native.sha256(root/'profile-weight-load.json'),
            runtime_environment=s.native.runtime_identity(s.native.runtime_environment('cpu'),'cpu'))
    def test_actual_control_hashes_and_changed_context_rejection(self):
        report=json.loads((CONTROL/'metrics.json').read_text());values=self.inputs()
        result=s.validate_control(CONTROL,values,'mps',report['input_evidence'])
        self.assertTrue(result['all_input_tensors_exact']);self.assertIn('Failed prior visual inspection',result['visual_quality'])
        values['positive']=values['positive'].clone();values['positive'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'every exact original'):s.validate_control(CONTROL,values,'mps',report['input_evidence'])
    def test_nonlinear50_step_oracle_input_identity_and_original_factory(self):
        values=self.inputs();hashes={k:s.tensor_sha256(v)for k,v in values.items()};calls=[];outputs=[]
        factory=s.native.make_scheduler
        def field(x,t,pos):return x.square()*.001+torch.sin(x)*.004+int(t)*1e-6+(.003 if pos else -.002)
        def model(xs,t,contexts,seq):
            pos=len(calls)%2==0;calls.append(int(t[0,-1]))
            self.assertTrue(torch.equal(contexts[0],values['positive' if pos else 'negative']))
            self.assertTrue(torch.equal(xs[0][:,:1],values['observation'][0]));self.assertTrue((t[:,:144]==0).all())
            return [field(xs[0],t[0,-1],pos)]
        actual,rows=s.integrate(model,values['initial_noise'],values['observation'],values['positive'],values['negative'],on_step=lambda row,x:outputs.append(x.clone()))
        oracle=s.FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False);oracle.set_timesteps(50,device='cpu',shift=3.)
        expected=values['initial_noise'].clone()
        for i,t in enumerate(oracle.timesteps):
            expected[:,:1]=values['observation'][0];p,n=field(expected,t,True),field(expected,t,False)
            expected=oracle.step((n+5*(p-n))[None],t,expected[None],return_dict=False)[0][0];expected[:,:1]=values['observation'][0]
            self.assertTrue(torch.equal(outputs[i],expected))
        self.assertTrue(torch.equal(actual,expected));self.assertEqual((len(calls),len(rows)),(100,50))
        self.assertEqual(hashes,{k:s.tensor_sha256(v)for k,v in values.items()});self.assertIs(factory,s.native.make_scheduler)
        self.assertEqual(calls[:2],[999,999]);self.assertEqual(s.native.SETTINGS['shift'],5.)
    def test_worker_success_and_failed_second_step_artifacts(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);config=self.config(root);calls=[]
                def model(xs,*args):
                    calls.append(True)
                    if fail and len(calls)==3:raise RuntimeError('injected second step')
                    return [xs[0]*.002]
                loaded=json.loads((root/'profile-weight-load.json').read_text())
                with mock.patch.object(s.native,'Guard',FakeGuard),mock.patch.object(s.native,'load_core',return_value=(model,loaded)):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'injected second'):s.child(config)
                    else:s.child(config)
                report=json.loads((root/'result/metrics.json').read_text())
                self.assertEqual(report['status'],'failed' if fail else 'passed')
                self.assertEqual(report['ablation_source_sha256'],s.source_hashes(independent=True))
                self.assertTrue((root/'result/last-latent.safetensors').is_file())
                self.assertEqual((root/'result/latents.safetensors').exists(),not fail)
                if fail:self.assertEqual(len(report['steps']),1);self.assertNotIn('completed_core_calls',report)
                else:
                    self.assertEqual((len(calls),report['completed_steps'],report['completed_core_calls']),(100,50,100))
                    for n,h in report['output_sha256'].items():self.assertEqual(s.native.sha256(root/'result'/n),h)
    def test_interrupted_load_retains_failed_report_and_no_latent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.config(root)
            with mock.patch.object(s.native,'Guard',FakeGuard),mock.patch.object(s.native,'load_core',side_effect=KeyboardInterrupt('injected load interruption')):
                with self.assertRaises(KeyboardInterrupt):s.child(config)
            self.assertEqual(json.loads((root/'result/metrics.json').read_text())['status'],'failed')
            self.assertFalse((root/'result/latents.safetensors').exists())
            self.assertFalse((root/'result/last-latent.safetensors').exists())
    def test_stale_source_fails_before_model_or_decoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.config(Path(tmp));config['ablation_source_sha256']=dict(config['ablation_source_sha256'],**{'sample.py':'0'*64})
            with mock.patch.object(s.native,'load_core')as load,mock.patch.object(s.native,'child')as decode:
                for stage in ('core','decode'):
                    with self.assertRaisesRegex(ValueError,'isolated shift3'):s.child(dict(config,stage=stage))
            self.assertFalse(load.called);self.assertFalse(decode.called)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve existing report')
    before=s.source_hashes(independent=True);native={n:s.native.sha256(s.native.HERE/n)for n in s.native.SOURCES};reused={n:s.native.sha256(s.native.REPO/n)for n in s.native.REUSED_NAMES}
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    assert before==s.source_hashes(independent=True) and native=={n:s.native.sha256(s.native.HERE/n)for n in s.native.SOURCES} and reused=={n:s.native.sha256(s.native.REPO/n)for n in s.native.REUSED_NAMES}
    report=dict(status='passed' if result.wasSuccessful()else'failed',tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),seconds=time.monotonic()-start,
        source_sha256=before,native_source_sha256=native,reused_source_sha256=reused,source_unchanged_after_tests=True,device='cpu',gpu_used=False,real_model_constructed=False,official_weight_values_loaded=False,
        scope='Actual control hashes, nonlinear CPU solver oracle, stand-in worker success/failure and interruption. No model quality claim.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
