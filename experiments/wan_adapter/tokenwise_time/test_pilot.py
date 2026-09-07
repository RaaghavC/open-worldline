# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixed-pilot tests. No external weights or GPU are loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from safetensors import safe_open
from tokenwise_time import pilot_common as pc
from tokenwise_time.sample_pilot import estimate_total,retained_noise
from tokenwise_time.test_training import fixture
from tokenwise_time.training import optimizer_update
from native_control.clamp_loop import integrate_clamped
from native_control.sampling import initial_noise


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_schedule_distinct_repeat_draws_and_diagnostics(self):
        windows={n:{'target':torch.zeros(1,16,3,4,6)} for n in pc.WINDOWS}
        values,records=pc.prepare_inputs(windows)
        repeat,other=pc.prepare_inputs(windows)
        self.assertEqual(len(records),24);self.assertEqual(records,other)
        self.assertEqual([r['window'] for r in records if r['kind']=='train'],list(pc.WINDOWS)*2)
        self.assertTrue(all(torch.equal(v,repeat[k]) for k,v in values.items()))
        self.assertFalse(torch.equal(values['train.00.noise'],values['train.08.noise']))
        self.assertFalse(torch.equal(values['train.00.noise'],values['diagnostic.00.noise']))
        with self.assertRaises(ValueError):pc.prepare_inputs(dict(reversed(list(windows.items()))))

    def test_diagnostics_keep_exact_inputs_after_optimizer(self):
        core,adapter,x,v,t,obs,actions,context=fixture()
        target=torch.randn_like(x);window={'target':target,'observation':obs,'actions':actions}
        values={'d.noise':torch.randn_like(x),'d.k':torch.tensor([550])}
        before,pred0=pc.diagnostic(core,adapter,window,values,'d',context[0])
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4)
        optimizer_update(core,adapter,optimizer,*pc.example(window,values,'d'),context)
        after,pred1=pc.diagnostic(core,adapter,window,values,'d',context[0])
        for key in ('noisy_latent_sha256','target_velocity_sha256','token_times_sha256'):
            self.assertEqual(before[key],after[key])
        self.assertFalse(torch.equal(pred0,pred1));self.assertFalse(adapter._hooks)

    def test_atomic_recovery_preserves_previous_completed_bundle(self):
        _,adapter,*_=fixture();optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4)
        with tempfile.TemporaryDirectory() as temp:
            digest=pc.save_recovery(temp,adapter,optimizer,0);path=Path(temp)/'recovery-last.pt'
            def fail(*args,**kwargs):raise KeyboardInterrupt('interrupted save')
            with mock.patch.object(torch,'save',side_effect=fail),self.assertRaises(KeyboardInterrupt):pc.save_recovery(temp,adapter,optimizer,1)
            self.assertEqual(pc.sha(path),digest)
            state=torch.load(path,weights_only=True);self.assertEqual(state['completed_updates'],0)
            self.assertEqual(state['schedule_completed'],[])
            with torch.no_grad():next(adapter.parameters()).fill_(float('nan'))
            with self.assertRaises(ValueError):pc.save_recovery(temp,adapter,optimizer,1)
            self.assertEqual(pc.sha(path),digest)

    def test_inference_cache_materializes_only_observation_and_actions(self):
        accessed=[]
        class Spy:
            def __init__(self,*args,**kwargs):self.inner=safe_open(*args,**kwargs)
            def __enter__(self):self.f=self.inner.__enter__();return self
            def __exit__(self,*args):return self.inner.__exit__(*args)
            def get_tensor(self,key):
                accessed.append(key)
                if key=='target':raise AssertionError('Future target forbidden')
                return self.f.get_tensor(key)
        with mock.patch('native_control.clamp_cache.safe_open',Spy),mock.patch.object(pc,'safe_open',Spy):
            observation,actions,provenance=pc.read_conditions(pc.PARENT/'data_cache')
        self.assertEqual(accessed,['observation','actions'])
        self.assertEqual(observation.shape,(16,1,36,64));self.assertEqual(actions.shape,(1,16,6))
        self.assertFalse(provenance['target_materialized'])

    def test_solver_token_times_cfg_conditions_and_caller_storage(self):
        noise=initial_noise();saved=noise.clone();obs=torch.randn(16,1,36,64);obs_saved=obs.clone()
        actions=torch.randn(1,16,6);action_saved=actions.clone();neg=torch.zeros(1,2);pos=torch.ones(2,2)
        adapter=type('Adapter',(),{'_hooks':[]})()
        calls=[];steps=[]
        def predicted(core,adapter,noisy,times,observed,commands,contexts):
            calls.append((times.clone(),float(contexts[0].mean())))
            self.assertTrue(torch.equal(noisy[:,:,:1],obs.unsqueeze(0)))
            self.assertTrue(torch.equal(observed,obs.unsqueeze(0)));self.assertTrue(torch.equal(commands,actions))
            return .03*noisy+.25+.1*contexts[0].mean()
        with mock.patch.object(pc,'predict',side_effect=predicted):
            latent,checks=pc.sample_latents(None,adapter,noise,obs,actions,neg,pos,device='cpu',callback=lambda i,t,x,row:steps.append((int(t),row)))
        class Oracle:
            def __call__(self,values,t,contexts,seq_len):return [.03*values[0]+.25+.1*contexts[0].mean()]
        expected=integrate_clamped(Oracle(),noise,obs,neg,pos,device='cpu')
        torch.testing.assert_close(latent,expected,atol=0,rtol=0)
        self.assertEqual(len(calls),100);self.assertEqual(len(steps),50)
        for i,(t,row) in enumerate(steps):
            self.assertEqual(row['clamped_prefix_max_abs_difference'],0.)
            for times,text in calls[2*i:2*i+2]:
                self.assertEqual(times.shape,(1,2880));self.assertEqual(times[:,:576].sum(),0)
                self.assertTrue((times[:,576:]==t).all())
            self.assertEqual([c[1] for c in calls[2*i:2*i+2]],[0.,1.])
        self.assertTrue(torch.equal(noise,saved));self.assertTrue(torch.equal(obs,obs_saved));self.assertTrue(torch.equal(actions,action_saved))
        self.assertEqual(checks['model_calls'],100)

    def test_noise_preservation_time_budget_and_output_isolation(self):
        noise,provenance=retained_noise(pc.PARENT/'native_control/results/clip50')
        self.assertTrue(torch.equal(noise,initial_noise()));self.assertTrue(provenance['noise_file_sha256'])
        self.assertEqual(estimate_total(40.,14.),786.)
        for bad in (float('nan'),float('inf'),0.,-1.):
            with self.assertRaises(ValueError):estimate_total(40.,bad)
        with self.assertRaises(RuntimeError):estimate_total(80.,20.)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):pc.reject_output(temp)
            pc.reject_output(Path(temp)/'new')
        with self.assertRaises(ValueError):pc.reject_output(pc.HERE/'never-create')

    def test_stale_preflight_and_interrupt_evidence(self):
        hashes={n:pc.sha(pc.PARENT/n) for n in pc.PILOT_NAMES+pc.SHARED_NAMES}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'review.json';path.write_text(json.dumps({'status':'passed','tests':7,'source_sha256':hashes}))
            pc.check_pilot_report(path)
            hashes[pc.PILOT_NAMES[0]]='changed';path.write_text(json.dumps({'status':'passed','tests':7,'source_sha256':hashes}))
            with self.assertRaises(ValueError):pc.check_pilot_report(path)
        from tokenwise_time import train_pilot,sample_pilot
        class FakeEvidence:
            last=None
            def __init__(self,directory):self.report={};FakeEvidence.last=self
            def close(self,error):self.error=error;self.report['status']='failed' if error else 'passed'
        for module in (train_pilot,sample_pilot):
            argv=['program']
            for option in ('weights','capture-cache','text-cache','output','cpu-report','independent-report'):
                argv+=['--'+option,'/tmp/unused-tokenwise-pilot-test']
            if module is sample_pilot:argv+=['--training-run','/tmp/unused','--arm','zero']
            with mock.patch.object(sys,'argv',argv),mock.patch.object(pc,'reject_output'),mock.patch.object(torch.backends.mps,'is_available',return_value=True),mock.patch.object(module,'Evidence',FakeEvidence),mock.patch.object(pc,'validate_preflight',side_effect=KeyboardInterrupt('test')),mock.patch.object(pc,'check_pilot_report',side_effect=KeyboardInterrupt('test')):
                with self.assertRaises(KeyboardInterrupt):module.main()
            self.assertEqual(FakeEvidence.last.report['status'],'failed')
            self.assertIsInstance(FakeEvidence.last.error,KeyboardInterrupt)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('CPU evidence output must be new')
    args.output.parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning);warnings.filterwarnings('ignore',message="User provided device_type of 'cuda'.*")
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    r={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'seconds':time.perf_counter()-started,'device':'cpu','external_weights_loaded':False,'torch':torch.__version__,
        'source_sha256':{n:pc.sha(pc.PARENT/n) for n in pc.PILOT_NAMES+pc.SHARED_NAMES}}
    args.output.write_text(json.dumps(r,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
