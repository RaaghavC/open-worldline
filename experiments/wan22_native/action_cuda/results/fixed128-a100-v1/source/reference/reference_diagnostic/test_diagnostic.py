"""Small injected-predictor checks; never load foundation weights or CUDA."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import torch
import diagnostic as d

class DiagnosticTests(unittest.TestCase):
    def fixture(self):
        shape=(1,2,5,4,4); initial=torch.full(shape,.7);times=torch.tensor([[0]*4+[999]*16])
        values={'initial_latent':initial[0],'token_times':times}
        contexts={'atrium':torch.ones(2,3),'native_negative':torch.full((2,3),2.)}
        commands={'closed':torch.zeros(1,16,6),'open':torch.zeros(1,16,6)};commands['open'][0,0,5]=1
        cases={}
        for arm,n in [('closed',1.),('open',2.)]:
            cases.update({arm+'_noisy':torch.full(shape,n),arm+'_times':torch.tensor([[0]*4+[506]*16]),arm+'_target':torch.full(shape,3.)})
        return cases,values,contexts,commands

    def test_all_calls_matched_inputs_order_and_effect(self):
        cases,values,contexts,commands=self.fixture();calls=[];saved={}
        def native(x,t,c):calls.append(('native',x.clone(),t.clone(),c.clone(),None));return x+c[0,0]
        def adapt(cp,x,t,c,a):
            calls.append((cp,x.clone(),t.clone(),c.clone(),a.clone()))
            if t[0,-1]==506:return torch.full_like(x,1. if cp=='0' else 2.)
            return x+c[0,0]+(a[0,0,5]*c[0,0]*.01 if cp=='16' else 0)
        report=d.evaluate(native,adapt,cases,values,contexts,commands,lambda n,v:saved.update({n:v.clone()}),lambda:None)
        self.assertEqual(report['predictions'],14);self.assertEqual(len(calls),14);self.assertEqual(len(report['calls']),14)
        self.assertEqual(len(saved),18);self.assertEqual([r[0] for r in calls],['native']*2+['0']*6+['16']*6)
        for row in calls:
            if row[2][0,-1]==999:self.assertTrue(torch.equal(row[1],values['initial_latent'][None]))
        for offset in (2,8):
            self.assertTrue(torch.equal(calls[offset][1],cases['closed_noisy']))
            self.assertTrue(torch.equal(calls[offset+1][1],cases['open_noisy']))
        self.assertEqual(report['objective']['0']['closed'],4.);self.assertEqual(report['objective']['16']['closed'],1.)
        self.assertEqual(report['relative_objective_improvement']['open'],.75)
        self.assertEqual(report['causal']['0']['guided_command_effect']['max_absolute'],0.)
        self.assertAlmostEqual(report['causal']['16']['guided_command_effect']['max_absolute'],.03,places=5)
        altered=values['initial_latent'][None].clone();altered[:,:,:1]+=100
        self.assertEqual(d.contrast(altered,values['initial_latent'][None])['rms'],0.)

    def test_zero_parity_failure_retains_prediction(self):
        cases,values,contexts,commands=self.fixture();saved={}
        with self.assertRaisesRegex(ValueError,'Zero checkpoint'):
            d.evaluate(lambda x,t,c:x,lambda cp,x,t,c,a:x+.1,cases,values,contexts,commands,lambda n,v:saved.update({n:v}),lambda:None)
        self.assertIn('causal-0-closed-atrium',saved)
        self.assertFalse(any('16-' in n for n in saved))

    def test_malformed_prediction_is_retained_then_rejected(self):
        cases,values,contexts,commands=self.fixture();saved={}
        with self.assertRaises(FloatingPointError):
            d.evaluate(lambda x,t,c:torch.full_like(x,float('nan')),lambda *a:None,cases,values,contexts,commands,lambda n,v:saved.update({n:v}),lambda:None)
        self.assertEqual(len(saved),1)

    def test_exact_admission_and_parent_failure_single_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary).resolve();plan={'source_sha256':{'a':'b'},'cpu_report_sha256':'c'}
            (out/'plan.json').write_text('{}')
            a={'schema':'worldline-action-fixed-input-diagnostic-admission-v1','decision':'admit','issued_by':'parent-agent','plan_sha256':d.vi.sha(out/'plan.json'),'source_sha256':plan['source_sha256'],'limits':d.limits('pair'),'training_admitted':False,'generation_admitted':False,'predictions':14,'cpu_report_sha256':'c'}
            decision=out/'decision.json';decision.write_text(json.dumps(a));d.admission(decision,out,plan)
            a['training_admitted']=True;decision.write_text(json.dumps(a))
            with self.assertRaises(ValueError):d.admission(decision,out,plan)
            proc=Mock();proc.returncode=-15
            with patch.object(d,'read_prepared',return_value=(plan,)),patch.object(d,'admission',return_value=d.vi.sha(decision)),patch.object(d.subprocess,'Popen',return_value=proc),patch.object(d,'supervise',side_effect=RuntimeError('mock timeout')),patch.object(d,'stop_child') as stop:
                with self.assertRaisesRegex(RuntimeError,'mock timeout'):d.execute(out,out,decision)
                stop.assert_called_once_with(proc)
                self.assertEqual(d.vi.read(out/'metrics.json')['status'],'failed')
                self.assertTrue((out/'terminal.json').is_file())
                with self.assertRaises(FileExistsError):d.execute(out,out,decision)

if __name__=='__main__':unittest.main()
