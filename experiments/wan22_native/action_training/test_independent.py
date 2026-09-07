# SPDX-License-Identifier: Apache-2.0
"""Independent algebra and evidence tests. No foundation models or GPU calls."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import torch
from torch import nn

from . import objective as o, runtime as gate
from ..load_weights import sha256


class AlgebraAdapter(nn.Module):
    """Two scalars exercise the training helper; this is not a world model."""
    def __init__(self):
        super().__init__();self.command_gru=nn.Linear(1,1,bias=False);self.output=nn.Linear(1,1,bias=False)
        with torch.no_grad():self.command_gru.weight.fill_(.23);self.output.weight.fill_(.37)


class AlgebraWrapper(nn.Module):
    def __init__(self):
        super().__init__();self.adapter=AlgebraAdapter();self.core=nn.Linear(1,1,bias=False)
        self.core.requires_grad_(False);self.calls=[]
        with torch.no_grad():self.core.weight.fill_(.4)
    def forward(self,noisy,times,contexts,*,commands,observation,seq_len):
        self.calls.append({'noisy':noisy.detach().clone(),'times':times.clone(),'commands':commands.clone(),'observation':observation.clone()})
        a=self.adapter.command_gru.weight.reshape(());b=self.adapter.output.weight.reshape(())
        return noisy*self.core.weight.reshape(())+b*(a*(commands.sum()+noisy)+observation.mean())


def windows():
    first=torch.arange(12,dtype=torch.float32).reshape(1,1,3,2,2)/10
    second=first.clone();second[:,:,1:]+=.2
    commands=torch.zeros(1,8,6);commands[:,:,3]=.1
    opened=commands.clone();opened[:,0,5]=1.
    return [{'target':value,'observation':value[:,:,:1].clone(),'commands':command}
            for value,command in ((first,commands),(second,opened))]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_flow_tokens_loss_and_input_storage(self):
        window=windows()[0];noise=torch.linspace(-1,1,12).reshape(1,1,3,2,2);saved={k:v.clone()for k,v in window.items()};before=noise.clone()
        noisy,times,velocity=o.flow_inputs(window['target'],window['observation'],noise,250)
        self.assertTrue(torch.equal(noisy[:,:,1:],.75*window['target'][:,:,1:]+.25*noise[:,:,1:]))
        self.assertTrue(torch.equal(noisy[:,:,:1],window['observation']));self.assertEqual(times.tolist(),[[0,250,250]])
        self.assertTrue(torch.equal(velocity,noise-window['target']));self.assertTrue(torch.equal(noise,before))
        for k in window:self.assertTrue(torch.equal(window[k],saved[k]))
        prediction=velocity.clone();prediction[:,:,1:]+=.5;expected=.25
        self.assertEqual(float(o.future_flow_mse(prediction,velocity)),expected)
        prediction[:,:,:1]=1e20;self.assertEqual(float(o.future_flow_mse(prediction,velocity)),expected)

    def test_two_half_backwards_match_batch_mean_and_one_optimizer_step(self):
        actual=AlgebraWrapper();reference=copy.deepcopy(actual);data=windows();noise=torch.linspace(-.9,.9,12).reshape(1,1,3,2,2);context=torch.zeros(1,1)
        optimizer=torch.optim.SGD(actual.adapter.parameters(),lr=.02)
        reference_optimizer=torch.optim.SGD(reference.adapter.parameters(),lr=.02)
        losses=[]
        for window in data:
            x,t,v=o.flow_inputs(window['target'],window['observation'],noise,400)
            p=reference(x,t,[context],commands=window['commands'],observation=window['observation'],seq_len=t.shape[1])
            losses.append((p[:,:,1:]-v[:,:,1:]).square().flatten())
        batch_loss=torch.cat(losses).mean();batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.adapter.parameters(),1.,error_if_nonfinite=True)
        expected_gradients=[p.grad.clone()for p in reference.adapter.parameters()];reference_optimizer.step()
        with mock.patch.object(optimizer,'step',wraps=optimizer.step)as steps:
            record=o.paired_update(actual,data,noise,400,context,optimizer)
        self.assertEqual(steps.call_count,1);self.assertEqual(len(actual.calls),2)
        self.assertAlmostEqual(record['paired_mean_future_flow_mse'],float(batch_loss),delta=1e-7)
        for p,q,g in zip(actual.adapter.parameters(),reference.adapter.parameters(),expected_gradients):
            self.assertTrue(torch.allclose(p,q,atol=1e-7,rtol=1e-6));self.assertTrue(torch.allclose(p.grad,g,atol=1e-7,rtol=1e-6))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in actual.core.parameters()))
        self.assertTrue(torch.equal(actual.calls[0]['times'],actual.calls[1]['times']))
        self.assertTrue(torch.equal(actual.calls[0]['observation'],actual.calls[1]['observation']))

    def test_second_branch_failure_never_steps_optimizer(self):
        model=AlgebraWrapper();data=windows();noise=torch.ones_like(data[0]['target']);optimizer=torch.optim.SGD(model.adapter.parameters(),lr=.1)
        before=[p.clone()for p in model.adapter.parameters()];calls=0
        original=model.forward
        def fail(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('injected second branch failure')
            return original(*args,**kwargs)
        with mock.patch.object(model,'forward',side_effect=fail),mock.patch.object(optimizer,'step',wraps=optimizer.step)as step:
            with self.assertRaisesRegex(RuntimeError,'second branch'):o.paired_update(model,data,noise,300,torch.zeros(1,1),optimizer)
        self.assertEqual(step.call_count,0)
        self.assertTrue(all(torch.equal(p,q)for p,q in zip(model.adapter.parameters(),before)))

    def test_fresh_initialization_and_draws_ignore_global_rng_and_probe_mutation(self):
        configuration=dict(hidden_dim=8,observation_channels=1,width=4)
        torch.manual_seed(11);before=torch.get_rng_state().clone();first=o.fresh_adapter(test_configuration=configuration)
        self.assertTrue(torch.equal(torch.get_rng_state(),before));reference={n:p.clone()for n,p in first.named_parameters()}
        with torch.no_grad():
            for p in first.parameters():p.add_(3.)
        torch.manual_seed(98);second=o.fresh_adapter(test_configuration=configuration)
        self.assertTrue(all(torch.equal(p,reference[n])for n,p in second.named_parameters()))
        probe_rows,probe=o.make_draws('probe',shape=(1,1,3,2,2));torch.randn(100)
        full_rows,full=o.make_draws('fixed16',shape=(1,1,3,2,2))
        self.assertEqual(probe_rows,full_rows[:2]);self.assertEqual([r['start']for r in full_rows],[0,8,32,49]*4)
        for name,v in probe.items():self.assertTrue(torch.equal(v,full[name]))
        self.assertEqual(len(full_rows),16);self.assertTrue(all(50<=r['k']<=950 and r['sigma']==r['k']/1000. for r in full_rows))

    def test_frozen_parameter_records_detect_value_and_gradient_mutation(self):
        core=nn.Linear(2,2,bias=False).requires_grad_(False);before=o.parameter_records(core,expected_count=1)
        with torch.no_grad():core.weight[0,0]+=1
        with self.assertRaisesRegex(RuntimeError,'value changed'):o.parameter_records(core,expected_count=1,expected=before)
        core.weight.grad=torch.zeros_like(core.weight)
        with self.assertRaisesRegex(RuntimeError,'gradient'):o.parameter_records(core,expected_count=1)

    def test_poisoned_optimizer_does_not_advance_last_valid_checkpoint(self):
        adapter=AlgebraAdapter();optimizer=torch.optim.AdamW(adapter.parameters(),lr=.01)
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);rng=torch.Generator().manual_seed(1).get_state()
            o.save_checkpoint(root,adapter,optimizer,0,identity={'test':True},draw_rng_state=rng)
            previous=(root/'last-valid.json').read_bytes()
            first=next(adapter.parameters());optimizer.state[first]['exp_avg']=torch.full_like(first,float('inf'))
            with self.assertRaises((ValueError,FloatingPointError,RuntimeError)):
                o.save_checkpoint(root,adapter,optimizer,1,identity={'test':True},draw_rng_state=rng)
            self.assertEqual((root/'last-valid.json').read_bytes(),previous);self.assertFalse((root/'checkpoint-0001').exists())

    def test_visual_admission_required_and_tied_to_exact_mode_and_evidence(self):
        inputs={'cache_manifest_sha256':'c','roundtrip_metrics_sha256':'r'}
        with self.assertRaises(ValueError):gate.validate_admission(None,'probe',inputs)
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);evidence=root/'review.json';evidence.write_text('{}')
            record={'schema':'worldline-wan22-action-training-admission-v1','decision':'admit','issued_by':'parent-agent',
                'mode':'probe','scope':'optimizer-feasibility-only','foundation_visual_status':'failed',
                'source_sha256':{'fixture':'checked'},'cache_manifest_sha256':'c','roundtrip_metrics_sha256':'r',
                'visual_review':'Explicit fixture review','foundation_evidence':[{'file':'review.json','sha256':sha256(evidence)}]}
            path=root/'admission.json';path.write_text(json.dumps(record))
            with mock.patch.object(gate,'source_hashes',return_value={'fixture':'checked'}):
                self.assertFalse(gate.validate_admission(path,'probe',inputs)['quality_inferred_from_cpu_or_probe'])
                with self.assertRaises(ValueError):gate.validate_admission(path,'fixed16',inputs)
                rejected=dict(record,mode='fixed16',scope='fixed16-action-pilot');path.write_text(json.dumps(rejected))
                with self.assertRaises(ValueError):gate.validate_admission(path,'fixed16',inputs)
                path.write_text(json.dumps(record))
                evidence.write_text('{"changed":true}')
                with self.assertRaises(ValueError):gate.validate_admission(path,'probe',inputs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('New independent evidence file required')
    a.output.parent.mkdir(parents=True,exist_ok=True);before=gate.source_hashes();started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));after=gate.source_hashes()
    report={'status':'passed'if result.wasSuccessful()and before==after else'failed','tests':result.testsRun,
        'seconds':time.monotonic()-started,'errors':len(result.errors),'failures':len(result.failures),
        'source_sha256':before,'source_unchanged_after_tests':before==after,'independent_review':True,
        'independent_test_sha256':sha256(Path(__file__)),'foundation_models_loaded':0,'gpu_operations':0,
        'scope':'Algebra/CPU fixtures and admission checks; no actual foundation runtime, training study or visual-quality evidence'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
