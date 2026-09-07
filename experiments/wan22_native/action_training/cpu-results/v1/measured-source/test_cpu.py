# SPDX-License-Identifier: Apache-2.0
"""Small CPU fixtures only. No external parameter values or GPU execution."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..portable import create_model
from ..action_adapter.wrapper import NativeActionWrapper
from . import objective as o, runtime as r, train as t

SMALL=dict(model_type='ti2v',in_dim=4,out_dim=4,dim=32,ffn_dim=64,freq_dim=8,text_dim=16,text_len=8,num_heads=4,num_layers=2)
SMALL_ADAPTER=dict(hidden_dim=32,observation_channels=4,width=8)


def fixture(policy='fp32'):
    torch.manual_seed(11)
    core=create_model(device='cpu',policy=policy,configuration=SMALL)
    torch.nn.init.normal_(core.head.head.weight,std=.12)
    adapter=o.fresh_adapter(test_configuration=SMALL_ADAPTER)
    wrapper=NativeActionWrapper(core,adapter,test_only=True)
    target=torch.randn(1,4,3,4,4);other=target.clone();other[:,:,1:]+=.3
    commands=torch.zeros(1,8,6);commands[0,0,5]=1
    windows=[dict(target=target,observation=target[:,:,:1].clone(),commands=commands*0),
             dict(target=other,observation=target[:,:,:1].clone(),commands=commands)]
    return wrapper,windows,torch.randn_like(target),torch.randn(3,16)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_exact_rng_draw_order_and_probe_prefix_of_fixed16(self):
        rows,values=o.make_draws('fixed16',shape=(1,4,3,4,4));probe,pv=o.make_draws('probe',shape=(1,4,3,4,4))
        rng=torch.Generator().manual_seed(o.SEED)
        for i,row in enumerate(rows):
            k=int(torch.randint(50,951,(1,),generator=rng));noise=torch.randn((1,4,3,4,4),generator=rng)
            self.assertEqual(row['k'],k);self.assertTrue(torch.equal(values[row['noise_key']],noise))
            self.assertEqual(row['noise_sha256'],o.tensor_sha256(noise));self.assertEqual(row['start'],o.STARTS[i%4])
            self.assertEqual(row['branches'],[f'closed-{row["start"]:04d}',f'open-{row["start"]:04d}'])
        self.assertEqual(probe,rows[:2]);self.assertTrue(all(torch.equal(v,values[k])for k,v in pv.items()))

    def test_fresh_initializer_has_exact_count_and_ignores_global_rng(self):
        torch.manual_seed(0);before=torch.get_rng_state().clone();a=o.fresh_adapter();after=torch.get_rng_state()
        torch.randn(17);b=o.fresh_adapter()
        self.assertTrue(torch.equal(before,after));self.assertEqual(sum(p.numel()for p in a.parameters()),947712)
        self.assertTrue(all(torch.equal(x,y)for x,y in zip(a.parameters(),b.parameters())))
        self.assertFalse(torch.count_nonzero(a.output.weight));self.assertFalse(torch.count_nonzero(a.output.bias))

    def test_flow_equations_clean_prefix_times_and_input_immutability(self):
        _,windows,noise,_=fixture();w=windows[0];before=[x.clone()for x in (w['target'],w['observation'],noise)]
        noisy,times,target=o.flow_inputs(w['target'],w['observation'],noise,321)
        self.assertTrue(torch.equal(noisy[:,:,1:],.679*w['target'][:,:,1:]+.321*noise[:,:,1:]))
        self.assertTrue(torch.equal(noisy[:,:,:1],w['observation']));self.assertTrue(torch.equal(target,noise-w['target']))
        self.assertTrue((times[:,:4]==0).all());self.assertTrue((times[:,4:]==321).all())
        self.assertTrue(all(torch.equal(x,y)for x,y in zip(before,(w['target'],w['observation'],noise))))

    def test_future_loss_excludes_prefix_and_has_exact_gradient(self):
        pred=torch.arange(12,dtype=torch.float32).reshape(1,1,3,2,2).requires_grad_();target=torch.zeros_like(pred)
        loss=o.future_flow_mse(pred,target);expected=pred[:,:,1:].square().sum()/8
        self.assertEqual(float(loss),float(expected));loss.backward()
        self.assertTrue((pred.grad[:,:,:1]==0).all());self.assertTrue(torch.equal(pred.grad[:,:,1:],2*pred.detach()[:,:,1:]/8))

    def test_live_sequential_half_loss_matches_independent_accumulation(self):
        for policy in ('fp32','selective_bf16'):
            with self.subTest(policy=policy):
                w,windows,noise,text=fixture(policy);ref=copy.deepcopy(w)
                opt=torch.optim.AdamW(w.adapter.parameters(),**o.OPTIMIZER);ropt=torch.optim.AdamW(ref.adapter.parameters(),**o.OPTIMIZER)
                called=[];original=w.forward
                def spy(*args,**kwargs):called.append(kwargs['commands'].clone());return original(*args,**kwargs)
                w.forward=spy
                record=o.paired_update(w,windows,noise,400,text,opt)
                losses=[]
                for window in windows:
                    x,times,v=o.flow_inputs(window['target'],window['observation'],noise,400)
                    pred=ref(x,times,[text],commands=window['commands'],observation=window['observation'],seq_len=12)
                    loss=(pred[:,:,1:]-v[:,:,1:]).square().mean();(loss*.5).backward();losses.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(ref.adapter.parameters(),1.,error_if_nonfinite=True);ropt.step()
                self.assertTrue(all(torch.equal(a,b)for a,b in zip(w.adapter.parameters(),ref.adapter.parameters())))
                self.assertEqual(record['paired_mean_future_flow_mse'],sum(losses)/2)
                self.assertTrue(all(torch.equal(a,b['commands'])for a,b in zip(called,windows)))
                self.assertEqual(record['live_sequential_forwards'],2)

    def test_second_paired_update_reaches_gru_and_every_core_hash_is_unchanged(self):
        w,windows,noise,text=fixture();count=len(list(w.core.parameters()))
        before=o.parameter_records(w.core,expected_count=count);opt=torch.optim.AdamW(w.adapter.parameters(),**o.OPTIMIZER)
        first=o.paired_update(w,windows,noise,700,text,opt);second=o.paired_update(w,windows,noise,650,text,opt)
        self.assertEqual(first['command_gru_gradient_l2'],0.);self.assertGreater(first['output_gradient_l2'],0.)
        self.assertGreater(second['command_gru_gradient_l2'],0.)
        self.assertEqual(before,o.parameter_records(w.core,expected_count=count,expected=before))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in w.core.parameters()))

    def test_frozen_hash_validation_detects_value_gradient_and_coverage_changes(self):
        w,_,_,_=fixture();count=len(list(w.core.parameters()));before=o.parameter_records(w.core,expected_count=count)
        with self.assertRaises(ValueError):o.parameter_records(w.core,expected_count=count+1)
        p=next(w.core.parameters());p.requires_grad_(True)
        with self.assertRaises(RuntimeError):o.parameter_records(w.core,expected_count=count)
        p.requires_grad_(False)
        with torch.no_grad():p.flatten()[0]+=1
        with self.assertRaisesRegex(RuntimeError,'value changed'):o.parameter_records(w.core,expected_count=count,expected=before)

    def test_no_teacher_target_argument_reaches_the_wrapper(self):
        w,windows,noise,text=fixture();seen=[];base=w.forward
        def spy(*args,**kwargs):
            self.assertEqual(set(kwargs),{'commands','observation','seq_len'});self.assertEqual(len(args),3)
            self.assertFalse(torch.equal(args[0],windows[len(seen)]['target']));seen.append(True)
            return base(*args,**kwargs)
        w.forward=spy;o.paired_update(w,windows,noise,500,text,torch.optim.AdamW(w.adapter.parameters(),**o.OPTIMIZER))
        self.assertEqual(len(seen),2)

    def test_last_valid_checkpoint_survives_nonfinite_optimizer_and_overwrite(self):
        w,windows,noise,text=fixture();opt=torch.optim.AdamW(w.adapter.parameters(),**o.OPTIMIZER)
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);rng=torch.Generator().manual_seed(o.SEED).get_state()
            initial=o.save_checkpoint(root,w.adapter,opt,0,identity={'mode':'probe'},draw_rng_state=rng)
            o.paired_update(w,windows,noise,600,text,opt)
            valid=o.save_checkpoint(root,w.adapter,opt,1,identity={'mode':'probe'},draw_rng_state=rng)
            before=(root/'last-valid.json').read_bytes()
            with self.assertRaises(ValueError):o.save_checkpoint(root,w.adapter,opt,1,identity={},draw_rng_state=rng)
            next(iter(opt.state.values()))['exp_avg'].flatten()[0]=float('nan')
            with self.assertRaises(FloatingPointError):o.save_checkpoint(root,w.adapter,opt,2,identity={},draw_rng_state=rng)
            self.assertEqual(before,(root/'last-valid.json').read_bytes());self.assertFalse((root/'checkpoint-0002').exists())
            restored=torch.load(root/valid['directory']/'optimizer-and-rng.pt',weights_only=True,map_location='cpu')
            self.assertEqual(restored['completed_updates'],1);o.require_finite_tree(restored)
            self.assertFalse(initial['external_core_weights_included']);self.assertFalse(valid['external_core_weights_included'])

    def test_invalid_inputs_and_optimizer_ownership_are_rejected(self):
        w,windows,noise,text=fixture()
        for k in (True,49,951,float('nan')):
            with self.assertRaises(ValueError):o.flow_inputs(windows[0]['target'],windows[0]['observation'],noise,k)
        wrong=windows[0]['observation'].clone()+1
        with self.assertRaises(ValueError):o.flow_inputs(windows[0]['target'],wrong,noise,500)
        poisoned=noise.clone();poisoned.flatten()[0]=float('inf')
        with self.assertRaises(ValueError):o.flow_inputs(windows[0]['target'],windows[0]['observation'],poisoned,500)
        with self.assertRaises(ValueError):o.paired_update(w,windows,noise,500,text,torch.optim.AdamW([next(w.adapter.parameters())],lr=.001))

    def test_visual_admission_is_separate_mode_bound_and_hash_bound(self):
        inputs={'cache_manifest_sha256':'c','roundtrip_metrics_sha256':'r'}
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);visual=root/'visual.txt';visual.write_text('A parent-reviewed fixture only')
            decision={'schema':'worldline-wan22-action-training-admission-v1','decision':'admit','issued_by':'parent-agent','mode':'probe',
                'source_sha256':r.source_hashes(),'cache_manifest_sha256':'c','roundtrip_metrics_sha256':'r','visual_review':'fixture, not a real admission',
                'foundation_evidence':[{'file':'visual.txt','sha256':o.sha256(visual)}]}
            path=root/'decision.json';path.write_text(json.dumps(decision))
            self.assertFalse(r.validate_admission(path,'probe',inputs)['quality_inferred_from_cpu_or_probe'])
            with self.assertRaises(ValueError):r.validate_admission(None,'probe',inputs)
            with self.assertRaises(ValueError):r.validate_admission(path,'fixed16',inputs)
            visual.write_text('changed')
            with self.assertRaises(ValueError):r.validate_admission(path,'probe',inputs)

    def test_saved_draw_identity_and_source_report_gate(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);rows,draws=o.make_draws('probe');save_file(draws,str(root/'draws.safetensors'))
            actual,_=t.verify_saved_draws(root,'probe',rows);self.assertEqual(actual,rows)
            draws['noise_0000']=draws['noise_0000']+1;save_file(draws,str(root/'draws.safetensors'))
            with self.assertRaises(ValueError):t.verify_saved_draws(root,'probe',rows)
            report=root/'cpu.json';report.write_text(json.dumps({'status':'passed','tests':100,'source_sha256':{},'source_unchanged_after_tests':True}))
            with self.assertRaises(ValueError):r.validate_cpu(report)
            with self.assertRaises(ValueError):o.make_draws('resume')

    def test_actual_cache_roundtrip_and_original_data_pass_without_visual_admission(self):
        capture=r.REPO.parents[1]/'work/atrium-pilot/dense-pair'
        cache=r.HERE.parent/'action_data/results/cache-v1/result'
        roundtrip=r.HERE.parent/'action_data/results/roundtrip-v1'
        text=r.REPO/'experiments/wan_adapter/text_cache/native-results'
        evidence,positive=r.validate_inputs(capture,cache,roundtrip,text)
        self.assertEqual(len(evidence['windows']),8);self.assertEqual(positive.shape[1],4096)
        self.assertEqual(evidence['cache_manifest_sha256'],'450d2c5d07881a0f42bb027bba666105921c50ad7c9259bf914883eaeeb3e39e')
        with self.assertRaises(ValueError):r.validate_admission(None,'probe',evidence)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve existing test evidence')
    before=r.source_hashes();start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    assert before==r.source_hashes()
    report=dict(schema='worldline-wan22-action-training-cpu-v1',status='passed' if result.wasSuccessful()else'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-start,source_sha256=before,source_unchanged_after_tests=True,
        device='cpu',gpu_execution=False,external_weights_loaded=False,actual_training_executed=False,
        scope='Tiny randomly initialized native-shaped fixtures and actual read-only cache/provenance checks. No visual admission or learned quality claim.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
