# SPDX-License-Identifier: Apache-2.0
"""Small CPU mechanics and gradient tests; no downloaded weights or GPU use."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from tokenwise_time.portable import create_model
from tokenwise_time.training import (ActionObservationAdapter,training_pair,future_loss,
    sample_training_inputs,conditioning,checkpoint_blocks,optimizer_update,predict)

CONFIG=dict(model_type='t2v',dim=256,ffn_dim=384,freq_dim=64,num_heads=2,num_layers=4,
            text_dim=32,text_len=8,in_dim=16,out_dim=16)


def fixture():
    torch.manual_seed(384)
    core=create_model(**CONFIG).eval().requires_grad_(False)
    with torch.no_grad():core.head.head.weight.normal_(std=.025)
    adapter=ActionObservationAdapter(256,block_indices=(0,1,2),width=16)
    target=torch.randn(1,16,3,4,6);noise=torch.randn_like(target);obs=torch.randn(1,16,1,4,6)
    noisy,velocity,times=training_pair(target,noise,obs,torch.tensor([750]))
    commands=torch.randn(1,8,6);text=[torch.randn(3,32)*.2]
    return core,adapter,noisy,velocity,times,obs,commands,text


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_noising_times_loss_and_future_target_separation(self):
        target=torch.randn(2,16,3,4,6);noise=torch.randn_like(target);obs=torch.randn(2,16,1,4,6)
        k=torch.tensor([50,950]);x,v,t=training_pair(target,noise,obs,k)
        sigma=k.float().view(2,1,1,1,1)/1000
        torch.testing.assert_close(x[:,:,1:],(1-sigma)*target[:,:,1:]+sigma*noise[:,:,1:],rtol=0,atol=0)
        self.assertTrue(torch.equal(x[:,:,:1],obs));self.assertEqual(t[:,:6].sum(),0)
        self.assertTrue(torch.equal(t[:,6:],k[:,None].expand(-1,12)))
        changed=target.clone();changed[:,:,:1]+=1000
        x2,v2,t2=training_pair(changed,noise,obs,k)
        self.assertTrue(torch.equal(x,x2) and torch.equal(v,v2) and torch.equal(t,t2))
        prediction=v.clone().requires_grad_();prediction.data[:,:,:1]=1234
        loss=future_loss(prediction,v);loss.backward()
        self.assertEqual(loss.item(),0);self.assertEqual(prediction.grad[:,:,:1].abs().sum().item(),0)

    def test_noise_rng_is_reproducible_and_integer_time_is_retained(self):
        target=torch.zeros(2,16,3,4,6)
        a=sample_training_inputs(target,torch.Generator().manual_seed(123))
        b=sample_training_inputs(target,torch.Generator().manual_seed(123))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
        self.assertEqual(a[0].dtype,torch.int64)
        self.assertTrue(((a[0]>=50)&(a[0]<=950)).all())

    def test_zero_init_and_two_updates_preserve_frozen_base(self):
        core,adapter,x,v,t,obs,act,ctx=fixture()
        saved={k:p.clone() for k,p in core.state_dict().items()}
        with torch.no_grad():
            original=torch.stack(core(list(x.unbind(0)),t,ctx,18))
            zero=predict(core,adapter,x,t,obs,act,ctx)
        self.assertTrue(torch.equal(original,zero))
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4)
        for _ in range(2):
            r=optimizer_update(core,adapter,optimizer,x,v,t,obs,act,ctx)
            self.assertTrue(all(n>0 for n in r['residual_gradient_norms']))
            self.assertGreater(r['block_forward_and_recompute_calls'][-1],1)
            self.assertFalse(adapter._hooks)
        self.assertTrue(all(torch.equal(p,saved[k]) for k,p in core.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in core.parameters()))

    def test_checkpoint_gradients_match_uncheckpointed_active_adapter(self):
        core,adapter,x,v,t,obs,act,ctx=fixture()
        with torch.no_grad():
            for residual in adapter.residuals:residual.output.weight.normal_(std=.01)
        def gradients(checkpointed):
            adapter.zero_grad(set_to_none=True)
            with conditioning(adapter,core,act,obs,(3,2,3)):
                if checkpointed:
                    with checkpoint_blocks(core):
                        pred=torch.stack(core(list(x.unbind(0)),t,ctx,18));loss=future_loss(pred,v);loss.backward()
                else:
                    pred=torch.stack(core(list(x.unbind(0)),t,ctx,18));loss=future_loss(pred,v);loss.backward()
            return loss.detach(),{n:p.grad.clone() for n,p in adapter.named_parameters()}
        loss1,g1=gradients(False);loss2,g2=gradients(True)
        torch.testing.assert_close(loss1,loss2,atol=0,rtol=0)
        for name in g1:torch.testing.assert_close(g1[name],g2[name],atol=2e-6,rtol=2e-5,msg=name)
        self.assertGreater(g2['residuals.0.query.weight'].abs().sum().item(),0)

    def test_error_removes_hooks_and_restores_methods(self):
        core,adapter,x,v,t,obs,act,ctx=fixture();old=[b.forward for b in core.blocks]
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with conditioning(adapter,core,act,obs,(3,2,3)),checkpoint_blocks(core):raise RuntimeError('deliberate')
        self.assertFalse(adapter._hooks)
        self.assertTrue(all(b.forward==f for b,f in zip(core.blocks,old)))
        bad=ActionObservationAdapter(256,block_indices=(0,999),width=16)
        with self.assertRaises(IndexError):
            with conditioning(bad,core,act,obs,(3,2,3)):pass
        self.assertFalse(bad._hooks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Evidence output must be new')
    a.output.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning);warnings.filterwarnings('ignore',message="User provided device_type of 'cuda'.*")
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    here=Path(__file__).parent
    names=['tokenwise_time/'+n for n in ['test_training.py','training.py','portable.py']]+['adapter.py','native_control/portable.py','native_control/vendor/model.py']
    report={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'seconds':time.perf_counter()-start,'device':'cpu','pretrained_weights_loaded':False,'torch':torch.__version__,
            'source_sha256':{n:hashlib.sha256((here.parent/n).read_bytes()).hexdigest() for n in names},
            'scope':'Noising/loss, checkpoint gradients, fresh adapter mechanics and hook cleanup; no video or generalization result'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
