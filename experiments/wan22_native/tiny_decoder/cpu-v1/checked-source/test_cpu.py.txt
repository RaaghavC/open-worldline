# SPDX-License-Identifier: Apache-2.0
"""Random small-spatial CPU fixtures; no official weight values or accelerator."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import torch
from . import codec,inputs,run
from .vendor import abot_taehv


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(673)
        cls.model=codec.make_model()
        with torch.no_grad():cls.model.decoder[-1].bias.fill_(.4)
        cls.decoder=codec.TinyDecoder.from_model(cls.model)
        cls.latent=torch.randn(1,48,6,1,2)

    @classmethod
    def tearDownClass(cls):
        cls.decoder.reset();torch.set_num_threads(cls.previous)

    def test_exact17_frames_matches_upstream_sequential_decode(self):
        value=self.latent[:,:,:5].clone();before=value.clone()
        with torch.inference_mode():expected=self.model.decode_video(value.permute(0,2,1,3,4),parallel=False,show_progress_bar=False)
        actual,rows=self.decoder.decode(value,chunks=(3,2))
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(actual.shape,(1,17,3,16,32))
        self.assertGreater(float(actual.std()),0.)
        self.assertEqual([r['output_frames'] for r in rows],[9,8]);self.assertTrue(torch.equal(value,before))
        self.assertTrue(self.decoder.cache_is_clear())
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),11_418_108)
        self.assertEqual(sum(p.numel() for p in self.model.decoder.parameters()),9_923_532)

    def test_published9_12_startup_and_legacy_chunk_wrapper_agree(self):
        old=abot_taehv.TAEHV(checkpoint_path=None,patch_size=2,latent_channels=48).float().eval().requires_grad_(False)
        old.load_state_dict(self.model.state_dict(),strict=True);stream=abot_taehv.StreamingTAEHV(old)
        with torch.inference_mode():
            first=stream.decode(self.latent[:,:,:3].permute(0,2,1,3,4));last=stream.decode(self.latent[:,:,3:].permute(0,2,1,3,4))
        actual,rows=self.decoder.decode(self.latent,chunks=(3,3))
        self.assertEqual((first.shape[1],last.shape[1]),(9,12))
        self.assertEqual([r['output_frames'] for r in rows],[9,12])
        self.assertTrue(torch.equal(actual,torch.cat([first,last],1)))

    def test_A_B_A_reset_isolation_and_all17_chunk_partitions(self):
        a=self.latent[:,:,:5];first,_=self.decoder.decode(a,chunks=(3,2))
        other,_=self.decoder.decode(-a,chunks=(2,3));self.assertFalse(torch.equal(first,other));again,_=self.decoder.decode(a,chunks=(1,1,1,1,1))
        self.assertTrue(torch.equal(first,again));self.assertTrue(self.decoder.cache_is_clear())
        self.assertEqual([x for x in self.decoder.stream.decoder_memory if x is not None],[])

    def test_direct_normalized_latents_no_inverse_scale_and_FP32_autocast(self):
        latent=self.latent[:,:,:5];expected,_=self.decoder.decode(latent)
        captured=[];dtypes=[];original=self.decoder.stream.decode
        def observe(value=None):
            if value is not None:captured.append(value.detach().cpu().clone())
            return original(value)
        handles=[m.register_forward_hook(lambda m,args,out:dtypes.append(out.dtype)) for m in self.model.decoder.modules() if isinstance(m,torch.nn.Conv2d)]
        try:
            with mock.patch.object(self.decoder.stream,'decode',side_effect=observe),torch.autocast('cpu',dtype=torch.bfloat16):
                actual,_=self.decoder.decode(latent)
        finally:
            for h in handles:h.remove()
        self.assertTrue(torch.equal(torch.cat(captured,1),latent.permute(0,2,1,3,4)))
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(set(dtypes),{torch.float32})
        self.assertFalse(self.decoder.provenance['full_vae_inverse_mean_std_applied'])

    def test_callback_failure_clears_stream_and_preserves_input(self):
        value=self.latent[:,:,:5].clone();before=value.clone()
        def fail(*unused):raise KeyboardInterrupt('fixture interruption')
        with self.assertRaises(KeyboardInterrupt):self.decoder.decode(value,on_chunk=fail)
        self.assertTrue(self.decoder.cache_is_clear());self.assertTrue(torch.equal(value,before))
        output,_=self.decoder.decode(value);self.assertTrue(torch.isfinite(output).all())

    def test_shape_chunks_dtype_and_weight_identity_rejections(self):
        for value,chunks in ((self.latent[:,:16,:5],(3,2)),(self.latent[:,:,:5].half(),(3,2)),(self.latent[:,:,:5],(3,3)),(self.latent[:,:,:5],(True,4))):
            with self.assertRaises(ValueError):self.decoder.decode(value,chunks=chunks)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.pth';path.write_bytes(b'not the prescribed weights')
            with mock.patch.object(torch,'load',side_effect=AssertionError('No unverified weight deserialization')):
                with self.assertRaises(ValueError):codec.TinyDecoder(path)

    def test_unexpected_latent_header_rejected_before_materialization(self):
        calls=[]
        class Poison:
            def __init__(self,*a,**k):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def keys(self):return ['latent','action']
            def get_tensor(self,key):calls.append(key);raise AssertionError('No action materialization')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input';path.write_bytes(b'fixture')
            with mock.patch.object(inputs,'safe_open',Poison):
                with self.assertRaises(ValueError):inputs.read_tensor(path,'latent',{'latent'},(1,48,5,18,32),torch.float32,2**21)
        self.assertEqual(calls,[])

    def test_stale_cpu_gate_and_original_guard_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tests.json';path.write_text(json.dumps({'status':'passed','tests':99,'source_sha256':{},'reused_source_sha256':{}}))
            with self.assertRaises(ValueError):run.check_cpu(path)
        self.assertIs(run.OneChildGuard.__init__,run.limits.CombinedGuard.__init__)
        self.assertIs(run.OneChildGuard.terminate,run.limits.CombinedGuard.terminate)
        self.assertEqual(run.limits.CAP_SECONDS,900.)
        self.assertEqual(run.ENVIRONMENT['PYTORCH_ENABLE_MPS_FALLBACK'],'0')

    def test_handoff_interruption_terminates_and_records_failure(self):
        class Process:
            returncode=None
            def poll(self):return self.returncode
            def communicate(self,data):raise BrokenPipeError('fixture handoff')
            def terminate(self):self.returncode=-15
            def wait(self,timeout=None):return self.returncode
        process=Process();guard=object.__new__(run.OneChildGuard);guard.process=None;guard.deadline=time.monotonic()+900
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'worker'
            with mock.patch.object(run.subprocess,'Popen',return_value=process):
                with self.assertRaises(BrokenPipeError):guard.launch({},out)
            result=json.loads((out/'terminal.json').read_text());self.assertEqual(result['status'],'failed');self.assertEqual(process.returncode,-15)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('Fresh CPU evidence required')
    before,reused=run.source_hashes(),run.reused_hashes();started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    assert before==run.source_hashes() and reused==run.reused_hashes()
    report={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'seconds':time.monotonic()-started,'source_sha256':before,'reused_source_sha256':reused,'source_unchanged_after_tests':True,
        'device':'cpu','official_weight_values_loaded':False,'gpu_used':False,
        'scope':'Random original TAEHV architecture at1x2 latent spatial size, streaming equations and failure boundaries. No real reconstruction or performance claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
