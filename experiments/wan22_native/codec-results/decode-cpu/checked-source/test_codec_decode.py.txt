# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the bounded full-shape synthetic decode profile."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from wan22_native import codec_decode_profile as d
from wan22_native.codec import Wan22Codec,make_model,cleanup_temporal_chunks,HERE,sha,CONFIG,WEIGHT_SHA256,SOURCE_SHA256


def make_observation_fixture(path):
    observation=torch.full((1,48,1,18,32),.125)
    save_file({'observation':observation},str(path/'observation.safetensors'))
    r={'status':'passed','finite_output':True,'input_image_sha256':d.IMAGE_SHA256,'input_rgb_frames':1,'future_rgb_read':False,
       'expected_latent_shape':[1,48,1,18,32],'dtype':'float32','cache_clear_after_encode':True,'cache_clear_after_decode':True,
       'codec':{'weight_sha256':WEIGHT_SHA256,'source_sha256':SOURCE_SHA256,'config':CONFIG},
       'source_sha256':{name:sha(HERE/name) for name in d.INITIAL_SOURCES},
       'output_sha256':{'observation.safetensors':sha(path/'observation.safetensors')},'latent_tensor_sha256':d.tensor_sha(observation)}
    (path/'metrics.json').write_text(json.dumps(r))
    return observation,r


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_saved_noise_is_canonical_without_rng_regeneration(self):
        with mock.patch.object(torch,'randn',side_effect=AssertionError('Seed regeneration is not the identity check')):
            noise=d.canonical_noise()
        self.assertEqual(noise.shape,d.SHAPE);self.assertEqual(d.tensor_sha(noise),d.NOISE_TENSOR_SHA256)
        self.assertEqual(sha(HERE/'codec-inputs/synthetic-noise.safetensors'),d.NOISE_SHA256)

    def test_synthetic_prefix_suffix_and_caller_storage(self):
        noise=d.canonical_noise();saved=noise.clone();obs=torch.zeros(1,48,1,18,32)
        result=d.synthetic_input(noise,obs)
        self.assertTrue(torch.equal(result[:,:,:1],obs));self.assertTrue(torch.equal(result[:,:,1:],noise[:,:,1:]))
        self.assertTrue(torch.equal(saved,noise));self.assertFalse(result.data_ptr()==noise.data_ptr())
        result.fill_(1);self.assertTrue(torch.equal(saved,noise));self.assertEqual(obs.sum().item(),0)
        with self.assertRaises(ValueError):d.synthetic_input(noise,torch.zeros(1,16,1,36,64))

    def test_only_observation_materialized(self):
        accessed=[]
        class Spy:
            def __init__(self,*a,**k):self.inner=safe_open(*a,**k)
            def __enter__(self):self.f=self.inner.__enter__();return self
            def __exit__(self,*a):return self.inner.__exit__(*a)
            def keys(self):return self.f.keys()
            def get_tensor(self,key):
                accessed.append(key)
                if key!='observation':raise AssertionError('Future/action input forbidden')
                return self.f.get_tensor(key)
        with tempfile.TemporaryDirectory() as temp:
            expected,_=make_observation_fixture(Path(temp))
            with mock.patch.object(d,'safe_open',Spy):actual,provenance=d.read_observation(temp)
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(accessed,['observation'])
        self.assertFalse(provenance['rgb_files_read']);self.assertFalse(provenance['actions_read']);self.assertFalse(provenance['future_targets_read'])

    def test_failed_stale_or_changed_first_image_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);_,record=make_observation_fixture(path)
            for field,value in [('status','failed'),('finite_output',False),('dtype','float16'),('future_rgb_read',True),('input_image_sha256','other')]:
                bad=copy.deepcopy(record);bad[field]=value;(path/'metrics.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):d.read_observation(path)
            bad=copy.deepcopy(record);bad['source_sha256']['codec.py']='changed';(path/'metrics.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):d.read_observation(path)
            (path/'metrics.json').write_text(json.dumps(record));(path/'observation.safetensors').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):d.read_observation(path)

    def test_all_five_decoder_chunks_keep_native_output(self):
        torch.manual_seed(664);codec=Wan22Codec.from_model(make_model(small=True,device='cpu'))
        latent=torch.randn(1,48,5,2,2);plain=codec.decode(latent)
        with mock.patch.object(codec.model.encoder,'forward',side_effect=AssertionError('Encoder forbidden')):
            with d.chunk_timings(codec,'cpu') as chunks,cleanup_temporal_chunks(codec) as counts:measured=codec.decode(latent)
        self.assertTrue(torch.equal(plain,measured));self.assertEqual(measured.shape,(1,3,17,32,32))
        self.assertEqual([r['first_chunk'] for r in chunks],[True,False,False,False,False])
        self.assertEqual([r['output_shape'][2] for r in chunks],[1,4,4,4,4])
        self.assertTrue(all(r['seconds']>0 for r in chunks));self.assertEqual(counts,{'encoder':0,'decoder':5})
        self.assertTrue(codec.cache_is_clear());self.assertFalse(codec.model.decoder._forward_hooks);self.assertFalse(codec.model.decoder._forward_pre_hooks)

    def test_timing_hooks_removed_after_partial_registration_or_error(self):
        codec=Wan22Codec.from_model(make_model(small=True,device='cpu'))
        with mock.patch.object(codec.model.decoder,'register_forward_hook',side_effect=RuntimeError('hook failure')):
            with self.assertRaises(RuntimeError):
                with d.chunk_timings(codec,'cpu'):pass
        self.assertFalse(codec.model.decoder._forward_pre_hooks)
        with self.assertRaises(KeyboardInterrupt):
            with d.chunk_timings(codec,'cpu'):raise KeyboardInterrupt('test')
        self.assertFalse(codec.model.decoder._forward_hooks);self.assertFalse(codec.model.decoder._forward_pre_hooks)

    def test_profile_rejects_stale_cpu_report_and_records_interrupt(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cpu.json';report={'status':'passed','tests':7,'canonical_noise_sha256':d.NOISE_SHA256,
                'source_sha256':{name:sha(HERE/name) for name in d.SOURCE_NAMES}}
            path.write_text(json.dumps(report));d.validate_decode_cpu(path)
            report['source_sha256']['codec_decode_profile.py']='other';path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):d.validate_decode_cpu(path)
        class FakeGuard:
            last=None
            def __init__(self,*a,**k):self.report={};FakeGuard.last=self
            def close(self,error):self.error=error;self.report['status']='failed' if error else 'passed'
        args=['test','--weights','unused','--observation-run','unused','--cpu-report','unused','--output','unused']
        with mock.patch.object(sys,'argv',args),mock.patch.object(d,'Guard',FakeGuard),mock.patch.object(d,'validate_cpu',side_effect=KeyboardInterrupt('test')):
            with self.assertRaises(KeyboardInterrupt):d.main()
        self.assertEqual(FakeGuard.last.report['status'],'failed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('CPU report must be new')
    args.output.parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    r={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
       'seconds':time.perf_counter()-started,'device':'cpu','external_weight_values_loaded':False,
       'canonical_noise_sha256':d.NOISE_SHA256,'source_sha256':{name:sha(HERE/name) for name in d.SOURCE_NAMES},
       'scope':'Canonical synthetic inputs, one-observation access and native five-chunk decode timing hooks on tiny CPU model; no full-shape pretrained timing or quality claim'}
    args.output.write_text(json.dumps(r,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
