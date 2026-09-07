# SPDX-License-Identifier: Apache-2.0
"""Tiny native 48-channel CPU encode/decode and source-bound data preflight."""
import argparse
import json
from pathlib import Path
import time
import unittest
from unittest import mock

import torch

from ..codec import Wan22Codec, make_model, cleanup_temporal_chunks
from ..codec_memory import cleanup_causal_convolutions
from .operations import observation_only, reconstruct_target, exact_prefix
from .runtime import source_hashes
from .test_data import DataTests

MEASUREMENTS={}


def fixture():
    torch.manual_seed(762)
    return Wan22Codec.from_model(make_model(small=True,device='cpu')),torch.rand(1,3,17,32,32)*2-1


class CodecPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_full_17_encode_and_decode_cleanup_exact(self):
        codec,video=fixture();raw=video.clone();reference=codec.encode(video);decoded=codec.decode(reference)
        with cleanup_temporal_chunks(codec,lambda:None) as chunks,cleanup_causal_convolutions(codec,lambda:None) as layers:
            encoded=codec.encode(video);actual=codec.decode(encoded)
        self.assertTrue(torch.equal(encoded,reference));self.assertTrue(torch.equal(actual,decoded))
        self.assertEqual(encoded.shape,(1,48,5,2,2));self.assertEqual(actual.shape,(1,3,17,32,32))
        self.assertEqual(chunks,{'encoder':5,'decoder':5});self.assertTrue(layers['hooks_removed'])
        self.assertTrue(torch.equal(video,raw));self.assertTrue(codec.cache_is_clear())
        self.assertEqual(layers['cleanup_calls'],layers['completed_cleanups']);self.assertGreater(layers['cleanup_calls'],129)
        MEASUREMENTS.update(full_17_encode_cleanup_max_abs=float((encoded-reference).abs().max()),
            full_17_decode_cleanup_max_abs=float((actual-decoded).abs().max()),
            cleanup_registered_layers=layers['registered_layers'],encode_decode_cleanup_calls=layers['cleanup_calls'])

    def test_initial_encode_is_independent_and_future_perturbation_causal(self):
        codec,video=fixture()
        with cleanup_temporal_chunks(codec,lambda:None),cleanup_causal_convolutions(codec,lambda:None):
            first=observation_only(codec,video);full=codec.encode(video)
            changed=video.clone();changed[:,:,1:]=-changed[:,:,1:];altered=codec.encode(changed)
            repeated=observation_only(codec,video)
        self.assertTrue(exact_prefix(full,first,name='full')['passed'])
        self.assertTrue(exact_prefix(altered,first,name='future')['passed'])
        self.assertTrue(torch.equal(repeated,first));self.assertTrue(codec.cache_is_clear())
        MEASUREMENTS['first_latent_causal_max_abs']=float((first-full[:,:,:1]).abs().max())

    def test_encoder_failure_removes_both_cleanup_hooks(self):
        codec,video=fixture()
        with mock.patch.object(codec.model.encoder,'forward',side_effect=KeyboardInterrupt('injected encode error')):
            with self.assertRaises(KeyboardInterrupt):
                with cleanup_temporal_chunks(codec,lambda:None),cleanup_causal_convolutions(codec,lambda:None) as layers:
                    codec.encode(video)
        self.assertTrue(layers['hooks_removed']);self.assertTrue(codec.cache_is_clear())
        self.assertTrue(all(not module._forward_hooks for module in codec.model.modules()))
        self.assertTrue(torch.isfinite(codec.encode(video)).all())

    def test_decoder_only_receives_encoded_latent(self):
        codec=mock.Mock();latent=torch.zeros(1,48,5,2,2);codec.decode.return_value='decoded'
        self.assertEqual(reconstruct_target(codec,latent),'decoded');codec.decode.assert_called_once_with(latent)
        codec.encode.assert_not_called()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('New CPU evidence file required')
    args.output.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (DataTests,CodecPathTests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,'errors':len(result.errors),
        'failures':len(result.failures),'seconds':time.monotonic()-started,'source_sha256':source_hashes(),'measurements':MEASUREMENTS,
        'device':'cpu','external_weight_values_loaded':False,'actual_gpu_operations':False,
        'scope':'Original data rules plus tiny native 48-channel codec equations, all 17 frames; not actual-weight reconstruction or model quality'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
