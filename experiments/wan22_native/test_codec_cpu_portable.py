# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixture: exact same-length causality, bounded cross-length rounding.

This separate test does not replace production exact_prefix, relax real cache
admission, or edit the retained strict fixture. It uses the identical tiny
native codec and seed, without external weights or GPU execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import unittest

import torch

from .codec import cleanup_temporal_chunks
from .codec_memory import cleanup_causal_convolutions
from .action_data.operations import observation_only, exact_prefix
from .action_data.test_codec_path import fixture
from .codec_cpu_diagnostic import comparison, state_sha

HERE=Path(__file__).resolve().parent
# Absolute units of the normalized FP32 fixture latent, with no relative term.
# Eight FP32 machine epsilons, declared only for this cross-length comparison.
CROSS_LENGTH_ATOL=8*torch.finfo(torch.float32).eps
SOURCES=('test_codec_cpu_portable.py','codec_cpu_diagnostic.py','codec.py','codec-source.json','codec_memory.py',
         'vendor/vae2_2.py','action_data/operations.py','action_data/test_codec_path.py')
MEASUREMENTS={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads();torch.set_num_threads(2)
        codec,video=fixture();before=state_sha(codec.model)
        with cleanup_temporal_chunks(codec,lambda:None),cleanup_causal_convolutions(codec,lambda:None):
            cls.first=observation_only(codec,video)
            cls.full=codec.encode(video)
            changed=video.clone();changed[:,:,1:]=-changed[:,:,1:]
            cls.perturbed=codec.encode(changed)
            cls.repeat=observation_only(codec,video)
        cls.state_unchanged=before==state_sha(codec.model)
        cls.cache_clear=codec.cache_is_clear()
        MEASUREMENTS.update(fixture_state_sha256=before,fixture_state_unchanged=cls.state_unchanged,
            cross_length=comparison(cls.full[:,:,:1],cls.first),
            same_length_future_perturbation=comparison(cls.full[:,:,:1],cls.perturbed[:,:,:1]),
            same_length_initial_repeat=comparison(cls.first,cls.repeat),
            declared_cross_length_atol=CROSS_LENGTH_ATOL,declared_cross_length_rtol=0.,
            tolerance_scope='Only full-clip versus standalone initial latent in this fixed tiny CPU fixture',
            num_threads=torch.get_num_threads(),mkldnn_available=torch.backends.mkldnn.is_available(),
            mkldnn_enabled=torch.backends.mkldnn.enabled)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.previous_threads)

    def test_same_length_future_perturbation_is_bit_exact(self):
        self.assertTrue(torch.equal(self.full[:,:,:1],self.perturbed[:,:,:1]))
        self.assertTrue(MEASUREMENTS['same_length_future_perturbation']['bytes_equal'])
        self.assertFalse(torch.equal(self.full[:,:,1:],self.perturbed[:,:,1:]))

    def test_same_length_initial_repeat_is_bit_exact(self):
        self.assertTrue(torch.equal(self.first,self.repeat))
        self.assertTrue(MEASUREMENTS['same_length_initial_repeat']['bytes_equal'])
        self.assertTrue(self.cache_clear);self.assertTrue(self.state_unchanged)

    def test_cross_length_comparison_has_only_declared_fp32_absolute_tolerance(self):
        self.assertEqual(self.first.shape,(1,48,1,2,2))
        self.assertEqual(self.first.dtype,torch.float32)
        self.assertTrue(torch.isfinite(self.first).all()and torch.isfinite(self.full).all())
        self.assertLessEqual(MEASUREMENTS['cross_length']['max_abs'],CROSS_LENGTH_ATOL)
        # No isclose/default-relative tolerance is used for any causality check.

    def test_production_prefix_check_still_rejects_even_small_difference(self):
        target=torch.zeros(1,48,5,2,2);observation=torch.zeros(1,48,1,2,2)
        target[0,0,0,0,0]=torch.finfo(torch.float32).eps
        report=exact_prefix(target,observation,name='production_strictness_fixture')
        self.assertFalse(report['passed']);self.assertFalse(report['target_prefix_replaced'])
        self.assertLess(report['max_abs'],CROSS_LENGTH_ATOL)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('New CPU evidence file required')
    a.output.parent.mkdir(parents=True,exist_ok=True);before={n:sha(HERE/n)for n in SOURCES};started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    unchanged=before=={n:sha(HERE/n)for n in SOURCES}
    report={'status':'passed'if result.wasSuccessful()and unchanged else'failed','tests':result.testsRun,
        'seconds':time.monotonic()-started,'failures':len(result.failures),'errors':len(result.errors),
        'source_sha256':before,'source_unchanged':unchanged,'torch_version':torch.__version__,
        'platform':platform.platform(),'measurements':MEASUREMENTS,'actual_gpu_operations':False,
        'external_weight_values_loaded':False,'production_exact_prefix_relaxed':False,
        'same_length_causality_and_repeat':'Exact tensor and byte equality required',
        'scope':'Platform CPU fixture only; no actual cache or production inference tolerance change'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
