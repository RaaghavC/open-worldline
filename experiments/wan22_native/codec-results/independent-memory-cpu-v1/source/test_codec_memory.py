# SPDX-License-Identifier: Apache-2.0
"""CPU equality and hook lifecycle for optional per-convolution cleanup."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import unittest
from unittest import mock

import torch

from .codec import Wan22Codec, make_model, HERE, sha, cleanup_temporal_chunks
from .codec_memory import cleanup_causal_convolutions
from .vendor.vae2_2 import CausalConv3d

SOURCES = ("codec.py", "codec-source.json", "vendor/vae2_2.py", "codec_memory.py", "test_codec_memory.py")
MEASUREMENTS = {}


def fixture():
    torch.manual_seed(752)
    codec = Wan22Codec.from_model(make_model(small=True, device="cpu"))
    latent = torch.randn(1,48,5,2,2)
    return codec, latent


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_all_17_frames_exact_and_native_layer_coverage(self):
        codec, latent = fixture(); before = latent.clone(); native = []
        handles = [m.register_forward_hook(lambda module, args, output, name=name: native.append(name))
            for name,m in codec.model.named_modules() if isinstance(m,CausalConv3d)]
        try: reference = codec.decode(latent)
        finally:
            for handle in handles: handle.remove()
        events = [];retained=[]
        with cleanup_temporal_chunks(codec,lambda:None) as chunks,cleanup_causal_convolutions(codec, lambda: events.append("cleanup"), retained.append) as report:
            actual = codec.decode(latent)
        self.assertEqual(actual.shape,(1,3,17,32,32)); self.assertTrue(torch.equal(actual,reference))
        self.assertEqual(chunks,{"encoder":0,"decoder":5})
        self.assertTrue(torch.equal(latent,before)); self.assertTrue(codec.cache_is_clear())
        actual_counts = {name:row["calls"] for name,row in report["layers"].items() if row["calls"]}
        self.assertEqual(actual_counts,dict(Counter(native)))
        self.assertEqual(report["registered_layers"],sum(isinstance(m,CausalConv3d) for m in codec.model.modules()))
        self.assertEqual(report["cleanup_calls"],len(native));self.assertEqual(report["completed_cleanups"],len(native))
        self.assertEqual(len(events),len(native));self.assertTrue(report["hooks_removed"])
        self.assertEqual([row["layer"] for row in report["events"]],native)
        self.assertEqual(retained,report["events"])
        self.assertTrue(all(row["cleanup_seconds"]>=0 for row in report["events"]))
        self.assertTrue(all(not m._forward_hooks for m in codec.model.modules()))
        MEASUREMENTS.update(registered_layers=report["registered_layers"],observed_native_calls=len(native),
            full_17_frame_max_abs=float((actual-reference).abs().max()),native_call_counts=actual_counts)

    def test_callback_error_removes_all_hooks_and_clears_caches(self):
        codec,latent=fixture(); calls=[]
        def failed():
            calls.append(1)
            if len(calls)==3: raise RuntimeError("injected cleanup failure")
        with self.assertRaisesRegex(RuntimeError,"injected"):
            with cleanup_causal_convolutions(codec,failed) as report: codec.decode(latent)
        self.assertEqual(report["cleanup_calls"],3);self.assertEqual(report["completed_cleanups"],2)
        self.assertEqual(report["events"][-1]["status"],"failed");self.assertTrue(report["hooks_removed"])
        self.assertTrue(codec.cache_is_clear());self.assertTrue(all(not m._forward_hooks for m in codec.model.modules()))
        self.assertTrue(torch.isfinite(codec.decode(latent)).all())

    def test_partial_registration_and_native_exception_leave_no_hooks(self):
        codec,latent=fixture()
        modules=[m for m in codec.model.modules() if isinstance(m,CausalConv3d)]
        with mock.patch.object(modules[2],"register_forward_hook",side_effect=RuntimeError("registration failure")):
            with self.assertRaises(RuntimeError):
                with cleanup_causal_convolutions(codec,lambda:None): pass
        self.assertTrue(all(not m._forward_hooks for m in modules));self.assertTrue(codec.cache_is_clear())
        with mock.patch.object(codec.model.decoder,"forward",side_effect=KeyboardInterrupt("native interruption")):
            with self.assertRaises(KeyboardInterrupt):
                with cleanup_causal_convolutions(codec,lambda:None) as report: codec.decode(latent)
        self.assertTrue(report["hooks_removed"]);self.assertTrue(all(not m._forward_hooks for m in modules))
        self.assertTrue(codec.cache_is_clear())

    def test_mps_cleanup_order_is_explicit_and_cpu_callback_avoids_mps(self):
        codec,latent=fixture()
        events=[]; codec.device=torch.device("mps")
        module=codec.model.conv2
        with mock.patch.object(torch.mps,"synchronize",side_effect=lambda:events.append("sync")),mock.patch.object(torch.mps,"empty_cache",side_effect=lambda:events.append("empty")):
            with cleanup_causal_convolutions(codec) as report: module(latent)
        self.assertEqual(events,["sync","empty"]);self.assertEqual(report["completed_cleanups"],1)
        codec.device=torch.device("cpu")
        with mock.patch.object(torch.mps,"synchronize",side_effect=AssertionError("No GPU")),mock.patch.object(torch.mps,"empty_cache",side_effect=AssertionError("No GPU")):
            with cleanup_causal_convolutions(codec,lambda:None):codec.decode(latent)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error("Evidence file must be new")
    args.output.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report={"status":"passed" if result.wasSuccessful() else "failed","tests":result.testsRun,
        "errors":len(result.errors),"failures":len(result.failures),"seconds":time.monotonic()-started,
        "source_sha256":{name:sha(HERE/name) for name in SOURCES},"measurements":MEASUREMENTS,
        "device":"cpu","actual_mps_operations":False,"external_weight_values_loaded":False,
        "scope":"Tiny native48-channel codec with all17 output frames; exact equation/hook equality, not actual memory savings or full-weight timing"}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=="__main__":main()
