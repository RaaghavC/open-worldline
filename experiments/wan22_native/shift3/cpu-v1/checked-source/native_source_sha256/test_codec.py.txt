# SPDX-License-Identifier: Apache-2.0
"""Tiny independent CPU checks of the native Wan2.2 VAE wrapper and cache."""
import argparse
import ast
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
import numpy as np
from PIL import Image
from wan22_native.codec import Wan22Codec,make_model,scales,sha,HERE,SOURCE_SHA256,cleanup_temporal_chunks
from wan22_native.vendor.vae2_2 import patchify,unpatchify

NAMES=['codec.py','codec-source.json','codec_profile.py','test_codec.py','vendor/vae2_2.py']
MEASUREMENTS={}


def fixture():
    torch.manual_seed(823)
    return Wan22Codec.from_model(make_model(small=True,device='cpu'))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_opaque_rgba_preserves_every_rgb_byte_and_rejects_transparency(self):
        from wan22_native import codec_profile as profile
        rgb=(np.arange(288*512*3,dtype=np.uint32)%251).astype(np.uint8).reshape(288,512,3)
        rgba=np.concatenate((rgb,np.full((288,512,1),255,dtype=np.uint8)),axis=2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.png';Image.fromarray(rgba,'RGBA').save(path)
            with mock.patch.object(profile,'IMAGE_SHA256',sha(path)):
                pixels,video=profile.read_image(path)
            self.assertTrue(np.array_equal(pixels,rgb))
            expected=torch.from_numpy(rgb).permute(2,0,1)[None,:,None].float()/127.5-1
            self.assertTrue(torch.equal(video,expected));self.assertEqual(pixels.dtype,np.uint8)
            rgba[123,321,3]=254;Image.fromarray(rgba,'RGBA').save(path)
            with mock.patch.object(profile,'IMAGE_SHA256',sha(path)):
                with self.assertRaisesRegex(ValueError,'fully opaque'):profile.read_image(path)
            Image.fromarray(rgb,'RGB').save(path)
            with mock.patch.object(profile,'IMAGE_SHA256',sha(path)):
                plain,_=profile.read_image(path)
            self.assertTrue(np.array_equal(plain,rgb))

    def test_literal_source_and_48_native_scales(self):
        self.assertEqual(sha(HERE/'vendor/vae2_2.py'),SOURCE_SHA256)
        tree=ast.parse((HERE/'vendor/vae2_2.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Wan2_2_VAE')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        numbers={n.targets[0].id:ast.literal_eval(n.value.args[0]) for n in init.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('mean','std')}
        mean,inv=scales();self.assertEqual(mean.dtype,torch.float32)
        torch.testing.assert_close(mean,torch.tensor(numbers['mean']),atol=0,rtol=0)
        torch.testing.assert_close(inv,1/torch.tensor(numbers['std']),atol=0,rtol=0)

    def test_spatial_patch_order_manual_oracle(self):
        x=torch.arange(1*3*5*32*48).reshape(1,3,5,32,48).float()
        independent=x.reshape(1,3,5,16,2,24,2).permute(0,1,6,4,2,3,5).reshape(1,12,5,16,24)
        self.assertTrue(torch.equal(patchify(x,2),independent))
        self.assertTrue(torch.equal(unpatchify(independent,2),x))

    def test_official_equation_normalization_and_first_chunk(self):
        codec=fixture();raw=copy.deepcopy(codec.model);x=torch.rand(1,3,5,32,48)*2-1
        with torch.no_grad():
            mu=raw.encode(x,[0.,1.]);expected=(mu-codec.scale[0].view(1,48,1,1,1))*codec.scale[1].view(1,48,1,1,1)
            z=codec.encode(x);torch.testing.assert_close(z,expected,atol=0,rtol=0)
            restored=z/codec.scale[1].view(1,48,1,1,1)+codec.scale[0].view(1,48,1,1,1)
            expected_rgb=raw.decode(restored,[0.,1.]).float().clamp(-1,1)
            flags=[]
            handle=codec.model.decoder.register_forward_pre_hook(lambda m,a,k:flags.append(k.get('first_chunk',False)),with_kwargs=True)
            try:y=codec.decode(z)
            finally:handle.remove()
        self.assertEqual(z.shape,(1,48,2,2,3));self.assertEqual(y.shape,x.shape)
        self.assertEqual(flags,[True,False]);torch.testing.assert_close(y,expected_rgb,atol=0,rtol=0)
        self.assertTrue(codec.cache_is_clear())

    def test_repeated_encode_decode_and_caller_storage(self):
        codec=fixture();a=torch.rand(1,3,5,32,32)*2-1;b=torch.rand_like(a)*2-1;saved=a.clone()
        za=codec.encode(a);codec.encode(b);za2=codec.encode(a)
        self.assertTrue(torch.equal(za,za2));self.assertTrue(torch.equal(a,saved))
        ya=codec.decode(za);codec.decode(torch.randn_like(za));ya2=codec.decode(za)
        self.assertTrue(torch.equal(ya,ya2));self.assertTrue(codec.cache_is_clear())

    def test_initial_latent_causality(self):
        codec=fixture();a=torch.rand(1,3,5,32,32)*2-1
        alone=codec.encode(a[:,:,:1]);complete=codec.encode(a)
        changed=a.clone();changed[:,:,1:]=torch.rand_like(changed[:,:,1:])*2-1
        future_changed=codec.encode(changed)
        MEASUREMENTS['first_latent_alone_vs_full_max_abs']=float((alone-complete[:,:,:1]).abs().max())
        MEASUREMENTS['first_latent_after_future_change_max_abs']=float((complete[:,:,:1]-future_changed[:,:,:1]).abs().max())
        torch.testing.assert_close(alone,complete[:,:,:1],atol=2e-6,rtol=2e-6)
        self.assertTrue(torch.equal(complete[:,:,:1],future_changed[:,:,:1]))
        self.assertFalse(torch.equal(complete[:,:,1:],future_changed[:,:,1:]))

    def test_allocator_hooks_preserve_exact_output(self):
        codec=fixture();x=torch.rand(1,3,9,32,32)*2-1
        z=codec.encode(x);y=codec.decode(z);events=[]
        with cleanup_temporal_chunks(codec,lambda:events.append('cleanup')) as count:
            z2=codec.encode(x);y2=codec.decode(z2)
        self.assertEqual(count,{'encoder':3,'decoder':3});self.assertEqual(len(events),6)
        self.assertTrue(torch.equal(z,z2));self.assertTrue(torch.equal(y,y2))
        self.assertFalse(codec.model.encoder._forward_hooks);self.assertFalse(codec.model.decoder._forward_hooks)
        self.assertTrue(codec.cache_is_clear())

    def test_exceptions_clear_cache_and_remove_hooks(self):
        codec=fixture();x=torch.rand(1,3,5,32,32)*2-1;z=codec.encode(x)
        for operation,value in ((codec.encode,x),(codec.decode,z)):
            with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
                with cleanup_temporal_chunks(codec,lambda:(_ for _ in ()).throw(RuntimeError('cleanup failed'))):operation(value)
            self.assertTrue(codec.cache_is_clear())
            self.assertFalse(codec.model.encoder._forward_hooks);self.assertFalse(codec.model.decoder._forward_hooks)
        with mock.patch.object(codec.model,'encode',side_effect=TypeError('native failure')):
            with self.assertRaisesRegex(TypeError,'native failure'):codec.encode(x)
        self.assertTrue(codec.cache_is_clear());self.assertIsNotNone(codec.encode(x))

    def test_invalid_shapes_and_nan_rejected(self):
        codec=fixture()
        for x in (torch.zeros(1,3,4,32,32),torch.zeros(1,3,1,24,32),torch.full((1,3,1,32,32),float('nan'))):
            with self.assertRaises(ValueError):codec.encode(x)
        with self.assertRaises(ValueError):codec.decode(torch.zeros(1,16,1,2,2))
        with self.assertRaises(ValueError):codec.decode(torch.full((1,48,1,2,2),float('inf')))
        self.assertTrue(codec.cache_is_clear())

    def test_loader_uses_safe_mmap_and_profiler_rejects_stale_checks(self):
        from wan22_native import codec as module
        from wan22_native.codec_profile import validate_cpu
        state=make_model(small=True,device='cpu').state_dict()
        with mock.patch.object(module,'verify_weight_file'),mock.patch.object(module,'make_model',side_effect=lambda:make_model(small=True)),mock.patch.object(torch,'load',return_value=state) as load:
            loaded=Wan22Codec('/not-a-real-weight-file','cpu')
        self.assertEqual(load.call_args.kwargs,{'map_location':'cpu','weights_only':True,'mmap':True})
        self.assertEqual(loaded.provenance['loaded_keys'],len(state))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'report.json';hashes={name:sha(HERE/name) for name in NAMES}
            path.write_text(json.dumps({'status':'passed','tests':9,'source_sha256':hashes}));validate_cpu(path)
            hashes['codec.py']='changed';path.write_text(json.dumps({'status':'passed','tests':9,'source_sha256':hashes}))
            with self.assertRaises(ValueError):validate_cpu(path)

    def test_ambient_autocast_cannot_change_fp32_codec_computation(self):
        codec=fixture();x=torch.rand(1,3,5,32,32)*2-1
        latent=codec.encode(x);video=codec.decode(latent);dtypes=[]
        handles=[module.register_forward_hook(lambda m,a,o:dtypes.append(o.dtype)) for module in codec.model.modules() if isinstance(module,torch.nn.Conv3d)]
        try:
            with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
                under_context=codec.encode(x);decoded=codec.decode(under_context)
        finally:
            for handle in handles:handle.remove()
        self.assertTrue(dtypes and all(dtype==torch.float32 for dtype in dtypes))
        self.assertTrue(torch.equal(latent,under_context));self.assertTrue(torch.equal(video,decoded))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Evidence file must be new')
    args.output.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
        'seconds':time.perf_counter()-start,'device':'cpu','pretrained_weight_values_loaded':False,'torch':torch.__version__,
        'source_sha256':{name:sha(HERE/name) for name in NAMES},
        'measurements':MEASUREMENTS,
        'limits':'Tiny dim4/decoder4, native48 latent channels and native spatial/temporal stages; not actual-weight quality or MPS equality'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
