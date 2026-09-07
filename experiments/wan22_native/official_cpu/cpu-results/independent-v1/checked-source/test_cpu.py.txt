# SPDX-License-Identifier: Apache-2.0
"""Small CPU mechanics tests. No external weight values or GPU use."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file
from . import reference as r
from . import streaming as s
from . import run
from .inputs import read_exact,load_inputs

SMALL=dict(model_type='ti2v',in_dim=4,out_dim=4,dim=32,ffn_dim=64,freq_dim=8,
           text_dim=16,text_len=8,num_heads=4,num_layers=2)


def fixture():
    model,_=r.create_meta(SMALL);g=torch.Generator().manual_seed(763)
    state={n:torch.randn(p.shape,generator=g,dtype=torch.float32)*.1 for n,p in model.named_parameters()}
    inputs=(torch.randn((4,3,4,4),generator=g),torch.tensor([[0]*4+[999]*8]),torch.randn((3,16),generator=g))
    return model,state,inputs


class DictSource:
    def __init__(self,state):self.state=state;self.calls=[]
    def load(self,name,shape):
        self.calls.append(name);value=self.state[name]
        assert tuple(shape)==tuple(value.shape)
        return value.clone()


def resident(state):
    model,_=r.create_meta(SMALL)
    for name,value in state.items():
        parent,key=name.rsplit('.',1);setattr(model.get_submodule(parent),key,torch.nn.Parameter(value.clone(),requires_grad=False))
    return model


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_only_seven_precision_calls_are_translated(self):
        source,report=r.translated_source();self.assertEqual(len(report['changed_calls']),7)
        self.assertTrue(report['complete_ast_reversal_exact']);self.assertFalse(report['other_ast_nodes_changed'])
        self.assertEqual({x['line']for x in report['changed_calls']},{27,38,238,246,254,286,462})
        self.assertIn('torch.view_as_complex',source);self.assertIn('torch.float64',source)

    def test_resident_and_streamed_outputs_and_native_dtypes_exact(self):
        model,state,inputs=fixture();ref=resident(state);before={n:s.tensor_sha(v)for n,v in state.items()}
        expected=r.forward(ref,*inputs,12);provider=DictSource(state)
        with s.StreamedModules(model,provider)as stream:
            actual=r.forward(model,*inputs,12);record=stream.completed()
        self.assertTrue(torch.equal(actual,expected));self.assertGreater(float(actual.abs().max()),0.)
        self.assertEqual(len(provider.calls),len(state));self.assertEqual(set(provider.calls),set(state))
        self.assertEqual(record['released_parameter_owners'],len(state));self.assertFalse(stream.handles)
        self.assertTrue(all(p.device.type=='meta'for p in model.parameters()));self.assertEqual(model.freqs.device.type,'cpu')
        types={x['group']:x['output']['dtype']for x in stream.rows if x['event']=='evicted'}
        self.assertEqual(types['patch_embedding'],'torch.bfloat16');self.assertEqual(types['text_embedding'],'torch.bfloat16')
        self.assertTrue(all(types[n]=='torch.float32'for n in ['time_embedding','time_projection','blocks.0','blocks.1','head']))
        self.assertEqual(before,{n:s.tensor_sha(v)for n,v in state.items()})

    def test_repeated_streams_and_autocast_cache_control(self):
        model,state,inputs=fixture();ref=resident(state)
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=True):
            cached=ref([inputs[0]],inputs[1],[inputs[2]],12)[0]
        self.assertTrue(torch.equal(cached,r.forward(ref,*inputs,12)))
        outcomes=[]
        for _ in range(2):
            with s.StreamedModules(model,DictSource(state))as stream:
                outcomes.append(r.forward(model,*inputs,12));stream.completed()
        self.assertTrue(torch.equal(outcomes[0],outcomes[1]));self.assertTrue(torch.equal(cached,outcomes[0]))

    def test_failed_source_native_and_event_paths_release_hooks_and_parameters(self):
        for failure in ['load','native','event']:
            model,state,inputs=fixture();provider=DictSource(state);base=provider.load
            def load(name,shape):
                if name.startswith('blocks.1.')and failure=='load':raise RuntimeError('fixture load failure')
                return base(name,shape)
            provider.load=load
            def event(row):
                if row['group']=='blocks.1'and row['event']=='loaded'and failure=='event':raise RuntimeError('fixture event failure')
            patch=mock.patch.object(model.blocks[1],'forward',side_effect=RuntimeError('fixture native failure'))if failure=='native'else mock.patch.dict({}, {})
            with patch:
                with self.assertRaises(RuntimeError):
                    with s.StreamedModules(model,provider,event=event)as stream:r.forward(model,*inputs,12)
            self.assertFalse(stream.handles);self.assertTrue(all(p.device.type=='meta'for p in model.parameters()))
            self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))

    def test_partial_hook_registration_is_removed(self):
        model,state,_=fixture()
        with mock.patch.object(model.time_projection,'register_forward_pre_hook',side_effect=RuntimeError('fixture registration failure')):
            with self.assertRaises(RuntimeError):
                with s.StreamedModules(model,DictSource(state)):pass
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))

    def test_actual_retained_input_and_text_only(self):
        base=r.HERE.parent
        values,contexts,identity=load_inputs(base/'core-results/cpu-pair-v1',base.parent/'wan_adapter/text_cache/native-results')
        self.assertEqual(values['initial_latent'].shape,(48,5,18,32));self.assertEqual(contexts['atrium'].shape,(25,4096))
        self.assertFalse(identity['noise_regenerated']);self.assertFalse(identity['future_target_read'])

    def test_poisoned_tensor_file_rejected_before_materialization(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp)/'poison.safetensors';save_file({'good':torch.zeros(2),'target':torch.zeros(2)},str(p))
            with self.assertRaisesRegex(ValueError,'Unexpected keys'):read_exact(p,{'good':(2,)})

    def test_shard_verification_and_source_copies(self):
        model,state,_=fixture()
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);save_file(state,str(root/'weights.safetensors'));(root/'config.json').write_text('{}')
            index={'weight_map':{n:'weights.safetensors'for n in state}}
            (root/'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps(index))
            provenance={'config_sha256':s.sha(root/'config.json'),'index_sha256':s.sha(root/'diffusion_pytorch_model.safetensors.index.json'),
                'weights':[{'file':'weights.safetensors','bytes':(root/'weights.safetensors').stat().st_size,'sha256':s.sha(root/'weights.safetensors')}]}
            source=s.ShardSource(root,model,provenance)
            for n,v in state.items():self.assertTrue(torch.equal(source.load(n,v.shape),v))
            self.assertTrue(all(v['source_owner_released']for v in source.records.values()))
            plan=s.ShardSource(root,model,provenance,verify_bytes=False)
            with self.assertRaises(RuntimeError):plan.load(next(iter(state)),state[next(iter(state))].shape)
            with (root/'weights.safetensors').open('ab')as f:f.write(b'changed')
            with self.assertRaises(ValueError):s.ShardSource(root,model,provenance)

    def test_stale_preflight_is_rejected(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp)/'report.json';record={'status':'passed','tests':8,'source_sha256':run.hashes()};p.write_text(json.dumps(record));run.preflight(p)
            record['source_sha256']['reference.py']='changed';p.write_text(json.dumps(record))
            with self.assertRaises(ValueError):run.preflight(p)

    def test_child_ignoring_terminate_is_killed(self):
        process=mock.Mock();process.poll.return_value=None
        process.wait.side_effect=[run.subprocess.TimeoutExpired('fixture',3),-9]
        run.stop_child(process)
        process.terminate.assert_called_once();process.kill.assert_called_once();self.assertEqual(process.wait.call_count,2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh CPU evidence required')
    a.output.parent.mkdir(parents=True,exist_ok=True);before=run.hashes();started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    after=run.hashes();record={'status':'passed'if result.wasSuccessful()and before==after else'failed',
        'tests':result.testsRun,'seconds':time.monotonic()-started,'failures':len(result.failures),'errors':len(result.errors),
        'source_sha256':before,'source_unchanged':before==after,'external_weight_values_loaded':False,'gpu_operations':False,
        'scope':'Small random-weight official model, original ownership/precision contracts and retained input artifact reads only'}
    run.atomic(a.output,record)
    for name,digest in before.items():
        dst=a.output.parent/'checked-source'/(name+'.txt');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((r.HERE/name).read_bytes())
        if s.sha(dst)!=digest:raise RuntimeError('Snapshot changed')
    if record['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
