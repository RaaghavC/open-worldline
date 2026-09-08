# SPDX-License-Identifier: Apache-2.0
"""Small CPU arrays and stand-in codec outputs; no weights or CUDA."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
import cache
import packet
import run
from experiments.atrium_factorial import reader
from experiments.wan22_native.official_cpu.streaming import sha,tensor_sha
from experiments.wan22_native.spatial_reference.guards import atomic


@pytest.fixture(autouse=True)
def cpu_only():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    yield
    torch.set_num_threads(previous)
    assert not torch.cuda.is_initialized()


def window(arm):
    index=reader.ARMS.index(arm)
    initial=np.full((3,2,4),.5,dtype=np.float32)
    future=np.full((16,3,2,4),index/10,dtype=np.float32)
    commands=np.zeros((16,6),dtype=np.float32)
    commands[:,3]=0 if index<2 else (.02 if index<4 else -.02)
    commands[0,5]=index%2
    return reader.RGBActionWindow(initial,future,commands)


def fake_encoder(rgb):
    frames=1 if rgb.shape[2]==1 else 5
    result=torch.full((1,48,frames,44,78),float(rgb.mean()),dtype=torch.float32)
    result[:,:,:1]=rgb[:,:,:1].mean()
    return result


def identity():
    return dict(source_sha256=packet.sources(),manifest_sha256=packet.MANIFEST,
                commands_sha256={a:tensor_sha(torch.from_numpy(window(a).commands)) for a in reader.ARMS})


def test_axis_mapping_keeps_exact_pixels_commands_and_separate_storage():
    w=window('left_interact');before=copy.deepcopy(w)
    rgb,commands=cache.video(w)
    assert rgb.shape==(1,3,17,2,4) and commands.shape==(1,16,6)
    assert torch.equal(rgb[0,:,0],torch.from_numpy(w.initial_rgb))
    assert torch.equal(rgb[0,:,1:].permute(1,0,2,3),torch.from_numpy(w.future_rgb_targets))
    rgb.zero_();commands.zero_()
    assert np.array_equal(w.initial_rgb,before.initial_rgb)
    assert np.array_equal(w.future_rgb_targets,before.future_rgb_targets)
    assert np.array_equal(w.commands,before.commands)


def test_all_eight_calls_twelve_gates_reader_and_canonical_arrays(tmp_path):
    calls=[]
    def encode(rgb):calls.append(rgb.shape[2]);return fake_encoder(rgb)
    out=tmp_path/'result'
    report=cache.encode_sequence(out,window,encode,identity=identity())
    assert calls==[1,17,17,17,17,17,17,1]
    assert report['status']=='passed' and len(report['checks'])==12
    assert all(r['passed'] and r['bit_exact_equal'] for r in report['checks'])
    assert len(report['files'])==8 and report['encoder_attempts']==8
    values,provenance=cache.read_window(out,'left_interact')
    assert set(values)=={'target','observation','commands'}
    assert torch.equal(values['target'][:,:,:1],values['observation'])
    assert torch.equal(values['commands'][0],torch.from_numpy(window('left_interact').commands))
    assert provenance['materialized_tensor_keys']==['target','observation','commands']


@pytest.mark.parametrize('kind',['nonfinite','exception','prefix','mutate'])
def test_encoder_failure_preserves_completed_outputs(tmp_path,kind):
    calls=0
    def encode(rgb):
        nonlocal calls
        calls+=1
        if calls==2:
            if kind=='exception':raise RuntimeError('interrupted encoder')
            value=fake_encoder(rgb)
            if kind=='nonfinite':value.flatten()[0]=float('nan')
            if kind=='prefix':value[:,:,:1]+=.01
            if kind=='mutate':rgb.zero_()
            return value
        return fake_encoder(rgb)
    out=tmp_path/'result'
    with pytest.raises((ValueError,RuntimeError)):
        cache.encode_sequence(out,window,encode,identity=identity())
    report=packet.evidence.read_json(out/'completion.json')
    assert report['status']=='failed' and report['encoder_attempts']==2
    assert (out/'observation.safetensors').exists()
    assert (out/'stationary_closed.safetensors').exists()==(kind!='exception')
    with pytest.raises(ValueError):cache.read_window(out,'stationary_closed')


def test_reader_recomputes_prefix_despite_forged_passed_flags(tmp_path):
    out=tmp_path/'result';cache.encode_sequence(out,window,fake_encoder,identity=identity())
    path=out/'left_closed.safetensors';record=packet.evidence.read_json(out/'completion.json')
    values,_=cache.read_window(out,'left_closed');values['target'][:,:,:1]+=.01
    path.unlink();record['files'][path.name]=cache.save(path,values)
    atomic(out/'completion.json',record)
    with pytest.raises(ValueError,match='prefix evidence'):cache.read_window(out,'left_closed')


def test_reader_binds_command_bytes_to_prepared_receipt(tmp_path):
    out=tmp_path/'result';cache.encode_sequence(out,window,fake_encoder,identity=identity())
    values,_=cache.read_window(out,'right_interact');values['commands'][0,0,5]=0
    path=out/'right_interact.safetensors';path.unlink();record=packet.evidence.read_json(out/'completion.json')
    record['files'][path.name]=cache.save(path,values);atomic(out/'completion.json',record)
    with pytest.raises(ValueError,match='Commands differ'):cache.read_window(out,'right_interact')


def test_conditioning_only_does_not_materialize_future_target(tmp_path,monkeypatch):
    out=tmp_path/'result';cache.encode_sequence(out,window,fake_encoder,identity=identity())
    original=cache.safe_open;materialized=[]
    class Handle:
        def __init__(self,*a,**kw):self.context=original(*a,**kw)
        def __enter__(self):self.actual=self.context.__enter__();return self
        def __exit__(self,*a):return self.context.__exit__(*a)
        def keys(self):return self.actual.keys()
        def get_slice(self,name):return self.actual.get_slice(name)
        def get_tensor(self,name):
            materialized.append(name)
            assert name!='target'
            return self.actual.get_tensor(name)
    monkeypatch.setattr(cache,'safe_open',Handle)
    values,_=cache.read_window(out,'stationary_interact',conditioning_only=True)
    assert set(values)=={'observation','commands'} and 'target' not in materialized


def test_prepare_and_read_reject_changed_png_packet(tmp_path,monkeypatch):
    source=tmp_path/'capture';source.mkdir();(source/'frame.png').write_bytes(b'original')
    inputs={'files':{'frame.png':{'bytes':8,'sha256':sha(source/'frame.png')}}}
    def input_plan(root):
        if (Path(root)/'frame.png').read_bytes()!=b'original':raise ValueError('PNG changed')
        return inputs
    monkeypatch.setattr(packet,'input_plan',input_plan)
    monkeypatch.setattr(packet,'sources',lambda:{})
    monkeypatch.setattr(packet,'source_paths',lambda:{})
    cpu=tmp_path/'cpu.json';atomic(cpu,{'status':'passed','tests':5,'source_sha256':{},'sources_unchanged':True,
                                      'pytest_exit_code':0,'failures':0,'errors':0,'skipped':0})
    out=tmp_path/'prepared';packet.prepare(source,cpu,out)
    assert packet.read_prepared(out)['status']=='prepared'
    (out/'capture/frame.png').write_bytes(b'changed!')
    with pytest.raises(ValueError,match='PNG changed'):packet.read_prepared(out)


def test_parent_deadline_and_primary_cleanup_error_retained(tmp_path,monkeypatch):
    root=tmp_path/'prepared';root.mkdir();atomic(root/'plan.json',{})
    admission=tmp_path/'admission.json';atomic(admission,{})
    monkeypatch.setattr(packet,'read_prepared',lambda p:{'source_sha256':{}})
    monkeypatch.setattr(packet,'admission',lambda *a:{'sha256':'a'*64})
    monkeypatch.setattr(run,'_deadline',lambda *a:(_ for _ in ()).throw(RuntimeError('expired')))
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**kw:pytest.fail('Expired worker started'))
    with pytest.raises(RuntimeError,match='expired'):run.execute(root,tmp_path,'fake',admission)
    assert packet.evidence.read_json(root/'metrics.json')['model_execution'] is False
    with pytest.raises(ValueError,match='Single-use'):run.execute(root,tmp_path,'fake',admission)


def test_admission_rejects_old_family_or_expanded_limits(tmp_path):
    atomic(tmp_path/'plan.json',{});plan={'source_sha256':{},'cpu_report_sha256':'b'*64}
    record=dict(schema=packet.SCHEMA,scope=packet.SCOPE,decision='admit',plan_sha256=sha(tmp_path/'plan.json'),
                source_sha256={},manifest_sha256=packet.MANIFEST,cpu_report_sha256='b'*64,
                expected_gpu='gpu',limits=run.limits('codec'),core_model_loaded=False,training_admitted=False)
    path=tmp_path/'admission.json';atomic(path,record)
    assert packet.admission(path,tmp_path,plan,'gpu')['sha256']==sha(path)
    for key,value in [('schema','old-cache'),('training_admitted',True),('limits',{})]:
        bad=dict(record);bad[key]=value;atomic(path,bad)
        with pytest.raises(ValueError):packet.admission(path,tmp_path,plan,'gpu')
