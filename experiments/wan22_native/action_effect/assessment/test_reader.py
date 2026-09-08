import importlib.util
import json
import struct
import sys
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('heldout_reader',Path(__file__).with_name('reader.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
ROOT=Path(__file__).resolve().parents[2]

def test_actual_packet():
    m,noises=r.read_heldout(ROOT/'work/wan22-action-effect-heldout-v1')
    assert len(noises)==4 and m['seed']==2026090801
    assert all(a.shape==r.SHAPE and a.dtype==np.float32 for a in noises.values())
    with pytest.raises(ValueError):r.read_heldout(ROOT/'work/wan22-action-effect-heldout-v1','0'*64)

def test_header_and_payload_fail_closed(tmp_path):
    p=tmp_path/'x.safetensors'
    def write(header,payload=b'\0'*4):
        h=json.dumps(header).encode();p.write_bytes(struct.pack('<Q',len(h))+h+payload)
    write({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}})
    assert r.read_tensor_file(p,{'x':((1,),'F32')},r.sha(p))['x'][0]==0
    for h in ({'x':{'dtype':'F32','shape':[2],'data_offsets':[0,4]}},
              {'target':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},
              {'x':{'dtype':'F32','shape':[1],'data_offsets':[1,5]}}):
        write(h)
        with pytest.raises(ValueError):r.read_tensor_file(p,{'x':((1,),'F32')},r.sha(p))
    write({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},struct.pack('<f',float('nan')))
    with pytest.raises(ValueError):r.read_tensor_file(p,{'x':((1,),'F32')},r.sha(p))
    with pytest.raises(ValueError):r.parse('{"x":1,"x":2}')

def test_endpoint_scoring_sign_guidance_prefix_and_magnitudes():
    zero=np.zeros(r.SHAPE,np.float32);target=np.ones(r.SHAPE,np.float32)*5
    predictions={k:zero.copy() for k in ('closed-positive','closed-negative','open-positive','open-negative')}
    predictions['open-positive'][:]=-1
    good=r.score(predictions,target)
    assert good['future_contrast_mse']==0 and good['normalized_contrast_mse']==0
    assert abs(good['cosine']-1)<1e-12 and good['magnitudes']['open-guided']['rms']==5
    target[:,:,:1]=999;predictions['open-positive'][:,:,:1]=-777
    assert r.score(predictions,target)['future_contrast_mse']==0
    predictions['open-positive'][:]=0
    failed=r.score(predictions,target)
    assert failed['normalized_contrast_mse']==1 and failed['cosine'] is None
    predictions['open-negative'][:]=np.finfo(np.float32).max
    with np.errstate(over='ignore',invalid='ignore'),pytest.raises(ValueError):r.score(predictions,target)

def fixture():
    noises={f'noise-{i:04d}.safetensors':np.full(r.SHAPE,i,np.float32) for i in range(4)}
    obs=np.full((1,48,1,44,78),7,np.float32)
    commands={a:np.zeros((1,16,6),np.float32) for a in r.ARMS};commands['open'][0,0,5]=1
    contexts={'positive':np.zeros((25,4096),np.float32),'negative':np.ones((126,4096),np.float32)}
    return noises,obs,commands,contexts

def test_eight_feature_extracts_reused_by_all_three_checkpoints():
    noises,obs,commands,contexts=fixture();extracts=[];predictions=[];saved=[]
    def extract(name,text,x,times,context):
        assert np.array_equal(x[:,:,:1],obs) and np.array_equal(x[:,:,1:],noises[name][:,:,1:])
        assert np.all(times[:,:858]==0) and np.all(times[:,858:]==999)
        handle=object();extracts.append((handle,name,text));x[:]=123;context[:]=999
        return handle
    def predict(cp,features,command,observation):
        assert features in [x[0] for x in extracts]
        assert np.array_equal(observation,obs);predictions.append((cp,features,int(command[0,0,5])))
        command[:]=666;observation[:]=888
        return np.zeros(r.SHAPE,np.float32)
    report=r.evaluate_matrix(noises,obs,commands,contexts,extract,predict,lambda name,v:saved.append(name))
    assert report['feature_extracts']==len(extracts)==8 and len(predictions)==len(saved)==48
    assert all(sum(p[1] is e[0] for p in predictions)==6 for e in extracts)
    assert np.all(obs==7) and commands['open'][0,0,5]==1 and np.all(contexts['positive']==0)
    assert report['checkpoint_admission_checked_here'] is False

def test_failure_retains_completed_predictions_and_stops():
    values=fixture();saved=[];calls=[]
    def predict(*args):
        calls.append(1)
        if len(calls)==3:raise RuntimeError('injected prediction failure')
        return np.zeros(r.SHAPE,np.float32)
    with pytest.raises(RuntimeError):r.evaluate_matrix(*values,lambda *x:object(),predict,lambda n,v:saved.append(n))
    assert len(saved)==2 and len(calls)==3

def test_saved_reader_rejects_missing_inventory():
    with pytest.raises(ValueError):r.read_saved_scores(Path('.'),{},[],np.ones(r.SHAPE,np.float32))

def test_reader_imports_no_torch():
    # The test environment may import Torch elsewhere. The reader source must
    # contain no Torch import, generator or model-loading operation.
    import ast
    tree=ast.parse(Path(r.__file__).read_text())
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and ('torch' in ast.unparse(n)) for n in ast.walk(tree))
