"""Portable orchestration checks only; no foundation load, GPU or provider."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
import engine
import run_profile as cli


def test_default_plan_does_not_import_torch_or_dispatch():
    command = "import sys; import run_profile; run_profile.main([]); assert 'torch' not in sys.modules; assert 'native_worker' not in sys.modules"
    result = subprocess.run([sys.executable,'-c',command],cwd=Path(__file__).parent,capture_output=True,text=True,check=True)
    value=json.loads(result.stdout)
    assert value['model_execution'] is False and value['zero_gate']['comparisons']==24
    assert [r['auxiliary'] for r in value['updates']]==[True,False]


def args_for(tmp_path):
    for name in ('repository','controller_source','training_source','prepared','weights'):
        (tmp_path/name).mkdir()
    (tmp_path/'prepared/inputs.json').write_text('{}')
    now=datetime(2026,9,8,tzinfo=timezone.utc)
    return SimpleNamespace(**{n:tmp_path/n for n in ('repository','controller_source','training_source','prepared','weights')},
        output=tmp_path/'output',inputs_sha256=cli.sha(tmp_path/'prepared/inputs.json'),expected_gpu='declared GPU',
        deadline_utc=(now+timedelta(seconds=600)).isoformat()),now


def test_execution_requires_all_explicit_arguments_and_future_short_deadline(tmp_path):
    args,now=args_for(tmp_path)
    assert cli.execution_arguments(args,now)==600
    for replacement in (None,''):
        args.expected_gpu=replacement
        with pytest.raises(ValueError):cli.execution_arguments(args,now)
    args.expected_gpu='declared GPU'
    for offset in (-1,0,901):
        args.deadline_utc=(now+timedelta(seconds=offset)).isoformat()
        with pytest.raises(ValueError):cli.execution_arguments(args,now)
    args.deadline_utc='2026-09-08T00:05:00'
    with pytest.raises(ValueError):cli.execution_arguments(args,now)


def test_execution_rejects_existing_output_and_input_nested_output(tmp_path):
    args,now=args_for(tmp_path);args.output.mkdir()
    with pytest.raises(ValueError):cli.execution_arguments(args,now)
    args.output=args.prepared/'new-result'
    with pytest.raises(ValueError):cli.execution_arguments(args,now)


class Bridge:
    def __init__(self,bad_path=None):
        self.controller=torch.nn.Linear(1,1,bias=False)
        self.bad_path=bad_path
    def __call__(self,x,*args,**kwargs):return x.clone()+(1 if self.bad_path=='full' else 0)
    def extract_features(self,x,*args):return x.clone()
    def predict_from_features(self,x,*args,**kwargs):return x.clone()+(1 if self.bad_path=='cached' else 0)


def execute_fake(bad_path=None):
    bridge=Bridge(bad_path);events=[]
    tensor=torch.ones(1,1,2,2,2,dtype=torch.float32)
    data=dict(windows={a:dict(target=tensor.clone(),observation=tensor[:,:,:1].clone(),commands=torch.zeros(1,16,6)) for a in engine.ARMS},
              positive=torch.ones(1,1),negative=torch.zeros(1,1))
    rows=[dict(update=1,k=506,auxiliary_edge=['stationary_closed','stationary_interact']),dict(update=2,k=200,auxiliary_edge=None)]
    def optimizer():events.append('optimizer');return SimpleNamespace(state={},zero_grad=lambda **kw:None)
    def step(row,noise,opt,retain):
        events.append(row['update'])
        bridge.controller.weight.grad=torch.ones_like(bridge.controller.weight)
        return dict(optimizer_updates=1,main_predictions=2,auxiliary={'predictions':4 if row['update']==1 else 0})
    kwargs=dict(noise=lambda i:tensor,flow_inputs=lambda *a:(tensor,torch.zeros(1,2,dtype=torch.int64),tensor),
                native_predict=lambda x,*a:x.clone(),tensor_sha=lambda v:hashlib.sha256(v.numpy().tobytes()).hexdigest(),
                make_optimizer=optimizer,run_step=step,retain=lambda *a:None,progress=lambda *a:None,
                verify_core=lambda name:events.append(name))
    return lambda:engine.execute(bridge,data,rows,**kwargs),events


@pytest.mark.parametrize('path',['full','cached'])
def test_failed_parity_never_creates_optimizer_or_updates(path):
    run,events=execute_fake(path)
    with pytest.raises(RuntimeError,match='before optimizer creation'):run()
    assert 'optimizer' not in events and not any(isinstance(v,int) for v in events)


def test_all_parities_precede_exact_two_update_schedule():
    run,events=execute_fake();report=run()
    assert len(report['parity'])==24 and all(r['exact_equal'] for r in report['parity'])
    assert events==['after-parity','optimizer',1,2,'after-updates']
    assert report['completed_updates']==2 and report['status']=='passed'


def test_gradient_summary_rejects_missing_nonfinite_and_zero_gradients():
    controller=torch.nn.Linear(1,1,bias=False)
    for gradient in (None,torch.tensor([[float('nan')]]),torch.zeros(1,1)):
        controller.weight.grad=gradient
        with pytest.raises(FloatingPointError):engine.gradient_summary(controller)
    controller.weight.grad=torch.ones(1,1)
    assert engine.gradient_summary(controller)['all_present_finite']


def test_bit_exact_parity_rejects_different_signed_zero_bytes():
    digest=lambda x:hashlib.sha256(x.numpy().tobytes()).hexdigest()
    reference=torch.tensor([0.],dtype=torch.float32)
    actual=torch.tensor([-0.],dtype=torch.float32)
    assert torch.equal(reference,actual)
    assert engine.comparison(reference,actual,digest)['exact_equal'] is False
