"""One tiny MPS pooling operation and its gradient. No model or weights."""
from pathlib import Path
import json,hashlib,os,sys,time,shutil
from unittest import mock
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
base=Path(__file__).resolve().parent.parent;repo=base/'outputs/open-worldline';sys.path.insert(0,str(repo))
from experiments.wan22_native.action_adapter import pooling
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=repo/'experiments/wan22_native/action_adapter/pooling.py';model=source.parent/'model.py'
assert sha(source)=='a6c1623aeab10c492a530dee6fbdd14e0089ee79330099025324fc9048374f5b'
assert sha(model)=='74d6584c8762530f5c1fa7568607a605c29a1a8b9fe97c79d09afde590f17375'
assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
cpu=base/'work/wan22-action-pooling-cpu-v1/report.json';assert sha(cpu)=='6db45df9560bbc7de1c3469fd303eaa9222c1a0cd65fa499f24250dd0b290d34'
assert json.loads(cpu.read_text())['status']=='passed'
out=base/'work/wan22-action-pooling-mps-v1';out.mkdir()
for p in (source,model,Path(__file__),cpu):shutil.copyfile(p,out/p.name)
report=dict(schema='worldline-action-adaptive-pooling-mps-v1',status='running',source_sha256={str(p.relative_to(repo)):sha(p)for p in (source,model)},
    cpu_report_sha256=sha(cpu),script_sha256=sha(Path(__file__)),shape=[1,48,18,32],output_size=[4,8],dtype='float32',
    device='mps',fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),atol=2e-7,rtol=3e-6,tolerances_chosen_before_execution=True,
    foundation_weights_loaded=False,model_inference=False,optimizer_updates=0,shape_oracle='CPU torch.nn.functional.adaptive_avg_pool2d',
    python=sys.version,torch=torch.__version__)
try:
    assert torch.backends.mps.is_available();torch.set_num_threads(1)
    rng=torch.Generator().manual_seed(20260907)
    original=torch.randn(1,48,18,32,generator=rng);upstream=torch.randn(1,48,4,8,generator=rng)*.25
    reference=original.clone().requires_grad_();expected=F.adaptive_avg_pool2d(reference,(4,8))
    reference_gradient,=torch.autograd.grad(expected,reference,upstream)
    operand=original.to('mps').requires_grad_();up=upstream.to('mps');torch.mps.synchronize();start=time.perf_counter()
    with mock.patch.object(F,'adaptive_avg_pool2d',side_effect=AssertionError('MPS dispatch called unsupported native adaptive-pooling kernel')):
        actual=pooling.observation_pool2d(operand,(4,8))
        gradient,=torch.autograd.grad(actual,operand,up)
    torch.mps.synchronize();report['forward_and_backward_seconds']=time.perf_counter()-start
    assert actual.device.type==gradient.device.type=='mps'
    actual_cpu=actual.detach().cpu();gradient_cpu=gradient.detach().cpu()
    assert torch.equal(operand.detach().cpu(),original)
    tensors=dict(input=original,upstream=upstream,oracle_output=expected.detach(),oracle_input_gradient=reference_gradient,
        mps_output=actual_cpu,mps_input_gradient=gradient_cpu)
    save_file(tensors,str(out/'tensors.safetensors'))
    report['tensor_sha256']={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()for k,v in tensors.items()}
    report['tensor_file_sha256']=sha(out/'tensors.safetensors')
    report['finite']=all(bool(torch.isfinite(v).all())for v in tensors.values());assert report['finite']
    def compare(a,b):
        d=(a-b).double();return dict(max_absolute=float(d.abs().max()),relative_l2=float(d.norm()/b.double().norm()))
    report['forward']=compare(actual_cpu,expected.detach());report['input_gradient']=compare(gradient_cpu,reference_gradient)
    torch.testing.assert_close(actual_cpu,expected.detach(),atol=2e-7,rtol=3e-6)
    torch.testing.assert_close(gradient_cpu,reference_gradient,atol=2e-7,rtol=3e-6)
    report.update(status='passed',gradient_nonzero=bool(torch.count_nonzero(gradient_cpu)),unsupported_kernel_called=False,
        source_unchanged=sha(source)==report['source_sha256'][str(source.relative_to(repo))],mps_driver_bytes_after=torch.mps.driver_allocated_memory())
    assert report['source_unchanged']
except BaseException as error:
    report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
finally:
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
