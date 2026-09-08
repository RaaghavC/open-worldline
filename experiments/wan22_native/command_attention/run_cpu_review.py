"""Run only the declared small CPU suite and retain source/report evidence."""
from pathlib import Path
import hashlib,json,platform,sys,time,importlib.metadata
import torch
import pytest

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    import experiments.wan22_native.cuda_reference.vendor.model as native
    repo=Path(native.__file__).resolve().parents[4]
    paths={f'local/{n}':HERE/n for n in ('controller.py','bridge.py','test_cpu.py','run_cpu_review.py')}
    for n in ('model.py','attention.py'):
        paths['vendor/'+n]=Path(native.__file__).parent/n
    paths['fixture/test_bridge.py']=repo/'experiments/wan22_native/intermediate_action/test_bridge.py'
    before={n:sha(v) for n,v in paths.items()}
    records=[]
    class Plugin:
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.failed:records.append({'name':report.nodeid,'phase':report.when,'outcome':report.outcome})
    if torch.cuda.is_initialized():raise RuntimeError('CPU review must not initialize CUDA')
    start=time.monotonic()
    result=pytest.main([str(HERE/'test_cpu.py'),'-q','-p','no:cacheprovider','--junitxml='+str(out/'junit.xml')],plugins=[Plugin()])
    after={n:sha(v) for n,v in paths.items()}
    passed=sum(r['phase']=='call' and r['outcome']=='passed' for r in records)
    report={'schema':'worldline-command-attention-controller-cpu-v1','status':'passed' if result==0 and before==after else 'failed',
        'pytest_exit_code':int(result),'tests_passed':passed,'test_records':records,'source_sha256':before,'sources_unchanged':before==after,
        'cuda_initialized':torch.cuda.is_initialized(),'model_weights_loaded':False,'literal_tiny_native_model_executed':True,
        'production_cuda_behavior_measured':False,'python':platform.python_version(),
        'packages':{x:importlib.metadata.version(x) for x in ('torch','numpy','diffusers','pytest')},
        'default_trainable_parameter_count':4_936_448,'elapsed_seconds':time.monotonic()-start,
        'limits':['CPU attention fixture is mathematical FP32 attention, not FlashAttention2 or CUDA BF16 validation.','No high-resolution model, resource test, training result or quality claim.']}
    for n,v in paths.items():
        dest=out/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(v.read_bytes())
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('report_sha256',sha(out/'report.json'))
    raise SystemExit(result if before==after else 1)

if __name__=='__main__':main()
