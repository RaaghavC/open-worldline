from pathlib import Path
import contextlib,hashlib,json,os,platform,shutil,sys,time
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
import numpy,PIL,pytest,safetensors,torch
repo=Path.cwd(); out=repo.parent.parent/'work/wan22-action-cuda-independent-v1'
if out.exists(): raise RuntimeError('Fresh report directory required')
author=repo/'experiments/wan22_native/action_cuda/cpu-results/v1/report.json'
rel=list(json.loads(author.read_text())['source_sha256'])+['experiments/wan22_native/action_cuda/test_independent.py']
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before={n:sha(repo/n) for n in rel}; out.mkdir()
class Records:
    def __init__(self): self.results=[]
    def pytest_runtest_logreport(self, report):
        if report.when=='call' or report.failed: self.results.append({'name':report.nodeid,'stage':report.when,'outcome':report.outcome,'duration_seconds':report.duration})
records=Records(); start=time.monotonic()
with (out/'pytest-output.txt').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    exitcode=pytest.main(['-q','experiments/wan22_native/action_cuda/test_independent.py','-p','no:cacheprovider'],plugins=[records])
after={n:sha(repo/n) for n in rel}
report={'status':'passed' if exitcode==0 and before==after else 'failed','tests':len([r for r in records.results if r['stage']=='call']),'pytest_exit_code':int(exitcode),'elapsed_seconds':time.monotonic()-start,'test_results':records.results,'source_sha256':after,'sources_unchanged':before==after,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__,'safetensors':safetensors.__version__,'pytest':pytest.__version__,'pytest_location':str(Path(pytest.__file__).parent),'pytest_plugin_autoload_disabled':True},'author_report_sha256':sha(author),'output_sha256':sha(out/'pytest-output.txt'),'scope':'CPU-only small literal upstream zero-attention-block model/head/unpatchify and existing adapter. No external weights, CUDA/FA2 inference, training study, cloud calls or quality claim.','pre_test_setup_failure':'Initial invocation stopped before tests at ModuleNotFoundError: pytest in isolated environment. This invocation appended the existing system pytest path; no packages installed.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for n in rel:
    p=out/'checked-source'/(n+'.txt');p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/n,p)
shutil.copyfile(Path(__file__),out/'run-review.py')
print(json.dumps({'status':report['status'],'tests':report['tests'],'elapsed_seconds':report['elapsed_seconds'],'report_sha256':sha(out/'report.json'),'report':str(out/'report.json')}))
if exitcode: print((out/'pytest-output.txt').read_text())
raise SystemExit(int(exitcode) if exitcode else (0 if before==after else 1))
