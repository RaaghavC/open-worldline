"""One source-bound run of the tiny CPU reader fixtures."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,platform,importlib.metadata,shutil
H=Path(__file__).resolve().parent;O=H/'cpu-v1';O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=['audit.py','checks.py','test_checks.py','pins.json','reference/fixed128-audit.py','run_review.py']
before={n:sha(H/n)for n in names}
for n in names:
 p=O/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(H/n,p)
t=time.monotonic();p=subprocess.run([sys.executable,str(H/'test_checks.py')],capture_output=True,text=True);elapsed=time.monotonic()-t
(O/'output.txt').write_text(p.stdout+p.stderr)
after={n:sha(H/n)for n in names};passed=p.returncode==0 and before==after and 'Ran 10 tests' in p.stderr and p.stderr.rstrip().endswith('OK')
report={'schema':'worldline-action-effect128-audit-cpu-v1','status':'passed'if passed else'failed','tests':10,'failures':0 if passed else None,'errors':0 if passed else None,'skipped':0,'exit_code':p.returncode,'elapsed_seconds':elapsed,'source_sha256':before,'sources_unchanged':before==after,'output_sha256':sha(O/'output.txt'),'python':platform.python_version(),'numpy':importlib.metadata.version('numpy'),'producer_source_sha256':json.loads((H/'pins.json').read_text())['source_sha256'],'prepared_plan_sha256':json.loads((H/'pins.json').read_text())['prepared_plan_sha256'],'actual_run_audited':False,'model_execution':False,'cloud_operations':False,'fixture_scope':'Tiny NumPy analytic cases plus deliberate source-preflight failure before Torch import; no model weights, CUDA or model backward.'}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'tests':10,'seconds':elapsed,'report_sha256':sha(O/'report.json')}));sys.exit(0 if passed else 1)
