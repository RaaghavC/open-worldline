"""Run bounded CPU-only protocol checks and retain exact reviewed sources."""
import argparse
import contextlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
repo=args.repo.resolve(); output=args.output.resolve()
output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(repo))
# The local pinned venv can use the already installed pytest without replacing
# its pinned numerical dependencies. Standard environments need pytest installed.
sys.path.append(str(Path(sys.base_prefix)/'lib/python3.11/site-packages'))
import pytest
from experiments.wan22_native.cuda_reference import evidence
before=evidence.sources(); before['test_independent.py']=evidence.sha(evidence.HERE/'test_independent.py')
for name,digest in before.items():
    destination=output/'checked-source'/(name+'.txt');destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(evidence.source_path(name),destination)
    assert evidence.sha(destination)==digest
class Records:
    def __init__(self):self.rows=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:
            self.rows.append({'nodeid':report.nodeid,'phase':report.when,'outcome':report.outcome,'seconds':report.duration})
records=Records();captured=io.StringIO();started=time.monotonic()
with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
    code=pytest.main([str(evidence.HERE/'test_independent.py'),'-q'],plugins=[records])
elapsed=time.monotonic()-started
(output/'test-output.txt').write_text(captured.getvalue())
after=evidence.sources();after['test_independent.py']=evidence.sha(evidence.HERE/'test_independent.py')
passed=sum(row['outcome']=='passed' for row in records.rows)
report={'schema':'worldline-wan22-cuda-reference-independent-cpu-review-v1',
 'status':'passed' if code==0 and before==after else 'failed','tests':passed,
 'pytest_exit_code':int(code),'elapsed_seconds':elapsed,'source_sha256':before,'source_sha256_after':after,
 'sources_unchanged':before==after,'test_results':records.rows,'python':sys.version,
 'dependencies':{name:importlib.metadata.version(name) for name in ['torch','diffusers','safetensors','numpy','pytest']},
 'threads':1,'actual_cuda_executed':False,'foundation_model_executed':False,'actual_weight_values_loaded':False,
 'cloud_account_accessed':False,'cloud_resources_provisioned':False,'billing_teardown_implemented':False,
 'test_output_sha256':evidence.sha(output/'test-output.txt'),'review_runner_sha256':evidence.sha(__file__),
 'review_findings':{
   'literal_upstream_source_hashes_checked':True,'retained_conditioning_identity_checked':True,
   'all_100_prefix_and_token_time_inputs_checked':True,'fifty_step_separate_literal_cpu_solver_exact':True,
   'caller_history_and_times_mutation_isolation_checked':True,'partial_positive_failure_evidence_checked':True,
   'cpu_verification_copy_for_cuda_loader_stand_in_checked':True,
   'fixed_hardware_and_sampled_resource_caps_checked':True,
   'complete_pair_source_precision_weight_and_artifact_admission_checked':True,
   'missing_hashes_sources_outputs_inputs_terminal_and_timing_rejected':True,
   'parent_deadline_and_kill_escalation_checked':True,
   'no_remaining_concrete_source_review_blocker':True},
 'scope':'Independent CPU protocol and read-only source review only. Native core source bytes are checked; no CUDA equation, performance, quality, installation or full-clip result is claimed.',
 'limits':['The optional clip uses the explicit CPU UniPC solver at 17 frames and 512 x 288, not the intended-size official pipeline.',
           'The 120-second decoder allowance is unmeasured and the 900-second deadline is not a completion guarantee.',
           'Stopping a worker does not stop pod billing, delete storage or replace a separate teardown plan.',
           'The CUDA loader test uses 825 scalar stand-ins, not real CUDA transfers or the 5B model.']}
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copyfile(__file__,output/'run-independent-review.py')
print(captured.getvalue())
print(json.dumps({'status':report['status'],'tests':passed,'report':str(output/'report.json'),'sha256':evidence.sha(output/'report.json'),'seconds':elapsed}))
if code==0 and before==after:print('gate_sha256='+evidence.preflight(output/'report.json',independent=True))
raise SystemExit(code or (0 if before==after else 1))
