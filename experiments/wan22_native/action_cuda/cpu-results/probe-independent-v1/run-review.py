"""One CPU-only independent probe review; preserves exact checked sources."""
from pathlib import Path
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import sys
import time

BASE = Path(__file__).resolve().parent.parent
REPO = BASE / "outputs/open-worldline"
OUT = BASE / "work/wan22-action-cuda-probe-independent-v1"
OUT.mkdir(exist_ok=False)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
import torch
import pytest
from experiments.wan22_native.action_cuda import probe_evidence

before = probe_evidence.source_hashes(independent=True)
for name, source in probe_evidence.source_paths(independent=True).items():
    target = OUT / 'checked-source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before[name]
shutil.copyfile(__file__, OUT / 'run-review.py')

class Counts:
    def __init__(self):
        self.tests = self.failures = self.errors = self.skipped = 0
    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.tests += 1
        if report.failed:
            if report.when == 'call': self.failures += 1
            else: self.errors += 1
        if report.skipped: self.skipped += 1
    def pytest_collectreport(self, report):
        if report.failed: self.errors += 1

counts = Counts()
capture = io.StringIO()
started = time.monotonic()
with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
    code = int(pytest.main(['experiments/wan22_native/action_cuda/test_probe_independent.py',
                           '-q', '-p', 'no:cacheprovider'], plugins=[counts]))
elapsed = time.monotonic() - started
after = probe_evidence.source_hashes(independent=True)
(OUT / 'pytest-output.txt').write_text(capture.getvalue())
report = {
    'schema': 'worldline-wan22-action-cuda-probe-independent-v1',
    'status': 'passed' if code == 0 and before == after and counts.tests == 5 else 'failed',
    'tests': counts.tests, 'pytest_exit_code': code,
    'failures': counts.failures, 'errors': counts.errors, 'skipped': counts.skipped,
    'elapsed_seconds': elapsed, 'independent_review': True,
    'source_sha256': before, 'source_sha256_after': after,
    'sources_unchanged': before == after,
    'python': platform.python_version(), 'platform': platform.platform(),
    'versions': {name: importlib.metadata.version(name) for name in
                 ['torch', 'numpy', 'safetensors', 'pytest']},
    'cuda_initialized': torch.cuda.is_initialized(),
    'scope': 'Tiny CPU fixtures only; no external model weights, CUDA, cloud or quality test.',
    'checks': [
        'Both fixed zero-adapter bounds independently reject failure, including zero reference and NaN.',
        'Two paired AdamW updates exactly match an independently written future-only flow objective.',
        'Original frozen parameter values and gradients remain unchanged; all adapter gradients exist.',
        'A bounded nonexact cross-length target prefix remains unchanged while noisy prefix equals independent observation.',
        'Parity failure and bridge exception retain completed native output and stop before optimizer mutation.'
    ],
    'limitations': ['Literal tiny native head runs on CPU without attention blocks.',
                    'Actual CUDA zero-adapter identity and memory remain unmeasured.',
                    'Numerical update correctness does not establish learned action control.'],
    'test_dependency_note': 'Used existing system pytest through an appended search path; no packages installed.'
}
(OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
if report['status'] == 'passed':
    probe_evidence.review(OUT / 'report.json', independent=True)
print(capture.getvalue())
print(json.dumps({'status': report['status'], 'tests': counts.tests,
                  'elapsed_seconds': elapsed, 'report': str(OUT / 'report.json'),
                  'sha256': hashlib.sha256((OUT / 'report.json').read_bytes()).hexdigest()}))
raise SystemExit(code or (0 if report['status'] == 'passed' else 1))
