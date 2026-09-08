"""Run bounded visual-reader CPU checks and retain the source identity."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path.cwd()))
import inputs as inp
import test_run

out=Path(sys.argv[1]).resolve()
if out.exists():raise ValueError('Fresh CPU report path required')
before=inp.sources(); stream=io.StringIO()
with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
    result=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_run))
after=inp.sources()
report={'schema':'worldline-action-visual-cpu-v1','status':'passed' if result.wasSuccessful() and before==after else 'failed',
        'source_sha256':before,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'source_unchanged':before==after,'native_model_execution':False,'cloud_actions':False}
out.write_text(json.dumps(report,indent=2)+'\n')
out.with_suffix('.txt').write_text(stream.getvalue())
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
if report['status']!='passed':raise SystemExit(1)
