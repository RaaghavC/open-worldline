# SPDX-License-Identifier: Apache-2.0
"""Run synthetic CPU audit checks and retain a source-bound report."""
import argparse
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import time
import unittest
import torch
from .run import sources
from .weights import sha

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(HERE):
        raise ValueError('Fresh CPU review outside the audit source package required')
    source_map = sources(); tests = {p.name: sha(p) for p in sorted(HERE.glob('test_*.py'))}
    suite = unittest.defaultTestLoader.loadTestsFromNames([__package__ + '.' + p.stem for p in sorted(HERE.glob('test_*.py'))])
    stream = io.StringIO(); started = time.monotonic()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    unchanged = source_map == sources() and tests == {p.name: sha(p) for p in sorted(HERE.glob('test_*.py'))}
    passed = result.wasSuccessful() and not result.skipped and unchanged and not torch.cuda.is_initialized()
    report = {'schema': 'wan22-spatial-audit-cpu-review-v1', 'status': 'passed' if passed else 'failed',
              'source_sha256': source_map, 'test_sha256': tests, 'sources_unchanged': unchanged,
              'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'elapsed_seconds': time.monotonic() - started,
              'cuda_initialized': torch.cuda.is_initialized(), 'new_model_execution': False,
              'original_weight_values_loaded': False, 'python': sys.version,
              'dependencies': {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'Pillow', 'safetensors', 'diffusers')},
              'log': stream.getvalue()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'tests_run', 'failures', 'errors', 'skipped', 'elapsed_seconds')}))
    if not passed:
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == '__main__':
    main()
