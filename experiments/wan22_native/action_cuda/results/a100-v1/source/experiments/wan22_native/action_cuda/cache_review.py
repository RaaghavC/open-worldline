# SPDX-License-Identifier: Apache-2.0
"""Run the bounded CPU cache-contract tests and bind the exact checked sources."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from ..spatial_reference.guards import atomic
from .cache_run import CPU_SCHEMA, HERE, sources, tests


class Outcomes:
    def __init__(self):
        self.tests_run = 0
        self.failures = 0
        self.errors = 0
        self.skipped = 0

    def pytest_collection_finish(self, session):
        self.tests_run = len(session.items)

    def pytest_collectreport(self, report):
        if report.failed:
            self.errors += 1

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped += 1
        if report.failed:
            if report.when == 'call':
                self.failures += 1
            else:
                self.errors += 1


def review(output):
    import pytest
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Fresh regular CPU report required')
    if torch.cuda.is_initialized():
        raise RuntimeError('Start CPU review in a fresh process without initialized CUDA')
    torch.set_num_threads(1)
    before, test_before = sources(), tests()
    outcomes = Outcomes()
    began = time.monotonic()
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    code = pytest.main(['-q', str(HERE / 'test_data.py'), str(HERE / 'test_cache_run.py')], plugins=[outcomes])
    unchanged = before == sources() and test_before == tests()
    passed = code == 0 and outcomes.tests_run >= 5 and not any((outcomes.failures, outcomes.errors, outcomes.skipped))
    passed = passed and unchanged and not torch.cuda.is_initialized()
    result = {'schema': CPU_SCHEMA, 'status': 'passed' if passed else 'failed',
              'tests_run': outcomes.tests_run, 'failures': outcomes.failures,
              'errors': outcomes.errors, 'skipped': outcomes.skipped,
              'source_sha256': before, 'test_sha256': test_before,
              'source_and_tests_unchanged': unchanged, 'cuda_initialized': torch.cuda.is_initialized(),
              'elapsed_seconds': time.monotonic() - began, 'pytest_exit_code': int(code),
              'torch': torch.__version__, 'model_weights_loaded': False,
              'limitation': 'Bounded CPU fixtures establish software contracts, not native CUDA codec equality or image quality.'}
    atomic(output, result)
    if not passed:
        raise RuntimeError('Cache CPU review failed; retained report identifies outcome')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = review(parser.parse_args().output)
    print(json.dumps({key: result[key] for key in ('status', 'tests_run', 'elapsed_seconds', 'cuda_initialized')}))
