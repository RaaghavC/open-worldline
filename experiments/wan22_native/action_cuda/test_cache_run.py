# SPDX-License-Identifier: Apache-2.0
"""Independent cache-parent admission and cleanup checks. No CUDA or models."""
import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from . import cache_run as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def plan_fixture(review):
    return {'schema': runner.SCHEMA, 'mode': 'cache', 'profile': 'spatial',
            'expected_gpu': 'NVIDIA A100-SXM4-80GB', 'limits': runner.limits('codec'),
            'source_sha256': {}, 'cpu_report_sha256': runner.sha(review),
            'input_plan': {'original': 'synthetic protocol fixture'},
            'input_plan_sha256': runner.canonical_sha({'original': 'synthetic protocol fixture'}),
            'model_execution': False, 'quality_assessed': False}


def test_preflight_rejects_stale_sources_skipped_checks_and_cuda_initialized(tmp_path):
    review = tmp_path / 'review.json'
    source, tests = {'data.py': 'a'*64}, {'test_data.py': 'b'*64, 'test_cache_run.py': 'c'*64}
    good = dict(schema=runner.CPU_SCHEMA, status='passed', source_sha256=source,
                test_sha256=tests, tests_run=5, failures=0, errors=0, skipped=0,
                cuda_initialized=False)
    with mock.patch.object(runner, 'sources', return_value=source), mock.patch.object(runner, 'tests', return_value=tests):
        write(review, good)
        assert runner.preflight(review) == runner.sha(review)
        for key, value in [('schema', 'other'), ('status', 'failed'), ('source_sha256', {}),
                           ('test_sha256', {}), ('tests_run', 4), ('tests_run', True),
                           ('failures', 1), ('errors', 1), ('skipped', 1), ('cuda_initialized', True)]:
            write(review, {**good, key: value})
            with pytest.raises(ValueError): runner.preflight(review)


def test_plan_binds_original_input_and_rejects_source_change(tmp_path):
    original = {'profile': 'spatial', 'source_images': ['verified fixture']}
    with mock.patch.object(runner, 'preflight', return_value='a'*64), \
         mock.patch.object(runner.data, 'plan', return_value=original), \
         mock.patch.object(runner, 'sources', return_value={'file': 'b'*64}):
        result = runner.make_plan(tmp_path, 'spatial', 'NVIDIA A100-SXM4-80GB', tmp_path/'review')
        assert result['input_plan'] == original
        assert result['input_plan_sha256'] == runner.canonical_sha(original)
        assert result['model_execution'] is False
        for profile, gpu in [('other', 'NVIDIA A100-SXM4-80GB'), ('spatial', 'Other GPU')]:
            with pytest.raises(ValueError): runner.make_plan(tmp_path, profile, gpu, tmp_path/'review')
    with mock.patch.object(runner, 'preflight', return_value='a'*64), \
         mock.patch.object(runner.data, 'plan', return_value=original), \
         mock.patch.object(runner, 'sources', side_effect=[{'file': 'b'*64}, {'file': 'c'*64}]):
        with pytest.raises(ValueError, match='changed during planning'):
            runner.make_plan(tmp_path, 'spatial', 'NVIDIA A100-SXM4-80GB', tmp_path/'review')


def run_arguments(root, review, output):
    return dict(capture=root/'capture', profile='spatial', weights=root/'weights',
                cpu_report=review, output=output, expected_gpu='NVIDIA A100-SXM4-80GB')


def test_plan_only_and_bad_explicit_input_hash_never_launch_worker(tmp_path):
    root = tmp_path.resolve(); review = root/'review.json'; write(review, {'fixture': True})
    prepared = plan_fixture(review)
    with mock.patch.object(runner, 'make_plan', return_value=prepared), mock.patch.object(runner, 'launch') as launch:
        result = runner.run(**run_arguments(root, review, root/'plan'))
        assert result['status'] == 'planned' and result['model_execution'] is False
        assert json.loads((root/'plan/metrics.json').read_text())['status'] == 'planned'
        with pytest.raises(ValueError, match='matching input-plan'):
            runner.run(**run_arguments(root, review, root/'bad'), execute=True, input_plan_sha256='0'*64)
        assert json.loads((root/'bad/metrics.json').read_text())['status'] == 'failed'
        launch.assert_not_called()
    with pytest.raises(ValueError): runner.fresh_output(root/'plan', root/'capture')


def completed_fixture(out, expected):
    write(out/'terminal.json', {'status': 'complete', 'exit_code': 0, 'mode': 'codec', 'cleanup_error': None})
    write(out/'worker/monitor-terminal.json', {'status': 'complete', 'sample_count': 1})
    write(out/'result/manifest.json', {'fixture': 'not actual CUDA evidence'})
    write(out/'result/completion.json', {'status': 'passed', 'manifest_sha256': runner.sha(out/'result/manifest.json')})
    worker = {key: expected[key] for key in ('profile','source_sha256','input_plan_sha256','cpu_report_sha256')}
    worker.update(status='passed', mode='cache', model_execution=True,
                  result_completion_sha256=runner.sha(out/'result/completion.json'))
    write(out/'worker/metrics.json', worker)


def test_completed_parent_requires_terminal_monitor_no_stop_and_exact_identity(tmp_path):
    out=tmp_path.resolve(); expected={'profile':'spatial','source_sha256':{'source':'a'*64},
        'input_plan_sha256':'b'*64,'cpu_report_sha256':'c'*64}
    completed_fixture(out,expected)
    with mock.patch.object(runner.data, 'read_window', return_value=None) as read_window:
        runner.validate_completed(out,expected)
        assert read_window.call_count == len(runner.data.SELECTION) == 8
        assert [call.args[1] for call in read_window.call_args_list] == [f'{arm}-{start:04d}' for arm,start in runner.data.SELECTION]
        paths=['terminal.json','worker/monitor-terminal.json','worker/metrics.json','result/completion.json']
        originals={name:json.loads((out/name).read_text()) for name in paths}
        mutations=[('terminal.json','exit_code',1),('terminal.json','status','failed'),
            ('terminal.json','cleanup_error','stubborn child'),('worker/monitor-terminal.json','sample_count',0),
            ('worker/metrics.json','input_plan_sha256','changed'),('result/completion.json','manifest_sha256','changed')]
        for name,key,value in mutations:
            write(out/name,{**originals[name],key:value})
            with pytest.raises((ValueError,RuntimeError)): runner.validate_completed(out,expected)
            write(out/name,originals[name])
        write(out/'worker/nested/watchdog-stop.json',{'status':'stopped'})
        with pytest.raises(RuntimeError): runner.validate_completed(out,expected)


class Process:
    def __init__(self): self.returncode=None


@pytest.mark.parametrize('failure_stage',['startup','supervisor'])
def test_launch_failure_stops_owned_child_and_preserves_terminal(tmp_path,failure_stage):
    out=tmp_path.resolve(); failure=KeyboardInterrupt('injected '+failure_stage)
    proc=Process()
    def stop(child):
        assert child is proc; child.returncode=-9
    def create(command, **kwargs):
        assert kwargs['stdin'] == runner.subprocess.DEVNULL
        assert command[-2:] == ['--config',str(out/'launch.json')]
        if failure_stage=='startup': raise failure
        return proc
    with mock.patch.object(runner.subprocess,'Popen',side_effect=create), \
         mock.patch.object(runner,'supervise',side_effect=failure) as supervise, \
         mock.patch.object(runner,'stop_child',side_effect=stop) as cleanup, \
         mock.patch.object(runner,'validate_completed') as validate:
        with pytest.raises(KeyboardInterrupt) as caught: runner.launch({'deadline':500.},out)
        assert caught.value is failure
        if failure_stage=='startup':
            supervise.assert_not_called(); cleanup.assert_not_called()
        else: cleanup.assert_called_once_with(proc)
        validate.assert_not_called()
    terminal=json.loads((out/'terminal.json').read_text())
    assert terminal['status']=='failed'
    assert terminal['exit_code'] == (None if failure_stage=='startup' else -9)


def test_worker_failure_preserves_actual_partial_model_execution(tmp_path):
    root=tmp_path.resolve(); review=root/'review.json'; write(review,{'fixture':True})
    prepared=plan_fixture(review)
    def launch(config,out):
        write(out/'worker/metrics.json',{'status':'failed','model_execution':True})
        raise RuntimeError('injected failure after model work')
    with mock.patch.object(runner,'make_plan',return_value=prepared), \
         mock.patch.object(runner,'launch',side_effect=launch):
        with pytest.raises(RuntimeError):
            runner.run(**run_arguments(root,review,root/'partial'),execute=True,
                       input_plan_sha256=prepared['input_plan_sha256'])
    report=json.loads((root/'partial/metrics.json').read_text())
    assert report['status']=='failed' and report['model_execution'] is True


def test_parent_rejects_deadline_crossed_during_final_verification(tmp_path):
    root=tmp_path.resolve(); review=root/'review.json'; write(review,{'fixture':True})
    prepared=plan_fixture(review); clock=[0.]
    worker={'hardware':{'fixture':True},'result_completion_sha256':'d'*64}
    def launch(config,out):
        assert config['deadline']==600.
        write(out/'worker/metrics.json',worker)
        return worker
    def final_plan(*args):
        clock[0]=601.
        return prepared['input_plan']
    with mock.patch.object(runner,'make_plan',return_value=prepared), \
         mock.patch.object(runner,'launch',side_effect=launch), \
         mock.patch.object(runner,'sources',return_value={}), \
         mock.patch.object(runner.data,'plan',side_effect=final_plan), \
         mock.patch.object(runner.time,'monotonic',side_effect=lambda:clock[0]):
        with pytest.raises(RuntimeError,match='deadline'):
            runner.run(**run_arguments(root,review,root/'late'),execute=True,
                       input_plan_sha256=prepared['input_plan_sha256'])
    assert json.loads((root/'late/metrics.json').read_text())['status']=='failed'


def test_planning_exhausted_deadline_rejects_before_worker_launch(tmp_path):
    root=tmp_path.resolve(); review=root/'review.json'; write(review,{'fixture':True})
    prepared=plan_fixture(review); clock=[0.]
    def plan(*args):
        clock[0]=601.
        return prepared
    with mock.patch.object(runner,'make_plan',side_effect=plan), \
         mock.patch.object(runner,'launch') as launch, \
         mock.patch.object(runner.time,'monotonic',side_effect=lambda:clock[0]):
        with pytest.raises(RuntimeError,match='deadline'):
            runner.run(**run_arguments(root,review,root/'planning-too-long'),execute=True,
                       input_plan_sha256=prepared['input_plan_sha256'])
        launch.assert_not_called()
    report=json.loads((root/'planning-too-long/metrics.json').read_text())
    assert report['status']=='failed' and report['model_execution'] is False
