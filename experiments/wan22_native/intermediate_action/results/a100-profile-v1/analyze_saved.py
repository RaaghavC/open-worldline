"""Read retained profile evidence; no model execution or generated predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from experiments.action_effect_diagnostics import analyze as diagnostic_source
from experiments.action_effect_diagnostics.analyze import guided_difference, scalar_fit, spatial_measurements

OLD_INDEX = '2e231b055914202d0d6a34ae7b7a7026a24fa339e5f78f199b3b80be5b749241'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def compare(a, b):
    assert a.keys() == b.keys()
    rows = {}
    for name in sorted(a):
        x, y = a[name], b[name]
        assert x.shape == y.shape and x.dtype == y.dtype
        assert np.isfinite(x).all() and np.isfinite(y).all()
        d = x.astype(np.float64) - y.astype(np.float64)
        scale = float(np.linalg.norm(x.astype(np.float64).reshape(-1)))
        rows[name] = dict(exact=bool(np.array_equal(x,y)), max_abs=float(np.max(np.abs(d))),
                          relative_l2=float(np.linalg.norm(d.reshape(-1)))/scale if scale else None)
    return rows

def analyze(profile, old_download, output):
    assert not output.exists()
    result = profile/'result'
    metrics = json.loads((result/'metrics.json').read_text())
    parent = json.loads((profile/'metrics.json').read_text())
    assert parent['status'] == metrics['status'] == 'passed'
    assert sha(result/'metrics.json') == parent['result_metrics_sha256']
    assert metrics['completed_updates'] == 4 and metrics['zero_gate_passed']
    assert len(metrics['parity']) == 16 and all(r['exact_equal'] for r in metrics['parity'])
    for name, digest in metrics['output_sha256'].items():
        p = result/name
        assert p.resolve().is_relative_to(result.resolve()) and not p.is_symlink() and sha(p) == digest
    consumed = {}
    def read_result(name):
        assert name in metrics['output_sha256']
        consumed[name] = metrics['output_sha256'][name]
        return load_file(result/name)
    assert sha(old_download/'index.json') == OLD_INDEX
    old_index = json.loads((old_download/'index.json').read_text())['files']
    def old_file(name):
        key = 'action-results/effect128-spatial-v1/training/'+name
        spec = old_index[key]; p = old_download/'recovered'/key
        assert p.stat().st_size == spec['bytes'] and sha(p) == spec['sha256']
        consumed['historical/'+key] = spec['sha256']
        return load_file(p)
    control = {}
    for arm in ('closed','open'):
        control['main/'+arm] = compare(old_file('prediction-0001-'+arm+'.safetensors'),
                                     read_result('block29/main-0001-'+arm+'.safetensors'))
        for text in ('positive','negative'):
            name = 'auxiliary-0001-'+arm+'-'+text+'.safetensors'
            control['auxiliary/'+arm+'/'+text] = compare(old_file(name),read_result('block29/'+name))
    control['gradients'] = compare(old_file('gradients-after-clip-0001.safetensors'),
                                  read_result('block29/gradients-after-clip-0001.safetensors'))
    assert sha(profile/'plan.json') == parent['plan_sha256'] == metrics['plan_sha256']
    plan = json.loads((profile/'plan.json').read_text())
    targets = {}
    for arm in ('closed','open'):
        name = 'action-results/cache-spatial-run-v1/result/'+arm+'-0000.safetensors'
        p = profile/'inputs'/name
        assert sha(p) == plan['input_files'][name]['sha256']
        consumed['inputs/'+name] = sha(p)
        targets[arm] = load_file(p)['target']
    target = np.subtract(targets['open'], targets['closed'], dtype=np.float32)
    responses = {}
    for block in (28,29):
        values = {}
        for arm in ('closed','open'):
            for text in ('positive','negative'):
                values[arm+'-'+text] = read_result(f'block{block}/auxiliary-0002-{arm}-{text}.safetensors')['velocity']
        prediction = guided_difference(values)
        responses[str(block)] = dict(after_completed_updates=1, saved_draw='noise_0004',
            ordinary=scalar_fit(prediction[:,:,1:],target[:,:,1:]),
            spatial=spatial_measurements(prediction,target))
    report = dict(status='passed', model_execution=False,
        source_sha256={'analyze_saved.py':sha(Path(__file__)),
                       'experiments/action_effect_diagnostics/analyze.py':sha(Path(diagnostic_source.__file__))},
        parent_metrics_sha256=sha(profile/'metrics.json'), result_metrics_sha256=sha(result/'metrics.json'),
        original_control_update1=control, one_update_saved_endpoint_response=responses,
        consumed=consumed, limitations=[
            'Historical control comparison is descriptive and does not relax the native parity requirement.',
            'Response uses already-seen training targets and one saved draw after one completed update.',
            'The target-fitted scalar is a post hoc diagnostic, not a deployable model or a video quality score.',
            'No new forward, training, sampling or decoding was performed by this analyzer.'])
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--old-download',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.profile,a.old_download,a.output)
    print(json.dumps({'status':'passed','output':str(a.output)}))
